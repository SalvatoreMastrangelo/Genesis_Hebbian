"""
measure_sigma_rank — quantify (and validate) the WP2 evaluation ranking noise.
==============================================================================

CMA-ES ranks individuals, so what limits progress is not the absolute per-
individual fitness noise (``sigma_ind``) but the noise on the *difference*
between individuals — ``sigma_rank`` — i.e. the part of the noise that does NOT
cancel when two individuals are compared head-to-head.  This script measures it
directly and doubles as the correctness check for the common-random-numbers
(CRN) implementation.

Method
------
Replicate ONE genome across all ``P`` population slots in a single generation
eval.  Every slot then runs the *same* controller, so the spread of the ``P``
per-individual fitnesses is pure evaluation noise — exactly ``sigma_rank`` for
that controller.  We sweep the two switches that govern that noise:

* ``crn``        — share per-slot domain-randomization draws across individuals
                   flying the same forest (forest layout, mass/COM, joint
                   bias/step noise, latency, aero params, obs noise).
* ``stochastic`` — sample actions from the policy (per-slot) vs use the mean.

The 2x2 sweep decomposes the noise:

    crn=OFF, stoch=ON   -> full sigma_ind   (the legacy / current-run regime)
    crn=ON,  stoch=ON   -> residual from per-slot ACTION sampling (DR shared)
    crn=ON,  stoch=OFF  -> residual from the in-kernel Taichi aero FORCE noise
                           (the one DR source CRN cannot share) — should be small
    crn=OFF, stoch=OFF  -> DR contribution without action sampling

A near-zero ``crn=ON, stoch=OFF`` number is the proof that the torch-side CRN is
complete (every shareable draw is actually shared).  Zero rules (genome=0.5) is
the default probe: with identical controllers AND full sharing, fitnesses should
collapse to a single value.

Usage
-----
.. code-block:: bash

    PYTHONPATH=src python src/WP2/measure_sigma_rank.py \
        --cfg src/WP2/experiments/batch_1/random_period_4/run.yaml \
        --cfg.checkpoint_path        <wp1_actor.pt> \
        --cfg.checkpoint_config_path <wp1_config.yaml> \
        --repeats 3

Notes
-----
* Needs a GPU (runs Genesis), so run it locally with a GPU or as a short cluster
  job.  Cost ≈ a handful of generation-evals.
* ``--cfg.evaluation.num_eval_envs`` controls F (forests/individual); shrink it
  for a fast smoke test, keep the production value for a representative number.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("GS_PARA_LEVEL", "4")

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure WP2 ranking noise (sigma_rank) and validate CRN."
    )
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to a HebbianEvolutionConfig YAML.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Eval repeats per (crn, stochastic) cell; the "
                             "reported sigma_rank is averaged over repeats.")
    parser.add_argument("--genome", type=str, default="zero",
                        choices=["zero", "random"],
                        help="'zero' = zero Hebbian rules (identical base "
                             "controllers; isolates env noise). 'random' = one "
                             "random non-zero rule set replicated across slots.")
    args, remaining = parser.parse_known_args()

    import numpy as np
    import torch

    from WP2.config import HebbianEvolutionConfig
    from WP2.evolve_cma import HebbianCMAES
    from WP2.evaluate import evaluate_population_multi_urdf
    from WP2.utils import seed_everything
    from WP2.run import _configure_cache_root

    _configure_cache_root()

    cfg = HebbianEvolutionConfig.from_yaml(args.cfg)
    cfg.apply_cli_overrides(remaining)

    # Infer last-layer dims from the checkpoint so the genome length is correct.
    from WP2.frozen_actor import last_actor_linear_key
    _ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    _sd = _ckpt.get("model_state_dict", _ckpt) if isinstance(_ckpt, dict) else _ckpt
    _last_key = last_actor_linear_key(_sd)
    if _last_key is not None:
        cfg.hebbian.num_actions = _sd[_last_key].shape[0]
        cfg.hebbian.hidden_dim = _sd[_last_key].shape[1]
    del _ckpt, _sd

    seed_everything(cfg.seed)

    runner = HebbianCMAES(cfg)
    runner._build_env_once()
    if runner._env is None:
        print("[sigma_rank] ERROR: eval env failed to build.")
        sys.exit(1)

    P = cfg.cmaes.population_size if cfg.cmaes.population_size > 0 else \
        int(4 + 3 * np.log(runner.n_genes))
    E = runner._env.E
    F = E // P
    if F < 1:
        print(f"[sigma_rank] ERROR: env has E={E} envs/drone but P={P} "
              f"individuals (F={F}). Grow num_eval_envs or lower popsize.")
        runner._cleanup_env()
        sys.exit(1)

    # One genome, replicated across all P slots -> identical controllers, so the
    # spread of the P fitnesses is pure evaluation (ranking) noise.
    if args.genome == "zero":
        base = np.full(runner.n_genes, 0.5)
    else:
        base = np.random.uniform(0.0, 1.0, runner.n_genes)
    solutions = [base.copy() for _ in range(P)]

    # Set the shared forest assignment up-front so refresh_forests() between
    # repeats draws fresh scenarios while keeping the slot->forest mapping.
    fixed = torch.arange(F, device=cfg.device, dtype=torch.long).repeat(P)
    runner._env._fixed_forest_ids = fixed
    runner._env._eval_speed_grid = torch.linspace(
        float(cfg.evaluation.vmin), float(cfg.evaluation.vmax), F,
        device=cfg.device, dtype=torch.float32,
    ).repeat(P)

    print(f"\n[sigma_rank] genome={args.genome}  P={P} individuals  "
          f"F={F} forests/(urdf,ind)  N={len(runner._urdf_paths)} URDFs  "
          f"E={E} envs/drone")
    print(f"[sigma_rank] All P slots run the SAME controller; std across the P "
          f"per-individual fitnesses = sigma_rank.\n")

    rows = []
    for crn in (False, True):
        for stoch in (True, False):
            cfg.evaluation.crn = crn
            cfg.evaluation.stochastic = stoch
            runner._env._crn_enabled = crn

            stds, means = [], []
            for _ in range(max(1, args.repeats)):
                runner._env.refresh_forests()  # fresh scenario draw, same mapping
                fitnesses, _ = evaluate_population_multi_urdf(
                    solutions, cfg, runner._model_and_layer, runner._wp1_cfg,
                    urdf_paths=runner._urdf_paths,
                    existing_env=(runner._env, runner._env_urdf_path),
                    verbose=False,
                )
                stds.append(float(np.std(fitnesses)))
                means.append(float(np.mean(fitnesses)))
            rows.append((crn, stoch, float(np.mean(stds)), float(np.mean(means))))
            print(f"  crn={'ON ' if crn else 'OFF'}  stochastic={'ON ' if stoch else 'OFF'}  "
                  f"-> sigma_rank = {np.mean(stds):8.3f}   (mean fitness {np.mean(means):8.2f})")

    runner._cleanup_env()

    # Interpretation
    d = {(c, s): v for c, s, v in [(r[0], r[1], r[2]) for r in rows]}
    print("\n[sigma_rank] ---- interpretation ----")
    print(f"  full sigma_ind          (crn OFF, stoch ON ) = {d[(False, True)]:.3f}")
    print(f"  action-sampling residual(crn ON , stoch ON ) = {d[(True, True)]:.3f}")
    print(f"  aero-force residual     (crn ON , stoch OFF) = {d[(True, False)]:.3f}  "
          f"<- in-kernel Taichi noise (not CRN-shareable)")
    print(f"  DR-only (no sampling)   (crn OFF, stoch OFF) = {d[(False, False)]:.3f}")
    base_noise = d[(False, True)]
    crn_noise = d[(True, True)]
    if base_noise > 1e-9:
        print(f"\n  CRN cuts ranking noise by {100.0 * (1.0 - crn_noise / base_noise):.0f}% "
              f"({base_noise:.2f} -> {crn_noise:.2f}) with action sampling still on.")
    print("  (crn ON, stoch OFF) near 0 ⇒ torch-side CRN is complete; the small "
          "residual is the\n  in-kernel aero force noise. If it is large, consider "
          "sharing the policy action noise too.")


if __name__ == "__main__":
    main()
