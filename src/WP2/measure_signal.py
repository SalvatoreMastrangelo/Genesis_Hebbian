"""
measure_signal — signal-vs-noise check for WP2 ranking.
=======================================================

σ_rank tells us the NOISE floor (spread of identical controllers). This script
adds the missing half: the genome SIGNAL — the fitness gap between two genuinely
different rule-sets — measured on the SAME forests so the comparison is clean.

Design (co-located, common random numbers):
  population of P individuals = first P/2 are ZERO rules (base controller),
  last P/2 are an EVOLVED rule-set loaded from genome.npy. All P fly the same F
  forests (CRN), so:
    signal      = mean(evolved) - mean(zero)        # forest variance cancels
    sigma_rank  = std across the P/2 replicas        # the chaos+aero+DR floor
    SNR         = |signal| / sigma_rank              # >1 ⇒ CMA can rank them

Swept over crn{OFF,ON} × stochastic{ON,OFF}, repeats averaged.

IMPORTANT: decode the evolved genome with the SAME plasticity settings it was
evolved under (eta, decay, use_oja) or its online behaviour changes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("GS_PARA_LEVEL", "4")
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--genome", required=True, help="path to evolved genome.npy")
    ap.add_argument("--repeats", type=int, default=3)
    args, remaining = ap.parse_known_args()

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

    # Faithful decode settings for the evolved genome (must match its run).
    cfg.hebbian.eta = 0.005
    cfg.hebbian.decay = 0.05
    cfg.hebbian.use_oja_coefficient = False
    cfg.hebbian.evolve_eta = False
    cfg.hebbian.evolve_decay = False

    from WP2.frozen_actor import last_actor_linear_key
    ck = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    last_key = last_actor_linear_key(sd)
    if last_key is not None:
        cfg.hebbian.num_actions = sd[last_key].shape[0]
        cfg.hebbian.hidden_dim = sd[last_key].shape[1]
    del ck, sd
    seed_everything(cfg.seed)

    runner = HebbianCMAES(cfg)
    n = runner.n_genes
    evolved = np.load(args.genome).astype(float).reshape(-1)
    if evolved.shape[0] != n:
        print(f"[signal] ERROR: genome dim {evolved.shape[0]} != expected {n} "
              f"(num_actions={cfg.hebbian.num_actions}, hidden_dim={cfg.hebbian.hidden_dim})")
        sys.exit(1)
    zero = np.full(n, 0.5)

    runner._build_env_once()
    if runner._env is None:
        print("[signal] ERROR: env build failed"); sys.exit(1)
    env = runner._env

    P = cfg.cmaes.population_size
    half = P // 2
    F = env.E // P
    solutions = [zero.copy() for _ in range(half)] + [evolved.copy() for _ in range(half)]

    fixed = torch.arange(F, device=cfg.device, dtype=torch.long).repeat(P)
    env._fixed_forest_ids = fixed
    env._eval_speed_grid = torch.linspace(
        float(cfg.evaluation.vmin), float(cfg.evaluation.vmax), F,
        device=cfg.device, dtype=torch.float32).repeat(P)

    print(f"\n[signal] P={P} ({half} zero + {half} evolved)  F={F}  N={len(runner._urdf_paths)} URDFs  "
          f"rollouts/indiv={len(runner._urdf_paths) * F}")
    print(f"[signal] eta={cfg.hebbian.eta} decay={cfg.hebbian.decay} use_oja={cfg.hebbian.use_oja_coefficient}\n")
    hdr = f"{'crn':>4} {'stoch':>6} | {'mean_zero':>10} {'mean_evolved':>13} | {'signal Δ':>9} {'σ_rank':>7} {'SNR=Δ/σ':>8}"
    print(hdr); print("-" * len(hdr))

    rows = []
    for crn in (False, True):
        for stoch in (True, False):
            cfg.evaluation.crn = crn
            cfg.evaluation.stochastic = stoch
            env._crn_enabled = crn
            mz, me, sz, se, sig = [], [], [], [], []
            for _ in range(max(1, args.repeats)):
                env.refresh_forests()
                fit, _ = evaluate_population_multi_urdf(
                    solutions, cfg, runner._model_and_layer, runner._wp1_cfg,
                    urdf_paths=runner._urdf_paths,
                    existing_env=(runner._env, runner._env_urdf_path), verbose=False)
                zf, ef = np.asarray(fit[:half]), np.asarray(fit[half:])
                mz.append(zf.mean()); me.append(ef.mean())
                sz.append(zf.std()); se.append(ef.std()); sig.append(ef.mean() - zf.mean())
            mz, me = float(np.mean(mz)), float(np.mean(me))
            sigma_rank = float((np.mean(sz) + np.mean(se)) / 2)
            signal = float(np.mean(sig))
            snr = abs(signal) / sigma_rank if sigma_rank > 1e-9 else float("inf")
            rows.append((crn, stoch, mz, me, signal, sigma_rank, snr))
            print(f"{'ON' if crn else 'OFF':>4} {'ON' if stoch else 'OFF':>6} | "
                  f"{mz:10.2f} {me:13.2f} | {signal:9.2f} {sigma_rank:7.2f} {snr:8.2f}", flush=True)

    runner._cleanup_env()
    print("\n[signal] SNR = |evolved − zero| / per-individual σ_rank. SNR≫1 ⇒ the genome")
    print("[signal] effect clears the noise floor, so CMA-ES can rank reliably at this scale.")
    print("[signal] Compare crn ON vs OFF rows to see how much CRN sharpens the ranking.")


if __name__ == "__main__":
    main()
