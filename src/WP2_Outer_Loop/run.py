"""
WP2 outer-loop entry point — persistent CMA-ES + NSGA-II URDF co-evolution.
===========================================================================

Runs ``NSGA2MorphCMAES`` (see ``nsga_cma.py``): one continuous inner CMA-ES
over Hebbian rules whose URDF population is evolved by NSGA-II every
``catalog.refresh_urdfs_every`` CMA generations, using per-URDF objectives
(fitness, cost of transport, …) harvested from the rollouts the CMA-ES
already runs.

The config is a SINGLE file: a plain WP2 ``HebbianEvolutionConfig`` YAML
(same layout as ``src/WP2/configs/cma_es_rules_only.yaml``) plus one
``outer:`` section with the NSGA-II knobs. The run behaves exactly like an
inner-loop multi-URDF run with periodic URDF refresh, except the refresh is
an NSGA-II update instead of random resampling:

* ``catalog.num_urdfs``            → NSGA-II population size N
* ``catalog.refresh_urdfs_every``  → CMA generations per phase
* ``evolution.num_generations``    → total CMA generations
* ``evaluation.num_eval_envs``     → total env slots (N × H × F)

The legacy nested implementation (fresh inner run per outer generation) is
preserved in ``legacy_run.py`` / ``outer_loop.py`` but deprecated.

Usage
-----
.. code-block:: bash

    PYTHONPATH=src python -m WP2_Outer_Loop.run \\
        --cfg src/WP2_Outer_Loop/configs/outer_nsga_default.yaml \\
        --cfg.checkpoint_path        <WP1 actor .pt> \\
        --cfg.checkpoint_config_path <WP1 config.yaml>

    # ONE override namespace — inner and outer fields alike:
    #   --cfg.cmaes.population_size 16  --cfg.catalog.num_urdfs 4
    #   --cfg.outer.n_elites 3          --cfg.outer.score_top_frac 0.25
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Optimise Genesis scene compilation parallelisation
os.environ["GS_PARA_LEVEL"] = "4"

# Ensure src/ is on the import path (match WP2/run.py behaviour).
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def _configure_cache_root() -> Path:
    """Force Taichi/genesis cache into a writable location. Matches WP2/run.py."""
    cache_root = (Path("logs") / ".cache" / "gstaichi").expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    for env_key in ("XDG_CACHE_HOME", "TI_CACHE_DIR", "TAICHI_CACHE_DIR", "GSTAICHI_CACHE_DIR"):
        os.environ[env_key] = str(cache_root)
    mpl_dir = cache_root / "mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    return cache_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WP2 Outer Loop — persistent CMA-ES + NSGA-II URDF co-evolution"
    )
    parser.add_argument(
        "--cfg", type=str, default=None,
        help="Path to a single OuterNSGA2Config YAML (inner-loop config + outer: section).",
    )
    args, _ = parser.parse_known_args()

    from WP2_Outer_Loop.config import OuterNSGA2Config

    if args.cfg:
        cfg = OuterNSGA2Config.from_yaml(args.cfg)
    else:
        cfg = OuterNSGA2Config()
    cfg.apply_cli_overrides()

    # --- Checkpoint sanity + Hebbian-dim inference (mirror WP2/run.py) ---
    if not cfg.checkpoint_path or not Path(cfg.checkpoint_path).is_file():
        print(f"[ERROR] checkpoint_path not set or not found: {cfg.checkpoint_path!r}")
        print("  Set it in the YAML or via --cfg.checkpoint_path.")
        sys.exit(1)
    if (not cfg.checkpoint_config_path
            or not Path(cfg.checkpoint_config_path).is_file()):
        print(f"[ERROR] checkpoint_config_path not set or not found: "
              f"{cfg.checkpoint_config_path!r}")
        sys.exit(1)

    import torch as _torch
    from WP2.frozen_actor import last_actor_linear_key
    _ckpt = _torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    _sd = _ckpt.get("model_state_dict", _ckpt) if isinstance(_ckpt, dict) else _ckpt
    _last_key = last_actor_linear_key(_sd)
    if _last_key is not None:
        cfg.hebbian.num_actions = _sd[_last_key].shape[0]
        cfg.hebbian.hidden_dim = _sd[_last_key].shape[1]
    del _ckpt, _sd

    # Validate AFTER dim inference (the env-budget check needs the effective
    # CMA population size, which can depend on the genome dimension).
    try:
        cfg.validate()
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    _configure_cache_root()

    from WP2.utils import seed_everything
    seed_everything(cfg.seed)

    N = int(cfg.catalog.num_urdfs)
    H = cfg.cma_population_size()
    F = (int(cfg.evaluation.num_eval_envs) // N) // H
    phase_len = int(cfg.catalog.refresh_urdfs_every)
    n_phases = max(1, int(cfg.evolution.num_generations) // phase_len)

    print("\n" + "=" * 70)
    print("  WP2 Outer Loop — persistent CMA-ES + NSGA-II URDF co-evolution")
    print("=" * 70)
    print(f"  Experiment:    {cfg.exp_name}")
    print(f"  Checkpoint:    {cfg.checkpoint_path}")
    print(f"  Genome dim:    {cfg.hebbian_genome_dim()} "
          f"(last layer {cfg.hebbian.num_actions}×{cfg.hebbian.hidden_dim})")
    print(f"  URDF pop (N):  {N}   elites: {cfg.outer.n_elites}")
    print(f"  Generations:   {cfg.evolution.num_generations} CMA gens "
          f"= {n_phases} phases × {phase_len}")
    print(f"  CMA popsize:   {H}   sigma0: {cfg.cmaes.sigma0}   "
          f"sigma_reinflate: {cfg.cmaes.sigma_reinflate}")
    print(f"  Env slots:     {cfg.evaluation.num_eval_envs} total → F={F} "
          f"forests/(URDF, individual)")
    print(f"  Objectives:    "
          f"{[(o.name, o.direction) for o in cfg.outer.objectives]}")
    print("=" * 70 + "\n")

    from WP2_Outer_Loop.nsga_cma import NSGA2MorphCMAES
    runner = NSGA2MorphCMAES(cfg)
    runner.run()
    print(f"\n[run] All done. Results in: {runner.run_dir}")

    # Inner-loop-style plots (population/summary CSVs are the same format).
    try:
        from WP2.plot_metrics import plot_metrics
        plot_metrics(runner.run_dir)
    except Exception as exc:
        print(f"[run] Warning: metrics plot failed — {exc}")
    try:
        from WP2.plot_metrics import plot_validation
        plot_validation(runner.run_dir)
    except Exception as exc:
        print(f"[run] Warning: validation plot failed — {exc}")
    try:
        from WP2.plot_cma import analyze_run
        analyze_run(runner.run_dir)
    except Exception as exc:
        print(f"[run] Warning: CMA plots failed — {exc}")

    # Outer-loop plots: Pareto front evolution + per-URDF metric curves.
    try:
        from WP2_Outer_Loop.pareto_plots import plot_outer_run
        plot_outer_run(runner.run_dir)
    except Exception as exc:
        print(f"[run] Warning: outer-loop plots failed — {exc}")


if __name__ == "__main__":
    main()
