"""
WP2 main entry point — Hebbian rules evolution with CMA-ES.
===========================================================

Usage
-----
.. code-block:: bash

    # From a YAML config
    python -m WP2.run --cfg src/WP2/configs/cma_es_rules_only.yaml

    # With CLI overrides
    python -m WP2.run --cfg src/WP2/configs/cma_es_rules_only.yaml \\
        --cfg.evolution.num_generations 100 \\
        --cfg.hebbian.eta 0.0001

    # Resume from a previous run
    python -m WP2.run --resume logs/runs_hebbian/2026-04-15_10-00-00_cma_es_rules \\
        --from-gen 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Optimize Genesis scene compilation parallelization
os.environ["GS_PARA_LEVEL"] = "4"

# Ensure src/ is on the import path
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def _configure_cache_root() -> Path:
    """Force Taichi/genesis cache into a writable location."""
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
        description="WP2: Hebbian Plasticity Evolution with CMA-ES"
    )
    parser.add_argument(
        "--cfg", type=str, default=None,
        help="Path to a HebbianEvolutionConfig YAML file.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to an existing run directory to resume from.",
    )
    parser.add_argument(
        "--from-gen", type=int, default=None,
        help="Generation number to resume from (requires --resume).",
    )
    parser.add_argument(
        "-v", "--vis", action="store_true",
        help="Enable Genesis viewer (not recommended for evolution).",
    )
    parser.add_argument(
        "--baseline-ckpt", type=str, default=None,
        help=("Optional separate WP1 checkpoint used ONLY for the zero-rules "
              "baseline evaluation. If omitted, the baseline reuses the frozen "
              "actor checkpoint (--cfg.checkpoint_path)."),
    )
    parser.add_argument(
        "--baseline-ckpt-cfg", type=str, default=None,
        help=("Path to the WP1 config YAML matching --baseline-ckpt. Required "
              "when --baseline-ckpt is set."),
    )
    parser.add_argument(
        "--specialist", type=str, default=None,
        help=("Optional second baseline controller (a 'specialist'): a WP1 "
              "checkpoint evaluated with zero Hebbian rules on the SAME "
              "forests/speeds/URDFs as the population, every generation "
              "(cadence: evaluation.specialist_every). Passing this flag "
              "auto-enables evaluation.run_specialist=True."),
    )
    parser.add_argument(
        "--specialist-cfg", type=str, default=None,
        help=("Path to the WP1 config YAML matching --specialist. Required "
              "when --specialist is set."),
    )

    args, remaining = parser.parse_known_args()

    # --- Load or create config ---
    from WP2.config import HebbianEvolutionConfig

    if args.resume:
        resume_dir = Path(args.resume)
        cfg_path = resume_dir / "reproducibility" / "config.yaml"
        if not cfg_path.is_file():
            print(f"[ERROR] Cannot find config at {cfg_path}")
            sys.exit(1)
        cfg = HebbianEvolutionConfig.from_yaml(cfg_path)
        cfg.base_dir = str(resume_dir.parent)
        print(f"[run] Resuming from {resume_dir}")
    elif args.cfg:
        cfg = HebbianEvolutionConfig.from_yaml(args.cfg)
    else:
        cfg = HebbianEvolutionConfig()

    cfg.apply_cli_overrides(remaining)

    # CLI flags take precedence over YAML for the baseline checkpoint pair.
    if args.baseline_ckpt is not None:
        cfg.baseline_checkpoint_path = args.baseline_ckpt
    if args.baseline_ckpt_cfg is not None:
        cfg.baseline_checkpoint_config_path = args.baseline_ckpt_cfg

    # Specialist: passing --specialist auto-enables run_specialist=True so the
    # CLI alone is sufficient ("just give me a path"), while YAML configs can
    # still drive it independently.
    if args.specialist is not None:
        cfg.specialist_checkpoint_path = args.specialist
        cfg.evaluation.run_specialist = True
    if args.specialist_cfg is not None:
        cfg.specialist_checkpoint_config_path = args.specialist_cfg

    # --- Validation ---
    if not cfg.checkpoint_path or not Path(cfg.checkpoint_path).is_file():
        print(f"[ERROR] checkpoint_path not set or not found: {cfg.checkpoint_path}")
        print("  Set it in the YAML config or via --cfg.checkpoint_path <path>")
        sys.exit(1)

    if not cfg.checkpoint_config_path or not Path(cfg.checkpoint_config_path).is_file():
        print(f"[ERROR] checkpoint_config_path not set or not found: {cfg.checkpoint_config_path}")
        print("  Set it in the YAML config or via --cfg.checkpoint_config_path <path>")
        sys.exit(1)

    # If a baseline-specific checkpoint is requested, both path + config must exist.
    if cfg.baseline_checkpoint_path or cfg.baseline_checkpoint_config_path:
        if not cfg.baseline_checkpoint_path or not Path(cfg.baseline_checkpoint_path).is_file():
            print(f"[ERROR] baseline_checkpoint_path not set or not found: {cfg.baseline_checkpoint_path}")
            print("  Provide both --baseline-ckpt and --baseline-ckpt-cfg, or neither.")
            sys.exit(1)
        if not cfg.baseline_checkpoint_config_path or not Path(cfg.baseline_checkpoint_config_path).is_file():
            print(f"[ERROR] baseline_checkpoint_config_path not set or not found: {cfg.baseline_checkpoint_config_path}")
            print("  Provide both --baseline-ckpt and --baseline-ckpt-cfg, or neither.")
            sys.exit(1)

    # Same rule for the specialist pair.
    if (
        cfg.specialist_checkpoint_path
        or cfg.specialist_checkpoint_config_path
        or cfg.evaluation.run_specialist
    ):
        if not cfg.specialist_checkpoint_path or not Path(cfg.specialist_checkpoint_path).is_file():
            print(f"[ERROR] specialist_checkpoint_path not set or not found: {cfg.specialist_checkpoint_path}")
            print("  Provide both --specialist and --specialist-cfg, or neither.")
            sys.exit(1)
        if not cfg.specialist_checkpoint_config_path or not Path(cfg.specialist_checkpoint_config_path).is_file():
            print(f"[ERROR] specialist_checkpoint_config_path not set or not found: {cfg.specialist_checkpoint_config_path}")
            print("  Provide both --specialist and --specialist-cfg, or neither.")
            sys.exit(1)

    if cfg.total_genome_dim() == 0:
        print("[ERROR] Total genome dimension is 0. Enable hebbian in config.")
        sys.exit(1)

    # --- Setup ---
    _configure_cache_root()

    from WP2.utils import seed_everything
    seed_everything(cfg.seed)

    # Infer last-layer dims from checkpoint so genome size is correct
    import torch as _torch
    from WP2.frozen_actor import last_actor_linear_key
    _ckpt = _torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    _sd = _ckpt.get("model_state_dict", _ckpt) if isinstance(_ckpt, dict) else _ckpt
    _last_key = last_actor_linear_key(_sd)
    if _last_key is not None:
        cfg.hebbian.num_actions = _sd[_last_key].shape[0]
        cfg.hebbian.hidden_dim = _sd[_last_key].shape[1]
    del _ckpt, _sd

    # --- Print summary ---
    n_pop_cfg = cfg.cmaes.population_size
    import numpy as _np
    expected_pop = (
        n_pop_cfg if n_pop_cfg > 0
        else int(4 + 3 * _np.log(cfg.total_genome_dim()))
    )
    catalog_info = cfg.catalog.path if cfg.catalog.path else "default URDF"

    print("\n" + "=" * 70)
    print("  WP2: Hebbian Rules Evolution — CMA-ES")
    print("=" * 70)
    print(f"  Experiment:    {cfg.exp_name}")
    print(f"  Checkpoint:    {cfg.checkpoint_path}")
    print(f"  WP1 Config:    {cfg.checkpoint_config_path}")
    if cfg.baseline_checkpoint_path:
        print(f"  Baseline ckpt: {cfg.baseline_checkpoint_path}")
        print(f"  Baseline cfg:  {cfg.baseline_checkpoint_config_path}")
    if cfg.specialist_checkpoint_path:
        print(f"  Specialist:    {cfg.specialist_checkpoint_path}")
        print(f"  Specialist cfg:{cfg.specialist_checkpoint_config_path}")
    print(f"  Seed:          {cfg.seed}")
    print(f"  Device:        {cfg.device}")
    print(f"  Genome dim:    {cfg.total_genome_dim()}")
    print(f"  Catalog:       {catalog_info}")
    print(f"  CMA sigma0:    {cfg.cmaes.sigma0}")
    print(f"  CMA popsize:   {'auto ≈ ' + str(expected_pop) if n_pop_cfg == 0 else n_pop_cfg}")
    print(f"  Generations:   {cfg.evolution.num_generations}")
    print(f"  Eval envs:     {cfg.evaluation.num_eval_envs}")
    print("=" * 70 + "\n")

    # --- Launch CMA-ES evolution ---
    from WP2.evolve_cma import HebbianCMAES

    runner = HebbianCMAES(cfg)

    resume_gen = None
    if args.resume and args.from_gen is not None:
        runner.run_dir = Path(args.resume)
        runner.gen_dir = runner.run_dir / "generations"
        runner.results_dir = runner.run_dir / "results"
        resume_gen = args.from_gen

    runner.run(resume_from_gen=resume_gen)
    print(f"\n[run] All done. Results in: {runner.run_dir}")

    try:
        from WP2.plot_metrics import plot_metrics, plot_validation
        plot_metrics(runner.run_dir)
        plot_validation(runner.run_dir)
    except Exception as exc:  # never crash the run over a plotting failure
        print(f"[run] Warning: metrics plot failed — {exc}")

    try:
        from WP2.plot_cma import analyze_run
        analyze_run(runner.run_dir)
    except Exception as exc:
        print(f"[run] Warning: CMA plots failed — {exc}")


if __name__ == "__main__":
    main()
