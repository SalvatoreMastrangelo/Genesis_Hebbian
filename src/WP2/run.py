"""
WP2 main entry point — launch Hebbian + morphology co-optimisation.
===================================================================

Usage
-----
.. code-block:: bash

    # From a YAML config
    python -m WP2.run --cfg src/WP2/configs/full_codesing.yaml

    # With CLI overrides
    python -m WP2.run --cfg src/WP2/configs/full_codesing.yaml \\
        --cfg.evolution.population_size 80 \\
        --cfg.hebbian.eta 0.005

    # Resume from a previous run
    python -m WP2.run --resume logs/runs_hebbian/2026-03-12_14-30-00_hebbian_codesing \\
        --from-gen 15
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
        description="WP2: Hebbian Plasticity + Evolutionary Co-Optimisation"
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

    args, remaining = parser.parse_known_args()

    # --- Load or create config ---
    from WP2.config import HebbianEvolutionConfig

    if args.resume:
        # Load config from the existing run
        resume_dir = Path(args.resume)
        cfg_path = resume_dir / "reproducibility" / "config.yaml"
        if not cfg_path.is_file():
            print(f"[ERROR] Cannot find config at {cfg_path}")
            sys.exit(1)
        cfg = HebbianEvolutionConfig.from_yaml(cfg_path)
        # Override base_dir to point to the parent of the run dir
        cfg.base_dir = str(resume_dir.parent)
        print(f"[run] Resuming from {resume_dir}")
    elif args.cfg:
        cfg = HebbianEvolutionConfig.from_yaml(args.cfg)
    else:
        cfg = HebbianEvolutionConfig()

    # Apply CLI overrides from remaining args
    cfg.apply_cli_overrides(remaining)

    # --- Validation ---
    if not cfg.checkpoint_path or not Path(cfg.checkpoint_path).is_file():
        print(f"[ERROR] checkpoint_path not set or not found: {cfg.checkpoint_path}")
        print("  Set it in the YAML config or via --cfg.checkpoint_path <path>")
        sys.exit(1)

    if not cfg.checkpoint_config_path or not Path(cfg.checkpoint_config_path).is_file():
        print(f"[ERROR] checkpoint_config_path not set or not found: {cfg.checkpoint_config_path}")
        print("  Set it in the YAML config or via --cfg.checkpoint_config_path <path>")
        sys.exit(1)

    if cfg.total_genome_dim() == 0:
        print("[ERROR] Total genome dimension is 0. Enable hebbian and/or morphology evolution.")
        sys.exit(1)

    # --- Setup ---
    _configure_cache_root()

    from WP2.utils import seed_everything
    seed_everything(cfg.seed)

    # Infer last-layer dims from checkpoint so genome size is correct
    import torch as _torch
    _ckpt = _torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    _sd = _ckpt.get("model_state_dict", _ckpt) if isinstance(_ckpt, dict) else _ckpt
    if "actor.4.weight" in _sd:
        cfg.hebbian.num_actions = _sd["actor.4.weight"].shape[0]
        cfg.hebbian.hidden_dim = _sd["actor.4.weight"].shape[1]
    del _ckpt, _sd

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("  WP2: Hebbian Plasticity + Evolutionary Co-Optimisation")
    print("=" * 70)
    print(f"  Experiment:    {cfg.exp_name}")
    print(f"  Checkpoint:    {cfg.checkpoint_path}")
    print(f"  WP1 Config:    {cfg.checkpoint_config_path}")
    print(f"  Seed:          {cfg.seed}")
    print(f"  Device:        {cfg.device}")
    print(f"  Genome dim:    {cfg.total_genome_dim()} "
          f"(Hebbian={cfg.hebbian_genome_dim()}, Morph={cfg.morphology_genome_dim()})")
    print(f"  Objectives:    {cfg.active_objective_names()}")
    print(f"  Population:    {cfg.evolution.population_size}")
    print(f"  Generations:   {cfg.evolution.num_generations}")
    print(f"  Crossover:     {'ON' if cfg.evolution.enable_crossover else 'OFF'}")
    print(f"  Hebbian:       {'ON' if cfg.hebbian.enabled else 'OFF'}")
    print(f"  Morphology:    {'EVOLVE' if cfg.morphology.evolve else 'FIXED'}")
    print(f"  Eval episodes: {cfg.evaluation.num_eval_episodes}")
    print(f"  Eval envs:     {cfg.evaluation.num_eval_envs}")
    print("=" * 70 + "\n")

    # --- Launch evolution ---
    from WP2.evolve import HebbianCodesignDEAP

    ga = HebbianCodesignDEAP(cfg)

    resume_gen = None
    if args.resume and args.from_gen is not None:
        # Point to the existing run directory
        ga.run_dir = Path(args.resume)
        ga.gen_dir = ga.run_dir / "generations"
        ga.results_dir = ga.run_dir / "results"
        ga.plots_dir = ga.run_dir / "plots"
        resume_gen = args.from_gen

    final_pop = ga.run(resume_from_gen=resume_gen)

    # --- Post-analysis ---
    try:
        from WP2.plotting import analyze_run
        analyze_run(str(ga.run_dir))
    except Exception as exc:
        print(f"[run] Plotting failed (non-fatal): {exc}")

    print(f"\n[run] All done. Results in: {ga.run_dir}")


if __name__ == "__main__":
    main()
