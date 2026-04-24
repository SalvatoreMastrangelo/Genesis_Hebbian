"""
WP2 outer-loop entry point — NSGA-II URDF co-evolution.
========================================================

Usage
-----
.. code-block:: bash

    # From a YAML config, picking WP1 controller + inner-loop template on the CLI
    python -m WP2_Outer_Loop.run \\
        --cfg src/WP2_Outer_Loop/configs/outer_loop_default.yaml \\
        --cfg.checkpoint_path        logs/runs/<RUN>/tb/model_1999.pt \\
        --cfg.checkpoint_config_path logs/runs/<RUN>/config.yaml \\
        --cfg.inner_cfg_template     src/WP2/configs/cma_es_rules_only.yaml

    # With loop-sizing overrides
    python -m WP2_Outer_Loop.run --cfg ... \\
        --cfg.outer_generations 30 \\
        --cfg.population_size 32 \\
        --cfg.num_eval_envs 512 \\
        --cfg.nsga2.sbx_eta 20
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
        description="WP2 Outer Loop — NSGA-II URDF co-evolution + inner CMA-ES"
    )
    parser.add_argument(
        "--cfg", type=str, default=None,
        help="Path to an OuterLoopConfig YAML file.",
    )
    args, remaining = parser.parse_known_args()

    from WP2_Outer_Loop.config import OuterLoopConfig

    if args.cfg:
        cfg = OuterLoopConfig.from_yaml(args.cfg)
    else:
        cfg = OuterLoopConfig()
    cfg.apply_cli_overrides(remaining)
    cfg.validate()

    # Validate inner template exists before building the env, which is slow.
    template_path = Path(cfg.inner_cfg_template)
    if not template_path.is_file():
        print(f"[ERROR] inner_cfg_template not found: {template_path}")
        sys.exit(1)

    _configure_cache_root()

    print(f"[WP2_Outer_Loop] Loaded config from {args.cfg or '<defaults>'}")
    print(f"[WP2_Outer_Loop] exp_name={cfg.exp_name}  base_dir={cfg.base_dir}")
    print(f"[WP2_Outer_Loop] outer_gens={cfg.outer_generations} "
          f"inner_gens={cfg.inner_generations}  pop={cfg.population_size}  "
          f"envs={cfg.num_eval_envs}  seed={cfg.seed}")

    from WP2_Outer_Loop.outer_loop import OuterLoop
    loop = OuterLoop(cfg)
    loop.run()


if __name__ == "__main__":
    main()
