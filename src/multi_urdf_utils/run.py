"""
Multi-URDF Utils main entry point — launch multi-drone benchmark.
==================================================================

Usage
-----
.. code-block:: bash

    python -m multi_urdf_utils.run --cfg src/multi_urdf_utils/configs/benchmark.yaml

    # With CLI overrides
    python -m multi_urdf_utils.run --cfg src/multi_urdf_utils/configs/benchmark.yaml \\
        --cfg.benchmark.N 8 --cfg.benchmark.S 5 --cfg.benchmark.E 128
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
        description="WP2.5: Multi-Drone-Per-Scene Benchmark"
    )
    parser.add_argument(
        "--cfg", type=str, default=None,
        help="Path to a BenchmarkConfig YAML file.",
    )

    args, remaining = parser.parse_known_args()

    from multi_urdf_utils.config import BenchmarkConfig

    if args.cfg:
        cfg = BenchmarkConfig.from_yaml(args.cfg)
    else:
        cfg = BenchmarkConfig()

    cfg.apply_cli_overrides(remaining)

    # Validation
    if not cfg.checkpoint.model_path or not Path(cfg.checkpoint.model_path).is_file():
        print(f"[ERROR] checkpoint.model_path not set or not found: {cfg.checkpoint.model_path}")
        sys.exit(1)
    if not cfg.checkpoint.config_path or not Path(cfg.checkpoint.config_path).is_file():
        print(f"[ERROR] checkpoint.config_path not set or not found: {cfg.checkpoint.config_path}")
        sys.exit(1)

    _configure_cache_root()

    # Seed
    from WP2.utils import seed_everything
    seed_everything(cfg.benchmark.seed)

    # Create output directory
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg.base_dir) / f"{ts}_{cfg.exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    cfg.to_yaml(run_dir / "config.yaml")

    # Run benchmark
    from multi_urdf_utils.benchmark import run_benchmark, save_results, save_table, print_results

    results = run_benchmark(cfg)
    csv_path = save_results(results, run_dir)
    txt_path = save_table(results, run_dir)
    print_results(results)

    print(f"[run] Results saved to: {csv_path}")
    print(f"[run] Table saved to:   {txt_path}")
    print(f"[run] Run directory: {run_dir}")


if __name__ == "__main__":
    main()
