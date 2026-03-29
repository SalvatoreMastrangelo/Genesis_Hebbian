"""SNE grid-search benchmark entry point.

Runs every (S, N, E) combination and records compilation and throughput
metrics to a CSV file.  Designed to be distributed across multiple SLURM
array jobs via ``--job-idx`` / ``--num-jobs``.

Usage
-----
.. code-block:: bash

    # Single-machine full grid
    python -m WP1.benchmark.grid_search --cfg src/WP1/benchmark/configs/sne_benchmark.yaml

    # SLURM array job (8 parallel jobs)
    python -m WP1.benchmark.grid_search \\
        --cfg src/WP1/benchmark/configs/sne_benchmark.yaml \\
        --job-idx 3 --num-jobs 8 \\
        --S-values 1 2 4 8 \\
        --N-values 1 2 4 \\
        --E-values 64 128 256 512
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set thread limits BEFORE importing heavy packages (numpy/torch)
# to avoid multiprocessing dispatcher conflicts
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_URDF = 384  # Skip runs where S*N > MAX_URDF


# ---------------------------------------------------------------------------
# CSV column definitions (fixed order)
# ---------------------------------------------------------------------------

CSV_COLUMNS: List[str] = [
    "status",
    "S",
    "N",
    "E",
    "total_envs",
    "wall_time_s",
    "compile_wall_time_s",
    "sim_time_total_s",
    "train_time_total_s",
    "compile_time_min_s",
    "compile_time_mean_s",
    "compile_time_max_s",
    "sim_time_per_iter_min_s",
    "sim_time_per_iter_mean_s",
    "sim_time_per_iter_max_s",
    "steps_per_sec_avg",
    "steps_per_sec_per_iter",
    "iter_times_s",
    "ram_per_scene_mean_mb",
    "ram_per_scene_max_mb",
    "vram_allocated_per_scene_mean_mb",
    "vram_allocated_per_scene_max_mb",
    "vram_allocated_total_mb",
    "vram_reserved_per_scene_mean_mb",
    "vram_reserved_per_scene_max_mb",
    "error",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _build_grid(
    S_values: List[int],
    N_values: List[int],
    E_values: List[int],
) -> List[Tuple[int, int, int]]:
    """Return all (S, N, E) combinations sorted by (S, N, E)."""
    return sorted(itertools.product(S_values, N_values, E_values))


def _filter_by_max_urdf(
    grid: List[Tuple[int, int, int]],
    max_urdf: int,
) -> List[Tuple[int, int, int]]:
    """Remove configs where S*N > max_urdf."""
    filtered = [(S, N, E) for S, N, E in grid if S * N <= max_urdf]
    skipped = len(grid) - len(filtered)
    if skipped > 0:
        print(f"[grid_search] Skipped {skipped} configs with S*N > {max_urdf}")
    return filtered


def _this_job_subset(
    grid: List[Tuple[int, int, int]],
    job_idx: int,
    num_jobs: int,
) -> List[Tuple[int, int, int]]:
    return [combo for i, combo in enumerate(grid) if i % num_jobs == job_idx]


def _timeout_result(S: int, N: int, E: int) -> Dict[str, Any]:
    return {
        "status": "TIMEOUT",
        "S": S,
        "N": N,
        "E": E,
        "total_envs": S * N * E,
        **{k: None for k in CSV_COLUMNS if k not in ("status", "S", "N", "E", "total_envs")},
        "error": f"Subprocess timed out after configured timeout_s",
    }


def _write_csv_row(writer: csv.DictWriter, result: Dict[str, Any]) -> None:
    """Write one result row, filling missing columns with empty string."""
    row = {col: result.get(col, "") for col in CSV_COLUMNS}
    # Normalise None → empty string for CSV
    for k, v in row.items():
        if v is None:
            row[k] = ""
    writer.writerow(row)


def _build_catalog_if_needed(
    cfg,
    job_subset: List[Tuple[int, int, int]],
    output_dir: Path,
) -> Optional[str]:
    """Build a URDF catalog in the main process if any N>1 config is in the job.

    Returns the catalog directory path as a string, or ``None`` if not needed.
    """
    # Check if any config in this job requires a catalog
    needs_catalog = any(N > 1 for _, N, _ in job_subset)

    # If a catalog_dir is already specified in cfg, use it directly
    if cfg.catalog_dir is not None:
        print(f"[grid_search] Using existing catalog_dir: {cfg.catalog_dir}")
        return cfg.catalog_dir

    if not needs_catalog:
        print("[grid_search] No N>1 configs in this job subset — skipping catalog build.")
        return None

    # Build a fresh catalog
    max_s = max(S for S, N, E in job_subset)
    max_n = max(N for S, N, E in job_subset)
    n_urdf = max(cfg.n_urdf, max_s * max_n)

    catalog_dir = output_dir / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    print(f"[grid_search] Building URDF catalog: n={n_urdf}, seed={cfg.urdf_seed} → {catalog_dir}")
    try:
        from general_policy.catalog import build_catalog
        build_catalog(catalog_dir, n=n_urdf, seed=cfg.urdf_seed)
        print(f"[grid_search] Catalog built: {catalog_dir}")
    except Exception as e:
        print(f"[grid_search] WARNING: catalog build failed: {e}")
        print("[grid_search] N>1 configurations will fail unless a pre-built catalog is provided.")
        return None

    return str(catalog_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SNE grid-search benchmark for WP1 multi-scene training."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="src/WP1/benchmark/configs/sne_benchmark.yaml",
        help="Path to SNEBenchmarkConfig YAML file.",
    )
    parser.add_argument(
        "--job-idx",
        type=int,
        default=0,
        help="Index of this job in the array (0-based).",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="Total number of parallel jobs.",
    )
    parser.add_argument(
        "--S-values",
        type=int,
        nargs="+",
        default=None,
        help="Override S_values from YAML (space-separated integers).",
    )
    parser.add_argument(
        "--N-values",
        type=int,
        nargs="+",
        default=None,
        help="Override N_values from YAML (space-separated integers).",
    )
    parser.add_argument(
        "--E-values",
        type=int,
        nargs="+",
        default=None,
        help="Override E_values from YAML (space-separated integers).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help=(
            "Number of GPUs available. If > 1, scenes are distributed across GPUs "
            "(each scene worker gets CUDA_VISIBLE_DEVICES set to one GPU)."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load config and apply CLI overrides
    # ------------------------------------------------------------------
    from WP1.benchmark.config import SNEBenchmarkConfig
    from WP1.benchmark.benchmark_runner import run_sne_config

    cfg = SNEBenchmarkConfig.from_yaml(args.cfg)

    # Resolve base_cfg_path to absolute: try CWD first, then walk up from the
    # benchmark config YAML's location (handles Apptainer where CWD != repo root).
    base_p = Path(cfg.base_cfg_path)
    if not base_p.is_absolute():
        if base_p.exists():
            cfg.base_cfg_path = str(base_p.resolve())
        else:
            # Walk up from the benchmark config YAML directory to find repo root
            search_root = Path(args.cfg).resolve().parent
            for _ in range(6):
                candidate = search_root / cfg.base_cfg_path
                if candidate.exists():
                    cfg.base_cfg_path = str(candidate.resolve())
                    break
                search_root = search_root.parent
    print(f"[grid_search] base_cfg_path resolved to: {cfg.base_cfg_path}")

    if args.S_values is not None:
        cfg.S_values = args.S_values
    if args.N_values is not None:
        cfg.N_values = args.N_values
    if args.E_values is not None:
        cfg.E_values = args.E_values

    # Multi-GPU setup
    if args.num_gpus > 1:
        print(f"[grid_search] Multi-GPU mode: {args.num_gpus} GPUs (scenes will be distributed)")

    # ------------------------------------------------------------------
    # Build full grid, filter by MAX_URDF, and distribute to jobs
    # ------------------------------------------------------------------
    full_grid = _build_grid(cfg.S_values, cfg.N_values, cfg.E_values)
    full_grid = _filter_by_max_urdf(full_grid, MAX_URDF)
    job_subset = _this_job_subset(full_grid, args.job_idx, args.num_jobs)

    print(
        f"[grid_search] Job {args.job_idx}/{args.num_jobs}: "
        f"{len(job_subset)}/{len(full_grid)} configs assigned."
    )
    for combo in job_subset:
        print(f"  S={combo[0]}, N={combo[1]}, E={combo[2]}")

    if not job_subset:
        print("[grid_search] Nothing to do for this job index.")
        return

    # ------------------------------------------------------------------
    # Prepare output directory and CSV
    # ------------------------------------------------------------------
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"job_{args.job_idx}_results.csv"
    print(f"[grid_search] Results will be written to: {csv_path}")

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    csv_writer.writeheader()
    csv_file.flush()

    # ------------------------------------------------------------------
    # Set up environment for Genesis/CUDA (before catalog build)
    # ------------------------------------------------------------------
    os.environ["GS_PARA_LEVEL"] = "4"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Configure Taichi/Genesis cache root
    from winged_drone_train.train import _configure_cache_root
    _configure_cache_root()

    # ------------------------------------------------------------------
    # Pre-build URDF catalog in main process (if needed)
    # ------------------------------------------------------------------
    catalog_dir_override = _build_catalog_if_needed(cfg, job_subset, output_dir)

    # ------------------------------------------------------------------
    # Run each (S, N, E) configuration sequentially
    # ------------------------------------------------------------------
    spawn_ctx = mp.get_context("spawn")
    total_start = time.perf_counter()

    for combo_idx, (S, N, E) in enumerate(job_subset):
        print(
            f"\n[grid_search] === Config {combo_idx + 1}/{len(job_subset)}: "
            f"S={S}, N={N}, E={E} (total_envs={S * N * E}) ==="
        )

        # For N==1 we don't need a catalog; pass None so the worker uses
        # the default URDF path.
        this_catalog = catalog_dir_override if N > 1 else None

        result_q = spawn_ctx.Queue()
        proc = spawn_ctx.Process(
            target=run_sne_config,
            kwargs=dict(
                S=S,
                N=N,
                E=E,
                cfg_path=cfg.base_cfg_path,
                num_iterations=cfg.num_iterations,
                catalog_dir_override=this_catalog,
                result_queue=result_q,
                num_gpus=args.num_gpus,
            ),
            daemon=False,
        )

        proc_start = time.perf_counter()
        proc.start()
        print(f"[grid_search] Subprocess PID {proc.pid} started.")

        # Wait for result with timeout
        result: Optional[Dict[str, Any]] = None
        timed_out = False
        try:
            result = result_q.get(timeout=cfg.subprocess_timeout_s)
        except Exception:
            timed_out = True

        proc.join(timeout=30)
        if proc.is_alive():
            print(f"[grid_search] Subprocess still alive after join — terminating.")
            proc.terminate()
            proc.join(timeout=10)

        elapsed = time.perf_counter() - proc_start

        if timed_out or result is None:
            print(
                f"[grid_search] TIMEOUT after {elapsed:.1f}s "
                f"(limit={cfg.subprocess_timeout_s}s)"
            )
            result = _timeout_result(S, N, E)
        elif proc.exitcode not in (0, None) and result.get("status") == "OK":
            # Subprocess crashed after putting result — trust the result dict
            print(f"[grid_search] WARNING: subprocess exitcode={proc.exitcode}")

        status = result.get("status", "ERROR")
        print(
            f"[grid_search] Config S={S},N={N},E={E} finished: "
            f"status={status}, elapsed={elapsed:.1f}s"
        )
        if status == "OK":
            sps = result.get("steps_per_sec_avg")
            if sps is not None:
                print(f"[grid_search]   steps/s avg = {sps:.0f}")

        _write_csv_row(csv_writer, result)
        csv_file.flush()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - total_start
    print(
        f"\n[grid_search] Job {args.job_idx} finished. "
        f"Processed {len(job_subset)} configs in {total_elapsed:.1f}s. "
        f"Results: {csv_path}"
    )

    csv_file.close()

    # Print a quick summary table
    print("\n[grid_search] Summary:")
    print(f"{'S':>4} {'N':>4} {'E':>6} {'total_envs':>12} {'status':>8} {'steps/s':>10}")
    print("-" * 52)

    # Re-read CSV for summary
    import csv as _csv
    with open(csv_path, "r") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            sps = row.get("steps_per_sec_avg", "")
            try:
                sps_str = f"{float(sps):.0f}" if sps else "N/A"
            except ValueError:
                sps_str = "N/A"
            print(
                f"{row['S']:>4} {row['N']:>4} {row['E']:>6} "
                f"{row['total_envs']:>12} {row['status']:>8} {sps_str:>10}"
            )


if __name__ == "__main__":
    main()
