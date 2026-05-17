"""
Reconstruct the ``best_individual/`` folder from per-generation checkpoints.

When a CMA-ES run is interrupted, ``_finalize()`` never runs and
``best_individual/`` is missing. All data needed to recover it is still
on disk:

* ``results/cma_population.csv``      — every metric for every individual
* ``generations/gen_XXX/solutions.npy`` — the genomes themselves
* ``reproducibility/config.yaml``     — config (needed to decode Hebbian rules)

This script picks the best individual per tracked metric over the
completed generations and writes the same files
(``genome.npy`` / ``hebbian_rules.yaml`` / ``fitness.yaml``) that
``_save_best_individual()`` would have produced.

Usage
-----
    python -m WP2.recover_best <run_dir> [--config PATH] [--overwrite]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from WP2.config import HebbianEvolutionConfig
from WP2.utils import decode_hebbian_genes


# (csv column, tracker folder name, higher_is_better)
_METRIC_SPECS: Tuple[Tuple[str, str, bool], ...] = (
    ("fitness",     "fitness",            True),
    ("progress",    "progress",           True),
    ("v_deviation", "velocity_deviation", False),
    ("cot",         "cost_of_transport",  False),
    ("crash_rate",  "crash_rate",         False),
)

# Mapping from CSV column → field name written into fitness.yaml,
# matching what _save_best_individual()/extra_fields uses.
_FIELD_NAME = {
    "fitness":     "fitness",
    "progress":    "progress",
    "v_deviation": "velocity_deviation",
    "cot":         "cost_of_transport",
    "crash_rate":  "crash_rate",
}


def _load_genome(run_dir: Path, gen: int, idx: int) -> np.ndarray:
    sol_path = run_dir / "generations" / f"gen_{gen:03d}" / "solutions.npy"
    if not sol_path.is_file():
        raise FileNotFoundError(
            f"Missing solutions for generation {gen}: {sol_path}"
        )
    solutions = np.load(sol_path)
    if idx >= solutions.shape[0]:
        raise IndexError(
            f"Individual idx={idx} out of range for gen {gen} "
            f"(population size {solutions.shape[0]})"
        )
    return solutions[idx].astype(np.float64, copy=True)


def _save_best_individual(
    subdir: Path,
    genome: np.ndarray,
    gen: int,
    idx: int,
    extra_fields: Dict,
    cfg: HebbianEvolutionConfig,
) -> None:
    """Mirror of HebbianCMAES._save_best_individual."""
    subdir.mkdir(parents=True, exist_ok=True)
    np.save(subdir / "genome.npy", genome)

    rules = decode_hebbian_genes(
        list(np.clip(genome, 0.0, 1.0)),
        cfg.hebbian,
        out_features=cfg.hebbian.num_actions,
        in_features=cfg.hebbian.hidden_dim,
    )
    rules_dict = {k: v.cpu().numpy().tolist() for k, v in rules.items()}
    with open(subdir / "hebbian_rules.yaml", "w") as f:
        yaml.dump(rules_dict, f, sort_keys=False)

    record = {"generation": int(gen), "individual_idx": int(idx), **extra_fields}
    with open(subdir / "fitness.yaml", "w") as f:
        yaml.dump(record, f, sort_keys=False)


def recover_best(
    run_dir: Path | str,
    config_path: Path | str | None = None,
    overwrite: bool = False,
) -> None:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory not found: {run_dir}")

    csv_path = run_dir / "results" / "cma_population.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    if config_path is None:
        config_path = run_dir / "reproducibility" / "config.yaml"
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")

    cfg = HebbianEvolutionConfig.from_yaml(config_path)
    df = pd.read_csv(csv_path)

    missing_cols = [c for c, _, _ in _METRIC_SPECS if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"CSV {csv_path} is missing required columns: {missing_cols}"
        )

    best_dir = run_dir / "best_individual"
    print(f"[recover_best] run_dir = {run_dir}")
    print(f"[recover_best] {len(df)} rows across "
          f"{df['generation'].nunique()} generations")

    # For every individual we'll also bake all five metrics into fitness.yaml,
    # so prefetch the full row for the winning index.
    for col, folder, higher_is_better in _METRIC_SPECS:
        arr = df[col].to_numpy()
        idx_row = int(np.nanargmax(arr) if higher_is_better else np.nanargmin(arr))
        row = df.iloc[idx_row]
        gen = int(row["generation"])
        ind = int(row["individual_idx"])
        value = float(row[col])

        subdir = best_dir / folder
        if subdir.exists() and not overwrite:
            print(f"[recover_best] {folder}: exists, skipping "
                  f"(use --overwrite to replace)")
            continue

        try:
            genome = _load_genome(run_dir, gen, ind)
        except (FileNotFoundError, IndexError) as e:
            print(f"[recover_best] {folder}: SKIPPED — {e}")
            continue

        extra_fields = {
            _FIELD_NAME[csv_col]: float(row[csv_col])
            for csv_col, _, _ in _METRIC_SPECS
        }
        _save_best_individual(subdir, genome, gen, ind, extra_fields, cfg)
        direction = "max" if higher_is_better else "min"
        print(f"[recover_best] {folder}: {direction}={value:.6g} "
              f"(gen {gen}, ind {ind}) → {subdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("run_dir", help="Path to a WP2 run directory")
    ap.add_argument("--config", default=None,
                    help="Path to config YAML "
                         "(default: <run_dir>/reproducibility/config.yaml)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace existing best_individual/<metric>/ folders")
    args = ap.parse_args()
    recover_best(args.run_dir, args.config, args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
