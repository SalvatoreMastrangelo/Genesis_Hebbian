"""
Plot per-generation mean ± std for the four evaluation metrics.

Usage
-----
    python -m WP2.plot_metrics <run_dir>

Also callable programmatically::

    from WP2.plot_metrics import plot_metrics
    plot_metrics(run_dir)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# (csv_column, display_label, lower_is_better)
_METRICS = [
    ("fitness",    "Fitness (reward)", False),
    ("velocity",   "Velocity [m/s]",   False),
    ("progress",   "Progress [m]",     False),
    ("crash_rate", "Crash Rate",       True),
]


def plot_metrics(run_dir: Path | str) -> None:
    """Read cma_population.csv and plot mean ± std per generation for all metrics."""
    run_dir = Path(run_dir)
    csv_path = run_dir / "results" / "cma_population.csv"

    if not csv_path.is_file():
        print(f"[plot_metrics] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    grouped = df.groupby("generation")
    means = grouped.mean(numeric_only=True)
    stds  = grouped.std(ddof=0, numeric_only=True)
    gens  = means.index.to_numpy()

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Metrics Evolution — mean ± std", fontsize=13)

    mean_colour = "#1f77b4"
    best_colour = "#d62728"

    for ax, (col, label, lower_is_better) in zip(axes.flat, _METRICS):
        mu  = means[col].to_numpy()
        sig = stds[col].to_numpy()

        # Per-generation best: min for lower-is-better, max otherwise
        best = grouped[col].min().to_numpy() if lower_is_better else grouped[col].max().to_numpy()

        # Crash rate is a rate — clip mean and best to [0, 1]
        if col == "crash_rate":
            mu   = np.clip(mu,   0.0, 1.0)
            best = np.clip(best, 0.0, 1.0)
            sig  = np.minimum(sig, 1.0 - mu)  # keep upper band from exceeding 1

        ax.plot(gens, mu,   color=mean_colour, linewidth=1.8, label="mean")
        ax.plot(gens, best, color=best_colour,  linewidth=1.4, label="best")
        ax.fill_between(
            gens, mu - sig, mu + sig,
            alpha=0.25, color=mean_colour, label="±1 std",
        )

        title = f"{label}  (↓ better)" if lower_is_better else label
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Generation")
        ax.set_ylabel(label)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    out = plots_dir / "metrics_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot_metrics] Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m WP2.plot_metrics <run_dir>")
        sys.exit(1)
    plot_metrics(sys.argv[1])
