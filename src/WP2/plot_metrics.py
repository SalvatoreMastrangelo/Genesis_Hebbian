"""
Plot per-generation spread for the evaluation metrics.

Usage
-----
    python -m WP2.plot_metrics <run_dir> [--percentile]

Also callable programmatically::

    from WP2.plot_metrics import plot_metrics
    plot_metrics(run_dir, use_percentile=True)
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
    ("fitness",     "Fitness (reward)",      False),
    ("velocity",    "Velocity [m/s]",        False),
    ("progress",    "Progress [m]",          False),
    ("crash_rate",  "Crash Rate",            True),
    ("cot",         "Cost of Transport",     True),
    ("v_deviation", "Vel. Deviation [m/s]",  True),
]


def plot_metrics(run_dir: Path | str, use_percentile: bool = False) -> None:
    """Read cma_population.csv and plot per-generation spread for all metrics.

    With ``use_percentile=False`` (default) the band is mean ± 1 std.
    With ``use_percentile=True`` the band is the IQR (25–75th percentile) and an
    extra line shows the top decile (90th pct for higher-is-better metrics,
    10th pct for lower-is-better). The per-generation best is always plotted.
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "results" / "cma_population.csv"

    if not csv_path.is_file():
        print(f"[plot_metrics] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    grouped = df.groupby("generation")
    means = grouped.mean(numeric_only=True)
    stds  = grouped.std(ddof=0, numeric_only=True)
    medians = grouped.median(numeric_only=True)
    q25 = grouped.quantile(0.25, numeric_only=True)
    q75 = grouped.quantile(0.75, numeric_only=True)
    q10 = grouped.quantile(0.10, numeric_only=True)
    q90 = grouped.quantile(0.90, numeric_only=True)
    gens  = means.index.to_numpy()

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    single_dir = plots_dir / "metrics"
    single_dir.mkdir(parents=True, exist_ok=True)

    mean_colour       = "#1f77b4"
    best_colour       = "#d62728"
    baseline_colour   = "#2ca02c"
    specialist_colour = "#9467bd"
    top_colour        = "#ff7f0e"

    # Load baseline + specialist per-generation data if available
    baseline_csv   = run_dir / "results" / "baseline_summary.csv"
    specialist_csv = run_dir / "results" / "specialist_summary.csv"
    baseline_df    = pd.read_csv(baseline_csv)   if baseline_csv.is_file()   else None
    specialist_df  = pd.read_csv(specialist_csv) if specialist_csv.is_file() else None

    def _draw_metric(ax, col, label, lower_is_better):
        # Per-generation best: min for lower-is-better, max otherwise
        best = grouped[col].min().to_numpy() if lower_is_better else grouped[col].max().to_numpy()

        if use_percentile:
            centre = medians[col].to_numpy()
            lo  = q25[col].to_numpy()
            hi  = q75[col].to_numpy()
            top = (q10[col] if lower_is_better else q90[col]).to_numpy()
            centre_label = "median"
            band_label   = "IQR (25–75%)"
            top_label    = "10th pct" if lower_is_better else "90th pct"
        else:
            centre = means[col].to_numpy()
            sig    = stds[col].to_numpy()
            lo, hi = centre - sig, centre + sig
            top    = None
            centre_label = "mean"
            band_label   = "±1 std"

        # Crash rate is a rate — clip to [0, 1]
        if col == "crash_rate":
            centre = np.clip(centre, 0.0, 1.0)
            best   = np.clip(best,   0.0, 1.0)
            lo     = np.clip(lo,     0.0, 1.0)
            hi     = np.clip(hi,     0.0, 1.0)
            if top is not None:
                top = np.clip(top, 0.0, 1.0)

        ax.plot(gens, centre, color=mean_colour, linewidth=1.8, label=centre_label)
        ax.plot(gens, best,   color=best_colour, linewidth=1.4, label="best")
        ax.fill_between(gens, lo, hi, alpha=0.25, color=mean_colour, label=band_label)
        if top is not None:
            ax.plot(gens, top, color=top_colour, linewidth=1.2,
                    linestyle=":", label=top_label)

        if baseline_df is not None and col in baseline_df.columns:
            bl = baseline_df.set_index("generation")[col].reindex(gens).dropna()
            ax.plot(
                bl.index.to_numpy(), bl.to_numpy(),
                color=baseline_colour, linewidth=1.4,
                linestyle="--", marker="o", markersize=4,
                label="baseline",
            )

        if specialist_df is not None and col in specialist_df.columns:
            sp = specialist_df.set_index("generation")[col].reindex(gens).dropna()
            ax.plot(
                sp.index.to_numpy(), sp.to_numpy(),
                color=specialist_colour, linewidth=1.4,
                linestyle="-.", marker="s", markersize=4,
                label="specialist",
            )

        title = f"{label}  (↓ better)" if lower_is_better else label
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Generation")
        ax.set_ylabel(label)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    suptitle = (
        "Metrics Evolution — median, IQR & top decile"
        if use_percentile
        else "Metrics Evolution — mean ± std"
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(suptitle, fontsize=13)

    for ax, (col, label, lower_is_better) in zip(axes.flat, _METRICS):
        _draw_metric(ax, col, label, lower_is_better)

    fig.tight_layout()
    out = plots_dir / "metrics_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot_metrics] Saved {out}")
    plt.close(fig)

    for col, label, lower_is_better in _METRICS:
        fig_s, ax_s = plt.subplots(figsize=(6, 4))
        _draw_metric(ax_s, col, label, lower_is_better)
        fig_s.tight_layout()
        out_s = single_dir / f"{col}.png"
        fig_s.savefig(out_s, dpi=150, bbox_inches="tight")
        print(f"[plot_metrics] Saved {out_s}")
        plt.close(fig_s)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    if not args:
        print("Usage: python -m WP2.plot_metrics <run_dir> [--percentile]")
        sys.exit(1)
    plot_metrics(args[0], use_percentile=("--percentile" in flags))
