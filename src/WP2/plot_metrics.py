"""
Plot per-generation spread for the evaluation metrics.

Usage
-----
    python -m WP2.plot_metrics <run_dir> [--std]

Also callable programmatically::

    from WP2.plot_metrics import plot_metrics
    plot_metrics(run_dir, use_percentile=False)
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


def plot_metrics(run_dir: Path | str, use_percentile: bool = True) -> None:
    """Read cma_population.csv and plot per-generation spread for all metrics.

    With ``use_percentile=True`` (default) the band is the IQR (25–75th
    percentile) and an extra line shows the top decile (90th pct for
    higher-is-better metrics, 10th pct for lower-is-better).
    With ``use_percentile=False`` the band is mean ± 1 std.
    The per-generation best is always plotted.
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "results" / "cma_population.csv"

    if not csv_path.is_file():
        print(f"[plot_metrics] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[plot_metrics] CSV is empty (header only): {csv_path}")
        return
    grouped = df.groupby("generation")
    means = grouped.mean(numeric_only=True)
    stds  = grouped.std(ddof=0, numeric_only=True)
    medians = grouped.median(numeric_only=True)
    q25 = grouped.quantile(0.25, numeric_only=True)
    q75 = grouped.quantile(0.75, numeric_only=True)
    q10 = grouped.quantile(0.10, numeric_only=True)
    q90 = grouped.quantile(0.90, numeric_only=True)
    gens  = means.index.to_numpy()

    # Fitness-best individual per generation — its values are used for the "best"
    # line across ALL metrics (so every red curve refers to the same individual,
    # the one that would actually be selected by the optimizer).
    best_idx  = grouped["fitness"].idxmax()
    best_rows = df.loc[best_idx].set_index("generation")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    single_dir = plots_dir / "metrics"
    single_dir.mkdir(parents=True, exist_ok=True)
    norm_dir = plots_dir / "metrics_normalized"

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

    def _draw_metric(ax, col, label, lower_is_better, normalize=False):
        # "Best" = the fitness-winner individual of each generation, evaluated on
        # this metric. Same individual across all six plots.
        best = best_rows[col].reindex(gens).to_numpy()

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

        # Crash rate is a rate — clip to [0, 1] (skip in normalized mode: values are ratios)
        if col == "crash_rate" and not normalize:
            centre = np.clip(centre, 0.0, 1.0)
            best   = np.clip(best,   0.0, 1.0)
            lo     = np.clip(lo,     0.0, 1.0)
            hi     = np.clip(hi,     0.0, 1.0)
            if top is not None:
                top = np.clip(top, 0.0, 1.0)

        # Pull baseline + specialist series aligned to gens. The baseline is
        # typically only evaluated every Nth generation, so reindexing to every
        # gen leaves NaN gaps. Linearly interpolate those (in generation-space)
        # so normalization has a defined denominator everywhere; keep a mask of
        # the genuinely-evaluated generations so markers only sit on real data.
        bl_vals = None
        bl_real_mask = None
        if baseline_df is not None and col in baseline_df.columns:
            bl_series = baseline_df.set_index("generation")[col].reindex(gens)
            bl_real_mask = ~bl_series.isna().to_numpy()
            bl_vals = bl_series.interpolate(
                method="index", limit_direction="both"
            ).to_numpy()
        sp_vals = None
        if specialist_df is not None and col in specialist_df.columns:
            sp_vals = specialist_df.set_index("generation")[col].reindex(gens).to_numpy()

        if normalize:
            if bl_vals is None:
                return  # cannot normalize without a baseline
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = np.where(np.abs(bl_vals) > 1e-12, bl_vals, np.nan)
                centre = centre / denom
                best   = best   / denom
                lo     = lo     / denom
                hi     = hi     / denom
                if top is not None:
                    top = top / denom
                if sp_vals is not None:
                    sp_vals = sp_vals / denom
                bl_plot = bl_vals / denom  # constant 1.0 where defined
        else:
            bl_plot = bl_vals

        ax.plot(gens, centre, color=mean_colour, linewidth=1.8, label=centre_label)
        best_label = "best" if col == "fitness" else "fitness-best"
        ax.plot(gens, best,   color=best_colour, linewidth=1.4, label=best_label)
        ax.fill_between(gens, lo, hi, alpha=0.25, color=mean_colour, label=band_label)
        if top is not None:
            ax.plot(gens, top, color=top_colour, linewidth=1.2,
                    linestyle=":", label=top_label)

        if bl_plot is not None:
            mask = ~np.isnan(bl_plot)
            # Continuous (interpolated) solid line.
            ax.plot(
                gens[mask], bl_plot[mask],
                color=baseline_colour, linewidth=1.4,
                linestyle="-", label="baseline",
            )

        if sp_vals is not None:
            mask = ~np.isnan(sp_vals)
            ax.plot(
                gens[mask], sp_vals[mask],
                color=specialist_colour, linewidth=1.4,
                linestyle="-.", marker="s", markersize=4,
                label="specialist",
            )

        title = f"{label}  (↓ better)" if lower_is_better else label
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Generation")
        ax.set_ylabel(f"{label} / baseline" if normalize else label)
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

    # ------------------------------------------------------------------
    # Normalized-vs-baseline plots (only if a baseline summary is present)
    # ------------------------------------------------------------------
    if baseline_df is None:
        return

    norm_dir.mkdir(parents=True, exist_ok=True)
    norm_suptitle = suptitle + "  (normalized to baseline)"
    fig_n, axes_n = plt.subplots(2, 3, figsize=(15, 7))
    fig_n.suptitle(norm_suptitle, fontsize=13)
    for ax, (col, label, lower_is_better) in zip(axes_n.flat, _METRICS):
        _draw_metric(ax, col, label, lower_is_better, normalize=True)
    fig_n.tight_layout()
    out_n = norm_dir / "metrics_evolution.png"
    fig_n.savefig(out_n, dpi=150, bbox_inches="tight")
    print(f"[plot_metrics] Saved {out_n}")
    plt.close(fig_n)

    for col, label, lower_is_better in _METRICS:
        fig_s, ax_s = plt.subplots(figsize=(6, 4))
        _draw_metric(ax_s, col, label, lower_is_better, normalize=True)
        fig_s.tight_layout()
        out_s = norm_dir / f"{col}.png"
        fig_s.savefig(out_s, dpi=150, bbox_inches="tight")
        print(f"[plot_metrics] Saved {out_s}")
        plt.close(fig_s)


def plot_validation(run_dir: Path | str) -> None:
    """Plot the held-out validation curves: best individual vs zero-rules
    baseline, generation-by-generation, for each metric.

    Reads ``results/validation_summary.csv`` (written by the inner loop when
    ``validation.enable`` is set) and saves a 2×3 grid plus per-metric figures
    under ``plots/validation/``.
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "results" / "validation_summary.csv"
    if not csv_path.is_file():
        return  # validation was not enabled for this run

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[plot_validation] CSV is empty (header only): {csv_path}")
        return

    gens = df["generation"].to_numpy()
    best_colour = "#1f77b4"      # Hebbian: blue
    baseline_colour = "#2ca02c"  # baseline: green

    out_dir = run_dir / "plots" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _rolling(y, window=10):
        s = pd.Series(y)
        return s.rolling(window=window, min_periods=1, center=False).mean().to_numpy()

    def _draw(ax, col, label, lower_is_better):
        best_col = f"best_{col}"
        base_col = f"baseline_{col}"
        if best_col not in df.columns or base_col not in df.columns:
            return
        best = df[best_col].to_numpy()
        base = df[base_col].to_numpy()
        if col == "crash_rate":
            best = np.clip(best, 0.0, 1.0)
            base = np.clip(base, 0.0, 1.0)
        # Raw curves, faded in the background.
        ax.plot(gens, best, color=best_colour, linewidth=1.0, alpha=0.25)
        ax.plot(gens, base, color=baseline_colour, linewidth=1.0, alpha=0.25)
        # 5-generation rolling averages on top.
        ax.plot(gens, _rolling(best), color=best_colour, linewidth=1.8,
                label="best (Hebbian)")
        ax.plot(gens, _rolling(base), color=baseline_colour, linewidth=1.8,
                label="baseline")
        title = f"{label}  (↓ better)" if lower_is_better else label
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Generation")
        ax.set_ylabel(label)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle("Held-out Validation — best individual vs baseline", fontsize=13)
    for ax, (col, label, lower_is_better) in zip(axes.flat, _METRICS):
        _draw(ax, col, label, lower_is_better)
    fig.tight_layout()
    out = out_dir / "validation_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot_validation] Saved {out}")
    plt.close(fig)

    for col, label, lower_is_better in _METRICS:
        fig_s, ax_s = plt.subplots(figsize=(6, 4))
        _draw(ax_s, col, label, lower_is_better)
        fig_s.tight_layout()
        out_s = out_dir / f"{col}.png"
        fig_s.savefig(out_s, dpi=150, bbox_inches="tight")
        print(f"[plot_validation] Saved {out_s}")
        plt.close(fig_s)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    if not args:
        print("Usage: python -m WP2.plot_metrics <run_dir> [--std]")
        sys.exit(1)
    plot_metrics(args[0], use_percentile=("--std" not in flags))
    plot_validation(args[0])
