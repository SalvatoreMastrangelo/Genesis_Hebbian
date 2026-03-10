from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


FITNESS_COLUMNS = ["ff_0", "ff_1", "ff_2"]
EVAL_REWARD_COLUMN = "eval_reward_mean"
FITNESS_LABELS = {
    "ff_0": "Speed",
    "ff_1": "Cost Of Transport",
    "ff_2": "Progress",
    EVAL_REWARD_COLUMN: "Reward accumulated during Evaluation",
}

# Custom reference point (BIX3) for Pareto plots.
BIX3_POINT = np.array([21.6, 0.3, 325.0], dtype=float)

# Sentinel values defined in src/morph_evolution/evolution_nsga.py
INVALID_V = {0.0}
INVALID_E = {-10.0}
INVALID_P = {0.0}
INVALID_PROGRESS_THRESHOLD = 250.0


def load_nsga_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"generation", *FITNESS_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns in {csv_path}: {missing_list}")
    return df


def _resolve_analysis_dir(input_path: Path) -> Path | None:
    path = input_path.expanduser().resolve()
    if path.is_dir():
        if (path / "population_history.csv").exists():
            return path
        if (path / "analysis" / "population_history.csv").exists():
            return (path / "analysis").resolve()
        return None

    parent = path.parent
    if path.name in {"population_history.csv", "pareto_history.csv", "generation_summary.csv", "nsga.csv"}:
        return parent if (parent / "population_history.csv").exists() else None
    if parent.name == "analysis" and (parent / "population_history.csv").exists():
        return parent.resolve()
    if (parent / "analysis" / "population_history.csv").exists():
        return (parent / "analysis").resolve()
    return None


def _resolve_run_dir(input_path: Path, analysis_dir: Path | None) -> Path:
    path = input_path.expanduser().resolve()
    if path.is_dir():
        if analysis_dir is not None and path == analysis_dir:
            return analysis_dir.parent.resolve()
        return path
    if analysis_dir is not None:
        return analysis_dir.parent.resolve()
    return path.parent.resolve()


def load_run_tables(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, Path, Path]:
    analysis_dir = _resolve_analysis_dir(input_path)
    run_dir = _resolve_run_dir(input_path, analysis_dir)
    if analysis_dir is not None:
        population_path = analysis_dir / "population_history.csv"
        summary_path = analysis_dir / "generation_summary.csv"
        pop_df = load_nsga_csv(population_path)
        summary_df = pd.read_csv(summary_path) if summary_path.exists() else None
        return pop_df, summary_df, population_path, run_dir

    csv_path = input_path.expanduser().resolve()
    pop_df = load_nsga_csv(csv_path)
    return pop_df, None, csv_path, run_dir


def _extract_genome_name_series(df: pd.DataFrame) -> pd.Series:
    if "exp_name" not in df.columns and "rep_exp_names" not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype="object")

    def _extract(row: pd.Series) -> str | None:
        for col in ("exp_name", "rep_exp_names"):
            if col not in row:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            name = str(val).split("|")[0]
            match = re.search(r"\[[^\]]+\]", name)
            if match:
                return match.group(0)
        return None

    return df.apply(_extract, axis=1)


def _invalid_row_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["ff_0"].isin(INVALID_V)
        | df["ff_1"].isin(INVALID_E)
        | df["ff_2"].isin(INVALID_P)
    )


def invalid_repetition_mask(
    df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame | None = None,
) -> pd.Series:
    ref_df = reference_df if reference_df is not None else df
    invalid_row_ref = _invalid_row_mask(ref_df)
    genome_series_ref = _extract_genome_name_series(ref_df)
    if genome_series_ref.notna().any():
        invalid_genomes = set(genome_series_ref[invalid_row_ref].dropna().unique())
        genome_series = _extract_genome_name_series(df)
        invalid_by_genome = genome_series.isin(invalid_genomes)
        return _invalid_row_mask(df) | invalid_by_genome
    return _invalid_row_mask(df)


def filter_sentinels(df: pd.DataFrame, *, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    invalid_mask = invalid_repetition_mask(df, reference_df=reference_df)
    return df.loc[~invalid_mask].copy()


def apply_invalid_repetition_values(
    df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()
    invalid_mask = invalid_repetition_mask(df, reference_df=reference_df)
    df.loc[invalid_mask, "ff_0"] = 0.0
    df.loc[invalid_mask, "ff_1"] = float(min(INVALID_E))
    return df


def filter_to_agg(df: pd.DataFrame) -> pd.DataFrame:
    if "row_kind" not in df.columns:
        return df.copy()
    row_kind = df["row_kind"].astype(str).str.lower()
    return df.loc[row_kind == "agg"].copy()


def _scaled_ylim(values: np.ndarray, padding_ratio: float = 0.08) -> tuple[float, float]:
    if values.size == 0:
        return (0.0, 1.0)
    lower = np.nanmin(values)
    upper = np.nanmax(values)
    if lower == upper:
        padding = 1.0 if lower == 0 else abs(lower) * 0.1
        return (lower - padding, upper + padding)
    padding = (upper - lower) * padding_ratio
    return (lower - padding, upper + padding)


def _scaled_ylim_from_series(values: list[np.ndarray], padding_ratio: float = 0.1) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    combined = np.concatenate([v for v in values if v.size])
    return _scaled_ylim(combined, padding_ratio=padding_ratio)


def _apply_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")


def _fitness_columns_for_plots(df: pd.DataFrame) -> list[str]:
    cols = [col for col in FITNESS_COLUMNS if col in df.columns]
    if EVAL_REWARD_COLUMN in df.columns:
        cols.append(EVAL_REWARD_COLUMN)
    return cols


def _fitness_frame_for_plot(
    df: pd.DataFrame,
    fitness: str,
    *,
    is_raw: bool = False,
) -> pd.DataFrame:
    temp = df[["generation", fitness]].copy()
    if is_raw and fitness == "ff_0":
        temp.loc[temp[fitness].isin(INVALID_V), fitness] = 0.0
    if is_raw and fitness == "ff_1":
        temp.loc[temp[fitness].isin(INVALID_E), fitness] = float(min(INVALID_E))
    if fitness == "ff_1":
        temp[fitness] = -temp[fitness]
    return temp


def _fitness_points_for_plot(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    points = df[cols].to_numpy(dtype=float)
    for idx, col in enumerate(cols):
        if col == "ff_1":
            points[:, idx] = -points[:, idx]
    return points


def _pareto_points_from_plot(points: np.ndarray, cols: list[str]) -> np.ndarray:
    pareto_points = points.copy()
    for idx, col in enumerate(cols):
        if col == "ff_1":
            pareto_points[:, idx] = -pareto_points[:, idx]
    return pareto_points


def _parse_chromosome_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    if "chromosome" not in df.columns:
        raise ValueError("Missing 'chromosome' column for genome PCA.")

    parsed = []
    indices = []
    for idx, val in df["chromosome"].items():
        if pd.isna(val):
            continue
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val, dtype=float)
        elif isinstance(val, str):
            try:
                arr = np.asarray(ast.literal_eval(val), dtype=float)
            except (SyntaxError, ValueError):
                continue
        else:
            continue
        if arr.ndim != 1:
            continue
        parsed.append(arr)
        indices.append(idx)

    if not parsed:
        raise ValueError("No valid genomes found in chromosome column.")

    lengths = np.array([len(arr) for arr in parsed], dtype=int)
    target_len = int(np.bincount(lengths).argmax())
    keep_mask = lengths == target_len
    if not np.any(keep_mask):
        raise ValueError("No genomes with consistent length for PCA.")

    matrix = np.vstack([arr for arr, keep in zip(parsed, keep_mask) if keep])
    idx_keep = [idx for idx, keep in zip(indices, keep_mask) if keep]
    return matrix, df.loc[idx_keep].copy()


def _pca_fit_transform_unique(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.size == 0:
        raise ValueError("No points available for PCA.")
    unique_points = np.unique(points, axis=0)
    mean = np.mean(unique_points, axis=0)
    centered = unique_points - mean
    if centered.shape[0] < 2:
        raise ValueError("Need at least two unique genomes for PCA.")
    _, singular_vals, vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = (singular_vals**2) / max(centered.shape[0] - 1, 1)
    total_var = float(np.sum(explained_var))
    explained_ratio = explained_var / total_var if total_var > 0 else np.zeros_like(explained_var)
    return mean, vt, explained_ratio


def plot_genome_pca_generations(df: pd.DataFrame, output_path: Path) -> None:
    matrix, aligned_df = _parse_chromosome_matrix(df)
    mean, vt, explained_ratio = _pca_fit_transform_unique(matrix)
    pcs = (matrix - mean) @ vt.T
    pc1_var = float(explained_ratio[0]) if explained_ratio.size > 0 else 0.0
    pc2_var = float(explained_ratio[1]) if explained_ratio.size > 1 else 0.0
    pc3_var = float(explained_ratio[2]) if explained_ratio.size > 2 else 0.0
    pc4_var = float(explained_ratio[3]) if explained_ratio.size > 3 else 0.0
    pc5_var = float(explained_ratio[4]) if explained_ratio.size > 4 else 0.0

    generations = [0, 5, 10, 15, 20, 25]
    _apply_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)

    for ax, gen in zip(axes.flat, generations):
        mask = aligned_df["generation"] == gen
        if np.any(mask):
            ax.scatter(pcs[mask, 0], pcs[mask, 1], s=18, alpha=0.5, color="#1f77b4")
            ax.set_title(f"Generation {gen} (n={int(mask.sum())})")
        else:
            ax.text(0.5, 0.5, f"Generation {gen}\nNo data", ha="center", va="center")
            ax.set_title(f"Generation {gen}")
        ax.grid(alpha=0.35)
        ax.set_xlabel(f"PC1 ({pc1_var * 100:.2f}%)")
        ax.set_ylabel(f"PC2 ({pc2_var * 100:.2f}%)")

    fig.suptitle(
        "Genome PCA — "
        f"PC1 {pc1_var * 100:.2f}% | "
        f"PC2 {pc2_var * 100:.2f}% | "
        f"PC3 {pc3_var * 100:.2f}% | "
        f"PC4 {pc4_var * 100:.2f}% | "
        f"PC5 {pc5_var * 100:.2f}%",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"PCA variance explained: PC1={pc1_var * 100:.2f}% PC2={pc2_var * 100:.2f}%")


def plot_genome_fitness_correlation(df: pd.DataFrame, output_path: Path) -> None:
    matrix, aligned_df = _parse_chromosome_matrix(df)
    fitness_cols = _fitness_columns_for_plots(aligned_df)
    fitness_df = aligned_df[fitness_cols].copy()
    if "ff_1" in fitness_df.columns:
        fitness_df["ff_1"] = -fitness_df["ff_1"]
    gene_cols = [f"g{i}" for i in range(matrix.shape[1])]
    try:
        from morph_evolution.chromosome_drone import Chromosome_Drone
    except Exception:
        import sys
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from morph_evolution.chromosome_drone import Chromosome_Drone

    param_names = [param.name for param in Chromosome_Drone.PARAMS]
    if len(param_names) == matrix.shape[1]:
        gene_cols = param_names
    genome_df = pd.DataFrame(matrix, columns=gene_cols, index=aligned_df.index)

    combined = pd.concat([genome_df, fitness_df], axis=1)
    combined = combined.apply(pd.to_numeric, errors="coerce")
    corr = combined.corr().loc[gene_cols, fitness_cols]
    corr = corr.fillna(0.0)

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, max(4.5, 0.32 * len(gene_cols))))
    im = ax.imshow(corr.to_numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")

    ax.set_xticks(np.arange(len(fitness_cols)))
    ax.set_xticklabels([FITNESS_LABELS[c] for c in fitness_cols], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(gene_cols)))
    ax.set_yticklabels(gene_cols)
    ax.set_xlabel("Fitness")
    ax.set_ylabel("Genome gene")
    ax.set_title("Correlation: Genome vs Fitness")

    if len(gene_cols) <= 25:
        for i in range(len(gene_cols)):
            for j in range(len(fitness_cols)):
                ax.text(
                    j,
                    i,
                    f"{corr.iat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _get_gene_names(n_genes: int) -> list[str]:
    gene_cols = [f"g{i}" for i in range(n_genes)]
    try:
        from morph_evolution.chromosome_drone import Chromosome_Drone
    except Exception:
        import sys
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from morph_evolution.chromosome_drone import Chromosome_Drone

    param_names = [param.name for param in Chromosome_Drone.PARAMS]
    if len(param_names) == n_genes:
        gene_cols = param_names
    return gene_cols


def plot_top5pct_gene_means(df: pd.DataFrame, output_path: Path) -> None:
    matrix, aligned_df = _parse_chromosome_matrix(df)
    gene_cols = _get_gene_names(matrix.shape[1])
    genome_df = pd.DataFrame(matrix, columns=gene_cols, index=aligned_df.index)

    fitness_cols = _fitness_columns_for_plots(aligned_df)
    fitness_df = aligned_df[fitness_cols].copy()
    if "ff_1" in fitness_df.columns:
        fitness_df["ff_1"] = -fitness_df["ff_1"]

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(16, 7.5))
    color_map = {
        "ff_0": "#1f77b4",  # Speed
        "ff_1": "#ff7f0e",  # CoT
        "ff_2": "#2ca02c",  # Progress
        EVAL_REWARD_COLUMN: "#9467bd",  # Reward accumulated during Evaluation
    }
    default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]
    colors = {
        fitness: color_map.get(fitness, default_palette[idx % len(default_palette)])
        for idx, fitness in enumerate(fitness_cols)
    }
    if len(fitness_cols) <= 1:
        offset_values = np.array([0.0])
    else:
        offset_values = np.linspace(-0.27, 0.27, len(fitness_cols))
    offsets = dict(zip(fitness_cols, offset_values))

    for fitness in fitness_cols:
        values = fitness_df[fitness].to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            continue

        finite_vals = values[finite_mask]
        if fitness == "ff_1":
            threshold = np.nanpercentile(finite_vals, 5)
            select_mask = finite_mask & (values <= threshold)
        else:
            threshold = np.nanpercentile(finite_vals, 95)
            select_mask = finite_mask & (values >= threshold)

        if not np.any(select_mask):
            continue

        subset = genome_df.loc[select_mask]
        mean_genes = subset.mean(axis=0).to_numpy()
        std_genes = subset.std(axis=0).fillna(0.0).to_numpy()
        x = np.arange(len(gene_cols)) + offsets[fitness]
        ax.errorbar(
            x,
            mean_genes,
            yerr=std_genes,
            fmt="o",
            markersize=6.8,
            linestyle="none",
            color=colors[fitness],
            ecolor=colors[fitness],
            elinewidth=1.2,
            capsize=3,
            label=f"Top 5% {FITNESS_LABELS[fitness]}",
        )
    ax.set_ylabel("Mean gene value")
    ax.grid(alpha=0.35)
    ax.legend(loc="best")
    ax.set_xticks(range(len(gene_cols)))
    ax.set_xticklabels(gene_cols, rotation=25, ha="right")
    ax.set_xlabel("Genome gene")
    ax.set_title("Top 5% Genome Means by Fitness", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_fitness_trends(
    df: pd.DataFrame,
    output_path: Path,
    raw_df: pd.DataFrame | None = None,
    summary_df: pd.DataFrame | None = None,
) -> None:
    grouped = df.groupby("generation", sort=True)
    generations = grouped.size().index.to_numpy()
    if generations.size == 0:
        raise ValueError("No valid rows available after filtering sentinel values.")

    if summary_df is not None:
        required_summary_cols = {"generation"}
        missing_summary = required_summary_cols - set(summary_df.columns)
        if missing_summary:
            missing_list = ", ".join(sorted(missing_summary))
            raise ValueError(f"Missing columns in generation summary: {missing_list}")

    base_df = raw_df if raw_df is not None else df
    fitness_cols = [col for col in _fitness_columns_for_plots(base_df) if col in FITNESS_COLUMNS]
    if EVAL_REWARD_COLUMN in base_df.columns:
        fitness_cols.append(EVAL_REWARD_COLUMN)
    if not fitness_cols:
        raise ValueError("No fitness columns available for plotting.")

    _apply_plot_style()
    total_plots = len(fitness_cols) + 1
    ncols = 2
    nrows = int(np.ceil(total_plots / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.3 * ncols, 5.5 * nrows),
        sharex=True,
    )
    axes_array = np.atleast_1d(axes)
    axes_flat = axes_array.flatten()
    for ax, fitness in zip(axes_flat, fitness_cols):
        if summary_df is not None and fitness in FITNESS_COLUMNS:
            summary = summary_df.sort_values("generation").copy()
            if fitness == "ff_0":
                center_col = "median_vel"
                extreme_col = "best_vel"
                band_low_col = "q1_vel"
                band_high_col = "q3_vel"
            elif fitness == "ff_1":
                center_col = "median_eff"
                extreme_col = "best_eff"
                band_low_col = "q1_eff"
                band_high_col = "q3_eff"
            else:
                center_col = "median_prog"
                extreme_col = "best_prog"
                band_low_col = "q1_prog"
                band_high_col = "q3_prog"

            needed = {
                "generation",
                center_col,
                extreme_col,
                band_low_col,
                band_high_col,
            }
            missing = needed - set(summary.columns)
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(f"Missing summary columns for {fitness}: {missing_list}")

            stats = pd.DataFrame(
                {
                    "generation": summary["generation"].to_numpy(),
                    "mean": summary[center_col].to_numpy(),
                    "median": summary[center_col].to_numpy(),
                    "extreme": summary[extreme_col].to_numpy(),
                    "band_low": summary[band_low_col].to_numpy(),
                    "band_high": summary[band_high_col].to_numpy(),
                }
            )
            if fitness == "ff_1":
                for col in ("mean", "median", "extreme", "band_low", "band_high"):
                    stats[col] = -stats[col]
            ylim_values = stats[["band_low", "band_high", "extreme"]].to_numpy(dtype=float).ravel()
            band_lower = stats["band_low"]
            band_upper = stats["band_high"]
            extreme_series = stats["extreme"]
            extreme_label = "best"
        elif raw_df is not None:
            temp = _fitness_frame_for_plot(raw_df, fitness, is_raw=True)
            stats = temp.groupby("generation", sort=True)[fitness].agg(
                ["mean", "median", "std", "max", "min"]
            ).reset_index()
            ylim_values = temp[fitness].to_numpy()
            band_lower = stats["mean"] - stats["std"].fillna(0.0)
            band_upper = stats["mean"] + stats["std"].fillna(0.0)
            extreme_col = "min" if fitness == "ff_1" else "max"
            extreme_series = stats[extreme_col]
            extreme_label = "min" if fitness == "ff_1" else "max"
        else:
            temp = _fitness_frame_for_plot(df, fitness, is_raw=False)
            stats = temp.groupby("generation", sort=True)[fitness].agg(
                ["mean", "median", "std", "max", "min"]
            ).reset_index()
            ylim_values = temp[fitness].to_numpy()
            stats["std"] = stats["std"].fillna(0.0)
            band_lower = stats["mean"] - stats["std"]
            band_upper = stats["mean"] + stats["std"]
            extreme_col = "min" if fitness == "ff_1" else "max"
            extreme_series = stats[extreme_col]
            extreme_label = "min" if fitness == "ff_1" else "max"
        ax.fill_between(
            stats["generation"],
            band_lower,
            band_upper,
            color="#9ecae1",
            alpha=0.35,
            label="IQR" if summary_df is not None and fitness in FITNESS_COLUMNS else "mean ± std",
        )
        ax.plot(
            stats["generation"],
            stats["mean"],
            color="#1f77b4",
            linewidth=2.5,
            label="median" if summary_df is not None and fitness in FITNESS_COLUMNS else "mean",
        )
        ax.plot(
            stats["generation"],
            stats["median"],
            color="#2ca02c",
            linewidth=2.2,
            label="median" if summary_df is None or fitness not in FITNESS_COLUMNS else "median (same as center)",
        )
        ax.plot(
            stats["generation"],
            extreme_series,
            color="#d62728",
            linewidth=2.0,
            linestyle="--",
            label=extreme_label,
        )

        ax.set_title(FITNESS_LABELS[fitness])
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel(FITNESS_LABELS[fitness], fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(alpha=0.35)
        curve_values = [
            stats["mean"].to_numpy(dtype=float),
            stats["median"].to_numpy(dtype=float),
            np.asarray(extreme_series, dtype=float),
        ]
        ax.set_ylim(_scaled_ylim_from_series(curve_values, padding_ratio=0.1))
        ax.set_xticks(generations)

    count_ax = axes_flat[len(fitness_cols)]
    count_source_df = raw_df if raw_df is not None else df
    if summary_df is not None and {"generation", "minimal_p"}.issubset(summary_df.columns):
        threshold_df = summary_df.loc[:, ["generation", "minimal_p"]].copy()
    else:
        threshold_df = pd.DataFrame(
            {
                "generation": sorted(count_source_df["generation"].unique()),
                "minimal_p": float(INVALID_PROGRESS_THRESHOLD),
            }
        )

    count_df = count_source_df.loc[:, ["generation", "ff_2"]].copy()
    count_df = count_df[np.isfinite(count_df["ff_2"].to_numpy(dtype=float))]
    count_stats = []
    threshold_map = dict(zip(threshold_df["generation"], threshold_df["minimal_p"]))
    for generation in sorted(count_df["generation"].unique()):
        gen_ff2 = count_df.loc[count_df["generation"] == generation, "ff_2"].to_numpy(dtype=float)
        threshold = float(threshold_map.get(generation, INVALID_PROGRESS_THRESHOLD))
        count_stats.append(
            {
                "generation": generation,
                "count_above_threshold": int(np.sum(gen_ff2 >= threshold)),
                "minimal_p": threshold,
            }
        )
    count_stats_df = pd.DataFrame(count_stats)
    count_ax.bar(
        count_stats_df["generation"],
        count_stats_df["count_above_threshold"],
        width=0.7,
        color="#6baed6",
        alpha=0.85,
        label="count >= minimal_p",
    )
    count_ax.plot(
        count_stats_df["generation"],
        count_stats_df["count_above_threshold"],
        color="#08519c",
        linewidth=2.2,
        marker="o",
        markersize=4.8,
    )
    count_ax.set_title("Individuals Above Minimal Progress")
    count_ax.set_xlabel("Generation", fontsize=12)
    count_ax.set_ylabel("Count", fontsize=13)
    count_ax.grid(alpha=0.35)
    count_ax.set_xticks(generations)
    count_ax.set_ylim(_scaled_ylim(count_stats_df["count_above_threshold"].to_numpy(dtype=float), padding_ratio=0.12))

    for ax in axes_flat[total_plots:]:
        ax.set_visible(False)

    axes_flat[0].legend(loc="best")
    fig.suptitle("Fitness Trends Across Generations", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def pareto_mask(points: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    mask = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not mask[i]:
            continue
        point = points[i]
        dominates = np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        if np.any(dominates):
            mask[i] = False
            continue
        dominated = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        mask[dominated] = False
        mask[i] = True
    return mask


def pareto_mask_finite(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    finite_mask = np.isfinite(points).all(axis=1)
    if not np.any(finite_mask):
        return np.zeros(points.shape[0], dtype=bool)
    mask = np.zeros(points.shape[0], dtype=bool)
    mask[finite_mask] = pareto_mask(points[finite_mask])
    return mask


def plot_pareto_fronts(df: pd.DataFrame, output_path: Path, generation: int) -> None:
    pairs = [("ff_0", "ff_1"), ("ff_0", "ff_2"), ("ff_1", "ff_2")]
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(19.8, 5.8))
    scatter_handle = None

    for ax, (x_col, y_col) in zip(axes, pairs):
        all_points = _fitness_points_for_plot(df, [x_col, y_col])
        all_generations = df["generation"].to_numpy()
        if all_points.size == 0:
            raise ValueError("No points available for Pareto fronts.")
        finite_mask = np.isfinite(all_points).all(axis=1)
        all_points = all_points[finite_mask]
        all_generations = all_generations[finite_mask]
        global_pareto_points = _pareto_points_from_plot(all_points, [x_col, y_col])
        global_front_mask_2d = pareto_mask_finite(global_pareto_points)
        total_points = int(all_points.shape[0])
        front_points = int(np.sum(global_front_mask_2d))
        scatter_handle = ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            c=all_generations,
            cmap="viridis",
            s=18,
            alpha=0.28,
            label="all generations",
            zorder=1,
        )
        ax.scatter(
            all_points[global_front_mask_2d, 0],
            all_points[global_front_mask_2d, 1],
            s=42,
            facecolor="none",
            edgecolor="#d62728",
            linewidth=1.8,
            label="global 2D front",
            zorder=3,
        )
        bix3_x = BIX3_POINT[FITNESS_COLUMNS.index(x_col)]
        bix3_y = BIX3_POINT[FITNESS_COLUMNS.index(y_col)]
        ax.scatter(
            bix3_x,
            bix3_y,
            s=180,
            marker="*",
            color="#ffbf00",
            edgecolor="#7a5a00",
            linewidth=1.2,
            label="BIX3",
            zorder=5,
        )
        ax.set_xlabel(FITNESS_LABELS[x_col])
        ax.set_ylabel(FITNESS_LABELS[y_col])
        ax.set_title(
            f"{FITNESS_LABELS[x_col]} vs {FITNESS_LABELS[y_col]} "
            f"(global front {front_points}/{total_points})"
        )
        ax.grid(alpha=0.35)

    axes[0].legend(loc="best")
    if scatter_handle is not None:
        cbar_ax = fig.add_axes([0.93, 0.18, 0.02, 0.64])
        fig.colorbar(scatter_handle, cax=cbar_ax, label="Generation")
    fig.suptitle("Pareto Fronts (All Generations)", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 1.0])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pareto_front_3d(df: pd.DataFrame, output_path: Path) -> None:
    df_plot = df.copy()
    df_plot["ff_1"] = -df_plot["ff_1"]
    points = _fitness_points_for_plot(df, FITNESS_COLUMNS)
    if points.size == 0:
        raise ValueError("No points available for 3D Pareto front.")
    finite_mask = np.isfinite(points).all(axis=1)
    total_points = int(np.sum(finite_mask))
    generations = df["generation"].to_numpy()
    pareto_points = _pareto_points_from_plot(points, FITNESS_COLUMNS)
    mask = pareto_mask_finite(pareto_points)
    front_df = df_plot.loc[mask, ["generation", *FITNESS_COLUMNS]].copy()
    points_csv = output_path.with_name(f"{output_path.stem}_points.csv")
    front_df.to_csv(points_csv, index=False)

    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        print("plotly not installed; skipping 3D Pareto HTML.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            name="population",
            marker=dict(
                size=3,
                color=generations,
                colorscale="Viridis",
                colorbar=dict(title="Generation"),
                opacity=0.7,
            ),
            hovertemplate=(
                "Generation: %{marker.color}<br>"
                f"{FITNESS_LABELS['ff_0']}: %{{x}}<br>"
                f"{FITNESS_LABELS['ff_1']}: %{{y}}<br>"
                f"{FITNESS_LABELS['ff_2']}: %{{z}}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=front_df["ff_0"],
            y=front_df["ff_1"],
            z=front_df["ff_2"],
            mode="markers",
            name="pareto front",
            marker=dict(
                size=5,
                color=front_df["generation"],
                colorscale="Viridis",
                line=dict(color="#d62728", width=2),
                showscale=False,
                opacity=0.9,
            ),
            hovertemplate=(
                "Generation: %{marker.color}<br>"
                f"{FITNESS_LABELS['ff_0']}: %{{x}}<br>"
                f"{FITNESS_LABELS['ff_1']}: %{{y}}<br>"
                f"{FITNESS_LABELS['ff_2']}: %{{z}}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[float(BIX3_POINT[0])],
            y=[float(BIX3_POINT[1])],
            z=[float(BIX3_POINT[2])],
            mode="markers",
            name="BIX3",
            marker=dict(
                size=8,
                color="#ffbf00",
                symbol="diamond",
                line=dict(color="#7a5a00", width=3),
            ),
            hovertemplate=(
                "BIX3<br>"
                f"{FITNESS_LABELS['ff_0']}: %{{x}}<br>"
                f"{FITNESS_LABELS['ff_1']}: %{{y}}<br>"
                f"{FITNESS_LABELS['ff_2']}: %{{z}}<extra></extra>"
            ),
        )
    )
    if front_df.shape[0] >= 3:
        fig.add_trace(
            go.Mesh3d(
                x=front_df["ff_0"],
                y=front_df["ff_1"],
                z=front_df["ff_2"],
                alphahull=0,
                opacity=0.2,
                color="#d62728",
                name="pareto surface",
                showscale=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Pareto Front (3D) — {front_df.shape[0]}/{total_points} points",
        scene=dict(
            xaxis_title=FITNESS_LABELS["ff_0"],
            yaxis_title=FITNESS_LABELS["ff_1"],
            zaxis_title=FITNESS_LABELS["ff_2"],
        ),
        legend=dict(
            orientation="h",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="bottom",
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    fig.write_html(output_path)
    png_path = output_path.with_suffix(".png")
    try:
        fig.write_image(png_path, scale=2)
    except Exception as exc:
        print(f"plotly static export failed ({exc}); PNG not generated.")

def _normalize_points(points: np.ndarray, mins: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    safe_ranges = np.where(ranges == 0.0, 1.0, ranges)
    norm = (points - mins) / safe_ranges
    return np.clip(norm, 0.0, 1.0)


def _crowding_distances(points: np.ndarray) -> np.ndarray:
    n_points, n_dims = points.shape
    if n_points == 0:
        return np.array([])
    distances = np.zeros(n_points, dtype=float)
    for dim in range(n_dims):
        order = np.argsort(points[:, dim])
        distances[order[0]] += 0.0
        distances[order[-1]] += 0.0
        for i in range(1, n_points - 1):
            prev_val = points[order[i - 1], dim]
            next_val = points[order[i + 1], dim]
            distances[order[i]] += next_val - prev_val
    return distances


def _nearest_neighbor_stats(points: np.ndarray) -> tuple[float, float]:
    n_points = points.shape[0]
    if n_points < 2:
        return 0.0, 0.0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    mean_nn = float(np.mean(nearest))
    spacing = float(np.sqrt(np.mean((nearest - mean_nn) ** 2)))
    max_gap = float(np.max(nearest))
    return spacing, max_gap


def _mean_pairwise_distance(points: np.ndarray) -> float:
    n_points = points.shape[0]
    if n_points < 2:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    upper = dist[np.triu_indices(n_points, k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _entropy_simpson(points: np.ndarray, bins: int = 8) -> tuple[float, float]:
    n_points = points.shape[0]
    if n_points == 0:
        return 0.0, 0.0
    hist, _ = np.histogramdd(points, bins=bins, range=[(0, 1)] * points.shape[1])
    counts = hist.flatten()
    total = np.sum(counts)
    if total == 0:
        return 0.0, 0.0
    probs = counts[counts > 0] / total
    entropy = -np.sum(probs * np.log2(probs))
    simpson = 1.0 - np.sum(probs ** 2)
    return float(entropy), float(max(simpson, 0.0))


def _min_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.array([])
    diff = a[:, None, :] - b[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return np.min(dist, axis=1)


def _hypervolume(points: np.ndarray, grid_res: int = 25) -> float:
    n_points = points.shape[0]
    if n_points == 0:
        return 0.0
    grid = np.linspace(0.0, 1.0, grid_res)
    mesh = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    covered = np.zeros(mesh.shape[0], dtype=bool)
    for point in points:
        covered |= np.all(mesh <= point, axis=1)
    return float(np.mean(covered))


def compute_evolutionary_metrics(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    all_points = df[FITNESS_COLUMNS].to_numpy()
    if all_points.size == 0:
        raise ValueError("No points available to compute evolutionary metrics.")

    all_points = all_points[np.isfinite(all_points).all(axis=1)]
    if all_points.size == 0:
        raise ValueError("No finite points available to compute evolutionary metrics.")
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    ranges = maxs - mins

    global_front = all_points[pareto_mask(all_points)]
    global_front_norm = _normalize_points(global_front, mins, ranges)

    rows = []
    for generation in sorted(df["generation"].unique()):
        gen_df = df[df["generation"] == generation]
        points = gen_df[FITNESS_COLUMNS].to_numpy()
        points = points[np.isfinite(points).all(axis=1)]
        n_points = int(points.shape[0])
        if n_points == 0:
            rows.append(
                dict(
                    generation=generation,
                    hypervolume=0.0,
                    spacing=0.0,
                    max_gap=0.0,
                    crowding_mean=0.0,
                    diversity=0.0,
                    entropy=0.0,
                    simpson=0.0,
                    gd=0.0,
                    igd=0.0,
                    n_points=0,
                    n_front_points=0,
                )
            )
            continue

        front_points = points[pareto_mask(points)]
        front_norm = _normalize_points(front_points, mins, ranges)

        crowding = _crowding_distances(front_norm)
        crowding_mean = float(np.mean(crowding)) if crowding.size else 0.0
        spacing, max_gap = _nearest_neighbor_stats(front_norm)
        diversity = _mean_pairwise_distance(front_norm)
        entropy, simpson = _entropy_simpson(front_norm)

        gd_vals = _min_distances(front_norm, global_front_norm)
        igd_vals = _min_distances(global_front_norm, front_norm)
        gd = float(np.mean(gd_vals)) if gd_vals.size else 0.0
        igd = float(np.mean(igd_vals)) if igd_vals.size else 0.0

        hypervolume_norm = _hypervolume(front_norm)
        hypervolume = float(hypervolume_norm * np.prod(np.where(ranges == 0.0, 1.0, ranges)))

        rows.append(
            dict(
                generation=generation,
                hypervolume=hypervolume,
                spacing=spacing,
                max_gap=max_gap,
                crowding_mean=crowding_mean,
                diversity=diversity,
                entropy=entropy,
                simpson=simpson,
                gd=gd,
                igd=igd,
                n_points=n_points,
                n_front_points=int(front_points.shape[0]),
            )
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df


def plot_evolutionary_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        "hypervolume",
        "spacing",
        "max_gap",
        "crowding_mean",
        "diversity",
        "entropy",
        "simpson",
        "gd",
        "igd",
    ]
    required = {"generation", *metrics}
    missing = required - set(metrics_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns in evolutionary metrics: {missing_list}")

    plot_df = metrics_df.sort_values("generation").copy()
    plot_df[metrics] = plot_df[metrics].fillna(0.0)
    generations = plot_df["generation"].to_numpy()

    _apply_plot_style()
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    for ax, metric in zip(axes.flatten(), metrics):
        values = plot_df[metric].to_numpy()
        ax.plot(generations, values, marker="o", linewidth=2.0, markersize=4.5)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Generation")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.35)
        ax.set_ylim(_scaled_ylim(values))
        ax.set_xticks(generations)

    fig.suptitle("Evolutionary Metrics Across Generations", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_dominance_solutions(
    df: pd.DataFrame,
    output_path: Path,
    *,
    summary_df: pd.DataFrame | None = None,
) -> None:
    if summary_df is not None and {"generation", "pareto_size"}.issubset(summary_df.columns):
        plot_df = summary_df.loc[:, ["generation", "pareto_size"]].copy()
        plot_df = plot_df.sort_values("generation")
        plot_df = plot_df.rename(columns={"pareto_size": "nondominated_count"})
    else:
        if "generation" not in df.columns:
            raise ValueError("Missing 'generation' column for dominance plot.")
        if "is_pareto" in df.columns:
            plot_df = (
                df.groupby("generation", sort=True)["is_pareto"]
                .sum()
                .reset_index(name="nondominated_count")
            )
        else:
            rows = []
            for generation in sorted(df["generation"].unique()):
                gen_df = df[df["generation"] == generation]
                points = gen_df[FITNESS_COLUMNS].to_numpy(dtype=float)
                points = points[np.isfinite(points).all(axis=1)]
                rows.append(
                    {
                        "generation": generation,
                        "nondominated_count": int(np.sum(pareto_mask(points))) if points.size else 0,
                    }
                )
            plot_df = pd.DataFrame(rows)

    if "is_pareto" in df.columns:
        pareto_df = df.loc[df["is_pareto"].astype(int) == 1].copy()
    else:
        rows = []
        for generation in sorted(df["generation"].unique()):
            gen_df = df[df["generation"] == generation].copy()
            points = gen_df[FITNESS_COLUMNS].to_numpy(dtype=float)
            finite_mask = np.isfinite(points).all(axis=1)
            gen_df = gen_df.loc[finite_mask].copy()
            points = points[finite_mask]
            if points.size == 0:
                continue
            gen_df["_is_pareto_tmp"] = pareto_mask(points).astype(int)
            rows.append(gen_df.loc[gen_df["_is_pareto_tmp"] == 1].drop(columns="_is_pareto_tmp"))
        pareto_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)

    if not pareto_df.empty:
        genome_name = _extract_genome_name_series(pareto_df)
        pareto_df = pareto_df.copy()
        pareto_df["_dominance_key"] = genome_name
        missing_key_mask = pareto_df["_dominance_key"].isna()
        if missing_key_mask.any():
            rounded_points = pareto_df.loc[missing_key_mask, FITNESS_COLUMNS].round(8)
            pareto_df.loc[missing_key_mask, "_dominance_key"] = rounded_points.apply(
                lambda row: tuple(row[col] for col in FITNESS_COLUMNS),
                axis=1,
            )
        first_seen = (
            pareto_df.groupby("_dominance_key", sort=False)["generation"]
            .min()
            .reset_index(name="first_generation")
        )
        discovered_df = (
            first_seen.groupby("first_generation", sort=True)
            .size()
            .reset_index(name="new_nondominated_count")
            .rename(columns={"first_generation": "generation"})
        )
    else:
        discovered_df = pd.DataFrame(
            {"generation": plot_df["generation"].to_numpy(dtype=int), "new_nondominated_count": 0}
        )

    plot_df = plot_df.merge(discovered_df, on="generation", how="left")
    plot_df["new_nondominated_count"] = plot_df["new_nondominated_count"].fillna(0.0)

    generations = plot_df["generation"].to_numpy(dtype=int)
    values = plot_df["nondominated_count"].to_numpy(dtype=float)
    discovered_values = plot_df["new_nondominated_count"].to_numpy(dtype=float)

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(
        generations,
        discovered_values,
        width=0.65,
        color="#9ecae1",
        alpha=0.75,
        label="new Pareto solutions discovered",
        zorder=1,
    )
    ax.plot(
        generations,
        values,
        marker="o",
        linewidth=2.4,
        markersize=5.5,
        color="#d62728",
        label="Pareto solutions in generation",
        zorder=3,
    )
    ax.fill_between(generations, 0.0, values, color="#fcae91", alpha=0.20, zorder=2)
    ax.set_title("Pareto Dominance Across Generations", fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Count")
    ax.set_xticks(generations)
    ax.set_ylim(_scaled_ylim(np.concatenate([values, discovered_values]), padding_ratio=0.12))
    ax.grid(alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def report_duplicate_stats(df: pd.DataFrame) -> None:
    genome_series = _extract_genome_name_series(df)
    if genome_series.isna().all():
        raise ValueError("Missing 'exp_name'/'rep_exp_names' columns for genome parsing.")
    missing_genome = int(genome_series.isna().sum())
    if missing_genome:
        print(f"Rows without genome in exp_name: {missing_genome}")

    df = df.copy()
    df["genome_name"] = genome_series
    df = df[df["genome_name"].notna()]
    dup_mask = df.duplicated(subset=["genome_name"], keep=False)
    dup_count = int(dup_mask.sum())
    print(f"Duplicate individuals (by genome filename): {dup_count}")
    if dup_count == 0:
        return
    dup_df = df.loc[dup_mask, ["genome_name", "generation", *FITNESS_COLUMNS]].copy()
    grouped = dup_df.groupby("genome_name", sort=False)
    counts = grouped.size().rename("n_evals")
    means = grouped[FITNESS_COLUMNS].mean().add_prefix("mean_")
    stds = grouped[FITNESS_COLUMNS].std().fillna(0.0).add_prefix("std_")
    generations = grouped["generation"].apply(
        lambda series: ",".join(str(val) for val in sorted(set(series)))
    ).rename("generations")
    summary = pd.concat([counts, means, stds, generations], axis=1)
    summary = summary.loc[summary["n_evals"] > 1]
    print("Duplicate genomes (evaluated >1):")
    print(summary.to_string(float_format="%.6f"))


def _extract_urdf_name(df: pd.DataFrame) -> pd.Series:
    if "exp_name" not in df.columns and "rep_exp_names" not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype="object")

    def _extract(row: pd.Series) -> str | None:
        for col in ("exp_name", "rep_exp_names"):
            if col not in row:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            name = str(val).split("|")[0]
            match = re.search(r"\[[^\]]+\]", name)
            if match:
                return match.group(0)
            if ".urdf" in name:
                return Path(name).stem
            if name:
                return name
        return None

    return df.apply(_extract, axis=1)


def export_pareto_csv(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    points = df[FITNESS_COLUMNS].to_numpy()
    if points.size == 0:
        raise ValueError("No points available to compute Pareto front.")
    pareto_mask_full = pareto_mask_finite(points)
    if not np.any(pareto_mask_full):
        raise ValueError("No finite points available to compute Pareto front.")
    pareto_df = df.loc[pareto_mask_full].copy()
    pareto_df["urdf_name"] = _extract_urdf_name(pareto_df)
    cols = ["urdf_name", "exp_name", "generation", *FITNESS_COLUMNS]
    existing_cols = [c for c in cols if c in pareto_df.columns]
    pareto_df = pareto_df[existing_cols].sort_values(["generation", "urdf_name"])
    pareto_df.to_csv(output_path, index=False)
    return pareto_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate post-evolution plots from legacy nsga.csv or a new NSGA run directory."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_processing/nsga_GP"),
        help=(
            "Path to a legacy nsga.csv, to population_history.csv, to an analysis directory, "
            "or to the root folder of a new NSGA run."
        ),
    )
    parser.add_argument(
        "--run-dir",
        dest="input",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--csv",
        dest="input",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag to name the output folder as post_evolution_plots_<tag>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: <run-dir>/post_evolution_plots[_<tag>].",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=None,
        help="Generation to use for Pareto fronts (default: last generation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_raw, summary_df, source_csv, run_dir = load_run_tables(args.input)
    df_agg_raw = filter_to_agg(df_raw)
    df = filter_sentinels(df_agg_raw, reference_df=df_raw)
    trends_raw = apply_invalid_repetition_values(df_agg_raw, reference_df=df_raw)

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        tag = args.tag.strip() if isinstance(args.tag, str) and args.tag.strip() else ""
        safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_")
        dir_name = "post_evolution_plots"
        if safe_tag:
            dir_name = f"{dir_name}_{safe_tag}"
        output_dir = run_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded evolution data from: {source_csv}")
    print(f"Using run directory: {run_dir}")
    report_duplicate_stats(df)
    plot_fitness_trends(
        df,
        output_dir / "fitness_trends.png",
        raw_df=trends_raw,
        summary_df=summary_df,
    )

    if args.generation is None:
        generation = int(df["generation"].max())
    else:
        generation = args.generation
    plot_pareto_fronts(df, output_dir / "pareto_fronts.png", generation)
    plot_pareto_front_3d(df, output_dir / "pareto_front_3d.html")
    export_pareto_csv(df, output_dir / "pareto.csv")
    metrics_df = compute_evolutionary_metrics(df, output_dir / "evolutionary_metrics.csv")
    plot_evolutionary_metrics(metrics_df, output_dir / "evolutionary_metrics.png")
    plot_dominance_solutions(
        df,
        output_dir / "dominance_solutions.png",
        summary_df=summary_df,
    )
    plot_genome_pca_generations(df, output_dir / "genome_pca_generations.png")
    plot_genome_fitness_correlation(df, output_dir / "genome_fitness_correlation.png")
    plot_top5pct_gene_means(df, output_dir / "genome_top5pct_gene_means.png")


if __name__ == "__main__":
    main()
