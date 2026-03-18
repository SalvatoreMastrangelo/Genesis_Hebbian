from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
AIR_DENSITY_KG_M3 = 1.225
AIR_DYNAMIC_VISCOSITY_KG_M_S = 1.81e-5
NACA4_CSV_PATH = Path(__file__).resolve().parents[1] / "naca_generation" / "naca4.csv"


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


def _load_chromosome_drone_class():
    try:
        from morph_evolution.chromosome_drone import Chromosome_Drone
    except Exception:
        import sys
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from morph_evolution.chromosome_drone import Chromosome_Drone
    return Chromosome_Drone


def _wing_reynolds_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    matrix, aligned_df = _parse_chromosome_matrix(df)
    Chromosome_Drone = _load_chromosome_drone_class()
    naca_df = pd.read_csv(NACA4_CSV_PATH)
    naca_df["airfoil_code"] = naca_df["airfoil"].astype(str).str.strip().str.zfill(4)
    re_nom_by_airfoil = naca_df.drop_duplicates("airfoil_code").set_index("airfoil_code")["ReNom"]

    phys = np.asarray([Chromosome_Drone.to_physical(row) for row in matrix], dtype=float)
    wing_span = phys[:, 0]
    wing_ar = phys[:, 1]
    chord = np.divide(
        wing_span,
        wing_ar,
        out=np.full_like(wing_span, np.nan, dtype=float),
        where=np.isfinite(wing_ar) & (np.abs(wing_ar) > 1e-12),
    )
    eff_v = pd.to_numeric(aligned_df["eff_v"], errors="coerce").to_numpy(dtype=float)
    reynolds = (AIR_DENSITY_KG_M3 * eff_v * chord) / AIR_DYNAMIC_VISCOSITY_KG_M_S
    wing_airfoil_codes = np.array(
        [
            str(Chromosome_Drone.naca_from_physical(row) or "").strip().zfill(4)
            for row in phys
        ],
        dtype=object,
    )
    wing_re_nom = pd.Series(wing_airfoil_codes).map(re_nom_by_airfoil).to_numpy(dtype=float)
    wing_re_delta = reynolds - wing_re_nom
    wing_re_ratio_pct = np.divide(
        100.0 * reynolds,
        wing_re_nom,
        out=np.full_like(reynolds, np.nan, dtype=float),
        where=np.isfinite(wing_re_nom) & (np.abs(wing_re_nom) > 1e-12),
    )

    out = aligned_df.copy()
    out["cost_of_transport"] = -pd.to_numeric(out["ff_1"], errors="coerce")
    out["wing_span"] = wing_span
    out["wing_aspect_ratio"] = wing_ar
    out["wing_chord"] = chord
    out["wing_airfoil_code"] = wing_airfoil_codes
    out["wing_re_nom_a1"] = wing_re_nom
    out["wing_reynolds_eff_v"] = reynolds
    out["wing_reynolds_minus_nom_a1"] = wing_re_delta
    out["wing_reynolds_over_nom_pct_a1"] = wing_re_ratio_pct
    return out


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


def _select_equally_spaced_generations(df: pd.DataFrame, n_generations: int = 6) -> list[int]:
    unique_generations = np.sort(pd.to_numeric(df["generation"], errors="coerce").dropna().unique().astype(int))
    if unique_generations.size == 0:
        raise ValueError("No valid generations available for genome PCA plot.")
    if unique_generations.size <= n_generations:
        return unique_generations.tolist()

    targets = np.linspace(unique_generations[0], unique_generations[-1], n_generations)
    selected: list[int] = []
    used: set[int] = set()

    for idx, target in enumerate(targets):
        if idx == n_generations - 1:
            gen = int(unique_generations[-1])
        else:
            order = np.argsort(np.abs(unique_generations - target))
            gen = None
            for candidate_idx in order:
                candidate = int(unique_generations[candidate_idx])
                if candidate not in used:
                    gen = candidate
                    break
            if gen is None:
                gen = int(unique_generations[order[0]])
        selected.append(gen)
        used.add(gen)

    return selected


def plot_genome_pca_generations(df: pd.DataFrame, output_path: Path) -> None:
    matrix, aligned_df = _parse_chromosome_matrix(df)
    mean, vt, explained_ratio = _pca_fit_transform_unique(matrix)
    pcs = (matrix - mean) @ vt.T
    pc1_var = float(explained_ratio[0]) if explained_ratio.size > 0 else 0.0
    pc2_var = float(explained_ratio[1]) if explained_ratio.size > 1 else 0.0
    pc3_var = float(explained_ratio[2]) if explained_ratio.size > 2 else 0.0
    pc4_var = float(explained_ratio[3]) if explained_ratio.size > 3 else 0.0
    pc5_var = float(explained_ratio[4]) if explained_ratio.size > 4 else 0.0

    generations = _select_equally_spaced_generations(aligned_df, n_generations=6)
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

    for ax in axes.flat[len(generations):]:
        ax.axis("off")

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


def plot_top10pct_gene_means(df: pd.DataFrame, output_path: Path) -> None:
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
            threshold = np.nanpercentile(finite_vals, 10)
            select_mask = finite_mask & (values <= threshold)
        else:
            threshold = np.nanpercentile(finite_vals, 90)
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
            label=f"Top 10% {FITNESS_LABELS[fitness]}",
        )
    ax.set_ylabel("Mean gene value")
    ax.grid(alpha=0.35)
    ax.legend(loc="best")
    ax.set_xticks(range(len(gene_cols)))
    ax.set_xticklabels(gene_cols, rotation=25, ha="right")
    ax.set_xlabel("Genome gene")
    ax.set_title("Top 10% Genome Means by Fitness", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_fitness_trends(
    df: pd.DataFrame,
    output_path: Path,
    raw_df: pd.DataFrame | None = None,
    summary_df: pd.DataFrame | None = None,
    generated_valid_df: pd.DataFrame | None = None,
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
                "selected_count_above_threshold": int(np.sum(gen_ff2 >= threshold)),
                "minimal_p": threshold,
            }
        )
    count_stats_df = pd.DataFrame(count_stats)
    generated_counts_map: dict[int, int] = {}
    if generated_valid_df is not None and not generated_valid_df.empty:
        gen_count_df = generated_valid_df.loc[:, ["generation", "ff_2"]].copy()
        gen_count_df = gen_count_df[np.isfinite(gen_count_df["ff_2"].to_numpy(dtype=float))]
        for generation in sorted(gen_count_df["generation"].unique()):
            gen_ff2 = gen_count_df.loc[gen_count_df["generation"] == generation, "ff_2"].to_numpy(dtype=float)
            threshold = float(threshold_map.get(generation, INVALID_PROGRESS_THRESHOLD))
            generated_counts_map[int(generation)] = int(np.sum(gen_ff2 >= threshold))
    count_stats_df["generated_count_above_threshold"] = count_stats_df["generation"].map(
        lambda g: generated_counts_map.get(int(g), np.nan)
    )
    bar_values = count_stats_df["generated_count_above_threshold"]
    bar_label = "generated valid >= minimal_p"
    bar_color = "#d62728"
    if not bar_values.notna().any():
        bar_values = count_stats_df["selected_count_above_threshold"]
        bar_label = "selected population >= minimal_p"
        bar_color = "#6baed6"
    count_ax.bar(
        count_stats_df["generation"],
        bar_values,
        width=0.7,
        color=bar_color,
        alpha=0.85,
        label=bar_label,
    )
    count_ax.set_title("Generated Valid Individuals Above Minimal Progress")
    count_ax.set_xlabel("Generation", fontsize=12)
    count_ax.set_ylabel("Count", fontsize=13)
    count_ax.grid(alpha=0.35)
    count_ax.set_xticks(generations)
    ylim_series = [bar_values.to_numpy(dtype=float)]
    max_count = float(np.nanmax(ylim_series[0])) if ylim_series and ylim_series[0].size else 1.0
    count_ax.set_ylim(0.0, max(1.0, max_count * 1.08))

    for ax in axes_flat[total_plots:]:
        ax.set_visible(False)

    axes_flat[0].legend(loc="best")
    fig.suptitle("Fitness Trends Across Generations (Selected Population)", fontsize=14, weight="bold")
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


def _non_dominated_sort_ranks(points: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    if n_points == 0:
        return np.array([], dtype=int)

    domination_counts = np.zeros(n_points, dtype=int)
    dominates_list = [[] for _ in range(n_points)]
    fronts: list[list[int]] = [[]]

    for i in range(n_points):
        for j in range(i + 1, n_points):
            i_dominates_j = np.all(points[i] >= points[j]) and np.any(points[i] > points[j])
            j_dominates_i = np.all(points[j] >= points[i]) and np.any(points[j] > points[i])

            if i_dominates_j:
                dominates_list[i].append(j)
                domination_counts[j] += 1
            elif j_dominates_i:
                dominates_list[j].append(i)
                domination_counts[i] += 1

    for i in range(n_points):
        if domination_counts[i] == 0:
            fronts[0].append(i)

    ranks = np.zeros(n_points, dtype=int)
    current_rank = 1
    current_front = fronts[0]
    while current_front:
        next_front: list[int] = []
        for idx in current_front:
            ranks[idx] = current_rank
            for dominated_idx in dominates_list[idx]:
                domination_counts[dominated_idx] -= 1
                if domination_counts[dominated_idx] == 0:
                    next_front.append(dominated_idx)
        current_front = next_front
        current_rank += 1

    return ranks


def plot_pareto_rank_distribution(df: pd.DataFrame, output_path: Path) -> None:
    if "generation" not in df.columns:
        raise ValueError("Missing 'generation' column for Pareto rank distribution plot.")

    work_df = df.copy()
    finite_mask = np.isfinite(work_df[FITNESS_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    work_df = work_df.loc[finite_mask].copy()
    generations = np.array(sorted(work_df["generation"].unique()), dtype=int)
    # Recompute local Pareto ranks from the raw DEAP fitness values.
    # ff_1 is already stored as -energy / -CoT in the optimization, so all
    # three objectives are maximized here and no sign flip is applied.
    rank_values = np.zeros(len(work_df), dtype=int)
    for generation in generations:
        gen_mask = work_df["generation"] == generation
        gen_points = work_df.loc[gen_mask, FITNESS_COLUMNS].to_numpy(dtype=float)
        rank_values[gen_mask.to_numpy()] = _non_dominated_sort_ranks(gen_points)
    work_df["local_pareto_rank"] = rank_values

    rank_counts = (
        work_df.groupby(["generation", "local_pareto_rank"], sort=True)
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )
    rank_counts = rank_counts.reindex(generations, fill_value=0)
    rank_columns = [int(col) for col in rank_counts.columns.tolist()]

    cmap = plt.get_cmap("viridis", max(len(rank_columns), 2))
    colors = [cmap(i) for i in range(len(rank_columns))]
    if colors:
        colors[0] = "#d62728"
    if len(colors) > 1:
        colors[1] = "#ff7f0e"
    if len(colors) > 2:
        colors[2] = "#f2c14e"

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    bottom = np.zeros(len(rank_counts), dtype=float)
    for idx, rank in enumerate(rank_columns):
        values = rank_counts[rank].to_numpy(dtype=float)
        ax.bar(
            generations,
            values,
            bottom=bottom,
            width=0.72,
            color=colors[idx],
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += values

    if rank_columns:
        front1_values = rank_counts[rank_columns[0]].to_numpy(dtype=float)
        for generation, value in zip(generations, front1_values):
            if value > 0:
                ax.text(
                    generation,
                    value * 0.5,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    weight="bold",
                )

    ax.set_title("Local Pareto Front Distribution (Selected Population)", fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Individuals")
    ax.set_xticks(generations)
    max_count = float(np.max(bottom)) if bottom.size else 1.0
    ax.set_ylim(0.0, max(1.0, max_count * 1.05))
    ax.grid(alpha=0.35, axis="y")
    legend_handles = [
        Patch(facecolor=colors[idx], edgecolor="white", label=("front 1 (Pareto)" if idx == 0 else f"front {idx + 1}"))
        for idx in range(len(rank_columns))
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        ncol=1,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_prev_current_generation_rank_distribution(df: pd.DataFrame, output_path: Path) -> None:
    if "generation" not in df.columns:
        raise ValueError("Missing 'generation' column for previous/current generation rank plot.")

    work_df = df.copy()
    finite_mask = np.isfinite(work_df[FITNESS_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    work_df = work_df.loc[finite_mask].copy()
    generations = np.array(sorted(work_df["generation"].unique()), dtype=int)
    if generations.size == 0:
        raise ValueError("No finite rows available for previous/current generation rank plot.")

    rows = []
    for generation in generations:
        if generation == int(generations[0]):
            pool_df = work_df.loc[work_df["generation"] == generation].copy()
        else:
            pool_df = work_df.loc[work_df["generation"].isin([generation - 1, generation])].copy()
        pool_points = pool_df[FITNESS_COLUMNS].to_numpy(dtype=float)
        pool_ranks = _non_dominated_sort_ranks(pool_points)
        pool_df["pair_generation_rank"] = pool_ranks
        current_df = pool_df.loc[pool_df["generation"] == generation].copy()
        rows.append(current_df.loc[:, ["generation", "pair_generation_rank"]])

    ranked_df = pd.concat(rows, ignore_index=True)
    rank_counts = (
        ranked_df.groupby(["generation", "pair_generation_rank"], sort=True)
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )
    rank_counts = rank_counts.reindex(generations, fill_value=0)
    rank_columns = [int(col) for col in rank_counts.columns.tolist()]

    cmap = plt.get_cmap("viridis", max(len(rank_columns), 2))
    colors = [cmap(i) for i in range(len(rank_columns))]
    if colors:
        colors[0] = "#d62728"
    if len(colors) > 1:
        colors[1] = "#ff7f0e"
    if len(colors) > 2:
        colors[2] = "#f2c14e"

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    bottom = np.zeros(len(rank_counts), dtype=float)
    for idx, rank in enumerate(rank_columns):
        values = rank_counts[rank].to_numpy(dtype=float)
        ax.bar(
            generations,
            values,
            bottom=bottom,
            width=0.72,
            color=colors[idx],
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += values

    if rank_columns:
        front1_values = rank_counts[rank_columns[0]].to_numpy(dtype=float)
        for generation, value in zip(generations, front1_values):
            if value > 0:
                ax.text(
                    generation,
                    value * 0.5,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    weight="bold",
                )

    ax.set_title("Generated Individuals Ranked In Pool (g-1 + g)", fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Individuals generated in current generation")
    ax.set_xticks(generations)
    max_count = float(np.max(bottom)) if bottom.size else 1.0
    ax.set_ylim(0.0, max(1.0, max_count * 1.05))
    ax.grid(alpha=0.35, axis="y")
    legend_handles = [
        Patch(
            facecolor=colors[idx],
            edgecolor="white",
            label=("front 1 (in g-1 + g pool)" if idx == 0 else f"front {idx + 1}"),
        )
        for idx in range(len(rank_columns))
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        ncol=1,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_newcomer_rank_distribution(df: pd.DataFrame, output_path: Path) -> None:
    if "generation" not in df.columns:
        raise ValueError("Missing 'generation' column for newcomer rank plot.")

    work_df = df.copy()
    finite_mask = np.isfinite(work_df[FITNESS_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    work_df = work_df.loc[finite_mask].copy()
    generations = np.array(sorted(work_df["generation"].unique()), dtype=int)
    if generations.size == 0:
        raise ValueError("No finite rows available for newcomer rank plot.")

    if "uid" not in work_df.columns:
        raise ValueError("Missing 'uid' column for newcomer rank plot.")

    rows = []
    for generation in generations:
        current_df = work_df.loc[work_df["generation"] == generation].copy()
        if generation == int(generations[0]):
            newcomer_df = current_df.copy()
            pool_df = current_df.copy()
        else:
            prev_df = work_df.loc[work_df["generation"] == generation - 1].copy()
            prev_uids = set(prev_df["uid"].tolist())
            newcomer_df = current_df.loc[~current_df["uid"].isin(prev_uids)].copy()
            pool_df = work_df.loc[work_df["generation"].isin([generation - 1, generation])].copy()

        if newcomer_df.empty:
            continue

        pool_points = pool_df[FITNESS_COLUMNS].to_numpy(dtype=float)
        pool_df = pool_df.copy()
        pool_df["pair_generation_rank"] = _non_dominated_sort_ranks(pool_points)
        newcomer_ranked = pool_df.loc[pool_df["uid"].isin(newcomer_df["uid"])].copy()
        rows.append(newcomer_ranked.loc[:, ["generation", "pair_generation_rank"]])

    if not rows:
        raise ValueError("No newcomer individuals found for newcomer rank plot.")

    ranked_df = pd.concat(rows, ignore_index=True)
    rank_counts = (
        ranked_df.groupby(["generation", "pair_generation_rank"], sort=True)
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )
    rank_counts = rank_counts.reindex(generations, fill_value=0)
    rank_columns = [int(col) for col in rank_counts.columns.tolist()]

    cmap = plt.get_cmap("viridis", max(len(rank_columns), 2))
    colors = [cmap(i) for i in range(len(rank_columns))]
    if colors:
        colors[0] = "#d62728"
    if len(colors) > 1:
        colors[1] = "#ff7f0e"
    if len(colors) > 2:
        colors[2] = "#f2c14e"

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    bottom = np.zeros(len(rank_counts), dtype=float)
    for idx, rank in enumerate(rank_columns):
        values = rank_counts[rank].to_numpy(dtype=float)
        ax.bar(
            generations,
            values,
            bottom=bottom,
            width=0.72,
            color=colors[idx],
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += values

    if rank_columns:
        front1_values = rank_counts[rank_columns[0]].to_numpy(dtype=float)
        for generation, value in zip(generations, front1_values):
            if value > 0:
                ax.text(
                    generation,
                    value * 0.5,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    weight="bold",
                )

    newcomer_counts = rank_counts.sum(axis=1).to_numpy(dtype=float)
    ax.set_title("Newcomer Rank In Pool (g-1 + g)", fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("New individuals in current generation")
    ax.set_xticks(generations)
    ax.set_ylim(0.0, max(1.0, float(np.max(newcomer_counts)) * 1.08))
    ax.grid(alpha=0.35, axis="y")
    legend_handles = [
        Patch(
            facecolor=colors[idx],
            edgecolor="white",
            label=("front 1 (in g-1 + g pool)" if idx == 0 else f"front {idx + 1}"),
        )
        for idx in range(len(rank_columns))
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        ncol=1,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _genealogy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"generation", "uid", "parent_uid_a", "parent_uid_b"}
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns for genealogy plots: {missing_list}")

    work_df = df.copy()
    finite_mask = np.isfinite(work_df[FITNESS_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    work_df = work_df.loc[finite_mask].copy()
    work_df = work_df.sort_values(["generation", "uid"]).drop_duplicates(subset="uid", keep="first")
    if work_df.empty:
        raise ValueError("No finite rows available for genealogy plots.")

    work_df["uid"] = work_df["uid"].astype(int)
    work_df["generation"] = work_df["generation"].astype(int)
    work_df["parent_uid_a"] = work_df["parent_uid_a"].fillna(-1).astype(int)
    work_df["parent_uid_b"] = work_df["parent_uid_b"].fillna(-1).astype(int)
    if "pareto_rank" in work_df.columns:
        work_df["pareto_rank"] = pd.to_numeric(work_df["pareto_rank"], errors="coerce").fillna(0).astype(int)
    else:
        work_df["pareto_rank"] = 0
    return work_df


def _founder_contribution_table(genealogy_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    founders = genealogy_df.loc[
        (genealogy_df["generation"] == int(genealogy_df["generation"].min()))
        | ((genealogy_df["parent_uid_a"] < 0) & (genealogy_df["parent_uid_b"] < 0)),
        "uid",
    ].drop_duplicates().astype(int).tolist()
    if not founders:
        raise ValueError("No founder individuals available for genealogy plots.")

    founder_index = {uid: idx for idx, uid in enumerate(founders)}
    uid_to_vec: dict[int, np.ndarray] = {}
    for row in genealogy_df.itertuples(index=False):
        uid = int(row.uid)
        parent_ids = [int(row.parent_uid_a), int(row.parent_uid_b)]
        valid_parents = [pid for pid in parent_ids if pid >= 0 and pid in uid_to_vec]
        if not valid_parents:
            vec = np.zeros(len(founders), dtype=float)
            vec[founder_index[uid]] = 1.0
            uid_to_vec[uid] = vec
            continue
        vec = np.zeros(len(founders), dtype=float)
        weight = 1.0 / float(len(valid_parents))
        for pid in valid_parents:
            vec += uid_to_vec[pid] * weight
        total = float(np.sum(vec))
        if total > 0.0:
            vec /= total
        uid_to_vec[uid] = vec

    contrib_columns = [f"founder_uid_{uid}" for uid in founders]
    contrib_df = pd.DataFrame.from_dict(uid_to_vec, orient="index", columns=contrib_columns)
    contrib_df.index.name = "uid"
    return contrib_df, founders


def _primary_parent_forest(genealogy_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, list[int]], dict[int, int]]:
    parent_map: dict[int, int] = {}
    for row in genealogy_df.itertuples(index=False):
        uid = int(row.uid)
        parent_uid = int(row.parent_uid_a) if int(row.parent_uid_a) >= 0 else int(row.parent_uid_b)
        parent_map[uid] = parent_uid if parent_uid >= 0 else -1

    founder_map: dict[int, int] = {}

    def founder_of(uid: int) -> int:
        if uid in founder_map:
            return founder_map[uid]
        parent_uid = parent_map.get(uid, -1)
        if parent_uid < 0:
            founder_map[uid] = uid
            return uid
        founder_map[uid] = founder_of(parent_uid)
        return founder_map[uid]

    for uid in parent_map:
        founder_of(uid)

    children_map: dict[int, list[int]] = {uid: [] for uid in parent_map}
    for uid, parent_uid in parent_map.items():
        if parent_uid >= 0 and parent_uid in children_map:
            children_map[parent_uid].append(uid)
    for uid in children_map:
        children_map[uid] = sorted(children_map[uid])

    return parent_map, children_map, founder_map


def _descendant_counts_from_children(children_map: dict[int, list[int]]) -> pd.Series:
    memo: dict[int, set[int]] = {}

    def collect_descendants(uid: int) -> set[int]:
        if uid in memo:
            return memo[uid]
        descendants: set[int] = set()
        for child_uid in children_map.get(uid, []):
            descendants.add(child_uid)
            descendants |= collect_descendants(child_uid)
        memo[uid] = descendants
        return descendants

    counts = {uid: len(collect_descendants(int(uid))) for uid in children_map}
    return pd.Series(counts, name="descendant_count", dtype=float)


def _scalar_fitness_proxy(df: pd.DataFrame) -> pd.Series:
    points = df[FITNESS_COLUMNS].to_numpy(dtype=float).copy()
    points[:, 1] = -points[:, 1]
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    ranges = np.where((maxs - mins) == 0.0, 1.0, maxs - mins)
    norm = np.clip((points - mins) / ranges, 0.0, 1.0)
    return pd.Series(np.mean(norm, axis=1), index=df.index, dtype=float, name="fitness_proxy")


def plot_phylogenetic_tree_like(
    df: pd.DataFrame,
    output_path: Path,
    *,
    top_lineages: int = 8,
) -> None:
    genealogy_df = _genealogy_dataframe(df)
    parent_map, children_map, founder_map = _primary_parent_forest(genealogy_df)
    desc_counts = _descendant_counts_from_children(children_map)
    work_df = genealogy_df.copy()
    work_df["dominant_founder_uid"] = work_df["uid"].map(lambda uid: founder_map[int(uid)]).astype(int)
    work_df["fitness_proxy"] = _scalar_fitness_proxy(work_df)
    work_df = work_df.join(desc_counts, on="uid")
    work_df["descendant_count"] = work_df["descendant_count"].fillna(0.0)

    final_generation = int(work_df["generation"].max())
    final_lineage = (
        work_df.loc[work_df["generation"] == final_generation, "dominant_founder_uid"]
        .value_counts()
        .sort_values(ascending=False)
    )
    top_founders = final_lineage.head(max(1, top_lineages)).index.astype(int).tolist()
    work_df = work_df.loc[work_df["dominant_founder_uid"].isin(top_founders)].copy()
    uid_rows = {int(row.uid): row for row in work_df.itertuples(index=False)}
    selected_children_map = {uid: [child for child in children_map.get(uid, []) if child in uid_rows] for uid in uid_rows}

    uid_to_y: dict[int, float] = {}
    lineage_centers: dict[int, float] = {}
    next_y = 0.0

    def assign_subtree_y(uid: int) -> float:
        nonlocal next_y
        kids = selected_children_map.get(uid, [])
        if not kids:
            y_val = next_y
            uid_to_y[uid] = y_val
            next_y += 1.0
            return y_val
        child_ys = [assign_subtree_y(child_uid) for child_uid in kids]
        y_val = float(np.mean(child_ys))
        uid_to_y[uid] = y_val
        return y_val

    root_order = sorted(top_founders, key=lambda uid: (-int(final_lineage.get(uid, 0)), int(uid)))
    for founder_uid in root_order:
        start_y = next_y
        assign_subtree_y(founder_uid)
        lineage_centers[founder_uid] = 0.5 * (start_y + next_y - 1.0) if next_y > start_y else start_y
        next_y += 2.0

    sizes = 16.0 + 180.0 * np.sqrt(work_df["descendant_count"].to_numpy(dtype=float) + 1.0) / np.sqrt(
        float(max(work_df["descendant_count"].max(), 1.0)) + 1.0
    )
    alphas = 0.15 + 0.85 * work_df["fitness_proxy"].to_numpy(dtype=float)

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12.8, 8.8))

    for row in work_df.itertuples(index=False):
        child_uid = int(row.uid)
        child_x = float(row.generation)
        child_y = uid_to_y[child_uid]
        parent_uid = parent_map.get(child_uid, -1)
        if parent_uid < 0 or parent_uid not in uid_to_y:
            continue
        parent_row = uid_rows[parent_uid]
        parent_x = float(parent_row.generation)
        parent_y = uid_to_y[parent_uid]
        edge_alpha = 0.06 + 0.34 * float(row.fitness_proxy)
        ax.plot(
            [parent_x, child_x],
            [parent_y, child_y],
            color="black",
            alpha=edge_alpha,
            linewidth=0.8,
            zorder=1,
        )

    ax.scatter(
        work_df["generation"].to_numpy(dtype=float),
        np.array([uid_to_y[int(uid)] for uid in work_df["uid"].tolist()], dtype=float),
        s=sizes,
        c="black",
        alpha=alphas,
        edgecolors="none",
        zorder=3,
    )

    for founder_uid in root_order:
        ax.text(
            final_generation + 0.55,
            lineage_centers[founder_uid],
            f"uid {founder_uid} ({int(final_lineage.get(founder_uid, 0))})",
            va="center",
            ha="left",
            fontsize=9,
        )

    caption = (
        "Each point is one individual from one of the strongest surviving lineages in the final generation, traced "
        "through a primary-parent genealogy to mimic the lineage trees in Fig. 2c of the paper. Point size is "
        "proportional to the number of descendants in this primary-parent forest; point opacity is a scalar fitness "
        "proxy obtained by averaging normalized speed, efficiency, and progress. Because this NSGA-II run uses "
        "two-parent crossover, following the primary recorded parent is an approximation introduced to recover a "
        "tree-like representation comparable to the paper."
    )

    ax.set_title("Phylogenetic Tree of Top Surviving Lineages", fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Branch ordering")
    generations = np.array(sorted(work_df["generation"].unique()), dtype=int)
    ax.set_xticks(generations)
    ax.grid(alpha=0.20, axis="x")
    ax.grid(False, axis="y")
    ax.set_yticks([])
    fig.text(
        0.03,
        0.02,
        caption,
        ha="left",
        va="bottom",
        fontsize=9,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#bbbbbb", alpha=0.95),
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_two_parent_genealogy(
    df: pd.DataFrame,
    output_path: Path,
    *,
    top_lineages: int = 8,
) -> None:
    genealogy_df = _genealogy_dataframe(df)
    parent_map, children_map, founder_map = _primary_parent_forest(genealogy_df)
    desc_counts = _descendant_counts_from_children(children_map)
    work_df = genealogy_df.copy()
    work_df["dominant_founder_uid"] = work_df["uid"].map(lambda uid: founder_map[int(uid)]).astype(int)
    work_df["fitness_proxy"] = _scalar_fitness_proxy(work_df)
    work_df = work_df.join(desc_counts, on="uid")
    work_df["descendant_count"] = work_df["descendant_count"].fillna(0.0)

    final_generation = int(work_df["generation"].max())
    final_lineage = (
        work_df.loc[work_df["generation"] == final_generation, "dominant_founder_uid"]
        .value_counts()
        .sort_values(ascending=False)
    )
    top_founders = final_lineage.head(max(1, top_lineages)).index.astype(int).tolist()
    work_df = work_df.loc[work_df["dominant_founder_uid"].isin(top_founders)].copy()

    root_order = sorted(top_founders, key=lambda uid: (-int(final_lineage.get(uid, 0)), int(uid)))
    founder_offset = {uid: idx for idx, uid in enumerate(root_order)}
    uid_to_y: dict[int, float] = {}
    lineage_centers: dict[int, float] = {}
    block_gap = 3.0

    for founder_uid in root_order:
        founder_df = work_df.loc[work_df["dominant_founder_uid"] == founder_uid].copy()
        y_base = founder_offset[founder_uid] * (founder_df["generation"].max() + 8.0 + block_gap)
        local_positions = {}
        for generation in sorted(founder_df["generation"].unique()):
            gen_df = founder_df.loc[founder_df["generation"] == generation].sort_values(
                ["descendant_count", "fitness_proxy", "uid"],
                ascending=[False, False, True],
            )
            for idx, row in enumerate(gen_df.itertuples(index=False)):
                local_positions[int(row.uid)] = y_base + idx
        uid_to_y.update(local_positions)
        if local_positions:
            ys = np.array(list(local_positions.values()), dtype=float)
            lineage_centers[founder_uid] = float(np.mean([np.min(ys), np.max(ys)]))
        else:
            lineage_centers[founder_uid] = y_base

    sizes = 12.0 + 120.0 * np.sqrt(work_df["descendant_count"].to_numpy(dtype=float) + 1.0) / np.sqrt(
        float(max(work_df["descendant_count"].max(), 1.0)) + 1.0
    )
    alphas = 0.18 + 0.82 * work_df["fitness_proxy"].to_numpy(dtype=float)
    uid_rows = {int(row.uid): row for row in work_df.itertuples(index=False)}

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(13.5, 9.0))

    for row in work_df.itertuples(index=False):
        child_uid = int(row.uid)
        child_x = float(row.generation)
        child_y = uid_to_y[child_uid]
        for parent_uid, linestyle, linewidth, alpha_scale in (
            (int(row.parent_uid_a), "-", 0.95, 1.0),
            (int(row.parent_uid_b), "--", 0.8, 0.72),
        ):
            if parent_uid < 0 or parent_uid not in uid_to_y:
                continue
            parent_row = uid_rows[parent_uid]
            parent_x = float(parent_row.generation)
            parent_y = uid_to_y[parent_uid]
            edge_alpha = (0.06 + 0.30 * float(row.fitness_proxy)) * alpha_scale
            ax.plot(
                [parent_x, child_x],
                [parent_y, child_y],
                color="black",
                alpha=edge_alpha,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=1,
            )

    ax.scatter(
        work_df["generation"].to_numpy(dtype=float),
        np.array([uid_to_y[int(uid)] for uid in work_df["uid"].tolist()], dtype=float),
        s=sizes,
        c="black",
        alpha=alphas,
        edgecolors="none",
        zorder=3,
    )

    for founder_uid in root_order:
        ax.text(
            final_generation + 0.6,
            lineage_centers[founder_uid],
            f"uid {founder_uid} ({int(final_lineage.get(founder_uid, 0))})",
            va="center",
            ha="left",
            fontsize=9,
        )

    caption = (
        "This plot shows the real logged genealogy for the top surviving final lineages in nsga_GP. Each point is one "
        "individual; solid edges connect parent A and dashed edges connect parent B, so most offspring visibly have "
        "two parents. Point size is proportional to the number of descendants in the primary-parent forest used only "
        "for sizing, while point opacity is a scalar fitness proxy from normalized speed, efficiency, and progress. "
        "Unlike the paper-style tree, this figure is not forced into a single-parent lineage interpretation."
    )

    ax.set_title("Two-Parent Genealogy of Top Surviving Lineages", fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Lineage blocks")
    generations = np.array(sorted(work_df["generation"].unique()), dtype=int)
    ax.set_xticks(generations)
    ax.grid(alpha=0.20, axis="x")
    ax.grid(False, axis="y")
    ax.set_yticks([])
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black", markersize=6, label="node = individual"),
        Line2D([0, 1], [0, 0], linestyle="-", color="black", linewidth=1.0, label="solid edge = parent A"),
        Line2D([0, 1], [0, 0], linestyle="--", color="black", linewidth=1.0, label="dashed edge = parent B"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")
    fig.text(
        0.03,
        0.02,
        caption,
        ha="left",
        va="bottom",
        fontsize=9,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#bbbbbb", alpha=0.95),
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_fractional_lineage_muller(
    df: pd.DataFrame,
    output_path: Path,
    *,
    top_lineages: int = 10,
) -> None:
    genealogy_df = _genealogy_dataframe(df)
    _, _, founder_map = _primary_parent_forest(genealogy_df)
    work_df = genealogy_df.copy()
    work_df["dominant_founder_uid"] = work_df["uid"].map(lambda uid: founder_map[int(uid)]).astype(int)
    work_df["fitness_proxy"] = _scalar_fitness_proxy(work_df)
    generations = np.array(sorted(work_df["generation"].unique()), dtype=int)
    lineage_counts = (
        work_df.groupby(["generation", "dominant_founder_uid"])
        .size()
        .unstack(fill_value=0.0)
        .reindex(generations, fill_value=0.0)
    )
    pop_sizes = work_df.groupby("generation").size().reindex(generations, fill_value=1).astype(float)
    lineage_frac = lineage_counts.div(pop_sizes, axis=0)

    final_share = lineage_frac.iloc[-1].sort_values(ascending=False)
    top_founders = final_share.head(max(1, top_lineages)).index.astype(int).tolist()
    plot_df = lineage_frac.loc[:, top_founders].copy()
    plot_df.columns = [f"founder uid {uid}" for uid in top_founders]

    cmap = plt.get_cmap("tab10", max(len(top_founders), 1))
    colors = [cmap(idx % cmap.N) for idx in range(len(top_founders))]
    labels = plot_df.columns.tolist()

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.stackplot(
        generations,
        *[plot_df[col].to_numpy(dtype=float) for col in plot_df.columns],
        colors=colors,
        alpha=0.95,
        linewidth=0.5,
        edgecolor="white",
    )
    cumulative = np.zeros(len(generations), dtype=float)
    band_centers = []
    for col in plot_df.columns:
        vals = plot_df[col].to_numpy(dtype=float)
        band_centers.append(cumulative + 0.5 * vals)
        cumulative += vals
    for idx, founder_uid in enumerate(top_founders):
        if plot_df.iloc[-1, idx] <= 0.0:
            continue
        ax.text(
            generations[-1] + 0.35,
            band_centers[idx][-1],
            str(idx + 1),
            va="center",
            ha="left",
            fontsize=9,
            weight="bold",
        )

    caption = (
        "Muller-style diagram approximating Fig. 2f of the paper. Each colored band is one of the top final surviving "
        "lineages, defined here by the founder reached through the primary recorded parent at each reproduction event; "
        "band thickness is that lineage's population share in each generation. Only the tracked top final lineages are "
        "drawn, so white space corresponds to all remaining lineages outside this final top set. Numbers at the right "
        "identify the largest final lineages. The paper can additionally mark successful topology-changing mutations "
        "with stars, but those events are not explicitly logged in the current NSGA-II output, so this plot focuses on "
        "lineage abundance over time."
    )

    ax.set_title("Muller Diagram of Top Surviving Lineages", fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population share")
    ax.set_xticks(generations)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.text(
        0.03,
        0.02,
        caption,
        ha="left",
        va="bottom",
        fontsize=9,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#bbbbbb", alpha=0.95),
    )
    fig.tight_layout(rect=[0.0, 0.08, 0.84, 1.0])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    csv_df = plot_df.copy()
    csv_df.insert(0, "generation", generations)
    csv_df.to_csv(output_path.with_suffix(".csv"), index=False)


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


def plot_cot_vs_wing_reynolds(df: pd.DataFrame, output_path: Path) -> None:
    re_df = _wing_reynolds_dataframe(df)
    front_mask = pareto_mask_finite(re_df[FITNESS_COLUMNS].to_numpy(dtype=float))
    re_df = re_df.loc[front_mask].copy()
    plot_df = re_df[
        [
            "cost_of_transport",
            "wing_reynolds_over_nom_pct_a1",
            "eff_v",
            "wing_airfoil_code",
            "wing_span",
            "wing_aspect_ratio",
            "wing_chord",
            "wing_re_nom_a1",
            "wing_reynolds_eff_v",
            "wing_reynolds_minus_nom_a1",
            "exp_name",
        ]
    ].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["cost_of_transport", "wing_reynolds_over_nom_pct_a1", "eff_v"])
    if plot_df.empty:
        raise ValueError("No valid points available for CoT vs Reynolds-limit plot.")

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    scatter = ax.scatter(
        plot_df["cost_of_transport"],
        plot_df["wing_reynolds_over_nom_pct_a1"],
        c=plot_df["eff_v"],
        cmap="viridis",
        s=30,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel("Cost Of Transport")
    ax.set_ylabel("Wing Reynolds / Re_nom [%]")
    ax.set_title("Final Global Pareto Front: CoT vs Wing Reynolds / Re_nom [%]")
    ax.set_ylim(80.0, 300.0)
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("eff_v")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    plot_df.sort_values(["cost_of_transport", "wing_reynolds_over_nom_pct_a1", "eff_v"]).to_csv(csv_path, index=False)


def plot_cot_vs_wing_chord(df: pd.DataFrame, output_path: Path) -> None:
    re_df = _wing_reynolds_dataframe(df)
    front_mask = pareto_mask_finite(re_df[FITNESS_COLUMNS].to_numpy(dtype=float))
    re_df = re_df.loc[front_mask].copy()
    plot_df = re_df[
        [
            "cost_of_transport",
            "wing_chord",
            "eff_v",
            "wing_airfoil_code",
            "wing_span",
            "wing_aspect_ratio",
            "wing_re_nom_a1",
            "wing_reynolds_eff_v",
            "wing_reynolds_over_nom_pct_a1",
            "exp_name",
        ]
    ].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["cost_of_transport", "wing_chord", "eff_v"])
    if plot_df.empty:
        raise ValueError("No valid points available for CoT vs wing-chord plot.")

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    scatter = ax.scatter(
        plot_df["cost_of_transport"],
        plot_df["wing_chord"],
        c=plot_df["eff_v"],
        cmap="viridis",
        s=30,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel("Cost Of Transport")
    ax.set_ylabel("Wing chord [m]")
    ax.set_title("Final Global Pareto Front: CoT vs Wing Chord")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("eff_v")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    plot_df.sort_values(["cost_of_transport", "wing_chord", "eff_v"]).to_csv(csv_path, index=False)


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

    fig.suptitle("Evolutionary Metrics Across Generations (Selected Population)", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_dominance_solutions(
    df: pd.DataFrame,
    output_path: Path,
    *,
    summary_df: pd.DataFrame | None = None,
) -> None:
    if "generation" not in df.columns:
        raise ValueError("Missing 'generation' column for dominance plot.")
    all_generations = np.array(sorted(df["generation"].unique()), dtype=int)

    work_df = df.copy()
    finite_mask = np.isfinite(work_df[FITNESS_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    work_df = work_df.loc[finite_mask].copy()
    genome_name = _extract_genome_name_series(work_df)
    work_df["_dominance_key"] = genome_name
    missing_key_mask = work_df["_dominance_key"].isna()
    if missing_key_mask.any():
        rounded_points = work_df.loc[missing_key_mask, FITNESS_COLUMNS].round(8)
        work_df.loc[missing_key_mask, "_dominance_key"] = rounded_points.apply(
            lambda row: tuple(row[col] for col in FITNESS_COLUMNS),
            axis=1,
        )

    # Keep the first occurrence of each unique solution; global front size is computed
    # on all unique solutions discovered up to each generation, not by summing local fronts.
    unique_df = (
        work_df.sort_values(["generation", "uid"] if "uid" in work_df.columns else ["generation"])
        .drop_duplicates(subset="_dominance_key", keep="first")
        .copy()
    )
    unique_df["discovery_generation"] = unique_df["generation"].astype(int)

    final_points = unique_df[FITNESS_COLUMNS].to_numpy(dtype=float)
    final_front_mask = pareto_mask(final_points) if final_points.size else np.array([], dtype=bool)
    final_global_front_keys = set(unique_df.loc[final_front_mask, "_dominance_key"].tolist())

    rows = []
    cumulative_seen_keys: set[object] = set()
    cumulative_global_front_keys: set[object] = set()
    for generation in all_generations:
        eligible = unique_df.loc[unique_df["discovery_generation"] <= generation].copy()
        if eligible.empty:
            rows.append(
                {
                    "generation": generation,
                    "new_unique_solutions": 0,
                    "global_pareto_front_size": 0,
                    "new_global_front_solutions": 0,
                }
            )
            continue

        points = eligible[FITNESS_COLUMNS].to_numpy(dtype=float)
        front_mask = pareto_mask(points)
        front_df = eligible.loc[front_mask].copy()
        front_keys = set(front_df["_dominance_key"].tolist())
        discovered_now = set(
            eligible.loc[eligible["discovery_generation"] == generation, "_dominance_key"].tolist()
        )
        new_front_now = front_keys - cumulative_global_front_keys
        final_front_now = new_front_now & final_global_front_keys

        rows.append(
            {
                "generation": generation,
                "new_unique_solutions": len(discovered_now - cumulative_seen_keys),
                "global_pareto_front_size": len(front_keys),
                "new_global_front_solutions": len(new_front_now),
                "new_solutions_in_final_global_front": len(final_front_now),
            }
        )
        cumulative_seen_keys |= discovered_now
        cumulative_global_front_keys = front_keys

    plot_df = pd.DataFrame(rows)

    generations = plot_df["generation"].to_numpy(dtype=int)
    values = plot_df["global_pareto_front_size"].to_numpy(dtype=float)
    discovered_values = plot_df["new_global_front_solutions"].to_numpy(dtype=float)
    final_values = plot_df["new_solutions_in_final_global_front"].to_numpy(dtype=float)

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(
        generations,
        discovered_values,
        width=0.65,
        color="#9ecae1",
        alpha=0.75,
        label="new solutions entering global Pareto front",
        zorder=1,
    )
    ax.bar(
        generations,
        final_values,
        width=0.42,
        color="#2ca02c",
        alpha=0.90,
        label="of those, still in final global Pareto front",
        zorder=2,
    )
    ax.plot(
        generations,
        values,
        marker="o",
        linewidth=2.4,
        markersize=5.5,
        color="#d62728",
        label="global Pareto front size",
        zorder=3,
    )
    ax.fill_between(generations, 0.0, values, color="#fcae91", alpha=0.20, zorder=2)
    ax.set_title("Global Pareto Front Growth Across Generations", fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Count")
    ax.set_xticks(generations)
    ax.set_ylim(_scaled_ylim(np.concatenate([values, discovered_values, final_values]), padding_ratio=0.12))
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
    nsga_generated_df = df_agg_raw
    nsga_generated_valid = df
    nsga_csv_path = source_csv.parent / "nsga.csv"
    if nsga_csv_path.exists():
        nsga_full_df = load_nsga_csv(nsga_csv_path)
        nsga_generated_df = filter_to_agg(nsga_full_df)
        nsga_generated_valid = filter_sentinels(nsga_generated_df, reference_df=nsga_full_df)

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
        generated_valid_df=nsga_generated_valid,
    )

    if args.generation is None:
        generation = int(df["generation"].max())
    else:
        generation = args.generation
    plot_pareto_fronts(nsga_generated_valid, output_dir / "pareto_fronts.png", generation)
    plot_pareto_front_3d(nsga_generated_valid, output_dir / "pareto_front_3d.html")
    plot_cot_vs_wing_reynolds(nsga_generated_valid, output_dir / "cot_vs_wing_reynolds.png")
    plot_cot_vs_wing_chord(nsga_generated_valid, output_dir / "cot_vs_wing_chord.png")
    export_pareto_csv(nsga_generated_valid, output_dir / "pareto.csv")
    metrics_df = compute_evolutionary_metrics(df, output_dir / "evolutionary_metrics.csv")
    plot_evolutionary_metrics(metrics_df, output_dir / "evolutionary_metrics.png")
    plot_dominance_solutions(
        nsga_generated_valid,
        output_dir / "dominance_solutions.png",
        summary_df=summary_df,
    )
    plot_pareto_rank_distribution(df_agg_raw, output_dir / "pareto_rank_distribution.png")
    plot_prev_current_generation_rank_distribution(
        nsga_generated_valid,
        output_dir / "pair_generation_rank_distribution.png",
    )
    plot_newcomer_rank_distribution(
        nsga_generated_valid,
        output_dir / "newcomer_rank_distribution.png",
    )
    plot_phylogenetic_tree_like(
        nsga_generated_valid,
        output_dir / "phylogenetic_tree_like.png",
    )
    plot_two_parent_genealogy(
        nsga_generated_valid,
        output_dir / "two_parent_genealogy.png",
    )
    plot_fractional_lineage_muller(
        nsga_generated_valid,
        output_dir / "lineage_muller_fractional.png",
    )
    plot_genome_pca_generations(df, output_dir / "genome_pca_generations.png")
    plot_genome_fitness_correlation(df, output_dir / "genome_fitness_correlation.png")
    plot_top10pct_gene_means(df, output_dir / "genome_top10pct_gene_means.png")


if __name__ == "__main__":
    main()
