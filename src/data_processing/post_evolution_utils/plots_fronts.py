from __future__ import annotations

from .common import *  # noqa: F401,F403
from .descriptors import *  # noqa: F401,F403

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
    generated_valid_df: pd.DataFrame | None = None,
    bix3_point: np.ndarray | None = None,
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
        figsize=(8.6 * ncols, 4.4 * nrows),
        sharex=True,
    )
    axes_array = np.atleast_1d(axes)
    axes_flat = axes_array.flatten()
    for ax, fitness in zip(axes_flat, fitness_cols):
        if raw_df is not None:
            temp = _fitness_frame_for_plot(raw_df, fitness, is_raw=True)
        else:
            temp = _fitness_frame_for_plot(df, fitness, is_raw=False)

        stats = temp.groupby("generation", sort=True)[fitness].agg(
            ["mean", "median", "std", "max", "min"]
        ).reset_index()
        stats["std"] = stats["std"].fillna(0.0)
        ylim_values = temp[fitness].to_numpy()
        band_lower = stats["mean"] - stats["std"]
        band_upper = stats["mean"] + stats["std"]
        extreme_col = "min" if fitness == "ff_1" else "max"
        extreme_series = stats[extreme_col]
        extreme_label = "min" if fitness == "ff_1" else "best"

        ax.set_facecolor("#fbfbfd")
        ax.fill_between(
            stats["generation"],
            band_lower,
            band_upper,
            color="#a8c7ff",
            alpha=0.28,
            label="mean ± std",
        )
        ax.plot(
            stats["generation"],
            stats["mean"],
            color="#1d4ed8",
            linewidth=2.8,
            marker="o",
            markersize=4.2,
            label="mean",
        )
        ax.plot(
            stats["generation"],
            stats["median"],
            color="#16a34a",
            linewidth=2.4,
            linestyle="-.",
            label="median",
        )
        ax.plot(
            stats["generation"],
            extreme_series,
            color="#d97706",
            linewidth=2.0,
            linestyle="--",
            label=extreme_label,
        )
        if bix3_point is not None and fitness in FITNESS_COLUMNS:
            bix3_value = float(bix3_point[FITNESS_COLUMNS.index(fitness)])
            ax.axhline(
                bix3_value,
                color=PLOT_COLOR_CYCLE["baseline"],
                linewidth=2.0,
                linestyle=(0, (6, 3)),
                label="Baseline drone",
            )

        ax.set_title(FITNESS_LABELS[fitness], fontsize=15, weight="bold")
        ax.set_xlabel("Generation", fontsize=14)
        ax.set_ylabel(FITNESS_LABELS[fitness], fontsize=15)
        ax.tick_params(axis="x", labelsize=12.5)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="y", alpha=0.32, linestyle="--", linewidth=0.8)
        ax.grid(axis="x", alpha=0.12, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(
            _fitness_trend_ylim(
                stats["median"].to_numpy(dtype=float),
                np.asarray(extreme_series, dtype=float),
                lower_from_best=(fitness == "ff_1"),
                margin_ratio=0.05,
            )
        )
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
    count_ax.set_facecolor("#fbfbfd")
    count_ax.set_title("# of drones achieving minimal Progress", fontsize=15, weight="bold")
    count_ax.set_xlabel("Generation", fontsize=14)
    count_ax.set_ylabel("Count", fontsize=15)
    count_ax.tick_params(axis="x", labelsize=12.5)
    count_ax.tick_params(axis="y", labelsize=13)
    count_ax.grid(axis="y", alpha=0.32, linestyle="--", linewidth=0.8)
    count_ax.grid(axis="x", alpha=0.12, linestyle=":")
    count_ax.spines["top"].set_visible(False)
    count_ax.spines["right"].set_visible(False)
    count_ax.set_xticks(generations)
    ylim_series = [bar_values.to_numpy(dtype=float)]
    max_count = float(np.nanmax(ylim_series[0])) if ylim_series and ylim_series[0].size else 1.0
    count_ax.set_ylim(0.0, max(1.0, max_count * 1.08))

    for ax in axes_flat[total_plots:]:
        ax.set_visible(False)

    axes_flat[0].legend(
        loc="best",
        fontsize=13,
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.96,
    )
    fig.suptitle(
        "Fitness Trends Across Generations",
        fontsize=20,
        weight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    _write_plot_note(
        output_path,
        title="Fitness trends across generations",
        importance=(
            "This is the first sanity-check plot for an evolutionary run: it shows whether the selected population is actually moving "
            "generation after generation on the objectives that define the search, and whether that movement is beating the BIX3 reference."
        ),
        prior_work=[
            f"Embodied intelligence via learning and evolution follows performance change across evolutionary time rather than only final snapshots ({PAPER_LINKS['derl']}).",
            f"Unconventional Hexacopters via Evolution and Learning presents generation-wise performance improvements before interpreting the discovered morphologies ({PAPER_LINKS['hexa']}).",
        ],
        what_to_read=[
            "The central curve tells you where the bulk of the selected population is moving, while the dashed extreme curve tells you what the frontier is doing.",
            "The dotted BIX3 baseline tells you whether improvement is only relative inside the run or already meaningful against a known design.",
            "Divergence between median and best highlights whether progress is population-wide or concentrated in a few elites.",
        ],
        takeaways=[
            "This plot answers whether the run is genuinely progressing and on which objectives.",
            "The BIX3 line turns the figure from an internal optimization diagnostic into an external performance comparison.",
        ],
    )

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

def plot_pareto_fronts(
    df: pd.DataFrame,
    output_path: Path,
    generation: int,
    bix3_point: np.ndarray,
) -> None:
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
            s=22,
            alpha=0.36,
            label="all generations",
            zorder=1,
        )
        ax.scatter(
            all_points[global_front_mask_2d, 0],
            all_points[global_front_mask_2d, 1],
            s=48,
            facecolor="none",
            edgecolor="#e15759",
            linewidth=2.0,
            label="global 2D front",
            zorder=3,
        )
        bix3_x = bix3_point[FITNESS_COLUMNS.index(x_col)]
        bix3_y = bix3_point[FITNESS_COLUMNS.index(y_col)]
        ax.scatter(
            bix3_x,
            bix3_y,
            s=300,
            marker="*",
            color=PLOT_COLOR_CYCLE["baseline"],
            edgecolor=PLOT_COLOR_CYCLE["baseline"],
            linewidth=1.6,
            label="Baseline drone",
            zorder=5,
        )
        ax.set_xlabel(FITNESS_LABELS[x_col])
        ax.set_ylabel(FITNESS_LABELS[y_col])
        ax.set_title(
            f"{FITNESS_LABELS[x_col]} vs {FITNESS_LABELS[y_col]} "
            f"(global front {front_points}/{total_points})",
            fontsize=15,
            weight="bold",
        )
        ax.set_xlabel(FITNESS_LABELS[x_col], fontsize=14)
        ax.set_ylabel(FITNESS_LABELS[y_col], fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.35)

    axes[0].legend(loc="best", fontsize=13, frameon=True)
    if scatter_handle is not None:
        cbar_ax = fig.add_axes([0.93, 0.18, 0.02, 0.64])
        cbar = fig.colorbar(scatter_handle, cax=cbar_ax, label="Generation")
        cbar.ax.tick_params(labelsize=11.5)
        cbar.set_label("Generation", fontsize=13)
    fig.suptitle("Pareto Fronts (All Generations)", fontsize=19, weight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 1.0])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

def plot_pareto_front_3d(df: pd.DataFrame, output_path: Path, bix3_point: np.ndarray) -> None:
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
            x=[float(bix3_point[0])],
            y=[float(bix3_point[1])],
            z=[float(bix3_point[2])],
            mode="markers",
            name="Baseline drone",
            marker=dict(
                size=10,
                color=PLOT_COLOR_CYCLE["baseline"],
                symbol="diamond",
                line=dict(color=PLOT_COLOR_CYCLE["baseline"], width=3),
            ),
            hovertemplate=(
                "Baseline drone<br>"
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
    re_df = _wing_reynolds_dataframe(df, velocity_col="eff_v")
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

def plot_progress_vs_wing_reynolds(df: pd.DataFrame, output_path: Path) -> None:
    re_df = _wing_reynolds_dataframe(df, velocity_col="prog_v")
    front_mask = pareto_mask_finite(re_df[FITNESS_COLUMNS].to_numpy(dtype=float))
    re_df = re_df.loc[front_mask].copy()
    plot_df = re_df[
        [
            "ff_2",
            "wing_reynolds_over_nom_pct_a1",
            "prog_v",
            "wing_airfoil_code",
            "wing_span",
            "wing_aspect_ratio",
            "wing_chord",
            "wing_re_nom_a1",
            "wing_reynolds_prog_v",
            "wing_reynolds_minus_nom_a1",
            "exp_name",
        ]
    ].copy()
    plot_df = plot_df.rename(columns={"ff_2": "progress"})
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["progress", "wing_reynolds_over_nom_pct_a1", "prog_v"])
    if plot_df.empty:
        raise ValueError("No valid points available for progress vs Reynolds-limit plot.")

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    scatter = ax.scatter(
        plot_df["progress"],
        plot_df["wing_reynolds_over_nom_pct_a1"],
        c=plot_df["prog_v"],
        cmap="viridis",
        s=30,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel("Progress")
    ax.set_ylabel("Wing Reynolds / Re_nom [%]")
    ax.set_title("Final Global Pareto Front: Progress vs Wing Reynolds / Re_nom [%]")
    ax.set_ylim(80.0, 300.0)
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("prog_v")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    plot_df.sort_values(["progress", "wing_reynolds_over_nom_pct_a1", "prog_v"]).to_csv(csv_path, index=False)
    _write_plot_note(
        output_path,
        title="Progress vs operating Reynolds ratio",
        importance=(
            "This plot asks whether progress-specialist solutions operate in a characteristic Reynolds regime relative to the nominal Reynolds value of the selected wing airfoil."
        ),
        prior_work=[
            f"Unconventional Hexacopters via Evolution and Learning motivates interpreting objective specialists through physically meaningful morphology descriptors ({PAPER_LINKS['hexa']}).",
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots is relevant because objective and controller pressure change which body parameters are favored ({PAPER_LINKS['controller_learning']}).",
        ],
        what_to_read=[
            "If high-progress designs cluster around one Reynolds band, progress may depend on operating close to a preferred aero regime.",
            "If progress rises across a wide Reynolds range, other geometric factors may dominate more than Reynolds matching.",
            "Color by effective speed helps separate Reynolds effects due to speed from those due to chord and airfoil choice.",
        ],
        takeaways=[
            "This is the progress-side companion of the CoT vs Reynolds plot.",
            "It helps tell whether forward mission performance is tied to a distinct aerodynamic operating regime.",
        ],
    )

def plot_cot_vs_wing_chord(df: pd.DataFrame, output_path: Path) -> None:
    re_df = _wing_reynolds_dataframe(df, velocity_col="eff_v")
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

def plot_progress_vs_wing_chord(df: pd.DataFrame, output_path: Path) -> None:
    re_df = _wing_reynolds_dataframe(df, velocity_col="prog_v")
    front_mask = pareto_mask_finite(re_df[FITNESS_COLUMNS].to_numpy(dtype=float))
    re_df = re_df.loc[front_mask].copy()
    plot_df = re_df[
        [
            "ff_2",
            "wing_chord",
            "prog_v",
            "wing_airfoil_code",
            "wing_span",
            "wing_aspect_ratio",
            "wing_re_nom_a1",
            "wing_reynolds_prog_v",
            "wing_reynolds_over_nom_pct_a1",
            "exp_name",
        ]
    ].copy()
    plot_df = plot_df.rename(columns={"ff_2": "progress"})
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["progress", "wing_chord", "prog_v"])
    if plot_df.empty:
        raise ValueError("No valid points available for progress vs wing-chord plot.")

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    scatter = ax.scatter(
        plot_df["progress"],
        plot_df["wing_chord"],
        c=plot_df["prog_v"],
        cmap="viridis",
        s=30,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel("Progress")
    ax.set_ylabel("Wing chord [m]")
    ax.set_title("Final Global Pareto Front: Progress vs Wing Chord")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("prog_v")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    plot_df.sort_values(["progress", "wing_chord", "prog_v"]).to_csv(csv_path, index=False)
    _write_plot_note(
        output_path,
        title="Progress vs wing chord",
        importance=(
            "This figure tests whether forward-mission specialists prefer a characteristic wing-chord scale, separating geometric effects from the Reynolds-based interpretation."
        ),
        prior_work=[
            f"Unconventional Hexacopters via Evolution and Learning motivates reading objective specialists through interpretable morphology descriptors ({PAPER_LINKS['hexa']}).",
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots is relevant because objective pressure changes which body dimensions are selected ({PAPER_LINKS['controller_learning']}).",
        ],
        what_to_read=[
            "If high-progress designs cluster around one chord band, mission performance may favor a specific wing scale.",
            "If the cloud is broad, chord alone is not enough and must be read together with airfoil and operating speed.",
            "Color by prog_v separates purely geometric preferences from preferences tied to the progress operating point.",
        ],
        takeaways=[
            "This is the progress-side companion of the CoT vs wing-chord plot.",
            "It helps distinguish whether progress specialists are defined mainly by mission speed, by chord scale, or by both together.",
        ],
    )

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
