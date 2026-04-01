from __future__ import annotations

from .common import *  # noqa: F401,F403
from .descriptors import *  # noqa: F401,F403


def _plot_pca_gene_contributions(
    vt: np.ndarray,
    explained_ratio: np.ndarray,
    output_path: Path,
    *,
    title: str,
    top_n: int = 10,
) -> None:
    n_components = min(5, vt.shape[0])
    if n_components <= 0:
        return

    gene_names = _get_gene_names(vt.shape[1])
    contributions = np.square(vt[:n_components, :])
    row_sums = contributions.sum(axis=1, keepdims=True)
    contributions = np.divide(
        contributions,
        row_sums,
        out=np.zeros_like(contributions),
        where=row_sums > 0,
    ) * 100.0

    _apply_plot_style()
    fig, axes = plt.subplots(n_components, 1, figsize=(13.8, 3.2 * n_components), constrained_layout=True)
    if n_components == 1:
        axes = [axes]

    bar_color = '#2b6cb0'
    tail_color = '#94a3b8'

    for comp_idx, ax in enumerate(axes):
        values = np.asarray(contributions[comp_idx], dtype=float)
        order = np.argsort(values)[::-1]
        keep = order[: min(top_n, len(order))]
        kept_values = values[keep]
        kept_names = [gene_names[idx] for idx in keep]
        other_share = float(max(0.0, 100.0 - np.sum(kept_values)))
        if other_share > 1e-9 and len(order) > len(keep):
            kept_values = np.append(kept_values, other_share)
            kept_names.append('other_genes')

        plot_order = np.arange(len(kept_names))[::-1]
        colors = [bar_color] * len(kept_names)
        if kept_names and kept_names[-1] == 'other_genes':
            colors[-1] = tail_color

        ax.barh(plot_order, kept_values, color=colors, edgecolor='none', alpha=0.94)
        ax.set_yticks(plot_order)
        ax.set_yticklabels(kept_names, fontsize=8.5)
        ax.set_xlim(0.0, max(40.0, float(np.max(kept_values)) * 1.15))
        ax.grid(axis='x', alpha=0.28)
        variance = float(explained_ratio[comp_idx]) if comp_idx < explained_ratio.size else 0.0
        ax.set_title(
            f'PC{comp_idx + 1}: gene contribution share ({variance * 100:.2f}% variance explained)',
            fontsize=11.5,
            weight='bold',
        )
        ax.set_xlabel('Contribution within PC [% of squared loading mass]')

        for y, val in zip(plot_order, kept_values):
            ax.text(min(val + 0.6, ax.get_xlim()[1] - 0.5), y, f'{val:.1f}%', va='center', fontsize=8.2, color='#1f2937')

    fig.suptitle(title, fontsize=14, weight='bold')
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_pca_gene_representation_heatmap(
    vt: np.ndarray,
    explained_ratio: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    n_components = min(5, vt.shape[0])
    if n_components <= 0:
        return

    gene_names = _get_gene_names(vt.shape[1])
    absolute_capture = np.square(vt[:n_components, :]).T * 100.0
    total_capture = absolute_capture.sum(axis=1)
    dominant_pc = np.argmax(absolute_capture, axis=1)
    dominant_share = np.max(absolute_capture, axis=1)
    order = np.lexsort((dominant_pc, -dominant_share, -total_capture))
    absolute_capture = absolute_capture[order]
    total_capture = total_capture[order]
    ordered_names = [gene_names[idx] for idx in order]

    vmax = float(np.nanmax(absolute_capture)) if absolute_capture.size else 100.0
    vmax = max(20.0, min(100.0, vmax))

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10.8, max(5.0, 0.42 * len(ordered_names) + 1.4)))
    im = ax.imshow(absolute_capture, aspect='auto', cmap='YlGnBu', vmin=0.0, vmax=vmax)

    ax.set_xticks(np.arange(n_components))
    ax.set_xticklabels([f'PC{i + 1}' for i in range(n_components)], fontsize=10)
    ax.set_yticks(np.arange(len(ordered_names)))
    ax.set_yticklabels(ordered_names, fontsize=8.5)
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Gene')

    variance_labels = [
        f'{float(explained_ratio[i]) * 100:.1f}%' if i < explained_ratio.size else '0.0%'
        for i in range(n_components)
    ]
    for i, label in enumerate(variance_labels):
        ax.text(i, -0.85, label, ha='center', va='bottom', fontsize=8.5, color='#334155')
    ax.text(
        n_components - 0.02,
        -1.45,
        'header: variance explained by each PC',
        ha='right',
        va='bottom',
        fontsize=8.2,
        color='#64748b',
    )

    for row_idx in range(absolute_capture.shape[0]):
        for col_idx in range(absolute_capture.shape[1]):
            value = absolute_capture[row_idx, col_idx]
            if value >= 5.0:
                ax.text(
                    col_idx,
                    row_idx,
                    f'{value:.0f}%',
                    ha='center',
                    va='center',
                    fontsize=7.8,
                    color='white' if value >= max(0.55 * vmax, 12.0) else '#0f172a',
                )

    capture_labels = [f'{val:.0f}%' for val in total_capture]
    for row_idx, label in enumerate(capture_labels):
        ax.text(
            n_components - 0.02 + 0.62,
            row_idx,
            label,
            ha='left',
            va='center',
            fontsize=8.0,
            color='#0f172a',
        )
    ax.text(
        n_components - 0.02 + 0.62,
        -0.85,
        'captured\nby PC1-5',
        ha='left',
        va='bottom',
        fontsize=8.1,
        color='#334155',
    )

    ax.set_xlim(-0.5, n_components - 0.5 + 1.35)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('Absolute gene capture by each PC [% of total gene loading mass]')

    fig.suptitle(title, fontsize=14, weight='bold')
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)

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
    _plot_pca_gene_contributions(
        vt,
        explained_ratio,
        output_path.with_name('genome_pca_gene_contributions_top5pcs.png'),
        title='Genome PCA Gene Contributions Across the First 5 PCs',
    )
    _plot_pca_gene_representation_heatmap(
        vt,
        explained_ratio,
        output_path.with_name('genome_pca_gene_representation_top5pcs.png'),
        title='Genome PCA Gene Representation Across the First 5 PCs',
    )
    print(f"PCA variance explained: PC1={pc1_var * 100:.2f}% PC2={pc2_var * 100:.2f}%")

def plot_joint_genome_pca_generations(
    run_dfs: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    pc_x: int = 0,
    pc_y: int = 1,
) -> None:
    parsed_runs: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for run_name, df in run_dfs.items():
        matrix, aligned_df = _parse_chromosome_matrix(df)
        parsed_runs[run_name] = (matrix, aligned_df)

    combined_matrix = np.vstack([matrix for matrix, _ in parsed_runs.values()])
    mean, vt, explained_ratio = _pca_fit_transform_unique(combined_matrix)
    pc1_var = float(explained_ratio[0]) if explained_ratio.size > 0 else 0.0
    pc2_var = float(explained_ratio[1]) if explained_ratio.size > 1 else 0.0
    pc3_var = float(explained_ratio[2]) if explained_ratio.size > 2 else 0.0
    pc4_var = float(explained_ratio[3]) if explained_ratio.size > 3 else 0.0
    pc5_var = float(explained_ratio[4]) if explained_ratio.size > 4 else 0.0

    projected_runs: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    generation_slots: dict[str, list[int]] = {}
    for run_name, (matrix, aligned_df) in parsed_runs.items():
        projected_runs[run_name] = ((matrix - mean) @ vt.T, aligned_df)
        generation_slots[run_name] = _select_equally_spaced_generations(aligned_df, n_generations=6)

    _apply_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)

    run_order = list(run_dfs.keys())
    controller_labels = {
        "nsga_GP": "Platform\nIndependent",
        "nsga_SP": "Platform\nDependent",
    }
    for slot_idx, ax in enumerate(axes.flat):
        title_text = None
        for run_name in run_order:
            pcs, aligned_df = projected_runs[run_name]
            generations = generation_slots[run_name]
            if slot_idx >= len(generations):
                continue
            generation = generations[slot_idx]
            mask = aligned_df["generation"] == generation
            count = int(mask.sum())
            if run_name == "nsga_SP":
                title_text = f"Generation {generation}"
            if count > 0:
                x_vals = pcs[mask, pc_x]
                y_vals = pcs[mask, pc_y]
                ax.scatter(
                    x_vals,
                    y_vals,
                    s=18,
                    alpha=0.55,
                    color=EVOLUTION_COLORS.get(run_name, None),
                    label=run_name,
                )
                ax.scatter(
                    float(np.mean(x_vals)),
                    float(np.mean(y_vals)),
                    s=150,
                    marker="X",
                    color=EVOLUTION_COLORS.get(run_name, None),
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=4,
                )
        if title_text is not None:
            ax.set_title(title_text, fontsize=11.5, weight="bold")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"Slot {slot_idx + 1}")
        ax.grid(alpha=0.35)
        ax.set_xlabel(_pc_label(explained_ratio, pc_x))
        ax.set_ylabel(_pc_label(explained_ratio, pc_y))

    legend_handles = []
    for run_name in run_order:
        color = EVOLUTION_COLORS.get(run_name, "#333333")
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=9,
                markerfacecolor=color,
                markeredgecolor=color,
                label=f"{controller_labels.get(run_name, run_name)} individuals",
                alpha=0.85,
            )
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="X",
                linestyle="none",
                markersize=11,
                markerfacecolor=color,
                markeredgecolor="black",
                label=f"{controller_labels.get(run_name, run_name)} centroid",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        fontsize=14,
        title="Controller Type",
        title_fontsize=15,
        borderpad=1.0,
        labelspacing=0.9,
        handlelength=2.4,
        ncol=2,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.87])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    if pc_x == 0 and pc_y == 1:
        _plot_pca_gene_contributions(
            vt,
            explained_ratio,
            output_path.with_name('joint_genome_pca_gene_contributions_top5pcs.png'),
            title='Joint Genome PCA Gene Contributions Across the First 5 PCs',
        )
        _plot_pca_gene_representation_heatmap(
            vt,
            explained_ratio,
            output_path.with_name('joint_genome_pca_gene_representation_top5pcs.png'),
            title='Joint Genome PCA Gene Representation Across the First 5 PCs',
        )
    print(f"Joint PCA variance explained: PC1={pc1_var * 100:.2f}% PC2={pc2_var * 100:.2f}%")

def plot_comparative_evolutionary_metrics(
    metrics_by_run: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
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
    for run_name, metrics_df in metrics_by_run.items():
        missing = required - set(metrics_df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in evolutionary metrics for {run_name}: {missing_list}")

    _apply_plot_style()
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=False)

    for ax, metric in zip(axes.flatten(), metrics):
        all_values = []
        for run_name, metrics_df in metrics_by_run.items():
            plot_df = metrics_df.sort_values("generation").copy()
            plot_df[metric] = plot_df[metric].fillna(0.0)
            generations = plot_df["generation"].to_numpy(dtype=float)
            values = plot_df[metric].to_numpy(dtype=float)
            all_values.append(values)
            ax.plot(
                generations,
                values,
                marker="o",
                linewidth=2.1,
                markersize=4.5,
                color=EVOLUTION_COLORS.get(run_name, None),
                label=run_name,
            )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Generation")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.35)
        ax.set_ylim(_scaled_ylim_from_series(all_values))

    handles = [
        Line2D(
            [0],
            [0],
            color=EVOLUTION_COLORS.get(run_name, "#333333"),
            marker="o",
            linewidth=2.1,
            markersize=5,
            label=run_name,
        )
        for run_name in metrics_by_run.keys()
    ]
    fig.legend(handles=handles, loc="upper right", frameon=True)
    fig.suptitle("Comparative Evolutionary Metrics", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 0.96, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

def plot_comparative_evolutionary_metrics_core(
    metrics_by_run: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    metrics = [
        ("hypervolume", "Hypervolume"),
        ("diversity", "Diversity"),
        ("gd", "GD"),
        ("igd", "IGD"),
    ]
    required = {"generation", *(metric for metric, _ in metrics)}
    for run_name, metrics_df in metrics_by_run.items():
        missing = required - set(metrics_df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in evolutionary metrics for {run_name}: {missing_list}")

    _apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.4), sharex=False)
    axes_flat = axes.flatten()

    controller_labels = {
        "nsga_GP": "Platform\nIndependent",
        "nsga_SP": "Platform\nDependent",
    }

    for ax, (metric, title) in zip(axes_flat, metrics):
        all_values = []
        for run_name, metrics_df in metrics_by_run.items():
            plot_df = metrics_df.sort_values("generation").copy()
            plot_df[metric] = plot_df[metric].fillna(0.0)
            generations = plot_df["generation"].to_numpy(dtype=float)
            values = plot_df[metric].to_numpy(dtype=float)
            all_values.append(values)
            color = EVOLUTION_COLORS.get(run_name, None)
            ax.plot(
                generations,
                values,
                marker="o",
                linewidth=2.4,
                markersize=4.8,
                color=color,
                label=run_name,
            )
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel(title)
        ax.grid(alpha=0.35)
        ax.set_ylim(_scaled_ylim_from_series(all_values))

    handles = [
        Line2D(
            [0],
            [0],
            color=EVOLUTION_COLORS.get(run_name, "#333333"),
            marker="o",
            linewidth=2.4,
            markersize=8,
            label=controller_labels.get(run_name, run_name),
        )
        for run_name in metrics_by_run.keys()
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.76),
        frameon=True,
        fontsize=15,
        title="Controller Type",
        title_fontsize=16,
        borderpad=1.0,
        labelspacing=0.9,
        handlelength=2.4,
    )
    fig.suptitle("Comparative Evolutionary Metrics", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 0.96, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

def plot_comparative_runtime_per_individual(
    runtime_by_run: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    required = {
        "generation",
        "n_individuals",
        "mean_train_duration_s",
        "mean_eval_duration_s",
        "mean_runtime_total_s",
        "std_runtime_total_s",
    }
    for run_name, runtime_df in runtime_by_run.items():
        missing = required - set(runtime_df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in runtime summary for {run_name}: {missing_list}")

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5.8))

    all_values = []
    export_frames = []
    for run_name, runtime_df in runtime_by_run.items():
        plot_df = runtime_df.sort_values("generation").copy()
        generations = plot_df["generation"].to_numpy(dtype=float)
        values = plot_df["mean_runtime_total_s"].to_numpy(dtype=float)
        std_values = plot_df["std_runtime_total_s"].fillna(0.0).to_numpy(dtype=float)
        lower = np.clip(values - std_values, 0.0, None)
        upper = values + std_values
        all_values.extend([lower, upper])
        color = EVOLUTION_COLORS.get(run_name, None)
        ax.plot(
            generations,
            values,
            marker="o",
            linewidth=2.2,
            markersize=4.8,
            color=color,
            label=run_name,
        )
        ax.fill_between(
            generations,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        export_df = plot_df.copy()
        export_df.insert(0, "run_name", run_name)
        export_frames.append(export_df)

    ax.set_title("Mean Computational Time Per Individual by Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean runtime per individual [s]")
    ax.grid(alpha=0.35)
    ax.set_ylim(_scaled_ylim_from_series(all_values))
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    if export_frames:
        pd.concat(export_frames, ignore_index=True).to_csv(output_path.with_suffix(".csv"), index=False)

def plot_comparative_steps90_by_generation(
    steps_by_run: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    required = {"generation", "mean_steps90_ratio", "std_steps90_ratio"}
    for run_name, steps_df in steps_by_run.items():
        missing = required - set(steps_df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in steps90 summary for {run_name}: {missing_list}")

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5.8))
    all_values: list[np.ndarray] = []
    export_frames = []

    for run_name, steps_df in steps_by_run.items():
        plot_df = steps_df.sort_values("generation").copy()
        generations = plot_df["generation"].to_numpy(dtype=float)
        mean_vals = plot_df["mean_steps90_ratio"].to_numpy(dtype=float)
        std_vals = plot_df["std_steps90_ratio"].fillna(0.0).to_numpy(dtype=float)
        lower = np.clip(mean_vals - std_vals, 0.0, None)
        upper = np.clip(mean_vals + std_vals, 0.0, 1.0)
        color = EVOLUTION_COLORS.get(run_name, None)
        all_values.extend([lower, upper])
        ax.plot(
            generations,
            mean_vals,
            marker="o",
            linewidth=2.2,
            markersize=4.8,
            color=color,
            label=run_name,
        )
        ax.fill_between(generations, lower, upper, color=color, alpha=0.18, linewidth=0)
        export_df = plot_df.copy()
        export_df.insert(0, "run_name", run_name)
        export_frames.append(export_df)

    ax.set_title("Comparative Steps to 90% Reward by Generation", fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Steps to 90% reward / total training steps")
    ax.grid(alpha=0.35)
    ax.set_ylim(_scaled_ylim_from_series(all_values))
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    if export_frames:
        pd.concat(export_frames, ignore_index=True).to_csv(output_path.with_suffix(".csv"), index=False)

def plot_joint_pca_cluster_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    required = {
        "progress",
        "generation_nsga_GP",
        "generation_nsga_SP",
        "centroid_distance_all_pcs_weighted",
    }
    missing = required - set(metrics_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns in joint PCA cluster metrics: {missing_list}")

    plot_df = metrics_df.sort_values("progress").copy()
    progress = plot_df["progress"].to_numpy(dtype=float)
    centroid = plot_df["centroid_distance_all_pcs_weighted"].to_numpy(dtype=float)

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5.2))

    ax.plot(progress, centroid, marker="o", linewidth=2.2, markersize=5, color="#ff7f0e")
    ax.set_title("Joint-PCA Cluster Separation Using All PCs")
    ax.set_xlabel("Normalized evolution progress")
    ax.set_ylabel("Weighted centroid distance")
    ax.grid(alpha=0.35)
    ax.set_ylim(_scaled_ylim(centroid))

    gp_labels = plot_df["generation_nsga_GP"].astype(int).astype(str).to_list()
    sp_labels = plot_df["generation_nsga_SP"].astype(int).astype(str).to_list()
    tick_labels = [f"GP {g} | SP {s}" for g, s in zip(gp_labels, sp_labels)]
    ax.set_xticks(progress)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

def plot_comparative_gene_distributions(
    run_dfs: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    subset_mode: str = "final_generation_pareto",
) -> None:
    Chromosome_Drone = _load_chromosome_drone_class()
    param_names = [param.name for param in Chromosome_Drone.PARAMS]
    phys_min = np.asarray(Chromosome_Drone.PHYS_MIN, dtype=float)
    phys_max = np.asarray(Chromosome_Drone.PHYS_MAX, dtype=float)
    subset_title = _comparison_subset_title(subset_mode)
    subset_dfs = _comparison_subset_by_run(run_dfs, subset_mode)
    phys_by_run: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, float | int | str]] = []

    for run_name, subset_df in subset_dfs.items():
        matrix, _ = _parse_chromosome_matrix(subset_df)
        phys = np.asarray([Chromosome_Drone.to_physical(row) for row in matrix], dtype=float)
        phys_by_run[run_name] = phys
        for gene_idx, gene_name in enumerate(param_names):
            values = phys[:, gene_idx]
            finite = values[np.isfinite(values)]
            summary_rows.append(
                {
                    "run_name": run_name,
                    "subset_mode": subset_mode,
                    "gene": gene_name,
                    "n_samples": int(finite.size),
                    "median": float(np.nanmedian(finite)) if finite.size else np.nan,
                    "q1": float(np.nanpercentile(finite, 25)) if finite.size else np.nan,
                    "q3": float(np.nanpercentile(finite, 75)) if finite.size else np.nan,
                    "mean": float(np.nanmean(finite)) if finite.size else np.nan,
                    "std": float(np.nanstd(finite, ddof=1)) if finite.size > 1 else np.nan,
                }
            )

    gp_phys = phys_by_run["nsga_GP"]
    sp_phys = phys_by_run["nsga_SP"]

    _apply_plot_style()
    fig, axes = plt.subplots(5, 3, figsize=(15.8, 18.8))
    axes_flat = axes.flatten()
    color_map = {run: EVOLUTION_COLORS.get(run, "#4c566a") for run in run_dfs}
    label_order = ["nsga_GP", "nsga_SP"]
    y_pos = [1, 0]

    rng = np.random.default_rng(7)
    export_rows: list[dict[str, float | int | str]] = []
    for gene_idx, ax in enumerate(axes_flat):
        if gene_idx >= len(param_names):
            ax.axis("off")
            continue
        gene_name = param_names[gene_idx]
        gp_vals = gp_phys[:, gene_idx]
        sp_vals = sp_phys[:, gene_idx]
        data = [gp_vals[np.isfinite(gp_vals)], sp_vals[np.isfinite(sp_vals)]]
        if not any(arr.size for arr in data):
            ax.axis("off")
            continue

        bp = ax.boxplot(
            data,
            vert=False,
            positions=y_pos,
            widths=0.52,
            patch_artist=True,
            whis=(10, 90),
            showfliers=False,
            medianprops=dict(color="#111827", linewidth=2.0),
            whiskerprops=dict(color="#475569", linewidth=1.2),
            capprops=dict(color="#475569", linewidth=1.2),
        )
        for patch, run_name in zip(bp["boxes"], label_order):
            patch.set_facecolor(color_map[run_name])
            patch.set_alpha(0.35)
            patch.set_edgecolor(color_map[run_name])
            patch.set_linewidth(1.3)

        for row_pos, run_name, vals in zip(y_pos, label_order, data):
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=vals.size)
            ax.scatter(
                vals,
                row_pos + jitter,
                s=18,
                alpha=0.35,
                color=color_map[run_name],
                edgecolors="none",
                zorder=3,
            )

        stats = _two_sample_distribution_stats(data[0], data[1])
        median_delta = float(stats["median_delta_sp_minus_gp"])
        effect = float(stats["standardized_delta"])
        p_value = float(stats["p_value"])
        export_rows.append(
            {
                "subset_mode": subset_mode,
                "gene": gene_name,
                "x_min": float(phys_min[gene_idx]),
                "x_max": float(phys_max[gene_idx]),
                **stats,
            }
        )

        ax.set_title(gene_name.replace("_", " "), fontsize=11.5, weight="bold")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_comparison_run_label(name) for name in label_order], fontsize=9.2)
        ax.grid(alpha=0.22, axis="x")
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlim(float(phys_min[gene_idx]), float(phys_max[gene_idx]))
        ax.text(
            0.98,
            0.10,
            (
                f"Difference: {median_delta:+.3g}\nstd delta: {effect:+.2f}\np: {_format_p_value(p_value)}"
                if np.isfinite(effect) else f"Difference: {median_delta:+.3g}\np: {_format_p_value(p_value)}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", boxstyle="round,pad=0.28"),
        )
        if gene_idx % 3 == 0:
            ax.set_ylabel("Controller regime")
        if gene_idx >= 12:
            ax.set_xlabel("Physical gene value")

    fig.suptitle(f"Gene distributions by controller regime\n{subset_title}", fontsize=17, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.982))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    effect_df = pd.DataFrame(export_rows)
    if not effect_df.empty and "p_value" in effect_df.columns:
        effect_df["p_value_fdr_bh"] = _benjamini_hochberg(pd.to_numeric(effect_df["p_value"], errors="coerce").to_numpy(dtype=float))
    summary_df.to_csv(output_path.with_suffix(".csv"), index=False)
    effect_df.to_csv(output_path.with_name(output_path.stem + "_effect_sizes.csv"), index=False)
    _write_plot_note(
        output_path,
        title=f"Gene distributions by controller regime | {subset_title}",
        importance=(
            "This plot answers the most direct genotype question: for each physical gene, do the two controller regimes occupy the same value range or settle into different bands under the selected subset of individuals?"
        ),
        prior_work=[
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots is directly relevant because different controller regimes can bias morphology search toward different parameter regions ({PAPER_LINKS['controller_learning']}).",
            f"Unconventional Hexacopters via Evolution and Learning also interprets divergence through morphology-family statistics rather than only through latent projections ({PAPER_LINKS['hexa']}).",
        ],
        what_to_read=[
            f"This specific version uses: {subset_title.lower()}.",
            "Each panel compares one physical gene across the two controller regimes, with the x-axis fixed to the exact physical genome bounds for that gene.",
            "The box summarizes the interquartile range, while faint dots show individual solutions without overcrowding the plot.",
        ],
        takeaways=[
            "Use the all-generations version to see the full explored support.",
            "Use the Global Pareto Front version to compare the non-dominated designs accumulated by each run across its entire history.",
            "Use the final-generation-all version to see where each search converged as a population.",
            "Use the Final Pareto Fronts version to compare the two surviving trade-off sets directly, one front per run.",
        ],
        caveats=[
            "Discrete NACA genes appear as stepped bands by construction.",
            "The fixed x-limits are the true physical genome bounds, so some panels will intentionally contain empty margins when evolution explored only part of the allowed range.",
        ],
    )

def plot_comparative_gene_distributions_core(
    run_dfs: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    subset_mode: str = "global_pareto_front",
) -> None:
    Chromosome_Drone = _load_chromosome_drone_class()
    phys_min = np.asarray(Chromosome_Drone.PHYS_MIN, dtype=float)
    phys_max = np.asarray(Chromosome_Drone.PHYS_MAX, dtype=float)
    subset_title = _comparison_subset_title(subset_mode)
    subset_dfs = _comparison_subset_by_run(run_dfs, subset_mode)

    elevator_area_limits = (
        float((phys_min[5] ** 2) / phys_max[6]),
        float((phys_max[5] ** 2) / phys_min[6]),
    )
    rudder_area_limits = (
        float((phys_min[7] ** 2) / phys_max[8]),
        float((phys_max[7] ** 2) / phys_min[8]),
    )
    metric_specs = [
        ("fuselage_length", "Fuselage Length [m]", (float(phys_min[2]), float(phys_max[2]))),
        ("cg_x_ratio", "CG Position Ratio [-]", (float(phys_min[3]), float(phys_max[3]))),
        ("attach_x_ratio", "Wing Attach Position Ratio [-]", (float(phys_min[4]), float(phys_max[4]))),
        ("elevator_area", "Elevator Planform Area [m^2]", elevator_area_limits),
        ("rudder_area", "Rudder Planform Area [m^2]", rudder_area_limits),
        ("dihedral_deg", "Dihedral Angle [deg]", (float(phys_min[9]), float(phys_max[9]))),
        ("sweep_multiplier", "Sweep Multiplier [-]", (float(phys_min[10]), float(phys_max[10]))),
        ("twist_multiplier", "Twist Multiplier [-]", (float(phys_min[11]), float(phys_max[11]))),
    ]

    values_by_run: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, float | int | str]] = []
    for run_name, subset_df in subset_dfs.items():
        matrix, _ = _parse_chromosome_matrix(subset_df)
        phys = np.asarray([Chromosome_Drone.to_physical(row) for row in matrix], dtype=float)
        desc_df = _aero_descriptor_frame_from_physical(phys)
        metric_df = pd.DataFrame({
            "fuselage_length": desc_df["fuselage_length"],
            "cg_x_ratio": phys[:, 3],
            "attach_x_ratio": phys[:, 4],
            "elevator_area": desc_df["elevator_area"],
            "rudder_area": desc_df["rudder_area"],
            "dihedral_deg": desc_df["dihedral_deg"],
            "sweep_multiplier": desc_df["sweep_multiplier"],
            "twist_multiplier": desc_df["twist_multiplier"],
        })
        values_by_run[run_name] = metric_df
        for metric_name, label, limits in metric_specs:
            values = pd.to_numeric(metric_df[metric_name], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            summary_rows.append(
                {
                    "run_name": run_name,
                    "subset_mode": subset_mode,
                    "metric": metric_name,
                    "label": label,
                    "x_min": float(limits[0]),
                    "x_max": float(limits[1]),
                    "n_samples": int(finite.size),
                    "median": float(np.nanmedian(finite)) if finite.size else np.nan,
                    "q1": float(np.nanpercentile(finite, 25)) if finite.size else np.nan,
                    "q3": float(np.nanpercentile(finite, 75)) if finite.size else np.nan,
                }
            )

    _apply_plot_style()
    fig, axes = plt.subplots(2, 4, figsize=(16.8, 8.8), sharey=True)
    axes_flat = axes.flatten()
    color_map = {run: EVOLUTION_COLORS.get(run, "#4c566a") for run in run_dfs}
    label_order = ["nsga_GP", "nsga_SP"]
    y_pos = [1, 0]
    rng = np.random.default_rng(23)
    export_rows: list[dict[str, float | int | str]] = []

    for ax, spec in zip(axes_flat, metric_specs):
        metric_name, label, limits = spec
        data = []
        for run_name in label_order:
            vals = pd.to_numeric(values_by_run[run_name][metric_name], errors="coerce").to_numpy(dtype=float)
            data.append(vals[np.isfinite(vals)])

        bp = ax.boxplot(
            data,
            vert=False,
            positions=y_pos,
            widths=0.52,
            patch_artist=True,
            whis=(10, 90),
            showfliers=False,
            medianprops=dict(color="#111827", linewidth=2.0),
            whiskerprops=dict(color="#475569", linewidth=1.2),
            capprops=dict(color="#475569", linewidth=1.2),
        )
        for patch, run_name in zip(bp["boxes"], label_order):
            patch.set_facecolor(color_map[run_name])
            patch.set_alpha(0.35)
            patch.set_edgecolor(color_map[run_name])
            patch.set_linewidth(1.3)

        for row_pos, run_name, vals in zip(y_pos, label_order, data):
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=vals.size)
            ax.scatter(
                vals,
                row_pos + jitter,
                s=18,
                alpha=0.35,
                color=color_map[run_name],
                edgecolors="none",
                zorder=3,
            )

        stats = _two_sample_distribution_stats(data[0], data[1])
        median_delta = float(stats["median_delta_sp_minus_gp"])
        effect = float(stats["standardized_delta"])
        p_value = float(stats["p_value"])
        export_rows.append(
            {
                "subset_mode": subset_mode,
                "metric": metric_name,
                "label": label,
                "x_min": float(limits[0]),
                "x_max": float(limits[1]),
                **stats,
            }
        )

        ax.set_title(label, fontsize=11.5, weight="bold")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_comparison_run_label(name) for name in label_order], fontsize=9.2)
        ax.grid(alpha=0.22, axis="x")
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlim(*limits)
        ax.text(
            0.98,
            0.10,
            (
                f"Difference: {median_delta:+.3g}\nstd delta: {effect:+.2f}\np: {_format_p_value(p_value)}"
                if np.isfinite(effect) else f"Difference: {median_delta:+.3g}\np: {_format_p_value(p_value)}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.4,
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", boxstyle="round,pad=0.28"),
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("Controller regime")
    for ax in axes[-1, :]:
        ax.set_xlabel("Value")
    for ax in axes_flat[len(metric_specs):]:
        ax.axis("off")

    fig.suptitle(f"Core gene and geometry distributions by controller regime\n{subset_title}", fontsize=17, weight="bold", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(output_path.with_suffix(".csv"), index=False)
    effect_df = pd.DataFrame(export_rows)
    if not effect_df.empty and "p_value" in effect_df.columns:
        effect_df["p_value_fdr_bh"] = _benjamini_hochberg(pd.to_numeric(effect_df["p_value"], errors="coerce").to_numpy(dtype=float))
    effect_df.to_csv(output_path.with_name(output_path.stem + "_effect_sizes.csv"), index=False)
    _write_plot_note(
        output_path,
        title=f"Core gene and geometry distributions by controller regime | {subset_title}",
        importance=(
            "This compact version strips the genome comparison down to the most interpretable longitudinal-balance and tail-sizing quantities, so the GP-SP divergence can be read quickly without scanning the full 15-gene panel."
        ),
        prior_work=[
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots motivates morphology comparisons under different controller assumptions ({PAPER_LINKS['controller_learning']}).",
            f"Unconventional Hexacopters via Evolution and Learning favors interpretable family-level morphology summaries over latent-only views ({PAPER_LINKS['hexa']}).",
        ],
        what_to_read=[
            f"This specific version uses: {subset_title.lower()}.",
            "Panels are limited to fuselage length, CG position, wing attach position, tail planform areas, dihedral, sweep and twist, because these are among the easiest quantities to connect to stability, trim and control authority.",
            "Axes use theoretical design-space bounds, including analytically derived bounds for elevator and rudder planform area.",
        ],
        takeaways=[
            "Use this view when the full gene plot is too dense and you want a quick read on balance, tail sizing and lateral geometry.",
        ],
        caveats=[
            "The compact plot now fills the 2x4 layout with eight interpretable quantities, adding sweep to the earlier balance-tail-geometry subset.",
        ],
    )

def plot_comparative_gene_shift_heatmap(
    run_dfs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    Chromosome_Drone = _load_chromosome_drone_class()
    gene_names = [param.name for param in Chromosome_Drone.PARAMS]
    rows = []
    for run_name, df in run_dfs.items():
        final_df = _comparison_subset(df, "final_generation_pareto")
        matrix, _ = _parse_chromosome_matrix(final_df)
        phys = np.asarray([Chromosome_Drone.to_physical(row) for row in matrix], dtype=float)
        rows.append(pd.DataFrame(phys, columns=gene_names).assign(run_name=run_name))

    combined_df = pd.concat(rows, ignore_index=True)
    heat_rows = []
    for gene_name in gene_names:
        gp_vals = pd.to_numeric(
            combined_df.loc[combined_df["run_name"] == "nsga_GP", gene_name],
            errors="coerce",
        ).to_numpy(dtype=float)
        sp_vals = pd.to_numeric(
            combined_df.loc[combined_df["run_name"] == "nsga_SP", gene_name],
            errors="coerce",
        ).to_numpy(dtype=float)
        pooled_all = pd.to_numeric(combined_df[gene_name], errors="coerce").to_numpy(dtype=float)
        center = float(np.nanmedian(pooled_all))
        scale = float(np.nanstd(pooled_all, ddof=1))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        gp_median = float(np.nanmedian(gp_vals))
        sp_median = float(np.nanmedian(sp_vals))
        delta_std = (sp_median - gp_median) / scale
        iqr_gp = float(np.nanpercentile(gp_vals, 75) - np.nanpercentile(gp_vals, 25))
        iqr_sp = float(np.nanpercentile(sp_vals, 75) - np.nanpercentile(sp_vals, 25))
        heat_rows.append(
            {
                "gene": gene_name,
                "gp_median_z": (gp_median - center) / scale,
                "sp_median_z": (sp_median - center) / scale,
                "delta_std": delta_std,
                "gp_iqr": iqr_gp,
                "sp_iqr": iqr_sp,
                "abs_delta": abs(delta_std),
            }
        )

    heat_df = pd.DataFrame(heat_rows).sort_values("abs_delta", ascending=False).reset_index(drop=True)
    heat = heat_df[["gp_median_z", "sp_median_z", "delta_std"]].to_numpy(dtype=float)
    vmax = max(1.0, float(np.nanmax(np.abs(heat))))

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, max(6.2, 0.45 * len(heat_df) + 1.4)))
    im = ax.imshow(heat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Platform\nIndependent\nmedian (z)", "Platform\nDependent\nmedian (z)", "Difference\n(z-delta)"])
    ax.set_yticks(np.arange(len(heat_df)))
    ax.set_yticklabels(heat_df["gene"].str.replace("_", " ", regex=False).to_list())
    ax.set_title("Which genes separate the controller regimes the most?", fontsize=15, weight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(
                j,
                i,
                f"{heat[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=8.6,
                color="black",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Standardized position relative to pooled final front")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    heat_df.to_csv(output_path.with_suffix(".csv"), index=False)
    _write_plot_note(
        output_path,
        title="Gene-shift heatmap by controller regime",
        importance=(
            "PCA tells you that two clouds separate, but not which coordinates are driving that separation. This heatmap ranks genes by how much their final medians move between the two controller regimes."
        ),
        prior_work=[
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots is relevant because the core question is exactly whether controller regime shifts morphology optima ({PAPER_LINKS['controller_learning']}).",
            f"DERL-style morphology analysis also motivates reading evolution through interpretable body descriptors rather than only through latent axes ({PAPER_LINKS['derl']}).",
        ],
        what_to_read=[
            "Rows are sorted so the strongest controller-regime separations appear first.",
            "The first two columns show where each run's median lies relative to the pooled final frontier.",
            "The last column is the cleanest summary: positive means the dependent-controller run prefers larger values, negative means the independent-controller run does.",
        ],
        takeaways=[
            "This is the fastest way to identify the genes that really explain divergence.",
            "It complements the distribution plot by showing ranking and direction in one compact panel.",
        ],
        caveats=[
            "The scale is standardized within the pooled final-front data, so large values mean strong separation inside this experiment, not universal effect sizes across studies.",
        ],
    )

def plot_comparative_pareto_fronts_2d(
    run_dfs: dict[str, pd.DataFrame],
    bix3_points_by_run: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    pairs = [("ff_0", "ff_1"), ("ff_0", "ff_2"), ("ff_1", "ff_2")]
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(19.8, 8.2))

    for ax, (x_col, y_col) in zip(axes, pairs):
        for run_name, df in run_dfs.items():
            all_points = _fitness_points_for_plot(df, [x_col, y_col])
            if all_points.size == 0:
                continue
            finite_mask = np.isfinite(all_points).all(axis=1)
            all_points = all_points[finite_mask]
            pareto_points = _pareto_points_from_plot(all_points, [x_col, y_col])
            front_mask = pareto_mask_finite(pareto_points)
            front_points = all_points[front_mask]
            if front_points.size == 0:
                continue
            front_points_unique = np.unique(front_points, axis=0)
            order = np.argsort(front_points_unique[:, 0], kind="mergesort")
            front_points_sorted = front_points_unique[order]
            ax.plot(
                front_points_sorted[:, 0],
                front_points_sorted[:, 1],
                linewidth=1.8,
                alpha=0.9,
                color=EVOLUTION_COLORS.get(run_name, None),
                zorder=2,
            )
            ax.scatter(
                front_points_sorted[:, 0],
                front_points_sorted[:, 1],
                s=42,
                alpha=0.72,
                color=EVOLUTION_COLORS.get(run_name, None),
                label=run_name,
                zorder=3,
            )
            if run_name in bix3_points_by_run:
                bix3_point = bix3_points_by_run[run_name]
                ax.scatter(
                    bix3_point[FITNESS_COLUMNS.index(x_col)],
                    bix3_point[FITNESS_COLUMNS.index(y_col)],
                    s=260,
                    marker="*",
                    color=EVOLUTION_COLORS.get(run_name, PLOT_COLOR_CYCLE["baseline"]),
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=4,
                )

        ax.set_xlabel(FITNESS_LABELS[x_col])
        ax.set_ylabel(FITNESS_LABELS[y_col])
        ax.set_title(f"{FITNESS_LABELS[x_col]} vs {FITNESS_LABELS[y_col]}")
        ax.grid(alpha=0.35)

    controller_labels = {
        "nsga_GP": "Platform\nIndependent",
        "nsga_SP": "Platform\nDependent",
    }

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=EVOLUTION_COLORS.get(run_name, "#333333"),
            markeredgecolor=EVOLUTION_COLORS.get(run_name, "#333333"),
            label=controller_labels.get(run_name, run_name),
            alpha=0.85,
        )
        for run_name in run_dfs.keys()
    ]
    for run_name in run_dfs.keys():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markersize=14,
                markerfacecolor=EVOLUTION_COLORS.get(run_name, PLOT_COLOR_CYCLE["baseline"]),
                markeredgecolor="black",
                label=f"{controller_labels.get(run_name, run_name)} baseline drone",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        fontsize=15,
        title="Runs and Baselines",
        title_fontsize=16,
        borderpad=1.0,
        labelspacing=0.9,
        handlelength=2.4,
        ncol=2,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.81])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
