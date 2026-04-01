from __future__ import annotations

from .common import *  # noqa: F401,F403
from .descriptors import *  # noqa: F401,F403

def plot_steps90_by_generation(
    steps_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    color: str = "#1f77b4",
    baseline_ratio: float | None = None,
) -> None:
    required = {"generation", "mean_steps90_ratio", "std_steps90_ratio"}
    missing = required - set(steps_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns in steps90 summary: {missing_list}")

    plot_df = steps_df.sort_values("generation").copy()
    generations = plot_df["generation"].to_numpy(dtype=float)
    mean_vals = plot_df["mean_steps90_ratio"].to_numpy(dtype=float)
    std_vals = plot_df["std_steps90_ratio"].fillna(0.0).to_numpy(dtype=float)
    lower = np.clip(mean_vals - std_vals, 0.0, None)
    upper = np.clip(mean_vals + std_vals, 0.0, 1.0)

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(
        generations,
        mean_vals,
        marker="o",
        linewidth=2.2,
        markersize=4.8,
        color=color,
        label="Population mean",
    )
    ax.fill_between(
        generations,
        lower,
        upper,
        color=color,
        alpha=0.18,
        linewidth=0,
        label="Population std",
    )
    if baseline_ratio is not None and np.isfinite(baseline_ratio) and baseline_ratio > 0.0:
        ax.axhline(
            float(baseline_ratio),
            color=PLOT_COLOR_CYCLE["baseline"],
            linewidth=2.0,
            linestyle=(0, (6, 3)),
            label="Baseline drone",
        )
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Steps to 90% reward / total training steps")
    ax.grid(alpha=0.35)
    ax.set_ylim(_scaled_ylim_from_series([lower, upper]))
    if baseline_ratio is not None and np.isfinite(baseline_ratio) and baseline_ratio > 0.0:
        ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    plot_df.to_csv(output_path.with_suffix(".csv"), index=False)

def plot_training_speed90_by_generation(
    speed_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    color: str = "#8c564b",
) -> None:
    required = {"generation", "mean_training_speed90", "std_training_speed90"}
    missing = required - set(speed_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns in training-speed summary: {missing_list}")

    plot_df = speed_df.sort_values("generation").copy()
    generations = plot_df["generation"].to_numpy(dtype=float)
    mean_vals = plot_df["mean_training_speed90"].to_numpy(dtype=float)
    std_vals = plot_df["std_training_speed90"].fillna(0.0).to_numpy(dtype=float)
    lower = np.clip(mean_vals - std_vals, 0.0, None)
    upper = mean_vals + std_vals

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(generations, mean_vals, marker="o", linewidth=2.2, markersize=4.8, color=color)
    ax.fill_between(generations, lower, upper, color=color, alpha=0.18, linewidth=0)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("0.9 * individual max reward / normalized steps to 90% reward")
    ax.grid(alpha=0.35)
    ax.set_ylim(_scaled_ylim_from_series([lower, upper]))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    plot_df.to_csv(output_path.with_suffix(".csv"), index=False)

def plot_learning_proxy_panel_by_generation(summary_df: pd.DataFrame, output_path: Path, *, title: str) -> None:
    metrics = [
        ("steps90_ratio", "Steps to 90% / budget", PLOT_COLOR_CYCLE["efficiency"], (0.0, 1.0)),
        ("burnin_ratio_proxy", "Burn-in proxy", PLOT_COLOR_CYCLE["innovation"], (0.0, 1.0)),
        ("volatility_proxy", "Volatility proxy", PLOT_COLOR_CYCLE["reward"], None),
        ("early_reward_proxy", "Early reward (10%)", "#0f766e", None),
        ("final_reward_proxy", "Final reward (100%)", "#334155", None),
        ("reward_gain_proxy", "Reward gain", "#ca8a04", None),
    ]

    _apply_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(13.4, 12.0), sharex=True)
    axes_flat = axes.flatten()

    for ax, (metric, label, color, fixed_ylim) in zip(axes_flat, metrics):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in summary_df.columns:
            ax.text(0.5, 0.5, "missing metric", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        plot_df = summary_df.loc[:, ["generation", mean_col, std_col]].copy()
        x = plot_df["generation"].to_numpy(dtype=float)
        y = plot_df[mean_col].to_numpy(dtype=float)
        std = plot_df[std_col].fillna(0.0).to_numpy(dtype=float)
        lower = y - std
        upper = y + std
        finite_mask = np.isfinite(y)
        if not finite_mask.any():
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=12.5, weight="bold")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            continue

        ax.plot(x, y, color=color, linewidth=2.5, marker="o", markersize=4.5)
        ax.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)
        ax.set_title(label, fontsize=12.5, weight="bold")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)
        else:
            finite_lower = lower[np.isfinite(lower)]
            finite_upper = upper[np.isfinite(upper)]
            finite_y = y[np.isfinite(y)]
            ax.set_ylim(_scaled_ylim_from_series([finite_lower, finite_upper, finite_y], padding_ratio=0.1))

    for ax in axes_flat[-2:]:
        ax.set_xlabel("Generation")

    fig.suptitle(title, fontsize=16, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    summary_df.to_csv(output_path.with_suffix(".csv"), index=False)
    _write_plot_note(
        output_path,
        title=title,
        importance=(
            "This figure checks whether evolution is making morphologies easier to train, not only stronger at the end. "
            "That is the closest statement in this project to embodied intelligence and Baldwin-like effects."
        ),
        prior_work=[
            f"Embodied intelligence via learning and evolution links morphology to learning speed and adaptation ({PAPER_LINKS['derl']}).",
            f"Unconventional Hexacopters via Evolution and Learning reports learning speed, burn-in and volatility descriptors ({PAPER_LINKS['hexa']}).",
            f"Morpho-evolution with learning using a controller archive studies how learning support changes search efficiency ({PAPER_LINKS['inheritance']}).",
        ],
        what_to_read=[
            "Steps90 going down means the population is becoming easier to train.",
            "Burn-in proxy going down means useful learning starts earlier.",
            "Volatility proxy near 1 means smoother monotonic learning; larger values mean more oscillatory learning.",
            "Reward gain separates morphologies that only start well from morphologies that keep improving during training.",
        ],
        takeaways=[
            "If final reward improves but steps90 and burn-in do not, evolution is finding stronger but not easier morphologies.",
            "If reward gain rises while volatility falls, controller adaptation is becoming both richer and cleaner.",
            "In GP these proxies are expected to be degenerate, because there is no lifetime training phase.",
        ],
        caveats=[
            "Burn-in and volatility are proxies built from decile reward checkpoints, not from the full TensorBoard curve.",
        ],
    )

def plot_bix3_beating_curves(df: pd.DataFrame, output_path: Path, bix3_point: np.ndarray) -> None:
    work_df = df.copy()
    work_df["generation"] = pd.to_numeric(work_df["generation"], errors="coerce")
    work_df = work_df[np.isfinite(work_df["generation"])].copy()
    work_df["cost_of_transport"] = -pd.to_numeric(work_df["ff_1"], errors="coerce")
    work_df["beats_speed"] = pd.to_numeric(work_df["ff_0"], errors="coerce") > float(bix3_point[0])
    work_df["beats_efficiency"] = work_df["cost_of_transport"] < float(bix3_point[1])
    work_df["beats_progress"] = pd.to_numeric(work_df["ff_2"], errors="coerce") > float(bix3_point[2])
    work_df["beats_two_of_three"] = (
        work_df[["beats_speed", "beats_efficiency", "beats_progress"]].sum(axis=1) >= 2
    )

    panels: list[tuple[str, pd.DataFrame]] = [("Population", work_df)]
    if "is_pareto" in work_df.columns and work_df["is_pareto"].astype(float).gt(0).any():
        pareto_df = work_df.loc[work_df["is_pareto"].astype(float) > 0].copy()
        panels.append(("Pareto set", pareto_df))

    rows = []
    for panel_name, panel_df in panels:
        grouped = panel_df.groupby("generation", sort=True)
        for generation, group in grouped:
            rows.append(
                {
                    "panel": panel_name,
                    "generation": int(generation),
                    "speed_frac": float(group["beats_speed"].mean()),
                    "efficiency_frac": float(group["beats_efficiency"].mean()),
                    "progress_frac": float(group["beats_progress"].mean()),
                    "two_of_three_frac": float(group["beats_two_of_three"].mean()),
                    "n_samples": int(len(group)),
                }
            )
    plot_df = pd.DataFrame(rows).sort_values(["panel", "generation"])
    if plot_df.empty:
        raise ValueError("No valid rows available for BIX3 comparison plot.")

    _apply_plot_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(7.1 * len(panels), 5.6), sharey=True, sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, (panel_name, _) in zip(axes_flat, panels):
        panel_df = plot_df.loc[plot_df["panel"] == panel_name].copy()
        x = panel_df["generation"].to_numpy(dtype=float)
        ax.plot(x, panel_df["speed_frac"], color=PLOT_COLOR_CYCLE["speed"], linewidth=2.4, marker="o", label="Beat BIX3 on speed")
        ax.plot(x, panel_df["efficiency_frac"], color=PLOT_COLOR_CYCLE["efficiency"], linewidth=2.4, marker="s", label="Beat BIX3 on efficiency")
        ax.plot(x, panel_df["progress_frac"], color=PLOT_COLOR_CYCLE["progress"], linewidth=2.4, marker="^", label="Beat BIX3 on progress")
        ax.plot(
            x,
            panel_df["two_of_three_frac"],
            color=PLOT_COLOR_CYCLE["reward"],
            linewidth=2.8,
            linestyle="--",
            marker="D",
            label="Beat BIX3 on at least 2/3 objectives",
        )
        ax.set_title(panel_name, fontsize=13, weight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
    axes_flat[0].set_ylabel("Fraction of designs")
    axes_flat[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("How Often Evolution Beats BIX3", fontsize=16, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    plot_df.to_csv(output_path.with_suffix(".csv"), index=False)
    _write_plot_note(
        output_path,
        title="How Often Evolution Beats BIX3",
        importance=(
            "A Pareto front shows that trade-offs moved, but not whether the evolved population is actually becoming practically "
            "better than a known reference design. This plot turns the story into a baseline-beating probability over generations."
        ),
        prior_work=[
            f"Unconventional Hexacopters via Evolution and Learning compares evolved morphologies against a conventional human design ({PAPER_LINKS['hexa']}).",
            f"Embodied intelligence via learning and evolution uses longitudinal evolutionary evidence rather than a single end-point comparison ({PAPER_LINKS['derl']}).",
        ],
        what_to_read=[
            "If speed and progress fractions rise but efficiency does not, the search is shifting the trade-off rather than dominating everywhere.",
            "The Pareto panel tells you whether the frontier itself is improving, while the population panel tells you how common the improvement is.",
            "The 2-of-3 curve is often more informative than strict 3-of-3 domination when the baseline is strong.",
        ],
        takeaways=[
            "This plot is the cleanest answer to: are evolved designs actually beating the standard design, and on what axes?",
            "It also shows whether the improvement is rare frontier behavior or widespread across the selected population.",
        ],
    )

def plot_energy_vs_learnability(df: pd.DataFrame, output_path: Path) -> None:
    proxy_df = _row_learning_proxy_frame(df)
    proxy_df["cost_of_transport"] = -pd.to_numeric(proxy_df["ff_1"], errors="coerce")
    proxy_df["generation"] = pd.to_numeric(proxy_df["generation"], errors="coerce")

    panels = [
        ("steps90_ratio", "Steps to 90% / budget"),
        ("burnin_ratio_proxy", "Burn-in proxy"),
        ("reward_gain_proxy", "Reward gain"),
    ]

    _apply_plot_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.8), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    export_rows: list[pd.DataFrame] = []
    scatter_handle = None

    for ax, (metric, label) in zip(axes_flat, panels):
        base_cols = ["generation", "cost_of_transport", metric]
        if "uid" in proxy_df.columns:
            base_cols.append("uid")
        plot_df = proxy_df.loc[:, base_cols].copy()
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["cost_of_transport", metric, "generation"])
        if plot_df.empty:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=12.5, weight="bold")
            ax.grid(alpha=0.3)
            continue

        scatter_handle = ax.scatter(
            plot_df["cost_of_transport"],
            plot_df[metric],
            c=plot_df["generation"],
            cmap="viridis",
            s=32,
            alpha=0.82,
            edgecolors="black",
            linewidths=0.25,
        )
        x = plot_df["cost_of_transport"].to_numpy(dtype=float)
        y = plot_df[metric].to_numpy(dtype=float)
        if len(plot_df) >= 3 and np.nanstd(x) > 1e-12 and np.nanstd(y) > 1e-12:
            coeffs = np.polyfit(x, y, deg=1)
            x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
            y_line = coeffs[0] * x_line + coeffs[1]
            corr = float(np.corrcoef(x, y)[0, 1])
            ax.plot(x_line, y_line, color="#111827", linewidth=2.0, linestyle="--")
            ax.text(
                0.03,
                0.97,
                f"r = {corr:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d1d5db", alpha=0.9),
            )
        ax.set_title(label, fontsize=12.5, weight="bold")
        ax.set_xlabel("Cost Of Transport")
        ax.grid(alpha=0.3)
        export_df = plot_df.copy()
        export_df["panel_metric"] = metric
        export_rows.append(export_df)

    axes_flat[0].set_ylabel("Learnability proxy")
    if scatter_handle is not None:
        cbar = fig.colorbar(scatter_handle, ax=axes_flat.tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("Generation")
    fig.suptitle("Energy vs Learnability", fontsize=16, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    if export_rows:
        pd.concat(export_rows, ignore_index=True).to_csv(output_path.with_suffix(".csv"), index=False)
    _write_plot_note(
        output_path,
        title="Energy vs Learnability",
        importance=(
            "This figure tests the embodied-intelligence hypothesis directly inside your data: do energetically favorable morphologies "
            "also become easier to train, or are those two axes largely independent in your search?"
        ),
        prior_work=[
            f"Embodied intelligence via learning and evolution links energy efficiency and body properties to downstream learnability ({PAPER_LINKS['derl']}).",
            f"Unconventional Hexacopters via Evolution and Learning uses learning descriptors to explain why some morphologies are controller-friendly ({PAPER_LINKS['hexa']}).",
        ],
        what_to_read=[
            "A negative CoT-vs-steps90 relation means efficient designs also learn faster.",
            "A positive CoT-vs-reward-gain relation means energetically expensive designs may still unlock richer learning.",
            "Generation coloring tells you whether the relation is static or whether evolution is moving the whole cloud.",
        ],
        takeaways=[
            "This is one of the cleanest ways to argue that morphology is acting as an inductive bias for control learning.",
            "If no clear relation appears, that is also informative: evolution may be decoupling energetic quality from training ease.",
        ],
        caveats=[
            "This figure uses summary learnability proxies, not full training curves.",
        ],
    )

def plot_lineage_takeover_metrics(work_df: pd.DataFrame, output_path: Path) -> None:
    required = {"generation", "lineage_id"}
    missing = required - set(work_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns for lineage takeover plot: {missing_list}")

    rows = []
    for generation, group in work_df.groupby("generation", sort=True):
        lineage_counts = group["lineage_id"].value_counts()
        probs = lineage_counts.to_numpy(dtype=float) / max(float(len(group)), 1.0)
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())
        entropy_norm = entropy / np.log(len(probs)) if len(probs) > 1 else 0.0
        dominant_share = float(lineage_counts.iloc[0] / max(len(group), 1))
        unique_airfoils = int(group["airfoil_signature"].astype(str).replace("", np.nan).dropna().nunique()) if "airfoil_signature" in group.columns else np.nan
        rows.append(
            {
                "generation": int(generation),
                "population_size": int(len(group)),
                "unique_lineages": int(lineage_counts.size),
                "unique_airfoils": unique_airfoils,
                "dominant_lineage_share": dominant_share,
                "lineage_entropy_norm": float(entropy_norm),
            }
        )
    plot_df = pd.DataFrame(rows).sort_values("generation")
    _apply_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.4), sharex=True)

    axes[0].plot(
        plot_df["generation"],
        plot_df["unique_lineages"],
        color=PLOT_COLOR_CYCLE["takeover"],
        linewidth=2.5,
        marker="o",
        label="Unique lineages",
    )
    if np.isfinite(plot_df["unique_airfoils"]).any():
        axes[0].plot(
            plot_df["generation"],
            plot_df["unique_airfoils"],
            color=PLOT_COLOR_CYCLE["diversity"],
            linewidth=2.3,
            marker="s",
            label="Unique airfoil signatures",
        )
    axes[0].set_ylabel("Count")
    axes[0].set_title("Diversity counts", fontsize=13, weight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        plot_df["generation"],
        plot_df["dominant_lineage_share"],
        color=PLOT_COLOR_CYCLE["baseline"],
        linewidth=2.5,
        marker="D",
        label="Dominant lineage share",
    )
    axes[1].plot(
        plot_df["generation"],
        plot_df["lineage_entropy_norm"],
        color=PLOT_COLOR_CYCLE["efficiency"],
        linewidth=2.3,
        marker="o",
        label="Normalized lineage entropy",
    )
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Share / normalized entropy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Takeover intensity", fontsize=13, weight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Lineage Takeover Metrics", fontsize=16, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    plot_df.to_csv(output_path.with_suffix(".csv"), index=False)
    _write_plot_note(
        output_path,
        title="Lineage Takeover Metrics",
        importance=(
            "Tree plots are visually rich but hard to summarize quantitatively. This figure compresses genealogy into a few numbers "
            "that tell you whether the run is still exploring many lineages or whether a small number of founders has taken over the search."
        ),
        prior_work=[
            f"Embodied intelligence via learning and evolution uses phylogenetic analysis to reason about how morphology families spread ({PAPER_LINKS['derl']}).",
            f"Morpho-evolution with learning using a controller archive analyzes how search support changes evolutionary dynamics over generations ({PAPER_LINKS['inheritance']}).",
        ],
        what_to_read=[
            "Unique lineages dropping quickly indicates strong takeover.",
            "Dominant lineage share rising toward 1 means one family is monopolizing the population.",
            "Entropy distinguishes healthy multi-lineage coexistence from near-clonal convergence.",
        ],
        takeaways=[
            "This plot is the quantitative counterpart of the genealogy and Muller figures.",
            "If performance improves while entropy collapses, the run is exploiting hard; if both stay broad, the run is still exploring.",
        ],
    )

def plot_airfoil_takeover(work_df: pd.DataFrame, output_path: Path, top_n: int = 6) -> None:
    required = {"generation", "airfoil_signature"}
    missing = required - set(work_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns for airfoil takeover plot: {missing_list}")

    clean_df = work_df.copy()
    clean_df["airfoil_signature"] = clean_df["airfoil_signature"].astype(str).str.strip()
    clean_df = clean_df.loc[clean_df["airfoil_signature"] != ""].copy()
    if clean_df.empty:
        raise ValueError("No airfoil signatures available for takeover plot.")

    counts = (
        clean_df.groupby(["generation", "airfoil_signature"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    totals = counts.sum(axis=0).sort_values(ascending=False)
    top_labels = totals.head(top_n).index.tolist()
    other_cols = [col for col in counts.columns if col not in top_labels]
    frac_df = counts.div(counts.sum(axis=1), axis=0)
    plot_df = frac_df.loc[:, top_labels].copy()
    if other_cols:
        plot_df["other"] = frac_df.loc[:, other_cols].sum(axis=1)
    plot_df = plot_df.fillna(0.0)

    summary_rows = []
    for generation, row in plot_df.iterrows():
        unique_airfoils = int(counts.loc[generation].astype(bool).sum())
        dominant_share = float(frac_df.loc[generation].max())
        summary_rows.append(
            {
                "generation": int(generation),
                "unique_airfoils": unique_airfoils,
                "dominant_airfoil_share": dominant_share,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("generation")

    _apply_plot_style()
    fig = plt.figure(figsize=(11.2, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.3], hspace=0.18)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    x = plot_df.index.to_numpy(dtype=float)
    colors = list(plt.get_cmap("tab20").colors[: len(plot_df.columns)])
    ax0.stackplot(x, [plot_df[col].to_numpy(dtype=float) for col in plot_df.columns], labels=plot_df.columns.tolist(), colors=colors, alpha=0.9)
    ax0.set_ylabel("Population share")
    ax0.set_ylim(0.0, 1.0)
    ax0.set_title("Airfoil-family takeover", fontsize=13, weight="bold")
    ax0.grid(alpha=0.18)
    ax0.legend(loc="upper left", ncol=2, fontsize=8.7)

    ax1.plot(
        summary_df["generation"],
        summary_df["unique_airfoils"],
        color=PLOT_COLOR_CYCLE["diversity"],
        linewidth=2.4,
        marker="o",
        label="Unique airfoils",
    )
    ax1.set_ylabel("Unique airfoils")
    ax1.grid(alpha=0.3)
    ax1.set_xlabel("Generation")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        summary_df["generation"],
        summary_df["dominant_airfoil_share"],
        color=PLOT_COLOR_CYCLE["baseline"],
        linewidth=2.4,
        marker="D",
        label="Dominant share",
    )
    ax1_twin.set_ylabel("Dominant share")
    ax1_twin.set_ylim(0.0, 1.0)
    lines = ax1.get_lines() + ax1_twin.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")

    fig.suptitle("Airfoil Signature Dynamics", fontsize=16, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    plot_df.reset_index().rename(columns={"index": "generation"}).to_csv(output_path.with_suffix(".csv"), index=False)
    _write_plot_note(
        output_path,
        title="Airfoil Signature Dynamics",
        importance=(
            "This plot shows whether evolution is converging toward a few airfoil families and when that takeover happens. "
            "In this project that matters especially because the current topology signal is really an airfoil-class signal."
        ),
        prior_work=[
            f"Unconventional Hexacopters via Evolution and Learning studies how morphology families and asymmetries emerge over generations ({PAPER_LINKS['hexa']}).",
            f"Embodied intelligence via learning and evolution studies morphological family spread through phylogenetic dynamics ({PAPER_LINKS['derl']}).",
        ],
        what_to_read=[
            "A stable stacked area means an airfoil family became entrenched.",
            "Late turnovers indicate that airfoil family identity remained under active search pressure.",
            "Unique-airfoil count plus dominant-share tells you whether convergence is soft or abrupt.",
        ],
        takeaways=[
            "This is the honest version of the current topology story: here the changing discrete family is the NACA signature.",
            "If a small number of signatures dominate while performance keeps improving, the main search is happening inside geometry and controller space.",
        ],
        caveats=[
            "This is not structural topology evolution. It is airfoil-family takeover.",
        ],
    )

def plot_innovation_payoff(selection_df: pd.DataFrame, output_path: Path) -> None:
    required = {"origin", "primary_parent_uid", "chromosome", "offspring_vs_best_parent_scalar_delta"}
    missing = required - set(selection_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns for innovation-payoff plot: {missing_list}")

    Chromosome_Drone = _load_chromosome_drone_class()
    scale = np.asarray(Chromosome_Drone.PHYS_MAX - Chromosome_Drone.PHYS_MIN, dtype=float)
    scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)

    uid_to_phys: dict[int, np.ndarray] = {}
    for row in selection_df.itertuples(index=False):
        uid = int(getattr(row, "uid"))
        try:
            chrom = np.asarray(ast.literal_eval(getattr(row, "chromosome")), dtype=float)
        except Exception:
            continue
        uid_to_phys[uid] = np.asarray(Chromosome_Drone.to_physical(chrom), dtype=float)

    rows = []
    offspring_df = selection_df.loc[selection_df["origin"].astype(str) == "offspring"].copy()
    for row in offspring_df.itertuples(index=False):
        parent_uid = int(getattr(row, "primary_parent_uid", -1))
        child_uid = int(getattr(row, "uid"))
        if parent_uid < 0 or parent_uid not in uid_to_phys or child_uid not in uid_to_phys:
            continue
        parent = uid_to_phys[parent_uid]
        child = uid_to_phys[child_uid]
        dist = float(np.sqrt(np.mean(((child - parent) / scale) ** 2)))
        delta = float(pd.to_numeric(getattr(row, "offspring_vs_best_parent_scalar_delta"), errors="coerce"))
        selected = int(getattr(row, "selected_next_generation", getattr(row, "selected", 0)))
        if not np.isfinite(dist) or not np.isfinite(delta):
            continue
        rows.append(
            {
                "generation": int(getattr(row, "generation")),
                "uid": child_uid,
                "primary_parent_uid": parent_uid,
                "innovation_distance": dist,
                "scalar_delta_vs_best_parent": delta,
                "selected_next_generation": selected,
                "improved_vs_best_parent": int(delta > 0.0),
                "airfoil_mutation": int(getattr(row, "airfoil_mutation", 0)),
            }
        )
    plot_df = pd.DataFrame(rows).sort_values("innovation_distance")
    if plot_df.empty:
        raise ValueError("No valid offspring-parent pairs available for innovation-payoff plot.")

    quantiles = np.linspace(0.0, 1.0, 7)
    edges = np.unique(np.quantile(plot_df["innovation_distance"], quantiles))
    if edges.size < 3:
        edges = np.linspace(float(plot_df["innovation_distance"].min()), float(plot_df["innovation_distance"].max()) + 1e-9, 4)
    plot_df["distance_bin"] = pd.cut(plot_df["innovation_distance"], bins=edges, include_lowest=True, duplicates="drop")
    bins = (
        plot_df.groupby("distance_bin", observed=False)
        .agg(
            x_center=("innovation_distance", "mean"),
            median_delta=("scalar_delta_vs_best_parent", "median"),
            survival_rate=("selected_next_generation", "mean"),
            improvement_rate=("improved_vs_best_parent", "mean"),
            n_samples=("innovation_distance", "size"),
        )
        .reset_index(drop=True)
    )

    _apply_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.8), sharex=True)
    sel_mask = plot_df["selected_next_generation"].astype(int) == 1
    axes[0].scatter(
        plot_df.loc[~sel_mask, "innovation_distance"],
        plot_df.loc[~sel_mask, "scalar_delta_vs_best_parent"],
        color=PLOT_COLOR_CYCLE["rejected"],
        alpha=0.55,
        s=26,
        edgecolors="none",
        label="Not selected",
    )
    axes[0].scatter(
        plot_df.loc[sel_mask, "innovation_distance"],
        plot_df.loc[sel_mask, "scalar_delta_vs_best_parent"],
        color=PLOT_COLOR_CYCLE["selected"],
        alpha=0.85,
        s=30,
        edgecolors="white",
        linewidths=0.25,
        label="Selected",
    )
    axes[0].plot(
        bins["x_center"],
        bins["median_delta"],
        color="#111827",
        linewidth=2.4,
        linestyle="--",
        marker="o",
        label="Binned median delta",
    )
    axes[0].axhline(0.0, color="#334155", linewidth=1.4, linestyle=":")
    axes[0].set_ylabel("Delta vs best parent")
    axes[0].set_title("Payoff of larger morphological moves", fontsize=13, weight="bold")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        bins["x_center"],
        bins["survival_rate"],
        color=PLOT_COLOR_CYCLE["selected"],
        linewidth=2.4,
        marker="o",
        label="Selection rate",
    )
    axes[1].plot(
        bins["x_center"],
        bins["improvement_rate"],
        color=PLOT_COLOR_CYCLE["innovation"],
        linewidth=2.4,
        marker="s",
        label="Improvement rate",
    )
    axes[1].set_xlabel("Parent-child innovation distance")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Selection and improvement by distance bin", fontsize=13, weight="bold")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Innovation Payoff", fontsize=16, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    plot_df.to_csv(output_path.with_suffix(".csv"), index=False)
    _write_plot_note(
        output_path,
        title="Innovation Payoff",
        importance=(
            "This plot measures whether large morphological moves are useful in your search or whether progress mostly comes from small local refinements. "
            "That is a key missing piece if you want to explain how morphology search behaves, not only where it ends."
        ),
        prior_work=[
            f"Unconventional Hexacopters via Evolution and Learning motivates the question by showing that unusual morphologies can be beneficial ({PAPER_LINKS['hexa']}).",
            f"Morpho-evolution with learning using a controller archive studies how search-support mechanisms change exploration versus exploitation ({PAPER_LINKS['inheritance']}).",
        ],
        what_to_read=[
            "If median delta stays positive at larger distances, bold morphology changes are paying off.",
            "If survival rate collapses with distance, the search is mainly exploiting locally.",
            "If improvement rate stays high but survival falls, large changes create rare but important breakthroughs.",
        ],
        takeaways=[
            "This plot is a direct answer to whether innovation is being rewarded or merely filtered out.",
            "It is especially useful to interpret mutation and airfoil-change events beyond raw counts.",
        ],
    )
