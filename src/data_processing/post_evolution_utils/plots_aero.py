from __future__ import annotations

from .common import *  # noqa: F401,F403
from .descriptors import *  # noqa: F401,F403

def plot_aero_parameter_evolution(df: pd.DataFrame, output_path: Path) -> None:
    desc_df = _aero_descriptor_dataframe(df)
    desc_df["generation"] = pd.to_numeric(desc_df["generation"], errors="coerce")
    desc_df = desc_df[np.isfinite(desc_df["generation"])].copy()
    descriptors = [
        ("static_margin_pct_mac", "Static margin [% MAC]"),
        ("neutral_point_pct_mac", "Neutral point [% MAC]"),
        ("cg_pct_mac", "CG position [% MAC]"),
        ("horizontal_tail_volume_coeff", "Horizontal tail volume coefficient"),
        ("vertical_tail_volume_coeff", "Vertical tail volume coefficient"),
        ("wing_loading_n_m2", "Wing loading [N/m^2]"),
        ("wing_clcd_max_2d", "Wing airfoil max L/D [-]"),
        ("aircraft_ld_max_est", "Aircraft max L/D estimate [-]"),
        ("downwash_gradient", "Downwash gradient [-]"),
        ("aircraft_cd0_est", "Aircraft CD0 estimate [-]"),
    ]
    summaries = _descriptor_summary_by_generation(desc_df, [col for col, _ in descriptors])
    colors = [
        "#0f766e",
        "#0284c7",
        "#6d28d9",
        "#b45309",
        "#b91c1c",
        "#7c3aed",
        "#1d4ed8",
        "#475569",
    ]

    _apply_plot_style()
    fig, axes = plt.subplots(5, 2, figsize=(13.6, 17.0), sharex=True)
    axes_flat = axes.flatten()
    export_frames = []
    for ax, (color, (col, label)) in zip(axes_flat, zip(colors, descriptors)):
        summary = summaries[col].copy()
        x = summary["generation"].to_numpy(dtype=float)
        med = summary["median"].to_numpy(dtype=float)
        q1 = summary["q1"].to_numpy(dtype=float)
        q3 = summary["q3"].to_numpy(dtype=float)
        ax.fill_between(x, q1, q3, color=color, alpha=0.16, linewidth=0)
        ax.plot(x, med, color=color, linewidth=2.5, marker="o", markersize=4.2)
        ax.set_title(label, fontsize=12.2, weight="bold")
        ax.grid(alpha=0.3)
        ax.set_ylim(_scaled_ylim_from_series([q1[np.isfinite(q1)], q3[np.isfinite(q3)], med[np.isfinite(med)]], padding_ratio=0.1))
        export_df = summary.copy()
        export_df["descriptor"] = col
        export_frames.append(export_df)
    for ax in axes_flat[-2:]:
        ax.set_xlabel("Generation")
    fig.suptitle("Evolution of Derived Aerodynamic Parameters", fontsize=16, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    pd.concat(export_frames, ignore_index=True).to_csv(output_path.with_suffix(".csv"), index=False)
    _write_plot_note(
        output_path,
        title="Evolution of Derived Aerodynamic Parameters",
        importance=(
            "Raw genes are hard to interpret physically. This figure converts genome coordinates into geometry and solver-level wing parameters, "
            "so you can tell what aerodynamic phenotype the search is actually preferring over time."
        ),
        prior_work=[
            f"Unconventional Hexacopters via Evolution and Learning interprets performance through morphology descriptors rather than only raw genes ({PAPER_LINKS['hexa']}).",
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots compares how different controller regimes select different body parameters ({PAPER_LINKS['controller_learning']}).",
        ],
        what_to_read=[
            "Wing loading, wing airfoil max L/D, and aircraft max L/D summarize how each morphology trades supportable lift against parasite and induced drag.",
            "CG position, neutral point, and static margin are expressed on the wing MAC using the exact total CG reconstructed from all URDF links.",
            "Horizontal and vertical tail volume coefficients summarize how much pitch and yaw authority the layout gets from both surface area and moment arm.",
            "The NACA-derived cd0, alpha0, alpha_stall and Re_nom are the solver-effective wing parameters, while the tail follows the solver's fixed 0216 override.",
        ],
        takeaways=[
            "This is the plot that turns genome evolution into aerodynamic evolution.",
            "If geometry moves a lot but NACA-derived parameters stay stable, search is refining layout around one preferred airfoil family.",
            "If NACA-derived parameters drift strongly, the airfoil choice itself is an active part of the evolutionary story.",
        ],
        caveats=[
            "The tail airfoil is fixed by the solver override path, so the varying NACA-derived parameters here are wing-side quantities.",
            "Static margin and neutral point are estimated with a classical wing-tail-fuselage stick-fixed formulation and the exact total CG reconstructed from all URDF links.",
            "Horizontal tail volume is the classical dimensionless coefficient S_t l_t / (S_w c_bar).",
        ],
    )

def plot_aero_parameter_top10pct_by_fitness(df: pd.DataFrame, output_path: Path) -> None:
    desc_df = _aero_descriptor_dataframe(df)
    desc_df["generation"] = pd.to_numeric(desc_df["generation"], errors="coerce")
    desc_df = desc_df[np.isfinite(desc_df["generation"])].copy()

    descriptors = [
        {
            "cols": ["static_margin_pct_mac"],
            "title": "Static Margin [% MAC]",
        },
        {
            "cols": ["horizontal_tail_volume_coeff"],
            "title": "Horizontal Tail Volume Coefficient",
        },
        {
            "cols": ["wing_loading_n_m2"],
            "title": "Wing Loading [N/m^2]",
        },
        {
            "cols": ["aircraft_ld_max_est"],
            "title": "Aircraft max L/D estimate [-]",
        },
        {
            "cols": ["downwash_gradient"],
            "title": "Downwash Gradient [-]",
        },
        {
            "cols": ["aircraft_cd0_est"],
            "title": "Aircraft CD0 estimate [-]",
        },
    ]
    elite_specs = [
        ("ff_0", "Average value across top 10% speed (4/40)", PLOT_COLOR_CYCLE["speed"]),
        ("ff_1", "Average value across top 10% efficiency (4/40)", PLOT_COLOR_CYCLE["efficiency"]),
        ("ff_2", "Average value across top 10% progress (4/40)", PLOT_COLOR_CYCLE["progress"]),
    ]

    rows: list[dict[str, float | int | str]] = []
    for generation, group in desc_df.groupby("generation", sort=True):
        for fitness, label, _ in elite_specs:
            values = pd.to_numeric(group[fitness], errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(values)
            if not finite_mask.any():
                continue
            n_pick = max(1, int(np.ceil(np.sum(finite_mask) * 0.10)))
            if fitness == "ff_1":
                metric_values = -values
                order = np.argsort(metric_values[finite_mask], kind="mergesort")
                selected_local = np.where(finite_mask)[0][order[:n_pick]]
            else:
                order = np.argsort(values[finite_mask], kind="mergesort")[::-1]
                selected_local = np.where(finite_mask)[0][order[:n_pick]]
            elite_mask = np.zeros(len(group), dtype=bool)
            elite_mask[selected_local] = True
            if not elite_mask.any():
                continue

            elite_df = group.loc[elite_mask].copy()
            row: dict[str, float | int | str] = {
                "generation": int(generation),
                "elite_group": label,
                "fitness": fitness,
                "n_selected": int(len(elite_df)),
            }
            for spec in descriptors:
                for col in spec["cols"]:
                    row[col] = float(pd.to_numeric(elite_df[col], errors="coerce").mean())
            rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(["fitness", "generation"])
    if summary_df.empty:
        raise ValueError("No elite aerodynamic-parameter summaries available.")

    _apply_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 12.2), sharex=True)
    axes_flat = axes.flatten()

    for ax, spec in zip(axes_flat, descriptors):
        y_series = []
        for fitness, elite_label, color in elite_specs:
            plot_df = summary_df.loc[summary_df["fitness"] == fitness].copy()
            if plot_df.empty:
                continue
            x = plot_df["generation"].to_numpy(dtype=float)
            variant_labels = spec.get("variant_labels", [None] * len(spec["cols"]))
            variant_styles = spec.get("variant_styles", ["-"] * len(spec["cols"]))
            for col, variant_label, linestyle in zip(spec["cols"], variant_labels, variant_styles):
                y = pd.to_numeric(plot_df[col], errors="coerce").to_numpy(dtype=float)
                finite_mask = np.isfinite(y)
                if not finite_mask.any():
                    continue
                line_label = elite_label if len(spec["cols"]) == 1 else f"{elite_label} | {variant_label}"
                ax.plot(
                    x[finite_mask],
                    y[finite_mask],
                    color=color,
                    linewidth=2.8 if linestyle == "-" else 2.4,
                    linestyle=linestyle,
                    marker="o" if linestyle == "-" else None,
                    markersize=5.2,
                    label=line_label,
                )
                y_series.append(y[finite_mask])
        ax.set_title(spec["title"], fontsize=15, weight="bold")
        ax.set_ylabel(spec["title"], fontsize=13, fontweight="normal")
        ax.set_xlabel("Generation", fontsize=13)
        ax.tick_params(axis="both", labelsize=11.5)
        ax.grid(alpha=0.3)
        if y_series:
            ax.set_ylim(_scaled_ylim_from_series(y_series, padding_ratio=0.1))

    axes_flat[0].legend(loc="best", fontsize=13, frameon=True)
    fig.suptitle(
        "Aerodynamic Parameters of Top-10% Objective Specialists",
        fontsize=19,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    summary_df.to_csv(output_path.with_suffix(".csv"), index=False)
    _write_plot_note(
        output_path,
        title="Aerodynamic parameters of top-10% objective specialists",
        importance=(
            "This figure does not ask how the whole population moves, but how the elite niches move. It shows which aerodynamic phenotype is preferred "
            "over time by the top 10% speed, efficiency and progress specialists."
        ),
        prior_work=[
            f"Unconventional Hexacopters via Evolution and Learning interprets different high-performing morphology families rather than only population averages ({PAPER_LINKS['hexa']}).",
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots is directly relevant because elite morphologies differ by optimization regime and objective pressure ({PAPER_LINKS['controller_learning']}).",
        ],
        what_to_read=[
            "The three curves show how the elite phenotype for each objective changes generation by generation.",
            "If the curves separate strongly, the three objectives are selecting genuinely different aerodynamic niches.",
            "If the curves converge late in evolution, the search may be discovering a shared good design region despite different objectives.",
        ],
        takeaways=[
            "This is the aerodynamic analogue of the top-10% specialist view, but much more interpretable physically.",
            "It helps you say not only that objectives differ, but exactly how their preferred bodies differ in aerodynamic terms.",
        ],
        caveats=[
            "The top-10% is computed separately inside each generation, so this is a moving-elite view, not a fixed-lineage view.",
        ],
    )

def plot_nose_cg_minus_nose_wing_ac_top10_by_fitness(df: pd.DataFrame, output_path: Path) -> None:
    desc_df = _aero_descriptor_dataframe(df)
    desc_df["generation"] = pd.to_numeric(desc_df["generation"], errors="coerce")
    desc_df = desc_df[np.isfinite(desc_df["generation"])].copy()

    elite_specs = [
        ("ff_0", "Top 10% speed", PLOT_COLOR_CYCLE["speed"]),
        ("ff_1", "Top 10% efficiency", PLOT_COLOR_CYCLE["efficiency"]),
        ("ff_2", "Top 10% progress", PLOT_COLOR_CYCLE["progress"]),
    ]
    rows: list[dict[str, float | int | str]] = []
    for generation, group in desc_df.groupby("generation", sort=True):
        for fitness, label, _ in elite_specs:
            values = pd.to_numeric(group[fitness], errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(values)
            if not finite_mask.any():
                continue
            n_pick = max(1, int(np.ceil(np.sum(finite_mask) * 0.10)))
            if fitness == "ff_1":
                metric_values = -values
                order = np.argsort(metric_values[finite_mask], kind="mergesort")
                selected_local = np.where(finite_mask)[0][order[:n_pick]]
            else:
                order = np.argsort(values[finite_mask], kind="mergesort")[::-1]
                selected_local = np.where(finite_mask)[0][order[:n_pick]]
            elite_mask = np.zeros(len(group), dtype=bool)
            elite_mask[selected_local] = True
            if not elite_mask.any():
                continue

            elite_df = group.loc[elite_mask].copy()
            row: dict[str, float | int | str] = {
                "generation": int(generation),
                "elite_group": label,
                "fitness": fitness,
                "n_selected": int(len(elite_df)),
            }
            for col in ["static_margin_pct_mac", "cg_pct_mac", "neutral_point_pct_mac"]:
                row[col] = float(pd.to_numeric(elite_df[col], errors="coerce").mean())
            rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(["fitness", "generation"])
    if summary_df.empty:
        raise ValueError("No static-margin / CG / neutral-point summaries available.")

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12.0, 6.8))

    y_series = []
    for fitness, elite_label, color in elite_specs:
        plot_df = summary_df.loc[summary_df["fitness"] == fitness].copy()
        if plot_df.empty:
            continue
        x = plot_df["generation"].to_numpy(dtype=float)
        center = pd.to_numeric(plot_df["static_margin_pct_mac"], errors="coerce").to_numpy(dtype=float)
        lower = pd.to_numeric(plot_df["cg_pct_mac"], errors="coerce").to_numpy(dtype=float)
        upper = pd.to_numeric(plot_df["neutral_point_pct_mac"], errors="coerce").to_numpy(dtype=float)
        finite_center = np.isfinite(center)
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        if finite_center.any():
            ax.plot(
                x[finite_center],
                center[finite_center],
                color=color,
                linewidth=3.0,
                marker="o",
                markersize=5.2,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=f"{elite_label} | static margin",
            )
            y_series.append(center[finite_center])
        if finite_lower.any():
            ax.plot(
                x[finite_lower],
                lower[finite_lower],
                color=color,
                linewidth=2.1,
                linestyle="--",
                alpha=0.95,
                label=f"{elite_label} | CG",
            )
            y_series.append(lower[finite_lower])
        if finite_upper.any():
            ax.plot(
                x[finite_upper],
                upper[finite_upper],
                color=color,
                linewidth=2.1,
                linestyle=":",
                alpha=0.95,
                label=f"{elite_label} | neutral point",
            )
            y_series.append(upper[finite_upper])

    ax.set_title("CG, Neutral Point, and Static Margin of Objective Specialists", fontsize=14, weight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("[% MAC]")
    ax.grid(alpha=0.28)
    if y_series:
        ax.set_ylim(_scaled_ylim_from_series(y_series, padding_ratio=0.12))
    ax.legend(loc="best", ncol=3, fontsize=8.8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    summary_df.to_csv(output_path.with_suffix(".csv"), index=False)
    _write_plot_note(
        output_path,
        title="CG, neutral point, and static margin of objective specialists",
        importance=(
            "This figure tracks the exact CG position, the estimated neutral point, and the resulting static margin for each elite objective niche."
        ),
        prior_work=[
            f"Unconventional Hexacopters via Evolution and Learning motivates interpreting elite morphology families through concrete mechanical degrees of freedom ({PAPER_LINKS['hexa']}).",
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots is relevant because objective pressure changes which body mechanisms and parameter ranges are selected ({PAPER_LINKS['controller_learning']}).",
        ],
        what_to_read=[
            "The solid line is the estimated static margin.",
            "The dashed and dotted lines are the CG and neutral-point locations expressed on the wing MAC.",
            "The separation between CG and neutral point is the static-margin reserve available to each elite family.",
        ],
        takeaways=[
            "It is the clearest plot for seeing whether each objective family keeps a real static-margin reserve or converges toward marginal stability.",
            "Reading CG and neutral point together shows whether a stability change comes from mass distribution, tail authority, fuselage destabilization, or a combination of them.",
        ],
        caveats=[
            "This is still a compact stick-fixed estimate, not a full dynamic stability derivative identification or a trimmed flight-condition solve.",
        ],
    )

def plot_aero_archetype_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    desc_df = _aero_descriptor_dataframe(df)
    desc_df["generation"] = pd.to_numeric(desc_df["generation"], errors="coerce")
    desc_df = desc_df[np.isfinite(desc_df["generation"])].copy()
    final_generation = int(desc_df["generation"].max())
    final_df = desc_df.loc[desc_df["generation"] == final_generation].copy()
    if final_df.empty:
        raise ValueError("No final-generation rows available for archetype heatmap.")

    if "is_pareto" in final_df.columns and final_df["is_pareto"].astype(float).gt(0).any():
        base_df = final_df.loc[final_df["is_pareto"].astype(float) > 0].copy()
    else:
        base_df = final_df.copy()

    descriptor_cols = [
        "static_margin_pct_mac",
        "neutral_point_pct_mac",
        "horizontal_tail_volume_coeff",
        "vertical_tail_volume_coeff",
        "wing_loading_n_m2",
        "wing_clcd_max_2d",
        "aircraft_ld_max_est",
        "downwash_gradient",
        "aircraft_cd0_est",
        "fuselage_cm_alpha_est",
    ]
    descriptor_labels = [
        "Static margin [% MAC]",
        "Neutral point [% MAC]",
        "Horizontal tail volume",
        "Vertical tail volume",
        "Wing loading [N/m^2]",
        "Wing airfoil max L/D",
        "Aircraft max L/D est.",
        "Downwash gradient",
        "Aircraft CD0 est.",
        "Fuselage Cmalpha est.",
    ]

    n_pick = max(3, int(np.ceil(len(base_df) * 0.1)))
    score_df = base_df.copy()
    points = _fitness_points_for_plot(score_df, FITNESS_COLUMNS)
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    spans = np.where(np.abs(maxs - mins) > 1e-12, maxs - mins, 1.0)
    normalized = (points - mins) / spans
    score_df["balanced_score"] = normalized.sum(axis=1)

    groups = {
        "Speed specialists": score_df.nlargest(n_pick, "ff_0"),
        "Efficiency specialists": score_df.nlargest(n_pick, "ff_1"),
        "Progress specialists": score_df.nlargest(n_pick, "ff_2"),
        "Balanced knee": score_df.nlargest(n_pick, "balanced_score"),
    }

    bix3_phys = np.asarray(ast.literal_eval(TARGET_BIX3_URDF_PARAMS), dtype=float).reshape(1, -1)
    bix3_desc = _aero_descriptor_frame_from_physical(bix3_phys)
    bix3_desc.index = ["BIX3"]

    pop_mean = final_df[descriptor_cols].apply(pd.to_numeric, errors="coerce").mean()
    pop_std = final_df[descriptor_cols].apply(pd.to_numeric, errors="coerce").std().replace(0.0, 1.0).fillna(1.0)

    heat_rows = []
    heat_index = []
    for name, group in groups.items():
        med = group[descriptor_cols].apply(pd.to_numeric, errors="coerce").median()
        heat_rows.append(((med - pop_mean) / pop_std).to_numpy(dtype=float))
        heat_index.append(name)
    heat_rows.append(((bix3_desc.iloc[0][descriptor_cols].astype(float) - pop_mean) / pop_std).to_numpy(dtype=float))
    heat_index.append("BIX3")

    heat = np.clip(np.vstack(heat_rows), -2.5, 2.5)

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    im = ax.imshow(heat, cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(descriptor_labels)))
    ax.set_xticklabels(descriptor_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(heat_index)))
    ax.set_yticklabels(heat_index)
    ax.set_title(f"Final-generation aerodynamic archetypes (generation {final_generation})", fontsize=14, weight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:+.1f}", ha="center", va="center", fontsize=8.8, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Descriptor z-score vs final population")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    pd.DataFrame(heat, index=heat_index, columns=descriptor_cols).to_csv(output_path.with_suffix(".csv"))
    _write_plot_note(
        output_path,
        title="Aerodynamic archetypes of the final generation",
        importance=(
            "A Pareto front tells you that different niches exist, but not what kind of body lives in each niche. "
            "This heatmap summarizes the phenotype of speed, efficiency, progress and balanced specialists in one figure."
        ),
        prior_work=[
            f"Unconventional Hexacopters via Evolution and Learning interprets evolved designs through morphology categories and unconventional asymmetries ({PAPER_LINKS['hexa']}).",
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots shows that controller regime changes which morphology families are preferred ({PAPER_LINKS['controller_learning']}).",
        ],
        what_to_read=[
            "Positive z-scores mean the archetype is above the final-population mean on that descriptor; negative means below.",
            "Compare speed and efficiency specialists to see whether they differ mostly in geometry, in airfoil-derived solver parameters, or both.",
            "Compare the balanced knee and BIX3 to see whether your compromise solution is conventional or truly unconventional.",
        ],
        takeaways=[
            "This figure explains why there are different winners on the same Pareto front.",
            "It is a compact way to translate optimization niches into interpretable aerodynamic phenotypes.",
        ],
    )

def plot_comparative_top10pct_aero_distributions(
    run_dfs: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    subset_mode: str = "final_generation_pareto",
) -> None:
    descriptor_specs = [
        ("static_margin_pct_mac", "Static Margin [% MAC]"),
        ("horizontal_tail_volume_coeff", "Horizontal Tail Volume Coefficient"),
        ("wing_loading_n_m2", "Wing Loading [N/m^2]"),
        ("aircraft_ld_max_est", "Aircraft max L/D estimate [-]"),
        ("downwash_gradient", "Downwash Gradient [-]"),
        ("aircraft_cd0_est", "Aircraft CD0 estimate [-]"),
    ]
    subset_title = _comparison_subset_title(subset_mode)
    subset_dfs = _comparison_subset_by_run(run_dfs, subset_mode)
    desc_by_run: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, float | int | str]] = []

    for run_name, subset_df in subset_dfs.items():
        desc_df = _aero_descriptor_dataframe(subset_df)
        desc_by_run[run_name] = desc_df
        for col, label in descriptor_specs:
            values = pd.to_numeric(desc_df[col], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            summary_rows.append(
                {
                    "run_name": run_name,
                    "subset_mode": subset_mode,
                    "descriptor": col,
                    "label": label,
                    "n_samples": int(finite.size),
                    "median": float(np.nanmedian(finite)) if finite.size else np.nan,
                    "q1": float(np.nanpercentile(finite, 25)) if finite.size else np.nan,
                    "q3": float(np.nanpercentile(finite, 75)) if finite.size else np.nan,
                    "mean": float(np.nanmean(finite)) if finite.size else np.nan,
                    "std": float(np.nanstd(finite, ddof=1)) if finite.size > 1 else np.nan,
                }
            )

    _apply_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 12.6))
    axes_flat = axes.flatten()
    label_order = ["nsga_GP", "nsga_SP"]
    y_pos = [1, 0]
    rng = np.random.default_rng(11)
    export_rows: list[dict[str, float | int | str]] = []

    for ax, (col, label) in zip(axes_flat, descriptor_specs):
        vals_by_run = {}
        for run_name in label_order:
            vals = pd.to_numeric(desc_by_run[run_name][col], errors="coerce").to_numpy(dtype=float)
            vals_by_run[run_name] = vals[np.isfinite(vals)]
        if not any(vals.size for vals in vals_by_run.values()):
            ax.axis("off")
            continue

        data = [vals_by_run["nsga_GP"], vals_by_run["nsga_SP"]]
        combined = np.concatenate([vals for vals in data if vals.size], axis=0)
        xlim = _scaled_ylim(combined, padding_ratio=0.06)

        bp = ax.boxplot(
            data,
            vert=False,
            positions=y_pos,
            widths=0.5,
            patch_artist=True,
            whis=(10, 90),
            showfliers=False,
            medianprops=dict(color="#111827", linewidth=2.0),
            whiskerprops=dict(color="#475569", linewidth=1.2),
            capprops=dict(color="#475569", linewidth=1.2),
        )
        for patch, run_name in zip(bp["boxes"], label_order):
            patch.set_facecolor(EVOLUTION_COLORS[run_name])
            patch.set_alpha(0.35)
            patch.set_edgecolor(EVOLUTION_COLORS[run_name])
            patch.set_linewidth(1.3)

        for row_pos, run_name, vals in zip(y_pos, label_order, data):
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=vals.size)
            ax.scatter(
                vals,
                row_pos + jitter,
                s=22,
                alpha=0.35,
                color=EVOLUTION_COLORS[run_name],
                edgecolors="none",
                zorder=3,
            )

        gp_vals = data[0]
        sp_vals = data[1]
        stats = _two_sample_distribution_stats(gp_vals, sp_vals)
        median_delta = float(stats["median_delta_sp_minus_gp"])
        effect = float(stats["standardized_delta"])
        p_value = float(stats["p_value"])
        export_rows.append(
            {
                "subset_mode": subset_mode,
                "descriptor": col,
                "x_min": float(xlim[0]),
                "x_max": float(xlim[1]),
                **stats,
            }
        )

        ax.set_title(label, fontsize=12.5, weight="bold")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_comparison_run_label(name) for name in label_order], fontsize=9.1)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlim(xlim)
        ax.grid(alpha=0.22, axis="x")
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

    fig.suptitle(f"Aerodynamic descriptor distributions by controller regime\n{subset_title}", fontsize=17, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(output_path.with_suffix(".csv"), index=False)
    effect_df = pd.DataFrame(export_rows)
    if not effect_df.empty and "p_value" in effect_df.columns:
        effect_df["p_value_fdr_bh"] = _benjamini_hochberg(pd.to_numeric(effect_df["p_value"], errors="coerce").to_numpy(dtype=float))
    effect_df.to_csv(output_path.with_name(output_path.stem + "_effect_sizes.csv"), index=False)
    _write_plot_note(
        output_path,
        title=f"Aerodynamic descriptor distributions by controller regime | {subset_title}",
        importance=(
            "This plot mirrors the gene-distribution comparison, but on six aerodynamic quantities that summarize longitudinal stability, tail authority, drag build-up, and overall efficiency."
        ),
        prior_work=[
            f"Unconventional Hexacopters via Evolution and Learning argues for comparing evolved families through interpretable morphology descriptors ({PAPER_LINKS['hexa']}).",
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots motivates comparing morphology under different controller assumptions ({PAPER_LINKS['controller_learning']}).",
        ],
        what_to_read=[
            f"This specific version uses: {subset_title.lower()}.",
            "Each panel compares the two controller regimes on one descriptor, using the same light dot plus compact box summary format as the gene plots.",
            "The annotation shows median shift and standardized shift, so you can see both direction and rough separation strength.",
        ],
        takeaways=[
            "Use this family of plots when you want a physical reading of controller-regime divergence without scanning 15 raw genes.",
            "They are especially useful together with the gene plots: first locate where divergence exists, then translate it into stability reserve, drag build-up, and control-authority consequences.",
        ],
        caveats=[
            "These descriptors are derived quantities, so unlike raw genes they do not have one simple fixed design-space bound directly encoded in Chromosome_Drone.",
            "For readability, x-limits here follow the pooled observed support in the selected subset rather than a closed-form global descriptor bound.",
        ],
    )

def plot_comparative_dihedral_stability_margin(
    run_dfs: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    subset_mode: str = "final_generation_pareto",
) -> None:
    subset_title = _comparison_subset_title(subset_mode)
    descriptor_specs = [
        ("dihedral_deg", "Dihedral Angle [deg]", (-4.0, 4.0)),
        ("static_margin_pct_mac", "Static Margin [% MAC]", (-20.0, 40.0)),
    ]
    subset_dfs = _comparison_subset_by_run(run_dfs, subset_mode)
    desc_by_run: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, float | int | str]] = []

    for run_name, subset_df in subset_dfs.items():
        desc_df = _aero_descriptor_dataframe(subset_df)
        desc_by_run[run_name] = desc_df
        for col, label, limits in descriptor_specs:
            values = pd.to_numeric(desc_df[col], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            summary_rows.append(
                {
                    "run_name": run_name,
                    "subset_mode": subset_mode,
                    "descriptor": col,
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
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharey=True)
    label_order = ["nsga_GP", "nsga_SP"]
    y_pos = [1, 0]
    rng = np.random.default_rng(19)
    export_rows: list[dict[str, float | int | str]] = []

    for ax, (col, label, limits) in zip(axes, descriptor_specs):
        data = []
        for run_name in label_order:
            vals = pd.to_numeric(desc_by_run[run_name][col], errors="coerce").to_numpy(dtype=float)
            data.append(vals[np.isfinite(vals)])

        bp = ax.boxplot(
            data,
            vert=False,
            positions=y_pos,
            widths=0.5,
            patch_artist=True,
            whis=(10, 90),
            showfliers=False,
            medianprops=dict(color="#111827", linewidth=2.0),
            whiskerprops=dict(color="#475569", linewidth=1.2),
            capprops=dict(color="#475569", linewidth=1.2),
        )
        for patch, run_name in zip(bp["boxes"], label_order):
            patch.set_facecolor(EVOLUTION_COLORS[run_name])
            patch.set_alpha(0.35)
            patch.set_edgecolor(EVOLUTION_COLORS[run_name])
            patch.set_linewidth(1.3)

        for row_pos, run_name, vals in zip(y_pos, label_order, data):
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=vals.size)
            ax.scatter(
                vals,
                row_pos + jitter,
                s=22,
                alpha=0.35,
                color=EVOLUTION_COLORS[run_name],
                edgecolors="none",
                zorder=3,
            )

        gp_vals, sp_vals = data
        stats = _two_sample_distribution_stats(gp_vals, sp_vals)
        median_delta = float(stats["median_delta_sp_minus_gp"])
        effect = float(stats["standardized_delta"])
        p_value = float(stats["p_value"])
        export_rows.append(
            {
                "descriptor": col,
                "x_min": float(limits[0]),
                "x_max": float(limits[1]),
                **stats,
            }
        )

        ax.set_title(label, fontsize=13.0, weight="bold")
        ax.set_xlim(float(limits[0]), float(limits[1]))
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_comparison_run_label(name) for name in label_order], fontsize=9.2)
        ax.grid(alpha=0.22, axis="x")
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

    fig.suptitle(f"Dihedral and stability margin by controller regime\n{subset_title}", fontsize=17, weight="bold", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(output_path.with_suffix('.csv'), index=False)
    effect_df = pd.DataFrame(export_rows)
    if not effect_df.empty and "p_value" in effect_df.columns:
        effect_df["p_value_fdr_bh"] = _benjamini_hochberg(pd.to_numeric(effect_df["p_value"], errors="coerce").to_numpy(dtype=float))
    effect_df.to_csv(output_path.with_name(output_path.stem + '_effect_sizes.csv'), index=False)
    _write_plot_note(
        output_path,
        title="Dihedral and stability margin by controller regime | Final generation, Pareto individuals",
        importance=(
            "This is a compact version of the comparative descriptor plot, focused only on two stability-relevant quantities: dihedral angle and the full stick-fixed static margin."
        ),
        prior_work=[
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots is directly relevant because controller regime can shift preferred stability-related body parameters ({PAPER_LINKS['controller_learning']}).",
            f"Interpretable morphology comparisons are also aligned with the descriptor-based reading used in Unconventional Hexacopters via Evolution and Learning ({PAPER_LINKS['hexa']}).",
        ],
        what_to_read=[
            "The left panel shows the final Pareto distribution of dihedral angle.",
            "The right panel shows the stick-fixed static margin estimated from total CG, wing AC, tail-volume coefficient, downwash, and fuselage destabilizing contribution.",
            "Both x-axes are fixed to comparison windows rather than autoscaled, so controller-regime shifts remain visually comparable across reruns.",
        ],
        takeaways=[
            "This plot is useful when you want a clean controller-regime comparison with almost no visual clutter.",
            "It isolates whether the controller regime is changing lateral-stability preference, longitudinal-balance preference, or both.",
        ],
        caveats=[
            "The stability quantity here is the full stick-fixed static margin estimate based on total CG, wing AC, tail volume, downwash, and fuselage destabilizing contribution.",
        ],
    )

def plot_comparative_aero_descriptor_panels(
    run_dfs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    descriptor_specs = [
        ("static_margin_pct_mac", "Static margin [% MAC]"),
        ("neutral_point_pct_mac", "Neutral point [% MAC]"),
        ("horizontal_tail_volume_coeff", "Horizontal tail volume coefficient"),
        ("vertical_tail_volume_coeff", "Vertical tail volume coefficient"),
        ("wing_loading_n_m2", "Wing loading [N/m^2]"),
        ("wing_clcd_max_2d", "Wing airfoil max L/D [-]"),
        ("aircraft_ld_max_est", "Aircraft max L/D estimate [-]"),
        ("downwash_gradient", "Downwash gradient [-]"),
        ("aircraft_cd0_est", "Aircraft CD0 estimate [-]"),
        ("fuselage_cm_alpha_est", "Fuselage Cmalpha estimate [-]"),
    ]
    desc_by_run: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, float | int | str]] = []
    for run_name, df in run_dfs.items():
        final_df = _comparison_subset(df, "final_generation_pareto")
        desc_df = _aero_descriptor_dataframe(final_df)
        desc_by_run[run_name] = desc_df
        for col, label in descriptor_specs:
            vals = pd.to_numeric(desc_df[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            summary_rows.append(
                {
                    "run_name": run_name,
                    "descriptor": col,
                    "label": label,
                    "n_samples": int(vals.size),
                    "median": float(np.nanmedian(vals)) if vals.size else np.nan,
                    "q1": float(np.nanpercentile(vals, 25)) if vals.size else np.nan,
                    "q3": float(np.nanpercentile(vals, 75)) if vals.size else np.nan,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    _apply_plot_style()
    fig, axes = plt.subplots(5, 2, figsize=(14.4, 18.0))
    axes_flat = axes.flatten()
    label_order = ["nsga_GP", "nsga_SP"]
    y_map = {"nsga_GP": 1.0, "nsga_SP": 0.0}
    export_rows: list[dict[str, float | str]] = []
    comparison_rows: list[dict[str, float | int | str]] = []

    for ax, (col, label) in zip(axes_flat, descriptor_specs):
        vals_by_run = {}
        for run_name in label_order:
            vals = pd.to_numeric(desc_by_run[run_name][col], errors="coerce").to_numpy(dtype=float)
            vals_by_run[run_name] = vals[np.isfinite(vals)]

        combined = np.concatenate([vals for vals in vals_by_run.values() if vals.size], axis=0)
        if combined.size == 0:
            ax.axis("off")
            continue
        ax.set_title(label, fontsize=12.8, weight="bold")
        for run_name in label_order:
            vals = vals_by_run[run_name]
            if vals.size == 0:
                continue
            median = float(np.nanmedian(vals))
            q1 = float(np.nanpercentile(vals, 25))
            q3 = float(np.nanpercentile(vals, 75))
            y = y_map[run_name]
            ax.hlines(y, q1, q3, color=EVOLUTION_COLORS[run_name], linewidth=8.0, alpha=0.35)
            ax.hlines(y, float(np.nanmin(vals)), float(np.nanmax(vals)), color=EVOLUTION_COLORS[run_name], linewidth=1.6, alpha=0.65)
            ax.scatter(median, y, s=90, color=EVOLUTION_COLORS[run_name], edgecolors="black", linewidths=0.5, zorder=4)
            export_rows.append(
                {
                    "descriptor": col,
                    "run_name": run_name,
                    "median": median,
                    "q1": q1,
                    "q3": q3,
                    "min": float(np.nanmin(vals)),
                    "max": float(np.nanmax(vals)),
                }
            )

        gp_vals = vals_by_run["nsga_GP"]
        sp_vals = vals_by_run["nsga_SP"]
        stats = _two_sample_distribution_stats(gp_vals, sp_vals)
        med_delta = float(stats["median_delta_sp_minus_gp"])
        effect = float(stats["standardized_delta"])
        p_value = float(stats["p_value"])
        comparison_rows.append({"descriptor": col, **stats})
        ax.text(
            0.98,
            0.10,
            (
                f"Difference: {med_delta:+.3g}\nstd delta: {effect:+.2f}\np: {_format_p_value(p_value)}"
                if np.isfinite(effect) else f"Difference: {med_delta:+.3g}\np: {_format_p_value(p_value)}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.6,
            bbox=dict(facecolor="white", alpha=0.84, edgecolor="none", boxstyle="round,pad=0.28"),
        )
        ax.set_yticks([1, 0])
        ax.set_yticklabels([_comparison_run_label("nsga_GP"), _comparison_run_label("nsga_SP")], fontsize=9.2)
        ax.grid(alpha=0.22, axis="x")
        ax.set_xlim(_scaled_ylim(combined))

    fig.suptitle("Final-front aerodynamic descriptors by controller regime", fontsize=17, weight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    summary_df.to_csv(output_path.with_suffix(".csv"), index=False)
    ranges_df = pd.DataFrame(export_rows)
    ranges_df.to_csv(output_path.with_name(output_path.stem + "_ranges.csv"), index=False)
    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty and "p_value" in comparison_df.columns:
        comparison_df["p_value_fdr_bh"] = _benjamini_hochberg(pd.to_numeric(comparison_df["p_value"], errors="coerce").to_numpy(dtype=float))
    comparison_df.to_csv(output_path.with_name(output_path.stem + "_effect_sizes.csv"), index=False)
    _write_plot_note(
        output_path,
        title="Final-front aerodynamic descriptor comparison by controller regime",
        importance=(
            "This figure converts genotype differences into aerodynamic and geometric consequences, so you can read whether the two controller regimes diverge in layout, control authority, or airfoil operating regime."
        ),
        prior_work=[
            f"Unconventional Hexacopters via Evolution and Learning argues for comparing evolved families through interpretable morphology descriptors ({PAPER_LINKS['hexa']}).",
            f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots motivates comparing morphology under different controller assumptions ({PAPER_LINKS['controller_learning']}).",
        ],
        what_to_read=[
            "Each panel shows median, interquartile range, and full range for one descriptor.",
            "Thick bars are the middle 50% of final Pareto solutions, thin lines show the full spread, and the dot is the median.",
            "This format keeps the comparison readable even when the two controller regimes partially overlap.",
        ],
        takeaways=[
            "This is the most interpretable comparison if you care about physical meaning rather than raw genes.",
            "It helps distinguish whether divergence is mostly geometric, stability-related, or airfoil-regime related.",
        ],
        caveats=[
            "These descriptors are reconstructed from URDF geometry, total link masses, and solver-consistent airfoil tables; they are not a substitute for a full trim solve or full dynamic-derivative identification.",
        ],
    )
