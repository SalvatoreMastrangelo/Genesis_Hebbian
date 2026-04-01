from .common import *
from .stats import *



class _URDFHistogramPlotterMainPlotsMixin:
    def plot(self, general_policy_idx=None, save_path=None, show=False):
        gp_idx = self._resolve_policy_index(
            general_policy_idx if general_policy_idx is not None else self.general_policy_idx,
            self.general_policy_count,
            "General",
        )
        bix3_index = self._find_bix3_index(self.df)

        metric_order = self._metric_order_hist()
        metric_labels = self._metric_labels_with_norm()
        legend_labels = [
            "Progress / Forest Length",
            "# Drones > Minimal Progress / Total Drones",
            "Speed / Maximum Commanded Speed",
            "Cost of Transport / 1",
            f"Evaluation Reward / {self._reward_norm_label()}",
            f"Steps to {self._steps_target_pct_label()} Reward / Total Training Steps",
        ]

        colors = {
            "thr_count": "#17becf",
            "speed":  "#1f77b4",
            "cot":   "#ff7f0e",
            "prog":   "#2ca02c",
            "reward": "#d62728",
            "steps90": "#9467bd",
        }

        # Transparent GP/SP backgrounds
        policy_bg_colors = {
            "gp": (0.2, 0.4, 1.0, 0.08),   # light blue
            "sp": (1.0, 0.3, 0.3, 0.08),   # light red
        }

        n_drones = len(self.df)
        fig, ax = plt.subplots(figsize=(20, 7))

        # ------------------------------------------------------------
        # Block geometry
        # ------------------------------------------------------------
        group_spacing = 2.5      # distance between drones
        gp_sp_gap = 1.1          # distance between GP and SP
        bar_spacing = 0.22       # distance between metrics
        bar_width = 0.12
        prog_idx = metric_order.index("prog")
        prog_segment_width = bar_width * 1.8

        xticks = []
        xticklabels = []

        # ============================================================
        #   COMPUTE MEANS EXACTLY AS IN THE PLOT
        # ============================================================

        general_bar_values = {m: [] for m in metric_order}
        special_bar_values = {m: [] for m in metric_order}
        per_drone_GP = []
        per_drone_SP = []
        gp_threshold_rates, sp_threshold_rate = self._compute_threshold_rates()
        selected_gp_threshold_rate = gp_threshold_rates[max(0, gp_idx - 1)] if gp_threshold_rates else 0.0

        for _, row in self.df.iterrows():
            metrics = self._process_row(row, general_policy_idx=general_policy_idx)
            gp = metrics["baseline_selected"]
            sp = metrics["trained"]

            drone_gp_vals = {}
            drone_sp_vals = {}

            for m in metric_order:
                if m == "thr_count":
                    general_bar_values[m].append(selected_gp_threshold_rate)
                    special_bar_values[m].append(sp_threshold_rate)
                    drone_gp_vals[m] = selected_gp_threshold_rate
                    drone_sp_vals[m] = sp_threshold_rate
                    continue

                # === GENERAL (baseline) ===
                gp_val = gp[m]
                if gp_val > 0:      # only bars > 0
                    general_bar_values[m].append(gp_val)
                    drone_gp_vals[m] = gp_val
                else:
                    drone_gp_vals[m] = 0

                # === SPECIALIZED POLICY (SP) ===
                vals = sp[m]

                valid_mask = self._is_valid_metric(m, vals)

                valid_vals = vals[valid_mask]

                if len(valid_vals) > 0:
                    sp_mean = valid_vals.mean()
                    special_bar_values[m].append(sp_mean)
                    drone_sp_vals[m] = sp_mean
                else:
                    drone_sp_vals[m] = 0

            per_drone_GP.append(drone_gp_vals)
            per_drone_SP.append(drone_sp_vals)

        # === FINAL BAR MEANS ===
        final_general = {m: (np.mean(general_bar_values[m]) if len(general_bar_values[m]) else 0)
                        for m in metric_order}
        final_special = {m: (np.mean(special_bar_values[m]) if len(special_bar_values[m]) else 0)
                        for m in metric_order}

        print("Final General Policy Means:")
        for m in metric_order:
            print(f"  {m}: {final_general[m]:.4f}")

        print("Final Specialized Policies Means:")
        for m in metric_order:
            print(f"  {m}: {final_special[m]:.4f}")


        # ============================================================
        #   DELTA % GP vs SP (same as plot_mean_delta_vs_sp)
        # ============================================================
        print(
            "\n=== DELTA % GP vs SP (consistent with plot_mean_delta_vs_sp) ==="
        )
        print(
            "Deviation formula per URDF: "
            "delta% = 100 * (GP_selected - mean(SP_valid_repetitions)) / mean(SP_valid_repetitions)"
        )
        print(
            "Reported mean value: 100 * sum(GP_selected - mean(SP_valid_repetitions)) / "
            "sum(mean(SP_valid_repetitions)); std over per-URDF delta% values."
        )
        delta_metric_order = ["speed", "cot", "prog", "reward", "steps90"]
        gp_delta_data = self._compute_gp_delta_vs_trained(metric_order=delta_metric_order)
        gp_delta_vals = gp_delta_data["ratios"]
        gp_delta_num = gp_delta_data["numerators"]
        gp_delta_den = gp_delta_data["denominators"]
        gp_print_idx = max(0, gp_idx - 1)
        for m in delta_metric_order:
            vals = gp_delta_vals[m][gp_print_idx] if gp_print_idx < len(gp_delta_vals[m]) else []
            nums = gp_delta_num[m][gp_print_idx] if gp_print_idx < len(gp_delta_num[m]) else []
            dens = gp_delta_den[m][gp_print_idx] if gp_print_idx < len(gp_delta_den[m]) else []
            if len(vals) == 0:
                mean_pct = 0.0
                std_pct = 0.0
            else:
                mean_val = self._compute_mean_delta_ratio(nums, dens)
                mean_pct = float(mean_val * 100.0) if np.isfinite(mean_val) else 0.0
                std_val = self._compute_weighted_delta_std(vals, dens)
                std_pct = float(std_val * 100.0) if np.isfinite(std_val) else 0.0
            print(
                f"{m.upper():>8}: delta = {mean_pct:+.2f}% "
                f"(std = {std_pct:.2f}%, n = {len(vals)})"
            )
        print("===============================================================\n")
        self._print_delta4_vs_steps90_correlation(general_policy_idx=gp_idx)
        self._print_minimal_threshold_counts()
        self._print_above_threshold_metric_summaries(general_policy_idx=gp_idx)



        # ------------------------------------------------------------
        # DRONE LOOP (each row = one drone)
        # ------------------------------------------------------------
        for idx, (_, row) in enumerate(self.df.iterrows(), start=1):

            metrics = self._process_row(row, general_policy_idx=general_policy_idx)

            # drone center
            base_x = idx * group_spacing

            # centers of the two groups
            gp_center = base_x - gp_sp_gap / 2
            sp_center = base_x + gp_sp_gap / 2

            # main xtick
            xticks.append(base_x)
            label = f"DRONE {idx}"
            if bix3_index is not None and (idx - 1) == bix3_index:
                label = "BIX3"

            xticklabels.append(label)


            # --------------------------------------------------------
            # Semi-transparent GP and SP backgrounds
            # --------------------------------------------------------
            gp_left, gp_right = self._policy_background_span(gp_center, len(metric_order), bar_spacing)
            ax.axvspan(gp_left, gp_right,
                       facecolor=policy_bg_colors["gp"],
                       edgecolor=None)

            sp_left, sp_right = self._policy_background_span(sp_center, len(metric_order), bar_spacing)
            ax.axvspan(sp_left, sp_right,
                       facecolor=policy_bg_colors["sp"],
                       edgecolor=None)

            # --------------------------------------------------------
            #   General Policy (baseline)
            # --------------------------------------------------------
            for mi, metric in enumerate(metric_order):
                x = gp_center + self._metric_offset(mi, len(metric_order), bar_spacing)
                if metric == "thr_count":
                    val = selected_gp_threshold_rate
                else:
                    val = metrics["baseline_selected"][metric]

                if val == 0 and metric != "thr_count":
                    ax.text(x, 0.01, "X",
                            ha="center", va="bottom", fontsize=9)
                else:
                    ax.bar(x, val, width=bar_width,
                           color=colors[metric], alpha=0.9)

            # --------------------------------------------------------
            #   Specialized Policy (SP)
            # --------------------------------------------------------
            for mi, metric in enumerate(metric_order):
                x = sp_center + self._metric_offset(mi, len(metric_order), bar_spacing)
                if metric == "thr_count":
                    mean_val = sp_threshold_rate
                    valid_mask = np.array([], dtype=bool)
                else:
                    vals = metrics["trained"][metric]

                    # ------------------------------
                    # FILTER INVALID VALUES
                    # ------------------------------
                    valid_mask = self._is_valid_metric(metric, vals)

                    valid_vals = vals[valid_mask]

                    # ------------------------------
                    # MEAN ONLY OVER VALID VALUES
                    # ------------------------------
                    if len(valid_vals) == 0:
                        mean_val = 0
                    else:
                        mean_val = valid_vals.mean()

                # ------------------------------
                # DRAW MEAN BAR
                # ------------------------------
                if mean_val == 0 and metric != "thr_count":
                    ax.text(x, 0.01, "X",
                            ha="center", va="bottom", fontsize=9)
                else:
                    ax.bar(x, mean_val, width=bar_width,
                        color=colors[metric], alpha=0.9)

                # ------------------------------
                # ALWAYS DRAW ALL POINTS
                # ------------------------------
                if metric != "thr_count" and np.any(valid_mask):
                    xs = np.full(np.sum(valid_mask), x)
                    ax.scatter(
                        xs,
                        vals[valid_mask],
                        color=colors[metric],
                        edgecolor="black",
                        alpha=0.5,
                    )

            prog_x_gp = gp_center + self._metric_offset(prog_idx, len(metric_order), bar_spacing)
            prog_x_sp = sp_center + self._metric_offset(prog_idx, len(metric_order), bar_spacing)
            self._draw_progress_threshold_segment(ax, prog_x_gp, prog_segment_width)
            self._draw_progress_threshold_segment(ax, prog_x_sp, prog_segment_width)


            # --------------------------------------------------------
            #   "General Policy" and "Specialized Policy" labels
            # --------------------------------------------------------
            ax.text(gp_center, -0.08, "Platform Independent",
                    ha="center", va="top", fontsize=10,
                    transform=ax.get_xaxis_transform())

            ax.text(sp_center, -0.08, "Platform Dependent",
                    ha="center", va="top", fontsize=10,
                    transform=ax.get_xaxis_transform())

        # ============================================================
        # FINAL MEAN COLUMN
        # ============================================================

        mean_base_x = (n_drones + 1) * group_spacing

        gp_center = mean_base_x - gp_sp_gap / 2
        sp_center = mean_base_x + gp_sp_gap / 2

        xticks.append(mean_base_x)
        xticklabels.append("MEAN")

        # GP/SP background
        gp_left, gp_right = self._policy_background_span(gp_center, len(metric_order), bar_spacing)
        ax.axvspan(gp_left, gp_right, facecolor=policy_bg_colors["gp"], edgecolor=None)

        sp_left, sp_right = self._policy_background_span(sp_center, len(metric_order), bar_spacing)
        ax.axvspan(sp_left, sp_right, facecolor=policy_bg_colors["sp"], edgecolor=None)

        # General Policy mean bars
        for mi, metric in enumerate(metric_order):
            x = gp_center + self._metric_offset(mi, len(metric_order), bar_spacing)
            val = final_general[metric]
            if val == 0 and metric != "thr_count":
                ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
            else:
                ax.bar(x, val, width=bar_width, color=colors[metric], alpha=0.9)

        # Specialized Policy mean bars
        for mi, metric in enumerate(metric_order):
            x = sp_center + self._metric_offset(mi, len(metric_order), bar_spacing)
            val = final_special[metric]
            if val == 0 and metric != "thr_count":
                ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
            else:
                ax.bar(x, val, width=bar_width, color=colors[metric], alpha=0.9)

        prog_x_gp = gp_center + self._metric_offset(prog_idx, len(metric_order), bar_spacing)
        prog_x_sp = sp_center + self._metric_offset(prog_idx, len(metric_order), bar_spacing)
        self._draw_progress_threshold_segment(ax, prog_x_gp, prog_segment_width)
        self._draw_progress_threshold_segment(ax, prog_x_sp, prog_segment_width)


        # ------------------------------------------------------------
        #   X AXIS, TITLE, LEGEND
        # ------------------------------------------------------------
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=self._metric_xtick_fontsize())
        ax.set_xlim(group_spacing - 2, group_spacing * (n_drones + 1) + 2)


        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{self._platform_independent_policy_label()} vs "
            f"{self._platform_dependent_policy_label()} — Normalized Metrics per Drone "
            f"(URDF 1–{n_drones})",
            fontsize=15)

        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, color=colors[m])
            for m in metric_order
        ]
        legend_patches.append(self._progress_threshold_legend_handle())
        legend_labels.append("Minimal Progress Threshold")
        ax.legend(legend_patches, legend_labels,
                  title="Metrics (normalized)")
        
        self._apply_dense_y_grid(ax)

        plt.tight_layout()

        # ------------------------------------------------------------
        #   SAVE PNG
        # ------------------------------------------------------------
        output_path = save_path if save_path is not None else self.save_path
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        print(f"\nPlot saved to: {os.path.abspath(output_path)}")

    def plot_bix3_vs_mean(self, save_path="drone_plot_bix3.png", general_policy_idx=None, show=False):
        gp_idx = self._resolve_policy_index(
            general_policy_idx if general_policy_idx is not None else self.general_policy_idx,
            self.general_policy_count,
            "General",
        )

        metric_order = self._metric_order_hist()

        colors = {
            "thr_count": "#17becf",
            "speed":  "#1f77b4",
            "cot":   "#ff7f0e",
            "prog":   "#2ca02c",
            "reward": "#d62728",
            "steps90": "#9467bd",
        }
        policy_bg_colors = {
            "gp": (0.2, 0.4, 1.0, 0.08),
            "sp": (1.0, 0.3, 0.3, 0.08),
        }

        # ------------------------------------------------------------
        #   SAME COMPUTATION AS MAIN PLOT
        # ------------------------------------------------------------
        general_bar_values = {m: [] for m in metric_order}
        special_bar_values = {m: [] for m in metric_order}
        per_drone_GP = []
        per_drone_SP = []
        gp_threshold_rates, sp_threshold_rate = self._compute_threshold_rates()
        selected_gp_threshold_rate = gp_threshold_rates[max(0, gp_idx - 1)] if gp_threshold_rates else 0.0

        for _, row in self.df.iterrows():
            metrics = self._process_row(row, general_policy_idx=general_policy_idx)
            gp = metrics["baseline_selected"]
            sp = metrics["trained"]

            drone_gp = {}
            drone_sp = {}

            for m in metric_order:
                if m == "thr_count":
                    general_bar_values[m].append(selected_gp_threshold_rate)
                    special_bar_values[m].append(sp_threshold_rate)
                    drone_gp[m] = selected_gp_threshold_rate
                    drone_sp[m] = sp_threshold_rate
                    continue

                # GP
                gp_val = gp[m]
                if gp_val > 0:
                    general_bar_values[m].append(gp_val)
                    drone_gp[m] = gp_val
                else:
                    drone_gp[m] = 0

                # SP
                vals = sp[m]
                valid_mask = self._is_valid_metric(m, vals)

                valid_vals = vals[valid_mask]
                if len(valid_vals) > 0:
                    sp_mean = valid_vals.mean()
                    special_bar_values[m].append(sp_mean)
                    drone_sp[m] = sp_mean
                else:
                    drone_sp[m] = 0

            per_drone_GP.append(drone_gp)
            per_drone_SP.append(drone_sp)

        # FINAL MEANS
        final_general = {
            m: (np.mean(general_bar_values[m]) if len(general_bar_values[m]) else 0)
            for m in metric_order
        }
        final_special = {
            m: (np.mean(special_bar_values[m]) if len(special_bar_values[m]) else 0)
            for m in metric_order
        }

        bix_index = self._find_bix3_index(self.df)
        if bix_index is None:
            print("[WARN] BIX3 URDF stem not found: using first available drone.")
            bix_index = 0
        if bix_index >= len(per_drone_GP):
            bix_index = max(0, len(per_drone_GP) - 1)
        bix_gp = per_drone_GP[bix_index] if per_drone_GP else {m: 0 for m in metric_order}
        bix_sp = per_drone_SP[bix_index] if per_drone_SP else {m: 0 for m in metric_order}

        # ------------------------------------------------------------
        #   PLOT
        # ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=self._metric_plot_figsize())

        group_spacing = 3.0
        gp_sp_gap = 1.1
        bar_spacing = 0.22
        bar_width = 0.12
        prog_idx = metric_order.index("prog")
        prog_segment_width = bar_width * 1.8

        xticks = []
        xticklabels = []

        groups = [
            ("BIX3", bix_gp, bix_sp),
            ("MEAN", final_general, final_special)
        ]

        # Correct loop
        for idx, (label, gp_vals, sp_vals) in enumerate(groups, start=1):

            base_x = idx * group_spacing
            gp_center = base_x - gp_sp_gap / 2
            sp_center = base_x + gp_sp_gap / 2
            # Label below GP/SP
            ax.text(gp_center, -0.08, "Platform Independent", ha="center", va="top",
                    fontsize=10, transform=ax.get_xaxis_transform())

            ax.text(sp_center, -0.08, "Platform Dependent", ha="center", va="top",
                    fontsize=10, transform=ax.get_xaxis_transform())

            xticks.append(base_x)
            xticklabels.append(label)

            # GP-SP backgrounds
            gp_left, gp_right = self._policy_background_span(gp_center, len(metric_order), bar_spacing)
            sp_left, sp_right = self._policy_background_span(sp_center, len(metric_order), bar_spacing)
            ax.axvspan(gp_left, gp_right, facecolor=policy_bg_colors["gp"], edgecolor=None)
            ax.axvspan(sp_left, sp_right, facecolor=policy_bg_colors["sp"], edgecolor=None)

            # ----- GP BARS + POINTS -----
            for mi, m in enumerate(metric_order):
                x = gp_center + self._metric_offset(mi, len(metric_order), bar_spacing)
                val = gp_vals[m]

                if val > 0 or m == "thr_count":
                    ax.bar(x, val, width=bar_width, color=colors[m])
                else:
                    ax.text(x, 0.01, "X", ha="center")

                # Raw points ONLY for BIX3
                if label == "BIX3":
                    if m == "thr_count":
                        continue
                    raw_val = self._process_row(
                        self.df.iloc[bix_index],
                        general_policy_idx=general_policy_idx
                    )["baseline_selected"][m]
                    if raw_val > 0:
                        ax.scatter(x, raw_val, color=colors[m], edgecolor="black", alpha=0.8)

            # ----- SP BARS + POINTS -----
            for mi, m in enumerate(metric_order):
                x = sp_center + self._metric_offset(mi, len(metric_order), bar_spacing)
                val = sp_vals[m]

                if val > 0 or m == "thr_count":
                    ax.bar(x, val, width=bar_width, color=colors[m])
                else:
                    ax.text(x, 0.01, "X", ha="center")

                # Raw points ONLY for BIX3
                if label == "BIX3":
                    if m == "thr_count":
                        continue
                    raw_vals = self._process_row(
                        self.df.iloc[bix_index],
                        general_policy_idx=general_policy_idx
                    )["trained"][m]

                    mask = self._is_valid_metric(m, raw_vals)

                    xs = np.full(len(raw_vals), x)
                    ax.scatter(xs[mask], raw_vals[mask],
                            color=colors[m], edgecolor="black", alpha=0.8)

            prog_x_gp = gp_center + self._metric_offset(prog_idx, len(metric_order), bar_spacing)
            prog_x_sp = sp_center + self._metric_offset(prog_idx, len(metric_order), bar_spacing)
            self._draw_progress_threshold_segment(ax, prog_x_gp, prog_segment_width)
            self._draw_progress_threshold_segment(ax, prog_x_sp, prog_segment_width)

        # final touches
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=self._metric_xtick_fontsize())
        ax.set_ylabel("Normalized Value")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"BIX3 (URDF {bix_index + 1}) vs Mean Performance — Normalized Metrics "
            f"(GP #{gp_idx})"
        )
        legend_labels = [
            "Progress / Forest Length",
            "# Drones > Minimal Progress / Total Drones",
            "Speed / Maximum Commanded Speed",
            "Cost of Transport / 1",
            f"Evaluation Reward / {self._reward_norm_label()}",
            f"Steps to {self._steps_target_pct_label()} Reward / Total Training Steps",
        ]

        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, color=colors[m])
            for m in metric_order
        ]
        legend_patches.append(self._progress_threshold_legend_handle())
        legend_labels.append("Minimal Progress Threshold")

        ax.legend(
            legend_patches,
            legend_labels,
            title="Metrics (normalized)",
            fontsize=10
        )
        self._apply_dense_y_grid(ax)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        print("\nBIX3 vs Mean plot saved to:", os.path.abspath(save_path))

    def plot_per_urdf_policies(self, save_dir=None, show=False, urdf_indices=None):
        metric_order = ["prog", "speed", "cot", "reward", "steps90"]
        metric_labels = self._metric_labels_with_norm()
        selected_urdfs = self._normalize_urdf_indices(urdf_indices)

        gp_count = max(1, self.general_policy_count)
        cmap = plt.get_cmap("tab10")
        gp_colors = [cmap(i) for i in range(gp_count)]
        trained_color = "#444444"

        bar_spacing = 0.12
        offsets = (np.arange(gp_count + 1) - gp_count / 2) * bar_spacing

        for row_idx, (_, row) in enumerate(self.df.iterrows(), start=1):
            if selected_urdfs is not None and row_idx not in selected_urdfs:
                continue
            metrics = self._process_row(row)
            gp_all = metrics["baseline_all"]
            gp_all_std = metrics["baseline_all_std"]
            tr_all = metrics["trained"]

            fig, ax = plt.subplots(figsize=self._metric_plot_figsize())
            group_centers = np.arange(len(metric_order))
            prog_idx = metric_order.index("prog")
            prog_center = group_centers[prog_idx]
            prog_span = max(
                bar_spacing * 0.9,
                (offsets[-1] - offsets[0]) + bar_spacing * 0.9,
            )

            for mi, metric in enumerate(metric_order):
                center = group_centers[mi]
                gp_vals = gp_all[metric]
                gp_std_vals = gp_all_std[metric]

                for pi in range(gp_count):
                    x = center + offsets[pi]
                    val = gp_vals[pi] if pi < len(gp_vals) else 0
                    std_val = gp_std_vals[pi] if pi < len(gp_std_vals) else np.nan
                    if val <= 0 or not np.isfinite(val):
                        ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
                    else:
                        yerr = None
                        if np.isfinite(std_val) and std_val > 0:
                            yerr = std_val
                        ax.bar(
                            x,
                            val,
                            width=bar_spacing * 0.9,
                            color=gp_colors[pi],
                            yerr=yerr,
                            capsize=4 if yerr is not None else 0,
                        )

                tr_vals = tr_all[metric]
                valid_mask = self._is_valid_metric(metric, tr_vals)
                valid_vals = tr_vals[valid_mask]
                if len(valid_vals) == 0:
                    mean_val = 0
                    std_val = 0
                else:
                    mean_val = valid_vals.mean()
                    std_val = valid_vals.std()

                x = center + offsets[-1]
                if mean_val <= 0 or not np.isfinite(mean_val):
                    ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
                else:
                    ax.bar(
                        x,
                        mean_val,
                        width=bar_spacing * 0.9,
                        color=trained_color,
                        yerr=std_val,
                        capsize=4,
                        alpha=0.9,
                    )

            self._draw_progress_threshold_segment(ax, prog_center, prog_span)

            ax.set_xticks(group_centers)
            ax.set_xticklabels(
                [metric_labels[m] for m in metric_order],
                fontsize=self._metric_xtick_fontsize(),
            )
            ax.set_ylabel("Normalized Value", fontsize=11)
            ax.set_ylim(0, 1)

            urdf_label = row["urdf_stem"] if "urdf_stem" in row.index else f"DRONE {row_idx}"
            urdf_label = str(urdf_label)
            if len(urdf_label) > 50:
                urdf_label = urdf_label[:47] + "..."
            ax.set_title(
                f"URDF {row_idx} — {self._platform_independent_policy_label(plural=True)} vs "
                f"{self._platform_dependent_policy_label()} (mean±std) — {urdf_label}",
                fontsize=12,
            )

            legend_labels = [
                f"{self._platform_independent_policy_label()} {i}"
                for i in range(1, gp_count + 1)
            ] + [f"{self._platform_dependent_policy_label()} mean±std"]
            legend_patches = [
                plt.Rectangle((0, 0), 1, 1, color=gp_colors[i])
                for i in range(gp_count)
            ]
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=trained_color))
            legend_patches.append(self._progress_threshold_legend_handle())
            legend_labels.append("Minimal Progress Threshold")
            ax.legend(legend_patches, legend_labels, title="Policies", fontsize=9)

            self._apply_dense_y_grid(ax)
            plt.tight_layout()

            if save_dir is not None:
                out_dir = self._urdf_output_dir(save_dir, row_idx=row_idx)
                filename = 'policy_metrics.png'
                fig.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")
                self._render_drone_picture(row_idx=row_idx, row=row, save_dir=save_dir)

            if show:
                plt.show()
            plt.close(fig)

    def plot_mean_policies(
        self,
        save_path=None,
        show=False,
        only_above_minimal_progress=False,
        progress_threshold_m=None,
    ):
        metric_order = self._metric_order_hist()
        metric_labels = self._metric_labels_with_norm()
        threshold_m = (
            self.MINIMAL_PROGRESS_M
            if progress_threshold_m is None
            else float(progress_threshold_m)
        )
        threshold_norm = threshold_m / self.PROGRESS_NORM

        gp_count = max(1, self.general_policy_count)
        cmap = plt.get_cmap("tab10")
        gp_colors = [cmap(i) for i in range(gp_count)]
        trained_color = "#444444"
        gp_threshold_rates, gp_threshold_stds, sp_threshold_rate, sp_threshold_std = (
            self._compute_threshold_rate_stats()
        )

        bar_spacing = 0.12
        offsets = (np.arange(gp_count + 1) - gp_count / 2) * bar_spacing

        # Accumulate valid values for global mean
        gp_acc = {m: [[] for _ in range(gp_count)] for m in metric_order}
        tr_acc = {m: [] for m in metric_order}

        for _, row in self.df.iterrows():
            metrics = self._process_row(row)
            gp_all = metrics["baseline_all"]
            tr_all = metrics["trained"]
            gp_prog_vals = np.asarray(gp_all["prog"], dtype=float)
            tr_prog_vals = np.asarray(tr_all["prog"], dtype=float)
            tr_prog_valid = tr_prog_vals[np.isfinite(tr_prog_vals) & (tr_prog_vals > 0)]
            tr_prog_mean = float(np.mean(tr_prog_valid)) if len(tr_prog_valid) > 0 else np.nan

            for m in metric_order:
                if m == "thr_count":
                    continue
                gp_vals = gp_all[m]
                for pi in range(gp_count):
                    if pi < len(gp_vals):
                        val = gp_vals[pi]
                        gp_prog_ok = (
                            pi < len(gp_prog_vals)
                            and np.isfinite(gp_prog_vals[pi])
                            and gp_prog_vals[pi] > threshold_norm
                        )
                        if np.isfinite(val) and val > 0 and (
                            (not only_above_minimal_progress) or gp_prog_ok
                        ):
                            gp_acc[m][pi].append(val)

                tr_vals = tr_all[m]
                tr_prog_ok = (
                    (not only_above_minimal_progress)
                    or (np.isfinite(tr_prog_mean) and tr_prog_mean > threshold_norm)
                )
                valid_mask = self._is_valid_metric(m, tr_vals)
                valid_vals = tr_vals[valid_mask]
                if len(valid_vals) > 0 and tr_prog_ok:
                    if only_above_minimal_progress:
                        # In the filtered plot use one contribution per URDF,
                        # otherwise min/max can reflect a failing repetition
                        # from an otherwise above-threshold drone.
                        tr_acc[m].append(float(np.mean(valid_vals)))
                    else:
                        tr_acc[m].extend(valid_vals.tolist())

        # Compute means + total std (across URDFs / valid repetitions)
        gp_means = {
            m: [np.mean(gp_acc[m][pi]) if gp_acc[m][pi] else 0 for pi in range(gp_count)]
            for m in metric_order
        }
        gp_stds = {
            m: [np.std(gp_acc[m][pi]) if len(gp_acc[m][pi]) > 1 else 0 for pi in range(gp_count)]
            for m in metric_order
        }
        gp_mins = {
            m: [np.min(gp_acc[m][pi]) if gp_acc[m][pi] else 0 for pi in range(gp_count)]
            for m in metric_order
        }
        gp_maxs = {
            m: [np.max(gp_acc[m][pi]) if gp_acc[m][pi] else 0 for pi in range(gp_count)]
            for m in metric_order
        }
        tr_means = {m: (np.mean(tr_acc[m]) if len(tr_acc[m]) else 0) for m in metric_order}
        tr_stds = {m: (np.std(tr_acc[m]) if len(tr_acc[m]) > 1 else 0) for m in metric_order}
        tr_mins = {m: (np.min(tr_acc[m]) if len(tr_acc[m]) else 0) for m in metric_order}
        tr_maxs = {m: (np.max(tr_acc[m]) if len(tr_acc[m]) else 0) for m in metric_order}
        gp_means["thr_count"] = [gp_threshold_rates[pi] if pi < len(gp_threshold_rates) else 0 for pi in range(gp_count)]
        gp_stds["thr_count"] = [gp_threshold_stds[pi] if pi < len(gp_threshold_stds) else 0 for pi in range(gp_count)]
        gp_mins["thr_count"] = [gp_threshold_rates[pi] if pi < len(gp_threshold_rates) else 0 for pi in range(gp_count)]
        gp_maxs["thr_count"] = [gp_threshold_rates[pi] if pi < len(gp_threshold_rates) else 0 for pi in range(gp_count)]
        tr_means["thr_count"] = sp_threshold_rate
        tr_stds["thr_count"] = sp_threshold_std
        tr_mins["thr_count"] = sp_threshold_rate
        tr_maxs["thr_count"] = sp_threshold_rate

        # Plot
        fig, ax = plt.subplots(figsize=self._metric_plot_figsize())
        group_centers = np.arange(len(metric_order))
        prog_idx = metric_order.index("prog")
        prog_center = group_centers[prog_idx]
        prog_span = max(
            bar_spacing * 0.9,
            (offsets[-1] - offsets[0]) + bar_spacing * 0.9,
        )

        for mi, metric in enumerate(metric_order):
            center = group_centers[mi]

            # GP bars
            for pi in range(gp_count):
                x = center + offsets[pi]
                val = gp_means[metric][pi]
                std_val = gp_stds[metric][pi]
                min_val = gp_mins[metric][pi]
                max_val = gp_maxs[metric][pi]
                if (val <= 0 or not np.isfinite(val)) and metric != "thr_count":
                    ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
                else:
                    ax.bar(
                        x,
                        val,
                        width=bar_spacing * 0.9,
                        color=gp_colors[pi],
                    )
                    if np.isfinite(std_val) and std_val > 0:
                        ax.errorbar(
                            x,
                            val,
                            yerr=std_val,
                            fmt="none",
                            ecolor="black",
                            elinewidth=2.2,
                            capsize=5,
                            capthick=2.2,
                            zorder=5,
                        )
                    lower = max(0.0, float(val - min_val))
                    upper = max(0.0, float(max_val - val))
                    if lower > 0 or upper > 0:
                        ax.errorbar(
                            x,
                            val,
                            yerr=np.array([[lower], [upper]], dtype=float),
                            fmt="none",
                            ecolor="black",
                            elinewidth=1.0,
                            capsize=3,
                            capthick=1.0,
                            alpha=0.9,
                            zorder=4.5,
                        )

            # SP mean
            x = center + offsets[-1]
            mean_val = tr_means[metric]
            std_val = tr_stds[metric]
            min_val = tr_mins[metric]
            max_val = tr_maxs[metric]
            if (mean_val <= 0 or not np.isfinite(mean_val)) and metric != "thr_count":
                ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
            else:
                ax.bar(
                    x,
                    mean_val,
                    width=bar_spacing * 0.9,
                    color=trained_color,
                    alpha=0.9,
                )
                if np.isfinite(std_val) and std_val > 0:
                    ax.errorbar(
                        x,
                        mean_val,
                        yerr=std_val,
                        fmt="none",
                        ecolor="black",
                        elinewidth=2.2,
                        capsize=5,
                        capthick=2.2,
                        zorder=5,
                    )
                lower = max(0.0, float(mean_val - min_val))
                upper = max(0.0, float(max_val - mean_val))
                if lower > 0 or upper > 0:
                    ax.errorbar(
                        x,
                        mean_val,
                        yerr=np.array([[lower], [upper]], dtype=float),
                        fmt="none",
                        ecolor="black",
                        elinewidth=1.0,
                        capsize=3,
                        capthick=1.0,
                        alpha=0.9,
                        zorder=4.5,
                    )

        self._draw_progress_threshold_segment(
            ax,
            prog_center,
            prog_span,
            threshold_norm=threshold_norm,
        )

        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            [metric_labels[m] for m in metric_order],
            fontsize=self._metric_xtick_fontsize(),
        )
        ax.set_ylabel("Normalized Value", fontsize=11)
        ax.set_ylim(0, 1)
        title = (
            "Performance Metrics Mean — "
            f"{self._platform_independent_policy_label(plural=True)} vs "
            f"{self._platform_dependent_policy_label()} (mean±std, min-max)"
        )
        if only_above_minimal_progress:
            if progress_threshold_m is None:
                title += " — Above Minimal Progress Only"
            else:
                title += f" — Above {threshold_m:.0f} m Progress Only"
        ax.set_title(title, fontsize=12)

        legend_labels = [
            f"{self._platform_independent_policy_label()} {i} mean±std + min/max"
            for i in range(1, gp_count + 1)
        ] + [f"{self._platform_dependent_policy_label()} mean±std + min/max"]
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, color=gp_colors[i])
            for i in range(gp_count)
        ]
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=trained_color))
        legend_patches.append(self._progress_threshold_legend_handle())
        legend_labels.append("Minimal Progress Threshold")
        ax.legend(legend_patches, legend_labels, title="Policies", fontsize=9)

        self._apply_dense_y_grid(ax)
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)

    def plot_selected_gp_vs_sp_distribution(
        self,
        save_path=None,
        general_policy_idx=None,
        show=False,
        progress_threshold_m=None,
        sp_use_all_repetitions=False,
    ):
        gp_idx = self._resolve_policy_index(
            general_policy_idx if general_policy_idx is not None else self.general_policy_idx,
            self.general_policy_count,
            "General",
        )
        threshold_m = (
            self.MINIMAL_PROGRESS_M
            if progress_threshold_m is None
            else float(progress_threshold_m)
        )
        threshold_norm = threshold_m / self.PROGRESS_NORM
        metric_order = ["prog", "cot", "speed", "reward", "steps90"]
        metric_labels = self._metric_labels_with_norm()

        gp_dist = {m: [] for m in metric_order}
        sp_dist = {m: [] for m in metric_order}

        for _, row in self.df.iterrows():
            metrics = self._process_row(row, general_policy_idx=gp_idx)
            gp_sel = metrics["baseline_selected"]
            sp_all = metrics["trained"]

            for metric in metric_order:
                gp_val = float(gp_sel[metric])
                valid_mask = self._is_valid_metric(metric, np.asarray([gp_val], dtype=float))
                if bool(valid_mask[0]):
                    gp_dist[metric].append(gp_val)

                sp_vals = np.asarray(sp_all[metric], dtype=float)
                valid_mask = self._is_valid_metric(metric, sp_vals)
                valid_vals = sp_vals[valid_mask]
                if len(valid_vals) > 0:
                    if sp_use_all_repetitions:
                        sp_dist[metric].extend(valid_vals.tolist())
                    else:
                        sp_dist[metric].append(float(np.mean(valid_vals)))

        distribution_stats = []
        raw_p_values = []
        for metric in metric_order:
            stats_row = mann_whitney_stats(gp_dist[metric], sp_dist[metric])
            stats_row["metric"] = metric
            distribution_stats.append(stats_row)
            raw_p_values.append(stats_row["p_value"])
        q_values = benjamini_hochberg(raw_p_values)
        for stats_row, q_value in zip(distribution_stats, q_values):
            stats_row["q_value_fdr_bh"] = q_value
        stats_by_metric = {row["metric"]: row for row in distribution_stats}

        fig, ax = plt.subplots(figsize=(15.8, 6.2))
        centers = np.arange(len(metric_order))
        half_offset = 0.14
        violin_width = 0.24
        gp_color = plt.get_cmap("tab10")(max(0, gp_idx - 1))
        sp_color = "#444444"
        rng = np.random.default_rng(0)

        def _draw_violin(values, pos, color):
            if len(values) == 0:
                ax.text(pos, 0.01, "X", ha="center", va="bottom", fontsize=9)
                return
            vp = ax.violinplot(
                [values],
                positions=[pos],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            body = vp["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.42)
            body.set_linewidth(1.0)

            vals = np.asarray(values, dtype=float)
            mean_val = float(np.mean(vals))
            med_val = float(np.median(vals))
            x0, x1 = pos - violin_width * 0.35, pos + violin_width * 0.35
            ax.plot([x0, x1], [mean_val, mean_val], color="black", linewidth=2.2, zorder=5)
            ax.plot([x0, x1], [med_val, med_val], color="black", linewidth=1.0, alpha=0.75, zorder=5)

            jitter = rng.uniform(-violin_width * 0.22, violin_width * 0.22, size=len(vals))
            x_pts = pos + jitter
            ax.scatter(
                x_pts,
                vals,
                s=16,
                color=color,
                alpha=0.72,
                edgecolor="black",
                linewidth=0.35,
                zorder=6,
            )

        for mi, metric in enumerate(metric_order):
            center = centers[mi]
            _draw_violin(gp_dist[metric], center - half_offset, gp_color)
            _draw_violin(sp_dist[metric], center + half_offset, sp_color)
            stats_row = stats_by_metric[metric]
            p_txt = format_p_value(stats_row["p_value"])
            q_txt = format_p_value(stats_row["q_value_fdr_bh"])
            marker = significance_marker(q_value=stats_row["q_value_fdr_bh"], p_value=stats_row["p_value"])
            ann = f"p={p_txt}\nq={q_txt}{marker}"
            ax.text(
                center,
                0.99,
                ann,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
                color="#333333",
            )

        prog_idx = metric_order.index("prog")
        self._draw_progress_threshold_segment(
            ax,
            centers[prog_idx],
            (half_offset * 2) + violin_width * 1.15,
            threshold_norm=threshold_norm,
        )

        ax.set_xticks(centers)
        ax.set_xticklabels(
            [metric_labels[m] for m in metric_order],
            fontsize=self._metric_xtick_fontsize(),
        )
        ax.set_ylabel("Normalized Value", fontsize=11)
        ax.set_ylim(0, 1)
        sp_label_suffix = (
            f"{self._platform_dependent_policy_label(plural=True)}: all seeds"
            if sp_use_all_repetitions
            else self._platform_dependent_policy_label()
        )
        ax.set_title(
            "Performance Distributions per Drone — "
            f"{self._platform_independent_policy_label()} vs {sp_label_suffix}",
            fontsize=14,
        )

        gp_n = len(gp_dist["prog"])
        sp_n = len(sp_dist["prog"])
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=gp_color, alpha=0.42),
            plt.Rectangle((0, 0), 1, 1, color=sp_color, alpha=0.42),
            Line2D([0], [0], color="black", linewidth=2.2),
            Line2D([0], [0], color="black", linewidth=1.0),
            self._progress_threshold_legend_handle(),
        ]
        legend_labels = [
            f"{self._platform_independent_policy_label()} distribution (n={gp_n})",
            f"{sp_label_suffix} distribution (n={sp_n})",
            "Mean",
            "Median",
            f"Progress threshold reference ({threshold_m:.0f} m)",
        ]
        ax.legend(
            legend_handles,
            legend_labels,
            title="Distributions",
            title_fontsize=13,
            fontsize=12,
            loc="upper center",
            bbox_to_anchor=(0.735, 0.98),
            frameon=True,
            facecolor="white",
            edgecolor="#bcbcbc",
            framealpha=0.96,
            borderpad=0.9,
            labelspacing=0.7,
            handlelength=2.2,
        )

        self._apply_dense_y_grid(ax)
        if sp_use_all_repetitions:
            footnote = "Each point is one seed-level evaluation for one drone."
        else:
            footnote = "Each point is one drone averaged over 5 training seeds."
        footnote += " Mann-Whitney U two-sided p-values are shown above each metric; q-values use FDR-BH correction."
        fig.text(
            0.5,
            0.015,
            footnote,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#4d4d4d",
        )
        plt.tight_layout(rect=(0, 0.05, 1, 1))

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

            save_stats_table(save_path, pd.DataFrame(distribution_stats), suffix="_distribution_stats.csv")
        if show:
            plt.show()
        plt.close(fig)

    def plot_mean_sp_policies(self, save_path=None, show=False):
        metric_order = self._metric_order_hist()
        metric_labels = self._metric_labels_with_norm()

        sp_color = "#444444"
        sp_acc = {m: [] for m in metric_order}
        _, _, sp_threshold_rate, sp_threshold_std = self._compute_threshold_rate_stats()

        for _, row in self.df.iterrows():
            metrics = self._process_row(row)
            sp_all = metrics["trained"]

            for m in metric_order:
                if m == "thr_count":
                    continue
                sp_vals = sp_all[m]
                valid_mask = self._is_valid_metric(m, sp_vals)
                valid_vals = sp_vals[valid_mask]
                if len(valid_vals) > 0:
                    sp_acc[m].extend(valid_vals.tolist())

        sp_means = {m: (np.mean(sp_acc[m]) if len(sp_acc[m]) else 0) for m in metric_order}
        sp_stds = {m: (np.std(sp_acc[m]) if len(sp_acc[m]) > 1 else 0) for m in metric_order}
        sp_means["thr_count"] = sp_threshold_rate
        sp_stds["thr_count"] = sp_threshold_std

        fig, ax = plt.subplots(figsize=self._metric_plot_figsize())
        group_centers = np.arange(len(metric_order))
        prog_idx = metric_order.index("prog")
        prog_center = group_centers[prog_idx]

        for mi, metric in enumerate(metric_order):
            center = group_centers[mi]
            mean_val = sp_means[metric]
            std_val = sp_stds[metric]
            if (mean_val <= 0 or not np.isfinite(mean_val)) and metric != "thr_count":
                ax.text(center, 0.01, "X", ha="center", va="bottom", fontsize=9)
            else:
                ax.bar(
                    center,
                    mean_val,
                    width=0.16,
                    color=sp_color,
                    yerr=std_val,
                    capsize=4,
                    alpha=0.9,
                )

        self._draw_progress_threshold_segment(ax, prog_center, 0.20)

        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            [metric_labels[m] for m in metric_order],
            fontsize=self._metric_xtick_fontsize(),
        )
        ax.set_ylabel("Normalized Value", fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(
            "Performance Metrics Mean — "
            f"{self._platform_dependent_policy_label(plural=True)} "
            "(mean±std)",
            fontsize=12,
        )

        legend_labels = [f"{self._platform_dependent_policy_label()} mean±std"]
        legend_patches = [plt.Rectangle((0, 0), 1, 1, color=sp_color)]
        legend_patches.append(self._progress_threshold_legend_handle())
        legend_labels.append("Minimal Progress Threshold")
        ax.legend(legend_patches, legend_labels, title="Policies", fontsize=9)

        self._apply_dense_y_grid(ax)
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)

    def plot_mean_delta_vs_trained(self, save_path=None, show=False):
        metric_order = ["speed", "cot", "prog", "reward", "steps90"]
        metric_labels = self._metric_labels_with_norm()

        gp_count = max(1, self.general_policy_count)
        cmap = plt.get_cmap("tab10")
        gp_colors = [cmap(i) for i in range(gp_count)]
        trained_color = "#444444"

        bar_spacing = 0.12
        offsets = (np.arange(gp_count + 1) - gp_count / 2) * bar_spacing

        gp_delta_data = self._compute_gp_delta_vs_trained(metric_order=metric_order)
        gp_delta_vals = gp_delta_data["ratios"]
        gp_delta_num = gp_delta_data["numerators"]
        gp_delta_den = gp_delta_data["denominators"]
        gp_delta_urdf_idx = gp_delta_data["urdf_indices"]

        gp_delta_mean = {
            m: [
                (
                    self._compute_mean_delta_ratio(gp_delta_num[m][pi], gp_delta_den[m][pi])
                    if len(gp_delta_vals[m][pi])
                    else 0
                )
                for pi in range(gp_count)
            ]
            for m in metric_order
        }
        gp_delta_std = {
            m: [
                (
                    self._compute_weighted_delta_std(
                        gp_delta_vals[m][pi],
                        gp_delta_den[m][pi],
                    )
                    if len(gp_delta_vals[m][pi]) > 1
                    else 0
                )
                for pi in range(gp_count)
            ]
            for m in metric_order
        }
        gp_delta_count = {
            m: [len(gp_delta_vals[m][pi]) for pi in range(gp_count)]
            for m in metric_order
        }
        tr_delta_mean = {m: 0 for m in metric_order}

        # Console summary requested at run time: progress deviation vs SP across all studied drones.
        prog_metric = "prog"
        print("\n=== Mean deviation relative to Specialized policy (Progress) ===")
        for pi in range(gp_count):
            mean_ratio = gp_delta_mean[prog_metric][pi]
            std_ratio = gp_delta_std[prog_metric][pi]
            n_valid = gp_delta_count[prog_metric][pi]
            mean_pct = float(mean_ratio * 100.0) if np.isfinite(mean_ratio) else 0.0
            std_pct = float(std_ratio * 100.0) if np.isfinite(std_ratio) else 0.0
            print(
                f"GP{pi + 1}: delta_progress = {mean_pct:+.2f}% "
                f"(std = {std_pct:.2f}%, n = {n_valid})"
            )
            per_drone_ratios = gp_delta_vals[prog_metric][pi]
            per_drone_indices = gp_delta_urdf_idx[prog_metric][pi]
            if len(per_drone_ratios) == 0:
                print("  No valid drones for progress delta.")
            else:
                for urdf_idx, ratio in zip(per_drone_indices, per_drone_ratios):
                    ratio_pct = float(ratio * 100.0)
                    print(f"  DRONE {urdf_idx}: delta_progress = {ratio_pct:+.2f}%")
        print("==============================================================\n")

        fig, ax = plt.subplots(figsize=self._metric_plot_figsize())
        group_centers = np.arange(len(metric_order))
        zero_line = ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

        for mi, metric in enumerate(metric_order):
            center = group_centers[mi]

            for pi in range(gp_count):
                x = center + offsets[pi]
                val = gp_delta_mean[metric][pi] * 100
                std_val = gp_delta_std[metric][pi] * 100
                if gp_delta_count[metric][pi] == 0 or not np.isfinite(val):
                    ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
                else:
                    ax.bar(
                        x,
                        val,
                        width=bar_spacing * 0.9,
                        color=gp_colors[pi],
                        yerr=std_val,
                        capsize=4,
                        alpha=0.9,
                    )

            x = center + offsets[-1]
            val = tr_delta_mean[metric] * 100
            if not np.isfinite(val):
                ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
            else:
                ax.bar(
                    x,
                    val,
                    width=bar_spacing * 0.9,
                    color=trained_color,
                    alpha=0.9,
                )

        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            [metric_labels[m] for m in metric_order],
            fontsize=self._metric_xtick_fontsize(),
        )
        ax.set_ylabel(
            "Mean deviation relative to Platform Dependent Policy (%)",
            fontsize=11,
        )
        ax.set_title(
            "Mean deviation on random drones between Platform Independent Policies "
            "and Platform Dependent Policy "
            "(normalized to Platform Dependent Policy, %)",
            fontsize=12,
        )

        legend_labels = [
            f"{self._platform_independent_policy_label()} {i}"
            for i in range(1, gp_count + 1)
        ] + [
            "Platform Dependent Policy (0% reference)"
        ]
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, color=gp_colors[i])
            for i in range(gp_count)
        ]
        legend_patches.append(zero_line)
        ax.legend(legend_patches, legend_labels, title="Policies", fontsize=9)

        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)

    def plot_reward_evolution_per_urdf(self, save_dir=None, show=False, urdf_indices=None):
        rep_mask = self._row_kind_mask(self.df_all, "rep")
        rep_df = self.df_all.loc[rep_mask].copy()
        if len(rep_df) == 0:
            print("[WARN] No 'rep' rows found for reward evolution plot.")
            return
        selected_urdfs = self._normalize_urdf_indices(urdf_indices)

        pct_cols = []
        for col in rep_df.columns:
            match = re.match(r"^rew_(\d+)pct$", str(col))
            if match:
                pct_cols.append((int(match.group(1)), col))
        if not pct_cols:
            print("[WARN] Reward evolution columns 'rew_*pct' not found.")
            return
        pct_cols.sort(key=lambda x: x[0])
        steps = np.array([p for p, _ in pct_cols], dtype=float)
        rew_cols = [c for _, c in pct_cols]
        steps_with_zero = np.concatenate(([0.0], steps))
        reward_matrix = rep_df[rew_cols].to_numpy(dtype=float)
        reward_matrix = reward_matrix + float(self.STEPS_REWARD_OFFSET)
        reward_matrix = np.concatenate(
            [np.zeros((reward_matrix.shape[0], 1), dtype=float), reward_matrix],
            axis=1,
        )
        reward_min = float(np.nanmin(reward_matrix)) if np.isfinite(reward_matrix).any() else 0.0
        reward_max = float(np.nanmax(reward_matrix)) if np.isfinite(reward_matrix).any() else 1.0
        if not np.isfinite(reward_min):
            reward_min = 0.0
        if not np.isfinite(reward_max) or reward_max <= reward_min:
            reward_max = reward_min + 1.0
        y_pad = 0.05 * (reward_max - reward_min)
        y_low = reward_min - y_pad
        y_high = reward_max + y_pad

        grouped = {}
        for _, row in rep_df.iterrows():
            key = self._canonical_urdf_key(row)
            if key is None:
                continue
            grouped.setdefault(key, []).append(row)

        agg_rows = self.df.copy()
        if len(agg_rows) == 0:
            print("[WARN] No 'agg' rows available to map URDFs.")
            return

        if save_dir is not None:
            self._urdf_output_dir(save_dir)

        for row_idx, (_, agg_row) in enumerate(agg_rows.iterrows(), start=1):
            if selected_urdfs is not None and row_idx not in selected_urdfs:
                continue
            key = self._canonical_urdf_key(agg_row)
            rep_rows = grouped.get(key, [])
            if len(rep_rows) == 0:
                continue

            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            curves = []
            for rep_i, rep_row in enumerate(rep_rows, start=1):
                y = np.array([rep_row.get(c, np.nan) for c in rew_cols], dtype=float)
                y = y + float(self.STEPS_REWARD_OFFSET)
                y = np.concatenate(([0.0], y))
                if not np.isfinite(y).any():
                    continue
                curves.append(y)
                ax.plot(
                    steps_with_zero,
                    y,
                    marker="o",
                    linewidth=1.3,
                    alpha=0.7,
                    label=f"rep {rep_i}",
                )

            if len(curves) == 0:
                plt.close(fig)
                continue

            mat = np.vstack(curves)
            mean_curve = np.nanmean(mat, axis=0)
            std_curve = np.nanstd(mat, axis=0)
            valid = np.isfinite(mean_curve)
            if np.any(valid):
                ax.plot(
                    steps_with_zero[valid],
                    mean_curve[valid],
                    color="black",
                    linewidth=2.4,
                    label="mean rep",
                )
                ax.fill_between(
                    steps_with_zero[valid],
                    mean_curve[valid] - std_curve[valid],
                    mean_curve[valid] + std_curve[valid],
                    color="black",
                    alpha=0.12,
                    linewidth=0,
                    label="mean ± std",
                )

            ax.set_xlabel("Training steps (% of total)")
            ax.set_ylabel(f"Reward (+{self.STEPS_REWARD_OFFSET:.1f} offset)")
            ax.set_title(f"URDF {row_idx} — Reward evolution across repetitions")
            ax.grid(alpha=0.3, linestyle="--")
            ax.set_xlim(float(np.min(steps_with_zero)), float(np.max(steps_with_zero)))
            ax.set_ylim(y_low, y_high)
            ax.legend(fontsize=8, ncol=2)
            plt.tight_layout()

            if save_dir is not None:
                out_dir = self._urdf_output_dir(save_dir, row_idx=row_idx)
                filename = 'reward_evolution.png'
                fig.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")

            if show:
                plt.show()
            plt.close(fig)

