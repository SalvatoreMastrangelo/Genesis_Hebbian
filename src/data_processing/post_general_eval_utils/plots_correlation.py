from .common import *
from .stats import *



class _URDFHistogramPlotterCorrelationPlotsMixin:
    def plot_genome_deviation_correlation(
        self,
        general_policy_idx=2,
        save_path=None,
        save_dir=None,
        show=False,
    ):
        gp_idx = self._resolve_policy_index(
            general_policy_idx if general_policy_idx is not None else self.general_policy_idx,
            self.general_policy_count,
            "General",
        )

        metric_order = ["speed", "cot", "prog", "reward", "steps90"]
        metric_labels = self._metric_labels_with_norm()

        # Use the in-memory dataframe because GP values are merged here from
        # evaluation_results_GP*.csv files.
        df = self.df_all

        genome_rows = []
        delta_rows = []

        for _, row in df.iterrows():
            params = row.get("urdf_params", [])
            if isinstance(params, str):
                try:
                    params = ast.literal_eval(params)
                except (SyntaxError, ValueError):
                    params = []
            if not isinstance(params, (list, tuple, np.ndarray)) or len(params) == 0:
                continue
            try:
                genome = np.asarray(params, dtype=float)
            except (TypeError, ValueError):
                continue

            metrics = self._process_row(row, general_policy_idx=gp_idx)
            gp = metrics["baseline_selected"]
            sp = metrics["trained"]

            deltas = {}
            for m in metric_order:
                gp_val = gp[m]
                vals = sp[m]
                valid_mask = self._is_valid_metric(m, vals)
                valid_vals = vals[valid_mask]
                sp_mean = valid_vals.mean() if len(valid_vals) > 0 else 0.0

                if gp_val > 0 and sp_mean > 0:
                    deltas[m] = (gp_val - sp_mean) / sp_mean * 100.0
                else:
                    deltas[m] = np.nan

            genome_rows.append(genome)
            delta_rows.append(deltas)

        if not genome_rows:
            print("[WARN] No valid genome found for correlation.")
            return
        lengths = np.array([len(g) for g in genome_rows], dtype=int)
        if lengths.size == 0:
            print("[WARN] No valid genome found for correlation.")
            return
        target_len = int(np.bincount(lengths).argmax())
        keep_idx = [i for i, length in enumerate(lengths) if length == target_len]
        if not keep_idx:
            print("[WARN] No genome with consistent length found.")
            return

        genome_matrix = np.vstack([genome_rows[i] for i in keep_idx])
        delta_df = pd.DataFrame([delta_rows[i] for i in keep_idx])

        n_genes = genome_matrix.shape[1]
        gene_labels = self._get_gene_names(n_genes)

        outlier_dirs = self._outlier_split_dirs(save_dir)
        default_filename = f"genome_performance_difference_correlation_gp{gp_idx}.png"
        base_filename = os.path.basename(save_path) if save_path is not None else default_filename
        base_root, base_ext = os.path.splitext(base_filename)
        if base_ext == "":
            base_ext = ".png"

        versions = [
            {"key": "yes", "filter_no_outliers": False, "title_suffix": ""},
            {"key": "no", "filter_no_outliers": True, "title_suffix": " (filtered: Progress/Reward y<=100%)"},
        ]

        for version_cfg in versions:
            filter_no_outliers = version_cfg["filter_no_outliers"]
            if filter_no_outliers:
                prog_vals = delta_df["prog"].to_numpy(dtype=float)
                reward_vals = delta_df["reward"].to_numpy(dtype=float)
                keep_mask = (
                    np.isfinite(prog_vals)
                    & np.isfinite(reward_vals)
                    & (prog_vals <= 100.0)
                    & (reward_vals <= 100.0)
                )
                genome_used = genome_matrix[keep_mask]
                delta_used = delta_df.loc[keep_mask].reset_index(drop=True)
            else:
                genome_used = genome_matrix
                delta_used = delta_df

            corr = np.full((n_genes, len(metric_order)), np.nan, dtype=float)
            p_values = np.full((n_genes, len(metric_order)), np.nan, dtype=float)
            counts = np.zeros((n_genes, len(metric_order)), dtype=int)
            for gi in range(n_genes):
                x = genome_used[:, gi]
                for mi, metric in enumerate(metric_order):
                    y = delta_used[metric].to_numpy(dtype=float)
                    stats_row = pearson_corr_stats(x, y)
                    corr[gi, mi] = stats_row["r"]
                    p_values[gi, mi] = stats_row["p_value"]
                    counts[gi, mi] = stats_row["n"]
            q_values = np.full_like(p_values, np.nan)
            finite_mask = np.isfinite(p_values)
            q_values[finite_mask] = benjamini_hochberg(p_values[finite_mask])

            fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * n_genes)))
            im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")

            ax.set_xticks(range(len(metric_order)))
            ax.set_xticklabels(
                [f"{metric_labels[m]} Δ (%)" for m in metric_order],
                rotation=15,
                ha="right",
            )
            ax.set_yticks(range(n_genes))
            ax.set_yticklabels(gene_labels)
            ax.set_xlabel(
                f"Performance difference (GP{gp_idx} vs SP, %)\n"
                r"$\Delta(\%) = 100 \cdot \frac{\mathrm{GP} - \mathrm{SP}_{mean}}{\mathrm{SP}_{mean}}$"
            )
            ax.set_ylabel("Genome parameter")
            ax.set_title(
                f"Genome vs performance difference between GP{gp_idx} and SP{version_cfg['title_suffix']}",
                fontsize=12,
            )
            fig.text(
                0.5,
                0.01,
                "Note: annotations show Pearson r; * means q<0.05 after FDR-BH correction. Full p/q tables are saved alongside the plot.",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#555555",
            )

            if n_genes <= 25:
                for i in range(n_genes):
                    for j in range(len(metric_order)):
                        if np.isfinite(corr[i, j]) and (abs(corr[i, j]) >= 0.20 or significance_marker(q_value=q_values[i, j], p_value=p_values[i, j])):
                            marker = significance_marker(q_value=q_values[i, j], p_value=p_values[i, j])
                            ax.text(j, i, f"{corr[i, j]:.2f}{marker}", ha="center", va="center", fontsize=8)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Pearson r")

            plt.tight_layout()

            if save_dir is not None:
                out_dir = outlier_dirs[version_cfg["key"]]
                target_path = os.path.join(out_dir, base_filename)
                fig.savefig(target_path, dpi=300, bbox_inches="tight")
                stats_df = matrix_stats_frame(gene_labels, metric_order, corr, p_values, q_values, counts, row_name="gene", col_name="metric")
                save_stats_table(target_path, stats_df)
            elif save_path is not None:
                version_suffix = "_outlier_no" if filter_no_outliers else "_outlier_yes"
                target_path = f"{base_root}{version_suffix}{base_ext}"
                fig.savefig(
                    target_path,
                    dpi=300,
                    bbox_inches="tight",
                )
                stats_df = matrix_stats_frame(gene_labels, metric_order, corr, p_values, q_values, counts, row_name="gene", col_name="metric")
                save_stats_table(target_path, stats_df)
            if show:
                plt.show()
            plt.close(fig)

    def plot_genome_specialized_correlation(
        self,
        save_path=None,
        show=False,
    ):
        metric_order = ["speed", "cot", "prog", "reward", "steps90"]
        metric_labels = self._metric_labels_with_norm()

        df = self.df_all
        genome_rows = []
        sp_rows = []

        for _, row in df.iterrows():
            params = row.get("urdf_params", [])
            if isinstance(params, str):
                try:
                    params = ast.literal_eval(params)
                except (SyntaxError, ValueError):
                    params = []
            if not isinstance(params, (list, tuple, np.ndarray)) or len(params) == 0:
                continue
            try:
                genome = np.asarray(params, dtype=float)
            except (TypeError, ValueError):
                continue

            metrics = self._process_row(row)
            sp = metrics["trained"]

            sp_values = {}
            for m in metric_order:
                vals = np.asarray(sp[m], dtype=float)
                valid_mask = self._is_valid_metric(m, vals)
                valid_vals = vals[valid_mask]
                sp_values[m] = valid_vals.mean() if len(valid_vals) > 0 else np.nan

            genome_rows.append(genome)
            sp_rows.append(sp_values)

        if not genome_rows:
            print("[WARN] No valid genome found for specialized correlation.")
            return

        lengths = np.array([len(g) for g in genome_rows], dtype=int)
        if lengths.size == 0:
            print("[WARN] No valid genome found for specialized correlation.")
            return
        target_len = int(np.bincount(lengths).argmax())
        keep_idx = [i for i, length in enumerate(lengths) if length == target_len]
        if not keep_idx:
            print("[WARN] No genome with consistent length found.")
            return

        genome_matrix = np.vstack([genome_rows[i] for i in keep_idx])
        sp_df = pd.DataFrame([sp_rows[i] for i in keep_idx])

        n_genes = genome_matrix.shape[1]
        corr = np.full((n_genes, len(metric_order)), np.nan, dtype=float)
        p_values = np.full((n_genes, len(metric_order)), np.nan, dtype=float)
        counts = np.zeros((n_genes, len(metric_order)), dtype=int)

        for gi in range(n_genes):
            x = genome_matrix[:, gi]
            for mi, metric in enumerate(metric_order):
                y = sp_df[metric].to_numpy()
                stats_row = pearson_corr_stats(x, y)
                corr[gi, mi] = stats_row["r"]
                p_values[gi, mi] = stats_row["p_value"]
                counts[gi, mi] = stats_row["n"]
        q_values = np.full_like(p_values, np.nan)
        finite_mask = np.isfinite(p_values)
        q_values[finite_mask] = benjamini_hochberg(p_values[finite_mask])

        gene_labels = self._get_gene_names(n_genes)
        fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * n_genes)))
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")

        ax.set_xticks(range(len(metric_order)))
        ax.set_xticklabels([metric_labels[m] for m in metric_order], rotation=15, ha="right")
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(gene_labels)
        ax.set_xlabel("Platform Dependent Policy metrics")
        ax.set_ylabel("Genome parameter")
        ax.set_title("Genome vs Platform Dependent Policy metrics", fontsize=12)
        fig.text(
            0.5,
            0.01,
            "Note: annotations show Pearson r; * means q<0.05 after FDR-BH correction. Full p/q table is saved alongside the plot.",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )

        if n_genes <= 25:
            for i in range(n_genes):
                for j in range(len(metric_order)):
                    if np.isfinite(corr[i, j]) and (abs(corr[i, j]) >= 0.20 or significance_marker(q_value=q_values[i, j], p_value=p_values[i, j])):
                        marker = significance_marker(q_value=q_values[i, j], p_value=p_values[i, j])
                        ax.text(j, i, f"{corr[i, j]:.2f}{marker}", ha="center", va="center", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            stats_df = matrix_stats_frame(gene_labels, metric_order, corr, p_values, q_values, counts, row_name="gene", col_name="metric")
            save_stats_table(save_path, stats_df)
        if show:
            plt.show()
        plt.close(fig)

    def plot_genome_general_policy_correlation(
        self,
        general_policy_idx=None,
        save_path=None,
        show=False,
    ):
        gp_idx = self._resolve_policy_index(
            general_policy_idx if general_policy_idx is not None else self.general_policy_idx,
            self.general_policy_count,
            "General",
        )
        metric_order = ["speed", "cot", "prog", "reward", "steps90"]
        metric_labels = self._metric_labels_with_norm()

        df = self.df_all
        genome_rows = []
        gp_rows = []

        for _, row in df.iterrows():
            params = row.get("urdf_params", [])
            if isinstance(params, str):
                try:
                    params = ast.literal_eval(params)
                except (SyntaxError, ValueError):
                    params = []
            if not isinstance(params, (list, tuple, np.ndarray)) or len(params) == 0:
                continue
            try:
                genome = np.asarray(params, dtype=float)
            except (TypeError, ValueError):
                continue

            metrics = self._process_row(row, general_policy_idx=gp_idx)
            gp = metrics["baseline_selected"]

            gp_values = {}
            for m in metric_order:
                val = float(gp[m])
                gp_values[m] = val if self._is_valid_metric(m, np.asarray([val], dtype=float))[0] else np.nan

            genome_rows.append(genome)
            gp_rows.append(gp_values)

        if not genome_rows:
            print("[WARN] No valid genome found for GP correlation.")
            return

        lengths = np.array([len(g) for g in genome_rows], dtype=int)
        if lengths.size == 0:
            print("[WARN] No valid genome found for GP correlation.")
            return
        target_len = int(np.bincount(lengths).argmax())
        keep_idx = [i for i, length in enumerate(lengths) if length == target_len]
        if not keep_idx:
            print("[WARN] No genome with consistent length found.")
            return

        genome_matrix = np.vstack([genome_rows[i] for i in keep_idx])
        gp_df = pd.DataFrame([gp_rows[i] for i in keep_idx])

        n_genes = genome_matrix.shape[1]
        corr = np.full((n_genes, len(metric_order)), np.nan, dtype=float)
        p_values = np.full((n_genes, len(metric_order)), np.nan, dtype=float)
        counts = np.zeros((n_genes, len(metric_order)), dtype=int)

        for gi in range(n_genes):
            x = genome_matrix[:, gi]
            for mi, metric in enumerate(metric_order):
                y = gp_df[metric].to_numpy()
                stats_row = pearson_corr_stats(x, y)
                corr[gi, mi] = stats_row["r"]
                p_values[gi, mi] = stats_row["p_value"]
                counts[gi, mi] = stats_row["n"]
        q_values = np.full_like(p_values, np.nan)
        finite_mask = np.isfinite(p_values)
        q_values[finite_mask] = benjamini_hochberg(p_values[finite_mask])

        gene_labels = self._get_gene_names(n_genes)
        fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * n_genes)))
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")

        ax.set_xticks(range(len(metric_order)))
        ax.set_xticklabels([metric_labels[m] for m in metric_order], rotation=15, ha="right")
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(gene_labels)
        ax.set_xlabel(f"{self._platform_independent_policy_label()} metrics")
        ax.set_ylabel("Genome parameter")
        ax.set_title(
            f"Genome vs {self._platform_independent_policy_label()} metrics",
            fontsize=12,
        )
        fig.text(
            0.5,
            0.01,
            "Note: annotations show Pearson r; * means q<0.05 after FDR-BH correction. Full p/q table is saved alongside the plot.",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )

        if n_genes <= 25:
            for i in range(n_genes):
                for j in range(len(metric_order)):
                    if np.isfinite(corr[i, j]) and (abs(corr[i, j]) >= 0.20 or significance_marker(q_value=q_values[i, j], p_value=p_values[i, j])):
                        marker = significance_marker(q_value=q_values[i, j], p_value=p_values[i, j])
                        ax.text(j, i, f"{corr[i, j]:.2f}{marker}", ha="center", va="center", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            stats_df = matrix_stats_frame(gene_labels, metric_order, corr, p_values, q_values, counts, row_name="gene", col_name="metric")
            save_stats_table(save_path, stats_df)
        if show:
            plt.show()
        plt.close(fig)

    def plot_policy_reward_metric_correlation(
        self,
        save_path=None,
        show=False,
    ):
        """Correlation between reward and core metrics across URDFs for each policy."""
        metric_order = ["prog", "speed", "cot"]
        metric_names = {
            "prog": "Progress",
            "speed": "Speed",
            "cot": "CoT",
        }

        policy_series = []

        for pi in range(1, self.general_policy_count + 1):
            x_by_metric = {m: [] for m in metric_order}
            y_reward = []
            for _, row in self.df.iterrows():
                metrics = self._process_row(row)
                gp_all = metrics["baseline_all"]
                reward_vals = np.asarray(gp_all["reward"], dtype=float)
                if pi - 1 >= len(reward_vals):
                    continue
                reward_val = float(reward_vals[pi - 1])
                reward_valid = self._is_valid_metric(
                    "reward",
                    np.asarray([reward_val], dtype=float),
                )[0]
                if not reward_valid:
                    continue

                candidate = {}
                ok = True
                for m in metric_order:
                    vals = np.asarray(gp_all[m], dtype=float)
                    if pi - 1 >= len(vals):
                        ok = False
                        break
                    val = float(vals[pi - 1])
                    valid = self._is_valid_metric(m, np.asarray([val], dtype=float))[0]
                    if not valid:
                        ok = False
                        break
                    candidate[m] = val

                if not ok:
                    continue

                for m in metric_order:
                    x_by_metric[m].append(candidate[m])
                y_reward.append(reward_val)

            policy_series.append((f"GP{pi}", x_by_metric, y_reward))

        x_by_metric = {m: [] for m in metric_order}
        y_reward = []
        for _, row in self.df.iterrows():
            metrics = self._process_row(row)
            sp_all = metrics["trained"]

            reward_vals = np.asarray(sp_all["reward"], dtype=float)
            reward_mask = self._is_valid_metric("reward", reward_vals)
            reward_valid = reward_vals[reward_mask]
            if len(reward_valid) == 0:
                continue
            reward_val = float(np.mean(reward_valid))

            candidate = {}
            ok = True
            for m in metric_order:
                vals = np.asarray(sp_all[m], dtype=float)
                mask = self._is_valid_metric(m, vals)
                valid = vals[mask]
                if len(valid) == 0:
                    ok = False
                    break
                candidate[m] = float(np.mean(valid))

            if not ok:
                continue

            for m in metric_order:
                x_by_metric[m].append(candidate[m])
            y_reward.append(reward_val)

        policy_series.append(("SP mean", x_by_metric, y_reward))

        if not policy_series:
            print("[WARN] No valid data found for policy reward-metric correlation plot.")
            return

        corr = np.full((len(policy_series), len(metric_order)), np.nan, dtype=float)
        p_values = np.full((len(policy_series), len(metric_order)), np.nan, dtype=float)
        counts = np.zeros((len(policy_series), len(metric_order)), dtype=int)

        for ri, (_, x_by_metric, y_reward) in enumerate(policy_series):
            y = np.asarray(y_reward, dtype=float)
            for ci, metric in enumerate(metric_order):
                x = np.asarray(x_by_metric[metric], dtype=float)
                stats_row = pearson_corr_stats(x, y)
                corr[ri, ci] = stats_row["r"]
                p_values[ri, ci] = stats_row["p_value"]
                counts[ri, ci] = stats_row["n"]
        q_values = np.full_like(p_values, np.nan)
        finite_mask = np.isfinite(p_values)
        q_values[finite_mask] = benjamini_hochberg(p_values[finite_mask])

        fig_h = max(4.8, 0.45 * len(policy_series))
        fig, ax = plt.subplots(figsize=(7.8, fig_h))
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")

        ax.set_xticks(range(len(metric_order)))
        ax.set_xticklabels([metric_names[m] for m in metric_order], rotation=0)
        ax.set_yticks(range(len(policy_series)))
        ax.set_yticklabels([name for name, _, _ in policy_series])
        ax.set_xlabel("Metric")
        ax.set_ylabel("Policy")
        ax.set_title(
            "Correlation across URDFs: Evaluation Reward vs Metric per Policy",
            fontsize=12,
        )

        for ri in range(len(policy_series)):
            for ci in range(len(metric_order)):
                val = corr[ri, ci]
                n = counts[ri, ci]
                p_val = p_values[ri, ci]
                q_val = q_values[ri, ci]
                if np.isfinite(val):
                    txt = f"{val:.2f}{significance_marker(q_value=q_val, p_value=p_val)}\np={format_p_value(p_val)}\nq={format_p_value(q_val)}\n(n={n})"
                else:
                    txt = f"nan\np={format_p_value(p_val)}\nq={format_p_value(q_val)}\n(n={n})"
                ax.text(ci, ri, txt, ha="center", va="center", fontsize=6.5)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            stats_df = matrix_stats_frame([name for name, _, _ in policy_series], metric_order, corr, p_values, q_values, counts, row_name="policy", col_name="metric")
            save_stats_table(save_path, stats_df)
        if show:
            plt.show()
        plt.close(fig)

    def plot_selected_gp_sp_mean_reward_metric_correlation(
        self,
        general_policy_idx=None,
        save_path=None,
        show=False,
    ):
        """Reward-vs-metric correlation across URDFs for selected GP and SP mean.

        Uses only URDFs where both selected GP and SP mean progress are above the
        minimal progress threshold.
        """
        gp_idx = self._resolve_policy_index(
            general_policy_idx if general_policy_idx is not None else self.general_policy_idx,
            self.general_policy_count,
            "General",
        )
        threshold = self._minimal_progress_threshold_norm()
        metric_order = ["prog", "speed", "cot"]
        metric_names = {
            "prog": "Progress",
            "speed": "Speed",
            "cot": "CoT",
        }

        gp_x_by_metric = {m: [] for m in metric_order}
        gp_y_reward = []
        sp_x_by_metric = {m: [] for m in metric_order}
        sp_y_reward = []

        for _, row in self.df.iterrows():
            metrics = self._process_row(row, general_policy_idx=gp_idx)
            gp_selected = metrics["baseline_selected"]
            sp_all = metrics["trained"]

            gp_prog = float(gp_selected["prog"])
            if not np.isfinite(gp_prog) or gp_prog <= threshold:
                continue

            sp_prog_vals = np.asarray(sp_all["prog"], dtype=float)
            sp_prog_mask = self._is_valid_metric("prog", sp_prog_vals)
            sp_prog_valid = sp_prog_vals[sp_prog_mask]
            if len(sp_prog_valid) == 0:
                continue
            sp_prog_mean = float(np.mean(sp_prog_valid))
            if not np.isfinite(sp_prog_mean) or sp_prog_mean <= threshold:
                continue

            gp_candidate = {}
            gp_ok = True
            for m in metric_order:
                val = float(gp_selected[m])
                valid = self._is_valid_metric(m, np.asarray([val], dtype=float))[0]
                if not valid:
                    gp_ok = False
                    break
                gp_candidate[m] = val
            gp_reward = float(gp_selected["reward"])
            gp_reward_valid = self._is_valid_metric(
                "reward",
                np.asarray([gp_reward], dtype=float),
            )[0]
            if not gp_ok or not gp_reward_valid:
                continue

            sp_candidate = {}
            sp_ok = True
            for m in metric_order:
                vals = np.asarray(sp_all[m], dtype=float)
                mask = self._is_valid_metric(m, vals)
                valid_vals = vals[mask]
                if len(valid_vals) == 0:
                    sp_ok = False
                    break
                sp_candidate[m] = float(np.mean(valid_vals))
            sp_reward_vals = np.asarray(sp_all["reward"], dtype=float)
            sp_reward_mask = self._is_valid_metric("reward", sp_reward_vals)
            sp_reward_valid_vals = sp_reward_vals[sp_reward_mask]
            if len(sp_reward_valid_vals) == 0 or not sp_ok:
                continue
            sp_reward = float(np.mean(sp_reward_valid_vals))

            for m in metric_order:
                gp_x_by_metric[m].append(gp_candidate[m])
                sp_x_by_metric[m].append(sp_candidate[m])
            gp_y_reward.append(gp_reward)
            sp_y_reward.append(sp_reward)

        policy_series = [
            (self._platform_independent_policy_label(), gp_x_by_metric, gp_y_reward),
            (f"{self._platform_dependent_policy_label()} mean", sp_x_by_metric, sp_y_reward),
        ]

        corr = np.full((len(policy_series), len(metric_order)), np.nan, dtype=float)
        p_values = np.full((len(policy_series), len(metric_order)), np.nan, dtype=float)
        counts = np.zeros((len(policy_series), len(metric_order)), dtype=int)

        for ri, (_, x_by_metric, y_reward) in enumerate(policy_series):
            y = np.asarray(y_reward, dtype=float)
            for ci, metric in enumerate(metric_order):
                x = np.asarray(x_by_metric[metric], dtype=float)
                stats_row = pearson_corr_stats(x, y)
                corr[ri, ci] = stats_row["r"]
                p_values[ri, ci] = stats_row["p_value"]
                counts[ri, ci] = stats_row["n"]
        q_values = np.full_like(p_values, np.nan)
        finite_mask = np.isfinite(p_values)
        q_values[finite_mask] = benjamini_hochberg(p_values[finite_mask])

        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")

        ax.set_xticks(range(len(metric_order)))
        ax.set_xticklabels([metric_names[m] for m in metric_order], rotation=0)
        ax.set_yticks(range(len(policy_series)))
        ax.set_yticklabels([name for name, _, _ in policy_series])
        ax.set_xlabel("Metric")
        ax.set_ylabel("Policy")
        ax.set_title(
            "Correlation across URDFs (filtered): "
            f"{self._platform_independent_policy_label()} vs "
            f"{self._platform_dependent_policy_label()} mean",
            fontsize=12,
        )

        for ri in range(len(policy_series)):
            for ci in range(len(metric_order)):
                val = corr[ri, ci]
                n = counts[ri, ci]
                p_val = p_values[ri, ci]
                q_val = q_values[ri, ci]
                if np.isfinite(val):
                    txt = f"{val:.2f}{significance_marker(q_value=q_val, p_value=p_val)}\np={format_p_value(p_val)}\nq={format_p_value(q_val)}\n(n={n})"
                else:
                    txt = f"nan\np={format_p_value(p_val)}\nq={format_p_value(q_val)}\n(n={n})"
                ax.text(ci, ri, txt, ha="center", va="center", fontsize=7)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")
        fig.text(
            0.5,
            0.01,
            (
                "Included URDFs: only those with Progress above minimal threshold "
                f"for both {self._platform_independent_policy_label()} and "
                f"{self._platform_dependent_policy_label()} mean."
            ),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

        plt.tight_layout(rect=(0, 0.04, 1, 1))

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            stats_df = matrix_stats_frame([name for name, _, _ in policy_series], metric_order, corr, p_values, q_values, counts, row_name="policy", col_name="metric")
            save_stats_table(save_path, stats_df)
        if show:
            plt.show()
        plt.close(fig)

    def plot_sp_metric_vs_steps_correlation_table(
        self,
        save_path=None,
        show=False,
    ):
        """Table plot: correlation between SP metrics and SP steps-to-threshold."""
        metric_order = ["prog", "cot", "speed"]
        metric_names = {
            "prog": "Progress (SP mean)",
            "cot": "CoT (SP mean)",
            "speed": "Speed (SP mean)",
        }
        threshold_pct = self._steps_target_pct_label()

        rows = []
        for _, row in self.df.iterrows():
            metrics = self._process_row(row)
            sp_all = metrics["trained"]

            steps_vals = np.asarray(sp_all["steps90"], dtype=float)
            steps_mask = self._is_valid_metric("steps90", steps_vals)
            steps_valid = steps_vals[steps_mask]
            if len(steps_valid) == 0:
                continue
            sp_steps_mean = float(np.mean(steps_valid))
            if not np.isfinite(sp_steps_mean):
                continue

            metric_means = {}
            valid = True
            for metric in metric_order:
                vals = np.asarray(sp_all[metric], dtype=float)
                mask = self._is_valid_metric(metric, vals)
                valid_vals = vals[mask]
                if len(valid_vals) == 0:
                    valid = False
                    break
                metric_means[metric] = float(np.mean(valid_vals))
            if not valid:
                continue

            metric_means["steps90"] = sp_steps_mean
            rows.append(metric_means)

        if len(rows) == 0:
            print("[WARN] No valid SP rows found for metric-vs-steps correlation table.")
            return

        corr_rows = []
        raw_p_values = []
        for metric in metric_order:
            x = np.asarray([r[metric] for r in rows], dtype=float)
            y = np.asarray([r["steps90"] for r in rows], dtype=float)
            stats_row = pearson_corr_stats(x, y)
            stats_row["metric"] = metric
            corr_rows.append(stats_row)
            raw_p_values.append(stats_row["p_value"])
        q_values = benjamini_hochberg(raw_p_values)
        for stats_row, q_value in zip(corr_rows, q_values):
            stats_row["q_value_fdr_bh"] = q_value

        fig_h = max(3.8, 1.8 + 0.65 * len(corr_rows))
        fig, ax = plt.subplots(figsize=(9.2, fig_h))
        ax.axis("off")

        col_labels = [
            "Metric",
            "Pearson r",
            "p-value",
            "q-value (FDR)",
            "N URDF",
            f"Steps to {threshold_pct} reward max (SP)",
        ]
        cell_text = []
        for stats_row in corr_rows:
            metric = stats_row["metric"]
            r_val = stats_row["r"]
            n = stats_row["n"]
            p_val = stats_row["p_value"]
            q_val = stats_row["q_value_fdr_bh"]
            r_txt = f"{r_val:+.3f}{significance_marker(q_value=q_val, p_value=p_val)}" if np.isfinite(r_val) else "nan"
            interpretation = "N/A"
            if np.isfinite(r_val):
                if abs(r_val) < 0.1:
                    interpretation = "Very weak"
                elif r_val > 0:
                    interpretation = "Positive"
                else:
                    interpretation = "Negative"
            cell_text.append([metric_names[metric], r_txt, format_p_value(p_val), format_p_value(q_val), str(n), interpretation])

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)

        ax.set_title(
            "SP correlation table: metrics vs steps-to-threshold reward during training",
            fontsize=12,
            pad=10,
        )
        fig.text(
            0.5,
            0.02,
            (
                "Steps metric is normalized by total training steps "
                "(computed from SP reward evolution per URDF)."
            ),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

        plt.tight_layout(rect=(0, 0.05, 1, 1))
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            save_stats_table(save_path, pd.DataFrame(corr_rows), suffix="_stats.csv")
        if show:
            plt.show()
        plt.close(fig)

    def plot_gp1_sp_mismatch_vs_sp_steps(self, general_policy_idx=None, save_dir=None, show=False):
        gp_idx = self._resolve_policy_index(
            general_policy_idx if general_policy_idx is not None else self.general_policy_idx,
            self.general_policy_count,
            "General",
        )
        threshold = self._minimal_progress_threshold_norm()
        metrics_to_plot = [
            ("prog", "Progress"),
            ("speed", "Speed"),
            ("cot", "CoT"),
            ("reward", "Evaluation Reward"),
        ]

        pairs = {m: {"x": [], "y": [], "num": [], "den": []} for m, _ in metrics_to_plot}

        for _, row in self.df.iterrows():
            metrics = self._process_row(row, general_policy_idx=gp_idx)

            gp_prog = float(metrics["baseline_selected"]["prog"])
            sp_prog_vals = np.asarray(metrics["trained"]["prog"], dtype=float)
            sp_prog_mask = self._is_valid_metric("prog", sp_prog_vals)
            sp_prog_valid = sp_prog_vals[sp_prog_mask]
            if len(sp_prog_valid) == 0:
                continue
            sp_prog_mean = float(np.mean(sp_prog_valid))

            sp_steps_vals = np.asarray(metrics["trained"]["steps90"], dtype=float)
            sp_steps_mask = self._is_valid_metric("steps90", sp_steps_vals)
            sp_steps_valid = sp_steps_vals[sp_steps_mask]
            if len(sp_steps_valid) == 0:
                continue
            sp_steps_mean = float(np.mean(sp_steps_valid))
            if not np.isfinite(sp_steps_mean) or sp_steps_mean <= 0:
                continue

            for m, _ in metrics_to_plot:
                use_minimal_progress_filter = m in ("speed", "cot")
                if use_minimal_progress_filter:
                    # Keep threshold filtering only for speed/CoT plots.
                    if not np.isfinite(gp_prog) or gp_prog <= threshold:
                        continue
                    if not np.isfinite(sp_prog_mean) or sp_prog_mean <= threshold:
                        continue

                gp_val = float(metrics["baseline_selected"][m])
                gp_valid = self._is_valid_metric(m, np.asarray([gp_val], dtype=float))[0]
                if not gp_valid:
                    continue

                sp_vals = np.asarray(metrics["trained"][m], dtype=float)
                sp_mask = self._is_valid_metric(m, sp_vals)
                sp_valid = sp_vals[sp_mask]
                if len(sp_valid) == 0:
                    continue
                sp_mean = float(np.mean(sp_valid))
                if not np.isfinite(sp_mean) or np.isclose(sp_mean, 0.0):
                    continue

                perf_diff_pct = (gp_val - sp_mean) / sp_mean * 100.0
                pairs[m]["x"].append(sp_steps_mean)
                pairs[m]["y"].append(perf_diff_pct)
                pairs[m]["num"].append(gp_val - sp_mean)
                pairs[m]["den"].append(sp_mean)

        outlier_dirs = self._outlier_split_dirs(save_dir)
        scatter_stats_rows = []

        for m, label in metrics_to_plot:
            x_all = np.asarray(pairs[m]["x"], dtype=float)
            y_all = np.asarray(pairs[m]["y"], dtype=float)
            num_all = np.asarray(pairs[m]["num"], dtype=float)
            den_all = np.asarray(pairs[m]["den"], dtype=float)

            versions = [{"name": "with_fit_mean", "filter_y_le_100": False}]
            if m in ("prog", "reward"):
                versions.append({"name": "with_fit_mean", "filter_y_le_100": True})

            for version_cfg in versions:
                version = version_cfg["name"]
                filter_y_le_100 = version_cfg["filter_y_le_100"]
                if filter_y_le_100:
                    valid_mask = np.isfinite(x_all) & np.isfinite(y_all) & (y_all <= 100.0)
                else:
                    valid_mask = np.isfinite(x_all) & np.isfinite(y_all)
                x = x_all[valid_mask]
                y = y_all[valid_mask]
                num = num_all[valid_mask]
                den = den_all[valid_mask]

                corr_stats = pearson_corr_stats(x, y)
                scatter_stats_rows.append({
                    "metric": m,
                    "version": version,
                    "filter_y_le_100": bool(filter_y_le_100),
                    "r": corr_stats["r"],
                    "p_value": corr_stats["p_value"],
                    "n": corr_stats["n"],
                })
                fig, ax = plt.subplots(figsize=(8.8, 4.8))
                use_minimal_progress_filter = m in ("speed", "cot")
                if len(x) > 0:
                    if use_minimal_progress_filter:
                        point_label = (
                            "Valid drones (Platform Independent Policy & "
                            "Platform Dependent Policy above minimal progress)"
                        )
                    else:
                        point_label = "All valid drones"
                    if filter_y_le_100:
                        point_label += ", y<=100%"
                    ax.scatter(
                        x,
                        y,
                        s=28,
                        alpha=0.85,
                        edgecolor="black",
                        linewidth=0.4,
                        label=point_label,
                    )
                else:
                    ax.text(
                        0.5, 0.5, "No valid drones for this metric",
                        ha="center", va="center", transform=ax.transAxes, fontsize=11
                    )

                ax.set_xlabel(
                    f"{self._platform_dependent_policy_label()} Steps to Learn "
                    f"(Steps to {self._steps_target_pct_label()} Reward / Total Training Steps)"
                )
                ax.set_ylabel(
                    "Performance difference "
                    f"{self._platform_independent_policy_label()} vs "
                    f"{self._platform_dependent_policy_label()} ({label}, %)\n"
                    r"$100 \cdot \frac{\mathrm{PIP}-\mathrm{PDP}}{\mathrm{PDP}}$"
                )
                title_suffix = ""
                if filter_y_le_100:
                    title_suffix = " (filtered: y<=100%)"
                ax.set_title(
                    f"{label} performance difference "
                    f"({self._platform_independent_policy_label()} vs "
                    f"{self._platform_dependent_policy_label()}) vs "
                    f"{self._platform_dependent_policy_label()} steps to learn"
                    f"{title_suffix}",
                    fontsize=12,
                )
                ax.grid(True, linestyle="--", alpha=0.6)
                corr_text = (
                    f"r={corr_stats['r']:+.3f}\np={format_p_value(corr_stats['p_value'])}\n(n={corr_stats['n']})"
                    if np.isfinite(corr_stats["r"])
                    else f"r=nan\np={format_p_value(corr_stats['p_value'])}\n(n={corr_stats['n']})"
                )
                ax.text(
                    0.02,
                    0.98,
                    corr_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"},
                )
                if len(x) > 0:
                    x_min = float(np.nanmin(x))
                    x_max = float(np.nanmax(x))
                    x_span = x_max - x_min
                    if x_span <= 1e-12:
                        x_margin = max(0.02, abs(x_min) * 0.05)
                    else:
                        x_margin = x_span * 0.05
                    ax.set_xlim(x_min - x_margin, x_max + x_margin)
                if version == "with_fit_mean" and len(x) >= 2:
                    fit = np.polyfit(x, y, 1)
                    x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 120)
                    y_line = fit[0] * x_line + fit[1]
                    ax.plot(
                        x_line,
                        y_line,
                        color="#ff7f0e",
                        linewidth=2.0,
                        label=f"Linear fit: y = {fit[0]:+.2f}x {fit[1]:+.2f}",
                    )
                if len(y) > 0:
                    ymin = min(-5.0, float(np.nanmin(y)) * 1.1)
                    ymax = max(5.0, float(np.nanmax(y)) * 1.1)
                    if ymin < ymax:
                        ax.set_ylim(ymin, ymax)
                    if version == "with_fit_mean":
                        mean_perf_diff = self._compute_mean_delta_ratio(
                            num,
                            den,
                        )
                        if np.isfinite(mean_perf_diff):
                            mean_perf_diff_pct = float(mean_perf_diff * 100.0)
                            ax.axhline(
                                mean_perf_diff_pct,
                                color="#2ca02c",
                                linestyle="--",
                                linewidth=1.8,
                                label=(
                                    "Mean performance difference "
                                    f"(sum delta / sum PDP): {mean_perf_diff_pct:+.2f}%"
                                ),
                            )
                if len(x) > 0:
                    ax.legend(fontsize=8, loc="best")
                fig.text(
                    0.5,
                    0.002,
                    (
                        "Included drones: only those above minimal progress in both "
                        "Platform Independent Policy and Platform Dependent Policy."
                        if use_minimal_progress_filter
                        else "Included drones: all valid drones (no minimal progress filter)."
                    ),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )
                plt.tight_layout(rect=(0, 0.05, 1, 1))

                if save_dir is not None:
                    suffix = '_filtered_y_le_100' if filter_y_le_100 else ''
                    filename = f'{m}_difference_vs_sp_steps{suffix}.png'
                    target_dir = outlier_dirs['yes']
                    if m in ('prog', 'reward') and filter_y_le_100:
                        target_dir = outlier_dirs['no']
                    fig.savefig(os.path.join(target_dir, filename), dpi=300, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close(fig)

        if scatter_stats_rows:
            q_values = benjamini_hochberg([row["p_value"] for row in scatter_stats_rows])
            for row, q_value in zip(scatter_stats_rows, q_values):
                row["q_value_fdr_bh"] = q_value
            if save_dir is not None:
                save_stats_table(os.path.join(save_dir, f"gp{gp_idx}_sp_mismatch_vs_sp_steps.png"), pd.DataFrame(scatter_stats_rows), suffix="_stats.csv")

