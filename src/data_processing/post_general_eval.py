import ast
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


class URDFHistogramPlotter:
    PROGRESS_NORM = 650.0
    REWARD_NORM = 500.0
    MINIMAL_PROGRESS_M = 250.0
    STEPS_REWARD_TARGET = 0.95
    MAX_GENERAL_POLICIES = 6
    BIX3_URDF_STEM = (
        0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4.0, 0.2, 2.0, 0.0, 2.0, 2.5, 3.0, 4.0, 16.0
    )

    def __init__(
        self,
        csv_path,
        save_path="drone_plot.png",
        general_policy_idx=1,
        gp_csv_path=None,
    ):
        self.csv_path = csv_path
        self.gp_csv_path = gp_csv_path
        self.df_all = self._load_df_with_genome(self.csv_path)
        self._fill_missing_baseline_from_gp_csv()
        self.df = self._filter_agg_rows(self.df_all)
        self._steps90_rep_cache = self._build_steps90_rep_cache(self.df_all)
        self.save_path = save_path
        self.general_policy_count = self._infer_policy_count("baseline")
        self.trained_policy_count = self._infer_policy_count("trained")
        self.general_policy_idx = general_policy_idx

    def _row_kind_mask(self, df, kind):
        if "row_kind" not in df.columns:
            return np.zeros(len(df), dtype=bool)
        return df["row_kind"].astype(str).str.lower().eq(kind.lower()).to_numpy()

    def _row_values_as_float_array(self, row, col):
        if col not in row.index:
            return None
        value = row[col]
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return None
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) == 0:
            return None
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if arr.ndim != 1 or not np.isfinite(arr).all():
            return None
        return arr

    def _find_bix3_index(self, df=None, atol=1e-8):
        target = np.asarray(self.BIX3_URDF_STEM, dtype=float)
        source_df = self.df if df is None else df
        for row_i, (_, row) in enumerate(source_df.iterrows()):
            stem = self._row_values_as_float_array(row, "urdf_stem")
            if stem is None or len(stem) != len(target):
                continue
            if np.allclose(stem, target, atol=atol, rtol=0.0):
                return row_i
        return None

    def _filter_agg_rows(self, df):
        if "row_kind" not in df.columns:
            print("[WARN] Missing 'row_kind' column: using all rows.")
            return df
        row_kind = df["row_kind"].astype(str).str.lower()
        agg_df = df[row_kind == "agg"].reset_index(drop=True)
        if len(agg_df) == 0:
            print("[WARN] No 'agg' rows found: using all rows.")
            return df.reset_index(drop=True)
        return agg_df

    def _infer_policy_count(self, policy_kind):
        pattern = re.compile(
            rf"(?:f_speed|f_negE|f_prog|reward_ep_mean)_{policy_kind}(\d+)$"
        )
        indices = []
        for col in self.df.columns:
            match = pattern.match(col)
            if match:
                indices.append(int(match.group(1)))
        count = max(indices) if indices else 0
        if policy_kind == "baseline":
            count = min(count, self.MAX_GENERAL_POLICIES)
        return count

    def _resolve_policy_index(self, policy_idx, max_idx, label):
        if max_idx <= 0:
            warn_key = f"_warned_missing_policy_{label.lower()}"
            if not getattr(self, warn_key, False):
                print(f"[WARN] No policy found for '{label}'.")
                setattr(self, warn_key, True)
            return 1
        if policy_idx is None:
            policy_idx = 1
        if policy_idx < 1:
            print(
                f"[WARN] {label} policy {policy_idx} out of range (1-{max_idx}). Using 1."
            )
            return 1
        if policy_idx > max_idx:
            print(
                f"[WARN] {label} policy {policy_idx} out of range (1-{max_idx}). "
                f"Using {max_idx}."
            )
            return max_idx
        return policy_idx

    def _extract_metric_values(self, row, prefix, count):
        values = []
        for i in range(1, count + 1):
            col = f"{prefix}{i}"
            values.append(row[col] if col in row.index else np.nan)
        return np.array(values, dtype=float)

    def _normalize_cot(self, values):
        cot = -np.array(values, dtype=float)
        cot[np.isclose(cot, 100)] = 0
        return cot

    def _select_policy_value(self, values, policy_idx):
        if len(values) == 0:
            return 0
        if policy_idx < 1 or policy_idx > len(values):
            return 0
        val = values[policy_idx - 1]
        return 0 if not np.isfinite(val) else val

    def _metric_labels_with_norm(self):
        return {
            "thr_count": r"$\frac{\mathbf{\#\ Drones\ >\ Minimal\ Progress}}{\mathrm{Total\ Drones}}$",
            "speed": r"$\frac{\mathbf{Speed}}{\mathrm{Maximum\ Commanded\ Speed}}$",
            "cot": r"$\frac{\mathbf{Cost\ of\ Transport}}{1}$",
            "prog": r"$\frac{\mathbf{Progress}}{\mathrm{Forest\ Length}}$",
            "reward": r"$\frac{\mathbf{Evaluation\ Reward}}{500}$",
            "steps90": r"$\frac{\mathbf{Steps\ to\ 95\%\ Reward}}{\mathrm{Total\ Training\ Steps}}$",
        }

    def _metric_order_hist(self):
        return ["prog", "thr_count", "speed", "cot", "reward", "steps90"]

    def _compute_threshold_rates(self):
        n_drones = len(self.df)
        gp_count = max(1, self.general_policy_count)
        if n_drones == 0:
            return [0.0] * gp_count, 0.0

        threshold = self._minimal_progress_threshold_norm()
        gp_counts = np.zeros(gp_count, dtype=int)
        sp_count = 0

        for _, row in self.df.iterrows():
            metrics = self._process_row(row)
            gp_prog = np.asarray(metrics["baseline_all"]["prog"], dtype=float)
            for pi in range(gp_count):
                if pi < len(gp_prog) and np.isfinite(gp_prog[pi]) and gp_prog[pi] > threshold:
                    gp_counts[pi] += 1

            sp_prog = np.asarray(metrics["trained"]["prog"], dtype=float)
            valid_mask = self._is_valid_metric("prog", sp_prog)
            valid_vals = sp_prog[valid_mask]
            if len(valid_vals) > 0 and valid_vals.mean() > threshold:
                sp_count += 1

        gp_rates = (gp_counts / float(n_drones)).tolist()
        sp_rate = float(sp_count / float(n_drones))
        return gp_rates, sp_rate

    def _minimal_progress_threshold_norm(self):
        return self.MINIMAL_PROGRESS_M / self.PROGRESS_NORM

    def _draw_progress_threshold_segment(self, ax, x_center, width):
        width = width * 1.20
        y = self._minimal_progress_threshold_norm()
        ax.plot(
            [x_center - width / 2, x_center + width / 2],
            [y, y],
            color="#2ca02c",
            linestyle="--",
            linewidth=2.6,
            alpha=0.95,
            zorder=5,
        )

    def _progress_threshold_legend_handle(self):
        return Line2D([0], [0], color="#2ca02c", linestyle="--", linewidth=2.6)

    def _metric_plot_figsize(self):
        # Wider and shorter canvas to give horizontal room to metric labels.
        return (14.0, 4.8)

    def _metric_xtick_fontsize(self):
        return 14

    def _apply_dense_y_grid(self, ax):
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    def _split_leading_lists(self, line, list_count=2):
        items = []
        idx = 0
        for _ in range(list_count):
            start = line.find("[", idx)
            if start == -1:
                break
            depth = 0
            end = None
            for pos in range(start, len(line)):
                if line[pos] == "[":
                    depth += 1
                elif line[pos] == "]":
                    depth -= 1
                    if depth == 0:
                        end = pos
                        break
            if end is None:
                break
            items.append(line[start:end + 1])
            idx = end + 1
        remainder = line[idx:].lstrip(",") if idx < len(line) else ""
        return items, remainder

    def _parse_eval_line(self, line, header):
        first_list_start = line.find("[")
        if first_list_start == -1:
            return None

        prefix = line[:first_list_start].rstrip(",")
        prefix_values = [v.strip() for v in prefix.split(",")] if prefix else []
        lists, remainder = self._split_leading_lists(line[first_list_start:], list_count=2)
        if len(lists) < 2:
            return None

        tail_values = [val.strip() for val in remainder.split(",")] if remainder else []
        values = prefix_values + lists + tail_values
        if len(values) < len(header):
            values += [""] * (len(header) - len(values))
        if len(values) > len(header):
            values = values[:len(header)]
        return dict(zip(header, values))

    def _compute_steps90_ratio(self, row, prefix="rew_"):
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)pct$")
        pct_cols = []
        for col in row.index:
            match = pattern.match(col)
            if match:
                pct_cols.append((int(match.group(1)), col))
        if not pct_cols:
            return np.nan

        pct_cols.sort(key=lambda x: x[0])
        max_pct = pct_cols[-1][0]
        if max_pct <= 0:
            return np.nan

        values = []
        for _, col in pct_cols:
            try:
                values.append(float(row[col]))
            except (TypeError, ValueError):
                values.append(np.nan)
        values = np.asarray(values, dtype=float)

        valid = np.isfinite(values)
        if not valid.any():
            return np.nan

        max_reward = np.nanmax(values)
        if not np.isfinite(max_reward) or max_reward <= 0:
            return np.nan

        threshold = self.STEPS_REWARD_TARGET * max_reward
        for (pct, _), val in zip(pct_cols, values):
            if np.isfinite(val) and val >= threshold:
                return pct / max_pct
        return np.nan

    def _canonical_urdf_key(self, row):
        # Prefer urdf_params because it is the most stable identifier across
        # post-processing files; keep urdf_stem as fallback.
        for col in ("urdf_params", "urdf_stem"):
            if col not in row.index:
                continue
            value = row[col]
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    value = []
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                key = []
                for v in value:
                    try:
                        key.append(round(float(v), 8))
                    except (TypeError, ValueError):
                        key.append(str(v))
                return (col, tuple(key))
        return None

    def _numeric_urdf_array(self, row):
        params = row.get("urdf_params", [])
        if isinstance(params, str):
            try:
                params = ast.literal_eval(params)
            except (SyntaxError, ValueError):
                return None
        if not isinstance(params, (list, tuple, np.ndarray)) or len(params) == 0:
            return None
        try:
            arr = np.asarray(params, dtype=float)
        except (TypeError, ValueError):
            return None
        if arr.ndim != 1 or not np.isfinite(arr).all():
            return None
        return arr

    def _match_gp_row_by_overlap(self, row, gp_rows_with_arrays):
        """Fallback matcher when exact URDF key is missing.

        Matches rows that share most URDF components exactly (after numeric
        parsing), then breaks ties with euclidean distance.
        """
        target = self._numeric_urdf_array(row)
        if target is None:
            return None

        required_overlap = max(3, int(np.ceil(0.7 * len(target))))
        best = None
        best_overlap = -1
        best_dist = np.inf

        for gp_row, gp_arr in gp_rows_with_arrays:
            if gp_arr is None or len(gp_arr) != len(target):
                continue

            overlap = int(np.sum(np.isclose(target, gp_arr, atol=1e-8, rtol=0.0)))
            if overlap < required_overlap:
                continue

            dist = float(np.linalg.norm(target - gp_arr))
            if overlap > best_overlap or (overlap == best_overlap and dist < best_dist):
                best = gp_row
                best_overlap = overlap
                best_dist = dist

        return best

    def _build_steps90_rep_cache(self, df):
        cache = {}
        rep_mask = self._row_kind_mask(df, "rep")
        if not rep_mask.any():
            return cache

        for _, row in df.loc[rep_mask].iterrows():
            key = self._canonical_urdf_key(row)
            if key is None:
                continue
            if key not in cache:
                cache[key] = {"trained": [], "baseline": []}
            tr_ratio = self._compute_steps90_ratio(row, prefix="rew_")
            gp_ratio = self._compute_steps90_ratio(row, prefix="rew_baseline_")
            cache[key]["trained"].append(tr_ratio)
            cache[key]["baseline"].append(gp_ratio)
        return cache

    def _get_steps90_values(self, row, kind, fallback_ratio, count):
        key = self._canonical_urdf_key(row)
        if key is not None and key in self._steps90_rep_cache:
            values = np.asarray(self._steps90_rep_cache[key].get(kind, []), dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            if len(values) > 0:
                return values
        return np.full(max(0, count), fallback_ratio, dtype=float)

    def _is_valid_metric(self, metric, values):
        mask = np.isfinite(values)
        if metric in ("speed", "cot", "steps90"):
            mask = mask & (values > 0)
        return mask

    def _metric_offset(self, metric_index, metric_count, spacing):
        return (metric_index - (metric_count - 1) / 2) * spacing

    def _policy_background_span(self, center, metric_count, spacing):
        half_width = ((metric_count - 1) / 2 + 0.6) * spacing
        return center - half_width, center + half_width

    def _print_minimal_threshold_counts(self):
        """Print how many URDFs exceed the minimal progress threshold per policy."""
        if len(self.df) == 0:
            print("[WARN] No URDFs available for threshold counting.")
            return

        prog_threshold_norm = self.MINIMAL_PROGRESS_M / self.PROGRESS_NORM
        gp_count = max(0, self.general_policy_count)
        gp_pass_counts = [0] * gp_count
        sp_pass_count = 0

        for _, row in self.df.iterrows():
            metrics = self._process_row(row)
            gp_prog = np.asarray(metrics["baseline_all"]["prog"], dtype=float)
            tr_prog = np.asarray(metrics["trained"]["prog"], dtype=float)

            for pi in range(gp_count):
                if pi >= len(gp_prog):
                    continue
                val = gp_prog[pi]
                if np.isfinite(val) and val > prog_threshold_norm:
                    gp_pass_counts[pi] += 1

            valid_mask = np.isfinite(tr_prog) & (tr_prog > 0)
            valid_vals = tr_prog[valid_mask]
            if len(valid_vals) > 0 and valid_vals.mean() > prog_threshold_norm:
                sp_pass_count += 1

        print(
            f"\n=== URDFs above minimal progress threshold "
            f"({self.MINIMAL_PROGRESS_M:.1f} m) ==="
        )
        for pi in range(gp_count):
            print(f"  GP{pi + 1}: {gp_pass_counts[pi]}/{len(self.df)}")
        print(f"  SP: {sp_pass_count}/{len(self.df)}")
        print("===============================================\n")

    def _compute_gp_delta_vs_trained(self, metric_order=None):
        """Compute per-URDF relative deviation used by plot_mean_delta_vs_sp.

        Formula per URDF:
            (GP - mean(SP_valid_repetitions)) / mean(SP_valid_repetitions)
        """
        if metric_order is None:
            metric_order = ["speed", "cot", "prog", "reward", "steps90"]

        gp_count = max(1, self.general_policy_count)
        gp_delta_vals = {m: [[] for _ in range(gp_count)] for m in metric_order}

        for _, row in self.df.iterrows():
            metrics = self._process_row(row)
            gp_all = metrics["baseline_all"]
            tr_all = metrics["trained"]

            for m in metric_order:
                tr_vals = np.asarray(tr_all[m], dtype=float)
                valid_mask = self._is_valid_metric(m, tr_vals)
                valid_vals = tr_vals[valid_mask]
                if len(valid_vals) == 0:
                    continue

                tr_mean_urdf = float(valid_vals.mean())
                if tr_mean_urdf <= 0 or not np.isfinite(tr_mean_urdf):
                    continue

                gp_vals = np.asarray(gp_all[m], dtype=float)
                for pi in range(gp_count):
                    if pi >= len(gp_vals):
                        continue
                    val = gp_vals[pi]
                    if np.isfinite(val) and val > 0:
                        gp_delta_vals[m][pi].append((val - tr_mean_urdf) / tr_mean_urdf)

        return gp_delta_vals

    def _print_delta4_vs_steps90_correlation(self, general_policy_idx=None):
        """Print correlation between GP-vs-SP deviation (4 metrics) and SP steps90."""
        gp_idx = self._resolve_policy_index(
            general_policy_idx if general_policy_idx is not None else self.general_policy_idx,
            self.general_policy_count,
            "General",
        )
        gp_array_idx = max(0, gp_idx - 1)
        delta_metrics = ["speed", "cot", "prog", "reward"]

        per_metric_pairs = {m: {"x": [], "y": []} for m in delta_metrics}
        mean_abs_x = []
        mean_abs_y = []

        for _, row in self.df.iterrows():
            metrics = self._process_row(row, general_policy_idx=gp_idx)
            gp_all = metrics["baseline_all"]
            tr_all = metrics["trained"]

            sp_steps_vals = np.asarray(tr_all["steps90"], dtype=float)
            sp_steps_mask = self._is_valid_metric("steps90", sp_steps_vals)
            sp_steps_valid = sp_steps_vals[sp_steps_mask]
            if len(sp_steps_valid) == 0:
                continue
            sp_steps_mean = float(np.mean(sp_steps_valid))
            if not np.isfinite(sp_steps_mean) or sp_steps_mean <= 0:
                continue

            per_urdf_abs_deltas = []
            for m in delta_metrics:
                sp_vals = np.asarray(tr_all[m], dtype=float)
                sp_mask = self._is_valid_metric(m, sp_vals)
                sp_valid = sp_vals[sp_mask]
                if len(sp_valid) == 0:
                    continue
                sp_mean = float(np.mean(sp_valid))
                if not np.isfinite(sp_mean) or sp_mean <= 0:
                    continue

                gp_vals = np.asarray(gp_all[m], dtype=float)
                if gp_array_idx >= len(gp_vals):
                    continue
                gp_val = float(gp_vals[gp_array_idx])
                if not np.isfinite(gp_val) or gp_val <= 0:
                    continue

                delta = (gp_val - sp_mean) / sp_mean
                per_metric_pairs[m]["x"].append(delta)
                per_metric_pairs[m]["y"].append(sp_steps_mean)
                per_urdf_abs_deltas.append(abs(delta))

            if len(per_urdf_abs_deltas) > 0:
                mean_abs_x.append(float(np.mean(per_urdf_abs_deltas)))
                mean_abs_y.append(sp_steps_mean)

        def _pearson_corr(x_vals, y_vals):
            x = np.asarray(x_vals, dtype=float)
            y = np.asarray(y_vals, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            n = len(x)
            if n < 2:
                return np.nan, n
            if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
                return np.nan, n
            return float(np.corrcoef(x, y)[0, 1]), n

        print(
            "\n=== CORRELATION: % deviation (GP vs SP) over 4 metrics "
            "vs steps@95% reward (SP) ==="
        )
        print(
            "Deviation used: "
            "delta = (GP_selected - mean(SP_valid_repetitions)) / mean(SP_valid_repetitions)"
        )
        print("Steps variable: mean(steps95_SP_valid_repetitions) per URDF.")
        for m in delta_metrics:
            corr, n = _pearson_corr(per_metric_pairs[m]["x"], per_metric_pairs[m]["y"])
            corr_txt = "nan" if not np.isfinite(corr) else f"{corr:+.4f}"
            print(f"{m.upper():>8}: r = {corr_txt}, n = {n}")

        agg_corr, agg_n = _pearson_corr(mean_abs_x, mean_abs_y)
        agg_txt = "nan" if not np.isfinite(agg_corr) else f"{agg_corr:+.4f}"
        print(f"{'MEAN|DELTA|':>8}: r = {agg_txt}, n = {agg_n}")

        def _corr_simple_explanation(corr, metric):
            if not np.isfinite(corr):
                return "insufficient data for interpretation."
            if abs(corr) < 0.1:
                return "very weak correlation: no clear trend emerges."
            if metric == "cot":
                # Lower CoT is better. Here delta>0 means GP has higher CoT -> worse.
                if corr > 0:
                    return (
                        "positive correlation: as steps90 increases, "
                        "delta(COT) increases, so GP tends to be worse than SP "
                        "(higher CoT)."
                    )
                return (
                    "negative correlation: as steps90 increases, "
                    "delta(COT) decreases, so GP tends to be better than SP "
                    "(relatively lower CoT)."
                )

            if metric in ("speed", "prog", "reward"):
                # Higher is better. Here delta>0 means GP better than SP.
                if corr > 0:
                    return (
                        "positive correlation: as steps90 increases, "
                        f"delta({metric.upper()}) increases, so GP tends to be better than SP."
                    )
                return (
                    "negative correlation: as steps90 increases, "
                    f"delta({metric.upper()}) decreases, so GP tends to be worse than SP."
                )

            # Aggregated |delta| case: only gap magnitude has meaning.
            if corr > 0:
                return (
                    "positive correlation: as steps90 increases, "
                    "the absolute GP-SP gap increases."
                )
            return (
                "negative correlation: as steps90 increases, "
                "the absolute GP-SP gap decreases."
            )

        print("Simple interpretation:")
        for m in delta_metrics:
            corr, _ = _pearson_corr(per_metric_pairs[m]["x"], per_metric_pairs[m]["y"])
            print(f"  {m.upper():>8}: {_corr_simple_explanation(corr, m)}")
        print(f"  {'MEAN|DELTA|':>8}: {_corr_simple_explanation(agg_corr, 'mean_abs')}")
        print("====================================================================\n")

    def _get_gene_names(self, n_genes):
        gene_cols = [f"g{i}" for i in range(n_genes)]
        try:
            from morph_evolution.chromosome_drone import Chromosome_Drone
        except Exception:
            import sys
            root = Path(__file__).resolve().parents[1]
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            try:
                from morph_evolution.chromosome_drone import Chromosome_Drone
            except Exception:
                return gene_cols
        param_names = [param.name for param in Chromosome_Drone.PARAMS]
        if len(param_names) == n_genes:
            return param_names
        return gene_cols

    def _load_df_with_genome(self, csv_path=None):
        target_csv = self.csv_path if csv_path is None else csv_path
        if not target_csv or not os.path.exists(target_csv):
            return pd.DataFrame()

        rows = []
        with open(target_csv, "r", encoding="utf-8") as handle:
            header_line = handle.readline()
            if not header_line:
                return pd.DataFrame()
            header = header_line.strip().split(",")
            if len(header) < 5:
                return pd.DataFrame(columns=header)

            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = self._parse_eval_line(line, header)
                if row is None:
                    continue
                for col in ("urdf_stem", "urdf_params"):
                    try:
                        row[col] = ast.literal_eval(row[col]) if isinstance(row[col], str) else []
                    except (SyntaxError, ValueError):
                        row[col] = []
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=header)

        parsed = pd.DataFrame(rows)
        for col in parsed.columns:
            if col in ("row_kind", "urdf_stem", "urdf_params"):
                continue
            parsed[col] = pd.to_numeric(parsed[col], errors="coerce")
        return parsed

    def _baseline_columns(self, df):
        cols = []
        for col in df.columns:
            if not isinstance(col, str):
                continue
            if re.match(r"^(?:f_speed|f_negE|f_prog|reward_ep_mean)_baseline\d+$", col):
                cols.append(col)
        return sorted(cols)

    def _resolve_gp_csv_path(self):
        if self.gp_csv_path:
            return self.gp_csv_path
        if not self.csv_path:
            return None
        base = Path(self.csv_path)
        candidate = base.with_name("evaluation_results_GP.csv")
        return str(candidate)

    def _fill_missing_baseline_from_gp_csv(self):
        gp_csv_path = self._resolve_gp_csv_path()
        if not gp_csv_path or not os.path.exists(gp_csv_path):
            return

        gp_df_all = self._load_df_with_genome(gp_csv_path)
        if len(gp_df_all) == 0:
            return

        gp_df = self._filter_agg_rows(gp_df_all).copy()
        if len(gp_df) == 0:
            return

        baseline_cols_gp = self._baseline_columns(gp_df)
        if not baseline_cols_gp:
            return

        for col in baseline_cols_gp:
            if col not in self.df_all.columns:
                self.df_all[col] = np.nan

        gp_by_key = {}
        gp_rows_with_arrays = []
        for _, gp_row in gp_df.iterrows():
            key = self._canonical_urdf_key(gp_row)
            if key is None:
                continue
            gp_by_key[key] = gp_row
            gp_rows_with_arrays.append((gp_row, self._numeric_urdf_array(gp_row)))

        if not gp_by_key:
            return

        filled_count = 0
        matched_rows = 0
        overlap_matched_rows = 0

        for idx, row in self.df_all.iterrows():
            key = self._canonical_urdf_key(row)
            if key is None:
                continue
            gp_row = gp_by_key.get(key)
            if gp_row is None:
                gp_row = self._match_gp_row_by_overlap(row, gp_rows_with_arrays)
                if gp_row is not None:
                    overlap_matched_rows += 1
            if gp_row is None:
                continue

            matched_rows += 1
            for col in baseline_cols_gp:
                current = self.df_all.at[idx, col]
                if pd.isna(current):
                    missing = True
                else:
                    try:
                        missing = not np.isfinite(float(current))
                    except (TypeError, ValueError):
                        missing = True
                if not missing:
                    continue
                gp_val = gp_row.get(col, np.nan)
                if pd.isna(gp_val):
                    continue
                try:
                    gp_val = float(gp_val)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(gp_val):
                    continue
                self.df_all.at[idx, col] = gp_val
                filled_count += 1

        if filled_count > 0:
            print(
                f"[INFO] GP fallback applied from '{gp_csv_path}': "
                f"matched rows={matched_rows} "
                f"(fallback-overlap={overlap_matched_rows}), "
                f"filled baseline values={filled_count}."
            )

    # ------------------------------------------------------------
    #   Per-row normalizations and transformations
    # ------------------------------------------------------------
    def _process_row(self, row, general_policy_idx=None):
        """Return all normalized metrics for one drone."""
        gp_idx = self._resolve_policy_index(
            general_policy_idx if general_policy_idx is not None else self.general_policy_idx,
            self.general_policy_count,
            "General",
        )

        # Raw values
        gp_speed = self._extract_metric_values(
            row, "f_speed_baseline", self.general_policy_count
        )
        gp_cot = self._extract_metric_values(
            row, "f_negE_baseline", self.general_policy_count
        )
        gp_prog = self._extract_metric_values(
            row, "f_prog_baseline", self.general_policy_count
        )
        gp_reward = self._extract_metric_values(
            row, "reward_ep_mean_baseline", self.general_policy_count
        )

        tr_speed = self._extract_metric_values(
            row, "f_speed_trained", self.trained_policy_count
        )
        tr_cot = self._extract_metric_values(
            row, "f_negE_trained", self.trained_policy_count
        )
        tr_prog = self._extract_metric_values(
            row, "f_prog_trained", self.trained_policy_count
        )
        tr_reward = self._extract_metric_values(
            row, "reward_ep_mean_trained", self.trained_policy_count
        )

        gp_steps90_ratio = self._compute_steps90_ratio(row, prefix="rew_baseline_")
        tr_steps90_ratio = self._compute_steps90_ratio(row, prefix="rew_")

        gp_steps90 = self._get_steps90_values(
            row=row,
            kind="baseline",
            fallback_ratio=gp_steps90_ratio,
            count=self.general_policy_count,
        )
        tr_steps90 = self._get_steps90_values(
            row=row,
            kind="trained",
            fallback_ratio=tr_steps90_ratio,
            count=self.trained_policy_count,
        )

        metrics = {
            "baseline_all": {
                "speed": gp_speed / 24,
                "cot": self._normalize_cot(gp_cot) / 1,
                "prog": gp_prog / self.PROGRESS_NORM,
                "reward": gp_reward / self.REWARD_NORM,
                "steps90": gp_steps90,
            },
            "trained": {
                "speed": tr_speed / 24,
                "cot": self._normalize_cot(tr_cot) / 1,
                "prog": tr_prog / self.PROGRESS_NORM,
                "reward": tr_reward / self.REWARD_NORM,
                "steps90": tr_steps90,
            },
        }

        metrics["baseline_selected"] = {
            m: self._select_policy_value(metrics["baseline_all"][m], gp_idx)
            for m in metrics["baseline_all"]
        }

        return metrics

    # ------------------------------------------------------------
    #   FULL PLOT
    # ------------------------------------------------------------
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
            "Evaluation Reward / 500",
            "Steps to 95% Reward / Total Training Steps",
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
        print("Reported value: mean(delta%) over valid URDFs; std over valid URDFs.")
        delta_metric_order = ["speed", "cot", "prog", "reward", "steps90"]
        gp_delta_vals = self._compute_gp_delta_vs_trained(metric_order=delta_metric_order)
        gp_print_idx = max(0, gp_idx - 1)
        for m in delta_metric_order:
            vals = gp_delta_vals[m][gp_print_idx] if gp_print_idx < len(gp_delta_vals[m]) else []
            if len(vals) == 0:
                mean_pct = 0.0
                std_pct = 0.0
            else:
                mean_pct = float(np.mean(vals) * 100.0)
                std_pct = float((np.std(vals) if len(vals) > 1 else 0.0) * 100.0)
            print(
                f"{m.upper():>8}: delta = {mean_pct:+.2f}% "
                f"(std = {std_pct:.2f}%, n = {len(vals)})"
            )
        print("===============================================================\n")
        self._print_delta4_vs_steps90_correlation(general_policy_idx=gp_idx)
        self._print_minimal_threshold_counts()



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
            ax.text(gp_center, -0.08, "General",
                    ha="center", va="top", fontsize=10,
                    transform=ax.get_xaxis_transform())

            ax.text(sp_center, -0.08, "Specialized",
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
            f"General Policy #{gp_idx} vs Specialized Policy — Normalized Metrics per Drone "
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
            ax.text(gp_center, -0.08, "General", ha="center", va="top",
                    fontsize=10, transform=ax.get_xaxis_transform())

            ax.text(sp_center, -0.08, "Specialized", ha="center", va="top",
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
            "Evaluation Reward / 500",
            "Steps to 95% Reward / Total Training Steps",
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

    def plot_per_urdf_policies(self, save_dir=None, show=False):
        metric_order = ["prog", "speed", "cot", "reward", "steps90"]
        metric_labels = self._metric_labels_with_norm()

        gp_count = max(1, self.general_policy_count)
        cmap = plt.get_cmap("tab10")
        gp_colors = [cmap(i) for i in range(gp_count)]
        trained_color = "#444444"

        bar_spacing = 0.12
        offsets = (np.arange(gp_count + 1) - gp_count / 2) * bar_spacing

        for row_idx, (_, row) in enumerate(self.df.iterrows(), start=1):
            metrics = self._process_row(row)
            gp_all = metrics["baseline_all"]
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

                for pi in range(gp_count):
                    x = center + offsets[pi]
                    val = gp_vals[pi] if pi < len(gp_vals) else 0
                    if val <= 0 or not np.isfinite(val):
                        ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
                    else:
                        ax.bar(x, val, width=bar_spacing * 0.9, color=gp_colors[pi])

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
                f"URDF {row_idx} — General Policies vs SP (mean±std) — {urdf_label}",
                fontsize=12,
            )

            legend_labels = [f"GP{i}" for i in range(1, gp_count + 1)] + ["SP mean±std"]
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
                os.makedirs(save_dir, exist_ok=True)
                filename = f"urdf_{row_idx:02d}.png"
                fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

            if show:
                plt.show()
            plt.close(fig)

    def plot_mean_policies(self, save_path=None, show=False):
        metric_order = self._metric_order_hist()
        metric_labels = self._metric_labels_with_norm()

        gp_count = max(1, self.general_policy_count)
        cmap = plt.get_cmap("tab10")
        gp_colors = [cmap(i) for i in range(gp_count)]
        trained_color = "#444444"
        gp_threshold_rates, sp_threshold_rate = self._compute_threshold_rates()

        bar_spacing = 0.12
        offsets = (np.arange(gp_count + 1) - gp_count / 2) * bar_spacing

        # Accumulate valid values for global mean
        gp_acc = {m: [[] for _ in range(gp_count)] for m in metric_order}
        tr_acc = {m: [] for m in metric_order}

        for _, row in self.df.iterrows():
            metrics = self._process_row(row)
            gp_all = metrics["baseline_all"]
            tr_all = metrics["trained"]

            for m in metric_order:
                if m == "thr_count":
                    continue
                gp_vals = gp_all[m]
                for pi in range(gp_count):
                    if pi < len(gp_vals):
                        val = gp_vals[pi]
                        if np.isfinite(val) and val > 0:
                            gp_acc[m][pi].append(val)

                tr_vals = tr_all[m]
                valid_mask = self._is_valid_metric(m, tr_vals)
                valid_vals = tr_vals[valid_mask]
                if len(valid_vals) > 0:
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
        tr_means = {m: (np.mean(tr_acc[m]) if len(tr_acc[m]) else 0) for m in metric_order}
        tr_stds = {m: (np.std(tr_acc[m]) if len(tr_acc[m]) > 1 else 0) for m in metric_order}
        gp_means["thr_count"] = [gp_threshold_rates[pi] if pi < len(gp_threshold_rates) else 0 for pi in range(gp_count)]
        gp_stds["thr_count"] = [0 for _ in range(gp_count)]
        tr_means["thr_count"] = sp_threshold_rate
        tr_stds["thr_count"] = 0

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
                if (val <= 0 or not np.isfinite(val)) and metric != "thr_count":
                    ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
                else:
                    ax.bar(
                        x,
                        val,
                        width=bar_spacing * 0.9,
                        color=gp_colors[pi],
                        yerr=std_val,
                        capsize=4,
                    )

            # SP mean
            x = center + offsets[-1]
            mean_val = tr_means[metric]
            std_val = tr_stds[metric]
            if (mean_val <= 0 or not np.isfinite(mean_val)) and metric != "thr_count":
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
        ax.set_title("Performance Metrics Mean — General Policies vs SP (mean±std)", fontsize=12)

        legend_labels = [f"GP{i} mean±std" for i in range(1, gp_count + 1)] + ["SP mean±std"]
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

    def plot_mean_sp_policies(self, save_path=None, show=False):
        metric_order = self._metric_order_hist()
        metric_labels = self._metric_labels_with_norm()

        sp_color = "#444444"
        sp_acc = {m: [] for m in metric_order}
        _, sp_threshold_rate = self._compute_threshold_rates()

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
        sp_stds["thr_count"] = 0

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
        ax.set_title("Performance Metrics Mean — Specialized Policies (SP, mean±std)", fontsize=12)

        legend_labels = ["SP mean±std"]
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

        gp_delta_vals = self._compute_gp_delta_vs_trained(metric_order=metric_order)

        gp_delta_mean = {
            m: [
                (np.mean(gp_delta_vals[m][pi]) if len(gp_delta_vals[m][pi]) else 0)
                for pi in range(gp_count)
            ]
            for m in metric_order
        }
        gp_delta_std = {
            m: [
                (np.std(gp_delta_vals[m][pi]) if len(gp_delta_vals[m][pi]) > 1 else 0)
                for pi in range(gp_count)
            ]
            for m in metric_order
        }
        gp_delta_count = {
            m: [len(gp_delta_vals[m][pi]) for pi in range(gp_count)]
            for m in metric_order
        }
        tr_delta_mean = {m: 0 for m in metric_order}

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
        ax.set_ylabel("Mean deviation relative to Specialized policy (%)", fontsize=11)
        ax.set_title(
            "Mean deviation on random drones between GPs and Specialized Policy "
            "(normalized to Specialized policy, %)",
            fontsize=12,
        )

        legend_labels = [f"GP{i}" for i in range(1, gp_count + 1)] + [
            "Specialized policy (0% reference)"
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

    def plot_reward_evolution_per_urdf(self, save_dir=None, show=False):
        rep_mask = self._row_kind_mask(self.df_all, "rep")
        rep_df = self.df_all.loc[rep_mask].copy()
        if len(rep_df) == 0:
            print("[WARN] No 'rep' rows found for reward evolution plot.")
            return

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
            os.makedirs(save_dir, exist_ok=True)

        for row_idx, (_, agg_row) in enumerate(agg_rows.iterrows(), start=1):
            key = self._canonical_urdf_key(agg_row)
            rep_rows = grouped.get(key, [])
            if len(rep_rows) == 0:
                continue

            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            curves = []
            for rep_i, rep_row in enumerate(rep_rows, start=1):
                y = np.array([rep_row.get(c, np.nan) for c in rew_cols], dtype=float)
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
            ax.set_ylabel("Reward")
            ax.set_title(f"URDF {row_idx} — Reward evolution across repetitions")
            ax.grid(alpha=0.3, linestyle="--")
            ax.set_xlim(float(np.min(steps_with_zero)), float(np.max(steps_with_zero)))
            ax.set_ylim(y_low, y_high)
            ax.legend(fontsize=8, ncol=2)
            plt.tight_layout()

            if save_dir is not None:
                filename = f"urdf_{row_idx:02d}_reward_evolution.png"
                fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")

            if show:
                plt.show()
            plt.close(fig)

    def plot_genome_deviation_correlation(
        self,
        general_policy_idx=2,
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

        # Use the in-memory dataframe because it may already include GP fallback
        # values injected from evaluation_results_GP.csv.
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
        corr = np.full((n_genes, len(metric_order)), np.nan, dtype=float)

        for gi in range(n_genes):
            x = genome_matrix[:, gi]
            for mi, metric in enumerate(metric_order):
                y = delta_df[metric].to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 2:
                    continue
                corr[gi, mi] = np.corrcoef(x[mask], y[mask])[0, 1]

        gene_labels = self._get_gene_names(n_genes)
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
            f"Genome vs performance difference between GP{gp_idx} and SP",
            fontsize=12,
        )
        fig.text(
            0.5,
            0.01,
            "Note: only |r| >= 0.25 is annotated (lower values are not shown).",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )

        if n_genes <= 25:
            for i in range(n_genes):
                for j in range(len(metric_order)):
                    if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= 0.25:
                        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
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

        for gi in range(n_genes):
            x = genome_matrix[:, gi]
            for mi, metric in enumerate(metric_order):
                y = sp_df[metric].to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 2:
                    continue
                corr[gi, mi] = np.corrcoef(x[mask], y[mask])[0, 1]

        gene_labels = self._get_gene_names(n_genes)
        fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * n_genes)))
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")

        ax.set_xticks(range(len(metric_order)))
        ax.set_xticklabels([metric_labels[m] for m in metric_order], rotation=15, ha="right")
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(gene_labels)
        ax.set_xlabel("Specialized policy metrics")
        ax.set_ylabel("Genome parameter")
        ax.set_title("Genome vs Specialized policy metrics", fontsize=12)
        fig.text(
            0.5,
            0.01,
            "Note: only |r| >= 0.25 is annotated (lower values are not shown).",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )

        if n_genes <= 25:
            for i in range(n_genes):
                for j in range(len(metric_order)):
                    if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= 0.25:
                        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
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

        for gi in range(n_genes):
            x = genome_matrix[:, gi]
            for mi, metric in enumerate(metric_order):
                y = gp_df[metric].to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 2:
                    continue
                corr[gi, mi] = np.corrcoef(x[mask], y[mask])[0, 1]

        gene_labels = self._get_gene_names(n_genes)
        fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * n_genes)))
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")

        ax.set_xticks(range(len(metric_order)))
        ax.set_xticklabels([metric_labels[m] for m in metric_order], rotation=15, ha="right")
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(gene_labels)
        ax.set_xlabel(f"General policy GP{gp_idx} metrics")
        ax.set_ylabel("Genome parameter")
        ax.set_title(f"Genome vs general policy GP{gp_idx} metrics", fontsize=12)
        fig.text(
            0.5,
            0.01,
            "Note: only |r| >= 0.25 is annotated (lower values are not shown).",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )

        if n_genes <= 25:
            for i in range(n_genes):
                for j in range(len(metric_order)):
                    if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= 0.25:
                        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
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

        pairs = {m: {"x": [], "y": []} for m, _ in metrics_to_plot}

        for _, row in self.df.iterrows():
            metrics = self._process_row(row, general_policy_idx=gp_idx)

            gp_prog = float(metrics["baseline_selected"]["prog"])
            sp_prog_vals = np.asarray(metrics["trained"]["prog"], dtype=float)
            sp_prog_mask = self._is_valid_metric("prog", sp_prog_vals)
            sp_prog_valid = sp_prog_vals[sp_prog_mask]
            if len(sp_prog_valid) == 0:
                continue
            sp_prog_mean = float(np.mean(sp_prog_valid))

            # Keep only drones above minimal progress in both selected GP and SP.
            if not np.isfinite(gp_prog) or gp_prog <= threshold:
                continue
            if not np.isfinite(sp_prog_mean) or sp_prog_mean <= threshold:
                continue

            sp_steps_vals = np.asarray(metrics["trained"]["steps90"], dtype=float)
            sp_steps_mask = self._is_valid_metric("steps90", sp_steps_vals)
            sp_steps_valid = sp_steps_vals[sp_steps_mask]
            if len(sp_steps_valid) == 0:
                continue
            sp_steps_mean = float(np.mean(sp_steps_valid))
            if not np.isfinite(sp_steps_mean) or sp_steps_mean <= 0:
                continue

            for m, _ in metrics_to_plot:
                gp_val = float(metrics["baseline_selected"][m])
                if not np.isfinite(gp_val) or gp_val <= 0:
                    continue

                sp_vals = np.asarray(metrics["trained"][m], dtype=float)
                sp_mask = self._is_valid_metric(m, sp_vals)
                sp_valid = sp_vals[sp_mask]
                if len(sp_valid) == 0:
                    continue
                sp_mean = float(np.mean(sp_valid))
                if not np.isfinite(sp_mean) or sp_mean <= 0:
                    continue

                perf_diff_pct = (gp_val - sp_mean) / sp_mean * 100.0
                pairs[m]["x"].append(sp_steps_mean)
                pairs[m]["y"].append(perf_diff_pct)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        for m, label in metrics_to_plot:
            x = np.asarray(pairs[m]["x"], dtype=float)
            y = np.asarray(pairs[m]["y"], dtype=float)

            versions = ["with_fit_mean"]
            if m == "prog":
                versions = ["points_only", "with_fit_mean"]

            for version in versions:
                fig, ax = plt.subplots(figsize=(8.8, 4.8))
                if len(x) > 0:
                    ax.scatter(
                        x,
                        y,
                        s=28,
                        alpha=0.85,
                        edgecolor="black",
                        linewidth=0.4,
                        label=f"Valid drones (GP{gp_idx} & SP above minimal progress)",
                    )
                else:
                    ax.text(
                        0.5, 0.5, "No valid drones for this metric",
                        ha="center", va="center", transform=ax.transAxes, fontsize=11
                    )

                ax.set_xlabel("SP Steps to Learn (Steps to 95% Reward / Total Training Steps)")
                ax.set_ylabel(
                    f"Performance difference GP{gp_idx} vs SP ({label}, %)\n"
                    r"$100 \cdot \frac{\mathrm{GP}-\mathrm{SP}}{\mathrm{SP}}$"
                )
                title_suffix = ""
                ax.set_title(
                    f"{label} performance difference (GP{gp_idx} vs SP) vs SP steps to learn{title_suffix}",
                    fontsize=12,
                )
                ax.grid(True, linestyle="--", alpha=0.6)
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
                        mean_perf_diff = float(np.nanmean(y))
                        ax.axhline(
                            mean_perf_diff,
                            color="#2ca02c",
                            linestyle="--",
                            linewidth=1.8,
                            label=f"Mean performance difference: {mean_perf_diff:+.2f}%",
                        )
                if len(x) > 0:
                    ax.legend(fontsize=8, loc="best")
                fig.text(
                    0.5,
                    0.002,
                    f"Included drones: only those above minimal progress in both GP{gp_idx} and SP.",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )
                plt.tight_layout(rect=(0, 0.05, 1, 1))

                if save_dir is not None:
                    if m == "prog":
                        filename = (
                            f"gp{gp_idx}_sp_performance_difference_vs_sp_steps_{m}_{version}.png"
                        )
                    else:
                        filename = f"gp{gp_idx}_sp_performance_difference_vs_sp_steps_{m}.png"
                    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close(fig)



if __name__ == "__main__":
    csv_path = "/home/andrea/Documents/Genesis/src/data_processing/evaluation_results.csv"  # Replace with your real CSV path
    plotter = URDFHistogramPlotter(csv_path)
    requested_general_policy_idx = 1
    general_policy_idx = plotter._resolve_policy_index(
        requested_general_policy_idx,
        plotter.general_policy_count,
        "General",
    )
    save_dir = "/home/andrea/Documents/Genesis/src/data_processing/evaluation_plots"
    os.makedirs(save_dir, exist_ok=True)
    plotter.plot(
        general_policy_idx=general_policy_idx,
        save_path=os.path.join(save_dir, "general_vs_specialized.png"),
        show=False,
    )
    plotter.plot_bix3_vs_mean(
        save_path=os.path.join(save_dir, "bix3_vs_mean.png"),
        general_policy_idx=general_policy_idx,
        show=False,
    )
    plotter.plot_per_urdf_policies(save_dir=save_dir, show=False)
    plotter.plot_reward_evolution_per_urdf(save_dir=save_dir, show=False)
    plotter.plot_mean_policies(
        save_path=os.path.join(save_dir, "mean_across_urdfs.png"),
        show=False,
    )
    plotter.plot_mean_sp_policies(
        save_path=os.path.join(save_dir, "mean_across_urdfs_sp.png"),
        show=False,
    )
    plotter.plot_mean_delta_vs_trained(
        save_path=os.path.join(save_dir, "mean_delta_vs_sp.png"),
        show=False,
    )
    plotter.plot_genome_deviation_correlation(
        general_policy_idx=general_policy_idx,
        save_path=os.path.join(save_dir, f"genome_performance_difference_correlation_gp{general_policy_idx}.png"),
        show=False,
    )
    plotter.plot_genome_specialized_correlation(
        save_path=os.path.join(save_dir, "genome_specialized_correlation.png"),
        show=False,
    )
    plotter.plot_genome_general_policy_correlation(
        general_policy_idx=general_policy_idx,
        save_path=os.path.join(save_dir, f"genome_general_policy_correlation_gp{general_policy_idx}.png"),
        show=False,
    )
    plotter.plot_gp1_sp_mismatch_vs_sp_steps(
        general_policy_idx=general_policy_idx,
        save_dir=save_dir,
        show=False,
    )
