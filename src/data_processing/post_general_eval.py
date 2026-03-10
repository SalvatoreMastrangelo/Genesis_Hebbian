import ast
import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


class URDFHistogramPlotter:
    PROGRESS_NORM = 650.0
    SPEED_NORM = 30.0
    REWARD_NORM = 500.0
    MINIMAL_PROGRESS_M = 250.0
    STEPS_REWARD_TARGET = 0.9
    STEPS_REWARD_OFFSET = 5.0
    MAX_GENERAL_POLICIES = 6
    INVALID_SPEED_VALUES = (0.0,)
    INVALID_PROGRESS_VALUES = (0.0,)
    INVALID_NEGE_VALUES = (-10.0, -100.0)
    BIX3_URDF_STEM = (
        0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4.0, 0.2, 2.0, 0.0, 2.0, 2.5, 3.0, 4.0, 16.0
    )

    def __init__(
        self,
        csv_path,
        save_path="drone_plot.png",
        general_policy_idx=1,
        gp_csv_paths=None,
    ):
        self.csv_path = self._resolve_sp_csv_path(csv_path)
        self.gp_csv_paths = self._resolve_gp_csv_paths(gp_csv_paths)
        self.df_all = self._load_df_with_genome(self.csv_path)
        self._gp_policy_count_from_csvs = 0
        self._gp_runtime_divisor_by_policy_idx = {}
        self._merge_gp_baselines_from_csvs()
        self.df = self._filter_agg_rows(self.df_all)
        self._steps90_rep_cache = self._build_steps90_rep_cache(self.df_all)
        self.save_path = save_path
        if self._gp_policy_count_from_csvs > 0:
            self.general_policy_count = self._gp_policy_count_from_csvs
        else:
            self.general_policy_count = self._infer_policy_count("baseline")
        self.trained_policy_count = self._infer_policy_count("trained")
        self.general_policy_idx = general_policy_idx
        self._load_eval_normalization_scales()

    def _load_eval_normalization_scales(self):
        """Load normalization constants from evaluation/env source files."""
        root = Path(__file__).resolve().parents[1]
        eval_path = root / "winged_drone_train" / "eval.py"
        env_path = root / "winged_drone_train" / "env.py"

        progress_norm = float(self.PROGRESS_NORM)
        speed_norm = float(self.SPEED_NORM)

        if eval_path.exists():
            try:
                txt = eval_path.read_text(encoding="utf-8")
                x_upper_match = re.search(r"\bx_upper\s*=\s*([-+]?\d*\.?\d+)", txt)
                base_x_match = re.search(
                    r"\bbase_init_pos\s*=\s*\[\s*([-+]?\d*\.?\d+)",
                    txt,
                )
                if x_upper_match:
                    x_upper = float(x_upper_match.group(1))
                    if base_x_match:
                        base_x = float(base_x_match.group(1))
                        candidate = x_upper - base_x
                        if np.isfinite(candidate) and candidate > 0:
                            progress_norm = candidate
                    elif np.isfinite(x_upper) and x_upper > 0:
                        progress_norm = x_upper
            except OSError:
                pass

        if env_path.exists():
            try:
                txt = env_path.read_text(encoding="utf-8")
                max_speed_matches = re.findall(
                    r'get\("max_speed",\s*([-+]?\d*\.?\d+)\)',
                    txt,
                )
                if max_speed_matches:
                    speeds = np.asarray([float(v) for v in max_speed_matches], dtype=float)
                    speeds = speeds[np.isfinite(speeds) & (speeds > 0)]
                    if len(speeds) > 0:
                        speed_norm = float(np.max(speeds))
            except OSError:
                pass

        self.PROGRESS_NORM = float(progress_norm)
        self.SPEED_NORM = float(speed_norm)
        print(
            "[INFO] Normalization scales loaded: "
            f"progress={self.PROGRESS_NORM:.3f} m, "
            f"max_speed={self.SPEED_NORM:.3f} m/s"
        )

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

    def _synthesize_missing_sp_agg_rows(self, df):
        if len(df) == 0 or "row_kind" not in df.columns:
            return df

        rep_mask = self._row_kind_mask(df, "rep")
        agg_mask = self._row_kind_mask(df, "agg")
        if not rep_mask.any():
            return df

        agg_keys = set()
        for _, row in df.loc[agg_mask].iterrows():
            key = self._canonical_urdf_key(row)
            if key is not None:
                agg_keys.add(key)

        synthetic_rows = []
        for key, group in df.loc[rep_mask].groupby(
            df.loc[rep_mask].apply(self._canonical_urdf_key, axis=1)
        ):
            if key is None or key in agg_keys:
                continue

            rep_rows = group.reset_index(drop=True)
            out = {}
            for col in df.columns:
                if col == "row_kind":
                    out[col] = "agg"
                    continue
                if col == "rep_idx":
                    out[col] = np.nan
                    continue
                series = rep_rows[col]
                non_na = series.dropna()
                if col in ("urdf_stem", "urdf_params"):
                    out[col] = non_na.iloc[0] if len(non_na) > 0 else []
                else:
                    out[col] = non_na.iloc[0] if len(non_na) > 0 else np.nan

            trained_metric_prefixes = (
                "f_speed_trained",
                "f_negE_trained",
                "f_prog_trained",
                "reward_ep_mean_trained",
            )
            for prefix in trained_metric_prefixes:
                values = []
                idx = 1
                while f"{prefix}{idx}" in df.columns:
                    val = out.get(f"{prefix}{idx}", np.nan)
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = np.nan
                    if np.isfinite(val):
                        values.append(val)
                    idx += 1
                mean_col = f"{prefix}_mean"
                if mean_col in df.columns and len(values) > 0:
                    out[mean_col] = float(np.mean(values))

            synthetic_rows.append(out)

        if not synthetic_rows:
            return df

        synth_df = pd.DataFrame(synthetic_rows, columns=df.columns)
        print(
            f"[INFO] Synthesized {len(synth_df)} missing SP aggregate row(s) from rep rows."
        )
        return pd.concat([df, synth_df], ignore_index=True)

    def _infer_policy_count(self, policy_kind):
        return self._infer_policy_count_from_df(self.df, policy_kind)

    def _infer_policy_count_from_df(self, df, policy_kind):
        pattern = re.compile(
            rf"(?:f_speed|f_negE|f_prog|reward_ep_mean)_{policy_kind}(\d+)$"
        )
        indices = []
        for col in df.columns:
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
        invalid_cot_values = -np.asarray(self.INVALID_NEGE_VALUES, dtype=float)
        if invalid_cot_values.size > 0:
            cot_flat = np.atleast_1d(cot)
            invalid_mask = np.isclose(
                cot_flat[:, None],
                invalid_cot_values[None, :],
            ).any(axis=1)
            cot_flat[invalid_mask] = 0
            cot = cot_flat.reshape(np.shape(cot))
        return cot

    def _matches_any_invalid(self, values, invalid_values):
        vals = np.asarray(values, dtype=float)
        invalid = np.asarray(tuple(invalid_values), dtype=float)
        if vals.ndim == 0:
            vals = vals.reshape(1)
        if invalid.size == 0:
            return np.zeros(vals.shape, dtype=bool)
        return np.isclose(vals[:, None], invalid[None, :]).any(axis=1)

    def _select_policy_value(self, values, policy_idx):
        if len(values) == 0:
            return 0
        if policy_idx < 1 or policy_idx > len(values):
            return 0
        val = values[policy_idx - 1]
        return 0 if not np.isfinite(val) else val

    def _metric_labels_with_norm(self):
        target_pct_tex = self._steps_target_pct_tex()
        return {
            "thr_count": r"$\frac{\mathbf{\#\ Drones\ >\ Minimal\ Progress}}{\mathrm{Total\ Drones}}$",
            "speed": r"$\frac{\mathbf{Speed}}{\mathrm{Maximum\ Commanded\ Speed}}$",
            "cot": r"$\frac{\mathbf{Cost\ of\ Transport}}{1}$",
            "prog": r"$\frac{\mathbf{Progress}}{\mathrm{Forest\ Length}}$",
            "reward": r"$\frac{\mathbf{Evaluation\ Reward}}{500}$",
            "steps90": (
                rf"$\frac{{\mathbf{{Steps\ to\ {target_pct_tex}\ Reward}}}}"
                r"{\mathrm{Total\ Training\ Steps}}$"
            ),
        }

    def _steps_target_pct(self):
        return float(self.STEPS_REWARD_TARGET) * 100.0

    def _steps_target_pct_label(self):
        pct = self._steps_target_pct()
        rounded = round(pct)
        if np.isclose(pct, rounded):
            return f"{int(rounded)}%"
        return f"{pct:.1f}%"

    def _steps_target_pct_tex(self):
        return self._steps_target_pct_label().replace("%", r"\%")

    def _metric_order_hist(self):
        return ["prog", "thr_count", "speed", "cot", "reward", "steps90"]

    def _compute_threshold_rates(self):
        gp_means, _, sp_mean, _ = self._compute_threshold_rate_stats()
        return gp_means, sp_mean

    def _trial_pass_rates_for_progress(self, df, prefix):
        if len(df) == 0:
            return []
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        indexed_cols = []
        for col in df.columns:
            if not isinstance(col, str):
                continue
            match = pattern.match(col)
            if match:
                indexed_cols.append((int(match.group(1)), col))
        indexed_cols.sort(key=lambda x: x[0])
        if not indexed_cols:
            return []

        # CSV progress columns are in raw meters, not normalized units.
        threshold = float(self.MINIMAL_PROGRESS_M)
        n_drones = len(df)
        rates = []
        for _, col in indexed_cols:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            pass_mask = np.isfinite(vals) & (vals > threshold)
            rate = float(np.sum(pass_mask) / float(n_drones))
            rates.append(rate)
        return rates

    def _compute_threshold_rate_stats(self):
        gp_count = max(1, self.general_policy_count)
        gp_means = [0.0] * gp_count
        gp_stds = [0.0] * gp_count

        # GP: use already-merged agg rows so policy indexing stays consistent
        # across both cases:
        #   1. evaluation_results_GP.csv -> baseline1, baseline2, ... are distinct policies
        #   2. evaluation_results_GP{idx}*.csv -> files with same idx are seed variants
        #      of the same policy and are already merged into baseline{idx}.
        threshold = float(self.MINIMAL_PROGRESS_M)
        n_drones = len(self.df)
        if n_drones > 0:
            for pi in range(gp_count):
                col = f"f_prog_baseline{pi + 1}"
                if col not in self.df.columns:
                    continue
                vals = pd.to_numeric(self.df[col], errors="coerce").to_numpy(dtype=float)
                pass_mask = np.isfinite(vals) & (vals > threshold)
                gp_means[pi] = float(np.sum(pass_mask) / float(n_drones))
                gp_stds[pi] = 0.0

        # SP: use trained trials directly from SP agg rows.
        sp_rates = self._trial_pass_rates_for_progress(self.df, "f_prog_trained")
        sp_mean = float(np.mean(sp_rates)) if len(sp_rates) > 0 else 0.0
        sp_std = float(np.std(sp_rates)) if len(sp_rates) > 1 else 0.0
        return gp_means, gp_stds, sp_mean, sp_std

    def _minimal_progress_threshold_norm(self):
        return self.MINIMAL_PROGRESS_M / self.PROGRESS_NORM

    def _draw_progress_threshold_segment(self, ax, x_center, width, threshold_norm=None):
        width = width * 1.20
        y = (
            self._minimal_progress_threshold_norm()
            if threshold_norm is None
            else float(threshold_norm)
        )
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
        values = values + float(self.STEPS_REWARD_OFFSET)

        valid = np.isfinite(values)
        if not valid.any():
            return np.nan

        max_reward = np.nanmax(values)
        if not np.isfinite(max_reward) or max_reward <= 0:
            return np.nan

        threshold = self.STEPS_REWARD_TARGET * max_reward

        pcts = np.asarray([pct for pct, _ in pct_cols], dtype=float)
        valid_mask = np.isfinite(values) & np.isfinite(pcts)
        pcts = pcts[valid_mask]
        vals = values[valid_mask]
        if len(pcts) == 0:
            return np.nan

        # Linear interpolation between adjacent sampled points (every 5%),
        # so the crossing step is not quantized to the discrete bins.
        for i in range(1, len(pcts)):
            x0, x1 = pcts[i - 1], pcts[i]
            y0, y1 = vals[i - 1], vals[i]
            crossed = (
                (y0 < threshold <= y1) or
                (y0 > threshold >= y1)
            )
            if not crossed:
                continue
            if np.isclose(y1, y0):
                x_cross = x1
            else:
                alpha = (threshold - y0) / (y1 - y0)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                x_cross = x0 + alpha * (x1 - x0)
            return float(x_cross / max_pct)

        # Fallback: if no segment crossing was detected, keep previous behavior.
        above_idx = np.where(vals >= threshold)[0]
        if len(above_idx) > 0:
            return float(pcts[int(above_idx[0])] / max_pct)
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
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(values)

        if metric == "speed":
            mask = mask & (values > 0) & (~self._matches_any_invalid(values, self.INVALID_SPEED_VALUES))
        elif metric == "cot":
            invalid_cot_values = -np.asarray(self.INVALID_NEGE_VALUES, dtype=float)
            mask = mask & (values > 0) & (~self._matches_any_invalid(values, invalid_cot_values))
        elif metric == "prog":
            mask = mask & (~self._matches_any_invalid(values, self.INVALID_PROGRESS_VALUES))
        elif metric == "steps90":
            mask = mask & (values > 0)
        return mask

    def _metric_offset(self, metric_index, metric_count, spacing):
        return (metric_index - (metric_count - 1) / 2) * spacing

    def _policy_background_span(self, center, metric_count, spacing):
        half_width = ((metric_count - 1) / 2 + 0.6) * spacing
        return center - half_width, center + half_width

    def _urdf_output_dir(self, save_dir):
        if save_dir is None:
            return None
        out_dir = os.path.join(save_dir, "urdf_xx")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _outlier_split_dirs(self, save_dir, yes_name="outlier_yes", no_name="outlier_no"):
        if save_dir is None:
            return {"yes": None, "no": None}
        yes_dir = os.path.join(save_dir, yes_name)
        no_dir = os.path.join(save_dir, no_name)
        os.makedirs(yes_dir, exist_ok=True)
        os.makedirs(no_dir, exist_ok=True)
        return {"yes": yes_dir, "no": no_dir}

    def _load_top_drone_visualizer_class(self):
        module_path = Path(__file__).resolve().parent / "top_drone_visualization.py"
        if not module_path.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location(
                "_top_drone_visualization_module",
                str(module_path),
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return getattr(module, "TopDroneVisualizer", None)
        except Exception as exc:
            print(f"[WARN] Could not load top_drone_visualization.py: {exc}")
            return None

    def _load_urdf_maker_class(self):
        module_path = Path(__file__).resolve().parents[1] / "drone_making.py"
        if not module_path.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location(
                "_drone_making_module",
                str(module_path),
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return getattr(module, "UrdfMaker", None)
        except Exception as exc:
            print(f"[WARN] Could not load drone_making.py: {exc}")
            return None

    def _render_drone_picture(self, row_idx, row, save_dir):
        out_dir = self._urdf_output_dir(save_dir)
        if out_dir is None:
            return
        stem = str(row.get("urdf_stem", "")).strip()
        if stem == "":
            print(f"[WARN] Missing urdf_stem for URDF {row_idx}; skipping drone picture.")
            return

        urdf_gen_dir = Path(__file__).resolve().parents[1] / "urdf_generated"
        urdf_path = urdf_gen_dir / f"{stem}.urdf"
        if not urdf_path.exists():
            params = row.get("urdf_params", [])
            if isinstance(params, str):
                try:
                    params = ast.literal_eval(params)
                except (SyntaxError, ValueError):
                    params = []
            if not isinstance(params, (list, tuple, np.ndarray)) or len(params) == 0:
                print(
                    f"[WARN] URDF file missing and urdf_params invalid for URDF {row_idx}; "
                    "skipping drone picture."
                )
                return
            if not hasattr(self, "_urdf_maker_cls"):
                self._urdf_maker_cls = self._load_urdf_maker_class()
            urdf_maker_cls = self._urdf_maker_cls
            if urdf_maker_cls is None:
                print("[WARN] URDF file missing and UrdfMaker unavailable; skipping drone picture.")
                return
            try:
                created_path = urdf_maker_cls(params, out_dir=urdf_gen_dir).create_urdf()
                urdf_path = Path(created_path).expanduser().resolve()
            except Exception as exc:
                print(
                    f"[WARN] Failed to generate missing URDF from params for URDF {row_idx}: {exc}"
                )
                return

        out_path = Path(out_dir) / f"urdf_{row_idx:02d}_picture.png"
        if out_path.exists():
            return

        if not hasattr(self, "_top_drone_visualizer_cls"):
            self._top_drone_visualizer_cls = self._load_top_drone_visualizer_class()
        visualizer_cls = self._top_drone_visualizer_cls

        if not hasattr(self, "_top_drone_visualizer_instance"):
            self._top_drone_visualizer_instance = None
        if self._top_drone_visualizer_instance is None and visualizer_cls is not None:
            try:
                self._top_drone_visualizer_instance = visualizer_cls()
            except Exception as exc:
                print(f"[WARN] Failed to initialize TopDroneVisualizer: {exc}")
                self._top_drone_visualizer_instance = None

        try:
            if self._top_drone_visualizer_instance is not None:
                self._top_drone_visualizer_instance.render_urdf(
                    str(urdf_path),
                    str(out_path),
                )
            else:
                self._render_drone_picture_fallback(urdf_path=urdf_path, out_path=out_path)
        except Exception as exc:
            print(f"[WARN] Failed to render drone picture for URDF {row_idx}: {exc}")

    def render_single_drone_picture(self, drone_idx, save_dir):
        idx = int(drone_idx)
        total = len(self.df)
        if idx < 1 or idx > total:
            raise ValueError(f"drone_idx {idx} out of range [1, {total}]")
        row = self.df.iloc[idx - 1]
        self._render_drone_picture(row_idx=idx, row=row, save_dir=save_dir)
        out_dir = self._urdf_output_dir(save_dir)
        if out_dir is not None:
            print(
                "[INFO] Drone picture path: "
                f"{os.path.join(out_dir, f'urdf_{idx:02d}_picture.png')}"
            )

    def _render_drone_picture_fallback(self, urdf_path, out_path):
        import genesis as gs
        from genesis.utils.misc import tensor_to_array

        camera_pos = (3.0, 1.8, 1.2)
        camera_lookat = (0.0, 0.0, 0.3)

        gs.init()
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1 / 60.0, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=30,
                camera_pos=camera_pos,
                camera_lookat=camera_lookat,
                res=(800, 800),
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
                show_world_frame=False,
                show_link_frame=False,
                background_color=(1.0, 1.0, 1.0),
                ambient_light=(1.0, 1.0, 1.0),
                shadow=False,
                plane_reflection=False,
            ),
            rigid_options=gs.options.RigidOptions(enable_collision=False, enable_joint_limit=False),
            show_viewer=False,
            renderer=gs.renderers.Rasterizer(),
        )
        try:
            scene.add_entity(
                gs.morphs.URDF(
                    file=str(urdf_path),
                    pos=(0.0, 0.0, 0.0),
                    quat=(1.0, 0.0, 0.0, 0.0),
                    collision=False,
                    merge_fixed_links=True,
                )
            )
            camera = scene.add_camera(
                res=(800, 800),
                pos=camera_pos,
                lookat=camera_lookat,
                up=(0.0, 0.0, 1.0),
                fov=35,
                near=0.05,
                far=20.0,
                debug=True,
            )
            scene.build(n_envs=0)
            rgb, *_ = camera.render(
                rgb=True,
                depth=False,
                segmentation=False,
                normal=False,
                antialiasing=True,
                force_render=True,
            )
            rgb_np = tensor_to_array(rgb)
            if rgb_np.ndim == 4:
                rgb_np = rgb_np[0]
            rgb_np = np.flip(rgb_np, axis=-1)
            if rgb_np.shape[-1] == 4:
                rgb_np = rgb_np[..., :3]
            rgb_np = np.asarray(rgb_np)
            if rgb_np.dtype != np.uint8:
                rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
            plt.imsave(out_path, rgb_np)
        finally:
            scene.destroy()
            gs.destroy()

    def _print_minimal_threshold_counts(self):
        """Print how many URDFs exceed the minimal progress threshold per policy."""
        if len(self.df) == 0:
            print("[WARN] No URDFs available for threshold counting.")
            return

        prog_threshold_norm = self.MINIMAL_PROGRESS_M / self.PROGRESS_NORM
        gp_count = max(0, self.general_policy_count)
        gp_pass_counts = [0] * gp_count
        sp_pass_count = 0
        gp_runtime_vals = [[] for _ in range(gp_count)]
        sp_runtime_vals = []

        for _, row in self.df.iterrows():
            metrics = self._process_row(row)
            gp_prog = np.asarray(metrics["baseline_all"]["prog"], dtype=float)
            tr_prog = np.asarray(metrics["trained"]["prog"], dtype=float)
            eval_duration_s = self._safe_row_float(row, "eval_duration_s")
            train_duration_s = self._safe_row_float(row, "train_duration_s")

            for pi in range(gp_count):
                if pi >= len(gp_prog):
                    continue
                val = gp_prog[pi]
                if np.isfinite(val) and val > prog_threshold_norm:
                    gp_pass_counts[pi] += 1
                    gp_runtime = self._safe_row_float(row, f"gp_runtime_s{pi + 1}")
                    if np.isfinite(gp_runtime):
                        gp_runtime_vals[pi].append(float(gp_runtime))

            valid_mask = np.isfinite(tr_prog) & (tr_prog > 0)
            valid_vals = tr_prog[valid_mask]
            if len(valid_vals) > 0 and valid_vals.mean() > prog_threshold_norm:
                sp_pass_count += 1
                rep_count = max(1, self._count_indexed_columns(row, "f_prog_trained"))
                sp_runtime = (train_duration_s + eval_duration_s) / float(rep_count)
                if np.isfinite(sp_runtime):
                    sp_runtime_vals.append(float(sp_runtime))

        print(
            f"\n=== URDFs above minimal progress threshold "
            f"({self.MINIMAL_PROGRESS_M:.1f} m) ==="
        )
        for pi in range(gp_count):
            runtime_arr = np.asarray(gp_runtime_vals[pi], dtype=float)
            runtime_arr = runtime_arr[np.isfinite(runtime_arr)]
            runtime_mean = float(np.mean(runtime_arr)) if len(runtime_arr) > 0 else np.nan
            runtime_std = float(np.std(runtime_arr)) if len(runtime_arr) > 1 else np.nan
            print(
                f"  GP{pi + 1}: {gp_pass_counts[pi]}/{len(self.df)} | "
                f"runtime mean={self._fmt_duration(runtime_mean)} | "
                f"std={self._fmt_duration(runtime_std)}"
            )
        sp_runtime_arr = np.asarray(sp_runtime_vals, dtype=float)
        sp_runtime_arr = sp_runtime_arr[np.isfinite(sp_runtime_arr)]
        sp_runtime_mean = float(np.mean(sp_runtime_arr)) if len(sp_runtime_arr) > 0 else np.nan
        sp_runtime_std = float(np.std(sp_runtime_arr)) if len(sp_runtime_arr) > 1 else np.nan
        print(
            f"  SP: {sp_pass_count}/{len(self.df)} | "
            f"runtime mean={self._fmt_duration(sp_runtime_mean)} | "
            f"std={self._fmt_duration(sp_runtime_std)}"
        )
        print("===============================================\n")

    def _print_gp_selected_below_threshold_details(self, general_policy_idx=None):
        """Print metrics for URDFs where selected GP progress is below threshold."""
        if len(self.df) == 0:
            print("[WARN] No URDFs available for below-threshold report.")
            return

        gp_idx = self._resolve_policy_index(
            general_policy_idx if general_policy_idx is not None else self.general_policy_idx,
            self.general_policy_count,
            "General",
        )
        threshold = self._minimal_progress_threshold_norm()
        metric_order = ["prog", "speed", "cot", "reward", "steps90"]

        def _sp_mean(metrics, metric):
            vals = np.asarray(metrics["trained"][metric], dtype=float)
            mask = self._is_valid_metric(metric, vals)
            valid_vals = vals[mask]
            if len(valid_vals) == 0:
                return np.nan
            return float(np.mean(valid_vals))

        def _fmt_metric(metric, gp_val_norm, sp_val_norm):
            if metric == "prog":
                gp_raw = gp_val_norm * self.PROGRESS_NORM if np.isfinite(gp_val_norm) else np.nan
                sp_raw = sp_val_norm * self.PROGRESS_NORM if np.isfinite(sp_val_norm) else np.nan
                return (
                    f"GP={gp_raw:.2f} m ({gp_val_norm:.4f}), "
                    f"SPmean={sp_raw:.2f} m ({sp_val_norm:.4f})"
                )
            if metric == "reward":
                gp_raw = gp_val_norm * self.REWARD_NORM if np.isfinite(gp_val_norm) else np.nan
                sp_raw = sp_val_norm * self.REWARD_NORM if np.isfinite(sp_val_norm) else np.nan
                return (
                    f"GP={gp_raw:.2f} ({gp_val_norm:.4f}), "
                    f"SPmean={sp_raw:.2f} ({sp_val_norm:.4f})"
                )

        rows = []
        runtime_vals = []
        for urdf_idx, (_, row) in enumerate(self.df.iterrows(), start=1):
            metrics = self._process_row(row, general_policy_idx=gp_idx)
            gp_selected = metrics["baseline_selected"]
            gp_prog = float(gp_selected["prog"])
            if np.isfinite(gp_prog) and gp_prog > threshold:
                continue

            summary = {"urdf_idx": urdf_idx, "metrics": {}}
            for metric in metric_order:
                gp_val = float(gp_selected[metric])
                sp_val = _sp_mean(metrics, metric)
                summary["metrics"][metric] = (gp_val, sp_val)
            rows.append(summary)

        print(
            f"\n=== URDFs below minimal progress threshold for selected GP{gp_idx} "
            f"({self.MINIMAL_PROGRESS_M:.1f} m) ==="
        )
        if len(rows) == 0:
            print("  None")
            print("===============================================================\n")
            return

        for item in rows:
            urdf_idx = item["urdf_idx"]
            gp_prog, sp_prog = item["metrics"]["prog"]
            print(
                f"  URDF {urdf_idx:02d}: "
                f"{_fmt_metric('prog', gp_prog, sp_prog)}"
            )
            print(f"    reward: {_fmt_metric('reward', *item['metrics']['reward'])}")
        print("===============================================================\n")

    def _print_sp_below_threshold_details(self):
        """Print SP metrics for URDFs where mean SP progress is below threshold."""
        if len(self.df) == 0:
            print("[WARN] No URDFs available for SP below-threshold report.")
            return

        threshold_m = float(self.MINIMAL_PROGRESS_M)
        rows = []
        runtime_vals = []
        runtime_vals = []
        runtime_vals = []

        for urdf_idx, (_, row) in enumerate(self.df.iterrows(), start=1):
            metrics = self._process_row(row)

            tr_prog = np.asarray(metrics["trained"]["prog"], dtype=float) * self.PROGRESS_NORM
            tr_speed = np.asarray(metrics["trained"]["speed"], dtype=float) * self.SPEED_NORM
            tr_cot = np.asarray(metrics["trained"]["cot"], dtype=float)
            tr_reward = np.asarray(metrics["trained"]["reward"], dtype=float) * self.REWARD_NORM

            prog_mask = np.isfinite(tr_prog) & (tr_prog > 0)
            prog_valid = tr_prog[prog_mask]
            if len(prog_valid) == 0:
                continue

            prog_mean = float(np.mean(prog_valid))
            if not np.isfinite(prog_mean) or prog_mean > threshold_m:
                continue

            speed_mask = self._is_valid_metric("speed", tr_speed)
            cot_mask = self._is_valid_metric("cot", tr_cot)
            reward_mask = self._is_valid_metric("reward", tr_reward)

            speed_valid = tr_speed[speed_mask]
            cot_valid = tr_cot[cot_mask]
            reward_valid = tr_reward[reward_mask]

            rows.append(
                dict(
                    urdf_idx=urdf_idx,
                    urdf_stem=row.get("urdf_stem", ""),
                    prog_mean=prog_mean,
                    prog_reps=tr_prog,
                    speed_mean=float(np.mean(speed_valid)) if len(speed_valid) > 0 else np.nan,
                    speed_reps=tr_speed,
                    cot_mean=float(np.mean(cot_valid)) if len(cot_valid) > 0 else np.nan,
                    cot_reps=tr_cot,
                    reward_mean=float(np.mean(reward_valid)) if len(reward_valid) > 0 else np.nan,
                    reward_reps=tr_reward,
                )
            )

        print(
            f"\n=== URDFs below minimal progress threshold for SP "
            f"({threshold_m:.1f} m) ==="
        )
        if len(rows) == 0:
            print("  None")
            print("===============================================================\n")
            return

        def _fmt_scalar(val, unit=""):
            if not np.isfinite(val):
                return "nan"
            return f"{val:.2f}{unit}"

        def _fmt_reps(vals):
            return ", ".join(
                "nan" if not np.isfinite(v) else f"{float(v):.2f}"
                for v in np.asarray(vals, dtype=float)
            )

        for item in rows:
            print(
                f"  URDF {item['urdf_idx']:02d}: progress_mean={_fmt_scalar(item['prog_mean'], ' m')} | "
                f"urdf={item['urdf_stem']}"
            )
            print(f"    progress reps: {_fmt_reps(item['prog_reps'])}")
            print(
                f"    speed mean: {_fmt_scalar(item['speed_mean'])} | "
                f"speed reps: {_fmt_reps(item['speed_reps'])}"
            )
            print(
                f"    cot mean: {_fmt_scalar(item['cot_mean'])} | "
                f"cot reps: {_fmt_reps(item['cot_reps'])}"
            )
            print(
                f"    reward mean: {_fmt_scalar(item['reward_mean'])} | "
                f"reward reps: {_fmt_reps(item['reward_reps'])}"
            )
        print("===============================================================\n")

    def _print_sp_steps_for_above_threshold(self):
        """Print SP steps@target reward for URDFs above minimal progress."""
        if len(self.df) == 0:
            print("[WARN] No URDFs available for SP steps report.")
            return

        threshold_m = float(self.MINIMAL_PROGRESS_M)
        rows = []
        runtime_vals = []

        for urdf_idx, (_, row) in enumerate(self.df.iterrows(), start=1):
            metrics = self._process_row(row)

            tr_prog = np.asarray(metrics["trained"]["prog"], dtype=float) * self.PROGRESS_NORM
            tr_steps = np.asarray(metrics["trained"]["steps90"], dtype=float)

            prog_mask = np.isfinite(tr_prog) & (tr_prog > 0)
            prog_valid = tr_prog[prog_mask]
            if len(prog_valid) == 0:
                continue

            prog_mean = float(np.mean(prog_valid))
            if not np.isfinite(prog_mean) or prog_mean <= threshold_m:
                continue

            steps_mask = self._is_valid_metric("steps90", tr_steps)
            steps_valid = tr_steps[steps_mask]
            if len(steps_valid) == 0:
                continue

            rep_count = max(1, self._count_indexed_columns(row, "f_prog_trained"))
            runtime_s = (
                self._safe_row_float(row, "train_duration_s")
                + self._safe_row_float(row, "eval_duration_s")
            ) / float(rep_count)
            if np.isfinite(runtime_s):
                runtime_vals.append(float(runtime_s))

            rows.append(
                dict(
                    urdf_idx=urdf_idx,
                    urdf_stem=row.get("urdf_stem", ""),
                    prog_mean=prog_mean,
                    steps_mean=float(np.mean(steps_valid)),
                    steps_reps=tr_steps,
                    runtime_s=runtime_s,
                )
            )

        print(
            "\n=== SP steps to reach "
            f"{self._steps_target_pct_label()} reward for URDFs above minimal progress "
            f"({threshold_m:.1f} m) ==="
        )
        print(
            "  Note: value reported as fraction of total training steps "
            "(the same quantity used in the plots)."
        )
        if len(rows) == 0:
            print("  None")
            print("===============================================================\n")
            return

        def _fmt_ratio(val):
            if not np.isfinite(val):
                return "nan"
            return f"{val:.4f} ({val * 100.0:.2f}%)"

        def _fmt_reps(vals):
            return ", ".join(
                "nan" if not np.isfinite(v) else f"{float(v):.4f} ({float(v) * 100.0:.2f}%)"
                for v in np.asarray(vals, dtype=float)
            )

        for item in rows:
            print(
                f"  URDF {item['urdf_idx']:02d}: progress_mean={item['prog_mean']:.2f} m | "
                f"steps_mean={_fmt_ratio(item['steps_mean'])} | "
                f"runtime={self._fmt_duration(item['runtime_s'])} | urdf={item['urdf_stem']}"
            )
            print(f"    steps reps: {_fmt_reps(item['steps_reps'])}")

        all_steps_means = np.asarray([item["steps_mean"] for item in rows], dtype=float)
        valid_global = all_steps_means[np.isfinite(all_steps_means) & (all_steps_means > 0)]
        global_mean = float(np.mean(valid_global)) if len(valid_global) > 0 else np.nan
        global_std = float(np.std(valid_global)) if len(valid_global) > 1 else np.nan
        runtime_arr = np.asarray(runtime_vals, dtype=float)
        runtime_arr = runtime_arr[np.isfinite(runtime_arr) & (runtime_arr > 0)]
        runtime_mean = float(np.mean(runtime_arr)) if len(runtime_arr) > 0 else np.nan
        runtime_std = float(np.std(runtime_arr)) if len(runtime_arr) > 1 else np.nan
        print(
            "  SP mean over URDFs above minimal progress only: "
            f"{_fmt_ratio(global_mean)} | std={_fmt_ratio(global_std)} "
            f"| runtime mean={self._fmt_duration(runtime_mean)} "
            f"| runtime std={self._fmt_duration(runtime_std)} "
            f"(n = {len(valid_global)})"
        )
        print("===============================================================\n")

    def _print_above_threshold_metric_summaries(self, general_policy_idx=None):
        """Print global summaries above minimal progress for steps and reward."""
        if len(self.df) == 0:
            print("[WARN] No URDFs available for above-threshold summaries.")
            return

        gp_idx = self._resolve_policy_index(
            general_policy_idx if general_policy_idx is not None else self.general_policy_idx,
            self.general_policy_count,
            "General",
        )
        threshold_m = float(self.MINIMAL_PROGRESS_M)
        gp_steps_vals = []
        sp_steps_vals = []
        gp_reward_vals = []
        sp_reward_vals = []

        for _, row in self.df.iterrows():
            metrics = self._process_row(row, general_policy_idx=gp_idx)

            gp_prog = float(metrics["baseline_selected"]["prog"]) * self.PROGRESS_NORM
            if np.isfinite(gp_prog) and gp_prog > threshold_m:
                gp_steps = float(metrics["baseline_selected"]["steps90"])
                gp_reward = float(metrics["baseline_selected"]["reward"]) * self.REWARD_NORM
                if self._is_valid_metric("steps90", np.asarray([gp_steps], dtype=float))[0]:
                    gp_steps_vals.append(gp_steps)
                if self._is_valid_metric("reward", np.asarray([gp_reward], dtype=float))[0]:
                    gp_reward_vals.append(gp_reward)

            tr_prog = np.asarray(metrics["trained"]["prog"], dtype=float) * self.PROGRESS_NORM
            tr_prog_valid = tr_prog[np.isfinite(tr_prog) & (tr_prog > 0)]
            if len(tr_prog_valid) == 0:
                continue
            tr_prog_mean = float(np.mean(tr_prog_valid))
            if not np.isfinite(tr_prog_mean) or tr_prog_mean <= threshold_m:
                continue

            tr_steps = np.asarray(metrics["trained"]["steps90"], dtype=float)
            tr_steps_valid = tr_steps[self._is_valid_metric("steps90", tr_steps)]
            if len(tr_steps_valid) > 0:
                sp_steps_vals.append(float(np.mean(tr_steps_valid)))

            tr_reward = np.asarray(metrics["trained"]["reward"], dtype=float) * self.REWARD_NORM
            tr_reward_valid = tr_reward[self._is_valid_metric("reward", tr_reward)]
            if len(tr_reward_valid) > 0:
                sp_reward_vals.append(float(np.mean(tr_reward_valid)))

        def _fmt_ratio_summary(vals):
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if len(arr) == 0:
                return "nan"
            mean = float(np.mean(arr))
            std = float(np.std(arr)) if len(arr) > 1 else np.nan
            return (
                f"{mean:.4f} ({mean * 100.0:.2f}%) | "
                f"std={std:.4f} ({std * 100.0:.2f}%) | n = {len(arr)}"
            )

        def _fmt_scalar_summary(vals):
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                return "nan"
            mean = float(np.mean(arr))
            std = float(np.std(arr)) if len(arr) > 1 else np.nan
            return f"{mean:.4f} | std={std:.4f} | n = {len(arr)}"

        print(
            f"\n=== Above-threshold summaries ({threshold_m:.1f} m minimal progress) ==="
        )
        print(
            f"  GP{gp_idx} steps@{self._steps_target_pct_label()}: "
            f"{_fmt_ratio_summary(gp_steps_vals)}"
        )
        print(
            f"  SP mean steps@{self._steps_target_pct_label()}: "
            f"{_fmt_ratio_summary(sp_steps_vals)}"
        )
        print(
            f"  GP{gp_idx} evaluation reward: "
            f"{_fmt_scalar_summary(gp_reward_vals)}"
        )
        print(
            "  SP mean evaluation reward: "
            f"{_fmt_scalar_summary(sp_reward_vals)}"
        )
        print("===============================================================\n")

    def _compute_mean_delta_ratio(self, numerators, denominators):
        """Compute aggregated deviation as sum(num) / sum(den)."""
        num = np.asarray(numerators, dtype=float)
        den = np.asarray(denominators, dtype=float)
        mask = np.isfinite(num) & np.isfinite(den) & (den > 0)
        if not mask.any():
            return np.nan
        den_sum = float(np.sum(den[mask]))
        if not np.isfinite(den_sum) or den_sum <= 0:
            return np.nan
        return float(np.sum(num[mask]) / den_sum)

    def _compute_weighted_delta_std(self, ratios, weights):
        """Compute weighted std of ratios with weights equal to SP means."""
        r = np.asarray(ratios, dtype=float)
        w = np.asarray(weights, dtype=float)
        mask = np.isfinite(r) & np.isfinite(w) & (w > 0)
        if not mask.any():
            return np.nan
        r = r[mask]
        w = w[mask]
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            return np.nan
        mu = float(np.sum(w * r) / w_sum)
        var = float(np.sum(w * (r - mu) ** 2) / w_sum)
        if not np.isfinite(var) or var < 0:
            return np.nan
        return float(np.sqrt(var))

    def _compute_gp_delta_vs_trained(self, metric_order=None):
        """Compute GP-vs-SP deviations used by mean-delta plots.

        For each metric and policy, stores:
          - ratios per URDF: (GP - SP_mean_urdf) / SP_mean_urdf
          - numerators per URDF: (GP - SP_mean_urdf)
          - denominators per URDF: SP_mean_urdf
          - urdf indices per value (1-based, aligned with ratios)
        """
        if metric_order is None:
            metric_order = ["speed", "cot", "prog", "reward", "steps90"]

        gp_count = max(1, self.general_policy_count)
        gp_delta_vals = {m: [[] for _ in range(gp_count)] for m in metric_order}
        gp_delta_num = {m: [[] for _ in range(gp_count)] for m in metric_order}
        gp_delta_den = {m: [[] for _ in range(gp_count)] for m in metric_order}
        gp_delta_urdf_idx = {m: [[] for _ in range(gp_count)] for m in metric_order}

        for urdf_idx, (_, row) in enumerate(self.df.iterrows(), start=1):
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
                        numerator = float(val - tr_mean_urdf)
                        denominator = float(tr_mean_urdf)
                        gp_delta_vals[m][pi].append(numerator / denominator)
                        gp_delta_num[m][pi].append(numerator)
                        gp_delta_den[m][pi].append(denominator)
                        gp_delta_urdf_idx[m][pi].append(urdf_idx)

        return {
            "ratios": gp_delta_vals,
            "numerators": gp_delta_num,
            "denominators": gp_delta_den,
            "urdf_indices": gp_delta_urdf_idx,
        }

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
            f"vs steps@{self._steps_target_pct_label()} reward (SP) ==="
        )
        print(
            "Deviation used: "
            "delta = (GP_selected - mean(SP_valid_repetitions)) / mean(SP_valid_repetitions)"
        )
        print(
            f"Steps variable: mean(steps@{self._steps_target_pct_label()}_SP_valid_repetitions) per URDF."
        )
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
        if re.search(r"evaluation_(?:general_)?SP\.csv$|evaluation_results_SP\.csv$", str(target_csv)):
            parsed = self._synthesize_missing_sp_agg_rows(parsed)
        return parsed

    def _resolve_sp_csv_path(self, csv_path):
        if csv_path and os.path.exists(csv_path):
            return csv_path
        base_dir = Path(__file__).resolve().parent
        preferred = base_dir / "evaluation_general_SP.csv"
        if preferred.exists():
            return str(preferred)
        fallback = base_dir / "evaluation_results_SP.csv"
        if fallback.exists():
            return str(fallback)
        return csv_path

    def _parse_gp_group_index(self, path_str):
        stem = Path(path_str).stem
        match = re.match(r"^evaluation_results_GP(\d+)(?:\D.*)?$", stem)
        if match:
            return int(match.group(1))
        return None

    def _gp_sort_key(self, path_str):
        explicit_idx = self._parse_gp_group_index(path_str)
        if explicit_idx is not None:
            return (0, explicit_idx, Path(path_str).name)
        return (1, 10_000, Path(path_str).name)

    def _resolve_gp_csv_paths(self, gp_csv_paths):
        if gp_csv_paths is not None:
            if isinstance(gp_csv_paths, (str, Path)):
                raw = [str(gp_csv_paths)]
            else:
                raw = [str(p) for p in gp_csv_paths if p]
        else:
            base_dir = Path(__file__).resolve().parent
            raw = [str(p) for p in base_dir.glob("evaluation_results_GP*.csv")]
        existing = []
        for p in raw:
            if os.path.exists(p):
                existing.append(p)
        existing = sorted(set(existing), key=self._gp_sort_key)
        return existing

    def _group_gp_csv_paths(self, gp_csv_paths):
        if not gp_csv_paths:
            return []

        grouped = {}
        sequential = []
        for path_str in gp_csv_paths:
            explicit_idx = self._parse_gp_group_index(path_str)
            if explicit_idx is None:
                sequential.append(
                    {
                        "policy_idx_hint": None,
                        "paths": [path_str],
                        "mode": "per_column",
                        "label": Path(path_str).stem,
                    }
                )
                continue
            grouped.setdefault(explicit_idx, []).append(path_str)

        out = []
        for explicit_idx in sorted(grouped):
            out.append(
                {
                    "policy_idx_hint": explicit_idx,
                    "paths": sorted(grouped[explicit_idx], key=self._gp_sort_key),
                    "mode": "aggregate",
                    "label": f"GP{explicit_idx}",
                }
            )
        out.extend(sequential)
        return out

    def _extract_gp_policy_value(self, row, prefix, policy_idx):
        col = f"{prefix}{policy_idx}"
        if col not in row.index:
            return np.nan
        try:
            val = float(row[col])
        except (TypeError, ValueError):
            return np.nan
        filtered = self._filter_gp_runs_like_sp(prefix, np.asarray([val], dtype=float))
        if len(filtered) == 0:
            return np.nan
        return float(filtered[0])

    def _extract_prefixed_series(self, row, prefix):
        values = []
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        indexed_cols = []
        for col in row.index:
            if not isinstance(col, str):
                continue
            match = pattern.match(col)
            if match:
                indexed_cols.append((int(match.group(1)), col))
        indexed_cols.sort(key=lambda x: x[0])
        for _, col in indexed_cols:
            try:
                values.append(float(row[col]))
            except (TypeError, ValueError):
                values.append(np.nan)
        arr = np.asarray(values, dtype=float)
        return arr[np.isfinite(arr)]

    def _filter_gp_runs_like_sp(self, prefix, raw_values):
        """Apply to GP runs the same validity logic used for SP metrics."""
        vals = np.asarray(raw_values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return vals

        if prefix == "f_speed_baseline":
            return vals[vals > 0]

        if prefix == "f_negE_baseline":
            cot_vals = self._normalize_cot(vals)
            valid_mask = self._is_valid_metric("cot", cot_vals)
            return vals[valid_mask]

        if prefix == "f_prog_baseline":
            valid_mask = self._is_valid_metric("prog", vals)
            return vals[valid_mask]

        if prefix == "reward_ep_mean_baseline":
            valid_mask = self._is_valid_metric("reward", vals)
            return vals[valid_mask]

        return vals

    def _aggregate_gp_runs_by_row(self, gp_df):
        metric_prefixes = [
            "f_speed_baseline",
            "f_negE_baseline",
            "f_prog_baseline",
            "reward_ep_mean_baseline",
        ]
        rows = []
        for _, row in gp_df.iterrows():
            out = {
                "eval_duration_s": self._safe_row_float(row, "eval_duration_s"),
                "urdf_key": self._canonical_urdf_key(row),
                "urdf_array": self._numeric_urdf_array(row),
                "metrics": {},
            }
            for prefix in metric_prefixes:
                vals = self._extract_prefixed_series(row, prefix)
                vals = self._filter_gp_runs_like_sp(prefix, vals)
                if len(vals) == 0:
                    mean_val = np.nan
                    std_val = np.nan
                else:
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals)) if len(vals) > 1 else 0.0
                out["metrics"][prefix] = {"series": vals, "mean": mean_val, "std": std_val}
            rows.append(out)
        return rows

    def _match_gp_entry(self, row, gp_rows_by_key, gp_rows_with_arrays):
        key = self._canonical_urdf_key(row)
        if key is not None and key in gp_rows_by_key:
            return gp_rows_by_key[key], False

        target = self._numeric_urdf_array(row)
        if target is None:
            return None, False

        required_overlap = max(3, int(np.ceil(0.7 * len(target))))
        best = None
        best_overlap = -1
        best_dist = np.inf
        for entry in gp_rows_with_arrays:
            gp_arr = entry["urdf_array"]
            if gp_arr is None or len(gp_arr) != len(target):
                continue
            overlap = int(np.sum(np.isclose(target, gp_arr, atol=1e-8, rtol=0.0)))
            if overlap < required_overlap:
                continue
            dist = float(np.linalg.norm(target - gp_arr))
            if overlap > best_overlap or (overlap == best_overlap and dist < best_dist):
                best = entry
                best_overlap = overlap
                best_dist = dist
        return best, best is not None

    def _merge_gp_baselines_from_csvs(self):
        if len(self.df_all) == 0 or len(self.gp_csv_paths) == 0:
            return

        metric_prefixes = [
            "f_speed_baseline",
            "f_negE_baseline",
            "f_prog_baseline",
            "reward_ep_mean_baseline",
        ]
        loaded_policies = 0
        total_filled = 0

        for group in self._group_gp_csv_paths(self.gp_csv_paths):
            if group.get("mode") == "per_column":
                gp_path = group["paths"][0]
                gp_df_all = self._load_df_with_genome(gp_path)
                gp_df = self._filter_agg_rows(gp_df_all).copy()
                if len(gp_df) == 0:
                    continue

                gp_rows_by_key = {}
                gp_rows_with_arrays = []
                for _, gp_row in gp_df.iterrows():
                    entry = {
                        "row": gp_row,
                        "urdf_key": self._canonical_urdf_key(gp_row),
                        "urdf_array": self._numeric_urdf_array(gp_row),
                    }
                    if entry["urdf_key"] is not None:
                        gp_rows_by_key[entry["urdf_key"]] = entry
                    if entry["urdf_array"] is not None:
                        gp_rows_with_arrays.append(entry)
                if not gp_rows_by_key and not gp_rows_with_arrays:
                    continue

                per_file_policy_count = self._infer_policy_count_from_df(gp_df, "baseline")
                if per_file_policy_count <= 0:
                    continue

                for source_policy_idx in range(1, per_file_policy_count + 1):
                    loaded_policies += 1
                    policy_idx = loaded_policies
                    runtime_col = f"gp_runtime_s{policy_idx}"
                    for prefix in metric_prefixes:
                        mean_col = f"{prefix}{policy_idx}"
                        std_col = f"{prefix}_std{policy_idx}"
                        if mean_col not in self.df_all.columns:
                            self.df_all[mean_col] = np.nan
                        if std_col not in self.df_all.columns:
                            self.df_all[std_col] = np.nan
                    if runtime_col not in self.df_all.columns:
                        self.df_all[runtime_col] = np.nan

                    matched_rows = 0
                    overlap_matches = 0
                    filled = 0

                    for idx, row in self.df_all.iterrows():
                        match, used_overlap = self._match_gp_entry(
                            row,
                            gp_rows_by_key,
                            gp_rows_with_arrays,
                        )
                        if match is None:
                            continue
                        matched_rows += 1
                        if used_overlap:
                            overlap_matches += 1

                        source_row = match["row"]
                        eval_runtime_s = self._safe_row_float(source_row, "eval_duration_s")
                        if np.isfinite(eval_runtime_s):
                            self.df_all.at[idx, runtime_col] = eval_runtime_s / float(max(1, per_file_policy_count))
                        for prefix in metric_prefixes:
                            mean_col = f"{prefix}{policy_idx}"
                            std_col = f"{prefix}_std{policy_idx}"
                            mean_val = self._extract_gp_policy_value(
                                source_row,
                                prefix,
                                source_policy_idx,
                            )
                            if np.isfinite(mean_val):
                                self.df_all.at[idx, mean_col] = mean_val
                                filled += 1
                            self.df_all.at[idx, std_col] = np.nan

                    total_filled += filled
                    print(
                        f"[INFO] Loaded GP policy #{policy_idx} from '{Path(gp_path).name}' "
                        f"(source column baseline{source_policy_idx}): "
                        f"matched rows={matched_rows}, overlap matches={overlap_matches}, "
                        f"filled values={filled}."
                    )
                    self._gp_runtime_divisor_by_policy_idx[policy_idx] = max(1, per_file_policy_count)
                continue

            group_matchers = []
            source_files = []

            for gp_path in group["paths"]:
                gp_df_all = self._load_df_with_genome(gp_path)
                gp_df = self._filter_agg_rows(gp_df_all).copy()
                if len(gp_df) == 0:
                    continue

                gp_entries = self._aggregate_gp_runs_by_row(gp_df)
                gp_rows_by_key = {}
                gp_rows_with_arrays = []
                for entry in gp_entries:
                    key = entry["urdf_key"]
                    if key is not None:
                        gp_rows_by_key[key] = entry
                    if entry["urdf_array"] is not None:
                        gp_rows_with_arrays.append(entry)
                if not gp_rows_by_key and not gp_rows_with_arrays:
                    continue
                group_matchers.append((gp_rows_by_key, gp_rows_with_arrays))
                source_files.append(gp_path)

            if not group_matchers:
                continue

            loaded_policies += 1
            policy_idx = loaded_policies
            runtime_col = f"gp_runtime_s{policy_idx}"
            for prefix in metric_prefixes:
                mean_col = f"{prefix}{policy_idx}"
                std_col = f"{prefix}_std{policy_idx}"
                if mean_col not in self.df_all.columns:
                    self.df_all[mean_col] = np.nan
                if std_col not in self.df_all.columns:
                    self.df_all[std_col] = np.nan
            if runtime_col not in self.df_all.columns:
                self.df_all[runtime_col] = np.nan

            matched_rows = 0
            overlap_matches = 0
            filled = 0

            for idx, row in self.df_all.iterrows():
                collected = {prefix: [] for prefix in metric_prefixes}
                runtime_series = []
                row_matched = False

                for gp_rows_by_key, gp_rows_with_arrays in group_matchers:
                    match, used_overlap = self._match_gp_entry(
                        row,
                        gp_rows_by_key,
                        gp_rows_with_arrays,
                    )
                    if match is None:
                        continue
                    row_matched = True
                    if used_overlap:
                        overlap_matches += 1
                    eval_runtime_s = float(match.get("eval_duration_s", np.nan))
                    if np.isfinite(eval_runtime_s):
                        runtime_series.append(float(eval_runtime_s))
                    for prefix in metric_prefixes:
                        series = np.asarray(match["metrics"][prefix].get("series", []), dtype=float)
                        series = series[np.isfinite(series)]
                        if len(series) > 0:
                            collected[prefix].append(series)

                if not row_matched:
                    continue
                matched_rows += 1
                if runtime_series:
                    self.df_all.at[idx, runtime_col] = float(np.mean(runtime_series))

                for prefix in metric_prefixes:
                    mean_col = f"{prefix}{policy_idx}"
                    std_col = f"{prefix}_std{policy_idx}"
                    if collected[prefix]:
                        merged = np.concatenate(collected[prefix])
                        mean_val = float(np.mean(merged))
                        std_val = float(np.std(merged)) if len(merged) > 1 else 0.0
                    else:
                        mean_val = np.nan
                        std_val = np.nan
                    if np.isfinite(mean_val):
                        self.df_all.at[idx, mean_col] = mean_val
                        filled += 1
                    if np.isfinite(std_val):
                        self.df_all.at[idx, std_col] = std_val

            total_filled += filled
            print(
                f"[INFO] Loaded GP policy #{policy_idx} from {len(source_files)} file(s) "
                f"({', '.join(Path(p).name for p in source_files)}): "
                f"matched rows={matched_rows}, overlap matches={overlap_matches}, "
                f"filled values={filled}."
            )
            self._gp_runtime_divisor_by_policy_idx[policy_idx] = 1

        self._gp_policy_count_from_csvs = loaded_policies
        if loaded_policies > 0:
            print(
                f"[INFO] GP merge complete: {loaded_policies} policy files loaded, "
                f"total filled baseline values={total_filled}."
            )

    def _extract_metric_std_values(self, row, prefix, count):
        values = []
        for i in range(1, count + 1):
            col = f"{prefix}_std{i}"
            values.append(row[col] if col in row.index else np.nan)
        return np.array(values, dtype=float)

    def _normalize_urdf_indices(self, urdf_indices):
        if urdf_indices is None:
            return None
        out = set()
        for idx in urdf_indices:
            try:
                out.add(int(idx))
            except (TypeError, ValueError):
                continue
        return out or None

    def _count_indexed_columns(self, row, prefix):
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        count = 0
        for col in row.index:
            if isinstance(col, str) and pattern.match(col):
                count += 1
        return count

    def _safe_row_float(self, row, col, default=np.nan):
        if col not in row.index:
            return float(default)
        try:
            return float(row[col])
        except (TypeError, ValueError):
            return float(default)

    def _fmt_duration(self, seconds):
        if not np.isfinite(seconds):
            return "nan"
        minutes = float(seconds) / 60.0
        return f"{float(seconds):.2f}s ({minutes:.2f} min)"

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
        gp_speed_std = self._extract_metric_std_values(
            row, "f_speed_baseline", self.general_policy_count
        )
        gp_cot_std = self._extract_metric_std_values(
            row, "f_negE_baseline", self.general_policy_count
        )
        gp_prog_std = self._extract_metric_std_values(
            row, "f_prog_baseline", self.general_policy_count
        )
        gp_reward_std = self._extract_metric_std_values(
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
                "speed": gp_speed / self.SPEED_NORM,
                "cot": self._normalize_cot(gp_cot) / 1,
                "prog": gp_prog / self.PROGRESS_NORM,
                "reward": gp_reward / self.REWARD_NORM,
                "steps90": gp_steps90,
            },
            "baseline_all_std": {
                "speed": gp_speed_std / self.SPEED_NORM,
                "cot": np.abs(gp_cot_std) / 1,
                "prog": gp_prog_std / self.PROGRESS_NORM,
                "reward": gp_reward_std / self.REWARD_NORM,
                "steps90": np.full(self.general_policy_count, np.nan, dtype=float),
            },
            "trained": {
                "speed": tr_speed / self.SPEED_NORM,
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
        metrics["baseline_selected_std"] = {
            m: self._select_policy_value(metrics["baseline_all_std"][m], gp_idx)
            for m in metrics["baseline_all_std"]
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
                out_dir = self._urdf_output_dir(save_dir)
                filename = f"urdf_{row_idx:02d}.png"
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
        title = "Performance Metrics Mean — General Policies vs SP (mean±std, min-max)"
        if only_above_minimal_progress:
            if progress_threshold_m is None:
                title += " — Above Minimal Progress Only"
            else:
                title += f" — Above {threshold_m:.0f} m Progress Only"
        ax.set_title(title, fontsize=12)

        legend_labels = [f"GP{i} mean±std + min/max" for i in range(1, gp_count + 1)] + ["SP mean±std + min/max"]
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
        metric_order = ["speed", "cot", "prog", "reward", "steps90"]
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

        fig, ax = plt.subplots(figsize=(14.5, 5.4))
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
        sp_label_suffix = "SP All Seeds" if sp_use_all_repetitions else "SP Mean per URDF"
        ax.set_title(
            f"Performance Distributions per URDF — GP{gp_idx} vs {sp_label_suffix} — All Valid URDFs",
            fontsize=12,
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
            f"GP{gp_idx} distribution (n={gp_n})",
            f"{sp_label_suffix} distribution (n={sp_n})",
            "Mean",
            "Median",
            f"Progress threshold reference ({threshold_m:.0f} m)",
        ]
        ax.legend(legend_handles, legend_labels, title="Distributions", fontsize=9)

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
                out_dir = self._urdf_output_dir(save_dir)
                filename = f"urdf_{row_idx:02d}_reward_evolution.png"
                fig.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")

            if show:
                plt.show()
            plt.close(fig)

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
            for gi in range(n_genes):
                x = genome_used[:, gi]
                for mi, metric in enumerate(metric_order):
                    y = delta_used[metric].to_numpy(dtype=float)
                    mask = np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 2:
                        continue
                    corr[gi, mi] = np.corrcoef(x[mask], y[mask])[0, 1]

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
                "Note: only |r| >= 0.20 is annotated (lower values are not shown).",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#555555",
            )

            if n_genes <= 25:
                for i in range(n_genes):
                    for j in range(len(metric_order)):
                        if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= 0.20:
                            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Pearson r")

            plt.tight_layout()

            if save_dir is not None:
                out_dir = outlier_dirs[version_cfg["key"]]
                fig.savefig(os.path.join(out_dir, base_filename), dpi=300, bbox_inches="tight")
            elif save_path is not None:
                version_suffix = "_outlier_no" if filter_no_outliers else "_outlier_yes"
                fig.savefig(
                    f"{base_root}{version_suffix}{base_ext}",
                    dpi=300,
                    bbox_inches="tight",
                )
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
            "Note: only |r| >= 0.20 is annotated (lower values are not shown).",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )

        if n_genes <= 25:
            for i in range(n_genes):
                for j in range(len(metric_order)):
                    if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= 0.20:
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
            "Note: only |r| >= 0.20 is annotated (lower values are not shown).",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )

        if n_genes <= 25:
            for i in range(n_genes):
                for j in range(len(metric_order)):
                    if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= 0.20:
                        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
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
        counts = np.zeros((len(policy_series), len(metric_order)), dtype=int)

        for ri, (_, x_by_metric, y_reward) in enumerate(policy_series):
            y = np.asarray(y_reward, dtype=float)
            for ci, metric in enumerate(metric_order):
                x = np.asarray(x_by_metric[metric], dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                n = int(mask.sum())
                counts[ri, ci] = n
                if n < 2:
                    continue
                if np.isclose(np.std(x[mask]), 0.0) or np.isclose(np.std(y[mask]), 0.0):
                    continue
                corr[ri, ci] = float(np.corrcoef(x[mask], y[mask])[0, 1])

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
                if np.isfinite(val):
                    txt = f"{val:.2f}\n(n={n})"
                else:
                    txt = f"nan\n(n={n})"
                ax.text(ci, ri, txt, ha="center", va="center", fontsize=7)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
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
            (f"GP{gp_idx} selected", gp_x_by_metric, gp_y_reward),
            ("SP mean", sp_x_by_metric, sp_y_reward),
        ]

        corr = np.full((len(policy_series), len(metric_order)), np.nan, dtype=float)
        counts = np.zeros((len(policy_series), len(metric_order)), dtype=int)

        for ri, (_, x_by_metric, y_reward) in enumerate(policy_series):
            y = np.asarray(y_reward, dtype=float)
            for ci, metric in enumerate(metric_order):
                x = np.asarray(x_by_metric[metric], dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                n = int(mask.sum())
                counts[ri, ci] = n
                if n < 2:
                    continue
                if np.isclose(np.std(x[mask]), 0.0) or np.isclose(np.std(y[mask]), 0.0):
                    continue
                corr[ri, ci] = float(np.corrcoef(x[mask], y[mask])[0, 1])

        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")

        ax.set_xticks(range(len(metric_order)))
        ax.set_xticklabels([metric_names[m] for m in metric_order], rotation=0)
        ax.set_yticks(range(len(policy_series)))
        ax.set_yticklabels([name for name, _, _ in policy_series])
        ax.set_xlabel("Metric")
        ax.set_ylabel("Policy")
        ax.set_title(
            f"Correlation across URDFs (filtered): GP{gp_idx} selected vs SP mean",
            fontsize=12,
        )

        for ri in range(len(policy_series)):
            for ci in range(len(metric_order)):
                val = corr[ri, ci]
                n = counts[ri, ci]
                if np.isfinite(val):
                    txt = f"{val:.2f}\n(n={n})"
                else:
                    txt = f"nan\n(n={n})"
                ax.text(ci, ri, txt, ha="center", va="center", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")
        fig.text(
            0.5,
            0.01,
            (
                "Included URDFs: only those with Progress above minimal threshold "
                f"for both GP{gp_idx} and SP mean."
            ),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

        plt.tight_layout(rect=(0, 0.04, 1, 1))

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
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
        for metric in metric_order:
            x = np.asarray([r[metric] for r in rows], dtype=float)
            y = np.asarray([r["steps90"] for r in rows], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            n = len(x)
            if n >= 2 and not np.isclose(np.std(x), 0.0) and not np.isclose(np.std(y), 0.0):
                pearson_r = float(np.corrcoef(x, y)[0, 1])
            else:
                pearson_r = np.nan
            corr_rows.append((metric, pearson_r, n))

        fig_h = max(3.8, 1.8 + 0.65 * len(corr_rows))
        fig, ax = plt.subplots(figsize=(9.2, fig_h))
        ax.axis("off")

        col_labels = [
            "Metric",
            "Pearson r",
            "N URDF",
            f"Steps to {threshold_pct} reward max (SP)",
        ]
        cell_text = []
        for metric, r_val, n in corr_rows:
            r_txt = f"{r_val:+.3f}" if np.isfinite(r_val) else "nan"
            interpretation = "N/A"
            if np.isfinite(r_val):
                if abs(r_val) < 0.1:
                    interpretation = "Very weak"
                elif r_val > 0:
                    interpretation = "Positive"
                else:
                    interpretation = "Negative"
            cell_text.append([metric_names[metric], r_txt, str(n), interpretation])

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

        for m, label in metrics_to_plot:
            x_all = np.asarray(pairs[m]["x"], dtype=float)
            y_all = np.asarray(pairs[m]["y"], dtype=float)
            num_all = np.asarray(pairs[m]["num"], dtype=float)
            den_all = np.asarray(pairs[m]["den"], dtype=float)

            versions = [{"name": "with_fit_mean", "filter_y_le_100": False}]
            if m == "prog":
                versions = [
                    {"name": "points_only", "filter_y_le_100": False},
                    {"name": "with_fit_mean", "filter_y_le_100": False},
                ]
            elif m == "reward":
                versions = [
                    {"name": "points_only", "filter_y_le_100": False},
                    {"name": "with_fit_mean", "filter_y_le_100": False},
                ]
            if m in ("prog", "reward"):
                if m == "prog":
                    versions.extend(
                        [
                            {"name": "points_only", "filter_y_le_100": True},
                            {"name": "with_fit_mean", "filter_y_le_100": True},
                        ]
                    )
                else:
                    versions.extend(
                        [
                            {"name": "points_only", "filter_y_le_100": True},
                            {"name": "with_fit_mean", "filter_y_le_100": True},
                        ]
                    )

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

                fig, ax = plt.subplots(figsize=(8.8, 4.8))
                use_minimal_progress_filter = m in ("speed", "cot")
                if len(x) > 0:
                    if use_minimal_progress_filter:
                        point_label = (
                            f"Valid drones (GP{gp_idx} & SP above minimal progress)"
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
                    f"SP Steps to Learn (Steps to {self._steps_target_pct_label()} Reward / Total Training Steps)"
                )
                ax.set_ylabel(
                    f"Performance difference GP{gp_idx} vs SP ({label}, %)\n"
                    r"$100 \cdot \frac{\mathrm{GP}-\mathrm{SP}}{\mathrm{SP}}$"
                )
                title_suffix = ""
                if filter_y_le_100:
                    title_suffix = " (filtered: y<=100%)"
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
                                    f"(sum delta / sum SP): {mean_perf_diff_pct:+.2f}%"
                                ),
                            )
                if len(x) > 0:
                    ax.legend(fontsize=8, loc="best")
                fig.text(
                    0.5,
                    0.002,
                    (
                        f"Included drones: only those above minimal progress in both GP{gp_idx} and SP."
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
                    if m == "prog":
                        suffix = "_y_le_100" if filter_y_le_100 else ""
                        filename = f"gp{gp_idx}_sp_performance_difference_vs_sp_steps_{m}_{version}{suffix}.png"
                    else:
                        suffix = "_y_le_100" if filter_y_le_100 else ""
                        if version != "with_fit_mean":
                            filename = f"gp{gp_idx}_sp_performance_difference_vs_sp_steps_{m}_{version}{suffix}.png"
                        else:
                            filename = f"gp{gp_idx}_sp_performance_difference_vs_sp_steps_{m}{suffix}.png"
                    target_dir = outlier_dirs["yes"]
                    if m in ("prog", "reward") and filter_y_le_100:
                        target_dir = outlier_dirs["no"]
                    fig.savefig(os.path.join(target_dir, filename), dpi=300, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-processing and plotting for GP/SP evaluation.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/andrea/Documents/Genesis/src/data_processing",
        help="Directory containing evaluation_results_*.csv files.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <data-dir>/evaluation_plots).",
    )
    parser.add_argument(
        "--gp-idx",
        type=int,
        default=2,
        help="General policy index used by selected-vs-SP plots.",
    )
    parser.add_argument(
        "--drone-idx",
        type=int,
        default=None,
        help="1-based URDF index. If set, generate only urdf_XX_picture.png and exit.",
    )
    parser.add_argument(
        "--urdf-idxs",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional 1-based URDF indices for per-URDF outputs. "
            "If omitted, no per-URDF plots or drone pictures are generated."
        ),
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    csv_path = os.path.join(data_dir, "evaluation_general_SP.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_dir, "evaluation_results_SP.csv")
    gp_csv_paths = sorted(
        [
            os.path.join(data_dir, name)
            for name in os.listdir(data_dir)
            if re.match(r"^evaluation_results_GP.*\.csv$", name)
        ]
    )
    plotter = URDFHistogramPlotter(csv_path, gp_csv_paths=gp_csv_paths)
    requested_general_policy_idx = int(args.gp_idx)
    general_policy_idx = plotter._resolve_policy_index(
        requested_general_policy_idx,
        plotter.general_policy_count,
        "General",
    )
    save_dir = (
        os.path.abspath(args.save_dir)
        if args.save_dir is not None
        else os.path.join(data_dir, "evaluation_plots")
    )
    os.makedirs(save_dir, exist_ok=True)

    if args.drone_idx is not None:
        plotter.render_single_drone_picture(drone_idx=args.drone_idx, save_dir=save_dir)
        raise SystemExit(0)

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
    selected_urdfs = plotter._normalize_urdf_indices(args.urdf_idxs)
    if selected_urdfs is not None:
        plotter.plot_per_urdf_policies(
            save_dir=save_dir,
            show=False,
            urdf_indices=selected_urdfs,
        )
        plotter.plot_reward_evolution_per_urdf(
            save_dir=save_dir,
            show=False,
            urdf_indices=selected_urdfs,
        )
        for urdf_idx in sorted(selected_urdfs):
            plotter.render_single_drone_picture(drone_idx=urdf_idx, save_dir=save_dir)
    plotter.plot_mean_policies(
        save_path=os.path.join(save_dir, "mean_across_urdfs.png"),
        show=False,
    )
    plotter.plot_mean_policies(
        save_path=os.path.join(save_dir, "mean_across_urdfs_>minimal.png"),
        show=False,
        only_above_minimal_progress=True,
    )
    plotter.plot_mean_policies(
        save_path=os.path.join(save_dir, "mean_across_urdfs_>300m.png"),
        show=False,
        only_above_minimal_progress=True,
        progress_threshold_m=300.0,
    )
    plotter.plot_selected_gp_vs_sp_distribution(
        save_path=os.path.join(
            save_dir,
            f"mean_across_urdfs_distribution_gp{general_policy_idx}.png",
        ),
        general_policy_idx=general_policy_idx,
        show=False,
    )
    plotter.plot_selected_gp_vs_sp_distribution(
        save_path=os.path.join(
            save_dir,
            f"mean_across_urdfs_distribution_gp{general_policy_idx}_sp_all_seeds.png",
        ),
        general_policy_idx=general_policy_idx,
        show=False,
        sp_use_all_repetitions=True,
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
        save_dir=save_dir,
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
    plotter.plot_policy_reward_metric_correlation(
        save_path=os.path.join(save_dir, "policy_reward_metric_correlation.png"),
        show=False,
    )
    plotter.plot_selected_gp_sp_mean_reward_metric_correlation(
        general_policy_idx=general_policy_idx,
        save_path=os.path.join(
            save_dir,
            f"gp{general_policy_idx}_selected_vs_sp_mean_reward_metric_correlation.png",
        ),
        show=False,
    )
    plotter.plot_sp_metric_vs_steps_correlation_table(
        save_path=os.path.join(save_dir, "sp_metric_vs_steps_correlation_table.png"),
        show=False,
    )
    plotter.plot_gp1_sp_mismatch_vs_sp_steps(
        general_policy_idx=general_policy_idx,
        save_dir=save_dir,
        show=False,
    )
