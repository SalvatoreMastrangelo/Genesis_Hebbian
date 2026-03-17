#!/usr/bin/env python3
"""
Plot utilities for evaluating WingedDroneEnv policies.

All plotting logic is centralized here in the `EvaluationPlotter` class, so that
the main evaluation script can stay focused on running rollouts and computing
metrics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402


class EvaluationPlotter:
    """Collection of plotting utilities and shared helpers."""

    def __init__(self, default_win_frac: float = 0.03) -> None:
        """
        Args:
            default_win_frac: default fraction of samples used for the
                moving-average window when not explicitly specified.
        """
        self.default_win_frac = float(default_win_frac)

    @staticmethod
    def _resolve_velocity_range(
        velocity_range: Optional[Tuple[float, float]],
        fallback_values: np.ndarray,
    ) -> Tuple[float, float]:
        if velocity_range is not None:
            v_min, v_max = map(float, velocity_range)
            if v_max < v_min:
                raise ValueError(
                    f"velocity_range must satisfy max >= min, got {(v_min, v_max)}"
                )
            return v_min, v_max

        vals = np.asarray(fallback_values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0, 1.0
        return float(vals.min()), float(vals.max())

    # ------------------------------------------------------------------ #
    # Shared helper: moving average                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def moving_avg(
        x: np.ndarray,
        y: np.ndarray,
        win_frac: float = 0.03,
        drop_edges: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute a moving average and standard deviation over y(x).

        The window size is chosen as:
            win = max(11, int(len(x) * win_frac)), forced to be odd.

        The points are first sorted by x; the same ordering is applied to y.

        Args:
            x: 1D array of x-values.
            y: 1D array of y-values (same shape as x).
            win_frac: fraction of the dataset length used for the window.
            drop_edges:
                - True: return only the indices where the window is "full"
                  (half-width at both sides).
                - False: return the full length, using truncated windows at
                  the borders.

        Returns:
            x_out, y_smooth, y_std
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        order = np.argsort(x)
        x_ord, y_ord = x[order], y[order]

        win = max(11, int(len(x_ord) * float(win_frac))) | 1  # ensure odd >= 11
        half = win // 2

        y_smooth = np.empty_like(y_ord, dtype=float)
        y_std = np.empty_like(y_ord, dtype=float)

        for i in range(len(x_ord)):
            lo = max(0, i - half)
            hi = min(len(x_ord), i + half + 1)
            seg = y_ord[lo:hi]
            y_smooth[i] = seg.mean()
            y_std[i] = seg.std(ddof=0)

        if drop_edges:
            sl = slice(half, len(x_ord) - half)
            return x_ord[sl], y_smooth[sl], y_std[sl]
        else:
            return x_ord, y_smooth, y_std

    # ------------------------------------------------------------------ #
    # Forest layout plot                                                 #
    # ------------------------------------------------------------------ #
    def plot_forest(self, env, out: str = "forest_layout.png") -> None:
        """
        Plot the 2D layout of the forest (cylinders) used in evaluation.

        Args:
            env: WingedDroneEnv instance (must expose `cylinders_array`).
            out: output PNG path.
        """
        cyl = env.cylinders_array[0].cpu().numpy()  # (N, 3): x, y, z center
        width = cyl[:, 0].max()
        height = cyl[:, 1].max() * 2.0
        fig, ax = plt.subplots(figsize=(6, 2.5))

        # Trees
        ax.scatter(cyl[:, 0], cyl[:, 1], s=2, c="forestgreen", label="Trees")

        # Starting point
        ax.scatter(-30, 0, s=20, c="blue", marker="o", label="Starting point")

        # Manual "wall" rectangle
        x0, y0 = 0, -height / 2

        # Right, top, bottom edges: solid
        ax.plot([x0, x0 + width], [y0, y0], color="red", linewidth=2)  # bottom
        ax.plot(
            [x0, x0 + width],
            [y0 + height, y0 + height],
            color="red",
            linewidth=2,
        )  # top
        ax.plot(
            [x0 + width, x0 + width],
            [y0, y0 + height],
            color="red",
            linewidth=2,
        )  # right

        # Left edge: dashed
        ax.plot(
            [x0, x0],
            [y0, y0 + height],
            color="red",
            linewidth=2,
            linestyle="--",
        )

        # Arrow indicating progress direction
        ax.annotate(
            "",
            xy=(1000, 55),
            xytext=(0, 55),
            arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5),
        )

        ax.plot([], [], color="black", linewidth=1.5, label="Progress direction")
        ax.plot([], [], color="red", linewidth=2, label="Wall")

        ax.set_xlim(-40, width + 20)
        ax.set_ylim(-(height / 2 + 10), height / 2 + 10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Evaluation Forest Example")
        ax.legend(loc="lower right", fontsize=10)

        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"✅ forest layout saved to {out}")

    # ------------------------------------------------------------------ #
    # Heatmap of joint behaviour                                         #
    # ------------------------------------------------------------------ #
    def plot_joint_diff_heatmap(
        self,
        traces_all: Dict[str, List],
        dof: str,
        out: str = "joint_behaviour_heatmap.png",
        s_bins: int = 20,
        v_bins: int = 16,
        command_speed_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Heatmap of mean joint behaviour as a function of distance and v_cmd.

        Args:
            traces_all:
                Dictionary produced by `run_eval`, with:
                  - "s": list of arrays, one per env, of covered distance.
                  - "j_pos": list of arrays (T_i, num_joints).
                  - "v_cmd": array of commanded velocities, one per env.
            dof:
                "sweep": use (j0 - j1)/2, positive = forward sweep.
                "twist": use -(j2 + j3)/2, positive = upward twist.
            out: output PNG path.
            s_bins: number of bins along distance.
            v_bins: number of bins along commanded velocity.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        v_cmd = np.asarray(traces_all["v_cmd"])
        v_min, v_max = self._resolve_velocity_range(command_speed_range, v_cmd)
        env_n = len(v_cmd)

        # Max distance across all envs
        max_s = max((s.max() if len(s) else 0.0) for s in traces_all["s"])
        if max_s <= 0.0:
            print("⚠️  No valid data for joint heatmap – skipping.")
            return

        s_edges = np.linspace(0.0, max_s, s_bins + 1)
        v_edges = np.linspace(v_min, v_max, v_bins + 1)
        v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

        heat = np.full((v_bins, s_bins), np.nan, dtype=float)
        count = np.zeros_like(heat, dtype=int)

        for env_idx in range(env_n):
            s_arr = np.asarray(traces_all["s"][env_idx], dtype=float)
            if s_arr.size == 0:
                continue
            jp_arr = np.asarray(traces_all["j_pos"][env_idx], dtype=float)
            if jp_arr.size == 0:
                continue

            if dof == "sweep":
                behaviour = np.rad2deg((jp_arr[:, 0] - jp_arr[:, 1]) / 2.0)
            elif dof == "twist":
                behaviour = np.rad2deg(-(jp_arr[:, 2] + jp_arr[:, 3]) / 2.0)
            else:
                raise ValueError(f"dof must be 'sweep' or 'twist', got '{dof}'")

            bins = np.searchsorted(s_edges, s_arr, side="right") - 1
            bins[bins == s_bins] = s_bins - 1

            v_val = v_cmd[env_idx]
            v_i = np.searchsorted(v_edges, v_val, side="right") - 1
            v_i = min(v_i, v_bins - 1)

            for b in range(s_bins):
                mask = bins == b
                if not mask.any():
                    continue
                m = float(behaviour[mask].mean())
                if np.isnan(heat[v_i, b]):
                    heat[v_i, b] = m
                else:
                    # running mean over episodes
                    heat[v_i, b] = (
                        heat[v_i, b] * count[v_i, b] + m
                    ) / float(count[v_i, b] + 1)
                count[v_i, b] += 1

        # Plot
        heat_masked = np.ma.masked_invalid(heat)
        vmin = float(np.nanmin(heat))
        vmax = float(np.nanmax(heat))

        plt.figure(figsize=(9, 5))
        extent = [s_edges[0], s_edges[-1], v_edges[0], v_edges[-1]]
        im = plt.imshow(
            heat_masked,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im, label="mean joint position [deg]")
        plt.xlabel("Distance covered s  [m]")
        plt.ylabel("Commanded velocity  [m/s]")
        plt.yticks(v_centers[:: max(1, v_bins // 8)])
        if dof == "sweep":
            plt.title("Joint-behaviour heatmap (+ is forward sweep)")
        else:
            plt.title("Joint-behaviour heatmap (+ is upward twist)")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"✅ joint behaviour heatmap saved to {out}")

    # ------------------------------------------------------------------ #
    # 3D scatter (Plotly)                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_3d_speed_energy_agility(
        v_mean: np.ndarray,
        E_tot: np.ndarray,
        progress: np.ndarray,
        v_cmd: np.ndarray,
        html_out: str = "speed_energy_progress_3D.html",
        cmap: str = "Viridis",
        velocity_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Interactive 3D scatter:

            x = v_mean       [m/s]
            y = E_tot        [J/m]
            z = progress     [m]
            color = v_cmd    [m/s]
        """
        import numpy as np
        import plotly.graph_objects as go

        v_mean = np.asarray(v_mean)
        E_tot = np.asarray(E_tot)
        progress = np.asarray(progress)
        v_cmd = np.asarray(v_cmd)

        mask = (
            np.isfinite(v_mean)
            & np.isfinite(E_tot)
            & np.isfinite(progress)
            & np.isfinite(v_cmd)
        )
        x, y, z, c = v_mean[mask], E_tot[mask], progress[mask], v_cmd[mask]

        fig = go.Figure(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=4,
                    color=c,
                    colorscale=cmap,
                    colorbar=dict(title="v_cmd [m/s]"),
                    opacity=0.8,
                ),
                name="data-points",
            )
        )

        fig.update_layout(
            title="Speed–Energy–Progress (3-D)",
            scene=dict(
                xaxis=dict(
                    title="v_mean [m/s]",
                    range=list(EvaluationPlotter._resolve_velocity_range(velocity_range, v_cmd)),
                ),
                yaxis=dict(title="E_tot [J/m]", range=[4, 10]),
                zaxis=dict(title="Progress [m]", range=[0, 1000]),
            ),
            margin=dict(l=0, r=0, t=35, b=0),
        )

        fig.write_html(html_out, include_plotlyjs="cdn")
        print(f"✅  3D scatter saved to {html_out}")

    @classmethod
    def plot_3d_speed_energy_progress_ma(
        cls,
        v_mean: np.ndarray,
        E_tot: np.ndarray,
        progress: np.ndarray,
        v_cmd: np.ndarray,
        win_frac: float = 0.03,
        html_out: str = "speed_energy_progress_ma_3D.html",
        cmap: str = "Viridis",
        velocity_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        3D scatter + moving-average curve linking:
            (v_mean, MA(E_tot), MA(progress)).
        """
        import numpy as np
        import plotly.graph_objects as go

        v_mean = np.asarray(v_mean)
        E_tot = np.asarray(E_tot)
        progress = np.asarray(progress)
        v_cmd = np.asarray(v_cmd)

        mask = (
            np.isfinite(v_mean)
            & np.isfinite(E_tot)
            & np.isfinite(progress)
            & np.isfinite(v_cmd)
        )
        x, y, z, c = v_mean[mask], E_tot[mask], progress[mask], v_cmd[mask]

        fig = go.Figure(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=4,
                    color=c,
                    colorscale=cmap,
                    colorbar=dict(title="v_cmd [m/s]"),
                    opacity=0.8,
                ),
                name="data-points",
            )
        )

        # 1) MA of E_tot vs v_mean
        x_ma, E_ma, _ = cls.moving_avg(x, y, win_frac)
        # 2) MA of progress vs v_mean
        _, P_ma, _ = cls.moving_avg(x, z, win_frac)

        fig.add_trace(
            go.Scatter3d(
                x=x_ma,
                y=E_ma,
                z=P_ma,
                mode="lines",
                line=dict(color="black", width=5),
                name=f"moving-avg (win={win_frac:.0%})",
            )
        )

        fig.update_layout(
            title="Speed–Energy–Progress (3-D) + moving average",
            scene=dict(
                xaxis=dict(
                    title="v_mean [m/s]",
                    range=list(cls._resolve_velocity_range(velocity_range, v_cmd)),
                ),
                yaxis=dict(title="E_tot [J/m]", range=[4, 10]),
                zaxis=dict(title="Progress [m]", range=[0, 1000]),
            ),
            margin=dict(l=0, r=0, t=35, b=0),
            legend=dict(x=0.02, y=0.98),
        )

        fig.write_html(html_out, include_plotlyjs="cdn")
        print(f"✅  3D scatter + MA saved to {html_out}")

    # ------------------------------------------------------------------ #
    # Total plot: MA curves + admissible region                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_custom_point(
        cp: Optional[object],
    ) -> Optional[Tuple[float, float, float]]:
        """Utility: parse custom (v, p, eff) point."""
        if cp is None:
            return None
        import numpy as np

        if isinstance(cp, dict):
            return (
                float(cp.get("v", np.nan)),
                float(cp.get("p", np.nan)),
                float(cp.get("eff", np.nan)),
            )
        if isinstance(cp, (tuple, list)) and len(cp) == 3:
            v, p, e = cp
            return float(v), float(p), float(e)
        raise ValueError(
            "custom_point must be (v, p, eff) or a dict with keys 'v', 'p', 'eff'"
        )

    @staticmethod
    def _find_intervals_below(
        xv: np.ndarray,
        yv: np.ndarray,
        thr: float,
    ) -> List[Tuple[float, float]]:
        """Find contiguous x-intervals where yv < thr."""
        import numpy as np

        if len(xv) == 0 or not np.isfinite(thr):
            return []
        below = yv < thr
        if not np.any(below):
            return []
        out: List[Tuple[float, float]] = []
        i, n = 0, len(xv)
        while i < n:
            if below[i]:
                xs, j = xv[i], i
                while j + 1 < n and below[j + 1]:
                    j += 1
                out.append((float(xs), float(xv[j])))
                i = j + 1
            else:
                i += 1
        return out

    @classmethod
    def total_plot(
        cls,
        v_mean,
        E_tot,
        v_cmd,
        progress,
        *,
        win_frac: float = 0.03,
        percentage: float = 0.10,
        minimal_p: float | None = None,
        custom_point=None,   # (v, p, eff) or {"v":..., "p":..., "eff":...}
        out: str = "total_plot.png",
        velocity_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        2×1 figure (shared x-axis).

        Top plot:
            - Moving-average of progress vs mean velocity.
            - Horizontal line at minimal progress threshold.
            - Grey vertical bands where MA(progress) < threshold.
            - Blue markers/lines for:
                * max progress (horizontal line to y-axis),
                * max velocity within admissible region (vertical line).
            - Optional red operational point (v, p).

        Bottom plot:
            - Moving-average of energy per meter vs mean velocity.
            - Same grey bands.
            - Blue markers/lines for:
                * min energy (only where progress is admissible).
                * same max-velocity vertical line as top plot.
            - Optional red operational point (v, eff).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        # ------------- helper: parse custom (v, p, eff) point -------------
        def _coerce_custom_point(cp):
            if cp is None:
                return None
            if isinstance(cp, dict):
                return (
                    float(cp.get("v", np.nan)),
                    float(cp.get("p", np.nan)),
                    float(cp.get("eff", np.nan)),
                )
            if isinstance(cp, (tuple, list)) and len(cp) == 3:
                return tuple(map(float, cp))
            raise ValueError(
                "custom_point must be (v, p, eff) or a dict with keys 'v', 'p', 'eff'"
            )

        # ------------------ filter valid samples ---------------------------
        v_mean = np.asarray(v_mean)
        E_tot = np.asarray(E_tot)
        v_cmd = np.asarray(v_cmd)
        progress = np.asarray(progress)

        mask = (
            np.isfinite(v_cmd)
            & np.isfinite(v_mean)
            & np.isfinite(E_tot)
            & np.isfinite(progress)
        )
        x = v_cmd[mask]      # reference for windowing (commanded speed)
        v = v_mean[mask]     # measured mean velocity
        E = E_tot[mask]      # energy per meter
        P = progress[mask]   # progress

        if x.size == 0:
            print("⚠️ total_plot: no valid samples, skipping.")
            return

        # ----------------- moving averages wrt v_cmd ----------------------
        # x_ref_* are the sorted v_cmd used for the window; for plotting
        # we treat x_line as the MA of v_mean (v_ma).
        x_ref_for_v, v_ma, v_std = cls.moving_avg(x, v, win_frac)
        x_ref_for_p, p_ma, p_std = cls.moving_avg(x, P, win_frac)
        x_ref_for_E, E_ma, E_std = cls.moving_avg(x, E, win_frac)

        # Same x for the MA curves (mean velocity)
        x_line = v_ma
        # Values used to color segments by commanded velocity
        c_line = x_ref_for_v

        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=np.nanmin(x), vmax=np.nanmax(x))

        # -------------------- threshold & intervals -----------------------
        max_p = float(np.nanmax(p_ma)) if p_ma.size else np.nan
        if minimal_p is None and np.isfinite(max_p):
            minimal_p = max_p * (1.0 - float(percentage))

        def _find_intervals_below(xv, yv, thr):
            xv = np.asarray(xv)
            yv = np.asarray(yv)
            if xv.size == 0 or not np.isfinite(thr):
                return []
            below = yv < thr
            if not np.any(below):
                return []
            out = []
            i = 0
            n = len(xv)
            while i < n:
                if below[i]:
                    xs = xv[i]
                    j = i
                    while j + 1 < n and below[j + 1]:
                        j += 1
                    out.append((float(xs), float(xv[j])))
                    i = j + 1
                else:
                    i += 1
            return out

        intervals = (
            _find_intervals_below(x_line, p_ma, minimal_p)
            if np.isfinite(minimal_p)
            else []
        )

        # ---------------------- blue peaks & zones ------------------------
        # Max progress (on MA curve)
        idx_p = int(np.nanargmax(p_ma)) if p_ma.size else None
        x_at_pmax = float(x_line[idx_p]) if idx_p is not None else np.nan
        y_pmax = float(p_ma[idx_p]) if idx_p is not None else np.nan

        # Max velocity within admissible region (p_ma > minimal_p)
        x_maxvel_zone = None
        y_at_xmax_zone = None
        y_start_maxvel = None
        if np.isfinite(minimal_p):
            ok = p_ma > minimal_p
            if np.any(ok):
                idxs_ok = np.where(ok)[0]
                last_ok = int(idxs_ok[-1])
                x_maxvel_zone = float(x_line[last_ok])
                y_at_xmax_zone = float(p_ma[last_ok])
                y_start_maxvel = y_at_xmax_zone
                # If the curve drops below the threshold right after, interpolate
                if last_ok + 1 < len(p_ma) and p_ma[last_ok + 1] <= minimal_p:
                    x0, x1 = x_line[last_ok], x_line[last_ok + 1]
                    y0, y1 = p_ma[last_ok], p_ma[last_ok + 1]
                    if y1 != y0:
                        t = (minimal_p - y0) / (y1 - y0)
                        x_maxvel_zone = float(x0 + t * (x1 - x0))
                        y_start_maxvel = float(minimal_p)

        # Min energy (only where progress admissible)
        x_at_emin = None
        y_emin = None
        if np.isfinite(minimal_p):
            okE = p_ma > minimal_p
            if np.any(okE):
                Emin_segment = E_ma[okE]
                x_segment = x_line[okE]
                j = int(np.nanargmin(Emin_segment))
                y_emin = float(Emin_segment[j])
                x_at_emin = float(x_segment[j])
        # Fallback: global min if no admissible region
        if x_at_emin is None or not np.isfinite(x_at_emin):
            if E_ma.size:
                j = int(np.nanargmin(E_ma))
                x_at_emin = float(x_line[j])
                y_emin = float(E_ma[j])

        # ---------------------- custom operational point ------------------
        cp = _coerce_custom_point(custom_point)
        if cp is None:
            v_c = p_c = eff_c = np.nan
        else:
            v_c, p_c, eff_c = cp

        # ---------------------------- figure ------------------------------
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(8.2, 7.6),
            sharex=True,
            gridspec_kw=dict(hspace=0.10),
        )

        # Common x-limits
        ax2.set_xlim(cls._resolve_velocity_range(velocity_range, x))
        xmin, xmax = ax2.get_xlim()

        Z_LINE = 6
        Z_MARK = 10

        # =======================
        #   SUBPLOT 1 — PROGRESS
        # =======================

        # Custom point on progress
        if np.isfinite(v_c) and np.isfinite(p_c):
            ax1.plot(
                v_c,
                p_c,
                "o",
                ms=6,
                color="tab:red",
                label="progress only -- Operational Point",
                zorder=Z_MARK,
            )
            ax1.plot(
                [xmin, v_c],
                [p_c, p_c],
                ls="--",
                lw=1.4,
                color="tab:red",
                label="progress only -- Fitness",
                zorder=Z_LINE,
            )
            ax1.plot(
                [xmin],
                [p_c],
                marker="*",
                ms=11,
                color="tab:red",
                zorder=Z_MARK,
                clip_on=False,
            )
            ax1.axvline(
                v_c,
                ls="--",
                lw=1.4,
                color="tab:red",
                zorder=Z_LINE,
            )

        # Moving-average curve of progress vs mean velocity (colored by v_cmd)
        pts1 = np.array([x_line, p_ma]).T.reshape(-1, 1, 2)
        segs1 = np.concatenate([pts1[:-1], pts1[1:]], axis=1)
        lc1 = LineCollection(segs1, cmap=cmap, norm=norm)
        lc1.set_array(0.5 * (c_line[:-1] + c_line[1:]))  # color = mean v_cmd of segment
        lc1.set_linewidth(2.2)
        lc1.set_zorder(Z_LINE)
        ax1.add_collection(lc1)

        # Legend entry for the MA curve
        proxy_line = Line2D(
            [0],
            [0],
            color="black",
            lw=2.2,
            label="Moving Average (colored by v_cmd)",
        )
        h1, l1 = ax1.get_legend_handles_labels()
        h1.append(proxy_line)
        ax1.legend(handles=h1, fontsize=9, loc="best")

        # Colorbar for commanded velocity
        cb1 = plt.colorbar(lc1, ax=ax1, pad=0.01)
        cb1.set_label("Commanded Velocity [m/s]")

        # ±1σ band around MA(progress)
        ax1.fill_between(
            x_line,
            p_ma - p_std,
            p_ma + p_std,
            color="purple",
            label="±1 σ",
            alpha=0.1,
        )

        # Grey forbidden bands (progress below threshold)
        for a, b in intervals:
            ax1.axvspan(a, b, color="gray", alpha=0.30, lw=0)

        # Horizontal blue line from max progress to y-axis
        if np.isfinite(y_pmax):
            ax1.plot(
                [xmin, x_at_pmax],
                [y_pmax, y_pmax],
                ls="--",
                lw=1.4,
                color="tab:blue",
                zorder=Z_LINE,
                label="velocity tracking -- Fitness",
            )
            ax1.plot(
                [xmin],
                [y_pmax],
                marker="s",
                ms=9,
                color="tab:blue",
                zorder=Z_MARK,
                clip_on=False,
            )

        # Horizontal threshold line
        if np.isfinite(minimal_p):
            ax1.axhline(
                minimal_p,
                ls="--",
                lw=1.4,
                color="tab:gray",
                label="minimal progress threshold",
                zorder=Z_LINE,
            )

        # Vertical blue line for max velocity in admissible region
        if x_maxvel_zone is not None and np.isfinite(y_start_maxvel):
            ax1.plot(
                [x_maxvel_zone, x_maxvel_zone],
                [y_start_maxvel, 0.0],
                ls="--",
                lw=1.4,
                color="tab:blue",
                zorder=Z_LINE,
            )

        ax1.set_ylim(0, 600)
        ax1.set_ylabel("Progress [m]")
        ax1.grid(alpha=0.25)

        # Add legend entry for grey bands
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles1.append(
            Patch(facecolor="gray", alpha=0.30, label="NOT admissible region")
        )
        ax1.legend(handles=handles1, fontsize=9, loc="best")

        # ==========================
        #   SUBPLOT 2 — ENERGY/met
        # ==========================

        # MA curve of energy vs mean velocity (same x_line)
        pts2 = np.array([x_line, E_ma]).T.reshape(-1, 1, 2)
        segs2 = np.concatenate([pts2[:-1], pts2[1:]], axis=1)
        lc2 = LineCollection(segs2, cmap=cmap, norm=norm)
        lc2.set_array(0.5 * (c_line[:-1] + c_line[1:]))
        lc2.set_linewidth(2.2)
        lc2.set_zorder(Z_LINE)
        ax2.add_collection(lc2)

        cb2 = plt.colorbar(lc2, ax=ax2, pad=0.01)
        cb2.set_label("Commanded Velocity [m/s]")

        # ±1σ band around MA(energy)
        ax2.fill_between(
            x_line,
            E_ma - E_std,
            E_ma + E_std,
            color="purple",
            label="±1 σ",
            alpha=0.1,
        )

        # Same grey forbidden bands
        for a, b in intervals:
            ax2.axvspan(a, b, color="gray", alpha=0.30, lw=0)

        # Horizontal blue line from min energy to y-axis
        if np.isfinite(y_emin):
            ax2.plot(
                [xmin, x_at_emin],
                [y_emin, y_emin],
                ls="--",
                lw=1.4,
                color="tab:blue",
                zorder=Z_LINE,
            )
            ax2.plot(
                [xmin],
                [y_emin],
                marker="s",
                ms=9,
                color="tab:blue",
                zorder=Z_MARK,
                clip_on=False,
            )

        # Vertical blue line for max velocity (continuation) + square on x-axis
        if x_maxvel_zone is not None and np.isfinite(y_at_xmax_zone):
            ax2.axvline(
                x_maxvel_zone,
                ls="--",
                lw=1.4,
                color="tab:blue",
                zorder=Z_LINE,
            )

        # Custom point on energy
        if np.isfinite(v_c) and np.isfinite(eff_c):
            ax2.plot(
                v_c,
                eff_c,
                "o",
                ms=6,
                color="tab:red",
                zorder=Z_MARK,
            )
            ax2.plot(
                [xmin, v_c],
                [eff_c, eff_c],
                ls="--",
                lw=1.4,
                color="tab:red",
                zorder=Z_LINE,
            )
            ax2.plot(
                [xmin],
                [eff_c],
                marker="*",
                ms=11,
                color="tab:red",
                zorder=Z_MARK,
                clip_on=False,
            )
            ax2.axvline(
                v_c,
                ls="--",
                lw=1.4,
                color="tab:red",
                zorder=Z_LINE,
            )
            ax2.plot(
                [v_c],
                [3.0],
                marker="*",
                ms=11,
                color="tab:red",
                zorder=Z_MARK,
                clip_on=False,
            )

        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Mean Velocity x̄ along progress direction [m/s]")
        ax2.set_ylabel("Cost of Transport [J/m]")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=9, loc="best")

        # Square marker at x-axis where the max-velocity line meets the axis
        if x_maxvel_zone is not None:
            y_axis = ax2.get_ylim()[0]
            ax2.plot(
                [x_maxvel_zone],
                [y_axis],
                marker="s",
                ms=9,
                color="tab:blue",
                zorder=Z_MARK,
                clip_on=False,
            )

        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"✅  total_plot saved to {out}")


    # ------------------------------------------------------------------ #
    # Total plot: variant with points instead of MA curves               #
    # ------------------------------------------------------------------ #
    @classmethod
    def total_plot_points_instead_of_ma(
        cls,
        v_mean: np.ndarray,
        E_tot: np.ndarray,
        v_cmd: np.ndarray,
        progress: np.ndarray,
        *,
        win_frac: float = 0.03,
        percentage: float = 0.10,
        minimal_p: Optional[float] = None,
        custom_point: Optional[object] = None,
        out: str = "total_plot_points_instead_of_ma.png",
        data_marker_size: float = 22.0,
        op_marker_size: float = 80.0,
        velocity_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Same frame/axes/legend as `total_plot`, but:
          - No moving averages and no blue peak markers.
          - Two scatter plots:
              [top]  progress vs mean velocity, colored by v_cmd
              [bot]  E_tot vs mean velocity, colored by v_cmd
          - Optional green/red operational point is plotted with a larger
            constant-size marker in the foreground.
          - Shaded bands where MA(progress) < minimal_p.
        """
        import numpy as np

        v_mean = np.asarray(v_mean, dtype=float)
        E_tot = np.asarray(E_tot, dtype=float)
        v_cmd = np.asarray(v_cmd, dtype=float)
        progress = np.asarray(progress, dtype=float)

        # Custom point
        v_c, p_c, eff_c = (np.nan, np.nan, np.nan)
        cp = cls._coerce_custom_point(custom_point)
        if cp is not None:
            v_c, p_c, eff_c = cp

        # Valid samples
        mask = (
            np.isfinite(v_mean)
            & np.isfinite(E_tot)
            & np.isfinite(progress)
            & np.isfinite(v_cmd)
        )
        x = v_mean[mask]
        E = E_tot[mask]
        P = progress[mask]
        C = v_cmd[mask]

        # Determine minimal_p from MA of progress if not provided
        x_ma, p_ma, _ = cls.moving_avg(x, P, win_frac)
        max_p = float(np.nanmax(p_ma)) if len(p_ma) else np.nan
        if minimal_p is None and np.isfinite(max_p):
            minimal_p = max_p * (1.0 - float(percentage))

        intervals = (
            cls._find_intervals_below(x_ma, p_ma, minimal_p)
            if np.isfinite(minimal_p)
            else []
        )

        fig = plt.figure(figsize=(8.2, 7.6))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.0, 0.08],
            hspace=0.10,
            wspace=0.25,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        cax1 = fig.add_subplot(gs[0, 1])
        cax2 = fig.add_subplot(gs[1, 1])

        ax2.set_xlim(cls._resolve_velocity_range(velocity_range, C))
        xmin, xmax = ax2.get_xlim()

        # Shaded bands
        if np.isfinite(minimal_p):
            for (xs, xe) in intervals:
                ax1.axvspan(xs, xe, color="0.85", alpha=0.6, zorder=1)
                ax2.axvspan(xs, xe, color="0.85", alpha=0.6, zorder=1)

        # Top: progress
        sc1 = ax1.scatter(
            x,
            P,
            c=C,
            cmap="viridis",
            s=data_marker_size,
            alpha=0.75,
            zorder=3,
        )
        if np.isfinite(minimal_p):
            ax1.axhline(
                minimal_p,
                ls="--",
                lw=1.4,
                color="tab:gray",
                zorder=4,
                label="minimal progress threshold",
            )

        cb1 = plt.colorbar(sc1, cax=cax1)
        cb1.set_label("Commanded Velocity [m/s]")

        if np.isfinite(v_c) and np.isfinite(p_c):
            ax1.scatter(
                [v_c],
                [p_c],
                s=op_marker_size,
                c=["tab:red"],
                edgecolors="none",
                zorder=12,
                label="progress only — Operational Point",
            )

        ax1.set_ylim(0, 600)
        ax1.set_ylabel("Progress [m]")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=9, loc="best")

        # Bottom: energy
        sc2 = ax2.scatter(
            x,
            E,
            c=C,
            cmap="viridis",
            s=data_marker_size,
            alpha=0.75,
            zorder=3,
        )
        cb2 = plt.colorbar(sc2, cax=cax2)
        cb2.set_label("Commanded Velocity [m/s]")

        if np.isfinite(v_c) and np.isfinite(eff_c):
            ax2.scatter(
                [v_c],
                [eff_c],
                s=op_marker_size,
                c=["tab:red"],
                edgecolors="none",
                zorder=12,
            )

        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Mean Velocity x̄ along progress direction [m/s]")
        ax2.set_ylabel("Cost of Transport [J/m]")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=9, loc="best")

        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"✅ total_plot_points_instead_of_ma saved to {out}")
