"""
Outer-loop plots: Pareto-front evolution + per-URDF metric curves.
==================================================================

Reads the CSVs written by ``NSGA2MorphCMAES``:

* ``results/outer_population.csv``       — per outer gen × URDF phase-mean
  objectives (the NSGA-II selection input / Pareto archive).
* ``results/outer_per_urdf_per_gen.csv`` — per inner gen × URDF diagnostics.

Usage
-----
    PYTHONPATH=src python -m WP2_Outer_Loop.pareto_plots <run_dir>

Also callable programmatically::

    from WP2_Outer_Loop.pareto_plots import plot_outer_run
    plot_outer_run(run_dir)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# (csv_column, display_label, lower_is_better) — mirrors WP2.plot_metrics.
_DIAG_METRICS = [
    ("fitness",           "Fitness (reward)",   False),
    ("velocity",          "Velocity [m/s]",     False),
    ("progress_m",        "Progress [m]",       False),
    ("crash_rate",        "Crash Rate",         True),
    ("cost_of_transport", "Cost of Transport",  True),
]


# ----------------------------------------------------------------------------
#  Config / data loading
# ----------------------------------------------------------------------------

def _load_objective_specs(run_dir: Path) -> List[Tuple[str, str]]:
    """Return [(name, direction), ...] from the saved outer config.

    Single-file runs save ``reproducibility/config.yaml`` with the objectives
    under ``outer:``; legacy runs save ``reproducibility/outer_config.yaml``
    with them at top level.
    """
    candidates = [
        (run_dir / "reproducibility" / "config.yaml", "outer"),
        (run_dir / "reproducibility" / "outer_config.yaml", None),
    ]
    for cfg_path, section in candidates:
        if not cfg_path.is_file():
            continue
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            if section is not None:
                cfg = cfg.get(section) or {}
            objs = cfg.get("objectives") or []
            specs = [(str(o["name"]), str(o["direction"])) for o in objs]
            if specs:
                return specs
        except Exception:
            pass
    return [("fitness", "maximize"), ("cost_of_transport", "minimize")]


def _load_min_progress(run_dir: Path) -> float:
    """``outer.min_progress_m`` from the saved config; 0.0 (gate off) when
    the config or the knob is absent (pre-gate runs)."""
    cfg_path = Path(run_dir) / "reproducibility" / "config.yaml"
    if not cfg_path.is_file():
        return 0.0
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        return float((cfg.get("outer") or {}).get("min_progress_m") or 0.0)
    except Exception:
        return 0.0


_PROGRESS_NAMES = ("progress_m", "progress")


def _filter_exam_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Keep only exam-scored rows when the run has any; ``(df, exam_only)``.

    Exam runs (``outer.rescore``) score phases on the exam forest
    distribution (``obj_source == "exam"``); rows tagged ``phase_mean`` are
    fallbacks scored on the (easier) inner-loop forests — e.g. the trailing
    partial phase, flushed after env teardown — so their objectives are not
    comparable and would corrupt fronts/hypervolumes. Runs with no exam rows
    (legacy or rescore off) pass through unchanged."""
    if "obj_source" not in df.columns:
        return df, False
    exam = df["obj_source"].astype(str) == "exam"
    if not exam.any():
        return df, False
    n_drop = int((~exam).sum())
    if n_drop:
        gens = sorted(df.loc[~exam, "outer_gen"].unique().tolist())
        print(f"[pareto_plots] Dropping {n_drop} non-exam rows "
              f"(outer gens {gens}): obj_source != 'exam' is scored on a "
              f"different forest distribution")
    return df[exam].reset_index(drop=True), True


def _read_results_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """DataFrame from ``csv_path``, or ``None`` (with a message) when the
    file is missing, zero-byte (interrupted sync), or has no rows."""
    if not csv_path.is_file():
        print(f"[pareto_plots] CSV not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if df.empty:
        print(f"[pareto_plots] CSV is empty: {csv_path}")
        return None
    return df


def _admission_mask(
    df: pd.DataFrame,
    specs: List[Tuple[str, str]],
    min_progress: Optional[float],
) -> np.ndarray:
    """Rows admitted to front / hypervolume computation under the
    minimum-progress gate (scatter always shows everything).

    Gate column: the ``obj_`` column when progress is a plotted objective
    (consistent with what selection saw), else the ``progress_m``
    diagnostic column. NaN progress is admitted; gate off or no progress
    column → everything admitted."""
    n = len(df)
    if min_progress is None or float(min_progress) <= 0.0:
        return np.ones(n, dtype=bool)
    col = None
    for name, _direction in specs:
        if name in _PROGRESS_NAMES and f"obj_{name}" in df.columns:
            col = f"obj_{name}"
            break
    if col is None and "progress_m" in df.columns:
        col = "progress_m"
    if col is None:
        print(f"[pareto_plots] min_progress={float(min_progress):g} requested "
              f"but no progress column in the CSV — gate disabled")
        return np.ones(n, dtype=bool)
    vals = df[col].to_numpy(dtype=float)
    return ~(vals < float(min_progress))  # NaN compares False → admitted


# Objective name → column in results/validation_summary.csv (baseline side).
_VALIDATION_BASELINE_COLS = {
    "fitness":            "baseline_fitness",
    "velocity":           "baseline_velocity",
    "progress_m":         "baseline_progress",
    "crash_rate":         "baseline_crash_rate",
    "cost_of_transport":  "baseline_cot",
    "velocity_deviation": "baseline_v_deviation",
    "v_deviation":        "baseline_v_deviation",
}


def _validation_ran_on_standard_drone(run_dir: Path) -> bool:
    """True when the run's held-out validation env held the standard mydrone.

    Validation must have been enabled AND left on the default catalog
    (``validation.validation_catalog`` empty/none, which builds the URDF from
    ``STANDARD_MYDRONE_GENOME``). Gates both reference points below, so a
    star is only ever drawn when the point really is the standard drone.
    """
    cfg_path = run_dir / "reproducibility" / "config.yaml"
    if not cfg_path.is_file():
        return False
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        val = cfg.get("validation") or {}
        if not val.get("enable"):
            return False
        vc = str(val.get("validation_catalog") or "").strip()
        return not vc or vc.lower() == "none"
    except Exception:
        return False


def _mean_columns(
    csv_path: Path, col_x: Optional[str], col_y: Optional[str],
) -> Optional[Tuple[float, float]]:
    """Mean of two named columns of ``csv_path``, or ``None`` when the file
    or either column is missing / unmapped / empty."""
    if col_x is None or col_y is None or not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty or col_x not in df.columns or col_y not in df.columns:
        return None
    x, y = df[col_x].mean(), df[col_y].mean()
    if pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y)


def _load_standard_drone_baseline(
    run_dir: Path, name_x: str, name_y: str,
) -> Optional[Tuple[float, float]]:
    """Mean (obj_x, obj_y) of the zero-rules baseline on the standard mydrone,
    from the held-out validation CSV — i.e. measured on the NOMINAL forests.

    Only valid for non-exam runs; exam runs use ``_load_exam_baseline``.
    """
    if not _validation_ran_on_standard_drone(run_dir):
        return None
    return _mean_columns(
        run_dir / "results" / "validation_summary.csv",
        _VALIDATION_BASELINE_COLS.get(name_x),
        _VALIDATION_BASELINE_COLS.get(name_y),
    )


def _load_exam_baseline(
    run_dir: Path, name_x: str, name_y: str,
) -> Optional[Tuple[float, float]]:
    """Mean (obj_x, obj_y) of the zero-rules generalist on the standard
    mydrone flown over the EXAM forests (``outer.exam_baseline``).

    This is the star for exam-scored runs: the same forest distribution the
    exam objectives were measured on, so the reference point is comparable.
    Averaged over every phase — the drone and the controller are fixed, so
    the per-phase spread is forest noise. ``None`` for runs predating the
    flag (no CSV) or whose validation env was not the standard drone.
    """
    if not _validation_ran_on_standard_drone(run_dir):
        return None
    # The CSV already uses canonical objective names as its metric columns.
    return _mean_columns(
        run_dir / "results" / "outer_exam_baseline.csv",
        "v_deviation" if name_x == "velocity_deviation" else name_x,
        "v_deviation" if name_y == "velocity_deviation" else name_y,
    )


def _nondominated_mask(points_max: np.ndarray) -> np.ndarray:
    """Boolean mask of nondominated rows; ``points_max`` is (n, m) in
    maximization space (all objectives flipped to higher-is-better)."""
    n = len(points_max)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (np.all(points_max[j] >= points_max[i])
                    and np.any(points_max[j] > points_max[i])):
                keep[i] = False
                break
    return keep


# Absolute worst-case value (raw objective space) for objectives with a
# physically fixed scale: zero progress / velocity, CoT 1, every drone
# crashed. Reward-shaped objectives (fitness) have no absolute scale.
_ABS_WORST_RAW = {
    "progress_m": 0.0,
    "progress": 0.0,
    "cost_of_transport": 1.0,
    "cot": 1.0,
}


def _hv_reference(
    specs: List[Tuple[str, str]], pts_max: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Hypervolume reference point in maximization space.

    When both objectives have an absolute worst bound (``_ABS_WORST_RAW``)
    the reference is fixed there — (0 m, CoT 1) for the standard
    progress/cot pair — so hypervolumes are comparable across runs.
    Otherwise it is run-relative (worst observed − 5 % of span per
    objective) and only the within-run curve is meaningful. Returns
    ``(ref, fixed)``.
    """
    names = [name for name, _ in specs[:2]]
    if all(n in _ABS_WORST_RAW for n in names):
        sign = np.array([1.0 if d == "maximize" else -1.0
                         for _, d in specs[:2]])
        return np.array([_ABS_WORST_RAW[n] for n in names]) * sign, True
    span = pts_max.max(axis=0) - pts_max.min(axis=0)
    return pts_max.min(axis=0) - 0.05 * np.where(span > 0, span, 1.0), False


def _hypervolume_2d(front_max: np.ndarray, ref: np.ndarray) -> float:
    """2-D hypervolume (maximization space) of a nondominated front w.r.t.
    reference point ``ref`` (worse than every front point)."""
    if len(front_max) == 0:
        return 0.0
    pts = front_max[np.argsort(-front_max[:, 0])]  # x desc ⇒ y asc on a front
    hv = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if x <= ref[0] or y <= prev_y:
            continue
        hv += (x - ref[0]) * (y - prev_y)
        prev_y = y
    return float(hv)


# ----------------------------------------------------------------------------
#  Pareto front evolution
# ----------------------------------------------------------------------------

def plot_pareto_front(
    run_dir: Path | str, min_progress: Optional[float] = None,
) -> Optional[dict]:
    """Objective-space scatter per outer generation + per-gen fronts +
    cumulative front, and a hypervolume-vs-generation curve.

    ``min_progress`` gates admission to the fronts and hypervolumes (the
    scatter always shows every point): ``None`` reads
    ``outer.min_progress_m`` from the run's saved config (absent/0 = no
    gate); pass a value explicitly to regenerate old runs with a gate.

    Returns ``{"hypervolume", "ref", "ref_fixed"}`` — the final cumulative
    hypervolume and its raw-space reference point (see ``_hv_reference``;
    cross-run comparable only when ``ref_fixed``) — or ``None`` when the
    run has no plottable data."""
    run_dir = Path(run_dir)
    csv_path = run_dir / "results" / "outer_population.csv"
    df = _read_results_csv(csv_path)
    if df is None:
        return
    df, exam_only = _filter_exam_rows(df)

    specs = _load_objective_specs(run_dir)
    if len(specs) < 2:
        print(f"[pareto_plots] Need ≥ 2 objectives, got {specs}")
        return
    (name_x, dir_x), (name_y, dir_y) = specs[0], specs[1]
    col_x, col_y = f"obj_{name_x}", f"obj_{name_y}"
    if col_x not in df.columns or col_y not in df.columns:
        print(f"[pareto_plots] Objective columns missing: {col_x}, {col_y}")
        return

    if min_progress is None:
        min_progress = _load_min_progress(run_dir)
    min_progress = float(min_progress)
    admit = _admission_mask(df, specs, min_progress)
    if not admit.any():
        print(f"[pareto_plots] min_progress={min_progress:g} excludes every "
              f"point — plotting ungated")
        admit = np.ones(len(df), dtype=bool)
        min_progress = 0.0

    sign = np.array([
        1.0 if dir_x == "maximize" else -1.0,
        1.0 if dir_y == "maximize" else -1.0,
    ])
    raw = df[[col_x, col_y]].to_numpy(dtype=float)
    pts_max = raw * sign  # maximization space
    gens = df["outer_gen"].to_numpy(dtype=int)
    uniq_gens = np.unique(gens)

    ref, ref_fixed = _hv_reference(specs, pts_max[admit])
    ref_raw = (ref * sign).tolist()

    cmap = plt.get_cmap("viridis")
    colors = {g: cmap(i / max(1, len(uniq_gens) - 1))
              for i, g in enumerate(uniq_gens)}

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    hv_per_gen, hv_cumulative = [], []
    seen_max = np.empty((0, 2))
    few_gens = len(uniq_gens) <= 8  # legend for few gens, colorbar otherwise

    for g in uniq_gens:
        sel = gens == g
        p_raw, p_max, p_adm = raw[sel], pts_max[sel], admit[sel]
        ax.scatter(p_raw[:, 0], p_raw[:, 1], color=colors[g], s=38,
                   alpha=0.85, edgecolors="none", zorder=3)
        # This generation's own front (admitted points only)
        mask = _nondominated_mask(p_max[p_adm])
        front = p_raw[p_adm][mask]
        order = np.argsort(front[:, 0])
        ax.plot(front[order, 0], front[order, 1], color=colors[g],
                linewidth=1.4, alpha=0.9, zorder=2,
                label=(f"gen {g}" if few_gens else None))
        hv_per_gen.append(_hypervolume_2d(p_max[p_adm][mask], ref))
        # Cumulative front over everything admitted so far
        seen_max = np.vstack([seen_max, p_max[p_adm]])
        cum_mask = _nondominated_mask(seen_max)
        hv_cumulative.append(_hypervolume_2d(seen_max[cum_mask], ref))

    # Final cumulative front in red on top
    cum_mask = _nondominated_mask(pts_max[admit])
    cum_front = raw[admit][cum_mask]
    order = np.argsort(cum_front[:, 0])
    ax.plot(cum_front[order, 0], cum_front[order, 1], color="#d62728",
            linewidth=2.2, marker="o", markersize=6, zorder=4,
            label="cumulative front")

    # Admission threshold marker when progress is one of the plotted axes.
    if min_progress > 0.0:
        if name_x in _PROGRESS_NAMES:
            ax.axvline(min_progress, color="gray", linestyle=":",
                       linewidth=1.4, zorder=1,
                       label=f"min progress {min_progress:g} m")
        elif name_y in _PROGRESS_NAMES:
            ax.axhline(min_progress, color="gray", linestyle=":",
                       linewidth=1.4, zorder=1,
                       label=f"min progress {min_progress:g} m")

    # Standard mydrone reference (zero-rules generalist), always measured on
    # the same forests as the plotted objectives: the exam-baseline rollout
    # for exam-scored runs, the held-out validation pass otherwise. Both need
    # validation to have run on the standard drone.
    if exam_only:
        star = _load_exam_baseline(run_dir, name_x, name_y)
        star_label = "standard mydrone (zero rules, exam forests)"
    else:
        star = _load_standard_drone_baseline(run_dir, name_x, name_y)
        star_label = "standard mydrone (zero rules)"
    if star is not None:
        ax.scatter([star[0]], [star[1]], marker="*", s=340, color="gold",
                   edgecolors="black", linewidths=0.9, zorder=5,
                   label=star_label)

    def _axis_label(name: str, direction: str) -> str:
        return f"{name}  ({'higher' if direction == 'maximize' else 'lower'} better)"

    ax.set_xlabel(_axis_label(name_x, dir_x))
    ax.set_ylabel(_axis_label(name_y, dir_y))
    ax.set_title("Outer-loop Pareto front evolution")
    ax.grid(True, linestyle="--", alpha=0.4)
    if few_gens:
        ax.legend(fontsize=8, ncols=2)
    else:
        ax.legend(fontsize=8, loc="best")
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=uniq_gens.min(), vmax=uniq_gens.max()),
        )
        fig.colorbar(sm, ax=ax, label="outer generation")
    fig.tight_layout()
    out = plots_dir / "pareto_front_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[pareto_plots] Saved {out}")

    # Zoomed variant: same figure cropped to the cumulative front's bounding
    # box (plus a small margin), so the front's extreme individuals sit at
    # the plot corners without their markers being clipped.
    if len(cum_front) >= 2:
        (x_lo, y_lo) = cum_front.min(axis=0)
        (x_hi, y_hi) = cum_front.max(axis=0)
        if x_hi > x_lo:
            pad = 0.03 * (x_hi - x_lo)
            ax.set_xlim(x_lo - pad, x_hi + pad)
        if y_hi > y_lo:
            pad = 0.03 * (y_hi - y_lo)
            ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_title("Outer-loop Pareto front evolution (zoomed to front)")
        out = plots_dir / "pareto_front_evolution_zoomed.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[pareto_plots] Saved {out}")
    plt.close(fig)

    # Hypervolume curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(uniq_gens, hv_cumulative, color="#d62728", linewidth=1.8,
            marker="o", label="cumulative front")
    ax.plot(uniq_gens, hv_per_gen, color="#1f77b4", linewidth=1.4,
            marker="s", markersize=4, label="per-generation front")
    ax.set_xlabel("Outer generation")
    ax.set_ylabel("Hypervolume (↑ better)")
    ref_note = (f"ref: {name_x}={ref_raw[0]:g}, {name_y}={ref_raw[1]:g}"
                if ref_fixed else "run-relative ref")
    ax.set_title(f"Pareto hypervolume — {name_x} vs {name_y}  ({ref_note})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = plots_dir / "pareto_hypervolume.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[pareto_plots] Saved {out}")
    print(f"[pareto_plots] cumulative hypervolume: {hv_cumulative[-1]:.4f} "
          f"({ref_note})")
    return {
        "hypervolume": float(hv_cumulative[-1]),
        "ref": ref_raw,
        "ref_fixed": ref_fixed,
        "star": star,
    }


# ----------------------------------------------------------------------------
#  Per-URDF metric curves (inner-loop plot style, morphology axis kept)
# ----------------------------------------------------------------------------

def plot_outer_metrics(run_dir: Path | str) -> None:
    """Per-inner-generation per-URDF diagnostics: one thin line per URDF slot,
    population mean bold, URDF-refresh vlines — mirrors WP2.plot_metrics."""
    run_dir = Path(run_dir)
    csv_path = run_dir / "results" / "outer_per_urdf_per_gen.csv"
    df = _read_results_csv(csv_path)
    if df is None:
        return

    # Phase length: single-file runs store it as catalog.refresh_urdfs_every;
    # legacy runs as top-level inner_generations in outer_config.yaml.
    refresh_every = 0
    for cfg_path, getter in [
        (run_dir / "reproducibility" / "config.yaml",
         lambda c: (c.get("catalog") or {}).get("refresh_urdfs_every")),
        (run_dir / "reproducibility" / "outer_config.yaml",
         lambda c: c.get("inner_generations")),
    ]:
        if not cfg_path.is_file():
            continue
        try:
            with open(cfg_path) as f:
                refresh_every = int(getter(yaml.safe_load(f) or {}) or 0)
            if refresh_every:
                break
        except Exception:
            pass

    gens = np.sort(df["generation"].unique())
    urdf_ids = np.sort(df["urdf_idx"].unique())
    mean_by_gen = df.groupby("generation").mean(numeric_only=True)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle("Outer loop — per-URDF metrics (thin: URDF slots, bold: mean)",
                 fontsize=13)

    slot_cmap = plt.get_cmap("winter")
    slot_colors = {u: slot_cmap(i / max(1, len(urdf_ids) - 1))
                   for i, u in enumerate(urdf_ids)}

    for ax, (col, label, lower_is_better) in zip(axes.flat, _DIAG_METRICS):
        for u in urdf_ids:
            sub = df[df["urdf_idx"] == u].set_index("generation")[col]
            ax.plot(sub.index.to_numpy(), sub.to_numpy(),
                    color=slot_colors[u], linewidth=0.8, alpha=0.5)
        ax.plot(mean_by_gen.index.to_numpy(), mean_by_gen[col].to_numpy(),
                color="#1f77b4", linewidth=2.0, label="mean over URDFs")
        if refresh_every > 0:
            vline_label = "URDF refresh"
            for g in range(refresh_every, int(gens.max()) + 1, refresh_every):
                ax.axvline(g, color="#ff7f0e", linestyle=":", linewidth=0.8,
                           alpha=0.7, label=vline_label)
                vline_label = None
        title = f"{label}  (↓ better)" if lower_is_better else label
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Inner generation")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Sixth panel: per-outer-gen population objective means (twin axes — the
    # objectives live on very different scales).
    ax = axes.flat[len(_DIAG_METRICS)]
    pop_csv = run_dir / "results" / "outer_population.csv"
    pop = _read_results_csv(pop_csv) if pop_csv.is_file() else None
    if pop is not None:
        pop, _ = _filter_exam_rows(pop)
        if not pop.empty:
            g_means = pop.groupby("outer_gen").mean(numeric_only=True)
            # From g_means, not pop: numeric-only mean drops non-numeric
            # obj_ columns (obj_source is a string tag, not an objective).
            obj_cols = [c for c in g_means.columns if c.startswith("obj_")]
            x = g_means.index.to_numpy()
            axes_pair = [ax, ax.twinx()] if len(obj_cols) >= 2 else [ax]
            palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
            for i, c in enumerate(obj_cols):
                a = axes_pair[min(i, len(axes_pair) - 1)]
                a.plot(x, g_means[c].to_numpy(), marker="o", linewidth=1.6,
                       color=palette[i % len(palette)],
                       label=c.replace("obj_", ""))
                a.set_ylabel(c.replace("obj_", ""),
                             color=palette[i % len(palette)])
                a.tick_params(axis="y", labelcolor=palette[i % len(palette)])
            ax.set_title("Phase-mean objectives per outer gen", fontsize=11)
            ax.set_xlabel("Outer generation")
            ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = plots_dir / "outer_metrics_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[pareto_plots] Saved {out}")


def plot_outer_run(
    run_dir: Path | str, min_progress: Optional[float] = None,
) -> None:
    """All outer-loop plots for a run directory."""
    plot_pareto_front(run_dir, min_progress=min_progress)
    plot_outer_metrics(run_dir)


if __name__ == "__main__":
    argv = sys.argv[1:]
    _min_progress: Optional[float] = None
    if "--min-progress" in argv:
        i = argv.index("--min-progress")
        try:
            _min_progress = float(argv[i + 1])
        except (IndexError, ValueError):
            print("--min-progress requires a numeric value (meters)")
            sys.exit(1)
        del argv[i:i + 2]
    if len(argv) < 1:
        print("Usage: python -m WP2_Outer_Loop.pareto_plots <run_dir> "
              "[--min-progress X]")
        sys.exit(1)
    plot_outer_run(argv[0], min_progress=_min_progress)
