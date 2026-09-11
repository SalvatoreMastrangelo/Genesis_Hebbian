"""
Outer-loop plots: Pareto front, exam champions, per-URDF metric curves.
=======================================================================

Reads the CSVs written by ``NSGA2MorphCMAES``:

* ``results/outer_population.csv``       — per outer gen × URDF phase-mean
  objectives (the NSGA-II selection input / Pareto archive).
* ``results/outer_per_urdf_per_gen.csv`` — per inner gen × URDF diagnostics.
* ``results/outer_exam_baseline.csv``    — per outer gen, the standard mydrone
  flown by the zero-rules generalist on that phase's exam forests.

Figures written to ``<run_dir>/plots``:

* ``pareto_front_evolution[_zoomed].png`` / ``pareto_hypervolume.png``
* ``<source>_champions.png`` — both objectives of the two record-holding
  morphologies vs outer generation, ``<source>`` being ``exam`` when the run
  has an exam rollout and ``phase_mean`` otherwise (``plot_champion_curves``)
* ``outer_metrics_evolution.png``
* ``champion_renders/`` — still renders (3/4 + top view) of the progress
  champion and of the cheapest morphology passing the ``min_progress_m``
  gate (``render_champions``; needs Genesis, best-effort, ``--no-render``
  or ``render=False`` to skip)

Usage
-----
    PYTHONPATH=src python -m WP2_Outer_Loop.pareto_plots <run_dir> [--min-progress X] [--no-render]

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
#  Config / data loading  (shared with the live front writer)
# ----------------------------------------------------------------------------
#
# These live in ``pareto_fronts`` — a matplotlib-free module the training
# process can import — so plots, the live per-phase front CSV, and the
# backfill CLI all share one definition of "the front". Re-exported here
# because callers and tests import them from ``pareto_plots``.

from .pareto_fronts import (  # noqa: E402  (kept next to its explanation)
    BIXLER_LABEL,
    _PROGRESS_NAMES,
    _admission_mask,
    _filter_exam_rows,
    _hv_reference,
    _hypervolume_2d,
    _load_min_progress,
    _load_objective_specs,
    _load_refresh_every,
    _nondominated_mask,
    _read_results_csv,
    build_pareto_front_csv,
)


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


def _exam_flew_nominal_forests(run_dir: Path) -> bool:
    """True when the exam rollout flew the inner loop's forest distribution —
    the same one the held-out validation pass flies.

    Mirrors ``config.ExamForestConfig.overrides()``: the exam distribution is
    its own only when ``outer.exam_forest.override_forest`` is on AND at least
    one field under it is non-null. Runs predating the section entirely have
    no way to override, so they flew nominal forests too.

    When this holds, the validation baseline is measured on exactly the
    distribution the exam objectives were scored on, which makes it a valid
    stand-in for a missing ``outer_exam_baseline.csv`` (below).
    """
    cfg_path = Path(run_dir) / "reproducibility" / "config.yaml"
    if not cfg_path.is_file():
        return False
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        exam_forest = (cfg.get("outer") or {}).get("exam_forest") or {}
    except Exception:
        return False
    if not exam_forest.get("override_forest"):
        return True
    return not any(v is not None for k, v in exam_forest.items()
                   if k != "override_forest")


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

    The reference for phase-mean runs, and the fallback for exam runs that
    flew nominal forests (``_exam_flew_nominal_forests``) without writing an
    exam baseline; an exam on its own forest distribution needs
    ``_load_exam_baseline`` instead.
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


def _axis_label(name: str, direction: str) -> str:
    return f"{name}  ({'higher' if direction == 'maximize' else 'lower'} better)"


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
        star_label = BIXLER_LABEL
        # Exam-scored run with no baseline CSV (it predates
        # ``outer.exam_baseline``): the validation pass is still the right
        # reference IF the exam flew the nominal forests, since then the two
        # rollouts sampled the same distribution.
        if star is None and _exam_flew_nominal_forests(run_dir):
            star = _load_standard_drone_baseline(run_dir, name_x, name_y)
            star_label = BIXLER_LABEL
    else:
        star = _load_standard_drone_baseline(run_dir, name_x, name_y)
        star_label = BIXLER_LABEL
    if star is not None:
        ax.scatter([star[0]], [star[1]], marker="*", s=340, color="gold",
                   edgecolors="black", linewidths=0.9, zorder=5,
                   label=star_label)

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
#  Champion curves (record-holding morphologies vs outer generation)
# ----------------------------------------------------------------------------

def _cumulative_champions(
    df: pd.DataFrame, specs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Per outer generation, the two record-holding morphologies so far.

    Champion ``a`` holds the best value of objective 0, champion ``b`` the
    best of objective 1, over every row with ``outer_gen <= g`` — cumulative
    and ungated (``outer.min_progress_m`` is deliberately not applied, so a
    barely-flying low-CoT morph can hold the CoT record).

    Each champion contributes BOTH of its objectives: ``a_obj1`` is what the
    objective-0 record-holder costs on objective 1, not the population's best
    objective 1. A record holder carries forward until beaten, so the curves
    are step-shaped by construction. NaN objectives never win a record.

    Returns a frame indexed by ``outer_gen``, columns
    ``a_obj0, a_obj1, a_urdf, b_obj0, b_obj1, b_urdf``.
    """
    cols = [f"obj_{name}" for name, _ in specs[:2]]
    signs = [1.0 if d == "maximize" else -1.0 for _, d in specs[:2]]
    vals = df[cols].to_numpy(dtype=float)
    urdfs = (df["urdf_file"].astype(str).to_numpy() if "urdf_file" in df.columns
             else np.full(len(df), "", dtype=object))
    gens = df["outer_gen"].to_numpy(dtype=int)

    best: List[Optional[int]] = [None, None]  # row index of each record holder
    rows = []
    for g in np.unique(gens):
        for i in np.flatnonzero(gens == g):
            for k in (0, 1):
                v = vals[i, k] * signs[k]
                if np.isnan(v):
                    continue
                if best[k] is None or v > vals[best[k], k] * signs[k]:
                    best[k] = int(i)
        row = {"outer_gen": int(g)}
        for k, tag in ((0, "a"), (1, "b")):
            i = best[k]
            row[f"{tag}_obj0"] = vals[i, 0] if i is not None else np.nan
            row[f"{tag}_obj1"] = vals[i, 1] if i is not None else np.nan
            row[f"{tag}_urdf"] = urdfs[i] if i is not None else ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("outer_gen")


def _load_exam_baseline_series(
    run_dir: Path, names: List[str],
) -> Optional[pd.DataFrame]:
    """Per-outer-gen standard-mydrone reference (``results/outer_exam_baseline
    .csv``) restricted to ``names``, indexed by ``outer_gen``.

    ``None`` when validation did not run on the standard mydrone (the drone
    would be mislabelled — same gate as the Pareto star), when the CSV is
    absent (runs predating ``outer.exam_baseline``), or when a requested
    metric is not a column.

    Rows written offline by ``exam_baseline_rerun`` carry ``outer_gen < 0``
    (whole-run reference, measured once rather than per phase) — see
    ``_baseline_is_constant``.
    """
    if not _validation_ran_on_standard_drone(run_dir):
        return None
    csv_path = Path(run_dir) / "results" / "outer_exam_baseline.csv"
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    cols = ["v_deviation" if n == "velocity_deviation" else n for n in names]
    if df.empty or "outer_gen" not in df.columns:
        return None
    if any(c not in df.columns for c in cols):
        return None
    return df.set_index("outer_gen")[cols]


def _baseline_is_constant(base: pd.DataFrame) -> bool:
    """True when the reference is a single whole-run measurement rather than
    one measurement per phase, so it belongs on the champion panels as a
    horizontal line instead of a curve.

    Signalled by ``outer_gen < 0`` (the sentinel ``exam_baseline_rerun``
    writes) or by there being only one row to plot.
    """
    return len(base) <= 1 or bool((base.index.to_numpy() < 0).all())


# Objective source → (figure title, caveat line under it).
_SOURCE_TITLES = {
    "exam": ("Exam champions", ""),
    "phase_mean": (
        "Phase-mean champions",
        "objectives = phase means over the inner-loop forests, not exam-scored",
    ),
}


def _objective_source(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Rows to plot and the label of the objective source they all share.

    Exam-scored rows win whenever the run has any (the ``phase_mean``
    fallback phases fly an easier forest distribution, so they are dropped
    rather than mixed in). Otherwise every row is kept: a run with no exam
    rows — ``outer.rescore`` off, or predating the ``obj_source`` column
    entirely — is uniformly phase-mean scored, so there is nothing to mix.
    """
    kept, exam_only = _filter_exam_rows(df)
    if exam_only:
        return kept, "exam"
    if "obj_source" not in df.columns:
        return df, "phase_mean"  # predates the column ⇒ all phase means
    sources = sorted(df["obj_source"].astype(str).unique())
    if len(sources) > 1:
        print(f"[pareto_plots] Mixed objective sources {sources} with no exam "
              f"rows — plotting them together; the curves compare different "
              f"measurements")
        return df, "mixed"
    return df, sources[0]


def _load_validation_baseline_series(
    run_dir: Path, names: List[str],
) -> Optional[pd.DataFrame]:
    """Per-outer-gen standard-mydrone reference for runs with no exam
    rollout, from the held-out validation pass (``validation_summary.csv``).

    That pass flies the nominal forests — the same distribution the
    phase-mean objectives are harvested on — which is what makes it the right
    reference here and the wrong one for exam-scored runs.

    The CSV is indexed by INNER generation, so each outer phase window
    (``catalog.refresh_urdfs_every`` inner gens) collapses to its mean. When
    the refresh period is unreadable the inner gens cannot be mapped to
    phases at all, so everything collapses to a single whole-run row under
    the ``outer_gen = -1`` sentinel and ``_baseline_is_constant`` puts it on
    the panels as a horizontal line.

    Same validity gate as the Pareto star: validation must have run on the
    standard mydrone. ``None`` when it did not, when the CSV is missing, or
    when an objective has no baseline column.
    """
    if not _validation_ran_on_standard_drone(Path(run_dir)):
        return None
    csv_path = Path(run_dir) / "results" / "validation_summary.csv"
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    cols = [_VALIDATION_BASELINE_COLS.get(n) for n in names]
    if df.empty or any(c is None or c not in df.columns for c in cols):
        return None

    every = _load_refresh_every(Path(run_dir))
    if every > 0 and "generation" in df.columns:
        outer = df["generation"].to_numpy(dtype=int) // every
        return df[cols].groupby(outer).mean()
    means = df[cols].mean()
    if means.isna().any():
        return None
    return pd.DataFrame([means.to_numpy()], index=[-1], columns=cols)


def plot_champion_curves(run_dir: Path | str) -> Optional[dict]:
    """Both objectives of the two record-holding morphologies vs outer gen.

    2×2: one row per champion, one column per objective, so for the standard
    progress/CoT run the top row is what the furthest-flying morph costs in
    CoT and the bottom row is how far the cheapest morph actually gets.

    Objectives come from the exam rollout when the run has one and from the
    phase-mean harvest otherwise (``_objective_source``); the figure is named
    and titled after that source, so a phase-mean plot is never mistaken for
    an exam one. The standard-mydrone baseline always comes from the rollout
    that flew the plotted objectives' forests: the exam-baseline pass on exam
    figures, the held-out validation pass on phase-mean ones — and, for an
    exam that flew nominal forests without writing a baseline CSV, that same
    validation pass (``_exam_flew_nominal_forests``).

    Returns ``{"champions", "baseline", "source"}``, or ``None`` when the run
    has nothing plottable.
    """
    run_dir = Path(run_dir)
    df = _read_results_csv(run_dir / "results" / "outer_population.csv")
    if df is None:
        return None
    df, source = _objective_source(df)

    specs = _load_objective_specs(run_dir)
    if len(specs) < 2:
        print(f"[pareto_plots] Need ≥ 2 objectives, got {specs}")
        return None
    (name0, dir0), (name1, dir1) = specs[0], specs[1]
    missing = [f"obj_{n}" for n in (name0, name1)
               if f"obj_{n}" not in df.columns]
    if missing:
        print(f"[pareto_plots] Objective columns missing: {missing}")
        return None

    champs = _cumulative_champions(df, specs)
    # Reference measured on the same forests as the plotted objectives: the
    # exam-baseline rollout for exam runs, the held-out validation pass
    # (nominal forests, like the phase-mean harvest) otherwise.
    if source == "exam":
        base = _load_exam_baseline_series(run_dir, [name0, name1])
        if base is None and _exam_flew_nominal_forests(run_dir):
            base = _load_validation_baseline_series(run_dir, [name0, name1])
    else:
        base = _load_validation_baseline_series(run_dir, [name0, name1])

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # One row per champion, one column per objective; y-scales left
    # independent so a crawling CoT champion is readable next to a
    # long-range progress champion instead of pinned to the axis floor.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = champs.index.to_numpy()
    champ_styles = [("a", f"best {name0} morph"), ("b", f"best {name1} morph")]
    # Colour keyed to the objective (column), not the champion (row): the two
    # panels showing the same quantity share a colour, and which morphology
    # they belong to is read off the row / panel title.
    objectives = [(name0, dir0, "#1f77b4"), (name1, dir1, "#d62728")]
    for r, (tag, label) in enumerate(champ_styles):
        for k, (name, direction, color) in enumerate(objectives):
            ax = axes[r][k]
            ax.plot(x, champs[f"{tag}_obj{k}"].to_numpy(), color=color,
                    linewidth=1.8, marker="o", markersize=4, label=label)
            if base is not None:
                if _baseline_is_constant(base):
                    ax.axhline(float(base.iloc[:, k].mean()), color="gray",
                               linestyle="--", linewidth=1.4,
                               label=BIXLER_LABEL)
                else:
                    ax.plot(base.index.to_numpy(), base.iloc[:, k].to_numpy(),
                            color="gray", linestyle="--", linewidth=1.4,
                            label=BIXLER_LABEL)
            ax.set_ylabel(_axis_label(name, direction))
            ax.set_title(f"{label} — {name}", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)
            if r == len(champ_styles) - 1:
                ax.set_xlabel("Outer generation")

    title, caveat = _SOURCE_TITLES.get(source, (f"{source} champions", ""))
    fig.suptitle(f"{title} — best-so-far morphologies (ungated), both "
                 f"objectives of each" + (f"\n{caveat}" if caveat else ""),
                 fontsize=13)
    fig.tight_layout()
    out = plots_dir / f"{source}_champions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[pareto_plots] Saved {out}")
    return {"champions": champs, "baseline": base is not None,
            "source": source}


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

    refresh_every = _load_refresh_every(run_dir)

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


def render_champions_safe(run_dir: Path | str) -> bool:
    """Still renders of the exam champions (``render_champions.render_run``)
    into ``plots/champion_renders/``; ``True`` on success.

    Needs a Genesis runtime, so it can only succeed inside the simulator
    environment (a live run, or the docker image). Anywhere else — or on any
    render error — it prints one line and returns ``False`` so the plots
    that came before it are never lost.
    """
    try:
        from WP2_Outer_Loop.render_champions import render_run
        render_run(run_dir)
        return True
    except Exception as exc:  # noqa: BLE001 — renders are best-effort
        print(f"[pareto_plots] champion renders skipped: {exc}")
        return False


def plot_outer_run(
    run_dir: Path | str, min_progress: Optional[float] = None,
    render: bool = True,
) -> None:
    """All outer-loop plots for a run directory.

    Also refreshes ``results/pareto_front.csv``, so re-plotting an old run
    backfills the per-generation front table it never wrote live. With
    ``render`` (the default) the two exam champion morphologies are rendered
    last via ``render_champions_safe`` — best-effort, needs Genesis.
    """
    build_pareto_front_csv(run_dir, min_progress=min_progress)
    plot_pareto_front(run_dir, min_progress=min_progress)
    plot_champion_curves(run_dir)
    plot_outer_metrics(run_dir)
    if render:
        render_champions_safe(run_dir)


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
    _render = True
    if "--no-render" in argv:
        argv.remove("--no-render")
        _render = False
    if len(argv) < 1:
        print("Usage: python -m WP2_Outer_Loop.pareto_plots <run_dir> "
              "[--min-progress X] [--no-render]")
        sys.exit(1)
    plot_outer_run(argv[0], min_progress=_min_progress, render=_render)
