"""
Per-generation Pareto fronts of the outer loop, as a CSV.
=========================================================

Reads ``results/outer_population.csv`` (the NSGA-II selection input written
by ``NSGA2MorphCMAES``) and writes ``results/pareto_front.csv``: one row per
Pareto-front member per outer generation, carrying the normalized genome so
any row's URDF can be regenerated with ``urdf_population.materialize_urdfs``.

The front is computed the way selection sees it, in three steps:

1. **Exam filter** (``_filter_exam_rows``) — when a run has exam-scored rows,
   ``phase_mean`` fallbacks are dropped: they are scored on the easier
   inner-loop forests and would forge a front tail.
2. **Admission gate** (``_admission_mask``) — ``outer.min_progress_m`` bars
   sub-threshold morphologies, matching ``nsga_cma.gated_select``. A missing
   config, a missing ``min_progress_m`` key, or a value <= 0 all mean the
   gate is OFF: every individual is admitted.
3. **Non-domination** (``_nondominated_mask``) — within each ``outer_gen``,
   over the ``obj_*`` columns of the run's configured objectives.

This module is matplotlib-free so the training process can import it: it owns
the pure helpers (config loading, filters, non-domination, hypervolume) and
``pareto_plots`` imports them back, so plots, the live writer, and the
backfill CLI all share one definition of "the front".

Usage
-----
Backfill an existing run (also refreshed by any ``plot_outer_run`` call)::

    PYTHONPATH=src python -m WP2_Outer_Loop.pareto_fronts <run_dir> \
        [--min-progress X]

Programmatically::

    from WP2_Outer_Loop.pareto_fronts import build_pareto_front_csv
    build_pareto_front_csv(run_dir)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


# Diagnostics carried through from outer_population.csv onto each front row
# (mirrors nsga_cma._DIAG_KEYS; absent columns are skipped, so legacy CSVs
# with a narrower set still work).
_DIAG_KEYS: Tuple[str, ...] = (
    "fitness", "cost_of_transport", "progress_m", "velocity", "crash_rate",
)

_FRONT_CSV_NAME = "pareto_front.csv"


# ----------------------------------------------------------------------------
#  Config / data loading
# ----------------------------------------------------------------------------

def _load_objective_specs(run_dir: Path) -> List[Tuple[str, str]]:
    """Return [(name, direction), ...] from the saved outer config.

    Single-file runs save ``reproducibility/config.yaml`` with the objectives
    under ``outer:``; legacy runs save ``reproducibility/outer_config.yaml``
    with them at top level.
    """
    run_dir = Path(run_dir)
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

# Legend label of the reference point in every outer-loop plot: the standard
# mydrone (the "Bixler") flown by the frozen WP1 generalist with zero rules.
BIXLER_LABEL = "Bixler (generalist controller)"


def _load_refresh_every(run_dir: Path) -> int:
    """Inner generations per outer phase, i.e. the URDF refresh period.

    Single-file runs store it as ``catalog.refresh_urdfs_every``; legacy runs
    as top-level ``inner_generations`` in ``outer_config.yaml``. 0 when
    neither is readable.
    """
    run_dir = Path(run_dir)
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
                every = int(getter(yaml.safe_load(f) or {}) or 0)
            if every:
                return every
        except Exception:
            pass
    return 0


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
        print(f"[pareto] Dropping {n_drop} non-exam rows "
              f"(outer gens {gens}): obj_source != 'exam' is scored on a "
              f"different forest distribution")
    return df[exam].reset_index(drop=True), True


def _read_results_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """DataFrame from ``csv_path``, or ``None`` (with a message) when the
    file is missing, zero-byte (interrupted sync), or has no rows."""
    if not csv_path.is_file():
        print(f"[pareto] CSV not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if df.empty:
        print(f"[pareto] CSV is empty: {csv_path}")
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
    diagnostic column. NaN progress is admitted; gate off (``None``, 0, or
    negative) or no progress column → everything admitted."""
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
        print(f"[pareto] min_progress={float(min_progress):g} requested "
              f"but no progress column in the CSV — gate disabled")
        return np.ones(n, dtype=bool)
    vals = df[col].to_numpy(dtype=float)
    return ~(vals < float(min_progress))  # NaN compares False → admitted


# ----------------------------------------------------------------------------
#  Non-domination / hypervolume
# ----------------------------------------------------------------------------

def _nondominated_mask_2d(points_max: np.ndarray) -> np.ndarray:
    """O(n log n) sweep for the two-objective case: sort by x descending
    (y descending within equal x) and keep a point iff its y beats the
    best y seen among strictly larger x. Same semantics as the general
    loop — a row is dropped iff some other row is >= in both objectives
    and > in at least one — so exact duplicates are all kept."""
    n = len(points_max)
    order = np.lexsort((-points_max[:, 1], -points_max[:, 0]))
    keep = np.zeros(n, dtype=bool)
    best_y = -np.inf
    i = 0
    while i < n:
        j = i
        x = points_max[order[i], 0]
        while j < n and points_max[order[j], 0] == x:
            j += 1
        block = order[i:j]
        ymax = points_max[block, 1].max()
        if ymax > best_y:
            keep[block[points_max[block, 1] == ymax]] = True
            best_y = ymax
        i = j
    return keep


def _nondominated_mask(points_max: np.ndarray) -> np.ndarray:
    """Boolean mask of nondominated rows; ``points_max`` is (n, m) in
    maximization space (all objectives flipped to higher-is-better).
    Two objectives take the O(n log n) sweep (``_nondominated_mask_2d``);
    any other m falls back to the pairwise loop."""
    points_max = np.asarray(points_max, dtype=float)
    n = len(points_max)
    if n and points_max.ndim == 2 and points_max.shape[1] == 2:
        return _nondominated_mask_2d(points_max)
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


# Fixed hypervolume anchor (raw objective space) for objectives with a
# meaningful absolute scale: the 80 m `outer.min_progress_m` admission gate
# and a CoT ceiling of 0.5, just above the worst front CoT observed across
# the Aug-2026 exam runs (0.391). The physical worst case (0 m, CoT 1) sits
# so far from where fronts live that every run shared a huge dead rectangle,
# compressing between-run differences to a few percent. Reward-shaped
# objectives (fitness) have no absolute scale and fall back to run-relative.
_FIXED_HV_REF_RAW = {
    "progress_m": 80.0,
    "progress": 80.0,
    "cost_of_transport": 0.5,
    "cot": 0.5,
}


def _hv_reference(
    specs: List[Tuple[str, str]], pts_max: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Hypervolume reference point in maximization space.

    When both objectives have a fixed anchor (``_FIXED_HV_REF_RAW``) the
    reference is fixed there — (80 m, CoT 0.5) for the standard
    progress/cot pair — so hypervolumes are comparable across runs. Front
    points worse than the anchor in either objective contribute nothing
    (``_hypervolume_2d`` clips them). Otherwise it is run-relative (worst
    observed − 5 % of span per objective) and only the within-run curve is
    meaningful. Returns ``(ref, fixed)``.
    """
    names = [name for name, _ in specs[:2]]
    if all(n in _FIXED_HV_REF_RAW for n in names):
        sign = np.array([1.0 if d == "maximize" else -1.0
                         for _, d in specs[:2]])
        return np.array([_FIXED_HV_REF_RAW[n] for n in names]) * sign, True
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
#  Front extraction
# ----------------------------------------------------------------------------

def _objective_columns(
    df: pd.DataFrame, specs: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str, str]]:
    """``[(column, name, direction), ...]`` for the configured objectives
    present in the CSV, in config order."""
    out: List[Tuple[str, str, str]] = []
    for name, direction in specs:
        col = f"obj_{name}"
        if col in df.columns:
            out.append((col, name, direction))
    return out


def _maximization_points(
    df: pd.DataFrame, obj_cols: Sequence[Tuple[str, str, str]],
) -> np.ndarray:
    """(n, m) objective matrix flipped so higher is better on every axis."""
    cols = [c for c, _, _ in obj_cols]
    signs = np.array(
        [1.0 if d == "maximize" else -1.0 for _, _, d in obj_cols]
    )
    return df[cols].to_numpy(dtype=float) * signs


def compute_front_rows(
    df: pd.DataFrame,
    specs: Sequence[Tuple[str, str]],
    min_progress: Optional[float],
) -> pd.DataFrame:
    """Per-generation Pareto fronts of ``df`` (an ``outer_population.csv``).

    Applies the exam filter, then the admission gate, then per-``outer_gen``
    non-domination. Returns the front rows with ``front_size``,
    ``front_rank``, ``hypervolume`` and ``hv_reference_fixed`` added; the two
    per-generation scalars are repeated on every row of that generation.

    ``hypervolume`` is NaN when the run has anything other than exactly two
    objectives — ``_hypervolume_2d`` is 2-D only.
    """
    specs = list(specs)
    df, _exam_only = _filter_exam_rows(df)
    obj_cols = _objective_columns(df, specs)
    if not obj_cols:
        print(f"[pareto] No obj_* columns for objectives "
              f"{[n for n, _ in specs]} — cannot compute fronts")
        return pd.DataFrame()

    admitted = _admission_mask(df, specs, min_progress)
    n_barred = int((~admitted).sum())
    if n_barred:
        print(f"[pareto] min_progress_m={float(min_progress):g} bars "
              f"{n_barred}/{len(df)} rows from the fronts")
    df = df[admitted].reset_index(drop=True)
    if df.empty:
        print("[pareto] No admissible rows — empty front CSV")
        return pd.DataFrame()

    # A fixed HV reference is generation-independent; a run-relative one is
    # derived once from every admitted point so the HV curve is comparable
    # across generations of this run.
    hv_ref, hv_fixed = (
        _hv_reference(specs, _maximization_points(df, obj_cols))
        if len(obj_cols) == 2 else (None, False)
    )

    pieces: List[pd.DataFrame] = []
    for gen, grp in df.groupby("outer_gen", sort=True):
        grp = grp.reset_index(drop=True)
        pts = _maximization_points(grp, obj_cols)
        finite = np.isfinite(pts).all(axis=1)
        if not finite.any():
            print(f"[pareto] outer_gen {gen}: no finite objectives — skipped")
            continue
        keep = np.zeros(len(grp), dtype=bool)
        keep[np.where(finite)[0][_nondominated_mask(pts[finite])]] = True

        front = grp[keep].copy()
        front_pts = pts[keep]
        # Sort by the first objective, best first — a stable index for
        # picking a spread along the front.
        order = np.argsort(-front_pts[:, 0], kind="stable")
        front = front.iloc[order].reset_index(drop=True)

        front["front_size"] = len(front)
        front["front_rank"] = np.arange(len(front))
        front["hypervolume"] = (
            _hypervolume_2d(front_pts[order], hv_ref)
            if hv_ref is not None else np.nan
        )
        front["hv_reference_fixed"] = bool(hv_fixed)
        pieces.append(front)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _ordered_columns(
    front: pd.DataFrame, specs: Sequence[Tuple[str, str]],
) -> List[str]:
    """Front CSV column order: identity, front stats, objectives,
    diagnostics, genome. Unknown extra columns are appended, never dropped."""
    identity = ["outer_gen", "urdf_idx", "urdf_file", "obj_source",
                "n_score_gens"]
    stats = ["front_size", "front_rank", "hypervolume", "hv_reference_fixed"]
    objectives = [f"obj_{name}" for name, _ in specs]
    genes = sorted(
        (c for c in front.columns
         if len(c) > 1 and c[0] == "g" and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    ordered = [
        c for c in identity + stats + objectives + list(_DIAG_KEYS) + genes
        if c in front.columns
    ]
    return ordered + [c for c in front.columns if c not in ordered]


def build_pareto_front_csv(
    run_dir: Path | str,
    specs: Optional[Sequence[Tuple[str, str]]] = None,
    min_progress: Optional[float] = None,
) -> Optional[Path]:
    """Write ``results/pareto_front.csv`` for ``run_dir``; return its path.

    Full rebuild from ``results/outer_population.csv``, so it is idempotent
    and self-healing: the live writer, a re-plot, and the backfill CLI all
    produce the same file. Returns ``None`` when the source CSV is missing,
    empty, or yields no front rows.

    ``specs`` / ``min_progress`` default to the run's saved config. The live
    writer passes them from the in-memory ``OuterNSGA2Config`` instead, so it
    never depends on ``reproducibility/config.yaml`` being parseable. A
    ``min_progress`` of ``None`` that resolves to a missing config key means
    the gate is off, not an error.
    """
    run_dir = Path(run_dir)
    src = run_dir / "results" / "outer_population.csv"
    df = _read_results_csv(src)
    if df is None:
        return None

    if specs is None:
        specs = _load_objective_specs(run_dir)
    if min_progress is None:
        min_progress = _load_min_progress(run_dir)

    front = compute_front_rows(df, specs, min_progress)
    if front.empty:
        return None

    front = front[_ordered_columns(front, specs)]
    out_path = run_dir / "results" / _FRONT_CSV_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    front.to_csv(out_path, index=False)
    n_gens = front["outer_gen"].nunique()
    print(f"[pareto] Wrote {out_path} — {len(front)} front members "
          f"across {n_gens} outer generations")
    return out_path


def build_pareto_front_csv_safe(
    run_dir: Path | str,
    specs: Optional[Sequence[Tuple[str, str]]] = None,
    min_progress: Optional[float] = None,
) -> Optional[Path]:
    """``build_pareto_front_csv`` that never raises.

    Used by the live writer inside ``NSGA2MorphCMAES``: a results artifact
    must not take down a multi-hour evolution run.
    """
    try:
        return build_pareto_front_csv(run_dir, specs, min_progress)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[pareto] Front CSV update failed ({type(exc).__name__}: "
              f"{exc}) — continuing")
        return None


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
        print("Usage: python -m WP2_Outer_Loop.pareto_fronts <run_dir> "
              "[--min-progress X]")
        sys.exit(1)
    sys.exit(0 if build_pareto_front_csv(argv[0], min_progress=_min_progress)
             else 1)
