"""
Cross-run overlay of cumulative exam Pareto fronts, with group means.
=====================================================================

Draws the cumulative exam front of every run in one or more *groups*
(e.g. the 8 co-design runs against the 8 morphology-only runs) as faint
lines, and one full-strength line per group: the **mean attainment curve**.

Definitions (identical to the batch briefs and to ``pareto_fronts``):

* **Cumulative front** of a run: the non-dominated set of every exam-scored
  row (``obj_source == "exam"``) with progress at or above the run's
  ``outer.min_progress_m`` admission gate, pooled over *all* outer
  generations — one line per run, not per phase.
* **Attainment curve** of a front: ``CoT(P)`` = the lowest cost of transport
  among front points whose progress is at least ``P``; undefined (NaN)
  beyond the front tip. This is the "CoT at 150 / 170 / 190 m" quantity
  already quoted in the briefs, evaluated on a fine grid.
* **Mean attainment** of a group: the arithmetic mean of the members'
  attainment curves at each grid point, over the members whose front
  *covers* P (has a point at or below and one at or above P). Drawn
  **solid** where at least ``--solid-min-runs`` members do (default 4, i.e.
  half of an 8-run group) and **dashed** where fewer do: the head (some
  runs have no cheap arm) always, the tail (some runs have ended) only in
  the ``_tail`` figure. One rule at both ends, so a dashed segment always
  means "fewer than the threshold".
* **Star**: the zero-rules WP1 generalist on the standard drone, averaged
  over every phase of every run's ``outer_exam_baseline.csv`` (the
  in-run exam baseline is the same to within 0.2 m across runs).
* **Gap area** (two groups only): the signed area between the two solid
  means, ``second − first`` in CoT integrated over progress (trapezoids on
  the grid) wherever *both* means are solid; dashed stretches never count.
  Positive where the first group is cheaper (shaded orange), negative where
  the second is (shaded light blue); the net is printed on the figure.

Usage
-----
::

    PYTHONPATH=src python -m WP2_Outer_Loop.pareto_overlay \\
        --group co-design red \\
            logs/remote/outer_nsga/outer_exam_4_64_64_300_r* \\
            logs/remote/outer_nsga/outer_exam_6_64_64_300_r* \\
            logs/remote/outer_nsga/outer_exam_4_64_64_300_extra_r* \\
        --group morphology-only blue \\
            logs/remote/outer_nsga/outer_morphology_only_exam*_r* \\
        --out logs/remote/outer_nsga/pareto_overlay/exam_fronts.png

Each ``--group`` takes ``LABEL COLOR RUN_DIR [RUN_DIR ...]``; a run dir may
be the timestamped run folder itself or the synced ``outer_<exp>_rX``
wrapper that contains exactly one. Writes ``<out>.png``, ``<out>.pdf``,
``<stem>_tail.{png,pdf}`` (same, with the mean also continued dashed past
the earliest tip), ``<stem>_area.{png,pdf}`` (the tail figure with the
gap between the two solid means shaded and its net area written on it;
only with exactly two groups) and ``<stem>_mean_attainment.csv`` (grid;
per group the solid mean, the covering-runs mean and the covering-run
count). ``--no-bixler-subdir [NAME]`` additionally re-plots every figure
without the star into ``NAME/`` (default ``no_bixler``) beside the
originals, for a caption that does not need the reference drone.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from WP2_Outer_Loop.pareto_fronts import (
    BIXLER_LABEL,
    _PROGRESS_NAMES,
    _admission_mask,
    _filter_exam_rows,
    _load_min_progress,
    _load_objective_specs,
    _maximization_points,
    _nondominated_mask,
    _objective_columns,
)

_COT_NAMES = ("cost_of_transport", "cot")


# ----------------------------------------------------------------------------
#  Data
# ----------------------------------------------------------------------------

def resolve_run_dir(path: Path | str) -> Path:
    """The folder holding ``results/outer_population.csv``: ``path`` itself,
    or its single child that has one (the synced ``outer_<exp>_rX`` wrapper
    around a timestamped run). Raises ``FileNotFoundError`` otherwise."""
    path = Path(path)
    if (path / "results" / "outer_population.csv").is_file():
        return path
    if path.is_dir():
        hits = sorted(
            p for p in path.iterdir()
            if (p / "results" / "outer_population.csv").is_file()
        )
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise FileNotFoundError(
                f"{path}: {len(hits)} run folders inside, pass one explicitly: "
                + ", ".join(h.name for h in hits)
            )
    raise FileNotFoundError(
        f"{path}: no results/outer_population.csv here or in a child folder"
    )


def cumulative_front(
    run_dir: Path | str, min_progress: Optional[float] = None,
) -> np.ndarray:
    """``(n, 2)`` array of ``[progress_m, cost_of_transport]`` front points,
    sorted by progress ascending. Exam rows only, gated at ``min_progress``
    (the run's own ``outer.min_progress_m`` when ``None``), non-dominated
    across every outer generation."""
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "results" / "outer_population.csv")
    df, _exam_only = _filter_exam_rows(df)
    specs = _load_objective_specs(run_dir)
    obj_cols = _objective_columns(df, specs)
    names = [n for _, n, _ in obj_cols]
    prog = next((c for c, n, _ in obj_cols if n in _PROGRESS_NAMES), None)
    cot = next((c for c, n, _ in obj_cols if n in _COT_NAMES), None)
    if len(obj_cols) != 2 or prog is None or cot is None:
        raise ValueError(
            f"{run_dir}: overlay needs the progress/cost-of-transport "
            f"objective pair, got {names}"
        )
    gate = _load_min_progress(run_dir) if min_progress is None else min_progress
    df = df[_admission_mask(df, specs, gate)].reset_index(drop=True)
    if df.empty:
        return np.zeros((0, 2))
    pts = _maximization_points(df, obj_cols)
    finite = np.isfinite(pts).all(axis=1)
    keep = np.zeros(len(df), dtype=bool)
    keep[np.where(finite)[0][_nondominated_mask(pts[finite])]] = True
    front = df.loc[keep, [prog, cot]].to_numpy(dtype=float)
    return front[np.argsort(front[:, 0], kind="stable")]


def exam_star(run_dirs: Sequence[Path | str]) -> Optional[Tuple[float, float]]:
    """``(progress_m, cost_of_transport)`` of the zero-rules generalist on
    the standard drone: mean over every phase of every run's
    ``results/outer_exam_baseline.csv``. ``None`` if no run has one."""
    frames = []
    for rd in run_dirs:
        p = Path(rd) / "results" / "outer_exam_baseline.csv"
        if p.is_file():
            frames.append(pd.read_csv(p)[["progress_m", "cost_of_transport"]])
    if not frames:
        return None
    allb = pd.concat(frames, ignore_index=True).dropna()
    if allb.empty:
        return None
    return float(allb["progress_m"].mean()), float(allb["cost_of_transport"].mean())


# ----------------------------------------------------------------------------
#  Attainment
# ----------------------------------------------------------------------------

def attainment_curve(front: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """``CoT(P)`` for each ``P`` in ``grid``: the lowest cost of transport
    among front points with progress >= P; NaN where no point reaches P."""
    front = np.asarray(front, dtype=float).reshape(-1, 2)
    grid = np.asarray(grid, dtype=float)
    out = np.full(grid.shape, np.nan)
    if front.shape[0] == 0:
        return out
    reaches = front[None, :, 0] >= grid[:, None]            # (G, n)
    cots = np.where(reaches, front[None, :, 1], np.inf)
    best = cots.min(axis=1)
    out[np.isfinite(best)] = best[np.isfinite(best)]
    return out


def _covering(fronts: Sequence[np.ndarray], grid: np.ndarray) -> np.ndarray:
    """``(R, G)`` bool: does run r's front *cover* grid point g, i.e. does it
    have a point at or below P and a point at or above P? Below its first
    point a run would only contribute its cheapest body (a flat extension),
    above its last point nothing at all; both are excluded so that the head
    and the tail of the mean follow one rule."""
    cov = []
    for f in fronts:
        f = np.asarray(f, dtype=float).reshape(-1, 2)
        if f.shape[0] == 0:
            cov.append(np.zeros(grid.shape, dtype=bool))
            continue
        cov.append((grid >= f[:, 0].min()) & (grid <= f[:, 0].max()))
    return np.stack(cov)


def partial_mean_attainment(
    fronts: Sequence[np.ndarray], grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(mean, n)``: mean attainment over the runs whose front covers each
    grid point, and how many they are. This is the dashed line at both ends
    of the group mean: before the last front has started and after the
    first one has ended it averages a subset of the group."""
    grid = np.asarray(grid, dtype=float)
    curves = np.stack([attainment_curve(f, grid) for f in fronts])
    cov = _covering(fronts, grid) & np.isfinite(curves)
    n = cov.sum(axis=0)
    total = np.where(cov, curves, 0.0).sum(axis=0)
    mean = np.where(n > 0, total / np.maximum(n, 1), np.nan)
    return mean, n


def _solid_threshold(n_runs: int, min_runs: Optional[int]) -> int:
    """Fronts that must cover P for the mean to count as solid: ``min_runs``
    clamped to the group size, or the whole group when ``None``."""
    return n_runs if min_runs is None else max(1, min(int(min_runs), n_runs))


def mean_attainment(
    fronts: Sequence[np.ndarray], grid: np.ndarray,
    min_runs: Optional[int] = None,
) -> np.ndarray:
    """The solid group mean: mean attainment over the fronts covering P,
    defined only where at least ``min_runs`` of them do (all of them when
    ``None``). NaN elsewhere, so the solid line never averages fewer runs
    than the caption promises."""
    mean, n = partial_mean_attainment(fronts, grid)
    return np.where(n >= _solid_threshold(len(fronts), min_runs), mean, np.nan)


@dataclass
class FrontGroup:
    label: str
    color: str
    fronts: List[np.ndarray]
    names: List[str] = field(default_factory=list)

    @property
    def tip(self) -> float:
        return max(float(f[:, 0].max()) for f in self.fronts if len(f))

    @property
    def shortest_tip(self) -> float:
        return min(float(f[:, 0].max()) for f in self.fronts if len(f))

    @property
    def latest_start(self) -> float:
        """Progress of the last front to start: the solid mean begins here."""
        return max(float(f[:, 0].min()) for f in self.fronts if len(f))


def attainment_grid(groups: Sequence[FrontGroup], step: float = 0.5) -> np.ndarray:
    """Shared progress grid: from the lowest front point (rounded down to
    ``step``) to the longest tip, inclusive."""
    lo = min(float(f[:, 0].min()) for g in groups for f in g.fronts if len(f))
    hi = max(g.tip for g in groups)
    lo = np.floor(lo / step) * step
    return np.arange(lo, hi + step / 2, step)


def mean_attainment_table(
    groups: Sequence[FrontGroup], grid: np.ndarray,
    min_runs: Optional[int] = None,
) -> pd.DataFrame:
    """``progress_m`` plus, per group label ``L``: ``L`` (the solid line:
    mean over the fronts covering P where at least ``min_runs`` do, NaN
    elsewhere), ``L_partial`` (the dashed line: the same mean wherever any
    front covers P) and ``L_n`` (how many those are)."""
    table = {"progress_m": grid}
    for g in groups:
        table[g.label] = mean_attainment(g.fronts, grid, min_runs)
        partial, n = partial_mean_attainment(g.fronts, grid)
        table[f"{g.label}_partial"] = partial
        table[f"{g.label}_n"] = n
    return pd.DataFrame(table)


@dataclass
class GapArea:
    """Signed area between two solid group means (see ``attainment_gap_area``).
    ``gap`` is ``second − first`` on the grid, NaN off the shared solid
    support ``[lo, hi]`` (both ``None`` when there is none); ``positive``
    (>= 0) and ``negative`` (<= 0) are the parts where the first group is
    cheaper / dearer, ``net`` their sum. Units: CoT × m."""
    gap: np.ndarray
    positive: float
    negative: float
    net: float
    lo: Optional[float]
    hi: Optional[float]


def _trapezoid_segments(grid: np.ndarray, y: np.ndarray) -> float:
    """Trapezoid-rule integral of ``y`` over ``grid``, summed over the runs
    of consecutive finite samples (a NaN breaks the integration)."""
    ok = np.isfinite(y[:-1]) & np.isfinite(y[1:])
    if not ok.any():
        return 0.0
    dx = np.diff(grid)[ok]
    return float((0.5 * (y[:-1][ok] + y[1:][ok]) * dx).sum())


def attainment_gap_area(
    groups: Sequence[FrontGroup], grid: np.ndarray,
    min_runs: Optional[int] = None,
) -> GapArea:
    """Signed area between the solid means of exactly two groups: the
    second group's mean attainment minus the first's, integrated over
    progress (trapezoids on ``grid``) only where **both** means are solid
    (each has at least ``min_runs`` fronts covering P). Positive = the first
    group is cheaper. Clipping each trapezoid at zero splits it into the
    positive and negative parts, whose sum is exactly the net."""
    if len(groups) != 2:
        raise ValueError(f"the gap area needs exactly two groups, got {len(groups)}")
    grid = np.asarray(grid, dtype=float)
    first = mean_attainment(groups[0].fronts, grid, min_runs)
    second = mean_attainment(groups[1].fronts, grid, min_runs)
    gap = second - first                                    # NaN where either is
    both = np.isfinite(gap)
    if not both.any():
        return GapArea(gap, 0.0, 0.0, 0.0, None, None)
    pos = _trapezoid_segments(grid, np.where(both, np.clip(gap, 0.0, None), np.nan))
    neg = _trapezoid_segments(grid, np.where(both, np.clip(gap, None, 0.0), np.nan))
    xs = grid[both]
    return GapArea(gap, pos, neg, pos + neg, float(xs.min()), float(xs.max()))


# ----------------------------------------------------------------------------
#  Figure
# ----------------------------------------------------------------------------

def draw_front_overlay(
    ax,
    groups: Sequence[FrontGroup],
    star: Optional[Tuple[float, float]] = None,
    grid_step: float = 0.5,
    member_alpha: float = 0.28,
    tail: bool = False,
    min_runs: Optional[int] = None,
    shade_gap: bool = False,
) -> None:
    """Draw the overlay onto ``ax``: faint per-run fronts, one group mean
    per group and the gold Bixler star, styled like
    ``pareto_plots.plot_pareto_front`` (dashed grid, framed legend).

    With ``shade_gap`` (exactly two groups) the area between the two solid
    means is filled orange where the first group is cheaper and light blue
    where the second is, and the net area (orange minus blue, see
    ``attainment_gap_area``) is written in the lower-right corner. Dashed
    stretches are never shaded or counted.

    The group mean averages the fronts that cover P. It is **solid** where
    at least ``min_runs`` fronts do (the whole group when ``None``) and
    **dashed** where fewer do: the head, before enough fronts have
    started, is always drawn; the tail, after too many have ended, only
    with ``tail=True``. Per-P run counts are in ``mean_attainment_table``.

    The legend lists only the mean lines and the star; the faint members and
    the dashed segments are described in the caption, not the box."""
    grid = attainment_grid(groups, grid_step)
    idx = np.arange(len(grid))
    dash = dict(lw=1.6, ls=(0, (4, 2.5)), zorder=3)
    if shade_gap:
        _shade_gap(ax, groups, grid, min_runs)
    for g in groups:
        for front in g.fronts:
            if len(front) == 0:
                continue
            ax.plot(front[:, 0], front[:, 1], color=g.color, lw=0.9,
                    alpha=member_alpha, zorder=2)
        partial, n = partial_mean_attainment(g.fronts, grid)
        full = n >= _solid_threshold(len(g.fronts), min_runs)
        solid = np.where(full, partial, np.nan)
        ax.plot(grid, solid, color=g.color, lw=2.4, zorder=4,
                label=f"{g.label} ({len(g.fronts)} runs)")
        # Dashed segments start/end on the first/last solid point so the
        # line is continuous; where nothing is fully covered, dash it all.
        first = int(idx[full][0]) if full.any() else len(grid)
        last = int(idx[full][-1]) if full.any() else -1
        head = (idx <= first) & (n > 0)
        if head.sum() > 1:  # a lone join point draws nothing: skip it
            ax.plot(grid, np.where(head, partial, np.nan), color=g.color, **dash)
        if tail:
            tail_mask = (idx >= last) & (n > 0)
            if tail_mask.sum() > 1:
                ax.plot(grid, np.where(tail_mask, partial, np.nan),
                        color=g.color, **dash)

    if star is not None:
        ax.scatter([star[0]], [star[1]], marker="*", s=340, color="gold",
                   edgecolors="black", linewidths=0.9, zorder=5,
                   label=BIXLER_LABEL)

    ax.set_xlabel("Exam progress [m]")
    ax.set_ylabel("Cost of transport")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)


GAP_COLORS = {"positive": "orange", "negative": "skyblue"}


def _shade_gap(ax, groups: Sequence[FrontGroup], grid: np.ndarray,
               min_runs: Optional[int]) -> GapArea:
    """Fill between the two solid means (orange: first group cheaper,
    light blue: second cheaper) and annotate the net area. Drawn below the
    lines; kept out of the legend."""
    area = attainment_gap_area(groups, grid, min_runs)
    first = mean_attainment(groups[0].fronts, grid, min_runs)
    second = mean_attainment(groups[1].fronts, grid, min_runs)
    both = np.isfinite(area.gap)
    fill = dict(alpha=0.35, lw=0, zorder=1, interpolate=True)
    ax.fill_between(grid, first, second, where=both & (area.gap > 0),
                    color=GAP_COLORS["positive"], **fill)
    ax.fill_between(grid, first, second, where=both & (area.gap < 0),
                    color=GAP_COLORS["negative"], **fill)
    support = (f" over {area.lo:.1f}–{area.hi:.1f} m"
               if area.lo is not None else " (no shared solid range)")
    ax.text(0.98, 0.03,
            f"orange: {groups[0].label} cheaper, blue: {groups[1].label} cheaper\n"
            f"net area (orange − blue) = {area.net:+.2f} CoT·m{support}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            zorder=6, bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                ec="0.6", alpha=0.9))
    return area


def plot_front_overlay(
    groups: Sequence[FrontGroup],
    out_path: Path | str,
    star: Optional[Tuple[float, float]] = None,
    grid_step: float = 0.5,
    title: Optional[str] = None,
    member_alpha: float = 0.28,
    tail: bool = False,
    min_runs: Optional[int] = None,
    shade_gap: bool = False,
) -> Path:
    """``draw_front_overlay`` on a fresh figure; saves ``out_path`` (PNG)
    and the same stem as PDF, returns ``out_path``."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_front_overlay(ax, groups, star=star, grid_step=grid_step,
                       member_alpha=member_alpha, tail=tail, min_runs=min_runs,
                       shade_gap=shade_gap)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def _parse_groups(parser: argparse.ArgumentParser, raw: Optional[List[List[str]]],
                  min_progress: Optional[float]) -> List[FrontGroup]:
    if not raw:
        parser.error("at least one --group LABEL COLOR RUN_DIR... is required")
    groups: List[FrontGroup] = []
    for tokens in raw:
        if len(tokens) < 3:
            parser.error(f"--group needs LABEL COLOR RUN_DIR [RUN_DIR ...], "
                         f"got {tokens}")
        label, color, *dirs = tokens
        fronts, names = [], []
        for d in dirs:
            rd = resolve_run_dir(d)
            fronts.append(cumulative_front(rd, min_progress))
            wrapper = rd.parent.name if rd.parent.name.startswith("outer_") else rd.name
            names.append(wrapper)
        groups.append(FrontGroup(label, color, fronts, names))
    return groups


def _report(groups: Sequence[FrontGroup], min_runs: Optional[int],
            levels=(150.0, 170.0, 190.0), grid_step: float = 0.5) -> None:
    lv = np.array(levels, dtype=float)
    grid = attainment_grid(groups, grid_step)
    for g in groups:
        k = _solid_threshold(len(g.fronts), min_runs)
        solid = mean_attainment(g.fronts, grid, min_runs)
        xs = grid[np.isfinite(solid)]
        rng = f"{xs.min():.1f}–{xs.max():.1f} m" if xs.size else "nowhere"
        print(f"[overlay] {g.label}: {len(g.fronts)} runs, solid where ≥ {k} "
              f"fronts cover P: {rng}; all {len(g.fronts)} cover "
              f"{g.latest_start:.1f}–{g.shortest_tip:.1f} m")
        for name, f in zip(g.names, g.fronts):
            cot = attainment_curve(f, lv)
            cells = "  ".join(f"CoT@{int(p)}={c:.3f}" for p, c in zip(lv, cot))
            print(f"    {name:<44s} tip={f[:, 0].max():6.1f} m  {cells}")
        mean, n = partial_mean_attainment(g.fronts, lv)
        cells = "  ".join(f"CoT@{int(p)}={c:.3f}({k}/{len(g.fronts)})"
                          for p, c, k in zip(lv, mean, n))
        print(f"    {'mean attainment (runs covering P)':<44s} {'':12s}{cells}")


def _write_figures(
    groups: Sequence[FrontGroup], out: Path, star: Optional[Tuple[float, float]],
    *, grid_step: float, title: Optional[str], min_runs: Optional[int],
) -> List[Path]:
    """The overlay at ``out``, its ``_tail`` sibling and, with exactly two
    groups, the shaded ``_area`` figure (each as PNG + PDF); returns the
    PNG paths."""
    common = dict(star=star, grid_step=grid_step, title=title, min_runs=min_runs)
    written = [
        plot_front_overlay(groups, out, **common),
        plot_front_overlay(groups, out.with_name(f"{out.stem}_tail{out.suffix}"),
                           tail=True, **common),
    ]
    if len(groups) == 2:
        written.append(plot_front_overlay(
            groups, out.with_name(f"{out.stem}_area{out.suffix}"),
            tail=True, shade_gap=True, **common))
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Overlay cumulative exam Pareto fronts of run groups "
                    "with per-group mean attainment curves.")
    parser.add_argument("--group", action="append", nargs="+",
                        metavar="TOKEN",
                        help="LABEL COLOR RUN_DIR [RUN_DIR ...]; repeatable")
    parser.add_argument("--out", required=True, type=Path,
                        help="output PNG path (PDF + mean CSV written beside it)")
    parser.add_argument("--min-progress", type=float, default=None,
                        help="override the runs' admission gate [m]")
    parser.add_argument("--grid-step", type=float, default=0.5,
                        help="attainment grid step [m] (default 0.5)")
    parser.add_argument("--no-star", action="store_true",
                        help="omit the WP1 generalist exam baseline star")
    parser.add_argument("--no-bixler-subdir", nargs="?", const="no_bixler",
                        default=None, metavar="NAME",
                        help="also re-plot every figure without the star into "
                             "NAME/ beside the originals (default name "
                             "no_bixler); skipped when no star is drawn")
    parser.add_argument("--solid-min-runs", type=int, default=4,
                        help="draw the group mean solid where at least this "
                             "many fronts cover P, dashed below (default 4; "
                             "clamped to the group size)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")  # headless CLI: never probe a GUI backend

    groups = _parse_groups(parser, args.group, args.min_progress)
    star = None
    if not args.no_star:
        run_dirs = [resolve_run_dir(d) for tokens in args.group for d in tokens[2:]]
        star = exam_star(run_dirs)
        if star is None:
            print("[overlay] no outer_exam_baseline.csv found: star omitted")
        else:
            print(f"[overlay] star: {star[0]:.1f} m, CoT {star[1]:.3f}")

    _report(groups, args.solid_min_runs, grid_step=args.grid_step)
    grid = attainment_grid(groups, args.grid_step)
    if len(groups) == 2:
        area = attainment_gap_area(groups, grid, args.solid_min_runs)
        rng = (f"{area.lo:.1f}–{area.hi:.1f} m" if area.lo is not None
               else "no shared solid range")
        print(f"[overlay] gap between solid means ({groups[1].label} − "
              f"{groups[0].label}) over {rng}: {groups[0].label} cheaper "
              f"{area.positive:+.3f}, {groups[1].label} cheaper "
              f"{area.negative:+.3f}, net area {area.net:+.2f} CoT·m")
    else:
        print(f"[overlay] area figure skipped: needs exactly two groups, "
              f"got {len(groups)}")
    out = Path(args.out)
    style = dict(grid_step=args.grid_step, title=args.title,
                 min_runs=args.solid_min_runs)
    written = _write_figures(groups, out, star, **style)
    if args.no_bixler_subdir:
        if star is None:
            print("[overlay] star-free copies skipped: no star drawn")
        else:
            sub = out.parent / args.no_bixler_subdir / out.name
            copies = _write_figures(groups, sub, None, **style)
            print(f"[overlay] wrote the same figures without the star to "
                  f"{sub.parent}: {', '.join(p.name for p in copies)} (+ .pdf)")
    csv_path = out.with_name(f"{out.stem}_mean_attainment.csv")
    mean_attainment_table(groups, grid, args.solid_min_runs).to_csv(csv_path, index=False)
    print(f"[overlay] wrote {', '.join(str(p) for p in written)} (+ .pdf), {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
