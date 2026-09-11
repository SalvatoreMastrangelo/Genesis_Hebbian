"""
Cross-run aggregate of exam hypervolumes: mean ± std per run group.
===================================================================

One curve per *group* (e.g. the 8 full co-design runs against the 8
morphology-only runs) of the exam-front hypervolume against the outer
phase, with a mean ± 1 sample-std band over the group's runs. Two
measures, drawn as two panels by default:

* **per-phase**: hypervolume of the exam front of that phase alone — the
  64 morphologies the phase-end exam scored, gated at
  ``outer.min_progress_m``, non-dominated within the phase;
* **cumulative**: hypervolume of the non-dominated set pooled over every
  exam phase up to and including that one (monotone non-decreasing; its
  last value is the "cumulative exam HV" quoted in the batch briefs).

Both are computed exactly as each run's own ``plots/pareto_hypervolume.png``
(``pareto_plots.plot_pareto_front``): exam rows only, the run's admission
gate, the fixed (80 m, CoT 0.5) reference from ``pareto_fronts``. The
x-axis is, by default, the outer phase index (``outer_gen`` of the exam
rows): one NSGA-II URDF refresh per phase in both conditions, so equal x
means an equal number of morphologies evaluated — not equal inner
generations (co-design runs 4–6 per phase, morphology-only 1) nor equal
wall-clock. ``--x-axis fraction`` instead normalises every run by its own
length (``phase / final phase``, 0 = first exam, 1 = last), so groups with
different phase counts line up start-to-end: each run is linearly
interpolated onto a common 201-point grid and every run then contributes
at every x (the mean is solid throughout; the legend states the phase
count behind "1"). ``--split-panels`` additionally writes each panel as
its own single-panel figure ``<stem>_per_phase`` / ``<stem>_cumulative``.

Groups may have different lengths (co-design 49 phases, morphology-only
87; the two ``exam_6`` co-design runs end at phase 32): each mean is over
the runs that reached the phase, the band needs at least two. The mean is
drawn **solid** where at least ``--solid-min-runs`` runs contribute (the
whole group by default) and **dashed** where fewer do, so a dashed tail
always means "some runs have ended" — the same rule as ``pareto_overlay``.

Usage
-----
::

    PYTHONPATH=src python -m WP2_Outer_Loop.hypervolume_aggregate \\
        --group co-design "#d62728" \\
            logs/remote/outer_nsga/outer_exam_4_64_64_300_r* \\
            logs/remote/outer_nsga/outer_exam_6_64_64_300_r* \\
            logs/remote/outer_nsga/outer_exam_4_64_64_300_extra_r* \\
        --group morphology-only "#1f77b4" \\
            logs/remote/outer_nsga/outer_morphology_only_exam*_r* \\
        --out logs/remote/outer_nsga/pareto_overlay/exam_hv_aggregate.png

Each ``--group`` takes ``LABEL COLOR RUN_DIR [RUN_DIR ...]``; a run dir may
be the timestamped run folder itself or the synced ``outer_<exp>_rX``
wrapper that contains exactly one. Writes ``<out>.png``, ``<out>.pdf`` and
``<stem>_aggregate.csv`` (per x, per group and measure: mean, std, n).
``--show-runs`` adds every run as a faint line; ``--measure`` picks one
panel instead of both; ``--x-axis fraction`` normalises each run to its
own length; ``--split-panels`` also saves each panel on its own;
``--solid-min-runs`` lowers the run count needed for a solid mean.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from WP2_Outer_Loop.pareto_fronts import (
    _admission_mask,
    _filter_exam_rows,
    _hv_reference,
    _hypervolume_2d,
    _load_min_progress,
    _load_objective_specs,
    _maximization_points,
    _nondominated_mask,
    _objective_columns,
)
from WP2_Outer_Loop.pareto_overlay import resolve_run_dir

MEASURES: Tuple[str, ...] = ("hv_per_phase", "hv_cumulative")
_MEASURE_TITLE: Dict[str, str] = {
    "hv_per_phase": "Per-phase exam front",
    "hv_cumulative": "Cumulative exam front",
}
_MEASURE_CLI: Dict[str, Tuple[str, ...]] = {
    "both": MEASURES,
    "per-phase": ("hv_per_phase",),
    "cumulative": ("hv_cumulative",),
}
X_AXES: Tuple[str, ...] = ("phase", "fraction")
FRACTION_GRID_POINTS = 201
_X_LABEL: Dict[str, str] = {
    "phase": "Outer phase",
    "fraction": "Fraction of run (phase / final phase)",
}


# ----------------------------------------------------------------------------
#  Per-run series
# ----------------------------------------------------------------------------

def hv_series(
    run_dir: Path | str, min_progress: Optional[float] = None,
) -> pd.DataFrame:
    """``phase, hv_per_phase, hv_cumulative`` per exam phase of one run.

    Mirrors ``pareto_plots.plot_pareto_front``: exam rows only
    (``_filter_exam_rows``), the first two configured objectives, admission
    gated at ``min_progress`` (the run's ``outer.min_progress_m`` when
    ``None``; ungated with a message if the gate admits nothing), hypervolume
    against ``pareto_fronts._hv_reference`` — fixed at (80 m, CoT 0.5) for
    the progress/CoT pair, so cross-run comparable. Rows with a non-finite
    objective are ignored. The cumulative pool is carried as its running
    front, which is the same non-dominated set as pooling every row.

    The reference is recorded in ``df.attrs['ref']`` (raw objective space)
    and ``df.attrs['ref_fixed']``."""
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "results" / "outer_population.csv")
    df, _exam_only = _filter_exam_rows(df)
    specs = _load_objective_specs(run_dir)
    obj_cols = _objective_columns(df, specs)
    if len(obj_cols) < 2:
        raise ValueError(
            f"{run_dir}: hypervolume needs two objectives, got "
            f"{[n for _, n, _ in obj_cols]}"
        )
    obj_cols = obj_cols[:2]
    specs = [(n, d) for _, n, d in obj_cols]
    pts_max = _maximization_points(df, obj_cols)
    finite = np.isfinite(pts_max).all(axis=1)
    gate = _load_min_progress(run_dir) if min_progress is None else min_progress
    admit = _admission_mask(df, specs, gate) & finite
    if not admit.any():
        print(f"[hv_aggregate] {run_dir.name}: min_progress={float(gate):g} "
              f"excludes every exam point — computing ungated")
        admit = finite
    ref, ref_fixed = _hv_reference(specs, pts_max[admit])
    if not ref_fixed:
        print(f"[hv_aggregate] {run_dir.name}: run-relative hypervolume "
              f"reference — NOT comparable across runs")
    sign = np.array([1.0 if d == "maximize" else -1.0 for _, d in specs])

    gens = df["outer_gen"].to_numpy(dtype=int)
    phases, per_phase, cumulative = [], [], []
    running = np.empty((0, 2))
    for g in np.unique(gens):
        sel = (gens == g) & admit
        p = pts_max[sel]
        front = p[_nondominated_mask(p)] if len(p) else p
        pool = np.vstack([running, front])
        running = pool[_nondominated_mask(pool)] if len(pool) else pool
        phases.append(int(g))
        per_phase.append(_hypervolume_2d(front, ref))
        cumulative.append(_hypervolume_2d(running, ref))
    out = pd.DataFrame({"phase": phases, "hv_per_phase": per_phase,
                        "hv_cumulative": cumulative})
    out.attrs["ref"] = (ref * sign).tolist()
    out.attrs["ref_fixed"] = bool(ref_fixed)
    out.attrs["objectives"] = [n for n, _ in specs]
    return out


def fraction_of_run(series: pd.DataFrame) -> np.ndarray:
    """``phase / final phase`` of one ``hv_series`` frame, in [0, 1]: 0 is
    the run's first exam phase, 1 its last, whatever the run's length. A
    run with a single phase sits at 0."""
    phase = series["phase"].to_numpy(dtype=float)
    last = phase.max() if len(phase) else 0.0
    return phase / last if last > 0 else np.zeros_like(phase)


def fraction_grid(n_points: int = FRACTION_GRID_POINTS) -> np.ndarray:
    """The common fraction-of-run grid: ``n_points`` evenly spaced values
    from 0 to 1 inclusive."""
    return np.linspace(0.0, 1.0, int(n_points))


# ----------------------------------------------------------------------------
#  Group aggregate
# ----------------------------------------------------------------------------

def aggregate_hv(
    series: Sequence[pd.DataFrame],
    measure: str,
    x_axis: str = "phase",
    grid: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """``<x_axis>, mean, std, n`` of ``measure`` across ``series`` (one
    ``hv_series`` frame per run).

    ``x_axis='phase'``: over the union of phases; the mean and the sample
    std (ddof=1, NaN with a single run) are taken over the runs that have
    the phase, ``n`` counts them.

    ``x_axis='fraction'``: each run is re-indexed by ``fraction_of_run``
    (its phase over its own final phase) and linearly interpolated onto
    ``grid`` (``fraction_grid()`` when ``None``), so runs of different
    lengths line up start-to-end; every run then contributes at every grid
    point (``n`` is the run count) and mean / std are over the interpolated
    values. The grid holds 0 and 1 exactly, so the first and last values
    are the runs' own, not interpolated."""
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")
    if x_axis not in X_AXES:
        raise ValueError(f"x_axis must be one of {X_AXES}, got {x_axis!r}")
    if x_axis == "phase":
        wide = pd.concat(
            [s.set_index("phase")[measure].rename(i) for i, s in enumerate(series)],
            axis=1,
        ).sort_index()
        x = wide.index.to_numpy(dtype=int)
    else:
        x = fraction_grid() if grid is None else np.asarray(grid, dtype=float)
        cols = []
        for s in series:
            if len(s) == 0:
                cols.append(np.full(len(x), np.nan))
                continue
            order = np.argsort(fraction_of_run(s), kind="stable")
            cols.append(np.interp(x, fraction_of_run(s)[order],
                                  s[measure].to_numpy(dtype=float)[order]))
        wide = pd.DataFrame(np.column_stack(cols), index=x)
    return pd.DataFrame({
        x_axis: x,
        "mean": wide.mean(axis=1, skipna=True).to_numpy(),
        "std": wide.std(axis=1, ddof=1, skipna=True).to_numpy(),
        "n": wide.notna().sum(axis=1).to_numpy(dtype=int),
    })


@dataclass
class HVGroup:
    label: str
    color: str
    series: List[pd.DataFrame]
    names: List[str] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.series)

    @property
    def last_phase(self) -> int:
        return max(int(s["phase"].max()) for s in self.series if len(s))

    @property
    def last_common_phase(self) -> int:
        """Last phase every run of the group reached."""
        return min(int(s["phase"].max()) for s in self.series if len(s))

    @property
    def phase_count_label(self) -> str:
        """``'49 phases'`` when every run has the same number of phases
        (final phase + 1), ``'33–49 phases'`` otherwise."""
        lo, hi = self.last_common_phase + 1, self.last_phase + 1
        return f"{lo} phases" if lo == hi else f"{lo}–{hi} phases"


def legend_label(group: HVGroup, x_axis: str = "phase") -> str:
    """``'co-design (8 runs)'``; on the fraction axis the phase count the
    runs were normalised by is appended: ``'co-design (8 runs, 33–49
    phases)'``."""
    if x_axis == "fraction":
        return f"{group.label} ({group.n_runs} runs, {group.phase_count_label})"
    return f"{group.label} ({group.n_runs} runs)"


def aggregate_table(groups: Sequence[HVGroup], x_axis: str = "phase") -> pd.DataFrame:
    """``<x_axis>`` plus, per group label ``L`` and measure ``M``:
    ``L_M_mean``, ``L_M_std``, ``L_M_n`` — over the union of all phases
    (NaN / 0 where a group has ended) or over the common fraction grid."""
    cols: Dict[str, pd.Series] = {}
    for g in groups:
        for m in MEASURES:
            agg = aggregate_hv(g.series, m, x_axis).set_index(x_axis)
            for stat in ("mean", "std", "n"):
                cols[f"{g.label}_{m}_{stat}"] = agg[stat]
    table = pd.DataFrame(cols).sort_index()
    for c in table.columns:
        if c.endswith("_n"):
            table[c] = table[c].fillna(0).astype(int)
    return table.rename_axis(x_axis).reset_index()


def _reference_note(groups: Sequence[HVGroup]) -> Optional[str]:
    """``'ref: progress_m = 80, cost_of_transport = 0.5'`` when every run
    shares one fixed reference, else ``None``."""
    refs = {
        (tuple(s.attrs.get("objectives", ())), tuple(s.attrs.get("ref", ())),
         bool(s.attrs.get("ref_fixed", False)))
        for g in groups for s in g.series
    }
    if len(refs) != 1:
        return None
    names, ref, fixed = next(iter(refs))
    if not fixed or not names:
        return None
    return "ref: " + ", ".join(f"{n} = {r:g}" for n, r in zip(names, ref))


# ----------------------------------------------------------------------------
#  Figure
# ----------------------------------------------------------------------------

def _solid_threshold(n_runs: int, min_runs: Optional[int]) -> int:
    """Runs that must contribute for the mean to count as solid:
    ``min_runs`` clamped to the group size, or the whole group when
    ``None``."""
    return n_runs if min_runs is None else max(1, min(int(min_runs), n_runs))


def draw_hv_aggregate(
    ax,
    groups: Sequence[HVGroup],
    measure: str,
    show_runs: bool = False,
    member_alpha: float = 0.25,
    min_runs: Optional[int] = None,
    x_axis: str = "phase",
) -> None:
    """One mean line per group with a mean ± 1 std band (drawn wherever
    the group has ≥ 2 runs at the x), optionally every run as a faint
    line, styled like ``pareto_plots`` (dashed grid, framed legend).

    The mean is **solid** where at least ``min_runs`` runs contribute (the
    whole group when ``None``) and **dashed** where fewer do — after the
    shortest runs have ended. The dashed segment starts on the last solid
    point so the line is continuous. The legend lists only the means, with
    the run count; the band and the dashed tail are for the caption.

    ``x_axis='fraction'`` draws every run and the aggregate against
    ``fraction_of_run`` (see ``aggregate_hv``): all runs then span 0–1, so
    the mean is solid throughout and the legend adds the phase count."""
    if x_axis not in X_AXES:
        raise ValueError(f"x_axis must be one of {X_AXES}, got {x_axis!r}")
    dash = dict(lw=1.6, ls=(0, (4, 2.5)), zorder=3)
    for g in groups:
        if show_runs:
            for s in g.series:
                xs = s["phase"] if x_axis == "phase" else fraction_of_run(s)
                ax.plot(xs, s[measure], color=g.color, lw=0.8,
                        alpha=member_alpha, zorder=2)
        agg = aggregate_hv(g.series, measure, x_axis)
        phase, mean, n = agg[x_axis].to_numpy(), agg["mean"].to_numpy(), agg["n"].to_numpy()
        lo, hi = mean - agg["std"].to_numpy(), mean + agg["std"].to_numpy()
        ax.fill_between(phase, lo, hi, color=g.color, alpha=0.18,
                        lw=0, zorder=1, label="_nolegend_")
        full = n >= _solid_threshold(g.n_runs, min_runs)
        ax.plot(phase, np.where(full, mean, np.nan), color=g.color, lw=2.2,
                zorder=4, label=legend_label(g, x_axis))
        partial = (~full) & (n > 0)
        if partial.any():
            # Extend the dashed mask by one point into the solid run on
            # each side so the two segments join.
            joined = partial.copy()
            joined[1:] |= partial[:-1]
            joined[:-1] |= partial[1:]
            joined &= n > 0
            ax.plot(phase, np.where(joined, mean, np.nan), color=g.color, **dash)
    ax.set_xlabel(_X_LABEL[x_axis])
    ax.set_ylabel("Exam hypervolume (↑ better)")
    ax.set_title(_MEASURE_TITLE[measure])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")


DEFAULT_TITLE = "Exam-front hypervolume, mean ± 1 std over runs"


def build_figure(
    groups: Sequence[HVGroup],
    measures: Sequence[str] = MEASURES,
    show_runs: bool = False,
    title: Optional[str] = None,
    member_alpha: float = 0.25,
    min_runs: Optional[int] = None,
    x_axis: str = "phase",
):
    """The figure: one panel per measure (shared y), a bare suptitle
    (``DEFAULT_TITLE`` unless ``title``; the hypervolume reference is
    reported on stdout, not in the title). Returns the matplotlib
    figure, unsaved."""
    import matplotlib.pyplot as plt

    n = len(measures)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n + 0.8, 4.6), sharey=True,
                             squeeze=False)
    for ax, m in zip(axes[0], measures):
        draw_hv_aggregate(ax, groups, m, show_runs=show_runs,
                          member_alpha=member_alpha, min_runs=min_runs,
                          x_axis=x_axis)
    for ax in axes[0][1:]:
        ax.set_ylabel("")
        ax.get_legend().remove()
    fig.suptitle(title or DEFAULT_TITLE)
    fig.tight_layout()
    return fig


def split_panel_path(out_path: Path | str, measure: str) -> Path:
    """``<stem>_per_phase.png`` / ``<stem>_cumulative.png`` beside
    ``out_path``."""
    out_path = Path(out_path)
    suffix = measure[len("hv_"):] if measure.startswith("hv_") else measure
    return out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix}")


def plot_hv_aggregate(
    groups: Sequence[HVGroup],
    out_path: Path | str,
    measures: Sequence[str] = MEASURES,
    show_runs: bool = False,
    title: Optional[str] = None,
    member_alpha: float = 0.25,
    min_runs: Optional[int] = None,
    x_axis: str = "phase",
    split_panels: bool = False,
) -> Path:
    """One panel per measure (shared y), saved as ``out_path`` (PNG) and
    the same stem as PDF, plus ``<stem>_aggregate.csv``
    (``aggregate_table`` on ``x_axis``). ``min_runs`` / ``x_axis`` as in
    ``draw_hv_aggregate``. ``split_panels`` also saves each measure as a
    single-panel figure at ``split_panel_path`` (PNG + PDF, no extra CSV)
    when there is more than one measure. Returns ``out_path``."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(show_runs=show_runs, title=title, member_alpha=member_alpha,
              min_runs=min_runs, x_axis=x_axis)
    targets = [(tuple(measures), out_path)]
    if split_panels and len(measures) > 1:
        targets += [((m,), split_panel_path(out_path, m)) for m in measures]
    for ms, path in targets:
        fig = build_figure(groups, ms, **kw)
        fig.savefig(path, dpi=200)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
    csv_path = out_path.with_name(f"{out_path.stem}_aggregate.csv")
    aggregate_table(groups, x_axis).to_csv(csv_path, index=False)
    return out_path


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def _parse_groups(parser: argparse.ArgumentParser, raw: Optional[List[List[str]]],
                  min_progress: Optional[float]) -> List[HVGroup]:
    if not raw:
        parser.error("at least one --group LABEL COLOR RUN_DIR... is required")
    groups: List[HVGroup] = []
    for tokens in raw:
        if len(tokens) < 3:
            parser.error(f"--group needs LABEL COLOR RUN_DIR [RUN_DIR ...], "
                         f"got {tokens}")
        label, color, *dirs = tokens
        series, names = [], []
        for d in dirs:
            rd = resolve_run_dir(d)
            series.append(hv_series(rd, min_progress))
            wrapper = rd.parent.name if rd.parent.name.startswith("outer_") else rd.name
            names.append(wrapper)
        groups.append(HVGroup(label, color, series, names))
    return groups


def _fmt(agg: pd.DataFrame, phase: float, x_axis: str = "phase") -> str:
    row = agg[np.isclose(agg[x_axis].to_numpy(dtype=float), float(phase))]
    if row.empty:
        return "—"
    m, s, n = float(row["mean"].iloc[0]), float(row["std"].iloc[0]), int(row["n"].iloc[0])
    sd = f" ± {s:.2f}" if np.isfinite(s) else ""
    return f"{m:.2f}{sd} (n={n})"


def _report(groups: Sequence[HVGroup], min_runs: Optional[int] = None,
            x_axis: str = "phase") -> None:
    note = _reference_note(groups)
    print(f"[hv_aggregate] {note}" if note else
          "[hv_aggregate] WARNING: hypervolume references differ across runs "
          "or are run-relative — curves NOT comparable")
    for g in groups:
        k = _solid_threshold(g.n_runs, min_runs)
        agg = aggregate_hv(g.series, "hv_cumulative")
        solid = agg.loc[agg["n"] >= k, "phase"]
        solid_to = int(solid.max()) if len(solid) else -1
        dashed = (f", dashed {solid_to + 1}–{g.last_phase} (< {k} runs)"
                  if solid_to < g.last_phase else "")
        print(f"[hv_aggregate] {g.label}: {g.n_runs} runs, phases 0–{g.last_phase}"
              f" (all runs reach {g.last_common_phase}); mean solid 0–{solid_to}"
              f"{dashed}")
        for m in MEASURES:
            agg = aggregate_hv(g.series, m)
            marks = sorted({0, 10, 20, 30, g.last_common_phase, g.last_phase})
            marks = [p for p in marks if p <= g.last_phase]
            cells = "  ".join(f"@{p}: {_fmt(agg, p)}" for p in marks)
            print(f"    {m:<14s} {cells}")
        if x_axis == "fraction":
            print(f"    fraction of run ({g.phase_count_label}):")
            for m in MEASURES:
                agg = aggregate_hv(g.series, m, "fraction")
                cells = "  ".join(f"@{f:.2f}: {_fmt(agg, f, 'fraction')}"
                                  for f in (0.0, 0.25, 0.5, 0.75, 1.0))
                print(f"    {m:<14s} {cells}")
        for name, s in zip(g.names, g.series):
            print(f"    {name:<44s} final per-phase {s['hv_per_phase'].iloc[-1]:6.2f}"
                  f"  cumulative {s['hv_cumulative'].iloc[-1]:6.2f}"
                  f"  ({len(s)} phases)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mean ± std of exam-front hypervolume vs outer phase, "
                    "one curve per run group.")
    parser.add_argument("--group", action="append", nargs="+",
                        metavar="TOKEN",
                        help="LABEL COLOR RUN_DIR [RUN_DIR ...]; repeatable")
    parser.add_argument("--out", required=True, type=Path,
                        help="output PNG path (PDF + aggregate CSV written beside it)")
    parser.add_argument("--min-progress", type=float, default=None,
                        help="override the runs' admission gate [m]")
    parser.add_argument("--measure", choices=sorted(_MEASURE_CLI), default="both",
                        help="which panel(s) to draw (default both)")
    parser.add_argument("--show-runs", action="store_true",
                        help="also draw every run as a faint line")
    parser.add_argument("--x-axis", choices=X_AXES, default="phase",
                        help="'phase': outer phase index (default); "
                             "'fraction': each run normalised by its own "
                             "final phase, 0–1, interpolated onto a common "
                             "grid so different-length runs line up")
    parser.add_argument("--split-panels", action="store_true",
                        help="also save each panel as its own figure, "
                             "<stem>_per_phase / <stem>_cumulative")
    parser.add_argument("--solid-min-runs", type=int, default=None,
                        help="draw the group mean solid where at least this "
                             "many runs contribute, dashed below (default: "
                             "the whole group; clamped to the group size)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")  # headless CLI: never probe a GUI backend

    groups = _parse_groups(parser, args.group, args.min_progress)
    _report(groups, args.solid_min_runs, args.x_axis)
    measures = _MEASURE_CLI[args.measure]
    out = plot_hv_aggregate(groups, args.out, measures=measures,
                            show_runs=args.show_runs, title=args.title,
                            min_runs=args.solid_min_runs, x_axis=args.x_axis,
                            split_panels=args.split_panels)
    print(f"[hv_aggregate] wrote {out} (+ .pdf), "
          f"{out.with_name(out.stem + '_aggregate.csv')}")
    if args.split_panels and len(measures) > 1:
        for m in measures:
            print(f"[hv_aggregate] wrote {split_panel_path(out, m)} (+ .pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
