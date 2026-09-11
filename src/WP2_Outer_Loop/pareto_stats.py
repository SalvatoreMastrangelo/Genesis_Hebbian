"""
Statistical significance of the difference between two conditions' exam
Pareto fronts (e.g. co-design vs morphology-only).
=====================================================================

The unit of analysis is one **run** (one seed, one Slurm job). Every test
is an *exact* permutation test over all ``C(n, n_a)`` relabellings of the
pooled runs, so nothing assumes normality and every p-value has a hard
floor of ``2 / C(n, n_a)`` (0.000155 for 8 vs 8): "p-value at the floor"
means the two groups do not overlap at all, not that the p-value vanishes.

Per run the object under test is the **cumulative exam front**
(``pareto_overlay.cumulative_front``: non-dominated set of every
exam-scored row above the run's ``outer.min_progress_m`` gate, pooled over
all outer generations). Four layers of tests read it:

1. **Unary indicators** (``indicator_tests``) — hypervolume w.r.t. the
   fixed (80 m, CoT 0.5) reference of ``pareto_fronts``, cost of transport
   at fixed progress levels (the briefs' CoT@150/170/190), tip progress and
   cheapest-arm progress; exact Mann–Whitney, Cliff's delta, the
   Hodges–Lehmann shift with its exact (Moses) 95 % CI, Holm across the
   indicators. A seed-paired Wilcoxon table (``paired_tests``) is added
   when the two groups share catalog seeds.
2. **Along the front** (``attainment_test``) — ``CoT(P)`` = cheapest front
   member reaching progress ``P`` on a fine grid, runs not reaching ``P``
   ranked worst; exact rank-sum test at every ``P`` with the max-T
   (Westfall–Young) family-wise correction across the grid, giving the
   band of progress where one condition is significantly cheaper. The same
   front read the other way (``progress_attainment_test``): ``Progress(C)``
   = farthest front member within a cost-of-transport budget ``C``, runs
   with no member that cheap ranked worst — the band of CoT budgets where
   one condition flies significantly farther. Progress-at-CoT indicators
   (``--cot-levels``) mirror the CoT-at-progress ones.
3. **Over the objective plane** (``eaf_test``, ``eaf_pvalue_maps``) — the
   empirical attainment function (EAF) of each group; the KS-type
   statistic ``max_z |EAF_A(z) − EAF_B(z)|`` with its exact permutation
   p-value (Fonseca et al.), and per-point maps of the raw (Fisher-exact)
   and family-wise-adjusted p-values (max-|ΔEAF| and min-p-value nulls).
4. **Hypervolume over the run** (``hv_trajectories``) — per-phase and
   cumulative-front HV against the fraction of the run's NSGA-II phases
   (the compute-matched axis when conditions run different phase counts),
   with the same max-T-corrected rank-sum test at every time point.

Usage
-----
::

    PYTHONPATH=src python -m WP2_Outer_Loop.pareto_stats \\
        --group co-design red \\
            logs/remote/outer_nsga/outer_exam_4_64_64_300_r* \\
            logs/remote/outer_nsga/outer_exam_6_64_64_300_r* \\
            logs/remote/outer_nsga/outer_exam_4_64_64_300_extra_r* \\
        --group morphology-only blue \\
            logs/remote/outer_nsga/outer_morphology_only_exam*_r* \\
        --out logs/remote/outer_nsga/pareto_stats

Exactly two ``--group LABEL COLOR RUN_DIR...`` (the first is "A": positive
effects, red shading and positive signs mean A is better). Writes into
``--out``: ``indicators_per_run.csv``, ``indicator_tests.csv``,
``paired_tests.csv``, ``attainment_test.csv``, ``eaf_maps.npz``,
``progress_attainment_test.csv``, ``hv_trajectories.csv``,
``hv_time_test.csv``, ``summary.json`` and the figures
``indicator_strips`` (CoT at ``--levels``), ``indicator_strips_progress``
(progress at ``--cot-levels``), ``hypervolume_strip``,
``attainment_difference``, ``attainment_difference_progress`` (max-T band,
Hodges–Lehmann shift) and their raw-p-value siblings
``attainment_difference_raw``, ``attainment_difference_progress_raw``,
``eaf_difference``, ``eaf_pvalue``, ``hypervolume_significance``
(PNG + PDF). ``--no-bixler-subdir [NAME]`` re-plots the two figures that
carry the Bixler star (``eaf_difference``, ``eaf_pvalue``) without it into
``--out/NAME/`` (default ``no_bixler``) as well. Pure
numpy/pandas/scipy/matplotlib — runs outside docker.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from WP2_Outer_Loop.pareto_fronts import (
    BIXLER_LABEL,
    _PROGRESS_NAMES,
    _admission_mask,
    _filter_exam_rows,
    _hv_reference,
    _hypervolume_2d,
    _load_min_progress,
    _load_objective_specs,
    _nondominated_mask,
    _objective_columns,
)
from WP2_Outer_Loop.pareto_overlay import (
    attainment_curve,
    cumulative_front,
    exam_star,
    resolve_run_dir,
)

_COT_NAMES = ("cost_of_transport", "cot")
DEFAULT_LEVELS: Tuple[float, ...] = (130.0, 150.0, 170.0, 190.0, 210.0)
DEFAULT_COT_LEVELS: Tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35)
ALPHA = 0.05


# ----------------------------------------------------------------------------
#  Per-run data
# ----------------------------------------------------------------------------

@dataclass
class RunStats:
    """One run's cumulative exam front and its scalar indicators."""
    run_dir: Path
    name: str
    seed: Optional[int]
    refresh: int
    front: np.ndarray                      # (n, 2) [progress, cot], progress asc
    hv: float
    tip: float
    arm: float
    cot_at: Dict[float, float]
    specs: List[Tuple[str, str]]
    prog_at: Dict[float, float] = field(default_factory=dict)

    def indicators(self) -> Dict[str, float]:
        out = {"hv": self.hv, "tip": self.tip, "arm": self.arm}
        for level, value in self.cot_at.items():
            out[f"cot_at_{int(level)}"] = value
        for level, value in self.prog_at.items():
            out[_prog_key(level)] = value
        return out


def _prog_key(cot_level: float) -> str:
    return f"prog_at_cot_{float(cot_level):g}"


@dataclass
class StatsGroup:
    label: str
    color: str
    runs: List[RunStats] = field(default_factory=list)

    @property
    def fronts(self) -> List[np.ndarray]:
        return [r.front for r in self.runs]


def _run_meta(run_dir: Path) -> Tuple[Optional[int], int]:
    cfg_path = run_dir / "reproducibility" / "config.yaml"
    seed, refresh = None, 0
    if cfg_path.is_file():
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            if cfg.get("seed") is not None:
                seed = int(cfg["seed"])
            refresh = int((cfg.get("catalog") or {}).get("refresh_urdfs_every") or 0)
        except Exception:
            pass
    return seed, refresh


def _run_name(run_dir: Path) -> str:
    """The synced ``outer_<exp>_rX`` wrapper name when the run sits in
    one, else the folder name."""
    parent = run_dir.parent.name
    return parent if parent.startswith("outer_") else run_dir.name


def front_hypervolume(front: np.ndarray, specs: Sequence[Tuple[str, str]]) -> float:
    """HV of a ``[progress, cot]`` front against ``pareto_fronts``'
    reference for these objectives (fixed (80 m, CoT 0.5) for the
    standard pair)."""
    front = np.asarray(front, dtype=float).reshape(-1, 2)
    if front.shape[0] == 0:
        return 0.0
    pm = np.column_stack([front[:, 0], -front[:, 1]])
    ref, _fixed = _hv_reference(list(specs), pm)
    return _hypervolume_2d(pm, ref)


def load_run(
    run_dir: Path | str,
    min_progress: Optional[float] = None,
    levels: Sequence[float] = DEFAULT_LEVELS,
    cot_levels: Sequence[float] = DEFAULT_COT_LEVELS,
) -> RunStats:
    """``RunStats`` of one run: cumulative exam front (run's own gate unless
    ``min_progress`` overrides it) plus HV, tip, cheapest-arm progress, CoT
    at each progress in ``levels`` and progress at each CoT budget in
    ``cot_levels``."""
    run_dir = resolve_run_dir(run_dir)
    front = cumulative_front(run_dir, min_progress)
    specs = _load_objective_specs(run_dir)
    seed, refresh = _run_meta(run_dir)
    levels = tuple(float(l) for l in levels)
    cot_at = dict(zip(levels, attainment_curve(front, np.array(levels))))
    cot_levels = tuple(float(c) for c in cot_levels)
    prog_at = dict(zip(cot_levels, progress_curve(front, np.array(cot_levels))))
    tip = float(front[:, 0].max()) if len(front) else float("nan")
    arm = float(front[:, 0].min()) if len(front) else float("nan")
    return RunStats(run_dir=run_dir, name=_run_name(run_dir), seed=seed,
                    refresh=refresh, front=front,
                    hv=front_hypervolume(front, specs), tip=tip, arm=arm,
                    cot_at=cot_at, specs=specs, prog_at=prog_at)


def indicator_specs(
    levels: Sequence[float] = DEFAULT_LEVELS,
    cot_levels: Sequence[float] = DEFAULT_COT_LEVELS,
) -> List[Tuple[str, str, str]]:
    """``[(key, better, display name), ...]`` in report order."""
    out = [("hv", "higher", "Hypervolume (ref 80 m / CoT 0.5)")]
    out += [(f"cot_at_{int(l)}", "lower", f"CoT at {int(l)} m") for l in levels]
    out += [(_prog_key(c), "higher", f"Progress at CoT {float(c):g} [m]") for c in cot_levels]
    out += [("tip", "higher", "Front tip progress [m]"),
            ("arm", "lower", "Cheapest-arm progress [m]")]
    return out


def progress_curve(front: np.ndarray, c_grid: np.ndarray) -> np.ndarray:
    """``Progress(C)`` for each CoT budget in ``c_grid``: the farthest
    progress among front members with CoT <= C; NaN where the front has no
    member that cheap (below its cheapest arm). The transpose of
    ``pareto_overlay.attainment_curve``."""
    front = np.asarray(front, dtype=float).reshape(-1, 2)
    c_grid = np.asarray(c_grid, dtype=float)
    out = np.full(c_grid.shape, np.nan)
    if front.shape[0] == 0:
        return out
    within = front[None, :, 1] <= c_grid[:, None]            # (G, n)
    progs = np.where(within, front[None, :, 0], -np.inf)
    best = progs.max(axis=1)
    out[np.isfinite(best)] = best[np.isfinite(best)]
    return out


# ----------------------------------------------------------------------------
#  Exact permutation core
# ----------------------------------------------------------------------------

def all_labelings(n: int, n_a: int) -> np.ndarray:
    """``(C(n, n_a), n_a)`` index array of every way to pick the group-A
    members out of ``n`` pooled runs; row 0 is ``[0, ..., n_a-1]``, the
    observed labelling when A's runs come first."""
    return np.array(list(itertools.combinations(range(n), n_a)), dtype=int)


def _avg_ranks(v: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(v, dtype=float), method="average", axis=0)


def mwu_exact(
    a: np.ndarray, b: np.ndarray,
    labelings: Optional[np.ndarray] = None,
    better: str = "higher",
) -> Dict[str, float]:
    """Exact two-sided Mann–Whitney test of ``a`` vs ``b`` over every
    relabelling, with effect sizes. ``better`` ("higher"/"lower") fixes the
    direction in which A counts as better: it signs Cliff's delta and the
    one-sided p-value, not the two-sided one.

    Returns ``p_two_sided``, ``p_one_sided`` (A better), ``cliffs_delta``
    (+1 = every A beats every B), ``prob_superiority``, ``hl_shift``
    (Hodges–Lehmann median of a−b), ``hl_ci_lo``/``hl_ci_hi`` (exact 95 %
    Moses interval from the enumerated U distribution) and ``p_floor``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a = len(a)
    v = np.concatenate([a, b])
    if labelings is None:
        labelings = all_labelings(len(v), n_a)
    r = _avg_ranks(v)
    ra_all = r[labelings].sum(axis=1)
    ra_obs = r[:n_a].sum()
    e = ra_all.mean()
    p = float(np.mean(np.abs(ra_all - e) >= np.abs(ra_obs - e) - 1e-12))
    if better == "lower":
        p_one = float(np.mean(ra_all <= ra_obs + 1e-12))
    else:
        p_one = float(np.mean(ra_all >= ra_obs - 1e-12))
    diffs = (a[:, None] - b[None, :]).ravel()
    sign = -1.0 if better == "lower" else 1.0
    delta = float(np.mean(np.sign(diffs) * sign))
    hl = float(np.median(diffs))
    # Moses CI: c = largest U with P(U <= c) <= alpha/2 under H0
    u_sorted = np.sort(ra_all - n_a * (n_a + 1) / 2)
    cdf = np.searchsorted(u_sorted, np.arange(0, len(diffs) + 1), side="right") / len(u_sorted)
    c = int(np.sum(cdf <= ALPHA / 2)) - 1
    ds = np.sort(diffs)
    if c < 0:
        lo, hi = ds[0], ds[-1]
    else:
        lo, hi = ds[c], ds[len(ds) - 1 - c]
    return dict(p_two_sided=p, p_one_sided=p_one, cliffs_delta=delta,
                prob_superiority=(delta + 1.0) / 2.0, hl_shift=hl,
                hl_ci_lo=float(lo), hl_ci_hi=float(hi),
                p_floor=2.0 / len(labelings))


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon_exact(d: np.ndarray) -> Dict[str, float]:
    """Exact two-sided Wilcoxon signed-rank test on paired differences
    (zeros dropped); enumerates all ``2^n`` sign patterns."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return dict(n=0, w=0.0, p_two_sided=1.0, p_floor=1.0)
    r = _avg_ranks(np.abs(d))
    w_obs = float(r[d > 0].sum())
    signs = np.array(list(itertools.product([0, 1], repeat=n)))
    w_all = (signs * r[None, :]).sum(axis=1)
    e = w_all.mean()
    p = float(np.mean(np.abs(w_all - e) >= abs(w_obs - e) - 1e-12))
    return dict(n=n, w=w_obs, p_two_sided=p, p_floor=2.0 / 2 ** n)


# ----------------------------------------------------------------------------
#  Along the front / along the run
# ----------------------------------------------------------------------------

def timewise_test(
    values: np.ndarray, n_a: int, labelings: np.ndarray, better: str = "higher",
) -> Dict[str, np.ndarray]:
    """Exact rank-sum permutation test at every column of ``values``
    (``(n, G)``, first ``n_a`` rows = group A) with the max-T
    (Westfall–Young) family-wise correction across columns.

    ``t_obs`` is the standardised rank-sum statistic signed so that
    positive means A is better (per ``better``); ``p_raw`` the per-column
    exact p-value; ``p_maxT`` = P(max over columns of |T| under
    relabelling >= |t_obs|); ``mean_diff`` = mean(A) − mean(B). Columns
    with all values tied (e.g. nobody reaches that progress) are null:
    ``t_obs`` 0, p-values 1."""
    values = np.asarray(values, dtype=float)
    sign = 1.0 if better == "higher" else -1.0
    r = _avg_ranks(values)
    ra_all = r[labelings].sum(axis=1)
    e = ra_all.mean(axis=0)
    sd = ra_all.std(axis=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    t_all = (ra_all - e) / sd_safe
    t_all[:, sd == 0] = 0.0
    t_obs = (r[:n_a].sum(axis=0) - e) / sd_safe
    t_obs[sd == 0] = 0.0
    p_raw = np.mean(np.abs(t_all) >= np.abs(t_obs)[None, :] - 1e-12, axis=0)
    max_t = np.abs(t_all).max(axis=1)
    p_adj = np.array([np.mean(max_t >= abs(t) - 1e-12) for t in t_obs])
    with np.errstate(invalid="ignore"):
        mean_diff = values[:n_a].mean(axis=0) - values[n_a:].mean(axis=0)
    return dict(t_obs=sign * t_obs, p_raw=p_raw, p_maxT=p_adj, mean_diff=mean_diff)


def attainment_test(
    fronts_a: Sequence[np.ndarray], fronts_b: Sequence[np.ndarray],
    grid: np.ndarray, labelings: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Along-the-front test: at every progress in ``grid``, ``CoT(P)`` of
    each run (``attainment_curve``; a run whose tip is below ``P`` counts
    as worst), exact rank-sum test with max-T correction over the grid.

    Columns: ``progress_m``, ``n_reach_a``/``n_reach_b`` (runs reaching
    P), ``mean_diff_cot`` and ``hl_shift_cot`` (A − B, only where every
    run reaches P, NaN otherwise), ``t_obs`` (positive = A cheaper),
    ``p_raw``, ``p_maxT``."""
    fronts_a, fronts_b = list(fronts_a), list(fronts_b)
    n_a = len(fronts_a)
    grid = np.asarray(grid, dtype=float)
    if labelings is None:
        labelings = all_labelings(n_a + len(fronts_b), n_a)
    curves = np.stack([attainment_curve(f, grid) for f in fronts_a + fronts_b])
    censored = np.where(np.isfinite(curves), curves, np.inf)
    res = timewise_test(censored, n_a, labelings, better="lower")
    reach_a = np.isfinite(curves[:n_a]).sum(axis=0)
    reach_b = np.isfinite(curves[n_a:]).sum(axis=0)
    both = np.isfinite(curves).all(axis=0)
    mean_diff = np.full(len(grid), np.nan)
    hl = np.full(len(grid), np.nan)
    for g in np.where(both)[0]:
        mean_diff[g] = curves[:n_a, g].mean() - curves[n_a:, g].mean()
        hl[g] = np.median((curves[:n_a, g][:, None] - curves[n_a:, g][None, :]).ravel())
    return pd.DataFrame(dict(progress_m=grid, n_reach_a=reach_a, n_reach_b=reach_b,
                             mean_diff_cot=mean_diff, hl_shift_cot=hl,
                             t_obs=res["t_obs"], p_raw=res["p_raw"], p_maxT=res["p_maxT"]))


def progress_attainment_test(
    fronts_a: Sequence[np.ndarray], fronts_b: Sequence[np.ndarray],
    c_grid: np.ndarray, labelings: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """The front read the other way: at every CoT budget in ``c_grid``,
    ``Progress(C)`` of each run (``progress_curve``; a run with no member
    that cheap counts as worst), exact rank-sum test with max-T correction
    over the grid.

    Columns: ``cot``, ``n_have_a``/``n_have_b`` (runs with a member within
    the budget), ``mean_diff_progress`` and ``hl_shift_progress`` (A − B in
    metres, only where every run has one, NaN otherwise), ``t_obs``
    (positive = A farther), ``p_raw``, ``p_maxT``."""
    fronts_a, fronts_b = list(fronts_a), list(fronts_b)
    n_a = len(fronts_a)
    c_grid = np.asarray(c_grid, dtype=float)
    if labelings is None:
        labelings = all_labelings(n_a + len(fronts_b), n_a)
    curves = np.stack([progress_curve(f, c_grid) for f in fronts_a + fronts_b])
    censored = np.where(np.isfinite(curves), curves, -np.inf)
    res = timewise_test(censored, n_a, labelings, better="higher")
    have_a = np.isfinite(curves[:n_a]).sum(axis=0)
    have_b = np.isfinite(curves[n_a:]).sum(axis=0)
    both = np.isfinite(curves).all(axis=0)
    mean_diff = np.full(len(c_grid), np.nan)
    hl = np.full(len(c_grid), np.nan)
    for g in np.where(both)[0]:
        mean_diff[g] = curves[:n_a, g].mean() - curves[n_a:, g].mean()
        hl[g] = np.median((curves[:n_a, g][:, None] - curves[n_a:, g][None, :]).ravel())
    return pd.DataFrame(dict(cot=c_grid, n_have_a=have_a, n_have_b=have_b,
                             mean_diff_progress=mean_diff, hl_shift_progress=hl,
                             t_obs=res["t_obs"], p_raw=res["p_raw"], p_maxT=res["p_maxT"]))


def _segments(xs: np.ndarray, step: float) -> List[List[float]]:
    """Contiguous runs of grid values (spacing ``step``) as [start, end]."""
    if len(xs) == 0:
        return []
    breaks = np.where(np.diff(xs) > step * 1.5)[0] + 1
    return [[float(s[0]), float(s[-1])] for s in np.split(xs, breaks)]


# ----------------------------------------------------------------------------
#  Objective plane: EAF test and p-value maps
# ----------------------------------------------------------------------------

def attain_matrix(
    fronts: Sequence[np.ndarray], p_grid: np.ndarray, c_grid: np.ndarray,
) -> np.ndarray:
    """``(R, P*C)`` bool: run r attains grid point (P, C) iff its front has
    a member with progress >= P and CoT <= C, i.e. ``CoT_r(P) <= C``.
    Flattened row-major over ``p_grid`` then ``c_grid``."""
    p_grid = np.asarray(p_grid, dtype=float)
    c_grid = np.asarray(c_grid, dtype=float)
    curves = np.stack([attainment_curve(f, p_grid) for f in fronts])
    curves = np.where(np.isfinite(curves), curves, np.inf)
    return (curves[:, :, None] <= c_grid[None, None, :]).reshape(len(fronts), -1)


def eaf_test(
    fronts_a: Sequence[np.ndarray], fronts_b: Sequence[np.ndarray],
    p_grid: np.ndarray, c_grid: np.ndarray,
    labelings: Optional[np.ndarray] = None, chunk: int = 512,
) -> Dict[str, object]:
    """Whole-front EAF test: ``ks`` = max over the grid of
    |EAF_A − EAF_B| (EAF = fraction of the group's runs attaining the
    point), ``p`` its exact permutation p-value, ``at_progress``/``at_cot``
    where the max sits, the two EAF maps and their difference (``(P, C)``
    arrays) and ``ks_null`` (the statistic under every relabelling)."""
    fronts_a, fronts_b = list(fronts_a), list(fronts_b)
    n_a, n_b = len(fronts_a), len(fronts_b)
    n = n_a + n_b
    if labelings is None:
        labelings = all_labelings(n, n_a)
    m = attain_matrix(fronts_a + fronts_b, p_grid, c_grid).astype(np.float32)
    eaf_a = m[:n_a].mean(axis=0)
    eaf_b = m[n_a:].mean(axis=0)
    diff = eaf_a - eaf_b
    ks_obs = float(np.abs(diff).max())
    ks_all = np.empty(len(labelings), dtype=np.float32)
    for s in range(0, len(labelings), chunk):
        idx = labelings[s:s + chunk]
        w = np.full((len(idx), n), -1.0 / n_b, dtype=np.float32)
        np.put_along_axis(w, idx, 1.0 / n_a, axis=1)
        ks_all[s:s + chunk] = np.abs(w @ m).max(axis=1)
    p = float(np.mean(ks_all >= ks_obs - 1e-6))
    shape = (len(p_grid), len(c_grid))
    where = int(np.abs(diff).argmax())
    pi, ci = divmod(where, len(c_grid))
    return dict(ks=ks_obs, p=p, at_progress=float(p_grid[pi]), at_cot=float(c_grid[ci]),
                eaf_a=eaf_a.reshape(shape), eaf_b=eaf_b.reshape(shape),
                diff=diff.reshape(shape), ks_null=ks_all)


def fisher_table(n_a: int, n_b: int) -> np.ndarray:
    """``tab[k, a]``: exact two-sided p-value that, of the ``k`` runs
    attaining a point, ``a`` belong to group A (hypergeometric, i.e. the
    per-point permutation test, ordered by |EAF_A − EAF_B|). NaN where
    the cell is impossible."""
    n = n_a + n_b
    tab = np.full((n + 1, n_a + 1), np.nan)
    for k in range(n + 1):
        a_lo, a_hi = max(0, k - n_b), min(n_a, k)
        aa = np.arange(a_lo, a_hi + 1)
        pr = np.array([comb(n_a, a) * comb(n_b, k - a) / comb(n, k) for a in aa])
        d = np.abs(aa / n_a - (k - aa) / n_b)
        for a, da in zip(aa, d):
            tab[k, a] = pr[d >= da - 1e-12].sum()
    return tab


def eaf_pvalue_maps(
    fronts_a: Sequence[np.ndarray], fronts_b: Sequence[np.ndarray],
    p_grid: np.ndarray, c_grid: np.ndarray,
    labelings: np.ndarray, ks_null: np.ndarray, chunk: int = 512,
) -> Dict[str, np.ndarray]:
    """Per-point p-value maps over the objective plane (``(P, C)`` arrays):

    * ``p_raw`` — exact per-point permutation (= Fisher) p-value that the
      fraction of runs attaining the point differs between groups;
    * ``p_maxd`` — family-wise adjusted, P(max_z |ΔEAF_perm| >= |ΔEAF_obs(z)|),
      using the EAF test's own null ``ks_null`` (its minimum is that
      test's p-value);
    * ``p_minp`` — Westfall–Young min-p-value adjustment,
      P(min_z p_perm(z) <= p_obs(z));
    * ``sign`` — +1 where more A runs attain the point, −1 where more B.

    ``minp_null`` (per relabelling) is returned too."""
    fronts_a, fronts_b = list(fronts_a), list(fronts_b)
    n_a, n_b = len(fronts_a), len(fronts_b)
    n = n_a + n_b
    m = attain_matrix(fronts_a + fronts_b, p_grid, c_grid).astype(np.float32)
    k = m.sum(axis=0).astype(int)
    a_obs = m[:n_a].sum(axis=0).astype(int)
    tab = fisher_table(n_a, n_b)
    p_raw = tab[k, a_obs]
    diff = a_obs / n_a - (k - a_obs) / n_b
    p_maxd = np.array([np.mean(ks_null >= d - 1e-6) for d in np.abs(diff)])
    minp_null = np.empty(len(labelings))
    for s in range(0, len(labelings), chunk):
        idx = labelings[s:s + chunk]
        w = np.zeros((len(idx), n), dtype=np.float32)
        np.put_along_axis(w, idx, 1.0, axis=1)
        a_perm = np.rint(w @ m).astype(int)
        minp_null[s:s + chunk] = np.nanmin(tab[k[None, :], a_perm], axis=1)
    p_minp = np.array([np.mean(minp_null <= pv + 1e-12) for pv in p_raw])
    shape = (len(p_grid), len(c_grid))
    return dict(p_raw=p_raw.reshape(shape), p_maxd=p_maxd.reshape(shape),
                p_minp=p_minp.reshape(shape), sign=np.sign(diff).reshape(shape),
                minp_null=minp_null)


# ----------------------------------------------------------------------------
#  Hypervolume over the run
# ----------------------------------------------------------------------------

def hv_trajectories(run_dir: Path | str, min_progress: Optional[float] = None) -> pd.DataFrame:
    """Per outer generation of a run: HV of that generation's exam front
    (``hv_phase``) and of the cumulative front pooled up to it
    (``hv_cum``); same exam filter, admission gate and fixed reference as
    ``pareto_fronts``."""
    run_dir = resolve_run_dir(run_dir)
    df = pd.read_csv(run_dir / "results" / "outer_population.csv")
    df, _exam_only = _filter_exam_rows(df)
    specs = _load_objective_specs(run_dir)
    obj_cols = _objective_columns(df, specs)
    prog = next(c for c, n, _ in obj_cols if n in _PROGRESS_NAMES)
    cot = next(c for c, n, _ in obj_cols if n in _COT_NAMES)
    gate = _load_min_progress(run_dir) if min_progress is None else min_progress
    df = df[_admission_mask(df, specs, gate)].reset_index(drop=True)
    rows = []
    running = np.zeros((0, 2))
    for gen, grp in df.groupby("outer_gen", sort=True):
        pts = np.column_stack([grp[prog].to_numpy(float), grp[cot].to_numpy(float)])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            continue
        phase_front = pts[_nondominated_mask(np.column_stack([pts[:, 0], -pts[:, 1]]))]
        pooled = np.vstack([running, pts])
        running = pooled[_nondominated_mask(np.column_stack([pooled[:, 0], -pooled[:, 1]]))]
        rows.append(dict(outer_gen=int(gen),
                         hv_phase=front_hypervolume(phase_front, specs),
                         hv_cum=front_hypervolume(running, specs)))
    return pd.DataFrame(rows, columns=["outer_gen", "hv_phase", "hv_cum"])


def sample_at_fraction(traj: pd.DataFrame, col: str, f_grid: np.ndarray) -> np.ndarray:
    """``traj[col]`` at the phase nearest to each fraction of the run."""
    n = len(traj)
    idx = np.rint(np.asarray(f_grid, dtype=float) * (n - 1)).astype(int)
    return traj[col].to_numpy(dtype=float)[idx]


# ----------------------------------------------------------------------------
#  Tables
# ----------------------------------------------------------------------------

def indicator_tests(
    runs_a: Sequence[RunStats], runs_b: Sequence[RunStats],
    labelings: Optional[np.ndarray] = None,
    levels: Sequence[float] = DEFAULT_LEVELS,
    cot_levels: Sequence[float] = DEFAULT_COT_LEVELS,
) -> pd.DataFrame:
    """One row per indicator: group medians/ranges, exact Mann–Whitney
    p-values, Cliff's delta, Hodges–Lehmann shift + CI, and Holm across
    the indicators (``p_holm``)."""
    if labelings is None:
        labelings = all_labelings(len(runs_a) + len(runs_b), len(runs_a))
    rows = []
    for key, better, name in indicator_specs(levels, cot_levels):
        a = np.array([r.indicators()[key] for r in runs_a], dtype=float)
        b = np.array([r.indicators()[key] for r in runs_b], dtype=float)
        res = mwu_exact(a, b, labelings, better)
        rows.append(dict(indicator=name, key=key, better=better,
                         n_a=len(a), n_b=len(b),
                         a_median=float(np.median(a)), a_min=float(a.min()), a_max=float(a.max()),
                         b_median=float(np.median(b)), b_min=float(b.min()), b_max=float(b.max()),
                         **res))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_two_sided"].to_numpy())
    return df


def paired_tests(
    runs_a: Sequence[RunStats], runs_b: Sequence[RunStats],
    levels: Sequence[float] = DEFAULT_LEVELS,
    cot_levels: Sequence[float] = DEFAULT_COT_LEVELS,
) -> pd.DataFrame:
    """Seed-paired exact Wilcoxon signed-rank tests: runs are paired by
    catalog ``seed`` across the groups; several runs of one seed within a
    group are averaged first. Empty when no seed is shared."""
    def by_seed(runs):
        out: Dict[int, List[RunStats]] = {}
        for r in runs:
            if r.seed is not None:
                out.setdefault(r.seed, []).append(r)
        return out

    sa, sb = by_seed(runs_a), by_seed(runs_b)
    seeds = sorted(set(sa) & set(sb))
    cols = ["indicator", "key", "n_pairs", "seeds", "mean_diff", "w", "p_two_sided", "p_floor"]
    if not seeds:
        return pd.DataFrame(columns=cols)
    rows = []
    for key, _better, name in indicator_specs(levels, cot_levels):
        d = np.array([np.mean([r.indicators()[key] for r in sa[s]])
                      - np.mean([r.indicators()[key] for r in sb[s]]) for s in seeds])
        res = wilcoxon_exact(d)
        rows.append(dict(indicator=name, key=key, n_pairs=len(seeds),
                         seeds=json.dumps(seeds), mean_diff=float(d.mean()),
                         w=res["w"], p_two_sided=res["p_two_sided"], p_floor=res["p_floor"]))
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------
#  Figures
# ----------------------------------------------------------------------------

def pvalue_label(p: float) -> str:
    """``p-value < 0.001`` below one per mille, else ``p-value = <3 s.f.>``."""
    return "p-value < 0.001" if p < 1e-3 else f"p-value = {p:.3g}"


def _draw_strips(ga: StatsGroup, gb: StatsGroup, ind: pd.DataFrame,
                 keys: Sequence[Tuple[str, str]], out_path: Path, suptitle: str,
                 panel_width: float = 2.4) -> None:
    """One strip panel per ``(key, title)``: jittered per-run dots for both
    groups, median bars, the exact Mann–Whitney p-value in the title."""
    import matplotlib.pyplot as plt

    t = ind.set_index("key")
    fig, axes = plt.subplots(1, len(keys), figsize=(panel_width * len(keys) + 0.8, 3.6))
    rng = np.random.default_rng(0)
    for ax, (k, name) in zip(np.atleast_1d(axes), keys):
        for x, g in ((0, ga), (1, gb)):
            v = np.array([r.indicators()[k] for r in g.runs])
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(v)), v, s=28, color=g.color,
                       alpha=0.85, edgecolors="white", linewidths=0.6, zorder=3)
            ax.hlines(np.median(v), x - 0.25, x + 0.25, color=g.color, lw=2, zorder=4)
        ax.set_title(f"{name}\n{pvalue_label(t.loc[k, 'p_two_sided'])}", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([ga.label, gb.label], fontsize=8)
        ax.set_xlim(-0.6, 1.6)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_indicator_strips(ga: StatsGroup, gb: StatsGroup, ind: pd.DataFrame,
                          out_path: Path, levels: Sequence[float] = DEFAULT_LEVELS) -> None:
    """Strips of CoT at each progress level."""
    keys = [(f"cot_at_{int(l)}", f"CoT at {int(l)} m") for l in levels]
    _draw_strips(ga, gb, ind, keys, out_path,
                 "Per-run CoT at fixed progress")


def plot_indicator_strips_progress(ga: StatsGroup, gb: StatsGroup, ind: pd.DataFrame,
                                   out_path: Path,
                                   cot_levels: Sequence[float] = DEFAULT_COT_LEVELS) -> None:
    """Strips of progress at each CoT budget."""
    keys = [(_prog_key(c), f"Progress at CoT {float(c):g} [m]") for c in cot_levels]
    _draw_strips(ga, gb, ind, keys, out_path,
                 "Per-run progress at fixed CoT budget")


def plot_hypervolume_strip(ga: StatsGroup, gb: StatsGroup, ind: pd.DataFrame,
                           out_path: Path, panel_width: float = 2.0) -> None:
    """The hypervolume strip on its own, as tight as one panel of the
    multi-panel indicator strips (``panel_width`` in inches; the figure is
    0.8 in wider for the y tick labels)."""
    _draw_strips(ga, gb, ind, [("hv", "Hypervolume (ref 80 m, CoT 0.5)")], out_path,
                 "Cumulative exam-front hypervolume", panel_width=panel_width)


def _alpha_line(ax, color: str) -> None:
    """The p-value = 0.05 threshold: a dashed red line with a red ``0.05``
    tag at its left end, on the log-scaled p-value panel."""
    ax.axhline(ALPHA, color=color, lw=0.9, ls=(0, (4, 2.5)))
    ax.text(0.005, ALPHA, f"{ALPHA:g}", transform=ax.get_yaxis_transform(),
            color="red", fontsize=8, ha="left", va="bottom")


def _attainment_axes(ga: StatsGroup, gb: StatsGroup, df: pd.DataFrame, x: np.ndarray,
                     diff_col: str, better: str, diff_floor: float, family_wise: bool,
                     mean_label: str, p_label: str):
    """Shared two-panel skeleton of the along-the-front figures: mean
    difference (+ Hodges–Lehmann shift and max-T shading when
    ``family_wise``, raw p-value < 0.05 shading otherwise) over the p-value
    panel. Returns ``(fig, ax1, ax2, lo, hi)``."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax1.axhline(0, color="0.4", lw=0.8)
    ax1.plot(x, df[diff_col], color="black", lw=1.8, label=mean_label)
    if family_wise:
        hl_col = diff_col.replace("mean_diff", "hl_shift")
        ax1.plot(x, df[hl_col], color="0.45", lw=1.2, ls=(0, (4, 2.5)),
                 label="Hodges–Lehmann shift")
        p_col, tag = "p_maxT", "family-wise p-value < 0.05 (max-T)"
    else:
        p_col, tag = "p_raw", "p-value < 0.05"
    sig_a = ((df[p_col] < ALPHA) & (df["t_obs"] > 0)).to_numpy()
    sig_b = ((df[p_col] < ALPHA) & (df["t_obs"] < 0)).to_numpy()
    finite = df[diff_col].to_numpy()[np.isfinite(df[diff_col])]
    lo = min(-diff_floor, float(finite.min()) * 1.25) if finite.size else -4 * diff_floor
    hi = max(diff_floor, float(finite.max()) * 1.25) if finite.size else 4 * diff_floor
    ax1.fill_between(x, lo, hi, where=sig_a, color=ga.color, alpha=0.12, lw=0,
                     label=f"{ga.label} {better}, {tag}")
    if sig_b.any():
        ax1.fill_between(x, lo, hi, where=sig_b, color=gb.color, alpha=0.12, lw=0,
                         label=f"{gb.label} {better}, {tag}")
    ax1.set_ylim(lo, hi)
    ax1.grid(True, linestyle="--", alpha=0.4)
    if family_wise:
        ax2.plot(x, df["p_raw"], color="0.55", lw=1.2, label=f"raw p-value ({p_label})")
        ax2.plot(x, df["p_maxT"], color="black", lw=1.8, label="max-T adjusted p-value")
    else:
        ax2.plot(x, df["p_raw"], color="black", lw=1.8, label=f"p-value ({p_label})")
    _alpha_line(ax2, ga.color)
    ax2.set_yscale("log")
    ax2.set_ylim(min(1e-4, float(df["p_raw"].min()) / 2), 1.5)
    ax2.set_ylabel("p-value (exact rank-sum)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8, loc="lower right")
    return fig, ax1, ax2, lo, hi


def plot_attainment_difference_progress(ga: StatsGroup, gb: StatsGroup, patt: pd.DataFrame,
                                        out_path: Path, family_wise: bool = True) -> None:
    """Δ progress at a given CoT budget (A − B) with the significance band,
    and the p-value panel — the transpose of ``plot_attainment_difference``.
    ``family_wise=False`` is the raw-p-value sibling: no Hodges–Lehmann
    shift, no max-T curve, shading where the raw p-value < 0.05."""
    import matplotlib.pyplot as plt

    C = patt["cot"].to_numpy()
    fig, ax1, ax2, lo, hi = _attainment_axes(
        ga, gb, patt, C, "mean_diff_progress", "farther", 5.0, family_wise,
        f"mean progress(CoT), {ga.label} − {gb.label}", "per CoT budget")
    ax1.set_ylabel("Δ progress at a given CoT budget [m]")
    ax1.legend(fontsize=8, loc="upper right")
    ax2.set_xlabel("Cost of transport budget")
    n_have = patt[["n_have_a", "n_have_b"]].to_numpy()
    full = (n_have == n_have.max(axis=0)).all(axis=1)
    if full.any() and not full[0]:
        all_arms = C[np.argmax(full)]
        for ax in (ax1, ax2):
            ax.axvline(all_arms, color="0.6", lw=0.8, ls=":")
        ax1.text(all_arms + 0.002, hi * 0.95, "every run has a body", fontsize=7,
                 color="0.4", va="top")
    ax1.set_title("Progress-at-CoT difference between conditions", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_attainment_difference(ga: StatsGroup, gb: StatsGroup, att: pd.DataFrame,
                               out_path: Path, family_wise: bool = True) -> None:
    """Δ CoT at a given progress (A − B) with the significance band and the
    p-value panel. ``family_wise=False`` is the raw-p-value sibling: no
    Hodges–Lehmann shift, no max-T curve, shading where the raw p-value
    < 0.05."""
    import matplotlib.pyplot as plt

    P = att["progress_m"].to_numpy()
    fig, ax1, ax2, lo, hi = _attainment_axes(
        ga, gb, att, P, "mean_diff_cot", "cheaper", 0.01, family_wise,
        f"mean CoT(progress), {ga.label} − {gb.label}", "per progress level")
    ax1.set_ylabel("Δ CoT at a given progress")
    ax1.legend(fontsize=8, loc="lower left")
    ax2.set_xlabel("Exam progress [m]")
    n_reach = att[["n_reach_a", "n_reach_b"]].to_numpy()
    short = (n_reach < n_reach.max(axis=0)).any(axis=1)
    if short.any():
        first_tip = P[np.argmax(short)]
        for ax in (ax1, ax2):
            ax.axvline(first_tip, color="0.6", lw=0.8, ls=":")
        ax1.text(first_tip + 1, lo * 0.95, "first tip", fontsize=7, color="0.4", va="bottom")
    ax1.set_title("Attainment-curve difference between conditions", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_eaf_difference(ga: StatsGroup, gb: StatsGroup, eaf: Dict[str, object],
                        p_grid: np.ndarray, c_grid: np.ndarray, out_path: Path,
                        star: Optional[Tuple[float, float]] = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("ab", [gb.color, "#d9d9d9", ga.color])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(p_grid, c_grid, eaf["diff"].T, cmap=cmap,
                       norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), shading="auto",
                       rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(f"EAF({ga.label}) − EAF({gb.label})")
    for m, g in ((eaf["eaf_a"], ga), (eaf["eaf_b"], gb)):
        ax.contour(p_grid, c_grid, m.T, levels=[0.5 - 1e-6], colors=[g.color], linewidths=1.8)
        ax.plot([], [], color=g.color, lw=1.8,
                label=f"{g.label}: attained by ≥ half of the runs")
        for f in g.fronts:
            if len(f):
                ax.plot(f[:, 0], f[:, 1], color=g.color, lw=0.6, alpha=0.35)
    if star is not None:
        ax.scatter([star[0]], [star[1]], marker="*", s=300, color="gold",
                   edgecolors="black", linewidths=0.9, zorder=5, label=BIXLER_LABEL)
    ax.scatter([eaf["at_progress"]], [eaf["at_cot"]], marker="x", s=70, color="black",
               zorder=6, label=f"max |ΔEAF| = {eaf['ks']:.2f}, {pvalue_label(eaf['p'])}")
    ax.set_xlabel("Exam progress [m]")
    ax.set_ylabel("Cost of transport")
    ax.set_xlim(p_grid[0], p_grid[-1])
    ax.set_ylim(c_grid[0], c_grid[-1])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Empirical attainment function difference", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_eaf_pvalue(ga: StatsGroup, gb: StatsGroup, eaf: Dict[str, object],
                    pmaps: Dict[str, np.ndarray], p_grid: np.ndarray, c_grid: np.ndarray,
                    out_path: Path, star: Optional[Tuple[float, float]] = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

    ramp_b = LinearSegmentedColormap.from_list("b", ["#d9d9d9", gb.color, "#08306b"])
    ramp_a = LinearSegmentedColormap.from_list("a", ["#d9d9d9", ga.color, "#67000d"])
    colors = ([ramp_b(x) for x in (1.0, 0.8, 0.6, 0.35)] + ["#d9d9d9"]
              + [ramp_a(x) for x in (0.35, 0.6, 0.8, 1.0)])
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    edges = np.array([-4.5, -3, -2, np.log10(0.05), np.log10(0.2),
                      -np.log10(0.2), -np.log10(0.05), 2, 3, 4.5])
    norm = BoundaryNorm(edges, cmap.N)
    panels = [("p_raw", "Raw pointwise p-value (exact, per point)"),
              ("p_maxd", "Family-wise adjusted p-value (max |ΔEAF| over the plane)"),
              ("p_minp", "Family-wise adjusted p-value (min-p-value over the plane)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    im = None
    for ax, (key, ttl) in zip(axes, panels):
        pv = np.clip(np.where(np.isnan(pmaps[key]), 1.0, pmaps[key]), 1e-4, 1.0)
        signed = -np.log10(pv) * pmaps["sign"]
        im = ax.pcolormesh(p_grid, c_grid, signed.T, cmap=cmap, norm=norm,
                           shading="auto", rasterized=True)
        if (np.abs(signed) > -np.log10(ALPHA)).any():
            ax.contour(p_grid, c_grid, np.abs(signed).T, levels=[-np.log10(ALPHA)],
                       colors=["black"], linewidths=0.9)
        for m, g in ((eaf["eaf_a"], ga), (eaf["eaf_b"], gb)):
            ax.contour(p_grid, c_grid, m.T, levels=[0.5 - 1e-6], colors=[g.color],
                       linewidths=1.3, linestyles="--")
        if star is not None:
            ax.scatter([star[0]], [star[1]], marker="*", s=220, color="gold",
                       edgecolors="black", linewidths=0.8, zorder=5)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("Exam progress [m]")
        ax.set_xlim(p_grid[0], p_grid[-1])
        ax.set_ylim(c_grid[0], c_grid[-1])
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Cost of transport")
    for g in (ga, gb):
        axes[0].plot([], [], color=g.color, lw=1.3, ls="--",
                     label=f"{g.label}: attained by ≥ half of the runs")
    axes[0].plot([], [], color="black", lw=0.9, label="p-value = 0.05 contour")
    if star is not None:
        axes[0].scatter([], [], marker="*", s=120, color="gold", edgecolors="black",
                        linewidths=0.8, label=BIXLER_LABEL)
    axes[0].legend(fontsize=7.5, loc="upper left")
    cb = fig.colorbar(im, ax=axes, pad=0.015, fraction=0.03, ticks=edges[1:-1])
    cb.ax.set_yticklabels(["0.001", "0.01", "0.05", "0.2", "0.2", "0.05", "0.01", "0.001"],
                          fontsize=8)
    cb.set_label(f"p-value, signed: {ga.color} = {ga.label} attains the point in more runs, "
                 f"{gb.color} = {gb.label} does", fontsize=8)
    fig.suptitle("Where in objective space do the two conditions' fronts differ?\n"
                 f"Pointwise test of the fraction of runs attaining a progress and CoT point, "
                 f"{len(ga.runs)} vs {len(gb.runs)} runs", fontsize=10.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_hv_significance(ga: StatsGroup, gb: StatsGroup, hv: Dict[str, object],
                         f_grid: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    n_a = len(ga.runs)
    fig, ((ax_null, ax_dots), (ax_cum, ax_phs)) = plt.subplots(2, 2, figsize=(12.5, 9))

    ax_null.hist(hv["diff_null"], bins=80, color="0.72", edgecolor="white", lw=0.4)
    ax_null.axvline(hv["diff_obs"], color=ga.color, lw=2.2)
    ax_null.axvline(-hv["diff_obs"], color=ga.color, lw=1.0, ls=(0, (4, 2.5)))
    ax_null.set_xlabel(f"mean HV({ga.label}) − mean HV({gb.label})  [relabelled runs]")
    ax_null.set_ylabel(f"number of relabellings (of {len(hv['diff_null'])})")
    ax_null.set_title("(a) Exact permutation null of the HV difference", fontsize=10)
    n_extreme = int(round(hv["p_meandiff"] * len(hv["diff_null"])))
    ymax = ax_null.get_ylim()[1]
    ax_null.annotate(f"observed {hv['diff_obs']:+.2f}\nexact two-sided "
                     f"{pvalue_label(hv['p_meandiff'])}\n{n_extreme} of {len(hv['diff_null'])} "
                     f"relabellings as extreme",
                     xy=(hv["diff_obs"], ymax * 0.55),
                     xytext=(hv["diff_obs"] - 0.05 * abs(hv["diff_obs"]) - 0.1, ymax * 0.75),
                     ha="right", fontsize=8.5,
                     arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8))
    ax_null.grid(True, linestyle="--", alpha=0.4)

    rng = np.random.default_rng(0)
    for x, g, vals in ((0, ga, hv["a"]), (1, gb, hv["b"])):
        ax_dots.scatter(x + rng.uniform(-0.12, 0.12, len(vals)), vals, s=34, color=g.color,
                        alpha=0.9, edgecolors="white", linewidths=0.6, zorder=3)
        ax_dots.hlines(np.median(vals), x - 0.25, x + 0.25, color=g.color, lw=2.2, zorder=4)
    m = hv["mwu"]
    base = float(np.median(hv["b"]))
    ax_dots.errorbar([2.0], [base + m["hl_shift"]],
                     yerr=[[m["hl_shift"] - m["hl_ci_lo"]], [m["hl_ci_hi"] - m["hl_shift"]]],
                     fmt="D", color="black", ms=5, capsize=4, lw=1.2, zorder=4)
    ax_dots.hlines(base, 0.75, 2.25, color=gb.color, lw=0.8, ls=":", zorder=2)
    ax_dots.text(2.0, base + m["hl_ci_hi"] + 0.02 * max(1.0, abs(m["hl_ci_hi"])),
                 f"shift {m['hl_shift']:+.2f}\n95 % CI [{m['hl_ci_lo']:+.2f}, {m['hl_ci_hi']:+.2f}]",
                 ha="center", va="bottom", fontsize=8.5)
    ax_dots.set_xticks([0, 1, 2])
    ax_dots.set_xticklabels([ga.label, gb.label, f"{gb.label} median\n+ HL shift"], fontsize=8.5)
    ax_dots.set_xlim(-0.5, 2.6)
    ax_dots.set_ylabel("Hypervolume of the cumulative exam front\n(ref 80 m, CoT 0.5)")
    ax_dots.set_title(f"(b) Per-run HV, exact Mann–Whitney {pvalue_label(m['p_two_sided'])}, "
                      f"Cliff's δ = {m['cliffs_delta']:+.2f}", fontsize=10)
    ax_dots.grid(True, axis="y", linestyle="--", alpha=0.4)

    phases = (f"{ga.label} {'/'.join(str(n) for n in sorted({len(hv['traj'][r.name]) for r in ga.runs}))}, "
              f"{gb.label} {'/'.join(str(n) for n in sorted({len(hv['traj'][r.name]) for r in gb.runs}))}")
    for ax, mat, tst, ttl, ylab in (
        (ax_cum, hv["cum"], hv["t_cum"], "(c) Cumulative-front HV over the run", "cumulative HV"),
        (ax_phs, hv["phs"], hv["t_phs"], "(d) Per-phase-front HV over the run", "per-phase HV"),
    ):
        for i in range(len(mat)):
            ax.plot(f_grid, mat[i], color=ga.color if i < n_a else gb.color, lw=0.8, alpha=0.3)
        ax.plot(f_grid, mat[:n_a].mean(axis=0), color=ga.color, lw=2.4,
                label=f"{ga.label} mean ({n_a} runs)")
        ax.plot(f_grid, mat[n_a:].mean(axis=0), color=gb.color, lw=2.4,
                label=f"{gb.label} mean ({len(mat) - n_a} runs)")
        lo, hi = ax.get_ylim()
        sig = (tst["p_maxT"] < ALPHA) & (tst["t_obs"] > 0)
        ax.fill_between(f_grid, lo, hi, where=sig, color=ga.color, alpha=0.10, lw=0,
                        label=f"{ga.label} higher, family-wise p-value < 0.05 (max-T over time)")
        raw_only = (tst["p_raw"] < ALPHA) & ~sig & (tst["t_obs"] > 0)
        if raw_only.any():
            ax.fill_between(f_grid, lo, hi, where=raw_only, color=ga.color, alpha=0.04, lw=0,
                            hatch="///", edgecolor=ga.color, label="raw p-value < 0.05 only")
        sig_b = (tst["p_maxT"] < ALPHA) & (tst["t_obs"] < 0)
        if sig_b.any():
            ax.fill_between(f_grid, lo, hi, where=sig_b, color=gb.color, alpha=0.10, lw=0,
                            label=f"{gb.label} higher, family-wise p-value < 0.05")
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"fraction of the run (NSGA-II phases: {phases})")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"Hypervolume significance, {ga.label} vs {gb.label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------
#  Orchestrator
# ----------------------------------------------------------------------------

def _grids(ga: StatsGroup, gb: StatsGroup, grid_step: float, cot_step: float,
           min_progress: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    fronts = [f for f in ga.fronts + gb.fronts if len(f)]
    gates = [g for g in (_load_min_progress(r.run_dir) for r in ga.runs + gb.runs) if g > 0]
    if min_progress is not None and min_progress > 0:
        p_lo = float(min_progress)
    elif gates:
        p_lo = min(gates)
    else:
        p_lo = min(float(f[:, 0].min()) for f in fronts)
    p_lo = np.floor(p_lo / grid_step) * grid_step
    p_hi = np.ceil(max(float(f[:, 0].max()) for f in fronts) / grid_step) * grid_step + 2 * grid_step
    p_grid = np.arange(p_lo, p_hi + grid_step / 2, grid_step)
    c_lo = np.floor(min(float(f[:, 1].min()) for f in fronts) / cot_step) * cot_step
    c_hi = max(0.5, max(float(f[:, 1].max()) for f in fronts))
    c_grid = np.arange(c_lo, c_hi + cot_step / 2, cot_step)
    return p_grid, c_grid


def run_analysis(
    ga: StatsGroup, gb: StatsGroup, out_dir: Path | str, *,
    grid_step: float = 0.5, cot_step: float = 0.0025,
    levels: Sequence[float] = DEFAULT_LEVELS,
    cot_levels: Sequence[float] = DEFAULT_COT_LEVELS, f_points: int = 51,
    min_progress: Optional[float] = None, star: bool = True, figures: bool = True,
    no_star_subdir: Optional[str] = None,
) -> Dict[str, object]:
    """Run every layer for group A vs group B, write tables, maps, figures
    and ``summary.json`` into ``out_dir``, and return the summary dict.
    Group A is the reference direction: positive shifts, positive ``t_obs``
    and the A colour mean A is better. With ``no_star_subdir`` the two
    figures that draw the Bixler star (``eaf_difference``, ``eaf_pvalue``)
    are re-plotted without it into ``out_dir/<no_star_subdir>/`` as well;
    nothing else lives there."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_a, n_b = len(ga.runs), len(gb.runs)
    if n_a < 2 or n_b < 2:
        raise ValueError("each group needs at least two runs")
    lab = all_labelings(n_a + n_b, n_a)
    levels = tuple(float(l) for l in levels)
    cot_levels = tuple(float(c) for c in cot_levels)
    summary: Dict[str, object] = {
        "groups": [ga.label, gb.label], "n_runs": [n_a, n_b],
        "n_labelings": int(len(lab)), "p_floor": 2.0 / len(lab), "levels": list(levels),
        "cot_levels": list(cot_levels),
    }

    # per-run indicators + tables
    per_run = pd.DataFrame([dict(group=g.label, run=r.name, seed=r.seed, refresh=r.refresh,
                                 n_front=len(r.front), **r.indicators())
                            for g in (ga, gb) for r in g.runs])
    per_run.to_csv(out_dir / "indicators_per_run.csv", index=False)
    ind = indicator_tests(ga.runs, gb.runs, lab, levels, cot_levels)
    ind.to_csv(out_dir / "indicator_tests.csv", index=False)
    summary["indicators"] = ind.drop(columns=["indicator"]).set_index("key").to_dict("index")
    paired = paired_tests(ga.runs, gb.runs, levels, cot_levels)
    paired.to_csv(out_dir / "paired_tests.csv", index=False)
    summary["paired"] = paired.set_index("key").to_dict("index") if not paired.empty else {}

    # along the front
    p_grid, c_grid = _grids(ga, gb, grid_step, cot_step, min_progress)
    att = attainment_test(ga.fronts, gb.fronts, p_grid, lab)
    att.to_csv(out_dir / "attainment_test.csv", index=False)
    sig_a = att[(att["p_maxT"] < ALPHA) & (att["t_obs"] > 0)]["progress_m"].to_numpy()
    sig_b = att[(att["p_maxT"] < ALPHA) & (att["t_obs"] < 0)]["progress_m"].to_numpy()
    summary["attainment"] = {
        "grid_step": grid_step,
        "significant_band_m": [float(sig_a.min()), float(sig_a.max())] if len(sig_a) else None,
        "segments_a_cheaper": _segments(sig_a, grid_step),
        "segments_b_cheaper": _segments(sig_b, grid_step),
        "min_p_maxT": float(att["p_maxT"].min()),
        "at_levels": {
            str(int(l)): att.iloc[int(np.argmin(np.abs(p_grid - l)))]
            [["mean_diff_cot", "hl_shift_cot", "p_raw", "p_maxT"]].to_dict()
            for l in levels
        },
    }

    # the front read the other way: progress at a CoT budget
    patt = progress_attainment_test(ga.fronts, gb.fronts, c_grid, lab)
    patt.to_csv(out_dir / "progress_attainment_test.csv", index=False)
    csig_a = patt[(patt["p_maxT"] < ALPHA) & (patt["t_obs"] > 0)]["cot"].to_numpy()
    csig_b = patt[(patt["p_maxT"] < ALPHA) & (patt["t_obs"] < 0)]["cot"].to_numpy()
    summary["progress_attainment"] = {
        "cot_step": cot_step,
        "significant_band_cot": [float(csig_a.min()), float(csig_a.max())] if len(csig_a) else None,
        "segments_a_farther": _segments(csig_a, cot_step),
        "segments_b_farther": _segments(csig_b, cot_step),
        "min_p_maxT": float(patt["p_maxT"].min()),
        "at_levels": {
            f"{c:g}": patt.iloc[int(np.argmin(np.abs(c_grid - c)))]
            [["mean_diff_progress", "hl_shift_progress", "p_raw", "p_maxT"]].to_dict()
            for c in cot_levels
        },
    }

    # objective plane
    eaf = eaf_test(ga.fronts, gb.fronts, p_grid, c_grid, lab)
    pmaps = eaf_pvalue_maps(ga.fronts, gb.fronts, p_grid, c_grid, lab, eaf["ks_null"])
    np.savez(out_dir / "eaf_maps.npz", p_grid=p_grid, c_grid=c_grid,
             eaf_a=eaf["eaf_a"], eaf_b=eaf["eaf_b"], diff=eaf["diff"], ks_null=eaf["ks_null"],
             p_raw=pmaps["p_raw"], p_maxD=pmaps["p_maxd"], p_minP=pmaps["p_minp"],
             sign=pmaps["sign"], minp_null=pmaps["minp_null"])
    cell = grid_step * cot_step
    sig_map = pmaps["p_maxd"] < ALPHA
    summary["eaf"] = {
        "ks": eaf["ks"], "p": eaf["p"], "at_progress": eaf["at_progress"], "at_cot": eaf["at_cot"],
        "ks_null_quantiles": {q: float(np.quantile(eaf["ks_null"], float(q)))
                              for q in ("0.5", "0.95", "0.99")},
        "min_p_raw": float(np.nanmin(pmaps["p_raw"])),
        "min_p_minP": float(np.nanmin(pmaps["p_minp"])),
        "area_sig_a_m_x_cot": float((sig_map & (pmaps["sign"] > 0)).sum() * cell),
        "area_sig_b_m_x_cot": float((sig_map & (pmaps["sign"] < 0)).sum() * cell),
        "sig_progress_range_m": ([float(p_grid[np.where(sig_map.any(axis=1))[0][0]]),
                                  float(p_grid[np.where(sig_map.any(axis=1))[0][-1]])]
                                 if sig_map.any() else None),
    }

    # hypervolume over the run
    f_grid = np.linspace(0.0, 1.0, int(f_points))
    a = np.array([r.hv for r in ga.runs]); b = np.array([r.hv for r in gb.runs])
    v = np.concatenate([a, b])
    sel = np.zeros((len(lab), len(v)))
    np.put_along_axis(sel, lab, 1.0, axis=1)
    diff_null = (sel @ v) / n_a - ((1 - sel) @ v) / n_b
    diff_obs = float(a.mean() - b.mean())
    traj = {r.name: hv_trajectories(r.run_dir, min_progress) for r in ga.runs + gb.runs}
    cum = np.stack([sample_at_fraction(traj[r.name], "hv_cum", f_grid) for r in ga.runs + gb.runs])
    phs = np.stack([sample_at_fraction(traj[r.name], "hv_phase", f_grid) for r in ga.runs + gb.runs])
    hv = dict(a=a, b=b, diff_null=diff_null, diff_obs=diff_obs,
              p_meandiff=float(np.mean(np.abs(diff_null) >= abs(diff_obs) - 1e-12)),
              mwu=mwu_exact(a, b, lab, "higher"), traj=traj, cum=cum, phs=phs,
              t_cum=timewise_test(cum, n_a, lab, "higher"),
              t_phs=timewise_test(phs, n_a, lab, "higher"))
    pd.concat([traj[r.name].assign(run=r.name, group=g.label) for g in (ga, gb) for r in g.runs],
              ignore_index=True).to_csv(out_dir / "hv_trajectories.csv", index=False)
    pd.DataFrame(dict(fraction=f_grid,
                      cum_mean_diff=hv["t_cum"]["mean_diff"], cum_p_raw=hv["t_cum"]["p_raw"],
                      cum_p_maxT=hv["t_cum"]["p_maxT"],
                      phase_mean_diff=hv["t_phs"]["mean_diff"], phase_p_raw=hv["t_phs"]["p_raw"],
                      phase_p_maxT=hv["t_phs"]["p_maxT"])).to_csv(out_dir / "hv_time_test.csv",
                                                                  index=False)

    def first_sig(tst, key="p_maxT"):
        ok = np.where((tst[key] < ALPHA) & (tst["t_obs"] > 0))[0]
        return float(f_grid[ok[0]]) if len(ok) else None

    summary["hypervolume"] = {
        "mean_diff": diff_obs, "p_meandiff_exact": hv["p_meandiff"],
        "null_abs_diff_quantiles": {q: float(np.quantile(np.abs(diff_null), float(q)))
                                    for q in ("0.5", "0.95", "0.99")},
        "mwu": hv["mwu"],
        "cum_first_sig_fraction": first_sig(hv["t_cum"]),
        "cum_first_raw_fraction": first_sig(hv["t_cum"], "p_raw"),
        "cum_frac_of_run_sig": float(np.mean((hv["t_cum"]["p_maxT"] < ALPHA) & (hv["t_cum"]["t_obs"] > 0))),
        "phase_first_sig_fraction": first_sig(hv["t_phs"]),
        "phase_frac_of_run_sig": float(np.mean((hv["t_phs"]["p_maxT"] < ALPHA) & (hv["t_phs"]["t_obs"] > 0))),
        "phase_final_mean_diff": float(hv["t_phs"]["mean_diff"][-1]),
        "phase_final_p_maxT": float(hv["t_phs"]["p_maxT"][-1]),
    }

    star_xy = exam_star([r.run_dir for r in ga.runs + gb.runs]) if star else None
    summary["star"] = list(star_xy) if star_xy is not None else None

    if figures:
        plot_indicator_strips(ga, gb, ind, out_dir / "indicator_strips.png", levels)
        plot_indicator_strips_progress(ga, gb, ind, out_dir / "indicator_strips_progress.png",
                                       cot_levels)
        plot_hypervolume_strip(ga, gb, ind, out_dir / "hypervolume_strip.png")
        plot_attainment_difference(ga, gb, att, out_dir / "attainment_difference.png")
        plot_attainment_difference_progress(ga, gb, patt,
                                            out_dir / "attainment_difference_progress.png")
        plot_attainment_difference(ga, gb, att, out_dir / "attainment_difference_raw.png",
                                   family_wise=False)
        plot_attainment_difference_progress(ga, gb, patt,
                                            out_dir / "attainment_difference_progress_raw.png",
                                            family_wise=False)
        plot_eaf_difference(ga, gb, eaf, p_grid, c_grid, out_dir / "eaf_difference.png", star_xy)
        plot_eaf_pvalue(ga, gb, eaf, pmaps, p_grid, c_grid, out_dir / "eaf_pvalue.png", star_xy)
        plot_hv_significance(ga, gb, hv, f_grid, out_dir / "hypervolume_significance.png")
        if no_star_subdir and star_xy is not None:
            sub = out_dir / no_star_subdir
            sub.mkdir(parents=True, exist_ok=True)
            plot_eaf_difference(ga, gb, eaf, p_grid, c_grid, sub / "eaf_difference.png", None)
            plot_eaf_pvalue(ga, gb, eaf, pmaps, p_grid, c_grid, sub / "eaf_pvalue.png", None)
            summary["no_star_figures_dir"] = str(sub)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_jsonable)
    return summary


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def _report(ga: StatsGroup, gb: StatsGroup, summary: Dict[str, object]) -> None:
    print(f"[stats] {ga.label}: {len(ga.runs)} runs vs {gb.label}: {len(gb.runs)} runs; "
          f"{summary['n_labelings']} relabellings, p-value floor {summary['p_floor']:.2g}")
    for g in (ga, gb):
        for r in g.runs:
            ind = r.indicators()
            cots = "  ".join(f"{k}={v:.3f}" for k, v in ind.items() if k.startswith("cot_at"))
            print(f"    {g.label:<16s} {r.name:<44s} seed={r.seed} hv={ind['hv']:.2f} "
                  f"tip={ind['tip']:.1f} {cots}")
    print("[stats] indicators (exact Mann-Whitney, two-sided p-value; Holm across rows):")
    for key, row in summary["indicators"].items():
        print(f"    {key:<11s} A median {row['a_median']:.4g} vs B {row['b_median']:.4g}  "
              f"shift {row['hl_shift']:+.4g} [{row['hl_ci_lo']:+.4g}, {row['hl_ci_hi']:+.4g}]  "
              f"p-value {row['p_two_sided']:.2g} (Holm {row['p_holm']:.2g})  "
              f"Cliff's delta {row['cliffs_delta']:+.2f}")
    att = summary["attainment"]
    print(f"[stats] attainment: {ga.label} cheaper with family-wise p-value < 0.05 over "
          f"{att['segments_a_cheaper']} m; {gb.label} cheaper over {att['segments_b_cheaper']} m")
    patt = summary["progress_attainment"]
    print(f"[stats] progress at CoT budget: {ga.label} farther with family-wise p-value < 0.05 "
          f"over CoT {patt['segments_a_farther']}; {gb.label} farther over "
          f"{patt['segments_b_farther']}")
    eaf = summary["eaf"]
    print(f"[stats] EAF: max|dEAF| = {eaf['ks']:.2f} at {eaf['at_progress']:.1f} m / "
          f"CoT {eaf['at_cot']:.3f}, exact p-value {eaf['p']:.2g}")
    hv = summary["hypervolume"]
    print(f"[stats] HV over the run: cumulative HV significantly higher for {ga.label} from "
          f"fraction {hv['cum_first_sig_fraction']}, per-phase from {hv['phase_first_sig_fraction']}")
    if summary["paired"]:
        for key, row in summary["paired"].items():
            print(f"    paired {key:<11s} n={row['n_pairs']} mean diff {row['mean_diff']:+.4g} "
                  f"Wilcoxon p-value {row['p_two_sided']:.2g} (floor {row['p_floor']:.2g})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exact permutation tests of the difference between two run groups' "
                    "cumulative exam Pareto fronts.")
    parser.add_argument("--group", action="append", nargs="+", metavar="TOKEN",
                        help="LABEL COLOR RUN_DIR [RUN_DIR ...]; exactly two, A first")
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    parser.add_argument("--min-progress", type=float, default=None,
                        help="override the runs' admission gate [m]")
    parser.add_argument("--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS),
                        help="progress levels for CoT-at-progress indicators [m]")
    parser.add_argument("--cot-levels", type=float, nargs="+", default=list(DEFAULT_COT_LEVELS),
                        help="CoT budgets for progress-at-CoT indicators")
    parser.add_argument("--grid-step", type=float, default=0.5, help="progress grid step [m]")
    parser.add_argument("--cot-step", type=float, default=0.0025, help="CoT grid step")
    parser.add_argument("--no-star", action="store_true",
                        help="omit the WP1 generalist exam baseline star")
    parser.add_argument("--no-bixler-subdir", nargs="?", const="no_bixler",
                        default=None, metavar="NAME",
                        help="also re-plot the star-bearing figures (eaf_difference, "
                             "eaf_pvalue) without the star into OUT/NAME/ (default "
                             "name no_bixler); skipped when no star is drawn")
    args = parser.parse_args(argv)
    if not args.group or len(args.group) != 2:
        parser.error("exactly two --group LABEL COLOR RUN_DIR... are required")
    groups = []
    for tokens in args.group:
        if len(tokens) < 3:
            parser.error(f"--group needs LABEL COLOR RUN_DIR [RUN_DIR ...], got {tokens}")
        label, color, *dirs = tokens
        groups.append(StatsGroup(label, color,
                                 [load_run(d, args.min_progress, args.levels, args.cot_levels)
                                  for d in dirs]))

    import matplotlib
    matplotlib.use("Agg")

    summary = run_analysis(groups[0], groups[1], args.out, grid_step=args.grid_step,
                           cot_step=args.cot_step, levels=args.levels,
                           cot_levels=args.cot_levels, min_progress=args.min_progress,
                           star=not args.no_star, no_star_subdir=args.no_bixler_subdir)
    _report(groups[0], groups[1], summary)
    print(f"[stats] wrote tables, maps and figures to {args.out}")
    if args.no_bixler_subdir:
        if summary.get("no_star_figures_dir"):
            print(f"[stats] wrote eaf_difference and eaf_pvalue without the star to "
                  f"{summary['no_star_figures_dir']}")
        else:
            print("[stats] star-free copies skipped: no star drawn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
