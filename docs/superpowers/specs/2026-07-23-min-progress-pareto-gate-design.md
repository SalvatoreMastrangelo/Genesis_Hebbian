# Minimum-progress admission gate for the WP2 outer-loop Pareto front

**Date:** 2026-07-23
**Status:** Approved
**Motivation:** With `progress_m` (max) vs `cost_of_transport` (min), a
morphology that barely flies (~5–20 m) but does so "cheaply" is genuinely
nondominated, so it survives as an NSGA-II elite, breeds, and paints a
degenerate left arm on the Pareto front (see
`logs/remote/outer_nsga/outer_full_run_4_64_64_r0/.../pareto_front_evolution.png`).
A minimum-progress threshold gates admission to both selection and the
plotted fronts.

## Decisions (user-approved)

- **Scope:** gate applies to NSGA-II selection **and** plots.
- **Threshold:** fixed absolute meters via config knob (not
  baseline-relative, not adaptive).
- **Mechanism:** **hard filter** (user chose over Deb constraint-domination
  and objective clamping).

## 1. Config

New field on `OuterConfig` (`src/WP2_Outer_Loop/config.py`):

```yaml
outer:
  min_progress_m: 0.0   # meters; 0 = gate off (backward compatible)
```

Validated `>= 0` in `OuterNSGA2Config.validate()`. Saved automatically in
`reproducibility/config.yaml` like every other outer knob.

## 2. Gate metric

Feasibility is judged on per-URDF progress from **the same source that
produced the objectives**: the exam's progress when the phase-end exam
scored the phase, else the phase-mean progress. `_score_phase` returns this
gate vector alongside the objectives (today the exam's progress is dropped
when `progress_m` is not an objective).

## 3. Selection filter (`nsga_cma._refresh_urdfs`)

- `feasible = gate_progress >= min_progress_m`.
- **Elites:** `selNSGA2` over feasible individuals only, keeping
  `min(n_elites, n_feasible)`. A deficit becomes extra offspring slots —
  never filled by infeasible morphs.
- **Parents:** tournament pool = feasible individuals; if fewer than 2 are
  feasible, top up with the highest-progress infeasible morphs until the
  pool has 2 (SBX needs a pair; avoids breeding N−1 offspring from one
  clone).
- **Zero-feasible fallback:** loud warning, run this refresh ungated
  (plain NSGA-II over all N). Never crash, never select arbitrarily.
- The phase-score printout flags each URDF feasible/infeasible.
- Implemented as a pure, unit-testable helper.

## 4. Recorded data unchanged

CSV, archive rows, and `.npy` snapshots keep raw values. The gate shapes
selection and plotting only — never the record.

## 5. Plots (`pareto_plots.py`)

- `plot_pareto_front(run_dir, min_progress=None)`: `None` → read
  `outer.min_progress_m` from the run's saved config (absent/0 → no gate).
- CLI: `python -m WP2_Outer_Loop.pareto_plots <run_dir> [--min-progress X]`
  to regenerate old runs with a gate.
- Sub-threshold points stay in the scatter but are excluded from
  per-generation fronts, the cumulative front, and both hypervolume curves
  (HV reference computed from feasible points only).
- Dashed vertical line at the threshold when `progress_m` is the x-axis.
- Gate column: `obj_progress_m` when progress is an objective, else the
  `progress_m` diagnostic column.

## 6. Tests

Extend `tests/hebbian/test_outer_nsga.py`:
- pure filter helper: all-feasible / partially-feasible / one-feasible /
  none-feasible;
- plot gating on a synthetic CSV: front + hypervolume exclude
  sub-threshold rows, scatter keeps them.

## Out of scope

Legacy `outer_loop.py` / `legacy_run.py` (deprecated) stay untouched.
