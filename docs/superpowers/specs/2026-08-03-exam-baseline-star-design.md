# Exam-scored standard-mydrone reference ("the star")

**Date:** 2026-08-03
**Status:** approved, ready to implement
**Touches:** `WP2_Outer_Loop/config.py`, `WP2_Outer_Loop/nsga_cma.py`,
`WP2_Outer_Loop/pareto_plots.py`, `tests/hebbian/test_exam_forest.py`

## Problem

Since the phase-end exam (`outer.rescore`) became the source of NSGA-II
objectives, the gold "standard mydrone (zero rules)" star disappeared from
the Pareto plots. `pareto_plots._plot_front` suppresses it deliberately:

```python
star = None if exam_only else _load_standard_drone_baseline(run_dir, name_x, name_y)
```

The suppression is correct. The star's coordinates came from
`results/validation_summary.csv`, whose baseline flies the *nominal* forest
distribution, while exam objectives are measured on the
`outer.exam_forest` distribution (batch_3: `x_upper` 300 m instead of 100 m,
`dens_min` 1.0, `dens_max` 5.5). Plotting one against the other would put a
reference point on the chart that was never flown under the same conditions.

What is missing is not a plotting rule — it is a *measurement*: nobody flies
the standard mydrone on the exam forests.

## Solution

Fly it. At exam time the held-out validation env is already built, already
holds the standard mydrone, and is still alive; use it.

`validation.validation_catalog` empty (the default) means the validation env
is exactly one URDF built from `STANDARD_MYDRONE_GENOME`. It is a
`MultiSceneEvalEnv` exposing the same `apply_forest_overrides()` /
`refresh_forests()` API as the population eval env, and
`HebbianCMAES._evaluate_reference_actor(..., env_override=...)` already knows
how to run the zero-rules generalist on it. Crucially,
`_cleanup_validation_env()` runs *after* `_score_phase()` returns
(`nsga_cma._refresh_urdfs`), so both envs are live during the exam.

No scene is rebuilt and no Genesis runtime is initialised. The cost is one
extra single-genome rollout per phase on an env that already exists.

## Design

### 1. Config flag

New field on `OuterConfig` (the `outer:` section):

```python
exam_baseline: bool = True
```

Documented as: fly the standard-mydrone reference on the exam forests at the
end of each phase, reusing the held-out validation env. Implicitly inert when
`validation.enable` is false or the validation env failed to build. No
validation rule needed — it is a plain bool.

### 2. Measurement

In `NSGA2MorphCMAES._run_exam()`, inside the existing `try`/`finally` that
applies and restores `outer.exam_forest`:

1. apply the **same** `overrides` dict to `self._val_env`, capturing its own
   `prev` values for restore;
2. `self._val_env.refresh_forests()`;
3. `self._evaluate_reference_actor("Exam-Baseline", ckpt_path=
   self.cfg.baseline_checkpoint_path or None, ckpt_cfg_path=
   self.cfg.baseline_checkpoint_config_path or None,
   env_override=self._val_env, urdf_paths_override=self._val_urdf_paths)` —
   the existing zero-ABCD / zero-decay path;
4. restore the validation env's forest params in the `finally`, next to the
   eval env's restore.

Skipped silently when `outer.exam_baseline` is false or `self._val_env is
None`. Wrapped in its own `try/except` so a baseline failure never affects
the exam objectives: `_run_exam` still returns its `(objs, diag, k)`.

Ordering: the population exam rollout runs first (unchanged), the baseline
second. They are separate envs, so the order between them does not matter.

### 3. Output

New `results/outer_exam_baseline.csv`, header written by
`_init_outer_csvs()`, one row appended per phase:

```
outer_gen, inner_gen, n_forests, fitness, velocity, progress_m,
crash_rate, cost_of_transport, v_deviation
```

Column names are the canonical objective names (not `_pack_eval_result`'s
`progress` / `cot` keys) so the plot mapping is a straight lookup.
`n_forests` is the validation env's slot count — with P=1 every slot is one
forest.

`inner_gen` comes from a new `self._last_gen`, recorded in
`_after_generation` alongside `_last_solutions`.

### 4. Plot

New `_load_exam_baseline(run_dir, name_x, name_y)` in `pareto_plots.py`,
mirroring `_load_standard_drone_baseline`:

- same `reproducibility/config.yaml` guard — `validation.enable` true and
  `validation_catalog` empty/none — so the star is only drawn when the point
  really is the standard mydrone;
- returns the **mean** of the two objective columns over all phases. The
  standard drone with the frozen generalist is a fixed system; per-phase
  variation is forest noise, so averaging is the tightest estimate.

Wiring:

```python
star = (_load_exam_baseline(run_dir, name_x, name_y) if exam_only
        else _load_standard_drone_baseline(run_dir, name_x, name_y))
```

Legend reads `"standard mydrone (zero rules, exam forests)"` when
`exam_only`, so the distribution is explicit. Runs without the CSV (every
existing `batch_3` run) get `None` and no star — today's behaviour, no crash.

### 5. Known limitation (documented, not guarded)

The validation env is rebuilt at every URDF refresh. When
`validation.period > catalog.refresh_urdfs_every` a phase can end without the
validation env having flown, making the exam-baseline rollout its first —
cold Taichi aero state (`_thr_flt` starts at 0), i.e. the known unfixed
cold-start bias, which would make the star pessimistic. With
`period <= refresh_urdfs_every` (batch_3: 1 vs 4) the env is warm. This is
recorded in the `_run_exam` docstring and the `exam_baseline` config
docstring; no warm-up rollout is added.

Second, smaller caveat, worth knowing when reading the plot: the star is
averaged over the validation env's slot count, while each URDF's exam point
comes from `E/k` slots. Same distribution, so the means are comparable; the
standard errors are not the same.

## Testing

Extend `tests/hebbian/test_exam_forest.py` (its `_FakeExamEnv` +
monkeypatched `_run_exam` harness already covers the eval-env side):

- overrides applied to **both** envs and both restored on the happy path;
- both restored when the baseline eval raises, and exam objectives survive;
- `exam_baseline: false` → validation env untouched, no CSV row;
- no validation env → exam unchanged, no CSV row;
- CSV row schema and values from a stubbed `_evaluate_reference_actor`.

New `pareto_plots` cases: `_load_exam_baseline` returns column means; returns
`None` for a non-empty `validation_catalog`, for `validation.enable: false`,
and for a missing CSV.

Finally, a short real run (small `num_urdfs`, 2 phases, tiny env) on the
cluster image to confirm the CSV is produced and the star renders — the unit
tests stub Genesis entirely.
