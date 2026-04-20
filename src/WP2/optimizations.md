# WP2 Performance Optimizations

Code audit identifying bottlenecks and optimization opportunities across the WP2
evaluation pipeline. Updated 2026-04-20 after the multi-URDF refactor from
`multi_urdf_utils.multi_drone_env.MultiDroneEnv` (single shared Genesis scene) to
`WP2.multi_scene_eval_env.MultiSceneEvalEnv` (K independent `WingedDroneEnv`
scenes, one per URDF — mirrors WP1 `Gen_Env`).

The refactor fixed a cross-URDF NaN contamination bug but introduced a ~1.8×
runtime regression (measured: gen 0 went from 120s to 217s with `num_urdfs=4`,
`num_eval_envs=512`). The new MULTI-SCENE section below targets that regression
directly.

Organized by priority.

---

## Compatibility constraints — two scripts must keep working

Any optimization below must not break these two standalone entry points, which
bypass the CMA-ES driver but share most of the WP2 / WingedDroneEnv code:

```bash
# 1) Hebbian-genome visual evaluation (writes camera/depth/overlay videos)
PYTHONPATH=src python src/winged_drone_train/eval_visual.py \
    -h  logs/runs_hebbian/<RUN> \
    --genome logs/runs_hebbian/<RUN>/best_individual/<METRIC>/genome.npy

# 2) Standalone Hebbian-vs-baseline compare (no CMA-ES loop)
PYTHONPATH=src python src/WP2/evaluate.py \
    --run logs/runs_hebbian/<RUN> \
    --x-upper 1600 --num-envs=10000 --compare --best crash_rate
```

### What they touch

| Entry point | Env | Rollout | Actor build |
|---|---|---|---|
| `eval_visual.py::_run_hebbian` | `WingedDroneEnv(num_envs=1, eval=True)` direct ([eval_visual.py:1077](../winged_drone_train/eval_visual.py#L1077)) | `run_and_record` — `while not done.all(): env.step(act)` ([eval_visual.py:795](../winged_drone_train/eval_visual.py#L795)) | `build_isolated_population_actor(K=1, S=1)` ([eval_visual.py:1237](../winged_drone_train/eval_visual.py#L1237)) |
| `WP2/evaluate.py::__main__` | `_build_env` (single-URDF WingedDroneEnv) ([evaluate.py:1040](evaluate.py#L1040)) | `_rollout_episode_reward_sum` ([evaluate.py:97](evaluate.py#L97)) via `evaluate_population_cma_batched` | `build_isolated_population_actor(K=1, S=actual_envs)` |

Neither command touches `MultiSceneEvalEnv` or `_rollout_episode_multi_urdf`.
Both read concrete attributes off the env directly — `env.base_pos`,
`env.base_quat`, `env.base_lin_vel`, `env.base_euler`, `env.joint_position`,
`env.commands`, `env.last_reward_total`, `env.power`, `env.nan_envs`,
`env.reset_buf`, `env.thrust_log`, `env.depth`, `env.privileged_obs_buf`,
`env.alpha`, `env.beta`, `env.pre_collision`, `env.pre_wall_crash`,
`env.pre_angle_limit`, `env.MAX_DISTANCE`. Any rewrite of `WingedDroneEnv`
state extraction must keep these populated with the same values.

### Rules for each optimization

**Must stay behind a default-off flag (flip the default = break both scripts):**
- **OPT-A1** (auto-reset in eval) — default `auto_reset=True`. Both rollouts
  here rely on auto-reset; only `MultiSceneEvalEnv` should pass `False`.
- **OPT-A2** (skip critic obs) — default `skip_critic_obs=False`. `eval_visual`
  reads from `extras["observations"]["critic"]` in some code paths; keep
  `privileged_obs_buf` populated unless explicitly disabled.

**Safe IF numerical behaviour is preserved (both scripts exercise the path):**
- **OPT-04** (mask-multiply `.any()`/`.all()` syncs) — affects
  `_rollout_episode_reward_sum` AND `eval_visual.run_and_record`. The subtle
  trap: `dx_acc[alive] = env.base_pos[alive, 0] - x0[alive]` uses assignment,
  so after auto-reset puts `base_pos` back to init for a done env, a naive
  mask-multiply would overwrite `dx_acc` with `(init - x0)`. Use
  `dx_acc = torch.where(alive, new_val, dx_acc)` to freeze per-env values on
  first termination.
- **OPT-05** (layout change `P*D*F → D*P*F` in `BatchedHebbianLastLayer`) —
  `build_isolated_population_actor` is called by BOTH scripts (K=1).
  Internal weight indexing must be updated consistently across actor build,
  Hebbian update, and both rollouts.
- **OPT-11** (mask-multiply `nan_mask` branch in `WingedDroneEnv.step`) —
  straightforward but easy to forget zeroing `last_reward_components` or
  `privileged_obs_buf`; both scripts crash (or produce garbage videos) if
  NaN leaks through.
- **OPT-B1** (coalesce six `.get_*()` queries) — whatever attribute rewrite
  happens, the attributes listed in the table above must keep the same
  tensors at the same shapes / dtypes.
- **OPT-03** (drop actor `deepcopy`, cache model) — verified safe: neither
  `attach_hebbian` nor `HebbianLastLayer` mutates `last_layer.weight`
  ([hebbian.py:68](hebbian.py#L68), [frozen_actor.py:177](frozen_actor.py#L177)).
  Sharing the frozen backbone across callers is fine.

**Unconditionally safe (don't touch the two scripts' code paths):**
- OPT-A3, A4, A5, A6, 07, 08, 12, 13, 14, 15, 16, 17, B2, B3, B5.

### Regression gate

Before shipping anything in Tier S / Tier A, run both commands against a
pinned checkpoint (e.g. `logs/runs_hebbian/2026-04-19_15-53-59_...`) and diff
the reported metrics:

- `WP2/evaluate.py --compare` prints a 5-row Baseline/Hebbian/Delta table;
  deltas should match within rollout stochasticity (σ ≈ a few percent of
  reward).
- `eval_visual.py` emits videos + `pretty_print_stats`; the printed stats
  block should reproduce the same final reason (`success` / `collision` /
  `timeout`) and distance within the same noise band.

If either shifts meaningfully, the optimization changed semantics, not just
runtime — revert before moving on.

---

## MULTI-SCENE — Recover the refactor's runtime regression

### OPT-A1 · `WingedDroneEnv` auto-resets inside every `step()` during eval
**File:** `winged_drone_train/env.py:1374-1382`
**Estimated impact:** 20–40% rollout time on multi-URDF path (biggest single win)
**Status: DONE 2026-04-20.** Added `auto_reset=True` kwarg to
`WingedDroneEnv.__init__`; `step()` skips the reset block when `False`.
`MultiSceneEvalEnv` constructs sub-envs with `auto_reset=False`.

`WingedDroneEnv.step()` always auto-resets crashed envs inline:

```python
reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
self.reset_idx(reset_env_ids)
if reset_env_ids.numel() > 0:
    self.depth[reset_env_ids] = self.MAX_DISTANCE
    self._rebuild_observations()
```

In WP2 eval the rollout marks `done[d, e] = True` on first crash and ignores
that slot thereafter, so the auto-reset (`set_dofs_position` →
`collider.reset`/`clear` → forward kinematics → `_rebuild_observations`) is
pure waste. By mid-rollout ~3/4 of envs are crashed, which means every step
does hundreds of wasted resets × D sub-envs.

The legacy `MultiDroneEnv` used a `training_mode=False` branch that just
accumulated `ds.done` and skipped reset entirely.

**Fix:** Add an `auto_reset` flag (default `True`) to `WingedDroneEnv.__init__`
that, when `False`, gates the auto-reset block in `step()`. Set it to `False`
from `MultiSceneEvalEnv` at construction time:

```python
WingedDroneEnv(..., eval=True, auto_reset=False)
```

No changes to the rollout needed — it already tracks `done` externally.

---

### OPT-A2 · WingedDroneEnv builds critic obs every step; WP2 actor never reads it
**File:** `winged_drone_train/env.py:1208-1218, 1346-1347, 1360-1361`
**Estimated impact:** 5–10% rollout time

`_rebuild_observations()` computes both `obs_actor` and `obs_critic`; the
critic stream includes privileged features (joint positions/velocities,
angular velocity, etc.) that the frozen WP1 actor in WP2 never sees. The
critic computation is done twice per step (once in normal path, once in
post-reset rebuild) and also stacked into `privileged_obs_buf` and exposed in
`extras`.

**Fix:** Add a `skip_critic_obs` flag (default `False`) on `WingedDroneEnv`.
When set, call `obs_builder.build_observations` with a variant that returns
only the actor stream, and skip the `privileged_obs_buf` write. Set it to
`True` from `MultiSceneEvalEnv`.

---

### OPT-A3 · D-way Python loop in `MultiSceneEvalEnv.step`/`reset`
**File:** `WP2/multi_scene_eval_env.py:step, reset`
**Estimated impact:** Marginal (µs per step), but free to eliminate

`MultiSceneEvalEnv.step` iterates Python-side over `self.drones`; each
iteration dispatches one Genesis `scene.step()`, one obs copy, one reward
copy, etc. The sub-env step itself is the bottleneck, not the Python loop,
but two small wins are available:

- `obs_sub.to(self.device)` is a no-op when already on device but still pays
  dispatch cost. Use `self.obs_buf[d].copy_(obs_sub, non_blocking=True)`
  directly.
- Skip `rew` and `dones` tensor allocations inside `step()` — the rollout
  reads per-sub-env state via `env.drones[d]` anyway, so the stacked return
  values are unused.

---

### OPT-A4 · Each sub-env generates its own forest pool (memory + compile time)
**File:** `winged_drone_train/env.py` (forest generator per env)
**Estimated impact:** ~D× forest memory savings; faster scene build

Every `WingedDroneEnv` runs its own `_forest_generator` at init — with D=16
URDFs we generate 16 independent forest pools even though `MultiSceneEvalEnv`
synchronises them to identical contents anyway via `refresh_forests`.

**Fix:** In `MultiSceneEvalEnv.__init__`, after building the sub-envs,
overwrite `drones[k>=1].cylinders_array` with `drones[0].cylinders_array`
(a view/reference, not a copy) so all D sub-envs share one pool. Update
`refresh_forests` to write once to `drones[0]` and broadcast.

---

### OPT-A5 · `scene.step()` calls run sequentially; D independent scenes could overlap
**File:** `WP2/multi_scene_eval_env.py:step`
**Estimated impact:** Up to D× theoretical; realistic 1.3–1.8× if Genesis
cooperates

The D sub-scenes are independent, so their `scene.step()` calls could run
concurrently on the GPU via CUDA streams. Genesis uses Taichi, which may
serialize internally on a single stream — needs investigation. If Taichi
supports multi-stream or we can dispatch scenes to alternating streams, the
D-way latency becomes roughly 1× plus launch overhead.

**Fix (exploratory):**
```python
streams = [torch.cuda.Stream() for _ in range(self.D)]
for d, (sub, stream) in enumerate(zip(self.drones, streams)):
    with torch.cuda.stream(stream):
        sub.step(actions[d])
torch.cuda.synchronize()
```

Skip if Taichi serializes. Measure before shipping.

---

### OPT-A6 · Stochastic baseline re-evaluated every generation (unchanged from prior audit)
**File:** `WP2/evolve_cma.py:941`
**Estimated impact:** ~50% total runtime reduction when `run_baseline: true`

Unchanged recommendation. The baseline is a frozen actor with zero Hebbian
rules; its fitness varies only due to rollout stochasticity. Cache the result
and re-evaluate every N generations (e.g., 10) or set `run_baseline: false`
and compute once offline.

```python
# In HebbianCMAES.__init__:
self._cached_baseline = None

# In the generation loop (replaces line 941):
if self.cfg.evaluation.run_baseline:
    if self._cached_baseline is None or gen % 10 == 0:
        self._cached_baseline = self._evaluate_baseline(verbose=verbose)
    baseline = self._cached_baseline
```

---

## P0 — Near-Instant Wins

### OPT-01 · Baseline re-evaluated every generation
Moved to [OPT-A6](#opt-a6--stochastic-baseline-re-evaluated-every-generation-unchanged-from-prior-audit)
above.

---

### OPT-02 · ~~`torch.zeros(self.E, ...)` inside reward loop in `multi_drone_env.py`~~
**Status: OBSOLETE after 2026-04-20 refactor.**

The multi-URDF rollout no longer uses `multi_urdf_utils.multi_drone_env.py` —
rewards are now computed by `WingedDroneEnv._accumulate_rewards()` at
[`winged_drone_train/env.py:1183`](../winged_drone_train/env.py#L1183).
That function already uses pre-allocated buffers (`self.rew_buf`,
`self.last_reward_components`, `self.episode_sums[name]`) — the per-call
allocation problem described here only existed in the legacy
`_compute_rewards_drone`, which is no longer on the hot path.

**However**, each reward function callable
(`self.reward_functions[name]()`) may still allocate intermediate tensors —
worth auditing individually if profiling shows it. Superseded by the more
targeted OPT-B* items below.

---

## P1 — High Impact, Require Code Changes

### OPT-03 · Actor deepcopy and checkpoint reload every generation
**File:** `WP2/frozen_actor.py:370, 381` (lines shifted from prior audit)
**Estimated impact:** 100–500ms per generation; scales with number of generations

Still applicable. `build_isolated_population_actor()`:
1. Line 370 — `torch.load(checkpoint_path, ...)` (reads the checkpoint from
   disk to extract the obs normalizer).
2. Inside `load_frozen_actor()` at `frozen_actor.py:107` — a second
   `torch.load(...)` of the same file.
3. Line 381 — `copy.deepcopy(model)` to produce a per-generation copy.

**Fix (unchanged):** Load the checkpoint once in `HebbianCMAES.__init__`;
pass the cached model into `build_isolated_population_actor`. Remove the
deepcopy — the model is frozen (`requires_grad=False`), never mutated, and
the Hebbian W lives outside it in `BatchedHebbianLastLayer`.

---

### OPT-04 · `done.all()` / `alive.any()` GPU→CPU sync every step
**Files:**
- Single-URDF rollout: `WP2/evaluate.py:121, 129, 137`
- Multi-URDF rollout: `WP2/evaluate.py:587, 609, 619`

**Estimated impact:** 15–25% rollout time

Still applicable. Both rollouts call `done.all()`/`alive_d.any()`/`just_done.any()`
each step, which forces a CUDA sync. Over ~1500 steps × 2 generations × 50
generations = ~150k syncs per run.

**Fix 1 — Mask-multiply instead of boolean-gate:**
```python
alive_f = alive.float()
reward_sum += ds.last_reward_total * alive_f
t_acc += alive_f * dt
```
For the multi-URDF version, use `alive_d` as the mask.

**Fix 2 — Throttle `done.all()` checks:**
```python
for step in range(max_episode_steps):
    ...
    if step % 50 == 0 and done.all():
        break
```

Bound the loop by `max_episode_steps` explicitly (read from `env.dt` and
`env.episode_length_s`) and only sync every 50 steps to detect early exit.

---

### OPT-05 · Double `.permute(...).contiguous()` per step in multi-URDF rollout
**File:** `WP2/evaluate.py:588, 592`
**Estimated impact:** 10–15% rollout time on multi-URDF path

Still applicable — the new `MultiSceneEvalEnv` returns `(D, E, obs)` stacked
the same way, and the rollout still permutes to `(P, D, F, obs)` for the
actor:
```python
obs_pdf = obs.view(D, P, F, -1).permute(1, 0, 2, 3).contiguous()  # line 588
actor_obs = obs_pdf.view(P * D * F, -1)
actor_act = actor.act(actor_obs)
act_pdf = actor_act.view(P, D, F, -1).permute(1, 0, 2, 3).contiguous()  # line 592
```

With D=16, P=128, F=8, obs_dim=36, each permute+contiguous copies ~2MB; over
5000 steps that's ~20GB of no-op memory traffic.

**Fix:** Change `hebbian.W` layout from `(P*D*F, out, in)` to
`(D*P*F, out, in)` (matching the env's `(D, E)` layout). This requires
updating the stacking order in `build_isolated_population_actor` and the
per-env slot indexing in `BatchedHebbianLastLayer`. An alternative is to
have `MultiSceneEvalEnv` expose obs in `(P, D, F, obs)` order, which avoids
the first permute only.

---

### OPT-06 · ~~Six separate `.get_*()` calls per drone in `multi_drone_env.py:786-803`~~
**Status: OBSOLETE** — that file is no longer on the WP2 hot path.

**Superseded by OPT-B1** below — the same pattern still exists inside
`WingedDroneEnv.step()` (now called D times per rollout step from
`MultiSceneEvalEnv`), so the total query count is actually similar.

---

## P1-new — WingedDroneEnv now on the hot path

### OPT-B1 · Six `.get_*()` calls per sub-env per step
**File:** `winged_drone_train/env.py:1267-1297`
**Estimated impact:** High on multi-URDF path

After the refactor `WingedDroneEnv.step()` is called D times per rollout
step. Each call hits the rigid solver six times for state extraction:

```python
dofs_pos = self.drone.get_dofs_position()       # kernel 1
self.base_quat[:] = self.drone.get_quat()       # kernel 2
self.joint_velocity[:] = self.drone.get_dofs_velocity()[:, ...]  # kernel 3
self.torque[:] = self.drone.get_dofs_control_force(...)           # kernel 4
self.base_lin_vel[:] = self.rigid_solver.get_dofs_velocity()[:, :3]  # kernel 5
self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), ...)  # kernel 6
```

Total: 6 × D × T ≈ 6 × 16 × 5000 = 480k Genesis API calls per rollout, each
an individual Taichi kernel launch.

**Fix options:**
- Use a batched Genesis API if one exists (`scene.get_all_dof_states()` or
  similar).
- Cache `get_dofs_velocity()` — it's called twice in `step()` (lines 1288,
  1296) with different slicing.
- Read `base_lin_vel` as `dofs_velocity[:, :3]` from the same tensor already
  fetched instead of a second solver query.

---

### OPT-B2 · Episode dict rebuilt per `reset_idx` call
**File:** `winged_drone_train/env.py:1451-1462`
**Estimated impact:** Small (~0.1–0.5% rollout) but free

`reset_idx` rebuilds a Python dict `self.extras["episode"]` from scratch
every time it runs (once per step if any env crashes). In WP2 eval with
auto-reset off (see OPT-A1), this goes away. Otherwise pre-allocate the
episode dict keys once and just overwrite values.

---

### OPT-B3 · Two full observation rebuilds per step on crash-heavy rollouts
**File:** `winged_drone_train/env.py:1347 and 1380`
**Estimated impact:** 5–10% rollout time before OPT-A1 lands

`step()` calls `_rebuild_observations()` twice when any env resets (once
from the normal physics update, once after `reset_idx`). In eval this is
redundant because (a) the second rebuild only matters for the re-entered
trajectories and (b) OPT-A1 will remove in-step resets entirely.

After OPT-A1 the second rebuild is dead; before that, short-circuit when
`reset_env_ids.numel() == 0`.

---

## P2 — Medium Impact

### OPT-07 · Constant Hebbian terms recomputed every step
**File:** `WP2/hebbian.py:152`
**Estimated impact:** ~5% rollout time

Unchanged. `self.W.mul_(1.0 - self.lam).add_(dW).add_(self.lam * self.W_checkpoint.unsqueeze(0))`
recomputes `(1 - lam)` and `(lam * W_checkpoint)` per step. Both are constant
for the whole episode — move to `reset_weights()`.

---

### OPT-08 · Intermediate tensor allocations in `hebbian_update`
**File:** `WP2/hebbian.py:135, 144`
**Estimated impact:** 5–10% rollout time

Unchanged. Pre-allocate `_xy_buf` and `_dW_buf`; use `torch.bmm(..., out=)`
and in-place `add_`/`mul_` to eliminate per-step allocations. Or wrap
`hebbian_update` in `torch.compile(..., mode="reduce-overhead")` to let
PyTorch fuse it.

---

### OPT-09 · Python loop for MLP forward pass
**File:** `WP2/frozen_actor.py:312`
**Estimated impact:** ~5µs × steps; minor but free

```python
for layer in self._actor_layers[:-1]:
    x = layer(x)
```

Replace with `self._mlp = nn.Sequential(*self._actor_layers[:-1])` built once
in `__init__`, then `x = self._mlp(x)` in `act()`.

**Caveat:** `_actor_layers` iterates `_modules.values()` to preserve the
reused ELU instance. `nn.Sequential(*list)` keeps references, so this is
still safe — but verify ELU appears at both expected positions in the
resulting `Sequential`.

---

### OPT-10 · `Normal(y, std).rsample()` per step
**File:** `WP2/frozen_actor.py:327-329`
**Estimated impact:** ~1–2µs/step

Unchanged. Cache `self._std = model.std.detach().clone()` in `__init__`,
then inline:
```python
if self.stochastic and self._std is not None:
    action_raw = y + self._std * torch.randn_like(y)
```

---

### OPT-11 · `nan_mask.any()` GPU→CPU sync
**File:** `winged_drone_train/env.py:1351` (was `multi_drone_env.py:1044`,
now obsolete)
**Estimated impact:** T syncs × D sub-envs

In `WingedDroneEnv.step()`:
```python
nan_mask = self.nan_envs.bool()
if nan_mask.any():
    self.rew_buf[nan_mask] = 0.0
    ...
```

Replace with unconditional mask-multiply:
```python
safe_mask = (~self.nan_envs.bool()).float()
self.rew_buf *= safe_mask
self.last_reward_components *= safe_mask.unsqueeze(1)
self.last_reward_total *= safe_mask
self.obs_buf *= safe_mask.unsqueeze(1)
self.privileged_obs_buf *= safe_mask.unsqueeze(1)
self.reset_buf |= self.nan_envs.bool()
self.pre_nan |= self.nan_envs.bool()
```

No sync.

---

### OPT-B4 · Per-step `.to(device)` on actions/obs even when already on device
**Files:** `winged_drone_train/env.py:1242-1243` (actions), `WP2/multi_scene_eval_env.py:step` (obs_buf copy)
**Estimated impact:** Small but free

Each `sub.step(actions[d])` does:
```python
if actions.device != self.device:
    actions = actions.to(self.device)
```

The actor returns actions on `self.device` already, so this is a guard with
one Python branch per sub-env per step. Document the invariant and remove
the branch, or hoist to the caller.

---

## P3 — Low Hanging, Low Impact

### OPT-12 · Checkpoint loaded from disk three times per run
**Files:** `WP2/run.py:115`, `WP2/frozen_actor.py:107`, `WP2/frozen_actor.py:370`

Unchanged. Load once in `WP2/run.py:115`, pass the state dict (and inferred
weight shapes) to downstream functions.

---

### OPT-13 · Rule stacking uses triple Python loop
**File:** `WP2/frozen_actor.py:399-404`

Unchanged. Replace:
```python
for key in rule_keys:
    per_env = []
    for rules in hebbian_rules_per_individual:
        t = rules[key].to(device)
        for _ in range(S):
            per_env.append(t)
    expanded[key] = torch.stack(per_env, dim=0)
```

with:
```python
for key in rule_keys:
    stacked = torch.stack([rules[key].to(device) for rules in hebbian_rules_per_individual])
    expanded[key] = stacked.repeat_interleave(S, dim=0)
```

---

### OPT-14 · Constant tensors recreated every generation
**Files:** `WP2/evaluate.py:398, 408, 807-809`

Unchanged. Cache `torch.arange(S, ...)` / `torch.linspace(vmin, vmax, F, ...)`
on the `HebbianCMAES` runner at first use — they never change within a run.

---

### OPT-15 · Redundant `.clone()` in `_expand_rule` / `reset_weights`
**Files:** `WP2/hebbian.py:71, 92, 98`

Unchanged. `torch.stack` already returns a contiguous copy, so
`.contiguous().clone()` right after it creates a redundant duplicate. Drop
the `.clone()` at lines 71, 92, 98.

---

### OPT-16 · Six sequential `.cpu().numpy()` transfers at episode end
**Files:** `WP2/evaluate.py:161-166, 662-667`

Unchanged. Stack into a single tensor before `.cpu().numpy()` to collapse
six syncs to one. At multi-URDF rollout end (lines 662-667):

```python
stacked = torch.stack([
    reward_per_ind, t_per_ind, dx_per_ind, energy_per_ind, v_dev_per_ind, crash_per_ind.float()
])
arrs = stacked.cpu().numpy()
reward_arr, t_arr, dx_arr, energy_arr, v_dev_arr, crash_arr = arrs
```

---

### OPT-17 · Dangling `torch.no_grad().__enter__()` in worker
**File:** `general_policy/super_scene/worker.py:123`
**Type:** Correctness, not performance

Unchanged. The context manager is entered but never exited. Wrap the worker
loop in `with torch.no_grad():` instead.

---

### OPT-B5 · `WingedDroneEnv.step` not under `torch.inference_mode`
**File:** `winged_drone_train/env.py:1230`

`@torch.inference_mode()` is stricter than `@torch.no_grad()` and faster
(skips version counter updates). WP2 eval never backprops through `step`,
so decorating `WingedDroneEnv.step` with `@torch.inference_mode()` is safe
and ~2–5% faster on the rollout loop. Already used on `reset()` (line 1533).

---

## Ranked Summary (most important → least, after 2026-04-20 refactor)

Ranking is by *expected wall-clock reduction on a full production run* with
`num_urdfs=16, popsize≈64, num_generations=50`, weighted by implementation
cost. Top of the list is where to start.

### Tier S — Do first (each should move the needle by ≥15%)

| Rank | ID | File | Description | Why it wins |
|---:|---|---|---|---|
| 1 | OPT-A6 | `WP2/evolve_cma.py:941` | Cache baseline across generations | Baseline is a full rollout with the same frozen actor; today it runs once per generation. Trivial 3-line fix, single-digit % wall-clock savings per gen × 50 gens = ~50% of total runtime. |
| 2 | OPT-A1 | `winged_drone_train/env.py:1374-1382` | Disable auto-reset in eval | By mid-rollout ~75% of envs have crashed. Each `step()` does `set_dofs_position` + `collider.reset/clear` + forward kinematics + `_rebuild_observations` on all of them, per sub-env, for nothing — the rollout already ignores them. Add `auto_reset=False` flag, wire through. 20–40% rollout time. |
| 3 | OPT-04 | `WP2/evaluate.py:121, 129, 137, 587, 609, 619` | Mask-multiply instead of `.any()`/`.all()` syncs | 3–6 CUDA syncs per step × ~1500 steps × (D+1) × many gens. Replace masked scatter/gather with elementwise multiply; throttle `done.all()` to every 50 steps. 15–25% rollout time. |

### Tier A — Do next (each 5–15%)

| Rank | ID | File | Description | Why it wins |
|---:|---|---|---|---|
| 4 | OPT-B1 | `winged_drone_train/env.py:1267-1297` | Coalesce 6 state queries per step | After the refactor each `step()` is called D times per rollout step. 6 solver kernels × D × T ≈ 480k Genesis API calls per rollout; already one is duplicated (`get_dofs_velocity` twice). Easy dedupe, then hunt for a batched API. |
| 5 | OPT-05 | `WP2/evaluate.py:588, 592` | Eliminate double `permute().contiguous()` | ~20GB of no-op memory traffic per rollout for D=16, P=128, F=8. Needs `BatchedHebbianLastLayer` layout change (P×D×F → D×P×F), more invasive than A/4 items but a clean one-shot win. 10–15%. |
| 6 | OPT-A2 | `winged_drone_train/env.py:1208-1218, 1346-1347` | Skip critic obs in eval | WP2's frozen actor never reads `privileged_obs_buf`. Cheaper variant of `build_observations` + kill the post-reset rebuild. 5–10%, cheap once A1 lands. |
| 7 | OPT-11 | `winged_drone_train/env.py:1351` | Mask-multiply `nan_mask` branch | `nan_mask.any()` forces one sync per step per sub-env. Unconditional masked zero eliminates that class of stall. Pairs naturally with OPT-04. |
| 8 | OPT-08 | `WP2/hebbian.py:135, 144` | Pre-allocate Hebbian scratch (or `torch.compile`) | The per-step einsum+ABCD update allocates several `(N, out, in)` intermediates. `torch.compile(mode="reduce-overhead")` is a one-liner that usually fuses the whole thing; manual buffers are a safer fallback. 5–10%. |
| 9 | OPT-07 | `WP2/hebbian.py:152` | Precompute `(1-λ)` and `λ·W_ckpt` once/episode | Two constant tensors rebuilt every step. Move to `reset_weights()`. Tiny code change, ~5%. |
| 10 | OPT-B3 | `winged_drone_train/env.py:1347, 1380` | Avoid double obs rebuild on crash | Before A1: short-circuit the post-reset rebuild when `reset_env_ids.numel() == 0`. After A1: dead code. Only worth doing as a one-liner if A1 is deferred. |

### Tier B — Small (1–5%) but mostly one-line fixes

| Rank | ID | File | Description | Why it wins |
|---:|---|---|---|---|
| 11 | OPT-03 | `WP2/frozen_actor.py:370, 381` | Cache frozen actor; drop `deepcopy` + redundant `torch.load` | 100–500 ms per generation × 50 gens = 5–25 s total, but also saves transient GPU memory doubling. Cheap. |
| 12 | OPT-B5 | `winged_drone_train/env.py:1230` | `@torch.inference_mode()` on `step()` | One decorator. Faster than `no_grad()` (no version counter). 2–5%. |
| 13 | OPT-10 | `WP2/frozen_actor.py:327-329` | Cache `std` once; inline reparameterization | `Normal(y, std).rsample()` rebuilds a Python object per step. Replace with `y + std * randn_like(y)`. 1–2µs/step. |
| 14 | OPT-09 | `WP2/frozen_actor.py:312` | MLP loop → `nn.Sequential` | 4 Python dispatches/step → 1. Tiny but free. |
| 15 | OPT-13 | `WP2/frozen_actor.py:399-404` | `repeat_interleave` instead of triple Python loop | Runs once per generation, not per step — low priority but trivially cheaper. |
| 16 | OPT-15 | `WP2/hebbian.py:71, 92, 98` | Drop redundant `.contiguous().clone()` | Once-per-episode allocations, just drop the dead `.clone()`. |
| 17 | OPT-14 | `WP2/evaluate.py:398, 408, 807-809` | Cache constant `arange`/`linspace` | Once-per-generation allocations. Cheap to cache on the runner. |
| 18 | OPT-12 | `WP2/run.py:115`, `WP2/frozen_actor.py:107, 370` | Load checkpoint once | Startup-only; removes two disk reads. Near-free. |
| 19 | OPT-16 | `WP2/evaluate.py:161-166, 662-667` | Batch episode-end CPU transfers | 6 syncs → 1 at episode end. Episode-granularity overhead only. |

### Tier C — Cleanup / correctness / exploratory

| Rank | ID | File | Description | Why it's here |
|---:|---|---|---|---|
| 20 | OPT-A4 | `WP2/multi_scene_eval_env.py:__init__` | Share forest pool across sub-envs | Memory (D× redundant forests) + scene-build time. Not on the step-level hot path. |
| 21 | OPT-A3 | `WP2/multi_scene_eval_env.py:step/reset` | Drop redundant `.to()`, skip unused return buffers | Trivial cleanup; no measurable impact. |
| 22 | OPT-B4 | `winged_drone_train/env.py:1242-1243` | Drop `actions.device` guard | One Python branch per sub-env per step; immeasurable. |
| 23 | OPT-B2 | `winged_drone_train/env.py:1451-1462` | Pre-alloc episode dict in `reset_idx` | Mostly subsumed by OPT-A1; episode-dict dict churn is minor. |
| 24 | OPT-A5 | `WP2/multi_scene_eval_env.py:step` | CUDA streams for D sub-scene parallelism | Exploratory — Taichi may serialize internally. If it works, could be the biggest single win of all (near-D× speedup). Do last, measure carefully. |
| — | OPT-17 | `general_policy/super_scene/worker.py:123` | Fix dangling `no_grad().__enter__()` | Correctness, not perf. Ship whenever touching that file. |
| — | OPT-02 | — | *Obsolete* — `multi_drone_env.py` removed from hot path | Keep marker for reference. |

### Recommended implementation order

1. **OPT-A6** (1 hr, huge win, safe): baseline caching.
2. **OPT-A1** (2-3 hr, biggest per-step win): auto-reset flag on `WingedDroneEnv`.
3. **OPT-04 + OPT-11** together (3-4 hr): sync-point cleanup; both are mask-multiply patterns.
4. **OPT-B1** (2 hr): dedupe the duplicate `get_dofs_velocity` first, check for batched Genesis API second.
5. **OPT-A2** (2 hr, cheap after A1): skip critic obs build in eval.
6. Profile again. At this point expect gen runtime to be roughly back to pre-refactor levels.
7. **OPT-05** (~half day): layout change for `hebbian.W`. Only worth it if profiling still points there.
8. **OPT-08 + OPT-07** (1 hr): try `torch.compile(hebbian_update, mode="reduce-overhead")` first — if it works cleanly, both are done.
9. Tier B items in a sweep (1-2 hr total).
10. **OPT-A5** (half day of measurement): CUDA streams experiment if runtime still matters.
