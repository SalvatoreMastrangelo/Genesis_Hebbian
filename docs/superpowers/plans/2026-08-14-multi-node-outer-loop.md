# Multi-Node Outer Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shard the WP2 outer-loop population evaluation across K Slurm nodes by URDF, at rollout granularity, with per-generation genome-broadcast / metric-gather synchronization — while the existing single-node command (`bash src/WP2_Outer_Loop/slurm_jobs/run_parallel.sh <batch>/`) keeps working bit-identically.

**Architecture:** Rank 0 keeps everything global (CMA-ES, NSGA-II, validation env, exam baseline, CSVs); every rank (0 included) owns a contiguous URDF shard and runs the full existing `evaluate_population_multi_urdf` on it locally via a `ShardEvalSession`. Transport is `torch.distributed` gloo (`broadcast_object_list` / `gather_object`), commands are per-generation (build / refresh / overrides / evaluate / teardown), and rank 0 merges shard metric dicts into the exact monolithic format via three new per-URDF sufficient-statistic matrices added to `evaluate.py`.

**Tech Stack:** Python 3.10, torch.distributed (gloo), Slurm srun multi-task, existing WP2 stack (Genesis, DEAP, pycma). No new dependencies.

**Spec:** The design conversation of 2026-08-14 (this session). Key agreed points: shard the D=64 URDF axis (never P or F); every node holds all P Hebbian controllers; two syncs per generation; rank 0 = coordinator + shard; K=1 degenerates to today's exact code path; intra-node `ParallelMultiSceneEvalEnv` structure unchanged inside each rank.

## Global Constraints

- The former submission command with no new flags must behave exactly as today (single task → no distributed code active; no behavior change in `evaluate.py` outputs for existing keys).
- Multi-node is opt-in purely via sbatch flags: `--nodes=K --gpus-per-node=2` (train.slurm header switches `--ntasks=1`→`--ntasks-per-node=1`, `--gpus=1`→`--gpus-per-node=1`, which is allocation-identical for the default single-node case).
- Distributed activation detection: `WP2_DIST_WORLD_SIZE`/`WP2_DIST_RANK` env vars (explicit, for local testing) take precedence; else `SLURM_NTASKS>1` + `SLURM_PROCID`. Anything else → single-process.
- All forest-refresh seeds identical across ranks (preserves the shared-forest fairness semantics); evaluate seeds offset per rank (decorrelates actor sampling / DR).
- Per-rank Taichi cache dirs when world_size>1 (avoid cross-node cache races).
- Worker ranks never create run dirs, never write CSVs, never build validation envs.
- No commits unless the user asks (repo convention overrides skill default).
- E2E verification must run in the local `mygenesis:latest` docker image before this is trusted (5-day-run safety rule).

---

### Task 1: Sufficient-statistic matrices in `evaluate.py`

**Files:**
- Modify: `src/WP2/evaluate.py` (`_rollout_episode_multi_urdf` return dict ~line 975; `evaluate_population_multi_urdf` acc dict ~line 1134 and accumulation loop ~line 1169)
- Test: `tests/hebbian/test_dist_eval.py` (new file, first test)

**Interfaces:**
- Produces: three new keys in the metrics dict, each `(N, P)` float64 episode-averaged: `per_urdf_t` (mean over F of summed alive-time), `per_urdf_v_dev` (mean over F of the RAW v_dev integral, NOT normalized by t), `per_urdf_cot_mean` (mean over F of `cot_slot` — the 1e-6-clamped, zero-where-crashed per-slot CoT used by the per-individual `cots`). Existing keys untouched.

- [ ] **Step 1: Write the failing test** — synthetic-tensor test in the style of `test_per_urdf_reduction_consistency`: build random `(D,E)` accumulators, run the three new formulas as specified, and assert the keys exist in a rollout-shaped dict produced by a small pure helper OR (simpler, chosen): test at the merge level in Task 2 and here only assert via grep-level unit: call `_rollout_episode_multi_urdf`? Not possible without env → instead assert formulas directly:

```python
def test_new_per_urdf_matrices_formulas():
    """per_urdf_t / per_urdf_v_dev / per_urdf_cot_mean must be F-means of the
    raw accumulators (v_dev raw, NOT normalized; cot with 1e-6 clamp + zeroing)."""
    torch.manual_seed(0)
    D, P, F = 3, 4, 5
    t = torch.rand(D, P * F) * 60
    vdev = torch.rand(D, P * F) * 10
    energy = torch.rand(D, P * F) * 100
    dx = torch.rand(D, P * F) * 50
    mg = torch.rand(D, 1) * 9.81
    cot_slot = torch.where(dx > 1e-6, energy / (mg * dx.clamp(min=1e-6)),
                           torch.zeros_like(dx))
    per_urdf = lambda m: m.view(D, P, F).float().mean(dim=2)
    # merge-level identity: per-ind == mean over D of per-URDF
    assert torch.allclose(per_urdf(t).mean(0), t.view(D, P, F).mean(dim=(0, 2)), atol=1e-5)
    assert torch.allclose(per_urdf(cot_slot).mean(0),
                          cot_slot.view(D, P, F).mean(dim=(0, 2)), atol=1e-5)
```

(This pins the merge identity the implementation relies on; the presence of the keys in the real dict is covered by the docker E2E in Task 6.)

- [ ] **Step 2: Run test, verify it fails** (file doesn't exist yet → collection error, then passes trivially once written — acceptable for a math-pinning test).
- [ ] **Step 3: Implement** — in `_rollout_episode_multi_urdf`, after `pu_cot` (~line 947):

```python
    # Sufficient statistics for exact cross-shard merging (WP2.dist_eval):
    # raw F-means so per-individual reductions can be reconstructed as
    # D-means downstream. v_dev stays UNnormalized (global v_dev = mean
    # v_dev integral / mean t); cot_mean is the same cot_slot the
    # per-individual `cots` averages.
    pu_v_dev = _per_urdf(v_dev_acc).cpu().numpy()
    pu_cot_mean = _per_urdf(cot_slot).cpu().numpy()
```

and add to the return dict: `"per_urdf_t": pu_t, "per_urdf_v_dev": pu_v_dev, "per_urdf_cot_mean": pu_cot_mean` (note `pu_t` already exists at line 921). In `evaluate_population_multi_urdf`: add the three keys as `np.zeros((N, P))` to `acc` and to the accumulation tuple loop.

- [ ] **Step 4: Run test to verify it passes** — `docker run … pytest tests/hebbian/test_dist_eval.py -v` (torch-only test also runs on system python if torch present; use docker to be safe).
- [ ] **Step 5: No commit** (user reviews at end).

### Task 2: `WP2/dist_eval.py` — detection, sharding, exact merge

**Files:**
- Create: `src/WP2/dist_eval.py`
- Test: `tests/hebbian/test_dist_eval.py`

**Interfaces:**
- Produces:
  - `detect_dist_env() -> Optional[Tuple[int, int]]` — `(rank, world_size)` or `None` when world_size ≤ 1.
  - `shard_urdf_indices(n_urdfs: int, world_size: int) -> List[List[int]]` — contiguous `np.array_split`, raises `ValueError` if `world_size > n_urdfs`.
  - `merge_shard_metrics(shard_metrics: List[Dict[str, np.ndarray]], fitness_aggregator: str) -> Tuple[np.ndarray, Dict]` — reconstructs the exact monolithic `(fitnesses, metrics)`:
    - concat all `per_urdf_*` and `per_slot_*` along axis 0 in rank order;
    - `t = mean_d(per_urdf_t)`, `dx = mean_d(per_urdf_progress)`, `progresses = dx`;
    - `velocities = where(t > 1e-6, dx / t, 0)`;
    - `v_deviations = where(t > 1e-6, mean_d(per_urdf_v_dev) / t, 0)`;
    - `crash_flags = mean_d(per_urdf_crash)`; `cots = mean_d(per_urdf_cot_mean)`;
    - `reward_sums` = `mean_d(per_urdf_reward)` for aggregator "mean", else `np.median` over the concatenated `per_slot_reward` reshaped `(P, D_total*F)`;
    - `reward_components` = `Σ_s D_s · comp_s / D_total`; `reward_names` from the first shard that has them.

- [ ] **Step 1: Write failing tests** — oracle test: generate random per-slot raw data for full D, compute expected metrics with an in-test numpy oracle transcribing `evaluate.py`'s formulas; slice into K shards, build each shard's metrics dict with the same formulas (shard-local), then `merge_shard_metrics` must match the full-D oracle to `rtol=1e-6`. Cases: K=2 even split, K=3 uneven (D=7), aggregator mean and median, n_comp ∈ {0, 2}. Plus `shard_urdf_indices` coverage/order/balance/raise tests and `detect_dist_env` env-var matrix (none → None; SLURM_NTASKS=1 → None; SLURM_NTASKS=4+SLURM_PROCID=2 → (2,4); WP2_DIST_* override wins).
- [ ] **Step 2: Run, verify fail** (`ModuleNotFoundError: WP2.dist_eval`).
- [ ] **Step 3: Implement** the three functions exactly as specified in Interfaces.
- [ ] **Step 4: Run tests to verify pass.**
- [ ] **Step 5: No commit.**

### Task 3: Transport + sessions — `DistContext`, `ShardEvalSession`, `DistributedEvalCoordinator`, `DistEnvHandle`, `shard_worker_main`

**Files:**
- Modify: `src/WP2/dist_eval.py`
- Test: `tests/hebbian/test_dist_eval.py`

**Interfaces (all in `WP2.dist_eval`):**
- `DistContext(rank, world_size, timeout_s)` — wraps `dist.init_process_group("gloo", init_method="env://", …)`; methods `broadcast_cmd(obj) -> obj` (rank 0 sends, all return the object), `gather(obj) -> Optional[List]` (list on rank 0, None elsewhere), `close()`.
- `ShardEvalSession(cfg, wp1_cfg, model_and_layer, rank)` — owns one shard: `build(urdf_paths: List[str], envs_per_drone: int)` (waits up to 120 s for URDF files to appear on shared FS, then `_build_multi_urdf_env` with `cfg.evaluation.num_eval_workers` / `num_gpus`), `teardown()` (env `shutdown()` if present, `gs.destroy()` if initialized), `refresh_forests(seed)` (seed `random`/`np.random`/`torch.manual_seed` with the SAME seed on every rank, then `env.refresh_forests()`), `set_dens_min(v)`, `apply_forest_overrides(ov) -> prev`, `evaluate(solutions, cfg_override, seed, verbose) -> Dict` (seed with `seed + rank * 9973`, then `evaluate_population_multi_urdf(solutions, cfg_override or self.cfg, self.model_and_layer, self.wp1_cfg, urdf_paths=self.shard_paths, existing_env=(self.env, self.shard_paths))`, return the metrics dict).
- `DistributedEvalCoordinator(ctx, session)` — rank-0 driver. `build_env(urdf_paths, envs_per_drone) -> DistEnvHandle` (broadcasts `{"op": "build_env", "shards": [...], "envs_per_drone": E}`, builds local shard, gathers acks, raises on any `("error", msg)`); `evaluate_population(solutions, cfg_override=None, verbose=False) -> (np.ndarray, Dict)` (broadcast op + seed, local evaluate, gather, `merge_shard_metrics` with `_resolve_fitness_aggregator(cfg_override or cfg)`); `teardown_env()`, `refresh_forests()`, `set_dens_min(v)`, `apply_forest_overrides(ov) -> prev` (returns rank-0 local prev), `shutdown()` (broadcast `{"op": "shutdown"}` + `ctx.close()`).
- `DistEnvHandle(coordinator, D_total, E)` — duck-types the env for `evolve_cma`/`nsga_cma` call sites: properties `.E`, `.D`; methods `refresh_forests()`, `set_dens_min(v)`, `apply_forest_overrides(ov)`, `shutdown()` (→ `coordinator.teardown_env()`).
- `shard_worker_main(cfg, rank, world_size, session_factory=None) -> None` — worker entry: loads frozen actor + WP1 cfg (same as `HebbianCMAES.__init__` head, NO run dir), builds session via factory, `DistContext`, command loop dispatching ops; every op wrapped in try/except → gathers `("ok", payload)` / `("error", f"rank {rank}: {exc}")`; `shutdown` op breaks the loop.
- Command protocol (plain dicts): `build_env{shards, envs_per_drone}` → ack; `teardown_env{}` → ack; `refresh_forests{seed}` → ack; `set_dens_min{value}` → ack; `apply_forest_overrides{overrides}` → ack with prev; `evaluate{solutions, cfg_override, seed, verbose}` → ack with metrics dict; `shutdown{}` → no ack.
- All errors on the coordinator raise `RuntimeError` listing failing ranks (the run loop's existing try/except turns evaluate failures into zero-fitness generations, same as today).

- [ ] **Step 1: Write failing protocol tests** — real gloo over localhost: `torch.multiprocessing.spawn` (or `multiprocessing.get_context("spawn").Process`) with world_size=2, `MASTER_ADDR=127.0.0.1`, free `MASTER_PORT`, `FakeSession` recording calls (no Genesis): assert shard routing (rank 1 receives shard 1's paths), evaluate merge order (rank-0 metrics rows before rank-1 rows), refresh seed identical on both ranks / evaluate seed differing by `9973`, `apply_forest_overrides` returns local prev, a `FakeSession` raising in `evaluate` on rank 1 → coordinator raises `RuntimeError` mentioning "rank 1", `shutdown` terminates the worker process cleanly (exitcode 0).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests to verify pass** (in docker; gloo needs no GPU).
- [ ] **Step 5: No commit.**

### Task 4: Wiring — `evolve_cma.py`, `nsga_cma.py`, `WP2_Outer_Loop/run.py`

**Files:**
- Modify: `src/WP2/evolve_cma.py` (`__init__` adds `self._dist = None`; new method `_evaluate_population`; `_build_env_once` multi-URDF branch; `_evaluate_reference_actor` multi branch; `_uh_reevaluate` multi branch; run-loop eval call)
- Modify: `src/WP2_Outer_Loop/nsga_cma.py` (`_run_exam` eval call)
- Modify: `src/WP2_Outer_Loop/run.py` (rank dispatch + coordinator attach + per-rank cache dirs + finally-shutdown)
- Test: `tests/hebbian/test_dist_eval.py`

**Interfaces:**
- `HebbianCMAES.attach_distributed(coordinator)` — must be called before `run()`; sets `self._dist`.
- `HebbianCMAES._evaluate_population(solutions, cfg_override=None, verbose=False)`:

```python
    def _evaluate_population(self, solutions, cfg_override=None, verbose=False):
        """Multi-URDF population eval — single-process or distributed."""
        if self._dist is not None:
            return self._dist.evaluate_population(
                solutions, cfg_override=cfg_override, verbose=verbose)
        cfg = cfg_override if cfg_override is not None else self.cfg
        existing_env = (
            (self._env, self._env_urdf_path) if self._env is not None else None
        )
        return evaluate_population_multi_urdf(
            solutions, cfg, self._model_and_layer, self._wp1_cfg,
            urdf_paths=self._urdf_paths, existing_env=existing_env,
            verbose=verbose,
        )
```

- `_build_env_once` multi branch: after computing `envs_per_drone` exactly as today, `if self._dist is not None: self._env = self._dist.build_env(self._urdf_paths, envs_per_drone); self._env_urdf_path = list(self._urdf_paths); return` (before the current local-build block; keep prints).
- Call sites switched to `self._evaluate_population(...)`: run-loop multi branch, `_evaluate_reference_actor` multi branch with `env_override is None` (pass `cfg_override=ref_cfg`), `_uh_reevaluate` multi branch, `nsga_cma._run_exam`. Validation-env calls (`env_override` set) stay direct.
- `run.py`: after `cfg.validate()` and `_configure_cache_root()` (suffix `_rank{rank}` in dist mode): `dist_info = detect_dist_env()`; rank>0 → `seed_everything(cfg.seed + rank)`, `shard_worker_main(cfg, rank, world)`, return; rank 0 with world>1 → build `DistContext` + `ShardEvalSession` + `DistributedEvalCoordinator`, `runner.attach_distributed(coord)`, wrap `runner.run()` + plots in `try/finally: coord.shutdown()`.

- [ ] **Step 1: Write failing dispatch tests** — `object.__new__(HebbianCMAES)` with hand-set attrs: `_dist=None` → monkeypatched `evaluate_population_multi_urdf` receives the exact legacy argument tuple (cfg identity, urdf_paths identity, existing_env tuple); `_dist=RecordingStub` → stub called with same solutions and cfg_override, real evaluator NOT called. Second test: `cfg_override` plumbing for the reference actor path.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement all wiring.**
- [ ] **Step 4: Run the new tests + the whole existing `tests/hebbian/` suite in docker** (regression gate: `test_outer_nsga.py`, `test_exam_forest.py` must stay green; the two pre-broken `MorphologyConfig` files stay broken).
- [ ] **Step 5: No commit.**

### Task 5: Slurm launcher — `train.slurm` multi-node

**Files:**
- Modify: `src/WP2_Outer_Loop/slurm_jobs/train.slurm`
- Test: `tests/hebbian/test_dist_eval.py` (script-content checks)

**Interfaces:** submission contract — unchanged: `bash run_parallel.sh <batch>/`; multi-node: `bash run_parallel.sh <batch>/ --nodes=4 --gpus-per-node=2`.

- [ ] **Step 1: Write failing script-content test** — `bash -n` parses; header contains `--ntasks-per-node=1` and `--gpus-per-node=1` (not `--ntasks=1`/`--gpus=1`); body exports `MASTER_ADDR`/`MASTER_PORT`; staging block is rank-guarded (`SLURM_PROCID`).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** — header swap (allocation-identical for nodes=1); before `srun`: `MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)` (fallback `127.0.0.1`), `MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))`, export both; `srun --kill-on-bad-exit=1 apptainer …`; inside the container block: URDF-mesh staging only on `SLURM_PROCID=0` + `.stage_done` sentinel, other ranks poll for the sentinel (120 s timeout); update the header comment documenting multi-node usage and that Python auto-detects ranks from `SLURM_PROCID`/`SLURM_NTASKS`.
- [ ] **Step 4: Run test to verify pass.**
- [ ] **Step 5: No commit.**

### Task 6: Docker E2E — K=2 vs K=1 on a real tiny outer run

**Files:**
- Create: `tests/hebbian/test_dist_e2e.py` (skipped unless Genesis importable — same convention as other GPU tests)

**Interfaces:** consumes the committed real checkpoint `src/WP2_Outer_Loop/experiments/batch_0/full_run_4_64_64/{checkpoint.pt,wp1_config.yaml}`.

- [ ] **Step 1: Write the E2E test** — tiny config written to tmp (N=4 URDFs, CMA pop H=4, `num_eval_envs=32` → F=2, `num_generations=4`, `refresh_urdfs_every=2`, `rescore: true`, `exam_baseline: false`, `validation.enable: true` with `n_val_envs=8`, `num_eval_workers=1`, 60 s episodes are too slow → shrink via `forest.x_upper` small + rely on crashes; accept a few minutes). Launch K=2: two `subprocess.Popen([sys.executable, "-m", "WP2_Outer_Loop.run", …])` with `WP2_DIST_WORLD_SIZE=2`, `WP2_DIST_RANK={0,1}`, `MASTER_ADDR=127.0.0.1`, free `MASTER_PORT`, distinct cwd-safe base_dir. Assert: both exit 0; run dir exists only for rank 0; `results/outer_population.csv` has 4 URDF rows per outer gen with `obj_source == "exam"` for mid-run phases; `results/cma_summary.csv` has all generations; all objective values finite. Then K=1 control with the same YAML (no dist env vars): same row structure. Compare row counts and column sets equal.
- [ ] **Step 2: Run in docker** (`mygenesis:latest`, `--gpus all`, `PYTHONPATH=/workspace/bind/src`). Iterate until green.
- [ ] **Step 3: Run the FULL hebbian test suite in docker one more time** (final regression gate).
- [ ] **Step 4: No commit** — report results.

### Task 7: Documentation

**Files:**
- Modify: `.claude/CONTEXT.md` (outer-loop section: distributed evaluation paragraph — activation, topology, seeds, files)
- Modify: memory (`project_outer_loop_rewrite.md` pointer + new memory file for the multi-node feature)

- [ ] **Step 1: Update CONTEXT.md** with the dist_eval architecture, command protocol, launcher usage, and the K=1-unchanged guarantee.
- [ ] **Step 2: Write memory file + MEMORY.md index line.**

## Self-Review Notes

- Spec coverage: URDF-axis sharding (T3/T4), per-generation sync (T3), exact metric merge (T1/T2), exam + baseline + overrides plumbing (T4 via `_evaluate_population` + handle), rank-0-only validation (untouched code = rank 0 only), same-command compatibility (T5 header swap + detection defaults), test suite (T1–T6), cache isolation (T4 run.py), staging race (T5).
- Type consistency: `merge_shard_metrics` returns `(np.ndarray, Dict)` matching `evaluate_population_multi_urdf`; `DistEnvHandle.E/D` ints; `evaluate` gathers raw metrics dicts (numpy) — all verified against call sites read in session.
- Known accepted deviations (documented in code): median aggregator with `num_episodes>1` merges as median-of-means (production uses `num_episodes=1`); merged means computed in float64 vs monolithic float32 GPU means (K>1 only; K=1 path bit-identical).
