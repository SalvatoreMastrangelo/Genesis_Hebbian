# WP2 Outer Loop — Verification Findings & Possible Fixes

**Date:** 2026-07-07
**Branch:** `outer_loop`
**Scope:** Verify the outer loop (`src/WP2_Outer_Loop/`) still works logically after the
~10 inner-loop commits that landed *after* it was written, and surface hidden
bugs / missing features.

## Context

The outer loop was added in `6165c49 "added outer_loop (to be tested)"` (2026-04-24) and
last touched in `59c841c` (2026-05-30). Since then the inner loop (`WP2/evolve_cma.py`,
`WP2/config.py`, `WP2/evaluate.py`, `WP2/run.py`, `WP2/frozen_actor.py`) received ~10 more
commits **without touching the outer loop**:

```
d908932 removed per-generation cma pickle
ed19afe MLP controller implementation
e13d8be uniform weighting option to CMA-ES
b8c009d median fitness aggregator + min/max normalization + sigma re-inflate fix
1fc9c9e forest selection field
8ac3605 latin forest init + urdf MUTATION instead of refresh
5b7c28f include_standard_drone + validation in the inner loop
779aa8b per-output ABCD rules mode
61e9683 mean -> median in inner loop
dd6a8ea UH-CMA-ES
59c841c SNR analysis + CRN
caf4103 cka
9976429 multi-gpu support + PCA + CMA-ES state plots
ec5d71d random URDF refresh in inner loop
```

The outer loop calls into the inner loop, so any interface/semantic drift can break it.

## Verification method

- Read every outer-loop module and the inner-loop call targets.
- Compared current inner-loop signatures / config fields / metric keys against what the
  outer loop assumes.
- Used the repo venv (`.venv`, has `deap 1.4` + `cma 4.4.4`) to import the full module
  chain and empirically reproduce the crashes below.

---

## ✅ What still works (no drift)

- **Entry-point signatures still match:**
  - `HebbianCMAES.run(resume_from_gen=None, x0_override=None) -> (best_genome, best_fitness)`
    (`evolve_cma.py:1620`) — outer loop calls `run(x0_override=...)`. ✔
  - `evaluate_population_multi_urdf(solutions, cfg, model_and_layer, wp1_cfg, urdf_paths, existing_env=None, verbose=False) -> (fit, metrics)`
    (`evaluate.py:943`) — outer loop calls with those exact kwargs. ✔
  - `load_frozen_actor(checkpoint_path, wp1_cfg_path, device) -> (model, last_layer, num_actions, hidden_dim)`
    (`frozen_actor.py:148`) — outer loop unpacks a 4-tuple. ✔
- **All metric keys** the outer loop reads still exist in the eval output
  (`evaluate.py:1085-1090`): `reward_sums`, `progresses`, `velocities`, `crash_flags`,
  `cots`, `v_deviations`. The `_METRIC_ALIASES` map in `evaluation.py:38` is fully covered.
- **All config fields** the outer loop sets still exist: `evaluation.{run_baseline,
  refresh_forests_per_generation, num_eval_workers, num_eval_envs}`,
  `catalog.{path, num_urdfs, force_multi_urdf}`, `cmaes.sigma_reinflate`,
  `evolution.num_generations`, `hebbian.{num_actions, hidden_dim}`.
- **`catalog.path` still pins URDFs** correctly (`evolve_cma.py:534` → multi-URDF path).
- **Full module chain imports cleanly** through the venv — no drift-induced
  `ImportError`/`NameError`/`AttributeError`.
- **NSGA-II bookkeeping is internally consistent:** crowding distance is assigned via
  `selNSGA2` at the end of each generation (enabling `selTournamentDCD` next gen), and the
  `population_genomes` array stays in sync with `current_individuals` ordering across
  offspring generation, environmental selection, inner-loop eval, and re-sort.

---

## 🔴 BLOCKER 1 — `population_size` must be divisible by 4, but the default is 2

**Where:** `nsga2.py:155` (`make_offspring` → `toolbox.mating_select(parents, k)` where
`mating_select = tools.selTournamentDCD`, `k = len(parents) = population_size`).

**Problem:** DEAP's `selTournamentDCD` **requires `k` divisible by 4** when
`k == len(individuals)`. Reproduced via venv:

```
pop= 2 (even, %4==2):  CRASH: selTournamentDCD: k must be divisible by four
pop= 4 (even, %4==0):  OK
pop= 6 (even, %4==2):  CRASH
pop= 8 (even, %4==0):  OK
pop=16 (even, %4==0):  OK
```

- Shipped default `outer_loop_default.yaml:27` → `population_size: 2` **crashes at outer-gen
  1** (gen 0 has no offspring, so the run survives exactly one generation then dies inside
  `make_offspring`).
- `OuterLoopConfig.validate()` (`config.py:222-246`) only enforces *even* + *≥2*
  (`config.py:225`), so 2 / 6 / 10 / 14 all pass validation and then crash.
- Note the inconsistency: the dataclass default is `population_size: int = 16` (`config.py:96`,
  divisible by 4, OK) but the **YAML** default is `2` (broken).

**Fix:**
1. In `OuterLoopConfig.validate()` replace the `% 2` check with `% 4`:
   ```python
   if self.population_size % 4 != 0:
       raise ValueError(
           f"population_size must be divisible by 4 for DEAP selTournamentDCD "
           f"(got {self.population_size})"
       )
   ```
2. Change `outer_loop_default.yaml:27` to `population_size: 4` (or 16, matching the
   dataclass default).

---

## 🔴 BLOCKER 2 — Outer loop skips the checkpoint → Hebbian-dim inference

**Where:** `outer_loop.py` (`OuterLoop.__init__` / `_build_inner_cfg`) never infers layer
dims; `evaluation.py:100-104` and `162-170` build `model_and_layer` from
`inner_cfg.hebbian.{num_actions, hidden_dim}`.

**Problem:** The inner loop's genome size is computed from the config, not the checkpoint:
- `HebbianCMAES.__init__` computes `self.n_genes = cfg.hebbian_genome_dim()`
  (`evolve_cma.py:494`) **immediately**, and builds `model_and_layer` from
  `cfg.hebbian.num_actions/hidden_dim` (`evolve_cma.py:505-508`) — it **discards** the real
  dims that `load_frozen_actor` returns (`_, _` at `evolve_cma.py:501`).
- The dims are reconciled from the checkpoint in **`WP2/run.py:178-187`**, i.e. the inner
  loop's CLI entry point, *before* `HebbianCMAES` is constructed:
  ```python
  _last_key = last_actor_linear_key(_sd)
  if _last_key is not None:
      cfg.hebbian.num_actions = _sd[_last_key].shape[0]
      cfg.hebbian.hidden_dim  = _sd[_last_key].shape[1]
  ```
  (The similar block at `evolve_cma.py:1122-1132` is **baseline/specialist-only** — it edits
  a deepcopied `ref_cfg`, not `self.cfg`, so it does not rescue the main genome dim.)

**The outer loop bypasses `run.py` entirely** (it builds the inner cfg via `_build_inner_cfg`
and calls `HebbianCMAES(inner_cfg)` directly). It never runs the inference. `grep` confirms
zero dim-inference in `WP2_Outer_Loop/`. The template YAML (`cma_es_rules_only.yaml`) also
does **not** set `hebbian.num_actions/hidden_dim`, so they fall back to the dataclass
defaults **7 / 64**. Verified via venv:

```
template hebbian dims: num_actions=7 hidden_dim=64 -> n_genes=1792
```

The production controller is `Linear(32→7)` (hidden_dim=**32**, so n_genes should be **896**);
MLP / lstm-15 variants differ as well. With the wrong dims:
- Decoded rules are `(7,64)` but the real last layer is `(7,32)` → shape mismatch in the
  Hebbian update → the inner CMA-ES **crashes**.
- The per-URDF pass (`evaluate_urdfs_per_individual`) wraps eval in a `try/except` that
  returns `-inf` on failure (`evaluation.py:130-136`), so it **silently** feeds NSGA-II
  garbage instead of erroring.

The outer loop therefore only works when the checkpoint's last layer happens to be exactly
`(7, 64)`.

**Fix:** In `OuterLoop.__init__`, after resolving `self.inner_template.checkpoint_path`
(around `outer_loop.py:206-210`), infer the dims from the checkpoint and write them onto the
template so every derived inner cfg and every `evaluate_*` call inherits them:
```python
import torch
from WP2.frozen_actor import last_actor_linear_key
_sd = torch.load(self.inner_template.checkpoint_path, map_location="cpu", weights_only=False)
_sd = _sd.get("model_state_dict", _sd) if isinstance(_sd, dict) else _sd
_k = last_actor_linear_key(_sd)
if _k is not None:
    self.inner_template.hebbian.num_actions = _sd[_k].shape[0]
    self.inner_template.hebbian.hidden_dim  = _sd[_k].shape[1]
del _sd
```
(Mirror of `run.py:178-187`.)

---

## 🟠 HIGH — env-budget coupling is unvalidated (no shipped default runs end-to-end)

**Where:** `config.py:234` (`validate()`), interacting with `evaluate.py:1010-1019`.

**Problem:** The inner CMA-ES evaluates its **full** population in one call
(`evolve_cma.py:1804` `solutions = es.ask()` → `1845` `evaluate_population_multi_urdf(...)`),
so `P = cmaes.population_size` (template = **64**). Inside the evaluator:
```python
F = (num_eval_envs // N) // P      # N = number of URDFs = outer population_size
```
(`evaluate.py:1012`). `F == 0` raises `ValueError` (`evaluate.py:1013-1018`). So the inner
loop requires:

```
num_eval_envs  >=  cmaes.population_size (inner, =64)  x  population_size (outer)
```

But `OuterLoopConfig.validate()` only checks `num_eval_envs >= population_size`
(`config.py:234`) — too weak by a factor of ~64.

**Consequence — there is no working shipped default:**
- `pop=2, num_eval_envs=256` (shipped) → dies on **Blocker 1** first.
- The obvious "fix" `pop=16` → `F = (256 // 16) // 64 = 0` → inner **crash**.
- Minimal working combo: `pop=4, num_eval_envs>=256` — but that gives `F=1` (one forest per
  (URDF, individual)), which is extremely noisy inner fitness. Realistically you want
  `num_eval_envs = 64 * pop * F_desired`, e.g. `pop=4, F=8 -> num_eval_envs=2048`.

**Fix:**
1. In `validate()` add a check against the inner template's CMA population (load the template
   or pass the value in):
   ```python
   inner_pop = 64  # or read cmaes.population_size from the resolved inner template
   need = inner_pop * self.population_size
   if self.num_eval_envs < need:
       raise ValueError(
           f"num_eval_envs ({self.num_eval_envs}) < inner_cmaes_pop*population_size "
           f"({inner_pop}*{self.population_size}={need}); inner eval gets F=0 forests/pair"
       )
   ```
2. Update the default YAML + the usage examples in `run.py`/`outer_loop_default.yaml`
   header to numbers that actually run (and document the `64 * pop * F` rule of thumb).

---

## 🟡 MEDIUM — latent footguns from inner-loop features added after the outer loop

### M1. URDF refresh / mutation not neutralized
`catalog.refresh_urdfs_every` (`ec5d71d`) and `catalog.mutate` (`8ac3605`) make the inner
CMA-ES **resample random URDFs** (`evolve_cma.py:820 _refresh_urdfs`) or **mutate the current
genomes** (`evolve_cma.py:863 _mutate_urdfs`) mid-run — silently discarding the outer loop's
carefully materialized population, which is the whole premise of the outer loop. Off in the
default template (so latent, not currently firing), but `_build_inner_cfg` should defensively
force it off so a user's custom template can't break the contract:
```python
cfg.catalog.refresh_urdfs_every = 0
cfg.catalog.mutate = False
```
(Ironically `_mutate_urdfs` imports `materialize_urdfs`/`write_catalog_txt` from
`WP2_Outer_Loop.urdf_population`.)

### M2. `validation.enable: true` in the default template
With validation on (`cma_es_rules_only.yaml:110`), every inner CMA-ES builds a **separate
2048-env** validation pool on the standard mydrone (`validation.n_val_envs: 2048`) and
evaluates it every generation (`_build_validation_env_once`). Inside the outer loop this is a
lot of wasted compute per outer generation, and combined with `inner_num_eval_workers: 2`
it may strain VRAM. Not incorrect, but the outer loop should likely force
`validation.enable = False` for inner runs (decision needed — see open question).

### M3. `inner_sigma_reinflate` is a dead knob
`cmaes.sigma_reinflate` only fires on *in-inner-loop* URDF refresh (`evolve_cma.py:1793`),
which the outer loop keeps off. Since the outer loop restarts CMA fresh with `sigma0` each
generation *by design* (see the `outer_loop.py` module docstring: "The CMA-ES distribution
… is not carried"), the `inner_sigma_reinflate` override in `OuterLoopConfig`
(`config.py:116-120`, `outer_loop.py:225-231`) has **no effect**. Either remove it or wire the
carry-forward it implies. At minimum, document that it is currently a no-op.

---

## 🟢 LOW — robustness / polish

- **L1. `_build_inner_cfg` shallow-copy aliasing** (`outer_loop.py:99-109`): only
  `hebbian/evolution/evaluation/cmaes/catalog` are deep-copied; `forest` and `validation`
  are shared by reference with the template across all generations. Read-only today
  (benign), but fragile — deep-copy them too, or deep-copy the whole cfg.
- **L2. `±inf` objectives on eval failure** (`evaluation.py:135`) can yield NaN crowding
  distances in `selNSGA2`. Use a large finite penalty instead.
- **L3. No resume.** The outer loop writes `population_genomes.npy`, `objectives.npy`,
  `best_rules.npy`, `rng_state.pkl` per gen, but `run()` always starts at gen 0 — no
  `--resume` / `--from-gen` (the inner loop has this). Missing feature.
- **L4. Redundant URDF materialization.** Offspring URDFs are built in `gen_XXX/offspring/`
  (`outer_loop.py:327`), then the survivors are rebuilt in `gen_XXX/urdfs/`
  (`outer_loop.py:355`). Wasteful, not wrong.
- **L5. `PLAN.md` drift.** Module docstrings reference `PLAN.md`, which was deleted after its
  initial commit.

---

## Suggested fix priority

1. **Blocker 1** — `validate()` `%4` + default YAML `population_size` (trivial; unblocks gen 1).
2. **Blocker 2** — checkpoint dim inference in `OuterLoop.__init__` (small; unblocks real checkpoints).
3. **High** — env-budget validation + sane default numbers (small).
4. **Medium** — force refresh/mutation off (M1); decide validation (M2); document/remove
   `inner_sigma_reinflate` (M3).
5. **Low** — L1–L5 polish (optional).

## Open questions for the user

- **M2:** force `validation.enable = False` for inner runs, or keep validation on (useful
  diagnostics, but a 2048-env pool per inner generation)?
- **M3:** remove `inner_sigma_reinflate`, or actually carry the CMA covariance/step-size
  across outer generations (changes the algorithm's semantics)?
- Apply **all** tiers, or blockers + High only for now?
