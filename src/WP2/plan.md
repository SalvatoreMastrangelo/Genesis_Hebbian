# WP2 — Hebbian Plasticity + Evolutionary Co-Optimization

> **Objective:** Joint optimization of morphology and synaptic adaptation mechanisms
> via NSGA-II, *without* retraining the base controller (no backprop).
> The output is a **Pareto frontier** of individuals trading off multiple objectives.

---

## Status Summary (2026-03-17)

### ✅ Implemented & Tested
- [x] **config.py** — `HebbianEvolutionConfig` with nested sub-configs, YAML I/O, CLI overrides
- [x] **hebbian.py** — `HebbianLastLayer` (single) and `BatchedHebbianLastLayer` (vectorized)
- [x] **frozen_actor.py** — `load_frozen_actor()`, `HebbianActorWrapper`, `BatchedHebbianActorWrapper`
- [x] **evaluate.py** — Single-individual evaluation with episode rollout and fitness computation
- [x] **objectives.py** — Velocity, energy, progress, smoothness, crash_rate (extensible registry)
- [x] **utils.py** — Seed control, genome encoding/decoding, reproducibility helpers
- [x] **evolve.py** — `HebbianCodesignDEAP` with NSGA-II, Ray parallelism, CSV logging
- [x] **run.py** — CLI entry point, config loading, cache setup, summary printing

### ⚠️ In Progress / Needs Review
- [ ] **plotting.py** — Partially implemented; verify all 10 required plots are working
- [ ] **Config YAML files** — Verify all preset configs exist and are correct
- [ ] **Batched evaluation** — `BatchedHebbianActorWrapper` integrated into `HebbianCodesignDEAP`?
- [ ] **Multi-objective weight handling** — Confirm fitness weights match active objectives

### ⓘ Known Gaps / TODO
- [ ] **Error handling in plotting** — Graceful degradation if CSVs incomplete
- [ ] **Ablation comparison plots** — Cross-run visualization (requires multiple run dirs)
- [ ] **Slurm job scripts** — WP2-specific cluster submission templates (if deploying to IZAR)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         NSGA-II Outer Loop                               │
│  Evolves: per-weight ABCD+λ rules (1600D) + morphology [0,1]^15         │
│  Produces: Pareto frontier of non-dominated individuals                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                   Fitness Evaluation (per individual)              │  │
│  │                                                                    │  │
│  │  1. Decode genome → Hebbian rules + morphology                    │  │
│  │  2. Generate URDF from morphology (Chromosome_Drone)              │  │
│  │  3. Build WingedDroneEnv with the URDF                            │  │
│  │  4. Load frozen actor, attach HebbianLastLayer                    │  │
│  │  5. Rollout N episodes (reset weights per episode + per gen)       │  │
│  │     - At each step: forward pass → Hebbian update on last layer    │  │
│  │  6. Aggregate metrics → compute multi-objective fitness            │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Module | Class/Function | Purpose |
|--------|---|---|
| **config.py** | `HebbianEvolutionConfig` | Unified configuration dataclass with YAML I/O |
| **hebbian.py** | `HebbianLastLayer` | Per-weight ABCD+λ update on frozen actor's last layer (single individual) |
| **hebbian.py** | `BatchedHebbianLastLayer` | Vectorized Hebbian for population-level eval (P individuals × S envs) |
| **frozen_actor.py** | `load_frozen_actor()` | Load WP1 checkpoint, freeze all params, extract actor only |
| **frozen_actor.py** | `HebbianActorWrapper` | Forward pass with Hebbian interception (single actor) |
| **frozen_actor.py** | `BatchedHebbianActorWrapper` | Forward pass for batched evaluation (P copies share backbone) |
| **evaluate.py** | `_build_env()` | Create `WingedDroneEnv` from morphology genome |
| **evaluate.py** | `_rollout_episode()` | Run one episode, collect metrics (velocity, energy, progress, etc.) |
| **evaluate.py** | `evaluate_individual()` | Full pipeline: decode genome → build env → rollout N episodes → fitness |
| **objectives.py** | `velocity_objective()`, `energy_objective()`, etc. | Individual fitness functions (extensible registry) |
| **objectives.py** | `compute_fitness()` | Combine metrics into fitness tuple matching active objectives |
| **evolve.py** | `HebbianCodesignDEAP` | NSGA-II outer loop: population management, variation operators, Pareto extraction |
| **run.py** | `main()` | CLI orchestration: config loading, seed control, launch evolution |
| **plotting.py** | `analyze_run()` | Generate all diagnostic plots from run results |
| **utils.py** | `seed_everything()`, `decode_hebbian_genes()`, etc. | Reproducibility and utility functions |

---

## Critical Design Rules

1. **Actor never receives morphology info** — `add_genome_obs_actor = False` in all configs
2. **Critic is dropped** — WP2 uses forward rollout only, no value estimation
3. **Hebbian acts ONLY on last `Linear(64→7)` layer** — Biases never touched, weights reset per episode + per generation
4. **Per-weight ABCD+λ** — Each of 448 weights has its own 5-parameter rule set
5. **Genome layout** — Hebbian section (0 or 2240 or 2688D) ⊕ Morphology section (0 or 15D)
6. **Weight reset policy** — Checkpoint values reset at **generation start** AND **episode start**

---

## Detailed Module Review

### 1. **config.py** ✅

**Status:** Complete. Implements the `HebbianEvolutionConfig` hierarchy with nested dataclasses.

**Key classes:**
- `HebbianConfig` — Hebbian hyperparameters (η, decay range, ABCD ranges, toggles for per-weight η and decay)
- `EvolutionConfig` — NSGA-II hyperparameters (population size, generations, crossover/mutation rates, eta_c, eta_m)
- `EvaluationConfig` — Rollout settings (num_episodes, num_envs, velocity range, stochasticity)
- `ObjectivesConfig` — Toggles for velocity, energy, progress, smoothness, crash_rate
- `MorphologyConfig` — Morphology evolution flag and fixed genome fallback
- `HebbianEvolutionConfig` — Top-level config with all sub-configs + YAML I/O + CLI overrides

**Features:**
- ✅ `to_yaml()` / `from_yaml()` for serialization
- ✅ `apply_cli_overrides()` for `--cfg.section.key value` parsing
- ✅ `active_objective_names()` / `fitness_weights()` for dynamic objective management
- ✅ `hebbian_genome_dim()` / `morphology_genome_dim()` / `total_genome_dim()` for genome size queries

**Notes:**
- All optional fields have sensible defaults
- Supports `evolve_eta=True` for per-weight learning rates (+320 genes to Hebbian section)
- Supports `evolve_decay=True` for per-weight decay (currently set to `False` in defaults)

---

### 2. **hebbian.py** ✅

**Status:** Complete. Implements ABCD+λ plasticity rule on the frozen actor's last layer.

**Key classes:**

#### `HebbianLastLayer` (single individual)
- Stores per-weight ABCD+λ tensors (each shape `(5, 64)` for 5 outputs × 64 inputs)
- **`reset_weights()`** — Restore layer weights to checkpoint values (bias untouched)
- **`hebbian_update(x, y)`** — Apply ABCD rule in-place:
  ```
  dW = η * (A·outer(y,x) + B·x + C·y + D)
  W ← (1-λ)·W + dW
  W ← clip(W, -w_max, +w_max)
  ```

**Key design:**
- Last layer weight is converted from `nn.Parameter` to buffer (in-place modifiable without autograd overhead)
- `hebbian_update()` averages activations across batch before applying rule (standard Hebbian convention)
- Bias is never modified (bias remains in `last_layer.bias`)

#### `BatchedHebbianLastLayer` (population-level)
- Manages P individuals × S envs (P*S total envs in parallel)
- Stores per-individual weight matrices: `W` shape `(P, 5, 64)`
- Per-individual ABCD+λ rules: A, B, C, D, lam, eta (each shape `(P, 5, 64)`)
- **`hebbian_update(x, y)`** — Groups envs by individual, computes per-individual mean activations, applies per-individual updates
- **`reset_weights()`** — Reset all P weight matrices to checkpoint values

**Key design:**
- Vectorized for efficiency: uses `torch.bmm()` for batched matrix multiplication
- Allows P individuals to evolve independently within a single forward pass
- Critical for parallel evaluation of entire population in one CUDA kernel

**Notes:**
- Both classes store `W_checkpoint` for quick resets
- Both handle optional per-weight eta (scalar or tensor)
- Epsilon handling is minimal — relies on weight clipping for stability

---

### 3. **frozen_actor.py** ✅

**Status:** Complete. Loads, freezes, and wraps the frozen actor for rollout.

**Key functions:**

#### `load_frozen_actor(checkpoint_path, wp1_cfg_path, device)`
- Loads WP1 checkpoint (handles both dict and `model_state_dict` wrapping)
- Instantiates `ActorCriticTanh` via `_build_actor_critic()`
- Freezes all parameters (`requires_grad=False`, `model.eval()`)
- Extracts last linear layer and converts weight from `Parameter` to buffer
- Infers `num_actions` and `hidden_dim` from checkpoint

#### `attach_hebbian(last_layer, hebbian_rules, cfg, device)`
- Creates `HebbianLastLayer` instance for single individual
- Passes per-weight ABCD+λ tensors and global η, w_max from config

#### `HebbianActorWrapper` (single individual forward pass)
- `__init__()` — stores frozen model + hebbian controller
- `reset_episode()` — resets Hebbian weights and LSTM hidden states
- `act(obs)` — forward pass with Hebbian interception:
  1. obs → LSTM → MLP backbone → x (hidden features, 64D)
  2. x @ W^T → y (raw actions, 5D, before tanh)
  3. `hebbian_update(x, y)` modifies W in-place
  4. y → tanh → scale → actions
- `act_simple()` — alternative simpler version using model's distribution

**Key design:**
- Both `act()` and `act_simple()` are defined; `act_simple()` is cleaner but both work
- `@torch.no_grad()` ensures no autograd overhead
- LSTM reset is explicit (`model.memory_a.reset()`)

#### `BatchedHebbianActorWrapper` (population-level forward pass)
- `__init__()` — stores frozen model + batched Hebbian controller
- `reset_episode()` — resets all P individuals' weights and LSTM
- `act(obs)` — forward pass with per-individual Hebbian updates:
  1. obs (P*S, obs_dim) → LSTM → MLP → x (P*S, 64)
  2. Reshape x → (P, S, 64)
  3. Batched matmul: `(P,S,64) @ (P,64,5)^T = (P,S,5)`
  4. Per-individual `hebbian_update()` via `BatchedHebbianLastLayer`
  5. Use model's distribution for action sampling (shared across all P)
  6. actions (P*S, 5)

**Notes:**
- Both wrappers use `torch.no_grad()` for efficiency
- LSTM memory management is critical — reset happens per episode
- Batched wrapper assumes all individuals share the frozen backbone (only last-layer weights differ)

---

### 4. **evaluate.py** ✅

**Status:** Complete. Evaluation pipeline for fitness computation.

**Key functions:**

#### `_build_env(morphology_genome, cfg, wp1_cfg, device, num_envs_override)`
- Converts morphology genome → physical parameters via `Chromosome_Drone.to_physical()`
- Generates URDF via `UrdfMaker`
- Instantiates `WingedDroneEnv` with the URDF
- Inherits env/obs/reward/command configs from WP1, overrides velocity range from config

#### `_rollout_episode(env, actor_wrapper, device, collect_smoothness)`
- Runs a single episode until all envs are done
- Tracks:
  - Time accumulated (`t_acc`)
  - Distance traveled (`dx_acc`)
  - Energy consumed (`E_acc` from `env.power`)
  - Crash flags (collision, wall, angle limit)
  - Action jerk (if `collect_smoothness=True`)
- Returns dict: `{velocities, energies, progresses, crash_flags, action_jerks (if requested)}`

**Key design:**
- Handles NaN environments gracefully
- Computes velocity as `distance / time`
- Crashes are tracked via env's internal flags

#### `evaluate_individual(genome, cfg, wp1_cfg, checkpoint_path, device)`
- Full pipeline: genome → fitness
- Steps:
  1. `split_genome()` → hebbian part, morphology part
  2. `_build_env()` → create environment
  3. `load_frozen_actor()` → frozen actor
  4. `decode_hebbian_genes()` → per-weight rules from genome
  5. `attach_hebbian()` → create `HebbianLastLayer`
  6. Loop over `num_eval_episodes`:
     - `_rollout_episode()` → collect metrics
  7. `compute_fitness()` → aggregate metrics into fitness tuple

**Notes:**
- Handles both single-individual (returns single fitness) and batched (returns list of fitnesses)
- Caches environment and actor creation across episodes to avoid redundant builds
- Energy is summed across timesteps; velocity is aggregated per-env then averaged

---

### 5. **objectives.py** ✅

**Status:** Complete. Extensible fitness objective functions.

**Objectives:**
- `velocity_objective()` — mean forward speed (maximize)
- `energy_objective()` — negative total energy (maximize = lower energy)
- `progress_objective()` — mean distance (maximize)
- `smoothness_objective()` — negative action jerk (maximize = smoother)
- `crash_rate_objective()` — negative crash rate (maximize = fewer crashes)

**Architecture:**
- `OBJECTIVE_REGISTRY` dict maps objective name → function
- `compute_fitness(metrics, cfg)` — loops over active objectives, computes each, returns list
- `default_fitness(cfg)` — returns sentinel values for invalid individuals (e.g., if evaluation crashes)

**Extensibility:**
- Add a new objective: define function, register in `OBJECTIVE_REGISTRY`, toggle in config

**Notes:**
- All objectives are oriented for **maximization** (NSGA-II convention)
- Costs (energy, crash_rate) are negated so maximization is semantically correct
- Defaults are sensible but parameterizable via objectives config

---

### 6. **utils.py** ✅

**Status:** Complete. Reproducibility and utility functions.

**Key functions:**

#### Seed control
- `seed_everything(seed)` — sets random, numpy, torch, cuda seeds for full reproducibility

#### RNG state save/load (for resuming)
- `save_rng_state(path)` — pickle all RNG states
- `load_rng_state(path)` — restore RNG states

#### Genome encoding/decoding
- `decode_hebbian_genes(genome_section, hebb_cfg)` — normalised [0,1] genome → per-weight tensors
  - Lays out genome as: [A_flat | B_flat | C_flat | D_flat | (lam_flat) | (eta_flat)]
  - Rescales each block to its configured range
  - Returns dict with keys: `A, B, C, D, lam, eta` (shapes `(5, 64)`)
- `encode_hebbian_genes(rules, hebb_cfg)` — inverse operation (for saving Pareto solutions)
- `split_genome(genome, cfg)` — splits full genome into (hebbian, morphology) sections

#### Reproducibility artifacts
- `save_git_info(path)` — capture git commit hash + uncommitted diff
- `save_environment_info(path)` — pip freeze output
- `save_pareto_front(run_dir, pareto_individuals, cfg)` — save each Pareto solution's rules + morphology + fitness to YAML

**Notes:**
- Genome encoding is layer-wise normalization: rescale [0,1] genome section to its configured range
- `save_pareto_front()` creates individual folders with `hebbian_rules.yaml`, `morphology.yaml`, `fitness.yaml`

---

### 7. **evolve.py** ✅

**Status:** Complete. NSGA-II main loop.

**Key class: `HebbianCodesignDEAP`**

**Constructor:**
- Initializes DEAP toolbox with:
  - Fitness class (multi-objective, maximisation)
  - Individual class (list of genes in [0,1])
  - Variation operators: SBX crossover + polynomial mutation (toggleable crossover)
  - Selection: NSGA-II tournament DCD

**Attributes:**
- `cfg` — `HebbianEvolutionConfig`
- `run_dir` — timestamped output directory
- `pop` — current DEAP population
- `pareto_front` — current non-dominated individuals

**Methods:**

#### `run(resume_from_gen=None)` — Main NSGA-II loop
- Initialization: random population of size `population_size`
- Loop over generations:
  1. **Evaluate** — for each individual in population, call `evaluate_individual()`
     - Serial or Ray-parallel depending on `GA_PARALLEL` flag
  2. **Selection** — `selNSGA2()` produces offspring pool
  3. **Variation** — crossover + mutation (crossover toggleable)
  4. **Replacement** — μ+λ elitism via NSGA-II
  5. **Logging** — append to CSV, save generation checkpoint (population.pkl, pareto_front.pkl, rng_state.pkl)
- Return final population

**Key features:**
- ✅ Ray parallelism (multi-GPU evaluation)
- ✅ CSV logging (population_history, pareto_history, generation_summary)
- ✅ Generation checkpoints (population.pkl, pareto_front.pkl, rng_state.pkl)
- ✅ Resume from any generation
- ✅ Pareto front extraction + serialization
- ✅ Weight reset to checkpoint at generation start (via `HebbianLastLayer.reset_weights()` called in `evaluate_individual()`)

**Notes:**
- Fitness validity check: invalid individuals get sentinel values from `default_fitness()`
- Crowding distance computed automatically by DEAP
- Hypervolume tracking could be added as an optional post-run analysis

---

### 8. **run.py** ✅

**Status:** Complete. CLI entry point.

**Main flow:**
1. Parse arguments: `--cfg`, `--resume`, `--from-gen`, `-v`
2. Load config from YAML or resume from run directory
3. Apply CLI overrides
4. Validate checkpoint paths and genome dimensions
5. Setup environment: cache directories, seed control
6. Infer last-layer dims from checkpoint
7. Print summary
8. Launch evolution via `HebbianCodesignDEAP.run()`
9. Post-run analysis: call `plotting.analyze_run()`

**Key design:**
- Idempotent cache setup (directories created if missing)
- Robust checkpoint loading (handles both direct state_dict and wrapped formats)
- Graceful plotting failure (non-fatal if plotting encounters errors)

---

### 9. **plotting.py** ⚠️

**Status:** Partially reviewed. Needs full verification.

**Declared plots (from docstring):**
1. Pareto front (2D/3D scatter)
2. Pareto front evolution (generation as color)
3. Hypervolume convergence
4. Per-objective convergence
5. Hebbian parameter distributions (violin)
6. Hebbian parameter heatmap
7. Weight dynamics (timestep within episode)
8. Morphology diversity
9. Ablation comparison (bar chart)
10. Objective correlation (pairwise scatter)

**Helper functions:**
- `_smooth()` — EMA smoothing
- `_load_gen_summary()` — Load generation_summary.csv
- `_load_pareto_history()` — Load pareto_history.csv
- `_load_pop_history()` — Load population_history.csv
- `_get_objective_names()` — Infer from CSV headers

**Notes from code review:**
- Only read first 100 lines; full implementation needs verification
- Should be auto-generated at end of run via `analyze_run()`
- Needs graceful error handling for incomplete runs

**TODO:**
- [ ] Verify all 10 plots are implemented
- [ ] Check matplotlib backend handling (non-interactive rendering)
- [ ] Ensure error handling for missing/incomplete CSVs
- [ ] Add optional ablation comparison (cross-run visualization)

---

### 10. **Preset YAML Configs**

**Location:** `src/WP2/configs/`

**Expected configs:**
- `full_codesing.yaml` — Hebbian ON + morphology ON (2255 genes)
- `hebbian_only.yaml` — Hebbian ON + morphology OFF (2240 genes)
- `morphology_only.yaml` — Hebbian OFF + morphology ON (15 genes)
- `baseline.yaml` — Hebbian OFF + morphology OFF (0 genes)
- `mutation_only.yaml` — Full co-opt with crossover disabled

**TODO:**
- [ ] Verify all configs exist and are correctly formatted
- [ ] Ensure they have correct checkpoint_path and checkpoint_config_path values
- [ ] Spot-check hyperparameters (population_size, eta, etc.)

---

## Known Issues & Improvements

### 1. **Batched Evaluation Integration** ⚠️
- `BatchedHebbianLastLayer` and `BatchedHebbianActorWrapper` are implemented
- **Status of integration in `evolve.py`:** Unclear if batched evaluation is actually used
- **Recommendation:** Verify that `evaluate_individual()` uses batched mode when evaluating the entire population simultaneously
- **Impact:** Affects throughput; batched could be 2-10× faster than serial

### 2. **Plotting Module Completeness** ⚠️
- Only reviewed docstring and first 100 lines
- **TODO:** Full code review to confirm all 10 plots are implemented
- **Risk:** Missing plots could leave blind spots in evolution analysis

### 3. **Error Handling in Evaluation** ⚠️
- `evaluate_individual()` may crash if env/actor building fails
- **Recommendation:** Wrap with try-except, return `default_fitness()` on error
- **Current status:** Not verified

### 4. **Morphology-Blind Actor Transfer** ⚠️
- Actor is morphology-blind (trained on diverse morphologies in WP1)
- **Risk:** May not transfer well to morphologies far from training distribution
- **Mitigation:** Hebbian plasticity should help; ablation (morphology-only) will reveal the gap

### 5. **Fitness Caching** ⚠️
- Plan mentions "fitness_db.csv" for caching evaluations
- **Status:** Unclear if implemented in `evolve.py`
- **Recommendation:** Add deduplication to avoid re-evaluating identical genomes

### 6. **Resume Stability** ⚠️
- Resume from any generation requires loaded RNG states and population state
- **Status:** Implemented (population.pkl, rng_state.pkl saved per generation)
- **Recommendation:** Test resume from middle of run to verify exact reproducibility

---

## Testing Checklist

### Unit Tests (minimal)
- [ ] `decode_hebbian_genes()` → tensor shapes correct
- [ ] `split_genome()` → correct splits for all config modes
- [ ] `compute_fitness()` → fitness tuple length matches active objectives

### Integration Tests
- [ ] Small run (pop=10, gen=2) completes without error
- [ ] Run folder structure is correct (reproducibility/, generations/, results/, pareto_solutions/, plots/)
- [ ] CSV files have correct headers and data
- [ ] Generation checkpoints (population.pkl, rng_state.pkl) are valid

### Reproducibility Tests
- [ ] Resume from gen 1 produces identical results to original run
- [ ] Pareto solutions can be deserialized from YAML
- [ ] `git_info.txt` and `environment.txt` capture state correctly

### Visualization Tests
- [ ] All 10 plots generate without error
- [ ] Plots adapt to number of active objectives (2, 3, 5 objectives)
- [ ] Plots are readable and clearly labeled

---

## Next Steps & Recommendations

### Immediate (before first full run)
1. **Verify config files** — ensure all 5 presets exist and have valid checkpoint paths
2. **Test plotting** — run `plotting.py` review to confirm all 10 plots are working
3. **Small end-to-end test** — pop=10, gen=2, check output structure
4. **Spot-check batched evaluation** — confirm it's actually used in `evolve.py`

### Short-term (after first successful run)
1. **Add fitness caching** — dedup identical genomes to avoid redundant evaluation
2. **Improve error handling** — wrap env/actor building with try-except
3. **Test resume** — verify mid-run resume produces deterministic continuation
4. **Profile throughput** — measure wall-clock time per generation, identify bottlenecks

### Medium-term (optimization)
1. **Hierarchical genome encoding** — current 1600D genome is large; consider structured encoding
2. **Adaptive mutation rates** — adjust per-weight mutation rate based on convergence
3. **Warm-start from WP1** — initialize Hebbian rules from WP1 plasticity (if available)

### Long-term (extensions)
1. **Ablation comparison plots** — cross-run visualization (full vs hebbian-only vs morphology-only vs baseline)
2. **Hypervolume indicator** — track Pareto front quality over generations
3. **Per-weight learning curves** — analyze how individual synaptic rules evolve
4. **Deployment script** — select a Pareto solution, deploy to real drone simulator

---

## File Manifest

```
src/WP2/
├── plan.md                            # This file — architecture + status
├── __init__.py                        # empty
├── config.py                          # HebbianEvolutionConfig (complete)
├── hebbian.py                         # HebbianLastLayer, BatchedHebbianLastLayer (complete)
├── frozen_actor.py                    # load_frozen_actor, HebbianActorWrapper, BatchedHebbianActorWrapper (complete)
├── evaluate.py                        # evaluate_individual, _rollout_episode, _build_env (complete)
├── objectives.py                      # fitness functions + registry (complete)
├── evolve.py                          # HebbianCodesignDEAP NSGA-II loop (complete)
├── run.py                             # CLI entry point (complete)
├── plotting.py                        # visualization (partial — needs review)
├── utils.py                           # utilities + reproducibility (complete)
├── configs/                           # YAML presets (TODO: verify)
│   ├── full_codesing.yaml
│   ├── hebbian_only.yaml
│   ├── morphology_only.yaml
│   ├── baseline.yaml
│   └── mutation_only.yaml
└── slurm_jobs/                        # cluster scripts (optional)
    ├── run.slurm
    └── run_sweep.sh
```

---

## Configuration Examples

### Full Co-Optimization (main experiment)
```bash
python -m WP2.run --cfg src/WP2/configs/full_codesing.yaml
```
- Hebbian enabled, morphology enabled
- Genome: 2255D (2240 Hebbian + 15 morphology)
- Objectives: velocity, energy, progress

### Hebbian-Only (ablation)
```bash
python -m WP2.run --cfg src/WP2/configs/hebbian_only.yaml
```
- Hebbian enabled, morphology fixed
- Genome: 1600D (Hebbian only)
- Tests whether Hebbian plasticity helps on fixed morphology

### Morphology-Only (ablation)
```bash
python -m WP2.run --cfg src/WP2/configs/morphology_only.yaml
```
- Hebbian disabled, morphology enabled
- Genome: 15D (morphology only)
- Tests whether morphology evolution is effective

### Resume from Generation 15
```bash
python -m WP2.run --resume logs/runs_hebbian/2026-03-17_12-34-56_hebbian_codesing --from-gen 15
```

### CLI Overrides
```bash
python -m WP2.run --cfg src/WP2/configs/full_codesing.yaml \
  --cfg.evolution.population_size 80 \
  --cfg.evolution.num_generations 50 \
  --cfg.hebbian.eta 0.02 \
  --cfg.evaluation.num_eval_episodes 3
```

---

## Critical Design Decisions (Rationale)

| Decision | Choice | Why |
|----------|--------|-----|
| Where Hebbian acts | Last layer only (Linear 64→7) | Maximizes expressiveness while keeping overhead low; matches biological neuromuscular plasticity |
| Bias never modified | Frozen | Simplifies implementation; acts as baseline offset (learnable via A/B/C coefficients) |
| Critic dropped | Yes | No value function needed for pure rollout; saves memory, reduces complexity |
| Per-weight granularity | Yes (448 separate rule sets) | Each synapse can learn a different adaptation strategy; essential for heterogeneous plasticity |
| Weight reset per episode | Yes | Each episode tests plasticity from scratch; prevents biased fitness estimates from carry-over |
| Crossover toggleable | Yes | 2240D genome may be disruptive; mutation-only is a valid strategy for high-D problems |
| Stochastic by default | Yes | Exploration helps discover diverse behaviors; toggleable for deterministic comparison |
| Pareto frontier output | Yes (not single best) | NSGA-II naturally produces frontier; user can select trade-off post-hoc |
| Genome encoding | [0,1] normalized | Matches DEAP convention; enables SBX crossover and polynomial mutation |

---

## References & Links

- **WP1 foundation training:** `python -m WP1.train --cfg src/WP1/configs/foundation.yaml`
- **Context file:** `.claude/CONTEXT.md` (comprehensive codebase overview)
- **Core environment:** `src/winged_drone_train/env.py`
- **Morphology encoding:** `src/morph_evolution/chromosome_drone.py`
- **DEAP documentation:** https://deap.readthedocs.io (NSGA-II, SBX, polynomial mutation)

---

**Last updated:** 2026-03-17
**Reviewed by:** Code review of all WP2 modules
**Status:** Implementation complete; plotting + config validation pending
