# WP2 — Hebbian Plasticity + Evolutionary Co-Optimization

> **Objective:** Joint optimization of morphology and synaptic adaptation mechanisms
> via NSGA-II, *without* retraining the base controller (no backprop).
> The output is a **Pareto frontier** of individuals trading off multiple objectives.

---

## Constraints & Ground Rules

- [ ] Do **not** modify code in other folders — only expand `src/WP2/`
- [ ] The pretrained controller (from a WP1 run) is **frozen** — no gradient updates, no backprop
- [ ] The **critic is dropped** entirely — only the actor is used in WP2 (forward pass only, no value estimation)
- [ ] Reuse WP1 infrastructure (`RunConfig`, `WingedDroneEnv`, `Gen_Env`, CSV logging, plotting) wherever possible
- [ ] Also reuse existing codebase utilities (`Chromosome_Drone`, NSGA-II patterns from `morph_evolution/`)
- [ ] All runs must be **fully reproducible** — every random seed (numpy, torch, DEAP, genesis) is controlled and saved
- [ ] Code must be modular: evolutionary loop, Hebbian module, evaluation, analysis are cleanly separated

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         NSGA-II Outer Loop                               │
│  Evolves: per-weight ABCD+λ rules (1600D) + morphology [0,1]^15         │
│  Produces: Pareto frontier of non-dominated individuals                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                   Fitness Evaluation (per individual)              │  │
│  │                                                                    │  │
│  │  1. Reset last-layer weights to checkpoint values                  │  │
│  │  2. Load frozen actor (no critic) + inject Hebbian rules           │  │
│  │  3. Build morphology URDF from genome (Chromosome_Drone)           │  │
│  │  4. Rollout N episodes in Genesis (forward pass only)              │  │
│  │     - At each step: forward pass → Hebbian update on last layer    │  │
│  │     - Weights reset to checkpoint values between episodes          │  │
│  │  5. Compute multi-objective fitness                                │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Insight

The pretrained LSTM actor has a final linear layer `Linear(64→5)` that maps
hidden features to actions (1 throttle + 4 servos). The **Hebbian rules
modify the weights of this existing layer in-place** at each simulation
timestep. No new layer is added — the ABCD rule directly modulates the
320 synaptic weights (64×5) of the pretrained output layer.

NSGA-II evolves per-weight plasticity rules (A, B, C, D, λ for each of
the 320 weights). The result is not a single best individual but a
**Pareto frontier** of solutions that represent optimal trade-offs between
the competing objectives.

---

## 2. Hebbian Plasticity Module

### 2.1 ABCD Hebbian Rule

Each weight `w_ij` in the actor's last linear layer is updated at every
simulation step `t` according to the **generalized ABCD Hebbian rule**:

```
Δw_ij(t) = η · [ A_ij · x_i(t) · y_j(t)      ← classical Hebb
               + B_ij · x_i(t)                 ← presynaptic
               + C_ij · y_j(t)                 ← postsynaptic
               + D_ij ]                         ← bias drift
```

Where:
- `x_i(t)` = presynaptic activation (input to last layer, from frozen MLP)
- `y_j(t)` = postsynaptic activation (output of last layer, before tanh scaling)
- `η` = global learning rate (**fixed** tunable hyperparameter, toggleable per-weight variant)
- `A_ij, B_ij, C_ij, D_ij` = per-weight rule coefficients (**evolved**)

Weight update with decay:
```
w_ij(t+1) = (1 - λ_ij) · w_ij(t) + Δw_ij(t)
w_ij(t+1) = clip( w_ij(t+1),  -w_max, +w_max )
```

Where:
- `λ_ij` = per-weight decay rate (**evolved**)
- `w_max` = weight clipping bound (**fixed** tunable hyperparameter, not evolved)

### 2.2 Evolved vs Fixed Parameters

**Evolved per-weight** (via NSGA-II):

| Parameter | Per-weight count | Total (64×5=320 weights) | Range | Description |
|-----------|-----------------|--------------------------|-------|-------------|
| `A_ij` | 320 | 320 | [-1, 1] | Hebbian correlation coefficient |
| `B_ij` | 320 | 320 | [-1, 1] | Presynaptic coefficient |
| `C_ij` | 320 | 320 | [-1, 1] | Postsynaptic coefficient |
| `D_ij` | 320 | 320 | [-1, 1] | Bias/drift coefficient |
| `λ_ij` | 320 | 320 | [0, 0.1] | Weight decay rate |

**Total evolved Hebbian genes: 1600** (stored in normalized [0,1] space)

**Optionally evolved per-weight** (toggled by `evolve_eta: bool`):

| Parameter | Per-weight count | Total | Range | Description |
|-----------|-----------------|-------|-------|-------------|
| `η_ij` | 320 | 320 | [0, 0.1] | Per-weight learning rate (when `evolve_eta=True`) |

When `evolve_eta=False` (default), a single global `η` is used for all weights.
When `evolve_eta=True`, the Hebbian genome grows by 320 genes (total: 1920).

**Fixed tunable hyperparameters** (set in config, never evolved):

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `η` | 0.01 | (0, 0.1] | Global learning rate (used when `evolve_eta=False`) |
| `w_max` | 3.0 | (0, ∞) | Symmetric weight clipping bound |

**Explicitly excluded:** Bias terms of the last layer are **never** modified
by Hebbian rules — only the weight matrix `W` (64×5) is plastic.

### 2.3 Implementation: In-Place Last-Layer Modification

The Hebbian plasticity operates **directly on the existing last linear layer**
of the actor network. No new layer is created — the pretrained `Linear(64→5)`
weight matrix is modified in-place at each forward pass. **Biases are never
touched.**

```python
class HebbianLastLayer:
    """
    Wraps the actor's existing last Linear layer with Hebbian plasticity.
    Modifies ONLY .weight in-place each step. Bias is never modified.

    This is NOT an nn.Module — it is a controller that mutates the existing
    layer's weight buffer during rollout.
    """

    def __init__(self, linear_layer: nn.Linear, hebbian_genes: Tensor,
                 eta: float | Tensor, w_max: float):
        self.layer = linear_layer
        self.w_max = w_max

        # eta: either scalar (global) or (5, 64) tensor (per-weight)
        self.eta = eta

        # Store original checkpoint weights for reset (bias untouched)
        self.W_checkpoint = linear_layer.weight.data.clone()

        # Decode per-weight ABCD + λ from evolved genome
        # hebbian_genes shape: (5, out_features, in_features) = (5, 5, 64)
        self.A = hebbian_genes[0]   # (5, 64)
        self.B = hebbian_genes[1]
        self.C = hebbian_genes[2]
        self.D = hebbian_genes[3]
        self.lam = hebbian_genes[4]

    def reset_weights(self):
        """Reset layer weights to checkpoint values.
        Called at: episode start AND generation start.
        Bias is never modified or reset."""
        self.layer.weight.data.copy_(self.W_checkpoint)

    def hebbian_update(self, x: Tensor, y: Tensor):
        """
        Apply ABCD rule to modify last layer weights in-place.
        Called AFTER each forward pass through the layer.

        Args:
            x: presynaptic activations (batch, 64) — input to last layer
            y: postsynaptic activations (batch, 5)  — output of last layer
        """
        # For batched envs, average activations across batch
        x_mean = x.mean(dim=0)  # (64,)
        y_mean = y.mean(dim=0)  # (5,)

        # ABCD update: dW shape (5, 64) matches layer.weight
        dW = self.eta * (
            self.A * torch.outer(y_mean, x_mean) +   # (5, 64)
            self.B * x_mean.unsqueeze(0) +             # broadcast (5, 64)
            self.C * y_mean.unsqueeze(1) +             # broadcast (5, 64)
            self.D                                      # (5, 64)
        )

        # Decay + update (bias untouched)
        self.layer.weight.data.mul_(1.0 - self.lam).add_(dW)
        self.layer.weight.data.clamp_(-self.w_max, self.w_max)
```

### 2.4 Integration with Frozen Actor (No Critic)

The frozen actor's forward pass is intercepted to inject Hebbian updates:

```
Original actor forward pass (WP1):
    obs → LSTM(128) → MLP[64,64] → Linear(64→5) → tanh → scale → action
                                    ↑ this layer
                                    ↑ weights modified in-place by ABCD rule

WP2 forward pass (critic dropped):
    obs → LSTM(128) → MLP[64,64] → x → Linear(64→5) → y → tanh → scale → action
                                    ↓         ↑ W modified         ↓
                                    └── hebbian_update(x, y) ──────┘
```

Steps:
1. Load WP1 checkpoint → extract **actor only** (drop critic entirely)
2. Freeze all actor parameters (`requires_grad=False`, `torch.no_grad()`)
3. Convert last layer weights from `nn.Parameter` to buffer (in-place modifiable)
4. Wrap with `HebbianLastLayer` using evolved per-weight rules
5. At each generation start and each episode start → `reset_weights()` to checkpoint values

---

## 3. Evolutionary Optimization (NSGA-II)

### 3.1 Genome Structure

The full individual genome is a concatenation of two optional parts:

```
genome = [hebbian_genes (1600–1920D)] ⊕ [morphology_genes (15D)]
          ↑ if hebbian_enabled             ↑ if evolve_morphology
          ↑ 1600 base + 320 if evolve_eta
```

Hebbian genome dimension depends on `evolve_eta`:
- `evolve_eta=False` (default): **1600** genes (A,B,C,D,λ × 320 weights)
- `evolve_eta=True`: **1920** genes (A,B,C,D,λ,η × 320 weights)

| Mode | Hebbian | Morphology | evolve_eta | Genome dim | Purpose |
|------|---------|------------|------------|------------|---------|
| **Full co-optimization** | ON | ON | OFF | 1615 | Main experiment |
| **Full + per-weight η** | ON | ON | ON | 1935 | Extended experiment |
| **Hebbian-only** | ON | OFF (fixed) | OFF | 1600 | Ablation: plasticity on fixed body |
| **Morphology-only** | OFF | ON | — | 15 | Ablation: body shape without plasticity |
| **Baseline** | OFF | OFF | — | 0 | Control: frozen policy on fixed morphology |

### 3.2 Fitness Objectives (Multi-Objective, Extensible)

NSGA-II evolves a **Pareto frontier** — a set of non-dominated solutions
representing optimal trade-offs between objectives. The objectives are
configurable and extensible.

**Default objectives** (reused from `morph_evolution/`):

| # | Objective | Direction | Description |
|---|-----------|-----------|-------------|
| 1 | Mean velocity | **maximize** | Average forward speed across evaluation episodes |
| 2 | Energy efficiency | **maximize** | Negative total energy consumption (-E_tot) |
| 3 | Progress | **maximize** | Distance covered through the forest corridor |

**Additional objectives** (can be enabled via config):

| # | Objective | Direction | Description |
|---|-----------|-----------|-------------|
| 4 | Smoothness | **maximize** | Negative action jerk (penalizes erratic control) |
| 5 | Crash rate | **minimize** | Fraction of episodes ending in crash |
| 6 | Adaptability | **maximize** | Performance variance across morphologies (lower = more robust) |
| 7 | Custom | configurable | User-defined fitness function via callback |

Fitness weights are specified in config as a tuple matching the number of
active objectives. Adding/removing objectives only requires updating the
weights tuple and the fitness computation function.

### 3.3 Evaluation Pipeline

For each individual in the population:

```
1. DECODE genome
   ├── Extract per-weight hebbian rules (A,B,C,D,λ)_ij  (if hebbian_enabled)
   └── Extract morphology_genome (if evolve_morphology, else use fixed_morphology)

2. BUILD environment
   ├── Chromosome_Drone.to_physical(morphology_genome) → URDF
   └── Instantiate WingedDroneEnv with the URDF

3. LOAD frozen actor (no critic)
   ├── Load ActorCriticTanh from WP1 checkpoint, extract actor only
   ├── Freeze all parameters
   ├── Reset last-layer weights to checkpoint values
   └── Attach HebbianLastLayer with decoded per-weight rules

4. ROLLOUT (no gradient, stochastic by default)
   ├── For each episode:
   │   ├── reset_weights() → restore checkpoint values
   │   ├── Reset environment
   │   ├── For each timestep:
   │   │   ├── obs → frozen_backbone → x (hidden features, 64D)
   │   │   ├── x → last_layer → y (raw actions, 5D)
   │   │   ├── hebbian_update(x, y)  ← modifies last layer weights
   │   │   ├── action = tanh(y) → scale
   │   │   └── env.step(action)
   │   └── Collect: velocity, energy, progress, (smoothness, crashes...)
   └── Aggregate across episodes

5. RETURN fitness tuple matching active objectives
```

### 3.4 NSGA-II Operators

| Operator | Method | Parameters | Notes |
|----------|--------|------------|-------|
| **Crossover** | Simulated Binary (SBX) | `eta_c=20`, `p_cx=0.9`, bounded [0,1] | **Toggleable** — can be disabled for mutation-only evolution |
| **Mutation** | Polynomial bounded | `eta_m=20`, `p_mut=1.0`, `indpb=1/n_genes` | Always active |
| **Selection** | NSGA-II (`selNSGA2`) | Tournament DCD for mating | Produces Pareto frontier |
| **Replacement** | μ+λ via NSGA-II | Elitist — best from parents+offspring | Preserves frontier quality |

When crossover is **disabled** (`enable_crossover: false`), the variation
loop only applies mutation. This is useful for very high-dimensional genomes
(1600D) where crossover may be disruptive.

### 3.5 Weight Reset Policy

- **At each generation start:** all last-layer weights reset to checkpoint values
  (no carrying over final weights from previous generation's rollouts)
- **At each episode start** (within an individual's evaluation): weights reset
  to checkpoint values — each episode starts from the same baseline, with
  Hebbian rules shaping the weights from scratch during that episode

### 3.6 Parallelism

- **Serial mode** (single GPU): evaluate individuals sequentially
- **Ray mode** (multi-GPU): distribute evaluations across GPUs (same pattern as `evolution_nsga.py`)
- Controlled by `GA_PARALLEL` environment variable

---

## 4. Configuration System

### 4.1 `HebbianEvolutionConfig` Dataclass

```python
@dataclass
class HebbianConfig:
    """Hebbian plasticity settings."""
    enabled: bool = True                    # toggle Hebbian rules on/off
    eta: float = 0.01                       # global learning rate (used when evolve_eta=False)
    evolve_eta: bool = False                # if True, η becomes per-weight evolved (+320 genes)
    w_max: float = 3.0                      # FIXED weight clipping bound (never evolved)
    A_range: Tuple[float, float] = (-1.0, 1.0)   # evolved per-weight
    B_range: Tuple[float, float] = (-1.0, 1.0)   # evolved per-weight
    C_range: Tuple[float, float] = (-1.0, 1.0)   # evolved per-weight
    D_range: Tuple[float, float] = (-1.0, 1.0)   # evolved per-weight
    decay_range: Tuple[float, float] = (0.0, 0.1) # evolved per-weight (λ)
    eta_range: Tuple[float, float] = (0.0, 0.1)   # per-weight η range (when evolve_eta=True)

@dataclass
class EvolutionConfig:
    """NSGA-II hyperparameters."""
    population_size: int = 40
    num_generations: int = 30
    enable_crossover: bool = True           # toggle crossover on/off
    crossover_probability: float = 0.9      # ignored if enable_crossover=False
    mutation_probability: float = 1.0
    eta_c: float = 20.0                     # SBX spread
    eta_m: float = 20.0                     # polynomial mutation spread
    weights: Tuple[float, ...] = (1.0, 1.0, 1.0)  # per-objective weights (extensible)

@dataclass
class EvaluationConfig:
    """Rollout settings for fitness evaluation."""
    num_eval_episodes: int = 5
    num_eval_envs: int = 8192
    vmin: float = 6.0
    vmax: float = 30.0
    stochastic: bool = True                 # stochastic policy (sample from distribution)

@dataclass
class ObjectivesConfig:
    """Toggle individual fitness objectives."""
    velocity: bool = True
    energy: bool = True
    progress: bool = True
    smoothness: bool = False
    crash_rate: bool = False
    # Weights auto-derived from active objectives + evolution.weights

@dataclass
class MorphologyConfig:
    """Morphology evolution settings."""
    evolve: bool = True                     # toggle morphology evolution on/off
    fixed_genome: Optional[List[float]] = None  # used when evolve=False

@dataclass
class HebbianEvolutionConfig:
    """Top-level config for WP2 runs."""
    exp_name: str = "hebbian_codesing"
    checkpoint_path: str = ""               # path to frozen WP1 actor checkpoint
    checkpoint_config_path: str = ""        # path to the WP1 run's config.yaml

    hebbian: HebbianConfig = field(default_factory=HebbianConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    objectives: ObjectivesConfig = field(default_factory=ObjectivesConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)

    seed: int = 42
    device: str = "cuda:0"
    base_dir: str = "logs/runs_hebbian"

    # YAML I/O (same pattern as WP1 RunConfig)
    def to_yaml(self, path) -> str: ...
    @classmethod
    def from_yaml(cls, path) -> "HebbianEvolutionConfig": ...
    def apply_cli_overrides(self, argv) -> None: ...
```

### 4.2 Preset YAML Configs

```
src/WP2/configs/
├── full_codesing.yaml          # Hebbian ON + morphology ON   (main experiment)
├── hebbian_only.yaml        # Hebbian ON + morphology OFF  (ablation)
├── morphology_only.yaml     # Hebbian OFF + morphology ON  (ablation)
├── baseline.yaml            # Hebbian OFF + morphology OFF (control)
└── mutation_only.yaml       # Full co-opt with crossover disabled
```

---

## 5. Reproducibility & Serialization

### 5.1 Seed Control

At the start of every run:
```python
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
# DEAP uses random module internally — covered by random.seed()
```

The seed is saved in the frozen config YAML snapshot inside the run folder.

### 5.2 Run Folder Structure

All runs are stored under `logs/runs_hebbian/` (parallel to WP1's `logs/runs/`).
Each run folder is **fully self-contained** — it includes every artifact needed
to reproduce the run from scratch.

```
logs/runs_hebbian/<YYYY-MM-DD_HH-MM-SS>_{exp_name}/
│
│── reproducibility/
│   ├── config.yaml                    # frozen HebbianEvolutionConfig snapshot
│   ├── wp1_config.yaml                # copy of the WP1 config that produced the checkpoint
│   ├── wp1_actor.pt                   # copy of the frozen WP1 actor weights (no critic)
│   ├── git_info.txt                   # git commit hash + diff (if any uncommitted changes)
│   └── environment.txt                # pip freeze / conda env export
│
├── generations/
│   ├── gen_000/
│   │   ├── population.pkl             # full DEAP population (genomes + fitness)
│   │   ├── pareto_front.pkl           # non-dominated individuals this generation
│   │   └── rng_state.pkl              # random/numpy/torch RNG states for exact resume
│   ├── gen_001/
│   │   └── ...
│   └── gen_N/
│       └── ...
│
├── results/
│   ├── population_history.csv         # all individuals across all generations
│   ├── pareto_history.csv             # Pareto front evolution across generations
│   ├── generation_summary.csv         # per-generation aggregated stats
│   └── fitness_db.csv                 # cached fitness evaluations
│
├── pareto_solutions/
│   ├── individual_000/
│   │   ├── hebbian_rules.yaml         # per-weight A,B,C,D,λ for this solution
│   │   ├── morphology.yaml            # genome (normalized + physical) + NACA code
│   │   └── fitness.yaml               # all objective values
│   ├── individual_001/
│   │   └── ...
│   └── summary.csv                    # all Pareto-optimal solutions in one table
│
└── plots/
    ├── pareto_front.png
    ├── fitness_convergence.png
    ├── hebbian_param_evolution.png
    ├── morphology_diversity.png
    └── ...
```

### 5.3 Reproducibility Checklist

Every run folder must contain all data needed to **exactly reproduce** the run.
The run manager validates this at startup and logs warnings for missing items.

| Artifact | Location | Purpose |
|----------|----------|---------|
| WP2 config (frozen) | `reproducibility/config.yaml` | All hyperparameters: Hebbian, evolution, evaluation, objectives, morphology, seeds |
| WP1 config (frozen) | `reproducibility/wp1_config.yaml` | Architecture of the frozen actor (hidden dims, LSTM size, action scaling) |
| WP1 actor weights | `reproducibility/wp1_actor.pt` | Exact checkpoint weights used — the frozen baseline for all Hebbian modifications |
| Git commit + diff | `reproducibility/git_info.txt` | Code version; includes uncommitted diff if working tree is dirty |
| Python environment | `reproducibility/environment.txt` | `pip freeze` output — exact package versions (torch, deap, genesis, numpy, etc.) |
| RNG states per generation | `generations/gen_NNN/rng_state.pkl` | `random`, `numpy`, and `torch` RNG states — allows resuming from any generation |
| Full population per generation | `generations/gen_NNN/population.pkl` | Complete DEAP population with genomes and fitness values |
| Fitness cache | `results/fitness_db.csv` | All evaluated (genome → fitness) pairs across the entire run |
| Fixed morphology genome | `reproducibility/config.yaml` | When `evolve_morphology=False`, the fixed genome is stored in config |

**Saving git info at run start:**
```python
import subprocess

def save_git_info(path):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    diff = subprocess.check_output(["git", "diff"]).decode()
    with open(path, "w") as f:
        f.write(f"commit: {commit}\n")
        if diff:
            f.write(f"\n--- uncommitted changes ---\n{diff}")
```

### 5.4 Save/Load of Pareto Solutions

```python
# Save entire Pareto front
save_pareto_front(run_dir, pareto_individuals)
# → creates pareto_solutions/individual_NNN/ for each non-dominated solution

# Load a specific Pareto solution for deployment
hebbian_rules = load_hebbian_rules("pareto_solutions/individual_003/hebbian_rules.yaml")
morphology = load_morphology("pareto_solutions/individual_003/morphology.yaml")

# Deploy: frozen actor + Hebbian rules on a specific morphology
actor = load_frozen_actor("reproducibility/wp1_actor.pt")
hebbian = HebbianLastLayer(actor.last_layer, hebbian_rules, eta=cfg.eta, w_max=cfg.w_max)
```

### 5.5 Resuming a Run

A run can be resumed from any generation checkpoint:
```python
# Resume from generation 15
python -m WP2.run --resume logs/runs_hebbian/2026-03-12_14-30-00_hebbian_codesing --from-gen 15
```
This loads `generations/gen_015/population.pkl` and `rng_state.pkl`, restoring
the exact evolutionary state to continue from where it left off.

---

## 6. Analysis & Plotting

### 6.1 Required Plots

| # | Plot | X-axis | Y-axis | Purpose |
|---|------|--------|--------|---------|
| 1 | **Pareto front** (2D/3D scatter) | objective 1 | objective 2 (+ obj 3 as color) | Visualizes the trade-off frontier |
| 2 | **Pareto front evolution** | objectives | generation as color/animation | Shows frontier improvement over generations |
| 3 | **Hypervolume convergence** | Generation | Hypervolume indicator | Tracks overall Pareto front quality |
| 4 | **Per-objective convergence** | Generation | Mean / best / worst per objective | Shows which objectives improve fastest |
| 5 | **Hebbian parameter distributions** | Generation | A, B, C, D, λ values (violin) | Tracks convergence of rule parameters |
| 6 | **Hebbian parameter heatmap** | Weight index (i,j) | ABCD value | Spatial structure of evolved rules for best solutions |
| 7 | **Weight dynamics** | Timestep within episode | W_ij values | How Hebbian updates reshape weights during rollout |
| 8 | **Morphology diversity** | Generation | Genome std / pairwise distance | Population diversity tracking |
| 9 | **Ablation comparison** (bar chart) | Condition | Fitness per objective | Full / hebbian-only / morphology-only / baseline |
| 10 | **Objective correlation** | Objective i | Objective j | Pairwise scatter — reveals conflict/harmony between objectives |

### 6.2 Implementation

Follow WP1's `plotting.py` pattern:
- Each plot is a standalone function accepting a results directory path
- An `analyze_run(run_dir)` function generates all plots at once
- Plots are saved to `{run_dir}/plots/` as PNG + PDF
- Pareto front plots auto-adapt to the number of active objectives (2D scatter, 3D scatter, or pairwise matrix)

---

## 7. File Structure

```
src/WP2/
├── plan.md                            # this file
├── config.py                          # HebbianEvolutionConfig dataclass + YAML I/O
├── hebbian.py                         # HebbianLastLayer — ABCD rule on existing layer
├── frozen_actor.py                    # load checkpoint, drop critic, freeze, attach hebbian
├── evaluate.py                        # rollout frozen+hebbian actor, compute fitness
├── objectives.py                      # fitness objective functions (extensible)
├── evolve.py                          # NSGA-II loop (HebbianCodesignDEAP class)
├── run.py                             # main entry point
├── plotting.py                        # all analysis/visualization functions
├── utils.py                           # seed control, serialization, genome encoding
├── configs/
│   ├── full_codesing.yaml
│   ├── hebbian_only.yaml
│   ├── morphology_only.yaml
│   ├── baseline.yaml
│   └── mutation_only.yaml
└── slurm_jobs/                        # cluster submission scripts (if needed)
```

---

## 8. Implementation Order

### Phase 1 — Core Infrastructure
1. **`config.py`** — Define `HebbianEvolutionConfig` with YAML I/O and CLI overrides
2. **`utils.py`** — Seed control, genome encoding/decoding for per-weight Hebbian params
3. **`hebbian.py`** — Implement `HebbianLastLayer` with per-weight ABCD+λ rule

### Phase 2 — Policy Integration & Evaluation
4. **`frozen_actor.py`** — Load WP1 checkpoint, drop critic, freeze weights, attach Hebbian
5. **`objectives.py`** — Modular fitness functions (velocity, energy, progress, + extensible)
6. **`evaluate.py`** — Rollout loop: reset weights per episode, collect multi-objective fitness

### Phase 3 — Evolutionary Loop
7. **`evolve.py`** — `HebbianCodesignDEAP` class:
   - Genome = per-weight Hebbian rules ⊕ morphology (conditioned on config toggles)
   - DEAP toolbox with toggleable SBX + polynomial mutation
   - Pareto front extraction, population I/O, generation logging
   - Weight reset to checkpoint values at each generation start
8. **`run.py`** — CLI entry point: parse config, seed everything, launch evolution

### Phase 4 — Analysis & Configs
9. **`plotting.py`** — All visualization functions (Pareto fronts, convergence, heatmaps)
10. **`configs/`** — YAML presets for all experimental conditions
11. **Validation** — End-to-end test: small population, few generations, verify reproducibility

---

## 9. Critical Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Where Hebbian acts | Modify existing last layer weights in-place | Not a new layer — directly changes the pretrained `Linear(64→5)` weight matrix each timestep |
| Bias terms | **Never modified** | Only the weight matrix W is plastic; bias vector is frozen and untouched |
| Critic | Dropped entirely | No value estimation needed — pure forward rollout for fitness |
| ABCD granularity | **Per-weight** (each of 320 weights has its own A,B,C,D,λ) | Maximizes expressiveness — each synapse can learn a different adaptation strategy |
| η (learning rate) | Fixed global by default; **toggleable** per-weight evolution | `evolve_eta=False` (default) uses a single η; `evolve_eta=True` adds 320 genes for per-weight η |
| w_max | Fixed tunable config param (never evolved) | Global safety bound, not a per-synapse property |
| Weight reset | Checkpoint values at every generation AND episode start | Each episode tests Hebbian adaptation from scratch; no leakage across episodes/generations |
| Crossover | **Toggleable** (default ON) | With 1600D genome, crossover can be disruptive — mutation-only is a valid strategy |
| Policy stochasticity | Default **stochastic** (sample from distribution) | Adds environment exploration; toggleable for deterministic comparison |
| Fitness structure | Pareto frontier (not single best) | NSGA-II naturally produces a frontier — we expose all non-dominated solutions |
| Objectives | Extensible via config | Start with 3 defaults; additional objectives can be toggled on without code changes |
| Population size | Tunable via config (default 40) | High-dimensional genomes may require larger populations; user adjusts per experiment |
| Morphology encoding | Reuse `Chromosome_Drone` | Proven [0,1]^15 encoding with NACA snapping already exists |

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 1600–1920D Hebbian genome is large for NSGA-II | Slow convergence, poor exploration | Mutation-only mode; increase `population_size` in config; consider hierarchical/structured encoding as future extension |
| Hebbian weights diverge during rollout | Actions become garbage → meaningless fitness | Fixed `w_max` clipping + evolved decay `λ_ij` bounds weight growth |
| Stochastic fitness is noisy | Unreliable selection signal | Average over `num_eval_episodes`; increase episodes for noisy regimes |
| Fitness evaluation is slow (full rollout per individual per generation) | Long wall-clock time | Ray parallelism across GPUs; fitness caching in CSV database |
| Morphology changes invalidate frozen actor | Frozen weights may not transfer to new body shapes | Actor was trained on diverse morphologies (WP1 catalog); Hebbian adaptation may also help bridge the gap |
| Per-weight rules overfit to specific morphology | Poor generalization | Test with morphology-only and hebbian-only ablations to isolate effects |

---

## 11. Resolved Design Questions

| # | Question | Resolution |
|---|----------|------------|
| 1 | Bias terms | **Never modified** — only the weight matrix W (64×5) is plastic |
| 2 | Per-weight η | **Toggleable** via `evolve_eta` flag (default `False` = single global η) |
| 3 | Cross-episode weight carry-over | **Not implemented** — weights reset at every episode start |
| 4 | Additional objectives | Start with the default 3; extensible architecture is ready for future additions |
| 5 | Population sizing | **Tunable** via `evolution.population_size` — user sets per experiment |
