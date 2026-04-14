# WP2 CMA-ES Evolution

This module provides CMA-ES-based evolution of Hebbian plasticity rules for WP2, replacing the multi-objective NSGA-II approach with a simpler, single-objective optimization using the same fitness function as WP1 (mean forward velocity).

## Quick Start

### 1. Basic CLI Usage

After training WP1, run the evolution:

```bash
python -m WP2.run_cmaes \
    --wp1-checkpoint logs/runs/<timestamp>_<exp>/model.pt \
    --wp1-cfg src/WP1/configs/foundation.yaml \
    --pop-size 64 \
    --num-generations 100 \
    --num-eval-envs 256
```

Fitness is automatically computed as the **weighted combination of WP1 reward components** (progress, crash, energy, smoothness) using the reward scales from `wp1_cfg.reward`.

### 2. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--wp1-checkpoint` | **Required** | Path to WP1 trained actor |
| `--wp1-cfg` | **Required** | Path to WP1 config YAML |
| `--pop-size` | 64 | CMA-ES population size |
| `--num-generations` | 100 | Number of evolution generations |
| `--num-eval-envs` | 256 | Total parallel evaluation environments |
| `--n-episodes` | 1 | Episodes per individual (for averaging) |
| `--add-decay` | True | Evolve per-weight decay (λ) |
| `--add-eta` | False | Evolve per-weight learning rates (η) |
| `--seed` | None | RNG seed for reproducibility |
| `--log-dir` | Auto | Output directory (default: `logs/runs_hebbian/<timestamp>/`) |
| `--device` | `cuda` | PyTorch device |

### 3. Programmatic Usage

```python
from WP2.cmaes_integration import run_cmaes_evolution

# Fitness is automatically WP1's weighted reward
evolution, run_dir = run_cmaes_evolution(
    wp1_checkpoint="logs/runs/.../model.pt",
    wp1_cfg="src/WP1/configs/foundation.yaml",
    pop_size=64,
    num_generations=100,
    num_eval_envs=256,
    add_decay=True,
    device="cuda",
)

# Access results
best_controller = evolution.get_best_controller()
best_rules = evolution.get_best_rules()
print(f"Best fitness history: {evolution.best_fitness_history}")
print(f"Best cumulative reward: {evolution.best_fitness_history[-1]:.3f}")
```

## Architecture

### Files

| File | Purpose |
|------|---------|
| `CMA_ES_loop.py` | Core CMA-ES evolution class (`CMAESHebbianEvolution`) |
| `cmaes_integration.py` | Integration with `RulesEvolutionEnv` for fitness evaluation |
| `run_cmaes.py` | CLI entry point (similar to `WP1.train`) |
| `rules_evolution.py` | Batched population evaluation environment |
| `checkpoint_loader.py` | Load and freeze WP1 actor |
| `hebbian.py` | Hebbian controller and rules management |

### Fitness Computation

The evolution uses **single-objective optimization** via CMA-ES, with fitness computed as the **cumulative WP1 weighted reward**:

```
reward = (progress * w_progress + 
          -crash * w_crash + 
          -energy * w_energy + 
          -smoothness * w_smoothness + ...)
```

Where weights (`w_*`) come from `wp1_cfg.reward`. This is the **same reward function used during WP1 training**, ensuring consistency across the pipeline.

**Reward components** (from WP1 environment):
- **Progress**: Gaussian reward for matching target forward velocity
- **Crash**: Penalty for collisions/crashes
- **Energy**: Penalty for power consumption
- **Smoothness**: Penalty for large action changes

### Population Evaluation

Each generation:

1. **CMA-ES asks** for candidate solutions (genomes)
2. **Create controllers** from genomes (Hebbian rules decoded)
3. **Evaluate in parallel**:
   - `RulesEvolutionEnv` steps all individuals simultaneously on GPU
   - Each individual tested across E different forest scenarios
   - Fitness aggregated as mean across scenarios
4. **CMA-ES tells** fitness values
5. **Track history** and save checkpoints

**Batching**: With `pop_size=64` and `num_eval_envs=256`, each environment slot evaluates multiple individuals efficiently.

## Genome Representation

The flat genome encodes Hebbian rules for the last layer `Linear(64→7)`:

```
Genome = [A_0, ..., A_447, B_0, ..., B_447, C_0, ..., C_447, D_0, ..., D_447, λ_0, ..., λ_447]
                                                              ↓
                                                    (optional if add_decay=True)
```

- **A, B, C, D**: Each is 448D (7 actions × 64 hidden neurons)
  - A: postsynaptic × presynaptic interaction
  - B: presynaptic bias
  - C: postsynaptic bias
  - D: constant drift
- **λ (decay)**: Per-weight coefficient pulling weights toward checkpoint (∈[0, 0.1])
- **η (learning rate)**: (optional) Per-weight learning rate for Hebbian updates

### Constraints Applied During Evolution

- **A, B, C, D**: Clamped to [-1, 1] (standard CMA-ES range)
- **λ**: Clamped to [0, 0.1] via sigmoid
- **η**: Positive via softplus (if evolved)

## Output Structure

Results saved to `logs/runs_hebbian/<timestamp>/`:

```
logs/runs_hebbian/2026-04-11_14-30-25_cmaes/
├── evolution_gen010.pt      # Checkpoint at generation 10
├── evolution_gen020.pt      # Checkpoint at generation 20
├── evolution_final.pt       # Final state
└── (future) plots/, metrics.csv, videos/  # TODO
```

Checkpoints contain:
- Population genome history
- CMA-ES internal state (mean, covariance, step-size)
- Best individual and fitness history
- Evolution statistics per generation

## Resume From Checkpoint

```python
from WP2.CMA_ES_loop import CMAESHebbianEvolution

evolution = CMAESHebbianEvolution(...)
evolution.load_state("logs/runs_hebbian/.../evolution_gen050.pt")

# Continue from generation 50
for gen in range(50, 150):
    best_ind, best_fit, stats = evolution.step()
    ...
```

## Comparison: CMA-ES vs NSGA-II

| Aspect | CMA-ES | NSGA-II |
|--------|--------|---------|
| **Objective** | Single-objective (maximize fitness) | Multi-objective (Pareto front) |
| **Fitness** | WP1 weighted reward (progress, crash, energy, smoothness) | Multi-objective scalarization |
| **Sample efficiency** | Generally faster convergence for 1D | Better for trade-offs (3+ objectives) |
| **Genome** | Flat real-valued vector | Can mix discrete/real genes |
| **Implementation** | Simpler (external `cma` library) | Complex (DEAP, custom operators) |
| **WP2 use** | Current (this module) | Previous implementation |

For WP2, CMA-ES is simpler and more sample-efficient when the goal is to **maximize WP1's reward function** (consistent with WP1 training objectives).

## Troubleshooting

### OOM Errors
Reduce `num_eval_envs` or `pop_size`:
```bash
python -m WP2.run_cmaes ... --num-eval-envs 128 --pop-size 32
```

### Slow Evaluation
- Increase `num_eval_envs` to better utilize GPU parallelism
- Reduce `n-episodes` to 1 (or increase it for stability)
- Use multiple GPUs (not yet supported; TODO)

### Evolution Not Converging
- Increase `pop_size` (larger population = better exploration)
- Check WP1 checkpoint quality
- Verify forest config is the same between WP1 and WP2

## Future Extensions

- [ ] Multi-objective CMA-ES (MOEA/D or NSGA-III)
- [ ] Morphology co-evolution (genome = Hebbian + URDF genes)
- [ ] Visualization of evolved rules and fitness progression
- [ ] Distributed evaluation across multiple GPUs
- [ ] Adaptive fitness weighting (e.g., progress vs energy trade-off)

## References

- **CMA-ES**: Hansen & Ostermeier (2001), "Completely derandomized self-adaptation..."
- **WP1**: PPO training of morphology-blind policy
- **WP2 goal**: Plasticity rules + (optionally) morphology via evolutionary optimization
