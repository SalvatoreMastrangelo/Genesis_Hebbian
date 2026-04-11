# CMA-ES Evolution Loop for Hebbian Rules

## Overview

`CMA_ES_loop.py` implements **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** for evolving Hebbian plasticity rules (ABCD + decay/eta coefficients).

### Key Features

- **Flat genome representation**: Hebbian rules flattened to 1D vector for CMA-ES optimization
- **Two controller strategies**:
  - **Create new batch per generation** (clean, flexible)
  - **Reuse & reset** (memory-efficient, ~2x faster)
- **Batched evaluation** via `HebbianControllerBatch` for parallel forward passes
- **Flexible reward functions** — bring your own fitness evaluation
- **State checkpointing** — save/load evolution progress
- **Multi-objective support** — weighted aggregation of multiple objectives

## Installation

Requires CMA-ES library:
```bash
pip install cma
```

## Quick Start

### 1. Define Your Reward Function

```python
from src.WP2.CMA_ES_loop import CMAESHebbianEvolution
from src.WP2.hebbian import HebbianControllerBatch

def evaluate_fitness(controllers):
    """
    Evaluate controllers and return reward array.
    
    Args:
        controllers: List[HebbianController]
    
    Returns:
        rewards: np.ndarray of shape (pop_size,), higher is better
    """
    # Run your environment simulation here
    batch = HebbianControllerBatch(controllers)
    
    # Simulate episode(s)
    obs_batch = env.reset(num_envs=len(controllers))
    batch.reset_hidden_states()
    
    rewards = np.zeros(len(controllers), dtype=np.float32)
    
    for step in range(episode_length):
        actions = batch.forward_batch(obs_batch)
        obs_batch, step_rewards, dones, _ = env.step(actions)
        
        # Update Hebbian rules (if your architecture supports it)
        # x_batch, y_batch = extract_activations(...)
        # batch.hebbian_update_batch(x_batch, y_batch)
        
        rewards += step_rewards
    
    return rewards
```

### 2. Create Evolution Loop

```python
from src.WP2.checkpoint_loader import load_wp1_actor

# Load frozen actor
actor = load_wp1_actor(checkpoint_path, wp1_cfg_path, device="cuda")

# Extract checkpoint weights
last_layer = list(actor.actor.modules())[-1]  # Last Linear layer
w_checkpoint = last_layer.weight.data.clone()

# Create evolution loop
evolution = CMAESHebbianEvolution(
    actor=actor,
    w_checkpoint=w_checkpoint,
    reward_fn=evaluate_fitness,
    pop_size=50,
    add_decay=True,      # Evolve decay coefficients
    add_eta=False,       # Use fixed eta (0.01)
    reuse_batch=True,    # Memory-efficient: reset & reuse batch
    device="cuda",
    seed=42
)
```

### 3. Run Evolution

**Synchronous (simple):**
```python
for gen in range(num_generations):
    best_genome, best_fitness, stats = evolution.step()
    print(f"Gen {gen}: best={stats.best_fitness:.3f}, "
          f"mean={stats.mean_fitness:.3f}")

# Access results
best_controller = evolution.get_best_controller()
best_rules = evolution.get_best_rules()
```

**Manual control (distributed evaluation):**
```python
for gen in range(num_generations):
    # Get candidate solutions
    genomes = evolution.ask()  # Shape: (pop_size, genome_dim)
    
    # Evaluate in parallel (your code)
    fitnesses = parallel_evaluate(genomes, env)  # Your function
    
    # Advance generation
    best_genome, best_fitness, stats = evolution.tell(fitnesses)
```

## Architecture

### Genome Structure

Flat vector encoding Hebbian rules:

```
[A_flat (448) | B_flat (448) | C_flat (448) | D_flat (448) | lam_flat (448) | eta_flat (448)]
                 └─ 7×64 reshaped
```

Genome dimensions:
- **Base (A,B,C,D)**: 4 × 448 = 1,792
- **With decay**: +448 = 2,240
- **With eta**: +448 = 2,688

### HebbianController

Individual controller with:
- **Frozen actor** (same for all individuals)
- **Per-controller rules** (A, B, C, D, lam, eta) — evolved by CMA-ES
- **Per-controller weights** (W) — reset at each generation start
- **Hebbian update** — plasticity during episode

### HebbianControllerBatch

Batch manager for `pop_size` controllers:
- Batched forward passes through shared actor
- Independent LSTM hidden states per controller
- Vectorized Hebbian updates
- Efficient reset for new generations

## Two Strategies for Controller Management

### Strategy 1: Create New Batch (default if `reuse_batch=False`)

```python
evolution = CMAESHebbianEvolution(..., reuse_batch=False)

for gen in range(num_generations):
    best_genome, best_fitness, stats = evolution.step(create_new_batch=True)
    # New HebbianControllerBatch created each generation
```

**Pros**: Clean separation, no state coupling
**Cons**: Memory overhead, slower due to allocations

### Strategy 2: Reuse & Reset Batch (memory-efficient, default)

```python
evolution = CMAESHebbianEvolution(..., reuse_batch=True)

for gen in range(num_generations):
    best_genome, best_fitness, stats = evolution.step(create_new_batch=False)
    # Existing batch reused, weights/rules reset each generation
```

**Pros**: ~2x faster, lower memory
**Cons**: Slightly more complex state management

## Customization

### Multi-Episode Evaluation

```python
def evaluate_multi_episode(controllers, num_episodes=3):
    rewards = np.zeros(len(controllers))
    for ep in range(num_episodes):
        ep_rewards = run_single_episode(controllers)
        rewards += ep_rewards
    return rewards / num_episodes
```

### Reward Shaping

```python
def evaluate_with_penalties(controllers):
    rewards = run_episode(controllers)  # (pop_size,)
    
    # Example: penalize high weight magnitudes
    for idx, ctrl in enumerate(controllers):
        weight_penalty = 0.01 * torch.norm(ctrl.W)
        rewards[idx] -= weight_penalty
    
    return rewards
```

### Multi-Objective (weighted sum)

```python
def evaluate_multi_objective(controllers):
    # Objective 1: trajectory length
    lengths = run_episodes(controllers)
    
    # Objective 2: energy efficiency
    energies = compute_energy_costs(controllers)
    
    # Weighted sum (CMA-ES expects single scalar)
    w1, w2 = 0.7, 0.3
    rewards = w1 * lengths + w2 * (1 - energies)
    
    return rewards
```

## API Reference

### CMAESHebbianEvolution

**Methods:**

- `step(create_new_batch=None)` → (genome, fitness, stats)
  - Run one generation, uses reward_fn

- `ask()` → genomes (pop_size, genome_dim)
  - Request candidates (manual control)

- `tell(fitnesses)` → (genome, fitness, stats)
  - Provide fitness values (manual control)

- `get_best_controller()` → HebbianController
  - Access best individual as controller

- `get_best_rules()` → dict
  - Access best Hebbian rules (A, B, C, D, lam, eta)

- `get_best_individual()` → np.ndarray
  - Access best genome (flat vector)

- `save_state(filepath)` → None
  - Checkpoint evolution state

- `load_state(filepath)` → None
  - Resume from checkpoint

**Properties:**

- `generation` (int) — current generation number
- `pop_size` (int) — population size
- `genome_dim` (int) — flat genome dimension
- `best_fitness_history` (list) — best fitness per generation
- `mean_fitness_history` (list) — mean fitness per generation

### EvolutionStats (returned each step)

```python
@dataclass
class EvolutionStats:
    generation: int
    pop_size: int
    best_fitness: float
    worst_fitness: float
    mean_fitness: float
    std_fitness: float
    median_fitness: float
```

## Examples

See `CMA_ES_example.py` for:
1. Simple synchronous evaluation
2. Batched evaluation with HebbianControllerBatch
3. Multi-objective evaluation
4. Full Genesis integration template
5. Manual ask/tell pattern for distributed evaluation

## Configuration

### Recommended Settings

**For fast iteration (small pop_size):**
```python
evolution = CMAESHebbianEvolution(
    ..., 
    pop_size=20,
    reuse_batch=True,  # Fast reset
    add_decay=True,
    add_eta=False
)
```

**For robust optimization (larger pop_size):**
```python
evolution = CMAESHebbianEvolution(
    ...,
    pop_size=100,
    reuse_batch=True,  # Still efficient
    add_decay=True,
    add_eta=True  # More expressive rules
)
```

**For exploration (high diversity):**
```python
# CMA-ES will auto-adjust sigma; set loose initial x0
```

## Notes

### Weight Constraints

- Decay (lam): automatically clamped to [0, 1] via sigmoid
- Eta: automatically positive via softplus
- Weights (W): clamped to [-w_max, w_max] during updates

### Hebbian Update Details

Per-controller update:
```
dW = eta * k * (A*outer(y,x) + B*x + C*y + D)
W_new = W * (1 - lam) + lam * W_checkpoint + dW
```

Where:
- `k` is Oja coefficient (if enabled)
- `outer(y,x)` is postsynaptic-presynaptic correlation

### Reproducibility

Use `seed` parameter:
```python
evolution = CMAESHebbianEvolution(..., seed=42)
```

## Troubleshooting

**"cma package not installed"**
```bash
pip install cma
```

**OOM during evaluation**
- Reduce `pop_size`
- Use `reuse_batch=True` (default)
- Reduce `num_episodes` in reward function

**Evolution stagnating**
- Increase `pop_size`
- Check reward function (may have all similar values)
- Try higher initial sigma in CMA-ES

**Slow evolution**
- Profile reward function — likely bottleneck
- Consider distributed evaluation with `ask()`/`tell()`
- Use `reuse_batch=True` for faster reset

## References

- CMA-ES: Hansen et al. (2003)
- Hebbian plasticity: Hebbian learning rules
- Genesis Physics Simulator: https://genesis-world.readthedocs.io/
