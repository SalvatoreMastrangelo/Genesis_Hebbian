# Environment-Isolated Hebbian Plasticity

## Overview

`EnvironmentHebbianLastLayer` provides **per-environment weight matrices** with **independent plasticity updates**. Each environment has its own controller copy and weights are updated based **only on that environment's activations alone** — no averaging or coupling between environments.

## Use Case

For evolutionary algorithms where:
- Population size: 256 individuals
- Environments per individual: 32 parallel environments
- Total environments: 256 × 32 = 8192
- **Result: 8192 independent weight matrices, one per environment**

Each environment tests the plasticity rules under different conditions (different morphologies, wind, terrain, etc.), and weights evolve independently in each scenario.

## Architecture

```
Population (256 individuals)
    │
    ├─ Individual 1 → [32 environments, each with own weight matrix]
    │                 W[0], W[1], ..., W[31]  ← 32 separate (7, 64) matrices
    │
    ├─ Individual 2 → [32 environments, each with own weight matrix]
    │                 W[32], W[33], ..., W[63]
    │
    ...
    │
    └─ Individual 256 → [32 environments, each with own weight matrix]
                        W[8160], W[8161], ..., W[8191]

Total: W shape (8192, 7, 64) ← 8192 independent weight matrices
```

## API

### Initialization

```python
from WP2 import create_hebbian_rules, get_actor_last_layer
from WP2 import EnvironmentHebbianLastLayer

# Load checkpoint
actor = load_wp1_actor(ckpt_path, cfg_path, device="cuda")
last_layer = get_actor_last_layer(actor)

# Create shared rules (same ABCD for all environments)
rules = create_hebbian_rules(
    actor,
    init_method="uniform",
    init_range=(-0.05, 0.05),
    add_decay=True,
    device="cuda"
)

# Create environment-isolated Hebbian wrapper
# 256 individuals × 32 environments = 8192 total
num_environments = 256 * 32  # 8192

hebbian_env = EnvironmentHebbianLastLayer(
    linear_layer=last_layer,
    hebbian_rules=rules,
    num_environments=num_environments,
    eta=0.01,
    w_max=3.0,
    use_oja_coefficient=True,
    device="cuda"
)
```

### Weight Update (Per-Environment)

```python
# Each environment produces its own activations
x = torch.randn(8192, 64, device="cuda")  # (num_envs, hidden_dim)
y = torch.randn(8192, 7, device="cuda")   # (num_envs, num_actions)

# Update: each environment's weights based ONLY on its activations
hebbian_env.hebbian_update(x, y)

# Result: each of 8192 weight matrices updated independently
# W[0] ← updated based on x[0], y[0] only
# W[1] ← updated based on x[1], y[1] only
# ...
# W[8191] ← updated based on x[8191], y[8191] only
```

### Access Individual Weights

```python
# Get all per-environment weights
W_all = hebbian_env.W  # Shape: (8192, 7, 64)

# Access weights for a specific individual's environments
individual_idx = 5  # Individual 5
env_start = individual_idx * 32
env_end = env_start + 32

W_individual_5 = W_all[env_start:env_end]  # Shape: (32, 7, 64)

# Access weights for a specific environment
env_global_idx = 150
W_env_150 = W_all[env_global_idx]  # Shape: (7, 64)
```

### Reset Weights

```python
# Reset all environments to checkpoint weights
hebbian_env.reset_weights()
# All W[i] → W_checkpoint for i in 0..8191
```

## Mathematical Operation

For **each environment independently** (no averaging):

```
For environment e with activations x[e] and y[e]:

1. Compute Hebb term:
   outer[e] = y[e] ⊗ x[e]  (shape: 7×64)

2. Optional Oja coefficient:
   k[e] = 1 - (y[e]² · (W[e] - W_checkpoint)) / (outer[e] + eps)

3. Weight delta:
   dW[e] = η · k[e] · (A·outer[e] + B·x[e] + C·y[e] + D)

4. Update weights (with decay):
   W[e] = W[e]·(1-λ) + W_checkpoint·λ + dW[e]
   W[e] = clamp(W[e], -w_max, w_max)
```

**Key difference from batch averaging:**
```python
# Batch average (old way):
x_mean = x.mean(dim=0)        # Average all 8192 environments
dW = eta * (A * (y_mean ⊗ x_mean) + ...)
W = W + dW                    # Single weight matrix updated

# Environment isolation (new way):
# For each environment e:
x_e = x[e]                    # Just that environment's input
y_e = y[e]                    # Just that environment's output
dW_e = eta * (A * (y_e ⊗ x_e) + ...)
W[e] = W[e] + dW_e           # 8192 weight matrices updated independently
```

## Complete Evolution Loop Example

```python
import torch
from WP2 import (
    load_wp1_actor,
    get_actor_last_layer,
    create_hebbian_rules,
    EnvironmentHebbianLastLayer,
)

# Configuration
population_size = 256
num_envs_per_individual = 32
num_total_envs = population_size * num_envs_per_individual  # 8192
episode_length = 1000

# Load actor and create rules
actor = load_wp1_actor(ckpt, cfg, device="cuda")
last_layer = get_actor_last_layer(actor)
rules = create_hebbian_rules(actor, init_method="uniform", device="cuda")

# Create environment-isolated plasticity
hebbian_env = EnvironmentHebbianLastLayer(
    linear_layer=last_layer,
    hebbian_rules=rules,
    num_environments=num_total_envs,
    eta=0.01,
    device="cuda"
)

# Evolution loop
for generation in range(num_generations):
    # Reset weights for all environments
    hebbian_env.reset_weights()
    
    # Run episode in all 8192 environments in parallel
    for step in range(episode_length):
        # Get activations from all environments
        # (Your simulation code produces these)
        x, y = simulator.get_activations(num_total_envs)  # (8192, 64), (8192, 7)
        
        # INDEPENDENT update for each environment
        hebbian_env.hebbian_update(x, y)
        
        # Take actions (from W[e] for each environment e)
        # ...
    
    # Evaluate fitness for each environment
    fitness = evaluate_all_environments(num_total_envs)
    
    # Group fitness by individual (average 32 environments per individual)
    fitness_per_individual = fitness.view(population_size, num_envs_per_individual).mean(dim=1)
    
    # Evolutionary update (NSGA-II)
    new_rules = evolve_population(rules, fitness_per_individual)
    rules = new_rules
```

## Advantages

✅ **Robustness Testing** — Rules tested across 32 diverse scenarios per individual  
✅ **Environmental Generalization** — Can evolve rules that work universally  
✅ **No Averaging Artifacts** — Each environment updates independently  
✅ **Population Diversity** — Each individual explores different morphological niches  

## Comparison: Batch vs Environment-Isolated

| Aspect | Batch Averaging | Environment-Isolated |
|--------|-----------------|----------------------|
| **Weight matrices** | 1 per individual (256 total) | 1 per environment (8192 total) |
| **Averaging** | Yes, across all 8192 envs | No, each env updates alone |
| **Update rule** | Single dW per generation | 8192 different dW values |
| **Memory** | 256 × (7×64) = 114K params | 8192 × (7×64) = 3.6M params |
| **Robustness** | High (smoothed learning) | Very high (all scenarios tested) |
| **Complexity** | Simple | More complex (more matrices) |

## Performance Notes

- **Memory**: 8192 weight matrices requires ~3.6MB (fp32)
- **Computation**: ~8192× more Hebbian updates per generation (but parallelizable)
- **Speed**: Still fast on GPU (vectorized einsum operations)

## See Also

- [HEBBIAN_RULES.md](src/WP2/HEBBIAN_RULES.md) — Rules creation API
- [hebbian.py](src/WP2/hebbian.py) — Implementation source code
