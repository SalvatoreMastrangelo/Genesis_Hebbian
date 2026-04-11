# Hebbian Rules Creator — Usage Guide

## Overview

The `hebbian.py` module provides utilities to create and initialize Hebbian ABCD plasticity rules for the last layer of a loaded actor checkpoint. The rules can be used with `HebbianLastLayer` or `BatchedHebbianLastLayer` from the existing Hebbian plasticity system.

## Quick Start

### Basic Example: Create Hebbian Rules (Zero-Initialized)

```python
from src.WP2.checkpoint_loader import load_wp1_actor
from src.WP2.hebbian import create_hebbian_rules

# Load a trained WP1 actor
actor = load_wp1_actor(
    checkpoint_path="logs/runs/model_checkpoint.pt",
    wp1_cfg_path="src/WP1/configs/foundation.yaml",
    device="cuda"
)

# Create Hebbian ABCD rules initialized to zero
rules = create_hebbian_rules(
    actor,
    init_method="zero",
    add_decay=True,
    device="cuda"
)

# rules is a dict with keys: A, B, C, D, lam (all shape [7, 64])
print(rules.keys())  # dict_keys(['A', 'B', 'C', 'D', 'lam'])
print(rules["A"].shape)  # torch.Size([7, 64])
```

### Example: Create Rules with Uniform Initialization

```python
# Create ABCD rules with uniform random initialization
rules = create_hebbian_rules(
    actor,
    init_method="uniform",
    init_range=(-0.05, 0.05),  # Sample A, B, C, D from this range
    add_decay=True,
    decay_init="uniform",
    decay_range=(0.0, 0.1),    # Sample lam from this range
    add_eta=True,              # Also include per-weight learning rates
    eta_value=0.01,
    device="cuda"
)

print(f"A range: [{rules['A'].min():.4f}, {rules['A'].max():.4f}]")
```

### Example: Attach Rules Directly to Actor

```python
from src.WP2.hebbian import attach_hebbian_rules_to_actor

# Attach rules as buffers to the actor
actor = attach_hebbian_rules_to_actor(
    actor,
    init_method="uniform",
    init_range=(-0.05, 0.05),
    add_decay=True,
    device="cuda"
)

# Access rules via the actor's hebbian_rules attribute
A = actor.hebbian_rules["A"]
B = actor.hebbian_rules["B"]
print(A.shape)  # torch.Size([7, 64])
```

## API Reference

### `create_hebbian_rules()`

Create Hebbian ABCD (and optionally decay/eta) tensors for the actor's last layer.

**Parameters:**
- `actor` (nn.Module): The loaded actor network (from `load_wp1_actor`)
- `init_method` (str): `"uniform"` or `"zero"` — initialization strategy for A, B, C, D
- `init_range` (tuple): Range for uniform init: `(low, high)`. Default: `(-0.1, 0.1)`
- `add_decay` (bool): Whether to add decay (lambda) tensor. Default: `True`
- `decay_init` (str): `"uniform"` or `"zero"` for decay initialization
- `decay_range` (tuple): Range for decay uniform init. Default: `(0.0, 0.1)`
- `add_eta` (bool): Whether to add per-weight learning rate tensor. Default: `False`
- `eta_value` (float): Default eta value if `add_eta=True`. Default: `0.01`
- `device` (str): Device for tensors. Default: `"cpu"`

**Returns:**
A dictionary with keys:
- `"A"`: (num_actions, hidden_dim) Hebbian coefficient
- `"B"`: (num_actions, hidden_dim) presynaptic bias
- `"C"`: (num_actions, hidden_dim) postsynaptic bias
- `"D"`: (num_actions, hidden_dim) constant drift
- `"lam"`: (num_actions, hidden_dim) decay coefficient [if `add_decay=True`]
- `"eta"`: (num_actions, hidden_dim) learning rate [if `add_eta=True`]

### `attach_hebbian_rules_to_actor()`

Attach Hebbian rules as buffers to the actor module.

**Parameters:**
- `actor` (nn.Module): The actor network
- `rules` (dict, optional): Pre-computed rules. If `None`, created via `create_hebbian_rules`
- `**create_kwargs`: Additional kwargs passed to `create_hebbian_rules` if rules is `None`

**Returns:**
The same actor module with `hebbian_rules` attribute.

### `extract_hebbian_rules_from_actor()`

Extract Hebbian rules from an actor that has them attached.

**Parameters:**
- `actor` (nn.Module): Actor with rules attached

**Returns:**
Dictionary of Hebbian rules (same format as `create_hebbian_rules`)

### `get_hebbian_genome_dim()`

Calculate the total Hebbian genome dimension for evolution.

**Parameters:**
- `actor` (nn.Module): The actor network
- `add_decay` (bool): Whether decay is included. Default: `True`
- `add_eta` (bool): Whether eta is included. Default: `False`

**Returns:**
Integer genome dimension.

**Example:**
```python
dim = get_hebbian_genome_dim(actor, add_decay=True, add_eta=False)
# For Linear(64→7): 5 × (7 × 64) = 2240
```

## Integration with HebbianLastLayer

Once you have created the rules, pass them to the existing Hebbian plasticity system:

```python
from src.WP2_old.hebbian import HebbianLastLayer
from src.WP2.checkpoint_loader import get_actor_last_layer

# Get the actor's last layer
last_layer = get_actor_last_layer(actor)

# Create Hebbian rules
rules = create_hebbian_rules(actor, init_method="zero", device="cuda")

# Create the Hebbian wrapper
hebbian_layer = HebbianLastLayer(
    linear_layer=last_layer,
    hebbian_rules=rules,
    eta=0.01,
    w_max=3.0,
    use_oja_coefficient=True,
    device="cuda"
)

# Now use hebbian_layer.hebbian_update() in your training loop
```

## Initialization Options

### Zero Initialization
All ABCD and decay values are set to 0. Useful for ablation studies or when you want to start plasticity from scratch.

```python
rules = create_hebbian_rules(actor, init_method="zero")
```

### Uniform Initialization
All ABCD and decay values are uniformly sampled from specified ranges. Useful for evolutionary algorithms that will mutate these values.

```python
rules = create_hebbian_rules(
    actor,
    init_method="uniform",
    init_range=(-0.1, 0.1),     # ABCD range
    decay_init="uniform",
    decay_range=(0.0, 0.1),     # decay range
)
```

## Dimensions

For a standard Genesis Hebbian drone with:
- **Hidden dimension:** 64 (LSTM output)
- **Action dimension:** 7 (throttle, sweeps, twists, elevator, rudder)
- **Last layer weight matrix:** 64 × 7 = 448 weights

The Hebbian rules have shape `(7, 64)` for each coefficient:
- **A, B, C, D:** 4 × 448 = 1792 scalars each
- **lam (decay):** 448 scalars
- **eta (learning rate):** 448 scalars (optional)

**Genome dimensions for evolution:**
- Base (A, B, C, D): 4 × 448 = **1792D**
- With decay: 5 × 448 = **2240D**
- With decay + eta: 6 × 448 = **2688D**

## Notes

1. **Biases are never modified:** The Hebbian rule only operates on weights. Biases remain frozen from the checkpoint.
2. **Rules are registered as buffers:** When attached to an actor, rules are buffers (not parameters), so they won't be updated by gradients during training.
3. **Ready for evolution:** The returned rules dictionary is ready to be encoded/decoded in evolutionary algorithms (DEAP, etc.).
4. **Device agnostic:** Rules are automatically placed on the specified device (CPU, CUDA, etc.).
