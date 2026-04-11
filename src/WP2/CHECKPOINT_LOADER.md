# WP2 Checkpoint Loader

Load a WP1-trained policy checkpoint and extract the actor for use in WP2 evolution.

## Overview

The checkpoint loader module handles:
- Loading WP1 `.pt` checkpoint files
- Building the model architecture from WP1 config
- Extracting the **actor** and discarding the **critic**
- Freezing the LSTM backbone + MLP layers
- Converting the last linear layer to a modifiable buffer

## Quick Start

```python
from WP2 import load_wp1_actor, get_actor_last_layer, get_actor_dimensions

# Load the actor
actor = load_wp1_actor(
    checkpoint_path="logs/runs/<timestamp>/checkpoints/model_1000.pt",
    wp1_cfg_path="src/WP1/configs/foundation.yaml",
    device="cuda"
)

# Extract the last layer (for Hebbian modifications)
last_layer = get_actor_last_layer(actor)
hidden_dim, num_actions = get_actor_dimensions(actor)

print(f"Actor last layer: {hidden_dim} → {num_actions}")
```

## Functions

### `load_wp1_actor(checkpoint_path, wp1_cfg_path, device="cpu") → nn.Module`

Load a WP1 checkpoint and extract the frozen actor.

**Parameters:**
- `checkpoint_path` (str or Path): Path to the `.pt` checkpoint
- `wp1_cfg_path` (str or Path): Path to the WP1 config YAML
- `device` (str): Device to load onto ('cpu', 'cuda', etc.)

**Returns:**
- `actor` (nn.Module): Frozen ActorCriticTanh with critic removed

**Properties of returned actor:**
- LSTM backbone parameters: frozen (`requires_grad=False`)
- MLP hidden layers: frozen (`requires_grad=False`)
- Last linear layer: weights stored as **buffer** (modifiable in-place without autograd)
- Critic: completely removed
- Model is in evaluation mode (`model.eval()`)

### `get_actor_last_layer(actor) → nn.Linear`

Extract the last linear layer from the actor.

**Parameters:**
- `actor` (nn.Module): The actor network

**Returns:**
- `last_layer` (nn.Linear): The final linear layer (hidden_dim → num_actions)

### `get_actor_dimensions(actor) → Tuple[int, int]`

Get the last layer's input and output dimensions.

**Parameters:**
- `actor` (nn.Module): The actor network

**Returns:**
- `hidden_dim` (int): Input dimension of the last layer
- `num_actions` (int): Output dimension (typically 7 for the drone)

## Usage Patterns

### Pattern 1: Extract and inspect

```python
from WP2 import load_wp1_actor, get_actor_dimensions

actor = load_wp1_actor(checkpoint_path, wp1_cfg_path, device="cuda")
hidden_dim, num_actions = get_actor_dimensions(actor)

print(f"Loaded actor with {hidden_dim}→{num_actions} output layer")
```

### Pattern 2: Modify last layer weights

```python
from WP2 import load_wp1_actor, get_actor_last_layer

actor = load_wp1_actor(checkpoint_path, wp1_cfg_path, device="cuda")
last_layer = get_actor_last_layer(actor)

# Weights are stored as a buffer, modifiable in-place
with torch.no_grad():
    last_layer.weight.mul_(0.5)  # Scale to 50%
    last_layer.weight.add_(0.1)  # Add small noise
```

### Pattern 3: Forward pass through the actor

```python
from WP2 import load_wp1_actor

actor = load_wp1_actor(checkpoint_path, wp1_cfg_path, device="cuda")

# obs shape: (batch, obs_dim)
obs = torch.randn(32, obs_dim, device="cuda")

with torch.no_grad():
    actions = actor.act_inference(obs)  # (32, num_actions)
```

### Pattern 4: Prepare for Hebbian evolution

```python
from WP2 import load_wp1_actor, get_actor_last_layer, get_actor_dimensions

actor = load_wp1_actor(checkpoint_path, wp1_cfg_path, device="cuda")
last_layer = get_actor_last_layer(actor)
hidden_dim, num_actions = get_actor_dimensions(actor)

# Calculate Hebbian genome dimension
num_weights = hidden_dim * num_actions
hebbian_params_per_weight = 4  # A, B, C, D
hebbian_genome_dim = hebbian_params_per_weight * num_weights

print(f"Hebbian genome size: {hebbian_genome_dim}")
# Now attach HebbianLastLayer or similar mechanism
```

## Checkpoint Formats Supported

The loader handles multiple checkpoint formats:

1. **RSL-RL wrapper format**: `{"model_state_dict": {...}}`
2. **Direct state dict**: `{"actor.0.weight": ..., "actor.0.bias": ..., ...}`
3. **Raw state dict**: The checkpoint itself is the state dict

The loader automatically detects and handles all formats.

## Architecture

The loaded model has this structure (simplified):

```
ActorCriticTanh (frozen, eval mode)
├── memory_a (LSTM backbone)          [frozen]
├── actor (MLP sequence)
│   ├── Linear(obs_dim, 64)           [frozen]
│   ├── ELU
│   ├── Linear(64, 64)                [frozen]
│   ├── ELU
│   └── Linear(64, num_actions)       [modifiable buffer]
└── memory_c (Critic LSTM)            [NOT LOADED - removed]
```

## Testing

Run the integration tests:

```bash
pytest tests/integration/test_checkpoint_loader.py -v
```

Note: Most tests require an actual WP1 checkpoint file and will skip if not available.

## Common Issues

### FileNotFoundError on checkpoint
```
FileNotFoundError: Checkpoint not found: logs/runs/.../model_1000.pt
```

Check that the checkpoint path exists. WP1 checkpoints are saved in:
```
logs/runs/<timestamp>_<exp_name>/checkpoints/model_*.pt
```

### FileNotFoundError on WP1 config
```
FileNotFoundError: WP1 config not found: src/WP1/configs/foundation.yaml
```

Check that the config path is correct. Default location is `src/WP1/configs/foundation.yaml`.

### RuntimeError: Could not find last Linear layer
```
RuntimeError: Could not find last Linear layer in actor network
```

This indicates the actor structure is different from expected. Verify that:
1. The checkpoint is from a WP1 training run
2. The WP1 config matches the checkpoint architecture

## Integration with WP2 Workflow

Typical WP2 setup:

```python
from WP2 import load_wp1_actor, get_actor_last_layer, get_actor_dimensions
from WP2.hebbian import HebbianLastLayer  # (not yet implemented)
from WP2.evolution import EvolutionStrategy  # (not yet implemented)

# 1. Load pretrained actor
actor = load_wp1_actor(checkpoint, config, device="cuda")
last_layer = get_actor_last_layer(actor)
hidden_dim, num_actions = get_actor_dimensions(actor)

# 2. Create Hebbian rules for population
hebbian_pop = create_hebbian_population(
    pop_size=256,
    hidden_dim=hidden_dim,
    num_actions=num_actions,
)

# 3. Attach Hebbian to actor
# hebbian = HebbianLastLayer(last_layer, hebbian_rules[0], ...)
# wrapper = HebbianActorWrapper(actor, hebbian)

# 4. Run evolution
# optimizer.evolve(actor, hebbian_pop, env, ...)
```

See WP2 planning docs for full integration details.
