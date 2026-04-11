# Quick Guide: Testing Checkpoint Architecture

Use this test to inspect the controller architecture loaded from a WP1 checkpoint.

## Quick Start

### 1. Find your latest checkpoint

```bash
ls -lht logs/runs/*/checkpoints/ | head -20
```

This shows the most recent checkpoint files. Pick one, e.g.:
```
logs/runs/2026-03-01_12-00-00_my-exp/checkpoints/model_1000.pt
```

### 2. Run the architecture test

```bash
pytest tests/integration/test_checkpoint_architecture.py -v -s \
    --checkpoint logs/runs/2026-03-01_12-00-00_my-exp/checkpoints/model_1000.pt \
    --config src/WP1/configs/foundation.yaml
```

## What the Test Shows

The test displays:

✓ **Basic Information**
  - Model type (ActorCriticTanh)
  - Recurrency status
  - Current device (CPU/CUDA)
  - Training mode

✓ **Layer Dimensions**
  - Input dimension (hidden features from LSTM)
  - Output dimension (7 actions for drone)

✓ **Parameters**
  - Total parameter count
  - Frozen vs trainable parameters
  - Parameter names and shapes

✓ **Submodules**
  - Actor MLP layers
  - LSTM backbone (memory_a)
  - Critic status (should be removed)

✓ **Last Layer Inspection**
  - Shape, dtype, device
  - Weight statistics (mean, std, min, max)
  - Whether weights are modifiable buffers

✓ **Forward Pass Test**
  - Runs dummy input through the network
  - Shows output shape and value range
  - Verifies the model is functional

## Common Checkpoint Locations

After running `python -m WP1.train ...`, checkpoints are saved in:

```
logs/runs/
├── 2026-03-01_12-00-00_my-exp/
│   ├── config.yaml           # Frozen config
│   ├── checkpoints/
│   │   ├── model_100.pt      ← Latest checkpoint at iter 100
│   │   ├── model_200.pt
│   │   ├── model_500.pt
│   │   └── model_1000.pt     ← Final checkpoint
│   ├── tb/                   # TensorBoard logs
│   ├── eval/
│   └── plots/
└── 2026-02-15_14-30-00_another-exp/
    └── checkpoints/
        └── model_500.pt
```

## Useful Variations

### Test with default paths (if you have recent runs)

```bash
pytest tests/integration/test_checkpoint_architecture.py -v -s
```

The test will try to find the most recent checkpoint in `logs/runs/`.

### Test with custom config

```bash
pytest tests/integration/test_checkpoint_architecture.py -v -s \
    --checkpoint logs/runs/2026-03-01_12-00-00_exp/checkpoints/model_1000.pt \
    --config src/WP1/configs/custom_config.yaml
```

### Test on CPU (if CUDA not available)

```bash
pytest tests/integration/test_checkpoint_architecture.py -v -s \
    --checkpoint logs/runs/.../checkpoints/model_1000.pt \
    --device cpu
```

## Expected Output

A successful run should show:

```
================================================================================
WP1 CHECKPOINT ARCHITECTURE INSPECTION
================================================================================

Checkpoint: logs/runs/2026-03-01_12-00-00_my-exp/checkpoints/model_1000.pt
Config:     src/WP1/configs/foundation.yaml
Device:     cuda

--------------------------------------------------------------------------------
LOADING CHECKPOINT...
--------------------------------------------------------------------------------
✓ Checkpoint loaded successfully

--------------------------------------------------------------------------------
BASIC INFORMATION
--------------------------------------------------------------------------------
Model type:        ActorCriticTanh
Recurrent:         True
Training mode:     False
Device:            cuda:0

[... more sections ...]

ACTOR MLP ARCHITECTURE
Layers in actor.actor (MLP):
  [0] Linear            100 → 64
  [1] ELU
  [2] Linear             64 → 64
  [3] ELU
  [4] Linear             64 →  7

LAST LINEAR LAYER (MODIFIABLE)
Layer type:         Linear
Input dimension:    64
Output dimension:   7
Weight shape:       torch.Size([7, 64])
Weight dtype:       torch.float32
Weight is buffer:   True

[... final summary ...]

✓ Checkpoint loaded successfully
✓ Actor extracted (critic removed)
✓ Backbone frozen (all parameters non-trainable)
✓ Last layer weights are modifiable buffers
```

## Troubleshooting

### FileNotFoundError: Checkpoint not found

**Error:**
```
FileNotFoundError: Checkpoint not found: logs/runs/.../model_1000.pt
```

**Solution:**
- Verify the checkpoint path exists: `ls -la logs/runs/.../checkpoints/`
- Use the exact path including `.pt` extension

### FileNotFoundError: WP1 config not found

**Error:**
```
FileNotFoundError: WP1 config not found: src/WP1/configs/foundation.yaml
```

**Solution:**
- Check that WP1 config exists: `ls src/WP1/configs/`
- If using a different config, pass it with `--config path/to/config.yaml`

### ImportError or RuntimeError during load

**Error:**
```
RuntimeError: Could not find last Linear layer in actor network
```

**Solution:**
- Verify the checkpoint is from a WP1 training run
- Check that the WP1 config matches the checkpoint architecture
- Ensure no version mismatches between config and checkpoint

### CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Use `--device cpu` to test on CPU instead
- Close other GPU processes
- Try with a different checkpoint if it's particularly large

## Integration with WP2

After verifying the checkpoint loads correctly, you can use it in WP2:

```python
from WP2 import load_wp1_actor, get_actor_last_layer, get_actor_dimensions

# Use the same checkpoint path you tested
checkpoint = "logs/runs/2026-03-01_12-00-00_my-exp/checkpoints/model_1000.pt"
config = "src/WP1/configs/foundation.yaml"

actor = load_wp1_actor(checkpoint, config, device="cuda")
last_layer = get_actor_last_layer(actor)
hidden_dim, num_actions = get_actor_dimensions(actor)

# Now attach Hebbian rules, evolutionary updates, etc.
```

## See Also

- [../../src/WP2/CHECKPOINT_LOADER.md](../../src/WP2/CHECKPOINT_LOADER.md) — Full checkpoint loader documentation
- [test_checkpoint_loader.py](test_checkpoint_loader.py) — Integration test examples
- WP1 training docs — How to generate checkpoints
