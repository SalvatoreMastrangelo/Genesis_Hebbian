# Human-Runnable Scripts

Simple Python scripts that can be executed directly without pytest.

## Available Scripts

### `test_hebbian_rules.py` — Test Hebbian rules creation and integration

Create Hebbian ABCD plasticity rules for a loaded WP1 checkpoint and verify they integrate correctly with the actor. Tests initialization strategies, weight modifications, and forward passes.

**Usage:**

```bash
# Find and test with latest checkpoint (uniform initialization)
python human_runnable/test_hebbian_rules.py

# Test with zero initialization (no weight changes)
python human_runnable/test_hebbian_rules.py --init-method zero

# Test specific checkpoint
python human_runnable/test_hebbian_rules.py \
    --checkpoint logs/runs/2026-03-01_12-00-00_exp/checkpoints/model_1000.pt \
    --config logs/runs/2026-03-01_12-00-00_exp/config.yaml

# Use CPU instead of CUDA
python human_runnable/test_hebbian_rules.py --device cpu

# Quiet mode (no output)
python human_runnable/test_hebbian_rules.py --quiet
```

**Output:**

Displays:
- ✓ Actor architecture structure
- ✓ Last layer dimensions (input/output sizes)
- ✓ Hebbian rule shapes (A, B, C, D, lam)
- ✓ Genome dimensions for evolution
- ✓ Forward pass validation
- ✓ Hebbian weight update verification
- ✓ Before/after weight statistics

**Arguments:**

```
--checkpoint PATH        Path to checkpoint (.pt file). If not specified, finds latest
--config PATH           Path to WP1 config (YAML). Default: src/WP1/configs/foundation.yaml
--device DEVICE         Device to load onto (cuda, cpu). Default: auto-detect
--init-method {zero,uniform}  Rule initialization method. Default: uniform
--quiet                 Suppress output
--help                  Show help message
```

**Example output:**

```
==========================================================================================
 HEBBIAN RULES TEST — Creation & Integration
==========================================================================================

Checkpoint: logs/runs/2026-03-01_12-00-00_exp/checkpoints/model_1000.pt
Config:     src/WP1/configs/foundation.yaml
Device:     cuda

[...steps 1-7...]

STEP 8: Applying Hebbian Update & Verifying Weight Changes
──────────────────────────────────────────────────────────────────────────────────────────
✓ Hebbian update applied successfully

Weight change statistics:
  Max change:  0.001215
  Mean change: 0.000285
  Num changed: 895/896
  ✓ WEIGHTS MODIFIED (as expected)

==========================================================================================
SUMMARY
==========================================================================================
✓ Hebbian rules successfully created and attached to actor
✓ Last layer: Linear(128 → 7)
✓ Genome dimensions: 3,584D (base) to 4,480D (with decay)
✓ Ready for WP2 evolution with NSGA-II + Hebbian plasticity
==========================================================================================
```

---

### `inspect_checkpoint.py` — Inspect WP1 checkpoint architecture

Load a WP1 checkpoint and display detailed information about the actor architecture.

**Usage:**

```bash
# Find and inspect the latest checkpoint
python human_runnable/inspect_checkpoint.py

# Inspect a specific checkpoint
python human_runnable/inspect_checkpoint.py \
    --checkpoint logs/runs/2026-03-01_12-00-00_exp/checkpoints/model_1000.pt \
    --config logs/runs/2026-03-01_12-00-00_exp/config.yaml

# Use CPU instead of CUDA
python human_runnable/inspect_checkpoint.py --device cpu

# Quiet mode (no output)
python human_runnable/inspect_checkpoint.py --quiet
```

**Output:**

Displays:
- ✓ Model type and status
- ✓ Layer dimensions (input/output sizes)
- ✓ Parameter count (frozen vs trainable)
- ✓ Actor MLP structure
- ✓ Last layer inspection
- ✓ Forward pass validation
- ✓ Weight statistics

**Example output:**

```
================================================================================
WP1 CHECKPOINT ARCHITECTURE INSPECTION
================================================================================

Checkpoint: logs/runs/2026-03-09_15-42-40_drone-forest/tb/model_1999.pt
Config:     logs/runs/2026-03-09_15-42-40_drone-forest/config.yaml
Device:     cuda

---
LOADING CHECKPOINT...
---
✓ Checkpoint loaded successfully

---
BASIC INFORMATION
---
Model type:        ActorCriticTanh
Recurrent:         True
Training mode:     False
Device:            cuda:0

---
LAYER DIMENSIONS
---
Last layer input:  64
Last layer output: 7 (actions)

[... more sections ...]

✓ Checkpoint loaded successfully
✓ Actor extracted (critic removed)
✓ Backbone frozen (all parameters non-trainable)
✓ Last layer weights are modifiable buffers
✓ Model ready for WP2 use
```

**Arguments:**

```
--checkpoint PATH     Path to checkpoint (.pt file). If not specified, finds latest
--config PATH         Path to WP1 config (YAML). Default: src/WP1/configs/foundation.yaml
--device DEVICE       Device to load onto (cuda, cpu). Default: auto-detect
--quiet              Suppress output
--help               Show help message
```

## Finding Your Checkpoint

Checkpoints are saved in:
```
logs/runs/
├── 2026-03-01_12-00-00_my-exp/
│   ├── config.yaml
│   ├── checkpoints/
│   │   ├── model_100.pt
│   │   ├── model_500.pt
│   │   └── model_1000.pt    ← Final checkpoint
│   └── ...
└── ...
```

List recent checkpoints:
```bash
ls -lht logs/runs/*/checkpoints/ | head -20
```

Or list checkpoints from a specific run:
```bash
ls logs/runs/2026-03-09_15-42-40_drone-forest/checkpoints/
```

## Quick Start Examples

### 1. Check latest checkpoint

```bash
python human_runnable/inspect_checkpoint.py
```

### 2. Check specific training run

```bash
# Find the run
ls logs/runs/ | grep drone-forest
# Output: 2026-03-09_15-42-40_drone-forest

# Inspect it
python human_runnable/inspect_checkpoint.py \
    --checkpoint logs/runs/2026-03-09_15-42-40_drone-forest/tb/model_1999.pt \
    --config logs/runs/2026-03-09_15-42-40_drone-forest/config.yaml
```

### 3. Check on CPU (for debugging)

```bash
python human_runnable/inspect_checkpoint.py --device cpu
```

## Integration with WP2

After inspecting the checkpoint, use it in WP2 with Hebbian plasticity:

```python
from WP2 import (
    load_wp1_actor, 
    get_actor_last_layer, 
    get_actor_dimensions,
    create_hebbian_rules,
    attach_hebbian_rules_to_actor,
    get_hebbian_genome_dim,
)

# Use the same checkpoint path you inspected
checkpoint = "logs/runs/2026-03-01_12-00-00_exp/checkpoints/model_1000.pt"
config = "logs/runs/2026-03-01_12-00-00_exp/config.yaml"

# Load actor
actor = load_wp1_actor(checkpoint, config, device="cuda")

# Create Hebbian ABCD rules for the last layer
rules = create_hebbian_rules(
    actor,
    init_method="uniform",      # or "zero"
    init_range=(-0.05, 0.05),
    add_decay=True,
    device="cuda"
)

# Attach rules to actor
actor = attach_hebbian_rules_to_actor(actor, rules=rules)

# Get genome dimension for evolution
genome_dim = get_hebbian_genome_dim(actor, add_decay=True)
print(f"Genome dimension for NSGA-II: {genome_dim}")

# Now integrate with evolutionary algorithm (DEAP)
```

**Workflow:**
1. **Inspect** — Use `inspect_checkpoint.py` to verify actor structure
2. **Test Hebbian Rules** — Use `test_hebbian_rules.py` to verify integration
3. **Evolve** — Use `python -m WP2.run` with loaded actor + rules

## Troubleshooting

### ModuleNotFoundError: No module named 'WP2'

Solution: Add `src` to PYTHONPATH or run from repo root:
```bash
cd /path/to/Genesis_Hebbian
python human_runnable/inspect_checkpoint.py
```

### FileNotFoundError: Checkpoint not found

Solution: Check the path exists:
```bash
ls logs/runs/.../checkpoints/model_1000.pt
```

### CUDA out of memory

Solution: Use CPU instead:
```bash
python human_runnable/inspect_checkpoint.py --device cpu
```

### Python import errors

Solution: Make sure you're in the repo root and have the virtual environment activated:
```bash
cd /path/to/Genesis_Hebbian
source .venv/bin/activate
python human_runnable/inspect_checkpoint.py
```

## See Also

- [../tests/integration/CHECKPOINT_TEST_GUIDE.md](../tests/integration/CHECKPOINT_TEST_GUIDE.md) — Pytest version
- [../src/WP2/CHECKPOINT_LOADER.md](../src/WP2/CHECKPOINT_LOADER.md) — Loader documentation
