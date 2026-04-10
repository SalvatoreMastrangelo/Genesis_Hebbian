# Hebbian Zero Initialization Test Report

## Summary
✅ **VERIFIED**: The `initialize_rules_to_zero: true` configuration correctly sets Hebbian rules to zero.

## What Was Tested

### 1. Genome Creation (`test_zero_initialization.py`)
**Test**: Does `create_zero_initialized_genome()` set A,B,C,D to 0.5 in normalized [0,1] space?

**Result**: ✓ PASS
- All A, B, C, D values are exactly 0.5 in [0,1] space
- Genome size: 1792 (7 actions × 64 hidden + 4 ABCD blocks)

### 2. Genome Decoding (`test_zero_initialization.py`)
**Test**: When decoded from [0,1] → actual ranges, do 0.5 values become 0.0?

**Result**: ✓ PASS
- A, B, C, D all decode to exactly **0.0000000000**
- Decoding formula: `value = normalized * (max - min) + min`
  - For 0.5 with range [-10.0, 10.0]: `0.5 * 20.0 + (-10.0) = 0.0` ✓

**Decoded Rules Structure**:
```
A: torch.Size([7, 64]) → all zeros
B: torch.Size([7, 64]) → all zeros
C: torch.Size([7, 64]) → all zeros
D: torch.Size([7, 64]) → all zeros
lam (decay): torch.Size([7, 64]) → uses config default (0.001)
```

### 3. Hebbian Weight Updates (`test_zero_hebbian_updates.py`)
**Test**: Do zero rules prevent actual weight changes during Hebbian updates?

**Result**: ✓ PASS - Weights stay frozen

**Test Parameters**:
- 10 simulated Hebbian update steps
- Random activations at each step
- Config: eta=0.0001, decay=0.001, use_oja_coefficient=False

**Measured Changes**:
| Metric | Value | Status |
|--------|-------|--------|
| Per-step weight change | 1.49e-08 | ✓ Minimal |
| Total weight drift (10 steps) | 1.49e-07 | ✓ Negligible |
| Bias drift | 0.00e+00 | ✓ No change |
| Max single-step change | 1.49e-08 | ✓ Negligible |

### 4. Comparison Test: Non-Zero Rules
**Test**: Do non-zero rules produce measurable weight changes?

**Result**: ✓ PASS - Non-zero rules work
- Random rules produce per-step weight change: **1.00e-01**
- This confirms the Hebbian mechanism is functional
- Zero rules are special case, not a broken system

## Technical Explanation

### Hebbian Update Equation
```python
# ABCD update term
dW = eta * k * (A*xy + B*x + C*y + D)

# Full weight update
W = W * (1 - lambda) + lambda * W_checkpoint + dW
```

### What Happens with Zero Rules
When A = B = C = D = 0:
```python
dW = eta * k * (0 + 0 + 0 + 0) = 0
```

Therefore:
```python
W = W * (1 - lambda) + lambda * W_checkpoint + 0
```

**Result**: Weights only decay towards checkpoint at rate lambda (0.001), producing negligible drift (~1e-7 over 10 steps).

## Configuration Verified

From `src/WP2/configs/custom.yaml`:
```yaml
hebbian:
  enabled: true
  eta: 0.0001
  decay: 0.001
  evolve_decay: false
  initialize_rules_to_zero: true  # ← This works correctly
  A_range: [-10.0, 10.0]
  B_range: [-10.0, 10.0]
  C_range: [-10.0, 10.0]
  D_range: [-10.0, 10.0]
```

## Code Path Verified

1. **Initialization** (`src/WP2/evolve.py:732-737`):
   ```python
   if self.cfg.hebbian.initialize_rules_to_zero:
       from WP2.utils import create_zero_initialized_genome
       zero_genome = create_zero_initialized_genome(self.cfg)
       for ind in pop:
           for i in range(len(ind)):
               ind[i] = zero_genome[i]
   ```

2. **Decoding** (`src/WP2/utils.py:70-138`):
   - Maps [0,1] genome to actual ABCD ranges
   - 0.5 → 0.0 for symmetric ranges

3. **Hebbian Update** (`src/WP2/hebbian.py:85-127`):
   - Applies ABCD update with zero rules
   - Results in dW = 0, weights frozen

## Conclusion

✅ **The initialization is working correctly.** When `initialize_rules_to_zero: true`:

1. **Genome is created** with A=0.5, B=0.5, C=0.5, D=0.5 in [0,1] space
2. **Rules decode to** A=0.0, B=0.0, C=0.0, D=0.0 in actual ranges
3. **Weights are frozen** with only negligible decay drift (~1e-7)
4. **Bias is never modified** (0.0 change observed)
5. **The actor behaves as a fixed policy**, exactly as intended

### What This Means for Your Experiment
- When using `initialize_rules_to_zero: true`, you get a **frozen actor with no plasticity**
- This is useful as a **baseline/control condition**
- Morphology evolution can still proceed independently
- The evolved population will differ only in their morphology and initial conditions, not in Hebbian plasticity

## Test Files
- `test_zero_initialization.py` - Genome creation and decoding tests
- `test_zero_hebbian_updates.py` - Weight update simulation tests
- This report - Summary and verification

Run either test with:
```bash
python -m pytest tests/hebbian/test_zero_initialization.py
python -m pytest tests/hebbian/test_zero_hebbian_updates.py
```

Or directly:
```bash
python tests/hebbian/test_zero_initialization.py
python tests/hebbian/test_zero_hebbian_updates.py
```
