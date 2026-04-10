# Hebbian Plasticity Tests

Tests for Hebbian learning mechanisms, rule initialization, and weight updates.

## Overview

This test suite verifies that the Hebbian plasticity system works correctly, particularly focusing on:
- Genome initialization with zero rules
- Rule decoding from normalized [0,1] space to actual ranges
- Hebbian weight updates with various configurations
- Comparison between zero and non-zero rules

## Test Files

### 1. `test_zero_initialization.py`

**Purpose**: Verify the full pipeline of zero-initialized genome creation and decoding.

**Tests**:
1. **Genome Creation** — Checks that `create_zero_initialized_genome()` produces genomes with A=B=C=D=0.5 in [0,1] space
2. **Genome Decoding** — Verifies that 0.5 values decode to 0.0 in actual [-10.0, 10.0] ranges for ABCD
3. **Network Integration** — Confirms decoded rules can be applied to network layers without errors

**Run**:
```bash
python tests/hebbian/test_zero_initialization.py
```

**Expected Output**:
```
✓ PASS: Genome Creation
✓ PASS: Genome Decoding
✓ PASS: Network Integration
✓ ALL TESTS PASSED - Zero initialization is working correctly!
```

### 2. `test_zero_hebbian_updates.py`

**Purpose**: Simulate Hebbian weight updates with zero and non-zero rules to verify behavior.

**Tests**:
1. **Zero Rules → No Weight Changes** — 10 simulated Hebbian update steps with zero A,B,C,D rules
   - Verifies weights remain frozen (drift < 1e-7)
   - Confirms bias is never modified
   - Checks that decay-only weight changes are negligible

2. **Non-Zero Rules → Weight Changes** — Comparison test with random rules
   - Confirms non-zero rules DO produce measurable weight changes
   - Validates that the Hebbian mechanism is functional

**Run**:
```bash
python tests/hebbian/test_zero_hebbian_updates.py
```

**Expected Output**:
```
✓ PASS: Zero rules → No weight changes
✓ PASS: Non-zero rules → Weight changes
✓ ALL TESTS PASSED
Conclusion: initialize_rules_to_zero: true is working correctly.
```

### 3. `ZERO_INITIALIZATION_TEST_REPORT.md`

**Purpose**: Detailed documentation of test results, technical explanation, and conclusions.

**Contents**:
- Test summary with pass/fail status
- Individual test details with metrics
- Technical explanation of Hebbian update equations
- Configuration verification
- Code path verification
- Conclusions and implications for experiments

## Configuration

Tests use the following key configuration:

```yaml
hebbian:
  enabled: true
  eta: 0.0001
  decay: 0.001
  evolve_decay: false
  initialize_rules_to_zero: true  # Key config being tested
  A_range: [-10.0, 10.0]
  B_range: [-10.0, 10.0]
  C_range: [-10.0, 10.0]
  D_range: [-10.0, 10.0]
```

## Technical Details

### Genome Structure

A typical Hebbian genome contains:
- **A, B, C, D blocks**: 4 × (num_actions × hidden_dim) = 4 × (7 × 64) = 1792 values
- **Optional decay**: 7 × 64 = 448 values (if `evolve_decay: true`)
- **Optional eta**: 7 × 64 = 448 values (if `evolve_eta: true`)

Total genome size = 1792 + optional components

### Hebbian Update Equation

```python
# ABCD contribution to weight change
dW = eta * k * (A*xy + B*x + C*y + D)

# Full weight update with decay towards checkpoint
W_new = W_old * (1 - lambda) + lambda * W_checkpoint + dW
```

Where:
- `x`: presynaptic activation (batch_size, hidden_dim)
- `y`: postsynaptic activation (batch_size, num_actions)
- `eta`: learning rate
- `lambda`: decay coefficient
- `W_checkpoint`: initial weight (before plasticity)

### Zero Rules Case

When A = B = C = D = 0:
```python
dW = eta * k * (0 + 0 + 0 + 0) = 0
```

Weight update becomes:
```python
W_new = W_old * (1 - lambda) + lambda * W_checkpoint
```

This means weights decay towards the checkpoint at rate `lambda`. With small lambda (0.001), this produces negligible drift.

## Key Insights

✅ **Working Correctly**:
- Zero-initialized genomes are created with all ABCD at 0.5 [0,1]
- Decoding correctly transforms 0.5 → 0.0 in [-10, 10] ranges
- Zero rules result in frozen weights (drift ~ 1e-7)
- Biases are never modified by Hebbian updates
- Non-zero rules produce expected weight changes

⚠️ **Important Notes**:
- `initialize_rules_to_zero: true` creates a frozen actor (no plasticity)
- This is useful as a **control condition** for evolution
- Morphology evolution can still proceed independently
- The actor behaves as a fixed policy with only weight decay effects

## Adding More Hebbian Tests

When adding new Hebbian tests:

1. **Create test file**: `test_hebbian_[feature].py`
2. **Follow structure**:
   ```python
   def test_feature():
       # Setup config
       # Create genome/rules
       # Execute test
       # Assert results
       return pass/fail
   
   def main():
       results = [("Test Name", test_feature())]
       # Print summary
       return 0 if all passed else 1
   ```

3. **Add to pytest**: Tests are automatically discovered if named `test_*.py`
4. **Document results**: Create corresponding `.md` report

## Troubleshooting

### Import errors
```bash
# Make sure you're in the project root
cd /path/to/Genesis_Hebbian
python tests/hebbian/test_zero_initialization.py
```

### Config not found
Tests create minimal configs programmatically, so config files aren't required. If loading actual configs fails, check:
- `src/WP2/configs/custom.yaml` exists
- YAML parser is installed (`pip install pyyaml`)

### Large weight drift
If you see weight drift > 1e-5:
- Check that `decay` is small (0.001 is expected)
- Verify ABCD rules are indeed zero (check decoded rule values)
- Check that `use_oja_coefficient: false` in config

## Running with pytest

```bash
# Install pytest
pip install pytest

# Run all Hebbian tests
pytest tests/hebbian/ -v

# Run specific test file
pytest tests/hebbian/test_zero_initialization.py -v

# Run and show print statements
pytest tests/hebbian/ -s

# Run with detailed output on failure
pytest tests/hebbian/ -vv --tb=long
```

---

For detailed results, see [`ZERO_INITIALIZATION_TEST_REPORT.md`](ZERO_INITIALIZATION_TEST_REPORT.md).
