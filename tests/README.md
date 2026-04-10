# Genesis Hebbian Test Suite

This directory contains all functional tests for the Genesis Hebbian project. Tests are organized by module to ensure comprehensive coverage of different components.

## Structure

```
tests/
├── README.md                          (this file)
├── hebbian/                           (Hebbian plasticity tests)
│   ├── test_zero_initialization.py    (genome creation & decoding)
│   ├── test_zero_hebbian_updates.py   (weight update verification)
│   └── ZERO_INITIALIZATION_TEST_REPORT.md (detailed test results)
└── (morphology/, integration/, etc. to be added)
```

## Running Tests

### Run all tests
```bash
python -m pytest tests/
```

### Run a specific test module
```bash
python -m pytest tests/hebbian/
```

### Run a specific test
```bash
python -m pytest tests/hebbian/test_zero_initialization.py
```

### Run with verbose output
```bash
python -m pytest tests/ -v
```

### Direct execution (without pytest)
```bash
python tests/hebbian/test_zero_initialization.py
python tests/hebbian/test_zero_hebbian_updates.py
```

## Test Categories

### 1. Hebbian Plasticity Tests (`hebbian/`)

Tests for the Hebbian learning mechanisms, including rule initialization, genome decoding, and weight updates.

| Test File | Purpose |
|-----------|---------|
| `test_zero_initialization.py` | Verify zero-initialized genomes decode correctly to A=B=C=D=0 |
| `test_zero_hebbian_updates.py` | Confirm zero rules prevent weight changes during updates |
| `ZERO_INITIALIZATION_TEST_REPORT.md` | Comprehensive results and technical analysis |

**Key Tests**:
- ✅ Genome creation with `initialize_rules_to_zero: true`
- ✅ Decoding [0,1] normalized values to actual ABCD ranges
- ✅ Hebbian weight update with zero rules (weights frozen)
- ✅ Non-zero rules comparison (weights do change)

**Expected Results**:
- Zero rules produce negligible weight drift (~1e-7 over 10 steps)
- Bias is never modified
- Non-zero rules produce measurable weight changes

---

## Adding New Tests

When adding tests for new features:

1. **Create appropriate subdirectory** (e.g., `tests/morphology/`)
2. **Create `__init__.py`** to mark as Python package
3. **Name test files** as `test_*.py` (pytest convention)
4. **Update this README** with test categories and instructions
5. **Add test report** (`.md`) if results should be documented

### Test Template

```python
#!/usr/bin/env python3
"""
Test module for [component name].
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import what you need
from WP2.config import Config

def test_feature_works():
    """Test that [feature] works as expected."""
    # Setup
    # Execute
    # Assert
    assert result == expected

def main():
    results = []
    results.append(("Feature Test", test_feature_works()))
    
    # Print summary
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    return 0 if all(p for _, p in results) else 1

if __name__ == "__main__":
    exit(main())
```

---

## Configuration

Some tests may require specific configurations. Check individual test files for:
- Required Python packages
- Configuration files needed
- GPU/CPU requirements
- Memory requirements

Most tests use the configuration in `src/WP2/configs/` and load dynamically.

---

## Dependencies

Tests require:
- Python 3.10+
- PyTorch
- NumPy
- YAML

Install with:
```bash
pip install -r requirements.txt
```

Or with pytest integration:
```bash
pip install pytest
```

---

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```bash
# Run all tests with exit code 0 only if all pass
python -m pytest tests/ --tb=short
```

---

## Test Reports

Test results and detailed analysis are documented in:
- [`hebbian/ZERO_INITIALIZATION_TEST_REPORT.md`](hebbian/ZERO_INITIALIZATION_TEST_REPORT.md) — Hebbian plasticity verification

New test categories should include similar reports documenting:
- What was tested
- Test parameters and configuration
- Results (passed/failed)
- Technical explanation of findings
- Code paths verified

---

## Troubleshooting

### Import errors
If tests fail with import errors, ensure:
- You're running from the project root: `cd /path/to/Genesis_Hebbian`
- `src/` directory exists and contains the code modules
- Python path is correctly set in test files

### Config errors
Some tests require valid config files. Check:
- `src/WP2/configs/custom.yaml` exists (for WP2 tests)
- `src/WP1/configs/` exists (for WP1 tests)

### GPU availability
Tests default to CPU. To use GPU:
- Modify the test's `device="cpu"` to `device="cuda"` if available
- Ensure CUDA is properly installed

---

## Contributing

When fixing bugs or adding features:
1. Write tests first (TDD approach recommended)
2. Run tests to ensure they fail initially
3. Implement the feature/fix
4. Re-run tests to verify they pass
5. Update test reports with results

---

For questions or issues, refer to individual test files or the project documentation.
