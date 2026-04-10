# Genesis Hebbian Testing Guide

Complete guide to the test suite for the Genesis Hebbian project.

## Quick Start

```bash
# Run all tests
python -m pytest tests/ -v

# Run Hebbian tests only
python -m pytest tests/hebbian/ -v

# Run a specific test
python tests/hebbian/test_zero_initialization.py

# Run with coverage
python -m pytest tests/ --cov=src
```

## Test Suite Organization

```
tests/
├── README.md                    (test suite overview)
├── TESTING_GUIDE.md             (this file)
│
├── hebbian/                     (✓ Implemented - Hebbian plasticity tests)
│   ├── __init__.py
│   ├── README.md
│   ├── test_zero_initialization.py         (✓ genome creation & decoding)
│   ├── test_zero_hebbian_updates.py        (✓ weight update verification)
│   └── ZERO_INITIALIZATION_TEST_REPORT.md  (✓ detailed results)
│
├── morphology/                  (⏳ Planned - morphology evolution tests)
│   ├── __init__.py
│   └── README.md
│
└── integration/                 (⏳ Planned - end-to-end tests)
    ├── __init__.py
    └── README.md
```

## Test Categories

### 1. ✅ Hebbian Plasticity Tests (`tests/hebbian/`)

**Status**: Complete and verified

**What's Tested**:
- Zero-initialized genome creation
- Rule decoding from [0,1] to actual ranges
- Hebbian weight updates with zero rules (frozen weights)
- Comparison with non-zero rules (weight changes)

**Key Files**:
- [`test_zero_initialization.py`](hebbian/test_zero_initialization.py) — Genome and decoding tests
- [`test_zero_hebbian_updates.py`](hebbian/test_zero_hebbian_updates.py) — Weight update simulation
- [`ZERO_INITIALIZATION_TEST_REPORT.md`](hebbian/ZERO_INITIALIZATION_TEST_REPORT.md) — Results analysis

**Run**:
```bash
python -m pytest tests/hebbian/ -v
```

**Results**:
- ✅ All tests passing
- ✅ Zero rules freeze weights (drift < 1e-7)
- ✅ Biases unchanged
- ✅ Non-zero rules produce changes

---

### 2. ⏳ Morphology Evolution Tests (`tests/morphology/`)

**Status**: Placeholder - to be implemented

**Planned Tests**:
- Morphology genome creation and validation
- Parameter decoding and ranges
- Multi-URDF generation
- Fitness evaluation
- Evolution convergence

**When to Add**:
- After morphology-specific bugs are identified
- When validation is needed for new morphology features
- Before major morphology changes in production

---

### 3. ⏳ Integration Tests (`tests/integration/`)

**Status**: Placeholder - to be implemented

**Planned Tests**:
- WP1 training pipeline (PPO)
- WP2 evolution pipeline (NSGA-II)
- Environment setup and stepping
- Policy evaluation
- Multi-environment simulation
- Log/checkpoint output validation

**When to Add**:
- After full pipeline changes
- For regression testing of major features
- Before releases

---

## Running Tests

### Basic Commands

```bash
# All tests
pytest tests/

# Specific category
pytest tests/hebbian/

# Specific test file
pytest tests/hebbian/test_zero_initialization.py

# Specific test function
pytest tests/hebbian/test_zero_initialization.py::test_zero_genome_creation

# With output
pytest tests/ -s

# Verbose output
pytest tests/ -v

# Very verbose (show all assertions)
pytest tests/ -vv

# Stop on first failure
pytest tests/ -x

# Show 10 slowest tests
pytest tests/ --durations=10
```

### Direct Execution (No pytest)

Tests can also be run directly without pytest:

```bash
python tests/hebbian/test_zero_initialization.py
python tests/hebbian/test_zero_hebbian_updates.py
```

### Checking Coverage

```bash
# Install coverage
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## Test Structure

All tests follow a consistent structure:

```python
#!/usr/bin/env python3
"""
Module docstring describing what is tested.
"""

import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import modules
from WP2.config import Config
from WP2.utils import some_function

def test_feature_one():
    """Test that [feature] works."""
    # Setup
    config = create_test_config()
    
    # Execute
    result = some_function(config)
    
    # Assert
    assert result is not None
    return True

def test_feature_two():
    """Test that [other feature] works."""
    # Similar structure
    return True

def main():
    """Run all tests in this module."""
    results = []
    results.append(("Feature One", test_feature_one()))
    results.append(("Feature Two", test_feature_two()))
    
    # Print summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(p for _, p in results)
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
```

---

## Writing New Tests

### Step 1: Choose a Category

- **Hebbian**: Rule initialization, genome decoding, weight updates
- **Morphology**: Genome creation, parameter ranges, URDF generation
- **Integration**: Multi-component pipelines, end-to-end workflows

### Step 2: Create Test File

```bash
# Create test file
touch tests/[category]/test_[feature].py
```

### Step 3: Implement Tests

Follow the template above. Key points:
- One test function per concept
- Clear setup/execute/assert flow
- Meaningful error messages
- Return True/False

### Step 4: Add Documentation

Create a corresponding `.md` file documenting:
- What was tested
- Test parameters
- Results and metrics
- Technical explanation
- Conclusions

Example: [`ZERO_INITIALIZATION_TEST_REPORT.md`](hebbian/ZERO_INITIALIZATION_TEST_REPORT.md)

### Step 5: Update README

Add entry to appropriate category's README (e.g., `tests/hebbian/README.md`)

---

## Test Configuration

Most tests create their own minimal configs. Some tests reference:
- `src/WP2/configs/custom.yaml` — WP2 evolution config
- `src/WP1/configs/foundation.yaml` — WP1 training config

Tests verify functionality without requiring external configs to exist.

---

## Dependencies

```bash
# Core
python >= 3.10
torch
numpy
pyyaml

# Testing
pytest              # Test runner
pytest-cov          # Coverage reporting

# Install all
pip install -r requirements.txt
pip install pytest pytest-cov
```

---

## CI/CD Integration

To integrate into CI/CD pipeline:

```bash
# GitHub Actions example
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/ -v
```

---

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'WP2'
```

**Solution**: Run from project root
```bash
cd /path/to/Genesis_Hebbian
python tests/hebbian/test_zero_initialization.py
```

### Config Not Found

Tests create minimal configs programmatically, so config files aren't required.

If you need to load actual config files:
```python
from pathlib import Path
import yaml

config_path = Path(__file__).parent.parent.parent / "src/WP2/configs/custom.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
```

### GPU/Device Issues

Tests default to CPU. To use GPU:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Test Metrics

Current test coverage:

| Category | Tests | Status | Coverage |
|----------|-------|--------|----------|
| Hebbian | 4 | ✅ Complete | High |
| Morphology | 0 | ⏳ Planned | — |
| Integration | 0 | ⏳ Planned | — |
| **Total** | **4** | — | — |

---

## Maintenance

### Regular Tasks

- **Weekly**: Run full test suite before merging PRs
- **Monthly**: Review test coverage and identify gaps
- **Quarterly**: Add integration tests for new features
- **As needed**: Update tests when fixing bugs

### Best Practices

✅ **Do**:
- Write tests for bugs before fixing them (TDD)
- Keep tests independent and isolated
- Use clear, descriptive test names
- Document expected behavior
- Test edge cases
- Verify error handling

❌ **Don't**:
- Skip failing tests
- Create flaky tests (dependent on timing)
- Mix multiple concepts in one test
- Ignore test failures
- Remove test code to make tests pass

---

## Reporting Issues

When tests fail:
1. **Collect information**:
   - Full test output
   - Configuration used
   - Python version
   - Platform (OS, GPU)
2. **Reproduce**:
   - Run test in isolation
   - Check if consistent
   - Try with fresh checkout
3. **Create issue** with findings

---

## Related Documentation

- [`../README.md`](README.md) — Test suite overview
- [`hebbian/README.md`](hebbian/README.md) — Hebbian test details
- [`.claude/CONTEXT.md`](../../.claude/CONTEXT.md) — Full project context
- [`src/WP2/config.py`](../../src/WP2/config.py) — Configuration classes

---

## Quick Reference

| Task | Command |
|------|---------|
| Run all tests | `pytest tests/ -v` |
| Run one category | `pytest tests/hebbian/ -v` |
| Run one file | `pytest tests/hebbian/test_zero_initialization.py` |
| Run one test | `pytest tests/hebbian/test_zero_initialization.py::test_zero_genome_creation` |
| Show prints | `pytest tests/ -s` |
| Stop on failure | `pytest tests/ -x` |
| Coverage report | `pytest tests/ --cov=src` |
| Direct execution | `python tests/hebbian/test_zero_initialization.py` |

---

For detailed information, see the README in each test category.
