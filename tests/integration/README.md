# Integration Tests

End-to-end integration tests covering multiple components together.

## Status

⏳ **To be implemented** — Placeholder for integration tests

## Planned Tests

- WP1 training pipeline (PPO)
- WP2 evolution pipeline (NSGA-II + Hebbian)
- Environment initialization and stepping
- Policy evaluation with morphology
- Multi-environment simulation
- Log output and checkpoint saving

## Directory Structure

```
integration/
├── __init__.py
├── README.md (this file)
├── test_wp1_training.py (to be created)
├── test_wp2_evolution.py (to be created)
├── test_environment_integration.py (to be created)
└── INTEGRATION_TEST_REPORT.md (to be created)
```

## Adding Tests

When implementing integration tests:
1. Test complete pipelines with actual configs
2. Verify data flow between components
3. Check output artifacts (logs, checkpoints)
4. Validate numerical results

Follow the structure in [`../hebbian/`](../hebbian/) as a template.
