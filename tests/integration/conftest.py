"""
Pytest configuration for integration tests.

Registers custom command-line options for checkpoint testing.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for checkpoint architecture test."""
    parser.addoption(
        "--checkpoint",
        action="store",
        default=None,
        help="Path to WP1 checkpoint (.pt file)",
    )
    parser.addoption(
        "--config",
        action="store",
        default="src/WP1/configs/foundation.yaml",
        help="Path to WP1 config (YAML)",
    )
    parser.addoption(
        "--device",
        action="store",
        default=None,
        help="Device to load onto (cuda, cpu)",
    )


@pytest.fixture
def checkpoint_paths(request):
    """Fixture to pass checkpoint path from command line."""
    return {
        "checkpoint": request.config.getoption("--checkpoint"),
        "config": request.config.getoption("--config"),
        "device": request.config.getoption("--device"),
    }
