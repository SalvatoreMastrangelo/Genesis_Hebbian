"""
Tests for WP2 checkpoint loader.
================================

Tests for loading WP1 checkpoints, extracting the actor, and preparing
it for modifications in WP2.

Usage
-----
Run with pytest:
    pytest tests/integration/test_checkpoint_loader.py -v

Or test specific examples:
    pytest tests/integration/test_checkpoint_loader.py::test_basic_load -v
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from WP2.checkpoint_loader import (
    load_wp1_actor,
    get_actor_last_layer,
    get_actor_dimensions,
)


# ============================================================================
#  Example: Basic load from a checkpoint
# ============================================================================

def test_basic_load():
    """Load actor from checkpoint and inspect structure.

    This example demonstrates:
    - Loading a WP1 checkpoint
    - Extracting the actor
    - Verifying the model structure
    """
    # NOTE: Requires actual WP1 checkpoint. Skip if not available.
    pytest.skip("Requires actual WP1 checkpoint file")

    checkpoint_path = "logs/runs/2026-03-01_12-00-00_my-training/checkpoints/model_1000.pt"
    wp1_cfg_path = "src/WP1/configs/foundation.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the actor
    actor = load_wp1_actor(checkpoint_path, wp1_cfg_path, device=device)

    # Verify it's the right model
    assert hasattr(actor, "actor"), "Actor should have 'actor' submodule"
    assert hasattr(actor, "memory_a"), "Actor should have LSTM backbone"
    assert not hasattr(actor, "critic"), "Critic should be dropped"


def test_get_actor_last_layer():
    """Extract and inspect the last linear layer.

    This example demonstrates:
    - Finding the last layer in the actor
    - Getting its dimensions
    """
    pytest.skip("Requires actual WP1 checkpoint file")

    checkpoint_path = "logs/runs/2026-03-01_12-00-00_my-training/checkpoints/model_1000.pt"
    wp1_cfg_path = "src/WP1/configs/foundation.yaml"

    actor = load_wp1_actor(checkpoint_path, wp1_cfg_path)
    last_layer = get_actor_last_layer(actor)

    # Verify it's a linear layer
    assert isinstance(last_layer, nn.Linear)
    assert hasattr(last_layer, "weight")


def test_actor_dimensions():
    """Get the last layer's dimensions.

    This example demonstrates:
    - Extracting hidden and action dimensions
    - Using them for downstream components (e.g., Hebbian rules)
    """
    pytest.skip("Requires actual WP1 checkpoint file")

    checkpoint_path = "logs/runs/2026-03-01_12-00-00_my-training/checkpoints/model_1000.pt"
    wp1_cfg_path = "src/WP1/configs/foundation.yaml"

    actor = load_wp1_actor(checkpoint_path, wp1_cfg_path)
    hidden_dim, num_actions = get_actor_dimensions(actor)

    # Verify dimensions match
    last_layer = get_actor_last_layer(actor)
    assert hidden_dim == last_layer.weight.shape[1]
    assert num_actions == last_layer.weight.shape[0]


def test_actor_frozen():
    """Verify that the actor backbone is frozen.

    This example demonstrates:
    - Checking that all backbone parameters are frozen
    - Only the last layer is modifiable
    """
    pytest.skip("Requires actual WP1 checkpoint file")

    checkpoint_path = "logs/runs/2026-03-01_12-00-00_my-training/checkpoints/model_1000.pt"
    wp1_cfg_path = "src/WP1/configs/foundation.yaml"

    actor = load_wp1_actor(checkpoint_path, wp1_cfg_path)

    # Count frozen and trainable parameters
    frozen_count = sum(1 for p in actor.parameters() if not p.requires_grad)
    trainable_count = sum(1 for p in actor.parameters() if p.requires_grad)

    # All parameters should be frozen
    assert trainable_count == 0, "Actor backbone should be fully frozen"
    assert frozen_count > 0, "Actor should have some parameters"


def test_last_layer_modifiable():
    """Verify that the last layer weights can be modified in-place.

    This example demonstrates:
    - The last layer is a buffer (not a parameter)
    - It can be modified without triggering autograd
    - Modifications persist
    """
    pytest.skip("Requires actual WP1 checkpoint file")

    checkpoint_path = "logs/runs/2026-03-01_12-00-00_my-training/checkpoints/model_1000.pt"
    wp1_cfg_path = "src/WP1/configs/foundation.yaml"

    actor = load_wp1_actor(checkpoint_path, wp1_cfg_path)
    last_layer = get_actor_last_layer(actor)

    # Store original weights
    original_weights = last_layer.weight.clone()

    # Modify in-place
    with torch.no_grad():
        last_layer.weight.mul_(0.5)  # Reduce to 50%

    # Verify modification persisted
    assert not torch.allclose(last_layer.weight, original_weights)
    assert torch.allclose(last_layer.weight, original_weights * 0.5)


def test_forward_pass():
    """Run a forward pass through the actor.

    This example demonstrates:
    - Passing observations through the frozen actor
    - Getting action predictions
    """
    pytest.skip("Requires actual WP1 checkpoint file")

    checkpoint_path = "logs/runs/2026-03-01_12-00-00_my-training/checkpoints/model_1000.pt"
    wp1_cfg_path = "src/WP1/configs/foundation.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    actor = load_wp1_actor(checkpoint_path, wp1_cfg_path, device=device)

    # Create dummy observations (must match obs_dim from config)
    obs_dim = 100  # Replace with actual from WP1 config
    batch_size = 4
    obs = torch.randn(batch_size, obs_dim, device=device)

    # Run inference
    with torch.no_grad():
        actions = actor.act_inference(obs)

    # Verify output shape
    hidden_dim, num_actions = get_actor_dimensions(actor)
    assert actions.shape == (batch_size, num_actions)


def test_checkpoint_not_found():
    """Verify proper error when checkpoint doesn't exist."""
    checkpoint_path = "/nonexistent/checkpoint.pt"
    wp1_cfg_path = "src/WP1/configs/foundation.yaml"

    with pytest.raises(FileNotFoundError):
        load_wp1_actor(checkpoint_path, wp1_cfg_path)


def test_config_not_found():
    """Verify proper error when config doesn't exist."""
    checkpoint_path = "logs/runs/2026-03-01_12-00-00_my-training/checkpoints/model_1000.pt"
    wp1_cfg_path = "/nonexistent/config.yaml"

    pytest.skip("Requires actual WP1 checkpoint file")

    with pytest.raises(FileNotFoundError):
        load_wp1_actor(checkpoint_path, wp1_cfg_path)


# ============================================================================
#  Example: Integration with Hebbian rules
# ============================================================================

def test_prepare_for_hebbian():
    """Prepare actor for Hebbian modifications.

    This example demonstrates:
    - Extracting actor ready for Hebbian attachment
    - Getting dimensions needed for Hebbian genome
    """
    pytest.skip("Requires actual WP1 checkpoint file")

    checkpoint_path = "logs/runs/2026-03-01_12-00-00_my-training/checkpoints/model_1000.pt"
    wp1_cfg_path = "src/WP1/configs/foundation.yaml"

    actor = load_wp1_actor(checkpoint_path, wp1_cfg_path)
    last_layer = get_actor_last_layer(actor)
    hidden_dim, num_actions = get_actor_dimensions(actor)

    # Verify actor is ready for Hebbian
    assert actor is not None
    assert last_layer is not None
    assert hidden_dim > 0 and num_actions > 0

    # Calculate Hebbian genome dimension (A, B, C, D per weight, plus optional params)
    num_weights = hidden_dim * num_actions
    hebbian_genome_base = 4 * num_weights  # A, B, C, D for each weight
    print(f"Actor ready for Hebbian:")
    print(f"  Last layer: {hidden_dim}×{num_actions} ({num_weights} weights)")
    print(f"  Hebbian genome base size: {hebbian_genome_base}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
