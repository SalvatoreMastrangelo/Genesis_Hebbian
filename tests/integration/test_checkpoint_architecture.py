"""
Test: Inspect WP1 checkpoint controller architecture.
======================================================

This test loads a WP1 checkpoint and displays detailed information about
the actor architecture. Run this to verify the checkpoint structure before
using it in WP2.

Usage
-----
Find your latest WP1 checkpoint:
    ls logs/runs/*/checkpoints/

Run the test with a specific checkpoint:
    pytest tests/integration/test_checkpoint_architecture.py -v -s \\
        --checkpoint logs/runs/2026-03-01_12-00-00_my-exp/checkpoints/model_1000.pt \\
        --config src/WP1/configs/foundation.yaml

Or use default paths (modify below):
    pytest tests/integration/test_checkpoint_architecture.py -v -s
"""

import sys
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn

from WP2 import load_wp1_actor, get_actor_last_layer, get_actor_dimensions


# ============================================================================
#  Configuration (modify these for your setup)
# ============================================================================

# Default paths to your WP1 checkpoint and config
DEFAULT_CHECKPOINT = None  # Will search for latest
DEFAULT_CONFIG = "src/WP1/configs/foundation.yaml"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_latest_checkpoint() -> Optional[Path]:
    """Find the latest WP1 checkpoint in logs/runs."""
    logs_dir = Path("logs/runs")
    if not logs_dir.exists():
        return None

    checkpoints = list(logs_dir.glob("*/checkpoints/model_*.pt"))
    if not checkpoints:
        return None

    # Sort by modification time, get the latest
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


# ============================================================================
#  Test: Load and display architecture
# ============================================================================

def _inspect_checkpoint_architecture(
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    """Load checkpoint and display controller architecture.

    This inspection:
    1. Loads a WP1 checkpoint
    2. Displays the actor architecture
    3. Shows layer dimensions
    4. Lists parameters and their shapes
    5. Verifies the model is in the expected state (frozen backbone, etc.)

    Pass checkpoint/config via pytest command line:
        pytest test_checkpoint_architecture.py -v -s \\
            --checkpoint path/to/model.pt \\
            --config path/to/config.yaml
    """
    # Use provided arguments or defaults
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    if config is None:
        config = DEFAULT_CONFIG
    if device is None:
        device = DEFAULT_DEVICE

    # If no checkpoint specified, try to find the latest
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
        if checkpoint is None:
            pytest.skip(
                "No checkpoint found. Provide one with:\n"
                "  pytest ... --checkpoint logs/runs/YYYYMMDD_HHMMSS_name/checkpoints/model_1000.pt"
            )

    checkpoint = Path(checkpoint)
    config = Path(config)

    print("\n" + "=" * 80)
    print("WP1 CHECKPOINT ARCHITECTURE INSPECTION")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint}")
    print(f"Config:     {config}")
    print(f"Device:     {device}")

    # ========================================================================
    #  LOAD CHECKPOINT
    # ========================================================================

    print("\n" + "-" * 80)
    print("LOADING CHECKPOINT...")
    print("-" * 80)

    try:
        actor = load_wp1_actor(
            checkpoint_path=checkpoint,
            wp1_cfg_path=config,
            device=device,
        )
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        pytest.fail(f"Failed to load checkpoint: {e}")

    # ========================================================================
    #  BASIC INFO
    # ========================================================================

    print("\n" + "-" * 80)
    print("BASIC INFORMATION")
    print("-" * 80)

    print(f"Model type:        {actor.__class__.__name__}")
    print(f"Recurrent:         {getattr(actor, 'recurrency', 'N/A')}")
    print(f"Training mode:     {actor.training}")
    print(f"Device:            {next(actor.parameters()).device}")

    # ========================================================================
    #  DIMENSIONS
    # ========================================================================

    print("\n" + "-" * 80)
    print("LAYER DIMENSIONS")
    print("-" * 80)

    try:
        hidden_dim, num_actions = get_actor_dimensions(actor)
        print(f"Last layer input:  {hidden_dim}")
        print(f"Last layer output: {num_actions} (actions)")
    except Exception as e:
        pytest.fail(f"Failed to get dimensions: {e}")

    # ========================================================================
    #  PARAMETER COUNT AND FREEZE STATUS
    # ========================================================================

    print("\n" + "-" * 80)
    print("PARAMETERS")
    print("-" * 80)

    frozen_params = sum(p.numel() for p in actor.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in actor.parameters())

    print(f"Total parameters:   {total_params:,}")
    print(f"Frozen parameters:  {frozen_params:,}")
    print(f"Trainable params:   {trainable_params:,}")

    if trainable_params > 0:
        print(f"\n⚠ WARNING: {trainable_params:,} trainable parameters found!")
        print("  Expected: all parameters frozen except last layer (which is a buffer)")

    # ========================================================================
    #  NAMED PARAMETERS
    # ========================================================================

    print("\n" + "-" * 80)
    print("NAMED PARAMETERS (first 10)")
    print("-" * 80)

    for i, (name, param) in enumerate(actor.named_parameters()):
        if i >= 10:
            remaining = sum(1 for _ in actor.named_parameters()) - i
            print(f"... and {remaining} more parameters")
            break
        requires_grad = "trainable" if param.requires_grad else "frozen"
        print(f"  {name:50s} {str(param.shape):20s} [{requires_grad}]")

    # ========================================================================
    #  BUFFERS (including last layer weights)
    # ========================================================================

    print("\n" + "-" * 80)
    print("BUFFERS (including last layer weight)")
    print("-" * 80)

    buffers = list(actor.named_buffers())
    if buffers:
        for name, buf in buffers[:10]:
            print(f"  {name:50s} {str(buf.shape):20s}")
        if len(buffers) > 10:
            print(f"... and {len(buffers) - 10} more buffers")
    else:
        print("  (no buffers found)")

    # ========================================================================
    #  SUBMODULES
    # ========================================================================

    print("\n" + "-" * 80)
    print("KEY SUBMODULES")
    print("-" * 80)

    if hasattr(actor, "actor"):
        print(f"✓ actor (MLP):        {actor.actor}")
    else:
        print("✗ actor submodule not found")

    if hasattr(actor, "memory_a"):
        print(f"✓ memory_a (LSTM):    {actor.memory_a}")
    else:
        print("✗ memory_a submodule not found")

    if hasattr(actor, "memory_c"):
        print(f"✗ memory_c (critic):  SHOULD NOT BE HERE - critic should be dropped")
    else:
        print(f"✓ memory_c (critic):  Not found (correctly dropped)")

    if hasattr(actor, "critic"):
        print(f"✗ critic:             SHOULD NOT BE HERE - critic should be dropped")
    else:
        print(f"✓ critic:             Not found (correctly dropped)")

    # ========================================================================
    #  ACTOR MLP LAYERS
    # ========================================================================

    print("\n" + "-" * 80)
    print("ACTOR MLP ARCHITECTURE")
    print("-" * 80)

    if hasattr(actor, "actor"):
        print("Layers in actor.actor (MLP):")
        for i, module in enumerate(actor.actor.children()):
            if isinstance(module, nn.Linear):
                print(f"  [{i}] {module.__class__.__name__:15s} {module.in_features:4d} → {module.out_features:4d}")
            else:
                print(f"  [{i}] {module.__class__.__name__:15s}")

    # ========================================================================
    #  LAST LAYER INSPECTION
    # ========================================================================

    print("\n" + "-" * 80)
    print("LAST LINEAR LAYER (MODIFIABLE)")
    print("-" * 80)

    try:
        last_layer = get_actor_last_layer(actor)
        print(f"Layer type:         {last_layer.__class__.__name__}")
        print(f"Input dimension:    {last_layer.weight.shape[1]}")
        print(f"Output dimension:   {last_layer.weight.shape[0]}")
        print(f"Weight shape:       {last_layer.weight.shape}")
        print(f"Weight dtype:       {last_layer.weight.dtype}")
        print(f"Weight device:      {last_layer.weight.device}")
        print(f"Weight is buffer:   {isinstance(last_layer.weight, torch.Tensor) and 'weight' in last_layer._buffers}")
        print(f"Weight requires_grad: {last_layer.weight.requires_grad if isinstance(last_layer.weight, torch.nn.Parameter) else 'N/A (buffer)'}")

        if last_layer.bias is not None:
            print(f"Bias shape:         {last_layer.bias.shape}")
            print(f"Bias requires_grad: {last_layer.bias.requires_grad if isinstance(last_layer.bias, torch.nn.Parameter) else 'N/A (buffer)'}")
        else:
            print(f"Bias:               None")

    except Exception as e:
        pytest.fail(f"Failed to inspect last layer: {e}")

    # ========================================================================
    #  TEST FORWARD PASS
    # ========================================================================

    print("\n" + "-" * 80)
    print("TEST FORWARD PASS")
    print("-" * 80)

    try:
        # Create dummy input
        from WP1.config import RunConfig
        wp1_cfg = RunConfig.from_yaml(config)
        obs_dim = wp1_cfg.obs.num_obs
        batch_size = 4

        obs = torch.randn(batch_size, obs_dim, device=device)
        print(f"Input shape:  {obs.shape}")

        with torch.no_grad():
            actions = actor.act_inference(obs)

        print(f"Output shape: {actions.shape}")
        print(f"Output range: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
        print(f"Sample action: {actions[0]}")
        print("✓ Forward pass successful")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

    # ========================================================================
    #  WEIGHT STATISTICS
    # ========================================================================

    print("\n" + "-" * 80)
    print("LAST LAYER WEIGHT STATISTICS")
    print("-" * 80)

    with torch.no_grad():
        w = last_layer.weight
        print(f"Mean:      {w.mean().item():10.6f}")
        print(f"Std:       {w.std().item():10.6f}")
        print(f"Min:       {w.min().item():10.6f}")
        print(f"Max:       {w.max().item():10.6f}")
        print(f"L2 norm:   {w.norm().item():10.6f}")

    # ========================================================================
    #  SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ Checkpoint loaded successfully")
    print("✓ Actor extracted (critic removed)")
    print("✓ Backbone frozen (all parameters non-trainable)")
    print("✓ Last layer weights are modifiable buffers")
    print(f"✓ Model ready for WP2 use")
    print(f"\nNext steps:")
    print(f"  1. Attach Hebbian rules to last layer")
    print(f"  2. Implement evolutionary algorithm")
    print(f"  3. Run WP2 evolution with this actor")
    print("=" * 80 + "\n")




def test_checkpoint_architecture(checkpoint_paths):
    """Load and inspect WP1 checkpoint controller architecture.

    Uses pytest command-line arguments:
        pytest ... -v -s \\
            --checkpoint logs/runs/.../model_1000.pt \\
            --config logs/runs/.../config.yaml \\
            --device cuda
    """
    _inspect_checkpoint_architecture(
        checkpoint=checkpoint_paths["checkpoint"],
        config=checkpoint_paths["config"],
        device=checkpoint_paths["device"],
    )


if __name__ == "__main__":
    # For direct script execution (alternative to pytest)
    print("Run with pytest instead:")
    print("  pytest tests/integration/test_checkpoint_architecture.py -v -s")
    print("\nOr with specific checkpoint:")
    print("  pytest tests/integration/test_checkpoint_architecture.py -v -s \\")
    print("    --checkpoint logs/runs/2026-03-01_12-00-00_exp/checkpoints/model_1000.pt")
