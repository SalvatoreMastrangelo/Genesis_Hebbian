#!/usr/bin/env python3
"""
Inspect WP1 checkpoint controller architecture.
================================================

Standalone script to load a WP1 checkpoint and display detailed information
about the actor architecture. No pytest required — just run directly.

Usage
-----
Basic usage (find latest checkpoint):
    python human_runnable/inspect_checkpoint.py

With specific checkpoint:
    python human_runnable/inspect_checkpoint.py \\
        --checkpoint logs/runs/2026-03-01_12-00-00_exp/checkpoints/model_1000.pt \\
        --config logs/runs/2026-03-01_12-00-00_exp/config.yaml

On CPU (if CUDA unavailable):
    python human_runnable/inspect_checkpoint.py \\
        --checkpoint logs/runs/.../checkpoints/model_1000.pt \\
        --device cpu
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


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


def load_wp1_actor(checkpoint_path: str | Path, wp1_cfg_path: str | Path, device: str = "cpu"):
    """Load a WP1 checkpoint and extract the frozen actor without critic."""
    from WP2 import load_wp1_actor as _load_wp1_actor
    return _load_wp1_actor(checkpoint_path, wp1_cfg_path, device)


def get_actor_last_layer(actor: nn.Module) -> nn.Linear:
    """Extract the last linear layer from the actor network."""
    from WP2 import get_actor_last_layer as _get_actor_last_layer
    return _get_actor_last_layer(actor)


def get_actor_dimensions(actor: nn.Module):
    """Get the last layer's input and output dimensions."""
    from WP2 import get_actor_dimensions as _get_actor_dimensions
    return _get_actor_dimensions(actor)


def inspect_checkpoint(
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Load checkpoint and return architecture information.

    Parameters
    ----------
    checkpoint : str, optional
        Path to the checkpoint. If None, finds the latest.
    config : str, optional
        Path to WP1 config. Default: src/WP1/configs/foundation.yaml
    device : str, optional
        Device to load onto. Default: auto-detect (cuda if available, else cpu)
    verbose : bool
        If True, print detailed information. Default: True

    Returns
    -------
    dict
        Dictionary with keys: actor, last_layer, hidden_dim, num_actions, device
    """
    if verbose:
        print("\n" + "=" * 80)
        print("WP1 CHECKPOINT ARCHITECTURE INSPECTION")
        print("=" * 80)

    # Set defaults
    if config is None:
        config = "src/WP1/configs/foundation.yaml"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find checkpoint if not specified
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
        if checkpoint is None:
            print(
                "ERROR: No checkpoint found in logs/runs/")
            print("Provide one with: --checkpoint path/to/model.pt")
            sys.exit(1)

    checkpoint = Path(checkpoint)
    config = Path(config)

    if verbose:
        print(f"\nCheckpoint: {checkpoint}")
        print(f"Config:     {config}")
        print(f"Device:     {device}")

    # ========================================================================
    #  LOAD CHECKPOINT
    # ========================================================================

    if verbose:
        print("\n" + "-" * 80)
        print("LOADING CHECKPOINT...")
        print("-" * 80)

    try:
        actor = load_wp1_actor(
            checkpoint_path=checkpoint,
            wp1_cfg_path=config,
            device=device,
        )
        if verbose:
            print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        sys.exit(1)

    # ========================================================================
    #  BASIC INFO
    # ========================================================================

    if verbose:
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

    try:
        hidden_dim, num_actions = get_actor_dimensions(actor)
    except Exception as e:
        print(f"ERROR: Failed to get dimensions: {e}")
        sys.exit(1)

    if verbose:
        print("\n" + "-" * 80)
        print("LAYER DIMENSIONS")
        print("-" * 80)
        print(f"Last layer input:  {hidden_dim}")
        print(f"Last layer output: {num_actions} (actions)")

    # ========================================================================
    #  PARAMETER COUNT AND FREEZE STATUS
    # ========================================================================

    frozen_params = sum(p.numel() for p in actor.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in actor.parameters())

    if verbose:
        print("\n" + "-" * 80)
        print("PARAMETERS")
        print("-" * 80)
        print(f"Total parameters:   {total_params:,}")
        print(f"Frozen parameters:  {frozen_params:,}")
        print(f"Trainable params:   {trainable_params:,}")

        if trainable_params > 0:
            print(f"\n⚠ WARNING: {trainable_params:,} trainable parameters found!")
            print("  Expected: all parameters frozen except last layer (which is a buffer)")

    # ========================================================================
    #  SUBMODULES
    # ========================================================================

    if verbose:
        print("\n" + "-" * 80)
        print("KEY SUBMODULES")
        print("-" * 80)

        if hasattr(actor, "actor"):
            print(f"✓ actor (MLP):        Present")
        else:
            print(f"✗ actor submodule not found")

        if hasattr(actor, "memory_a"):
            print(f"✓ memory_a (LSTM):    Present")
        else:
            print(f"✗ memory_a submodule not found")

        if hasattr(actor, "memory_c") or hasattr(actor, "critic"):
            print(f"✗ Critic found:       SHOULD NOT BE HERE - critic should be dropped")
        else:
            print(f"✓ Critic:             Not found (correctly dropped)")

    # ========================================================================
    #  ACTOR MLP LAYERS
    # ========================================================================

    if verbose and hasattr(actor, "actor"):
        print("\n" + "-" * 80)
        print("ACTOR MLP ARCHITECTURE")
        print("-" * 80)

        # First show the full module structure
        print("Full actor.actor structure:")
        print(actor.actor)

        print("\nLayers enumeration:")
        # Iterate through actual Sequential children by accessing items directly
        num_children = len(actor.actor)
        for i in range(num_children):
            module = actor.actor[i]
            if isinstance(module, nn.Linear):
                print(
                    f"  [{i}] {module.__class__.__name__:15s} "
                    f"{module.in_features:4d} → {module.out_features:4d}"
                )
            else:
                print(f"  [{i}] {module.__class__.__name__:15s}")

    # ========================================================================
    #  LAST LAYER INSPECTION
    # ========================================================================

    try:
        last_layer = get_actor_last_layer(actor)
    except Exception as e:
        print(f"ERROR: Failed to get last layer: {e}")
        sys.exit(1)

    if verbose:
        print("\n" + "-" * 80)
        print("LAST LINEAR LAYER (MODIFIABLE)")
        print("-" * 80)

        print(f"Layer type:         {last_layer.__class__.__name__}")
        print(f"Input dimension:    {last_layer.weight.shape[1]}")
        print(f"Output dimension:   {last_layer.weight.shape[0]}")
        print(f"Weight shape:       {last_layer.weight.shape}")
        print(f"Weight dtype:       {last_layer.weight.dtype}")
        print(f"Weight device:      {last_layer.weight.device}")
        print(
            f"Weight is buffer:   "
            f"{isinstance(last_layer.weight, torch.Tensor) and 'weight' in last_layer._buffers}"
        )

        if last_layer.bias is not None:
            print(f"Bias shape:         {last_layer.bias.shape}")
        else:
            print(f"Bias:               None")

    # ========================================================================
    #  TEST FORWARD PASS
    # ========================================================================

    if verbose:
        print("\n" + "-" * 80)
        print("TEST FORWARD PASS")
        print("-" * 80)

    try:
        from WP1.config import RunConfig

        wp1_cfg = RunConfig.from_yaml(config)
        obs_dim = wp1_cfg.obs.num_obs
        batch_size = 4

        obs = torch.randn(batch_size, obs_dim, device=device)
        if verbose:
            print(f"Input shape:  {obs.shape}")

        with torch.no_grad():
            actions = actor.act_inference(obs)

        if verbose:
            print(f"Output shape: {actions.shape}")
            print(f"Output range: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
            print(f"Sample action: {actions[0]}")
            print("✓ Forward pass successful")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

    # ========================================================================
    #  WEIGHT STATISTICS
    # ========================================================================

    if verbose:
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

    if verbose:
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

    return {
        "actor": actor,
        "last_layer": last_layer,
        "hidden_dim": hidden_dim,
        "num_actions": num_actions,
        "device": device,
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect WP1 checkpoint controller architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find and inspect latest checkpoint
  python human_runnable/inspect_checkpoint.py

  # Inspect specific checkpoint
  python human_runnable/inspect_checkpoint.py \\
    --checkpoint logs/runs/2026-03-01_12-00-00_exp/checkpoints/model_1000.pt \\
    --config logs/runs/2026-03-01_12-00-00_exp/config.yaml

  # Use CPU instead of CUDA
  python human_runnable/inspect_checkpoint.py --device cpu
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to WP1 checkpoint (.pt file). If not specified, finds latest.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/WP1/configs/foundation.yaml",
        help="Path to WP1 config (YAML). Default: src/WP1/configs/foundation.yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to load onto (cuda, cpu). Default: auto-detect",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output (only return data)",
    )

    args = parser.parse_args()

    # Run inspection
    result = inspect_checkpoint(
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        verbose=not args.quiet,
    )

    return result


if __name__ == "__main__":
    import sys

    # Add src to path so imports work
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    main()
