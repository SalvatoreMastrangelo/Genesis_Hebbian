"""
Load WP1 actor checkpoint for WP2 — discard critic, make actor modifiable.
===========================================================================

This module handles loading a trained WP1 policy checkpoint and extracting
just the actor component (frozen backbone with modifiable output layer) for
use in WP2 evolution.

The typical workflow is:
    actor = load_wp1_actor(checkpoint_path, wp1_cfg_path, device="cuda")
    # actor is a nn.Module with:
    #  - frozen LSTM backbone + MLP layers
    #  - last Linear layer weights are modifiable (registered as buffer, not parameter)
    #  - critic fully discarded
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def load_wp1_actor(
    checkpoint_path: str | Path,
    wp1_cfg_path: str | Path,
    device: str = "cpu",
) -> nn.Module:
    """Load a WP1 checkpoint and extract the frozen actor without critic.

    The returned actor has:
    - All LSTM + MLP backbone parameters frozen (requires_grad=False)
    - Last Linear layer weights converted to buffer (modifiable in-place)
    - Critic completely removed from model

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the WP1 model checkpoint (.pt file).
    wp1_cfg_path : str or Path
        Path to the WP1 config file (YAML) used during training.
    device : str
        Device to load model onto ('cpu', 'cuda', etc.). Default: 'cpu'.

    Returns
    -------
    actor : nn.Module
        Frozen ActorCriticTanh model with:
        - actor.weight (buffer, not parameter) on last layer
        - all parameters frozen except last-layer weights can be modified in-place

    Raises
    ------
    FileNotFoundError
        If checkpoint_path or wp1_cfg_path do not exist.
    RuntimeError
        If checkpoint format is unrecognized or last layer cannot be found.
    """
    checkpoint_path = Path(checkpoint_path)
    wp1_cfg_path = Path(wp1_cfg_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not wp1_cfg_path.exists():
        raise FileNotFoundError(f"WP1 config not found: {wp1_cfg_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle multiple checkpoint formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("actor") for k in ckpt):
        state_dict = ckpt
    else:
        state_dict = ckpt

    # Build the model from WP1 config
    model = _build_model_from_config(wp1_cfg_path, device, state_dict)

    # Load the checkpoint weights
    model.load_state_dict(state_dict, strict=False)

    # Remove critic modules to reduce model size and avoid confusion
    _remove_critic_modules(model)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Convert the last linear layer's weight to a buffer so it can be
    # modified in-place without triggering autograd bookkeeping
    _convert_last_layer_to_buffer(model)

    return model


def get_actor_last_layer(actor: nn.Module) -> nn.Linear:
    """Extract the last linear layer from the actor network.

    Parameters
    ----------
    actor : nn.Module
        The actor network (typically from load_wp1_actor).

    Returns
    -------
    last_layer : nn.Linear
        The final linear layer (hidden_dim -> num_actions).

    Raises
    ------
    RuntimeError
        If no linear layer is found in the actor.
    """
    if not hasattr(actor, "actor"):
        raise RuntimeError("Actor model does not have an 'actor' submodule")

    # Find the last linear layer in the actor MLP
    last_layer = None
    for module in reversed(list(actor.actor.modules())):
        if isinstance(module, nn.Linear):
            last_layer = module
            break

    if last_layer is None:
        raise RuntimeError("Could not find last Linear layer in actor network")

    return last_layer


def _build_model_from_config(
    wp1_cfg_path: str | Path,
    device: str,
    state_dict: dict | None = None,
) -> nn.Module:
    """Build ActorCriticTanh model from WP1 config and inferred dimensions.

    Parameters
    ----------
    wp1_cfg_path : str or Path
        Path to WP1 config YAML.
    device : str
        Target device.
    state_dict : dict, optional
        Checkpoint state dict. If provided, dimensions are inferred from it.

    Returns
    -------
    model : nn.Module
        Uninitialised ActorCriticTanh model.
    """
    from WP1.config import RunConfig
    import builtins

    # Import and register ActorCriticTanh for unpickling
    from winged_drone_train.rl.A2C_modified import ActorCriticTanh

    builtins.ActorCriticTanh = ActorCriticTanh

    # Load WP1 config
    wp1_cfg = RunConfig.from_yaml(wp1_cfg_path)
    train_cfg = wp1_cfg.to_train_cfg()
    policy_cfg = train_cfg["policy"]

    # Extract dimensions
    num_obs = wp1_cfg.obs.num_obs
    num_actions = wp1_cfg.env.num_actions
    num_critic_obs = num_obs

    actor_rnn_hidden = policy_cfg.get("rnn_hidden_size", 128)
    critic_rnn_hidden = policy_cfg.get("critic_rnn_hidden_size", None)

    # Override with checkpoint dimensions if available
    if state_dict is not None:
        if "actor.4.weight" in state_dict:
            num_actions = state_dict["actor.4.weight"].shape[0]
        if "memory_a.rnn.weight_ih_l0" in state_dict:
            actor_rnn_hidden = state_dict["memory_a.rnn.weight_ih_l0"].shape[0] // 4
        if "memory_c.rnn.weight_ih_l0" in state_dict:
            num_critic_obs = state_dict["memory_c.rnn.weight_ih_l0"].shape[1]
            critic_rnn_hidden = state_dict["memory_c.rnn.weight_ih_l0"].shape[0] // 4

    # Build model
    model = ActorCriticTanh(
        num_actor_obs=num_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=policy_cfg["actor_hidden_dims"],
        critic_hidden_dims=policy_cfg["critic_hidden_dims"],
        activation=policy_cfg["activation"],
        rnn_type=policy_cfg.get("rnn_type", "lstm"),
        rnn_hidden_size=actor_rnn_hidden,
        critic_rnn_hidden_size=critic_rnn_hidden,
        rnn_num_layers=policy_cfg.get("rnn_num_layers", 1),
        init_noise_std=policy_cfg.get("init_noise_std", 0.3),
        max_servo=policy_cfg.get("max_servo", 1.0),
        max_throttle=policy_cfg.get("max_throttle", 1.0),
    )

    return model.to(device)


def _remove_critic_modules(model: nn.Module) -> None:
    """Remove critic-related modules from the model.

    This reduces model size and memory usage by deleting submodules that
    are not needed for inference in WP2.

    Parameters
    ----------
    model : nn.Module
        The model to clean up.
    """
    # Remove critic memory (LSTM backbone)
    if hasattr(model, "memory_c"):
        delattr(model, "memory_c")

    # Remove critic MLP
    if hasattr(model, "critic"):
        delattr(model, "critic")


def _convert_last_layer_to_buffer(model: nn.Module) -> None:
    """Convert the actor's last layer weight from Parameter to buffer.

    This allows in-place weight modifications (e.g., by Hebbian rules) without
    triggering autograd bookkeeping or requiring requires_grad=True.

    Parameters
    ----------
    model : nn.Module
        The actor model.

    Raises
    ------
    RuntimeError
        If the last linear layer cannot be found.
    """
    last_layer = get_actor_last_layer(model)

    # Store original weight
    weight_data = last_layer.weight.data.clone()

    # Remove the Parameter
    del last_layer.weight

    # Register as buffer (persistent tensor, not a parameter)
    last_layer.register_buffer("weight", weight_data)


def get_actor_dimensions(actor: nn.Module) -> Tuple[int, int]:
    """Get the last layer's input and output dimensions.

    Parameters
    ----------
    actor : nn.Module
        The actor network.

    Returns
    -------
    hidden_dim : int
        Input dimension of the last linear layer.
    num_actions : int
        Output dimension of the last linear layer.
    """
    last_layer = get_actor_last_layer(actor)
    num_actions = last_layer.weight.shape[0]
    hidden_dim = last_layer.weight.shape[1]
    return hidden_dim, num_actions
