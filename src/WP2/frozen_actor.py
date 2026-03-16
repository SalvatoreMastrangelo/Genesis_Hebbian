"""
Frozen actor loading — load WP1 checkpoint, drop critic, freeze, attach Hebbian.
================================================================================

The pretrained actor's forward pass is intercepted so that:

1. All parameters are frozen (no gradients).
2. The critic is dropped entirely.
3. The last linear layer's weights are managed by ``HebbianLastLayer``.
4. The forward pass exposes pre-activation features ``x`` and raw outputs ``y``
   for the Hebbian update.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from WP2.config import HebbianEvolutionConfig
from WP2.hebbian import HebbianLastLayer


# ============================================================================
#  Actor loading
# ============================================================================

def _build_actor_critic(wp1_cfg_path: str | Path, device: str = "cpu", state_dict=None):
    """Instantiate an ActorCriticTanh with the same architecture as WP1.

    Reads the WP1 config to get hidden dims, RNN params, etc.
    Returns an uninitialised model (weights will be loaded from checkpoint).
    """
    from WP1.config import RunConfig
    import builtins
    from winged_drone_train.rl.A2C_modified import ActorCriticTanh

    # Ensure ActorCriticTanh is available for unpickling
    builtins.ActorCriticTanh = ActorCriticTanh

    wp1_cfg = RunConfig.from_yaml(wp1_cfg_path)

    train_cfg = wp1_cfg.to_train_cfg()
    policy_cfg = train_cfg["policy"]

    # The recurrent actor needs: num_actor_obs, num_critic_obs, num_actions
    # In the recurrent variant, actor MLP input = rnn_hidden_dim (NOT num_obs)
    # but the constructor takes num_actor_obs = actual obs dim for the RNN input
    num_obs = wp1_cfg.obs.num_obs
    num_actions = wp1_cfg.env.num_actions
    num_critic_obs = num_obs

    # Infer dims from checkpoint if available (config may be stale)
    if state_dict is not None:
        if "actor.4.weight" in state_dict:
            num_actions = state_dict["actor.4.weight"].shape[0]
        if "memory_c.rnn.weight_ih_l0" in state_dict:
            num_critic_obs = state_dict["memory_c.rnn.weight_ih_l0"].shape[1]

    model = ActorCriticTanh(
        num_actor_obs=num_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=policy_cfg["actor_hidden_dims"],
        critic_hidden_dims=policy_cfg["critic_hidden_dims"],
        activation=policy_cfg["activation"],
        rnn_type=policy_cfg.get("rnn_type", "lstm"),
        rnn_hidden_size=policy_cfg.get("rnn_hidden_size", 128),
        rnn_num_layers=policy_cfg.get("rnn_num_layers", 1),
        init_noise_std=policy_cfg.get("init_noise_std", 0.3),
        max_servo=policy_cfg.get("max_servo", 1.0),
        max_throttle=policy_cfg.get("max_throttle", 1.0),
    )
    return model, wp1_cfg


def load_frozen_actor(
    checkpoint_path: str | Path,
    wp1_cfg_path: str | Path,
    device: str = "cpu",
) -> Tuple[nn.Module, nn.Linear]:
    """Load a WP1 checkpoint, freeze all weights, and return the actor.

    Returns
    -------
    model : ActorCriticTanh
        The full model with all parameters frozen.
    last_layer : nn.Linear
        Reference to the actor's last linear layer (64 -> 5).
    """
    # Load checkpoint first to infer architecture
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Handle both direct state_dict and wrapped formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("actor") for k in ckpt):
        state_dict = ckpt
    else:
        state_dict = ckpt

    model, wp1_cfg = _build_actor_critic(wp1_cfg_path, device, state_dict=state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Identify the last linear layer in the actor
    # actor is nn.Sequential: [Linear, act, Linear, act, Linear(hidden->actions)]
    last_layer = None
    for module in reversed(list(model.actor.modules())):
        if isinstance(module, nn.Linear):
            last_layer = module
            break

    if last_layer is None:
        raise RuntimeError("Could not find last Linear layer in actor network")

    # Convert last layer weight from Parameter to buffer so we can modify in-place
    # without triggering autograd bookkeeping
    weight_data = last_layer.weight.data.clone()
    del last_layer.weight
    last_layer.register_buffer("weight", weight_data)

    # Infer actual last-layer dimensions from the loaded weights
    num_actions = last_layer.weight.shape[0]
    hidden_dim = last_layer.weight.shape[1]

    return model, last_layer, num_actions, hidden_dim


def attach_hebbian(
    last_layer: nn.Linear,
    hebbian_rules: Dict[str, Tensor],
    cfg: HebbianEvolutionConfig,
    device: str = "cpu",
) -> HebbianLastLayer:
    """Create a HebbianLastLayer wrapper for the actor's last layer.

    Parameters
    ----------
    last_layer : nn.Linear
        The actor's last linear layer (already frozen, weight is a buffer).
    hebbian_rules : dict
        Per-weight ABCD + lambda tensors from ``decode_hebbian_genes``.
    cfg : HebbianEvolutionConfig
        Config for eta and w_max.
    device : str
        Target device.

    Returns
    -------
    HebbianLastLayer
        Controller that will modify the layer's weights in-place.
    """
    eta = cfg.hebbian.eta  # scalar global eta (default)
    return HebbianLastLayer(
        linear_layer=last_layer,
        hebbian_rules=hebbian_rules,
        eta=eta,
        w_max=cfg.hebbian.w_max,
        device=device,
    )


# ============================================================================
#  Forward pass with Hebbian interception
# ============================================================================

class HebbianActorWrapper:
    """Wraps a frozen ActorCriticTanh + HebbianLastLayer for rollout.

    Performs the forward pass and intercepts pre/post activations of the
    last layer for the Hebbian update.

    Parameters
    ----------
    model : nn.Module
        Frozen ActorCriticTanh.
    hebbian : HebbianLastLayer
        Hebbian controller for the last layer.
    stochastic : bool
        If True, sample from the policy distribution.  If False, use mean.
    """

    def __init__(
        self,
        model: nn.Module,
        hebbian: HebbianLastLayer,
        stochastic: bool = True,
    ) -> None:
        self.model = model
        self.hebbian = hebbian
        self.stochastic = stochastic

    def reset_episode(self, num_envs: int, device: str | torch.device = "cpu") -> None:
        """Reset LSTM hidden states and Hebbian weights for a new episode."""
        self.hebbian.reset_weights()
        # Reset LSTM internal hidden states (inference mode: masks=None)
        if hasattr(self.model, "memory_a"):
            self.model.memory_a.reset()

    @torch.no_grad()
    def act(self, obs: Tensor) -> Tensor:
        """Forward pass with Hebbian update on the last layer.

        Steps:
        1. obs -> LSTM -> MLP backbone -> x (hidden features)
        2. x -> last_layer -> y (raw actions, before tanh)
        3. hebbian_update(x, y)
        4. action = tanh(y) -> scale

        Returns
        -------
        actions : Tensor
            Scaled physical actions (throttle + servos).
        """
        model = self.model

        # --- Step 1: Run through LSTM backbone (inference mode: masks=None) ---
        if hasattr(model, "memory_a") and model.recurrency:
            inp = model.memory_a(obs)  # uses internal hidden states
        else:
            inp = obs

        # --- Step 2: Run through actor MLP backbone (all layers except last) ---
        # actor is nn.Sequential: [Linear, act, Linear, act, Linear(last)]
        actor_layers = list(model.actor.children())
        x = inp.squeeze(0) if inp.dim() == 3 else inp

        # Forward through all layers except the last Linear
        for layer in actor_layers[:-1]:
            x = layer(x)

        # x is now the presynaptic activation (batch, 64)
        last_layer = actor_layers[-1]
        y = x @ last_layer.weight.t()  # (batch, 5) raw output
        if last_layer.bias is not None:
            y = y + last_layer.bias

        # --- Step 3: Hebbian update ---
        self.hebbian.hebbian_update(x, y)

        # --- Step 4: Tanh scaling ---
        # update_distribution runs the full actor MLP, so pass LSTM output (inp)
        mlp_inp = inp.squeeze(0) if inp.dim() == 3 else inp
        model.update_distribution(mlp_inp)
        z = model.distribution.mean if not self.stochastic else model.distribution.rsample()
        a = torch.tanh(z)
        actions = model._scale(a)

        return actions

    @torch.no_grad()
    def act_simple(self, obs: Tensor) -> Tensor:
        """Simplified forward: use model.act() then do Hebbian update.

        This avoids duplicating the distribution logic by leveraging the
        model's own act() method, but still intercepts the last layer for
        Hebbian updates.
        """
        model = self.model
        device = obs.device
        num_envs = obs.shape[0]

        # Run the LSTM backbone (inference mode: masks=None)
        if hasattr(model, "memory_a") and model.recurrency:
            inp = model.memory_a(obs)  # uses internal hidden states
        else:
            inp = obs

        # Get pre-last-layer features
        actor_layers = list(model.actor.children())
        x = inp.squeeze(0) if inp.dim() == 3 else inp
        for layer in actor_layers[:-1]:
            x = layer(x)

        # Compute raw last-layer output
        last_layer = actor_layers[-1]
        y = x @ last_layer.weight.t()
        if last_layer.bias is not None:
            y = y + last_layer.bias

        # Hebbian update
        self.hebbian.hebbian_update(x, y)

        # Use model's distribution for action sampling
        # update_distribution runs the full actor MLP, so pass LSTM output (inp)
        mlp_inp = inp.squeeze(0) if inp.dim() == 3 else inp
        model.update_distribution(mlp_inp)
        if self.stochastic:
            z = model.distribution.rsample()
        else:
            z = model.distribution.mean
        a = torch.tanh(z)
        actions = model._scale(a)

        return actions


class BatchedHebbianActorWrapper:
    """Vectorized actor wrapper for population-level evaluation.

    Wraps a frozen ActorCriticTanh + BatchedHebbianLastLayer for parallel rollout
    of P individuals across P*S shared environments.

    All individuals share the frozen LSTM backbone and MLP layers, but each
    individual has its own last-layer weight matrix and Hebbian rules.

    Parameters
    ----------
    model : nn.Module
        Frozen ActorCriticTanh.
    batched_hebbian : BatchedHebbianLastLayer
        Batched Hebbian controller with per-individual weight matrices.
    stochastic : bool
        If True, sample actions. If False, use mean.
    """

    def __init__(
        self,
        model: nn.Module,
        batched_hebbian,
        stochastic: bool = True,
    ) -> None:
        self.model = model
        self.hebbian = batched_hebbian
        self.stochastic = stochastic

    def reset_episode(self, num_envs: int, device: str | torch.device = "cpu") -> None:
        """Reset Hebbian weights and LSTM hidden states."""
        self.hebbian.reset_weights()
        if hasattr(self.model, "memory_a"):
            self.model.memory_a.reset()

    @torch.no_grad()
    def act(self, obs: Tensor) -> Tensor:
        """Forward pass with batched per-individual Hebbian updates.

        Parameters
        ----------
        obs : Tensor
            Observations (P*S, obs_dim) where P is population size, S is slice size.

        Returns
        -------
        actions : Tensor
            Scaled actions (P*S, action_dim).
        """
        from WP2.hebbian import BatchedHebbianLastLayer

        if not isinstance(self.hebbian, BatchedHebbianLastLayer):
            raise TypeError(f"Expected BatchedHebbianLastLayer, got {type(self.hebbian)}")

        model = self.model
        P = self.hebbian.pop_size
        S = self.hebbian.slice_size
        device = obs.device

        # --- LSTM backbone ---
        if hasattr(model, "memory_a") and model.recurrency:
            inp = model.memory_a(obs)  # (P*S, lstm_out)
        else:
            inp = obs

        # --- MLP backbone except last layer ---
        x = inp.squeeze(0) if inp.dim() == 3 else inp
        actor_layers = list(model.actor.children())
        for layer in actor_layers[:-1]:
            x = layer(x)  # x: (P*S, 64)

        # --- Batched last layer forward ---
        # Reshape x into per-individual batches and apply per-individual weights via bmm
        x_batched = x.view(P, S, -1)       # (P, S, 64)
        W = self.hebbian.W                 # (P, out, in)
        last_layer = actor_layers[-1]
        out_features = last_layer.weight.shape[0]

        # y = x @ W^T: (P, S, 64) @ (P, 64, out) = (P, S, out)
        y_batched = torch.bmm(x_batched, W.transpose(-2, -1))  # (P, S, out)

        # Add bias if present (broadcast)
        if last_layer.bias is not None:
            y_batched = y_batched + last_layer.bias.unsqueeze(0).unsqueeze(0)

        # Flatten back to (P*S, out) for Hebbian update
        y = y_batched.view(P * S, out_features)

        # --- Hebbian update ---
        self.hebbian.hebbian_update(x, y)

        # --- Action distribution and sampling ---
        # Use model's distribution logic (frozen, so deterministic)
        mlp_inp = inp.squeeze(0) if inp.dim() == 3 else inp
        model.update_distribution(mlp_inp)

        if self.stochastic:
            z = model.distribution.rsample()
        else:
            z = model.distribution.mean

        a = torch.tanh(z)
        actions = model._scale(a)

        return actions
