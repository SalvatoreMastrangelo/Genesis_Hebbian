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
from torch.distributions import Normal

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

    actor_rnn_hidden  = policy_cfg.get("rnn_hidden_size", 128)
    critic_rnn_hidden = policy_cfg.get("critic_rnn_hidden_size", None)

    # Infer dims from checkpoint if available (config may be stale)
    if state_dict is not None:
        if "actor.4.weight" in state_dict:
            num_actions = state_dict["actor.4.weight"].shape[0]
        if "memory_a.rnn.weight_ih_l0" in state_dict:
            # LSTM weight_ih shape: (4*hidden, input) → hidden = shape[0] // 4
            actor_rnn_hidden = state_dict["memory_a.rnn.weight_ih_l0"].shape[0] // 4
        if "memory_c.rnn.weight_ih_l0" in state_dict:
            num_critic_obs    = state_dict["memory_c.rnn.weight_ih_l0"].shape[1]
            critic_rnn_hidden = state_dict["memory_c.rnn.weight_ih_l0"].shape[0] // 4

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
        Reference to the actor's last linear layer (hidden_dim -> num_actions).
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
    num_envs: int = 1,
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
    num_envs : int
        Number of environments. If > 1, maintains per-environment weight matrices.
        Default: 1 (backward compatible).

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
        use_oja_coefficient=cfg.hebbian.use_oja_coefficient,
        device=device,
        num_envs=num_envs,
    )


# ============================================================================
#  Forward pass with Hebbian interception
# ============================================================================

class HebbianActorWrapper:
    """Wraps a frozen ActorCriticTanh + HebbianLastLayer for rollout.

    Performs the forward pass and intercepts pre/post activations of the
    last layer for the Hebbian update.

    Now supports **per-environment** updates where each environment has its own
    weight matrix and LSTM hidden state.

    Parameters
    ----------
    model : nn.Module
        Frozen ActorCriticTanh.
    hebbian : HebbianLastLayer
        Hebbian controller for the last layer (may have per-env weights).
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

        # Per-environment LSTM hidden states (if needed)
        self._lstm_h_states = None  # (num_envs, num_layers, hidden_dim)
        self._lstm_c_states = None  # (num_envs, num_layers, hidden_dim)
        self._num_envs = hebbian.num_envs

    def reset_episode(self, num_envs: int, device: str | torch.device = "cpu") -> None:
        """Reset LSTM hidden states and Hebbian weights for a new episode.

        Parameters
        ----------
        num_envs : int
            Number of environments (should match hebbian.num_envs).
        device : str or torch.device
            Device for tensors.
        """
        self.hebbian.reset_weights()

        # Reset per-environment LSTM hidden states
        if hasattr(self.model, "memory_a") and self.model.recurrency:
            # Get RNN hidden size from the LSTM module
            if hasattr(self.model.memory_a, "rnn"):
                rnn = self.model.memory_a.rnn
            else:
                rnn = self.model.memory_a

            hidden_size = rnn.hidden_size if hasattr(rnn, "hidden_size") else rnn.hidden_dim
            num_layers = rnn.num_layers if hasattr(rnn, "num_layers") else 1

            # Initialize per-env LSTM states: (num_envs, num_layers, hidden_size)
            # Note: stored without batch dim; batch dim added when calling LSTM
            self._lstm_h_states = torch.zeros(
                num_envs, num_layers, hidden_size, device=device, dtype=torch.float32
            )
            self._lstm_c_states = torch.zeros(
                num_envs, num_layers, hidden_size, device=device, dtype=torch.float32
            )

    @torch.no_grad()
    def act(self, obs: Tensor) -> Tensor:
        """Forward pass with Hebbian update on the last layer.

        Per-environment forward:
        1. obs -> LSTM (per-env state) -> MLP backbone -> x (hidden features)
        2. x -> last_layer (per-env weights) -> y (raw actions, before tanh)
        3. hebbian_update(x, y) per-environment
        4. action = sample or mean from Normal(y, log_std)
        5. action = tanh(action) -> scale

        If num_envs > 1, processes each environment independently to maintain
        per-environment LSTM hidden states and weight matrices.

        Returns
        -------
        actions : Tensor
            Scaled physical actions (throttle + servos) of shape (num_envs, num_actions).
        """
        model = self.model
        num_envs = self.hebbian.num_envs

        if num_envs == 1:
            # Single environment: use batched forward (backward compatible)
            return self._act_single_env(obs)
        else:
            # Multiple environments: per-environment forward
            return self._act_multi_env(obs)

    def _act_single_env(self, obs: Tensor) -> Tensor:
        """Forward pass for single environment (original batched logic)."""
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
        y = x @ last_layer.weight.t()  # (batch, num_actions) raw output
        if last_layer.bias is not None:
            y = y + last_layer.bias

        # --- Step 3: Hebbian update ---
        self.hebbian.hebbian_update(x, y)

        # --- Step 4: Stochastic or deterministic action sampling ---
        if self.stochastic:
            if hasattr(model, "std"):
                std = model.std.detach()
            elif hasattr(model, "log_std"):
                log_std = model.log_std.detach()
                std = torch.exp(log_std)
            else:
                std = None

            if std is not None:
                dist = Normal(y, std)
                action_raw = dist.rsample()
            else:
                action_raw = y
        else:
            action_raw = y

        # --- Step 5: Tanh + scale (reproducing training pipeline) ---
        a = torch.tanh(action_raw)
        return self.model._scale(a)

    def _act_multi_env(self, obs: Tensor) -> Tensor:
        """Forward pass for multiple environments with per-environment LSTM states.

        Each environment maintains its own LSTM hidden state across timesteps.
        """
        model = self.model
        actor_layers = list(model.actor.children())
        num_envs = self.hebbian.num_envs
        device = obs.device

        # Collect per-environment results
        all_x = []
        all_y = []

        # Process each environment independently with its own LSTM hidden state
        for env_idx in range(num_envs):
            # Get this environment's observation
            if obs.shape[0] == num_envs:
                obs_i = obs[env_idx : env_idx + 1]  # (1, obs_dim)
            else:
                obs_i = obs[env_idx : env_idx + 1]

            # --- LSTM forward per-environment with stored hidden state ---
            if hasattr(model, "memory_a") and model.recurrency:
                # Get the underlying RNN module
                if hasattr(model.memory_a, "rnn"):
                    rnn = model.memory_a.rnn
                    obs_for_rnn = obs_i
                else:
                    rnn = model.memory_a
                    obs_for_rnn = obs_i

                # Retrieve stored hidden states for this environment
                h_i = self._lstm_h_states[env_idx]  # (num_layers, hidden_size)
                c_i = self._lstm_c_states[env_idx]  # (num_layers, hidden_size)

                # Forward through RNN with explicit hidden state
                if isinstance(rnn, nn.LSTM):
                    # obs_for_rnn shape: (1, obs_dim)
                    # LSTM expects input (seq_len, batch, input_size)
                    if obs_for_rnn.dim() == 2:
                        obs_for_rnn = obs_for_rnn.unsqueeze(0)  # (1, 1, obs_dim)

                    # Add batch dim to hidden states: (num_layers, hidden_size) -> (num_layers, 1, hidden_size)
                    h_i_batched = h_i.unsqueeze(1)
                    c_i_batched = c_i.unsqueeze(1)

                    rnn_out, (h_new, c_new) = rnn(obs_for_rnn, (h_i_batched, c_i_batched))

                    # Remove seq_len dimension, keep batch: (1, 1, hidden_size) -> (1, hidden_size)
                    inp_i = rnn_out.squeeze(0)

                    # Store hidden states without batch dim for next timestep
                    self._lstm_h_states[env_idx] = h_new.squeeze(1)  # (num_layers, 1, hidden_size) -> (num_layers, hidden_size)
                    self._lstm_c_states[env_idx] = c_new.squeeze(1)
                else:
                    # Fallback for other RNN types (GRU, etc.)
                    inp_i = rnn(obs_i)
            else:
                inp_i = obs_i

            # --- MLP backbone per-environment ---
            x_i = inp_i.squeeze(0) if inp_i.dim() == 3 else inp_i

            for layer in actor_layers[:-1]:
                x_i = layer(x_i)

            # --- Last layer per-environment using per-env weights ---
            last_layer = actor_layers[-1]
            if self.hebbian.num_envs > 1:
                # Use per-env weight matrix
                W_i = self.hebbian.W[env_idx]  # (num_actions, hidden_dim)
                y_i = x_i @ W_i.t()
            else:
                y_i = x_i @ last_layer.weight.t()

            if last_layer.bias is not None:
                y_i = y_i + last_layer.bias

            all_x.append(x_i)
            all_y.append(y_i)

        # --- Batched Hebbian update (per-environment, no averaging) ---
        x_batch = torch.cat(all_x, dim=0)  # (num_envs, hidden_dim)
        y_batch = torch.cat(all_y, dim=0)  # (num_envs, num_actions)
        self.hebbian.hebbian_update(x_batch, y_batch)

        # --- Stochastic or deterministic action sampling ---
        if self.stochastic:
            if hasattr(model, "std"):
                std = model.std.detach()
            elif hasattr(model, "log_std"):
                log_std = model.log_std.detach()
                std = torch.exp(log_std)
            else:
                std = None

            if std is not None:
                dist = Normal(y_batch, std)
                action_raw = dist.rsample()
            else:
                action_raw = y_batch
        else:
            action_raw = y_batch

        # --- Tanh + scale ---
        a = torch.tanh(action_raw)
        return self.model._scale(a)

    @torch.no_grad()
    def act_simple(self, obs: Tensor) -> Tensor:
        """Simplified forward with stochastic sampling support.

        Intercepts the last layer for Hebbian updates while handling
        both stochastic (sampled) and deterministic (mean) actions.
        """
        # Same as act() for now, since we handle per-env in act()
        return self.act(obs)


class BatchedHebbianActorWrapper:
    """Vectorized actor wrapper with per-environment updates.

    Wraps a frozen ActorCriticTanh + BatchedHebbianLastLayer for parallel rollout
    of P*N total environments (either as P individuals × N envs each, or N independent envs).

    Each environment has its own last-layer weight matrix and Hebbian rules.
    All environments share the frozen LSTM backbone and MLP layers.

    Parameters
    ----------
    model : nn.Module
        Frozen ActorCriticTanh.
    batched_hebbian : BatchedHebbianLastLayer
        Batched Hebbian controller with per-environment weight matrices.
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
        """Reset Hebbian weights and LSTM hidden states.

        Parameters
        ----------
        num_envs : int
            Number of environments (should match hebbian.num_envs).
        device : str or torch.device
            Device for tensors.
        """
        self.hebbian.reset_weights()
        if hasattr(self.model, "memory_a"):
            self.model.memory_a.reset()

    @torch.no_grad()
    def act(self, obs: Tensor) -> Tensor:
        """Forward pass with per-environment Hebbian updates.

        Each of N environments has its own weight matrix and receives its own
        Hebbian update based on individual activations (no averaging).

        Parameters
        ----------
        obs : Tensor
            Observations (N, obs_dim) where N is total number of environments.

        Returns
        -------
        actions : Tensor
            Scaled actions (N, action_dim).
        """
        from WP2.hebbian import BatchedHebbianLastLayer

        if not isinstance(self.hebbian, BatchedHebbianLastLayer):
            raise TypeError(f"Expected BatchedHebbianLastLayer, got {type(self.hebbian)}")

        model = self.model
        N = self.hebbian.num_envs
        device = obs.device

        # --- LSTM backbone ---
        if hasattr(model, "memory_a") and model.recurrency:
            inp = model.memory_a(obs)  # (N, lstm_out)
        else:
            inp = obs

        # --- MLP backbone except last layer ---
        x = inp.squeeze(0) if inp.dim() == 3 else inp  # (N, hidden_dim)
        actor_layers = list(model.actor.children())
        for layer in actor_layers[:-1]:
            x = layer(x)  # x: (N, 64)

        # --- Per-environment last layer forward ---
        # Apply per-environment weights: (N, hidden) @ (N, out, hidden)^T = (N, out)
        W = self.hebbian.W  # (N, out, in)
        last_layer = actor_layers[-1]
        out_features = last_layer.weight.shape[0]

        # Compute y per-environment using einsum
        # x: (N, hidden), W: (N, out, hidden) -> (N, out)
        y = torch.einsum("nj,nij->ni", x, W)  # (N, out)

        # Add bias if present (broadcast)
        if last_layer.bias is not None:
            y = y + last_layer.bias.unsqueeze(0)  # (1, out) broadcasts to (N, out)

        # --- Hebbian update: per-environment (no averaging) ---
        self.hebbian.hebbian_update(x, y)

        # --- Stochastic or deterministic action sampling ---
        if self.stochastic:
            if hasattr(model, "std"):
                std = model.std.detach()
            elif hasattr(model, "log_std"):
                log_std = model.log_std.detach()
                std = torch.exp(log_std)
            else:
                std = None

            if std is not None:
                dist = Normal(y, std)
                action_raw = dist.rsample()
            else:
                action_raw = y
        else:
            action_raw = y

        # Tanh + scale (reproducing training pipeline)
        a = torch.tanh(action_raw)
        return self.model._scale(a)
