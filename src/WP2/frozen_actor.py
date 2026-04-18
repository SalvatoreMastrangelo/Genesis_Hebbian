"""
Frozen actor loading + IsolatedPopulationActor for WP2 evaluation.
==================================================================

We load the pretrained WP1 actor once, freeze it, and wrap it in an
``IsolatedPopulationActor`` that maintains:

1. ONE shared frozen backbone (LSTM + MLP layers 1..N-1 + last-layer bias).
2. ``K*S`` independent LSTM hidden states, one per environment slot.
3. ``K*S`` independent last-layer weight matrices (managed by
   ``HebbianLastLayer``), with ABCD rules replicated from K individuals.

No mutable state is shared between environment slots — the fitness seen by
the outer evolutionary loop is therefore a clean function of each individual's
own genome.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
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

    builtins.ActorCriticTanh = ActorCriticTanh

    wp1_cfg = RunConfig.from_yaml(wp1_cfg_path)

    train_cfg = wp1_cfg.to_train_cfg()
    policy_cfg = train_cfg["policy"]

    num_obs = wp1_cfg.obs.num_obs
    num_actions = wp1_cfg.env.num_actions
    num_critic_obs = num_obs

    actor_rnn_hidden = policy_cfg.get("rnn_hidden_size", 128)
    critic_rnn_hidden = policy_cfg.get("critic_rnn_hidden_size", None)

    if state_dict is not None:
        if "actor.4.weight" in state_dict:
            num_actions = state_dict["actor.4.weight"].shape[0]
        if "memory_a.rnn.weight_ih_l0" in state_dict:
            actor_rnn_hidden = state_dict["memory_a.rnn.weight_ih_l0"].shape[0] // 4
        if "memory_c.rnn.weight_ih_l0" in state_dict:
            num_critic_obs = state_dict["memory_c.rnn.weight_ih_l0"].shape[1]
            critic_rnn_hidden = state_dict["memory_c.rnn.weight_ih_l0"].shape[0] // 4

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
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
) -> Tuple[nn.Module, nn.Linear, int, int]:
    """Load a WP1 checkpoint, freeze all weights, and return the actor.

    Returns
    -------
    model : ActorCriticTanh
        The full model with all parameters frozen.
    last_layer : nn.Linear
        Reference to the actor's last linear layer (hidden_dim -> num_actions).
    num_actions : int
    hidden_dim : int
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("actor") for k in ckpt):
        state_dict = ckpt
    else:
        state_dict = ckpt

    model, _wp1_cfg = _build_actor_critic(wp1_cfg_path, device, state_dict=state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    if hasattr(model, "memory_a"):
        model.memory_a.rnn.flatten_parameters()

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    last_layer = None
    for module in reversed(list(model.actor.modules())):
        if isinstance(module, nn.Linear):
            last_layer = module
            break

    if last_layer is None:
        raise RuntimeError("Could not find last Linear layer in actor network")

    # Convert last-layer weight from Parameter to buffer so in-place ops are cheap.
    weight_data = last_layer.weight.data.clone()
    del last_layer.weight
    last_layer.register_buffer("weight", weight_data)

    num_actions = last_layer.weight.shape[0]
    hidden_dim = last_layer.weight.shape[1]

    return model, last_layer, num_actions, hidden_dim


def _load_obs_normalizer(ckpt_dict: dict, num_obs: int, device: str) -> nn.Module:
    """Load the empirical observation normalizer from a checkpoint dict.

    Returns an EmpiricalNormalization in eval mode if ``obs_norm_state_dict``
    is present in the checkpoint, otherwise ``nn.Identity``.
    """
    if not isinstance(ckpt_dict, dict) or "obs_norm_state_dict" not in ckpt_dict:
        return nn.Identity().to(device)
    try:
        from rsl_rl.modules import EmpiricalNormalization  # type: ignore
        norm = EmpiricalNormalization(shape=[num_obs], until=1.0e8)
        norm.load_state_dict(ckpt_dict["obs_norm_state_dict"])
        norm.eval()
        norm.to(device)
        return norm
    except Exception as exc:
        print(f"[frozen_actor] obs_normalizer load failed ({exc}); using identity")
        return nn.Identity().to(device)


def attach_hebbian(
    last_layer: nn.Linear,
    hebbian_rules: Dict[str, Tensor],
    cfg: HebbianEvolutionConfig,
    device: str = "cpu",
    num_envs: int = 1,
) -> HebbianLastLayer:
    """Create a ``HebbianLastLayer`` wrapping the frozen last layer's weights.

    The returned controller owns its own ``W`` tensor — the frozen
    ``last_layer.weight`` buffer is used only as the checkpoint reference.
    """
    W_checkpoint = last_layer.weight.data
    return HebbianLastLayer(
        W_checkpoint=W_checkpoint,
        hebbian_rules=hebbian_rules,
        eta=cfg.hebbian.eta,
        w_max=cfg.hebbian.w_max,
        use_oja_coefficient=cfg.hebbian.use_oja_coefficient,
        device=device,
        num_envs=num_envs,
    )


# ============================================================================
#  IsolatedPopulationActor — K*S isolated environments, one shared backbone
# ============================================================================

class IsolatedPopulationActor:
    """K individuals × S envs with fully-isolated per-env state.

    The frozen backbone (LSTM + MLP layers 1..N-1 + last-layer bias) is
    shared across all ``K*S`` environment slots because it is never
    mutated.  Each slot owns its own LSTM hidden/cell state and its own
    last-layer weight matrix; nothing crosses the boundary between slots.

    Parameters
    ----------
    model : nn.Module
        Frozen ActorCriticTanh (must be a deep copy — not shared with any
        other actor).
    hebbian : HebbianLastLayer
        Per-env Hebbian controller with ``num_envs == K * S``.
    K : int
        Number of individuals.
    S : int
        Environments per individual (``K * S`` total slots).
    stochastic : bool
        If True, sample from ``N(y, std)``; otherwise use the mean.
    """

    def __init__(
        self,
        model: nn.Module,
        hebbian: HebbianLastLayer,
        K: int,
        S: int,
        stochastic: bool = True,
        obs_normalizer: Optional[nn.Module] = None,
    ) -> None:
        if hebbian.num_envs != K * S:
            raise ValueError(
                f"hebbian.num_envs ({hebbian.num_envs}) must equal K*S ({K * S})"
            )
        if not hasattr(model, "memory_a") or not getattr(model, "recurrency", False):
            raise ValueError("IsolatedPopulationActor requires a recurrent model")

        self.model = model
        self.hebbian = hebbian
        self.K = K
        self.S = S
        self.stochastic = stochastic
        # Observation normalizer loaded from the WP1 checkpoint (empirical normalization).
        # If the checkpoint was trained without normalization this is nn.Identity.
        self.obs_normalizer: nn.Module = obs_normalizer if obs_normalizer is not None else nn.Identity()

        rnn = model.memory_a.rnn if hasattr(model.memory_a, "rnn") else model.memory_a
        if not isinstance(rnn, nn.LSTM):
            raise ValueError("IsolatedPopulationActor only supports LSTM backbones")
        self._rnn = rnn

        self._hidden_size = rnn.hidden_size
        self._num_layers = rnn.num_layers

        # Cache references to actor layers.
        # NOTE: `model.actor.children()` deduplicates by module identity, which
        # drops repeated activation instances (rsl_rl reuses a single ELU across
        # positions). We must iterate `_modules.values()` to preserve the full
        # forward order of nn.Sequential.
        self._actor_layers = list(model.actor._modules.values())
        self._last_layer_bias = self._actor_layers[-1].bias

        # Owned LSTM states: (num_layers, K*S, hidden) — NOT shared with model.memory_a
        self._h: Tensor | None = None
        self._c: Tensor | None = None

    def reset_episode(self, device: str | torch.device = "cpu") -> None:
        """Zero the LSTM states for ALL slots and reset per-env weights to checkpoint."""
        dev = torch.device(device)
        N = self.K * self.S
        self._h = torch.zeros(
            self._num_layers, N, self._hidden_size, device=dev, dtype=torch.float32
        )
        self._c = torch.zeros(
            self._num_layers, N, self._hidden_size, device=dev, dtype=torch.float32
        )
        self.hebbian.reset_weights()

    def reset_individual(self, k: int, device: str | torch.device = "cpu") -> None:
        """Zero the LSTM states for individual ``k``'s S slots and reset their weights."""
        if self._h is None or self._c is None:
            raise RuntimeError("reset_individual called before reset_episode")
        start, end = k * self.S, (k + 1) * self.S
        self._h[:, start:end, :].zero_()
        self._c[:, start:end, :].zero_()
        self.hebbian.reset_weights_individual(k, self.S)

    @torch.no_grad()
    def act(self, obs: Tensor) -> Tensor:
        """Forward pass with per-env LSTM state and per-env last-layer weights.

        Parameters
        ----------
        obs : Tensor
            Observations ``(K*S, obs_dim)``.

        Returns
        -------
        actions : Tensor
            Scaled actions ``(K*S, num_actions)``.
        """
        if self._h is None or self._c is None:
            raise RuntimeError("act called before reset_episode")

        # --- 0. Normalize observations (matches WP1 inference pipeline) ---
        obs = self.obs_normalizer(obs)

        # --- 1. LSTM with our owned hidden states (bypass memory_a.hidden_states) ---
        # obs: (N, obs_dim) → (1, N, obs_dim) for (seq_len, batch, input) layout
        rnn_in = obs.unsqueeze(0)
        rnn_out, (h_new, c_new) = self._rnn(rnn_in, (self._h, self._c))
        self._h, self._c = h_new, c_new
        inp = rnn_out.squeeze(0)  # (N, lstm_out)

        # --- 2. Shared frozen MLP backbone (all layers except last Linear) ---
        x = inp
        for layer in self._actor_layers[:-1]:
            x = layer(x)
        # x: (N, hidden_dim)

        # --- 3. Per-env last layer using hebbian.W ---
        W = self.hebbian.W  # (N, out, in)
        y = torch.einsum("ni,noi->no", x, W)
        if self._last_layer_bias is not None:
            y = y + self._last_layer_bias.unsqueeze(0)

        # --- 4. Per-env Hebbian update (updates hebbian.W in-place) ---
        self.hebbian.hebbian_update(x, y)

        # --- 5. Stochastic or deterministic sampling ---
        if self.stochastic:
            std = self._get_std()
            if std is not None:
                action_raw = Normal(y, std).rsample()
            else:
                action_raw = y
        else:
            action_raw = y

        # --- 6. Tanh + scale (reproducing training pipeline) ---
        a = torch.tanh(action_raw)
        return self.model._scale(a)

    def _get_std(self) -> Tensor | None:
        if hasattr(self.model, "std"):
            return self.model.std.detach()
        if hasattr(self.model, "log_std"):
            return torch.exp(self.model.log_std.detach())
        return None


def build_isolated_population_actor(
    checkpoint_path: str | Path,
    wp1_cfg_path: str | Path,
    hebbian_rules_per_individual: Sequence[Dict[str, Tensor]],
    cfg: HebbianEvolutionConfig,
    K: int,
    S: int,
    device: str = "cpu",
    stochastic: bool = True,
) -> IsolatedPopulationActor:
    """Load the checkpoint, deep-copy the model, and build an isolated actor.

    ``hebbian_rules_per_individual`` is a list of ``K`` rule dicts (each with
    keys ``A, B, C, D, lam`` and optionally ``eta``, each of shape ``(out, in)``).
    Rules are replicated ``S`` times per individual so every environment slot
    gets the rules of its owning individual.
    """
    if len(hebbian_rules_per_individual) != K:
        raise ValueError(
            f"Expected {K} rule dicts, got {len(hebbian_rules_per_individual)}"
        )

    # Load checkpoint once; extract obs_normalizer before the model is built.
    ckpt_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model, last_layer, _num_actions, _hidden_dim = load_frozen_actor(
        checkpoint_path, wp1_cfg_path, device=device
    )

    # Load observation normalizer (matches WP1 get_inference_policy).
    num_obs = model.memory_a.rnn.input_size
    obs_normalizer = _load_obs_normalizer(ckpt_dict, num_obs, device)

    # Deep-copy to fully isolate this actor from the cached/template model
    model = copy.deepcopy(model)
    # Re-find the matching last layer in the deep-copied model so the
    # W_checkpoint reference is to the copy's buffer, not the template's.
    last_layer = None
    for module in reversed(list(model.actor.modules())):
        if isinstance(module, nn.Linear):
            last_layer = module
            break
    if last_layer is None:
        raise RuntimeError("Could not find last Linear layer after deepcopy")

    # Pre-expand K rule sets to K*S by replicating each set S times
    rule_keys = set()
    for r in hebbian_rules_per_individual:
        rule_keys.update(r.keys())

    expanded: Dict[str, Tensor] = {}
    for key in rule_keys:
        per_env: List[Tensor] = []
        for rules in hebbian_rules_per_individual:
            t = rules[key].to(device)
            for _ in range(S):
                per_env.append(t)
        expanded[key] = torch.stack(per_env, dim=0)  # (K*S, out, in)

    hebbian = attach_hebbian(last_layer, expanded, cfg, device=device, num_envs=K * S)
    return IsolatedPopulationActor(
        model, hebbian, K=K, S=S, stochastic=stochastic, obs_normalizer=obs_normalizer
    )
