"""
MultiDroneActorManager — D independent frozen actor + Hebbian instances.
========================================================================

Each drone entity gets its own copy of the frozen WP1 actor with independent
LSTM hidden states and Hebbian plasticity rules. This ensures zero
cross-contamination between drones during evaluation.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from torch import nn

from WP2.frozen_actor import load_frozen_actor, HebbianActorWrapper
from WP2.hebbian import HebbianLastLayer
from WP2.config import HebbianConfig


class BaselineActorWrapper:
    """Thin wrapper around a frozen WP1 actor (no Hebbian plasticity).

    Exposes the same ``act`` / ``reset_episode`` interface as
    ``HebbianActorWrapper`` so it can be used interchangeably in
    ``MultiDroneActorManager``.
    """

    def __init__(self, model: nn.Module, stochastic: bool = False):
        self.model = model
        self.stochastic = stochastic

    def reset_episode(self, num_envs: int, device: str | torch.device = "cpu") -> None:
        if hasattr(self.model, "memory_a"):
            self.model.memory_a.reset()

    @torch.no_grad()
    def act(self, obs: Tensor) -> Tensor:
        """Forward pass: LSTM → MLP → tanh → _scale (deterministic)."""
        return self.model.act_inference(obs)


def random_hebbian_rules(
    hebb_cfg: HebbianConfig,
    out_features: int = 7,
    in_features: int = 64,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, Tensor]:
    """Generate random Hebbian rules for benchmarking."""
    rng = np.random.RandomState(seed)
    n_weights = out_features * in_features

    def _rand(lo, hi):
        return torch.tensor(
            rng.uniform(lo, hi, size=(out_features, in_features)),
            dtype=torch.float32, device=device,
        )

    return {
        "A": _rand(*hebb_cfg.A_range),
        "B": _rand(*hebb_cfg.B_range),
        "C": _rand(*hebb_cfg.C_range),
        "D": _rand(*hebb_cfg.D_range),
        "lam": _rand(*hebb_cfg.decay_range),
    }


class MultiDroneActorManager:
    """Manages D independent frozen actor + Hebbian wrappers.

    Parameters
    ----------
    D : int
        Number of drone entities.
    checkpoint_path : str
        Path to frozen WP1 actor .pt file.
    checkpoint_config_path : str
        Path to WP1 config YAML.
    hebbian_rules_list : list[dict]
        D dicts of Hebbian rules (A, B, C, D, lam tensors).
    hebb_cfg : HebbianConfig
        Hebbian config for eta and w_max.
    stochastic : bool
        Sample actions (True) or use mean (False).
    device : str
        Torch device.
    """

    def __init__(
        self,
        D: int,
        checkpoint_path: str,
        checkpoint_config_path: str,
        hebbian_rules_list: Optional[List[Dict[str, Tensor]]] = None,
        hebb_cfg: Optional[HebbianConfig] = None,
        stochastic: bool = True,
        device: str = "cpu",
        use_hebbian: bool = True,
    ):
        self.D = D
        self.device = device
        self.actors: list = []

        for i in range(D):
            model, last_layer, num_actions, hidden_dim = load_frozen_actor(
                checkpoint_path, checkpoint_config_path, device=device,
            )

            if use_hebbian:
                hebbian = HebbianLastLayer(
                    linear_layer=last_layer,
                    hebbian_rules=hebbian_rules_list[i],
                    eta=hebb_cfg.eta,
                    w_max=hebb_cfg.w_max,
                    device=device,
                )
                wrapper = HebbianActorWrapper(model, hebbian, stochastic=stochastic)
            else:
                wrapper = BaselineActorWrapper(model, stochastic=stochastic)

            self.actors.append(wrapper)

    def act(self, obs_all: Tensor) -> Tensor:
        """Forward pass for all D drones.

        Parameters
        ----------
        obs_all : (D, E, obs_dim)

        Returns
        -------
        actions : (D, E, action_dim)
        """
        actions = []
        for i, actor in enumerate(self.actors):
            a = actor.act(obs_all[i])  # (E, action_dim)
            actions.append(a)
        return torch.stack(actions)  # (D, E, action_dim)

    def reset_episode(self, E: int, device: str):
        """Reset LSTM hidden states and Hebbian weights for all actors."""
        for actor in self.actors:
            actor.reset_episode(num_envs=E, device=device)
