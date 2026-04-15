"""
MultiDroneActorManager — D independent frozen actor + Hebbian instances.
========================================================================

Each drone entity gets its own copy of the frozen WP1 actor with independent
LSTM hidden states and Hebbian plasticity rules. This ensures zero
cross-contamination between drones during evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from torch import nn

from WP2.frozen_actor import (
    IsolatedPopulationActor,
    build_isolated_population_actor,
    load_frozen_actor,
)
from WP2.config import HebbianConfig, HebbianEvolutionConfig


class BaselineActorWrapper:
    """Thin wrapper around a frozen WP1 actor (no Hebbian plasticity)."""

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


def _adapt_hebb_cfg(hebb_cfg: HebbianConfig) -> HebbianEvolutionConfig:
    """Wrap a HebbianConfig inside a HebbianEvolutionConfig shell so that
    ``build_isolated_population_actor`` (which expects the evolution-level
    config) can read ``cfg.hebbian.*`` directly."""
    shell = HebbianEvolutionConfig.__new__(HebbianEvolutionConfig)
    shell.hebbian = hebb_cfg
    return shell


class MultiDroneActorManager:
    """Manages D independent frozen actor + Hebbian wrappers.

    Parameters
    ----------
    D : int
        Number of drone entities.
    num_envs_per_drone : int
        Number of parallel environments each drone runs in (E).  Each env
        gets its own LSTM state and last-layer weight matrix.
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
    use_hebbian : bool
        If False, use baseline actor wrappers (no plasticity).
    """

    def __init__(
        self,
        D: int,
        num_envs_per_drone: int,
        checkpoint_path: str,
        checkpoint_config_path: str,
        hebbian_rules_list: Optional[List[Dict[str, Tensor]]] = None,
        hebb_cfg: Optional[HebbianConfig] = None,
        stochastic: bool = True,
        device: str = "cpu",
        use_hebbian: bool = True,
    ):
        self.D = D
        self.E = num_envs_per_drone
        self.device = device
        self.actors: list = []

        if use_hebbian:
            if hebbian_rules_list is None or hebb_cfg is None:
                raise ValueError("hebbian_rules_list and hebb_cfg required when use_hebbian=True")

            cfg_shell = _adapt_hebb_cfg(hebb_cfg)
            for i in range(D):
                wrapper = build_isolated_population_actor(
                    checkpoint_path=checkpoint_path,
                    wp1_cfg_path=checkpoint_config_path,
                    hebbian_rules_per_individual=[hebbian_rules_list[i]],
                    cfg=cfg_shell,
                    K=1,
                    S=num_envs_per_drone,
                    device=device,
                    stochastic=stochastic,
                )
                self.actors.append(wrapper)
        else:
            for _ in range(D):
                model, _last_layer, _na, _hd = load_frozen_actor(
                    checkpoint_path, checkpoint_config_path, device=device,
                )
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
        if E != self.E:
            raise ValueError(
                f"reset_episode called with E={E} but manager was built with E={self.E}"
            )
        for actor in self.actors:
            if isinstance(actor, IsolatedPopulationActor):
                actor.reset_episode(device=device)
            else:
                actor.reset_episode(num_envs=E, device=device)
