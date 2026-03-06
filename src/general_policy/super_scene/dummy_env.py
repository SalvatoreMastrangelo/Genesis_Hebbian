from __future__ import annotations

from typing import Dict, Tuple

import torch


class BootstrapVecEnv:
    """
    Minimal VecEnv-like object used only to bootstrap OnPolicyRunner/algorithm objects.

    It intentionally exposes the attributes/methods usually consumed during runner init,
    without requiring Genesis scene creation in the learner process.
    """

    def __init__(
        self,
        num_envs: int,
        num_obs: int,
        num_privileged_obs: int,
        num_actions: int,
        max_episode_length: int,
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.num_obs = int(num_obs)
        self.num_privileged_obs = int(num_privileged_obs)
        self.num_actions = int(num_actions)
        self.max_episode_length = int(max_episode_length)

        self.dt = 0.04
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.privileged_obs_buf = torch.zeros(
            (self.num_envs, self.num_privileged_obs), device=self.device
        )
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device)
        self.reset_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int64)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.extras: Dict = {
            "observations": {"critic": self.privileged_obs_buf},
            "time_outs": torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32),
        }

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        self.obs_buf.zero_()
        self.privileged_obs_buf.zero_()
        self.rew_buf.zero_()
        self.reset_buf.zero_()
        self.episode_length_buf.zero_()
        self.extras["observations"]["critic"] = self.privileged_obs_buf
        self.extras["time_outs"].zero_()
        return self.obs_buf, self.extras

    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> Tuple[torch.Tensor, Dict]:
        return self.privileged_obs_buf, {}

    def step(self, actions: torch.Tensor):
        # Should never be used by the logical-super-scene runner.
        _ = actions
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def close(self) -> None:
        return None
