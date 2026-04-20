"""
MultiSceneEvalEnv — K independent Genesis scenes for WP2 multi-URDF evaluation.
===============================================================================

Why this module exists
----------------------
The previous ``multi_urdf_utils.multi_drone_env.MultiDroneEnv`` packs D URDFs
into a single Genesis scene so the physics step runs once for all of them.
That is efficient but couples D URDFs through a single Taichi rigid/aero
state: when one URDF's environment goes NaN (a crash-mode physics blowup),
that NaN persists in shared solver fields and contaminates every other env
on the next reset + step.  Diagnostics confirmed this: post-reset state is
clean, but the first ``scene.step()`` produces NaN in every env in every
drone's slot.

WP1 sidesteps this by building one Genesis scene per URDF (``Gen_Env``).
Each scene has its own rigid solver, its own aero Taichi fields, its own
contact/constraint state.  NaN in one URDF's scene cannot reach any other
URDF's scene, and auto-reset within each scene cleans its own state.

This module mirrors that construction for WP2 eval.  It builds K
``WingedDroneEnv`` scenes — one per URDF — and exposes the
``MultiDroneEnv``-shaped API that the WP2 rollout loop expects:

* ``env.D``, ``env.E``             – K URDFs, E envs per URDF
* ``env.drones[d]``                – the k-th WingedDroneEnv (also used for
                                     per-drone state access: ``.base_pos``,
                                     ``.reset_buf``, ``.last_reward_total``,
                                     ``.power``, ``.pre_collision``, ...)
* ``env.reset()``                  – shape (D, E, obs_dim)
* ``env.step(actions)``            – actions shape (D, E, num_actions)
* ``env.refresh_forests()``        – regenerate & synchronise forests across D scenes
* ``env._fixed_forest_ids``,
  ``env._eval_speed_grid``         – propagated to every sub-env
* ``env._compute_rewards_drone``   – no-op (WingedDroneEnv already computes
                                     rewards inside step)

The per-slot forest and target speed assignment is SHARED across all D
sub-envs: slot ``e`` in every URDF sees the same forest and target speed,
so (urdf, individual) fitness comparisons remain fair.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch

from winged_drone_train.env import WingedDroneEnv
from winged_drone_train.noise_config import configure_solver_noise


class MultiSceneEvalEnv:
    """Wrap K independent ``WingedDroneEnv`` scenes behind a MultiDroneEnv-like API."""

    def __init__(
        self,
        urdf_paths: List[str],
        num_envs_per_drone: int,
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        command_cfg: dict,
        device: str,
    ) -> None:
        if not urdf_paths:
            raise ValueError("MultiSceneEvalEnv requires at least one URDF path.")

        self.D = len(urdf_paths)
        self.E = int(num_envs_per_drone)
        self.device = device

        self.drones: List[WingedDroneEnv] = []
        for k, urdf_file in enumerate(urdf_paths):
            # Each sub-env takes its own copies so per-scene mutations don't leak.
            sub_env_cfg = dict(env_cfg)
            sub_obs_cfg = dict(obs_cfg)
            sub_reward_cfg = dict(reward_cfg)
            sub_command_cfg = dict(command_cfg)

            sub = WingedDroneEnv(
                num_envs=self.E,
                env_cfg=sub_env_cfg,
                obs_cfg=sub_obs_cfg,
                reward_cfg=sub_reward_cfg,
                command_cfg=sub_command_cfg,
                urdf_file=urdf_file,
                show_viewer=False,
                eval=True,
                device=device,
                auto_reset=False,
            )
            configure_solver_noise(sub, sub_env_cfg)
            self.drones.append(sub)

        # Sanity: all sub-envs must share obs/action dims so we can build one
        # shared actor batch across them.
        obs_dims = {s.num_obs for s in self.drones}
        act_dims = {s.num_actions for s in self.drones}
        if len(obs_dims) != 1 or len(act_dims) != 1:
            raise RuntimeError(
                f"MultiSceneEvalEnv: sub-env dim mismatch  obs={obs_dims}  act={act_dims}"
            )
        self.num_obs = self.drones[0].num_obs
        self.num_actions = self.drones[0].num_actions
        self.dt = float(self.drones[0].dt)

        # Shared stacked obs buffer — (D, E, obs_dim)
        self.obs_buf = torch.zeros(self.D, self.E, self.num_obs, device=self.device)

        # Slot-aligned eval overrides.  Setting these on the parent propagates
        # to every sub-env so every URDF sees the same per-slot forest id and
        # target speed (required for fair comparisons).
        self._fixed_forest_ids_buf: Optional[torch.Tensor] = None
        self._eval_speed_grid_buf: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    #  Eval override properties — propagate to each sub-env
    # ------------------------------------------------------------------

    @property
    def _fixed_forest_ids(self) -> Optional[torch.Tensor]:
        return self._fixed_forest_ids_buf

    @_fixed_forest_ids.setter
    def _fixed_forest_ids(self, value: Optional[torch.Tensor]) -> None:
        self._fixed_forest_ids_buf = value
        for sub in self.drones:
            sub._fixed_forest_ids = value

    @property
    def _eval_speed_grid(self) -> Optional[torch.Tensor]:
        return self._eval_speed_grid_buf

    @_eval_speed_grid.setter
    def _eval_speed_grid(self, value: Optional[torch.Tensor]) -> None:
        self._eval_speed_grid_buf = value
        for sub in self.drones:
            sub._eval_speed_grid = value

    @property
    def cylinders_array(self):
        """Expose sub-env 0's forest pool for code that reads it after setup."""
        return self.drones[0].cylinders_array if self.drones else None

    @property
    def forest_ids(self):
        return self.drones[0].forest_ids

    @property
    def cylinders_xy(self):
        return self.drones[0].cylinders_xy

    # ------------------------------------------------------------------
    #  Forest regeneration
    # ------------------------------------------------------------------

    def refresh_forests(self) -> None:
        """Regenerate the forest pool and synchronise it across all sub-envs.

        To keep per-slot forest identical across URDFs, we regenerate once in
        sub-env 0 and copy the resulting ``cylinders_array`` into every other
        sub-env.  Then each sub-env re-applies its forest ids (either its
        ``_fixed_forest_ids`` or fresh random ones shared across sub-envs).
        """
        head = self.drones[0]
        if head._forest_generator is None or head.cylinders_array is None:
            return

        new_cylinders, _ = head._forest_generator.generate()

        # Shared per-slot forest assignment (either fixed or a fresh random draw).
        if self._fixed_forest_ids_buf is not None:
            shared_ids = self._fixed_forest_ids_buf
        else:
            n_forests = new_cylinders.shape[0]
            shared_ids = torch.randint(
                0, n_forests, (self.E,), device=self.device, dtype=torch.long,
            )

        for sub in self.drones:
            sub.cylinders_array = new_cylinders
            sub.forest_ids[:] = shared_ids
            sub.cylinders_xy = new_cylinders[sub.forest_ids, :, :2]

    # ------------------------------------------------------------------
    #  reset / step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset(self) -> Tuple[torch.Tensor, dict]:
        """Reset every sub-env and return a stacked (D, E, obs_dim) obs tensor."""
        for d, sub in enumerate(self.drones):
            obs_sub, _ = sub.reset()
            self.obs_buf[d] = obs_sub.to(self.device)
        return self.obs_buf, {}

    @torch.no_grad()
    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step every sub-env with its ``(E, num_actions)`` slice."""
        if actions.shape != (self.D, self.E, self.num_actions):
            raise ValueError(
                f"MultiSceneEvalEnv.step: expected actions shape "
                f"({self.D}, {self.E}, {self.num_actions}), got {tuple(actions.shape)}"
            )
        actions = actions.to(self.device)

        rew = torch.zeros(self.D, self.E, device=self.device)
        dones = torch.zeros(self.D, self.E, dtype=torch.bool, device=self.device)

        for d, sub in enumerate(self.drones):
            obs_sub, rew_sub, done_sub, _ = sub.step(actions[d])
            self.obs_buf[d] = obs_sub.to(self.device)
            rew[d] = rew_sub.to(self.device)
            dones[d] = done_sub.to(torch.bool).to(self.device)

        return self.obs_buf, rew, dones, {}

    # ------------------------------------------------------------------
    #  Reward API compatibility shim
    # ------------------------------------------------------------------

    def _compute_rewards_drone(self, drone_idx: int, ds) -> None:
        """No-op shim.

        The legacy ``MultiDroneEnv`` ran in ``training_mode=False`` and
        required the rollout to call ``_compute_rewards_drone`` explicitly to
        populate ``ds.last_reward_total``.  ``WingedDroneEnv`` already
        computes rewards inside ``step()`` (via ``_accumulate_rewards``) and
        writes ``last_reward_total``, so nothing is needed here — but we keep
        the method so the rollout code is unchanged.
        """
        return
