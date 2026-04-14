"""
Rules Evolution — Batched URDF population evaluation with HebbianControllers.
=============================================================================

Builds a single Genesis scene to evaluate a population of drone URDFs, each
paired with a HebbianController, across multiple independent forest environments.

Uses WP1's weighted reward function (progress, crash, energy, smoothness, etc.)
as the fitness metric for evolution.

Scene layout
------------
  - D = pop_size   drone entities (one per URDF)
  - E = envs_per_urdf = floor(n_envs / pop_size)   environments per entity
  - Total: D × E drone instances stepped simultaneously on GPU

Forest layout
-------------
  - Exactly E distinct forests are generated (one per env slot).
  - Env slot e uses forest e for all D drones (deterministic, not random).
  - Each drone therefore experiences all E distinct forest scenarios.

Controller layout
-----------------
  - D HebbianControllerBatch objects, one per URDF.
  - Each batch has E "copies" sharing the same Hebbian rules (A, B, C, D, lam)
    but with independent weight matrices and LSTM hidden states (one per env slot).
  - This models: "the same Hebbian rules applied to E independent forest runs."

Fitness (WP1 reward)
--------------------
  Computes the same weighted reward as WP1 training:
    reward = progress*w_p + crash*w_c + energy*w_e + smoothness*w_s + ...

  The weights come from wp1_cfg.reward config.

Typical usage
-------------
  from WP2.rules_evolution import RulesEvolutionEnv

  env = RulesEvolutionEnv(
      urdf_paths=my_urdf_paths,       # list of pop_size URDF file paths
      controllers=my_controllers,     # list of pop_size HebbianController objects
      n_envs=256,
      wp1_cfg=wp1_run_config,
  )
  fitness_matrix = env.evaluate_population(n_episodes=3)
  # fitness_matrix.shape == (pop_size, envs_per_urdf)
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from multi_urdf_utils.multi_drone_env import MultiDroneEnv
from WP2.hebbian import HebbianController, HebbianControllerBatch


# ---------------------------------------------------------------------------
# Controller replication helper
# ---------------------------------------------------------------------------

def _replicate_controller(
    ctrl: HebbianController,
    n_copies: int,
    device: torch.device,
) -> HebbianControllerBatch:
    """Create n_copies of a HebbianController with shared rules but independent weights.

    All copies share the same Hebbian rules (A, B, C, D, lam, eta) but each
    has its own independent copy of the weight matrix W (initialised to
    W_checkpoint).  This models evaluating the same rule set in n_copies
    independent environments.

    Parameters
    ----------
    ctrl : HebbianController
        Source controller whose rules and checkpoint are copied.
    n_copies : int
        Number of independent environment copies (= envs_per_urdf).
    device : torch.device
        Target device for all tensors.

    Returns
    -------
    HebbianControllerBatch
        Batch of n_copies controllers ready for parallel evaluation.
    """
    copies: List[HebbianController] = []
    for _ in range(n_copies):
        rules: Dict[str, Tensor | float] = {
            "A": ctrl.A.clone().to(device),
            "B": ctrl.B.clone().to(device),
            "C": ctrl.C.clone().to(device),
            "D": ctrl.D.clone().to(device),
            "lam": ctrl.lam.clone().to(device),
        }
        if isinstance(ctrl.eta, Tensor):
            rules["eta"] = ctrl.eta.clone().to(device)
        else:
            rules["eta"] = ctrl.eta  # scalar — HebbianController handles this

        copy = HebbianController(
            actor=ctrl.actor,
            rules=rules,
            w_checkpoint=ctrl.W_checkpoint.clone().to(device),
            w_max=ctrl.w_max,
            use_oja_coefficient=ctrl.use_oja_coefficient,
            device=device,
        )
        copies.append(copy)

    return HebbianControllerBatch(copies, device=device)


# ---------------------------------------------------------------------------
# Combined forward + Hebbian update (one drone, E parallel environments)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward_and_update(
    batch: HebbianControllerBatch,
    obs_batch: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Run forward pass and Hebbian weight update for one drone's controller batch.

    This combines the forward pass from HebbianControllerBatch.forward_batch
    with in-place Hebbian weight updates, exposing the intermediate presynaptic
    activations (x) and postsynaptic activations (y) needed for ABCD updates.

    Parameters
    ----------
    batch : HebbianControllerBatch
        E controllers for one drone: same rules, independent weights + LSTM states.
    obs_batch : Tensor
        Shape (E, obs_dim) — observations for this drone's E environments.
    eps : float
        Numerical stability constant for Oja coefficient. Default: 1e-8.

    Returns
    -------
    actions_batch : Tensor
        Shape (E, num_actions) — scaled actions to apply to the environment.
    """
    E = batch.num_controllers
    actor = batch.controllers[0].actor
    obs_batch = obs_batch.to(batch.device)

    # ── LSTM + MLP backbone (shared across all E environments) ──────────
    if not batch._hidden_states_initialized:
        actor.memory_a.hidden_states = None
        hidden_in = actor.memory_a(obs_batch)
        if hidden_in.dim() == 3:
            hidden_in = hidden_in.squeeze(0)
        h, c = actor.memory_a.hidden_states
        batch._h_batch = h.transpose(0, 1).clone()
        batch._c_batch = c.transpose(0, 1).clone()
        batch._hidden_states_initialized = True
    else:
        h_fmt = batch._h_batch.transpose(0, 1)
        c_fmt = batch._c_batch.transpose(0, 1)
        actor.memory_a.hidden_states = (h_fmt, c_fmt)
        hidden_in = actor.memory_a(obs_batch)
        if hidden_in.dim() == 3:
            hidden_in = hidden_in.squeeze(0)
        h_new, c_new = actor.memory_a.hidden_states
        batch._h_batch = h_new.transpose(0, 1).clone()
        batch._c_batch = c_new.transpose(0, 1).clone()

    # MLP layers before the last linear → x: presynaptic activations (E, H)
    x = hidden_in
    for layer in actor.actor.children():
        if isinstance(layer, nn.Linear) and layer.out_features == batch.num_actions:
            break
        x = layer(x)
    # x: (E, hidden_dim)

    # ── Per-controller last layer (each env has its own weight matrix) ──
    W_batch = torch.stack([ctrl.W for ctrl in batch.controllers])           # (E, A, H)
    W_cp_batch = torch.stack([ctrl.W_checkpoint for ctrl in batch.controllers])  # (E, A, H)
    b_batch = torch.stack([ctrl.last_layer.bias for ctrl in batch.controllers])  # (E, A)

    # z[e] = x[e] @ W[e]^T + b[e]
    z = torch.einsum("eh,enh->en", x, W_batch) + b_batch   # (E, A)
    y = torch.tanh(z)                                        # postsynaptic activations
    actions_batch = actor._scale(y)                          # (E, A) — scaled actions

    # ── Batched Hebbian update ───────────────────────────────────────────
    # outer[e, i, j] = y[e, i] * x[e, j]   (postsynaptic × presynaptic)
    outer = torch.einsum("ei,ej->eij", y, x)   # (E, A, H)

    if batch.controllers[0].use_oja_coefficient:
        y2 = y.unsqueeze(2) ** 2               # (E, A, 1)
        w_diff = W_batch - W_cp_batch           # (E, A, H)
        k = 1.0 - (y2 * w_diff) / (outer + eps)  # (E, A, H)
    else:
        k = 1.0

    for i, ctrl in enumerate(batch.controllers):
        dW = ctrl.eta * k[i] * (
            ctrl.A * outer[i]
            + ctrl.B * x[i].unsqueeze(0)   # (1, H) → broadcast to (A, H)
            + ctrl.C * y[i].unsqueeze(1)   # (A, 1) → broadcast to (A, H)
            + ctrl.D
        )
        # W = (1 - lam) * W + lam * W_checkpoint + dW
        ctrl.W.mul_(1.0 - ctrl.lam).add_(dW).add_(ctrl.lam * ctrl.W_checkpoint)
        ctrl.W.clamp_(-ctrl.w_max, ctrl.w_max)

    return actions_batch


# ---------------------------------------------------------------------------
# Main evaluation environment
# ---------------------------------------------------------------------------

class RulesEvolutionEnv:
    """Batched evaluation of a population of URDFs with HebbianControllers.

    Creates a single Genesis scene containing D = pop_size drone entities,
    each with E = envs_per_urdf = floor(n_envs / pop_size) parallel environments.
    Exactly E distinct forests are generated and assigned deterministically:
    env slot e always uses forest e.  Each URDF therefore experiences all E
    distinct forest scenarios within a single episode.

    Parameters
    ----------
    urdf_paths : list[str]
        Paths to D URDF files, one per individual in the population.
    controllers : list[HebbianController]
        Hebbian controllers, one per URDF (len must equal len(urdf_paths)).
    n_envs : int
        Total number of environments.  Actual value used is
        ``pop_size * (n_envs // pop_size)``.
    wp1_cfg : RunConfig
        WP1 configuration (env, obs, reward params, etc.).
    device : str
        PyTorch device string. Default: "cuda".
    vmin, vmax : float
        Range for the randomly sampled forward-speed command.  Default: 6–20 m/s.
    """

    def __init__(
        self,
        urdf_paths: List[str],
        controllers: List[HebbianController],
        n_envs: int,
        wp1_cfg,
        device: str = "cuda",
        vmin: float = 6.0,
        vmax: float = 20.0,
    ) -> None:
        if len(urdf_paths) != len(controllers):
            raise ValueError(
                f"urdf_paths ({len(urdf_paths)}) and controllers ({len(controllers)}) "
                "must have the same length."
            )

        self.pop_size = len(urdf_paths)
        self.envs_per_urdf = n_envs // self.pop_size
        if self.envs_per_urdf < 1:
            raise ValueError(
                f"n_envs ({n_envs}) must be >= pop_size ({self.pop_size}). "
                f"Got envs_per_urdf = {self.envs_per_urdf}."
            )

        self.device = torch.device(device)

        print(
            f"[RulesEvolutionEnv] pop_size={self.pop_size}  "
            f"envs_per_urdf={self.envs_per_urdf}  "
            f"total={self.pop_size * self.envs_per_urdf} instances"
        )

        # ── Build underlying multi-drone scene ───────────────────────────
        # unique_forests_eval=True → generates exactly envs_per_urdf forests
        # (one per env slot); training_mode=False → eval termination + no auto-reset
        self._env = MultiDroneEnv(
            urdf_paths=list(urdf_paths),
            num_envs=self.envs_per_urdf,
            wp1_cfg=wp1_cfg,
            device=device,
            vmin=vmin,
            vmax=vmax,
            training_mode=False,
        )

        # ── Deterministic forest assignment: env slot e → forest e ───────
        E = self.envs_per_urdf
        self._env.forest_ids = torch.arange(E, device=self.device, dtype=torch.long)
        if self._env.cylinders_array is not None:
            self._env.cylinders_xy = self._env.cylinders_array[
                self._env.forest_ids, :, :2
            ]  # (E, T, 2)

        # ── Controller batches: D batches × E copies each ─────────────────
        # Each batch has E independent copies of the URDF's Hebbian controller
        self._ctrl_batches: List[HebbianControllerBatch] = []
        for ctrl in controllers:
            batch = _replicate_controller(ctrl, self.envs_per_urdf, self.device)
            self._ctrl_batches.append(batch)

        # ── Metric accumulators (reset in each call to reset()) ───────────
        D, E = self.pop_size, self.envs_per_urdf
        self._initial_x = torch.zeros(D, E, device=self.device)
        self._cumulative_velocity = torch.zeros(D, E, device=self.device)
        self._cumulative_energy = torch.zeros(D, E, device=self.device)
        self._cumulative_reward = torch.zeros(D, E, device=self.device)
        self._last_actions = None  # For smoothness penalty
        self._active_steps = torch.zeros(D, E, device=self.device)
        # done_flags[d, e] = True once env e of drone d has terminated
        self._done_flags = torch.zeros(D, E, dtype=torch.bool, device=self.device)

        # Extract reward scales from WP1 config
        self.reward_scales = wp1_cfg.reward if hasattr(wp1_cfg, 'reward') else self._default_reward_scales()

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def max_episode_length(self) -> int:
        return int(self._env.max_episode_length)

    @property
    def num_actions(self) -> int:
        return int(self._env.num_actions)

    @property
    def dt(self) -> float:
        return float(self._env.dt)

    def _default_reward_scales(self):
        """Return default reward scales matching WP1 defaults."""
        from WP1.config import RewardConfig
        return RewardConfig()

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the scene and all controllers.

        Resets:
        - Physics state of all drones (positions, velocities).
        - Hebbian weight matrices back to W_checkpoint.
        - LSTM hidden states.
        - Episode metric accumulators.
        """
        # Reset scene (all envs for all drones)
        self._env.reset()

        # Zero metric accumulators
        D, E = self.pop_size, self.envs_per_urdf
        self._cumulative_velocity.zero_()
        self._cumulative_energy.zero_()
        self._cumulative_reward.zero_()
        self._active_steps.zero_()
        self._done_flags.zero_()
        self._last_actions = None

        # Record initial x positions (used for velocity computation)
        for d, ds in enumerate(self._env.drones):
            self._initial_x[d] = ds.base_pos[:, 0]

        # Reset each drone's controller batch: weights back to checkpoint + LSTM zeroed
        for batch in self._ctrl_batches:
            batch.reset_controller_state_and_weights()

    # ── Step ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self) -> Tensor:
        """Run one physics timestep.

        1. Queries each drone's HebbianControllerBatch for actions (E envs).
        2. Applies Hebbian weight updates inline.
        3. Steps the Genesis scene.
        4. Accumulates metric increments for active (non-done) envs.

        Returns
        -------
        done_flags : Tensor
            Shape (pop_size, envs_per_urdf) bool — True for envs that have
            terminated (cumulative across the episode).
        """
        D, E = self.pop_size, self.envs_per_urdf
        obs_buf = self._env.obs_buf   # (D, E, obs_dim) — populated by last reset/step

        # ── Compute actions via controllers, apply Hebbian updates ───────
        all_actions = torch.zeros(D, E, self.num_actions, device=self.device)
        for d, batch in enumerate(self._ctrl_batches):
            actions_d = _forward_and_update(batch, obs_buf[d])  # (E, num_actions)
            all_actions[d] = actions_d

        # ── Advance physics ───────────────────────────────────────────────
        _, _, dones, _ = self._env.step(all_actions)
        # dones: (D, E) bool from MultiDroneEnv eval mode

        # ── Accumulate metrics for environments still active ─────────────
        for d, ds in enumerate(self._env.drones):
            active = ~self._done_flags[d]   # (E,) bool

            # Forward velocity (vx in world frame)
            vx = ds.base_lin_vel[:, 0].clamp(min=0.0)   # (E,)
            self._cumulative_velocity[d] += vx * active.float()

            # Power consumption
            self._cumulative_energy[d] += ds.power * active.float()
            self._active_steps[d] += active.float()

            # Compute WP1 rewards
            reward = self._compute_wp1_reward(d, all_actions[d], dones[d])
            self._cumulative_reward[d] += reward * active.float()

            # Update done flags
            self._done_flags[d] |= dones[d]

        # Store actions for next smoothness penalty
        self._last_actions = all_actions.clone()

        return self._done_flags

    def _compute_wp1_reward(
        self,
        drone_idx: int,
        actions: Tensor,
        dones: Tensor,
    ) -> Tensor:
        """Compute WP1 reward for a single drone.

        Parameters
        ----------
        drone_idx : int
            Index of the drone.
        actions : Tensor
            Shape (E, num_actions) — actions taken.
        dones : Tensor
            Shape (E,) — whether episodes terminated.

        Returns
        -------
        reward : Tensor
            Shape (E,) — instantaneous reward for this timestep.
        """
        E = self.envs_per_urdf
        ds = self._env.drones[drone_idx]
        device = self.device
        dt = self.dt

        reward = torch.zeros(E, device=device)

        # Progress reward: forward speed tracking (Gaussian centered on target command)
        if self.reward_scales.progress != 0.0:
            v_xy = ds.base_lin_vel[:, :2]
            v_proj = v_xy[:, 0]
            v_tgt = self._env.commands[:, 0].clamp(min=1e-3)
            x = v_proj / v_tgt
            sigma = 0.25
            r_progress = torch.exp(-0.5 * (x / sigma) ** 2)
            reward += r_progress * self.reward_scales.progress

        # Energy penalty
        if self.reward_scales.energy != 0.0:
            reward -= ds.power * self.reward_scales.energy

        # Smoothness penalty (action changes)
        if self.reward_scales.smooth != 0.0 and self._last_actions is not None:
            action_diff = torch.sum((actions - self._last_actions[drone_idx]) ** 2, dim=1)
            reward -= action_diff * self.reward_scales.smooth

        # Crash penalty
        if self.reward_scales.crash != 0.0:
            crash = torch.zeros(E, device=device)
            # Check if episode just ended (crashed)
            crash[dones] = 1.0
            reward -= crash * self.reward_scales.crash

        # Scale by dt and convention (matching WP1)
        reward = reward * dt * 50.0

        return reward

    # ── Episode evaluation ───────────────────────────────────────────────

    def evaluate_population(
        self,
        n_episodes: int = 1,
    ) -> Tensor:
        """Run n_episodes and return per-URDF fitness (WP1 weighted reward).

        The episode runs for at most ``max_episode_length`` timesteps.  It
        stops early if all (drone, env) pairs have terminated.

        Each call to this method fully resets the scene and controllers.

        Fitness is computed as the cumulative WP1 reward (weighted combination of
        progress, energy, smoothness, and crash penalties).

        Parameters
        ----------
        n_episodes : int
            Number of independent episodes to average over. Default: 1.

        Returns
        -------
        fitness_matrix : Tensor
            Shape (pop_size, envs_per_urdf).
            ``fitness_matrix[d, e]`` = cumulative WP1 reward for URDF d in
            environment (forest) e, averaged over n_episodes.
        """
        accumulated = torch.zeros(
            self.pop_size, self.envs_per_urdf, device=self.device
        )

        for episode in range(n_episodes):
            self.reset()

            for _step in range(self.max_episode_length):
                done_flags = self.step()
                if done_flags.all():
                    break

            # Cumulative WP1 reward is the fitness
            fitness = self._cumulative_reward

            accumulated += fitness
            print(
                f"[RulesEvolutionEnv] episode {episode + 1}/{n_episodes}  "
                f"mean_reward={fitness.mean().item():.3f}  "
                f"steps_taken={_step + 1}"
            )

        return accumulated / n_episodes

    # ── Auxiliary metrics (for analysis) ─────────────────────────────────

    def evaluate_population_with_aux_metrics(
        self,
        n_episodes: int = 1,
    ) -> Dict[str, Tensor]:
        """Run n_episodes and return WP1 reward + auxiliary metrics.

        Returns
        -------
        metrics : dict[str, Tensor]
            All tensors have shape (pop_size, envs_per_urdf).

            - ``"wp1_reward"``: cumulative WP1 weighted reward (fitness).
            - ``"mean_velocity"``: mean forward velocity (m/s).
            - ``"progress"``: total forward displacement (m).
            - ``"mean_energy"``: mean power consumption (W).
        """
        acc_reward = torch.zeros(self.pop_size, self.envs_per_urdf, device=self.device)
        acc_vel = torch.zeros(self.pop_size, self.envs_per_urdf, device=self.device)
        acc_prog = torch.zeros(self.pop_size, self.envs_per_urdf, device=self.device)
        acc_energy = torch.zeros(self.pop_size, self.envs_per_urdf, device=self.device)

        for episode in range(n_episodes):
            self.reset()

            for _step in range(self.max_episode_length):
                done_flags = self.step()
                if done_flags.all():
                    break

            eps_safe = self._active_steps.clamp(min=1.0)

            # Fitness: WP1 reward
            acc_reward += self._cumulative_reward

            # Auxiliary metrics
            acc_vel += self._cumulative_velocity / eps_safe

            final_x = torch.stack(
                [ds.base_pos[:, 0] for ds in self._env.drones]
            )  # (D, E)
            acc_prog += final_x - self._initial_x

            acc_energy += self._cumulative_energy / eps_safe

            print(
                f"[RulesEvolutionEnv] episode {episode + 1}/{n_episodes}  "
                f"wp1_reward={acc_reward.mean().item() / (episode + 1):.3f}  "
                f"velocity={acc_vel.mean().item() / (episode + 1):.3f} m/s  "
                f"steps={_step + 1}"
            )

        return {
            "wp1_reward": acc_reward / n_episodes,
            "mean_velocity": acc_vel / n_episodes,
            "progress": acc_prog / n_episodes,
            "mean_energy": acc_energy / n_episodes,
        }


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def evaluate_population_batched(
    urdf_paths: List[str],
    controllers: List[HebbianController],
    n_envs: int,
    wp1_cfg,
    n_episodes: int = 1,
    device: str = "cuda",
    vmin: float = 6.0,
    vmax: float = 20.0,
) -> Tensor:
    """Build a RulesEvolutionEnv and run the full evaluation in one call.

    Parameters
    ----------
    urdf_paths : list[str]
        URDF file paths for the population (length = pop_size).
    controllers : list[HebbianController]
        Hebbian controllers, one per URDF.
    n_envs : int
        Total environments.  ``envs_per_urdf = n_envs // pop_size``.
    wp1_cfg : RunConfig
        WP1 run configuration.
    n_episodes : int
        Number of episodes to average the fitness over. Default: 1.
    device : str
        PyTorch device. Default: ``"cuda"``.
    vmin, vmax : float
        Speed command range. Default: 6–20 m/s.

    Returns
    -------
    fitness_matrix : Tensor
        Shape (pop_size, envs_per_urdf).
        ``fitness_matrix[d, e]`` = WP1 reward for URDF d in forest e.
    """
    env = RulesEvolutionEnv(
        urdf_paths=urdf_paths,
        controllers=controllers,
        n_envs=n_envs,
        wp1_cfg=wp1_cfg,
        device=device,
        vmin=vmin,
        vmax=vmax,
    )
    return env.evaluate_population(n_episodes=n_episodes)


__all__ = [
    "RulesEvolutionEnv",
    "evaluate_population_batched",
]
