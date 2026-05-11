"""Observation construction utilities for RL control of a winged drone.

This module focuses on building compact, well-structured observations for
policy (actor) and critic networks.  It is intentionally decoupled from the
rest of the environment so that it can be reused or unit-tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class ObsScaling:
    """Simple container for feature scaling parameters.

    Attributes:
        altitude_center: Reference altitude in meters (z=altitude_center maps to 0).
        altitude_scale: Altitude scale in meters (delta_z / altitude_scale).
        vel_scale: Per–axis velocity scale [vx, vy, vz] in m/s.
        command_speed_scale: Scale for forward–speed command in m/s.
        max_depth: Maximum depth sensor distance in meters used for
            normalisation (near obstacles -> values close to 1).
    """

    altitude_center: float = 10.0
    altitude_scale: float = 10.0
    vel_scale: Tuple[float, float, float] = (20.0, 10.0, 10.0)
    command_speed_scale: float = 25.0
    max_depth: float = 30.0


@dataclass
class PrivilegedObsScaling:
    """Scaling parameters for critic-only privileged features."""

    base_ang_vel: float = 5.0
    joint_position: float = 1.0
    joint_velocity: float = 5.0
    actual_thrust: str | float = "max_thrust"


class ObservationBuilder:
    """Build observations for actor and critic networks.

    The layout of the actor observation (without genome features) is:

        [ z_norm(1),
          quat(4),
          lin_vel(3),
          depth(num_actor_sectors)   (optional),
          last_actions(num_actions),
          forward_speed_cmd(1) ]

    Optionally, a genome vector of dimension ``G`` can be appended separately
    to actor and/or critic observations (controlled by add_genome_obs_actor
    and add_genome_obs_critic).

    The critic observation mirrors the actor observation but without noise and
    may append extra privileged features configured through ``obs_cfg``.
    """

    def __init__(
        self,
        num_actions: int,
        num_sectors_actor: int,
        *,
        joint_limits_max: Optional[torch.Tensor] = None,
        obs_cfg: Optional[Dict] = None,
        add_genome_obs_actor: bool = False,
        add_genome_obs_critic: bool = False,
        include_joint_pos_critic: bool = False,
        include_joint_vel_critic: bool = False,
        include_ang_vel_critic: bool = False,
        include_effective_thrust_critic: bool = False,
        genome_vec: Optional[torch.Tensor] = None,
        genome_min: Optional[list] = None,
        genome_max: Optional[list] = None,
        num_servos: int = 6,
        include_depth: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_actions = int(num_actions)
        self.num_actor_sectors = int(num_sectors_actor)
        self.include_depth = bool(include_depth)
        self.device = torch.device(device)

        # Critic-only features
        self.include_joint_pos_critic = bool(include_joint_pos_critic)
        self.include_joint_vel_critic = bool(include_joint_vel_critic)
        self.include_ang_vel_critic = bool(include_ang_vel_critic)
        self.include_effective_thrust_critic = bool(include_effective_thrust_critic)
        self.num_servos = int(num_servos)

        # Joint limits used to normalise last joint actions to roughly [-1, 1]
        if joint_limits_max is not None:
            jmax = torch.as_tensor(joint_limits_max, dtype=torch.float32, device=self.device)
            if jmax.ndim != 1 or jmax.numel() != self.num_actions - 1:
                raise ValueError(
                    "joint_limits_max must be 1D with length num_actions-1 "
                    f"(got shape {tuple(jmax.shape)} for num_actions={self.num_actions})"
                )
            self.joint_limits_max = jmax
        else:
            # Fallback: assume last actions already in [-1,1]
            self.joint_limits_max = torch.ones(self.num_actions - 1, device=self.device)
        self._joint_limits_max_safe = self.joint_limits_max.clamp(min=1e-6).unsqueeze(0)

        # Configuration and noise
        cfg = {} if obs_cfg is None else dict(obs_cfg)
        self.add_noise: bool = bool(cfg.get("add_noise", False))
        self.noise_std: Dict[str, float] = dict(cfg.get("noise_std", {}))
        priv_cfg = dict(cfg.get("privileged_obs", {}))
        self.include_priv_base_ang_vel = bool(priv_cfg.get("base_ang_vel", False))
        self.include_priv_joint_position = bool(priv_cfg.get("joint_position", False))
        self.include_priv_joint_velocity = bool(priv_cfg.get("joint_velocity", False))
        self.include_priv_actual_thrust = bool(priv_cfg.get("actual_thrust", False))
        priv_scaling_cfg = dict(cfg.get("privileged_scaling", {}))
        self.priv_scaling = PrivilegedObsScaling(
            base_ang_vel=float(priv_scaling_cfg.get("base_ang_vel", 5.0)),
            joint_position=float(priv_scaling_cfg.get("joint_position", 1.0)),
            joint_velocity=float(priv_scaling_cfg.get("joint_velocity", 5.0)),
            actual_thrust=priv_scaling_cfg.get("actual_thrust", "max_thrust"),
        )
        self._inv_priv_base_ang_vel_scale = 1.0 / max(1e-6, self.priv_scaling.base_ang_vel)
        self._inv_priv_joint_position_scale = 1.0 / max(1e-6, self.priv_scaling.joint_position)
        self._inv_priv_joint_velocity_scale = 1.0 / max(1e-6, self.priv_scaling.joint_velocity)

        # Feature scaling parameters
        scaling_cfg = cfg.get("scaling", {})
        self.scaling = ObsScaling(
            altitude_center=float(scaling_cfg.get("altitude_center", 10.0)),
            altitude_scale=float(scaling_cfg.get("altitude_scale", 10.0)),
            vel_scale=tuple(scaling_cfg.get("vel_scale", (20.0, 10.0, 10.0))),
            command_speed_scale=float(scaling_cfg.get("command_speed_scale", 25.0)),
            max_depth=float(scaling_cfg.get("max_depth", 30.0)),
        )
        self._inv_altitude_scale = 1.0 / max(1e-6, self.scaling.altitude_scale)
        self._inv_vel_scale = (
            1.0 / max(1e-6, self.scaling.vel_scale[0]),
            1.0 / max(1e-6, self.scaling.vel_scale[1]),
            1.0 / max(1e-6, self.scaling.vel_scale[2]),
        )
        self._inv_command_speed_scale = 1.0 / max(1e-6, self.scaling.command_speed_scale)
        self._inv_max_depth = 1.0 / max(1e-6, self.scaling.max_depth)
        self._noise_scratch: Optional[torch.Tensor] = None

        # Genome configuration ------------------------------------------------
        self.add_genome_obs_actor = bool(add_genome_obs_actor)
        self.add_genome_obs_critic = bool(add_genome_obs_critic)
        self.genome_vec = genome_vec  # may be None initially
        self.genome_dim: Optional[int] = None
        self.genome_min: Optional[torch.Tensor] = None
        self.genome_max: Optional[torch.Tensor] = None
        self._genome_denom: Optional[torch.Tensor] = None

        # Initialize genome if either actor or critic needs it
        if self.add_genome_obs_actor or self.add_genome_obs_critic:
            if genome_vec is not None:
                self.genome_dim = int(genome_vec.shape[1])
            elif genome_min is not None and genome_max is not None:
                if len(genome_min) != len(genome_max):
                    raise ValueError("genome_min and genome_max must have the same length")
                self.genome_dim = len(genome_min)
            else:
                raise ValueError(
                    "add_genome_obs_actor or add_genome_obs_critic is True but neither genome_vec nor (genome_min, genome_max) were provided"
                )

            if genome_min is not None and genome_max is not None:
                min_arr = np.asarray(genome_min, dtype=np.float32)
                max_arr = np.asarray(genome_max, dtype=np.float32)
                if min_arr.shape[0] != self.genome_dim or max_arr.shape[0] != self.genome_dim:
                    raise ValueError("Genome min/max length does not match genome dimension.")
                self.genome_min = torch.from_numpy(min_arr).to(self.device).unsqueeze(0)  # (1, G)
                self.genome_max = torch.from_numpy(max_arr).to(self.device).unsqueeze(0)  # (1, G)
                self._genome_denom = (self.genome_max - self.genome_min).clamp(min=1e-6)

        # ------------------------------------------------------------------
        # Precompute observation dimensions
        # ------------------------------------------------------------------
        base_kin_dim = 8  # z_norm(1) + quat(4) + lin_vel(3)
        depth_dim_actor = self.num_actor_sectors if self.include_depth else 0
        last_act_dim = self.num_actions  # throttle + all joints
        cmd_dim = 1  # forward speed command

        self.actor_obs_dim = base_kin_dim + depth_dim_actor + last_act_dim + cmd_dim
        self.priv_obs_dim = self.actor_obs_dim
        if self.include_priv_base_ang_vel:
            self.priv_obs_dim += 3
        if self.include_priv_joint_position:
            self.priv_obs_dim += self.num_actions - 1
        if self.include_priv_joint_velocity:
            self.priv_obs_dim += self.num_actions - 1
        if self.include_priv_actual_thrust:
            self.priv_obs_dim += 1

        # Critic-only features (not added to actor observation)
        critic_extra_dim = 0
        if self.include_joint_pos_critic:
            critic_extra_dim += self.num_servos
        if self.include_joint_vel_critic:
            critic_extra_dim += self.num_servos
        if self.include_ang_vel_critic:
            critic_extra_dim += 3  # 3D angular velocity
        if self.include_effective_thrust_critic:
            critic_extra_dim += 1  # scalar thrust in Newtons

        if self.add_genome_obs_actor and self.genome_dim is not None:
            self.actor_obs_dim += self.genome_dim
        if self.add_genome_obs_critic and self.genome_dim is not None:
            self.priv_obs_dim += self.genome_dim

        self.priv_obs_dim += critic_extra_dim

        self._base_actor_obs_dim = base_kin_dim + depth_dim_actor + last_act_dim + cmd_dim
        self._actor_scratch: Optional[torch.Tensor] = None
        self._critic_scratch: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_observations(
        self,
        base_pos: torch.Tensor,
        base_quat: torch.Tensor,
        base_lin_vel: torch.Tensor,
        last_actions: torch.Tensor,
        commands: torch.Tensor,
        *,
        depth_actor: Optional[torch.Tensor] = None,
        base_ang_vel: Optional[torch.Tensor] = None,
        joint_position: Optional[torch.Tensor] = None,
        joint_velocity: Optional[torch.Tensor] = None,
        actual_thrust: Optional[torch.Tensor] = None,
        actual_thrust_scale: Optional[torch.Tensor] = None,
        genome_vec: Optional[torch.Tensor] = None,
        joint_positions: Optional[torch.Tensor] = None,
        joint_velocities: Optional[torch.Tensor] = None,
        effective_thrust: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct actor and critic observations.

        Args:
            base_pos: ``(B, 3)`` base position in world frame.
            base_quat: ``(B, 4)`` orientation quaternion (w, x, y, z).
            base_lin_vel: ``(B, 3)`` linear velocity in world frame.
            last_actions: ``(B, num_actions)`` last applied *scaled* actions
                (throttle + servo angles).
            commands: ``(B, M)`` command vector.  Only the forward–speed target
                at index 0 is currently used, but the extra components are
                accepted for future extensions.
            depth_actor: Optional ``(B, num_actor_sectors)`` tensor with depth
                readings in meters for the actor FOV.
            genome_vec: Optional ``(B, G)`` genome tensor.  If provided, it
                overrides the genome passed at construction time.
            joint_positions: Optional ``(B, num_servos)`` joint positions.
                Only appended to critic if include_joint_pos_critic=True.
            joint_velocities: Optional ``(B, num_servos)`` joint velocities.
                Only appended to critic if include_joint_vel_critic=True.
            base_ang_vel: Optional ``(B, 3)`` base angular velocity in body frame.
                Only appended to critic if include_ang_vel_critic=True.
            effective_thrust: Optional ``(B, 1)`` effective thrust in Newtons.
                Only appended to critic if include_effective_thrust_critic=True.

        Returns:
            A tuple ``(obs_actor, obs_critic)``.
        """
        device = self.device
        if base_pos.device != device:
            base_pos = base_pos.to(device)
        if base_quat.device != device:
            base_quat = base_quat.to(device)
        if base_lin_vel.device != device:
            base_lin_vel = base_lin_vel.to(device)
        if last_actions.device != device:
            last_actions = last_actions.to(device)
        if commands.device != device:
            commands = commands.to(device)
        if depth_actor is not None:
            if depth_actor.device != device:
                depth_actor = depth_actor.to(device)
        if base_ang_vel is not None and base_ang_vel.device != device:
            base_ang_vel = base_ang_vel.to(device)
        if joint_position is not None and joint_position.device != device:
            joint_position = joint_position.to(device)
        if joint_velocity is not None and joint_velocity.device != device:
            joint_velocity = joint_velocity.to(device)
        if actual_thrust is not None and actual_thrust.device != device:
            actual_thrust = actual_thrust.to(device)
        if actual_thrust_scale is not None and actual_thrust_scale.device != device:
            actual_thrust_scale = actual_thrust_scale.to(device)
        if genome_vec is not None:
            self.genome_vec = genome_vec if genome_vec.device == device else genome_vec.to(device)

        B = base_pos.shape[0]

        # ----------------------- Base kinematic features --------------------
        # Altitude (z) normalised around a reference height
        z = base_pos[:, 2:3]
        z_norm = (z - self.scaling.altitude_center) * self._inv_altitude_scale

        quat = base_quat  # (B, 4)

        vx = base_lin_vel[:, 0:1]
        vy = base_lin_vel[:, 1:2]
        vz = base_lin_vel[:, 2:3]
        vx_norm = vx * self._inv_vel_scale[0]
        vy_norm = vy * self._inv_vel_scale[1]
        vz_norm = vz * self._inv_vel_scale[2]

        # -------------------------- Depth features --------------------------
        depth_feat_actor = None
        if self.include_depth and depth_actor is not None:
            # Map [0, max_depth] -> [0, 1] with 1 = very close, 0 = no obstacle.
            depth_feat_actor = 1.0 - depth_actor * self._inv_max_depth

        # ---------------------- Last action features ------------------------
        if last_actions.shape[1] != self.num_actions:
            raise ValueError(
                f"last_actions second dimension must be num_actions={self.num_actions}, "
                f"got {last_actions.shape[1]}"
            )

        last_thr = last_actions[:, 0:1]  # already in physical [min,max] range
        if self.num_actions > 1:
            last_jnts_raw = last_actions[:, 1:]
            # Normalise by the configured maximum joint range
            last_jnts = last_jnts_raw / self._joint_limits_max_safe
        else:
            last_jnts = torch.empty((B, 0), device=device)

        # ------------------------ Command features --------------------------
        v_tgt = commands[:, 0].unsqueeze(1)
        v_tgt_norm = v_tgt * self._inv_command_speed_scale

        obs_actor, obs_critic = self._ensure_obs_scratch(B, device)
        idx = 0
        obs_actor[:, idx : idx + 1].copy_(z_norm)
        idx += 1
        obs_actor[:, idx : idx + 4].copy_(quat)
        idx += 4
        obs_actor[:, idx : idx + 1].copy_(vx_norm)
        idx += 1
        obs_actor[:, idx : idx + 1].copy_(vy_norm)
        idx += 1
        obs_actor[:, idx : idx + 1].copy_(vz_norm)
        idx += 1
        if depth_feat_actor is not None:
            depth_dim = depth_feat_actor.shape[1]
            obs_actor[:, idx : idx + depth_dim].copy_(depth_feat_actor)
            idx += depth_dim
        obs_actor[:, idx : idx + 1].copy_(last_thr)
        idx += 1
        if self.num_actions > 1:
            obs_actor[:, idx : idx + self.num_actions - 1].copy_(last_jnts)
        idx += self.num_actions - 1
        obs_actor[:, idx : idx + 1].copy_(v_tgt_norm)
        # Snapshot clean base features for the critic before noise is added.
        obs_clean = obs_actor.clone()
        # --------------------------- Add noise ------------------------------
        if self.add_noise and self.noise_std:
            std_cfg = self.noise_std
            idx = 0
            noise_buf = self._get_noise_scratch(B, self._base_actor_obs_dim, device)

            # z_norm
            if std_cfg.get("z", 0.0) > 0.0:
                noise = noise_buf[:, :1]
                noise.normal_()
                noise *= std_cfg["z"]
                obs_actor[:, idx : idx + 1] += noise
            idx += 1

            # quat (4)
            if std_cfg.get("quat", 0.0) > 0.0:
                noise = noise_buf[:, :4]
                noise.normal_()
                noise *= std_cfg["quat"]
                obs_actor[:, idx : idx + 4] += noise
            idx += 4

            # vel (3)
            if std_cfg.get("vel", 0.0) > 0.0:
                noise = noise_buf[:, :3]
                noise.normal_()
                noise *= std_cfg["vel"]
                obs_actor[:, idx : idx + 3] += noise
            idx += 3

            # depth
            if depth_feat_actor is not None:
                depth_dim = depth_feat_actor.shape[1]
                if std_cfg.get("depth", 0.0) > 0.0:
                    noise = noise_buf[:, :depth_dim]
                    noise.normal_()
                    noise *= (std_cfg["depth"] * depth_actor * self._inv_max_depth)
                    obs_actor[:, idx : idx + depth_dim] += noise
                idx += depth_dim

            # last actions (throttle + joints)
            if std_cfg.get("last_actions", 0.0) > 0.0:
                dim_la = self.num_actions
                noise = noise_buf[:, :dim_la]
                noise.normal_()
                noise *= std_cfg["last_actions"]
                obs_actor[:, idx : idx + dim_la] += noise
            idx += self.num_actions

            # command speed
            if std_cfg.get("command", 0.0) > 0.0:
                noise = noise_buf[:, :1]
                noise.normal_()
                noise *= std_cfg["command"]
                obs_actor[:, idx : idx + 1] += noise
            idx += 1

        # Critic gets clean observation; it may also see extra depth sectors and critic-only features.
        obs_critic = obs_clean

        # ------------------------- Genome features --------------------------
        if self.add_genome_obs_actor or self.add_genome_obs_critic:
            if self.genome_vec is None:
                raise RuntimeError("add_genome_obs_actor or add_genome_obs_critic=True but no genome_vec has been provided.")
            genome = self.genome_vec
            if genome.device != device:
                genome = genome.to(device)
            if genome.shape[0] != B:
                # Broadcast single genome vector to all environments if needed.
                if genome.shape[0] == 1:
                    genome = genome.expand(B, -1)
                else:
                    raise ValueError(
                        f"Genome batch dimension {genome.shape[0]} does not match B={B}"
                    )

            if self.genome_min is not None and self.genome_max is not None:
                # Normalise to roughly [0, 1]
                denom = self._genome_denom
                if denom is None:
                    denom = (self.genome_max - self.genome_min).clamp(min=1e-6)
                    self._genome_denom = denom
                genome_norm = (genome - self.genome_min) / denom
            else:
                genome_norm = genome

            if self.add_genome_obs_actor:
                obs_actor = torch.cat((obs_actor, genome_norm), dim=1)
            if self.add_genome_obs_critic:
                obs_critic = torch.cat((obs_critic, genome_norm), dim=1)

            # print each different observation component of the first env separately for debugging
            '''
            print("Obs components (first env):")
            idx = 0
            print(f" z_norm: {obs_actor[0, idx:idx+1].cpu().numpy()}")
            idx += 1
            print(f" quat: {obs_actor[0, idx:idx+4].cpu().numpy()}")
            idx += 4
            print(f" vel: {obs_actor[0, idx:idx+3].cpu().numpy()}")
            idx += 3
            if depth_feat_actor is not None:
                depth_dim = depth_feat_actor.shape[1]
                print(f" depth: {obs_actor[0, idx:idx+depth_dim].cpu().numpy()}")
                idx += depth_dim
            print(f" last actions: {obs_actor[0, idx:idx+self.num_actions].cpu().numpy()}")
            idx += self.num_actions
            print(f" command speed: {obs_actor[0, idx:idx+1].cpu().numpy()}")
            idx += 1
            print(f" genome: {obs_actor[0, idx:idx+self.genome_dim].cpu().numpy()}")
            '''

        # ----------------------- Critic-only features -----------------------
        # Append joint positions, joint velocities, and base angular velocity to critic only
        critic_features = []

        if self.include_joint_pos_critic:
            if joint_positions is None:
                raise ValueError("include_joint_pos_critic=True but joint_positions not provided")
            jp = joint_positions
            if jp.device != device:
                jp = jp.to(device)
            if self.add_noise and self.noise_std:
                std_cfg = self.noise_std
                if std_cfg.get("joint_pos", 0.0) > 0.0:
                    jp = jp.clone()
                    noise = torch.randn_like(jp, device=device)
                    noise *= std_cfg["joint_pos"]
                    jp = jp + noise
            critic_features.append(jp)

        if self.include_joint_vel_critic:
            if joint_velocities is None:
                raise ValueError("include_joint_vel_critic=True but joint_velocities not provided")
            jv = joint_velocities
            if jv.device != device:
                jv = jv.to(device)
            if self.add_noise and self.noise_std:
                std_cfg = self.noise_std
                if std_cfg.get("joint_vel", 0.0) > 0.0:
                    jv = jv.clone()
                    noise = torch.randn_like(jv, device=device)
                    noise *= std_cfg["joint_vel"]
                    jv = jv + noise
            critic_features.append(jv)

        if self.include_ang_vel_critic:
            if base_ang_vel is None:
                raise ValueError("include_ang_vel_critic=True but base_ang_vel not provided")
            av = base_ang_vel
            if av.device != device:
                av = av.to(device)
            if self.add_noise and self.noise_std:
                std_cfg = self.noise_std
                if std_cfg.get("ang_vel", 0.0) > 0.0:
                    av = av.clone()
                    noise = torch.randn_like(av, device=device)
                    noise *= std_cfg["ang_vel"]
                    av = av + noise
            critic_features.append(av)

        if self.include_effective_thrust_critic:
            if effective_thrust is None:
                raise ValueError("include_effective_thrust_critic=True but effective_thrust not provided")
            et = effective_thrust
            if et.device != device:
                et = et.to(device)
            # Ensure it's 2D (B, 1)
            if et.ndim == 1:
                et = et.unsqueeze(1)
            if self.add_noise and self.noise_std:
                std_cfg = self.noise_std
                if std_cfg.get("effective_thrust", 0.0) > 0.0:
                    et = et.clone()
                    noise = torch.randn_like(et, device=device)
                    noise *= std_cfg["effective_thrust"]
                    et = et + noise
            critic_features.append(et)

        if critic_features:
            obs_critic = torch.cat((obs_critic,) + tuple(critic_features), dim=1)

        return obs_actor, obs_critic

    def _ensure_obs_scratch(self, B: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._actor_scratch is None
            or self._actor_scratch.shape != (B, self.actor_obs_dim)
            or self._actor_scratch.device != device
        ):
            self._actor_scratch = torch.empty((B, self.actor_obs_dim), device=device, dtype=torch.float32)
        if (
            self._critic_scratch is None
            or self._critic_scratch.shape != (B, self.priv_obs_dim)
            or self._critic_scratch.device != device
        ):
            self._critic_scratch = torch.empty((B, self.priv_obs_dim), device=device, dtype=torch.float32)
        return self._actor_scratch, self._critic_scratch

    def _get_noise_scratch(self, B: int, D: int, device: torch.device) -> torch.Tensor:
        if (
            self._noise_scratch is None
            or self._noise_scratch.shape[0] != B
            or self._noise_scratch.shape[1] < D
            or self._noise_scratch.device != device
        ):
            self._noise_scratch = torch.empty((B, D), device=device, dtype=torch.float32)
        return self._noise_scratch
