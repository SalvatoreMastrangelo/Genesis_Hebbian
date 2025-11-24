"""Observation construction utilities for RL control of a winged drone.

This module focuses on building compact, well–structured observations for
policy (actor) and critic networks.  It is intentionally decoupled from the
rest of the environment so that it can be reused or unit–tested independently.
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


class ObservationBuilder:
    """Build observations for actor and critic networks.

    The layout of the actor observation (without genome features) is:

        [ z_norm(1),
          quat(4),
          lin_vel(3),
          depth(num_actor_sectors)   (optional),
          last_actions(num_actions),
          forward_speed_cmd(1) ]

    Optionally, a genome vector of dimension ``G`` can be appended at the end
    of both actor and critic observations.

    The critic observation mirrors the actor observation but without noise.
    """

    def __init__(
        self,
        num_actions: int,
        num_sectors_actor: int,
        *,
        joint_limits_max: Optional[torch.Tensor] = None,
        obs_cfg: Optional[Dict] = None,
        add_genome_obs: bool = False,
        genome_vec: Optional[torch.Tensor] = None,
        genome_min: Optional[list] = None,
        genome_max: Optional[list] = None,
        include_depth: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_actions = int(num_actions)
        self.num_actor_sectors = int(num_sectors_actor)
        self.include_depth = bool(include_depth)
        self.device = torch.device(device)

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

        # Configuration and noise
        cfg = {} if obs_cfg is None else dict(obs_cfg)
        self.add_noise: bool = bool(cfg.get("add_noise", False))
        self.noise_std: Dict[str, float] = dict(cfg.get("noise_std", {}))

        # Feature scaling parameters
        scaling_cfg = cfg.get("scaling", {})
        self.scaling = ObsScaling(
            altitude_center=float(scaling_cfg.get("altitude_center", 10.0)),
            altitude_scale=float(scaling_cfg.get("altitude_scale", 10.0)),
            vel_scale=tuple(scaling_cfg.get("vel_scale", (20.0, 10.0, 10.0))),
            command_speed_scale=float(scaling_cfg.get("command_speed_scale", 25.0)),
            max_depth=float(scaling_cfg.get("max_depth", 30.0)),
        )

        # Genome configuration ------------------------------------------------
        self.add_genome_obs = bool(add_genome_obs)
        self.genome_vec = genome_vec  # may be None initially
        self.genome_dim: Optional[int] = None
        self.genome_min: Optional[torch.Tensor] = None
        self.genome_max: Optional[torch.Tensor] = None

        if self.add_genome_obs:
            if genome_vec is not None:
                self.genome_dim = int(genome_vec.shape[1])
            elif genome_min is not None and genome_max is not None:
                if len(genome_min) != len(genome_max):
                    raise ValueError("genome_min and genome_max must have the same length")
                self.genome_dim = len(genome_min)
            else:
                raise ValueError(
                    "add_genome_obs=True but neither genome_vec nor (genome_min, genome_max) were provided"
                )

            if genome_min is not None and genome_max is not None:
                min_arr = np.asarray(genome_min, dtype=np.float32)
                max_arr = np.asarray(genome_max, dtype=np.float32)
                if min_arr.shape[0] != self.genome_dim or max_arr.shape[0] != self.genome_dim:
                    raise ValueError("Genome min/max length does not match genome dimension.")
                self.genome_min = torch.from_numpy(min_arr).to(self.device).unsqueeze(0)  # (1, G)
                self.genome_max = torch.from_numpy(max_arr).to(self.device).unsqueeze(0)  # (1, G)

        # ------------------------------------------------------------------
        # Precompute observation dimensions
        # ------------------------------------------------------------------
        base_kin_dim = 8  # z_norm(1) + quat(4) + lin_vel(3)
        depth_dim_actor = self.num_actor_sectors if self.include_depth else 0
        last_act_dim = self.num_actions  # throttle + all joints
        cmd_dim = 1  # forward speed command

        self.actor_obs_dim = base_kin_dim + depth_dim_actor + last_act_dim + cmd_dim
        self.priv_obs_dim = self.actor_obs_dim

        if self.add_genome_obs and self.genome_dim is not None:
            self.actor_obs_dim += self.genome_dim
            self.priv_obs_dim += self.genome_dim

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
        genome_vec: Optional[torch.Tensor] = None,
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

        Returns:
            A tuple ``(obs_actor, obs_critic)``.
        """
        device = self.device
        base_pos = base_pos.to(device)
        base_quat = base_quat.to(device)
        base_lin_vel = base_lin_vel.to(device)
        last_actions = last_actions.to(device)
        commands = commands.to(device)
        if depth_actor is not None:
            depth_actor = depth_actor.to(device)
        if genome_vec is not None:
            self.genome_vec = genome_vec.to(device)

        B = base_pos.shape[0]

        # ----------------------- Base kinematic features --------------------
        # Altitude (z) normalised around a reference height
        z = base_pos[:, 2:3]
        z_norm = (z - self.scaling.altitude_center) / max(1e-6, self.scaling.altitude_scale)

        quat = base_quat  # (B, 4)

        vx = base_lin_vel[:, 0:1]
        vy = base_lin_vel[:, 1:2]
        vz = base_lin_vel[:, 2:3]
        vx_norm = vx / max(1e-6, self.scaling.vel_scale[0])
        vy_norm = vy / max(1e-6, self.scaling.vel_scale[1])
        vz_norm = vz / max(1e-6, self.scaling.vel_scale[2])

        # -------------------------- Depth features --------------------------
        depth_feat_actor = None
        if self.include_depth and depth_actor is not None:
            # Map [0, max_depth] -> [0, 1] with 1 = very close, 0 = no obstacle.
            depth_feat_actor = 1.0 - depth_actor / self.scaling.max_depth

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
            last_jnts = last_jnts_raw / self.joint_limits_max.unsqueeze(0)
        else:
            last_jnts = torch.empty((B, 0), device=device)

        # ------------------------ Command features --------------------------
        v_tgt = commands[:, 0].unsqueeze(1)
        v_tgt_norm = v_tgt / max(1e-6, self.scaling.command_speed_scale)

        # ---------------------- Assemble clean features ---------------------
        components = [z_norm, quat, vx_norm, vy_norm, vz_norm]
        if depth_feat_actor is not None:
            components.append(depth_feat_actor)
        components += [last_thr, last_jnts, v_tgt_norm]

        obs_clean = torch.cat(components, dim=1)
        # --------------------------- Add noise ------------------------------
        obs_actor = obs_clean.clone()
        if self.add_noise and self.noise_std:
            std_cfg = self.noise_std
            idx = 0

            # z_norm
            if std_cfg.get("z", 0.0) > 0.0:
                noise = torch.randn((B, 1), device=device) * std_cfg["z"]
                obs_actor[:, idx : idx + 1] += noise
            idx += 1

            # quat (4)
            if std_cfg.get("quat", 0.0) > 0.0:
                noise = torch.randn((B, 4), device=device) * std_cfg["quat"]
                obs_actor[:, idx : idx + 4] += noise
            idx += 4

            # vel (3)
            if std_cfg.get("vel", 0.0) > 0.0:
                noise = torch.randn((B, 3), device=device) * std_cfg["vel"]
                obs_actor[:, idx : idx + 3] += noise
            idx += 3

            # depth
            if depth_feat_actor is not None:
                depth_dim = depth_feat_actor.shape[1]
                if std_cfg.get("depth", 0.0) > 0.0:
                    noise = torch.randn((B, depth_dim), device=device) * (std_cfg["depth"] * depth_actor / self.scaling.max_depth)
                    obs_actor[:, idx : idx + depth_dim] += noise
                idx += depth_dim

            # last actions (throttle + joints)
            if std_cfg.get("last_actions", 0.0) > 0.0:
                dim_la = self.num_actions
                noise = torch.randn((B, dim_la), device=device) * std_cfg["last_actions"]
                obs_actor[:, idx : idx + dim_la] += noise
            idx += self.num_actions

            # command speed
            if std_cfg.get("command", 0.0) > 0.0:
                noise = torch.randn((B, 1), device=device) * std_cfg["command"]
                obs_actor[:, idx : idx + 1] += noise
            idx += 1

        # Critic gets clean observation; it may also see extra depth sectors.
        obs_critic = obs_clean

        # ------------------------- Genome features --------------------------
        if self.add_genome_obs:
            if self.genome_vec is None:
                raise RuntimeError("add_genome_obs=True but no genome_vec has been provided.")
            genome = self.genome_vec.to(device)
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
                denom = (self.genome_max - self.genome_min).clamp(min=1e-6)
                genome_norm = (genome - self.genome_min) / denom
            else:
                genome_norm = genome

            genome_actor = genome_norm
            if self.add_noise and self.noise_std and self.noise_std.get("genome", 0.0) > 0.0:
                genome_noise = torch.randn_like(genome_norm) * self.noise_std["genome"]
                genome_actor = genome_actor + genome_noise

            obs_actor = torch.cat((obs_actor, genome_actor), dim=1)
            obs_critic = torch.cat((obs_critic, genome_norm), dim=1)

            print(genome_actor[0, :])

        return obs_actor, obs_critic
