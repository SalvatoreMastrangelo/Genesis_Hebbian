"""
MultiDroneEnv — multiple different drones in a single Genesis scene.
====================================================================

Places D = N × S drone entities (each a different URDF) into ONE Genesis
scene with E environments per entity, for a total of D × E drone instances
stepped by a single ``scene.step()`` call.

Each drone entity gets its own independent:
- Aero solver instance (forces applied only to its links)
- DOF indices (servo control)
- Collision detection (geometric, per-drone against shared forest)
- Observation builder (36D obs vector)
- Metric accumulators

Drones are spawned at the same position and do not interact
(``enable_collision=False``).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat
from genesis.assets.urdf.aero_model import DroneAeroModel

from morph_evolution.chromosome_drone import Chromosome_Drone
from winged_drone_train.aero_profile import resolve_aero_config
from winged_drone_train.env import _servo_gains_from_catalog
from winged_drone_train.perception import forest as forest_utils
from winged_drone_train.perception import depth as depth_utils
from winged_drone_train.perception import obs as obs_utils
from winged_drone_train.control.power import scale_and_clamp_actions, ActuatorDynamics, LatencyConfig
from winged_drone_train.control import power as power_utils


def _resolve_servo_joint_names(urdf_path: str) -> List[str]:
    """Auto-detect servo joint names from URDF (same logic as WingedDroneEnv)."""
    import xml.etree.ElementTree as ET

    def _classify(name: str):
        ln = name.lower()
        if "sweep" in ln:
            return "sweep_left" if "left" in ln else ("sweep_right" if "right" in ln else "sweep")
        if "twist" in ln:
            return "twist_left" if "left" in ln else ("twist_right" if "right" in ln else "twist")
        if "elevator" in ln:
            return "elevator"
        if "rudder" in ln:
            return "rudder"
        return None

    root = ET.parse(str(urdf_path)).getroot()
    role_to_name = {}
    for joint in root.findall(".//joint"):
        jtype = (joint.get("type") or "").strip().lower()
        if jtype not in ("revolute", "continuous"):
            continue
        name = joint.get("name")
        if not name:
            continue
        role = _classify(name)
        if role and role not in role_to_name:
            role_to_name[role] = name

    ordered = ["sweep_left", "sweep_right", "twist_left", "twist_right",
               "elevator", "rudder", "sweep", "twist"]
    return [role_to_name[r] for r in ordered if r in role_to_name]



class _DroneState:
    """Per-drone cached state and indices."""
    __slots__ = (
        "entity", "drone_model", "aero_solver",
        "servo_joint_names", "servo_dof_indices", "num_servos",
        "span", "nominal_mass", "naca_code",
        "base_pos", "base_quat", "base_euler", "base_lin_vel",
        "last_actions", "power", "nan_envs",
        "episode_length", "done",
        "pre_collision", "pre_wall_crash", "pre_angle_limit",
        "cylinders_xy", "forest_ids",
        "commands", "obs",
        "obs_builder", "_joint_limits_max", "_joint_limits_min",
    )


class MultiDroneEnv:
    """D different drones in one Genesis scene, E envs per drone.

    Parameters
    ----------
    urdf_paths : list[str]
        Paths to D different URDF files.
    num_envs : int
        Number of parallel environments per drone entity (E).
    wp1_cfg : RunConfig
        WP1 configuration for environment parameters.
    device : str
        Torch device.
    vmin, vmax : float
        Speed command range for evaluation.
    """

    NUM_SECTORS = 20
    CONE_DEG = 80.0
    MAX_DEPTH = 30.0

    def __init__(
        self,
        urdf_paths: List[str],
        num_envs: int,
        wp1_cfg,
        device: str = "cuda:0",
        vmin: float = 6.0,
        vmax: float = 30.0,
        record: bool = False,
    ):
        self.D = len(urdf_paths)
        self.E = num_envs
        self.urdf_paths = urdf_paths
        self.device = torch.device(device)
        self.vmin = vmin
        self.vmax = vmax

        # Extract env parameters from WP1 config
        env_cfg = wp1_cfg.to_env_cfg()
        obs_cfg = wp1_cfg.to_obs_cfg()
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg

        # Use dt from WP1 config if available; fallback to 25 Hz (matches obs/activation sampling)
        self.dt = float(env_cfg.get("dt", 0.04))

        # Compute substeps from dt and desired physics rate (100 Hz = 0.01 s per physics step)
        physics_dt = 0.01
        substeps = max(1, round(self.dt / physics_dt))

        episode_length_s = float(env_cfg.get("episode_length_s", 20.0))
        self.max_episode_length = math.ceil(episode_length_s / self.dt)

        self._tree_radius = float(env_cfg.get("tree_radius", 0.75))
        self._collision_tol = 0.01  # eval tolerance
        self._termination_abs_y_max = float(env_cfg.get("termination_if_y_greater_than", 100.0))
        self._termination_min_z = float(env_cfg.get("termination_if_close_to_ground", 0.1))
        self._roll_limit = math.radians(100.0)
        self._pitch_limit = math.radians(90.0)
        self._success_x = float(env_cfg.get("x_upper", 500.0))

        base_init_pos = torch.tensor(
            env_cfg.get("base_init_pos", [-30.0, 0.0, 15.0]),
            device=self.device, dtype=torch.float32,
        )
        base_init_quat = torch.tensor(
            env_cfg.get("base_init_quat", [1.0, 0.0, 0.0, 0.0]),
            device=self.device, dtype=torch.float32,
        )
        self.base_init_pos = base_init_pos
        self.base_init_quat = base_init_quat
        self.inv_base_init_quat = inv_quat(base_init_quat)

        # ------------------------------------------------------------------ #
        # Build Genesis scene with D drone entities                          #
        # ------------------------------------------------------------------ #
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=15,
                camera_pos=(-35.0, 0.0, 15.0),
                camera_lookat=(-28.0, 0.0, 10.0),
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
                show_world_frame=False,
                plane_reflection=False,
                shadow=False,
                enable_rendering=record,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.CG,
                enable_collision=False,
                enable_joint_limit=True,
            ),
            show_viewer=False,
            renderer=gs.renderers.Rasterizer(),
        )

        # Add all D drone entities
        self.drones: List[_DroneState] = []
        for i, urdf_path in enumerate(urdf_paths):
            ds = _DroneState()

            # Resolve aero config and model
            aero_config = resolve_aero_config("simple")
            aero_yaml = Path(urdf_path).parent / "aero_parameters.yaml"
            if aero_yaml.is_file():
                import yaml as _yaml
                with open(aero_yaml) as f:
                    loaded = _yaml.safe_load(f)
                if isinstance(loaded, dict):
                    aero_config = loaded

            ds.drone_model = DroneAeroModel(urdf_path, config_override=aero_config)
            ds.servo_joint_names = _resolve_servo_joint_names(urdf_path)

            urdf_args = {
                "file": urdf_path,
                "pos": base_init_pos.cpu().numpy(),
                "quat": base_init_quat.cpu().numpy(),
                "collision": False,
                "merge_fixed_links": True,
            }
            if ds.drone_model is not None:
                urdf_args["links_to_keep"] = ds.drone_model.required_links(ds.servo_joint_names)

            ds.entity = self.scene.add_entity(gs.morphs.URDF(**urdf_args))

            # NACA code from URDF filename
            match = re.search(r"\[([^\]]+)\]\.urdf$", urdf_path)
            ds.naca_code = None
            if match:
                try:
                    vals = [float(x) for x in match.group(1).split(",")]
                    ds.naca_code = Chromosome_Drone.naca_from_physical(vals)
                except Exception:
                    pass

            self.drones.append(ds)

        # Shared forest (geometry only)
        (
            self.cylinders_array,
            self.total_forests,
            self._forest_generator,
        ) = forest_utils.generate_forests(
            num_envs=self.E,
            evaluation=True,
            unique_forests_eval=True,
            growing_forest=bool(env_cfg.get("growing_forest", True)),
            env_cfg=env_cfg,
            device=self.device,
        )

        # Depth solver (shared across all drones)
        y_lower = float(env_cfg.get("y_lower", -50.0))
        y_upper = float(env_cfg.get("y_upper", 50.0))
        self.depth_solver = depth_utils.DepthSolver(
            num_sectors=self.NUM_SECTORS,
            cone_angle_deg=self.CONE_DEG,
            max_distance=self.MAX_DEPTH,
            short_range=2.0,
            tree_radius=self._tree_radius,
            y_lower=y_lower,
            y_upper=y_upper,
            torch_device=self.device,
        )

        # ------------------------------------------------------------------ #
        # Optional follow-camera (must be added before scene.build())        #
        # ------------------------------------------------------------------ #
        self.rec_cam = None
        if record:
            cam_res = tuple(env_cfg.get("camera_res", (1024, 768)))
            cam_fov = float(env_cfg.get("camera_fov", 80.0))
            pos0 = tuple(base_init_pos.cpu().tolist())
            pos0 = (pos0[0] - 3.0, pos0[1], pos0[2] + 1.0)
            self.rec_cam = self.scene.add_camera(
                res=cam_res,
                pos=pos0,
                lookat=tuple(base_init_pos.cpu().tolist()),
                fov=cam_fov,
                GUI=False,
            )
        self._rec_cam_pos_filtered: Optional[torch.Tensor] = None
        self._rec_cam_lookat_filtered: Optional[torch.Tensor] = None

        # Spawn forest tree visuals (before scene.build())
        self._spawn_tree_visuals(env_cfg)

        # ------------------------------------------------------------------ #
        # Build scene (replicates all D entities × E envs)                   #
        # ------------------------------------------------------------------ #
        print(f"[MultiDroneEnv] Building scene: D={self.D} entities, E={self.E} envs")
        self.scene.build(n_envs=self.E)
        self.rigid_solver = self.scene.sim.rigid_solver

        # ------------------------------------------------------------------ #
        # Per-drone post-build setup                                         #
        # ------------------------------------------------------------------ #
        from genesis.engine.solvers.drones.simple_drone import SimpleDroneAeroSolver

        for i, ds in enumerate(self.drones):
            # Create independent aero solver instance
            ds.aero_solver = SimpleDroneAeroSolver(self.scene, self.scene.sim)
            ds.aero_solver.add_target(ds.entity, drone_model=ds.drone_model)
            if ds.naca_code and hasattr(ds.aero_solver, "apply_naca_wing_override"):
                ds.aero_solver.apply_naca_wing_override(ds.naca_code)
            ds.aero_solver.build()
            self.scene.sim._active_solvers.append(ds.aero_solver)

            ds.span = ds.aero_solver.tip_to_tip
            ds.nominal_mass = float(sum(link.get_mass() for link in ds.entity.links))

            # Servo DOF indices
            ds.servo_dof_indices = []
            for name in ds.servo_joint_names:
                joint = ds.entity.get_joint(name)
                idx = getattr(joint, "dofs_idx_local", None)
                if idx is None:
                    idx = joint.dof_idx_local
                if isinstance(idx, (list, tuple)):
                    idx = idx[0]
                ds.servo_dof_indices.append(int(idx))
            ds.num_servos = len(ds.servo_dof_indices)

            # PD gains
            kp, kv = _servo_gains_from_catalog(ds.drone_model, ds.servo_joint_names, self.device)
            if ds.servo_dof_indices:
                ds.entity.set_dofs_kp(kp, ds.servo_dof_indices)
                ds.entity.set_dofs_kv(kv, ds.servo_dof_indices)

            # Joint limits for observation normalization and action scaling
            if ds.servo_dof_indices:
                joint_mins, joint_maxs = ds.entity.get_dofs_limit(ds.servo_dof_indices)
                ds._joint_limits_min = torch.as_tensor(joint_mins, device=self.device, dtype=torch.float32)
                ds._joint_limits_max = torch.as_tensor(joint_maxs, device=self.device, dtype=torch.float32)
            else:
                ds._joint_limits_min = torch.zeros((0,), device=self.device, dtype=torch.float32)
                ds._joint_limits_max = torch.zeros((0,), device=self.device, dtype=torch.float32)

            # Allocate per-drone state buffers
            ds.base_pos = torch.zeros(self.E, 3, device=self.device)
            ds.base_quat = torch.zeros(self.E, 4, device=self.device)
            ds.base_euler = torch.zeros(self.E, 3, device=self.device)
            ds.base_lin_vel = torch.zeros(self.E, 3, device=self.device)
            ds.last_actions = torch.zeros(self.E, 1 + ds.num_servos, device=self.device)
            ds.power = torch.zeros(self.E, device=self.device)
            ds.nan_envs = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.episode_length = torch.zeros(self.E, dtype=torch.long, device=self.device)
            ds.done = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.pre_collision = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.pre_wall_crash = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.pre_angle_limit = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.commands = torch.zeros(self.E, 1, device=self.device)

            # Forest assignment per env
            ds.forest_ids = torch.randint(0, self.total_forests, (self.E,),
                                          device=self.device, dtype=torch.long)
            if self.cylinders_array is not None:
                ds.cylinders_xy = self.cylinders_array[ds.forest_ids, :, :2]
            else:
                ds.cylinders_xy = None

            print(f"  drone[{i}]: span={ds.span:.3f}m  mass={ds.nominal_mass:.3f}kg  "
                  f"servos={ds.num_servos}  naca={ds.naca_code}")

        # Number of actions = max across all drones (throttle + servos)
        self.num_actions = max(1 + ds.num_servos for ds in self.drones)
        num_servos_max = self.num_actions - 1

        # Build ObservationBuilders with uniform num_actions so all drones
        # produce the same obs_dim (matching the WP1 checkpoint expectation).
        for ds in self.drones:
            # Pad joint limits to num_servos_max (default 1.0 for missing servos)
            jlim = ds._joint_limits_max
            if jlim.numel() < num_servos_max:
                pad = torch.ones(num_servos_max - jlim.numel(), device=self.device)
                jlim = torch.cat([jlim, pad])
            ds.obs_builder = obs_utils.ObservationBuilder(
                num_actions=self.num_actions,
                num_sectors_actor=self.NUM_SECTORS,
                joint_limits_max=jlim,
                obs_cfg=self.obs_cfg,
                include_depth=True,
                device=self.device,
            )

        # Obs dim from the ObservationBuilder (matches WP1 training)
        self.obs_dim = self.drones[0].obs_builder.actor_obs_dim

        # Create shared ActuatorDynamics with WP1 default latency config
        # (simulates realistic actuator response delays, matching WP1 training)
        latency_cfg = LatencyConfig(
            simulate_latency=bool(env_cfg.get("simulate_action_latency", True)),
            latency_min=int(env_cfg.get("action_latency_min_steps", 0)),
            latency_max=int(env_cfg.get("action_latency_max_steps", 1)),
            random_latency_per_step=bool(env_cfg.get("action_latency_random_per_step", False)),
        )
        self.actuator = ActuatorDynamics(
            num_envs=self.D * self.E,
            num_actions=self.num_actions,
            throttle_limit=(0.0, 1.0),
            joint_limits=None,  # will use defaults
            latency_cfg=latency_cfg,
            device=self.device,
        )

        # Allocate output buffers: (D, E, ...)
        self.obs_buf = torch.zeros(self.D, self.E, self.obs_dim, device=self.device)
        self.rew_buf = torch.zeros(self.D, self.E, device=self.device)
        self.done_buf = torch.zeros(self.D, self.E, dtype=torch.bool, device=self.device)

        print(f"[MultiDroneEnv] Ready: {self.D} drones × {self.E} envs = "
              f"{self.D * self.E} instances")

    # ================================================================== #
    #  Step                                                               #
    # ================================================================== #

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        """Advance all drones by one timestep.

        Parameters
        ----------
        actions : (D, E, num_actions)
            Per-drone, per-env actions. action[:, :, 0] = throttle [0,1],
            action[:, :, 1:] = servo targets.

        Returns
        -------
        obs : (D, E, obs_dim)
        rewards : (D, E)
        dones : (D, E) bool
        info : dict
        """
        # 1. Flatten actions to (D*E, num_actions) for batch processing through ActuatorDynamics
        B = self.D * self.E
        actions_flat = actions.reshape(B, self.num_actions).to(self.device)

        # Process all actions through ActuatorDynamics (applies scaling, clamping, and latency)
        # Returns: (servo_targets, throttle) where servo_targets is (B, num_actions-1), throttle is (B,)
        servo_targets_flat, throttle_flat = self.actuator.process_actions(actions_flat)

        # Reshape back to (D, E, ...)
        throttle = throttle_flat.reshape(self.D, self.E)
        servo_targets = servo_targets_flat.reshape(self.D, self.E, -1)

        # 2. Apply controls per drone
        for i, ds in enumerate(self.drones):
            thr_i = throttle[i]  # (E,)
            servo_i = servo_targets[i]  # (E, num_actions-1) padded

            # Extract this drone's actual servo targets (without padding)
            num_act_drone = 1 + ds.num_servos
            servo_targets_drone = servo_i[:, :ds.num_servos]

            # Apply to simulator
            ds.aero_solver.set_throttle(thr_i)
            if ds.num_servos > 0:
                ds.entity.control_dofs_position(servo_targets_drone, ds.servo_dof_indices)

            # Store applied actions in physical range (throttle + servo angles)
            ds.last_actions[:] = 0.0
            ds.last_actions[:, 0] = thr_i
            if ds.num_servos > 0:
                ds.last_actions[:, 1:] = servo_targets_drone

        # 2. Single physics step for ALL D×E instances
        self.scene.step()

        # 3. Update state, obs, termination, rewards per drone
        for i, ds in enumerate(self.drones):
            self._update_state(ds)
            self._compute_obs(i, ds)
            term = self._check_termination(ds)
            ds.episode_length += 1
            ds.done |= term
            self.done_buf[i] = ds.done

        return self.obs_buf, self.rew_buf, self.done_buf, {}

    # ================================================================== #
    #  Reset                                                              #
    # ================================================================== #

    @torch.no_grad()
    def reset(self):
        """Reset all drones and environments.

        Returns
        -------
        obs : (D, E, obs_dim)
        info : dict
        """
        base_euler_init = quat_to_xyz(self.base_init_quat.unsqueeze(0), rpy=True, degrees=False)

        for i, ds in enumerate(self.drones):
            # Build full DOF state: [base_pos(3), base_euler(3), joints...]
            n_dofs = ds.entity.n_dofs
            dof_pos = torch.zeros(self.E, n_dofs, device=self.device)
            dof_pos[:, :3] = self.base_init_pos.unsqueeze(0).expand(self.E, -1)
            dof_pos[:, 3:6] = base_euler_init.expand(self.E, -1)
            # joints (DOFs 6+) stay zero

            dof_vel = torch.zeros(self.E, n_dofs, device=self.device)
            # Initial forward velocity (like WP1 eval: 15 m/s)
            dof_vel[:, 0] = 15.0

            ds.entity.set_dofs_position(dof_pos)
            ds.entity.set_dofs_velocity(dof_vel)

            # Reset buffers
            ds.episode_length.zero_()
            ds.done.zero_()
            ds.nan_envs.zero_()
            ds.last_actions.zero_()
            ds.pre_collision.zero_()
            ds.pre_wall_crash.zero_()
            ds.pre_angle_limit.zero_()

            # Randomize speed commands
            ds.commands[:, 0] = torch.empty(self.E, device=self.device).uniform_(
                self.vmin, self.vmax
            )

            # Reassign forests
            ds.forest_ids = torch.randint(0, self.total_forests, (self.E,),
                                          device=self.device, dtype=torch.long)
            if self.cylinders_array is not None:
                ds.cylinders_xy = self.cylinders_array[ds.forest_ids, :, :2]

            self._update_state(ds)
            self._compute_obs(i, ds)

        # Reset actuator dynamics for all envs
        all_env_ids = torch.arange(self.D * self.E, device=self.device, dtype=torch.long)
        self.actuator.reset_envs(all_env_ids)

        return self.obs_buf, {}

    # ================================================================== #
    #  State extraction                                                   #
    # ================================================================== #

    def _update_state(self, ds: _DroneState):
        """Extract entity position, orientation, velocity from rigid solver."""
        ds.base_pos[:] = ds.entity.get_pos()      # (E, 3)
        ds.base_quat[:] = ds.entity.get_quat()    # (E, 4)
        ds.base_euler[:] = quat_to_xyz(ds.base_quat)

        # Velocity in world frame (matching WP1 training env convention)
        ds.base_lin_vel[:] = ds.entity.get_vel()[:, :3]  # (E, 3)

        # Check for NaN
        nan_mask = torch.isnan(ds.base_pos).any(dim=1) | torch.isnan(ds.base_quat).any(dim=1)
        ds.nan_envs[:] = nan_mask

        # Power
        ds.power[:] = self._compute_power(ds)

    def _compute_power(self, ds: _DroneState) -> torch.Tensor:
        """Simplified power computation from thrust."""
        cached_thrust = getattr(ds.aero_solver, "_thrust_n_buf", None)
        if torch.is_tensor(cached_thrust) and cached_thrust.shape[0] == self.E:
            thrust = cached_thrust.clone()
        else:
            thrust = torch.zeros(self.E, device=self.device)
        # Simple power model: P = T * v_axial (approximate)
        v_x = ds.base_lin_vel[:, 0].abs()
        return (thrust * v_x).clamp(min=0.0)

    # ================================================================== #
    #  Observation                                                        #
    # ================================================================== #

    def _compute_obs(self, drone_idx: int, ds: _DroneState):
        """Build observation for one drone using the same ObservationBuilder as WP1."""
        # Depth sectors
        if ds.cylinders_xy is not None:
            depth = self.depth_solver.compute_depth(
                base_pos=ds.base_pos,
                base_euler=ds.base_euler,
                cyl_xy_b=ds.cylinders_xy,
            )
        else:
            depth = torch.full(
                (self.E, self.NUM_SECTORS), self.MAX_DEPTH,
                device=self.device, dtype=torch.float32,
            )

        # Pad last_actions to uniform num_actions (throttle + max servos)
        la = torch.zeros(self.E, self.num_actions, device=self.device)
        la[:, :ds.last_actions.shape[1]] = ds.last_actions

        obs_actor, _ = ds.obs_builder.build_observations(
            base_pos=ds.base_pos,
            base_quat=ds.base_quat,
            base_lin_vel=ds.base_lin_vel,
            last_actions=la,
            commands=ds.commands,
            depth_actor=depth,
        )
        self.obs_buf[drone_idx] = obs_actor

    # ================================================================== #
    #  Termination                                                        #
    # ================================================================== #

    def _check_termination(self, ds: _DroneState) -> torch.Tensor:
        """Check per-env termination conditions for one drone."""
        term = torch.zeros(self.E, dtype=torch.bool, device=self.device)

        # Collision with trees
        if ds.cylinders_xy is not None:
            collided = self._check_collision(ds)
            ds.pre_collision[:] = collided
            term |= collided

        # Wall/ground crash
        wall = (
            (ds.base_pos[:, 1].abs() > self._termination_abs_y_max) |
            (ds.base_pos[:, 2] < self._termination_min_z)
        )
        ds.pre_wall_crash[:] = wall
        term |= wall

        # Angle limits
        roll_abs = ds.base_euler[:, 0].abs()
        pitch_abs = ds.base_euler[:, 1].abs()
        angle_term = (roll_abs > self._roll_limit) | (pitch_abs > self._pitch_limit)
        ds.pre_angle_limit[:] = angle_term
        term |= angle_term

        # Success
        term |= ds.base_pos[:, 0] > self._success_x

        # Timeout
        term |= ds.episode_length >= self.max_episode_length

        # NaN
        term |= ds.nan_envs

        return term

    def _check_collision(self, ds: _DroneState) -> torch.Tensor:
        """Geometric collision check: drone rectangle vs tree cylinders."""
        if ds.cylinders_xy is None:
            return torch.zeros(self.E, dtype=torch.bool, device=self.device)

        half_span = ds.span / 2.0
        r_tree = self._tree_radius + self._collision_tol

        # Relative positions: (E, T, 2)
        drone_xy = ds.base_pos[:, :2].unsqueeze(1)  # (E, 1, 2)
        diff = ds.cylinders_xy - drone_xy            # (E, T, 2)

        yaw = ds.base_euler[:, 2]
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)

        # Transform to body frame
        x_b = diff[..., 0] * cos_y.unsqueeze(1) + diff[..., 1] * sin_y.unsqueeze(1)
        y_b = -diff[..., 0] * sin_y.unsqueeze(1) + diff[..., 1] * cos_y.unsqueeze(1)

        roll = ds.base_euler[:, 0].abs()
        half_span_eff = half_span * torch.cos(roll).unsqueeze(1)

        hit_x = (x_b >= -r_tree) & (x_b <= r_tree)
        hit_y = y_b.abs() <= (half_span_eff + r_tree)

        return (hit_x & hit_y).any(dim=1)

    # ================================================================== #
    #  Follow-camera                                                      #
    # ================================================================== #

    def update_follow_camera(self, target_drone_idx: int = 0) -> None:
        """Smoothly move rec_cam to follow env-0 of the given drone index."""
        if self.rec_cam is None:
            return

        ds = self.drones[target_drone_idx]
        pos = ds.base_pos[0]  # env 0, shape (3,)

        dist_back = float(self.env_cfg.get("rec_cam_follow_distance", 3.0))
        height_offset = float(self.env_cfg.get("rec_cam_follow_height", 1.0))
        alpha = float(self.env_cfg.get("rec_cam_smooth_alpha", 0.25))

        target_cam_pos = torch.tensor(
            [pos[0].item() - dist_back, pos[1].item(), pos[2].item() + height_offset],
            device=self.device, dtype=torch.float32,
        )
        lookat_offset = torch.tensor([1.5, 0.0, 0.0], device=self.device, dtype=torch.float32)
        target_lookat = pos + lookat_offset

        if self._rec_cam_pos_filtered is None:
            self._rec_cam_pos_filtered = target_cam_pos.clone()
            self._rec_cam_lookat_filtered = target_lookat.clone()
        else:
            self._rec_cam_pos_filtered = (
                alpha * target_cam_pos + (1.0 - alpha) * self._rec_cam_pos_filtered
            )
            self._rec_cam_lookat_filtered = (
                alpha * target_lookat + (1.0 - alpha) * self._rec_cam_lookat_filtered
            )

        self.rec_cam.set_pose(
            pos=tuple(self._rec_cam_pos_filtered.detach().cpu().tolist()),
            lookat=tuple(self._rec_cam_lookat_filtered.detach().cpu().tolist()),
        )

    def start_recording(self, path: str) -> None:
        """Begin recording from the follow-camera."""
        if self.rec_cam is not None:
            self._rec_cam_pos_filtered = None
            self._rec_cam_lookat_filtered = None
            self.rec_cam.start_recording()
            self._recording_path = path

    def stop_recording(self, fps: int = 25) -> None:
        """Stop recording and save MP4."""
        if self.rec_cam is not None and hasattr(self, "_recording_path"):
            self.rec_cam.stop_recording(self._recording_path, fps=fps)

    def render_frame(self) -> None:
        """Render one frame from the follow-camera."""
        if self.rec_cam is not None:
            self.rec_cam.render()

    def _spawn_tree_visuals(self, env_cfg: Dict) -> None:
        """Create cylinder entities for forest visualization in Genesis.

        Called before scene.build() to add visual geometry for trees.
        This allows the Genesis renderer to display the forest in videos.
        """
        tree_radius = float(env_cfg.get("tree_radius", 0.75))
        tree_height = float(env_cfg.get("tree_height", 100.0))

        # Use the first forest layout for visuals
        if self.cylinders_array is None or self.cylinders_array.shape[1] == 0:
            return

        forest0 = self.cylinders_array[0]  # (T, 3)
        for pos in forest0:
            self.scene.add_entity(
                gs.morphs.Cylinder(
                    pos=(float(pos[0]), float(pos[1]), float(tree_height / 2.0)),
                    radius=tree_radius,
                    height=tree_height,
                    collision=False,
                    fixed=True,
                )
            )

