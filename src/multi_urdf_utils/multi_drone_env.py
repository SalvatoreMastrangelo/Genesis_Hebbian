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
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, xyz_to_quat
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
        # --- URDF and genome ---
        "urdf_path", "_genome_vec",
        # --- Entity / physics ---
        "entity", "drone_model", "aero_solver",
        "servo_joint_names", "servo_dof_indices", "num_servos",
        "span", "nominal_mass", "naca_code",
        # --- Kinematics ---
        "base_pos", "base_quat", "base_euler", "base_lin_vel", "base_ang_vel",
        "joint_position", "joint_velocity", "torque",
        # --- Actions / control ---
        "last_actions", "prev_actions", "power", "nan_envs",
        # --- Episode bookkeeping ---
        "episode_length", "done",   # legacy (benchmark compat)
        "reset_buf",                # (E,) bool — envs that fired reset this step
        "max_episode_length_per_env",
        # --- Termination flags ---
        "success", "collision", "wall_crash_condition", "angle_limit_condition",
        "pre_collision", "pre_wall_crash", "pre_angle_limit",
        "pre_success", "pre_nan",
        "_time_outs",
        # --- Rewards ---
        "rew_buf",
        "episode_sums", "last_reward_components", "last_reward_total",
        # --- Observations ---
        "priv_obs_buf",
        "commands", "obs",
        "obs_builder", "_joint_limits_max", "_joint_limits_min",
        "actuator",
        # --- Obstacle reward ray dirs ---
        "ray_dir_x", "ray_dir_y",
        # --- Height reward ---
        "target_height", "_success_x_limit_train",
        # --- Power model caches ---
        "_drone_name", "_aero_config_dict",
        "_thrust_buf", "_prop_rpm_buf", "_prop_axial_speed_buf",
        "thrust_log", "prop_rpm_log", "prop_axial_speed_log",
        "_power_prop_coeffs", "_power_servo_constants", "_power_torque_multipliers",
        "_prop_diameter",
        # Episode stat cache (filled during partial reset, consumed by step())
        "_ep_dict_cache",
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
        training_mode: bool = False,
    ):
        self.D = len(urdf_paths)
        self.E = num_envs
        self.urdf_paths = urdf_paths
        self.device = torch.device(device)
        self.vmin = vmin
        self.vmax = vmax
        self.training_mode = training_mode

        # Extract env parameters from WP1 config
        env_cfg = wp1_cfg.to_env_cfg()
        obs_cfg = wp1_cfg.to_obs_cfg()
        reward_cfg = wp1_cfg.to_reward_cfg()
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg

        # ------------------------------------------------------------------ #
        # Reward configuration (training mode only, but stored for all modes)#
        # ------------------------------------------------------------------ #
        self.reward_scales: Dict[str, float] = reward_cfg.get("reward_scales", {})
        self.reward_names: List[str] = list(self.reward_scales.keys())

        # WP1 WingedDroneEnv hardcodes control at 25 Hz (dt=0.04s, substeps=4) regardless of
        # the "dt" field in env_cfg, which stores the *physics* timestep (0.01s), not the
        # control timestep.  MultiDroneEnv must match WP1's control rate exactly.
        physics_dt = 0.01
        control_hz = 25
        self.dt = 1.0 / control_hz          # 0.04 s — matches WP1 hardcoded value
        substeps = round(self.dt / physics_dt)  # 4 — matches WP1 hardcoded substeps

        episode_length_s = float(env_cfg.get("episode_length_s", 20.0))
        self.max_episode_length = math.ceil(episode_length_s / self.dt)

        self._tree_radius = float(env_cfg.get("tree_radius", 0.75))
        # Collision tolerance: lenient during training (0.2), tight during eval (0.01)
        self._collision_tol = 0.2 if training_mode else 0.01
        self._termination_abs_y_max = float(env_cfg.get("termination_if_y_greater_than", 100.0))
        self._termination_min_z = float(env_cfg.get("termination_if_close_to_ground", 0.1))
        # Angle limits — training mode uses WingedDroneEnv training values
        self._roll_limit_train = math.radians(90.0)
        self._pitch_limit_train = math.radians(90.0)
        self._yaw_limit_train = math.radians(90.0)
        self._roll_limit_eval = math.radians(100.0)
        self._pitch_limit_eval = math.radians(90.0)
        self._yaw_limit_eval = math.radians(90.0)
        # Backward-compat aliases
        self._roll_limit = self._roll_limit_eval
        self._pitch_limit = self._pitch_limit_eval
        self._success_x = float(env_cfg.get("x_upper", 500.0))
        # Training uses forest_x_limit as success threshold (same as WingedDroneEnv)
        self._success_x_limit_train = float(env_cfg.get("forest_x_limit", 150.0))
        self._success_x_limit_eval = float(env_cfg.get("x_upper", 500.0))
        # Target height for height reward (init altitude)
        _init_pos = env_cfg.get("base_init_pos", [-30.0, 0.0, 15.0])
        self.target_height = float(_init_pos[2])
        # Drone type (for power_utils)
        self._drone_name = str(env_cfg.get("drone", "morphing_drone"))

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
            ds.urdf_path = urdf_path

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
        # In training mode use the same randomized forest generation as WingedDroneEnv
        (
            self.cylinders_array,
            self.total_forests,
            self._forest_generator,
        ) = forest_utils.generate_forests(
            num_envs=self.E,
            evaluation=not training_mode,
            unique_forests_eval=not training_mode,
            growing_forest=bool(env_cfg.get("growing_forest", True)),
            env_cfg=env_cfg,
            device=self.device,
        )

        # Scene-level shared forest assignment (all N drones in the same env slot see the same forest)
        self.forest_ids = torch.randint(
            0, self.total_forests, (self.E,), device=self.device, dtype=torch.long
        )
        if self.cylinders_array is not None:
            self.cylinders_xy = self.cylinders_array[self.forest_ids, :, :2]  # (E, T, 2)
        else:
            self.cylinders_xy = None

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

        # Ground plane (visual only, for rendering)
        x_min, x_max = -200.0, 1200.0
        y_min, y_max = -200.0, 200.0
        center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, 0.0)
        size_xy = (x_max - x_min, y_max - y_min)
        self.scene.add_entity(
            gs.morphs.Box(
                pos=center,
                size=(size_xy[0], size_xy[1], 0.01),
                collision=False,
                fixed=True,
            )
        )

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
            ds._prop_diameter = torch.tensor(
                [2.0 * float(ds.aero_solver.prop_radius)],
                device=self.device, dtype=torch.float32,
            )

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
            ds.prev_actions = torch.zeros(self.E, 1 + ds.num_servos, device=self.device)
            ds.power = torch.zeros(self.E, device=self.device)
            ds.nan_envs = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.episode_length = torch.zeros(self.E, dtype=torch.long, device=self.device)
            ds.done = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.pre_collision = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.pre_wall_crash = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.pre_angle_limit = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.commands = torch.zeros(self.E, 1, device=self.device)
            # --- Training-mode additions ---
            ds.base_ang_vel = torch.zeros(self.E, 3, device=self.device)
            # joint state buffers (filled in _update_state when num_servos > 0)
            _ns = ds.num_servos
            ds.joint_position = torch.zeros(self.E, _ns, device=self.device)
            ds.joint_velocity = torch.zeros(self.E, _ns, device=self.device)
            ds.torque = torch.zeros(self.E, _ns, device=self.device)
            # Termination flags
            ds.success = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.collision = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.wall_crash_condition = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.angle_limit_condition = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.pre_success = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.pre_nan = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds.reset_buf = torch.zeros(self.E, dtype=torch.bool, device=self.device)
            ds._time_outs = torch.zeros(self.E, device=self.device)
            ds.max_episode_length_per_env = torch.full(
                (self.E,), self.max_episode_length, device=self.device, dtype=torch.long
            )
            # Reward buffers (populated after obs builders are set up below)
            ds.rew_buf = torch.zeros(self.E, device=self.device)
            ds.last_reward_total = torch.zeros(self.E, device=self.device)
            ds.last_reward_components = torch.zeros(
                self.E, max(len(self.reward_names), 1), device=self.device
            )
            ds.episode_sums = {
                name: torch.zeros(self.E, device=self.device)
                for name in self.reward_names
            }
            # Target height and success threshold
            ds.target_height = self.target_height
            ds._success_x_limit_train = self._success_x_limit_train
            # Power model buffers (shapes finalised after aero solver build)
            ds._thrust_buf = torch.zeros(self.E, device=self.device)
            ds._prop_rpm_buf = torch.zeros(self.E, device=self.device)
            ds._prop_axial_speed_buf = torch.zeros(self.E, device=self.device)
            ds.thrust_log = torch.zeros(self.E, 1, device=self.device)
            ds.prop_rpm_log = torch.zeros(self.E, 1, device=self.device)
            ds.prop_axial_speed_log = torch.zeros(self.E, 1, device=self.device)
            ds._power_prop_coeffs = None
            ds._power_servo_constants = None
            ds._power_torque_multipliers = None
            ds._prop_diameter = None  # filled after aero solver build
            ds._drone_name = self._drone_name
            ds._ep_dict_cache = {}
            # aero_config resolved above per-drone; stored for power_utils
            ds._aero_config_dict = resolve_aero_config("simple")
            _aero_yaml = Path(urdf_paths[i]).parent / "aero_parameters.yaml"
            if _aero_yaml.is_file():
                import yaml as _yaml_power
                with open(_aero_yaml) as _f:
                    _loaded = _yaml_power.safe_load(_f)
                if isinstance(_loaded, dict):
                    ds._aero_config_dict = _loaded


            print(f"  drone[{i}]: span={ds.span:.3f}m  mass={ds.nominal_mass:.3f}kg  "
                  f"servos={ds.num_servos}  naca={ds.naca_code}")

        # Number of actions = max across all drones (throttle + servos)
        self.num_actions = max(1 + ds.num_servos for ds in self.drones)
        num_servos_max = self.num_actions - 1

        # Extract genome vectors from each drone's URDF filename and cache them
        genome_min, genome_max = Chromosome_Drone.genome_min_max()
        for ds in self.drones:
            ds._genome_vec: Optional[torch.Tensor] = None
            match = re.search(r"\[([^\]]+)\]\.urdf$", ds.urdf_path)
            if match:
                try:
                    values = [float(x) for x in match.group(1).split(",")]
                    g = torch.tensor(values, dtype=torch.float32, device=self.device)
                    ds._genome_vec = g.unsqueeze(0).repeat(self.E, 1)
                except Exception:
                    pass

        # Build ObservationBuilders with uniform num_actions so all drones
        # produce the same obs_dim (matching the WP1 checkpoint expectation).
        for ds in self.drones:
            # Pad joint limits to num_servos_max (default 1.0 for missing servos)
            jlim = ds._joint_limits_max
            if jlim.numel() < num_servos_max:
                pad = torch.ones(num_servos_max - jlim.numel(), device=self.device)
                jlim = torch.cat([jlim, pad])

            # Extract genome observation flags from obs_cfg
            add_genome_actor = self.obs_cfg.get("add_genome_obs_actor", False) and (ds._genome_vec is not None)
            add_genome_critic = self.obs_cfg.get("add_genome_obs_critic", False) and (ds._genome_vec is not None)

            ds.obs_builder = obs_utils.ObservationBuilder(
                num_actions=self.num_actions,
                num_sectors_actor=self.NUM_SECTORS,
                joint_limits_max=jlim,
                obs_cfg=self.obs_cfg,
                add_genome_obs_actor=add_genome_actor,
                add_genome_obs_critic=add_genome_critic,
                genome_vec=ds._genome_vec,
                genome_min=genome_min if ds._genome_vec is not None else None,
                genome_max=genome_max if ds._genome_vec is not None else None,
                include_depth=True,
                device=self.device,
            )

        # Obs dim from the ObservationBuilder (matches WP1 training)
        self.obs_dim = self.drones[0].obs_builder.actor_obs_dim
        self.priv_obs_dim = self.drones[0].obs_builder.priv_obs_dim
        self.num_obs = self.obs_dim
        self.num_privileged_obs = self.priv_obs_dim

        # Precomputed ray directions for obstacle reward (same geometry as depth solver)
        _angles = torch.linspace(
            -0.5 * math.radians(self.CONE_DEG),
             0.5 * math.radians(self.CONE_DEG),
            self.NUM_SECTORS,
            device=self.device,
            dtype=torch.float32,
        )
        _ray_dir_x = torch.cos(_angles)  # (NUM_SECTORS,)
        _ray_dir_y = torch.sin(_angles)  # (NUM_SECTORS,)
        for ds in self.drones:
            ds.ray_dir_x = _ray_dir_x
            ds.ray_dir_y = _ray_dir_y
            # per-drone privileged obs buffer
            ds.priv_obs_buf = torch.zeros(self.E, self.priv_obs_dim, device=self.device)

        # Per-drone ActuatorDynamics (each with correct URDF joint limits)
        latency_cfg = LatencyConfig(
            simulate_latency=bool(env_cfg.get("simulate_action_latency", True)),
            latency_min=int(env_cfg.get("action_latency_min_steps", 0)),
            latency_max=int(env_cfg.get("action_latency_max_steps", 1)),
            random_latency_per_step=bool(env_cfg.get("action_latency_random_per_step", False)),
        )
        for ds in self.drones:
            # Pad joint limits to num_servos_max (uniform action dim)
            jmin = ds._joint_limits_min
            jmax = ds._joint_limits_max
            if jmin.numel() < num_servos_max:
                pad_min = -torch.ones(num_servos_max - jmin.numel(), device=self.device)
                pad_max = torch.ones(num_servos_max - jmax.numel(), device=self.device)
                jmin = torch.cat([jmin, pad_min])
                jmax = torch.cat([jmax, pad_max])
            ds.actuator = ActuatorDynamics(
                num_envs=self.E,
                num_actions=self.num_actions,
                throttle_limit=(0.0, 1.0),
                joint_limits=(jmin.tolist(), jmax.tolist()),
                latency_cfg=latency_cfg,
                device=self.device,
            )

        # Allocate output buffers: (D, E, ...)
        self.obs_buf = torch.zeros(self.D, self.E, self.obs_dim, device=self.device)
        self.rew_buf = torch.zeros(self.D, self.E, device=self.device)
        self.done_buf = torch.zeros(self.D, self.E, dtype=torch.bool, device=self.device)
        # Depth buffer for video rendering: (D, E, NUM_SECTORS)
        self.depth_buf = torch.zeros(self.D, self.E, self.NUM_SECTORS, device=self.device)

        print(f"[MultiDroneEnv] Building scene: D={self.D} drones × E={self.E} shared envs "
              f"= {self.D * self.E} instances  (all drones in each env slot share the same forest)")

    # ================================================================== #
    #  Step                                                               #
    # ================================================================== #

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        """Advance all drones by one timestep.

        Parameters
        ----------
        actions : (D, E, num_actions)
            Per-drone, per-env actions. action[d, :, 0] = throttle [0,1],
            action[d, :, 1:] = servo targets (in physical joint-limit range).

        Returns (training_mode=True)
        -------
        obs        : (D, E, obs_dim)
        rewards    : (D, E)
        reset_buf  : (D, E) bool
        extras     : dict — "observations" → {"critic": (D,E,priv_obs_dim)},
                             "time_outs"   → (D, E),
                             "episode"     → dict of scalar metrics (when resets occur)

        Returns (training_mode=False, backward compat)
        -------
        obs     : (D, E, obs_dim)
        rewards : (D, E)
        dones   : (D, E) bool
        info    : {}
        """
        # 1. Per-drone action processing (each with its own joint limits / actuator dynamics)
        for i, ds in enumerate(self.drones):
            raw_i = actions[i].to(self.device)       # (E, num_actions)
            ds.prev_actions.copy_(ds.last_actions)   # save t-1 actions for obs

            servo_targets_i, throttle_i = ds.actuator.process_actions(raw_i)
            servo_targets_drone = servo_targets_i[:, :ds.num_servos]

            ds.aero_solver.set_throttle(throttle_i)
            if ds.num_servos > 0:
                ds.entity.control_dofs_position(servo_targets_drone, ds.servo_dof_indices)

            # Store applied (scaled + delayed) actions
            ds.last_actions[:] = 0.0
            ds.last_actions[:, 0] = throttle_i
            if ds.num_servos > 0:
                ds.last_actions[:, 1:1 + ds.num_servos] = servo_targets_drone

        # 2. Single physics step for all D×E instances simultaneously
        self.scene.step()

        # 3. Per-drone: state extraction, obs, termination, rewards, auto-reset
        for i, ds in enumerate(self.drones):
            ds.episode_length += 1
            self._update_state(ds)
            self._compute_obs(i, ds)

            if self.training_mode:
                self._compute_termination_flags(ds)
                self._compute_rewards_drone(i, ds)
                self.rew_buf[i] = ds.rew_buf
                self.done_buf[i] = ds.reset_buf

                # Auto-reset done environments (matching WingedDroneEnv.step)
                reset_env_ids = ds.reset_buf.nonzero(as_tuple=False).flatten()
                if reset_env_ids.numel() > 0:
                    self._reset_drone_idx(ds, reset_env_ids)
                    # Rebuild obs for reset envs so PPO sees a coherent next-obs
                    self._compute_obs(i, ds)
                    # Max depth for freshly reset envs (no obstacle info yet)
                    self.depth_buf[i][reset_env_ids] = self.MAX_DEPTH
            else:
                # Eval / benchmark backward-compat: accumulate done, no auto-reset
                term = self._check_termination(ds)
                ds.done |= term
                self.done_buf[i] = ds.done

        # 4. Build extras dict
        if self.training_mode:
            priv_obs = torch.stack([ds.priv_obs_buf for ds in self.drones], dim=0)
            time_outs = torch.stack([ds._time_outs for ds in self.drones], dim=0)
            extras: Dict[str, Any] = {
                "observations": {"critic": priv_obs},
                "time_outs": time_outs,
            }
            # Merge episode dicts from all drones that had resets this step
            ep_dicts = [ds._ep_dict_cache for ds in self.drones if ds._ep_dict_cache]
            if ep_dicts:
                merged: Dict[str, List[float]] = {}
                for ep in ep_dicts:
                    for k, v in ep.items():
                        merged.setdefault(k, []).append(v)
                extras["episode"] = {k: sum(v) / len(v) for k, v in merged.items()}
                for ds in self.drones:
                    ds._ep_dict_cache = {}
            return self.obs_buf, self.rew_buf, self.done_buf, extras
        else:
            return self.obs_buf, self.rew_buf, self.done_buf, {}

    # ================================================================== #
    #  Reset                                                              #
    # ================================================================== #

    @torch.no_grad()
    def reset(self):
        """Reset all drones and environments.

        Returns (training_mode=True)
        -------
        obs  : (D, E, obs_dim)
        info : {"observations": {"critic": (D, E, priv_obs_dim)}}

        Returns (training_mode=False)
        -------
        obs  : (D, E, obs_dim)
        info : {}
        """
        all_env_ids = torch.arange(self.E, device=self.device, dtype=torch.long)

        for i, ds in enumerate(self.drones):
            self._reset_drone_idx(ds, all_env_ids)

            # After reset, build obs with MAX_DEPTH (no obstacle info yet)
            depth = torch.full(
                (self.E, self.NUM_SECTORS), self.MAX_DEPTH,
                device=self.device, dtype=torch.float32,
            )
            self.depth_buf[i] = depth

            la = torch.zeros(self.E, self.num_actions, device=self.device)
            obs_actor, obs_critic = ds.obs_builder.build_observations(
                base_pos=ds.base_pos,
                base_quat=ds.base_quat,
                base_lin_vel=ds.base_lin_vel,
                last_actions=la,
                commands=ds.commands,
                depth_actor=depth,
            )
            self.obs_buf[i] = obs_actor
            ds.priv_obs_buf[:] = obs_critic

        if self.training_mode:
            priv_obs = torch.stack([ds.priv_obs_buf for ds in self.drones], dim=0)
            return self.obs_buf, {"observations": {"critic": priv_obs}}
        else:
            return self.obs_buf, {}

    # ================================================================== #
    #  State extraction                                                   #
    # ================================================================== #

    def _update_state(self, ds: _DroneState):
        """Extract entity position, orientation, velocity from rigid solver."""
        ds.base_pos[:] = ds.entity.get_pos()
        ds.base_quat[:] = ds.entity.get_quat()
        ds.base_euler[:] = quat_to_xyz(ds.base_quat, rpy=True, degrees=False)

        # Linear velocity from DOF solver (matching WP1)
        all_dof_vel = ds.entity.get_dofs_velocity()
        ds.base_lin_vel[:] = all_dof_vel[:, :3]

        # Angular velocity in body frame (matching WP1)
        inv_base = inv_quat(ds.base_quat)
        ds.base_ang_vel[:] = transform_by_quat(ds.entity.get_ang(), inv_base)

        # Joint state (servo positions, velocities, torques)
        if ds.num_servos > 0:
            dofs_pos = ds.entity.get_dofs_position()
            ds.joint_position[:] = dofs_pos[:, ds.servo_dof_indices]
            ds.joint_velocity[:] = all_dof_vel[:, ds.servo_dof_indices]
            ds.torque[:] = ds.entity.get_dofs_control_force(ds.servo_dof_indices)

        # NaN check
        nan_mask = (torch.isnan(ds.base_pos).any(dim=1) |
                    torch.isnan(ds.base_quat).any(dim=1))
        ds.nan_envs[:] = nan_mask

        # Full power model
        ds.power[:] = self._compute_power_drone(ds)

    def _compute_power_drone(self, ds: _DroneState) -> torch.Tensor:
        """Full power model matching WingedDroneEnv.power_consumption()."""
        # Thrust
        cached_thrust = getattr(ds.aero_solver, "_thrust_n_buf", None)
        if torch.is_tensor(cached_thrust) and cached_thrust.shape[0] == self.E:
            ds._thrust_buf.copy_(cached_thrust)
        else:
            ds._thrust_buf.zero_()
        torch.nan_to_num_(ds._thrust_buf, nan=0.0, posinf=0.0, neginf=0.0)
        ds._thrust_buf.clamp_(min=0.0)
        ds.thrust_log[:, 0] = ds._thrust_buf

        # Propeller RPM
        cached_rpm = getattr(ds.aero_solver, "_prop_rpm_buf", None)
        if torch.is_tensor(cached_rpm) and cached_rpm.shape[0] == self.E:
            ds._prop_rpm_buf.copy_(cached_rpm)
        else:
            ds._prop_rpm_buf.zero_()
        torch.nan_to_num_(ds._prop_rpm_buf, nan=0.0, posinf=0.0, neginf=0.0)
        ds._prop_rpm_buf.clamp_(min=0.0)
        ds.prop_rpm_log[:, 0] = ds._prop_rpm_buf

        # Axial inflow speed
        cached_speed = getattr(ds.aero_solver, "_prop_axial_speed_buf", None)
        if torch.is_tensor(cached_speed) and cached_speed.shape[0] == self.E:
            ds._prop_axial_speed_buf.copy_(cached_speed)
        else:
            ds._prop_axial_speed_buf.zero_()
        torch.nan_to_num_(ds._prop_axial_speed_buf, nan=0.0, posinf=0.0, neginf=0.0)
        ds._prop_axial_speed_buf.clamp_(min=0.0)
        ds.prop_axial_speed_log[:, 0] = ds._prop_axial_speed_buf

        # Lazy-init per-drone power coefficients
        if ds._power_prop_coeffs is None or ds._power_prop_coeffs.shape[0] != ds.thrust_log.shape[1]:
            ds._power_prop_coeffs = power_utils._default_prop_coeffs(
                ds._drone_name, self.device, ds.thrust_log.shape[1],
                aero_config=ds._aero_config_dict,
            )
        if ds._power_servo_constants is None or ds._power_servo_constants.shape[0] != ds.num_servos:
            ds._power_servo_constants = power_utils._default_servo_power_constants(
                ds._drone_name, self.device, ds.num_servos,
            )
        if ds._power_torque_multipliers is None or ds._power_torque_multipliers.shape[0] != ds.num_servos:
            ds._power_torque_multipliers = power_utils._default_torque_multipliers(
                ds._drone_name, self.device, ds.num_servos,
                sweep_multiplier=2.0, twist_multiplier=2.5, tail_multiplier=2.0,
            )

        return power_utils.compute_power_consumption(
            thrust=ds.thrust_log,
            servo_torque=ds.torque,
            servo_velocity=ds.joint_velocity,
            drone_name=ds._drone_name,
            aero_config=ds._aero_config_dict,
            prop_coefficients=ds._power_prop_coeffs,
            prop_rpm=ds.prop_rpm_log,
            prop_axial_speed=ds.prop_axial_speed_log,
            prop_diameters=ds._prop_diameter,
            servo_power_constants=ds._power_servo_constants,
            torque_multipliers=ds._power_torque_multipliers,
            device=self.device,
        )

    # ================================================================== #
    #  Observation                                                        #
    # ================================================================== #

    def _compute_obs(self, drone_idx: int, ds: _DroneState):
        """Build actor and critic observations for one drone."""
        if self.cylinders_xy is not None:
            depth = self.depth_solver.compute_depth(
                base_pos=ds.base_pos,
                base_euler=ds.base_euler,
                cyl_xy_b=self.cylinders_xy,
            )
        else:
            depth = torch.full(
                (self.E, self.NUM_SECTORS), self.MAX_DEPTH,
                device=self.device, dtype=torch.float32,
            )

        self.depth_buf[drone_idx] = depth

        # Use previous step's actions for observation (matches WP1 timing)
        la = torch.zeros(self.E, self.num_actions, device=self.device)
        la[:, :ds.prev_actions.shape[1]] = ds.prev_actions

        obs_actor, obs_critic = ds.obs_builder.build_observations(
            base_pos=ds.base_pos,
            base_quat=ds.base_quat,
            base_lin_vel=ds.base_lin_vel,
            last_actions=la,
            commands=ds.commands,
            depth_actor=depth,
        )
        self.obs_buf[drone_idx] = obs_actor
        ds.priv_obs_buf[:] = obs_critic

    # ================================================================== #
    #  Termination                                                        #
    # ================================================================== #

    def _compute_termination_flags(self, ds: _DroneState) -> None:
        """Update reset_buf, time_outs, and all termination flags for one drone (training)."""
        # Collision
        ds.collision[:] = self._check_collision(ds)

        # Success (training: forest_x_limit, eval: x_upper)
        success_limit = (
            ds._success_x_limit_train if self.training_mode else self._success_x_limit_eval
        )
        ds.success[:] = ds.base_pos[:, 0] > success_limit

        # Wall / ground crash
        ds.wall_crash_condition[:] = (
            (ds.base_pos[:, 1].abs() > self._termination_abs_y_max) |
            (ds.base_pos[:, 2] < self._termination_min_z)
        )

        # Angle limits (tighter during training to encourage stable flight)
        if self.training_mode:
            ds.angle_limit_condition[:] = (
                (ds.base_euler[:, 0].abs() > self._roll_limit_train) |
                (ds.base_euler[:, 1].abs() > self._pitch_limit_train) |
                (ds.base_euler[:, 2].abs() > self._yaw_limit_train)
            )
        else:
            ds.angle_limit_condition[:] = (
                (ds.base_euler[:, 0].abs() > self._roll_limit_eval) |
                (ds.base_euler[:, 1].abs() > self._pitch_limit_eval) |
                (ds.base_euler[:, 2].abs() > self._yaw_limit_eval)
            )

        nan_mask = ds.nan_envs.bool()

        ds.reset_buf[:] = (
            (ds.episode_length >= ds.max_episode_length_per_env) |
            ds.wall_crash_condition |
            ds.angle_limit_condition |
            ds.success |
            ds.collision |
            nan_mask
        )

        # Capture termination reason (for logging)
        ds.pre_wall_crash[ds.reset_buf] = ds.wall_crash_condition[ds.reset_buf]
        ds.pre_angle_limit[ds.reset_buf] = ds.angle_limit_condition[ds.reset_buf]
        ds.pre_collision[ds.reset_buf] = ds.collision[ds.reset_buf]
        ds.pre_success[ds.reset_buf] = ds.success[ds.reset_buf]
        ds.pre_nan[ds.reset_buf] = nan_mask[ds.reset_buf]

        # Time-out: episode ended by timeout (not crash/success/collision)
        timeout = (ds.episode_length >= ds.max_episode_length_per_env) & ~(
            ds.success | ds.collision | ds.wall_crash_condition | ds.angle_limit_condition
        )
        ds._time_outs.zero_()
        ds._time_outs[timeout] = 1.0

    # Backward-compat for eval code that still calls _check_termination
    def _check_termination(self, ds: _DroneState) -> torch.Tensor:
        """Eval-only termination check (backward compat). Returns bool tensor (E,)."""
        self._compute_termination_flags(ds)
        return ds.reset_buf.clone()

    def _compute_rewards_drone(self, drone_idx: int, ds: _DroneState) -> None:
        """Compute all reward components for one drone (matches WingedDroneEnv._accumulate_rewards)."""
        ds.rew_buf.zero_()
        ds.last_reward_components.zero_()

        if not self.reward_names:
            return

        depth_vals = self.depth_buf[drone_idx]  # (E, NUM_SECTORS)

        for i, name in enumerate(self.reward_names):
            scale = self.reward_scales[name]

            if name == "smooth":
                rew = torch.sum((ds.last_actions - ds.prev_actions) ** 2, dim=1)
            elif name == "angular":
                rew = torch.norm(ds.base_ang_vel, dim=1)
            elif name == "stability":
                rew = ds.base_euler[:, 0] ** 2 + ds.base_euler[:, 1] ** 2
            elif name == "crash":
                rew = torch.zeros(self.E, device=self.device)
                rew[ds.wall_crash_condition | ds.collision | ds.angle_limit_condition] = 1.0
                rew = rew / (self.dt * 100.0)
            elif name == "energy":
                rew = ds.power
            elif name == "progress":
                sigma = 0.25
                v_proj = ds.base_lin_vel[:, 0]
                v_tgt = ds.commands[:, 0].clamp(min=1e-3)
                x = v_proj / v_tgt
                rew = torch.exp(-0.5 * ((x - 1.0) / sigma) ** 2)
            elif name == "obstacle":
                dir_x = ds.ray_dir_x.unsqueeze(0)   # (1, S)
                dir_y = ds.ray_dir_y.unsqueeze(0)   # (1, S)
                x_loc = depth_vals * dir_x
                y_loc = depth_vals * dir_y
                aniso = torch.sqrt(x_loc ** 2 + 24.0 * y_loc ** 2) / 5.0
                rew = torch.sum(torch.exp(-1.5 * torch.clamp_min(aniso, 0.5)), dim=1)
            elif name == "height":
                dist = (ds.base_pos[:, 2] - ds.target_height) ** 2
                penalty = torch.zeros(self.E, device=self.device)
                low = ds.base_pos[:, 2] < ds.target_height
                penalty[low] = dist[low]
                rew = penalty
            elif name == "success":
                rew = torch.zeros(self.E, device=self.device)
                rew[ds.success] = 1.0
            elif name == "cosmetic":
                rew = torch.zeros(self.E, device=self.device)
                if ds.num_servos >= 4:
                    rew += (ds.joint_position[:, 0] + ds.joint_position[:, 1]) ** 2
                    rew += (ds.joint_position[:, 2] - ds.joint_position[:, 3]) ** 2
                if ds.num_servos > 5:
                    rew += ds.joint_position[:, 5] ** 2
            else:
                rew = torch.zeros(self.E, device=self.device)

            rew_comp = rew * scale * self.dt * 50.0
            ds.rew_buf += rew_comp
            if name in ds.episode_sums:
                ds.episode_sums[name] += rew_comp
            ds.last_reward_components[:, i] = rew_comp

        ds.last_reward_total[:] = ds.rew_buf

        # Zero out NaN envs
        nan_mask = ds.nan_envs.bool()
        if nan_mask.any():
            ds.rew_buf[nan_mask] = 0.0
            ds.last_reward_components[nan_mask] = 0.0
            ds.last_reward_total[nan_mask] = 0.0

    def _check_collision(self, ds: _DroneState) -> torch.Tensor:
        """Geometric collision check: drone rectangle vs tree cylinders."""
        if self.cylinders_xy is None:
            return torch.zeros(self.E, dtype=torch.bool, device=self.device)

        half_span = ds.span / 2.0
        r_tree = self._tree_radius + self._collision_tol

        # Relative positions: (E, T, 2)
        drone_xy = ds.base_pos[:, :2].unsqueeze(1)  # (E, 1, 2)
        diff = self.cylinders_xy - drone_xy          # (E, T, 2)

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
    #  Per-drone partial reset                                            #
    # ================================================================== #

    def _reset_drone_idx(self, ds: _DroneState, env_ids: torch.Tensor) -> None:
        """Reset a subset of environments for one drone.

        Matches WingedDroneEnv.reset_idx(): resamples commands and forest,
        logs episode statistics, applies initial state with training
        randomization, and resets all bookkeeping buffers.
        """
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()

        # Resample command (target forward speed)
        ds.commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
            self.vmin, self.vmax
        )

        # Resample forest layout (matches WingedDroneEnv behavior)
        if self.cylinders_array is not None:
            new_ids = torch.randint(0, self.total_forests, (n,), device=self.device, dtype=torch.long)
            self.forest_ids[env_ids] = new_ids
            self.cylinders_xy = self.cylinders_array[self.forest_ids, :, :2]

        # Log episode statistics for finished episodes (consumed by step())
        ep_dict: Dict[str, float] = {}
        for key, total in ds.episode_sums.items():
            ep_dict["rew_" + key] = float(total[env_ids].mean())
            total[env_ids] = 0.0
        ep_dict["final_x"] = float(ds.base_pos[env_ids, 0].mean())
        ep_dict["final_y"] = float(ds.base_pos[env_ids, 1].mean())
        ep_dict["final_z"] = float(ds.base_pos[env_ids, 2].mean())
        ep_dict["num_wall_crashed"] = float(ds.wall_crash_condition[env_ids].sum())
        ep_dict["num_angle_crashed"] = float(ds.angle_limit_condition[env_ids].sum())
        ep_dict["num_success"] = float(ds.success[env_ids].sum())
        ep_dict["num_collision"] = float(ds.collision[env_ids].sum())
        ds._ep_dict_cache = ep_dict

        # Initial state (base values before randomization)
        base_euler_init = quat_to_xyz(
            self.base_init_quat.unsqueeze(0), rpy=True, degrees=False
        ).squeeze(0)

        pos = self.base_init_pos.unsqueeze(0).expand(n, -1).clone()
        euler = base_euler_init.unsqueeze(0).expand(n, -1).clone()
        lin_vel = torch.zeros(n, 3, device=self.device)
        lin_vel[:, 0] = 15.0
        ang_vel = torch.zeros(n, 3, device=self.device)
        joint_pos = torch.zeros(n, ds.num_servos, device=self.device)
        joint_vel = torch.zeros(n, ds.num_servos, device=self.device)

        # Training-time state randomization (mirrors WingedDroneEnv._randomize_reset_state)
        if self.training_mode:
            r = torch.empty(n, device=self.device)
            pos[:, 0] += r.uniform_(0.0, 1.0) * 30.0 - 15.0
            pos[:, 1] += r.uniform_(0.0, 1.0) * 80.0 - 40.0
            pos[:, 2] += r.uniform_(0.0, 1.0) * 15.0 - 10.0
            lin_vel[:, 0] = r.uniform_(0.0, 1.0) * 22.0 + 4.0
            lin_vel[:, 1] = r.normal_().mul_(2.0).clamp_(-8.0, 8.0)
            lin_vel[:, 2] = r.normal_().mul_(2.0).clamp_(-8.0, 8.0)
            # Align attitude roughly with velocity
            euler[:, 1] = torch.atan2(-lin_vel[:, 2], lin_vel[:, 0])
            euler[:, 2] = torch.atan2(lin_vel[:, 1], lin_vel[:, 0])
            euler[:, 0] += r.normal_().mul_(0.2).clamp_(-0.8, 0.8)
            euler[:, 1] += r.normal_().mul_(0.05).clamp_(-0.2, 0.2)
            euler[:, 2] += r.normal_().mul_(0.05).clamp_(-0.2, 0.2)
            if ds.num_servos > 0:
                rs = torch.zeros(n, ds.num_servos, device=self.device)
                joint_pos += rs.normal_().mul_(0.01).clamp_(-0.04, 0.04)

        # Write back to drone state buffers
        ds.base_pos[env_ids] = pos
        ds.base_euler[env_ids] = euler
        ds.base_lin_vel[env_ids] = lin_vel
        ds.base_ang_vel[env_ids] = ang_vel
        if ds.num_servos > 0:
            ds.joint_position[env_ids] = joint_pos
            ds.joint_velocity[env_ids] = joint_vel
            ds.torque[env_ids] = 0.0
        # Re-derive quaternion from (possibly randomized) euler angles
        ds.base_quat[env_ids] = xyz_to_quat(euler, degrees=False)

        # Build DOF position/velocity vectors for this entity and apply
        n_dofs = ds.entity.n_dofs
        dof_pos = torch.zeros(n, n_dofs, device=self.device)
        dof_pos[:, :3] = pos
        dof_pos[:, 3:6] = euler
        if ds.num_servos > 0:
            for j, dof_idx in enumerate(ds.servo_dof_indices):
                if j < joint_pos.shape[1]:
                    dof_pos[:, dof_idx] = joint_pos[:, j]

        dof_vel = torch.zeros(n, n_dofs, device=self.device)
        dof_vel[:, :3] = lin_vel
        dof_vel[:, 3:6] = ang_vel
        if ds.num_servos > 0:
            for j, dof_idx in enumerate(ds.servo_dof_indices):
                if j < joint_vel.shape[1]:
                    dof_vel[:, dof_idx] = joint_vel[:, j]

        ds.entity.set_dofs_position(dof_pos, envs_idx=env_ids)
        ds.entity.set_dofs_velocity(dof_vel, envs_idx=env_ids)

        # Reset per-drone actuator dynamics (latency buffer)
        ds.actuator.reset_envs(env_ids)

        # Reset bookkeeping
        ds.last_actions[env_ids] = 0.0
        ds.prev_actions[env_ids] = 0.0
        ds.episode_length[env_ids] = 0
        ds.reset_buf[env_ids] = True
        ds.success[env_ids] = False
        ds.collision[env_ids] = False
        ds.wall_crash_condition[env_ids] = False
        ds.angle_limit_condition[env_ids] = False
        ds.nan_envs[env_ids] = False
        ds.pre_collision[env_ids] = False
        ds.pre_wall_crash[env_ids] = False
        ds.pre_angle_limit[env_ids] = False
        ds.pre_success[env_ids] = False
        ds.pre_nan[env_ids] = False

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

