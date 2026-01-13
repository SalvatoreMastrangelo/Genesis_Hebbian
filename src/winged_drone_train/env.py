# env.py
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, Tuple, Sequence, Optional, List

import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, xyz_to_quat
from genesis.assets.urdf.mydrone.drone import DroneAeroModel

from morph_evolution.chromosome_drone import Chromosome_Drone
from winged_drone_train.utils import depth as depth_utils
from winged_drone_train.utils import forest as forest_utils
from winged_drone_train.utils import obs as obs_utils
from winged_drone_train.utils import power as power_utils


def _servo_gains_from_catalog(
    drone_model: DroneAeroModel,
    joint_names: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fetch per-joint kp/kv gains using actuator assignments from DroneAeroModel."""
    if drone_model is None or not getattr(drone_model, "urdf_path", None):
        raise ValueError("servo gain loading requires a DroneAeroModel with a valid urdf_path.")

    csv_path = Path(str(drone_model.urdf_path)).parent / "actuators.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing actuator catalog: {csv_path}")

    catalog: Dict[Tuple[str, str], Dict[str, str]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            kind = (row.get("type") or "").strip().lower()
            if name and kind:
                catalog[(name, kind)] = row

    def get_servo_gains(actuator_name: str) -> Tuple[float, float]:
        row = catalog.get((actuator_name, "servo"))
        if row is None:
            raise ValueError(f"Servo actuator '{actuator_name}' not found in {csv_path}")
        kp = (row.get("kp") or "").strip()
        kv = (row.get("kv") or "").strip()
        if kp == "" or kv == "":
            raise ValueError(f"Servo actuator '{actuator_name}' missing kp/kv in {csv_path}")
        return float(kp), float(kv)

    def pick_actuator(frame: str, axis: Optional[str] = None) -> Optional[str]:
        info = getattr(drone_model, "actuators", {}).get(frame) if drone_model else None
        if info is None:
            return None
        if axis == "yaw" and getattr(info, "yaw_actuator", None):
            return info.yaw_actuator
        if axis == "pitch" and getattr(info, "pitch_actuator", None):
            return info.pitch_actuator
        return info.actuator

    act_left_wing_yaw = pick_actuator("aero_frame_left_wing", "yaw")
    act_left_wing_pitch = pick_actuator("aero_frame_left_wing", "pitch")
    act_right_wing_yaw = pick_actuator("aero_frame_right_wing", "yaw")
    act_right_wing_pitch = pick_actuator("aero_frame_right_wing", "pitch")
    act_elev_pitch = pick_actuator("aero_frame_elevator_left", "pitch") or pick_actuator(
        "aero_frame_elevator_right", "pitch"
    )
    act_rudd_yaw = pick_actuator("aero_frame_rudder", "yaw")

    kp_list: List[float] = []
    kv_list: List[float] = []
    for name in joint_names:
        act = None
        if "sweep_left" in name:
            act = act_left_wing_yaw
        elif "sweep_right" in name:
            act = act_right_wing_yaw
        elif "twist_left" in name:
            act = act_left_wing_pitch
        elif "twist_right" in name:
            act = act_right_wing_pitch
        elif "elevator" in name:
            act = act_elev_pitch
        elif "rudder" in name:
            act = act_rudd_yaw

        if not act:
            raise ValueError(
                f"Missing actuator assignment for joint '{name}' (check aero_parameters.yaml links actuators)."
            )
        kp_i, kv_i = get_servo_gains(act)
        kp_list.append(kp_i)
        kv_list.append(kv_i)

    kp = torch.tensor(kp_list, device=device, dtype=torch.float32)
    kv = torch.tensor(kv_list, device=device, dtype=torch.float32)
    return kp, kv


class WingedDroneEnv:
    """
    Genesis-based environment for a morphing winged drone.

    Key design choices:
    - Single high-level command: target forward speed in +X (m/s).
    - Depth sensing via Taichi `DepthSolver` (GPU, batched over all envs).
    - Observation construction delegated to `ObservationBuilder`.
    - Action scaling + latency handled by `ActuatorDynamics`.
    - Power consumption computed via `compute_power_consumption`.

    The class exposes an RSL-RL style API:
        - step(actions)  -> obs, reward, done, info
        - reset()        -> obs, info
    """

    # Core sizes
    BASE_OBS_SIZE = 8          # [z_norm, quat(4), v_body_xy(2), v_body_z(1)]
    NUM_SECTORS_ACTOR = 20     # Depth sectors used in actor observations
    CONE_ACTOR_DEG = 80.0      # Horizontal FOV of depth (degrees)

    MAX_DISTANCE = 30.0        # Depth max range (m)
    SHORT_RANGE = 0.0          # Extra safety bubble in front (m)

    # Genome parameter ranges for normalization (if present)
    GENOME_MIN, GENOME_MAX = Chromosome_Drone.genome_min_max()

    def __init__(
        self,
        num_envs: int,
        env_cfg: Dict,
        obs_cfg: Dict,
        reward_cfg: Dict,
        command_cfg: Dict,
        urdf_file: Optional[str] = None,
        show_viewer: bool = False,
        eval: bool = False,
        device: str = "cuda",
    ) -> None:
        # ------------------------------------------------------------------ #
        # Basic configuration                                               #
        # ------------------------------------------------------------------ #
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.evaluation = bool(eval)

        # Only ONE command: target forward speed along +X (m/s)
        self.num_commands = 1
        self.command_cfg = dict(command_cfg)
        self.command_cfg["num_commands"] = 1  # keep config consistent

        # Feature toggles
        self.env_cfg = dict(env_cfg)
        self.obs_cfg = dict(obs_cfg)
        self.reward_cfg = dict(reward_cfg)

        self.growing_forest = self.env_cfg.get("growing_forest", True)
        self.unique_forests_eval = self.env_cfg.get("unique_forests_eval", True)
        self.show_viewer = bool(show_viewer)

        # Action latency (delegated to ActuatorDynamics)
        self.simulate_action_latency = bool(self.env_cfg.get("simulate_action_latency", False))
        self.action_latency_min = int(self.env_cfg.get("action_latency_min_steps", 0))
        self.action_latency_max = int(self.env_cfg.get("action_latency_max_steps", 0))
        self.action_latency_random_per_step = bool(self.env_cfg.get("action_latency_random_per_step", False))

        # For evaluation we enforce a deterministic, fixed latency
        if self.evaluation:
            self.action_latency_min = 0
            self.action_latency_max = 0
            self.action_latency_random_per_step = False

        if self.action_latency_max < self.action_latency_min:
            self.action_latency_max = self.action_latency_min

        # Target height (used in height reward) is fixed, not commanded
        self.target_height = float(self.env_cfg.get("target_height", 3.0))

        # Time step setup: control at 25 Hz, physics at 100 Hz
        control_hz = 25
        physics_hz = 100
        self.dt = 1.0 / control_hz
        substeps = int(physics_hz / control_hz)

        # Episode length (seconds → steps)
        episode_length_s = float(self.env_cfg.get("episode_length_s", 500.0))
        self.max_episode_length = math.ceil(episode_length_s / self.dt)

        # Per-env episode length with small random variation for training
        self.max_episode_length_per_env = torch.full(
            (self.num_envs,), self.max_episode_length, device=self.device, dtype=torch.long
        )

        # Crash angle limits (can be overridden at runtime)
        if self.evaluation:
            self.set_angle_limit(self.env_cfg.get("eval_angle_limit_deg", 90.0))
        else:
            self.set_angle_limit(self.env_cfg.get("default_angle_limit_deg", 90.0))

        if urdf_file is None:
            urdf_file = "/home/andrea/Documents/Genesis/genesis/assets/urdf/mydrone/[0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4, 0.2, 2, 0, 0.25, 2, 2.5, 2, -3].urdf"
        self.urdf_file = str(urdf_file)
        self.drone_model = DroneAeroModel(self.urdf_file)

        # ------------------------------------------------------------------ #
        # Genesis scene                                                      #
        # ------------------------------------------------------------------ #
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=self.env_cfg.get("max_visualize_FPS", 60),
                camera_pos=tuple(self.env_cfg.get("camera_pos", (-35.0, 0.0, 15.0))),
                camera_lookat=tuple(self.env_cfg.get("camera_lookat", (-28.0, 0.0, 10.0))),
                res=self.env_cfg.get("camera_res", (360, 360)),
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(self.num_envs, 1))),
                show_world_frame=False,
                world_frame_size=1.0,
                show_link_frame=False,
                plane_reflection=False,
                ambient_light=(0.1, 0.1, 0.1),
                shadow=False,
                background_color=(0.04, 0.08, 0.12),
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=False,
                enable_joint_limit=True,
            ),
            show_viewer=self.show_viewer,
            renderer=gs.renderers.Rasterizer(),
        )

        # Optional ground plane (visual only)
        if self.evaluation:
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
        # Drone entity                                                       #
        # ------------------------------------------------------------------ #
        base_init_pos = torch.tensor(
            self.env_cfg.get("base_init_pos", [0.0, 0.0, 1.0]),
            device=self.device,
            dtype=torch.float32,
        )
        base_init_quat = torch.tensor(
            self.env_cfg.get("base_init_quat", [1.0, 0.0, 0.0, 0.0]),
            device=self.device,
            dtype=torch.float32,
        )

        self.base_init_pos = base_init_pos
        self.base_init_quat = base_init_quat
        self.inv_base_init_quat = inv_quat(base_init_quat)

        self.drone_name = self.env_cfg.get("drone", "morphing_drone")

        # Servo joints
        servo_joint_names = self.env_cfg.get("servo_joint_names", None)
        if servo_joint_names is None:
            servo_joint_names = [
                "joint_0_sweep_left_wing",
                "joint_0_sweep_right_wing",
                "joint_1_twist_left_wing",
                "joint_1_twist_right_wing",
                "elevator_pitch_joint",
                "rudder_yaw_joint",
            ]
        self.servo_joint_names = servo_joint_names

        urdf_args = {
            "file": self.urdf_file,
            "pos": base_init_pos.cpu().numpy(),
            "quat": base_init_quat.cpu().numpy(),
            "collision": False,
            "merge_fixed_links": True,
        }

        if "links_to_keep" in self.env_cfg:
            urdf_args["links_to_keep"] = self.env_cfg["links_to_keep"]
        else:
            urdf_args["links_to_keep"] = self.drone_model.required_links(self.servo_joint_names)

        self.drone = self.scene.add_entity(gs.morphs.URDF(**urdf_args))

        self.servo_dof_indices = []
        for name in self.servo_joint_names:
            joint = self.drone.get_joint(name)
            idx = getattr(joint, "dofs_idx_local", None)
            if idx is None:
                idx = joint.dof_idx_local
            if isinstance(idx, (list, tuple)):
                idx = idx[0] if idx else None
            if idx is None:
                raise ValueError(f"Joint '{name}' has no DOF index.")
            self.servo_dof_indices.append(int(idx))

        # Throttle + servos
        self.THROTTLE_SIZE = 1
        self.num_servos = len(self.servo_dof_indices)
        self.num_actions = self.THROTTLE_SIZE + self.num_servos
        self.throttle_limit: Tuple[float, float] = (0.0, 1.0)

        # ------------------------------------------------------------------ #
        # Forest generation (geometry only, no physics)                      #
        # ------------------------------------------------------------------ #
        (
            self.cylinders_array,
            self.total_forests,
            self._forest_generator,
        ) = forest_utils.generate_forests(
            num_envs=self.num_envs,
            evaluation=self.evaluation,
            unique_forests_eval=self.unique_forests_eval,
            growing_forest=self.growing_forest,
            env_cfg=self.env_cfg,
            device=self.device,
        )

        # Each environment gets a forest layout id
        self.forest_ids = torch.randint(
            low=0,
            high=self.total_forests,
            size=(self.num_envs,),
            device=self.device,
            dtype=torch.long,
        )

        # Optional visualization of trees (only for eval; training stays lean)
        if self.num_envs == 1:
            self._spawn_tree_visuals()

        # Always define the attribute so eval.py can check it.
        self.rec_cam = None

        # We only support recording when there is a single environment.
        if self.num_envs == 1 and self.evaluation:
            self._create_follow_camera()

        # ------------------------------------------------------------------ #
        # IMU sensor on the fuselage                                         #
        # ------------------------------------------------------------------ #
        imu_link = self.drone.get_link("fuselage")  # link già presente in links_to_keep

        # Nota: su alcune versioni di Genesis potresti dover usare
        # gs.sensors.imu.IMUOptions invece di gs.sensors.IMUOptions.
        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=self.drone.idx,
                link_idx_local=imu_link.idx_local,
                # Parti senza rumore per il debug
                acc_axes_skew=(0.0, 0.0, 0.0),
                gyro_axes_skew=(0.0, 0.0, 0.0),
                delay=0.0,
                jitter=0.0,
            )
        )
        # ------------------------------------------------------------------ #
        # Genome handling (optional)                                        #
        # ------------------------------------------------------------------ #
        self.add_genome_obs = bool(self.obs_cfg.get("add_genome_obs", False))
        self._genome_vec: Optional[torch.Tensor] = None
        self.noise_std = self.obs_cfg.get("noise_std", {})
        self._naca_code: Optional[str] = None

        if self.urdf_file:
            match = re.search(r"\[([^\]]+)\]\.urdf$", self.urdf_file)
            if match:
                try:
                    values = [float(x) for x in match.group(1).split(",")]
                    self._naca_code = Chromosome_Drone.naca_from_physical(values)
                    g = torch.tensor(values, dtype=torch.float32, device=self.device)
                    self._genome_vec = g.unsqueeze(0).repeat(self.num_envs, 1)
                except Exception:
                    self._genome_vec = None

        if self._genome_vec is not None and not self.evaluation:
            # Simple domain randomization of genome parameters during training
            noise_std = self.noise_std.get("genome", 0.1)
            noise = torch.randn_like(self._genome_vec) * noise_std
            gmin = torch.tensor(self.GENOME_MIN, device=self.device)
            gmax = torch.tensor(self.GENOME_MAX, device=self.device)
            noise *= (gmax - gmin).unsqueeze(0)
            for idx in Chromosome_Drone.NACA_GENE_INDICES:
                if idx < noise.shape[1]:
                    noise[:, idx] = 0.0
            self._genome_vec += noise
            self._genome_vec = torch.max(torch.min(self._genome_vec, gmax), gmin)

        # ------------------------------------------------------------------ #
        # Build scene and get solvers                                       #
        # ------------------------------------------------------------------ #
        self.scene.build(n_envs=self.num_envs)
        self.drone_model.validate_entity(
            self.drone,
            self.servo_joint_names,
            self.servo_dof_indices,
        )
        self.rigid_solver = self.scene.sim.rigid_solver
        self.aero_solver = self.scene.sim.aero_solver
        self.aero_solver.add_target(self.drone, drone_model=self.drone_model)
        naca_code = self._naca_code or str(self.env_cfg.get("naca", "") or "").strip()
        if naca_code and hasattr(self.aero_solver, "apply_naca_wing_override"):
            self.aero_solver.apply_naca_wing_override(naca_code)

        self.span = self.aero_solver.tip_to_tip
        self.nominal_mass = float(sum(link.get_mass() for link in self.drone.links))

        print(f"[WingedDroneEnv] Created with {self.num_envs} envs, drone '{self.drone_name}'")
        print(f"  - Action space size: {self.num_actions} (throttle + {self.num_servos} servos)")
        print(f"  - Drone span: {self.span:.3f} m")
        print(f"  - Nominal mass: {self.nominal_mass:.3f} kg")

        # ------------------------------------------------------------------ #
        # Setup drone actuators                                          #
        # ------------------------------------------------------------------ #

        # Joint limits for servos
        joint_mins, joint_maxs = self.drone.get_dofs_limit(self.servo_dof_indices)
        joint_mins = torch.as_tensor(joint_mins, device=self.device, dtype=torch.float32)
        joint_maxs = torch.as_tensor(joint_maxs, device=self.device, dtype=torch.float32)
        self.joint_limit_min = joint_mins
        self.joint_limit_max = joint_maxs

        # PD gains for servo position control
        kp, kv = _servo_gains_from_catalog(self.drone_model, self.servo_joint_names, self.device)
        self.drone.set_dofs_kp(kp, self.servo_dof_indices)
        self.drone.set_dofs_kv(kv, self.servo_dof_indices)

        # ------------------------------------------------------------------ #
        # Depth solver + precomputed ray directions                          #
        # ------------------------------------------------------------------ #
        tree_radius = float(self.env_cfg.get("tree_radius", 1.0))
        y_lower = float(self.env_cfg.get("y_lower", -50.0))
        y_upper = float(self.env_cfg.get("y_upper", 50.0))

        self.depth_solver = depth_utils.DepthSolver(
            num_sectors=self.NUM_SECTORS_ACTOR,
            cone_angle_deg=self.CONE_ACTOR_DEG,
            max_distance=self.MAX_DISTANCE,
            short_range=self.SHORT_RANGE,
            tree_radius=tree_radius,
            y_lower=y_lower,
            y_upper=y_upper,
            torch_device=self.device,
        )

        # Torch copy of ray directions (for obstacle reward, no Taichi needed)
        angles = torch.linspace(
            -0.5 * math.radians(self.CONE_ACTOR_DEG),
            0.5 * math.radians(self.CONE_ACTOR_DEG),
            self.NUM_SECTORS_ACTOR,
            device=self.device,
            dtype=torch.float32,
        )
        self.ray_dir_x = torch.cos(angles)  # (S,)
        self.ray_dir_y = torch.sin(angles)  # (S,)

        # Depth buffer
        self.depth = torch.full(
            (self.num_envs, self.NUM_SECTORS_ACTOR),
            self.MAX_DISTANCE,
            device=self.device,
            dtype=torch.float32,
        )

        # ------------------------------------------------------------------ #
        # Observation builder                                                #
        # ------------------------------------------------------------------ #
        self.obs_builder = obs_utils.ObservationBuilder(
            num_actions=self.num_actions,
            num_sectors_actor=self.NUM_SECTORS_ACTOR,
            joint_limits_max=self.joint_limit_max,
            obs_cfg=self.obs_cfg,
            add_genome_obs=self.add_genome_obs and (self._genome_vec is not None),
            genome_vec=self._genome_vec,
            genome_min=self.GENOME_MIN if self._genome_vec is not None else None,
            genome_max=self.GENOME_MAX if self._genome_vec is not None else None,
            include_depth=self.obs_cfg.get("include_depth", True),
            device=self.device,
        )

        # We want depth in both actor and critic observations
        self.include_depth = bool(self.obs_cfg.get("include_depth", True))

        # ------------------------------------------------------------------ #
        # Actuator dynamics (scaling + latency)                              #
        # ------------------------------------------------------------------ #
        latency_cfg = power_utils.LatencyConfig(
            simulate_latency=self.simulate_action_latency,
            latency_min=self.action_latency_min,
            latency_max=self.action_latency_max,
            random_latency_per_step=self.action_latency_random_per_step,
        )

        self.actuator = power_utils.ActuatorDynamics(
            num_envs=self.num_envs,
            num_actions=self.num_actions,
            throttle_limit=self.throttle_limit,
            joint_limits=(self.joint_limit_min.tolist(), self.joint_limit_max.tolist()),
            latency_cfg=latency_cfg,
            device=self.device,
        )

        # ------------------------------------------------------------------ #
        # Observation + reward dimensions                                    #
        # ------------------------------------------------------------------ #
        self.num_obs = self.obs_builder.actor_obs_dim
        self.num_privileged_obs = self.obs_builder.priv_obs_dim

        # Expose to config (useful for RL pipeline)
        self.obs_cfg["num_obs"] = self.num_obs
        self.env_cfg["num_actions"] = self.num_actions

        # ------------------------------------------------------------------ #
        # State, buffers, bookkeeping                                        #
        # ------------------------------------------------------------------ #
        # Core state
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.accelerations = torch.zeros((self.num_envs, 6), device=self.device)

        self.joint_position = torch.zeros((self.num_envs, self.num_servos), device=self.device)
        self.joint_velocity = torch.zeros((self.num_envs, self.num_servos), device=self.device)
        self.torque = torch.zeros((self.num_envs, self.num_servos), device=self.device)

        # Action buffers (normalized space, [-1, 1])
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)

        # Commands: single scalar forward speed per env
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device)

        # Logging extras
        self.thrust_log = torch.zeros((self.num_envs, 1), device=self.device)
        self.alpha = torch.zeros((self.num_envs, 1), device=self.device)
        self.beta = torch.zeros((self.num_envs, 1), device=self.device)
        self.d_cf_com_body = torch.zeros((self.num_envs, 3), device=self.device)

        if self.evaluation: 
            self.aero_solver._aero_log = True
        # Episode bookkeeping
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.reset_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.success = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.collision = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.wall_crash_condition = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.angle_limit_condition = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.nan_envs = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int64)

        self.pre_wall_crash = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.pre_angle_limit = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.pre_collision = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.pre_success = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.pre_nan = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        # Reward bookkeeping
        self.reward_scales: Dict[str, float] = self.reward_cfg.get("reward_scales", {})
        self.reward_names = list(self.reward_scales.keys())
        self.reward_functions = {}
        self.episode_sums = {}

        for name in self.reward_names:
            fn_name = f"_reward_{name}"
            if not hasattr(self, fn_name):
                raise AttributeError(f"Reward function '{fn_name}' is not defined.")
            self.reward_functions[name] = getattr(self, fn_name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device)

        self.last_reward_components = torch.zeros(
            (self.num_envs, len(self.reward_names)), device=self.device
        )
        self.last_reward_total = torch.zeros((self.num_envs,), device=self.device)

        # Observation / reward buffers exposed to the RL algorithm
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device)
        # Extras dictionary for logging (RSL-RL convention)
        self.extras: Dict = {"observations": {}}
        self._video_on = False
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
    # ---------------------------------------------------------------------- #
    # Video recording                                                      #
    # ---------------------------------------------------------------------- #
    def start_video(self, filename: str = "sim_record.mp4", fps: int | None = None):
        """
        Begin recording from the follow camera.

        Call this before stepping the environment. Only supported when
        a recording camera (`self.rec_cam`) has been created, i.e.,
        typically with num_envs == 1.
        """
        if self._video_on:
            return

        if self.rec_cam is None:
            raise RuntimeError(
                "Recording camera not created. "
                "Video recording is only supported when num_envs == 1 "
                "and env_cfg['enable_rec_camera'] is True."
            )

        self._video_file = filename
        self._video_fps = fps or int(1.0 / self.dt)

        # Start capturing all subsequent `render()` calls.
        self.rec_cam.start_recording()
        self._video_on = True

    def stop_video(self):
        """
        Stop recording from the follow camera and save the MP4 file.
        """
        if not self._video_on or self.rec_cam is None:
            return

        # Genesis API: save_to_filename + fps
        self.rec_cam.stop_recording(
            save_to_filename=self._video_file,
            fps=self._video_fps,
        )
        self._video_on = False


    def _create_follow_camera(self) -> None:
        """
        Create a rasterization camera that can record a video and
        (later) be updated to follow env-0 from behind.

        The pose will be updated every step by `_update_follow_camera`.
        """
        cam_res = tuple(self.env_cfg.get("camera_res", (1024, 768)))
        cam_fov = float(self.env_cfg.get("camera_fov", 80.0))

        # Initial dummy pose; will be overwritten on reset/step.
        pos0 = tuple(self.env_cfg.get("rec_cam_pos", (-10.0, 0.0, 3.0)))
        lookat0 = tuple(self.env_cfg.get("rec_cam_lookat", (0.0, 0.0, 1.0)))

        # This camera is independent from the viewer camera.
        self.rec_cam = self.scene.add_camera(
            res=cam_res,
            pos=pos0,
            lookat=lookat0,
            fov=cam_fov,
            GUI=False,   # required for recording
        )

    def _update_follow_camera(self) -> None:
        """
        Update the recording camera pose so that it smoothly follows env-0
        from behind.

        The camera target pose is computed from the drone position and yaw,
        then smoothed with an exponential filter:
            x_smoothed = alpha * x_target + (1 - alpha) * x_smoothed_prev
        """
        if self.num_envs == 0:
            return

        # Follow environment 0 (evaluation: num_envs == 1).
        pos = self.base_pos[0]       # (3,)
        euler = self.base_euler[0]   # (3,)
        yaw = float(euler[2].item())

        # Offsets in meters (body frame), can be tuned via env_cfg.
        dist_back = float(self.env_cfg.get("rec_cam_follow_distance", 3.0))
        height_offset = float(self.env_cfg.get("rec_cam_follow_height", 1.0))
        lateral_offset = float(self.env_cfg.get("rec_cam_follow_lateral", 0.0))

        # Smoothing factor for exponential filter (0 → very smooth, 1 → no filter).
        alpha = float(self.env_cfg.get("rec_cam_smooth_alpha", 0.25))

        # World position of the drone (we look at this point).
        px = float(pos[0].item())
        py = float(pos[1].item())
        pz = float(pos[2].item())

        # Desired camera position in world frame:
        # behind the drone along -heading, plus optional lateral offset and height.
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        back_x = px - dist_back
        back_y = py
        back_z = pz + height_offset

        target_pos = torch.tensor(
            [back_x, back_y, back_z],
            device=self.device,
            dtype=torch.float32,
        )
        target_lookat = pos + torch.tensor([1.5, 0, 0])
        # Lazy initialization of filtered pose on first call.
        if not hasattr(self, "_rec_cam_pos_filtered"):
            self._rec_cam_pos_filtered = target_pos.clone()
            self._rec_cam_lookat_filtered = target_lookat.clone()
        else:
            # Exponential smoothing (like your top_pos / look_at example).
            self._rec_cam_pos_filtered = (
                alpha * target_pos + (1.0 - alpha) * self._rec_cam_pos_filtered
            )
            self._rec_cam_lookat_filtered = (
                alpha * target_lookat + (1.0 - alpha) * self._rec_cam_lookat_filtered
            )

        # Push pose to Genesis camera (convert to plain Python tuples).
        cam_pos = tuple(self._rec_cam_pos_filtered.detach().cpu().tolist())
        cam_look = tuple(self._rec_cam_lookat_filtered.detach().cpu().tolist())

        if self.rec_cam is not None and self.evaluation:
            self.rec_cam.set_pose(pos=cam_pos, lookat=cam_look)

        if self.show_viewer and getattr(self.scene, "viewer", None) is not None:
            cam_pos_np = self._rec_cam_pos_filtered.detach().cpu().numpy()   # np.ndarray (float32, shape (3,))
            cam_look_np = self._rec_cam_lookat_filtered.detach().cpu().numpy() # np.ndarray

            self.scene.viewer.set_camera_pose(
                pos=cam_pos_np,
                lookat=cam_look_np,
            )

    # ---------------------------------------------------------------------- #
    # High-level command sampling                                            #
    # ---------------------------------------------------------------------- #
    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """
        Resample target forward speed for given environments.

        Command semantics:
            commands[b, 0] = v_target_x (m/s, along world +X).
        """

        if env_ids.numel() == 0:
            return
        
        if self.evaluation and self.num_envs == 1:
            # Use fixed velocity for evaluation
            v_eval = float(self.command_cfg.get("eval_speed", 12.0))
            self.commands[env_ids, 0] = v_eval
        elif self.evaluation and self.num_envs > 1:
            v_min = float(self.command_cfg.get("min_speed", 8.0))
            v_max = float(self.command_cfg.get("max_speed", 16.0))
            # linearly spaced speeds for all eval envs
            speeds = torch.linspace(v_min, v_max, self.num_envs, device=self.device)
            self.commands[env_ids, 0] = speeds[env_ids]
        else:
            v_min = float(self.command_cfg.get("min_speed", 6.0))
            v_max = float(self.command_cfg.get("max_speed", 24.0))
            u = torch.rand((env_ids.numel(),), device=self.device)
            self.commands[env_ids, 0] = v_min + (v_max - v_min) * u

    # ---------------------------------------------------------------------- #
    # Crash limits                                                           #
    # ---------------------------------------------------------------------- #
    def set_angle_limit(self, limit_deg: float) -> None:
        """Set symmetric angle limits in roll/pitch/yaw used for crash detection."""
        self.curr_limit = float(limit_deg)
        r = math.radians(limit_deg)
        self.roll_limit_rad = r
        self.pitch_limit_rad = r
        self.yaw_limit_rad = r

    # ---------------------------------------------------------------------- #
    # Step function                                                          #
    # ---------------------------------------------------------------------- #
    @torch.inference_mode()
    def step(self, actions: torch.Tensor):
        """
        Advance the simulation by one control step.

        Args:
            actions: (num_envs, num_actions) normalized in [-1, 1].

        Returns:
            obs_buf, rew_buf, reset_buf, extras
        """
        # ------------------------- Actions -------------------------------- #
        actions = actions.to(self.device)
        # ActuatorDynamics handles clamping, scaling and latency
        servo_targets, throttle = self.actuator.process_actions(actions)

        # Store applied (scaled + delayed) actions
        self.last_actions.copy_(self.actions)
        self.actions[:, 0] = throttle
        self.actions[:, 1:] = servo_targets

        # Apply to simulator
        self.drone.control_dofs_position(servo_targets, self.servo_dof_indices)
        self.aero_solver.set_throttle(throttle)

        # Print everything about the state for debugging
        if self.num_envs == 1:
            print(f"Step: {self.episode_length_buf[0].cpu().numpy()}")
            print(f"Pos: {self.base_pos[0].cpu().numpy()}")
            print(f"Euler: {self.base_euler[0].cpu().numpy()}")
            print(f"Lin Vel: {self.base_lin_vel[0].cpu().numpy()}")
            print(f"Ang Vel: {self.base_ang_vel[0].cpu().numpy()}")
            print(f"Joint Pos: {self.joint_position[0].cpu().numpy()}")
            print(f"Joint Vel: {self.joint_velocity[0].cpu().numpy()}")
            print(f"Torque: {self.torque[0].cpu().numpy()}")

        # ------------------------- Physics -------------------------------- #
        self.scene.step()

        # NaN check (simulation instability)
        self.nan_envs.fill_(0)
        dofs_pos = self.drone.get_dofs_position()
        if not torch.isfinite(dofs_pos).all():
            nan_idx = torch.isnan(dofs_pos).any(dim=1).nonzero(as_tuple=False).flatten()
            if len(nan_idx) > 0:
                print(f"[WingedDroneEnv] NaN detected at sim time {float(self.scene.t):.3f}, envs {nan_idx.tolist()}")
                self.nan_envs[nan_idx] = 1

        # Increase episode step counters
        self.episode_length_buf += 1

        # ------------------------- State update ---------------------------- #
        # Base pose
        self.base_pos[:] = dofs_pos[:, :3]
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler[:] = quat_to_xyz(self.base_quat, rpy=True, degrees=False)

        # Joint state
        self.joint_position[:] = dofs_pos[:, self.servo_dof_indices]
        self.joint_velocity[:] = self.drone.get_dofs_velocity()[:, self.servo_dof_indices]
        self.torque[:] = self.drone.get_dofs_control_force(self.servo_dof_indices)

        # Velocities (world/body)
        inv_base = inv_quat(self.base_quat)
        self.base_lin_vel[:] = self.rigid_solver.get_dofs_velocity()[:, :3]
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base)

        if self.evaluation:
            self.alpha = self.aero_solver.alpha_dbg.to_torch(device=self.device)[0,0]
            self.beta = self.aero_solver.beta_dbg.to_torch(device=self.device)[0,0]

        # ------------------------- IMU readings --------------------------- #
        '''
        imu_data = self.imu.read()
        lin_acc = imu_data.lin_acc      # shape: (num_envs, 3)
        ang_vel = imu_data.ang_vel      # shape: (num_envs, 3)

        # Se vuoi essere super sicuro del dtype:
        lin_acc = lin_acc.to(dtype=torch.float32, device=self.device)
        ang_vel = ang_vel.to(dtype=torch.float32, device=self.device)

        self.accelerations[:, 0:3] = lin_acc
        self.accelerations[:, 3:6] = ang_vel
        '''
        # ------------------------- Terminations ---------------------------- #
        self.collision = self.check_collision()
        self.success = self.check_success()

        self.wall_crash_condition = (
            (torch.abs(self.base_pos[:, 1]) > self.env_cfg.get("termination_if_y_greater_than", 100.0))
            | (self.base_pos[:, 2] < self.env_cfg.get("termination_if_close_to_ground", 1.0))
        )

        self.angle_limit_condition = (
            (torch.abs(self.base_euler[:, 0]) > getattr(self, "roll_limit_rad", math.radians(80.0)))
            | (torch.abs(self.base_euler[:, 1]) > getattr(self, "pitch_limit_rad", math.radians(80.0)))
            | (torch.abs(self.base_euler[:, 2]) > getattr(self, "yaw_limit_rad", math.radians(80.0)))
        )

        self.reset_buf = (
            (self.episode_length_buf >= self.max_episode_length_per_env)
            | self.wall_crash_condition
            | self.angle_limit_condition
            | self.success
            | self.collision
            | self.nan_envs.to(torch.bool)
        )

        just_reset = self.reset_buf.clone()  # envs that will be reset this step

        self.pre_wall_crash[just_reset] = self.wall_crash_condition[just_reset]
        self.pre_angle_limit[just_reset] = self.angle_limit_condition[just_reset]
        self.pre_collision[just_reset] = self.collision[just_reset]
        self.pre_success[just_reset] = self.success[just_reset]
        self.pre_nan[just_reset] = self.nan_envs[just_reset].to(torch.bool)

        # Time-out mask (episode ended without crash/success/collision)
        timeout = (self.episode_length_buf >= self.max_episode_length_per_env) & ~(
            self.success | self.collision | self.wall_crash_condition | self.angle_limit_condition
        )
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, dtype=torch.float32)
        self.extras["time_outs"][timeout] = 1.0

        # ------------------------- Depth sensing --------------------------- #
        cyl_xy = None
        if self.cylinders_array is not None:
            # cylinders_array: (F, T, 3) → (B, T, 2) via forest_ids
            cyl_xy = self.cylinders_array[self.forest_ids, :, :2]

        self.depth = self.depth_solver.compute_depth(
            base_pos=self.base_pos,
            base_euler=self.base_euler,
            cyl_xy_b=cyl_xy,
            noise_std=float(self.obs_cfg.get("depth_noise_std", 0.0)),
        )
        self.power = self.power_consumption()

        # ------------------------- Rewards -------------------------------- #
        self.rew_buf[:] = 0.0
        self.last_reward_components.zero_()

        if self.reward_names:
            for i, name in enumerate(self.reward_names):
                rew_comp = self.reward_functions[name]() * self.reward_scales[name] * self.dt * 50.0
                self.rew_buf += rew_comp
                self.episode_sums[name] += rew_comp
                self.last_reward_components[:, i] = rew_comp
            self.last_reward_total[:] = self.rew_buf

        # ------------------------- Observations ---------------------------- #
        depth_actor = self.depth if self.include_depth else None

        obs_actor, obs_critic = self.obs_builder.build_observations(
            base_pos=self.base_pos,
            base_quat=self.base_quat,
            base_lin_vel=self.base_lin_vel,
            last_actions=self.last_actions,
            commands=self.commands,
            depth_actor=depth_actor,
        )

        self.obs_buf.copy_(obs_actor)
        self.privileged_obs_buf.copy_(obs_critic)

        self.extras.setdefault("observations", {})
        self.extras["observations"]["critic"] = self.privileged_obs_buf
        self.extras["reward_components"] = self.last_reward_components
        self.extras["reward_total"] = self.last_reward_total

        # ------------------------- Follow-camera video -------------------- #
        if ((self.show_viewer and getattr(self.scene, "viewer", None) is not None)
            or (self._video_on and self.rec_cam is not None)):
            self._update_follow_camera()

        if self._video_on and self.rec_cam is not None:
            # Render one frame for the recording camera.
            self.rec_cam.render()

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ---------------------------------------------------------------------- #
    # Reset                                                                 #
    # ---------------------------------------------------------------------- #
    @torch.inference_mode()
    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset a subset of environments by index."""
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(self.device, dtype=torch.long)

        # Resample command (target speed) and forest layout
        self._resample_commands(env_ids)
        new_ids = torch.randint(
            low=0,
            high=self.cylinders_array.shape[0],
            size=(env_ids.numel(),),
            device=self.device,
            dtype=torch.long,
        )
        self.forest_ids[env_ids] = new_ids

        # Log episode statistics for finished episodes
        self.extras["episode"] = {}
        for key, total in self.episode_sums.items():
            self.extras["episode"]["rew_" + key] = float(total[env_ids].mean())
            self.episode_sums[key][env_ids] = 0.0

        self.extras["episode"]["final_x"] = float(self.base_pos[env_ids, 0].mean())
        self.extras["episode"]["final_y"] = float(self.base_pos[env_ids, 1].mean())
        self.extras["episode"]["final_z"] = float(self.base_pos[env_ids, 2].mean())
        self.extras["episode"]["num_wall_crashed"] = float(self.wall_crash_condition[env_ids].sum())
        self.extras["episode"]["num_angle_crashed"] = float(self.angle_limit_condition[env_ids].sum())
        self.extras["episode"]["num_success"] = float(self.success[env_ids].sum())
        self.extras["episode"]["num_collision"] = float(self.collision[env_ids].sum())

        # Reset state
        n = env_ids.numel()

        if self.evaluation:
            base_init_pos = torch.tensor(
                self.env_cfg.get("base_init_pos", [0.0, 0.0, 1.0]),
                device=self.device,
                dtype=torch.float32,
            )
            base_init_quat = torch.tensor(
                self.env_cfg.get("base_init_quat", [1.0, 0.0, 0.0, 0.0]),
                device=self.device,
                dtype=torch.float32,
            )
            self.base_init_pos = base_init_pos
            self.base_init_quat = base_init_quat

        self.base_pos[env_ids] = self.base_init_pos
        self.base_quat[env_ids] = self.base_init_quat.reshape(1, -1)

        # Euler angles and velocities
        base_euler_init = quat_to_xyz(self.base_init_quat, rpy=True, degrees=False)
        self.base_euler[env_ids] = base_euler_init.reshape(1, -1).expand(n, -1)
        self.joint_position[env_ids] = torch.zeros((len(env_ids), self.num_servos), device=self.device)
        self.joint_velocity[env_ids] = torch.zeros((len(env_ids), self.num_servos), device=self.device)
        self.torque[env_ids] = torch.zeros((len(env_ids), self.num_servos), device=self.device)

        self.base_lin_vel[env_ids] = torch.tensor([10.0, 0.0, 0.0], device=self.device).repeat(n, 1)
        self.base_ang_vel[env_ids] = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(n, 1)

        # Training: inject randomness in initial pose and speed
        if not self.evaluation:
            # Longitudinal position
            self.base_pos[env_ids, 0] += torch.rand(n, device=self.device) * 30.0 - 15.0
            # Lateral position
            self.base_pos[env_ids, 1] += torch.rand(n, device=self.device) * 80.0 - 40.0
            # Altitude
            self.base_pos[env_ids, 2] += torch.rand(n, device=self.device) * 8.0 - 4.0

            # Forward speed 
            self.base_lin_vel[env_ids, 0] = torch.rand(n, device=self.device) * 18.0 + 6.0
            # Lateral speed
            self.base_lin_vel[env_ids, 1] = torch.clamp(
                torch.randn(n, device=self.device) * 1.5, min=-6.0, max=6.0
            )
            # Vertical speed
            self.base_lin_vel[env_ids, 2] = torch.clamp(
                torch.randn(n, device=self.device) * 1.5, min=-6.0, max=6.0
            )

            self.base_euler[env_ids, 1] = torch.atan2(-self.base_lin_vel[env_ids, 2], self.base_lin_vel[env_ids, 0])
            self.base_euler[env_ids, 2] = torch.atan2(self.base_lin_vel[env_ids, 1], self.base_lin_vel[env_ids, 0])

            # Small attitude perturbations
            self.base_euler[env_ids, 0] += torch.clamp(torch.randn(n, device=self.device) * 0.04, min=-0.2, max=0.2)
            self.base_euler[env_ids, 1] += torch.clamp(torch.randn(n, device=self.device) * 0.04, min=-0.2, max=0.2)
            self.base_euler[env_ids, 2] += torch.clamp(torch.randn(n, device=self.device) * 0.04, min=-0.2, max=0.2)

            # Joint positions noise
            self.joint_position[env_ids] += torch.clamp(torch.randn(
                (n, self.num_servos), device=self.device
            ) * 0.004, min=-0.02, max=0.02)

        # Apply quaternion back from Euler
        self.base_quat[env_ids] = xyz_to_quat(self.base_euler[env_ids], degrees=False)

        # Apply to rigid solver
        initial_pos = torch.cat(
            (self.base_pos[env_ids], self.base_euler[env_ids], self.joint_position[env_ids]),
            dim=1,
        )
        initial_vel = torch.cat(
            (self.base_lin_vel[env_ids], self.base_ang_vel[env_ids], self.joint_velocity[env_ids]),
            dim=1,
        )

        self.rigid_solver.set_dofs_position(initial_pos, envs_idx=env_ids)
        self.rigid_solver.set_dofs_velocity(initial_vel, envs_idx=env_ids)

        # Optional aero parameter randomization
        if hasattr(self.aero_solver, "_enable_noise"):
            if hasattr(self.rigid_solver, "randomize_aero_params"):
                self.aero_solver.randomize_aero_params(env_ids)

        # Reset actuator dynamics state (latency buffer)
        self.actuator.reset_envs(env_ids)

        # Reset bookkeeping
        self.last_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = True
        self.success[env_ids] = False
        self.collision[env_ids] = False
        self.wall_crash_condition[env_ids] = False
        self.angle_limit_condition[env_ids] = False
        self.nan_envs[env_ids] = 0

    @torch.inference_mode()
    def reset(self):
        """Reset all environments and return initial observations."""
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_idx(env_ids)

        # Depth default: no obstacles seen at reset
        self.depth.fill_(self.MAX_DISTANCE)

        depth_actor = self.depth if self.include_depth else None

        obs_actor, obs_critic = self.obs_builder.build_observations(
            base_pos=self.base_pos,
            base_quat=self.base_quat,
            base_lin_vel=self.base_lin_vel,
            last_actions=self.last_actions,
            commands=self.commands,
            depth_actor=depth_actor,
        )

        self.obs_buf.copy_(obs_actor)
        self.privileged_obs_buf.copy_(obs_critic)

        self.extras.setdefault("observations", {})
        self.extras["observations"]["critic"] = self.privileged_obs_buf

        return self.obs_buf, self.extras

    # ---------------------------------------------------------------------- #
    # Power consumption                                                      #
    # ---------------------------------------------------------------------- #
    def power_consumption(self) -> torch.Tensor:
        """
        Approximate total power consumption for current state.

        - Propeller power: based on throttle fraction * max_thrust.
        - Servo power: torque * angular velocity.
        """
        self.thrust_log = self.extract_thrust().unsqueeze(1)  # (B, 1)

        total_power = power_utils.compute_power_consumption(
            thrust=self.thrust_log,
            servo_torque=self.torque,
            servo_velocity=self.joint_velocity,
            drone_name=self.drone_name,
            device=self.device,
        )
        return total_power
    
    # ---------------------------------------------------------------------- #
    # Thrust extraction                                                      #
    # ---------------------------------------------------------------------- #
    def extract_thrust(self) -> torch.Tensor:
        """
        Extract the *actual filtered thrust* [N] from the Taichi AeroSolver.

        Returns:
            torch.Tensor of shape (num_envs,) on the same device as env.device.
        """
        if not hasattr(self, "aero_solver"):
            raise RuntimeError("AeroSolver not initialized in this environment.")
        if not hasattr(self.aero_solver, "_thr_flt"):
            raise RuntimeError("AeroSolver._thr_flt not found (did you call add_target?).")

        # Convert Taichi fields to Torch tensors on the correct device
        thr_flt = self.aero_solver._thr_flt.to_torch(device=self.device)     # (B,)
        max_thr = self.aero_solver.max_thrust.to_torch(device=self.device)   # (B,)

        # Elementwise multiplication → actual thrust (in Newtons)
        thrust_N = thr_flt * max_thr

        # Optional: clamp NaNs or negatives
        thrust_N = torch.nan_to_num(thrust_N, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        return thrust_N

    # ---------------------------------------------------------------------- #
    # Convenience getters                                                    #
    # ---------------------------------------------------------------------- #
    def get_observations(self):
        """Return actor observations and critic observations (for logging)."""
        return self.obs_buf, {"observations": {"critic": self.privileged_obs_buf}}

    def get_privileged_observations(self):
        """Return critic observations and full extras dict."""
        return self.privileged_obs_buf, dict(self.extras)

    # ---------------------------------------------------------------------- #
    # Collision / success checks                                             #
    # ---------------------------------------------------------------------- #
    def check_collision(self, tol: float = 0.01) -> torch.Tensor:
        """
        Detect collisions with trunks using a rectangle in the body frame.

        The rectangle spans [-r_tree, +r_tree] along X, and +/- (span/2) along Y,
        expanded by `tol` and modulated by roll (wings appear narrower in bank).
        """
        half_span = self.span / 2.0
        r_tree = float(self.env_cfg.get("tree_radius", 1.0)) + tol

        cyl_xy = self.cylinders_array[self.forest_ids, :, :2]  # (B, T, 2)
        drone_xy = self.base_pos[:, :2].unsqueeze(1)           # (B, 1, 2)
        diff = cyl_xy - drone_xy                               # (B, T, 2)

        yaw = self.base_euler[:, 2].unsqueeze(1)
        roll = torch.abs(self.base_euler[:, 0])

        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)

        x_b = diff[..., 0] * cos_y + diff[..., 1] * sin_y
        y_b = -diff[..., 0] * sin_y + diff[..., 1] * cos_y

        half_span_eff = half_span * torch.cos(roll).unsqueeze(1)

        hit_x = (x_b >= -r_tree) & (x_b <= r_tree)
        hit_y = (y_b.abs() <= half_span_eff + r_tree)

        return (hit_x & hit_y).any(dim=1)

    def check_success(self) -> torch.Tensor:
        """
        Success when the drone passes beyond the forest along +X.
        """
        forest_x_limit = float(self.env_cfg.get("forest_x_limit", 250.0))
        if self.evaluation:
            forest_x_limit = float(self.env_cfg.get("x_upper", 500.0))
        return self.base_pos[:, 0] > forest_x_limit

    # ---------------------------------------------------------------------- #
    # Rewards                                                                #
    # ---------------------------------------------------------------------- #
    def _reward_smooth(self) -> torch.Tensor:
        """Penalize large action changes (smooth control)."""
        return torch.sum((self.actions - self.last_actions) ** 2, dim=1)

    def _reward_angular(self) -> torch.Tensor:
        """Penalize angular velocity magnitude."""
        return torch.norm(self.base_ang_vel, dim=1)

    def _reward_stability(self) -> torch.Tensor:
        """Penalize large roll/pitch angles."""
        return torch.abs(self.base_euler[:, 0]) + torch.abs(self.base_euler[:, 1])

    def _reward_crash(self) -> torch.Tensor:
        """Penalty for crash or collision (1 on crash/collision)."""
        crash = torch.zeros((self.num_envs,), device=self.device)
        crash[self.wall_crash_condition | self.collision | self.angle_limit_condition] = 1.0
        # Rescale so that a single crash produces ~O(1) penalty per episode
        return crash / (self.dt * 100.0)

    def _reward_energy(self) -> torch.Tensor:
        """Penalize energy consumption (higher power -> lower reward)."""
        energy = self.power
        return energy / (10.0 * self.dt)

    def _reward_progress(self, sigma: float = 0.15) -> torch.Tensor:
        """
        Reward for forward speed tracking.

        Target:
            v_proj (along +X) ≈ v_target (commands[:,0]).
        """
        # Desired direction fixed along +X (heading = 0)
        desired_dir = torch.tensor([1.0, 0.0], device=self.device)
        v_xy = self.base_lin_vel[:, :2]
        v_proj = torch.sum(v_xy * desired_dir, dim=1)

        v_tgt = self.commands[:, 0].clamp(min=1e-3)
        x = v_proj / v_tgt
        return torch.exp(-0.5 * ((x - 1.0) / sigma) ** 2)

    def _reward_obstacle(self) -> torch.Tensor:
        """
        Reward for staying away from obstacles.

        Uses an anisotropic "distance" in body frame, derived from depth:
            d = sqrt(x^2 + k * y^2) / scale
        and aggregates via an exponential kernel.
        """
        depth_vals = self.depth  # (B, S)

        # Ray directions in body frame (broadcast over batch)
        dir_x = self.ray_dir_x.unsqueeze(0)  # (1, S)
        dir_y = self.ray_dir_y.unsqueeze(0)  # (1, S)

        x_loc = depth_vals * dir_x
        y_loc = depth_vals * dir_y

        anisotropic_distance = torch.sqrt(x_loc**2 + 24.0 * y_loc**2) / 5.0
        alpha = 1.5
        clipped = torch.clamp_min(anisotropic_distance, 0.5)
        r_obs = torch.sum(torch.exp(-alpha * clipped), dim=1)
        return r_obs

    def _reward_height(self) -> torch.Tensor:
        """
        Penalize deviation from a fixed target height, especially when flying low.
        """
        dist = (self.base_pos[:, 2] - self.target_height) ** 2
        penalty = torch.zeros_like(dist, device=self.device)
        low = self.base_pos[:, 2] < self.target_height
        penalty[low] = dist[low]
        return penalty

    def _reward_success(self) -> torch.Tensor:
        """Reward for successful completion of the mission."""
        r = torch.zeros((self.num_envs,), device=self.device)
        r[self.success] = 1.0
        return r

    def _reward_cosmetic(self) -> torch.Tensor:
        """
        Cosmetic reward encouraging symmetric joint usage.

        This is optional and should have a small weight.
        """
        cosmetic = torch.zeros((self.num_envs,), device=self.device)
        if self.num_servos >= 4:
            cosmetic += (self.joint_position[:, 0] + self.joint_position[:, 1]) ** 2
            cosmetic += (self.joint_position[:, 2] - self.joint_position[:, 3]) ** 2
        if self.num_servos > 5:
            cosmetic += (self.joint_position[:, 5]) ** 2
        return cosmetic

    # ---------------------------------------------------------------------- #
    # Internal: tree visualization                                           #
    # ---------------------------------------------------------------------- #
    def _spawn_tree_visuals(self) -> None:
        """
        Create simple cylinders for visualization of trunks.

        Only called in evaluation mode with few envs, so cost is negligible
        compared to training.
        """
        tree_radius = float(self.env_cfg.get("tree_radius", 1.0))
        tree_height = float(self.env_cfg.get("tree_height", 20.0))

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
