# env.py
from __future__ import annotations

import csv
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Sequence, Optional, List

import torch
import genesis as gs
from genesis.utils import geom as gu
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, xyz_to_quat
from genesis.assets.urdf.aero_model import DroneAeroModel, SurfaceKind

from morph_evolution.chromosome_drone import Chromosome_Drone
from winged_drone_train.aero_profile import (
    configure_runtime_aero_solver,
    resolve_aero_config,
)
from winged_drone_train.perception import depth as depth_utils
from winged_drone_train.perception import forest as forest_utils
from winged_drone_train.perception import obs as obs_utils
from winged_drone_train.control import power as power_utils
from winged_drone_train.defaults import resolve_default_urdf_for_drone


LISPARROW_SERVO_JOINT_NAMES: tuple[str, ...] = (
    "joint_left_outer_wing_hinged",
    "joint_right_outer_wing_hinged",
    "joint_elevator_hinged",
    "joint_rudder_hinged",
)


def _scalar_like_to_float(value, device: torch.device | str | None = None) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "to_torch"):
        try:
            tensor = value.to_torch(device=device or "cpu")
            if torch.is_tensor(tensor) and tensor.numel() > 0:
                return float(tensor.reshape(-1)[0].item())
        except Exception:
            pass
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _scalar_like_to_float(value[0], device=device)
    try:
        return float(value)
    except Exception:
        return None


def _resolve_wingspan_from_aero_config(aero_config: Optional[Dict]) -> Optional[float]:
    if not aero_config:
        return None
    links = aero_config.get("links", {}) or {}
    total_span = 0.0
    for info in links.values():
        if not isinstance(info, dict):
            continue
        if str(info.get("type", "")).strip().lower() != "wing":
            continue
        span = _scalar_like_to_float(info.get("span"))
        if span is None or span <= 0.0:
            continue
        total_span += span
    if total_span <= 0.0:
        return None
    return float(total_span)


def _resolve_prop_diameter(
    aero_solver,
    aero_config: Optional[Dict],
    device: torch.device | str,
) -> torch.Tensor:
    radius = _scalar_like_to_float(getattr(aero_solver, "prop_radius", None), device=device)
    if radius is None or radius <= 0.0:
        global_cfg = (aero_config or {}).get("global", {}) or {}
        radius = _scalar_like_to_float(global_cfg.get("prop_radius"))
    if radius is None or radius <= 0.0:
        raise RuntimeError("Unable to resolve propeller radius from aero solver or aero config.")
    return torch.tensor([2.0 * radius], device=device, dtype=torch.float32)


def _infer_drone_profile(drone_key: str | None, urdf_path: str | None = None) -> str:
    key = (drone_key or "").strip().lower()
    if "lisparrow" in key:
        return "lisparrow"
    if urdf_path:
        up = str(urdf_path).strip().lower()
        if "lisparrow" in up:
            return "lisparrow"
    return "simple"


def _apply_drone_profile_defaults(env_cfg: Dict, urdf_path: str | None = None) -> Dict:
    cfg = dict(env_cfg)
    profile = _infer_drone_profile(cfg.get("drone"), urdf_path)
    if profile != "lisparrow":
        return cfg

    cfg["drone"] = "lisparrow"
    cfg.setdefault("aero_solver_kind", "lisparrow")
    cfg.setdefault("servo_joint_names", list(LISPARROW_SERVO_JOINT_NAMES))
    cfg.setdefault("fallback_servo_gains", (20.0, 2.0))
    if cfg.get("naca", None):
        cfg["naca"] = None
    return cfg


def _classify_joint_role(name: str) -> Optional[str]:
    lname = name.lower()
    if "sweep" in lname:
        if "left" in lname:
            return "sweep_left"
        if "right" in lname:
            return "sweep_right"
        return "sweep"
    if "outer_wing" in lname:
        if "left" in lname:
            return "sweep_left"
        if "right" in lname:
            return "sweep_right"
        return "sweep"
    if "twist" in lname:
        if "left" in lname:
            return "twist_left"
        if "right" in lname:
            return "twist_right"
        return "twist"
    if "elevator" in lname:
        return "elevator"
    if "rudder" in lname:
        return "rudder"
    return None


def _resolve_servo_joint_names(
    urdf_path: str,
    preferred_names: Optional[Sequence[str]] = None,
) -> List[str]:
    root = ET.parse(str(urdf_path)).getroot()
    joint_nodes = root.findall(".//joint")
    joint_names = [j.get("name") for j in joint_nodes if j.get("name")]
    joint_set = set(joint_names)

    if preferred_names is not None:
        missing = [name for name in preferred_names if name not in joint_set]
        if missing:
            raise ValueError(f"Missing servo joints in URDF: {missing}")
        return list(preferred_names)

    role_to_name: Dict[str, str] = {}
    for joint in joint_nodes:
        jtype = (joint.get("type") or "").strip().lower()
        if jtype not in ("revolute", "continuous"):
            continue
        name = joint.get("name")
        if not name:
            continue
        role = _classify_joint_role(name)
        if role and role not in role_to_name:
            role_to_name[role] = name

    ordered_roles = [
        "sweep_left",
        "sweep_right",
        "twist_left",
        "twist_right",
        "elevator",
        "rudder",
        "sweep",
        "twist",
    ]
    return [role_to_name[role] for role in ordered_roles if role in role_to_name]


def _servo_gains_from_catalog(
    drone_model: DroneAeroModel,
    joint_names: Sequence[str],
    device: torch.device,
    fallback_gains: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fetch per-joint kp/kv gains using actuator assignments from DroneAeroModel."""
    if not joint_names:
        empty = torch.zeros((0,), device=device, dtype=torch.float32)
        return empty, empty
    if drone_model is None or not getattr(drone_model, "urdf_path", None):
        raise ValueError("servo gain loading requires a DroneAeroModel with a valid urdf_path.")

    csv_path = Path(str(drone_model.urdf_path)).parent / "actuators.csv"
    if not csv_path.exists():
        if fallback_gains is None:
            raise FileNotFoundError(f"Missing actuator catalog: {csv_path}")
        kp_f, kv_f = fallback_gains
        kp = torch.full((len(joint_names),), float(kp_f), device=device, dtype=torch.float32)
        kv = torch.full((len(joint_names),), float(kv_f), device=device, dtype=torch.float32)
        return kp, kv

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

    def side_from_name(name: str) -> Optional[str]:
        lname = name.lower()
        if "left" in lname:
            return "left"
        if "right" in lname:
            return "right"
        return None

    frame_kind: Dict[str, SurfaceKind] = {}
    if hasattr(drone_model, "frames") and hasattr(drone_model, "surface_kinds"):
        for frame, kind in zip(drone_model.frames, drone_model.surface_kinds):
            frame_kind[frame] = kind

    act_by_kind_side = {}
    for frame, info in getattr(drone_model, "actuators", {}).items():
        kind = frame_kind.get(frame)
        if kind is None:
            continue
        side = side_from_name(frame)
        key = (kind, side)
        if key not in act_by_kind_side:
            act_by_kind_side[key] = info

    def pick_info(kind: SurfaceKind, side: Optional[str]):
        info = act_by_kind_side.get((kind, side))
        if info is not None:
            return info
        info = act_by_kind_side.get((kind, None))
        if info is not None:
            return info
        for key in ((kind, "left"), (kind, "right")):
            info = act_by_kind_side.get(key)
            if info is not None:
                return info
        return None

    kp_list: List[float] = []
    kv_list: List[float] = []
    for name in joint_names:
        lname = name.lower()
        side = side_from_name(name)
        act = None
        info = None
        if "sweep" in lname or "outer_wing" in lname:
            info = pick_info(SurfaceKind.WING, side)
            if info is not None:
                act = info.yaw_actuator or info.actuator
        elif "twist" in lname:
            info = pick_info(SurfaceKind.WING, side)
            if info is not None:
                act = info.pitch_actuator or info.actuator
        elif "elevator" in lname:
            info = pick_info(SurfaceKind.ELEVATOR, side)
            if info is not None:
                act = info.pitch_actuator or info.actuator
        elif "rudder" in lname:
            info = pick_info(SurfaceKind.RUDDER, side)
            if info is not None:
                act = info.yaw_actuator or info.actuator

        if not act:
            if fallback_gains is None:
                raise ValueError(
                    f"Missing actuator assignment for joint '{name}' (check aero solver actuator config)."
                )
            kp_i, kv_i = fallback_gains
        else:
            kp_i, kv_i = get_servo_gains(act)
        kp_list.append(float(kp_i))
        kv_list.append(float(kv_i))

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

    @staticmethod
    def _resolve_aero_config(solver_kind: str, urdf_file: Optional[str] = None) -> dict:
        candidates: List[Path] = []
        env_path = os.getenv("AERO_CONFIG_PATH", "").strip()
        if env_path:
            candidates.append(Path(env_path))
        if urdf_file:
            candidates.append(Path(str(urdf_file)).expanduser().resolve().parent / "aero_parameters.yaml")

        try:
            import yaml as _yaml  # type: ignore
        except Exception:
            _yaml = None

        if _yaml is not None:
            for path in candidates:
                try:
                    if path.is_file():
                        with open(path, "r") as f:
                            cfg = _yaml.safe_load(f)
                        if isinstance(cfg, dict):
                            return cfg
                except Exception:
                    pass

        return resolve_aero_config(solver_kind)

    def _resolve_property_randomization_cfg(self) -> Dict[str, float]:
        cfg = dict(self.env_cfg.get("property_randomization", {}) or {})
        noise_cfg = self._aero_config.get("noise", {}) or {}

        def _get_std(name: str, legacy_name: Optional[str] = None) -> float:
            if name in cfg and cfg.get(name) is not None:
                return float(cfg.get(name, 0.0) or 0.0)
            if legacy_name is not None:
                return float(noise_cfg.get(legacy_name, 0.0) or 0.0)
            return 0.0

        return {
            "mass_shift_std": _get_std("mass_shift_std", legacy_name="mass_shift"),
            "com_shift_std": _get_std("com_shift_std", legacy_name="com_shift"),
            "joint_target_episode_bias_std": _get_std("joint_target_episode_bias_std"),
            "joint_target_step_noise_std": _get_std("joint_target_step_noise_std"),
        }

    def _apply_dynamics_noise(self, env_ids: torch.Tensor) -> None:
        rand_cfg = self._property_rand_cfg
        sigma_mass = float(rand_cfg.get("mass_shift_std", 0.0))
        sigma_com = float(rand_cfg.get("com_shift_std", 0.0))

        n = env_ids.numel()
        if n == 0:
            return
        n_links = int(self.drone.n_links)

        mass_shift = self._mass_shift_scratch[:n]
        mass_shift.zero_()
        if sigma_mass > 0.0:
            mass_shift.normal_()
            mass_shift *= sigma_mass
            mass_shift *= self._link_masses.view(1, n_links)
        self.drone.set_mass_shift(mass_shift, envs_idx=env_ids)

        com_shift = self._com_shift_scratch[:n]
        com_shift.zero_()
        if sigma_com > 0.0:
            com_shift.normal_()
            com_shift *= sigma_com
        self.drone.set_COM_shift(com_shift, envs_idx=env_ids)


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
        auto_reset: bool = True,
    ) -> None:
        env_init_start = time.perf_counter()
        # ------------------------------------------------------------------ #
        # Basic configuration                                               #
        # ------------------------------------------------------------------ #
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.evaluation = bool(eval)
        self.auto_reset = bool(auto_reset)

        # Only ONE command: target forward speed along +X (m/s)
        self.num_commands = 1
        self.command_cfg = dict(command_cfg)
        self.command_cfg["num_commands"] = 1  # keep config consistent
        self._eval_speed_grid: Optional[torch.Tensor] = None

        # Feature toggles
        self.env_cfg = dict(env_cfg)
        self.obs_cfg = dict(obs_cfg)
        self.reward_cfg = dict(reward_cfg)
        self.debug = bool(self.env_cfg.get("debug", False))

        self.growing_forest = self.env_cfg.get("growing_forest", True)
        self.unique_forests_eval = self.env_cfg.get("unique_forests_eval", True)
        self.show_viewer = bool(show_viewer)
        self.enable_rendering = bool(self.env_cfg.get("enable_rendering", True))
        self._tree_radius = float(self.env_cfg.get("tree_radius", 1.0))
        self._collision_tol = 0.01 if self.evaluation else 0.2
        self._termination_abs_y_max = float(
            self.env_cfg.get("termination_if_y_greater_than", 100.0)
        )
        self._termination_min_z = float(
            self.env_cfg.get("termination_if_close_to_ground", 0.1)
        )
        self._roll_limit_train = math.radians(90.0)
        self._pitch_limit_train = math.radians(90.0)
        self._yaw_limit_train = math.radians(90.0)
        self._roll_limit_eval = math.radians(100.0)
        self._pitch_limit_eval = math.radians(90.0)
        self._yaw_limit_eval = math.radians(90.0)
        self._success_x_limit_train = float(self.env_cfg.get("forest_x_limit", 250.0))
        self._success_x_limit_eval = float(self.env_cfg.get("x_upper", 500.0))

        # Action latency (delegated to ActuatorDynamics)
        self.simulate_action_latency = bool(self.env_cfg.get("simulate_action_latency", False))
        self.action_latency_min = int(self.env_cfg.get("action_latency_min_steps", 0))
        self.action_latency_max = int(self.env_cfg.get("action_latency_max_steps", 0))
        self.action_latency_random_per_step = bool(self.env_cfg.get("action_latency_random_per_step", False))

        # Reset randomization toggles
        self.randomize_joint_pos = bool(self.env_cfg.get("randomize_joint_pos", True))

        # For evaluation we enforce a deterministic, fixed latency
        if self.evaluation:
            self.action_latency_min = 0
            self.action_latency_max = 1
            self.action_latency_random_per_step = False
            if self.num_envs > 1:
                v_min = float(self.command_cfg.get("min_speed", 5.0))
                v_max = float(self.command_cfg.get("max_speed", 25.0))
                self._eval_speed_grid = torch.linspace(
                    v_min, v_max, self.num_envs, device=self.device, dtype=torch.float32
                )

        if self.action_latency_max < self.action_latency_min:
            self.action_latency_max = self.action_latency_min

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

        if urdf_file is None:
            env_urdf = self.env_cfg.get("urdf_file")
            if env_urdf:
                urdf_file = env_urdf
            else:
                drone_key = str(self.env_cfg.get("drone", ""))
                urdf_file = str(resolve_default_urdf_for_drone(drone_key))
        self.urdf_file = str(urdf_file)
        if not Path(self.urdf_file).exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_file}")
        self.env_cfg = _apply_drone_profile_defaults(self.env_cfg, self.urdf_file)
        self.aero_solver_kind = str(self.env_cfg.get("aero_solver_kind", "simple")).strip().lower()
        configure_runtime_aero_solver(self.aero_solver_kind)
        self._aero_config = self._resolve_aero_config(self.aero_solver_kind, self.urdf_file)
        self._property_rand_cfg = self._resolve_property_randomization_cfg()
        self.drone_model = DroneAeroModel(self.urdf_file, config_override=self._aero_config)

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
                enable_rendering=self.enable_rendering,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.CG,
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

        # Target height (used in height reward) is fixed, not commanded
        self.target_height = 10

        self.base_init_pos = base_init_pos
        self.base_init_quat = base_init_quat
        self.inv_base_init_quat = inv_quat(base_init_quat)

        self.drone_name = self.env_cfg.get("drone", "morphing_drone")

        # Servo joints
        servo_joint_names = self.env_cfg.get("servo_joint_names", None)
        self.servo_joint_names = _resolve_servo_joint_names(self.urdf_file, servo_joint_names)

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
        # When set from outside (e.g. WP2 evaluation), reset_idx uses this
        # tiled assignment instead of random, so all individuals see the same
        # set of forests (env i of individual k always gets forest i).
        self._fixed_forest_ids: Optional[torch.Tensor] = None
        self.cylinders_xy: Optional[torch.Tensor] = None
        if self.cylinders_array is not None:
            self.cylinders_xy = self.cylinders_array[self.forest_ids, :, :2]

        # Optional visualization of trees (only for eval; training stays lean)
        if self.num_envs == 1:
            self._spawn_tree_visuals()

        # Always define the attribute so eval.py can check it.
        self.rec_cam = None

        # We only support recording when there is a single environment.
        if self.num_envs == 1 and self.evaluation and self.enable_rendering:
            self._create_follow_camera()

        # ------------------------------------------------------------------ #
        # Genome handling (optional)                                        #
        # ------------------------------------------------------------------ #
        self.add_genome_obs_actor = bool(self.obs_cfg.get("add_genome_obs_actor", False))
        self.add_genome_obs_critic = bool(self.obs_cfg.get("add_genome_obs_critic", False))
        self._genome_vec: Optional[torch.Tensor] = None
        self._genome_base_vec: Optional[torch.Tensor] = None
        self._genome_obs_scratch: Optional[torch.Tensor] = None
        self.noise_std = self.obs_cfg.get("noise_std", {})
        self._naca_code: Optional[str] = None
        genome_noise_cfg = self.obs_cfg.get("genome_obs_noise", {}) or {}
        self._genome_episode_noise_std = float(genome_noise_cfg.get("episode_std", 0.0) or 0.0)
        self._genome_step_noise_std = float(genome_noise_cfg.get("step_std", 0.0) or 0.0)
        self._genome_min_tensor = torch.tensor(self.GENOME_MIN, device=self.device, dtype=torch.float32)
        self._genome_max_tensor = torch.tensor(self.GENOME_MAX, device=self.device, dtype=torch.float32)
        self._genome_span_tensor = self._genome_max_tensor - self._genome_min_tensor

        if self.urdf_file:
            match = re.search(r"\[([^\]]+)\]\.urdf$", self.urdf_file)
            if match:
                try:
                    values = [float(x) for x in match.group(1).split(",")]
                    self._naca_code = Chromosome_Drone.naca_from_physical(values)
                    g = torch.tensor(values, dtype=torch.float32, device=self.device)
                    self._genome_vec = g.unsqueeze(0).repeat(self.num_envs, 1)
                    self._genome_base_vec = self._genome_vec.clone()
                except Exception:
                    self._genome_vec = None

        # ------------------------------------------------------------------ #
        # Build scene and get solvers                                       #
        # ------------------------------------------------------------------ #
        scene_build_start = time.perf_counter()
        self.scene.build(n_envs=self.num_envs)
        self.scene_build_elapsed = time.perf_counter() - scene_build_start

        if hasattr(self.scene.sim, "rigid_solver"):
            rigid_solver = self.scene.sim.rigid_solver

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
        self._servo_dof_idx_tensor = torch.as_tensor(
            self.servo_dof_indices, device=self.device, dtype=torch.long
        ).contiguous()


        # Throttle + servos
        self.THROTTLE_SIZE = 1
        self.num_servos = len(self.servo_dof_indices)
        self.num_actions = self.THROTTLE_SIZE + self.num_servos
        self.throttle_limit: Tuple[float, float] = (0.0, 1.0)
        self._power_prop_coeffs = power_utils._default_prop_coeffs(
            self.drone_name,
            self.device,
            1,
            aero_config=self._aero_config,
        )
        self._power_servo_constants = power_utils._default_servo_power_constants(
            self.drone_name, self.device, self.num_servos
        )
        self._power_torque_multipliers = power_utils._default_torque_multipliers(
            self.drone_name,
            self.device,
            self.num_servos,
            sweep_multiplier=2.0,
            twist_multiplier=2.5,
            tail_multiplier=2.0,
        )

        self.drone_model.validate_entity(
            self.drone,
            self.servo_joint_names,
            self.servo_dof_indices,
        )
        self.rigid_solver = self.scene.sim.rigid_solver
        self.aero_solver = self.scene.sim.aero_solver
        if hasattr(self.aero_solver, "_aero_log"):
            self.aero_solver._aero_log = bool(self.evaluation or self.debug)
        self.aero_solver.add_target(self.drone, drone_model=self.drone_model)
        naca_code = self._naca_code or str(self.env_cfg.get("naca", "") or "").strip()
        if naca_code and hasattr(self.aero_solver, "apply_naca_wing_override"):
            self.aero_solver.apply_naca_wing_override(naca_code)

        span_value = _scalar_like_to_float(getattr(self.aero_solver, "tip_to_tip", None), device=self.device)
        if span_value is None or span_value <= 0.0:
            span_value = _resolve_wingspan_from_aero_config(self._aero_config)
        if span_value is None or span_value <= 0.0:
            raise RuntimeError("Unable to resolve wingspan from aero solver or aero config.")
        self.span = float(span_value)
        self.nominal_mass = float(sum(link.get_mass() for link in self.drone.links))
        self._link_masses = torch.tensor(
            [link.get_mass() for link in self.drone.links],
            device=self.device,
            dtype=torch.float32,
        )
        n_links = int(self.drone.n_links)
        self._mass_shift_scratch = torch.empty(
            (self.num_envs, n_links), device=self.device, dtype=torch.float32
        )
        self._com_shift_scratch = torch.empty(
            (self.num_envs, n_links, 3), device=self.device, dtype=torch.float32
        )
        # ------------------------------------------------------------------ #
        # Setup drone actuators                                          #
        # ------------------------------------------------------------------ #

        # Joint limits for servos
        if self.servo_dof_indices:
            joint_mins, joint_maxs = self.drone.get_dofs_limit(self.servo_dof_indices)
            joint_mins = torch.as_tensor(joint_mins, device=self.device, dtype=torch.float32)
            joint_maxs = torch.as_tensor(joint_maxs, device=self.device, dtype=torch.float32)
            self.joint_limit_min = joint_mins
            self.joint_limit_max = joint_maxs
        else:
            self.joint_limit_min = torch.zeros((0,), device=self.device, dtype=torch.float32)
            self.joint_limit_max = torch.zeros((0,), device=self.device, dtype=torch.float32)

        # PD gains for servo position control
        fallback_gains = self.env_cfg.get("fallback_servo_gains")
        fallback_tuple = None
        if fallback_gains is not None:
            if not isinstance(fallback_gains, (list, tuple)) or len(fallback_gains) != 2:
                raise ValueError("fallback_servo_gains must be a (kp, kv) pair.")
            fallback_tuple = (float(fallback_gains[0]), float(fallback_gains[1]))
        kp, kv = _servo_gains_from_catalog(
            self.drone_model,
            self.servo_joint_names,
            self.device,
            fallback_gains=fallback_tuple,
        )
        self.base_kp = kp.clone()
        self.base_kv = kv.clone()
        if self.num_servos > 0:
            self.drone.set_dofs_kp(kp, self.servo_dof_indices)
            self.drone.set_dofs_kv(kv, self.servo_dof_indices)

        # ------------------------------------------------------------------ #
        # Depth solver + precomputed ray directions                          #
        # ------------------------------------------------------------------ #
        tree_radius = self._tree_radius
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
            backend=self.obs_cfg.get("depth_backend", "torch"),
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
            add_genome_obs_actor=self.add_genome_obs_actor and (self._genome_vec is not None),
            add_genome_obs_critic=self.add_genome_obs_critic and (self._genome_vec is not None),
            include_joint_pos_critic=bool(self.obs_cfg.get("include_joint_pos_critic", False)),
            include_joint_vel_critic=bool(self.obs_cfg.get("include_joint_vel_critic", False)),
            include_ang_vel_critic=bool(self.obs_cfg.get("include_ang_vel_critic", False)),
            include_effective_thrust_critic=bool(self.obs_cfg.get("include_effective_thrust_critic", False)),
            genome_vec=self._genome_vec,
            genome_min=self.GENOME_MIN if self._genome_vec is not None else None,
            genome_max=self.GENOME_MAX if self._genome_vec is not None else None,
            num_servos=self.num_servos,
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
        self._base_quat_inv_buf = torch.empty((self.num_envs, 4), device=self.device)
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._base_ang_vel_body_buf = torch.empty((self.num_envs, 3), device=self.device)
        # Reused at every reset to avoid repeated tensor allocations.
        self._reset_lin_vel = torch.tensor([15.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self._reset_ang_vel = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self._rec_cam_lookat_offset = torch.tensor([1.5, 0.0, 0.0], device=self.device, dtype=torch.float32)
        # Scratch random buffers reused in reset/randomization paths.
        self._rand_scalar_scratch = torch.empty((self.num_envs,), device=self.device, dtype=torch.float32)
        self._rand_servo_scratch = torch.empty((self.num_envs, self.num_servos), device=self.device, dtype=torch.float32)
        self._randint_scratch = torch.empty((self.num_envs,), device=self.device, dtype=torch.long)
        self._joint_target_episode_bias = torch.zeros(
            (self.num_envs, self.num_servos), device=self.device, dtype=torch.float32
        )

        self.joint_position = torch.zeros((self.num_envs, self.num_servos), device=self.device)
        self.joint_velocity = torch.zeros((self.num_envs, self.num_servos), device=self.device)
        self.torque = torch.zeros((self.num_envs, self.num_servos), device=self.device)
        self._dofs_pos_buf = torch.empty((self.num_envs, self.drone.n_dofs), device=self.device, dtype=torch.float32)
        # Scratch buffers reused on reset to avoid per-reset `torch.cat` allocations.
        self._reset_pos_scratch = torch.empty(
            (self.num_envs, 6 + self.num_servos), device=self.device, dtype=torch.float32
        )
        self._reset_vel_scratch = torch.empty(
            (self.num_envs, 6 + self.num_servos), device=self.device, dtype=torch.float32
        )

        # Action buffers (normalized space, [-1, 1])
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)

        # Commands: single scalar forward speed per env
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device)

        # Logging extras
        self.thrust_log = torch.zeros((self.num_envs, 1), device=self.device)
        self.prop_rpm_log = torch.zeros((self.num_envs, 1), device=self.device)
        self.prop_axial_speed_log = torch.zeros((self.num_envs, 1), device=self.device)
        self.alpha = torch.zeros((self.num_envs, 1), device=self.device)
        self.beta = torch.zeros((self.num_envs, 1), device=self.device)
        self.d_cf_com_body = torch.zeros((self.num_envs, 3), device=self.device)
        self._thr_flt_buf = torch.empty((self.num_envs,), device=self.device, dtype=torch.float32)
        self._max_thr_buf = torch.empty((self.num_envs,), device=self.device, dtype=torch.float32)
        self._thrust_buf = torch.empty((self.num_envs,), device=self.device, dtype=torch.float32)
        self._prop_rpm_buf = torch.empty((self.num_envs,), device=self.device, dtype=torch.float32)
        self._prop_axial_speed_buf = torch.empty((self.num_envs,), device=self.device, dtype=torch.float32)
        self._prop_diameter = _resolve_prop_diameter(self.aero_solver, self._aero_config, self.device)

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
        self._time_outs = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        # Extras dictionary for logging (RSL-RL convention)
        self.extras: Dict = {"observations": {}}
        self._video_on = False
        self._refresh_max_thrust_cache()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if bool(self.env_cfg.get("warmup_runtime_kernels", False)):
            self._warmup_runtime_kernels()
        self.scene_init_elapsed = time.perf_counter() - env_init_start

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

        # Offsets in meters (body frame), can be tuned via env_cfg.
        dist_back = float(self.env_cfg.get("rec_cam_follow_distance", 3.0))
        height_offset = float(self.env_cfg.get("rec_cam_follow_height", 1.0))

        # Smoothing factor for exponential filter (0 → very smooth, 1 → no filter).
        alpha = float(self.env_cfg.get("rec_cam_smooth_alpha", 0.25))

        # World position of the drone (we look at this point).
        px = float(pos[0].item())
        py = float(pos[1].item())
        pz = float(pos[2].item())

        # Desired camera position in world frame:
        # behind the drone along -heading, plus optional lateral offset and height.
        back_x = px - dist_back
        back_y = py
        back_z = pz + height_offset

        target_pos = torch.empty((3,), device=self.device, dtype=torch.float32)
        target_pos[0] = back_x
        target_pos[1] = back_y
        target_pos[2] = back_z
        target_lookat = pos + self._rec_cam_lookat_offset
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
            # Linearly spaced fixed speeds for all eval envs.
            if self._eval_speed_grid is None or self._eval_speed_grid.shape[0] != self.num_envs:
                v_min = float(self.command_cfg.get("min_speed", 5.0))
                v_max = float(self.command_cfg.get("max_speed", 25.0))
                self._eval_speed_grid = torch.linspace(
                    v_min, v_max, self.num_envs, device=self.device, dtype=torch.float32
                )
            self.commands[env_ids, 0] = self._eval_speed_grid[env_ids]
        else:
            v_min = float(self.command_cfg.get("min_speed", 5.0))
            v_max = float(self.command_cfg.get("max_speed", 25.0))
            u = self._rand_scalar_scratch[: env_ids.numel()]
            u.uniform_(0.0, 1.0)
            self.commands[env_ids, 0] = v_min + (v_max - v_min) * u

    def _update_cylinders_xy(self, env_ids: torch.Tensor) -> None:
        """Refresh cached cylinder XY positions for the selected environments."""
        if self.cylinders_array is None:
            return
        if self.cylinders_xy is None or self.cylinders_xy.shape[0] != self.num_envs:
            self.cylinders_xy = self.cylinders_array[self.forest_ids, :, :2]
            return
        self.cylinders_xy[env_ids] = self.cylinders_array[self.forest_ids[env_ids], :, :2]

    def refresh_forests(self) -> None:
        """Regenerate the full forest pool (new random tree positions).

        Cylinders are pure tensor obstacles (no Genesis physics bodies), so a
        refresh is just re-running the forest generator and reapplying the
        per-env forest assignment.
        """
        if self._forest_generator is None or self.cylinders_array is None:
            return
        new_cylinders, _ = self._forest_generator.generate()
        self.cylinders_array = new_cylinders
        if self._fixed_forest_ids is not None:
            self.forest_ids[:] = self._fixed_forest_ids
        else:
            self.forest_ids.random_(0, self.cylinders_array.shape[0])
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.cylinders_xy = self.cylinders_array[self.forest_ids, :, :2]
        self._update_cylinders_xy(all_ids)

    def _nonfinite_row_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Return a per-environment mask where at least one element is non-finite.
        """
        finite = torch.isfinite(tensor)
        if tensor.dim() <= 1:
            return ~finite
        # Reduce all non-batch dimensions via flattening for broad torch compatibility.
        finite_per_row = finite.reshape(finite.shape[0], -1).all(dim=1)
        return ~finite_per_row

    def _flag_nonfinite_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mark non-finite rows in `self.nan_envs` and return the row mask."""
        bad = self._nonfinite_row_mask(tensor)
        if bad.any():
            self.nan_envs[bad] = 1
        return bad

    def _sanitize_nonfinite_rows(self, tensor: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        """
        Replace non-finite rows with a safe value while preserving shape/dtype.
        """
        bad = self._flag_nonfinite_rows(tensor)
        if bad.any():
            tensor = tensor.clone()
            tensor[bad] = fill_value
        return tensor

    def _debug_print_step_state(self) -> None:
        """Print detailed state for env-0 when debug mode is enabled."""
        if self.debug and self.num_envs == 1:
            print(f"Step: {self.episode_length_buf[0].cpu().numpy()}")
            print(f"Pos: {self.base_pos[0].cpu().numpy()}")
            print(f"Euler: {self.base_euler[0].cpu().numpy()}")
            print(f"Lin Vel: {self.base_lin_vel[0].cpu().numpy()}")
            print(f"Ang Vel: {self.base_ang_vel[0].cpu().numpy()}")
            print(f"Joint Pos: {self.joint_position[0].cpu().numpy()}")
            print(f"Joint Vel: {self.joint_velocity[0].cpu().numpy()}")
            print(f"Torque: {self.torque[0].cpu().numpy()}")

    def _compute_termination_flags(self) -> None:
        """Update termination conditions, reset mask, and timeout bookkeeping."""
        self.collision = self.check_collision(tol=self._collision_tol)
        self.success = self.check_success()

        self.wall_crash_condition = (
            (torch.abs(self.base_pos[:, 1]) > self._termination_abs_y_max)
            | (self.base_pos[:, 2] < self._termination_min_z)
        )
        if not self.evaluation:
            self.angle_limit_condition = (
                (torch.abs(self.base_euler[:, 0]) > self._roll_limit_train)
                | (torch.abs(self.base_euler[:, 1]) > self._pitch_limit_train)
                | (torch.abs(self.base_euler[:, 2]) > self._yaw_limit_train)
            )
        else:
            # During evaluation, be more lenient on angle limits to allow for aggressive maneuvers.
            self.angle_limit_condition = (
                (torch.abs(self.base_euler[:, 0]) > self._roll_limit_eval)
                | (torch.abs(self.base_euler[:, 1]) > self._pitch_limit_eval)
                | (torch.abs(self.base_euler[:, 2]) > self._yaw_limit_eval)
            )

        nan_mask = self.nan_envs.bool()
        self.reset_buf = (
            (self.episode_length_buf >= self.max_episode_length_per_env)
            | self.wall_crash_condition
            | self.angle_limit_condition
            | self.success
            | self.collision
            | nan_mask
        )

        just_reset = self.reset_buf  # envs that will be reset this step

        self.pre_wall_crash[just_reset] = self.wall_crash_condition[just_reset]
        self.pre_angle_limit[just_reset] = self.angle_limit_condition[just_reset]
        self.pre_collision[just_reset] = self.collision[just_reset]
        self.pre_success[just_reset] = self.success[just_reset]
        self.pre_nan[just_reset] = nan_mask[just_reset]

        # Time-out mask (episode ended without crash/success/collision)
        timeout = (self.episode_length_buf >= self.max_episode_length_per_env) & ~(
            self.success | self.collision | self.wall_crash_condition | self.angle_limit_condition
        )
        self._time_outs.zero_()
        self._time_outs[timeout] = 1.0
        self.extras["time_outs"] = self._time_outs

    def _accumulate_rewards(self) -> None:
        """Compute total reward and reward components for the current step."""
        self.rew_buf[:] = 0.0
        self.last_reward_components.zero_()

        if self.reward_names:
            for i, name in enumerate(self.reward_names):
                rew_comp = self.reward_functions[name]() * self.reward_scales[name] * self.dt * 50.0
                self.rew_buf += rew_comp
                self.episode_sums[name] += rew_comp
                self.last_reward_components[:, i] = rew_comp
            self.last_reward_total[:] = self.rew_buf
        self._flag_nonfinite_rows(self.rew_buf)
        self._flag_nonfinite_rows(self.last_reward_components)

    def _rebuild_observations(self) -> None:
        """
        Recompute actor/critic observations from the current simulator state.

        This is needed both after physics stepping and after resetting finished
        environments, so the returned observation always matches the internal
        state used for the next action.
        """
        depth_actor = self.depth if self.include_depth else None
        genome_vec = self._get_genome_obs_tensor()

        obs_actor, obs_critic = self.obs_builder.build_observations(
            base_pos=self.base_pos,
            base_quat=self.base_quat,
            base_lin_vel=self.base_lin_vel,
            last_actions=self.last_actions,
            commands=self.commands,
            depth_actor=depth_actor,
            joint_positions=self.joint_position,
            joint_velocities=self.joint_velocity,
            base_ang_vel=self.base_ang_vel,
            effective_thrust=self.thrust_log,
        )

        self.obs_buf.copy_(obs_actor)
        self.privileged_obs_buf.copy_(obs_critic)
        self._flag_nonfinite_rows(self.obs_buf)
        self._flag_nonfinite_rows(self.privileged_obs_buf)

    def _apply_genome_noise_(self, genome: torch.Tensor, noise_std: float) -> torch.Tensor:
        if noise_std <= 0.0 or genome.numel() == 0:
            return genome
        noise = torch.randn_like(genome) * noise_std
        noise *= self._genome_span_tensor.unsqueeze(0)
        for idx in Chromosome_Drone.NACA_GENE_INDICES:
            if idx < noise.shape[1]:
                noise[:, idx] = 0.0
        genome.add_(noise)
        genome.clamp_(min=self._genome_min_tensor, max=self._genome_max_tensor)
        return genome

    def _resample_genome_episode_noise(self, env_ids: torch.Tensor) -> None:
        if (
            self._genome_vec is None
            or self._genome_base_vec is None
            or self.evaluation
            or env_ids.numel() == 0
        ):
            return
        genome = self._genome_base_vec[env_ids].clone()
        self._apply_genome_noise_(genome, self._genome_episode_noise_std)
        self._genome_vec[env_ids] = genome

    def _get_genome_obs_tensor(self) -> Optional[torch.Tensor]:
        if self._genome_vec is None:
            return None
        if self.evaluation or self._genome_step_noise_std <= 0.0:
            return self._genome_vec
        if (
            self._genome_obs_scratch is None
            or self._genome_obs_scratch.shape != self._genome_vec.shape
            or self._genome_obs_scratch.device != self._genome_vec.device
        ):
            self._genome_obs_scratch = torch.empty_like(self._genome_vec)
        self._genome_obs_scratch.copy_(self._genome_vec)
        self._apply_genome_noise_(self._genome_obs_scratch, self._genome_step_noise_std)
        return self._genome_obs_scratch

    def _warmup_runtime_kernels(self) -> None:
        """
        Compile late-bound runtime kernels once during env construction.

        This avoids paying first-use compilation during the first rollout step
        while keeping the simulator state unchanged.
        """
        zero_throttle = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.aero_solver.set_throttle(zero_throttle)
        self.aero_solver._aero_step()
        self.rigid_solver.clear_external_force()

        cyl_xy = self.cylinders_xy
        if cyl_xy is None and self.cylinders_array is not None:
            cyl_xy = self.cylinders_array[self.forest_ids, :, :2]
        self.depth_solver.compute_depth(
            base_pos=self.base_pos,
            base_euler=self.base_euler,
            cyl_xy_b=cyl_xy,
            noise_std=0.0,
        )
        self.depth.fill_(self.MAX_DISTANCE)

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
        self.nan_envs.fill_(0)
        if actions.device != self.device:
            actions = actions.to(self.device)
        actions = self._sanitize_nonfinite_rows(actions, fill_value=0.0)
        # ActuatorDynamics handles clamping, scaling and latency
        servo_targets, throttle = self.actuator.process_actions(actions)
        servo_targets = self._sanitize_nonfinite_rows(servo_targets, fill_value=0.0)
        throttle = self._sanitize_nonfinite_rows(throttle, fill_value=0.0)
        if self.num_servos > 0:
            servo_targets = servo_targets + self._joint_target_episode_bias
            sigma_step = float(self._property_rand_cfg.get("joint_target_step_noise_std", 0.0))
            if sigma_step > 0.0:
                servo_noise = self._rand_servo_scratch
                servo_noise.normal_()
                servo_targets = servo_targets + sigma_step * servo_noise
            servo_targets = torch.max(
                torch.min(servo_targets, self.joint_limit_max.unsqueeze(0)),
                self.joint_limit_min.unsqueeze(0),
            )

        # Store applied (scaled + delayed) actions
        self.last_actions.copy_(self.actions)
        self.actions[:, 0] = throttle
        self.actions[:, 1:] = servo_targets

        # Apply to simulator
        if self.num_servos > 0:
            self.drone.control_dofs_position(servo_targets, self.servo_dof_indices)
        self.aero_solver.set_throttle(throttle)

        # Print everything about the state for debugging
        self._debug_print_step_state()

        # ------------------------- Physics -------------------------------- #
        self.scene.step()

        # NaN check (simulation instability)
        self.rigid_solver.export_winged_drone_state(
            dofs_pos=self._dofs_pos_buf,
            base_quat=self.base_quat,
            base_lin_vel=self.base_lin_vel,
            base_ang_vel=self.base_ang_vel,
            joint_vel=self.joint_velocity,
            control_force=self.torque,
            servo_dofs_idx=self._servo_dof_idx_tensor,
            base_link_idx=self.drone.base_link_idx,
        )
        dofs_pos = self._dofs_pos_buf
        if not torch.isfinite(dofs_pos).all():
            nan_idx = torch.isnan(dofs_pos).any(dim=1).nonzero(as_tuple=False).flatten()
            if nan_idx.numel() > 0:
                print(f"[WingedDroneEnv] NaN detected at sim time {float(self.scene.t):.3f}, envs {nan_idx.tolist()}")
                self.nan_envs[nan_idx] = 1

        # Increase episode step counters
        self.episode_length_buf += 1

        # ------------------------- State update ---------------------------- #
        # Base pose
        self.base_pos[:] = dofs_pos[:, :3]
        self.base_euler[:] = quat_to_xyz(self.base_quat, rpy=True, degrees=False)
        self._flag_nonfinite_rows(self.base_quat)
        self._flag_nonfinite_rows(self.base_euler)

        # Joint state
        if self.num_servos > 0:
            self.joint_position[:] = dofs_pos[:, self.servo_dof_indices]
            self._flag_nonfinite_rows(self.joint_position)
            self._flag_nonfinite_rows(self.joint_velocity)
            self._flag_nonfinite_rows(self.torque)

        # Velocities (world/body)
        self._base_quat_inv_buf.copy_(self.base_quat)
        self._base_quat_inv_buf[:, 1:].neg_()
        gu._tc_transform_by_quat(self.base_ang_vel, self._base_quat_inv_buf, out=self._base_ang_vel_body_buf)
        self.base_ang_vel.copy_(self._base_ang_vel_body_buf)
        self._flag_nonfinite_rows(self.base_lin_vel)
        self._flag_nonfinite_rows(self.base_ang_vel)

        if self.evaluation:
            a0 = getattr(self.aero_solver, "_alpha_dbg0_buf", None)
            b0 = getattr(self.aero_solver, "_beta_dbg0_buf", None)
            if torch.is_tensor(a0) and torch.is_tensor(b0) and a0.numel() > 0 and b0.numel() > 0:
                self.alpha = a0[0]
                self.beta = b0[0]
            else:
                self.alpha = self.aero_solver.alpha_dbg.to_torch(device=self.device)[0, 0]
                self.beta = self.aero_solver.beta_dbg.to_torch(device=self.device)[0, 0]

        # ------------------------- Terminations ---------------------------- #
        self._compute_termination_flags()

        # ------------------------- Depth sensing --------------------------- #
        cyl_xy = self.cylinders_xy
        if cyl_xy is None and self.cylinders_array is not None:
            # cylinders_array: (F, T, 3) → (B, T, 2) via forest_ids
            cyl_xy = self.cylinders_array[self.forest_ids, :, :2]

        self.depth = self.depth_solver.compute_depth(
            base_pos=self.base_pos,
            base_euler=self.base_euler,
            cyl_xy_b=cyl_xy,
            noise_std=float(self.obs_cfg.get("depth_noise_std", 0.0)),
        )
        if self.depth is not None:
            self._flag_nonfinite_rows(self.depth)
        self.power = self.power_consumption()

        # ------------------------- Rewards -------------------------------- #
        self._accumulate_rewards()

        # ------------------------- Observations ---------------------------- #
        self._rebuild_observations()

        # If any env produced NaNs, force safe outputs and trigger reset.
        nan_mask = self.nan_envs.bool()
        if nan_mask.any():
            self.rew_buf[nan_mask] = 0.0
            self.last_reward_components[nan_mask] = 0.0
            self.last_reward_total[nan_mask] = 0.0
            self.obs_buf[nan_mask] = 0.0
            self.privileged_obs_buf[nan_mask] = 0.0
            self.reset_buf |= nan_mask
            self.pre_nan[nan_mask] = True

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

        if self.auto_reset:
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self.reset_idx(reset_env_ids)
            if reset_env_ids.numel() > 0:
                # Match the next observation to the freshly reset state so PPO
                # stores coherent transitions across episode boundaries.
                self.depth[reset_env_ids] = self.MAX_DISTANCE
                self._rebuild_observations()
                self.extras["observations"]["critic"] = self.privileged_obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ---------------------------------------------------------------------- #
    # Reset randomization helper                                             #
    # ---------------------------------------------------------------------- #
    def _randomize_reset_state(self, env_ids: torch.Tensor, n: int) -> None:
        """Apply training-time randomization to reset pose/velocity/joints."""
        r = self._rand_scalar_scratch[:n]

        # Longitudinal position
        r.uniform_(0.0, 1.0)
        self.base_pos[env_ids, 0] += r * 30.0 - 15.0
        # Lateral position
        r.uniform_(0.0, 1.0)
        self.base_pos[env_ids, 1] += r * 80.0 - 40.0
        # Altitude
        r.uniform_(0.0, 1.0)
        self.base_pos[env_ids, 2] += r * 15.0 - 10.0

        # Forward speed
        r.uniform_(0.0, 1.0)
        self.base_lin_vel[env_ids, 0] = r * 20.0 + 5.0
        # Lateral speed
        r.normal_()
        self.base_lin_vel[env_ids, 1] = torch.clamp(r * 1.0, min=-8.0, max=8.0)
        # Vertical speed
        r.normal_()
        self.base_lin_vel[env_ids, 2] = torch.clamp(r * 1.0, min=-8.0, max=8.0)

        self.base_euler[env_ids, 1] = torch.atan2(-self.base_lin_vel[env_ids, 2], self.base_lin_vel[env_ids, 0])
        self.base_euler[env_ids, 2] = torch.atan2(self.base_lin_vel[env_ids, 1], self.base_lin_vel[env_ids, 0])

        # Small attitude perturbations (if randomize_init_quat is enabled)
        if self.env_cfg.get("randomize_init_quat", True):
            r.normal_()
            self.base_euler[env_ids, 0] += torch.clamp(r * 0.2, min=-0.8, max=0.8)
            r.normal_()
            self.base_euler[env_ids, 1] += torch.clamp(r * 0.05, min=-0.2, max=0.2)
            r.normal_()
            self.base_euler[env_ids, 2] += torch.clamp(r * 0.05, min=-0.2, max=0.2)

        # Joint positions noise
        if self.num_servos > 0 and self.randomize_joint_pos:
            rs = self._rand_servo_scratch[:n]
            rs.normal_()
            self.joint_position[env_ids] += torch.clamp(rs * 0.0, min=-0.04, max=0.04)

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
        if self._fixed_forest_ids is not None:
            self.forest_ids[env_ids] = self._fixed_forest_ids[env_ids]
        else:
            new_ids = self._randint_scratch[: env_ids.numel()]
            new_ids.random_(0, self.cylinders_array.shape[0])
            self.forest_ids[env_ids] = new_ids
        self._update_cylinders_xy(env_ids)

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
            base_init_pos = self.env_cfg.get("base_init_pos", [0.0, 0.0, 1.0])
            self.base_init_pos[0] = float(base_init_pos[0])
            self.base_init_pos[1] = float(base_init_pos[1])
            self.base_init_pos[2] = float(base_init_pos[2])

            base_init_quat = self.env_cfg.get("base_init_quat", [1.0, 0.0, 0.0, 0.0])
            self.base_init_quat[0] = float(base_init_quat[0])
            self.base_init_quat[1] = float(base_init_quat[1])
            self.base_init_quat[2] = float(base_init_quat[2])
            self.base_init_quat[3] = float(base_init_quat[3])

        self.base_pos[env_ids] = self.base_init_pos
        self.base_quat[env_ids] = self.base_init_quat.reshape(1, -1)

        # Euler angles and velocities
        base_euler_init = quat_to_xyz(self.base_init_quat, rpy=True, degrees=False)
        self.base_euler[env_ids] = base_euler_init.reshape(1, -1).expand(n, -1)
        self.joint_position[env_ids] = 0.0
        self.joint_velocity[env_ids] = 0.0
        self.torque[env_ids] = 0.0

        self.base_lin_vel[env_ids] = self._reset_lin_vel.unsqueeze(0).expand(n, -1)
        self.base_ang_vel[env_ids] = self._reset_ang_vel.unsqueeze(0).expand(n, -1)

        # Training: inject randomness in initial pose and speed
        if not self.evaluation:
            self._randomize_reset_state(env_ids, n)

        # Apply quaternion back from Euler
        self.base_quat[env_ids] = xyz_to_quat(self.base_euler[env_ids], degrees=False)

        # Apply to rigid solver (same values as previous cat-based path).
        initial_pos = self._reset_pos_scratch[:n]
        initial_vel = self._reset_vel_scratch[:n]
        initial_pos[:, :3] = self.base_pos[env_ids]
        initial_pos[:, 3:6] = self.base_euler[env_ids]
        initial_pos[:, 6:] = self.joint_position[env_ids]
        initial_vel[:, :3] = self.base_lin_vel[env_ids]
        initial_vel[:, 3:6] = self.base_ang_vel[env_ids]
        initial_vel[:, 6:] = self.joint_velocity[env_ids]

        self.rigid_solver.set_dofs_position(initial_pos, envs_idx=env_ids)
        self.rigid_solver.set_dofs_velocity(initial_vel, envs_idx=env_ids)

        self._apply_dynamics_noise(env_ids)
        if self.num_servos > 0:
            episode_bias = self._joint_target_episode_bias[env_ids]
            episode_bias.zero_()
            sigma_episode = float(self._property_rand_cfg.get("joint_target_episode_bias_std", 0.0))
            if sigma_episode > 0.0:
                rs = self._rand_servo_scratch[:n]
                rs.normal_()
                episode_bias.add_(sigma_episode * rs)
        self._resample_genome_episode_noise(env_ids)

        # Optional aero parameter randomization
        if hasattr(self.aero_solver, "_enable_noise"):
            if hasattr(self.aero_solver, "randomize_aero_params"):
                self.aero_solver.randomize_aero_params(env_ids)
        self._refresh_max_thrust_cache(env_ids)

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
        genome_vec = self._get_genome_obs_tensor()

        depth_actor = self.depth if self.include_depth else None

        obs_actor, obs_critic = self.obs_builder.build_observations(
            base_pos=self.base_pos,
            base_quat=self.base_quat,
            base_lin_vel=self.base_lin_vel,
            last_actions=self.last_actions,
            commands=self.commands,
            depth_actor=depth_actor,
            joint_positions=self.joint_position,
            joint_velocities=self.joint_velocity,
            base_ang_vel=self.base_ang_vel,
            effective_thrust=self.thrust_log,
        )

        self.obs_buf.copy_(obs_actor)
        self.privileged_obs_buf.copy_(obs_critic)

        self.extras.setdefault("observations", {})
        self.extras["observations"]["critic"] = self.privileged_obs_buf
        if "episode" in self.extras:
            del self.extras["episode"]

        return self.obs_buf, self.extras

    # ---------------------------------------------------------------------- #
    # Power consumption                                                      #
    # ---------------------------------------------------------------------- #
    def power_consumption(self) -> torch.Tensor:
        """
        Approximate total power consumption for current state.

        - Propeller power: RPM/advance-ratio model when solver states are available.
        - Servo power: torque * angular velocity.
        """
        self.thrust_log[:, 0].copy_(self.extract_thrust())  # (B, 1)
        self.prop_rpm_log[:, 0].copy_(self.extract_prop_rpm())
        self.prop_axial_speed_log[:, 0].copy_(self.extract_prop_axial_speed())

        prop_coeffs = self._power_prop_coeffs
        if prop_coeffs is None or prop_coeffs.shape[0] != self.thrust_log.shape[1]:
            prop_coeffs = power_utils._default_prop_coeffs(
                self.drone_name,
                self.device,
                self.thrust_log.shape[1],
                aero_config=self._aero_config,
            )
            self._power_prop_coeffs = prop_coeffs
        servo_constants = self._power_servo_constants
        if servo_constants is None or servo_constants.shape[0] != self.num_servos:
            servo_constants = power_utils._default_servo_power_constants(
                self.drone_name, self.device, self.num_servos
            )
            self._power_servo_constants = servo_constants
        torque_multipliers = self._power_torque_multipliers
        if torque_multipliers is None or torque_multipliers.shape[0] != self.num_servos:
            torque_multipliers = power_utils._default_torque_multipliers(
                self.drone_name,
                self.device,
                self.num_servos,
                sweep_multiplier=2.0,
                twist_multiplier=2.5,
                tail_multiplier=2.0,
            )
            self._power_torque_multipliers = torque_multipliers

        total_power = power_utils.compute_power_consumption(
            thrust=self.thrust_log,
            servo_torque=self.torque,
            servo_velocity=self.joint_velocity,
            drone_name=self.drone_name,
            aero_config=self._aero_config,
            prop_coefficients=prop_coeffs,
            prop_rpm=self.prop_rpm_log,
            prop_axial_speed=self.prop_axial_speed_log,
            prop_diameters=self._prop_diameter,
            servo_power_constants=servo_constants,
            torque_multipliers=torque_multipliers,
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

        # Fast path: reuse solver-side cached thrust buffer (updated every aero step).
        cached_thrust = getattr(self.aero_solver, "_thrust_n_buf", None)
        if torch.is_tensor(cached_thrust) and cached_thrust.shape[0] == self.num_envs:
            self._thrust_buf.copy_(cached_thrust)
        else:
            # Fallback for solver variants that do not expose cached thrust.
            self._thr_flt_buf.copy_(self.aero_solver._thr_flt.to_torch(device=self.device))
            self._max_thr_buf.copy_(self.aero_solver.max_thrust.to_torch(device=self.device))
            self._thrust_buf.copy_(self._thr_flt_buf)
            self._thrust_buf.mul_(self._max_thr_buf)

        # Optional: clamp NaNs or negatives
        torch.nan_to_num_(self._thrust_buf, nan=0.0, posinf=0.0, neginf=0.0)
        self._thrust_buf.clamp_(min=0.0)
        return self._thrust_buf

    def extract_max_thrust(self) -> torch.Tensor:
        """Extract the per-env maximum propeller thrust used for normalization."""
        if not torch.isfinite(self._max_thr_buf).all():
            self._refresh_max_thrust_cache()
        torch.nan_to_num_(self._max_thr_buf, nan=0.0, posinf=0.0, neginf=0.0)
        self._max_thr_buf.clamp_(min=1e-6)
        return self._max_thr_buf

    def _refresh_max_thrust_cache(self, env_ids: Optional[torch.Tensor] = None) -> None:
        cached_max = getattr(self.aero_solver, "_max_thrust_buf", None)
        if torch.is_tensor(cached_max) and cached_max.shape[0] == self.num_envs:
            if env_ids is None:
                self._max_thr_buf.copy_(cached_max)
            elif env_ids.numel() > 0:
                self._max_thr_buf[env_ids] = cached_max[env_ids]
        else:
            max_thrust = self.aero_solver.max_thrust.to_torch(device=self.device)
            if env_ids is None:
                self._max_thr_buf.copy_(max_thrust)
            elif env_ids.numel() > 0:
                self._max_thr_buf[env_ids] = max_thrust[env_ids]
        torch.nan_to_num_(self._max_thr_buf, nan=0.0, posinf=0.0, neginf=0.0)
        self._max_thr_buf.clamp_(min=1e-6)

    def extract_prop_rpm(self) -> torch.Tensor:
        """Extract the current propeller RPM from the AeroSolver cache."""
        cached_rpm = getattr(self.aero_solver, "_prop_rpm_buf", None)
        if torch.is_tensor(cached_rpm) and cached_rpm.shape[0] == self.num_envs:
            self._prop_rpm_buf.copy_(cached_rpm)
        else:
            self._prop_rpm_buf.zero_()
        torch.nan_to_num_(self._prop_rpm_buf, nan=0.0, posinf=0.0, neginf=0.0)
        self._prop_rpm_buf.clamp_(min=0.0)
        return self._prop_rpm_buf

    def extract_prop_axial_speed(self) -> torch.Tensor:
        """Extract the positive axial inflow speed seen by the propeller."""
        cached_speed = getattr(self.aero_solver, "_prop_axial_speed_buf", None)
        if torch.is_tensor(cached_speed) and cached_speed.shape[0] == self.num_envs:
            self._prop_axial_speed_buf.copy_(cached_speed)
        else:
            self._prop_axial_speed_buf.zero_()
        torch.nan_to_num_(self._prop_axial_speed_buf, nan=0.0, posinf=0.0, neginf=0.0)
        self._prop_axial_speed_buf.clamp_(min=0.0)
        return self._prop_axial_speed_buf

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
        r_tree = self._tree_radius + tol

        cyl_xy = self.cylinders_xy
        if cyl_xy is None:
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
        forest_x_limit = (
            self._success_x_limit_eval if self.evaluation else self._success_x_limit_train
        )
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
        """Penalize large roll/pitch angles (quadratically)."""
        return self.base_euler[:, 0]**2 + self.base_euler[:, 1]**2

    def _reward_crash(self) -> torch.Tensor:
        """Penalty for crash or collision (1 on crash/collision)."""
        crash = torch.zeros((self.num_envs,), device=self.device)
        crash[self.wall_crash_condition | self.collision | self.angle_limit_condition] = 1.0
        # Rescale so that a single crash produces ~O(1) penalty per episode
        return crash / (self.dt * 100.0)

    def _reward_energy(self) -> torch.Tensor:
        """Penalize energy consumption (higher power -> lower reward)."""
        energy = self.power
        return energy

    def _reward_progress(self, sigma: float = 0.25) -> torch.Tensor:
        """
        Reward for forward speed tracking.

        Target:
            v_proj (along +X) ≈ v_target (commands[:,0]).
        """
        v_xy = self.base_lin_vel[:, :2]
        v_proj = v_xy[:, 0]

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
        tree_radius = self._tree_radius
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
