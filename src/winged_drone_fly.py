import os
import time
import threading
import signal
import argparse
import importlib
import importlib.util
from dataclasses import dataclass
from typing import List, Optional, Sequence
from pathlib import Path
import csv
import xml.etree.ElementTree as ET
from datetime import datetime

from pynput import keyboard
import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import (
    euler_to_quat,
    transform_by_quat,
    quat_to_xyz,
)
from genesis.assets.urdf.aero_model import DroneAeroModel, SurfaceKind
from winged_drone_train.aero_profile import (
    configure_runtime_aero_solver,
    resolve_aero_config,
)
import sys

# -------- Redirect ONLY print() output to file --------
log_file = open("winged_drone_output.txt", "w", buffering=1)
sys.stdout = log_file
CONSOLE = sys.__stderr__


# Batch size: keep 1 for now, but code is structured to extend to B > 1.
BATCH_SIZE = 1
AERO_SOLVER_KIND = os.environ.get("AERO_SOLVER_KIND", "simple").strip().lower()
KILL_ALTITUDE_Z = 2.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_FILE = PROJECT_ROOT / "src/winged_drone_prescribed_trajectory.py"

# Select which drone to fly.
DRONE_NAME = "lisparrow"  # "mydrone" or "lisparrow"

MYDRONE_URDF = PROJECT_ROOT / "genesis/assets/urdf/mydrone/[0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4, 0.2, 2, 0, 2, 2.5, 3, 4, 16].urdf"
LISPARROW_URDF = PROJECT_ROOT / "genesis/assets/urdf/lisparrow/lisparrow.urdf"

DRONE_CONFIGS = {
    "mydrone": {
        "urdf_path": MYDRONE_URDF,
        "naca": "3416",
        "servo_joint_names": None,
        "servo_role_names": None,
        "debug_links": ["fuselage", "left_wing", "right_wing", "elevator_hinge", "rudder"],
        "fallback_servo_gains": None,
        "aero_solver_kind": None,
        "sweep_command_signs": (1.0, -1.0),  # (left, right)
        "tail_command_signs": (1.0, 1.0),  # (elevator, rudder)
    },
    "lisparrow": {
        "urdf_path": LISPARROW_URDF,
        "naca": None,
        "servo_joint_names": [
            "joint_left_outer_wing_hinged",
            "joint_right_outer_wing_hinged",
            "joint_elevator_hinged",
            "joint_rudder_hinged",
        ],
        "servo_role_names": {
            "sweep_left": "joint_left_outer_wing_hinged",
            "sweep_right": "joint_right_outer_wing_hinged",
            "elevator": "joint_elevator_hinged",
            "rudder": "joint_rudder_hinged",
        },
        "debug_links": [
            "fuselage",
            "root_wing_fixed",
            "left_outer_wing_hinged",
            "right_outer_wing_hinged",
            "elevator_hinged",
            "rudder_hinged",
            "propeller_fixed",
        ],
        "fallback_servo_gains": (20.0, 2.0),
        "aero_solver_kind": "lisparrow",
        # The Lisparrow wing joints have mirrored axes in the URDF
        # (left: -Z, right: +Z), so equal target signs produce symmetric sweep.
        "sweep_command_signs": (1.0, 1.0),  # (left, right)
        # Lisparrow joints are mounted with opposite sign vs keyboard semantics.
        # q=up elevator, a=left rudder.
        "tail_command_signs": (-1.0, -1.0),  # (elevator, rudder)
    },
}


def _resolve_aero_config(solver_kind: str) -> dict:
    return resolve_aero_config(solver_kind)


def _configure_aero_solver(solver_kind: str) -> None:
    configure_runtime_aero_solver(solver_kind)


def _load_joint_position_limits_from_urdf(urdf_path: str, joint_names: List[str]) -> np.ndarray:
    """
    Return per-joint lower/upper position limits from the URDF (in radians).

    No defaults: every requested joint must exist and define a <limit lower= upper=>.
    """
    root = ET.parse(str(urdf_path)).getroot()
    out = []
    for name in joint_names:
        joint = root.find(f".//joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"URDF is missing joint '{name}'.")
        lim = joint.find("limit")
        if lim is None:
            raise ValueError(f"URDF joint '{name}' is missing a <limit> tag.")
        lower = lim.get("lower")
        upper = lim.get("upper")
        if lower is None or upper is None:
            raise ValueError(f"URDF joint '{name}' must define both 'lower' and 'upper' in <limit>.")
        lo = float(lower)
        hi = float(upper)
        out.append((lo, hi))
    return np.asarray(out, dtype=np.float32)


@dataclass
class ServoLayout:
    joint_names: List[str]
    role_index: dict[str, int]


def _resolve_drone_config(name: str) -> dict:
    key = (name or "").strip().lower()
    if key not in DRONE_CONFIGS:
        raise ValueError(f"Unknown drone '{name}'. Options: {sorted(DRONE_CONFIGS.keys())}")
    return dict(DRONE_CONFIGS[key])


def _classify_joint_role(name: str) -> str | None:
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


def _resolve_servo_layout(urdf_path: str, preferred_names: Sequence[str] | None = None) -> ServoLayout:
    root = ET.parse(str(urdf_path)).getroot()
    joint_nodes = root.findall(".//joint")
    joint_names = [j.get("name") for j in joint_nodes if j.get("name")]
    joint_set = set(joint_names)

    if preferred_names is not None:
        missing = [name for name in preferred_names if name not in joint_set]
        if missing:
            raise ValueError(f"Missing servo joints in URDF: {missing}")
        names = list(preferred_names)
        role_index: dict[str, int] = {}
        for idx, name in enumerate(names):
            role = _classify_joint_role(name)
            if role and role not in role_index:
                role_index[role] = idx
        return ServoLayout(joint_names=names, role_index=role_index)

    role_to_name: dict[str, str] = {}
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
    names = []
    role_index = {}
    for role in ordered_roles:
        name = role_to_name.get(role)
        if name:
            role_index[role] = len(names)
            names.append(name)

    return ServoLayout(joint_names=names, role_index=role_index)


def _servo_layout_from_role_names(
    urdf_path: str,
    role_names: dict[str, str] | None,
    preferred_names: Sequence[str] | None = None,
) -> ServoLayout | None:
    if not role_names:
        return None

    root = ET.parse(str(urdf_path)).getroot()
    joint_set = {
        j.get("name")
        for j in root.findall(".//joint")
        if j.get("name")
    }

    names = list(preferred_names) if preferred_names is not None else []
    for role in ("sweep_left", "sweep_right", "twist_left", "twist_right", "elevator", "rudder", "sweep", "twist"):
        name = role_names.get(role)
        if not name:
            continue
        if name not in joint_set:
            raise ValueError(f"Missing servo joint '{name}' for role '{role}' in URDF.")
        if name not in names:
            names.append(name)

    role_index: dict[str, int] = {}
    for role, name in role_names.items():
        if name not in names:
            continue
        role_index[role] = names.index(name)

    return ServoLayout(joint_names=names, role_index=role_index)


def _print_controls(layout: ServoLayout) -> None:
    print("\nWinged Drone Controls:")
    print("↑ / ↓   - Increase / decrease thrust (via AeroSolver)")
    if (
        "twist_left" in layout.role_index
        or "twist_right" in layout.role_index
        or "twist" in layout.role_index
    ):
        print("← / →   - Asymmetric twist (roll)")
        print("space   - Increase symmetric twist")
        print("shift   - Decrease symmetric twist")
    if (
        "sweep_left" in layout.role_index
        or "sweep_right" in layout.role_index
        or "sweep" in layout.role_index
    ):
        print("w / s   - Increase / decrease symmetric sweep")
    if "elevator" in layout.role_index:
        print("q / e   - Elevator up / down (pitch)")
    if "rudder" in layout.role_index:
        print("a / d   - Rudder left / right (yaw)")
    print("ESC     - Quit\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fly/debug a winged drone with optional prescribed actuator trajectories.")
    parser.add_argument(
        "--trajectory",
        nargs="?",
        const=str(DEFAULT_TRAJECTORY_FILE),
        default=None,
        help=(
            "Use a prescribed trajectory module. With no value, uses "
            f"{DEFAULT_TRAJECTORY_FILE}. A filesystem path or importable module name is accepted."
        ),
    )
    return parser.parse_args()


class PrescribedTrajectory:
    def __init__(self, module, source: str):
        self.module = module
        self.source = source
        if not hasattr(module, "command"):
            raise ValueError(f"Trajectory source '{source}' must define command(t).")
        self.initial_conditions = dict(getattr(module, "INITIAL_CONDITIONS", {}) or {})

    def command(self, t: float) -> dict:
        out = self.module.command(float(t))
        if out is None:
            return {}
        if not isinstance(out, dict):
            raise TypeError(f"Trajectory command(t) from '{self.source}' must return a dict.")
        return out


def _load_prescribed_trajectory(spec: str | None) -> PrescribedTrajectory | None:
    if not spec:
        return None

    candidate = Path(spec).expanduser()
    if not candidate.is_absolute():
        cwd_path = Path.cwd() / candidate
        project_path = PROJECT_ROOT / candidate
        src_path = PROJECT_ROOT / "src" / candidate
        if cwd_path.exists():
            candidate = cwd_path
        elif project_path.exists():
            candidate = project_path
        elif src_path.exists():
            candidate = src_path

    if candidate.exists():
        module_name = f"_winged_drone_trajectory_{abs(hash(candidate.resolve()))}"
        spec_obj = importlib.util.spec_from_file_location(module_name, candidate)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Could not load trajectory file: {candidate}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
        return PrescribedTrajectory(module, str(candidate))

    module = importlib.import_module(spec)
    return PrescribedTrajectory(module, spec)


def _as_vec3(value, default, name: str) -> np.ndarray:
    arr = np.asarray(value if value is not None else default, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"Trajectory initial condition '{name}' must have 3 values, got {arr}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Trajectory initial condition '{name}' contains non-finite values: {arr}.")
    return arr


def _first_local_dof_idx(joint_obj) -> int | None:
    idx = getattr(joint_obj, "dofs_idx_local", None)
    if idx is None:
        idx = getattr(joint_obj, "dof_idx_local", None)
    if idx is None:
        return None
    if isinstance(idx, (list, tuple, np.ndarray)):
        if len(idx) == 0:
            return None
        return int(idx[0])
    return int(idx)


def _joint_first_dof(drone, joint_name: str | None) -> int | None:
    if not joint_name:
        return None
    try:
        j = drone.get_joint(joint_name)
    except Exception:
        return None
    return _first_local_dof_idx(j)


def _extract_solver_dof_vector(aero_solver, attr_name: str) -> list[int] | None:
    vec = getattr(aero_solver, attr_name, None)
    if vec is None:
        return None
    if isinstance(vec, (list, tuple)):
        return [int(v) for v in vec]
    if isinstance(vec, np.ndarray):
        return [int(v) for v in vec.reshape(-1)]
    try:
        if hasattr(vec, "to_numpy"):
            arr = vec.to_numpy()
            return [int(v) for v in np.asarray(arr).reshape(-1)]
    except Exception:
        return None
    return None


def _log_aero_surface_joint_mapping(drone, drone_model, aero_solver) -> None:
    frames = list(getattr(aero_solver, "_aero_frames", []) or [])
    if not frames and drone_model is not None:
        frames = list(getattr(drone_model, "frames", []) or [])
    if not frames:
        print("[AERO MAP] no aerodynamic surfaces found.")
        return

    act_map = dict(getattr(drone_model, "actuators", {}) or {}) if drone_model is not None else {}
    surf_dof = _extract_solver_dof_vector(aero_solver, "_surf_dof")
    surf_dof_yaw = _extract_solver_dof_vector(aero_solver, "_surf_dof_yaw")
    surf_dof_pitch = _extract_solver_dof_vector(aero_solver, "_surf_dof_pitch")

    print("\n--- Aero surface -> joint/DOF mapping ---")
    print("frame | model_joint:model_dof | solver_primary/yaw/pitch")
    for i, frame in enumerate(frames):
        info = act_map.get(frame)
        model_joint = getattr(info, "joint_name", None) if info is not None else None
        model_dof = _joint_first_dof(drone, model_joint)

        s_main = surf_dof[i] if surf_dof is not None and i < len(surf_dof) else None
        s_yaw = surf_dof_yaw[i] if surf_dof_yaw is not None and i < len(surf_dof_yaw) else None
        s_pitch = surf_dof_pitch[i] if surf_dof_pitch is not None and i < len(surf_dof_pitch) else None

        model_joint_str = str(model_joint) if model_joint else "-"
        model_dof_str = str(model_dof) if model_dof is not None else "-"
        s_main_str = str(s_main) if s_main is not None else "-"
        s_yaw_str = str(s_yaw) if s_yaw is not None else "-"
        s_pitch_str = str(s_pitch) if s_pitch is not None else "-"

        print(
            f"{frame:30s} | {model_joint_str:24s}:{model_dof_str:>3s} | "
            f"{s_main_str:>3s}/{s_yaw_str:>3s}/{s_pitch_str:>3s}"
        )
    print("--- end aero mapping ---\n")


class DroneController:
    """
    High-level keyboard controller for the winged drone.

    - Keeps throttle (0..1) and aerodynamic control surface joints.
    - Exposes:
        * update_thrust(dt): integrate keys → throttle
        * apply_thrust(aero_solver, dt): update_thrust + set_throttle
        * apply_joint_commands(dt): integrate keys → servo DOF targets
    """

    def __init__(
        self,
        layout: ServoLayout,
        sweep_command_signs: tuple[float, float] = (1.0, -1.0),
        tail_command_signs: tuple[float, float] = (1.0, 1.0),
    ):
        # Initial spawn state for the drone
        self.init_pos = np.array([0.0, 0.0, 20.0], dtype=np.float32)
        self.init_vel = np.array([6.0, 0.0, 1.0], dtype=np.float32)
        self.init_euler = np.array([0.0, -15.0, 0.0], dtype=np.float32)
        self.init_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.init_joint_velocity = np.zeros(0, dtype=np.float32)
        self.init_joint_position = np.zeros(0, dtype=np.float32)

        # Runtime
        self.running: bool = True
        self.pressed_keys: set = set()
        self.trajectory: Optional[PrescribedTrajectory] = None
        self.trajectory_time: float = 0.0

        # Throttle state (broadcast inside AeroSolver to all envs)
        self.throttle: float = 0.0
        self._throttle_min: float = 0.0
        self._throttle_max: float = 1.0
        self._throttle_rate: float = 0.2  # change per second

        # Servo joints (configurable)
        self.servo_joint_names: List[str] = list(layout.joint_names)
        self._role_index: dict[str, int] = dict(layout.role_index)
        self._joint_index_by_name: dict[str, int] = {
            str(name): i for i, name in enumerate(self.servo_joint_names)
        }

        # Filled after build()
        self.drone: Optional[gs.engine.entities.RigidEntity] = None  # type: ignore
        self.servo_dof_indices: Optional[np.ndarray] = None

        # Servo command (target joint angles in radians)
        self.servo_cmd = np.zeros(len(self.servo_joint_names), dtype=np.float32)

        # Speeds (rad/s) and limits (rad)
        self._sweep_rate = 0.2
        self._twist_rate_sym = 0.2
        self._twist_rate_asym = 0.02
        self._tail_rate = 0.2
        self._sweep_left_sign = float(sweep_command_signs[0])
        self._sweep_right_sign = float(sweep_command_signs[1])
        self._elevator_sign = float(tail_command_signs[0])
        self._rudder_sign = float(tail_command_signs[1])

        # Filled from URDF in main() (no defaults).
        self._servo_lower_limits: Optional[np.ndarray] = None
        self._servo_upper_limits: Optional[np.ndarray] = None

        # Key aliases
        self._key_w = keyboard.KeyCode.from_char("w")
        self._key_s = keyboard.KeyCode.from_char("s")
        self._key_a = keyboard.KeyCode.from_char("a")
        self._key_d = keyboard.KeyCode.from_char("d")
        self._key_q = keyboard.KeyCode.from_char("q")
        self._key_e = keyboard.KeyCode.from_char("e")
        self._key_space = keyboard.Key.space
        self._shift_keys = {
            keyboard.Key.shift,
            keyboard.Key.shift_l,
            keyboard.Key.shift_r,
        }

    def set_prescribed_trajectory(self, trajectory: PrescribedTrajectory | None) -> None:
        self.trajectory = trajectory
        self.trajectory_time = 0.0
        if trajectory is not None:
            self.apply_trajectory_initial_conditions(trajectory.initial_conditions)

    def apply_trajectory_initial_conditions(self, initial: dict) -> None:
        if not initial:
            return
        self.init_pos = _as_vec3(initial.get("pos"), self.init_pos, "pos")
        self.init_vel = _as_vec3(initial.get("vel"), self.init_vel, "vel")
        self.init_euler = _as_vec3(initial.get("euler_deg", initial.get("euler")), self.init_euler, "euler_deg")
        self.init_ang_vel = _as_vec3(initial.get("ang_vel"), self.init_ang_vel, "ang_vel")
        if "throttle" in initial:
            self.throttle = float(np.clip(float(initial["throttle"]), self._throttle_min, self._throttle_max))

        controls = dict(initial.get("controls", {}) or {})
        for key in (
            "joint_targets",
            "joints",
            "servo_targets",
            "sweep_symmetric",
            "sweep_asymmetric",
            "elevator",
            "rudder",
        ):
            if key in initial and key not in controls:
                controls[key] = initial[key]
        if controls:
            self._apply_control_dict(controls)

    def _apply_control_dict(self, controls: dict) -> None:
        if self.servo_cmd.size:
            target = self.servo_cmd.copy()
            direct_targets = self._direct_joint_targets_from_controls(controls)
            for joint_name, value in direct_targets.items():
                target[self._joint_index_by_name[joint_name]] = float(value)

            if not direct_targets:
                target = self._apply_semantic_control_targets(target, controls)

            self.servo_cmd = target

        if "throttle" in controls:
            self.throttle = float(np.clip(float(controls["throttle"]), self._throttle_min, self._throttle_max))

    def _direct_joint_targets_from_controls(self, controls: dict) -> dict[str, float]:
        direct: dict[str, float] = {}
        for key in ("joint_targets", "joints", "servo_targets"):
            mapping = controls.get(key)
            if mapping is None:
                continue
            if not isinstance(mapping, dict):
                raise TypeError(f"Trajectory '{key}' must be a dict of joint_name -> target_rad.")
            for joint_name, value in mapping.items():
                joint_name = str(joint_name)
                if joint_name not in self._joint_index_by_name:
                    raise KeyError(
                        f"Trajectory commands unknown joint '{joint_name}'. "
                        f"Known servo joints: {self.servo_joint_names}"
                    )
                direct[joint_name] = float(value)

        for joint_name, value in controls.items():
            joint_name = str(joint_name)
            if joint_name in self._joint_index_by_name:
                direct[joint_name] = float(value)
        return direct

    def _apply_semantic_control_targets(self, target: np.ndarray, controls: dict) -> np.ndarray:
        if self.servo_cmd.size:
            sw_l = self._role_index.get("sweep_left")
            sw_r = self._role_index.get("sweep_right")
            sw = self._role_index.get("sweep")
            ele = self._role_index.get("elevator")
            rud = self._role_index.get("rudder")

            sym = float(controls.get("sweep_symmetric", 0.0))
            asym = float(controls.get("sweep_asymmetric", 0.0))
            if sw_l is not None:
                target[sw_l] = self._sweep_left_sign * (sym + asym)
            if sw_r is not None:
                target[sw_r] = self._sweep_right_sign * (sym - asym)
            if sw is not None and sw_l is None and sw_r is None:
                target[sw] = sym
            if ele is not None and "elevator" in controls:
                target[ele] = self._elevator_sign * float(controls["elevator"])
            if rud is not None and "rudder" in controls:
                target[rud] = self._rudder_sign * float(controls["rudder"])
        return target

    def update_prescribed_trajectory(self, dt: float) -> None:
        if self.trajectory is None:
            return
        if dt > 0.0:
            self.trajectory_time += float(dt)
        self._apply_control_dict(self.trajectory.command(self.trajectory_time))

    # ------------------------ Attach drone / indices ------------------------

    def attach_drone(self, drone, servo_dof_indices: List[int]):
        """Store drone handle and DOF indices (local)."""
        self.drone = drone
        self.servo_dof_indices = np.array(servo_dof_indices, dtype=np.int32)

    def init_joint_state(self, n_dofs: int) -> None:
        if n_dofs < 0:
            raise ValueError(f"Invalid DOF count: {n_dofs}")
        self.init_joint_velocity = np.zeros(n_dofs, dtype=np.float32)
        self.init_joint_position = np.zeros(n_dofs, dtype=np.float32)

    def set_servo_limits(self, limits: np.ndarray):
        arr = np.asarray(limits, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != len(self.servo_joint_names):
                raise ValueError(
                    f"Expected {len(self.servo_joint_names)} joint limits, got shape {arr.shape}."
                )
            lower = -arr
            upper = arr
        elif arr.shape == (len(self.servo_joint_names), 2):
            lower = arr[:, 0]
            upper = arr[:, 1]
        else:
            raise ValueError(
                f"Expected limits shape ({len(self.servo_joint_names)}, 2), got {arr.shape}."
            )
        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise ValueError(f"Invalid joint limits: {arr}")
        self._servo_lower_limits = lower
        self._servo_upper_limits = upper

    def _clip_servo_cmd_to_limits(self) -> None:
        if self._servo_lower_limits is None or self._servo_upper_limits is None:
            raise RuntimeError("Servo limits are not initialized. Load them from the URDF before running.")
        self.servo_cmd = np.clip(self.servo_cmd, self._servo_lower_limits, self._servo_upper_limits)

    def sync_initial_joint_state_from_commands(self, base_dofs: int) -> None:
        if self.servo_dof_indices is None or self.servo_cmd.size == 0:
            return
        if self._servo_lower_limits is not None and self._servo_upper_limits is not None:
            self._clip_servo_cmd_to_limits()
        for dof_idx, cmd in zip(self.servo_dof_indices.tolist(), self.servo_cmd.tolist()):
            joint_idx = int(dof_idx) - int(base_dofs)
            if 0 <= joint_idx < self.init_joint_position.shape[0]:
                self.init_joint_position[joint_idx] = float(cmd)
            if 0 <= joint_idx < self.init_joint_velocity.shape[0]:
                self.init_joint_velocity[joint_idx] = 0.0

    # ------------------------------ Keyboard --------------------------------

    def on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.running = False
                return False
            self.pressed_keys.add(key)
        except AttributeError:
            # Unknown key, ignore
            pass

    def on_release(self, key):
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)

    # ---------------------------- Throttle logic ----------------------------

    def update_thrust(self, dt: float):
        """Integrate keyboard input (↑/↓) into throttle [0..1]."""
        if self.trajectory is not None:
            return
        if dt <= 0.0:
            return
        if keyboard.Key.up in self.pressed_keys:
            self.throttle = min(
                self._throttle_max, self.throttle + self._throttle_rate * dt
            )
        if keyboard.Key.down in self.pressed_keys:
            self.throttle = max(
                self._throttle_min, self.throttle - self._throttle_rate * dt
            )

    def apply_thrust(self, aero_solver, dt: float):
        """
        Update throttle, then forward it to the aerodynamic solver.
        This is the ONLY way thrust is applied (no RPM hacks).
        """
        self.update_thrust(dt)
        aero_solver.set_throttle(self.throttle)

    # --------------------------- Control surfaces ---------------------------

    def _update_servo_targets(self, dt: float):
        """
        Integrate keyboard into target angles for all surfaces.

        Mapping:
          w / s         : symmetric sweep up / down (both wings)
          space / shift : symmetric twist up / down (both wings)
          ← / →         : asymmetric twist (roll)
          q / e         : elevator up / down (pitch)
          a / d         : rudder left / right (yaw)
        """
        if dt <= 0.0:
            return
        if self.servo_cmd.size == 0:
            return

        sw_l = self._role_index.get("sweep_left")
        sw_r = self._role_index.get("sweep_right")
        sw = self._role_index.get("sweep")
        tw_l = self._role_index.get("twist_left")
        tw_r = self._role_index.get("twist_right")
        tw = self._role_index.get("twist")
        ele = self._role_index.get("elevator")
        rud = self._role_index.get("rudder")

        # Symmetric sweep (w/s)
        if self._key_w in self.pressed_keys:
            if sw_l is not None:
                self.servo_cmd[sw_l] += self._sweep_left_sign * self._sweep_rate * dt
            if sw_r is not None:
                self.servo_cmd[sw_r] += self._sweep_right_sign * self._sweep_rate * dt
            if sw is not None and sw_l is None and sw_r is None:
                self.servo_cmd[sw] += self._sweep_rate * dt
        if self._key_s in self.pressed_keys:
            if sw_l is not None:
                self.servo_cmd[sw_l] -= self._sweep_left_sign * self._sweep_rate * dt
            if sw_r is not None:
                self.servo_cmd[sw_r] -= self._sweep_right_sign * self._sweep_rate * dt
            if sw is not None and sw_l is None and sw_r is None:
                self.servo_cmd[sw] -= self._sweep_rate * dt

        # Symmetric twist (space / shift)
        if self._key_space in self.pressed_keys:
            if tw_l is not None:
                self.servo_cmd[tw_l] -= self._twist_rate_sym * dt
            if tw_r is not None:
                self.servo_cmd[tw_r] -= self._twist_rate_sym * dt
            if tw is not None and tw_l is None and tw_r is None:
                self.servo_cmd[tw] -= self._twist_rate_sym * dt
        if any(k in self.pressed_keys for k in self._shift_keys):
            if tw_l is not None:
                self.servo_cmd[tw_l] += self._twist_rate_sym * dt
            if tw_r is not None:
                self.servo_cmd[tw_r] += self._twist_rate_sym * dt
            if tw is not None and tw_l is None and tw_r is None:
                self.servo_cmd[tw] += self._twist_rate_sym * dt

        # Asymmetric twist (←/→)
        if keyboard.Key.left in self.pressed_keys:
            if tw_l is not None:
                self.servo_cmd[tw_l] += self._twist_rate_asym * dt
            if tw_r is not None:
                self.servo_cmd[tw_r] -= self._twist_rate_asym * dt
        if keyboard.Key.right in self.pressed_keys:
            if tw_l is not None:
                self.servo_cmd[tw_l] -= self._twist_rate_asym * dt
            if tw_r is not None:
                self.servo_cmd[tw_r] += self._twist_rate_asym * dt

        # Elevator (q/e)
        if self._key_q in self.pressed_keys:
            if ele is not None:
                self.servo_cmd[ele] += self._elevator_sign * self._tail_rate * dt
        if self._key_e in self.pressed_keys:
            if ele is not None:
                self.servo_cmd[ele] -= self._elevator_sign * self._tail_rate * dt

        # Rudder (a/d)
        if self._key_a in self.pressed_keys:
            if rud is not None:
                self.servo_cmd[rud] += self._rudder_sign * self._tail_rate * dt
        if self._key_d in self.pressed_keys:
            if rud is not None:
                self.servo_cmd[rud] -= self._rudder_sign * self._tail_rate * dt

        # Clamp joint targets to safe range
        self._clip_servo_cmd_to_limits()

    def apply_joint_commands(self, dt: float):
        """Send PD position targets for the servo joints."""
        if (
            self.drone is None
            or self.servo_dof_indices is None
            or self.servo_cmd.size == 0
        ):
            return
        if self.trajectory is None:
            self._update_servo_targets(dt)
        else:
            self._clip_servo_cmd_to_limits()
        pairs = sorted(
            zip(self.servo_dof_indices.tolist(), self.servo_cmd.tolist()),
            key=lambda x: x[0],
        )
        idx_sorted = np.asarray([p[0] for p in pairs], dtype=np.int32)
        cmd_sorted = np.asarray([p[1] for p in pairs], dtype=np.float32)
        self.drone.control_dofs_position(
            cmd_sorted,
            idx_sorted,  # dofs_idx_local
        )


def _servo_gains_from_catalog(
    drone_model,
    joint_names: Sequence[str],
    fallback_gains: tuple[float, float] | None = None,
    solver_kind: str | None = None,
):
    """
    Fetch kp/kv for joints using actuator names from the aero config / actuators.csv.
    yaw -> sweep, pitch -> twist.
    """
    if not joint_names:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    if drone_model is None or not hasattr(drone_model, "urdf_path"):
        raise ValueError("servo gain loading requires a DroneAeroModel with a valid urdf_path.")

    def apply_solver_overrides(kp_arr: np.ndarray, kv_arr: np.ndarray):
        if (solver_kind or "").strip().lower() == "lisparrow":
            from genesis.engine.solvers.drones.lisparrow import lisparrow_servo_gain_override

            for i, name in enumerate(joint_names):
                override = lisparrow_servo_gain_override(name)
                if override is None:
                    continue
                kp_arr[i] = float(override[0])
                kv_arr[i] = float(override[1])
        return kp_arr, kv_arr

    csv_path = Path(str(drone_model.urdf_path)).parent / "actuators.csv"
    if not csv_path.exists():
        if fallback_gains is None:
            raise FileNotFoundError(f"Missing actuator catalog: {csv_path}")
        kp_f, kv_f = fallback_gains
        return apply_solver_overrides(
            np.full(len(joint_names), float(kp_f), dtype=np.float32),
            np.full(len(joint_names), float(kv_f), dtype=np.float32),
        )

    catalog = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            kind = (row.get("type") or "").strip().lower()
            if name and kind:
                catalog[(name, kind)] = row

    def get_servo_gains(actuator_name: str) -> tuple[float, float]:
        row = catalog.get((actuator_name, "servo"))
        if row is None:
            raise ValueError(f"Servo actuator '{actuator_name}' not found in {csv_path}")
        kp = (row.get("kp") or "").strip()
        kv = (row.get("kv") or "").strip()
        if kp == "" or kv == "":
            raise ValueError(f"Servo actuator '{actuator_name}' missing kp/kv in {csv_path}")
        return float(kp), float(kv)

    def side_from_name(name: str) -> str | None:
        lname = name.lower()
        if "left" in lname:
            return "left"
        if "right" in lname:
            return "right"
        return None

    frame_kind = {}
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

    def pick_info(kind, side: str | None):
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

    kp, kv = [], []
    for name in joint_names:
        lname = name.lower()
        side = side_from_name(name)
        info = None
        act = None
        if "sweep" in lname:
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
        kp.append(float(kp_i))
        kv.append(float(kv_i))

    kp_arr = np.array(kp, dtype=np.float32)
    kv_arr = np.array(kv, dtype=np.float32)

    return apply_solver_overrides(kp_arr, kv_arr)


class DroneModel:
    """
    Debug helper for the winged drone + custom AeroSolver.

    If self.model_debug = True, we:
      - fetch per-surface aerodynamic forces and centers of pressure,
      - decompose forces into drag & lift (based on link velocity),
      - draw 2 debug arrows per aero link:
            * red   = drag (parallel to -velocity)
            * blue  = lift (perpendicular to velocity)
      - print basic info (velocities, angles, lift, drag, etc.).
    """

    def __init__(self):
        # Master switch for debug visualization and prints
        self.model_debug: bool = True

        # Attached objects
        self.scene: Optional[gs.Scene] = None  # type: ignore
        self.drone: Optional[gs.engine.entities.RigidEntity] = None  # type: ignore
        self.aero_solver: Optional[object] = None
        self._root_link: Optional[gs.engine.entities.RigidLink] = None  # type: ignore
        self._root_link_name: Optional[str] = None

        # Aerodynamic frame names and links (one per surface)
        self._aero_frames: List[str] = []
        self._aero_links: List[Optional[gs.engine.entities.RigidLink]] = []  # type: ignore

        # Device for torch conversion (match solver device)
        self._aero_device = gs.device

        # Debug arrow visual settings
        # (scale force → arrow length; radius controls thickness)
        self.force_vis_scale: float = 0.2
        self.arrow_radius: float = 0.03  # thicker than default (0.01) for visibility
        self.velocity_vis_scale: float = 0.1
        self.velocity_color = (1.0, 0.5, 0.0, 0.95)  # orange = root velocity
        self.velocity_height_offset: float = 0.5

        # Colors for arrows (RGBA, 0..1), mostly opaque
        self.drag_color = (1.0, 0.0, 0.0, 0.95)   # red   = drag
        self.lift_color = (0.0, 0.6, 1.0, 0.95)   # blue  = lift

        # Print every N sim steps (avoid spamming)
        self._print_every_n_steps: int = 20
        self._step_counter: int = 0
        self._debug_links: List[str] = []
        self._history: dict[str, list] = {
            "time": [],
            "root_pos": [],
            "root_vel": [],
            "root_ang_vel": [],
            "root_rpy_deg": [],
            "throttle_cmd": [],
            "servo_cmd": [],
            "joint_pos": [],
            "joint_names": [],
            "surface_force_local": [],
            "surface_cp_local": [],
            "surface_alpha_deg": [],
            "surface_beta_deg": [],
            "surface_lift": [],
            "surface_drag": [],
            "surface_side": [],
            "surface_flow_local": [],
            "surface_joint_angle": [],
            "surface_names": [],
            "prop_thrust": [],
        }

    # -------------------------- Attach + setup ---------------------------

    def attach_to_scene(self, scene: gs.Scene, drone):
        """Store scene, drone and aero solver handles."""
        self.scene = scene
        self.drone = drone
        self.aero_solver = scene.sim.aero_solver

        # Aero frames defined in custom AeroSolver (order = surfaces)
        self._aero_frames = getattr(self.aero_solver, "_aero_frames", [])

        # Map frame names to actual RigidLink objects on the drone
        self._aero_links = []
        for name in self._aero_frames:
            try:
                link = drone.get_link(name)
            except Exception:
                link = None
            self._aero_links.append(link)

        try:
            self._root_link = drone.get_link("root_link")
            self._root_link_name = "root_link"
        except Exception:
            try:
                self._root_link = drone.get_link("base")
                self._root_link_name = "base"
            except Exception:
                self._root_link = None
                self._root_link_name = None

        # Device where aero solver keeps its tensors
        if hasattr(self.aero_solver, "_aero_device"):
            self._aero_device = self.aero_solver._aero_device  # type: ignore
        else:
            self._aero_device = gs.device

    def enable_debug(self, flag: bool = True):
        """Turn on/off logging in AeroSolver and local debug."""
        self.model_debug = flag
        if self.aero_solver is not None and hasattr(self.aero_solver, "_aero_log"):
            # This flag controls writing to force_b, cp_b, alpha_dbg, ...
            self.aero_solver._aero_log = bool(flag)  # type: ignore

    def set_debug_links(self, link_names: Sequence[str]) -> None:
        self._debug_links = [str(name) for name in link_names if name]

    def _snapshot_joint_positions(self, drone, joint_names: Sequence[str]) -> np.ndarray:
        if not joint_names:
            return np.zeros(0, dtype=np.float32)
        q = drone.scene.sim.rigid_solver.get_dofs_position()[0]
        out = []
        for name in joint_names:
            j = drone.get_joint(name)
            idx = getattr(j, "dofs_idx_local", None)
            if idx is None:
                idx = j.dof_idx_local
            if isinstance(idx, (list, tuple, np.ndarray)):
                idx = int(idx[0])
            out.append(float(q[idx]))
        return np.asarray(out, dtype=np.float32)

    def root_position(self) -> np.ndarray | None:
        if self.drone is None:
            return None
        if self._root_link is not None:
            root_pos_t = self._root_link.get_pos(envs_idx=0)
        else:
            root_pos_t = self.drone.get_pos(envs_idx=0)
        root_pos = root_pos_t.detach().cpu().numpy()
        if root_pos.ndim > 1:
            root_pos = root_pos[0]
        return np.asarray(root_pos, dtype=np.float32).reshape(-1)

    # ----------------------- Fetch solver debug data ----------------------

    def _get_debug_tensors(self):
        """Fetch debug tensors from AeroSolver and convert to numpy."""
        if (
            self.scene is None
            or self.aero_solver is None
            or not hasattr(self.aero_solver, "force_b")
        ):
            return None

        L = len(self._aero_frames)
        if L == 0:
            return None

        # Shapes: (B, n_links, 3) or (B, L, 3); we only use env 0 and first L
        fb = self.aero_solver.force_b.to_torch(device=self._aero_device)[0, :L, :]
        cp = self.aero_solver.cp_b.to_torch(device=self._aero_device)[0, :L, :]

        alpha = self.aero_solver.alpha_dbg.to_torch(device=self._aero_device)[0, :L]
        beta = self.aero_solver.beta_dbg.to_torch(device=self._aero_device)[0, :L]
        lift = self.aero_solver.lift_dbg.to_torch(device=self._aero_device)[0, :L]
        drag = self.aero_solver.drag_dbg.to_torch(device=self._aero_device)[0, :L]
        side_np = self.aero_solver.side_force_dbg.to_torch(device=self._aero_device)[0, :L]

        alpha_tail_raw = None
        downwash_eps = None
        cl_wing_tail = None
        k_eps_tail = None
        flow_l = None
        joint_angle = None
        if hasattr(self.aero_solver, "alpha_tail_raw_dbg"):
            alpha_tail_raw = self.aero_solver.alpha_tail_raw_dbg.to_torch(device=self._aero_device)[0, :L]
        if hasattr(self.aero_solver, "downwash_eps_dbg"):
            downwash_eps = self.aero_solver.downwash_eps_dbg.to_torch(device=self._aero_device)[0, :L]
        if hasattr(self.aero_solver, "cl_wing_for_tail_dbg"):
            cl_wing_tail = self.aero_solver.cl_wing_for_tail_dbg.to_torch(device=self._aero_device)[0, :L]
        if hasattr(self.aero_solver, "k_eps_tail_dbg"):
            k_eps_tail = self.aero_solver.k_eps_tail_dbg.to_torch(device=self._aero_device)[0, :L]
        if hasattr(self.aero_solver, "flow_dbg"):
            flow_l = self.aero_solver.flow_dbg.to_torch(device=self._aero_device)[0, :L, :]
        if hasattr(self.aero_solver, "joint_angle_dbg"):
            joint_angle = self.aero_solver.joint_angle_dbg.to_torch(device=self._aero_device)[0, :L]

        return (
            fb.detach().cpu().numpy(),
            cp.detach().cpu().numpy(),
            alpha.detach().cpu().numpy(),
            beta.detach().cpu().numpy(),
            lift.detach().cpu().numpy(),
            drag.detach().cpu().numpy(),
            side_np.detach().cpu().numpy(),
            None if alpha_tail_raw is None else alpha_tail_raw.detach().cpu().numpy(),
            None if downwash_eps is None else downwash_eps.detach().cpu().numpy(),
            None if cl_wing_tail is None else cl_wing_tail.detach().cpu().numpy(),
            None if k_eps_tail is None else k_eps_tail.detach().cpu().numpy(),
            None if flow_l is None else flow_l.detach().cpu().numpy(),
            None if joint_angle is None else joint_angle.detach().cpu().numpy(),
        )

    def record_step(self, controller: DroneController) -> None:
        if self.scene is None or self.drone is None or self.aero_solver is None:
            return

        tensors = self._get_debug_tensors()
        if tensors is None:
            return

        (
            fb_np,
            cp_np,
            alpha_np,
            beta_np,
            lift_np,
            drag_np,
            side_np,
            _alpha_tail_raw_np,
            _downwash_eps_np,
            _cl_wing_tail_np,
            _k_eps_tail_np,
            flow_l_np,
            joint_angle_np,
        ) = tensors

        if self._root_link is not None:
            root_pos_t = self._root_link.get_pos(envs_idx=0)
            root_vel_t = self._root_link.get_vel(envs_idx=0)
            root_ang_t = self._root_link.get_ang(envs_idx=0)
            root_quat_t = self._root_link.get_quat(envs_idx=0)
        else:
            root_pos_t = self.drone.get_pos(envs_idx=0)
            root_vel_t = self.drone.get_vel(envs_idx=0)
            root_ang_t = self.drone.get_ang(envs_idx=0)
            root_quat_t = self.drone.get_quat(envs_idx=0)

        root_pos = root_pos_t.detach().cpu().numpy()
        root_vel = root_vel_t.detach().cpu().numpy()
        root_ang = root_ang_t.detach().cpu().numpy()
        root_quat = root_quat_t.detach().cpu().numpy()
        if root_pos.ndim > 1:
            root_pos = root_pos[0]
        if root_vel.ndim > 1:
            root_vel = root_vel[0]
        if root_ang.ndim > 1:
            root_ang = root_ang[0]
        if root_quat.ndim > 1:
            root_quat = root_quat[0]
        root_rpy_deg = (
            quat_to_xyz(torch.from_numpy(root_quat.astype(np.float32)[None, :]), rpy=True, degrees=True)[0]
            .detach()
            .cpu()
            .numpy()
        )

        t = float(len(self._history["time"])) * float(getattr(self.scene.sim, "_substep_dt", 0.01))
        prop_idx = next((i for i, n in enumerate(self._aero_frames) if "prop" in n.lower()), None)
        prop_thrust = float(np.linalg.norm(fb_np[prop_idx])) if prop_idx is not None else 0.0

        self._history["time"].append(t)
        self._history["root_pos"].append(root_pos.astype(np.float32))
        self._history["root_vel"].append(root_vel.astype(np.float32))
        self._history["root_ang_vel"].append(root_ang.astype(np.float32))
        self._history["root_rpy_deg"].append(root_rpy_deg.astype(np.float32))
        self._history["throttle_cmd"].append(float(controller.throttle))
        self._history["servo_cmd"].append(np.asarray(controller.servo_cmd, dtype=np.float32).copy())
        self._history["joint_pos"].append(self._snapshot_joint_positions(self.drone, controller.servo_joint_names))
        self._history["joint_names"] = list(controller.servo_joint_names)
        self._history["surface_force_local"].append(np.asarray(fb_np, dtype=np.float32).copy())
        self._history["surface_cp_local"].append(np.asarray(cp_np, dtype=np.float32).copy())
        self._history["surface_alpha_deg"].append(np.degrees(alpha_np).astype(np.float32))
        self._history["surface_beta_deg"].append(np.degrees(beta_np).astype(np.float32))
        self._history["surface_lift"].append(np.asarray(lift_np, dtype=np.float32).copy())
        self._history["surface_drag"].append(np.asarray(drag_np, dtype=np.float32).copy())
        self._history["surface_side"].append(np.asarray(side_np, dtype=np.float32).copy())
        self._history["surface_flow_local"].append(
            np.asarray(flow_l_np, dtype=np.float32).copy() if flow_l_np is not None else np.zeros_like(fb_np, dtype=np.float32)
        )
        self._history["surface_joint_angle"].append(
            np.asarray(joint_angle_np, dtype=np.float32).copy()
            if joint_angle_np is not None
            else np.zeros(len(self._aero_frames), dtype=np.float32)
        )
        self._history["surface_names"] = list(self._aero_frames)
        self._history["prop_thrust"].append(prop_thrust)

    def save_summary_plots(self, output_dir: Path) -> Optional[Path]:
        if len(self._history["time"]) < 2:
            print("[PLOT] Not enough samples collected, skipping summary plot.")
            return None

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        import matplotlib

        has_display = bool(os.environ.get("DISPLAY"))
        if not has_display:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"winged_drone_summary_{ts}.png"
        csv_path = output_dir / f"winged_drone_data_{ts}.csv"

        t = np.asarray(self._history["time"], dtype=np.float32)
        root_pos = np.asarray(self._history["root_pos"], dtype=np.float32)
        root_vel = np.asarray(self._history["root_vel"], dtype=np.float32)
        root_ang = np.asarray(self._history["root_ang_vel"], dtype=np.float32)
        root_rpy = np.asarray(self._history["root_rpy_deg"], dtype=np.float32)
        throttle = np.asarray(self._history["throttle_cmd"], dtype=np.float32)
        servo_cmd = np.asarray(self._history["servo_cmd"], dtype=np.float32)
        joint_pos = np.asarray(self._history["joint_pos"], dtype=np.float32)
        alpha_deg = np.asarray(self._history["surface_alpha_deg"], dtype=np.float32)
        beta_deg = np.asarray(self._history["surface_beta_deg"], dtype=np.float32)
        lift = np.asarray(self._history["surface_lift"], dtype=np.float32)
        drag = np.asarray(self._history["surface_drag"], dtype=np.float32)
        side = np.asarray(self._history["surface_side"], dtype=np.float32)
        flow = np.asarray(self._history["surface_flow_local"], dtype=np.float32)
        cp_local = np.asarray(self._history["surface_cp_local"], dtype=np.float32)
        joint_angle = np.asarray(self._history["surface_joint_angle"], dtype=np.float32)
        prop_thrust = np.asarray(self._history["prop_thrust"], dtype=np.float32)
        surf_names = list(self._history["surface_names"])
        joint_names = list(self._history["joint_names"])

        self._save_history_csv(
            csv_path,
            t,
            root_pos,
            root_vel,
            root_ang,
            root_rpy,
            throttle,
            prop_thrust,
            servo_cmd,
            joint_pos,
            joint_names,
            surf_names,
            alpha_deg,
            beta_deg,
            lift,
            drag,
            side,
            flow,
            cp_local,
            joint_angle,
        )

        fig, axes = plt.subplots(6, 3, figsize=(24, 28), constrained_layout=True)
        axs = axes.reshape(-1)

        def _plot_xyz(ax, arr, title, ylabel):
            ax.plot(t, arr[:, 0], label="x")
            ax.plot(t, arr[:, 1], label="y")
            ax.plot(t, arr[:, 2], label="z")
            ax.set_title(title)
            ax.set_xlabel("time [s]")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        def _plot_surface_series(ax, arr, title, ylabel):
            for i, name in enumerate(surf_names):
                ax.plot(t, arr[:, i], label=name)
            ax.set_title(title)
            ax.set_xlabel("time [s]")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)

        def _plot_joint_series(ax, arr, title, ylabel):
            if arr.size == 0:
                ax.set_title(title)
                ax.text(0.5, 0.5, "no joints", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                return
            for i, name in enumerate(joint_names):
                ax.plot(t, arr[:, i], label=name)
            ax.set_title(title)
            ax.set_xlabel("time [s]")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        _plot_xyz(axs[0], root_pos, "Root Position", "m")
        _plot_xyz(axs[1], root_vel, "Root Linear Velocity", "m/s")
        _plot_xyz(axs[2], root_ang, "Root Angular Velocity", "rad/s")
        _plot_xyz(axs[3], root_rpy, "Root Euler Angles", "deg")

        axs[4].plot(t, throttle, label="throttle_cmd")
        axs[4].plot(t, prop_thrust, label="prop_thrust_norm [N]")
        axs[4].set_title("Throttle And Propeller Thrust")
        axs[4].set_xlabel("time [s]")
        axs[4].grid(True, alpha=0.3)
        axs[4].legend(fontsize=8)

        _plot_joint_series(axs[5], servo_cmd, "Servo Command Targets", "rad")
        _plot_joint_series(axs[6], joint_pos, "Actual Joint Positions", "rad")
        if servo_cmd.size and joint_pos.size and servo_cmd.shape == joint_pos.shape:
            _plot_joint_series(axs[7], joint_pos - servo_cmd, "Joint Tracking Error", "rad")
        else:
            axs[7].axis("off")

        _plot_surface_series(axs[8], alpha_deg, "Alpha Per Surface", "deg")
        _plot_surface_series(axs[9], beta_deg, "Beta Per Surface", "deg")
        _plot_surface_series(axs[10], lift, "Lift Debug Scalar Per Surface", "N")
        _plot_surface_series(axs[11], drag, "Drag Debug Scalar Per Surface", "N")
        _plot_surface_series(axs[12], side, "Side Force Debug Scalar Per Surface", "N")
        _plot_surface_series(axs[13], joint_angle, "Solver Surface Joint Angles", "rad")

        _plot_surface_series(axs[14], np.linalg.norm(flow, axis=2), "Local Flow Magnitude Per Surface", "m/s")
        _plot_surface_series(axs[15], np.linalg.norm(cp_local, axis=2), "CP Local Distance Per Surface", "m")

        axs[16].plot(t, np.linalg.norm(root_vel, axis=1), label="|v|")
        axs[16].plot(t, np.linalg.norm(root_ang, axis=1), label="|omega|")
        axs[16].set_title("Velocity Magnitudes")
        axs[16].set_xlabel("time [s]")
        axs[16].grid(True, alpha=0.3)
        axs[16].legend(fontsize=8)

        axs[17].plot(t, root_pos[:, 2], label="altitude z")
        axs[17].plot(t, root_vel[:, 0], label="forward vx")
        axs[17].set_title("Altitude And Forward Speed")
        axs[17].set_xlabel("time [s]")
        axs[17].grid(True, alpha=0.3)
        axs[17].legend(fontsize=8)

        fig.suptitle(f"Winged Drone Flight Summary: {DRONE_NAME}", fontsize=18)
        fig.savefig(out_path, dpi=180)

        if has_display and "PYTEST_VERSION" not in os.environ:
            try:
                plt.show()
            except Exception:
                plt.close(fig)
                pass
        else:
            plt.close(fig)
        print(f"[PLOT] saved summary plot to {out_path}")
        print(f"[PLOT] saved summary plot to {out_path}", file=CONSOLE, flush=True)
        print(f"[CSV] saved flight data to {csv_path}")
        print(f"[CSV] saved flight data to {csv_path}", file=CONSOLE, flush=True)
        return out_path

    def _save_history_csv(
        self,
        path: Path,
        t: np.ndarray,
        root_pos: np.ndarray,
        root_vel: np.ndarray,
        root_ang: np.ndarray,
        root_rpy: np.ndarray,
        throttle: np.ndarray,
        prop_thrust: np.ndarray,
        servo_cmd: np.ndarray,
        joint_pos: np.ndarray,
        joint_names: Sequence[str],
        surf_names: Sequence[str],
        alpha_deg: np.ndarray,
        beta_deg: np.ndarray,
        lift: np.ndarray,
        drag: np.ndarray,
        side: np.ndarray,
        flow: np.ndarray,
        cp_local: np.ndarray,
        joint_angle: np.ndarray,
    ) -> None:
        def clean_name(name: str) -> str:
            return str(name).strip().replace(" ", "_").replace("/", "_")

        header = [
            "time_s",
            "root_pos_x_m",
            "root_pos_y_m",
            "root_pos_z_m",
            "root_vel_x_mps",
            "root_vel_y_mps",
            "root_vel_z_mps",
            "root_ang_vel_x_radps",
            "root_ang_vel_y_radps",
            "root_ang_vel_z_radps",
            "root_roll_deg",
            "root_pitch_deg",
            "root_yaw_deg",
            "throttle_cmd",
            "prop_thrust_n",
        ]
        for name in joint_names:
            cname = clean_name(name)
            header.append(f"servo_cmd_{cname}_rad")
            header.append(f"joint_pos_{cname}_rad")
            header.append(f"joint_tracking_error_{cname}_rad")
        for name in surf_names:
            cname = clean_name(name)
            header.extend(
                [
                    f"surface_alpha_{cname}_deg",
                    f"surface_beta_{cname}_deg",
                    f"surface_lift_{cname}_n",
                    f"surface_drag_{cname}_n",
                    f"surface_side_{cname}_n",
                    f"surface_joint_angle_{cname}_rad",
                    f"surface_force_local_{cname}_x_n",
                    f"surface_force_local_{cname}_y_n",
                    f"surface_force_local_{cname}_z_n",
                    f"surface_flow_local_{cname}_x_mps",
                    f"surface_flow_local_{cname}_y_mps",
                    f"surface_flow_local_{cname}_z_mps",
                    f"surface_cp_local_{cname}_x_m",
                    f"surface_cp_local_{cname}_y_m",
                    f"surface_cp_local_{cname}_z_m",
                ]
            )

        force = np.asarray(self._history["surface_force_local"], dtype=np.float32)

        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(t)):
                row = [
                    float(t[i]),
                    float(root_pos[i, 0]),
                    float(root_pos[i, 1]),
                    float(root_pos[i, 2]),
                    float(root_vel[i, 0]),
                    float(root_vel[i, 1]),
                    float(root_vel[i, 2]),
                    float(root_ang[i, 0]),
                    float(root_ang[i, 1]),
                    float(root_ang[i, 2]),
                    float(root_rpy[i, 0]),
                    float(root_rpy[i, 1]),
                    float(root_rpy[i, 2]),
                    float(throttle[i]),
                    float(prop_thrust[i]),
                ]
                for j, _ in enumerate(joint_names):
                    cmd = float(servo_cmd[i, j]) if servo_cmd.size else 0.0
                    pos = float(joint_pos[i, j]) if joint_pos.size else 0.0
                    row.extend([cmd, pos, pos - cmd])
                for j, _ in enumerate(surf_names):
                    row.extend(
                        [
                            float(alpha_deg[i, j]),
                            float(beta_deg[i, j]),
                            float(lift[i, j]),
                            float(drag[i, j]),
                            float(side[i, j]),
                            float(joint_angle[i, j]),
                            float(force[i, j, 0]),
                            float(force[i, j, 1]),
                            float(force[i, j, 2]),
                            float(flow[i, j, 0]),
                            float(flow[i, j, 1]),
                            float(flow[i, j, 2]),
                            float(cp_local[i, j, 0]),
                            float(cp_local[i, j, 1]),
                            float(cp_local[i, j, 2]),
                        ]
                    )
                writer.writerow(row)

    # ---------------------------- Debug step ------------------------------

    def debug_step(self):
        """
        Draw arrows for aerodynamic forces and print per-link debug information.

        This should be called once per simulation step *after* scene.step().
        """
        if not self.model_debug:
            return

        tensors = self._get_debug_tensors()
        if tensors is None or self.scene is None:
            return

        (
            fb_np,
            cp_np,
            alpha_np,
            beta_np,
            lift_np,
            drag_np,
            side_np,
            alpha_tail_raw_np,
            downwash_eps_np,
            cl_wing_tail_np,
            k_eps_tail_np,
            flow_l_np,
            joint_angle_np,
        ) = tensors

        # Clear old arrows so we redraw fresh ones every frame
        try:
            self.scene.clear_debug_objects()
        except Exception:
            # If viewer is not built yet, fail silently
            pass

        # Increase step counter and check if we should print this frame
        self._step_counter += 1
        do_print = (self._step_counter % self._print_every_n_steps) == 0

        if self._root_link is not None or self.drone is not None:
            try:
                if self._root_link is not None:
                    root_pos_t = self._root_link.get_pos(envs_idx=0)
                    root_vel_t = self._root_link.get_vel(envs_idx=0)
                else:
                    root_pos_t = self.drone.get_pos(envs_idx=0)
                    root_vel_t = self.drone.get_vel(envs_idx=0)

                root_pos = root_pos_t.detach().cpu().numpy()
                root_vel = root_vel_t.detach().cpu().numpy()

                if root_pos.ndim > 1:
                    root_pos = root_pos[0]
                if root_vel.ndim > 1:
                    root_vel = root_vel[0]

                root_pos = root_pos.astype(np.float32)
                root_vel = root_vel.astype(np.float32)

                if np.linalg.norm(root_vel) > 1e-4:
                    arrow_pos = root_pos + np.array(
                        [0.0, 0.0, self.velocity_height_offset], dtype=np.float32
                    )
                    self.scene.draw_debug_arrow(
                        pos=arrow_pos,
                        vec=root_vel * self.velocity_vis_scale,
                        radius=self.arrow_radius,
                        color=self.velocity_color,
                    )
            except Exception:
                pass

        for idx, (link, name) in enumerate(zip(self._aero_links, self._aero_frames)):
            # Skip if the link does not exist (should not happen)
            if link is None:
                continue

            # Local force (link frame) and center of pressure (link frame)
            f_local = fb_np[idx]
            cp_local = cp_np[idx]

            # Do not draw tiny forces (numerical noise)
            if np.linalg.norm(f_local) < 1e-4:
                continue

            # Link pose and velocity in world frame, env 0
            pos_t = link.get_pos(envs_idx=0)
            quat_t = link.get_quat(envs_idx=0)
            vel_t = link.get_vel(envs_idx=0)
            ang_t = link.get_ang(envs_idx=0)

            pos_world = pos_t.detach().cpu().numpy()
            quat_world = quat_t.detach().cpu().numpy()
            vel_world = vel_t.detach().cpu().numpy()
            ang_world = ang_t.detach().cpu().numpy()

            # Handle possible leading env dimension
            if pos_world.ndim > 1:
                pos_world = pos_world[0]
            if quat_world.ndim > 1:
                quat_world = quat_world[0]
            if vel_world.ndim > 1:
                vel_world = vel_world[0]
            if ang_world.ndim > 1:
                ang_world = ang_world[0]

            # Convert to float32 numpy (safe for torch)
            cp_local = cp_local.astype(np.float32)
            f_local = f_local.astype(np.float32)
            quat_world = quat_world.astype(np.float32)

            # Use Genesis geom helper to rotate from link frame → world frame
            # (we use small torch tensors just for this rotation)
            cp_local_t = torch.from_numpy(cp_local[None, :])
            f_local_t = torch.from_numpy(f_local[None, :])
            quat_world_t = torch.from_numpy(quat_world[None, :])

            cp_world_offset_t = transform_by_quat(cp_local_t, quat_world_t)[0]
            f_world_t = transform_by_quat(f_local_t, quat_world_t)[0]

            cp_world = pos_world + cp_world_offset_t.detach().cpu().numpy()
            f_world = f_world_t.detach().cpu().numpy()

            is_prop = "prop" in name.lower()

            if is_prop:
                axis_local_t = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
                axis_world = transform_by_quat(axis_local_t, quat_world_t)[0].detach().cpu().numpy()
                axis_world /= (np.linalg.norm(axis_world) + 1e-12)

                thrust_vec = f_world * self.force_vis_scale
                axis_vec = axis_world * self.force_vis_scale

                try:
                    self.scene.draw_debug_arrow(
                        pos=cp_world,
                        vec=axis_vec,
                        radius=self.arrow_radius * 0.5,
                        color=(1.0, 1.0, 0.0, 1.0),  # yellow = prop +X axis
                    )
                    self.scene.draw_debug_arrow(
                        pos=cp_world,
                        vec=thrust_vec,
                        radius=self.arrow_radius,
                        color=(1.0, 0.0, 1.0, 1.0),  # magenta = prop thrust (applied force)
                    )
                except Exception:
                    pass

                if do_print and self.aero_solver is not None:
                    alpha_deg = float(np.degrees(alpha_np[idx]))
                    beta_deg = float(np.degrees(beta_np[idx]))
                    thrust_N = float(np.dot(f_world, axis_world))
                    fmag = float(np.linalg.norm(f_world))
                    dot = float(thrust_N / (fmag + 1e-12))
                    try:
                        kappa = float(self.aero_solver.kappa_prop.to_torch(device=self._aero_device)[0].item())
                    except Exception:
                        kappa = float("nan")
                    torque_Nm = -kappa * thrust_N
                    print(
                        f"[PropDebug] link={name} | alpha={alpha_deg:.2f} deg | beta={beta_deg:.2f} deg | "
                        f"thrust={thrust_N:.3f} N | |F|={fmag:.3f} N | dot(F,axis)={dot:.3f} | "
                        f"kappa={kappa:.6f} | reaction_torque_z={torque_Nm:.4f} N·m"
                    )

                continue

            # ---------- Visualize per-link wind axis + perpendicular axis ----------
            # Use the SAME alpha/beta used in the solver for this link
            a = float(alpha_np[idx])
            b = float(beta_np[idx])

            ca, sa = np.cos(a), np.sin(a)
            cb, sb = np.cos(b), np.sin(b)

            # This matches AeroSolver._rot_yz(alpha, beta)
            R = np.array(
                [
                    [cb * ca, -sb, cb * sa],
                    [sb * ca, cb, sb * sa],
                    [-sa, 0.0, ca],
                ],
                dtype=np.float32,
            )

            drag_axis_local = R[:, 0]
            side_axis_local = R[:, 1]
            lift_axis_local = R[:, 2]

            drag_axis_local /= (np.linalg.norm(drag_axis_local) + 1e-12)
            side_axis_local /= (np.linalg.norm(side_axis_local) + 1e-12)
            lift_axis_local /= (np.linalg.norm(lift_axis_local) + 1e-12)

            # Rotate axes to world frame for drawing
            drag_axis_world = transform_by_quat(torch.from_numpy(drag_axis_local[None, :]), quat_world_t)[
                0
            ].detach().cpu().numpy()
            side_axis_world = transform_by_quat(torch.from_numpy(side_axis_local[None, :]), quat_world_t)[
                0
            ].detach().cpu().numpy()
            lift_axis_world = transform_by_quat(torch.from_numpy(lift_axis_local[None, :]), quat_world_t)[
                0
            ].detach().cpu().numpy()

            # ---- Arrow 1: "velocity opposite to link" axis (relative wind axis)
            # Here we draw the axis direction, scaled for visibility
            wind_axis_vec = drag_axis_world * self.force_vis_scale  # axis-only arrow

            # ---- Arrow 2: perpendicular axis (lift axis)
            perp_axis_vec = lift_axis_world * self.force_vis_scale  # axis-only arrow

            # Decompose the ACTUAL applied force along those axes.
            # This stays consistent even if the solver changes internal debug scalars.
            drag_comp = float(np.dot(f_local, drag_axis_local))
            side_comp = float(np.dot(f_local, side_axis_local))
            lift_comp = float(np.dot(f_local, lift_axis_local))

            drag_vec = drag_axis_world * (drag_comp * self.force_vis_scale)
            side_vec = side_axis_world * (side_comp * self.force_vis_scale)
            lift_vec = lift_axis_world * (lift_comp * self.force_vis_scale)

            try:
                self.scene.draw_debug_arrow(
                    pos=cp_world,
                    vec=side_vec,
                    radius=self.arrow_radius,
                    color=(0.0, 0.8, 0.0, 1.0),  # green = side force
                )

                # Forces (thicker) — these should match what the solver applies
                self.scene.draw_debug_arrow(
                    pos=cp_world,
                    vec=drag_vec,
                    radius=self.arrow_radius,
                    color=self.drag_color,
                )
                self.scene.draw_debug_arrow(
                    pos=cp_world,
                    vec=lift_vec,
                    radius=self.arrow_radius,
                    color=self.lift_color,
                )
            except Exception:
                pass


            # Optional console logging (angles, forces, velocities)
            if do_print:
                # Convert world quaternion to roll/pitch/yaw (deg).
                # `quat_to_xyz(..., rpy=True)` returns intrinsic ZYX (roll-pitch-yaw style) angles.
                quat_world_t_single = torch.from_numpy(quat_world[None, :])
                rpy = (
                    quat_to_xyz(quat_world_t_single, rpy=True, degrees=True)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                roll, pitch, yaw = rpy

                if name == "aero_frame_fuselage":
                    axes_local = torch.tensor(
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        dtype=torch.float32,
                    )
                    axes_world = (
                        transform_by_quat(axes_local, quat_world_t)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    x_w, y_w, z_w = axes_world
                    print(f"[Axes] link={name} | x_w={x_w} | y_w={y_w} | z_w={z_w}")

                alpha_deg = float(np.degrees(alpha_np[idx]))
                beta_deg = float(np.degrees(beta_np[idx]))
                lift_val = float(lift_np[idx])
                drag_val = float(drag_np[idx])
                side_val = float(side_np[idx])

                extra = ""
                if (
                    "elevator" in name
                    and alpha_tail_raw_np is not None
                    and downwash_eps_np is not None
                ):
                    alpha_raw_deg = float(np.degrees(alpha_tail_raw_np[idx]))
                    eps_deg = float(np.degrees(downwash_eps_np[idx]))
                    cl_w = float(cl_wing_tail_np[idx]) if cl_wing_tail_np is not None else float("nan")
                    k_eps = float(k_eps_tail_np[idx]) if k_eps_tail_np is not None else float("nan")
                    extra = f" | alpha_raw={alpha_raw_deg:.2f} deg | eps_dw={eps_deg:.2f} deg | CL_w={cl_w:.3f} | k_eps={k_eps:.3f}"

                print(
                    f"[DroneModel] link={name} | "
                    f"pos={pos_world} | vel={vel_world} | ang_vel={ang_world} | "
                    f"rpy(deg)=({roll:.1f}, {pitch:.1f}, {yaw:.1f}) | "
                    f"alpha={alpha_deg:.2f} deg | beta={beta_deg:.2f} deg | "
                    f"Lift={lift_val:.3f} | Drag={drag_val:.3f} | Side={side_val:.3f}"
                    f"{extra}"
                )

                # Lisparrow-specific periodic verbose (same cadence as other stats).
                if (
                    self.aero_solver is not None
                    and self.aero_solver.__class__.__name__ == "LisparrowAeroSolver"
                ):
                    flow_txt = "n/a"
                    if flow_l_np is not None:
                        fl = flow_l_np[idx]
                        flow_txt = f"({fl[0]:+.4f},{fl[1]:+.4f},{fl[2]:+.4f})"
                    joint_txt = "none"
                    if (
                        joint_angle_np is not None
                        and hasattr(self.aero_solver, "_surf_joint_name")
                    ):
                        jnames = getattr(self.aero_solver, "_surf_joint_name", [])
                        jn = jnames[idx] if idx < len(jnames) else ""
                        if jn:
                            joint_txt = f"{jn}={float(joint_angle_np[idx]):+.4f} rad"
                    fmag_local = float(np.linalg.norm(f_local))
                    print(
                        f"[LisparrowVerbose] link={name} | joint={joint_txt} | "
                        f"cp_l=({cp_local[0]:+.4f},{cp_local[1]:+.4f},{cp_local[2]:+.4f}) | "
                        f"cp_w=({cp_world[0]:+.4f},{cp_world[1]:+.4f},{cp_world[2]:+.4f}) | "
                        f"F_l=({f_local[0]:+.4f},{f_local[1]:+.4f},{f_local[2]:+.4f}) | "
                        f"|F|={fmag_local:.4f} N | alpha={alpha_deg:+.2f} deg | "
                        f"beta={beta_deg:+.2f} deg | flow_l={flow_txt}"
                    )

    def print_joint_positions(self, drone, joint_names: List[str]):
        do_print = (self._step_counter % self._print_every_n_steps) == 0
        if not do_print:
            return

        if joint_names:
            print("\n--- Joint positions (DOFs) ---")
            for name in joint_names:
                j = drone.get_joint(name)

                # usa dofs_idx_local (nuovo) se c'è, altrimenti fallback
                idx = getattr(j, "dofs_idx_local", None)
                if idx is None:
                    idx = j.dof_idx_local  # deprecated ma ok per ora
                # idx può essere int o lista/np array (1 DOF -> prendi il primo)
                if isinstance(idx, (list, tuple, np.ndarray)):
                    idx = int(idx[0])

                # LEGGI POSIZIONE DOF (non qpos!)
                q = drone.scene.sim.rigid_solver.get_dofs_position()[0]  # env 0
                pos = float(q[idx])

                print(f"{name:30s} | dofs_idx={idx:2d} | q={pos:+.4f}")

            joint_pos = drone.get_dofs_position()
            print("Full DOF positions array:", joint_pos[0], "\n")

        for ln in self._debug_links:
            try:
                L = drone.get_link(ln)
            except Exception:
                continue
            p = L.get_pos(envs_idx=0)
            q = L.get_quat(envs_idx=0)
            print(ln, p, q)

# ------------------------------- Sim thread ---------------------------------


def run_sim(scene: gs.Scene, drone, controller: DroneController, model: DroneModel):
    """
    Background simulation loop.
    The viewer is managed by Genesis (already running in its own thread),
    so we do NOT call viewer.run() here.
    """
    aero_solver = scene.sim.aero_solver
    last_time = time.time()

    while controller.running:
        v = scene.viewer
        if v is not None and hasattr(v, "is_alive") and not v.is_alive():
            controller.running = False
            break

        now = time.time()
        dt = now - last_time
        last_time = now

        # Fallback for the very first frame
        if dt <= 0.0:
            dt = scene.sim._substep_dt if hasattr(scene.sim, "_substep_dt") else 0.01

        sim_dt = float(getattr(scene.sim, "_substep_dt", dt))
        controller.update_prescribed_trajectory(sim_dt)

        # 1) Control surfaces (servo joints)
        controller.apply_joint_commands(dt)
        if model._step_counter % model._print_every_n_steps == 0:
            print("pressed_keys:", controller.pressed_keys)
        # 2) Thrust via AeroSolver (ONLY path to apply thrust)
        controller.apply_thrust(aero_solver, dt)

        # 3) Step physics and refresh viewer
        scene.step()  # refresh_visualizer=True by default

        root_pos = model.root_position()
        if root_pos is not None and root_pos.shape[0] >= 3 and float(root_pos[2]) <= KILL_ALTITUDE_Z:
            print(
                f"[KILL] z={float(root_pos[2]):.3f} m <= {KILL_ALTITUDE_Z:.3f} m. Stopping simulation.",
                file=CONSOLE,
                flush=True,
            )
            controller.running = False
            break

        model.record_step(controller)

        # 4) Debug visualization of aero forces (lift + drag arrows)
        model.debug_step()
        model.print_joint_positions(drone, controller.servo_joint_names)

        # 5) Limit loop rate to viewer max FPS
        if v is not None and v.max_FPS > 0:
            time.sleep(1.0 / v.max_FPS)

        if "PYTEST_VERSION" in os.environ:
            break


# ---------------------------------- Main ------------------------------------
def main():
    args = _parse_args()
    config = _resolve_drone_config(DRONE_NAME)
    urdf_path = Path(config["urdf_path"]).expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Initialize Genesis (GPU backend if available)
    gs.init(backend=gs.cpu)

    solver_kind = config.get("aero_solver_kind") or AERO_SOLVER_KIND
    _configure_aero_solver(str(solver_kind))

    layout = _servo_layout_from_role_names(
        urdf_path,
        config.get("servo_role_names"),
        config.get("servo_joint_names"),
    )
    if layout is None:
        layout = _resolve_servo_layout(urdf_path, config.get("servo_joint_names"))
    controller = DroneController(
        layout,
        sweep_command_signs=tuple(config.get("sweep_command_signs", (1.0, -1.0))),
        tail_command_signs=tuple(config.get("tail_command_signs", (1.0, 1.0))),
    )
    trajectory = _load_prescribed_trajectory(args.trajectory)
    controller.set_prescribed_trajectory(trajectory)
    servo_joint_names = controller.servo_joint_names

    # Scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=False,
            enable_adjacent_collision=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-2.0, -2.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        show_viewer=True,
    )

    # Ground
    _ = scene.add_entity(gs.morphs.Plane())

    # URDF path (your parametrized winged drone)
    NACA = config.get("naca")  # used in the URDF

    aero_config = _resolve_aero_config(str(solver_kind))
    drone_model = DroneAeroModel(str(urdf_path), config_override=aero_config)
    links_to_keep = drone_model.required_links(servo_joint_names)
    # Drone as generic URDF (RigidEntity)
    drone = scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(urdf_path),
            pos=controller.init_pos,
            quat=euler_to_quat(controller.init_euler),
            collision=True,
            merge_fixed_links=True,
            links_to_keep=links_to_keep,
        )
    )

    # Camera follow (use viewer follow_entity from Genesis docs)
    if scene.viewer is not None:
        scene.viewer.follow_entity(drone)

    # Build batched scene (B=1 now, easy to scale later)
    scene.build(n_envs=BATCH_SIZE, env_spacing=(4.0, 4.0))

    base_dofs = 6 if int(drone.n_qs) - int(drone.n_dofs) == 1 else 0
    joint_qpos = int(drone.n_qs) - 7 if base_dofs == 6 else int(drone.n_qs)
    controller.init_joint_state(joint_qpos)

    # Map joints → DOF indices (local)
    servo_dof_indices = []
    for name in servo_joint_names:
        j = drone.get_joint(name)
        idx = getattr(j, "dofs_idx_local", None)
        if idx is None:
            idx = j.dof_idx_local
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = int(idx[0])
        servo_dof_indices.append(idx)

    drone_model.validate_entity(drone, servo_joint_names, servo_dof_indices)
    controller.attach_drone(drone, servo_dof_indices)

    if servo_joint_names:
        # Servo joint position limits come from the URDF (no hardcoded numbers).
        controller.set_servo_limits(
            _load_joint_position_limits_from_urdf(str(urdf_path), servo_joint_names)
        )
        # DEBUG: verify mapping between servo_cmd indices and actual DOF indices
        print("\n--- Command mapping (servo_cmd index -> joint -> dofs_idx) ---")
        for i, name in enumerate(controller.servo_joint_names):
            j = drone.get_joint(name)
            idx = getattr(j, "dofs_idx_local", None)
            if idx is None:
                idx = j.dof_idx_local
            if isinstance(idx, (list, tuple, np.ndarray)):
                idx = int(idx[0])
            print(f"cmd[{i}] -> {name:30s} -> dofs_idx={idx}")
        print("servo_dof_indices array:", controller.servo_dof_indices, "\n")

        # PD gains from actuator catalog
        kp, kv = _servo_gains_from_catalog(
            drone_model,
            servo_joint_names,
            fallback_gains=config.get("fallback_servo_gains"),
            solver_kind=config.get("aero_solver_kind") or AERO_SOLVER_KIND,
        )
        drone.set_dofs_kp(kp=kp, dofs_idx_local=servo_dof_indices)
        drone.set_dofs_kv(kv=kv, dofs_idx_local=servo_dof_indices)

    controller.sync_initial_joint_state_from_commands(base_dofs)

    # Initial state (position + orientation + joints)
    if base_dofs == 6:
        qpos_single = np.concatenate(
            (
                controller.init_pos,
                euler_to_quat(controller.init_euler),
                controller.init_joint_position,
            )
        )
        dofs_single = np.concatenate(
            (
                controller.init_vel,
                controller.init_ang_vel,
                controller.init_joint_velocity,
            )
        )
    else:
        qpos_single = controller.init_joint_position
        dofs_single = controller.init_joint_velocity

    initial_position = np.tile(qpos_single, (BATCH_SIZE, 1))
    initial_velocity = np.tile(dofs_single, (BATCH_SIZE, 1))

    drone.set_qpos(initial_position)
    drone.set_dofs_velocity(initial_velocity)

    # Register drone in AeroSolver AFTER build (batch size known)
    scene.sim.aero_solver.add_target(drone, drone_model=drone_model)
    if hasattr(scene.sim.aero_solver, "set_verbose_init"):
        # We print detailed Lisparrow lines periodically from DroneModel.debug_step.
        scene.sim.aero_solver.set_verbose_init(False)
    _log_aero_surface_joint_mapping(drone, drone_model, scene.sim.aero_solver)
    # Per-run NACA override (does not touch the aero config).
    if NACA and hasattr(scene.sim.aero_solver, "apply_naca_wing_override"):
        scene.sim.aero_solver.apply_naca_wing_override(NACA)

    # Create debug model and attach to scene + drone
    model = DroneModel()
    model.attach_to_scene(scene, drone)
    model.set_debug_links(config.get("debug_links", []))
    model.enable_debug(True)  # turn on aero logging + arrows

    # Keyboard listener (start after build)
    listener = keyboard.Listener(
        on_press=controller.on_press, on_release=controller.on_release
    )
    listener.start()

    # Help
    _print_controls(layout)
    if trajectory is not None:
        print(f"[TRAJECTORY] prescribed trajectory loaded from {trajectory.source}", file=CONSOLE, flush=True)

    # Simulation in background thread (viewer already running internally)
    sim_thread = threading.Thread(
        target=run_sim, args=(scene, drone, controller, model), daemon=True
    )
    sim_thread.start()

    # If DroneModel.model_debug = True, arrows show drag & lift for each aero link.
    # The arrows are thicker and colored so they stay visible even when partly hidden.

    # Keep main thread alive while viewer is open
    interrupted = False
    old_sigint = signal.getsignal(signal.SIGINT)
    try:
        def _handle_sigint(signum, frame):
            nonlocal interrupted
            interrupted = True
            controller.running = False
            print("[INFO] Ctrl+C received, stopping simulation and generating summary plot...", file=CONSOLE, flush=True)
            try:
                if scene.viewer is not None and hasattr(scene.viewer, "stop"):
                    scene.viewer.stop()
            except Exception:
                pass

        signal.signal(signal.SIGINT, _handle_sigint)
        while controller.running:
            if scene.viewer is not None and hasattr(scene.viewer, "is_alive") and not scene.viewer.is_alive():
                controller.running = False
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        interrupted = True
        controller.running = False
        print("[INFO] Ctrl+C received, stopping simulation and generating summary plot...", file=CONSOLE, flush=True)
    finally:
        try:
            signal.signal(signal.SIGINT, old_sigint)
        except Exception:
            pass
        controller.running = False
        try:
            if scene.viewer is not None and hasattr(scene.viewer, "stop"):
                scene.viewer.stop()
        except Exception:
            pass
        try:
            listener.stop()
        except NotImplementedError:
            pass
        try:
            sim_thread.join(timeout=2.0)
        except KeyboardInterrupt:
            interrupted = True
            controller.running = False
        plot_path = model.save_summary_plots(Path.cwd() / "winged_drone_fly_result")
        if plot_path is None:
            print("[PLOT] no plot generated.", file=CONSOLE, flush=True)
        try:
            log_file.close()
        except Exception:
            pass
        if interrupted:
            return



if __name__ == "__main__":
    main()
