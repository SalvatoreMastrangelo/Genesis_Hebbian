import os
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence
from pathlib import Path
import csv
import xml.etree.ElementTree as ET

from pynput import keyboard
import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import (
    euler_to_quat,
    transform_by_quat,
    quat_to_xyz,
)
from genesis.assets.urdf.mydrone.drone import DroneAeroModel, SurfaceKind
import sys

# -------- Redirect ONLY print() output to file --------
log_file = open("winged_drone_output.txt", "w", buffering=1)
sys.stdout = log_file


# Batch size: keep 1 for now, but code is structured to extend to B > 1.
BATCH_SIZE = 1
AERO_SOLVER_KIND = os.environ.get("AERO_SOLVER_KIND", "simple").strip().lower()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Select which drone to fly.
DRONE_NAME = "lisparrow"  # "mydrone" or "lisparrow"

MYDRONE_URDF = PROJECT_ROOT / "genesis/assets/urdf/mydrone/[0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4, 0.2, 2, 0, 2, 2.5, 3, 4, 16].urdf"
LISPARROW_URDF = PROJECT_ROOT / "genesis/assets/urdf/lisparrow/lisparrow.urdf"

DRONE_CONFIGS = {
    "mydrone": {
        "urdf_path": MYDRONE_URDF,
        "naca": "3416",
        "servo_joint_names": None,
        "debug_links": ["fuselage", "left_wing", "right_wing", "elevator_hinge", "rudder"],
        "fallback_servo_gains": None,
        "aero_solver_kind": None,
    },
    "lisparrow": {
        "urdf_path": LISPARROW_URDF,
        "naca": None,
        "servo_joint_names": None,
        "debug_links": [
            "fuselage",
            "center_wing",
            "left_outer_wing",
            "right_outer_wing",
            "elevator",
            "rudder",
        ],
        "fallback_servo_gains": (20.0, 2.0),
        "aero_solver_kind": "lisparrow",
    },
}


def _resolve_aero_config(solver_kind: str) -> dict:
    from genesis.engine.solvers.drones.simple_drone import SimpleDroneAeroParameters
    from genesis.engine.solvers.drones.lisparrow import LisparrowAeroParameters
    name = (solver_kind or "").strip().lower()
    if name in ("lisparrow", "cpp", "morphing"):
        return LisparrowAeroParameters.as_dict()
    return SimpleDroneAeroParameters.as_dict()


def _configure_aero_solver(solver_kind: str) -> None:
    name = (solver_kind or "").strip().lower()
    if name not in ("lisparrow", "cpp", "morphing"):
        return
    from genesis.engine.solvers.drones.lisparrow import LisparrowAeroSolver
    import genesis.engine.simulator as gs_sim
    import genesis.engine.solvers as gs_solvers

    gs_sim.AeroSolver = LisparrowAeroSolver
    gs_solvers.AeroSolver = LisparrowAeroSolver


def _load_joint_position_limits_from_urdf(urdf_path: str, joint_names: List[str]) -> np.ndarray:
    """
    Return per-joint absolute position limits from the URDF (in radians).

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
        out.append(max(abs(lo), abs(hi)))
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


class DroneController:
    """
    High-level keyboard controller for the winged drone.

    - Keeps throttle (0..1) and aerodynamic control surface joints.
    - Exposes:
        * update_thrust(dt): integrate keys → throttle
        * apply_thrust(aero_solver, dt): update_thrust + set_throttle
        * apply_joint_commands(dt): integrate keys → servo DOF targets
    """

    def __init__(self, layout: ServoLayout):
        # Initial spawn state for the drone
        self.init_pos = np.array([0.0, 0.0, 20.0], dtype=np.float32)
        self.init_vel = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        self.init_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.init_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.init_joint_velocity = np.zeros(0, dtype=np.float32)
        self.init_joint_position = np.zeros(0, dtype=np.float32)

        # Runtime
        self.running: bool = True
        self.pressed_keys: set = set()

        # Throttle state (broadcast inside AeroSolver to all envs)
        self.throttle: float = 0.25
        self._throttle_min: float = 0.0
        self._throttle_max: float = 1.0
        self._throttle_rate: float = 0.2  # change per second

        # Servo joints (configurable)
        self.servo_joint_names: List[str] = list(layout.joint_names)
        self._role_index: dict[str, int] = dict(layout.role_index)

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

        # Filled from URDF in main() (no defaults).
        self._servo_limits: Optional[np.ndarray] = None

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
        limits = np.asarray(limits, dtype=np.float32).reshape(-1)
        if limits.shape[0] != len(self.servo_joint_names):
            raise ValueError(
                f"Expected {len(self.servo_joint_names)} joint limits, got shape {limits.shape}."
            )
        if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError(f"Invalid joint limits: {limits}")
        self._servo_limits = limits

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
                self.servo_cmd[sw_l] += self._sweep_rate * dt
            if sw_r is not None:
                self.servo_cmd[sw_r] -= self._sweep_rate * dt
            if sw is not None and sw_l is None and sw_r is None:
                self.servo_cmd[sw] += self._sweep_rate * dt
        if self._key_s in self.pressed_keys:
            if sw_l is not None:
                self.servo_cmd[sw_l] -= self._sweep_rate * dt
            if sw_r is not None:
                self.servo_cmd[sw_r] += self._sweep_rate * dt
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
                self.servo_cmd[ele] += self._tail_rate * dt
        if self._key_e in self.pressed_keys:
            if ele is not None:
                self.servo_cmd[ele] -= self._tail_rate * dt

        # Rudder (a/d)
        if self._key_a in self.pressed_keys:
            if rud is not None:
                self.servo_cmd[rud] += self._tail_rate * dt
        if self._key_d in self.pressed_keys:
            if rud is not None:
                self.servo_cmd[rud] -= self._tail_rate * dt

        # Clamp joint targets to safe range
        if self._servo_limits is None:
            raise RuntimeError("Servo limits are not initialized. Load them from the URDF before running.")
        self.servo_cmd = np.clip(self.servo_cmd, -self._servo_limits, self._servo_limits)

    def apply_joint_commands(self, dt: float):
        """Send PD position targets for the servo joints."""
        if (
            self.drone is None
            or self.servo_dof_indices is None
            or self.servo_cmd.size == 0
        ):
            return
        self._update_servo_targets(dt)
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
):
    """
    Fetch kp/kv for joints using actuator names from the aero config / actuators.csv.
    yaw -> sweep, pitch -> twist.
    """
    if not joint_names:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    if drone_model is None or not hasattr(drone_model, "urdf_path"):
        raise ValueError("servo gain loading requires a DroneAeroModel with a valid urdf_path.")

    csv_path = Path(str(drone_model.urdf_path)).parent / "actuators.csv"
    if not csv_path.exists():
        if fallback_gains is None:
            raise FileNotFoundError(f"Missing actuator catalog: {csv_path}")
        kp_f, kv_f = fallback_gains
        return (
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

    return np.array(kp, dtype=np.float32), np.array(kv, dtype=np.float32)


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
        if hasattr(self.aero_solver, "alpha_tail_raw_dbg"):
            alpha_tail_raw = self.aero_solver.alpha_tail_raw_dbg.to_torch(device=self._aero_device)[0, :L]
        if hasattr(self.aero_solver, "downwash_eps_dbg"):
            downwash_eps = self.aero_solver.downwash_eps_dbg.to_torch(device=self._aero_device)[0, :L]
        if hasattr(self.aero_solver, "cl_wing_for_tail_dbg"):
            cl_wing_tail = self.aero_solver.cl_wing_for_tail_dbg.to_torch(device=self._aero_device)[0, :L]
        if hasattr(self.aero_solver, "k_eps_tail_dbg"):
            k_eps_tail = self.aero_solver.k_eps_tail_dbg.to_torch(device=self._aero_device)[0, :L]

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
        )

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
                axis_local_t = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
                axis_world = transform_by_quat(axis_local_t, quat_world_t)[0].detach().cpu().numpy()
                axis_world /= (np.linalg.norm(axis_world) + 1e-12)

                thrust_vec = f_world * self.force_vis_scale
                axis_vec = axis_world * self.force_vis_scale

                try:
                    self.scene.draw_debug_arrow(
                        pos=cp_world,
                        vec=axis_vec,
                        radius=self.arrow_radius * 0.5,
                        color=(1.0, 1.0, 0.0, 1.0),  # yellow = prop +Z axis
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
        time.sleep(2.5)
        now = time.time()
        dt = now - last_time
        last_time = now

        # Fallback for the very first frame
        if dt <= 0.0:
            dt = scene.sim._substep_dt if hasattr(scene.sim, "_substep_dt") else 0.01

        # 1) Control surfaces (servo joints)
        controller.apply_joint_commands(dt)
        if model._step_counter % model._print_every_n_steps == 0:
            print("pressed_keys:", controller.pressed_keys)
        # 2) Thrust via AeroSolver (ONLY path to apply thrust)
        controller.apply_thrust(aero_solver, dt)

        # 3) Step physics and refresh viewer
        scene.step()  # refresh_visualizer=True by default

        # 4) Debug visualization of aero forces (lift + drag arrows)
        model.debug_step()
        model.print_joint_positions(drone, controller.servo_joint_names)

        #time.sleep(0.02)  # yield to other threads

        # 5) Limit loop rate to viewer max FPS
        v = scene.viewer
        if v is not None and v.max_FPS > 0:
            time.sleep(1.0 / v.max_FPS)

        if "PYTEST_VERSION" in os.environ:
            break


# ---------------------------------- Main ------------------------------------
def main():
    config = _resolve_drone_config(DRONE_NAME)
    urdf_path = Path(config["urdf_path"]).expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Initialize Genesis (GPU backend if available)
    gs.init(backend=gs.gpu)

    solver_kind = config.get("aero_solver_kind") or AERO_SOLVER_KIND
    _configure_aero_solver(str(solver_kind))

    layout = _resolve_servo_layout(urdf_path, config.get("servo_joint_names"))
    controller = DroneController(layout)
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
        )
        drone.set_dofs_kp(kp=kp, dofs_idx_local=servo_dof_indices)
        drone.set_dofs_kv(kv=kv, dofs_idx_local=servo_dof_indices)

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

    # Simulation in background thread (viewer already running internally)
    sim_thread = threading.Thread(
        target=run_sim, args=(scene, drone, controller, model), daemon=True
    )
    sim_thread.start()

    # If DroneModel.model_debug = True, arrows show drag & lift for each aero link.
    # The arrows are thicker and colored so they stay visible even when partly hidden.

    # Keep main thread alive while viewer is open
    try:
        while controller.running:
            time.sleep(0.1)
    finally:
        controller.running = False
        try:
            listener.stop()
        except NotImplementedError:
            pass
        sim_thread.join(timeout=2.0)
        try:
            log_file.close()
        except Exception:
            pass



if __name__ == "__main__":
    main()
