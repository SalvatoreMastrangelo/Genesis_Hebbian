import os
import time
import threading
from typing import List, Optional

from pynput import keyboard
import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import (
    euler_to_quat,
    transform_by_quat,
    quat_to_xyz,
)


# Batch size: keep 1 for now, but code is structured to extend to B > 1.
BATCH_SIZE = 1


class DroneController:
    """
    High-level keyboard controller for the winged drone.

    - Keeps throttle (0..1) and aerodynamic control surface joints.
    - Exposes:
        * update_thrust(dt): integrate keys → throttle
        * apply_thrust(aero_solver, dt): update_thrust + set_throttle
        * apply_joint_commands(dt): integrate keys → servo DOF targets
    """

    def __init__(self):
        # Initial spawn state for the drone
        self.init_pos = np.array([0.0, 0.0, 20.0], dtype=np.float32)
        self.init_vel = np.array([8.0, 0.0, 0.0], dtype=np.float32)
        self.init_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.init_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.init_joint_velocity = np.zeros(6, dtype=np.float32)  # 6 servo joints
        self.init_joint_position = np.zeros(6, dtype=np.float32)

        # Runtime
        self.running: bool = True
        self.pressed_keys: set = set()

        # Throttle state (broadcast inside AeroSolver to all envs)
        self.throttle: float = 0.1
        self._throttle_min: float = 0.0
        self._throttle_max: float = 1.0
        self._throttle_rate: float = 0.2  # change per second

        # Servo joints: exact order / names from URDF
        self.servo_joint_names: List[str] = [
            "joint_0_sweep_left_wing",
            "joint_0_sweep_right_wing",
            "joint_1_twist_left_wing",
            "joint_1_twist_right_wing",
            "elevator_pitch_joint",
            "rudder_yaw_joint",
        ]

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

        # [sweep_L, sweep_R, twist_L, twist_R, elevator, rudder]
        self._servo_limits = np.array(
            [0.8, 0.8, 0.6, 0.6, 0.6, 0.6], dtype=np.float32
        )

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

        SW_L, SW_R, TW_L, TW_R, ELE, RUD = range(6)

        # Symmetric sweep (w/s)
        if self._key_w in self.pressed_keys:
            self.servo_cmd[SW_L] += self._sweep_rate * dt
            self.servo_cmd[SW_R] -= self._sweep_rate * dt
        if self._key_s in self.pressed_keys:
            self.servo_cmd[SW_L] -= self._sweep_rate * dt
            self.servo_cmd[SW_R] += self._sweep_rate * dt

        # Symmetric twist (space / shift)
        if self._key_space in self.pressed_keys:
            self.servo_cmd[TW_L] -= self._twist_rate_sym * dt
            self.servo_cmd[TW_R] -= self._twist_rate_sym * dt
        if any(k in self.pressed_keys for k in self._shift_keys):
            self.servo_cmd[TW_L] += self._twist_rate_sym * dt
            self.servo_cmd[TW_R] += self._twist_rate_sym * dt

        # Asymmetric twist (←/→)
        if keyboard.Key.left in self.pressed_keys:
            self.servo_cmd[TW_L] += self._twist_rate_asym * dt
            self.servo_cmd[TW_R] -= self._twist_rate_asym * dt
        if keyboard.Key.right in self.pressed_keys:
            self.servo_cmd[TW_L] -= self._twist_rate_asym * dt
            self.servo_cmd[TW_R] += self._twist_rate_asym * dt

        # Elevator (q/e)
        if self._key_q in self.pressed_keys:
            self.servo_cmd[ELE] += self._tail_rate * dt
        if self._key_e in self.pressed_keys:
            self.servo_cmd[ELE] -= self._tail_rate * dt

        # Rudder (a/d)
        if self._key_a in self.pressed_keys:
            self.servo_cmd[RUD] += self._tail_rate * dt
        if self._key_d in self.pressed_keys:
            self.servo_cmd[RUD] -= self._tail_rate * dt

        # Clamp joint targets to safe range
        self.servo_cmd = np.clip(self.servo_cmd, -self._servo_limits, self._servo_limits)

    def apply_joint_commands(self, dt: float):
        """Send PD position targets for the servo joints."""
        if self.drone is None or self.servo_dof_indices is None:
            return
        self._update_servo_targets(dt)
        self.drone.control_dofs_position(
            self.servo_cmd.astype(np.float32),
            self.servo_dof_indices,  # dofs_idx_local
        )


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

        # Aerodynamic frame names and links (one per surface)
        self._aero_frames: List[str] = []
        self._aero_links: List[Optional[gs.engine.entities.RigidLink]] = []  # type: ignore

        # Device for torch conversion (match solver device)
        self._aero_device = gs.device

        # Debug arrow visual settings
        # (scale force → arrow length; radius controls thickness)
        self.force_vis_scale: float = 0.2
        self.arrow_radius: float = 0.03  # thicker than default (0.01) for visibility

        # Colors for arrows (RGBA, 0..1), mostly opaque
        self.drag_color = (1.0, 0.0, 0.0, 0.95)   # red   = drag
        self.lift_color = (0.0, 0.6, 1.0, 0.95)   # blue  = lift

        # Print every N sim steps (avoid spamming)
        self._print_every_n_steps: int = 30
        self._step_counter: int = 0

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

        return (
            fb.detach().cpu().numpy(),
            cp.detach().cpu().numpy(),
            alpha.detach().cpu().numpy(),
            beta.detach().cpu().numpy(),
            lift.detach().cpu().numpy(),
            drag.detach().cpu().numpy(),
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

        fb_np, cp_np, alpha_np, beta_np, lift_np, drag_np = tensors

        # Clear old arrows so we redraw fresh ones every frame
        try:
            self.scene.clear_debug_objects()
        except Exception:
            # If viewer is not built yet, fail silently
            pass

        # Increase step counter and check if we should print this frame
        self._step_counter += 1
        do_print = (self._step_counter % self._print_every_n_steps) == 0

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

            # ---------- Decompose total force into drag + lift ----------

            # Direction of motion (world frame)
            v_world = vel_world.astype(np.float32)
            v_norm = float(np.linalg.norm(v_world))
            if v_norm < 1e-4:
                # If velocity is almost zero, pick a default axis
                vel_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                vel_dir = v_world / v_norm

            # Drag is the component of force parallel to -velocity
            drag_axis = -vel_dir  # unit vector
            drag_mag = float(np.dot(f_world, drag_axis))
            f_drag = drag_mag * drag_axis

            # Lift is the component of force perpendicular to velocity
            f_lift = f_world - f_drag

            # Scale vectors for visualization (longer arrows, more visible)
            drag_vec = f_drag * self.force_vis_scale
            lift_vec = f_lift * self.force_vis_scale

            # Draw two arrows at the center of pressure:
            #   - red   = drag   (along -velocity)
            #   - blue  = lift   (perpendicular to velocity)
            try:
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
                # Debug drawing is best-effort only
                pass

            # Optional console logging (angles, forces, velocities)
            if do_print:
                # Convert world quaternion to roll/pitch/yaw (rad → deg)
                quat_world_t_single = torch.from_numpy(quat_world[None, :])
                rpy = (
                    quat_to_xyz(quat_world_t_single)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                roll, pitch, yaw = rpy

                alpha_deg = float(np.degrees(alpha_np[idx]))
                beta_deg = float(np.degrees(beta_np[idx]))
                lift_val = float(lift_np[idx])
                drag_val = float(drag_np[idx])

                print(
                    f"[DroneModel] link={name} | "
                    f"pos={pos_world} | vel={vel_world} | ang_vel={ang_world} | "
                    f"rpy(deg)=({np.degrees(roll):.1f}, "
                    f"{np.degrees(pitch):.1f}, {np.degrees(yaw):.1f}) | "
                    f"alpha={alpha_deg:.2f} deg | beta={beta_deg:.2f} deg | "
                    f"Lift={lift_val:.3f} | Drag={drag_val:.3f}"
                )


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
        now = time.time()
        dt = now - last_time
        last_time = now

        # Fallback for the very first frame
        if dt <= 0.0:
            dt = scene.sim._substep_dt if hasattr(scene.sim, "_substep_dt") else 0.01

        # 1) Control surfaces (servo joints)
        controller.apply_joint_commands(dt)

        # 2) Thrust via AeroSolver (ONLY path to apply thrust)
        controller.apply_thrust(aero_solver, dt)

        # 3) Step physics and refresh viewer
        scene.step()  # refresh_visualizer=True by default

        # 4) Debug visualization of aero forces (lift + drag arrows)
        model.debug_step()

        # 5) Limit loop rate to viewer max FPS
        v = scene.viewer
        if v is not None and v.max_FPS > 0:
            time.sleep(1.0 / v.max_FPS)

        if "PYTEST_VERSION" in os.environ:
            break


# ---------------------------------- Main ------------------------------------


def main():
    # Initialize Genesis (GPU backend if available)
    gs.init(backend=gs.gpu)

    controller = DroneController()

    # Scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
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
    urdf_path = (
        "genesis/assets/urdf/mydrone/"
        "[0.7, 3.5, 0.73, 0.38, 0.38, 0.18, 1.3, 0.16, 1.3, 0, 0.25, 2, 2.5, 2, -3].urdf"
    )

    # Drone as generic URDF (RigidEntity)
    drone = scene.add_entity(
        morph=gs.morphs.URDF(
            file=urdf_path,
            pos=controller.init_pos,
            quat=euler_to_quat(controller.init_euler),
            collision=True,
            merge_fixed_links=True,
            links_to_keep=[
                "aero_frame_fuselage",
                "aero_frame_left_wing_prop",
                "aero_frame_left_wing_free",
                "aero_frame_right_wing_prop",
                "aero_frame_right_wing_free",
                "aero_frame_elevator_left",
                "aero_frame_elevator_right",
                "aero_frame_rudder",
                "elevator_left",
                "elevator_right",
                "rudder",
                "prop_frame_fuselage_0",
                "fuselage",
                "left_wing",
                "right_wing",
            ],
        )
    )

    # Camera follow (use viewer follow_entity from Genesis docs)
    if scene.viewer is not None:
        scene.viewer.follow_entity(drone)

    # Build batched scene (B=1 now, easy to scale later)
    scene.build(n_envs=BATCH_SIZE, env_spacing=(4.0, 4.0))

    # Map joints → DOF indices (local)
    servo_joint_names = controller.servo_joint_names
    servo_dof_indices = [drone.get_joint(name).dof_idx_local for name in servo_joint_names]
    controller.attach_drone(drone, servo_dof_indices)

    # PD gains (tune later)
    kp = np.array([8.0] * len(servo_dof_indices), dtype=np.float32)
    kv = np.array([2.0] * len(servo_dof_indices), dtype=np.float32)
    drone.set_dofs_kp(kp=kp, dofs_idx_local=servo_dof_indices)
    drone.set_dofs_kv(kv=kv, dofs_idx_local=servo_dof_indices)

    # Initial state (position + orientation + joints)
    initial_position = np.tile(
        np.concatenate(
            (
                controller.init_pos,
                euler_to_quat(controller.init_euler),
                controller.init_joint_position,
            )
        ),
        (BATCH_SIZE, 1),
    )

    initial_velocity = np.tile(
        np.concatenate(
            (
                controller.init_vel,
                controller.init_ang_vel,
                controller.init_joint_velocity,
            )
        ),
        (BATCH_SIZE, 1),
    )

    scene.sim.rigid_solver.set_qpos(initial_position)
    scene.sim.rigid_solver.set_dofs_velocity(initial_velocity)

    # Register drone in AeroSolver AFTER build (batch size known)
    scene.sim.aero_solver.add_target(drone, urdf_file=urdf_path)

    # Create debug model and attach to scene + drone
    model = DroneModel()
    model.attach_to_scene(scene, drone)
    model.enable_debug(True)  # turn on aero logging + arrows

    # Keyboard listener (start after build)
    listener = keyboard.Listener(
        on_press=controller.on_press, on_release=controller.on_release
    )
    listener.start()

    # Help
    print("\nWinged Drone Controls:")
    print("↑ / ↓   - Increase / decrease thrust (via AeroSolver)")
    print("← / →   - Asymmetric twist (roll)")
    print("space   - Increase symmetric twist")
    print("shift   - Decrease symmetric twist")
    print("w / s   - Increase / decrease symmetric sweep")
    print("q / e   - Elevator up / down (pitch)")
    print("a / d   - Rudder left / right (yaw)")
    print("ESC     - Quit\n")

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


if __name__ == "__main__":
    main()
