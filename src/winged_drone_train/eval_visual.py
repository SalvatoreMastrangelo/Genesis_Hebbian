#!/usr/bin/env python3
"""
Evaluation script for the winged drone in forest environments.

Features:
- Loads trained policy (PPO + ActorCriticTanh) from logs/<exp_name>/model_<ckpt>.pt
- Runs a single evaluation episode in the "eval forest" configuration
- Records:
    * Top-down multi-layer trajectory video with trees
    * Depth "heatmap" video (sectors as colored squares)
    * Camera + HUD overlay video (camera + depth + top-down + time series)
    * Camera + rewards video (camera + cumulative + component rewards)
- Saves all outputs into: logs/<exp_name>_eval/

All plotting / video generation is done with matplotlib + OpenCV (no GUI).
"""

import argparse
import copy
import math
import os
import pickle

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# Matplotlib non-GUI backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FFMpegWriter

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

# Local imports: environment + policy
try:
    from winged_drone_train.env import WingedDroneEnv, _apply_drone_profile_defaults
    from winged_drone_train.urdf_resolver import resolve_or_generate_urdf
    from winged_drone_train.defaults import default_mydrone_urdf_path
except ModuleNotFoundError:
    # Backward-compatible path when running this file directly.
    from env import WingedDroneEnv, _apply_drone_profile_defaults  # type: ignore
    from urdf_resolver import resolve_or_generate_urdf  # type: ignore
    from defaults import default_mydrone_urdf_path  # type: ignore
from winged_drone_train.rl.A2C_modified import ActorCriticTanh  # same as in train.py

import builtins
# Make the custom policy class discoverable by name ("ActorCriticTanh")
builtins.ActorCriticTanh = ActorCriticTanh

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

SUCCESS_TIME_SEC = 300.0  # required minimum flight time to count as "completed"
MINIMAL_PROGRESS_M = 250.0


# ---------------------------------------------------------------------------
# Aero debug helpers
# ---------------------------------------------------------------------------
def _disable_all_noise_except_obs(env: WingedDroneEnv) -> None:
    """Force-disable all noise sources except observation noise."""
    # Keep observation noise as configured (including depth noise if set).

    # SimpleDrone/Lisparrow config noise (mass/CoM + aero noise params)
    if isinstance(getattr(env, "_aero_config", None), dict):
        noise = dict(env._aero_config.get("noise", {}) or {})
        for key in ("sigma_mag", "sigma_dir", "sigma_param", "sigma_cp", "mass_shift", "com_shift"):
            if key in noise:
                noise[key] = 0.0
        env._aero_config["noise"] = noise

    # Aero solver noise flags and sigmas
    if hasattr(env, "aero_solver"):
        if hasattr(env.aero_solver, "_enable_noise"):
            env.aero_solver._enable_noise = False
        for key in ("noise_sigma_mag", "noise_sigma_dir", "noise_sigma_param", "noise_sigma_cp"):
            if hasattr(env.aero_solver, key):
                setattr(env.aero_solver, key, 0.0)

    # Drone model noise params (if present)
    if hasattr(env, "drone_model") and hasattr(env.drone_model, "noise_params"):
        env.drone_model.noise_params = {}


def _print_aero_geometry_debug(env: WingedDroneEnv) -> None:
    """Print aerodynamic surfaces parsed by SimpleDrone (kind/area/chord/span)."""
    solver = getattr(env, "aero_solver", None)
    if solver is None:
        return
    required = ("kind", "area", "chord", "span")
    if not all(hasattr(solver, name) for name in required):
        return

    try:
        dev = env.device
        kind = solver.kind.to_torch(device=dev).detach().cpu().numpy()
        area = solver.area.to_torch(device=dev).detach().cpu().numpy()
        chord = solver.chord.to_torch(device=dev).detach().cpu().numpy()
        span = solver.span.to_torch(device=dev).detach().cpu().numpy()
    except Exception as exc:
        print(f"[aero-debug] could not read surface fields: {exc}")
        return

    names = getattr(solver, "_aero_frames", [])
    kind_name = {0: "fuselage", 1: "wing", 2: "elevator", 3: "rudder", 4: "prop"}

    print("\n[aero-debug] surfaces used by simple_drone:")
    for i in range(len(kind)):
        frame = names[i] if i < len(names) else f"link_{i}"
        k = int(kind[i])
        print(
            f"  l={i:02d} frame={frame:>24s} kind={k}({kind_name.get(k, 'unknown')}) "
            f"S={float(area[i]):.6f} c={float(chord[i]):.6f} b={float(span[i]):.6f}"
        )


def _print_aero_step_debug(env: WingedDroneEnv, step_idx: int) -> None:
    """Print lift/drag summaries from per-surface debug fields (env 0)."""
    solver = getattr(env, "aero_solver", None)
    if solver is None:
        return
    required = ("kind", "lift_dbg", "drag_dbg", "Reynolds")
    if not all(hasattr(solver, name) for name in required):
        return

    try:
        dev = env.device
        kind = solver.kind.to_torch(device=dev).detach().cpu().numpy().astype(np.int32)
        lift = solver.lift_dbg.to_torch(device=dev)[0, : len(kind)].detach().cpu().numpy().astype(np.float32)
        drag = solver.drag_dbg.to_torch(device=dev)[0, : len(kind)].detach().cpu().numpy().astype(np.float32)
        reyn = solver.Reynolds.to_torch(device=dev)[0, : len(kind)].detach().cpu().numpy().astype(np.float32)
    except Exception as exc:
        print(f"[aero-debug] could not read per-step debug fields: {exc}")
        return

    wing = kind == 1
    elev = kind == 2
    rudd = kind == 3
    total_lift = float(np.nansum(lift[wing | elev]))
    total_drag = float(np.nansum(drag[wing | elev | rudd]))
    mean_re = float(np.nanmean(reyn[wing | elev | rudd])) if np.any(wing | elev | rudd) else float("nan")

    print(
        f"[aero-debug][step {step_idx:04d}] "
        f"L(w+e)={total_lift:.3f} N | D(w+e+r)={total_drag:.3f} N | Re_mean={mean_re:.1f}"
    )


def _extract_filtered_throttle(env: WingedDroneEnv) -> torch.Tensor:
    """Extract the actual filtered prop throttle from the AeroSolver."""
    if not hasattr(env, "aero_solver"):
        raise RuntimeError("AeroSolver not initialized in this environment.")

    cached_thr = getattr(env.aero_solver, "_thr_flt_buf", None)
    if torch.is_tensor(cached_thr) and cached_thr.shape[0] == env.num_envs:
        env._thr_flt_buf.copy_(cached_thr.to(device=env.device))
    else:
        thr_flt_field = getattr(env.aero_solver, "_thr_flt", None)
        if thr_flt_field is None:
            raise RuntimeError("AeroSolver._thr_flt not found (did you call add_target?).")
        env._thr_flt_buf.copy_(thr_flt_field.to_torch(device=env.device))

    torch.nan_to_num_(env._thr_flt_buf, nan=0.0, posinf=0.0, neginf=0.0)
    env._thr_flt_buf.clamp_(min=0.0, max=1.0)
    return env._thr_flt_buf


def _resolve_debug_surface_index(env: WingedDroneEnv, target_kind: int) -> Optional[int]:
    """Return the first aero-surface index whose solver kind matches ``target_kind``."""
    solver = getattr(env, "aero_solver", None)
    if solver is None or not hasattr(solver, "kind"):
        return None
    try:
        kind = solver.kind.to_torch(device=env.device).detach().view(-1).cpu()
    except Exception:
        return None
    matches = (kind == int(target_kind)).nonzero(as_tuple=False).flatten()
    if matches.numel() == 0:
        return None
    return int(matches[0].item())


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def compute_fov(
    pos_xy: np.ndarray,
    yaw: float,
    roll: float,
    fov_angle_nom: float = 40.0,
    fov_x_max: float = 30.0,
    n_points: int = 30,
) -> np.ndarray:
    """
    Compute a fan-shaped field-of-view polygon in the XY-plane.

    The fan (sector) is centered at pos_xy, aligned with yaw, and shrinks with roll:
    alpha_eff = fov_angle_nom * |cos(roll)|

    Args:
        pos_xy: (2,) array with [x, y] of the drone in world frame.
        yaw: heading angle [rad].
        roll: roll angle [rad].
        fov_angle_nom: nominal half-angle [deg].
        fov_x_max: maximum forward range of the FOV [m].
        n_points: number of points used to approximate the arc.

    Returns:
        Array of shape (n_points+1, 2) describing a polygon:
        [pos_xy, arc_point_0, ..., arc_point_{n_points-1}]
    """
    alpha_eff = max(fov_angle_nom * abs(math.cos(roll)), 2.0)
    a = math.radians(alpha_eff)

    # angles in body frame, then rotated by yaw
    thetas = np.linspace(-a, a, n_points) + yaw

    arc_points = np.stack(
        [fov_x_max * np.cos(thetas), fov_x_max * np.sin(thetas)],
        axis=1,
    )
    arc_points += pos_xy  # move arc endpoints to world position

    return np.vstack((pos_xy, arc_points))


# ---------------------------------------------------------------------------
# Video generation: top-down trajectory with forest
# ---------------------------------------------------------------------------

def create_topdown_video_multi(
    env: WingedDroneEnv,
    trajectories: List[Dict[str, np.ndarray]],
    save_path: str,
    dpi: int = 300,
    scale_px_per_m: float = 30.0,
) -> None:
    """
    Create a top-down video with one or more trajectories overlaid on the forest.

    The figure is tightly cropped to the forest bounding box and has:
      - minimal margins
      - a metric x axis for forward progress
      - reference lines at 50 m and 250 m

    Args:
        env: Environment instance (used to get forest layout and dt).
        trajectories: list of trajectory dicts (output of run_and_record).
        save_path: path to the output MP4.
        dpi: figure DPI for matplotlib.
        scale_px_per_m: conversion from meters to "pixels" for figure size.
    """
    x_world0, x_world1 = -40.0, env.env_cfg.get("x_upper", 400.0)
    x_offset = -x_world0
    x0, x1 = 0.0, 650.0
    y0 = env.env_cfg.get("y_lower", -60.0)
    y1 = env.env_cfg.get("y_upper", 60.0)
    W, H = (x1 - x0), (y1 - y0)
    fig_w, fig_h = W / scale_px_per_m, H / scale_px_per_m

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0.018, right=0.995, bottom=0.125, top=0.995)
    ax.set_aspect("equal", "box")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("x [m]", fontsize=16, labelpad=8)
    ax.set_yticks([])
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 650])
    ax.tick_params(axis="x", labelsize=13, width=1.2, length=5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_alpha(0.5)
    ax.spines["bottom"].set_linewidth(1.2)

    for x_ref in (50.0, 250.0):
        if x0 <= x_ref <= x1:
            ax.axvline(
                x_ref,
                color="limegreen",
                linestyle="--",
                linewidth=3.4,
                alpha=0.95,
                zorder=0,
            )

    # Forest obstacles (cylinders)
    cyl = env.cylinders_array.detach().cpu().numpy()
    obstacles = cyl[0] if cyl.ndim == 3 else cyl
    tree_r = 1.5
    for cx, cy, _ in obstacles:
        ax.add_patch(Circle((cx + x_offset, cy), tree_r, color="green", alpha=0.60, linewidth=0))

    # Trajectory plots
    N = len(trajectories)
    cmap = plt.get_cmap("tab10")
    lines, markers, polys = [], [], []
    for i in range(N):
        col = cmap(i % 10)
        lines.append(ax.plot([], [], lw=8.0, color=col)[0])
        markers.append(ax.plot([], [], "o", ms=10, color=col)[0])
        poly = Polygon(
            np.empty((0, 2)),
            closed=True,
            edgecolor=col,
            facecolor=col,
            alpha=0.60,
        )
        ax.add_patch(poly)
        polys.append(poly)

    # Common time grid
    t_max = max(tr["time_steps"][-1] for tr in trajectories)
    t_vals = np.arange(0.0, t_max, env.dt)

    writer = FFMpegWriter(fps=int(1.0 / env.dt), metadata=dict(artist="winged-drone"))
    with writer.saving(fig, save_path, dpi=dpi):
        for t in t_vals:
            for i, tr in enumerate(trajectories):
                ts = tr["time_steps"]
                if t > ts[-1]:
                    continue
                idx = max(np.searchsorted(ts, t) - 1, 0)
                pos = tr["positions"]
                pos_plot = pos.copy()
                pos_plot[:, 0] += x_offset
                yaw = tr["yaw"][idx]
                roll = tr["roll"][idx]

                lines[i].set_data(pos_plot[: idx + 1, 0], pos_plot[: idx + 1, 1])
                markers[i].set_data([pos_plot[idx, 0]], [pos_plot[idx, 1]])
                polys[i].set_xy(compute_fov(pos_plot[idx], yaw, roll))
            writer.grab_frame()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Video generation: depth map (sectors as colored squares)
# ---------------------------------------------------------------------------

def create_depth_video(
    depth_series: np.ndarray,
    max_distance: float,
    save_path: str,
    fps: int = 25,
    square_px: int = 18,
) -> None:
    """
    Render depth measurements as a simple 2D video.

    Each timestep is visualized as a horizontal row of S colored squares:
      - red means 0 m
      - green means max_distance
      - linear interpolation in between.

    Args:
        depth_series: array (T, S) with distances in meters.
        max_distance: maximum distance for normalization.
        save_path: path to the output MP4 file.
        fps: frames per second for the video.
        square_px: pixel size of each square.
    """
    import cv2

    depth = np.nan_to_num(
        depth_series.astype(np.float32),
        nan=0.0,
        posinf=max_distance,
        neginf=0.0,
    )

    T, S = depth.shape
    f = np.clip(depth / max_distance, 0.0, 1.0)
    # RGB: [1-f, f, 0]  => red→green
    color = np.stack([1.0 - f, f, np.zeros_like(f)], axis=-1)  # (T, S, 3)
    color = color[:, ::-1, :]  # optional: flip sectors for better viewing

    H = square_px
    W = S * square_px
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    for t in range(T):
        row = color[t]  # (S, 3)
        img = np.repeat(row[None, :, :], H, axis=0)  # (H, S, 3)
        img = np.repeat(img, square_px, axis=1)      # (H, S*square_px, 3)
        frame = (img[:, :, ::-1] * 255).astype(np.uint8)  # RGB->BGR
        vw.write(frame)

    vw.release()


# ---------------------------------------------------------------------------
# Video generation: overlay (camera + depth + top-down + HUD)
# ---------------------------------------------------------------------------

def create_overlay_video(
    cam_mp4: str,
    td_mp4: str,
    traj: Dict[str, np.ndarray],
    out_mp4: str,
    v_commanded: float = 12.0,
    dpi: int = 240,
    depth_mp4: Optional[str] = None,
) -> None:
    # Hebbian-only extra: optional per-step delta of the last-layer weights
    # (W_current - W_checkpoint), shape (T, num_actions, hidden_dim). If
    # present, a full-width heatmap is rendered at the bottom of the figure
    # with a fixed diverging scale centered at 0.
    weight_delta = traj.get("hebbian_weight_delta_history", None)
    has_heatmap = weight_delta is not None and weight_delta.size > 0
    """
    Create a composite video with:
      - Left column: camera view, depth squares, top-down render.
      - Right column: time-series plots (throttle + thrust, joints, velocity, alpha/beta).

    Args:
        cam_mp4: path to camera video.
        td_mp4: path to top-down video.
        traj: trajectory dict as returned by run_and_record.
        out_mp4: path to output overlay MP4.
        v_commanded: commanded forward speed (for HUD reference line).
        dpi: DPI for matplotlib.
        depth_mp4: optional path to depth video; if None, depth panel is blank.
    """
    import cv2
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    # Video sources
    cap_cam = cv2.VideoCapture(cam_mp4)
    cap_td = cv2.VideoCapture(td_mp4)
    cap_dp = cv2.VideoCapture(depth_mp4) if depth_mp4 else None

    fps_cam = cap_cam.get(cv2.CAP_PROP_FPS) or 25.0
    fps_td = cap_td.get(cv2.CAP_PROP_FPS) or fps_cam
    fps_dp = cap_dp.get(cv2.CAP_PROP_FPS) if cap_dp else fps_cam
    fps = fps_cam  # use camera FPS as reference

    nF_list = [
        int(cap_cam.get(cv2.CAP_PROP_FRAME_COUNT) or 1),
        int(cap_td.get(cv2.CAP_PROP_FRAME_COUNT) or 1),
    ]
    if cap_dp:
        nF_list.append(int(cap_dp.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
    nF = min(nF_list)
    dt = 1.0 / fps

    # Time-series data
    t_all = traj["time_steps"]
    throttle_series = traj.get("throttle", None)
    thrust_n_series = traj.get("thrust_n", None)
    # Backward compatibility with older trajectories where "thrust" actually
    # stored applied throttle fraction.
    if throttle_series is None and "thrust" in traj:
        throttle_series = traj["thrust"]
    if throttle_series is not None:
        throttle_sum = np.asarray(throttle_series).sum(axis=1)
    else:
        throttle_sum = np.zeros_like(t_all, dtype=np.float32)
    if thrust_n_series is not None:
        thrust_n_sum = np.asarray(thrust_n_series).sum(axis=1)
    else:
        thrust_n_sum = None
    max_thrust_total = traj.get("max_thrust", None)
    if max_thrust_total is not None:
        max_thrust_total = float(np.asarray(max_thrust_total).reshape(-1).sum())
    thrust_frac = None
    if (
        thrust_n_sum is not None
        and max_thrust_total is not None
        and np.isfinite(max_thrust_total)
        and max_thrust_total > 1e-6
    ):
        thrust_frac = np.clip(thrust_n_sum / max_thrust_total, 0.0, None)
    jp = traj["joint_positions"]
    joint_names = [str(name) for name in traj.get("joint_names", [])]
    n_joints = int(jp.shape[1]) if jp.ndim == 2 else 0
    vlin = traj["lin_vel"]
    alpha_deg = traj.get("alpha_deg", None)
    beta_deg = traj.get("beta_deg", None)
    vel_commanded = np.full_like(t_all, v_commanded, dtype=np.float32)

    # Layout: left column = videos, right column = plots; optional full-width
    # heatmap row appended at the bottom for Hebbian runs, plus an extra
    # time-series row below the heatmap for weight-drift / step-size curves.
    fig_h = 9 if not has_heatmap else 13
    fig = plt.figure(figsize=(16, fig_h), dpi=dpi)
    if has_heatmap:
        gs = GridSpec(
            nrows=9,
            ncols=3,
            width_ratios=[1.8, 0.06, 1.0],
            height_ratios=[1.3, 1.3, 1.3, 1.0, 1.0, 0.7, 0.2, 1.6, 1.0],
            wspace=0.08,
            hspace=0.30,
        )
    else:
        gs = GridSpec(
            nrows=7,
            ncols=3,
            width_ratios=[1.8, 0.06, 1.0],
            height_ratios=[1.3, 1.3, 1.3, 1.0, 1.0, 0.7, 0.2],
            wspace=0.08,
            hspace=0.30,
        )

    # Left column (col 0): camera spans rows 0-2, depth row 3, top-down rows 4-5
    ax_cam = fig.add_subplot(gs[0:3, 0])
    ax_cam.axis("off")
    ax_depth = fig.add_subplot(gs[3, 0])
    ax_depth.axis("off")
    ax_td = fig.add_subplot(gs[4:6, 0])
    ax_td.axis("off")

    # Right column (col 2): six time-series stacked one per row
    ax_T = fig.add_subplot(gs[0, 2])
    ax_J01 = fig.add_subplot(gs[1, 2])
    ax_J23 = fig.add_subplot(gs[2, 2])
    ax_J45 = fig.add_subplot(gs[3, 2])
    ax_VLIN = fig.add_subplot(gs[4, 2])
    ax_aero = fig.add_subplot(gs[5, 2])
    ts_axes = [ax_T, ax_J01, ax_J23, ax_J45, ax_VLIN, ax_aero]

    # Axis limits and grids
    ax_T.set_ylim(0.0, 1.0)
    ax_T.set_ylabel("Throttle / thrust ratio [-]")
    ax_T_right = ax_T.twinx()
    # Make the twinned axes coexist cleanly: same panel geometry, no opaque
    # patch on the right axis, and the throttle axis kept slightly in front.
    ax_T.set_zorder(2)
    ax_T_right.set_zorder(1)
    ax_T_right.patch.set_visible(False)
    thrust_n_limit = max_thrust_total
    if thrust_n_limit is not None and np.isfinite(thrust_n_limit) and thrust_n_limit > 0.0:
        ax_T_right.set_ylim(0.0, thrust_n_limit)
    elif thrust_n_sum is not None and thrust_n_sum.size:
        thrust_n_max = float(np.nanmax(thrust_n_sum))
        ax_T_right.set_ylim(0.0, max(1.0, thrust_n_max))
    else:
        ax_T_right.set_ylim(0.0, 1.0)
    ax_T_right.set_ylabel("Thrust [N]", labelpad=10)
    for ax in (ax_J01, ax_J23, ax_J45):
        ax.set_ylim(-0.3, 0.3)
    ax_VLIN.set_ylim(-2, v_commanded + 3)

    ax_aero.set_ylim(-10, 30)
    ax_aero.set_xlabel("t [s]")

    for ax in ts_axes:
        ax.grid(True, lw=0.35, alpha=0.35)
        ax.tick_params(labelsize=9)
    ax_T_right.grid(False)
    ax_T_right.tick_params(labelsize=9)

    for ax in ts_axes[:-1]:
        ax.tick_params(labelbottom=False)

    # Keep the full right column allocation, but shrink each plot a bit inside
    # that column so the right-side thrust label is not clipped.
    for ax in ts_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.92, pos.height])
    ax_T_right.set_position(ax_T.get_position())

    def _zeros_upto(i: int) -> np.ndarray:
        return np.zeros((i + 1,), dtype=np.float32)

    def _joint_series(idx: int) -> np.ndarray:
        if 0 <= idx < n_joints:
            return jp[: idx_frame + 1, idx]
        return _zeros_upto(idx_frame)

    # Time series lines
    lnThrCmd, = ax_T.plot([], [], lw=2.2, color="tab:blue", label="Throttle", zorder=4)
    lnThrustFrac = None
    if thrust_frac is not None:
        lnThrustFrac, = ax_T.plot(
            [],
            [],
            lw=2.2,
            color="tab:orange",
            linestyle="--",
            label="Thrust / T_max",
            zorder=5,
        )
    lnThrustN = None
    if thrust_n_sum is not None:
        lnThrustN, = ax_T_right.plot(
            [],
            [],
            lw=1.8,
            color="tab:red",
            linestyle=":",
            label="Thrust [N]",
            zorder=3,
        )
    is_lisparrow = (
        n_joints == 4
        and any("outer_wing" in name for name in joint_names)
        and any("elevator" in name for name in joint_names)
        and any("rudder" in name for name in joint_names)
    )
    if is_lisparrow:
        lnJ0, = ax_J01.plot([], [], lw=1.6, label="Sweep Left")
        lnJ1, = ax_J01.plot([], [], lw=1.6, label="Sweep Right")
        lnJ2, = ax_J23.plot([], [], lw=1.6, label="Elevator")
        lnJ3, = ax_J23.plot([], [], lw=1.6, label="Rudder")
        lnJ4, = ax_J45.plot([], [], lw=1.6, label="Sweep Mean")
        lnJ5, = ax_J45.plot([], [], lw=1.6, label="Sweep Diff")
    else:
        lnJ0, = ax_J01.plot([], [], lw=1.6, label="Sweep Mean")
        lnJ1, = ax_J01.plot([], [], lw=1.6, label="Sweep Diff")
        lnJ2, = ax_J23.plot([], [], lw=1.6, label="Twist Mean")
        lnJ3, = ax_J23.plot([], [], lw=1.6, label="Twist Diff")
        lnJ4, = ax_J45.plot([], [], lw=1.6, label="Elevator")
        lnJ5, = ax_J45.plot([], [], lw=1.6, label="Rudder")

    lnVx, = ax_VLIN.plot([], [], lw=1.6, label="vx")
    lnVy, = ax_VLIN.plot([], [], lw=1.6, label="vy")
    lnVz, = ax_VLIN.plot([], [], lw=1.6, label="vz")
    lnVCOM, = ax_VLIN.plot([], [], lw=1.6, label="vCOM")

    ln_alpha = ln_beta = None
    if alpha_deg is not None:
        ln_alpha, = ax_aero.plot([], [], lw=2.0, label="Alpha [deg]")
    if beta_deg is not None:
        ln_beta, = ax_aero.plot([], [], lw=2.0, label="Beta [deg]")

    for ax in ts_axes[1:]:
        ax.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=2, handlelength=1.8, columnspacing=1.0)
    thrust_handles = [lnThrCmd]
    if lnThrustFrac is not None:
        thrust_handles.append(lnThrustFrac)
    if lnThrustN is not None:
        thrust_handles.append(lnThrustN)
    thrust_labels = [h.get_label() for h in thrust_handles]
    ax_T.legend(thrust_handles, thrust_labels, fontsize=8.5, frameon=False, loc="upper left")

    # Optional Hebbian weight-delta heatmap (full width, bottom row)
    im_heatmap = None
    vmax_hm = 0.0
    if has_heatmap:
        vmax_hm = float(np.abs(weight_delta).max())
        if vmax_hm <= 0.0:
            vmax_hm = 1e-8  # avoid zero range; keeps colormap well-defined
        gs_hm = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[7, :], width_ratios=[1.0, 0.015], wspace=0.02
        )
        ax_heatmap = fig.add_subplot(gs_hm[0, 0])
        ax_cbar = fig.add_subplot(gs_hm[0, 1])
        num_actions, hidden_dim = weight_delta.shape[1], weight_delta.shape[2]
        im_heatmap = ax_heatmap.imshow(
            weight_delta[0],
            aspect="auto",
            cmap="seismic",
            vmin=-vmax_hm,
            vmax=vmax_hm,
            interpolation="nearest",
        )
        ax_heatmap.set_xlabel("Head neuron")
        ax_heatmap.set_ylabel("Actuator")
        actuator_names = [
            "throttle",
            "sweep_L",
            "sweep_R",
            "twist_L",
            "twist_R",
            "elevator",
            "rudder",
        ]
        if num_actions == len(actuator_names):
            ax_heatmap.set_yticks(range(num_actions))
            ax_heatmap.set_yticklabels(actuator_names, fontsize=8)
        ax_heatmap.set_title(
            f"Hebbian last-layer ΔW (max |ΔW| = {vmax_hm:.4f})",
            fontsize=10,
        )
        fig.colorbar(im_heatmap, cax=ax_cbar)

    # Hebbian-only dynamic curves: cumulative drift and per-step change of
    # the last-layer weights, summed over all (action, hidden) entries.
    # Uses a twin y-axis so the (small) per-step curve stays readable next
    # to the (much larger) cumulative-drift curve.
    ln_drift = ln_step = None
    ax_wstats = ax_wstats_step = None
    drift_sq = step_sq = None
    if has_heatmap:
        drift_sq = (weight_delta.astype(np.float64) ** 2).sum(axis=(1, 2))
        step_diff = np.diff(weight_delta.astype(np.float64), axis=0)
        step_sq = np.concatenate(
            [[0.0], (step_diff ** 2).sum(axis=(1, 2))]
        )

        gs_ws = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[8, :], width_ratios=[1.0, 0.015], wspace=0.02
        )
        ax_wstats = fig.add_subplot(gs_ws[0, 0])
        ax_wstats.grid(True, lw=0.3, alpha=0.4)
        ax_wstats.set_xlabel("t [s]")

        color_drift = "tab:blue"
        color_step = "tab:red"

        y_top_drift = float(drift_sq.max())
        if not np.isfinite(y_top_drift) or y_top_drift <= 0.0:
            y_top_drift = 1e-8
        ax_wstats.set_ylim(0.0, y_top_drift * 1.1)
        ax_wstats.set_ylabel("cumulative", color=color_drift)
        ax_wstats.tick_params(axis="y", labelcolor=color_drift)
        (ln_drift,) = ax_wstats.plot(
            [], [], lw=1.8, color=color_drift,
            label=r"$\Sigma (W - W_{ckpt})^2$",
        )

        ax_wstats_step = ax_wstats.twinx()
        y_top_step = float(step_sq.max())
        if not np.isfinite(y_top_step) or y_top_step <= 0.0:
            y_top_step = 1e-8
        ax_wstats_step.set_ylim(0.0, y_top_step * 1.1)
        ax_wstats_step.set_ylabel("per step", color=color_step)
        ax_wstats_step.tick_params(axis="y", labelcolor=color_step)
        (ln_step,) = ax_wstats_step.plot(
            [], [], lw=1.4, color=color_step,
            label=r"$\Sigma (W_t - W_{t-1})^2$",
        )

        ax_wstats.legend(
            handles=[ln_drift, ln_step],
            fontsize=9, frameon=False, loc="upper left", ncol=2,
        )

    # First frames
    okC, frm_cam = cap_cam.read()
    okT, frm_td = cap_td.read()
    if not (okC and okT):
        raise RuntimeError("Cannot read first frames from camera/top-down videos.")
    im_cam = ax_cam.imshow(cv2.cvtColor(frm_cam, cv2.COLOR_BGR2RGB))
    im_td = ax_td.imshow(cv2.cvtColor(frm_td, cv2.COLOR_BGR2RGB))

    im_depth = None
    if cap_dp:
        okD, frm_dp = cap_dp.read()
        if not okD:
            raise RuntimeError("Cannot read first frame from depth video.")
        im_depth = ax_depth.imshow(cv2.cvtColor(frm_dp, cv2.COLOR_BGR2RGB))

    writer = FFMpegWriter(fps=fps, metadata=dict(artist="winged-drone"))
    with writer.saving(fig, out_mp4, dpi=dpi):
        for k in range(nF):
            if k:
                rC, frm_cam = cap_cam.read()
                rT, frm_td = cap_td.read()
                if cap_dp:
                    rD, frm_dp = cap_dp.read()
                    if not (rC and rT and rD):
                        break
                    im_depth.set_data(cv2.cvtColor(frm_dp, cv2.COLOR_BGR2RGB))
                else:
                    if not (rC and rT):
                        break
                im_cam.set_data(cv2.cvtColor(frm_cam, cv2.COLOR_BGR2RGB))
                im_td.set_data(cv2.cvtColor(frm_td, cv2.COLOR_BGR2RGB))

            t_now = k * dt
            idx = max(np.searchsorted(t_all, t_now) - 1, 0)
            idx_frame = idx

            # Update time series
            lnThrCmd.set_data(t_all[: idx + 1], throttle_sum[: idx + 1])
            if lnThrustFrac is not None and thrust_frac is not None:
                lnThrustFrac.set_data(t_all[: idx + 1], thrust_frac[: idx + 1])
            if lnThrustN is not None and thrust_n_sum is not None:
                lnThrustN.set_data(t_all[: idx + 1], thrust_n_sum[: idx + 1])
            if is_lisparrow:
                left = _joint_series(0)
                right = _joint_series(1)
                elevator = _joint_series(2)
                rudder = _joint_series(3)
                lnJ0.set_data(t_all[: idx + 1], left)
                lnJ1.set_data(t_all[: idx + 1], right)
                lnJ2.set_data(t_all[: idx + 1], elevator)
                lnJ3.set_data(t_all[: idx + 1], rudder)
                lnJ4.set_data(t_all[: idx + 1], 0.5 * (left + right))
                lnJ5.set_data(t_all[: idx + 1], left - right)
            else:
                left = _joint_series(0)
                right = _joint_series(1)
                twist_left = _joint_series(2)
                twist_right = _joint_series(3)
                elev = _joint_series(4)
                rud = _joint_series(5)
                lnJ0.set_data(t_all[: idx + 1], 0.5 * (left - right))
                lnJ1.set_data(t_all[: idx + 1], left + right)
                lnJ2.set_data(t_all[: idx + 1], 0.5 * (twist_left + twist_right))
                lnJ3.set_data(t_all[: idx + 1], twist_left - twist_right)
                lnJ4.set_data(t_all[: idx + 1], elev)
                lnJ5.set_data(t_all[: idx + 1], rud)

            lnVx.set_data(t_all[: idx + 1], vlin[: idx + 1, 0])
            lnVy.set_data(t_all[: idx + 1], vlin[: idx + 1, 1])
            lnVz.set_data(t_all[: idx + 1], vlin[: idx + 1, 2])
            lnVCOM.set_data(t_all[: idx + 1], vel_commanded[: idx + 1])

            if ln_alpha is not None:
                ln_alpha.set_data(t_all[: idx + 1], alpha_deg[: idx + 1])
            if ln_beta is not None:
                ln_beta.set_data(t_all[: idx + 1], beta_deg[: idx + 1])

            for ax in ts_axes:
                ax.set_xlim(0, max(t_all[idx], 1e-6))

            if im_heatmap is not None:
                hm_idx = min(idx, weight_delta.shape[0] - 1)
                im_heatmap.set_data(weight_delta[hm_idx])

            if ln_drift is not None:
                ws_idx = min(idx, drift_sq.shape[0] - 1)
                ln_drift.set_data(t_all[: ws_idx + 1], drift_sq[: ws_idx + 1])
                ln_step.set_data(t_all[: ws_idx + 1], step_sq[: ws_idx + 1])
                ax_wstats.set_xlim(0, max(t_all[ws_idx], 1e-6))
                ax_wstats_step.set_xlim(0, max(t_all[ws_idx], 1e-6))

            writer.grab_frame()

    cap_cam.release()
    cap_td.release()
    if cap_dp:
        cap_dp.release()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Video generation: camera + rewards
# ---------------------------------------------------------------------------

def create_camera_rewards_video(
    cam_mp4: str,
    traj: Dict[str, np.ndarray],
    out_mp4: str,
    dpi: int = 240,
) -> None:
    """
    Create a video with:
      - top: camera image
      - bottom: large plot with total and component cumulative rewards.

    The legend is placed outside the plot on the right.
    """
    import cv2
    from matplotlib.animation import FFMpegWriter
    from matplotlib.gridspec import GridSpec

    # Reward data
    t_all = traj["time_steps"]
    R_tot = traj.get("reward_total", None)
    R_comp = traj.get("reward_components", None)
    R_names = traj.get("reward_names", None)

    # Camera video
    cap = cv2.VideoCapture(cam_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(t_all))
    dt = 1.0 / fps

    # Figure layout: 2 rows (bottom plot larger)
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    gs = GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[1.0, 1.8],
        hspace=0.08,
    )
    ax_cam = fig.add_subplot(gs[0, 0])
    ax_cam.axis("off")
    ax_rew = fig.add_subplot(gs[1, 0])

    # Extra space on the right for legend
    fig.subplots_adjust(right=0.82)

    # First camera frame
    ok, frm = cap.read()
    if not ok:
        cap.release()
        plt.close(fig)
        raise RuntimeError(f"Cannot read camera video: {cam_mp4}")
    im_cam = ax_cam.imshow(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Reward plot setup
    ax_rew.set_xlabel("t [s]")
    ax_rew.set_ylabel("reward (cumulative)")
    ax_rew.grid(True, lw=0.3, alpha=0.4)

    ln_tot = None
    ln_comp: List[plt.Line2D] = []
    if R_tot is not None:
        (ln_tot,) = ax_rew.plot([], [], lw=2.2, c="k", label="Total")
    if R_comp is not None and R_names is not None:
        cmap = plt.get_cmap("tab20")
        for i, nm in enumerate(list(R_names)):
            (ln,) = ax_rew.plot(
                [], [], lw=1.6, c=cmap(i % 20), label=str(nm)
            )
            ln_comp.append(ln)

    ax_rew.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=10,
        ncol=1,
    )

    # Writer
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="winged-drone"))
    with writer.saving(fig, out_mp4, dpi=dpi):
        nK = int(min(nF, math.ceil(t_all[-1] / dt)))
        for k in range(nK):
            if k > 0:
                ok, frm = cap.read()
                if not ok:
                    break
                im_cam.set_data(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            t_now = k * dt
            idx = max(np.searchsorted(t_all, t_now) - 1, 0)

            if ln_tot is not None:
                ln_tot.set_data(t_all[: idx + 1], R_tot[: idx + 1])
            for j, ln in enumerate(ln_comp):
                ln.set_data(t_all[: idx + 1], R_comp[: idx + 1, j])

            # Adaptive y-limits over all rewards up to current time
            vals = []
            if R_tot is not None:
                vals.append(R_tot[: idx + 1])
            if R_comp is not None:
                vals.append(R_comp[: idx + 1].ravel())

            if vals:
                allv = np.concatenate(vals)
                y_min = float(np.min(allv))
                y_max = float(np.max(allv))
                if y_max == y_min:
                    eps = (abs(y_max) + 1.0) * 1e-3
                    y_min -= eps
                    y_max += eps
                pad = 0.10 * abs(y_max)
                ax_rew.set_ylim(y_min - pad, y_max + pad)

            ax_rew.set_xlim(0, max(t_all[idx], 1e-6))
            writer.grab_frame()

    cap.release()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rollout helper: run one episode and record trajectory + stats
# ---------------------------------------------------------------------------

def run_and_record(env,
                   policy,
                   show_video: bool = False,
                   collect_video: bool = False,
                   video_cam_path: str = "camera_view.mp4",
                   debug_aero: bool = True,
                   hebbian_actor=None):
    """
    Roll out ONE evaluation episode (usually with num_envs = 1) and:
    - collect trajectory data (positions, velocities, joints, depth)
    - compute alpha / beta directly here
    - extract both applied throttle and physical thrust in Newtons
    - optionally record the drone camera video

    Returns:
        stats: dict of scalar statistics over all envs
        traj:  dict with time series (only if num_envs == 1, else None)
        cam_recording: bool, True if a camera video was actually recorded
    """
    device = env.device
    B = env.num_envs

    done = torch.zeros(B, dtype=torch.bool, device=device)

    # Episode-level accumulators (for stats over all envs)
    time_acc = torch.zeros(B, device=device)
    energy_acc = torch.zeros(B, device=device)
    energy_eff = torch.zeros(B, device=device)
    energy_propulsion = torch.zeros(B, device=device)
    energy_joints = torch.zeros(B, device=device)
    x_init = torch.zeros(B, device=device)
    x_progress = torch.zeros(B, device=device)
    x_eff = torch.zeros(B, device=device)
    z_at_min_progress = torch.full((B,), float("nan"), device=device)
    reached_min = torch.zeros(B, dtype=torch.bool, device=device)
    straight = torch.zeros(B, device=device)
    final_reason = [""] * B  # "collision", "wall_crash", "angle_limit", "success", "timeout", ...

    # Per-step trajectory buffers (used only when B == 1, i.e. single-env video)
    if B == 1:
        pos_b, yaw_b, roll_b, t_b = [], [], [], []
        throttle_b = []           # applied throttle (fraction 0..1)
        thrust_n_b = []           # physical thrust [N]
        joint_positions_b = []
        lin_vel_b = []
        alpha_deg_b, beta_deg_b = [], []
        reward_total_b = []
        reward_comp_b = []
        reward_names = None
        depth_b = []              # depth sectors at each time step
        # Hebbian-only: per-step delta of last-layer weights vs frozen checkpoint
        log_hebb_weights = hebbian_actor is not None and hasattr(hebbian_actor, "hebbian")
        weight_delta_b: List[np.ndarray] = []
        if log_hebb_weights:
            W_ckpt_np = hebbian_actor.hebbian.W_checkpoint.detach().cpu().numpy().astype(np.float32)

    # Camera video (if available and requested)
    if collect_video and getattr(env, "rec_cam", None) is not None and B == 1:
        env.start_video(filename=video_cam_path, fps=int(1.0 / env.dt))
        cam_recording = True
    else:
        cam_recording = False

    # Reset environment and record initial state
    obs, _ = env.reset()
    if debug_aero:
        _print_aero_geometry_debug(env)
    fuselage_dbg_idx = _resolve_debug_surface_index(env, target_kind=0)
    x_init[:] = env.base_pos[:, 0]
    t = torch.zeros(B, device=device)
    step_idx = 0

    # Main rollout loop
    while not done.all():
        # --------------------------------------------------------------
        #  Log state BEFORE stepping (synchronized with camera frames)
        # --------------------------------------------------------------
        if B == 1:
            # Position & attitude
            pos_b.append(env.base_pos[0, :2].detach().cpu().numpy())
            yaw_b.append(env.base_euler[0, 2].detach().cpu().item())
            roll_b.append(env.base_euler[0, 0].detach().cpu().item())
            t_b.append(t[0].item())

            # Joints
            joint_positions_b.append(env.joint_position[0].detach().cpu().numpy())

            # Linear velocity (world frame)
            v = env.base_lin_vel[0].detach().cpu().numpy()
            lin_vel_b.append(v.copy())

            alpha_deg_val = float(env.alpha.detach().cpu().item() * 180.0 / math.pi)
            beta_deg_val = float(env.beta.detach().cpu().item() * 180.0 / math.pi)
            if fuselage_dbg_idx is not None:
                try:
                    alpha_dbg = env.aero_solver.alpha_dbg.to_torch(device=device)
                    beta_dbg = env.aero_solver.beta_dbg.to_torch(device=device)
                    alpha_deg_val = float(alpha_dbg[0, fuselage_dbg_idx].detach().cpu().item() * 180.0 / math.pi)
                    beta_deg_val = float(beta_dbg[0, fuselage_dbg_idx].detach().cpu().item() * 180.0 / math.pi)
                except Exception:
                    pass
            alpha_deg_b.append(alpha_deg_val)
            beta_deg_b.append(beta_deg_val)

            # ----------------------------------------------------------
            #  Log propulsion state: applied throttle and physical thrust [N]
            # --------------------------------------------------------------
            throttle_val = 0.0
            try:
                throttle_filtered = _extract_filtered_throttle(env)
                if isinstance(throttle_filtered, torch.Tensor):
                    throttle_val = float(throttle_filtered[0].detach().cpu().reshape(-1)[0])
                else:
                    throttle_val = float(throttle_filtered)
            except Exception:
                throttle_val = 0.0

            thrust_n_val = 0.0
            try:
                thrust_n = env.extract_thrust()
                if isinstance(thrust_n, torch.Tensor):
                    thrust_n_val = float(thrust_n[0].detach().cpu().reshape(-1)[0])
                else:
                    thrust_n_val = float(thrust_n)
            except Exception:
                thrust_n_val = 0.0

            throttle_b.append(np.array([throttle_val], dtype=np.float32))
            thrust_n_b.append(np.array([thrust_n_val], dtype=np.float32))

            # Depth sectors (if present in the env)
            if getattr(env, "depth", None) is not None:
                depth_b.append(env.depth[0].detach().cpu().numpy())

        # --------------------------------------------------------------
        #  Policy inference and env step
        # --------------------------------------------------------------
        with torch.no_grad():
            act = policy(obs)
        # Capture post-update Hebbian weights (delta vs frozen checkpoint).
        if B == 1 and log_hebb_weights:
            W_now = hebbian_actor.hebbian.W[0].detach().cpu().numpy().astype(np.float32)
            weight_delta_b.append(W_now - W_ckpt_np)
        obs, _, term, _ = env.step(act)
        step_idx += 1

        if debug_aero and B == 1 and (step_idx <= 10 or step_idx % 50 == 0):
            _print_aero_step_debug(env, step_idx)

        # Reward components for first env (for reward videos)
        if B == 1:
            if reward_names is None:
                reward_names = env.reward_names
            comp_step = env.last_reward_components[0].detach().cpu().numpy()
            reward_total_b.append(float(comp_step.sum()))
            reward_comp_b.append(comp_step)

        # --------------------------------------------------------------
        #  Episode statistics over all envs
        # --------------------------------------------------------------
        terminated = term.to(torch.bool)
        nan_mask = torch.isnan(env.base_pos[:, 0])
        still_flying = (~done) & (~terminated) & (~nan_mask)

        time_acc[still_flying] += env.dt
        power = env.power_consumption()
        energy_acc[still_flying] += power[still_flying] * env.dt

        # If the env exposes cons_prop / cons_joint, accumulate them; otherwise skip
        if hasattr(env, "cons_prop") and hasattr(env, "cons_joint"):
            energy_propulsion[still_flying] += env.cons_prop[still_flying] * env.dt
            energy_joints[still_flying] += env.cons_joint[still_flying] * env.dt

        # Progress along +X and "straightness" (ratio of lateral to forward speed)
        x_progress[still_flying] = env.base_pos[still_flying, 0] - x_init[still_flying]
        straight[still_flying] += (
            torch.norm(env.base_lin_vel[still_flying, 1:], dim=1)
            / env.base_lin_vel[still_flying, 0].clamp_min(1e-6)
        )

        still_count = still_flying & (~reached_min)
        if still_count.any():
            energy_eff[still_count] += power[still_count] * env.dt
            x_eff[still_count] = x_progress[still_count]
            newly_reached = still_count & (x_progress >= MINIMAL_PROGRESS_M)
            if newly_reached.any():
                reached_min[newly_reached] = True
                x_eff[newly_reached] = MINIMAL_PROGRESS_M
                z_at_min_progress[newly_reached] = env.base_pos[newly_reached, 2]

        # --------------------------------------------------------------
        #  Final reason for envs that just terminated
        #  (we must infer it here, before any further rollouts)
        # --------------------------------------------------------------
        just_done = (~done) & terminated
        for j in just_done.nonzero(as_tuple=False).flatten().tolist():
            # Prefer episode counters from env.extras["episode"]
            ep_info = env.extras.get("episode", {})

            num_collision = float(ep_info.get("num_collision", 0.0))
            num_wall = float(ep_info.get("num_wall_crashed", 0.0))
            num_angle = float(ep_info.get("num_angle_crashed", 0.0))
            num_success = float(ep_info.get("num_success", 0.0))

            time_outs = env.extras.get("time_outs", None)
            is_timeout = False
            if time_outs is not None and isinstance(time_outs, torch.Tensor):
                is_timeout = bool(time_outs[j].item() > 0.5)

            if num_collision >= 1.0:
                final_reason[j] = "collision"
            elif num_wall >= 1.0:
                final_reason[j] = "wall_crash"
            elif num_angle >= 1.0:
                final_reason[j] = "angle_limit"
            elif num_success >= 1.0:
                final_reason[j] = "success"
            elif is_timeout:
                final_reason[j] = "timeout"
            else:
                final_reason[j] = "unknown"

        done |= terminated
        t += env.dt

    # Stop camera recording at the end of the rollout
    if cam_recording:
        env.stop_video()

    # ----------------------------------------------------------------------
    # Aggregate statistics
    # ----------------------------------------------------------------------
    success_mask = time_acc >= SUCCESS_TIME_SEC - 0.1
    n_completed_20s = int(success_mask.sum().item())

    mean_x = x_progress.mean().item()
    reached_z_mask = ~torch.isnan(z_at_min_progress)
    mean_z_at_progress_threshold = (
        z_at_min_progress[reached_z_mask].mean().item() if reached_z_mask.any() else float("nan")
    )
    mean_survival_time = time_acc.mean().item()
    mean_energy_total = energy_acc.mean().item()
    not_reached = ~reached_min
    if not_reached.any():
        energy_eff[not_reached] = energy_acc[not_reached]
        x_eff[not_reached] = x_progress[not_reached]
    mean_energy_per_m_x = (energy_eff / x_eff.clamp_min(1e-6)).mean().item()

    mean_en_propulsion = energy_propulsion.mean().item()
    mean_en_joints = energy_joints.mean().item()
    prop_to_en = (
        mean_en_propulsion / mean_energy_total if mean_energy_total > 0 else float("nan")
    )
    joints_to_en = (
        mean_en_joints / mean_energy_total if mean_energy_total > 0 else float("nan")
    )
    straightness = (straight / (time_acc / env.dt).clamp_min(1e-6)).mean().item()

    timeout_mask = torch.tensor(
        [fr == "timeout" for fr in final_reason],
        dtype=torch.bool,
        device=device,
    )
    mean_x_timeout = (
        x_progress[timeout_mask].mean().item() if timeout_mask.any() else float("nan")
    )

    n_obstacles_hit = sum(fr == "collision" for fr in final_reason)
    n_walls_hit = sum(fr == "wall_crash" for fr in final_reason)
    if n_obstacles_hit > 0:
        print(f"⚠️  {(n_obstacles_hit / B * 100):.2f}% episodes hit obstacles")
    if n_walls_hit > 0:
        print(f"⚠️  {(n_walls_hit / B * 100):.2f}% episodes hit lateral walls")
    if n_completed_20s > 0:
        print(f"✅  {(n_completed_20s / B * 100):.2f}% episodes survived {SUCCESS_TIME_SEC:.0f} s")

    # Print final reason for the first env for convenience in evaluation
    if B >= 1:
        print(f"[eval] final reason env 0: {final_reason[0]}")
    if debug_aero and B == 1:
        _print_aero_step_debug(env, step_idx)

    stats = dict(
        n_completed_20s=n_completed_20s,
        mean_x=mean_x,
        mean_z_at_progress_threshold=mean_z_at_progress_threshold,
        mean_survival_time=mean_survival_time,
        mean_energy_total=mean_energy_total,
        mean_energy_per_m_x=mean_energy_per_m_x,
        prop_to_en=prop_to_en,
        joints_to_en=joints_to_en,
        mean_x_timeout=mean_x_timeout,
        straightness=straightness,
    )

    # ----------------------------------------------------------------------
    # Build trajectory dict for B == 1 (used by video utilities)
    # ----------------------------------------------------------------------
    if B == 1:
        traj = dict(
            positions=np.vstack(pos_b),
            yaw=np.array(yaw_b, dtype=np.float32),
            roll=np.array(roll_b, dtype=np.float32),
            thrust=np.vstack(throttle_b).astype(np.float32),  # backward-compatible alias
            throttle=np.vstack(throttle_b).astype(np.float32),  # (T, 1) applied throttle fraction
            thrust_n=np.vstack(thrust_n_b).astype(np.float32),  # (T, 1) physical thrust [N]
            max_thrust=np.array(
                [float(env.aero_solver.max_thrust.to_torch(device=device)[0].detach().cpu().item())],
                dtype=np.float32,
            ),
            joint_positions=np.vstack(joint_positions_b).astype(np.float32),
            joint_names=np.array(list(getattr(env, "servo_joint_names", [])), dtype=object),
            lin_vel=np.vstack(lin_vel_b).astype(np.float32),
            time_steps=np.array(t_b, dtype=np.float32),
            end_reason=final_reason[0],
            reward_total=np.array(reward_total_b, dtype=np.float32),
            reward_components=np.vstack(reward_comp_b).astype(np.float32),
            reward_names=np.array(reward_names, dtype=object),
            alpha_deg=np.array(alpha_deg_b, dtype=np.float32),
            beta_deg=np.array(beta_deg_b, dtype=np.float32),
            depth_series=(
                np.vstack(depth_b).astype(np.float32) if len(depth_b) > 0 else None
            ),
            depth_max_distance=float(getattr(env, "MAX_DISTANCE", 30.0)),
            hebbian_weight_delta_history=(
                np.stack(weight_delta_b, axis=0) if log_hebb_weights and len(weight_delta_b) > 0 else None
            ),
        )
        return stats, traj, cam_recording

    return stats, None, cam_recording


# ---------------------------------------------------------------------------
# Utility: pretty-print stats
# ---------------------------------------------------------------------------

def pretty_print_stats(stats: Dict) -> None:
    """Print aggregated statistics in a compact, human-readable way."""
    print("\n=== Evaluation statistics (single episode batch) ===")
    print(f"- Completed {SUCCESS_TIME_SEC:.0f}s episodes: {stats['n_completed_20s']}")
    print(f"- Mean forward progress       : {stats['mean_x']:.1f} m")
    print(
        f"- Mean z at {MINIMAL_PROGRESS_M:.0f} m progress : "
        f"{stats['mean_z_at_progress_threshold']:.2f} m"
    )
    print(f"- Mean survival time          : {stats['mean_survival_time']:.1f} s")
    print(f"- Mean total energy           : {stats['mean_energy_total']:.1f} J")
    print(
        f"- Mean energy per meter (x)   : {stats['mean_energy_per_m_x']:.2f} J/m"
    )
    print(f"- Propulsion / total energy   : {stats['prop_to_en']:.3f}")
    print(f"- Joints / total energy       : {stats['joints_to_en']:.3f}")
    print(f"- Mean x at timeout           : {stats['mean_x_timeout']:.1f} m")
    print(f"- Mean straightness ratio     : {stats['straightness']:.3f}")
    print("===================================================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_eval_env(env_cfg, obs_cfg, reward_cfg, command_cfg, urdf_file, args, device) -> WingedDroneEnv:
    """Build and reset a single-env WingedDroneEnv with the standard eval overrides."""
    env_cfg_eval = dict(env_cfg)
    env_cfg_eval.update(
        dict(
            visualize_camera=False,
            visualize_target=False,
            max_visualize_FPS=25,
            unique_forests_eval=False,
            growing_forest=True,
            episode_length_s=SUCCESS_TIME_SEC,
            x_upper=600,
            forest_x_limit=600,
            tree_radius=env_cfg.get("tree_radius", 0.75),
            base_init_pos=env_cfg.get("base_init_pos", [-50.0, 0.0, 15.0]),
            aero_noise=False,
            aero_noise_sigma0=0.0,
            noise_sigma_param=0.0,
        )
    )
    command_cfg["eval_speed"] = args.vtgt

    if getattr(args, "dens_min", None) is not None:
        env_cfg_eval["dens_min"] = float(args.dens_min)
    if getattr(args, "dens_max", None) is not None:
        env_cfg_eval["dens_max"] = float(args.dens_max)

    obs_cfg_eval = dict(obs_cfg)

    print("\nEnvironment Configuration (eval):")
    print(env_cfg_eval)
    print("\nObservation Configuration (eval):")
    print(obs_cfg_eval)
    print("\nReward Configuration:")
    print(reward_cfg)
    print("\nCommand Configuration:")
    print(command_cfg)

    env = WingedDroneEnv(
        num_envs=1,
        env_cfg=env_cfg_eval,
        obs_cfg=obs_cfg_eval,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        urdf_file=urdf_file,
        show_viewer=args.visual,
        eval=True,
        device=str(device),
    )
    _disable_all_noise_except_obs(env)
    env.reset()
    return env


def _render_all_videos(env: WingedDroneEnv, eval_log_dir: str, cam_mp4: str, traj: Dict, cam_saved: bool) -> None:
    """Render the full set of evaluation videos (top-down, depth, overlay, rewards)."""
    topdown_mp4 = os.path.join(eval_log_dir, "eval_topdown.mp4")
    print("\nRendering top-down video …")
    create_topdown_video_multi(env, [traj], topdown_mp4)
    print(f"✅ Top-down video saved to: {topdown_mp4}")

    depth_video_path = None
    if traj.get("depth_series") is not None:
        depth_video_path = os.path.join(eval_log_dir, "depth_view.mp4")
        print("Rendering depth video …")
        create_depth_video(
            depth_series=traj["depth_series"],
            max_distance=traj.get("depth_max_distance", float(env.MAX_DISTANCE)),
            save_path=depth_video_path,
            fps=int(1.0 / env.dt),
        )
        print(f"✅ Depth video saved to: {depth_video_path}")
    else:
        print("No depth data recorded; skipping depth video.")

    if cam_saved:
        overlay_mp4 = os.path.join(eval_log_dir, "overlay.mp4")
        if hasattr(env, "commands") and env.commands.shape[1] >= 3:
            v_commanded = env.commands[0, 2].detach().cpu().item()
        else:
            v_commanded = env.commands[0, 0].detach().cpu().item()
        print("Rendering camera + HUD overlay video …")
        create_overlay_video(
            cam_mp4=cam_mp4,
            td_mp4=topdown_mp4,
            traj=traj,
            out_mp4=overlay_mp4,
            v_commanded=v_commanded,
            depth_mp4=depth_video_path,
        )
        print(f"✅ Overlay video saved to: {overlay_mp4}")

        camera_rewards_mp4 = os.path.join(eval_log_dir, "camera_rewards.mp4")
        print("Rendering camera + rewards video …")
        create_camera_rewards_video(
            cam_mp4=cam_mp4,
            traj=traj,
            out_mp4=camera_rewards_mp4,
            dpi=240,
        )
        print(f"✅ Camera + rewards video saved to: {camera_rewards_mp4}")
    else:
        print("Camera recording was not enabled or not supported; skipping overlay/videos based on camera.")


def _resolve_urdf(args) -> str:
    """Return the URDF path to use, generating a random one if --random-urdf is set."""
    if args.random_urdf:
        import tempfile
        from pathlib import Path as _Path
        from drone_making import UrdfMaker
        from morph_evolution.chromosome_drone import Chromosome_Drone
        norm_genome = np.random.uniform(0.0, 1.0, size=15).tolist()
        phys_genome = Chromosome_Drone.to_physical(norm_genome)
        urdf_dir = _Path(tempfile.mkdtemp(prefix="eval_random_urdf_"))
        urdf_path = _Path(UrdfMaker(phys_genome, out_dir=str(urdf_dir)).create_urdf()).resolve()
        print(f"[eval] Random morphology genome: {[f'{v:.3f}' for v in norm_genome]}")
        print(f"[eval] Generated random URDF: {urdf_path}")
        return str(urdf_path)
    return args.urdf if args.urdf is not None else str(default_mydrone_urdf_path())


def _run_hebbian(args) -> None:
    """Evaluate a Hebbian controller (WP2) with a given genome and render videos."""
    from pathlib import Path

    # Ensure src/ is importable for WP1/WP2 modules.
    _src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    import sys
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    from WP1.config import RunConfig
    from WP2.config import HebbianEvolutionConfig
    from WP2.frozen_actor import build_isolated_population_actor
    from WP2.utils import decode_hebbian_genes

    hebbian_run = Path(args.hebbian_run).resolve()
    genome_path = Path(args.genome).resolve()

    repro = hebbian_run / "reproducibility"
    wp2_cfg_path = repro / "config.yaml"
    wp1_cfg_path = repro / "wp1_config.yaml"
    wp1_ckpt_path = repro / "wp1_actor.pt"

    for p in (wp2_cfg_path, wp1_cfg_path, wp1_ckpt_path, genome_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    gs.init(logging_level="error", backend=gs.gpu)

    cfg = HebbianEvolutionConfig.from_yaml(wp2_cfg_path)
    # Repoint to the files saved inside the run's reproducibility folder
    # (the original paths in config.yaml are the ones used at training time and
    # may no longer exist or may be relative to a different working directory).
    cfg.checkpoint_path = str(wp1_ckpt_path)
    cfg.checkpoint_config_path = str(wp1_cfg_path)

    # Infer last-layer dims from checkpoint (robust to config drift)
    _ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    _sd = _ckpt.get("model_state_dict", _ckpt) if isinstance(_ckpt, dict) else _ckpt
    if "actor.4.weight" in _sd:
        cfg.hebbian.num_actions = _sd["actor.4.weight"].shape[0]
        cfg.hebbian.hidden_dim = _sd["actor.4.weight"].shape[1]
    del _ckpt, _sd

    # Build env from the WP1 config embedded with the run
    wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)
    env_cfg = wp1_cfg.to_env_cfg()
    obs_cfg = wp1_cfg.to_obs_cfg()
    reward_cfg = wp1_cfg.to_reward_cfg()
    command_cfg = wp1_cfg.to_command_cfg()
    # Hebbian evaluation always strips morphology genome from actor/critic obs.
    obs_cfg["add_genome_obs_actor"] = False
    obs_cfg["add_genome_obs_critic"] = False

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    urdf_file = _resolve_urdf(args)

    env = _build_eval_env(env_cfg, obs_cfg, reward_cfg, command_cfg, urdf_file, args, device)

    # Load and decode the genome
    genome_arr = np.load(genome_path)
    if genome_arr.ndim == 2:
        genome = genome_arr[0]
        print(f"[eval] Loaded genome row 0 from {genome_path.name} (shape {genome_arr.shape})")
    else:
        genome = genome_arr
        print(f"[eval] Loaded genome from {genome_path.name} (dim={genome.size})")

    hebb_part = list(np.clip(genome, 0.0, 1.0))
    rules = decode_hebbian_genes(
        hebb_part,
        cfg.hebbian,
        out_features=cfg.hebbian.num_actions,
        in_features=cfg.hebbian.hidden_dim,
    )

    actor = build_isolated_population_actor(
        checkpoint_path=cfg.checkpoint_path,
        wp1_cfg_path=cfg.checkpoint_config_path,
        hebbian_rules_per_individual=[rules],
        cfg=cfg,
        K=1,
        S=1,
        device=str(device),
        stochastic=cfg.evaluation.stochastic,
    )
    actor.reset_episode(device=device)

    policy = actor.act

    # Output directory: <hebbian_run>/eval_<genome_stem>/
    eval_log_dir = str(hebbian_run / f"eval_{genome_path.stem}")
    os.makedirs(eval_log_dir, exist_ok=True)

    print(
        f"\n[eval] Hebbian controller | stochastic={cfg.evaluation.stochastic} | "
        f"rules from {genome_path}"
    )
    print(f"[eval] Output directory: {eval_log_dir}")

    print("\nRunning evaluation episode …")
    cam_mp4 = os.path.join(eval_log_dir, "camera_view.mp4")
    stats, traj, cam_saved = run_and_record(
        env,
        policy,
        show_video=args.visual,
        collect_video=True,
        video_cam_path=cam_mp4,
        hebbian_actor=actor,
    )

    print("Final reason:", traj["end_reason"])
    pretty_print_stats(stats)

    if traj is None:
        print("No trajectory data recorded (num_envs > 1). Nothing to plot.")
        return

    _render_all_videos(env, eval_log_dir, cam_mp4, traj, cam_saved)
    print("\nEvaluation complete.")


def main() -> None:
    # add_help=False frees `-h` to be used as the short flag for --hebbian-run.
    parser = argparse.ArgumentParser(
        description="Evaluate a trained winged drone policy and generate videos.",
        add_help=False,
    )
    parser.add_argument(
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment (training log directory under ./logs). Required in PPO mode.",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=None,
        help="Checkpoint index to load (model_<ckpt>.pt). Required in PPO mode.",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Enable Genesis viewer while running the episode.",
    )
    parser.add_argument(
        "--vtgt",
        type=float,
        default=12.0,
        help="Commanded target Velocity.",
    )
    parser.add_argument(
        "--dens-min",
        dest="dens_min",
        type=float,
        default=None,
        help="Override forest density at x=x_lower [trees/m].",
    )
    parser.add_argument(
        "--dens-max",
        dest="dens_max",
        type=float,
        default=None,
        help="Override forest density at x=x_upper [trees/m].",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help=(
            "Base directory that contains the <exp_name> subfolder. "
            "Defaults to 'logs'. Override to point at tests/eval_visual or any other location."
        ),
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to the drone URDF file. Defaults to default_mydrone_urdf_path().",
    )
    parser.add_argument(
        "--random-urdf",
        action="store_true",
        help="Sample a random morphology genome and generate a fresh URDF for this evaluation.",
    )
    parser.add_argument(
        "-h",
        "--hebbian-run",
        dest="hebbian_run",
        type=str,
        default=None,
        help=(
            "Path to a WP2 (Hebbian) training directory, e.g. "
            "logs/runs_hebbian/2026-04-18_13-07-20_.... "
            "Enables Hebbian controller mode; requires --genome."
        ),
    )
    parser.add_argument(
        "--genome",
        type=str,
        default=None,
        help="Path to the genome .npy file to evaluate in Hebbian mode.",
    )
    args = parser.parse_args()

    # Hebbian mode takes precedence if -h/--hebbian-run is set.
    if args.hebbian_run is not None:
        if not args.genome:
            parser.error("--genome is required when -h/--hebbian-run is set.")
        _run_hebbian(args)
        return

    if args.exp_name is None or args.ckpt is None:
        parser.error("-e/--exp_name and --ckpt are required in PPO mode (or use -h for Hebbian mode).")

    # Initialize Genesis in high-performance mode (same backend as training)
    gs.init(logging_level="error", backend=gs.gpu)

    # Paths: training logs and evaluation outputs
    base_dir = args.log_dir if args.log_dir is not None else "logs"
    train_log_dir = os.path.join(base_dir, args.exp_name)
    eval_log_dir = os.path.join(base_dir, f"{args.exp_name}_eval")
    os.makedirs(eval_log_dir, exist_ok=True)

    # Load training configurations
    cfg_path = os.path.join(train_log_dir, "cfgs.pkl")
    yaml_path = os.path.join(train_log_dir, "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "rb") as f:
            cfg_data = pickle.load(f)
        if len(cfg_data) == 6:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, _ = cfg_data
        else:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfg_data
    elif os.path.exists(yaml_path):
        from WP1.config import RunConfig
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = RunConfig.from_yaml(yaml_path).to_legacy_cfgs()
    else:
        raise FileNotFoundError(f"Could not find cfgs.pkl or config.yaml in {train_log_dir}")

    urdf_file = _resolve_urdf(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = _build_eval_env(env_cfg, obs_cfg, reward_cfg, command_cfg, urdf_file, args, device)

    # Build runner and load policy
    runner_cfg = copy.deepcopy(train_cfg)
    runner = OnPolicyRunner(env, runner_cfg, train_log_dir, device=env.device)

    ckpt_path = os.path.join(train_log_dir, f"model_{args.ckpt}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\nLoading policy from: {ckpt_path}")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=env.device)

    # Run a single episode and record everything
    print("\nRunning evaluation episode …")
    cam_mp4 = os.path.join(eval_log_dir, "camera_view.mp4")
    stats, traj, cam_saved = run_and_record(
        env,
        policy,
        show_video=args.visual,
        collect_video=True,
        video_cam_path=cam_mp4,
    )

    print("Final reason:", traj["end_reason"])

    pretty_print_stats(stats)

    if traj is None:
        print("No trajectory data recorded (num_envs > 1). Nothing to plot.")
        return

    _render_all_videos(env, eval_log_dir, cam_mp4, traj, cam_saved)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
