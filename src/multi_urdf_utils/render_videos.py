#!/usr/bin/env python3
"""
Multi-URDF scene video rendering — overlay style (like WP1).
=============================================================

Renders composite overlay videos for each scene in a multi_urdf_utils benchmark
run. Each video combines the Genesis follow-camera view with a live top-down
trajectory plot and per-drone time-series panels (x-progress, altitude,
forward velocity).

Usage
-----
.. code-block:: bash

    python -m multi_urdf_utils.render_videos \\
        --cfg src/multi_urdf_utils/configs/benchmark.yaml \\
        --output logs/multi_urdf_videos

    # With specific parameters
    python -m multi_urdf_utils.render_videos \\
        --cfg src/multi_urdf_utils/configs/benchmark.yaml \\
        --output logs/multi_urdf_videos \\
        --max-steps 500
"""

from __future__ import annotations

import argparse
import builtins
import math
import os
os.environ["GS_PARA_LEVEL"] = "3"
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import fields as dc_fields

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter

import genesis as gs

from WP1.config import RunConfig
from WP2.config import HebbianConfig
from WP2.utils import seed_everything
from multi_urdf_utils.config import BenchmarkConfig
from multi_urdf_utils.multi_drone_env import MultiDroneEnv
from multi_urdf_utils.multi_drone_actor import MultiDroneActorManager, random_hebbian_rules
from general_policy.catalog import build_catalog


def _init_genesis() -> None:
    """Initialise Genesis (idempotent)."""
    if gs._initialized:
        return
    gs.init(logging_level="error", backend=gs.gpu)


def _load_hebbian_config(checkpoint_path: str, benchmark_cfg: BenchmarkConfig) -> HebbianConfig:
    """Load Hebbian config, inferring last-layer dimensions from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    hebb_cfg = HebbianConfig(
        enabled=benchmark_cfg.hebbian.enabled,
        eta=benchmark_cfg.hebbian.eta,
        w_max=benchmark_cfg.hebbian.w_max,
    )

    if "actor.4.weight" in sd:
        hebb_cfg.num_actions = sd["actor.4.weight"].shape[0]
        hebb_cfg.hidden_dim = sd["actor.4.weight"].shape[1]

    del ckpt, sd
    return hebb_cfg


def _collect_scene_trajectory(
    scene_idx: int,
    urdf_paths: List[str],
    wp1_cfg: RunConfig,
    hebb_cfg: HebbianConfig,
    benchmark_cfg: BenchmarkConfig,
    checkpoint_path: str,
    checkpoint_config_path: str,
    num_episodes: int = 1,
    max_steps: int = 500,
    video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a scene, record a follow-camera video, and return trajectory data."""
    N = len(urdf_paths)
    E = 1  # single env is sufficient for video rendering
    device = benchmark_cfg.device

    _init_genesis()
    env = MultiDroneEnv(
        urdf_paths=urdf_paths,
        num_envs=E,
        wp1_cfg=wp1_cfg,
        device=device,
        vmin=benchmark_cfg.env.vmin,
        vmax=benchmark_cfg.env.vmax,
        record=(video_path is not None),
    )

    use_hebbian = benchmark_cfg.hebbian.enabled

    if use_hebbian:
        hebbian_rules_list = [
            random_hebbian_rules(
                hebb_cfg,
                out_features=hebb_cfg.num_actions,
                in_features=hebb_cfg.hidden_dim,
                seed=benchmark_cfg.benchmark.seed + scene_idx * 100 + i,
                device=device,
            )
            for i in range(N)
        ]
    else:
        hebbian_rules_list = None

    actor_mgr = MultiDroneActorManager(
        D=N,
        num_envs_per_drone=E,
        checkpoint_path=checkpoint_path,
        checkpoint_config_path=checkpoint_config_path,
        hebbian_rules_list=hebbian_rules_list,
        hebb_cfg=hebb_cfg,
        stochastic=False,
        device=device,
        use_hebbian=use_hebbian,
    )

    all_positions = []      # (T, N, 3)
    all_orientations = []   # (T, N, 4)
    all_velocities = []     # (T, N, 3)
    all_throttle = []       # (T, N)
    all_joint_pos = []      # (T, N, num_servos_max)
    all_times = []
    all_depth = []          # (T, N) — depth measurements per drone

    num_servos_max = env.num_actions - 1  # first col is throttle

    # Track furthest drone across all episodes (by max x reached in env 0)
    best_x_per_drone = np.full(N, -np.inf)

    if video_path is not None:
        env.start_recording(video_path)

    forest_xy = None  # captured after first reset

    for ep in range(num_episodes):
        print(f"    Episode {ep+1}/{num_episodes}...")

        actor_mgr.reset_episode(E, device)
        obs, _ = env.reset()

        # Capture the forest assigned to drone 0, env 0 after reset
        if forest_xy is None:
            ds0 = env.drones[0]
            if hasattr(ds0, "cylinders_xy") and ds0.cylinders_xy is not None:
                forest_xy = ds0.cylinders_xy[0].detach().cpu().numpy()  # (C, 2)
        done = torch.zeros(N, E, dtype=torch.bool, device=torch.device(device))

        t = 0.0
        for step in range(max_steps):
            with torch.no_grad():
                actions = actor_mgr.act(obs)
            obs, rew, term, info = env.step(actions)
            done |= term

            # Positions: (N, 3) for env 0
            positions = torch.stack([ds.base_pos for ds in env.drones], dim=0)[:, 0, :].detach().cpu().numpy()
            orientations = torch.stack([ds.base_quat for ds in env.drones], dim=0)[:, 0, :].detach().cpu().numpy()
            velocities = torch.stack([ds.base_lin_vel for ds in env.drones], dim=0)[:, 0, :].detach().cpu().numpy()

            # last_actions: [throttle, servo_0, ..., servo_k] per drone, env 0
            throttle = np.array([ds.last_actions[0, 0].item() for ds in env.drones])  # (N,)
            jp = np.zeros((N, num_servos_max))
            for i, ds in enumerate(env.drones):
                ns = ds.num_servos
                if ns > 0:
                    jp[i, :ns] = ds.last_actions[0, 1:1+ns].detach().cpu().numpy()

            all_positions.append(positions)
            all_orientations.append(orientations)
            all_velocities.append(velocities)
            all_throttle.append(throttle)
            all_joint_pos.append(jp)
            all_times.append(t)

            # Capture depth readings per drone
            if hasattr(env, "depth_buf") and env.depth_buf is not None:
                # env.depth_buf shape: (D, E, NUM_SECTORS) where D=N, E=1 for rendering
                depth_readings = env.depth_buf[:N, 0, :].detach().cpu().numpy()  # (N, NUM_SECTORS)
                all_depth.append(depth_readings)
            else:
                all_depth.append(np.zeros((N, env.NUM_SECTORS)))  # fallback: no depth

            # Update best_x and pick follow target
            best_x_per_drone = np.maximum(best_x_per_drone, positions[:, 0])
            follow_idx = int(np.argmax(best_x_per_drone))

            if video_path is not None:
                env.update_follow_camera(follow_idx)
                env.render_frame()

            t += env.dt

            if done.all():
                print(f"      -> Episode finished after {step+1} steps")
                break

    if video_path is not None:
        env.stop_recording(fps=25)
        print(f"    Saved camera video: {video_path}")

    tree_radius = float(wp1_cfg.env.tree_radius) if hasattr(wp1_cfg.env, "tree_radius") else 1.5

    gs.destroy()

    # Depth: (T, N, num_sectors) or None
    depth_series = None
    depth_max_distance = 30.0
    if len(all_depth) > 0:
        depth_series = np.array(all_depth)  # (T, N, num_sectors)
        if hasattr(env, "MAX_DISTANCE"):
            depth_max_distance = float(env.MAX_DISTANCE)

    return {
        "positions": np.array(all_positions),       # (T, N, 3)
        "orientations": np.array(all_orientations), # (T, N, 4)
        "velocities": np.array(all_velocities),     # (T, N, 3)
        "throttle": np.array(all_throttle),         # (T, N)
        "joint_positions": np.array(all_joint_pos), # (T, N, num_servos_max)
        "times": np.array(all_times),               # (T,)
        "drone_names": [Path(p).stem for p in urdf_paths],
        "dt": env.dt,
        "episode_length_s": wp1_cfg.env.episode_length_s,
        "forest_xy": forest_xy,                     # (C, 2) or None
        "tree_radius": tree_radius,
        "depth_series": depth_series,               # (T, N, num_sectors) or None
        "depth_max_distance": depth_max_distance,
    }




def _create_depth_video(
    depth_series: Optional[np.ndarray],
    max_distance: float,
    save_path: str,
    fps: int = 25,
    square_px: int = 18,
) -> None:
    """Create a depth visualization video — red (close) to green (far).

    Args:
        depth_series: array (T, N, S) with depth measurements in meters, or None
                     T=timesteps, N=drones, S=sectors per drone
        max_distance: max distance for normalization
        save_path: output MP4 path
        fps: frames per second
        square_px: pixel size per sector square
    """
    import cv2

    # Handle missing depth data
    if depth_series is None or len(depth_series) == 0:
        # Create a blank placeholder video (1 frame, 8 sectors, 1 drone)
        T, N, S = 1, 1, 8
        depth_series = np.zeros((T, N, S), dtype=np.float32)

    T, N, S = depth_series.shape

    # Normalize: 0 -> red (1,0,0), max_distance -> green (0,1,0)
    depth_norm = np.nan_to_num(
        depth_series.astype(np.float32),
        nan=0.0,
        posinf=max_distance,
        neginf=0.0,
    )
    f = np.clip(depth_norm / max_distance, 0.0, 1.0)  # (T, N, S)
    # RGB: [1-f, f, 0] => red to green
    color = np.stack([1.0 - f, f, np.zeros_like(f)], axis=-1)  # (T, N, S, 3)

    # Stack drones horizontally: width = S*N sectors, height = 1 row per drone
    H = N * square_px
    W = S * square_px
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    for t in range(T):
        # For each drone, create its row of sectors
        frame_rows = []
        for n in range(N):
            row = color[t, n]  # (S, 3)
            # Expand to square_px height
            img = np.repeat(row[None, :, :], square_px, axis=0)  # (square_px, S, 3)
            # Expand each sector to square_px width
            img = np.repeat(img, square_px, axis=1)  # (square_px, S*square_px, 3)
            frame_rows.append(img)

        # Stack all drone rows vertically
        frame = np.vstack(frame_rows)  # (N*square_px, S*square_px, 3)
        frame_bgr = (frame[:, :, ::-1] * 255).astype(np.uint8)  # RGB->BGR
        vw.write(frame_bgr)

    vw.release()


def _create_overlay_video(
    cam_mp4: str,
    traj: Dict[str, Any],
    out_mp4: str,
    depth_mp4: Optional[str] = None,
    v_commanded: float = 12.0,
    dpi: int = 300,
) -> None:
    """Create a composite overlay video — identical layout to WP1's
    ``create_overlay_video`` in ``winged_drone_train/eval_visual.py``.

    Layout (7 rows x 3 columns):
      Left column  : camera view (rows 0-2), depth heatmap (row 3),
                     top-down trajectory (rows 4-5)
      Right column : Σ thrust (row 0), Sweep Mean/Diff (row 1),
                     Twist Mean/Diff (row 2), Elevator/Rudder (row 3),
                     vx/vy/vz/vCOM (row 4), Altitude (row 5)
    """
    import cv2

    # ------------------------------------------------------------------
    # Video sources
    # ------------------------------------------------------------------
    cap_cam = cv2.VideoCapture(cam_mp4)
    cap_dp = cv2.VideoCapture(depth_mp4) if depth_mp4 else None
    fps = cap_cam.get(cv2.CAP_PROP_FPS) or 25.0
    nF = int(cap_cam.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    if cap_dp:
        nF = min(nF, int(cap_dp.get(cv2.CAP_PROP_FRAME_COUNT) or nF))
    dt = 1.0 / fps

    # ------------------------------------------------------------------
    # Trajectory data
    # ------------------------------------------------------------------
    t_all = traj["times"]               # (T,)
    pos = traj["positions"]             # (T, N, 3)
    vel = traj["velocities"]            # (T, N, 3)
    thr = traj["throttle"]              # (T, N)
    jp  = traj["joint_positions"]       # (T, N, num_servos_max)
    names = traj["drone_names"]
    forest_xy = traj.get("forest_xy")   # (C, 2) or None
    tree_r = traj.get("tree_radius", 3.0)
    N = pos.shape[1]
    num_servos = jp.shape[2]
    nF = min(nF, len(t_all))
    vel_commanded = np.full_like(t_all, v_commanded, dtype=np.float32)

    # Per-drone colours — same cmap as WP1 top-down
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(N)]

    # Additional color palettes for series differentiation
    cmap_alt = plt.get_cmap("Set2")
    colors_alt = [cmap_alt(i % 8) for i in range(N)]
    cmap_vel = plt.get_cmap("Set1")
    colors_vel = [cmap_vel(i % 9) for i in range(N)]

    # ------------------------------------------------------------------
    # Figure — identical GridSpec to WP1
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    gs = GridSpec(
        nrows=7,
        ncols=3,
        width_ratios=[1.8, 0.06, 1.0],
        height_ratios=[1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 0.2],
        wspace=0.08,
        hspace=0.30,
    )

    # Left column — same slots as WP1
    ax_cam   = fig.add_subplot(gs[0:3, 0]);  ax_cam.axis("off")
    ax_depth = fig.add_subplot(gs[3:4, 0]);  ax_depth.axis("off")   # blank
    ax_td    = fig.add_subplot(gs[4:6, 0]);  ax_td.axis("off")

    # Right column — same 6 panels as WP1
    ax_T    = fig.add_subplot(gs[0, 2])
    ax_J01  = fig.add_subplot(gs[1, 2])
    ax_J23  = fig.add_subplot(gs[2, 2])
    ax_J45  = fig.add_subplot(gs[3, 2])
    ax_VLIN = fig.add_subplot(gs[4, 2])
    ax_aero = fig.add_subplot(gs[5, 2])
    ts_axes = [ax_T, ax_J01, ax_J23, ax_J45, ax_VLIN, ax_aero]

    # ------------------------------------------------------------------
    # Top-down axis — same style as WP1 create_topdown_video_multi
    # ------------------------------------------------------------------
    td_x0, td_x1 = -40.0, float(pos[:, :, 0].max()) + 40.0
    td_y0, td_y1 = -60.0, 60.0

    # Expand bounds to include forest
    if forest_xy is not None and len(forest_xy) > 0:
        fd_x_min, fd_x_max = float(forest_xy[:, 0].min()), float(forest_xy[:, 0].max())
        fd_y_min, fd_y_max = float(forest_xy[:, 1].min()), float(forest_xy[:, 1].max())
        td_x0 = min(td_x0, fd_x_min - 10)
        td_x1 = max(td_x1, fd_x_max + 10)
        td_y0 = min(td_y0, fd_y_min - 10)
        td_y1 = max(td_y1, fd_y_max + 10)

    ax_td.set_xlim(td_x0, td_x1)
    ax_td.set_ylim(td_y0, td_y1)
    ax_td.set_aspect("equal", "box")

    # Forest — green circles, same as WP1
    if forest_xy is not None:
        for cx, cy in forest_xy:
            ax_td.add_patch(Circle((cx, cy), tree_r,
                                   color="green", alpha=0.60, linewidth=3, fill=True))

    # Per-drone trajectory lines & markers — same lw/ms as WP1
    td_lines, td_markers = [], []
    for i in range(N):
        td_lines.append(ax_td.plot([], [], lw=8.0, color=colors[i])[0])
        td_markers.append(ax_td.plot([], [], "o", ms=10, color=colors[i])[0])

    # ------------------------------------------------------------------
    # Compute dynamic y-limits based on actual data
    # ------------------------------------------------------------------
    def compute_lim(data, pad_frac=0.1, min_range=0.1):
        """Compute y-limits with padding, ensuring min_range spread."""
        d_min, d_max = float(np.min(data)), float(np.max(data))
        d_range = max(d_max - d_min, min_range)
        pad = d_range * pad_frac
        return d_min - pad, d_max + pad

    # Thrust
    thr_min, thr_max = compute_lim(thr, pad_frac=0.1, min_range=0.5)
    ax_T.set_ylim(thr_min, thr_max)

    # Joint positions: compute mean/diff for sweep, twist, elevator/rudder
    if num_servos >= 2:
        sweep_mean = (jp[:, :, 0] + jp[:, :, 1]) / 2
        sweep_diff = (jp[:, :, 0] - jp[:, :, 1]) / 2
        j01_data = np.concatenate([sweep_mean, sweep_diff], axis=None)
        j01_min, j01_max = compute_lim(j01_data, pad_frac=0.1, min_range=0.05)
    else:
        j01_min, j01_max = -0.3, 0.3

    if num_servos >= 4:
        twist_mean = (jp[:, :, 2] + jp[:, :, 3]) / 2
        twist_diff = (jp[:, :, 2] - jp[:, :, 3]) / 2
        j23_data = np.concatenate([twist_mean, twist_diff], axis=None)
        j23_min, j23_max = compute_lim(j23_data, pad_frac=0.1, min_range=0.05)
    else:
        j23_min, j23_max = -0.3, 0.3

    if num_servos >= 5:
        elev_data = jp[:, :, 4]
        if num_servos >= 6:
            rudd_data = jp[:, :, 5]
            j45_data = np.concatenate([elev_data, rudd_data], axis=None)
        else:
            j45_data = elev_data
        j45_min, j45_max = compute_lim(j45_data, pad_frac=0.1, min_range=0.05)
    else:
        j45_min, j45_max = -0.3, 0.3

    ax_J01.set_ylim(j01_min, j01_max)
    ax_J23.set_ylim(j23_min, j23_max)
    ax_J45.set_ylim(j45_min, j45_max)

    # Linear velocity
    vel_data = vel[:, :, :3]  # vx, vy, vz
    vx_min, vx_max = compute_lim(vel_data[:, :, 0], pad_frac=0.1, min_range=1.0)
    vy_min, vy_max = compute_lim(vel_data[:, :, 1], pad_frac=0.1, min_range=1.0)
    vz_min, vz_max = compute_lim(vel_data[:, :, 2], pad_frac=0.1, min_range=1.0)
    all_vel = np.concatenate([vel_data, vel_commanded[:, np.newaxis]], axis=None)
    v_min, v_max = compute_lim(all_vel, pad_frac=0.1, min_range=2.0)
    ax_VLIN.set_ylim(v_min, v_max)
    ax_VLIN.grid(True, lw=0.3, alpha=0.4)
    ax_VLIN.set_xlabel("t [s]")

    # Altitude
    alt_data = pos[:, :, 2]
    alt_min, alt_max = compute_lim(alt_data, pad_frac=0.1, min_range=5.0)
    ax_aero.set_ylim(alt_min, alt_max)
    ax_aero.grid(True, lw=0.3, alpha=0.4)
    ax_aero.set_xlabel("t [s]")

    for ax in (ax_T, ax_J01, ax_J23, ax_J45):
        ax.grid(True, lw=0.3, alpha=0.4)

    # ------------------------------------------------------------------
    # Time-series lines — per drone, same labels/lw as WP1
    # ------------------------------------------------------------------
    # Row 0 — Σ thrust  (WP1 sums 4 motors; here we plot throttle per drone)
    lnT = [ax_T.plot([], [], lw=2.0, color=colors[i],
                      label=names[i])[0] for i in range(N)]

    # Row 1 — Sweep Mean / Diff
    lnJ0 = [ax_J01.plot([], [], lw=1.6, color=colors[i])[0] for i in range(N)]
    lnJ1 = [ax_J01.plot([], [], lw=1.6, color=colors_alt[i])[0] for i in range(N)]
    ax_J01.plot([], [], lw=1.6, color="grey",  label="Sweep Mean")
    ax_J01.plot([], [], lw=1.6, color="lightgrey", label="Sweep Diff")

    # Row 2 — Twist Mean / Diff
    lnJ2 = [ax_J23.plot([], [], lw=1.6, color=colors[i])[0] for i in range(N)]
    lnJ3 = [ax_J23.plot([], [], lw=1.6, color=colors_alt[i])[0] for i in range(N)]
    ax_J23.plot([], [], lw=1.6, color="grey",  label="Twist Mean")
    ax_J23.plot([], [], lw=1.6, color="lightgrey", label="Twist Diff")

    # Row 3 — Elevator / Rudder
    lnJ4 = [ax_J45.plot([], [], lw=1.6, color=colors[i])[0] for i in range(N)]
    lnJ5 = [ax_J45.plot([], [], lw=1.6, color=colors_alt[i])[0] for i in range(N)]
    ax_J45.plot([], [], lw=1.6, color="grey",  label="Elevator")
    ax_J45.plot([], [], lw=1.6, color="lightgrey", label="Rudder")

    # Row 4 — Linear velocity (vx, vy, vz, vCOM)
    lnVx = [ax_VLIN.plot([], [], lw=1.6, color=colors[i])[0] for i in range(N)]
    lnVy = [ax_VLIN.plot([], [], lw=1.6, color=colors_alt[i])[0] for i in range(N)]
    lnVz = [ax_VLIN.plot([], [], lw=1.6, color=colors_vel[i])[0] for i in range(N)]
    lnVCOM, = ax_VLIN.plot([], [], lw=1.6, color="orange", label="vCOM")
    ax_VLIN.plot([], [], lw=1.6, color="grey",  label="vx")
    ax_VLIN.plot([], [], lw=1.6, color="lightgrey", label="vy")
    ax_VLIN.plot([], [], lw=1.6, color="darkgrey",  label="vz")

    # Row 5 — Altitude (replaces Alpha/Beta which is unavailable)
    lnAlt = [ax_aero.plot([], [], lw=2.0, color=colors[i],
                           label=names[i])[0] for i in range(N)]

    # Legends — same as WP1
    for ax in ts_axes:
        ax.legend(fontsize=9, frameon=False, loc="upper right", ncol=2)

    # ------------------------------------------------------------------
    # First frames
    # ------------------------------------------------------------------
    okC, frm_cam = cap_cam.read()
    if not okC:
        cap_cam.release()
        if cap_dp:
            cap_dp.release()
        plt.close(fig)
        raise RuntimeError(f"Cannot read camera video: {cam_mp4}")
    im_cam = ax_cam.imshow(cv2.cvtColor(frm_cam, cv2.COLOR_BGR2RGB))

    im_depth = None
    if cap_dp:
        okD, frm_dp = cap_dp.read()
        if not okD:
            cap_cam.release()
            cap_dp.release()
            plt.close(fig)
            raise RuntimeError(f"Cannot read first depth frame: {depth_mp4}")
        im_depth = ax_depth.imshow(cv2.cvtColor(frm_dp, cv2.COLOR_BGR2RGB))

    # ------------------------------------------------------------------
    # Render loop — identical flow to WP1
    # ------------------------------------------------------------------
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="winged-drone"))
    with writer.saving(fig, out_mp4, dpi=dpi):
        for k in range(nF):
            if k:
                rC, frm_cam = cap_cam.read()
                if not rC:
                    break
                im_cam.set_data(cv2.cvtColor(frm_cam, cv2.COLOR_BGR2RGB))

                if cap_dp:
                    rD, frm_dp = cap_dp.read()
                    if not rD:
                        break
                    im_depth.set_data(cv2.cvtColor(frm_dp, cv2.COLOR_BGR2RGB))

            t_now = k * dt
            idx = max(np.searchsorted(t_all, t_now) - 1, 0)

            # Top-down
            for i in range(N):
                td_lines[i].set_data(pos[: idx + 1, i, 0],
                                     pos[: idx + 1, i, 1])
                td_markers[i].set_data([pos[idx, i, 0]],
                                       [pos[idx, i, 1]])

            # Time series
            for i in range(N):
                lnT[i].set_data(t_all[: idx + 1], thr[: idx + 1, i])

                if num_servos >= 2:
                    lnJ0[i].set_data(t_all[: idx + 1],
                                     (jp[: idx + 1, i, 0] - jp[: idx + 1, i, 1]) / 2)
                    lnJ1[i].set_data(t_all[: idx + 1],
                                     (jp[: idx + 1, i, 0] + jp[: idx + 1, i, 1]))
                if num_servos >= 4:
                    lnJ2[i].set_data(t_all[: idx + 1],
                                     (jp[: idx + 1, i, 2] + jp[: idx + 1, i, 3]) / 2)
                    lnJ3[i].set_data(t_all[: idx + 1],
                                     (jp[: idx + 1, i, 2] - jp[: idx + 1, i, 3]))
                if num_servos >= 5:
                    lnJ4[i].set_data(t_all[: idx + 1], jp[: idx + 1, i, 4])
                if num_servos >= 6:
                    lnJ5[i].set_data(t_all[: idx + 1], jp[: idx + 1, i, 5])

                lnVx[i].set_data(t_all[: idx + 1], vel[: idx + 1, i, 0])
                lnVy[i].set_data(t_all[: idx + 1], vel[: idx + 1, i, 1])
                lnVz[i].set_data(t_all[: idx + 1], vel[: idx + 1, i, 2])

                lnAlt[i].set_data(t_all[: idx + 1], pos[: idx + 1, i, 2])

            lnVCOM.set_data(t_all[: idx + 1], vel_commanded[: idx + 1])

            for ax in ts_axes:
                ax.set_xlim(0, max(t_all[idx], 1e-6))

            writer.grab_frame()

    cap_cam.release()
    if cap_dp:
        cap_dp.release()
    plt.close(fig)


def render_all_scenes(
    benchmark_cfg: BenchmarkConfig,
    output_dir: Optional[Path] = None,
    max_steps: Optional[int] = None,
) -> None:
    """Render videos for all scenes in the benchmark config.

    Parameters
    ----------
    benchmark_cfg : BenchmarkConfig
        The benchmark configuration.
    output_dir : Path, optional
        Where to save videos. Defaults to logs/multi_urdf_videos.
    max_steps : int, optional
        Overrides max_steps in config.
    """
    if output_dir is None:
        output_dir = Path("logs/multi_urdf_videos")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = 1  # single episode per scene is sufficient for video
    max_steps = max_steps or benchmark_cfg.benchmark.max_steps

    # Seed
    seed_everything(benchmark_cfg.benchmark.seed)

    # Load checkpoint config
    wp1_cfg = RunConfig.from_yaml(benchmark_cfg.checkpoint.config_path)
    wp1_cfg.env.episode_length_s = benchmark_cfg.env.episode_length_s
    wp1_cfg.env.dens_min = benchmark_cfg.env.forest_density_min
    wp1_cfg.env.dens_max = benchmark_cfg.env.forest_density_max
    wp1_cfg.env.growing_forest = benchmark_cfg.env.growing_forest
    wp1_cfg.env.tree_radius = benchmark_cfg.env.tree_radius
    wp1_cfg.env.tree_height = benchmark_cfg.env.tree_height
    wp1_cfg.env.base_init_pos = benchmark_cfg.env.base_init_pos
    wp1_cfg.env.base_init_quat = benchmark_cfg.env.base_init_quat
    # Apply aero noise config from benchmark (may differ from training config)
    wp1_cfg.env.aero_noise = benchmark_cfg.env.aero_noise

    # Load Hebbian config
    hebb_cfg = _load_hebbian_config(benchmark_cfg.checkpoint.model_path, benchmark_cfg)

    # Build URDF catalog
    N = benchmark_cfg.benchmark.N
    S = benchmark_cfg.benchmark.S
    D = N * S

    print(f"\n{'='*70}")
    print(f"  Multi-URDF Scene Video Rendering")
    print(f"{'='*70}")
    print(f"  N={N} URDFs/scene  S={S} scenes")
    print(f"  Episodes={num_episodes}  max_steps={max_steps}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    print("[Phase 1] Generating URDFs...")
    _init_genesis()

    catalog_dir = Path(benchmark_cfg.catalog.catalog_dir)
    urdf_paths = build_catalog(
        catalog_dir=catalog_dir,
        n=D,
        seed=benchmark_cfg.catalog.urdf_seed,
        include_standard_mydrone=True,
    )
    urdf_paths_str = [str(p) for p in urdf_paths]
    while len(urdf_paths_str) < D:
        urdf_paths_str.append(urdf_paths_str[len(urdf_paths_str) % len(urdf_paths)])

    gs.destroy()

    # Render each scene
    for scene_idx in range(S):
        scene_urdf_paths = urdf_paths_str[scene_idx * N : (scene_idx + 1) * N]

        print(f"\n[Scene {scene_idx + 1}/{S}]")
        print(f"  URDFs: {[Path(p).stem for p in scene_urdf_paths]}")

        cam_path = str(output_dir / f"scene_{scene_idx:02d}_camera.mp4")
        depth_path = str(output_dir / f"scene_{scene_idx:02d}_depth.mp4")
        overlay_path = str(output_dir / f"scene_{scene_idx:02d}_overlay.mp4")
        print(f"  Recording follow-camera video...")
        traj = _collect_scene_trajectory(
            scene_idx=scene_idx,
            urdf_paths=scene_urdf_paths,
            wp1_cfg=wp1_cfg,
            hebb_cfg=hebb_cfg,
            benchmark_cfg=benchmark_cfg,
            checkpoint_path=benchmark_cfg.checkpoint.model_path,
            checkpoint_config_path=benchmark_cfg.checkpoint.config_path,
            num_episodes=num_episodes,
            max_steps=max_steps,
            video_path=cam_path,
        )

        # Generate depth video (always)
        print(f"  Creating depth video...")
        _create_depth_video(
            depth_series=traj.get("depth_series"),
            max_distance=traj.get("depth_max_distance", 30.0),
            save_path=depth_path,
            fps=25,
        )

        print(f"  Creating overlay video...")
        _create_overlay_video(
            cam_mp4=cam_path,
            traj=traj,
            out_mp4=overlay_path,
            depth_mp4=depth_path,
            v_commanded=float(benchmark_cfg.env.vmin + benchmark_cfg.env.vmax) / 2.0,
        )
        print(f"    Saved overlay: {overlay_path}")

    print(f"\n✅ All videos saved to: {output_dir}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Render videos for multi_urdf_utils benchmark scenes."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="src/multi_urdf_utils/configs/benchmark.yaml",
        help="Path to benchmark config YAML.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/multi_urdf_videos",
        help="Output directory for videos.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps from config.",
    )
    parser.add_argument(
        "--scenes",
        type=int,
        nargs="*",
        default=None,
        help="Only render specific scenes (0-indexed). If not specified, renders all scenes.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output.",
    )

    args, remaining = parser.parse_known_args()

    # Load config
    try:
        cfg = BenchmarkConfig.from_yaml(args.cfg)
        cfg.apply_cli_overrides(remaining)
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load config: {e}")
        sys.exit(1)

    # Validate
    if not cfg.checkpoint.model_path or not Path(cfg.checkpoint.model_path).is_file():
        print(f"[ERROR] checkpoint.model_path not found: {cfg.checkpoint.model_path}")
        sys.exit(1)
    if not cfg.checkpoint.config_path or not Path(cfg.checkpoint.config_path).is_file():
        print(f"[ERROR] checkpoint.config_path not found: {cfg.checkpoint.config_path}")
        sys.exit(1)

    # Render videos
    try:
        render_all_scenes(
            benchmark_cfg=cfg,
            output_dir=args.output,
            max_steps=args.max_steps,
        )
    except Exception as e:
        print(f"[ERROR] Video rendering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
