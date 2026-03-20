#!/usr/bin/env python3
"""
Multi-URDF scene video rendering.
==================================

Renders videos for each scene in a multi_urdf_utils benchmark run, showing all drones
flying together from a far-away top-down perspective.

Usage
-----
.. code-block:: bash

    python -m WP2.multi_urdf_utils.render_videos \\
        --cfg src/WP2/multi_urdf_utils/configs/benchmark.yaml \\
        --output logs/multi_urdf_videos

    # With specific parameters
    python -m WP2.multi_urdf_utils.render_videos \\
        --cfg src/WP2/multi_urdf_utils/configs/benchmark.yaml \\
        --output logs/multi_urdf_videos \\
        --max-steps 500
"""

from __future__ import annotations

import argparse
import builtins
import os
os.environ["GS_PARA_LEVEL"] = "3"
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import fields as dc_fields

import numpy as np
import torch
import genesis as gs

from WP1.config import RunConfig
from WP2.config import HebbianConfig
from WP2.utils import seed_everything
from WP2.multi_urdf_utils.config import BenchmarkConfig
from WP2.multi_urdf_utils.multi_drone_env import MultiDroneEnv
from WP2.multi_urdf_utils.multi_drone_actor import MultiDroneActorManager, random_hebbian_rules
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
        checkpoint_path=checkpoint_path,
        checkpoint_config_path=checkpoint_config_path,
        hebbian_rules_list=hebbian_rules_list,
        hebb_cfg=hebb_cfg,
        stochastic=False,
        device=device,
        use_hebbian=use_hebbian,
    )

    all_positions = []
    all_orientations = []
    all_times = []

    # Track furthest drone across all episodes (by max x reached in env 0)
    best_x_per_drone = np.full(N, -np.inf)

    if video_path is not None:
        env.start_recording(video_path)

    for ep in range(num_episodes):
        print(f"    Episode {ep+1}/{num_episodes}...")

        actor_mgr.reset_episode(E, device)
        obs, _ = env.reset()
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

            all_positions.append(positions)
            all_orientations.append(orientations)
            all_times.append(t)

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

    gs.destroy()

    return {
        "positions": np.array(all_positions),   # (T, N, 3)
        "orientations": np.array(all_orientations),  # (T, N, 4)
        "times": np.array(all_times),            # (T,)
        "drone_names": [Path(p).stem for p in urdf_paths],
        "dt": wp1_cfg.env.dt,
        "episode_length_s": wp1_cfg.env.episode_length_s,
    }




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

        video_path = str(output_dir / f"scene_{scene_idx:02d}_camera.mp4")
        print(f"  Recording follow-camera video...")
        _collect_scene_trajectory(
            scene_idx=scene_idx,
            urdf_paths=scene_urdf_paths,
            wp1_cfg=wp1_cfg,
            hebb_cfg=hebb_cfg,
            benchmark_cfg=benchmark_cfg,
            checkpoint_path=benchmark_cfg.checkpoint.model_path,
            checkpoint_config_path=benchmark_cfg.checkpoint.config_path,
            num_episodes=num_episodes,
            max_steps=max_steps,
            video_path=video_path,
        )

    print(f"\n✅ All videos saved to: {output_dir}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Render videos for multi_urdf_utils benchmark scenes."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="src/WP2/multi_urdf_utils/configs/benchmark.yaml",
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
