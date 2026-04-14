#!/usr/bin/env python3
"""
Evaluation video generation for trained policies.
=================================================

Standalone script to generate evaluation videos for any trained checkpoint,
optionally with custom controllers. Supports both single-morphology and
multi-morphology (foundation) training runs.

Videos are saved to: logs/runs/<run_name>/videos/

Usage
-----
.. code-block:: bash

    # Generate videos for the latest checkpoint
    python -m WP1 eval-videos --run logs/runs/<run_name>

    # Generate videos for a specific checkpoint
    python -m WP1 eval-videos --run logs/runs/<run_name> --ckpt 1500

    # Use a custom controller (Python file with a controller class)
    python -m WP1 eval-videos --run logs/runs/<run_name> --controller my_controller.py

    # For multi-morphology runs, also evaluate with the best morphology
    python -m WP1 eval-videos --run logs/runs/<run_name> --best-morphology

    # All options together
    python -m WP1 eval-videos \\
        --run logs/runs/<run_name> \\
        --ckpt 1500 \\
        --controller custom.py \\
        --best-morphology \\
        --speed 15.0

    # Or run directly from the Python file
    python src/WP1/eval_videos.py --run logs/runs/<run_name>
"""

from __future__ import annotations

import argparse
import builtins
import os
os.environ["GS_PARA_LEVEL"] = "3"
import pickle
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import numpy as np
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from winged_drone_train.rl.A2C_modified import ActorCriticTanh
from winged_drone_train.env import WingedDroneEnv
from winged_drone_train.defaults import default_mydrone_urdf_path
from winged_drone_train.eval_visual import (
    run_and_record,
    create_topdown_video_multi,
    create_overlay_video,
    create_camera_rewards_video,
    create_depth_video,
)

from WP1.config import RunConfig

# RSL-RL resolves policy classes by name via builtins
builtins.ActorCriticTanh = ActorCriticTanh


def _init_genesis() -> None:
    """Initialise Genesis (idempotent)."""
    if gs._initialized:
        return
    gs.init(logging_level="error", backend=gs.gpu)


def _find_latest_checkpoint(run_dir: Path) -> Optional[int]:
    """Find the highest-numbered checkpoint in the run directory.

    Parameters
    ----------
    run_dir : Path
        The run directory containing checkpoints in tb/ subdirectory.

    Returns
    -------
    int or None
        The checkpoint number (e.g., 1999 for model_1999.pt), or None if
        no checkpoints are found.
    """
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return None

    checkpoints = sorted([
        int(f.stem.split("_")[1])
        for f in tb_dir.glob("model_*.pt")
    ])
    return checkpoints[-1] if checkpoints else None


def _load_run_config(run_dir: Path) -> RunConfig:
    """Load the RunConfig from a run directory.

    Parameters
    ----------
    run_dir : Path
        The run directory containing config.yaml.

    Returns
    -------
    RunConfig
        The frozen configuration from this training run.
    """
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {run_dir}")
    return RunConfig.from_yaml(config_path)


def _load_legacy_cfgs(run_dir: Path) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """Load legacy configuration dicts from a run directory.

    First tries to load from cfgs.pkl (for older runs), then falls back
    to loading config.yaml and converting via to_legacy_cfgs().

    Parameters
    ----------
    run_dir : Path
        The run directory.

    Returns
    -------
    tuple
        (env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)
    """
    pkl_path = run_dir / "cfgs.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    # Load from config.yaml
    cfg = _load_run_config(run_dir)
    return cfg.to_legacy_cfgs()


def _is_multi_morphology(run_dir: Path) -> bool:
    """Check if this is a multi-morphology training run.

    Parameters
    ----------
    run_dir : Path
        The run directory.

    Returns
    -------
    bool
        True if catalog.txt exists (indicating multiple morphologies).
    """
    return (run_dir / "catalog.txt").exists()


def _load_catalog(run_dir: Path) -> Optional[list[str]]:
    """Load the URDF list from the catalog directory in the run folder.

    Parameters
    ----------
    run_dir : Path
        The run directory.

    Returns
    -------
    list[str] or None
        List of URDF file paths, or None if not a multi-morphology run.
    """
    catalog_txt = run_dir / "catalog.txt"
    if not catalog_txt.exists():
        return None

    catalog_dir = run_dir / "catalog"
    if not catalog_dir.is_dir():
        print("[WP1.eval_videos] Warning: catalog.txt exists but catalog/ directory not found in run directory")
        return None

    # Read catalog.txt for the list of URDFs
    catalog_file = catalog_dir / "catalog.txt"
    if catalog_file.exists():
        lines = [s.strip() for s in catalog_file.read_text().splitlines() if s.strip()]
        resolved = []
        for s in lines:
            p = Path(s)
            if not p.is_absolute():
                p = catalog_dir / p
            resolved.append(str(p))
        return resolved

    # Fallback: all URDFs in the folder
    urdfs = sorted(str(p) for p in catalog_dir.glob("*.urdf"))
    return urdfs if urdfs else None


def _generate_eval_videos(
    run_dir: Path,
    ckpt_num: Optional[int] = None,
    custom_controller: Optional[Path] = None,
    eval_speed: float = 12.0,
    eval_best_morphology: bool = False,
    morphology_index: int = 0,
) -> None:
    """Generate evaluation videos for a trained policy.

    Creates one or more single-env evaluation episodes with video recording,
    generating top-down, depth, overlay, and camera+rewards videos.

    Parameters
    ----------
    run_dir : Path
        Path to the training run directory (logs/runs/<run_name>).
    ckpt_num : int, optional
        Checkpoint number to load. If None, uses the latest checkpoint.
    custom_controller : Path, optional
        Path to a Python file defining a custom controller. If provided,
        loads this instead of the trained policy.
    eval_speed : float
        Commanded evaluation speed (m/s). Default 12.0.
    eval_best_morphology : bool
        For multi-morphology runs, also evaluate with the best morphology
        (determined from the evaluation log). Default False.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    _init_genesis()

    # Create videos directory
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Load training configuration
    cfg = _load_run_config(run_dir)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = _load_legacy_cfgs(run_dir)

    # Determine checkpoint to load
    if ckpt_num is None:
        ckpt_num = _find_latest_checkpoint(run_dir)
        if ckpt_num is None:
            raise FileNotFoundError(f"No checkpoints found in {run_dir / 'tb'}")
        print(f"[WP1.eval_videos] Using latest checkpoint: {ckpt_num}")

    ckpt_path = run_dir / "tb" / f"model_{ckpt_num}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Determine if multi-morphology
    is_multi = _is_multi_morphology(run_dir)
    urdf_list = _load_catalog(run_dir) if is_multi else None

    # Select a single URDF for evaluation
    eval_urdf = None
    if urdf_list:
        if morphology_index >= len(urdf_list):
            raise ValueError(
                f"--morphology {morphology_index} is out of range. "
                f"Catalog has {len(urdf_list)} URDFs (0-{len(urdf_list) - 1})."
            )
        eval_urdf = urdf_list[morphology_index]
        print(f"[WP1.eval_videos] Using URDF [{morphology_index}]: {Path(eval_urdf).name}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build eval-specific env config
    env_cfg_eval = env_cfg.copy()
    env_cfg_eval.update({
        "visualize_camera": False,
        "visualize_target": False,
        "max_visualize_FPS": 25,
        "aero_noise": False,
        "aero_noise_sigma0": 0.0,
        "noise_sigma_param": 0.0,
        "episode_length_s": 300.0,
    })

    # Apply eval-specific overrides from config if they exist
    if cfg.eval.episode_length_s is not None:
        env_cfg_eval["episode_length_s"] = cfg.eval.episode_length_s
    if cfg.eval.aero_noise is not None:
        env_cfg_eval["aero_noise"] = cfg.eval.aero_noise
    if cfg.eval.forest_x_limit is not None:
        env_cfg_eval["forest_x_limit"] = cfg.eval.forest_x_limit
    if cfg.eval.x_upper is not None:
        env_cfg_eval["x_upper"] = cfg.eval.x_upper
    if cfg.eval.dens_min is not None:
        env_cfg_eval["dens_min"] = cfg.eval.dens_min
    if cfg.eval.dens_max is not None:
        env_cfg_eval["dens_max"] = cfg.eval.dens_max

    obs_cfg_eval = obs_cfg.copy()
    command_cfg_eval = command_cfg.copy()
    command_cfg_eval["eval_speed"] = eval_speed

    if is_multi and eval_urdf is not None:
        print(f"\n[WP1.eval_videos] Evaluating with morphology index {morphology_index}")
        _run_single_evaluation(
            env_cfg_eval, obs_cfg_eval, reward_cfg, command_cfg_eval, train_cfg,
            ckpt_path, videos_dir, eval_urdf, device, custom_controller
        )
    else:
        print(f"\n[WP1.eval_videos] Running single-morphology evaluation")
        _run_single_evaluation(
            env_cfg_eval, obs_cfg_eval, reward_cfg, command_cfg_eval, train_cfg,
            ckpt_path, videos_dir, None, device, custom_controller
        )

    print(f"\n[WP1.eval_videos] ✅ Videos saved to: {videos_dir}")


def _get_best_morphology(run_dir: Path) -> Optional[str]:
    """Determine the best morphology from evaluation logs.

    Looks for the morphology with the highest average reward in the training
    evaluation logs.

    Parameters
    ----------
    run_dir : Path
        The run directory.

    Returns
    -------
    str or None
        The best morphology name, or None if cannot determine.
    """
    # This would require parsing the CSV logs to find best morphology
    # For now, return None (implementation depends on log format)
    # User can manually specify via catalog.txt
    return None


def _run_single_evaluation(
    env_cfg: Dict,
    obs_cfg: Dict,
    reward_cfg: Dict,
    command_cfg: Dict,
    train_cfg: Dict,
    ckpt_path: Path,
    output_dir: Path,
    eval_urdf: Optional[str],
    device: torch.device,
    custom_controller: Optional[Path],
) -> None:
    """Run a single evaluation episode and generate all videos.

    Parameters
    ----------
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg : dict
        Configuration dictionaries for environment and training.
    ckpt_path : Path
        Path to the checkpoint file.
    output_dir : Path
        Directory where videos will be saved.
    eval_urdf : str, optional
        Path to a single URDF file (for multi-morphology eval).
    device : torch.device
        Device to run on (cuda/cpu).
    custom_controller : Path, optional
        Path to custom controller file.
    """
    # Create evaluation environment
    if eval_urdf is not None:
        from general_policy.env_gen import Gen_Env
        print(f"  Creating multi-morphology eval environment...")
        env = Gen_Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            urdf_list=[eval_urdf],
            max_scenes=None,
            show_viewer=False,
            eval=True,
            device=device,
        )
    else:
        print(f"  Creating single-morphology eval environment...")
        urdf_file = str(default_mydrone_urdf_path())
        env = WingedDroneEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            urdf_file=urdf_file,
            show_viewer=False,
            eval=True,
            device=device,
        )

    # Load policy or custom controller
    if custom_controller is not None:
        print(f"  Loading custom controller from {custom_controller}...")
        policy = _load_custom_controller(custom_controller, device)
    else:
        print(f"  Loading checkpoint: {ckpt_path}")
        runner = OnPolicyRunner(env, train_cfg, str(ckpt_path.parent.parent), device=device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=device)

    # Run evaluation with video recording
    try:
        cam_mp4 = str(output_dir / "camera_view.mp4")
        print(f"  Running evaluation episode...")
        stats, traj, cam_saved = run_and_record(
            env, policy,
            show_video=False,
            collect_video=True,
            video_cam_path=cam_mp4,
            debug_aero=False,
        )

        # Generate all videos
        _create_all_videos(env, traj, cam_mp4, output_dir, cam_saved)

    finally:
        try:
            env.close()
        except Exception:
            pass


def _load_custom_controller(controller_path: Path, device: torch.device) -> object:
    """Load a custom controller from a Python file.

    The file should define a class or function `get_controller(device)` that
    returns a callable policy.

    Parameters
    ----------
    controller_path : Path
        Path to the Python file.
    device : torch.device
        Device to run on.

    Returns
    -------
    object
        The controller (a callable that takes observations and returns actions).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("custom_controller", controller_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "get_controller"):
        return module.get_controller(device)
    elif hasattr(module, "Controller"):
        return module.Controller(device)
    else:
        raise ValueError(f"No get_controller or Controller class found in {controller_path}")


def _create_all_videos(
    env: object,
    traj: Dict[str, Any],
    cam_mp4: str,
    output_dir: Path,
    cam_saved: bool,
) -> None:
    """Generate all visualization videos from a trajectory.

    Parameters
    ----------
    env : object
        The evaluation environment.
    traj : dict
        Trajectory data from run_and_record.
    cam_mp4 : str
        Path to the camera video file.
    output_dir : Path
        Output directory for videos.
    cam_saved : bool
        Whether a camera video was actually recorded.
    """
    # Top-down trajectory video
    topdown_mp4 = str(output_dir / "eval_topdown.mp4")
    print(f"    Rendering top-down video...")
    create_topdown_video_multi(env, [traj], topdown_mp4)
    print(f"    ✅ {topdown_mp4}")

    # Depth video
    depth_mp4 = None
    if traj is not None and traj.get("depth_series") is not None:
        depth_mp4 = str(output_dir / "depth_view.mp4")
        print(f"    Rendering depth video...")
        create_depth_video(
            depth_series=traj["depth_series"],
            max_distance=traj.get("depth_max_distance", float(getattr(env, "MAX_DISTANCE", 30.0))),
            save_path=depth_mp4,
            fps=int(1.0 / env.dt),
        )
        print(f"    ✅ {depth_mp4}")

    # Overlay and camera+rewards videos (if camera was recorded)
    if cam_saved:
        overlay_mp4 = str(output_dir / "overlay.mp4")
        print(f"    Rendering overlay video...")
        try:
            if hasattr(env, "commands") and env.commands.shape[1] >= 3:
                v_commanded = env.commands[0, 2].detach().cpu().item()
            else:
                v_commanded = env.commands[0, 0].detach().cpu().item()
        except Exception:
            v_commanded = 12.0

        create_overlay_video(
            cam_mp4=cam_mp4,
            td_mp4=topdown_mp4,
            traj=traj,
            out_mp4=overlay_mp4,
            v_commanded=v_commanded,
            depth_mp4=depth_mp4,
        )
        print(f"    ✅ {overlay_mp4}")

        camera_rewards_mp4 = str(output_dir / "camera_rewards.mp4")
        print(f"    Rendering camera + rewards video...")
        create_camera_rewards_video(
            cam_mp4=cam_mp4,
            traj=traj,
            out_mp4=camera_rewards_mp4,
            dpi=240,
        )
        print(f"    ✅ {camera_rewards_mp4}")


def main() -> None:
    """CLI entry point for video generation."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation videos for a trained policy."
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to the run directory (logs/runs/<run_name>).",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=None,
        help="Checkpoint number to load (e.g., 1500 for model_1500.pt). "
             "If not specified, uses the latest checkpoint.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Path to a custom controller Python file. "
             "If specified, loads this instead of the trained checkpoint.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=12.0,
        help="Commanded evaluation speed (m/s). Default: 12.0",
    )
    parser.add_argument(
        "--best-morphology",
        action="store_true",
        help="For multi-morphology runs, also evaluate with the best morphology.",
    )
    parser.add_argument(
        "--morphology",
        type=int,
        default=0,
        help="Index of the URDF in the catalog to evaluate (0-based). Default: 0 (first).",
    )

    args = parser.parse_args()

    _generate_eval_videos(
        run_dir=args.run,
        ckpt_num=args.ckpt,
        custom_controller=Path(args.controller) if args.controller else None,
        eval_speed=args.speed,
        eval_best_morphology=args.best_morphology,
        morphology_index=args.morphology,
    )


if __name__ == "__main__":
    main()
