#!/usr/bin/env python
"""
Training script for the Genesis winged–drone forest task.

This script connects:
  - the Genesis environment `WingedDroneEnv`
  - the custom policy `ActorCriticTanh` (A&C_modified.py)
  - the RSL-RL on–policy runner (PPO)

All configurable parameters live in:
  - get_train_cfg()       → PPO / policy / runner hyperparameters
  - get_cfgs()            → environment, observations, rewards, commands

From the command line you only control:
  - experiment name
  - visualization flag
  - number of parallel environments
  - maximum training iterations
  - optional parent checkpoint for policy inheritance
  - optional wide depth for the critic
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("GS_PARA_LEVEL", "3")
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import genesis as gs
import torch

from rsl_rl.runners import OnPolicyRunner
from winged_drone_train.analysis.eval_plotter import EvaluationPlotter
from winged_drone_train.rl.A2C_modified import ActorCriticTanh
from winged_drone_train.rl.logging import RLTrainingLogger
from winged_drone_train.env import (
    LISPARROW_SERVO_JOINT_NAMES,
    WingedDroneEnv,
)
from winged_drone_train.noise_config import configure_solver_noise
from winged_drone_train.runtime_random import seed_runtime_randomness
from winged_drone_train.urdf_resolver import resolve_or_generate_urdf

import builtins

# RSL-RL resolves policy classes by name; expose our implementation on builtins
builtins.ActorCriticTanh = ActorCriticTanh

# ---------------------------------------------------------------------------
#  Cache handling (Taichi / Genesis)
# ---------------------------------------------------------------------------
def _configure_cache_root() -> Path:
    """
    Force Taichi/genesis cache into a writable location to avoid ROFS errors.
    """
    cache_root = (Path("logs") / ".cache" / "gstaichi").expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    for env_key in ("XDG_CACHE_HOME", "TI_CACHE_DIR", "TAICHI_CACHE_DIR", "GSTAICHI_CACHE_DIR"):
        os.environ[env_key] = str(cache_root)
    mpl_dir = cache_root / "mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    print(f"[train] cache dir set to {cache_root}")
    return cache_root

def _init_genesis_with_retry() -> None:
    """Initialize Genesis with retry/backoff for transient OS/CUDA errors."""
    if gs._initialized:
        return
    retries = int(os.getenv("GS_INIT_RETRIES", "3") or 3)
    backoff = float(os.getenv("GS_INIT_BACKOFF", "0.5") or 0.5)
    last_exc = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            gs.init(logging_level="error", backend=gs.gpu)
            return
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if (
                "CUDA_ERROR_OPERATING_SYSTEM" in msg
                or "Resource temporarily unavailable" in msg
                or "primary_context_retain" in msg
            ) and attempt < retries:
                time.sleep(backoff * (2 ** (attempt - 1)))
                continue
            raise
    if last_exc is not None:
        raise last_exc

# =============================================================================
#  TRAINING CONFIGURATION
# =============================================================================

def get_train_cfg(exp_name: str, max_iterations: int, seed: int) -> Dict[str, Any]:
    """
    Build the training configuration dictionary consumed by RSL-RL.

    If you want to tune PPO or the network structure, edit this function.
    """
    train_cfg_dict: Dict[str, Any] = {
        # Rollout length
        "num_steps_per_env": 25,       # T in PPO (steps per env per iteration)
        "save_interval": 100,           # checkpoint every N iterations

        # Runner / logging
        "runner_class_name": "OnPolicyRunner",
        "empirical_normalization": True,
        "seed": int(seed),
        "logger": "tensorboard",

        # PPO hyperparameters
        "algorithm": {
            "normalize_advantage_per_mini_batch": True,
            "class_name": "PPO",
            "clip_param": 0.15,
            "desired_kl": 0.006, #0.005
            "entropy_coef": 0.002,
            "gamma": 0.99,
            "lam": 0.9,
            "learning_rate": 1e-4, #1e-4,5e-5
            "max_grad_norm": 0.5,
            "num_learning_epochs": 2,
            "num_mini_batches": 32,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 0.3,
        },

        # Additional RSL-RL plumbing (kept minimal)
        "init_member_classes": {},

        # Policy network configuration
        "policy": {
            "class_name": "ActorCriticTanh",   # our custom policy
            "activation": "elu", # `elu`, `selu`, `relu`, `crelu` (= CELU), `lrelu`, `tanh`, `sigmoid`, `identity`.
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "init_noise_std": 0.3,
            "rnn_type": "lstm",
            "rnn_hidden_size": 64,
            "rnn_num_layers": 1,
            "max_servo": 1.0,
            "max_throttle": 1.0,
        },

        # Runner configuration (logging, checkpointing)
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 10,
            "max_iterations": max_iterations,
            "policy_class_name": "ActorCriticTanh",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "OnPolicyRunner",
        },
    }

    return train_cfg_dict


# =============================================================================
#  ENV / OBS / REWARD / COMMAND CONFIGURATION
# =============================================================================

def get_cfgs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Build configuration dictionaries for:
      - environment (env_cfg)
      - observations (obs_cfg)
      - reward scales (reward_cfg)
      - high-level commands (command_cfg)

    These are passed directly to WingedDroneEnv.
    """

    # --------------------------------------------------------------------- #
    # Environment configuration                                            #
    # --------------------------------------------------------------------- #
    env_cfg: Dict[str, Any] = {
        # Basic task / model selection
        "num_actions": 5,              # throttle + 4 servos (kept for reference)
        "dt": 0.01,                    # not used directly; env sets dt from control_hz
        "drone": "morphing_drone",
        "naca": "3416",

        # Termination criteria
        "termination_if_close_to_ground": 0.5,
        "termination_if_y_greater_than": 50.0,
        "termination_if_z_greater_than": 50.0,

        # Initial base pose
        "base_init_pos": [-30.0, 0.0, 15.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],

        # Episode duration
        "episode_length_s": 100.0,

        # Command behaviour (kept for future use)
        "at_target_threshold": 0.1,
        "resampling_time_s": 3.0,

        # Action latency configuration (handled inside env / utils.power)
        "simulate_action_latency": True,
        "action_latency_min_steps": 0,
        "action_latency_max_steps": 1,
        "action_latency_random_per_step": False,
        "clip_actions": 1.0,

        # Visualization options
        "visualize_target": False,
        "visualize_camera": True,
        "max_visualize_FPS": 15,

        # Forest geometry
        "tree_radius": 0.75,
        "tree_height": 100.0,
        "y_lower": -50.0,
        "y_upper": 50.0,
        "dens_min_min": 0.0,
        "dens_min_max": 3.0,
        # X-limit for success condition
        "forest_x_limit": 150.0,
        "x_upper": 150.0,

        # ---------- AERODYNAMIC NOISE -------------------------------------
        # Single group controlling BOTH:
        #   - noise on aerodynamic forces
        #   - noise / randomization of aerodynamic parameters
        "aero_noise": True,
        "aero_noise_sigma0": 0.05,    # base std for mag/dir noise on aero forces
        "noise_sigma_param": 0.1,

        # ---------- LIGHT PER-ENV PHYSICAL RANDOMIZATION ------------------
        "property_randomization": {
            "mass_shift_std": 0.02,      # additive std scaled by nominal link mass
            "com_shift_std": 0.004,       # additive std in meters
            "joint_target_episode_bias_std": 0.01,
            "joint_target_step_noise_std": 0.005,
        },
        "warmup_runtime_kernels": False,

    }

    # --------------------------------------------------------------------- #
    # Observation configuration                                            #
    # --------------------------------------------------------------------- #
    obs_cfg: Dict[str, Any] = {
        # This field will be overwritten by the environment after construction.
        "num_obs": 36,

        # Whether to add Gaussian noise to actor observations
        "add_noise": True,
        "actor_genome_obs": False,
        "critic_genome_obs": False,
        "privileged_obs": {
            "base_ang_vel": True,
            "joint_position": True,
            "joint_velocity": True,
            "actual_thrust": True,
        },
        "depth_backend": "taichi",  # "taichi" or "cpu" (Taichi is faster but may cause OOM on large batches)
        # Genome-observation noise.
        # `episode_std` is sampled once at every reset and persists for the
        # whole episode. `step_std` is sampled fresh every observation build.
        "genome_obs_noise": {
            "episode_std": 0.1,
            "step_std": 0.02,
        },
        # Per-feature noise standard deviations.
        # The keys are understood by the current ObservationBuilder / helper functions.
        "noise_std": {
            "z": 0.01,         # altitude
            "quat": 0.01,      # orientation
            "vel": 0.02,       # linear velocity
            "depth": 0.05,     # depth readings
            "last_thr": 0.0,
            "last_jnts": 0.0,
            "v_tgt": 0.0,
        },
    }

    # --------------------------------------------------------------------- #
    # Reward scaling configuration                                         #
    # --------------------------------------------------------------------- #
    reward_cfg: Dict[str, Any] = {
        "reward_scales": {
            "smooth": -1e-1,
            "angular": -5e-3,
            "crash": -10.0,
            "obstacle": -1e-1,
            "energy": -1e-3,#-5e-4,   
            "progress": 5e-1,
            "height": -3e-3, #-5e-3,
            "success": 0.0,
            "cosmetic": -1.0,
            "stability": -0,
        },
    }

    # --------------------------------------------------------------------- #
    # Command configuration                                                #
    # --------------------------------------------------------------------- #
    command_cfg: Dict[str, Any] = {
        "num_commands": 1,
        "min_speed": 5.0,
        "max_speed": 25.0,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def _write_cfg_snapshot(
    cfg_path: Path,
    env_cfg: Dict[str, Any],
    obs_cfg: Dict[str, Any],
    reward_cfg: Dict[str, Any],
    command_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> None:
    """Persist the full config tuple used for the run."""
    with cfg_path.open("wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)


def _apply_train_drone_overrides(
    env_cfg: Dict[str, Any],
    *,
    drone_key: Optional[str] = None,
    urdf_file: Optional[str | Path] = None,
) -> Dict[str, Any]:
    cfg = dict(env_cfg)
    key = str(drone_key or cfg.get("drone") or "").strip().lower()
    urdf_s = str(urdf_file or cfg.get("urdf_file") or "").strip().lower()
    is_lisparrow = ("lisparrow" in key) or ("lisparrow" in urdf_s)
    if not is_lisparrow:
        return cfg

    cfg["drone"] = "lisparrow"
    cfg["aero_solver_kind"] = "lisparrow"
    cfg["servo_joint_names"] = list(LISPARROW_SERVO_JOINT_NAMES)
    cfg["fallback_servo_gains"] = (20.0, 2.0)
    cfg["naca"] = None
    return cfg


def _build_runner(
    env: WingedDroneEnv,
    train_cfg: Dict[str, Any],
    log_dir: Path,
    device: str,
) -> OnPolicyRunner:
    """Create a configured RSL-RL runner."""
    return OnPolicyRunner(env, train_cfg, str(log_dir), device=device)


def _extract_model_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract a model state_dict from the checkpoint formats used in this repo."""
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "actor_critic_state_dict",
            "policy_state_dict",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            maybe_tensor_values = any(torch.is_tensor(v) for v in checkpoint.values())
            if maybe_tensor_values:
                return checkpoint
    raise RuntimeError("Unsupported checkpoint format: could not extract a model state_dict.")


def _load_checkpoint_compatible(
    runner: OnPolicyRunner,
    checkpoint_path: str | Path,
    tag: str,
) -> Dict[str, Any]:
    """
    Load a checkpoint strictly when possible, otherwise copy only shape-compatible tensors.

    This keeps warm-start robust when the source policy was trained with a slightly
    different observation space, for example a foundation policy with genome inputs.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    try:
        runner.load(str(checkpoint_path))
        return {
            "mode": "strict",
            "matched_keys": "all",
            "skipped_keys": [],
        }
    except Exception as exc:
        print(f"[{tag}] strict load failed for {checkpoint_path}: {exc}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    source_state = _extract_model_state_dict(checkpoint)
    target_model = runner.alg.actor_critic
    target_state = target_model.state_dict()

    matched: Dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, value in source_state.items():
        if key not in target_state:
            skipped.append(f"{key}:missing")
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            skipped.append(
                f"{key}:shape {tuple(value.shape)} -> {tuple(target_state[key].shape)}"
            )
            continue
        matched[key] = value

    if not matched:
        raise RuntimeError(
            "Checkpoint is incompatible with the target network: no parameter tensors matched."
        )

    target_state.update(matched)
    target_model.load_state_dict(target_state, strict=False)
    return {
        "mode": "compatible",
        "matched_keys": len(matched),
        "skipped_keys": skipped,
    }


def _maybe_load_parent_checkpoint(
    runner: OnPolicyRunner,
    parent_exp: Optional[str],
    parent_ckpt: Optional[int],
    parent_root: Path,
    tag: str,
) -> None:
    """Optionally warm-start a runner from a parent experiment checkpoint."""
    if parent_exp is None or parent_ckpt is None:
        return
    parent_dir = parent_root / parent_exp
    ckpt_path = parent_dir / f"model_{parent_ckpt}.pt"
    if ckpt_path.is_file():
        print(f"[{tag}] Inheriting weights from {ckpt_path}")
        runner.load(str(ckpt_path))
    else:
        print(f"[{tag}] ⚠ checkpoint {ckpt_path} not found – starting from scratch.")


def _maybe_load_init_checkpoint(
    runner: OnPolicyRunner,
    init_policy_path: Optional[str],
    tag: str,
) -> None:
    """Optionally warm-start a runner from an arbitrary checkpoint path."""
    if init_policy_path is None:
        return
    load_report = _load_checkpoint_compatible(runner, init_policy_path, tag=tag)
    print(f"[{tag}] Warm-started weights from {Path(init_policy_path).expanduser().resolve()}")
    print(f"[{tag}] Load report: {load_report}")


def training(
    exp_name: str,
    urdf_file: str,
    num_envs: int,
    max_iterations: int,
    parent_exp: Optional[str] = None,
    parent_ckpt: Optional[int] = None,
    device: str = "cuda:0",
    init_policy_path: Optional[str] = None,
) -> None:
    """
    Programmatic training entry point used by the evolutionary algorithm.

    Differences from the CLI `main()`:
      - Logs under logs/ea/<exp_name> so that the evolution scripts and
        eval.evaluation() can find the runs consistently.
      - Fully parameterized (no argparse).
    """
    if init_policy_path is not None and (parent_exp is not None or parent_ckpt is not None):
        raise ValueError(
            "init_policy_path is mutually exclusive with parent_exp/parent_ckpt."
        )

    _configure_cache_root()
    # Genesis init
    _init_genesis_with_retry()
    runtime_seed = seed_runtime_randomness(f"train:{exp_name}")

    # Log directory for evolution runs
    log_dir = Path("logs") / "ea" / exp_name
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build configs
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg = _apply_train_drone_overrides(env_cfg, urdf_file=urdf_file)
    if (
        obs_cfg.get("actor_genome_obs", False)
        or obs_cfg.get("critic_genome_obs", False)
    ):
        print("[train_single] genome observations enabled in cfg -> forcing off for evolution training.")
        obs_cfg["actor_genome_obs"] = False
        obs_cfg["critic_genome_obs"] = False
    train_cfg = get_train_cfg(exp_name, max_iterations, runtime_seed)

    # Save cfg snapshot
    cfg_path = log_dir / "cfgs.pkl"
    _write_cfg_snapshot(cfg_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)

    # Environment
    env = WingedDroneEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        urdf_file=urdf_file,
        show_viewer=False,
        eval=False,
        device=device,
    )

    configure_solver_noise(env, env_cfg)

    runner = _build_runner(env, train_cfg, log_dir, device=device)
    if init_policy_path is not None:
        _maybe_load_init_checkpoint(
            runner,
            init_policy_path=init_policy_path,
            tag="train_single",
        )
    else:
        _maybe_load_parent_checkpoint(
            runner,
            parent_exp=parent_exp,
            parent_ckpt=parent_ckpt,
            parent_root=Path("logs") / "ea",
            tag="train_single",
        )
    rl_logger = RLTrainingLogger(runner=runner, log_dir=log_dir, max_iterations=max_iterations)
    rl_logger.attach()
    try:
        runner.learn(
            num_learning_iterations=max_iterations,
            init_at_random_ep_len=False,
        )
    finally:
        rl_logger.close()

    try:
        gs.destroy()
    except Exception:
        pass



# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    # --------------------------------------------------------------------- #
    #  Command-line arguments (kept minimal as requested)                   #
    # --------------------------------------------------------------------- #
    parser = argparse.ArgumentParser(
        description="PPO training for a morphing winged drone flying in a forest."
    )
    parser.add_argument(
        "-e", "--exp_name", type=str, default="drone-forest",
        help="Name of the experiment (log directory: logs/<exp_name>).",
    )
    parser.add_argument(
        "-v", "--vis", action="store_true", default=False,
        help="Enable Genesis viewer visualization.",
    )
    parser.add_argument(
        "-B", "--num_envs", type=int, default=16384, # 32768,16384
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=1000,
        help="Maximum number of training iterations (PPO updates).",
    )
    parser.add_argument(
        "--parent_exp", type=str, default=None,
        help="Parent experiment name for policy inheritance.",
    )
    parser.add_argument(
        "--parent_ckpt", type=int, default=None,
        help="Parent checkpoint number for policy inheritance.",
    )
    parser.add_argument(
        "--init-policy-path",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path used to warm-start this run. "
            "When set, it overrides --parent_exp/--parent_ckpt."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable debug prints in the environment.",
    )
    parser.add_argument(
        "--drone",
        type=str,
        default=None,
        help="Drone key for a known default URDF, e.g. 'mydrone' or 'lisparrow'.",
    )
    parser.add_argument(
        "--urdf-file",
        type=str,
        default=None,
        help="Explicit URDF path. Overrides --drone if both are provided.",
    )
    # fastest urdf [0.483543, 3.94355, 0.646855, 0.384346, 0.386289, 0.209037, 3.22566, 0.194493, 3.78274, 1.26071, 2.8838, 1.8765, 4, 3, 20]
    # most efficient urdf [0.877019, 3.58858, 0.86315, 0.456303, 0.384385, 0.258249, 2.98578, 0.201869, 3.60696, 0.788425, 2.77441, 1.89443, 4, 3, 16]
    # most progress maker urdf [0.479983, 2.74563, 0.66078, 0.384355, 0.416095, 0.351919, 2.98202, 0.326091, 3.43132, 1.25988, 2.54392, 1.79791, 4, 4, 20]
    # bad drone example [0.700882, 4.69207, 0.642027, 0.587554, 0.473013, 0.211098, 3.34718, 0.35842, 1.74638, 3.70754, 2.62685, 2.9814, 1, 5, 17]
    args = parser.parse_args()

    # --------------------------------------------------------------------- #
    #  Genesis initialization                                              #
    # --------------------------------------------------------------------- #
    gs.init(
        logging_level="error",   # use "info" or "debug" for verbose logs
        backend=gs.gpu,
        #performance_mode=True,
    )
    runtime_seed = seed_runtime_randomness(f"train_cli:{args.exp_name}")

    # --------------------------------------------------------------------- #
    #  Logging directory                                                   #
    # --------------------------------------------------------------------- #
    log_dir = Path("logs") / args.exp_name
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- #
    #  Build configuration dictionaries                                    #
    # --------------------------------------------------------------------- #
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg["debug"] = bool(args.debug)
    if args.drone:
        env_cfg["drone"] = args.drone

    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, runtime_seed)

    urdf_file = resolve_or_generate_urdf(
        urdf_file=args.urdf_file,
        drone_key=args.drone or env_cfg.get("drone"),
    )
    env_cfg = _apply_train_drone_overrides(
        env_cfg,
        drone_key=args.drone or str(env_cfg.get("drone", "")),
        urdf_file=urdf_file,
    )
    cfg_path = log_dir / "cfgs.pkl"
    _write_cfg_snapshot(cfg_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)
    # --------------------------------------------------------------------- #
    #  Environment creation                                                #
    # --------------------------------------------------------------------- #
    env = WingedDroneEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        urdf_file=urdf_file,           # let env_cfg["drone"] select the model
        show_viewer=args.vis,
        eval=False,
        device="cuda:0",
    )

    # Configure aerodynamic noise and parameter randomization
    configure_solver_noise(env, env_cfg)
    plotter = EvaluationPlotter()
    plotter.plot_forest(env)

    # --------------------------------------------------------------------- #
    #  Runner setup                                                        #
    # --------------------------------------------------------------------- #
    runner = _build_runner(env, train_cfg, log_dir, device=gs.device)
    if args.init_policy_path is not None:
        if args.parent_exp is not None or args.parent_ckpt is not None:
            raise ValueError(
                "--init-policy-path cannot be combined with --parent_exp/--parent_ckpt."
            )
        _maybe_load_init_checkpoint(
            runner,
            init_policy_path=args.init_policy_path,
            tag="train",
        )
    else:
        _maybe_load_parent_checkpoint(
            runner,
            parent_exp=args.parent_exp,
            parent_ckpt=args.parent_ckpt,
            parent_root=Path("logs"),
            tag="train",
        )

    # --------------------------------------------------------------------- #
    #  Training loop                                                       #
    # --------------------------------------------------------------------- #
    rl_logger = RLTrainingLogger(runner=runner, log_dir=log_dir, max_iterations=args.max_iterations)
    rl_logger.attach()
    try:
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=False,
        )
    finally:
        rl_logger.close()


if __name__ == "__main__":
    main()
