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
os.environ["GS_PARA_LEVEL"] = "3"
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import genesis as gs

from rsl_rl.runners import OnPolicyRunner
from winged_drone_train.analysis.eval_plotter import EvaluationPlotter
from winged_drone_train.rl.A2C_modified import ActorCriticTanh
from winged_drone_train.rl.logging import RLTrainingLogger
from winged_drone_train.defaults import default_mydrone_urdf_path
from winged_drone_train.env import WingedDroneEnv

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

def get_train_cfg(exp_name: str, max_iterations: int) -> Dict[str, Any]:
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
        "seed": 1,
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
            "activation": "elu",
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
            "log_interval": 1,
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
        "termination_if_close_to_ground": 0.1,
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
        # X-limit for success condition
        "forest_x_limit": 150.0,
        "x_upper": 150.0,

        # ---------- AERODYNAMIC NOISE -------------------------------------
        # Single group controlling BOTH:
        #   - noise on aerodynamic forces
        #   - noise / randomization of aerodynamic parameters
        "aero_noise": True,
        "aero_noise_sigma0": 0.05,    # base std for mag/dir noise on aero forces
        "noise_sigma_param": 0.15,

    }

    # --------------------------------------------------------------------- #
    # Observation configuration                                            #
    # --------------------------------------------------------------------- #
    obs_cfg: Dict[str, Any] = {
        # This field will be overwritten by the environment after construction.
        "num_obs": 36,

        # Whether to add Gaussian noise to actor observations
        "add_noise": True,
        "add_genome_obs": False,

        # Per-feature noise standard deviations.
        # The keys are understood by the current ObservationBuilder / helper functions.
        "noise_std": {
            "z": 0.05,         # altitude
            "quat": 0.05,      # orientation
            "vel": 0.05,       # linear velocity
            "depth": 0.20,     # depth readings
            "last_thr": 0.0,
            "last_jnts": 0.0,
            "v_tgt": 0.0,
            "genome": 0.05,
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
            "obstacle": -0.1,
            "energy": -2e-3,#-5e-4,   
            "progress": 5e-1,
            "height": -3e-2, #-1e-1,
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
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


# =============================================================================
#  NOISE CONFIGURATION HELPERS
# =============================================================================

def configure_solver_noise(env: WingedDroneEnv, env_cfg: Dict[str, Any]) -> None:
    """
    Configure all three noise mechanisms in a single place:

    1) Mass / inertia randomization:
       - Controlled by env_cfg["robot_randomization"], "rand_mass_frac", "rand_inertia_frac".
       - Applied inside env._randomize_physical_props() at construction time.

    2) Aerodynamic parameter noise:
       - Controlled via the rigid solver flag `_enable_noise`.
       - When enabled, env.reset_idx() will call `aero_solver.randomize_aero_params(...)`
         if available in the current Genesis version.

    3) Aerodynamic force noise:
       - Configured by `aero_solver.noise_sigma_mag` and `aero_solver.noise_sigma_dir` when present.
       - These scale random perturbations on aerodynamic force magnitude and direction.
    """
    aero_solver = getattr(env, "aero_solver", None)
    if aero_solver is None:
        return

    # --- 2) aerodynamic parameter noise ---------------------------------- #
    aero_noise_enabled = bool(env_cfg.get("aero_noise", False))
    if hasattr(aero_solver, "_enable_noise"):
        aero_solver._enable_noise = aero_noise_enabled

    # --- 3) aerodynamic force noise -------------------------------------- #
    sigma0 = float(env_cfg.get("aero_noise_sigma0", 0.0))
    if aero_noise_enabled:
        sigma_mag = sigma0
        sigma_dir = sigma0
        sigma_param = float(env_cfg.get("noise_sigma_param", 0.0))
    else:
        sigma_mag = 0.0
        sigma_dir = 0.0
        sigma_param = 0.0

    if hasattr(env, "set_noise_settings"):
        env.set_noise_settings(
            aero_sigma_mag=sigma_mag,
            aero_sigma_dir=sigma_dir,
            aero_sigma_param=sigma_param,
            enable_aero_param_noise=aero_noise_enabled,
        )
    else:
        if hasattr(aero_solver, "noise_sigma_mag"):
            aero_solver.noise_sigma_mag = sigma_mag
        if hasattr(aero_solver, "noise_sigma_dir"):
            aero_solver.noise_sigma_dir = sigma_dir
        if hasattr(aero_solver, "noise_sigma_param"):
            aero_solver.noise_sigma_param = sigma_param

    if env_cfg.get("debug", False):
        print(
            f"[configure_solver_noise] Aero noise enabled: {aero_noise_enabled}, "
            f"sigma_mag: {getattr(aero_solver, 'noise_sigma_mag', 'N/A')}, "
            f"sigma_dir: {getattr(aero_solver, 'noise_sigma_dir', 'N/A')}, "
            f"sigma_param: {getattr(aero_solver, 'noise_sigma_param', 'N/A')}"
        )

    # --- 1) mass / inertia randomization --------------------------------- #
    # Mass randomization is handled inside the env via env.robot_randomization
    # and env_cfg["rand_mass_frac"]. We do not need additional wiring here.
    # `rand_inertia_frac` is kept in the config for future use by the env or
    # by a solver extension that also perturbs inertias.


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


def _build_runner(
    env: WingedDroneEnv,
    train_cfg: Dict[str, Any],
    log_dir: Path,
    device: str,
) -> OnPolicyRunner:
    """Create a configured RSL-RL runner."""
    return OnPolicyRunner(env, train_cfg, str(log_dir), device=device)


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


def training(
    exp_name: str,
    urdf_file: str,
    num_envs: int,
    max_iterations: int,
    parent_exp: Optional[str] = None,
    parent_ckpt: Optional[int] = None,
    device: str = "cuda:0",
) -> None:
    """
    Programmatic training entry point used by the evolutionary algorithm.

    Differences from the CLI `main()`:
      - Logs under logs/ea/<exp_name> so that the evolution scripts and
        eval.evaluation() can find the runs consistently.
      - Fully parameterized (no argparse).
    """
    _configure_cache_root()
    # Genesis init
    _init_genesis_with_retry()

    # Log directory for evolution runs
    log_dir = Path("logs") / "ea" / exp_name
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build configs
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    if obs_cfg.get("add_genome_obs", False):
        print("[train_single] add_genome_obs enabled in cfg → forcing off for evolution training.")
        obs_cfg["add_genome_obs"] = False
    train_cfg = get_train_cfg(exp_name, max_iterations)

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
    _maybe_load_parent_checkpoint(
        runner,
        parent_exp=parent_exp,
        parent_ckpt=parent_ckpt,
        parent_root=Path("logs") / "ea",
        tag="train_single",
    )

    runner.learn(
        num_learning_iterations=max_iterations,
        init_at_random_ep_len=False,
    )

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
        "-B", "--num_envs", type=int, default=32768, # 32768,16384
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=2000,
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
        "--debug", action="store_true", default=False,
        help="Enable debug prints in the environment.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------------- #
    #  Genesis initialization                                              #
    # --------------------------------------------------------------------- #
    gs.init(
        logging_level="error",   # use "info" or "debug" for verbose logs
        backend=gs.gpu,
        #performance_mode=True,
    )

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

    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Snapshot of all configurations for reproducibility
    cfg_path = log_dir / "cfgs.pkl"
    _write_cfg_snapshot(cfg_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)
    
    urdf_file = str(default_mydrone_urdf_path())
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
    rl_logger = RLTrainingLogger(runner=runner, log_dir=log_dir)
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
