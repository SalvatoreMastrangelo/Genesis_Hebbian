from __future__ import annotations

"""
General-policy training script for Genesis winged-drone environments.

Workflow:
---------
1) Optional: build a fresh URDF catalog with `--n-urdf` and `build_catalog`.
2) Use `Gen_Env` to create a multi-URDF mixture that uses **all** URDFs
   found in the catalog.
3) If no valid catalog is available, fall back to single-URDF `WingedDroneEnv`.
"""

import argparse
import os
os.environ["GS_PARA_LEVEL"] = "3"
import pickle
import shutil
import time
from pathlib import Path
from typing import Optional

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from winged_drone_train.train import get_cfgs, get_train_cfg
from winged_drone_train.env import WingedDroneEnv
from winged_drone_train.rl.logging import RLTrainingLogger
from general_policy.env_gen import Gen_Env
from general_policy.catalog import build_catalog


# --------------------------------------------------------------------------- #
# Catalog helpers                                                             #
# --------------------------------------------------------------------------- #
def _resolve_catalog_path(catalog_dir: Optional[str], n_urdf: Optional[int]) -> Optional[Path]:
    """
    Resolve the catalog directory with the same precedence used by `train`.
    """
    resolved_dir = catalog_dir
    if resolved_dir is None:
        env_val = os.getenv("URDF_CATALOG_DIR", "")
        if env_val:
            resolved_dir = env_val
        elif n_urdf and n_urdf > 0:
            resolved_dir = "urdf_generated"
        else:
            resolved_dir = ""

    return Path(resolved_dir).expanduser() if resolved_dir else None


# --------------------------------------------------------------------------- #
# Training                                                                   #
# --------------------------------------------------------------------------- #
def train(
    experiment_name: str,
    urdf_file: Optional[str],
    num_envs: int,
    max_iterations: int,
    catalog_dir: Optional[str] = None,
    device: Optional[str] = None,
    n_urdf: Optional[int] = None,
    urdf_seed: int = 0,
    vis: bool = False,
) -> None:
    """
    Train a general policy on the winged-drone environments.

    Modes
    -----
    1) Mixture mode (recommended for foundation training)
       - A URDF catalog directory exists (either via `catalog_dir` or
         the `URDF_CATALOG_DIR` environment variable).
       - Optionally, `--n-urdf` builds a fresh catalog before training.
       - `Gen_Env` is created and uses **all** URDFs in the catalog.

    2) Single-URDF mode (fallback)
       - If no valid catalog directory is found, we fall back to a
         single `WingedDroneEnv` using `urdf_file` (or a default drone
         from the config if `urdf_file` is None).
    """
    # ------------------------------------------------------------------ #
    # Device & Genesis initialization                                    #
    # ------------------------------------------------------------------ #
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gs.init(logging_level="error", backend=gs.gpu)

    # ------------------------------------------------------------------ #
    # Logging directory                                                  #
    # ------------------------------------------------------------------ #
    log_dir = Path("logs") / "ea" / experiment_name
    if log_dir.exists():
        print(f"[train] Removing existing log directory: {log_dir}")
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load configs                                                       #
    # ------------------------------------------------------------------ #
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(experiment_name, max_iterations)

    obs_cfg["add_genome_obs"] = True  # Always include genome observation

    cfg_snapshot_path = log_dir / "cfgs.pkl"
    with cfg_snapshot_path.open("wb") as f:
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[train] Config snapshot saved to {cfg_snapshot_path}")

    # ------------------------------------------------------------------ #
    # Catalog resolution + optional building                             #
    # ------------------------------------------------------------------ #
    catalog_path = _resolve_catalog_path(catalog_dir, n_urdf)

    # If requested, (re)build catalog first
    if n_urdf is not None and n_urdf > 0:
        if catalog_path is None:
            catalog_path = Path("urdf_generated")
        print(
            f"[train] Building URDF catalog ({n_urdf} entries) "
            f"in {catalog_path} with seed={urdf_seed}"
        )
        build_catalog(catalog_path, n=n_urdf, seed=urdf_seed)

    use_mixture = bool(catalog_path and catalog_path.is_dir())

    # ------------------------------------------------------------------ #
    # Environment creation                                               #
    # ------------------------------------------------------------------ #
    env_init_start = time.perf_counter()
    if use_mixture:
        print(f"[train] Using URDF catalog at: {catalog_path} → mixture mode")

        env = Gen_Env(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            catalog_dir=str(catalog_path),
            max_scenes=None,
            show_viewer=vis,
            eval=False,
            device=device,
        )
    else:
        print("[train] No valid catalog directory found → single `WingedDroneEnv` mode.")
        if catalog_dir:
            print(f"[train] (catalog_dir='{catalog_dir}' is not a directory)")

        env = WingedDroneEnv(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            urdf_file=urdf_file,
            show_viewer=vis,
            eval=False,
            device=device,
        )
    env_init_elapsed = time.perf_counter() - env_init_start
    per_env = env_init_elapsed / max(1, num_envs)
    print(
        "[train] Env init time: "
        f"{env_init_elapsed:.3f}s total, {per_env:.6f}s per env (num_envs={num_envs})"
    )

    # ------------------------------------------------------------------ #
    # RSL-RL runner                                                      #
    # ------------------------------------------------------------------ #
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=device)
    rl_logger = RLTrainingLogger(runner=runner, log_dir=log_dir)
    rl_logger.attach()
    try:
        runner.learn(
            num_learning_iterations=max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        rl_logger.close()

    # ------------------------------------------------------------------ #
    # Clean up Genesis                                                   #
    # ------------------------------------------------------------------ #
    try:
        gs.destroy()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for standalone training.
    """
    parser = argparse.ArgumentParser(
        description="Train a general policy on Genesis winged-drone environments."
    )

    parser.add_argument(
        "--exp-name",
        "-e",
        type=str,
        default="foundation-mixture",
        help="Name of the experiment (used for the log directory).",
    )
    parser.add_argument(
        "--num-envs",
        "-n",
        type=int,
        default=16384,
        help="Total number of parallel environments.",
    )
    parser.add_argument(
        "--max-iterations",
        "-i",
        type=int,
        default=4000,
        help="Maximum number of learning iterations.",
    )
    parser.add_argument(
        "--urdf-file",
        type=str,
        default=None,
        help="Path to a single URDF file (used if no catalog is available).",
    )
    parser.add_argument(
        "--catalog-dir",
        "-c",
        type=str,
        default=None,
        help=(
            "Directory containing a URDF catalog. If not provided, "
            "URDF_CATALOG_DIR is used, or 'urdf_generated' if --n-urdf "
            "is specified."
        ),
    )

    # Catalog building
    parser.add_argument(
        "--n-urdf",
        type=int,
        default=None,
        help="If set (>0), build a fresh catalog with this many URDFs before training.",
    )
    parser.add_argument(
        "--urdf-seed",
        type=int,
        default=0,
        help="Random seed used when generating URDFs with --n-urdf.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch/Genesis device string (e.g., 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "-v", "--vis", action="store_true", default=False,
        help="Enable Genesis viewer visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    train(
        experiment_name=args.exp_name,
        urdf_file=args.urdf_file,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        catalog_dir=args.catalog_dir,
        device=args.device,
        n_urdf=args.n_urdf,
        urdf_seed=args.urdf_seed,
        vis=args.vis,
    )


if __name__ == "__main__":
    main()
