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
import sys
os.environ.setdefault("GS_PARA_LEVEL", "3")
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from winged_drone_train.train import (
    _configure_cache_root,
    _init_genesis_with_retry,
    _maybe_load_init_checkpoint,
    get_cfgs,
    get_train_cfg,
)
from winged_drone_train.env import WingedDroneEnv
from winged_drone_train.noise_config import configure_solver_noise
from winged_drone_train.rl.logging import RLTrainingLogger
from winged_drone_train.runtime_random import seed_runtime_randomness
from general_policy.env_gen import Gen_Env
from general_policy.catalog import build_catalog
from general_policy.super_scene import run_logical_super_scene_training


def _apply_train_cfg_overrides(
    train_cfg: dict,
    num_mini_batches: Optional[int] = None,
    actor_hidden_dims: Optional[list[int]] = None,
    critic_hidden_dims: Optional[list[int]] = None,
    rnn_hidden_size: Optional[int] = None,
) -> None:
    """
    Apply optional hardcoded overrides to the train configuration in-place.
    """
    algorithm_cfg = train_cfg.setdefault("algorithm", {})
    policy_cfg = train_cfg.setdefault("policy", {})

    if num_mini_batches is not None:
        algorithm_cfg["num_mini_batches"] = int(num_mini_batches)

    if actor_hidden_dims is not None:
        policy_cfg["actor_hidden_dims"] = [int(dim) for dim in actor_hidden_dims]

    if critic_hidden_dims is not None:
        policy_cfg["critic_hidden_dims"] = [int(dim) for dim in critic_hidden_dims]

    if rnn_hidden_size is not None:
        policy_cfg["rnn_hidden_size"] = int(rnn_hidden_size)


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


def _load_cfg_snapshot(cfg_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the serialized config snapshot saved in a previous training logdir."""
    with cfg_path.open("rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


def _resolve_inherited_log_dir(path: Path) -> Path:
    """
    Accept either a direct training logdir or a run root directory and return the
    concrete logdir that contains cfgs.pkl and model_*.pt files.
    """
    path = path.expanduser().resolve()
    if (path / "cfgs.pkl").is_file():
        return path

    candidates = []
    for candidate in (path / "logs" / "ea").glob("*"):
        if not candidate.is_dir():
            continue
        if (candidate / "cfgs.pkl").is_file() and any(candidate.glob("model_*.pt")):
            candidates.append(candidate.resolve())

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve an inherited training logdir from: {path}"
        )
    raise RuntimeError(
        f"Multiple inherited training logdirs found under {path / 'logs' / 'ea'}: {candidates}"
    )


def _latest_checkpoint_path(log_dir: Path) -> Optional[Path]:
    """Return the highest-index model_*.pt checkpoint in a log directory."""
    candidates: list[tuple[int, Path]] = []
    for path in log_dir.glob("model_*.pt"):
        stem = path.stem
        try:
            step = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _resolve_inherited_catalog_dir(log_dir: Path) -> Optional[Path]:
    """
    Resolve the catalog directory associated with a previous run.

    Preference order:
      1. A nearby `urdf_generated/` directory in the run ancestors.
      2. `inherit_meta.pkl` saved in the logdir.

    We prefer the inherited run's own local `urdf_generated/` first because
    older metadata may contain container-local paths like `/workspace/out/...`.
    In a new inheritance job that path can exist again, but point to the new
    empty output directory instead of the source run.
    """
    for parent in [log_dir, *log_dir.parents]:
        candidate = parent / "urdf_generated"
        if candidate.is_dir() and ((candidate / "catalog.txt").is_file() or any(candidate.glob("*.urdf"))):
            return candidate.resolve()

    meta_path = log_dir / "inherit_meta.pkl"
    if meta_path.is_file():
        try:
            with meta_path.open("rb") as f:
                meta = pickle.load(f)
            catalog_dir = meta.get("catalog_dir")
            if catalog_dir:
                candidate = Path(str(catalog_dir)).expanduser().resolve()
                if candidate.is_dir() and (
                    (candidate / "catalog.txt").is_file() or any(candidate.glob("*.urdf"))
                ):
                    return candidate
        except Exception:
            pass
    return None


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
    logical_super_scene: bool = False,
    urdf_shard_size: int = 0,
    num_workers: int = 0,
    collection_gpus: int = 1,
    inherit_run: Optional[str] = None,
    init_policy_path: Optional[str] = None,
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
    _configure_cache_root()
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Helps reduce CUDA allocator fragmentation in long PPO runs.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    headless_no_gl = not bool(vis)
    if headless_no_gl:
        os.environ.setdefault("GS_HEADLESS_NO_GL", "1")

    node_name = os.uname().nodename
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "<unset>")
    if torch.cuda.is_available():
        gpu_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        print(
            f"[RUNTIME] Node={node_name} CUDA_VISIBLE_DEVICES={cuda_visible} "
            f"GPU[{gpu_idx}]={gpu_name}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[RUNTIME] Node={node_name} CUDA_VISIBLE_DEVICES={cuda_visible} GPU=<not available>",
            file=sys.stderr,
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # Logging directory                                                  #
    # ------------------------------------------------------------------ #
    log_dir = Path("logs") / "ea" / experiment_name
    if log_dir.exists():
        print(f"[train] Removing existing log directory: {log_dir}")
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if inherit_run is not None and init_policy_path is not None:
        raise ValueError("--inherit-run and --init-policy-path are mutually exclusive.")

    # ------------------------------------------------------------------ #
    # Load configs                                                       #
    # ------------------------------------------------------------------ #
    runtime_seed = seed_runtime_randomness(f"train_gen:{experiment_name}")
    if inherit_run is not None:
        inherit_source_dir = Path(inherit_run).expanduser().resolve()
        inherit_dir = _resolve_inherited_log_dir(inherit_source_dir)
        cfg_path = inherit_dir / "cfgs.pkl"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing cfgs.pkl in inherited run: {cfg_path}")
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = _load_cfg_snapshot(cfg_path)
        inherited_ckpt = _latest_checkpoint_path(inherit_dir)
        if inherited_ckpt is None:
            raise FileNotFoundError(f"No model_*.pt checkpoint found in inherited run: {inherit_dir}")
        init_policy_path = str(inherited_ckpt)
        inherited_catalog_dir = _resolve_inherited_catalog_dir(inherit_dir)
        if inherited_catalog_dir is None:
            raise FileNotFoundError(
                f"Could not resolve a catalog directory for inherited run: {inherit_source_dir}"
            )
        catalog_dir = str(inherited_catalog_dir)
        print(f"[train] Inheriting run source from {inherit_source_dir}")
        print(f"[train] Resolved inherited logdir to {inherit_dir}")
        print(f"[train] Inheriting cfgs from {cfg_path}")
        print(f"[train] Inheriting latest checkpoint from {inherited_ckpt}")
        print(f"[train] Inheriting catalog from {inherited_catalog_dir}")
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        train_cfg = get_train_cfg(experiment_name, max_iterations, runtime_seed)
        train_cfg_overrides = {
            "num_mini_batches": None,
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "rnn_hidden_size": 128,
        }
        _apply_train_cfg_overrides(
            train_cfg,
            num_mini_batches=train_cfg_overrides["num_mini_batches"],
            actor_hidden_dims=train_cfg_overrides["actor_hidden_dims"],
            critic_hidden_dims=train_cfg_overrides["critic_hidden_dims"],
            rnn_hidden_size=train_cfg_overrides["rnn_hidden_size"],
        )
        obs_cfg["actor_genome_obs"] = False
        obs_cfg["critic_genome_obs"] = True

    env_cfg = dict(env_cfg)
    obs_cfg = dict(obs_cfg)
    reward_cfg = dict(reward_cfg)
    command_cfg = dict(command_cfg)
    train_cfg = dict(train_cfg)
    runner_cfg = dict(train_cfg.get("runner", {}))
    runner_cfg["experiment_name"] = experiment_name
    runner_cfg["max_iterations"] = max_iterations
    runner_cfg["resume"] = False
    runner_cfg["resume_path"] = None
    train_cfg["runner"] = runner_cfg
    train_cfg["seed"] = runtime_seed

    if headless_no_gl:
        env_cfg["enable_rendering"] = False

    cfg_snapshot_path = log_dir / "cfgs.pkl"
    with cfg_snapshot_path.open("wb") as f:
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[train] Config snapshot saved to {cfg_snapshot_path}")
    inherit_meta_path = log_dir / "inherit_meta.pkl"
    with inherit_meta_path.open("wb") as f:
        pickle.dump(
            {
                "catalog_dir": str(Path(catalog_dir).expanduser().resolve()) if catalog_dir else None,
                "source_run_dir": str(Path(inherit_run).expanduser().resolve()) if inherit_run else None,
                "source_checkpoint": str(Path(init_policy_path).expanduser().resolve()) if init_policy_path else None,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

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
        build_catalog(
            catalog_path,
            n=n_urdf,
            seed=urdf_seed,
            include_standard_mydrone=False,
        )

    use_mixture = bool(catalog_path and catalog_path.is_dir())

    # ------------------------------------------------------------------ #
    # Logical super-scene mode (opt-in, mixture-only)                   #
    # ------------------------------------------------------------------ #
    if logical_super_scene:
        if not use_mixture:
            raise RuntimeError(
                "[train] --logical-super-scene requires a valid URDF catalog directory."
            )
        if urdf_shard_size <= 0:
            raise RuntimeError(
                "[train] --logical-super-scene requires --urdf-shard-size > 0."
            )
        if vis:
            print("[train] logical-super-scene mode: viewer enabled only on worker #0.")

        run_logical_super_scene_training(
            experiment_name=experiment_name,
            catalog_path=catalog_path,
            train_cfg=train_cfg,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            log_dir=log_dir,
            num_envs_total=num_envs,
            max_iterations=max_iterations,
            urdf_shard_size=urdf_shard_size,
            num_workers=num_workers,
            collection_gpus=collection_gpus,
            device=device,
            init_policy_path=init_policy_path,
            vis=vis,
        )
        return

    # Standard path keeps original behavior.
    _init_genesis_with_retry()

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
    configure_solver_noise(env, env_cfg)
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
    _maybe_load_init_checkpoint(runner, init_policy_path, tag="train_gen")
    rl_logger = RLTrainingLogger(runner=runner, log_dir=log_dir, max_iterations=max_iterations)
    rl_logger.attach()
    rl_logger.log_resource_usage(step=0, include_cuda=torch.cuda.is_available())
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
        default="foundation-mixture7",
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
        "--logical-super-scene",
        action="store_true",
        default=False,
        help=(
            "Enable logical super-scene mode: spawn one worker per URDF shard, "
            "collect rollout across all shards, then perform one global PPO update."
        ),
    )
    parser.add_argument(
        "--urdf-shard-size",
        type=int,
        default=0,
        help="Shard size used by --logical-super-scene (required when that mode is active).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of worker processes in logical-super-scene mode. "
            "If 0, auto-uses a conservative number of workers "
            "(typically 1 per rollout GPU, or 1 on CPU-only setups)."
        ),
    )
    parser.add_argument(
        "--collection-gpus",
        type=int,
        default=1,
        help=(
            "Number of visible GPUs to use for rollout shard collection in "
            "logical-super-scene mode. Default: 1."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch/Genesis device string (e.g., 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--inherit-run",
        type=str,
        default=None,
        help=(
            "Existing log directory to inherit from. Loads cfgs.pkl and the latest "
            "model_*.pt checkpoint from that run before starting a new run."
        ),
    )
    parser.add_argument(
        "--init-policy-path",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path used to warm-start training. "
            "Starts a new run but initializes matching model weights from this file."
        ),
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
        logical_super_scene=args.logical_super_scene,
        urdf_shard_size=args.urdf_shard_size,
        num_workers=args.num_workers,
        collection_gpus=args.collection_gpus,
        inherit_run=args.inherit_run,
        init_policy_path=args.init_policy_path,
        vis=args.vis,
    )


if __name__ == "__main__":
    main()
