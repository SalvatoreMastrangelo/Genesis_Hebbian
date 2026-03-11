#!/usr/bin/env python3
from __future__ import annotations

"""
Adapt a general-policy checkpoint on each URDF of a catalog, then evaluate it.

Workflow:
1. Resolve a catalog of URDFs exactly like the other general-policy scripts.
2. For each URDF, create a single-URDF PPO run that starts from a provided
   general-policy checkpoint.
3. Save the adapted checkpoint under ``logs/ea/<exp_name>/``.
4. Evaluate the adapted policy with the standard ``winged_drone_train.eval``
   path via ``general_policy.eval_gen.evaluate_single``.

The script intentionally reuses as much code as possible from:
  - ``general_policy.eval_gen`` for catalog handling, CSV logging, and eval
  - ``winged_drone_train.train`` for Genesis/RSL-RL setup
"""

import argparse
import builtins
import copy
import os
import pickle
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("GS_PARA_LEVEL", "3")

import torch
import genesis as gs

from rsl_rl.runners import OnPolicyRunner

from general_policy.catalog import build_catalog
from general_policy.eval_gen import (
    EA_ROOT,
    FitnessTriple,
    INVALID_ENERGY,
    LeanCSV,
    _configure_logging,
    _extract_reward_curve,
    copy_individual_policy_run,
    evaluate_single,
    list_urdfs,
    list_urdfs_from_nsga,
    parse_urdf_params,
)
from winged_drone_train.env import WingedDroneEnv
from winged_drone_train.noise_config import configure_solver_noise
from winged_drone_train.rl.A2C_modified import ActorCriticTanh
from winged_drone_train.rl.logging import RLTrainingLogger
from winged_drone_train.runtime_random import seed_runtime_randomness
from winged_drone_train.train import (
    _build_runner,
    _configure_cache_root,
    _init_genesis_with_retry,
)

# RSL-RL resolves policy classes by name when loading/saving checkpoints.
builtins.ActorCriticTanh = ActorCriticTanh


@dataclass
class AdaptedRun:
    exp_name: str
    log_dir: Path
    cfg_path: Path
    model_path: Path
    eval_ckpt: int
    load_report: Dict[str, Any]


def _resolve_cfg_path(
    general_policy_path: Path,
    cfg_path: Optional[Path],
) -> Path:
    candidates: List[Path] = []
    if cfg_path is not None:
        p = cfg_path.expanduser()
        if p.is_dir():
            p = p / "cfgs.pkl"
        candidates.append(p.resolve())

    model_dir = general_policy_path.expanduser().resolve().parent
    candidates.append((model_dir / "cfgs.pkl").resolve())
    candidates.append((model_dir.parent / "cfgs.pkl").resolve())

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve cfgs.pkl for checkpoint {general_policy_path}. Tried: {candidates}"
    )


def _load_cfg_tuple(cfg_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    with cfg_path.open("rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    return (
        copy.deepcopy(env_cfg),
        copy.deepcopy(obs_cfg),
        copy.deepcopy(reward_cfg),
        copy.deepcopy(command_cfg),
        copy.deepcopy(train_cfg),
    )


def _save_cfg_tuple(
    cfg_path: Path,
    env_cfg: Dict[str, Any],
    obs_cfg: Dict[str, Any],
    reward_cfg: Dict[str, Any],
    command_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> None:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("wb") as f:
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _extract_model_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
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


def _load_checkpoint_compatible(runner: OnPolicyRunner, checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()

    try:
        runner.load(str(checkpoint_path))
        return {
            "mode": "strict",
            "matched_keys": "all",
            "skipped_keys": [],
        }
    except Exception as exc:
        print(f"[adapt] strict load failed for {checkpoint_path}: {exc}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    source_state = _extract_model_state_dict(checkpoint)
    target_model = runner.alg.actor_critic
    target_state = target_model.state_dict()

    matched: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []

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


def _save_final_checkpoint(runner: OnPolicyRunner, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(runner, "save"):
        raise RuntimeError("OnPolicyRunner.save() is not available; cannot persist final checkpoint.")
    runner.save(str(model_path))


def adapt_policy(
    exp_name: str,
    urdf_file: Path,
    general_policy_path: Path,
    general_cfg_path: Path,
    num_envs: int,
    max_iterations: int,
    device: str,
) -> AdaptedRun:
    _configure_cache_root()
    _init_genesis_with_retry()
    runtime_seed = seed_runtime_randomness(f"adapt_gen:{exp_name}")

    log_dir = EA_ROOT / exp_name
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = _load_cfg_tuple(general_cfg_path)

    env_cfg["enable_rendering"] = False

    train_cfg["seed"] = int(runtime_seed)
    train_cfg.setdefault("runner", {})
    train_cfg["runner"]["experiment_name"] = exp_name
    train_cfg["runner"]["max_iterations"] = int(max_iterations)
    train_cfg["runner"]["resume"] = False
    train_cfg["runner"]["resume_path"] = None
    if "max_iterations" in train_cfg:
        train_cfg["max_iterations"] = int(max_iterations)

    try:
        env = WingedDroneEnv(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            urdf_file=str(urdf_file),
            show_viewer=False,
            eval=False,
            device=device,
        )
        configure_solver_noise(env, env_cfg)

        cfg_path = log_dir / "cfgs.pkl"
        _save_cfg_tuple(cfg_path, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)

        runner = _build_runner(env, train_cfg, log_dir, device=device)
        load_report = _load_checkpoint_compatible(runner, general_policy_path)
        print(f"[adapt] load report for {exp_name}: {load_report}")

        rl_logger = RLTrainingLogger(runner=runner, log_dir=log_dir, max_iterations=max_iterations)
        rl_logger.attach()
        try:
            if max_iterations > 0:
                runner.learn(
                    num_learning_iterations=max_iterations,
                    init_at_random_ep_len=False,
                )
        finally:
            rl_logger.close()

        final_model_idx = max(0, int(max_iterations) - 1)
        final_model_path = log_dir / f"model_{final_model_idx}.pt"
        _save_final_checkpoint(runner, final_model_path)
    finally:
        try:
            gs.destroy()
        except Exception:
            pass

    return AdaptedRun(
        exp_name=exp_name,
        log_dir=log_dir,
        cfg_path=cfg_path,
        model_path=final_model_path,
        eval_ckpt=final_model_idx + 1,
        load_report=load_report,
    )


def _resolve_urdf_catalog(
    catalog_dir: Path,
    n_urdf: int,
    urdf_seed: int,
    nsga_csv: Optional[Path],
) -> List[Path]:
    catalog_dir = catalog_dir.expanduser().resolve()
    catalog_dir.mkdir(parents=True, exist_ok=True)

    if nsga_csv is not None:
        urdf_list = list_urdfs_from_nsga(nsga_csv, catalog_dir)
    else:
        if n_urdf > 0:
            build_catalog(catalog_dir=catalog_dir, n=n_urdf, seed=urdf_seed)
        urdf_list = list_urdfs(catalog_dir)

    if not urdf_list:
        raise RuntimeError(f"No URDFs found in catalog {catalog_dir}")
    return urdf_list


def run_pipeline(
    *,
    general_policy_path: Path,
    general_cfg_path: Optional[Path],
    catalog_dir: Path,
    n_urdf: int,
    urdf_seed: int,
    nsga_csv: Optional[Path],
    exp_name: str,
    csv_path: Optional[Path],
    eval_envs: int,
    vmin: float,
    vmax: float,
    adapt_envs: int,
    adapt_iters: int,
    adapt_repeats: int,
    device: str,
) -> None:
    general_policy_path = general_policy_path.expanduser().resolve()
    if not general_policy_path.is_file():
        raise FileNotFoundError(f"General policy checkpoint not found: {general_policy_path}")

    resolved_cfg_path = _resolve_cfg_path(general_policy_path, general_cfg_path)
    urdf_list = _resolve_urdf_catalog(catalog_dir, n_urdf, urdf_seed, nsga_csv)

    if csv_path is None:
        csv_path = Path("logs") / f"{exp_name}_evaluation" / "analysis" / "adapt_and_eval_gen"

    csv_writer = LeanCSV(
        path=csv_path,
        n_baselines=0,
        n_trained=max(1, int(adapt_repeats)),
    )

    for idx, urdf in enumerate(urdf_list, start=1):
        print(f"[adapt] URDF {idx}/{len(urdf_list)}: {urdf.name}")
        clean_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", urdf.stem).strip("_") or f"urdf_{idx:03d}"
        urdf_params = parse_urdf_params(urdf)

        adapted_fitness = []
        adapted_rewards = []
        adapted_reward_curves = []

        train_t0 = time.time()
        for rep in range(max(1, int(adapt_repeats))):
            rep_t0 = time.time()
            rep_exp = f"{exp_name}_urdf{idx:03d}_rep{rep + 1}"

            try:
                run = adapt_policy(
                    exp_name=rep_exp,
                    urdf_file=urdf,
                    general_policy_path=general_policy_path,
                    general_cfg_path=resolved_cfg_path,
                    num_envs=int(adapt_envs),
                    max_iterations=int(adapt_iters),
                    device=device,
                )

                reward_curve = _extract_reward_curve(
                    run.log_dir,
                    train_iters=max(1, int(adapt_iters)),
                    n_points=20,
                    win_frac=0.05,
                )

                summary = evaluate_single(
                    exp_name=run.exp_name,
                    urdf_file=urdf,
                    ckpt=run.eval_ckpt,
                    eval_envs=int(eval_envs),
                    vmin=float(vmin),
                    vmax=float(vmax),
                    obs_genome=None,
                    model_path=run.model_path,
                    cfg_path=run.cfg_path,
                    eval_dir=run.log_dir / f"eval_{clean_stem}",
                    clean_urdf_stem=clean_stem,
                )

                copy_individual_policy_run(
                    exp_train=run.exp_name,
                    eval_name=exp_name,
                    urdf_stem=clean_stem,
                    urdf_idx=idx,
                    rep=rep,
                )

                rep_fitness = summary.fitness
                rep_reward = summary.reward_ep_mean
            except Exception as exc:
                print(f"[adapt][error] URDF={urdf.name} rep={rep + 1} failed: {exc}")
                rep_fitness = FitnessTriple(
                    speed=0.0,
                    neg_energy=-INVALID_ENERGY,
                    progress=0.0,
                )
                rep_reward = float("nan")
                reward_curve = {}

            adapted_fitness.append(rep_fitness)
            adapted_rewards.append(rep_reward)
            adapted_reward_curves.append(reward_curve)

            csv_writer.append_rep(
                urdf_stem=urdf.stem,
                urdf_params=str(urdf_params),
                baseline_fitness=[],
                baseline_rewards=[],
                rep_idx=rep + 1,
                rep_fitness=rep_fitness,
                rep_reward=rep_reward,
                rep_reward_curve=reward_curve,
                eval_duration_s=None,
                train_duration_s=time.time() - rep_t0,
            )

        csv_writer.append_agg(
            urdf_stem=urdf.stem,
            urdf_params=str(urdf_params),
            baseline_fitness=[],
            baseline_rewards=[],
            trained_fitness=adapted_fitness,
            trained_rewards=adapted_rewards,
            trained_reward_curves=adapted_reward_curves,
            eval_duration_s=None,
            train_duration_s=time.time() - train_t0,
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt a general-policy checkpoint on each URDF of a catalog and evaluate it."
    )
    parser.add_argument("--general-policy", type=Path, required=True, help="Path to the .pt general policy checkpoint.")
    parser.add_argument("--cfg-path", type=Path, default=None, help="Optional cfgs.pkl path (or parent dir).")
    parser.add_argument("--exp", type=str, required=True, help="Output experiment prefix.")
    parser.add_argument("--catalog-dir", type=Path, default=Path("urdf_foundation"), help="URDF catalog directory.")
    parser.add_argument("--n-urdf", type=int, default=0, help="If > 0, rebuild a random catalog with this many URDFs.")
    parser.add_argument("--urdf-seed", type=int, default=0, help="Seed used only when generating a random catalog.")
    parser.add_argument("--nsga-csv", type=Path, default=None, help="Optional CSV used to resolve the URDF catalog.")
    parser.add_argument("--eval-envs", type=int, default=2048, help="Parallel envs used by eval.py.")
    parser.add_argument("--vmin", type=float, default=6.0, help="Minimum commanded speed for evaluation.")
    parser.add_argument("--vmax", type=float, default=30.0, help="Maximum commanded speed for evaluation.")
    parser.add_argument("--adapt-envs", type=int, default=16384, help="Parallel envs used during adaptation PPO.")
    parser.add_argument("--adapt-iters", type=int, default=100, help="Number of PPO iterations for adaptation.")
    parser.add_argument("--adapt-repeats", type=int, default=1, help="Independent adaptation runs per URDF.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device, for example cuda:0 or cpu.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional output CSV path without extension.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity.")
    return parser.parse_args(args=argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(int(args.verbose))

    run_pipeline(
        general_policy_path=args.general_policy,
        general_cfg_path=args.cfg_path,
        catalog_dir=args.catalog_dir,
        n_urdf=int(args.n_urdf),
        urdf_seed=int(args.urdf_seed),
        nsga_csv=args.nsga_csv,
        exp_name=str(args.exp),
        csv_path=args.csv,
        eval_envs=int(args.eval_envs),
        vmin=float(args.vmin),
        vmax=float(args.vmax),
        adapt_envs=int(args.adapt_envs),
        adapt_iters=int(args.adapt_iters),
        adapt_repeats=int(args.adapt_repeats),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()
