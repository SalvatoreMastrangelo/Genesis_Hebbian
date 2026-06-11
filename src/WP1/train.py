#!/usr/bin/env python
"""
WP1 training entry point — thin YAML/CLI wrapper around
``winged_drone_train.train``.

The script:

  1. Loads a YAML config (optional) and parses ``--cfg.section.key`` CLI
     overrides.
  2. Deep-merges those onto the dicts produced by
     ``winged_drone_train.train.get_cfgs()`` / ``get_train_cfg()``.
  3. Creates a timestamped run folder via ``WP1.run_manager.RunManager``.
  4. Hands the resulting cfgs straight to
     ``winged_drone_train``'s helpers (``_configure_cache_root``,
     ``_init_genesis_with_retry``, ``_build_runner``,
     ``_apply_train_drone_overrides``, ``_maybe_load_init_checkpoint``,
     ``_maybe_load_parent_checkpoint``, ``_write_cfg_snapshot``,
     ``RLTrainingLogger``, ``configure_solver_noise``,
     ``resolve_or_generate_urdf``, ``seed_runtime_randomness``).

All training mechanics — env construction, runner setup, checkpoint
warm-start, RLTrainingLogger, learn loop — are reused verbatim from
``winged_drone_train.train``; this module only takes care of the
config plumbing and the run-folder layout.

Usage
-----
.. code-block:: bash

    # All defaults (matches winged_drone_train.train.main defaults)
    python -m WP1.train

    # From a YAML config
    python -m WP1.train --cfg src/WP1/configs/default.yaml

    # YAML + CLI overrides
    python -m WP1.train --cfg src/WP1/configs/custom.yaml \\
        --cfg.ppo.learning_rate 3e-4 \\
        --cfg.training.num_envs 8192 \\
        --cfg.reward.crash -20.0

    # Resume (creates <previous_run>_resumed)
    python -m WP1.train --cfg src/WP1/configs/default.yaml --resume
"""

from __future__ import annotations

import argparse
import builtins
import os
import random
os.environ.setdefault("GS_PARA_LEVEL", "3")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import genesis as gs

# winged_drone_train: reused verbatim
from winged_drone_train.train import (
    _apply_train_drone_overrides,
    _build_runner,
    _configure_cache_root,
    _init_genesis_with_retry,
    _maybe_load_init_checkpoint,
    _maybe_load_parent_checkpoint,
    _write_cfg_snapshot,
)
from winged_drone_train.env import WingedDroneEnv
from winged_drone_train.noise_config import configure_solver_noise
from winged_drone_train.rl.A2C_modified import ActorCriticTanh, ActorCriticTanhFF
from winged_drone_train.rl.logging import RLTrainingLogger
from winged_drone_train.runtime_random import seed_runtime_randomness
from winged_drone_train.urdf_resolver import resolve_or_generate_urdf

from WP1.config_loader import build_cfgs, parse_cli_overrides
from WP1.csv_logger import CSVLogger
from WP1.plotting import plot_run
from WP1.run_manager import RunManager

# RSL-RL resolves policy classes by name; expose ours on builtins so the
# runner can find it after pickle round-trips.
builtins.ActorCriticTanh = ActorCriticTanh
builtins.ActorCriticTanhFF = ActorCriticTanhFF


# --------------------------------------------------------------------------- #
# Post-training diagnostics helpers (verbatim from WP1_OLD)
# --------------------------------------------------------------------------- #


def _enrich_csv_with_tensorboard_rewards(csv_path: Path, tb_dir: Path) -> None:
    """Backfill the ``mean_reward`` column of ``training_log.csv`` from the
    ``Train/mean_reward`` scalar series emitted by RSL-RL into TensorBoard.

    This is best-effort: missing TensorBoard, missing scalars, or an empty
    CSV are all silently skipped. The runner's own buffers are usually
    enough, but TensorBoard guarantees we don't lose the curve when the
    buffer is not exposed at the moment ``_on_iteration_end`` fires.
    """
    import csv

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("[WP1.train] tensorboard not available — skipping reward enrichment")
        return

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    rewards_by_step: dict[int, float] = {}
    try:
        for event in ea.Scalars("Train/mean_reward"):
            rewards_by_step[event.step] = event.value
    except KeyError:
        print("[WP1.train] Train/mean_reward not found in TensorBoard")
        return
    if not rewards_by_step:
        print("[WP1.train] No Train/mean_reward events in TensorBoard")
        return

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return
    rows = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iter_num = int(row["iter"])
            if iter_num in rewards_by_step:
                row["mean_reward"] = f"{rewards_by_step[iter_num]:.6g}"
            rows.append(row)
    if not rows:
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"[WP1.train] Updated {csv_path.name} with "
        f"{len([r for r in rows if r.get('mean_reward')])} mean_reward values"
    )


def _install_csv_logger_hook(runner, env, csv_logger: CSVLogger) -> None:
    """Monkey-patch ``runner.alg.update`` so that each PPO iteration appends
    one row to ``csv_logger``.

    Captures every key found in ``env.extras["episode"]`` (so termination
    counts, reward components, final-x, etc. flow into the CSV) plus the
    PPO-side diagnostics exposed by RSL-RL.
    """
    alg = getattr(runner, "alg", None)
    if alg is None:
        return

    _state = {"i": 0}
    _orig_update = alg.update

    def _patched_update(*args, **kwargs):
        result = _orig_update(*args, **kwargs)
        it = _state["i"]
        try:
            extras = getattr(env, "extras", {}) or {}
            ppo_metrics: dict[str, float] = {}
            rewbuf = getattr(runner, "rewbuffer", None)
            if rewbuf is not None and len(rewbuf) > 0:
                ppo_metrics["mean_reward"] = sum(rewbuf) / len(rewbuf)
            lenbuf = getattr(runner, "lenbuffer", None)
            if lenbuf is not None and len(lenbuf) > 0:
                ppo_metrics["mean_episode_length"] = sum(lenbuf) / len(lenbuf)
            for metric_name in (
                "actor_loss", "critic_loss", "entropy", "ppo_loss", "value_loss",
            ):
                val = getattr(alg, metric_name, None)
                if val is not None:
                    ppo_metrics[metric_name] = val
            csv_logger.log(it, extras, ppo_metrics=ppo_metrics)
        except Exception as e:
            print(f"[WP1.train] CSV log error at iter {it}: {e}")
        _state["i"] = it + 1
        return result

    alg.update = _patched_update


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WP1 training entry point (wraps winged_drone_train.train)."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Path to a YAML config file. If omitted, all defaults are used.",
    )
    parser.add_argument(
        "-v", "--vis",
        action="store_true",
        help="Enable Genesis viewer visualisation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest run matching the experiment name.",
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
    parser.add_argument(
        "--parent-exp",
        type=str,
        default=None,
        help="Parent experiment name for policy inheritance (logs/<name>).",
    )
    parser.add_argument(
        "--parent-ckpt",
        type=int,
        default=None,
        help="Parent checkpoint number used together with --parent-exp.",
    )
    parser.add_argument(
        "--init-policy-path",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path used to warm-start this run. "
            "Mutually exclusive with --parent-exp / --parent-ckpt."
        ),
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Override the experiment name without editing the YAML.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set env_cfg['debug']=True for verbose env-side prints.",
    )
    return parser


# --------------------------------------------------------------------------- #
# Programmatic entry point
# --------------------------------------------------------------------------- #


def train(
    cfg_path: Optional[str] = None,
    *,
    vis: bool = False,
    resume: bool = False,
    drone_key: Optional[str] = None,
    urdf_file: Optional[str] = None,
    parent_exp: Optional[str] = None,
    parent_ckpt: Optional[int] = None,
    init_policy_path: Optional[str] = None,
    exp_name_override: Optional[str] = None,
    debug: bool = False,
    cli_overrides: Optional[dict] = None,
) -> Path:
    """Run a full WP1 training session.

    Returns the run directory.

    The body mirrors ``winged_drone_train.train.main`` step-by-step,
    swapping argparse for YAML/CLI overrides and ``logs/<exp>`` for a
    timestamped run folder.
    """
    if init_policy_path is not None and (parent_exp is not None or parent_ckpt is not None):
        raise ValueError(
            "--init-policy-path is mutually exclusive with --parent-exp / --parent-ckpt."
        )

    # ----- 1. Resolve config (defaults ← YAML ← CLI overrides). ---------- #
    cfg = build_cfgs(
        yaml_path=cfg_path,
        cli_overrides=cli_overrides,
        exp_name_override=exp_name_override,
    )

    exp_name: str = cfg["exp_name"]
    env_cfg = cfg["env_cfg"]
    obs_cfg = cfg["obs_cfg"]
    reward_cfg = cfg["reward_cfg"]
    command_cfg = cfg["command_cfg"]
    train_cfg = cfg["train_cfg"]
    num_envs = cfg["num_envs"]
    max_iterations = cfg["max_iterations"]
    device = cfg["device"]
    yaml_payload = cfg["yaml_payload"]
    user_seed_explicit = cfg["user_seed_explicit"]

    if debug:
        env_cfg["debug"] = True

    # ----- 2. Cache + Genesis init (reused). ----------------------------- #
    _configure_cache_root()
    # In LSS (sharded) mode Genesis runs inside worker subprocesses only;
    # the main / coordinator process must NOT call ``gs.init`` here, or it
    # will fight the workers for CUDA contexts.
    lss_enabled = bool(cfg["lss_cfg"].get("enabled", False))
    if not lss_enabled:
        _init_genesis_with_retry()

    # Seeding policy:
    #   - If the user pinned ``training.seed`` (YAML or CLI), honour it.
    #   - Otherwise, draw a fresh OS-entropy seed, matching
    #     ``winged_drone_train.train.main`` default behaviour.
    if user_seed_explicit:
        seed = int(train_cfg["seed"])
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[WP1.train] Using user-supplied seed: {seed}")
    else:
        seed = seed_runtime_randomness(f"train:{exp_name}")
        train_cfg["seed"] = int(seed)

    # ----- 3. Run folder. ----------------------------------------------- #
    run = RunManager(exp_name=exp_name, resume=resume)
    run.save_yaml_snapshot(yaml_payload)

    # ----- 4. Decide mode: single-URDF / multi-URDF (Gen_Env) / LSS. ---- #
    catalog_cfg = cfg["catalog_cfg"]
    lss_cfg = cfg["lss_cfg"]
    use_catalog = (
        (catalog_cfg.get("n_urdf") is not None and int(catalog_cfg["n_urdf"]) > 0)
        or (catalog_cfg.get("catalog_dir") is not None)
    )

    if lss_enabled and not use_catalog:
        raise RuntimeError(
            "lss.enabled=true requires catalog.n_urdf > 0 or catalog.catalog_dir."
        )
    if lss_enabled and int(lss_cfg.get("urdf_shard_size", 0)) <= 0:
        raise RuntimeError(
            "lss.enabled=true requires lss.urdf_shard_size > 0."
        )

    urdf_path: Optional[str] = None
    catalog_path: Optional[Path] = None

    if use_catalog:
        if catalog_cfg.get("catalog_dir") is not None:
            catalog_path = Path(catalog_cfg["catalog_dir"]).expanduser().resolve()
        else:
            catalog_path = run.run_dir / "catalog"

        n_urdf = catalog_cfg.get("n_urdf")
        if n_urdf is not None and int(n_urdf) > 0:
            from general_policy.catalog import build_catalog
            include_standard = bool(catalog_cfg.get("include_standard_mydrone", True))
            print(
                f"[WP1.train] Building catalog: n={n_urdf}, "
                f"seed={catalog_cfg.get('urdf_seed', 0)}, "
                f"include_standard_mydrone={include_standard}, "
                f"dir={catalog_path}"
            )
            build_catalog(
                catalog_path,
                n=int(n_urdf),
                seed=int(catalog_cfg.get("urdf_seed", 0)),
                include_standard_mydrone=include_standard,
            )
        else:
            print(f"[WP1.train] Using existing catalog at {catalog_path}")

        # In catalog mode we don't apply per-drone overrides — Gen_Env / LSS
        # workers use one WingedDroneEnv per URDF with the right drone key.
    else:
        # ----- Single-URDF: resolve / generate, apply drone overrides. --- #
        urdf_path = resolve_or_generate_urdf(
            urdf_file=urdf_file,
            drone_key=drone_key or env_cfg.get("drone"),
        )
        env_cfg = _apply_train_drone_overrides(
            env_cfg,
            drone_key=drone_key or str(env_cfg.get("drone", "")),
            urdf_file=urdf_path,
        )
        # Mirror the catalog layout for single-URDF runs so every run has a
        # ``catalog/`` folder with the URDF file + genome record.
        from general_policy.catalog import write_single_urdf_catalog
        write_single_urdf_catalog(
            run.run_dir / "catalog",
            urdf_path,
            drone_key=drone_key or env_cfg.get("drone"),
        )

    # ----- 5. Persist the legacy 5-tuple inside the run folder so the rest
    #         of the codebase (eval, plotters, custom controllers) can pick
    #         this run up without any extra plumbing.
    _write_cfg_snapshot(
        run.run_dir / "cfgs.pkl",
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        train_cfg,
    )

    # ----- 6. LSS branch: delegate the whole training loop. ------------- #
    if lss_enabled:
        from general_policy.super_scene import run_logical_super_scene_training
        print(
            f"[WP1.train] LSS mode: shards={lss_cfg.get('urdf_shard_size')}, "
            f"num_workers={lss_cfg.get('num_workers')}, "
            f"collection_gpus={lss_cfg.get('collection_gpus')}, "
            f"catalog={catalog_path}"
        )
        try:
            # Passing ``eval_dir`` makes the LSS runner write
            # ``training_log.csv`` via ``WP1.csv_logger.CSVLogger``,
            # mirroring what the non-LSS branch sets up below.
            run_logical_super_scene_training(
                experiment_name=exp_name,
                catalog_path=Path(catalog_path),
                train_cfg=train_cfg,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                log_dir=run.log_dir,
                num_envs_total=num_envs,
                max_iterations=max_iterations,
                urdf_shard_size=int(lss_cfg["urdf_shard_size"]),
                num_workers=int(lss_cfg.get("num_workers", 0)),
                collection_gpus=int(lss_cfg.get("collection_gpus", 1)),
                device=device,
                init_policy_path=init_policy_path,
                vis=vis,
                eval_dir=run.eval_dir,
            )
        finally:
            # Workers own Genesis; nothing to gs.destroy() in this process.
            pass
        _post_training(run)
        print(f"[WP1.train] Done. Results in: {run.run_dir}")
        return run.run_dir

    # ----- 7. Build the environment (single-URDF or single-process Gen_Env). #
    if use_catalog:
        from general_policy.env_gen import Gen_Env
        print(f"[WP1.train] Multi-URDF (Gen_Env) mode with catalog: {catalog_path}")
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
        # Gen_Env applies configure_solver_noise to each sub-env internally.
    else:
        env = WingedDroneEnv(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            urdf_file=urdf_path,
            show_viewer=vis,
            eval=False,
            device=device,
        )
        configure_solver_noise(env, env_cfg)

    # ----- 8. Build the runner + warm-start (reused helpers). ----------- #
    runner = _build_runner(env, train_cfg, run.log_dir, device=device)

    if init_policy_path is not None:
        _maybe_load_init_checkpoint(
            runner,
            init_policy_path=init_policy_path,
            tag="WP1.train",
        )
    else:
        _maybe_load_parent_checkpoint(
            runner,
            parent_exp=parent_exp,
            parent_ckpt=parent_ckpt,
            parent_root=Path("logs"),
            tag="WP1.train",
        )

    # ----- 9. Logger + CSV hook + learn. -------------------------------- #
    rl_logger = RLTrainingLogger(
        runner=runner,
        log_dir=run.log_dir,
        max_iterations=max_iterations,
    )
    rl_logger.attach()

    csv_logger = CSVLogger(run.eval_dir / "training_log.csv")
    _install_csv_logger_hook(runner, env, csv_logger)

    try:
        runner.learn(
            num_learning_iterations=max_iterations,
            # Multi-URDF rollouts benefit from random initial episode phases
            # so the parallel scenes don't reset in lock-step.
            init_at_random_ep_len=use_catalog,
        )
    finally:
        rl_logger.close()
        csv_logger.close()
        try:
            gs.destroy()
        except Exception:
            pass

    _post_training(run)
    print(f"[WP1.train] Done. Results in: {run.run_dir}")
    return run.run_dir


def _post_training(run: RunManager) -> None:
    """Backfill TB rewards into the CSV and render diagnostic plots.

    Best-effort: a missing CSV, missing TB events, or matplotlib import
    failure each only print a notice and skip their step.
    """
    csv_path = run.eval_dir / "training_log.csv"
    try:
        _enrich_csv_with_tensorboard_rewards(csv_path=csv_path, tb_dir=run.log_dir)
    except Exception as e:
        print(f"[WP1.train] TensorBoard reward extraction skipped: {e}")
    print("[WP1.train] Generating plots...")
    try:
        plot_run(run.run_dir)
    except Exception as e:
        print(f"[WP1.train] plot_run skipped: {e}")


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = _build_arg_parser()
    args, remaining = parser.parse_known_args()
    cli_overrides, leftover = parse_cli_overrides(remaining)
    if leftover:
        raise SystemExit(f"Unrecognised arguments: {leftover}")

    train(
        cfg_path=args.cfg,
        vis=args.vis,
        resume=args.resume,
        drone_key=args.drone,
        urdf_file=args.urdf_file,
        parent_exp=args.parent_exp,
        parent_ckpt=args.parent_ckpt,
        init_policy_path=args.init_policy_path,
        exp_name_override=args.exp_name,
        debug=args.debug,
        cli_overrides=cli_overrides,
    )


if __name__ == "__main__":
    main()
