"""
Logical-super-scene training for WP1.

This module provides the entry point for LSS (logical-super-scene) training,
which splits a URDF catalog into shards and spawns one worker process per shard
for independent rollout collection, followed by a global PPO update.

The actual LSS orchestration is handled by ``run_logical_super_scene_training()``
from ``general_policy.super_scene.runner``, which is reused here without modification.

Usage
-----
Called from WP1.train when ``cfg.lss.enabled = True``:

    python -m WP1.train --cfg configs/foundation.yaml \\
        --lss-cfg configs/logical_super_scene.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import genesis as gs

from WP1.config import RunConfig
from WP1.run_manager import RunManager
from winged_drone_train.train import configure_solver_noise, _configure_cache_root
from general_policy.catalog import build_catalog
from general_policy.super_scene import run_logical_super_scene_training


def _configure_torch_backends() -> None:
    """Enable TF32 and cuDNN autotuning for faster GPU compute."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True




def train_lss(cfg: RunConfig, vis: bool = False, resume: bool = False) -> None:
    """Train using logical-super-scene mode.

    LSS mode partitions the URDF catalog into shards and spawns one worker
    process per shard for independent rollout collection. All results are
    aggregated globally for a single PPO update.

    Parameters
    ----------
    cfg : RunConfig
        Fully-populated training configuration with ``lss.enabled = True``.
    vis : bool
        Enable the Genesis real-time viewer (only on worker #0).
    resume : bool
        If ``True``, resume from the latest run with the same experiment name.
    """
    _configure_cache_root()
    _configure_torch_backends()

    # Note: Genesis is NOT initialized in main process; workers own it.
    # Only initialize Genesis if needed for catalog building.
    if not gs._initialized:
        gs.init(logging_level="error", backend=gs.gpu)

    # --- Run manager (creates timestamped folder) ---
    run = RunManager(cfg, resume=resume)

    # --- Validate LSS config ---
    if not cfg.lss.enabled:
        raise RuntimeError("[lss_train] lss.enabled must be True")
    if cfg.lss.urdf_shard_size <= 0:
        raise RuntimeError(
            "[lss_train] lss.urdf_shard_size must be > 0; got "
            f"{cfg.lss.urdf_shard_size}"
        )

    # --- Build / resolve catalog path ---
    catalog_path: Optional[Path] = None
    if cfg.catalog.catalog_dir is not None:
        catalog_path = Path(cfg.catalog.catalog_dir)
        print(f"[lss_train] Using existing catalog_dir: {catalog_path}")

    if cfg.catalog.n_urdf is not None and cfg.catalog.n_urdf > 0:
        if catalog_path is None:
            catalog_path = run.run_dir / "catalog"
        print(
            f"[lss_train] Building catalog: n={cfg.catalog.n_urdf}, "
            f"seed={cfg.catalog.urdf_seed}"
        )
        build_catalog(catalog_path, n=cfg.catalog.n_urdf, seed=cfg.catalog.urdf_seed)
    elif catalog_path is None or not catalog_path.is_dir():
        raise RuntimeError(
            "[lss_train] LSS mode requires a valid URDF catalog. "
            "Set either catalog.n_urdf > 0 or catalog.catalog_dir to an existing path."
        )

    # Save catalog info to run folder
    run.save_catalog(catalog_path)

    # --- Build legacy config dicts ---
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfg.to_legacy_cfgs()

    # --- Print LSS config summary ---
    print(
        "[lss_train] **LOGICAL-SUPER-SCENE MODE ACTIVE**\n"
        f"  urdf_shard_size={cfg.lss.urdf_shard_size}\n"
        f"  num_workers={cfg.lss.num_workers} (0 = auto)\n"
        f"  collection_gpus={cfg.lss.collection_gpus}\n"
        f"  catalog_path={catalog_path}"
    )

    # --- Call LSS training function ---
    try:
        run_logical_super_scene_training(
            experiment_name=cfg.exp_name,
            catalog_path=catalog_path,
            train_cfg=train_cfg,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            log_dir=run.log_dir,
            num_envs_total=cfg.training.num_envs,
            max_iterations=cfg.training.max_iterations,
            urdf_shard_size=cfg.lss.urdf_shard_size,
            num_workers=cfg.lss.num_workers,
            collection_gpus=cfg.lss.collection_gpus,
            device=cfg.training.device,
            vis=vis,
            eval_dir=run.eval_dir,
        )
    finally:
        try:
            gs.destroy()
        except Exception:
            pass

    # --- Post-training artefacts ---

    # Generate evaluation videos
    print("[lss_train] Generating evaluation videos…")
    try:
        from WP1.eval_videos import _generate_eval_videos
        # Ensure Genesis is destroyed before evaluation
        try:
            gs.destroy()
        except Exception:
            pass
        _generate_eval_videos(run.run_dir)
        print("[lss_train] Evaluation videos generated")
    except Exception as exc:
        import traceback
        print(f"[lss_train] Video generation failed: {exc}")
        traceback.print_exc()

    # Generate plots
    print("[lss_train] Generating plots…")
    try:
        from WP1.plotting import plot_run
        plot_run(run.run_dir)
        print("[lss_train] Plots generated")
    except Exception as exc:
        print(f"[lss_train] Plot generation failed: {exc}")

    print(f"[lss_train] Done. Results in: {run.run_dir}")
