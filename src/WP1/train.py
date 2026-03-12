#!/usr/bin/env python
"""
WP1 training entry point — config-driven, fully-logged training runs.
======================================================================

This module replaces the ad-hoc training scripts (``winged_drone_train.train``
and ``general_policy.train_gen``) with a single entry point that:

1. Reads all hyperparameters from a ``RunConfig`` (YAML file + CLI overrides).
2. Creates a timestamped, self-contained run folder via ``RunManager``.
3. Converts the config to legacy dicts and creates the environment
   (``WingedDroneEnv`` for single-morphology, ``Gen_Env`` for mixture mode).
4. Trains with RSL-RL's ``OnPolicyRunner`` while logging per-iteration
   metrics to CSV via ``CSVLogger``.
5. Auto-generates diagnostic plots at the end of training.

The existing environment, policy, and runner code is used *unmodified* —
``RunConfig.to_legacy_cfgs()`` produces the exact dict format they expect.

Modes
-----
**Single-morphology** (default):
    Uses ``WingedDroneEnv`` with the standard morphing-drone URDF.
    Activated when ``catalog.n_urdf`` is ``None`` or ``0``.

**Multi-morphology / foundation training**:
    Uses ``Gen_Env`` with a URDF catalog.  Activated when
    ``catalog.n_urdf > 0`` or ``catalog.catalog_dir`` points to an
    existing catalog directory.  The catalog is optionally (re)built
    before training.

Usage
-----
.. code-block:: bash

    # Default single-morphology run
    python -m WP1.train

    # From a YAML config file
    python -m WP1.train --cfg configs/default.yaml

    # Foundation multi-morphology training
    python -m WP1.train --cfg configs/foundation.yaml

    # With CLI overrides on top of a YAML base
    python -m WP1.train --cfg configs/default.yaml \\
        --cfg.ppo.learning_rate 3e-4 \\
        --cfg.training.num_envs 4096 \\
        --cfg.reward.crash -20.0

    # Resume from the latest matching run
    python -m WP1.train --cfg configs/default.yaml --resume

    # With Genesis viewer enabled
    python -m WP1.train --cfg configs/default.yaml -v

Output
------
All artefacts are written to the run folder created by ``RunManager``::

    logs/runs/<timestamp>_<exp_name>/
    ├── config.yaml            # frozen config snapshot
    ├── catalog.txt            # URDF list (if multi-morph)
    ├── checkpoints/           # model_*.pt
    ├── tb/                    # TensorBoard events
    ├── eval/training_log.csv  # per-iteration metrics
    └── plots/                 # reward curve, termination breakdown, ...
"""

from __future__ import annotations

import argparse
import builtins
import os
os.environ["GS_PARA_LEVEL"] = "3"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from winged_drone_train.rl.A2C_modified import ActorCriticTanh
from winged_drone_train.rl.logging import RLTrainingLogger
from winged_drone_train.train import configure_solver_noise, _configure_cache_root
from winged_drone_train.env import WingedDroneEnv
from winged_drone_train.defaults import default_mydrone_urdf_path

from WP1.config import RunConfig
from WP1.run_manager import RunManager
from WP1.csv_logger import CSVLogger
from WP1.plotting import plot_run
from WP1.eval_videos import _generate_eval_videos

# RSL-RL resolves policy classes by name via builtins
builtins.ActorCriticTanh = ActorCriticTanh


def _configure_torch_backends() -> None:
    """Enable TF32 and cuDNN autotuning for faster GPU compute."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _init_genesis() -> None:
    """Initialise the Genesis simulator (idempotent).

    Calls ``gs.init()`` with GPU backend and error-level logging.
    Skips initialisation if Genesis is already initialised.
    """
    if gs._initialized:
        return
    gs.init(logging_level="error", backend=gs.gpu)


def _enrich_csv_with_tensorboard_rewards(csv_path: Path, tb_dir: Path) -> None:
    """Extract mean_reward values from TensorBoard and update CSV.

    Reads the TensorBoard event files from a training run to extract the
    'Train/mean_reward' scalar values logged by RSL-RL, then updates the
    CSV file to fill in the mean_reward column (which may be empty if the
    runner buffers were not exposed).

    Parameters
    ----------
    csv_path : Path
        Path to the training_log.csv file to update.
    tb_dir : Path
        Path to the TensorBoard log directory (containing event files).
    """
    import csv
    import tempfile
    from pathlib import Path

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[WP1.train] tensorboard not available — skipping reward enrichment")
        return

    # Load TensorBoard events
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    # Extract mean_reward scalar (iteration -> value)
    rewards_by_step = {}
    try:
        scalars = ea.Scalars("Train/mean_reward")
        for event in scalars:
            # event is a ScalarEvent with .step and .value attributes
            rewards_by_step[event.step] = event.value
    except KeyError:
        print("[WP1.train] Train/mean_reward not found in TensorBoard")
        return

    if not rewards_by_step:
        print("[WP1.train] No Train/mean_reward events in TensorBoard")
        return

    # Read CSV and update it
    csv_path = Path(csv_path)
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iter_num = int(row["iter"])
            if iter_num in rewards_by_step:
                row["mean_reward"] = f"{rewards_by_step[iter_num]:.6g}"
            rows.append(row)

    # Write updated CSV back
    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            print(f"[WP1.train] Updated {csv_path.name} with {len([r for r in rows if r.get('mean_reward')])} mean_reward values")




def train(cfg: RunConfig, vis: bool = False, resume: bool = False) -> None:
    """Run a complete training session using the WP1 infrastructure.

    This is the main programmatic entry point.  It orchestrates:

    - Run-folder creation (``RunManager``)
    - Legacy config conversion (``RunConfig.to_legacy_cfgs()``)
    - Optional URDF catalog building (``build_catalog``)
    - Environment creation (``WingedDroneEnv`` or ``Gen_Env``)
    - Aerodynamic noise configuration
    - RSL-RL runner setup with ``RLTrainingLogger`` and ``CSVLogger``
    - Training loop (``runner.learn()``)
    - Post-training plot generation (``plot_run()``)

    Parameters
    ----------
    cfg : RunConfig
        Fully-populated training configuration.
    vis : bool
        Enable the Genesis real-time viewer for visual debugging.
    resume : bool
        If ``True``, the ``RunManager`` will find the latest existing run
        with the same experiment name and create a ``_resumed`` folder.
    """
    _configure_cache_root()
    _configure_torch_backends()
    _init_genesis()

    # --- Run manager (creates timestamped folder) ---
    run = RunManager(cfg, resume=resume)

    # --- Build legacy config dicts ---
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfg.to_legacy_cfgs()

    # --- Decide single-URDF vs mixture mode ---
    use_mixture = False
    catalog_path: Optional[Path] = None

    if cfg.catalog.catalog_dir is not None:
        catalog_path = Path(cfg.catalog.catalog_dir)
    if cfg.catalog.n_urdf is not None and cfg.catalog.n_urdf > 0:
        if catalog_path is None:
            # Build new catalog inside the run folder
            catalog_path = run.run_dir / "catalog"
        # Build catalog
        from general_policy.catalog import build_catalog
        print(f"[WP1.train] Building catalog: n={cfg.catalog.n_urdf}, seed={cfg.catalog.urdf_seed}")
        build_catalog(catalog_path, n=cfg.catalog.n_urdf, seed=cfg.catalog.urdf_seed)
        use_mixture = True
    elif catalog_path is not None and catalog_path.is_dir():
        use_mixture = True

    # Save catalog info to run folder
    run.save_catalog(catalog_path)

    # --- Create environment ---
    t0 = time.perf_counter()
    if use_mixture:
        from general_policy.env_gen import Gen_Env
        print(f"[WP1.train] Mixture mode with catalog: {catalog_path}")
        env = Gen_Env(
            num_envs=cfg.training.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            catalog_dir=str(catalog_path),
            max_scenes=None,
            show_viewer=vis,
            eval=False,
            device=cfg.training.device,
        )
    else:
        urdf_file = str(default_mydrone_urdf_path())
        print(f"[WP1.train] Single-URDF mode: {urdf_file}")
        env = WingedDroneEnv(
            num_envs=cfg.training.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            urdf_file=urdf_file,
            show_viewer=vis,
            eval=False,
            device=cfg.training.device,
        )
        configure_solver_noise(env, env_cfg)

    elapsed = time.perf_counter() - t0
    print(f"[WP1.train] Env init: {elapsed:.2f}s ({cfg.training.num_envs} envs)")

    # --- RSL-RL runner ---
    runner = OnPolicyRunner(env, train_cfg, str(run.log_dir), device=cfg.training.device)

    # --- Compile policy for faster forward/backward passes ---
    try:
        runner.alg.policy = torch.compile(
            runner.alg.policy, mode="reduce-overhead"
        )
        # Patch runner.save so checkpoints use unwrapped keys (no _orig_mod. prefix)
        _orig_save = runner.save

        def _save_unwrapped(path, infos=None):
            policy = runner.alg.policy
            unwrapped = getattr(policy, "_orig_mod", policy)
            runner.alg.policy = unwrapped
            _orig_save(path, infos)
            runner.alg.policy = policy

        runner.save = _save_unwrapped
        print("[WP1.train] torch.compile enabled (reduce-overhead mode)")
    except Exception as e:
        print(f"[WP1.train] torch.compile skipped: {e}")

    # --- Attach PPO diagnostics logger (TensorBoard) ---
    rl_logger = RLTrainingLogger(runner=runner, log_dir=run.log_dir)
    rl_logger.attach()

    # --- CSV logger (per-iteration metrics) ---
    csv_logger = CSVLogger(run.eval_dir / "training_log.csv")

    # --- Hook into PPO update to capture metrics each iteration ---
    _iter_counter = {"i": 0}

    def _on_iteration_end() -> None:
        """Post-update callback: write one CSV row with current metrics."""
        it = _iter_counter["i"]
        try:
            extras = env.extras if hasattr(env, 'extras') else {}
            mean_rew = None
            mean_ep_len = None

            # Try to get mean reward from runner's internal tracking
            rewbuffer = getattr(runner, 'rewbuffer', None)
            if rewbuffer is not None and len(rewbuffer) > 0:
                mean_rew = sum(rewbuffer) / len(rewbuffer)
            lenbuffer = getattr(runner, 'lenbuffer', None)
            if lenbuffer is not None and len(lenbuffer) > 0:
                mean_ep_len = sum(lenbuffer) / len(lenbuffer)

            csv_logger.log(it, extras, mean_reward=mean_rew, mean_episode_length=mean_ep_len)
        except Exception as e:
            print(f"[WP1.train] CSV log error at iter {it}: {e}")
        _iter_counter["i"] = it + 1

    # Monkey-patch the algorithm's update to also log CSV
    alg = getattr(runner, 'alg', None)
    if alg is not None:
        _orig_alg_update = alg.update

        def _patched_update(*args, **kwargs):
            """Wrapped PPO update that appends CSV logging."""
            result = _orig_alg_update(*args, **kwargs)
            _on_iteration_end()
            return result

        alg.update = _patched_update

    # --- Train ---
    try:
        runner.learn(
            num_learning_iterations=cfg.training.max_iterations,
            init_at_random_ep_len=use_mixture,
        )
    finally:
        rl_logger.close()
        csv_logger.close()

    # --- Extract mean_reward from TensorBoard and update CSV ---
    try:
        _enrich_csv_with_tensorboard_rewards(
            csv_path=run.eval_dir / "training_log.csv",
            tb_dir=run.log_dir
        )
    except Exception as e:
        print(f"[WP1.train] TensorBoard reward extraction skipped: {e}")

    # --- Generate evaluation videos ---
    try:
        _generate_eval_videos(run.run_dir)
    except Exception as e:
        print(f"[WP1.train] Video generation skipped: {e}")

    # --- Auto-generate plots ---
    print("[WP1.train] Generating plots...")
    plot_run(run.run_dir)

    try:
        gs.destroy()
    except Exception:
        pass

    print(f"[WP1.train] Done. Results in: {run.run_dir}")


def main() -> None:
    """CLI entry point for WP1 training.

    Parses command-line arguments, builds a ``RunConfig`` (from YAML +
    CLI overrides), and calls :func:`train`.

    Arguments
    ---------
    --cfg PATH
        Path to a YAML config file.  If omitted, all defaults are used.
    -v, --vis
        Enable the Genesis real-time viewer.
    --resume
        Resume from the latest run matching the experiment name.
    --cfg.<section>.<key> VALUE
        Override any config field (processed by
        :meth:`RunConfig.apply_cli_overrides`).
    """
    parser = argparse.ArgumentParser(description="WP1 training entry point.")
    parser.add_argument("--cfg", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("-v", "--vis", action="store_true", help="Enable viewer.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest matching run.")
    args, remaining = parser.parse_known_args()

    # Build config
    if args.cfg is not None:
        # Load base config from YAML, then merge with default for missing fields
        base = RunConfig()
        loaded = RunConfig.from_yaml(args.cfg)
        cfg = loaded
    else:
        cfg = RunConfig()

    # Apply CLI overrides like --cfg.ppo.learning_rate 3e-4
    cfg.apply_cli_overrides(remaining)

    train(cfg, vis=args.vis, resume=args.resume)


if __name__ == "__main__":
    main()
