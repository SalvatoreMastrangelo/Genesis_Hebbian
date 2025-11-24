#!/usr/bin/env python3
"""
eval_gen.py

Evaluate a foundation (general) policy and per-morphology policies
on a catalog of Genesis winged-drone URDFs.

High-level workflow
-------------------
1. Build or load a catalog of URDF files (drone morphologies).
2. For each *baseline* checkpoint (foundation policy):
   - Stage the checkpoint and its ``cfgs.pkl`` snapshot under
     ``logs/ea/<baseline_exp>/`` so that :func:`eval.evaluation`
     can load it.
3. For each URDF in the catalog:
   a) Evaluate each baseline checkpoint with :func:`eval.evaluation`.
   b) Train one or more *per-URDF* policies using
      :func:`train.training`.
   c) Evaluate each per-URDF policy in the same way.
   d) Append a compact row to a CSV file containing, for this URDF:
        - three fitness components for each baseline checkpoint
          (speed, minus energy, progress)
        - three fitness components for each trained checkpoint
        - mean episode reward reported by the evaluator.
   e) Copy evaluation plots (total velocity/energy/progress and
      joint heatmaps) into a structured analysis directory.

The goal of this script is to make it easy to compare:
  - a general foundation policy trained on a mixture of morphologies,
  - vs. policies trained specifically on a single morphology.

The script is intentionally self-contained and only depends on:
  - ``train_gen.build_catalog``  (to generate URDF catalogs)
  - ``train.training``           (to train a single policy)
  - ``eval.evaluation``          (to evaluate a trained policy)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import random
import torch

from winged_drone_train.eval import evaluation
from train_gen import build_catalog
from winged_drone_train.train import training


# =============================================================================
#  Logging configuration
# =============================================================================

logger = logging.getLogger("eval_gen")


def _configure_logging(verbosity: int) -> None:
    """Configure a simple console logger."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )
    logger.setLevel(level)


# =============================================================================
#  Paths and constants
# =============================================================================

LOG_ROOT = Path("logs").expanduser().resolve()
ANALYSIS_ROOT = Path("analysis").expanduser().resolve()

# Sentinel used when something goes wrong with an energy measurement
INVALID_ENERGY = 100


# =============================================================================
#  URDF catalog helpers
# =============================================================================

def list_urdfs(catalog_dir: Path) -> List[Path]:
    """
    Return the list of URDF files that belong to a catalog.

    The function implements the same logic used in the training code:

    1. If ``catalog.txt`` exists in ``catalog_dir``, each non-empty line
       is interpreted as a relative or absolute path to a URDF file.
    2. Otherwise, all ``*.urdf`` files in ``catalog_dir`` are returned.

    Parameters
    ----------
    catalog_dir:
        Directory that contains a URDF catalog.

    Returns
    -------
    list of :class:`pathlib.Path`
        Sorted list of absolute URDF paths. The function does not check
        that the URDFs are valid, only that the files exist.
    """
    base = catalog_dir.expanduser().resolve()
    if not base.exists():
        return []

    catalog_txt = base / "catalog.txt"
    if catalog_txt.is_file():
        urdfs: List[Path] = []
        for line in catalog_txt.read_text().splitlines():
            s = line.strip()
            if not s:
                continue
            p = Path(s)
            if not p.is_absolute():
                p = base / p
            p = p.expanduser().resolve()
            if p.is_file():
                urdfs.append(p)
            else:
                logger.warning("URDF listed in catalog.txt not found: %s", p)
        return sorted(urdfs)

    # Fallback: use all URDF files in the directory.
    return sorted(p.expanduser().resolve() for p in base.glob("*.urdf"))


# =============================================================================
#  Baseline checkpoint staging
# =============================================================================

@dataclass
class StagedCheckpoint:
    """
    Description of a baseline checkpoint staged for evaluation.

    Attributes
    ----------
    exp_name:
        Name of the experiment under ``logs/ea/<exp_name>/``.
    ckpt_index:
        Checkpoint index to be passed to :func:`eval.evaluation`.
        Note that :mod:`eval` will internally load ``model_{ckpt-1}.pt``.
    model_path:
        Original, user-provided path to the checkpoint file.
    log_dir:
        Directory where the checkpoint and ``cfgs.pkl`` have been staged.
    """
    exp_name: str
    ckpt_index: int
    model_path: Path
    log_dir: Path


def stage_baseline_checkpoint(
    exp_name: str,
    model_path: Path,
    cfg_dir: Optional[Path],
) -> StagedCheckpoint:
    """
    Copy a baseline checkpoint and its config snapshot into ``logs/ea/``.

    The evaluation code expects runs to live under ``logs/ea/<exp_name>/``
    with files:

      - ``cfgs.pkl``        – environment / training configuration
      - ``model_<k>.pt``    – checkpoint at iteration ``k``

    and will load ``model_{ckpt-1}.pt`` given the ``ckpt`` integer.

    This helper:

      1. Creates ``logs/ea/<exp_name>/`` if needed.
      2. Copies ``cfgs.pkl`` from ``cfg_dir``.
      3. Copies ``model_*.pt`` from ``model_path``.
         If ``model_path`` already has the pattern ``model_<N>.pt``,
         the same index ``N`` is preserved. Otherwise we use ``N = 0``.

    Parameters
    ----------
    exp_name:
        Name of the experiment (directory under ``logs/ea/``).
    model_path:
        Path to the checkpoint to be evaluated.
    cfg_dir:
        Directory containing ``cfgs.pkl``. If ``None``, the parent
        directory of ``model_path`` is used.

    Returns
    -------
    :class:`StagedCheckpoint`
        Description of the staged checkpoint, including the checkpoint
        index to pass to :func:`eval.evaluation`.
    """
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    run_dir = (LOG_ROOT / "ea" / exp_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) cfgs.pkl ------------------------------------------------------
    if cfg_dir is None:
        cfg_dir = model_path.parent
    cfg_dir = cfg_dir.expanduser().resolve()
    cfg_src = cfg_dir / "cfgs.pkl"
    cfg_dst = run_dir / "cfgs.pkl"

    if not cfg_src.is_file():
        raise FileNotFoundError(
            f"cfgs.pkl not found in cfg_dir={cfg_dir}. "
            "Please point --cfg-dir to a run directory that contains it."
        )

    shutil.copy2(cfg_src, cfg_dst)
    logger.debug("Copied cfgs.pkl from %s to %s", cfg_src, cfg_dst)

    # --- 2) model_<N>.pt --------------------------------------------------
    # Try to infer N from the filename, falling back to N=0.
    m = re.search(r"model_(\d+)\.pt$", model_path.name)
    if m:
        index = int(m.group(1))
    else:
        index = 0

    model_dst = run_dir / f"model_{index}.pt"
    if model_dst.exists():
        model_dst.unlink()
    shutil.copy2(model_path, model_dst)

    logger.info(
        "Staged baseline checkpoint: %s → %s (index=%d, exp=%s)",
        model_path,
        model_dst,
        index,
        exp_name,
    )

    # evaluation() expects a 'ckpt' such that it loads model_{ckpt-1}.pt
    ckpt_for_eval = index + 1
    return StagedCheckpoint(
        exp_name=exp_name,
        ckpt_index=ckpt_for_eval,
        model_path=model_path,
        log_dir=run_dir,
    )


# =============================================================================
#  Evaluation helpers
# =============================================================================

@dataclass
class FitnessTriple:
    """
    Compact representation of a morphology's flight performance.

    We follow the same convention used by the evolutionary code:

      - speed:    mean forward velocity  (to maximize)
      - neg_energy: minus energy per meter (to minimize energy)
      - progress: mean distance flown    (to maximize)
    """
    speed: float
    neg_energy: float
    progress: float


@dataclass
class EvalSummary:
    """Summary of a single evaluation run."""
    fitness: FitnessTriple
    reward_ep_mean: float
    metadata: Dict[str, Any]


def evaluate_single(
    exp_name: str,
    urdf_file: Path,
    ckpt: int,
    eval_envs: int,
    vmin: float,
    vmax: float,
) -> EvalSummary:
    """
    Run :func:`eval.evaluation` and convert its output into a FitnessTriple.

    Parameters
    ----------
    exp_name:
        Name of the experiment under ``logs/ea/<exp_name>/``.
    urdf_file:
        URDF file describing the drone morphology.
    ckpt:
        Checkpoint index to evaluate. Internally, ``eval.evaluation``
        will load ``model_{ckpt-1}.pt`` from the log directory.
    eval_envs:
        Number of parallel environments to use during evaluation.
    vmin, vmax:
        Minimum and maximum commanded speed used for the evaluation sweep.

    Returns
    -------
    :class:`EvalSummary`
        Structured summary of the evaluation results.
    """
    logger.debug(
        "Evaluating exp=%s ckpt=%d on urdf=%s (envs=%d, vmin=%.1f, vmax=%.1f)",
        exp_name,
        ckpt,
        urdf_file,
        eval_envs,
        vmin,
        vmax,
    )

    # evaluation() returns:
    #   top_vel, top_eff, top_prog, max_p, extra
    top_vel, top_eff, top_prog, max_p, extra = evaluation(
        exp_name=str(exp_name),
        urdf_file=str(urdf_file),
        ckpt=int(ckpt),
        envs=int(eval_envs),
        vmin=float(vmin),
        vmax=float(vmax),
        return_arrays=True,
    )

    # Build a fitness triple:
    #   - maximize speed (velocity at best-speed operating point)
    #   - minimize energy: take MIN energy and flip sign
    #   - maximize progress (distance at best-progress operating point)
    speed = float(top_vel["mean_v"])
    neg_energy = -float(top_eff["mean_E"])
    progress = float(top_prog["mean_progress"])

    fitness = FitnessTriple(speed=speed, neg_energy=neg_energy, progress=progress)

    # Reward statistics and auxiliary info
    reward_ep_mean = float(extra.get("final_reward", float("nan")))
    metadata: Dict[str, Any] = {
        "top_vel": top_vel,
        "top_eff": top_eff,
        "top_prog": top_prog,
        "max_p": float(extra.get("max_p", max_p)),
    }
    metadata.update(extra)

    logger.info(
        "Evaluation exp=%s ckpt=%d urdf=%s → "
        "speed=%.3f, neg_energy=%.3f, progress=%.3f, mean_reward=%.3f",
        exp_name,
        ckpt,
        urdf_file.name,
        fitness.speed,
        fitness.neg_energy,
        fitness.progress,
        reward_ep_mean,
    )

    return EvalSummary(fitness=fitness, reward_ep_mean=reward_ep_mean, metadata=metadata)


def copy_eval_images(
    exp_name: str,
    urdf_file: Path,
    dst_dir: Path,
) -> None:
    """
    Copy evaluation plots for a given morphology into a destination directory.

    The evaluation script writes its plots into:

        logs/ea/<exp_name>/eval_<urdf_stem>/

    where ``<urdf_stem>`` is the stem of the URDF filename.

    This helper copies a small set of standard plots (if they exist):

      - ``total_plot.png``             – velocity, energy, progress summary
      - ``joint_heatmap_sweep.png``    – control sweep heatmap
      - ``joint_heatmap_twist.png``    – control twist heatmap

    Parameters
    ----------
    exp_name:
        Name of the experiment under ``logs/ea/<exp_name>/``.
    urdf_file:
        URDF file for which the evaluation was run.
    dst_dir:
        Directory where plots will be copied.
    """
    src_dir = LOG_ROOT / "ea" / exp_name / f"eval_{urdf_file.stem}"
    dst_dir = dst_dir.expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    plot_names = (
        "total_plot.png",
        "joint_heatmap_sweep.png",
        "joint_heatmap_twist.png",
    )

    for name in plot_names:
        src = src_dir / name
        if src.is_file():
            dst = dst_dir / name
            try:
                shutil.copy2(src, dst)
                logger.debug("Copied %s → %s", src, dst)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Failed to copy %s → %s: %s", src, dst, exc)


# =============================================================================
#  Lean CSV writer
# =============================================================================

class LeanCSV:
    """
    Minimal CSV writer: one row per URDF, a handful of scalar metrics.

    For ``B`` baseline checkpoints and ``T`` trained checkpoints the
    header has the following structure:

        urdf_stem,
        f_speed_baseline1, f_negE_baseline1, f_prog_baseline1, reward_ep_mean_baseline1,
        ...,
        f_speed_trained1,  f_negE_trained1,  f_prog_trained1,  reward_ep_mean_trained1,
        ...

    The class only appends rows; it never attempts to read or update
    existing contents.
    """

    def __init__(self, path: Path, n_baselines: int, n_trained: int) -> None:
        self.path = path.with_suffix(".csv").expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_baselines = max(0, int(n_baselines))
        self.n_trained = max(0, int(n_trained))

        if not self.path.exists():
            header = ["urdf_stem"]
            # Baseline columns
            for i in range(self.n_baselines):
                k = i + 1
                header += [
                    f"f_speed_baseline{k}",
                    f"f_negE_baseline{k}",
                    f"f_prog_baseline{k}",
                    f"reward_ep_mean_baseline{k}",
                ]
            # Trained columns
            for i in range(self.n_trained):
                k = i + 1
                header += [
                    f"f_speed_trained{k}",
                    f"f_negE_trained{k}",
                    f"f_prog_trained{k}",
                    f"reward_ep_mean_trained{k}",
                ]
            self.path.write_text(",".join(header) + "\n")

    # ------------------------------------------------------------------ #

    def append(
        self,
        urdf_stem: str,
        baseline_fitness: Sequence[FitnessTriple],
        baseline_rewards: Sequence[float],
        trained_fitness: Sequence[FitnessTriple],
        trained_rewards: Sequence[float],
    ) -> None:
        """
        Append a new CSV row for a single URDF.

        Parameters
        ----------
        urdf_stem:
            Stem of the URDF filename (without extension).
        baseline_fitness:
            Sequence of :class:`FitnessTriple` values, one per baseline
            checkpoint.
        baseline_rewards:
            Sequence of mean episode rewards for each baseline.
        trained_fitness:
            Sequence of :class:`FitnessTriple` values for trained policies.
        trained_rewards:
            Sequence of mean episode rewards for each trained policy.
        """
        row: List[str] = [urdf_stem]

        # Baseline metrics
        for i in range(self.n_baselines):
            if i < len(baseline_fitness):
                f = baseline_fitness[i]
                r = baseline_rewards[i] if i < len(baseline_rewards) else float("nan")
            else:
                f = FitnessTriple(speed=float("nan"), neg_energy=float("nan"), progress=float("nan"))
                r = float("nan")

            row += [
                f"{f.speed:.6f}",
                f"{f.neg_energy:.6f}",
                f"{f.progress:.6f}",
                f"{r:.6f}" if np.isfinite(r) else "nan",
            ]

        # Trained metrics
        for i in range(self.n_trained):
            if i < len(trained_fitness):
                f = trained_fitness[i]
                r = trained_rewards[i] if i < len(trained_rewards) else float("nan")
            else:
                f = FitnessTriple(speed=float("nan"), neg_energy=float("nan"), progress=float("nan"))
                r = float("nan")

            row += [
                f"{f.speed:.6f}",
                f"{f.neg_energy:.6f}",
                f"{f.progress:.6f}",
                f"{r:.6f}" if np.isfinite(r) else "nan",
            ]

        with self.path.open("a") as f:
            f.write(",".join(row) + "\n")


# =============================================================================
#  End-to-end pipeline
# =============================================================================

def run_pipeline(
    catalog_dir: Path,
    n_urdf: int,
    rng_seed: int,
    baseline_models: Sequence[Path],
    cfg_dir: Optional[Path],
    baseline_exp: str,
    csv_path: Optional[Path],
    eval_envs: int,
    vmin: float,
    vmax: float,
    train_envs: int,
    train_iters: int,
    train_repeats: int,
    device: str,
) -> None:
    """
    Full end-to-end workflow.

    Parameters
    ----------
    catalog_dir:
        Directory that contains or will contain the URDF catalog.
    n_urdf:
        If > 0, a fresh catalog with exactly ``n_urdf`` URDFs is built
        before evaluation. If 0, an existing catalog is used as-is.
    rng_seed:
        Global random seed for reproducibility (Python, NumPy, Torch).
    baseline_models:
        Sequence of checkpoint files implementing the foundation policy
        to be evaluated.
    cfg_dir:
        Directory from which to copy ``cfgs.pkl`` for the baseline
        models. If ``None``, the parent of the first model is used.
    baseline_exp:
        Experiment name used to stage baseline models in
        ``logs/ea/<baseline_exp>/``.
    csv_path:
        Output CSV path (without extension). If ``None``, defaults to
        ``analysis/foundation_eval_lean.csv``.
    eval_envs:
        Number of parallel environments used during evaluation.
    vmin, vmax:
        Commanded speed range for evaluation.
    train_envs:
        Number of environments used during training of per-URDF policies.
    train_iters:
        Number of PPO iterations for per-URDF policies. The final
        checkpoint is assumed to be ``model_{train_iters-1}.pt``.
    train_repeats:
        Number of independent training runs per URDF (with different
        random seeds).
    device:
        Device string for training, e.g. ``"cuda:0"`` or ``"cpu"``.
    """
    catalog_dir = catalog_dir.expanduser().resolve()
    catalog_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # RNG seeding                                                        #
    # ------------------------------------------------------------------ #
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)

    # ------------------------------------------------------------------ #
    # Build / load catalog                                               #
    # ------------------------------------------------------------------ #
    if n_urdf > 0:
        logger.info(
            "Building a fresh URDF catalog with %d entries into %s",
            n_urdf,
            catalog_dir,
        )
        build_catalog(catalog_dir=catalog_dir, n=n_urdf, seed=rng_seed)

    urdf_list = list_urdfs(catalog_dir)
    if not urdf_list:
        raise RuntimeError(
            f"No URDF files found in catalog directory: {catalog_dir}. "
            "Either generate a new catalog with --n-urdf or provide an "
            "existing directory containing *.urdf files or a catalog.txt."
        )
    logger.info("Found %d URDFs in catalog %s", len(urdf_list), catalog_dir)

    # ------------------------------------------------------------------ #
    # Stage baseline models                                              #
    # ------------------------------------------------------------------ #
    if not baseline_models:
        raise ValueError("At least one baseline model must be provided.")

    baseline_models = [p.expanduser().resolve() for p in baseline_models]
    if cfg_dir is None:
        cfg_dir = baseline_models[0].parent

    staged: List[StagedCheckpoint] = []
    for model_path in baseline_models:
        staged_ckpt = stage_baseline_checkpoint(
            exp_name=baseline_exp,
            model_path=model_path,
            cfg_dir=cfg_dir,
        )
        staged.append(staged_ckpt)

    # CSV initialization
    if csv_path is None:
        csv_path = ANALYSIS_ROOT / "foundation_eval_lean"

    csv_writer = LeanCSV(
        path=csv_path,
        n_baselines=len(staged),
        n_trained=train_repeats,
    )
    logger.info("Writing lean CSV to %s", csv_writer.path)

    # ------------------------------------------------------------------ #
    # Main loop over URDFs                                               #
    # ------------------------------------------------------------------ #
    for idx, urdf in enumerate(urdf_list, start=1):
        logger.info("=== [%d / %d] %s ===", idx, len(urdf_list), urdf.name)

        # A) Evaluate all baseline checkpoints
        baseline_fitness: List[FitnessTriple] = []
        baseline_rewards: List[float] = []

        for j, sc in enumerate(staged, start=1):
            try:
                summary = evaluate_single(
                    exp_name=sc.exp_name,
                    urdf_file=urdf,
                    ckpt=sc.ckpt_index,
                    eval_envs=eval_envs,
                    vmin=vmin,
                    vmax=vmax,
                )
            except Exception as exc:  # pragma: no cover - robust to individual failures
                logger.error(
                    "Baseline evaluation failed for urdf=%s (checkpoint=%s): %s",
                    urdf.name,
                    sc.model_path,
                    exc,
                )
                baseline_fitness.append(
                    FitnessTriple(
                        speed=0.0,
                        neg_energy=-INVALID_ENERGY,
                        progress=0.0,
                    )
                )
                baseline_rewards.append(float("nan"))
            else:
                baseline_fitness.append(summary.fitness)
                baseline_rewards.append(summary.reward_ep_mean)

        # Copy baseline plots from the *last* staged experiment
        copy_eval_images(
            exp_name=staged[-1].exp_name,
            urdf_file=urdf,
            dst_dir=ANALYSIS_ROOT / "foundation_eval_runs" / urdf.stem / "baseline",
        )

        # B) Train and evaluate per-URDF policies
        trained_fitness: List[FitnessTriple] = []
        trained_rewards: List[float] = []

        for rep in range(train_repeats):
            exp_train = urdf.stem if train_repeats == 1 else f"{urdf.stem}_rep{rep+1}"
            run_seed = rng_seed + rep

            logger.info(
                "[train %d/%d] exp=%s  seed=%d  envs=%d  iters=%d",
                rep + 1,
                train_repeats,
                exp_train,
                run_seed,
                train_envs,
                train_iters,
            )

            # Per-run RNG seed for robustness
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

            # Training: single URDF, evolution-friendly logging layout
            try:
                training(
                    exp_name=exp_train,
                    urdf_file=str(urdf),
                    num_envs=int(train_envs),
                    max_iterations=int(train_iters),
                    parent_exp=None,
                    parent_ckpt=None,
                    device=device,
                )
            except Exception as exc:  # pragma: no cover - robust to training failures
                logger.error(
                    "Training failed for urdf=%s (exp=%s): %s",
                    urdf.name,
                    exp_train,
                    exc,
                )
                trained_fitness.append(
                    FitnessTriple(
                        speed=0.0,
                        neg_energy=-INVALID_ENERGY,
                        progress=0.0,
                    )
                )
                trained_rewards.append(float("nan"))
                continue

            # Evaluate the final checkpoint (iteration 'train_iters').
            try:
                summary = evaluate_single(
                    exp_name=exp_train,
                    urdf_file=urdf,
                    ckpt=train_iters,
                    eval_envs=eval_envs,
                    vmin=vmin,
                    vmax=vmax,
                )
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "Evaluation of trained policy failed for urdf=%s (exp=%s): %s",
                    urdf.name,
                    exp_train,
                    exc,
                )
                trained_fitness.append(
                    FitnessTriple(
                        speed=0.0,
                        neg_energy=-INVALID_ENERGY,
                        progress=0.0,
                    )
                )
                trained_rewards.append(float("nan"))
            else:
                trained_fitness.append(summary.fitness)
                trained_rewards.append(summary.reward_ep_mean)

                # Copy plots only for the first repetition
                if rep == 0:
                    copy_eval_images(
                        exp_name=exp_train,
                        urdf_file=urdf,
                        dst_dir=ANALYSIS_ROOT
                        / "foundation_eval_runs"
                        / urdf.stem
                        / "trained",
                    )

        # C) Append a row to the CSV
        csv_writer.append(
            urdf_stem=urdf.stem,
            baseline_fitness=baseline_fitness,
            baseline_rewards=baseline_rewards,
            trained_fitness=trained_fitness,
            trained_rewards=trained_rewards,
        )

    logger.info("Pipeline finished successfully.")


# =============================================================================
#  CLI
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    This function is intentionally explicit: each parameter of the
    end-to-end pipeline has a corresponding CLI flag for reproducibility.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a foundation policy and per-URDF policies on a "
            "catalog of Genesis winged-drone morphologies."
        )
    )

    # Catalog and general options
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("urdf_foundation"),
        help="Directory used to store the URDF catalog.",
    )
    parser.add_argument(
        "--n-urdf",
        type=int,
        default=0,
        help="If > 0, build a fresh catalog with this many URDFs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for catalog generation and training.",
    )

    # Baseline checkpoints (up to 5 for convenience)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Legacy convenience flag for a single baseline checkpoint. "
            "Equivalent to --model-path1 when given."
        ),
    )
    for i in range(1, 6):
        parser.add_argument(
            f"--model-path{i}",
            type=Path,
            default=None,
            help=(
                f"Baseline checkpoint {i} (optional). "
                "Provide multiple to compare several foundation runs."
            ),
        )

    parser.add_argument(
        "--cfg-dir",
        type=Path,
        default=None,
        help=(
            "Directory that contains cfgs.pkl for the baseline checkpoints. "
            "Defaults to the parent directory of the first model."
        ),
    )
    parser.add_argument(
        "--baseline-exp",
        type=str,
        default="foundation-baseline",
        help="Experiment name under logs/ea/ used to stage baseline models.",
    )

    # CSV output
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path (without extension). "
            "Default: analysis/foundation_eval_lean.csv"
        ),
    )

    # Evaluation settings
    parser.add_argument(
        "--eval-envs",
        type=int,
        default=2048,
        help="Number of parallel environments used during evaluation.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=6.0,
        help="Minimum commanded speed for evaluation sweeps.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=24.0,
        help="Maximum commanded speed for evaluation sweeps.",
    )

    # Training settings for per-URDF policies
    parser.add_argument(
        "--train-envs",
        type=int,
        default=16384,
        help="Number of parallel environments used during training.",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=1000,
        help="Number of PPO iterations for per-URDF training.",
    )
    parser.add_argument(
        "--train-repeats",
        type=int,
        default=1,
        help="Number of independent training runs per URDF.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device used for training (e.g. 'cuda:0' or 'cpu').",
    )

    # Logging verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (repeat for more detail).",
    )

    args = parser.parse_args(args=argv)

    # Collect baseline models from flags
    baseline_models: List[Path] = []
    for i in range(1, 6):
        p = getattr(args, f"model_path{i}")
        if p is not None:
            baseline_models.append(p)

    # Fallback: legacy --model-path
    if not baseline_models and args.model_path is not None:
        baseline_models.append(args.model_path)

    if not baseline_models:
        parser.error(
            "You must provide at least one baseline checkpoint via "
            "--model-path or --model-path1..--model-path5."
        )

    args.baseline_models = baseline_models
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the command-line interface."""
    args = parse_args(argv)
    _configure_logging(args.verbose)

    run_pipeline(
        catalog_dir=args.catalog_dir,
        n_urdf=int(args.n_urdf),
        rng_seed=int(args.seed),
        baseline_models=args.baseline_models,
        cfg_dir=args.cfg_dir,
        baseline_exp=str(args.baseline_exp),
        csv_path=args.csv,
        eval_envs=int(args.eval_envs),
        vmin=float(args.vmin),
        vmax=float(args.vmax),
        train_envs=int(args.train_envs),
        train_iters=int(args.train_iters),
        train_repeats=int(args.train_repeats),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()
