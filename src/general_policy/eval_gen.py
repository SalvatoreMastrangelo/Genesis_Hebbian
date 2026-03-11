#!/usr/bin/env python3
"""
eval_gen.py

Evaluate a foundation (general) policy and per-morphology policies
on a catalog of Genesis winged-drone URDFs.

High-level workflow
-------------------
1. Build or load a catalog of URDF files (drone morphologies).
2. For each *baseline* checkpoint (foundation policy):
   - Resolve checkpoint and config paths so :func:`eval.evaluation`
     can load the baseline model in read-only mode.
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
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import genesis as gs
from filelock import FileLock
from tensorboard.backend.event_processing import event_accumulator

from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME
from winged_drone_train.eval import evaluation, safe_urdf_stem
from general_policy.catalog import build_catalog
from drone_making import UrdfMaker


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
# new constant for evolution-style logs
EA_ROOT = LOG_ROOT / "ea"
ANALYSIS_ROOT = Path("analysis").expanduser().resolve()
PLOT_FILES = (
    "total_plot.png",
    "joint_heatmap_sweep.png",
    "joint_heatmap_twist.png",
)

def get_eval_root(exp_name: str) -> Path:
    """
    Root of the human-readable evaluation directory:
    logs/<exp_name>_evaluation/
    """
    return LOG_ROOT / f"{exp_name}_evaluation"

# Sentinel used when something goes wrong with an energy measurement
INVALID_ENERGY = 10
# Minimal progress below which speed/energy are set to sentinel in CSV
MINIMAL_PROGRESS_CSV = 250.0
DEFAULT_EXTRA_GENOME = list(STANDARD_MYDRONE_GENOME)


# =============================================================================
#  PARALLELISM SELECTION (Ray or serial)
# =============================================================================


def _want_parallel() -> bool:
    """
    Decide whether to run in parallel (Ray) or serial.

    We keep serial when a single GPU is available, regardless of GA_PARALLEL.
    """
    if torch.cuda.device_count() <= 1:
        return False
    flag = os.getenv("GA_PARALLEL", "auto").lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    return True


USE_PARALLEL = _want_parallel()

if USE_PARALLEL:
    import ray  # type: ignore[import]

    ray_address = os.getenv("RAY_ADDRESS", "").strip()
    ray_log_to_driver = os.getenv("RAY_LOG_TO_DRIVER", "").strip().lower() in ("1", "true", "yes")
    if ray_address:
        ray.init(address=ray_address, log_to_driver=ray_log_to_driver)
    else:
        ray.init(log_to_driver=ray_log_to_driver)


def _prepare_device_env(device: str) -> str:
    """
    Normalize a device string and set CUDA visibility accordingly.
    """
    dev = device.strip()
    low = dev.lower()
    if low.startswith("cuda:"):
        _, _, idx = low.partition(":")
        if idx:
            if os.getenv("CUDA_VISIBLE_DEVICES"):
                return "cuda:0"
            os.environ["CUDA_VISIBLE_DEVICES"] = idx
            return f"cuda:{idx}"
        return "cuda"
    if low == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    return dev


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


def _extract_urdf_name(value: str) -> Optional[str]:
    if not value:
        return None
    name = str(value).strip()
    if not name:
        return None
    match = re.search(r"\[[^\]]+\]", name)
    if match:
        return match.group(0)
    if ".urdf" in name:
        return Path(name).stem
    return name


def _parse_params_from_name(name: str) -> Optional[List[float]]:
    match = re.search(r"\[[^\]]+\]", name)
    if not match:
        return None
    raw = match.group(0).strip("[]")
    parts = [p.strip() for p in raw.split(",")]
    params: List[float] = []
    for p in parts:
        if not p:
            continue
        try:
            params.append(float(p))
        except Exception:
            return None
    return params if params else None


def list_urdfs_from_nsga(nsga_csv: Path, catalog_dir: Path) -> List[Path]:
    """
    Build a URDF list from a CSV (pareto/nsga-style), matching names to catalog_dir.
    """
    nsga_csv = nsga_csv.expanduser().resolve()
    if not nsga_csv.is_file():
        raise FileNotFoundError(f"CSV not found: {nsga_csv}")

    catalog_dir = catalog_dir.expanduser().resolve()
    urdfs: List[Path] = []
    seen: set[Path] = set()

    import csv as _csv

    with nsga_csv.open(newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            for col in ("urdf_name", "exp_name", "rep_exp_names"):
                if col not in row:
                    continue
                raw = row.get(col)
                if not raw:
                    continue
                for token in str(raw).split("|"):
                    name = _extract_urdf_name(token)
                    if not name:
                        continue
                    if os.sep in name or name.endswith(".urdf"):
                        p = Path(name)
                        if not p.is_absolute():
                            p = catalog_dir / p
                    else:
                        p = catalog_dir / f"{name}.urdf"
                    p = p.expanduser().resolve()
                    if p.is_file():
                        if p not in seen:
                            urdfs.append(p)
                            seen.add(p)
                        continue

                    params = _parse_params_from_name(name)
                    if params is None:
                        logger.warning("URDF from CSV not found and cannot parse params: %s", name)
                        continue
                    if len(params) != 15:
                        logger.warning("URDF params must be length 15, got %d: %s", len(params), name)
                        continue
                    if not gs._initialized:
                        gs.init(logging_level="error", backend=gs.gpu)
                    try:
                        path_str = UrdfMaker(params, out_dir=catalog_dir).create_urdf()
                        p = Path(path_str).expanduser().resolve()
                    except Exception as exc:
                        logger.warning("URDF generation failed for %s: %s", name, exc)
                        continue
                    if p.is_file() and p not in seen:
                        urdfs.append(p)
                        seen.add(p)

    return urdfs

def parse_urdf_params(urdf_path: Path) -> List[float]:
    """
    Extract the parameter array from a URDF filename.
    Example:
        [0.7, 3.5, 0.73, ... , -3].urdf
    Returns a list of floats.
    """
    stem = urdf_path.stem  # e.g. "[0.7, 3.5, 0.73, ...]"
    # Strip surrounding brackets.
    clean = stem.strip("[]")
    # Split by comma.
    parts = clean.split(",")
    # Convert each token to float if possible.
    out = []
    for x in parts:
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out

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
    cfg_path:
        Optional path to ``cfgs.pkl`` used for evaluation.
    """
    exp_name: str
    ckpt_index: int
    model_path: Path
    log_dir: Path
    cfg_path: Optional[Path] = None


def stage_baseline_checkpoint(exp_name: str, model_path: Path, cfg_dir: Optional[Path]):
    """
    Read-only baseline resolver.

    No files are copied or staged; we only infer the checkpoint index from
    the filename and keep source paths for downstream evaluation.
    """
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    m = re.search(r"model_(\d+)\.pt$", model_path.name)
    index = int(m.group(1)) if m else 0

    return StagedCheckpoint(
        exp_name=exp_name,
        ckpt_index=index,   # evaluation loads model_{ckpt-1}.pt
        model_path=model_path,
        log_dir=model_path.parent,
    )


def _resolve_cfg_path(
    exp_name: str,
    model_path: Optional[Path],
    cfg_dir: Optional[Path],
) -> Optional[Path]:
    candidates: List[Path] = []
    if model_path is not None:
        model_dir = model_path.expanduser().resolve().parent
        candidates.append(model_dir / "cfgs.pkl")
        candidates.append(model_dir.parent / "cfgs.pkl")
    if cfg_dir is not None:
        candidates.append(Path(cfg_dir).expanduser().resolve() / "cfgs.pkl")
    candidates.append(EA_ROOT / exp_name / "cfgs.pkl")

    for path in candidates:
        if path.is_file():
            return path

    if candidates:
        logger.warning("No cfgs.pkl found for model=%s; tried %s", model_path, candidates)
    return None

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

def _extract_reward_curve(
    log_dir: Path,
    train_iters: int,
    n_points: int = 20,
    win_frac: float = 0.05,
) -> Dict[str, float]:
    """
    Extract reward values at 5%, 10%, …, 100% of training using a moving
    average over a window equal to `win_frac` of total training steps.

    Assumptions
    -----------
    - TensorBoard scalar tag is exactly "Train/mean_reward".
    - Window is defined in training steps (± half-window around target step).
    - If the window has no events (sparse logging), we fall back to the
      nearest event.

    Returns
    -------
    dict
        Keys are "rew_5pct", "rew_10pct", ..., "rew_100pct".
        If the curve cannot be extracted, an empty dict is returned.
    """
    if not log_dir.exists():
        return {}

    ea = event_accumulator.EventAccumulator(str(log_dir))
    try:
        ea.Reload()
    except Exception:
        return {}

    key = "Train/mean_reward"
    if key not in ea.Tags().get("scalars", []):
        return {}

    events = ea.Scalars(key)
    if not events:
        return {}

    steps = np.array([e.step for e in events], dtype=np.int64)
    vals = np.array([e.value for e in events], dtype=np.float64)

    win_steps = max(1, int(round(train_iters * win_frac)))
    half = win_steps // 2

    out: Dict[str, float] = {}
    for frac in range(1, n_points + 1):
        target = int(round(train_iters * frac / n_points))
        lo, hi = target - half, target + half
        mask = (steps >= lo) & (steps <= hi)

        if mask.any():
            out[f"rew_{frac * 5}pct"] = float(np.nanmean(vals[mask]))
        else:
            idx = int(np.argmin(np.abs(steps - target)))
            out[f"rew_{frac * 5}pct"] = float(vals[idx])

    return out


def _apply_progress_mask(
    top_vel: Dict[str, float],
    top_eff: Dict[str, float],
    top_prog: Dict[str, float],
    extra: Dict[str, Any],
    minimal_progress: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Re-select top velocity/efficiency using the smoothed progress mask.

    Matches the CLI logic in winged_drone_train/eval.py:
    - if max(p_s) <= minimal_progress: return all zeros
    - else: select velocity/energy/progress using only p_s >= minimal_progress
    """
    p_s = extra.get("p_s")
    v_s = extra.get("v_s")
    E_s = extra.get("E_s")
    if p_s is None or v_s is None or E_s is None:
        return top_vel, top_eff, top_prog

    p_s = np.asarray(p_s, dtype=float)
    v_s = np.asarray(v_s, dtype=float)
    E_s = np.asarray(E_s, dtype=float)
    if p_s.size == 0:
        return top_vel, top_eff, top_prog

    mask = np.where(p_s >= minimal_progress)[0]
    if mask.size == 0:
        invalid = {"mean_v": 0.0, "mean_E": float(INVALID_ENERGY), "mean_progress": 0.0}
        return invalid, invalid, top_prog

    idx_v = int(mask[np.argmax(v_s[mask])])
    idx_e = int(mask[np.argmin(E_s[mask])])
    top_vel = {
        "mean_v": float(v_s[idx_v]),
        "mean_E": float(E_s[idx_v]),
        "mean_progress": float(p_s[idx_v]),
    }
    top_eff = {
        "mean_v": float(v_s[idx_e]),
        "mean_E": float(E_s[idx_e]),
        "mean_progress": float(p_s[idx_e]),
    }
    return top_vel, top_eff, top_prog


def evaluate_single(
    exp_name: str,
    urdf_file: Path,
    ckpt: int,
    eval_envs: int,
    vmin: float,
    vmax: float,
    obs_genome: Optional[bool] = None,
    model_path: Optional[Path] = None,
    cfg_path: Optional[Path] = None,
    eval_dir: Optional[Path] = None,
    clean_urdf_stem: Optional[str] = None,
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
    clean_label = clean_urdf_stem or safe_urdf_stem(urdf_file)
    if eval_dir is None:
        eval_dir = EA_ROOT / exp_name / f"eval_{clean_label}"

    logger.debug(
        "Evaluating exp=%s ckpt=%d on urdf=%s (envs=%d, vmin=%.1f, vmax=%.1f, eval_dir=%s)",
        exp_name,
        ckpt,
        urdf_file,
        eval_envs,
        vmin,
        vmax,
        eval_dir,
    )
    print(
        f"[eval_gen] evaluation start exp={exp_name} ckpt={ckpt} "
        f"urdf={urdf_file.name} -> {eval_dir}"
    )

    # evaluation() returns:
    #   top_vel, top_eff, top_prog, max_p, extra
    # evaluation always loads from logs/ea/<exp_name>
    custom_policy_path = None
    if model_path is not None:
        custom_policy_path = str(Path(model_path).expanduser().resolve())

    top_vel, top_eff, top_prog, max_p, extra = evaluation(
        exp_name=str(exp_name),
        urdf_file=str(urdf_file),
        ckpt=int(ckpt),
        envs=int(eval_envs),
        vmin=float(vmin),
        vmax=float(vmax),
        return_arrays=True,
        obs_genome=obs_genome,
        cfg_path=str(cfg_path) if cfg_path is not None else None,
        save_plots=True,
        custom_policy_path=custom_policy_path,
        eval_dir=str(eval_dir),
    )
    top_vel, top_eff, top_prog = _apply_progress_mask(
        top_vel,
        top_eff,
        top_prog,
        extra,
        minimal_progress=MINIMAL_PROGRESS_CSV,
    )

    print("[EVAL 7] Evaluation loop completed")
    if extra.get("plot_paths"):
        print(f"[eval_gen] eval saved plots: {extra['plot_paths']}")
    if extra.get("eval_dir"):
        try:
            contents = sorted(p.name for p in Path(extra["eval_dir"]).iterdir())
            print(f"[eval_gen] eval dir contents {extra['eval_dir']}: {contents}")
        except Exception as exc:
            print(f"[eval_gen][warn] cannot list eval dir {extra.get('eval_dir')}: {exc}")

    # Build a fitness triple:
    #   - maximize speed (velocity at best-speed operating point)
    #   - minimize energy: take MIN energy and flip sign
    #   - maximize progress (distance at best-progress operating point)
    speed = float(top_vel["mean_v"])
    neg_energy = -float(top_eff["mean_E"])
    progress = float(top_prog["mean_progress"])

    fitness = FitnessTriple(speed=speed, neg_energy=neg_energy, progress=progress)

    # Reward statistics and auxiliary info
    eval_reward_mean = float(extra.get("eval_reward_mean", float("nan")))
    if not np.isfinite(eval_reward_mean):
        eval_reward_mean = float(extra.get("final_reward", float("nan")))
    if not np.isfinite(eval_reward_mean):
        eval_reward_mean = 0.0
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
        eval_reward_mean,
    )

    print("[EVAL 8] EvalSummary created")

    return EvalSummary(fitness=fitness, reward_ep_mean=eval_reward_mean, metadata=metadata)


def copy_baseline_eval_images(
    source_exp: str,
    eval_name: str,
    urdf_file: Path,
    urdf_idx: int,
    gp_idx: int,
) -> None:
    """
    Copy baseline evaluation plots into the human-readable tree.

    Source: logs/ea/<source_exp>/gpXX_eval_<clean_stem>/
    Dest:   logs/<eval_name>_evaluation/general_policy/gpXX/urdf_XXX_eval/
    """
    clean_stem = safe_urdf_stem(urdf_file)
    src_dir = EA_ROOT / source_exp / f"gp{gp_idx:02d}_eval_{clean_stem}"
    dst_dir = (
        get_eval_root(eval_name)
        / "general_policy"
        / f"gp{gp_idx:02d}"
        / f"urdf_{urdf_idx:03d}_eval"
    )

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"[copy][baseline] {src_dir} -> {dst_dir}")

    for name in PLOT_FILES:
        src = src_dir / name
        dst = dst_dir / name
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"[copy][baseline] copied {src} -> {dst}")
        else:
            print(f"[copy][baseline][missing] expected plot not found: {src}")
    try:
        contents = sorted(p.name for p in src_dir.iterdir())
        print(f"[copy][baseline] src contents: {src_dir} -> {contents}")
    except Exception as exc:
        print(f"[copy][baseline][warn] cannot list src {src_dir}: {exc}")


def copy_individual_policy_run(
    exp_train: str,
    eval_name: str,
    urdf_stem: str,
    urdf_idx: int,
    rep: int,
) -> None:

    clean_stem = safe_urdf_stem(urdf_stem, already_clean=True)
    src_root = EA_ROOT / exp_train
    dst_root = get_eval_root(eval_name) / "individual_policy" / f"urdf_{urdf_idx:03d}_{rep+1}"

    if not src_root.exists():
        print(f"[copy][trained][missing] source run not found: {src_root}")
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"[copy][trained] copying run {src_root} -> {dst_root}")

    for item in src_root.iterdir():
        dst = dst_root / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    # Copy evaluation plots
    eval_src = src_root / f"eval_{clean_stem}"
    eval_dst = dst_root / "eval"
    eval_dst.mkdir(parents=True, exist_ok=True)

    for name in PLOT_FILES:
        src = eval_src / name
        dst = eval_dst / name
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"[copy][trained] copied {src} -> {dst}")
        else:
            print(f"[copy][trained][missing] expected plot not found: {src}")
    try:
        contents = sorted(p.name for p in eval_src.iterdir())
        print(f"[copy][trained] src contents: {eval_src} -> {contents}")
    except Exception as exc:
        print(f"[copy][trained][warn] cannot list src {eval_src}: {exc}")



# =============================================================================
#  Lean CSV writer
# =============================================================================

class LeanCSV:
    """
    Minimal CSV writer: append-only, supports raw (per-repeat) and
    aggregate rows.

    For ``B`` baseline checkpoints and ``T`` trained checkpoints the
    header has the following structure:

        row_kind, rep_idx,
        urdf_stem,
        f_speed_baseline1, f_negE_baseline1, f_prog_baseline1, reward_ep_mean_baseline1,
        ...,
        f_speed_trained1,  f_negE_trained1,  f_prog_trained1,  reward_ep_mean_trained1,
        ...

    The class only appends rows; it never attempts to read or update
    existing contents.
    """

    def __init__(
        self,
        path: Path,
        n_baselines: int,
        n_trained: int,
        minimal_progress: float = MINIMAL_PROGRESS_CSV,
    ) -> None:
        self.path = path.with_suffix(".csv").expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_baselines = max(0, int(n_baselines))
        self.n_trained = max(0, int(n_trained))
        self.minimal_progress = float(minimal_progress)
        self.columns = self._build_columns()
        self._ensure_header()

    # ------------------------------------------------------------------ #
    # Header / schema helpers
    # ------------------------------------------------------------------ #

    def _build_columns(self) -> List[str]:
        header = [
            "row_kind",
            "rep_idx",
            "urdf_stem",
            "urdf_params",
            "eval_duration_s",
            "train_duration_s",
        ]
        # Training reward curve (5% steps, moving average)
        header += [f"rew_{i * 5}pct" for i in range(1, 21)]
        # Baseline columns
        for i in range(self.n_baselines):
            k = i + 1
            header += [
                f"f_speed_baseline{k}",
                f"f_negE_baseline{k}",
                f"f_prog_baseline{k}",
                f"reward_ep_mean_baseline{k}",
            ]
        # Trained columns (per repeat)
        for i in range(self.n_trained):
            k = i + 1
            header += [
                f"f_speed_trained{k}",
                f"f_negE_trained{k}",
                f"f_prog_trained{k}",
                f"reward_ep_mean_trained{k}",
            ]
        # Aggregate mean across repeats
        header += [
            "f_speed_trained_mean",
            "f_negE_trained_mean",
            "f_prog_trained_mean",
            "reward_ep_mean_trained_mean",
        ]
        return header

    def _ensure_header(self) -> None:
        lock = FileLock(str(self.path) + ".lock")
        with lock:
            if not self.path.exists():
                self.path.write_text(",".join(self.columns) + "\n")

    def _fmt(self, value: Optional[float], digits: int = 6) -> str:
        if value is None or not np.isfinite(value):
            return "nan"
        return f"{float(value):.{digits}f}"

    # ------------------------------------------------------------------ #

    def _gate(self, f: FitnessTriple) -> FitnessTriple:
        """
        If progress is below threshold, keep progress but set speed/energy to sentinel.
        """
        if f.progress < self.minimal_progress:
            return FitnessTriple(
                speed=0.0,
                neg_energy=-INVALID_ENERGY,
                progress=f.progress,
            )
        return f

    def _append_row(self, row: Dict[str, Any]) -> None:
        line = ",".join(str(row.get(col, "nan")) for col in self.columns)
        lock = FileLock(str(self.path) + ".lock")
        with lock:
            with self.path.open("a") as f:
                f.write(line + "\n")

    def _base_row(
        self,
        row_kind: str,
        rep_idx: int,
        urdf_stem: str,
        urdf_params: str,
        eval_duration_s: Optional[float],
        train_duration_s: Optional[float],
        baseline_fitness: Sequence[FitnessTriple],
        baseline_rewards: Sequence[float],
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "row_kind": row_kind,
            "rep_idx": int(rep_idx),
            "urdf_stem": urdf_stem,
            "urdf_params": urdf_params,
            "eval_duration_s": self._fmt(eval_duration_s, digits=3),
            "train_duration_s": self._fmt(train_duration_s, digits=3),
        }

        # Baseline metrics
        for i in range(self.n_baselines):
            k = i + 1
            if i < len(baseline_fitness):
                f = self._gate(baseline_fitness[i])
                r = baseline_rewards[i] if i < len(baseline_rewards) else float("nan")
            else:
                f = FitnessTriple(speed=float("nan"), neg_energy=float("nan"), progress=float("nan"))
                r = float("nan")

            row[f"f_speed_baseline{k}"] = self._fmt(f.speed)
            row[f"f_negE_baseline{k}"] = self._fmt(f.neg_energy)
            row[f"f_prog_baseline{k}"] = self._fmt(f.progress)
            row[f"reward_ep_mean_baseline{k}"] = self._fmt(r)
        return row

    def append_rep(
        self,
        urdf_stem: str,
        urdf_params: str,
        baseline_fitness: Sequence[FitnessTriple],
        baseline_rewards: Sequence[float],
        rep_idx: int,
        rep_fitness: FitnessTriple,
        rep_reward: float,
        rep_reward_curve: Optional[Dict[str, float]],
        eval_duration_s: Optional[float],
        train_duration_s: Optional[float],
    ) -> None:
        """
        Append a raw per-repeat row for a single URDF.
        """
        row = self._base_row(
            row_kind="rep",
            rep_idx=rep_idx,
            urdf_stem=urdf_stem,
            urdf_params=urdf_params,
            eval_duration_s=eval_duration_s,
            train_duration_s=train_duration_s,
            baseline_fitness=baseline_fitness,
            baseline_rewards=baseline_rewards,
        )

        # Initialize trained columns as NaN
        for i in range(self.n_trained):
            k = i + 1
            row[f"f_speed_trained{k}"] = "nan"
            row[f"f_negE_trained{k}"] = "nan"
            row[f"f_prog_trained{k}"] = "nan"
            row[f"reward_ep_mean_trained{k}"] = "nan"

        # Fill only the current repetition slot
        if 1 <= rep_idx <= self.n_trained:
            f = self._gate(rep_fitness)
            row[f"f_speed_trained{rep_idx}"] = self._fmt(f.speed)
            row[f"f_negE_trained{rep_idx}"] = self._fmt(f.neg_energy)
            row[f"f_prog_trained{rep_idx}"] = self._fmt(f.progress)
            row[f"reward_ep_mean_trained{rep_idx}"] = self._fmt(rep_reward)

        # Training reward curve for this repeat
        curve = rep_reward_curve or {}
        for i in range(1, 21):
            key = f"rew_{i * 5}pct"
            row[key] = self._fmt(curve.get(key, float("nan")))

        # Aggregate mean columns remain empty for raw rows
        row["f_speed_trained_mean"] = "nan"
        row["f_negE_trained_mean"] = "nan"
        row["f_prog_trained_mean"] = "nan"
        row["reward_ep_mean_trained_mean"] = "nan"

        self._append_row(row)
        print(f"[csv] appended rep row for {urdf_stem} (rep={rep_idx}) to {self.path}")

    def append_agg(
        self,
        urdf_stem: str,
        urdf_params: str,
        baseline_fitness: Sequence[FitnessTriple],
        baseline_rewards: Sequence[float],
        trained_fitness: Sequence[FitnessTriple],
        trained_rewards: Sequence[float],
        trained_reward_curves: Sequence[Dict[str, float]],
        eval_duration_s: Optional[float],
        train_duration_s: Optional[float],
    ) -> None:
        """
        Append an aggregate row (mean across repeats).
        """
        row = self._base_row(
            row_kind="agg",
            rep_idx=-1,
            urdf_stem=urdf_stem,
            urdf_params=urdf_params,
            eval_duration_s=eval_duration_s,
            train_duration_s=train_duration_s,
            baseline_fitness=baseline_fitness,
            baseline_rewards=baseline_rewards,
        )

        gated_trained: List[FitnessTriple] = []
        for i in range(self.n_trained):
            k = i + 1
            if i < len(trained_fitness):
                f = self._gate(trained_fitness[i])
                r = trained_rewards[i] if i < len(trained_rewards) else float("nan")
            else:
                f = FitnessTriple(speed=float("nan"), neg_energy=float("nan"), progress=float("nan"))
                r = float("nan")
            gated_trained.append(f)
            row[f"f_speed_trained{k}"] = self._fmt(f.speed)
            row[f"f_negE_trained{k}"] = self._fmt(f.neg_energy)
            row[f"f_prog_trained{k}"] = self._fmt(f.progress)
            row[f"reward_ep_mean_trained{k}"] = self._fmt(r)

        if gated_trained:
            speed_mean = np.nanmean([f.speed for f in gated_trained])
            negE_mean = np.nanmean([f.neg_energy for f in gated_trained])
            prog_mean = np.nanmean([f.progress for f in gated_trained])
        else:
            speed_mean = float("nan")
            negE_mean = float("nan")
            prog_mean = float("nan")

        reward_mean = np.nanmean(np.asarray(trained_rewards, dtype=float)) if trained_rewards else float("nan")

        row["f_speed_trained_mean"] = self._fmt(speed_mean)
        row["f_negE_trained_mean"] = self._fmt(negE_mean)
        row["f_prog_trained_mean"] = self._fmt(prog_mean)
        row["reward_ep_mean_trained_mean"] = self._fmt(reward_mean)

        # Aggregate training reward curve (mean across repeats)
        for i in range(1, 21):
            key = f"rew_{i * 5}pct"
            vals = []
            for curve in trained_reward_curves:
                if curve is None:
                    continue
                try:
                    val = float(curve.get(key, float("nan")))
                except Exception:
                    continue
                if np.isfinite(val):
                    vals.append(val)
            row[key] = self._fmt(np.nanmean(vals) if vals else float("nan"))

        self._append_row(row)
        print(f"[csv] appended agg row for {urdf_stem} to {self.path}")


# =============================================================================
#  End-to-end pipeline
# =============================================================================


def _process_urdf_impl(
    urdf: Path,
    idx: int,
    staged: Sequence[StagedCheckpoint],
    exp_name: str,
    saving_path: str,
    csv_path: Optional[Path],
    eval_envs: int,
    vmin: float,
    vmax: float,
    train_envs: int,
    train_iters: int,
    train_repeats: int,
    device: str,
) -> Dict[str, Any]:
    device = _prepare_device_env(device)
    csv_writer = (
        LeanCSV(
            path=csv_path,
            n_baselines=len(staged),
            n_trained=train_repeats,
            minimal_progress=MINIMAL_PROGRESS_CSV,
        )
        if csv_path is not None
        else None
    )

    logger.info("=== [%d] %s ===", idx, urdf.name)

    clean_stem = safe_urdf_stem(urdf)
    print(f"[eval_gen] URDF stem raw='{urdf.stem}' clean='{clean_stem}'")

    eval_t0 = time.time()
    baseline_fitness: List[FitnessTriple] = []
    baseline_rewards: List[float] = []

    for j, sc in enumerate(staged, start=1):
        try:
            summary = evaluate_single(
                exp_name=exp_name,
                urdf_file=urdf,
                ckpt=sc.ckpt_index,
                eval_envs=eval_envs,
                vmin=vmin,
                vmax=vmax,
                obs_genome=None,
                model_path=sc.model_path,
                cfg_path=sc.cfg_path,
                eval_dir=EA_ROOT / exp_name / f"gp{j:02d}_eval_{clean_stem}",
                clean_urdf_stem=clean_stem,
            )
        except Exception as exc:
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
        copy_baseline_eval_images(
            source_exp=exp_name,
            eval_name=saving_path,
            urdf_file=urdf,
            urdf_idx=idx,
            gp_idx=j,
        )

    print("[EVAL 9] Baseline evaluation completed for all checkpoints")
    eval_duration = time.time() - eval_t0
    print(f"[timing] baseline eval duration: {eval_duration:.2f}s for URDF {idx}")

    trained_fitness: List[FitnessTriple] = []
    trained_rewards: List[float] = []
    trained_reward_curves: List[Dict[str, float]] = []

    urdf_params = parse_urdf_params(urdf)
    train_t0 = time.time()
    for rep in range(train_repeats):
        exp_train = f"{saving_path}_urdf{idx:03d}_rep{rep+1}"
        rep_t0 = time.time()

        logger.info(
            "[train %d/%d] exp=%s  runtime_seed=os_entropy  envs=%d  iters=%d",
            rep + 1,
            train_repeats,
            exp_train,
            train_envs,
            train_iters,
        )
        print(f"[EVAL 10] Starting training for exp={exp_train} with runtime_seed=os_entropy")

        rep_fitness: FitnessTriple
        rep_reward: float
        rep_reward_curve: Dict[str, float] = {}
        try:
            cmd = [
                "python",
                "-c",
                (
                    "from winged_drone_train.train import training; "
                    f"training(exp_name={exp_train!r}, urdf_file={str(urdf)!r}, "
                    f"num_envs={int(train_envs)}, max_iterations={int(train_iters)}, "
                    "parent_exp=None, parent_ckpt=None, "
                    f"device={device!r})"
                ),
            ]
            env = os.environ.copy()
            env["TAICHI_CACHE_DIR"] = "/tmp/taichi_cache"
            subprocess.run(cmd, check=True, env=env)
        except Exception as exc:  # pragma: no cover - robust to training failures
            logger.error(
                "Training failed for urdf=%s (exp=%s): %s",
                urdf.name,
                exp_train,
                exc,
            )
            rep_fitness = FitnessTriple(
                speed=0.0,
                neg_energy=-INVALID_ENERGY,
                progress=0.0,
            )
            rep_reward = float("nan")
            trained_fitness.append(rep_fitness)
            trained_rewards.append(rep_reward)
            trained_reward_curves.append(rep_reward_curve)
            rep_duration = time.time() - rep_t0
            if csv_writer is not None:
                csv_writer.append_rep(
                    urdf_stem=urdf.stem,
                    urdf_params=str(urdf_params),
                    baseline_fitness=baseline_fitness,
                    baseline_rewards=baseline_rewards,
                    rep_idx=rep + 1,
                    rep_fitness=rep_fitness,
                    rep_reward=rep_reward,
                    rep_reward_curve=rep_reward_curve,
                    eval_duration_s=eval_duration,
                    train_duration_s=rep_duration,
                )
            continue

        rep_reward_curve = _extract_reward_curve(
            EA_ROOT / exp_train,
            train_iters,
            n_points=20,
            win_frac=0.05,
        )

        print(f"[EVAL 11] Starting evaluation for exp={exp_train} with checkpoint={train_iters}")

        try:
            summary = evaluate_single(
                exp_name=exp_train,
                urdf_file=urdf,
                ckpt=train_iters,
                eval_envs=eval_envs,
                vmin=vmin,
                vmax=vmax,
                obs_genome=None,
                eval_dir=EA_ROOT / exp_train / f"eval_{clean_stem}",
                clean_urdf_stem=clean_stem,
            )
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Evaluation of trained policy failed for urdf=%s (exp=%s): %s",
                urdf.name,
                exp_train,
                exc,
            )
            rep_fitness = FitnessTriple(
                speed=0.0,
                neg_energy=-INVALID_ENERGY,
                progress=0.0,
            )
            rep_reward = float("nan")
        else:
            rep_fitness = summary.fitness
            rep_reward = summary.reward_ep_mean

            copy_individual_policy_run(
                exp_train=exp_train,
                eval_name=saving_path,
                urdf_stem=clean_stem,
                urdf_idx=idx,
                rep=rep,
            )

        trained_fitness.append(rep_fitness)
        trained_rewards.append(rep_reward)
        trained_reward_curves.append(rep_reward_curve)
        rep_duration = time.time() - rep_t0
        if csv_writer is not None:
            csv_writer.append_rep(
                urdf_stem=urdf.stem,
                urdf_params=str(urdf_params),
                baseline_fitness=baseline_fitness,
                baseline_rewards=baseline_rewards,
                rep_idx=rep + 1,
                rep_fitness=rep_fitness,
                rep_reward=rep_reward,
                rep_reward_curve=rep_reward_curve,
                eval_duration_s=eval_duration,
                train_duration_s=rep_duration,
            )

    train_duration = time.time() - train_t0
    print(f"[timing] training+trained eval duration: {train_duration:.2f}s for URDF {idx}")

    if csv_writer is not None:
        csv_writer.append_agg(
            urdf_stem=urdf.stem,
            urdf_params=str(urdf_params),
            baseline_fitness=baseline_fitness,
            baseline_rewards=baseline_rewards,
            trained_fitness=trained_fitness,
            trained_rewards=trained_rewards,
            trained_reward_curves=trained_reward_curves,
            eval_duration_s=eval_duration,
            train_duration_s=train_duration,
        )

    return {
        "urdf_stem": urdf.stem,
        "urdf_params": str(urdf_params),
        "baseline_fitness": baseline_fitness,
        "baseline_rewards": baseline_rewards,
        "trained_fitness": trained_fitness,
        "trained_rewards": trained_rewards,
        "eval_duration_s": eval_duration,
        "train_duration_s": train_duration,
    }


if USE_PARALLEL:

    @ray.remote  # type: ignore[misc]
    def _process_urdf_remote(*args, **kwargs):
        return _process_urdf_impl(*args, **kwargs)


def run_pipeline(
    catalog_dir: Path,
    n_urdf: int,
    urdf_seed: int,
    baseline_models: Sequence[Path],
    cfg_dir: Optional[Path],
    exp_name: str,      # foundation-exp (per evaluation)
    saving_path: str,   # exp (per output)
    csv_path: Optional[Path],
    nsga_csv: Optional[Path],
    eval_baselines: bool,
    eval_envs,
    vmin,
    vmax,
    train_envs,
    train_iters,
    train_repeats,
    device,
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
    urdf_seed:
        Random seed used only for URDF catalog generation.
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
        Number of independent training runs per URDF.
    device:
        Device string for training, e.g. ``"cuda:0"`` or ``"cpu"``.
    """
    catalog_dir = catalog_dir.expanduser().resolve()
    catalog_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Copy meshes directory into catalog_dir                             #
    # ------------------------------------------------------------------ #
    # Required because URDFs reference meshes/xxx.obj
    meshes_src = Path("/workspace/Genesis/src/urdf_generated/meshes")
    meshes_dst = catalog_dir / "meshes"

    if meshes_src.is_dir():
        shutil.copytree(meshes_src, meshes_dst, dirs_exist_ok=True)
        logger.info("Copied meshes directory to %s", meshes_dst)
    else:
        logger.error("Meshes directory not found at %s", meshes_src)
        raise FileNotFoundError(f"Meshes directory missing: {meshes_src}")


    # ------------------------------------------------------------------ #
    # Build / load catalog                                               #
    # ------------------------------------------------------------------ #
    if nsga_csv is not None:
        urdf_list = list_urdfs_from_nsga(nsga_csv, catalog_dir)
        if not urdf_list:
            raise RuntimeError(
                f"No URDFs resolved from nsga.csv: {nsga_csv}. "
                "Check that exp_name/rep_exp_names entries match files in the catalog."
            )
        logger.info("Loaded %d URDFs from CSV %s", len(urdf_list), nsga_csv)
    else:
        if n_urdf <= 0:
            raise RuntimeError(
                "nsga.csv path is None, so a random catalog is required. "
                "Please set --n-urdf > 0."
            )
        logger.info(
            "Building a fresh URDF catalog with %d entries into %s",
            n_urdf,
            catalog_dir,
        )
        build_catalog(
            catalog_dir=catalog_dir,
            n=n_urdf,
            seed=urdf_seed,
            include_standard_mydrone=True,
        )
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
    if not eval_baselines:
        baseline_models = []
    else:
        baseline_models = [p.expanduser().resolve() for p in baseline_models]
    if baseline_models and cfg_dir is None:
        cfg_dir = baseline_models[0].parent

    if eval_baselines:
        print(f"[eval_gen] Baseline logs loaded from: {EA_ROOT / exp_name}")
    print(f"[eval_gen] Evaluation artifacts will be copied to: {get_eval_root(saving_path)}")

    # Baseline checkpoints: read-only, no staging
    staged = []
    for model_path in baseline_models:
        m = re.search(r"model_(\d+)\.pt$", model_path.name)
        ckpt_index = int(m.group(1)) if m else 0
        cfg_path = _resolve_cfg_path(exp_name, model_path, cfg_dir)
        staged.append(StagedCheckpoint(
            exp_name=exp_name,      # foundation-exp (folder in logs/ea/)
            ckpt_index=ckpt_index,
            model_path=model_path,
            log_dir=model_path.parent,  # read-only model dir
            cfg_path=cfg_path,
        ))

    # CSV initialization
    if csv_path is None:
        csv_path = get_eval_root(saving_path) / "analysis" / "foundation_eval_lean"

    csv_writer = LeanCSV(
        path=csv_path,
        n_baselines=len(staged),
        n_trained=train_repeats,
        minimal_progress=MINIMAL_PROGRESS_CSV,
    )
    logger.info("Writing lean CSV to %s", csv_writer.path)

    # ------------------------------------------------------------------ #
    # Main loop over URDFs                                               #
    # ------------------------------------------------------------------ #
    if USE_PARALLEL:
        gpu_needed = str(device).lower().startswith("cuda")
        refs = []
        max_in_flight = 1
        try:
            if gpu_needed:
                max_in_flight = int(ray.cluster_resources().get("GPU", 0))
            else:
                max_in_flight = int(ray.cluster_resources().get("CPU", 0))
        except Exception:
            max_in_flight = 0
        if max_in_flight <= 0:
            max_in_flight = torch.cuda.device_count() if gpu_needed else 1
        max_in_flight = max(1, max_in_flight)
        logger.info("Ray parallelism: max in-flight URDF tasks = %d", max_in_flight)

        for idx, urdf in enumerate(urdf_list, start=1):
            ref = _process_urdf_remote.options(num_gpus=1 if gpu_needed else 0).remote(
                urdf=urdf,
                idx=idx,
                staged=staged,
                exp_name=exp_name,
                saving_path=saving_path,
                csv_path=csv_writer.path,
                eval_envs=int(eval_envs),
                vmin=float(vmin),
                vmax=float(vmax),
                train_envs=int(train_envs),
                train_iters=int(train_iters),
                train_repeats=int(train_repeats),
                device=str(device),
            )
            refs.append(ref)

            if len(refs) >= max_in_flight:
                done, refs = ray.wait(refs, num_returns=1)
                ray.get(done[0])

        while refs:
            done, refs = ray.wait(refs, num_returns=1)
            ray.get(done[0])

        logger.info("Pipeline finished successfully.")
        return

    for idx, urdf in enumerate(urdf_list, start=1):
        _process_urdf_impl(
            urdf=urdf,
            idx=idx,
            staged=staged,
            exp_name=exp_name,
            saving_path=saving_path,
            csv_path=csv_writer.path,
            eval_envs=int(eval_envs),
            vmin=float(vmin),
            vmax=float(vmax),
            train_envs=int(train_envs),
            train_iters=int(train_iters),
            train_repeats=int(train_repeats),
            device=str(device),
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
        "--urdf-seed",
        "--seed",
        dest="urdf_seed",
        type=int,
        default=0,
        help="Random seed used only for URDF catalog generation.",
    )

    # Baseline checkpoints (up to 6 for convenience)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Legacy convenience flag for a single baseline checkpoint. "
            "Equivalent to --model-path1 when given."
        ),
    )
    for i in range(1, 7):
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
        "--exp",
        type=str,
        required=True,
        help="Name/id of the foundation experiment (e.g. 7). "
             "Used as model folder name AND baseline-exp."
    )
    parser.add_argument(
        "--foundation-exp",
        type=str,
        required=False,
        help="Experiment ID of the foundation policy (logs/ea/<id>/)."
    )
    parser.add_argument(
        "--eval-baselines",
        type=int,
        default=1,
        help="Set to 0 to skip baseline evaluations entirely.",
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
    parser.add_argument(
        "--nsga-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV path (e.g. pareto.csv). If provided, URDF names are "
            "read from the CSV and used as the evaluation catalog."
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
        default=30.0,
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
    for i in range(1, 7):
        p = getattr(args, f"model_path{i}")
        if p is not None:
            baseline_models.append(p)

    # Fallback: legacy --model-path
    if not baseline_models and args.model_path is not None:
        baseline_models.append(args.model_path)

    args.baseline_models = baseline_models
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the command-line interface."""
    args = parse_args(argv)
    _configure_logging(args.verbose)
    if bool(int(args.eval_baselines)) and not args.foundation_exp:
        raise SystemExit("--foundation-exp is required when --eval-baselines=1")

    run_pipeline(
        catalog_dir=args.catalog_dir,
        n_urdf=int(args.n_urdf),
        urdf_seed=int(args.urdf_seed),
        baseline_models=args.baseline_models,
        cfg_dir=args.cfg_dir,
        exp_name=(args.foundation_exp),
        saving_path=str(args.exp),
        csv_path=args.csv,
        nsga_csv=args.nsga_csv,
        eval_baselines=bool(int(args.eval_baselines)),
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
