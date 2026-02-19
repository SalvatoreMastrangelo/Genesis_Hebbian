#!/usr/bin/env python3
"""
evolution_nsga.py

NSGA-II co-design of drone morphology + control policy in a continuous
genome space, with:

  - Simulated Binary Crossover (SBX) in [0, 1]^D.
  - Polynomial mutation (bounded).
  - Policy inheritance between generations.
  - Fitness caching in a CSV "database".
  - Optional multi-GPU parallelism via Ray.
  - Post-hoc analysis and plotting utilities.

The genome lives in [0, 1]^D and is mapped to physical parameters by
`Chromosome_Drone.to_physical()`, which in turn drives `UrdfMaker`.
"""

from __future__ import annotations

import argparse
import gc
import datetime
import os
import shutil
import time
import random
import socket
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
import genesis as gs
import builtins
import psutil
from deap import base, creator, tools
from filelock import FileLock
from tensorboard.backend.event_processing import event_accumulator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Project imports (adapt to your package layout)
from drone_making import UrdfMaker
from chromosome_drone import Chromosome_Drone
from winged_drone_train.train import training
from winged_drone_train.eval import evaluation


# =============================================================================
#  GLOBAL GA CONFIGURATION (single place for all hyper-parameters)
# =============================================================================


@dataclass
class GAConfig:
    """
    All GA / NSGA-II hyper-parameters in one place.

    Edit this dataclass (or override fields from the CLI) to configure
    the search behaviour.
    """

    # --- Population & evolutionary budget ---------------------------------
    population_size: int = 40       # number of individuals per generation
    num_generations: int = 25       # number of generations to run

    # --- NSGA-II operators (continuous) -----------------------------------
    crossover_probability: float = 0.9   # probability of SBX crossover
    mutation_probability: float = 1.0 # 0.12    # probability of applying mutation
    eta_c: float = 20.0                  # SBX "spread" parameter (higher = more local)
    eta_m: float = 20.0                  # polynomial mutation parameter

    # --- RL training / evaluation -----------------------------------------
    gen_policy: bool = False
    policy_path: Optional[str] = None  # path to initial policy checkpoint
    train_iters_new: int = 700       # iterations for NEW morphologies
    train_iters_inherit: int = 200  # iterations when inheriting from a parent
    train_repetition: int = 1       # repeat train+eval N times (gen_policy=0)
    train_envs: int = 32768           # number of envs during training
    eval_envs: int = 8192            # number of envs during evaluation
    vmin: float = 6.0               # min commanded speed in evaluation
    vmax: float = 24.0              # max commanded speed in evaluation

    # --- Fitness shaping / invalid individuals ----------------------------
    # fail_value removed; fallback uses INVALID_* sentinels
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (vel, -energy, progress)

    # --- Progress threshold (minimal_p) -----------------------------------
    use_dynamic_p: bool = False      # if True: percentile-based threshold
    fixed_p: float = 250.0          # fallback / fixed threshold [m]
    pct_above: float = 50.0         # fraction of individuals above minimal_p

    # --- Policy inheritance -----------------------------------------------
    inherit_policy: bool = False    # if True: offspring can inherit parent policy

    # --- Output / logging -------------------------------------------------
    csv_basename: str = "nsga"       # CSV filename (".csv" added automatically)
    run_name: Optional[str] = None   # Optional custom run name (defaults to timestamp)
    base_dir: str = "nsga"           # Root directory for all artifacts
    device: str = "cuda:0"           # Device passed to training/evaluation


# Default config used when no custom config is provided
DEFAULT_GA_CONFIG = GAConfig()


# =============================================================================
#  PARALLELISM SELECTION (Ray or serial)
# =============================================================================


def _want_parallel() -> bool:
    """
    Decide whether to run in parallel (Ray) or serial.

    Rules
    -----
    - GA_PARALLEL=0 / "false" / "no"  → force serial.
    - GA_PARALLEL=1 / "true" / "yes"  → force parallel.
    - GA_PARALLEL unset / "auto"      → parallel only if ≥ 2 GPUs.
    """
    flag = os.getenv("GA_PARALLEL", "auto").lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    return torch.cuda.device_count() > 1


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

    This keeps training/eval consistent between processes and makes the
    requested GPU explicit (useful on clusters).
    """
    dev = device.strip()
    low = dev.lower()
    if low.startswith("cuda:"):
        _, _, idx = low.partition(":")
        if idx:
            # Respect pre-set CUDA visibility (e.g., Ray assigns GPUs per worker).
            if os.getenv("CUDA_VISIBLE_DEVICES"):
                return "cuda:0"
            os.environ["CUDA_VISIBLE_DEVICES"] = idx
            return f"cuda:{idx}"
        return "cuda"
    if low == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    return dev


@contextmanager
def _pushd(path: Path):
    """Temporarily change working directory."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# =============================================================================
#  SENTINEL VALUES FOR INVALID INDIVIDUALS
# =============================================================================

INVALID_V = {0.0}       # invalid average velocity
INVALID_E = {-100.0}    # negative energy sentinel (equivalent to +100 before flip)
INVALID_P = {0.0}       # invalid progress / maneuverability

_GS_INIT_LOCK = None
_GS_INIT_FILE_LOCK = None
_WORKER_DEBUG_PRINTED = False


def _default_fitness(_: Optional[Dict[str, Any]] = None) -> List[float]:
    """Return the sentinel fitness values defined by INVALID_* globals."""
    return [
        float(min(INVALID_V)),
        float(min(INVALID_E)),
        float(min(INVALID_P)),
    ]


def _failure_result(
    reason: str,
    cfg: Dict[str, Any],
    exp_name: Optional[str] = None,
    train_it: Optional[int] = None,
) -> Tuple[List[float], Dict[str, Any], Dict[str, np.ndarray]]:
    """Build a safe fallback result for failed train/eval steps."""
    ff = _default_fitness(cfg)
    meta = dict(
        exp_name=exp_name or "failed",
        train_it=int(train_it if train_it is not None else cfg.get("TRAIN_ITERS", 0)),
        train_repetition=int(cfg.get("TRAIN_REPETITION", 1)),
        rep_exp_names=exp_name or "",
        max_p=float("nan"),
        eval_reward_mean=0.0,
        failed=True,
        fail_reason=reason,
    )
    extra = dict(
        p_s=np.array([]),
        v_s=np.array([]),
        E_s=np.array([]),
    )
    return ff, meta, extra


def _set_thread_envs() -> None:
    """Ensure per-process thread envs are bounded (Ray workers included)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("TI_NUM_THREADS", "1")
    os.environ.setdefault("GSTAICHI_NUM_THREADS", "1")
    if not getattr(builtins, "_TORCH_THREADS_CONFIGURED", False):
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        builtins._TORCH_THREADS_CONFIGURED = True


def _log_worker_context(tag: str, cfg: Optional[Dict[str, Any]] = None) -> None:
    """Print useful per-worker context to stdout for debugging."""
    global _WORKER_DEBUG_PRINTED
    if _WORKER_DEBUG_PRINTED:
        return
    _WORKER_DEBUG_PRINTED = True
    host = socket.gethostname()
    pid = os.getpid()
    cuda_vis = os.getenv("CUDA_VISIBLE_DEVICES", "")
    env_urdf = os.getenv("URDF_DIR", "")
    env_ray_tmp = os.getenv("RAY_TMPDIR", "")
    env_cache = os.getenv("XDG_CACHE_HOME", "")
    env_gs_init = os.getenv("URDF_GS_INIT", "")
    env_mujoco_gl = os.getenv("MUJOCO_GL", "")
    print(
        f"[worker] tag={tag} host={host} pid={pid} "
        f"CUDA_VISIBLE_DEVICES={cuda_vis} URDF_DIR={env_urdf} "
        f"RAY_TMPDIR={env_ray_tmp} XDG_CACHE_HOME={env_cache} "
        f"URDF_GS_INIT={env_gs_init} MUJOCO_GL={env_mujoco_gl}"
    )
    try:
        print(
            f"[worker] torch.cuda.is_available={torch.cuda.is_available()} "
            f"device_count={torch.cuda.device_count()}"
        )
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            print(f"[worker] torch.cuda.current_device={idx} name={name}")
    except Exception:
        print("[worker] torch cuda info failed")
    if cfg:
        print(
            f"[worker] cfg DEVICE={cfg.get('DEVICE')} TRAIN_ENVS={cfg.get('TRAIN_ENVS')} "
            f"EVAL_ENVS={cfg.get('EVAL_ENVS')} BASE_DIR={cfg.get('BASE_DIR')} "
            f"LOGS_DIR={cfg.get('LOGS_DIR')} URDF_DIR={cfg.get('URDF_DIR')}"
        )

def _log_mem(tag: str) -> None:
    if os.getenv("MEM_LOG", "0").strip() not in ("1", "true", "yes"):
        return
    try:
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024 ** 2)
        vms_mb = proc.memory_info().vms / (1024 ** 2)
    except Exception:
        rss_mb = vms_mb = float("nan")
    try:
        if torch.cuda.is_available():
            cuda_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            cuda_rsv_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        else:
            cuda_mb = cuda_rsv_mb = 0.0
    except Exception:
        cuda_mb = cuda_rsv_mb = float("nan")
    print(
        f"[mem] tag={tag} pid={os.getpid()} "
        f"rss_mb={rss_mb:.1f} vms_mb={vms_mb:.1f} "
        f"cuda_alloc_mb={cuda_mb:.1f} cuda_reserved_mb={cuda_rsv_mb:.1f}"
    )

def _ensure_gs_initialized() -> None:
    """Initialize Genesis once per process (thread-safe best effort)."""
    global _GS_INIT_LOCK, _GS_INIT_FILE_LOCK
    _set_thread_envs()
    if _GS_INIT_LOCK is None:
        _GS_INIT_LOCK = __import__("threading").Lock()
    if gs._initialized:
        return
    with _GS_INIT_LOCK:
        if not gs._initialized:
            lock_dir = os.getenv("URDF_DIR", "").strip() or "/tmp"
            try:
                Path(lock_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                lock_dir = "/tmp"
            lock_path = Path(lock_dir) / ".gs_init.lock"
            if _GS_INIT_FILE_LOCK is None:
                _GS_INIT_FILE_LOCK = FileLock(str(lock_path))
            with _GS_INIT_FILE_LOCK:
                if not gs._initialized:
                    gs.init(logging_level="error", backend=gs.gpu)


def _should_init_gs_for_urdf(urdf_dir: Path) -> bool:
    """
    Decide whether URDF generation needs Genesis initialized.

    Auto mode:
      - If PyYAML is unavailable, fallback to Genesis (needs init).
      - If no aero_parameters.yaml is found, fallback to Genesis (needs init).
      - Otherwise skip Genesis init (URDF can be built from YAML only).
    """
    flag = os.getenv("URDF_GS_INIT", "").strip().lower()
    if flag in ("1", "true", "yes", "force"):
        return True
    if flag in ("0", "false", "no", "skip"):
        return False

    try:
        import yaml as _yaml  # noqa: F401
    except Exception:
        return True

    candidates: List[Path] = []
    env_path = os.getenv("AERO_CONFIG_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(urdf_dir / "aero_parameters.yaml")
    repo_default = (
        Path(__file__).resolve().parents[2]
        / "genesis"
        / "assets"
        / "urdf"
        / "mydrone"
        / "aero_parameters.yaml"
    )
    candidates.append(repo_default)
    return not any(p.is_file() for p in candidates)


def _create_urdf_with_retry(
    phys_genome: Sequence[float],
    urdf_dir: Path,
    max_attempts: int = 6,
    base_sleep: float = 0.2,
) -> Path:
    """Generate a URDF file with retry/backoff on EAGAIN (errno 11)."""
    urdf_dir.mkdir(parents=True, exist_ok=True)
    debug_urdf = os.getenv("DEBUG_URDF", "").strip().lower() in ("1", "true", "yes")
    lock_path = urdf_dir / ".urdf.lock"
    need_gs_init = _should_init_gs_for_urdf(urdf_dir)
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            if need_gs_init:
                _ensure_gs_initialized()
            if debug_urdf:
                print(f"[urdf] gs_init={'on' if need_gs_init else 'off'}")
            if debug_urdf:
                print(f"[urdf] attempt={attempt} dir={urdf_dir}")
            with FileLock(str(lock_path)):
                urdf_path = Path(UrdfMaker(phys_genome, out_dir=urdf_dir).create_urdf()).resolve()
            if debug_urdf:
                print(f"[urdf] ok path={urdf_path}")
            mirror_dir_raw = os.getenv("URDF_MIRROR_DIR", "").strip()
            if mirror_dir_raw:
                mirror_dir = Path(mirror_dir_raw).expanduser().resolve()
                mirror_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(urdf_path, mirror_dir / urdf_path.name)
                    sentinel = mirror_dir / ".meshes_copied"
                    if not sentinel.exists():
                        src_meshes = urdf_dir / "meshes"
                        dst_meshes = mirror_dir / "meshes"
                        if src_meshes.is_dir():
                            dst_meshes.mkdir(parents=True, exist_ok=True)
                            shutil.copytree(src_meshes, dst_meshes, dirs_exist_ok=True)
                        for fname in ("aero_parameters.yaml", "actuators.csv", "drone.py"):
                            src_f = urdf_dir / fname
                            if src_f.is_file():
                                shutil.copy2(src_f, mirror_dir / fname)
                        sentinel.write_text("ok")
                except Exception:
                    if debug_urdf:
                        print("[urdf] mirror copy failed")
            return urdf_path
        except OSError as exc:
            last_exc = exc
            if getattr(exc, "errno", None) == 11 and attempt < max_attempts:
                sleep_s = base_sleep * (2 ** (attempt - 1))
                time.sleep(sleep_s + random.uniform(0.0, base_sleep))
                continue
            raise
        except Exception as exc:
            last_exc = exc
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("URDF generation failed without exception.")


def _all_finite(vals: Sequence[float]) -> bool:
    for v in vals:
        try:
            if not np.isfinite(float(v)):
                return False
        except Exception:
            return False
    return True


# =============================================================================
#  TRAIN + EVAL (single GPU, programmatic)
# =============================================================================


def _extract_reward_curve(
    log_dir: Path,
    train_iters: int,
    n_points: int = 10,
    win_frac: float = 0.05,
) -> Dict[str, float]:
    """
    Extract reward values at 10%, 20%, …, 100% of training using a moving
    average over a window equal to `win_frac` of total training steps.

    Assumptions
    -----------
    - TensorBoard scalar tag is exactly `"Train/mean_reward"`.
    - Window is defined in *training steps* (± half-window around target step).
    - If the window has no events (sparse logging), we fall back to the
      nearest event.

    Returns
    -------
    dict
        Keys are "rew_10pct", "rew_20pct", ..., "rew_100pct".
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
            out[f"rew_{frac * 10}pct"] = float(np.nanmean(vals[mask]))
        else:
            idx = int(np.argmin(np.abs(steps - target)))
            out[f"rew_{frac * 10}pct"] = float(vals[idx])

    return out

def _eval_only_custom(
    genome_norm,
    policy_path,
    tag,
    cfg,
    return_arrays=True,
):
    _set_thread_envs()
    _log_worker_context("eval_only", cfg)
    try:
        phys_genome = Chromosome_Drone.to_physical(genome_norm)
        env_urdf_dir = os.getenv("URDF_DIR", "").strip()
        urdf_dir = (
            Path(env_urdf_dir).expanduser().resolve()
            if env_urdf_dir
            else Path(cfg["URDF_DIR"]).expanduser().resolve()
        )
        urdf_file = _create_urdf_with_retry(phys_genome, urdf_dir)
    except Exception as exc:
        traceback.print_exc()
        reason = f"urdf_generation_failed: {exc}"
        print(f"[safe_mode] {reason}")
        return _failure_result(reason, cfg, train_it=0)
    exp_name = urdf_file.stem
    gen_tag = cfg.get("GENERATION", None)
    try:
        gen_tag = int(gen_tag) if gen_tag is not None else None
    except Exception:
        gen_tag = None
    if gen_tag is not None:
        gen_prefix = f"g{gen_tag:03d}"
        if cfg.get("EXP_PREFIX"):
            exp_name = f"{cfg['EXP_PREFIX']}-{gen_prefix}-{exp_name}"
        else:
            exp_name = f"{gen_prefix}-{exp_name}"
    elif cfg.get("EXP_PREFIX"):
        exp_name = f"{cfg['EXP_PREFIX']}-{exp_name}"

    _prepare_device_env(cfg.get("DEVICE", "cuda:0"))
    base_dir = Path(cfg["BASE_DIR"]).expanduser().resolve()
    eval_dir = (Path(cfg["LOGS_DIR"]).expanduser().resolve() / "eval" / exp_name)

    policy_path = Path(policy_path).expanduser().resolve()
    cfg_src = policy_path.parent / "cfgs.pkl"
    cfg_dst = Path(cfg["LOG_ROOT"]).expanduser().resolve() / exp_name / "cfgs.pkl"
    cfg_dst.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_src.is_file():
        raise FileNotFoundError(f"cfgs.pkl not found next to policy: {cfg_src}")
    copied_cfg = False
    if not cfg_dst.is_file():
        shutil.copy2(cfg_src, cfg_dst)
        copied_cfg = True
    print(
        f"[eval_only] exp={exp_name} policy={policy_path} "
        f"cfgs={'copied' if copied_cfg else 'reuse'} eval_dir={eval_dir}"
    )

    # Usa evaluation ma caricando la policy custom
    with _pushd(base_dir):
        print(
            "[eval_only] running evaluation "
            f"(envs={cfg['EVAL_ENVS']} vmin={cfg['VMIN']} vmax={cfg['VMAX']})"
        )
        out = evaluation(
            exp_name=exp_name,
            urdf_file=urdf_file,
            ckpt=None,  # Ignorato
            envs=cfg["EVAL_ENVS"],
            vmin=cfg["VMIN"],
            vmax=cfg["VMAX"],
            return_arrays=return_arrays,
            obs_genome=None,
            custom_policy_path=policy_path,  # << PATCH IN eval.py
            eval_dir=eval_dir,
        )

    if return_arrays:
        v_dict, e_dict, p_dict, _, extra = out
        max_p = extra["max_p"]
    else:
        v_dict, e_dict, p_dict, _, max_p = out
        extra = None

    eval_reward_mean = float(extra.get("eval_reward_mean", np.nan)) if extra else float("nan")
    if not np.isfinite(eval_reward_mean):
        eval_reward_mean = 0.0
    meta = dict(
        vel_v=v_dict["mean_v"],
        vel_E=-v_dict["mean_E"],
        vel_P=v_dict["mean_progress"],
        eff_v=e_dict["mean_v"],
        eff_E=-e_dict["mean_E"],
        eff_P=e_dict["mean_progress"],
        prog_v=p_dict["mean_v"],
        prog_E=-p_dict["mean_E"],
        prog_P=p_dict["mean_progress"],
        train_it=0,
        train_repetition=1,
        exp_name=exp_name,
        rep_exp_names=exp_name,
        max_p=max_p,
        eval_reward_mean=eval_reward_mean,
        # reward curve zerata
        **{f"rew_{i*10}pct": 0.0 for i in range(1, 11)},
        final_reward=0.0,
        steps90_pct=0.0,
    )

    ff = [
        v_dict["mean_v"],
        -e_dict["mean_E"],
        p_dict["mean_progress"],
    ]
    print(f"[eval_only] done exp={exp_name} ff={ff} max_p={max_p:.2f}")

    return ff, meta, extra


def _train_and_eval_sync(
    genome_norm: Sequence[float],
    parent_info: Tuple[Optional[str], Optional[int]],
    tag: str,
    cfg: Dict[str, Any],
    return_arrays: bool = True,
):
    """
    Synchronous wrapper that:

      1. Maps normalized genome → physical parameters.
      2. Builds a URDF for the given physical genome.
      3. Trains a control policy (with optional inheritance).
      4. Runs evaluation over a speed range.
      5. Reads training reward curve from TensorBoard.

    Parameters
    ----------
    genome_norm : sequence of float
        Normalized genome in [0, 1]^D.
    parent_info : (parent_exp, parent_ckpt)
        If provided, we inherit policy weights from this experiment.
    tag : str
        Run tag for naming/logging (e.g. timestamp).
    cfg : dict
        Training / evaluation config.
    return_arrays : bool
        If True, also return raw arrays from evaluation.

    Returns
    -------
    ff : list[float]
        Final 3D fitness (velocity, -energy, progress).
    meta : dict
        Detailed metadata for CSV logging.
    extra : Optional[dict]
        When `return_arrays` is True, contains raw arrays:
            "p_s": smoothed progress,
            "v_s": smoothed mean velocities (aligned on v_cmd),
            "E_s": smoothed energy per meter (aligned on v_cmd).
    """
    parent_exp, parent_ckpt = parent_info

    # 1) Map genome to physical parameters and generate URDF
    try:
        _set_thread_envs()
        _log_mem("train_eval:start")
        _log_worker_context("train_and_eval", cfg)
        phys_genome = Chromosome_Drone.to_physical(genome_norm)
        env_urdf_dir = os.getenv("URDF_DIR", "").strip()
        urdf_dir = (
            Path(env_urdf_dir).expanduser().resolve()
            if env_urdf_dir
            else Path(cfg["URDF_DIR"]).expanduser().resolve()
        )
        urdf_file = _create_urdf_with_retry(phys_genome, urdf_dir)
    except Exception as exc:
        traceback.print_exc()
        reason = f"urdf_generation_failed: {exc}"
        print(f"[safe_mode] {reason}")
        return _failure_result(reason, cfg)

    base_exp_name = urdf_file.stem
    if cfg.get("EXP_PREFIX"):
        base_exp_name = f"{cfg['EXP_PREFIX']}-{base_exp_name}"

    device = _prepare_device_env(cfg.get("DEVICE", "cuda:0"))
    base_dir = Path(cfg["BASE_DIR"]).expanduser().resolve()

    # 2) Training iterations: shorter if inheriting a parent policy
    train_iters = (
        cfg["TRAIN_ITERS_INHERIT"]
        if (parent_exp is not None and parent_ckpt is not None)
        else cfg["TRAIN_ITERS"]
    )

    train_repetition = int(cfg.get("TRAIN_REPETITION", 1) or 1)
    if train_repetition < 1:
        print("[train_eval][warn] TRAIN_REPETITION < 1; forcing to 1")
        train_repetition = 1

    def _run_once(run_exp_name: str) -> Tuple[List[float], Dict[str, Any], Dict[str, Any]]:
        # 3) Train policy for this morphology
        try:
            jitter = float(os.getenv("WORKER_START_JITTER", "0") or 0.0)
            if jitter > 0:
                time.sleep(random.uniform(0.0, jitter))
            _log_mem(f"train_eval:{run_exp_name}:before_train")
            with _pushd(base_dir):
                training(
                    exp_name=run_exp_name,
                    urdf_file=urdf_file,
                    num_envs=cfg["TRAIN_ENVS"],
                    max_iterations=train_iters,
                    parent_exp=parent_exp,
                    parent_ckpt=parent_ckpt,
                    device=device,
                )
        except Exception as exc:
            traceback.print_exc()
            reason = f"training_failed exp={run_exp_name}: {exc}"
            print(f"[safe_mode] {reason}")
            return _failure_result(reason, cfg, exp_name=run_exp_name, train_it=train_iters)

        # 4) Evaluate policy at multiple commanded speeds
        eval_dir = (Path(cfg["LOGS_DIR"]).expanduser().resolve() / "eval" / run_exp_name)

        try:
            _log_mem(f"train_eval:{run_exp_name}:before_eval")
            with _pushd(base_dir):
                out = evaluation(
                    exp_name=run_exp_name,
                    urdf_file=urdf_file,
                    ckpt=train_iters,
                    envs=cfg["EVAL_ENVS"],
                    vmin=cfg["VMIN"],
                    vmax=cfg["VMAX"],
                    return_arrays=return_arrays,
                    eval_dir=eval_dir,
                )
        except Exception as exc:
            traceback.print_exc()
            reason = f"evaluation_failed exp={run_exp_name}: {exc}"
            print(f"[safe_mode] {reason}")
            return _failure_result(reason, cfg, exp_name=run_exp_name, train_it=train_iters)

        try:
            if return_arrays:
                v_dict, e_dict, p_dict, _, extra = out
                max_p = extra["max_p"]
            else:
                v_dict, e_dict, p_dict, _, max_p = out
                extra = None
        except Exception as exc:
            reason = f"eval_output_unpack_failed exp={run_exp_name}: {exc}"
            print(f"[safe_mode] {reason}")
            return _failure_result(reason, cfg, exp_name=run_exp_name, train_it=train_iters)

        if not _all_finite(
            [
                v_dict.get("mean_v"),
                e_dict.get("mean_E"),
                p_dict.get("mean_progress"),
                max_p,
            ]
        ):
            reason = (
                f"non_finite_metrics exp={run_exp_name} "
                f"v={v_dict.get('mean_v')} E={e_dict.get('mean_E')} "
                f"P={p_dict.get('mean_progress')} max_p={max_p}"
            )
            print(f"[safe_mode] {reason}")
            return _failure_result(reason, cfg, exp_name=run_exp_name, train_it=train_iters)

        if return_arrays:
            p_s = np.asarray(extra.get("p_s", []))
            v_s = np.asarray(extra.get("v_s", []))
            E_s = np.asarray(extra.get("E_s", []))
            if p_s.size == 0 or v_s.size == 0 or E_s.size == 0:
                reason = f"empty_eval_arrays exp={run_exp_name}"
                print(f"[safe_mode] {reason}")
                return _failure_result(reason, cfg, exp_name=run_exp_name, train_it=train_iters)
            if not (
                np.isfinite(p_s).all()
                and np.isfinite(v_s).all()
                and np.isfinite(E_s).all()
            ):
                reason = f"non_finite_eval_arrays exp={run_exp_name}"
                print(f"[safe_mode] {reason}")
                return _failure_result(reason, cfg, exp_name=run_exp_name, train_it=train_iters)

        # 5) Read TensorBoard logs and compute a smoothed reward curve
        tb_log_dir = Path(cfg["LOG_ROOT"]) / run_exp_name
        reward_curve = _extract_reward_curve(
            tb_log_dir,
            train_iters,
            n_points=10,
            win_frac=0.05,
        )

        eval_reward_mean = float(extra.get("eval_reward_mean", np.nan)) if extra else float("nan")
        if not np.isfinite(eval_reward_mean):
            eval_reward_mean = 0.0

        # 6) Build metadata dict for logging
        meta = dict(
            vel_v=v_dict["mean_v"],
            vel_E=-v_dict["mean_E"],
            vel_P=v_dict["mean_progress"],
            eff_v=e_dict["mean_v"],
            eff_E=-e_dict["mean_E"],
            eff_P=e_dict["mean_progress"],
            prog_v=p_dict["mean_v"],
            prog_E=-p_dict["mean_E"],
            prog_P=p_dict["mean_progress"],
            train_it=train_iters,
            exp_name=run_exp_name,
            max_p=max_p,
            eval_reward_mean=eval_reward_mean,
            **reward_curve,
        )

        ff = [
            v_dict["mean_v"],           # +velocity
            -e_dict["mean_E"],          # +(-energy)
            p_dict["mean_progress"],    # +progress
        ]

        # Optionally offload payload to disk to reduce RAM use when repeating.
        if return_arrays and train_repetition > 1:
            rep_dir = Path(cfg["BASE_DIR"]).expanduser().resolve() / "analysis" / "rep_payloads"
            rep_dir.mkdir(parents=True, exist_ok=True)
            payload_path = rep_dir / f"{run_exp_name}.npz"
            np.savez_compressed(payload_path, p_s=p_s, v_s=v_s, E_s=E_s)
            extra = {"payload_path": str(payload_path)}
            print(f"[train_eval] rep payload saved → {payload_path}")

        return ff, meta, extra

    def _cleanup_after_rep() -> None:
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if train_repetition == 1:
        ff, meta, extra = _run_once(base_exp_name)
        meta["train_repetition"] = 1
        meta["rep_exp_names"] = base_exp_name
        return ff, meta, extra

    rep_payloads: List[Dict[str, Any]] = []
    rep_exp_names: List[str] = []
    raw_ffs: List[List[float]] = []
    max_p_vals: List[float] = []

    for rep_idx in range(train_repetition):
        run_exp_name = base_exp_name
        if train_repetition > 1:
            run_exp_name = f"{base_exp_name}_r{rep_idx + 1:02d}"
            print(
                f"[train_eval] rep {rep_idx + 1}/{train_repetition} "
                f"exp={run_exp_name} train_iters={train_iters}"
            )
        ff, meta, extra = _run_once(run_exp_name)
        _log_mem(f"train_eval:{run_exp_name}:after_eval")
        _cleanup_after_rep()
        rep_payloads.append(dict(rep_idx=rep_idx, meta=meta, extra=extra))
        rep_exp_names.append(run_exp_name)

        if meta.get("failed"):
            raw_ffs.append(_default_fitness(cfg))
        else:
            raw_ffs.append(ff)
        try:
            max_p_vals.append(float(meta.get("max_p", np.nan)))
        except Exception:
            pass

        if train_repetition > 1:
            status = "failed" if meta.get("failed") else "ok"
            print(
                f"[train_eval] rep {rep_idx + 1}/{train_repetition} done "
                f"exp={run_exp_name} status={status}"
            )

    if raw_ffs:
        ff_mean = np.nanmean(np.asarray(raw_ffs, dtype=float), axis=0).tolist()
    else:
        ff_mean = _default_fitness(cfg)

    max_p_mean = float(np.nanmean(max_p_vals)) if max_p_vals else float("nan")

    meta = dict(
        exp_name=rep_exp_names[0] if rep_exp_names else base_exp_name,
        train_it=train_iters,
        train_repetition=train_repetition,
        rep_exp_names="|".join(rep_exp_names),
        max_p=max_p_mean,
    )
    extra = dict(rep_payloads=rep_payloads)

    _log_mem("train_eval:end")
    return ff_mean, meta, extra


# Parallel / serial dispatch wrapper
if USE_PARALLEL:

    # max_calls=1 forces Ray to recycle the worker process after each task,
    # which helps avoid memory growth across many train+eval runs.
    _ray_max_calls_raw = os.getenv("RAY_MAX_CALLS", "1").strip().lower()
    _ray_max_calls: Optional[int]
    if _ray_max_calls_raw in ("", "0", "none", "inf", "infinite"):
        _ray_max_calls = None
    else:
        try:
            _ray_max_calls = max(1, int(_ray_max_calls_raw))
        except Exception:
            _ray_max_calls = 1

    _ray_remote_kwargs = {"num_gpus": 1}
    if _ray_max_calls is not None:
        _ray_remote_kwargs["max_calls"] = _ray_max_calls

    @ray.remote(**_ray_remote_kwargs)
    def train_and_eval_remote(*args, **kwargs):
        return _train_and_eval_sync(*args, **kwargs)

    @ray.remote(**_ray_remote_kwargs)
    def eval_only_remote(*args, **kwargs):
        return _eval_only_custom(*args, **kwargs)

else:

    def train_and_eval_remote(*args, **kwargs):
        return _train_and_eval_sync(*args, **kwargs)

    def eval_only_remote(*args, **kwargs):
        return _eval_only_custom(*args, **kwargs)


# =============================================================================
#  POST-HOC ANALYSIS (plots)
# =============================================================================


class PostAnalyzer:
    """
    Helper to analyze the CSV produced by the GA and generate plots.

    It can:
      - Clean invalid sentinel values.
      - Plot best fronts per generation (velocity, energy, progress).
      - Plot final reward vs generation and learning speed.
    """

    def __init__(
        self,
        csv_path: str = "deap_temp.csv",
        stats_obj: Optional["Stats"] = None,
        pkl_path: Optional[str] = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)

        if "row_kind" in self.df.columns:
            row_kind = self.df["row_kind"].fillna("agg")
            self.df = self.df[row_kind != "rep"].copy()

        # Replace sentinel values with NaN so Matplotlib ignores them.
        for col in ("vel_v", "eff_v", "prog_v"):
            self.df.loc[self.df[col].isin(INVALID_V), col] = np.nan
        for col in ("vel_E", "eff_E", "prog_E"):
            self.df.loc[self.df[col].isin(INVALID_E), col] = np.nan
        for col in ("vel_P", "eff_P", "prog_P"):
            self.df.loc[self.df[col].isin(INVALID_P), col] = np.nan

        self.stats = stats_obj
        if self.stats is None and pkl_path and Path(pkl_path).is_file():
            import pickle

            with open(pkl_path, "rb") as f:
                self.stats = pickle.load(f)

        self.vel = self.df[["vel_v", "vel_E", "vel_P"]].rename(
            columns={"vel_v": "v", "vel_E": "E", "vel_P": "P"}
        )
        self.eff = self.df[["eff_v", "eff_E", "eff_P"]].rename(
            columns={"eff_v": "v", "eff_E": "E", "eff_P": "P"}
        )
        self.prog = self.df[["prog_v", "prog_E", "prog_P"]].rename(
            columns={"prog_v": "v", "prog_E": "E", "prog_P": "P"}
        )

    def final_reward_steps(self, out: str = "final_reward_steps.png") -> None:
        """
        Plot, for the best individual in each generation:
          - Final episodic reward (left y-axis).
          - Percentage of steps needed to reach 90% reward (right y-axis).
        """
        if "final_reward" not in self.df.columns:
            print("final_reward missing")
            return
        
        df_valid = self.df.dropna(subset=["final_reward", "steps90_pct"])
        if df_valid.empty:
            print("No valid reward data – skipping plot.")
            return

        idx = self.df.groupby("generation")["final_reward"].idxmax()
        gens = self.df.loc[idx, "generation"].to_numpy(dtype=float)
        final_vals = self.df.loc[idx, "final_reward"].to_numpy(dtype=float)
        steps_vals = self.df.loc[idx, "steps90_pct"].to_numpy(dtype=float)

        order = np.argsort(gens)
        gens = gens[order]
        final_vals = final_vals[order]
        steps_vals = steps_vals[order]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(gens, final_vals, label="Final reward")
        ax2.plot(gens, steps_vals, label="Steps to 90% (pct)")

        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Final reward (avg last 5%)")
        ax2.set_ylabel("Steps to 90% final reward [%]")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.title("Evolution of final reward and learning speed")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ final reward / learning speed plot →", out)

    def fronts_progress_V(self, out: str = "best_vel_per_gen.png") -> None:
        """Plot best velocity per generation."""
        if self.stats is None:
            print("No stats – skipping velocity progress plot.")
            return

        V = np.where(np.isin(self.stats.V, list(INVALID_V)), np.nan, self.stats.V)
        plt.plot(np.nanmax(V, axis=1), label="velocity ↑")
        plt.xlabel("Generation")
        plt.ylabel("Best value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ velocity progress plot →", out)

    def fronts_progress_P(self, out: str = "best_prog_per_gen.png") -> None:
        """Plot best progress per generation (and minimal_p if available)."""
        if self.stats is None:
            print("No stats – skipping progress plot.")
            return

        M = np.where(np.isin(self.stats.M, list(INVALID_P)), np.nan, self.stats.M)
        best_prog = np.nanmax(M, axis=1)

        plt.figure()
        plt.plot(best_prog, label="best progress ↑")

        if "minimal_p" in self.df.columns:
            min_p_ser = self.df.groupby("generation")["minimal_p"].first()
            min_p_curve = min_p_ser.reindex(range(len(best_prog)))
            plt.plot(min_p_curve, "--", label="minimal_p threshold")

        plt.xlabel("Generation")
        plt.ylabel("Meters")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ progress plot →", out)

    def fronts_progress_E(self, out: str = "best_eff_per_gen.png") -> None:
        """Plot best (-energy) per generation."""
        if self.stats is None:
            print("No stats – skipping energy plot.")
            return

        E = np.where(np.isin(self.stats.E, list(INVALID_E)), np.nan, self.stats.E)
        plt.plot(np.nanmax(E, axis=1), label="-energy ↑")
        plt.xlabel("Generation")
        plt.ylabel("Best value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ energy progress plot →", out)

    def analyze(self, prefix: str = "analysis") -> None:
        """Run all standard plots."""
        self.fronts_progress_V(f"{prefix}_progress_vel.png")
        self.fronts_progress_P(f"{prefix}_progress_prog.png")
        self.fronts_progress_E(f"{prefix}_progress_eff.png")
        self.final_reward_steps(f"{prefix}_reward_speed.png")


# =============================================================================
#  FITNESS DATABASE (CSV cache)
# =============================================================================


class FitnessDB:
    """
    Simple CSV-backed cache mapping chromosome → fitness + metadata.

    This avoids retraining individuals that have already been evaluated.
    """

    def __init__(self, name: str, n_obj: int, root: Path) -> None:
        self.n_obj = n_obj
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{name}.csv"
        self.df = pd.read_csv(self.path) if self.path.exists() else self._blank()
        if self.path.exists():
            if self._ensure_columns():
                lock = FileLock(str(self.path) + ".lock")
                with lock:
                    self.df.to_csv(self.path, index=False)
        else:
            self.df.to_csv(self.path, index=False)

    def lookup_fitness(self, chromo: Sequence[float]) -> Optional[List[float]]:
        """Return cached fitness values for the chromosome, if present."""
        df = self._filter_agg_rows(self.df)
        row = df[df.chromosome == str(list(chromo))]
        if row.empty:
            return None
        return [row[f"ff_{i}"].min() for i in range(self.n_obj)]

    def insert(self, chromo: Sequence[float], ff: Sequence[float], meta: Dict[str, Any]) -> None:
        """Append a new row (chromo, fitness, metadata) to the CSV."""
        print(f"   ↘ writing CSV  gen={meta.get('generation')}  ff={ff}")
        row_dict: Dict[str, Any] = {c: np.nan for c in self.df.columns}

        row_dict["timestamp"] = time.time()
        row_dict["chromosome"] = str(list(chromo))
        for i, val in enumerate(ff):
            row_dict[f"ff_{i}"] = val
        for k, v in meta.items():
            if k in row_dict:
                row_dict[k] = v
            else:
                print(f"Warning: meta key {k} not in columns")

        row_df = pd.DataFrame([row_dict])

        lock = FileLock(str(self.path) + ".lock")
        with lock:
            self.df = pd.concat([self.df, row_df], ignore_index=True)
            row_df.to_csv(self.path, mode="a", header=False, index=False)

    def _blank(self) -> pd.DataFrame:
        cols = (
            ["timestamp", "chromosome"]
            + [f"ff_{i}" for i in range(self.n_obj)]
            + [
                "generation",
                "uid",
                "parent_idx_a",
                "parent_idx_b",
                "parent_uid_a",
                "parent_uid_b",
                "parent_gen_a",
                "parent_gen_b",
                "row_kind",
                "rep_idx",
                "exp_name",
                "rep_exp_names",
                "train_it",
                "train_repetition",
                "max_p",
                "minimal_p",
                "vel_v",
                "vel_E",
                "vel_P",
                "eff_v",
                "eff_E",
                "eff_P",
                "prog_v",
                "prog_E",
                "prog_P",
                "failed",
                "fail_reason",
                "rew_10pct",
                "rew_20pct",
                "rew_30pct",
                "rew_40pct",
                "rew_50pct",
                "rew_60pct",
                "rew_70pct",
                "rew_80pct",
                "rew_90pct",
                "rew_100pct",
                "final_reward",
                "steps90_pct",
                "eval_reward_mean",
            ]
        )
        return pd.DataFrame(columns=cols)

    @staticmethod
    def _filter_agg_rows(df: pd.DataFrame) -> pd.DataFrame:
        if "row_kind" not in df.columns:
            return df
        row_kind = df["row_kind"].fillna("agg")
        return df[row_kind != "rep"]

    def _ensure_columns(self) -> bool:
        """
        Ensure the CSV has all expected columns (adds missing, reorders if needed).
        """
        expected = list(self._blank().columns)
        extra = [c for c in self.df.columns if c not in expected]
        missing = [c for c in expected if c not in self.df.columns]
        changed = False
        for col in missing:
            self.df[col] = np.nan
            changed = True
        new_cols = expected + extra
        if list(self.df.columns) != new_cols:
            self.df = self.df[new_cols]
            changed = True
        return changed

    def get_row(self, chromo: Sequence[float]) -> Optional[pd.Series]:
        """Return the entire row for the chromosome, or None if absent."""
        df = self._filter_agg_rows(self.df)
        row = df[df.chromosome == str(list(chromo))]
        return None if row.empty else row.iloc[0]


# =============================================================================
#  STATS CONTAINER
# =============================================================================


class Stats:
    """
    Container for per-generation distributions of the three objectives:
    velocity, energy, and progress.
    """

    def __init__(self, n_pop: int, n_gen: int, n_obj: int) -> None:
        self.arr = np.zeros((n_obj, n_gen + 1, n_pop))
        self.arr.fill(np.nan)

    def record(self, gen: int, pop: List["IndType"]) -> None:
        invalid_sets = (INVALID_V, INVALID_E, INVALID_P)
        for j in range(self.arr.shape[0]):
            vals = [ind.fitness.values[j] for ind in pop]
            vals = [np.nan if v in invalid_sets[j] else v for v in vals]
            self.arr[j, gen] = vals

    @property
    def V(self) -> np.ndarray:
        return self.arr[0]

    @property
    def E(self) -> np.ndarray:
        return self.arr[1]

    @property
    def M(self) -> np.ndarray:
        return self.arr[2]


IndType = TypeVar("IndType")


# =============================================================================
#  NSGA-II MAIN CLASS
# =============================================================================


class CodesignDEAP:
    """
    NSGA-II loop for morphology + controller co-design.

    Usage (programmatic)
    --------------------
        cfg = GAConfig()
        cfg.population_size = 40
        cfg.num_generations = 30
        ...
        ga = CodesignDEAP(cfg)
        final_pop = ga.run()
    """

    def __init__(self, config: GAConfig = DEFAULT_GA_CONFIG) -> None:
        self.cfg = config

        # Basic checks
        if self.cfg.population_size % 4 != 0:
            raise ValueError("population_size must be a multiple of 4 for tournamentDCD.")

        self.n_pop = self.cfg.population_size
        self.n_gen = self.cfg.num_generations
        self.cx_pb = self.cfg.crossover_probability
        self.mut_pb = self.cfg.mutation_probability
        self.inherit_policy = self.cfg.inherit_policy

        self.gen_policy = self.cfg.gen_policy
        self.policy_path = self.cfg.policy_path

        self.tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = self.cfg.run_name or self.tag
        self.base_dir = (Path(self.cfg.base_dir) / self.run_name).expanduser().resolve()
        self.urdf_dir = self.base_dir / "urdf_generated"
        self.logs_dir = self.base_dir / "logs"
        self.analysis_dir = self.base_dir / "analysis"
        for d in (self.urdf_dir, self.logs_dir, self.analysis_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.log_root = (self.logs_dir / "ea").resolve()
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.exp_prefix = self.run_name
        self.stats_path = (self.analysis_dir / "stats.pkl").resolve()

        self.db = FitnessDB(self.cfg.csv_basename, 3, root=self.analysis_dir)
        self.stats = Stats(self.n_pop, self.n_gen, 3)

        # Create DEAP fitness and individual types (only once)
        if "FitMulti" not in creator.__dict__:
            creator.create("FitMulti", base.Fitness, weights=self.cfg.weights)
        if "Chrom" not in creator.__dict__:
            creator.create("Chrom", list, fitness=creator.FitMulti)

        self.IndType = creator.Chrom

        self._uid_counter = 0

        # Bounds in normalized space (all genes ∈ [0, 1])
        low, up = Chromosome_Drone.get_bounds()
        self._low = low
        self._up = up
        n_genes = Chromosome_Drone.num_genes()

        # DEAP toolbox
        self.tb = base.Toolbox()
        self.tb.register("attr_float", random.random)
        self.tb.register("ind", tools.initRepeat, self.IndType, self.tb.attr_float, n=n_genes)
        self.tb.register("pop", tools.initRepeat, list, self.tb.ind)

        # Typical NSGA-II operators for continuous decision variables
        self.tb.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            low=self._low,
            up=self._up,
            eta=self.cfg.eta_c,
        )
        self.tb.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=self._low,
            up=self._up,
            eta=self.cfg.eta_m,
            indpb=1.0 / n_genes,
        )
        self.tb.register("select", tools.selNSGA2)
        self.tb.register("evaluate", self._evaluate)

    # ------------------------------------------------------------------ #
    # Fitness evaluation                                                 #
    # ------------------------------------------------------------------ #

    def _assign_uid(self, ind: "IndType") -> int:
        """Assign a new globally unique UID to an individual."""
        self._uid_counter += 1
        ind.uid = int(self._uid_counter)
        return ind.uid

    def _ensure_uids(self, population: Sequence["IndType"]) -> None:
        """Ensure all individuals have a UID; assign if missing or invalid."""
        max_uid = int(self._uid_counter)
        for ind in population:
            uid = getattr(ind, "uid", None)
            if uid is None:
                self._assign_uid(ind)
                max_uid = max(max_uid, int(ind.uid))
                continue
            try:
                uid_val = int(uid)
            except Exception:
                self._assign_uid(ind)
                max_uid = max(max_uid, int(ind.uid))
                continue
            if uid_val < 0:
                self._assign_uid(ind)
                max_uid = max(max_uid, int(ind.uid))
                continue
            max_uid = max(max_uid, uid_val)
        if max_uid > self._uid_counter:
            self._uid_counter = max_uid

    def _evaluate(self, indiv: "IndType") -> Tuple[float, float, float]:
        """
        DEAP evaluation hook – possibly spawns Ray jobs.

        The individual is a list of floats in [0, 1] (normalized genome).
        """
        chromo = list(indiv)
        mode = "gen_policy eval-only" if self.gen_policy else "train+eval"
        if not hasattr(indiv, "uid") or getattr(indiv, "uid") is None:
            self._assign_uid(indiv)
        print(
            f"[evaluate] gen={getattr(self, '_gen', 0)} mode={mode} "
            f"uid={getattr(indiv, 'uid', -1)} "
            f"parent_uid=({getattr(indiv, 'parent_uid_a', -1)}, "
            f"{getattr(indiv, 'parent_uid_b', -1)}) "
            f"chr={chromo}"
        )

        if self.gen_policy and not self.policy_path:
            raise ValueError("gen_policy requires a valid --policy_path")
        if self.gen_policy and self.cfg.train_repetition > 1:
            print(
                "   ↪ GEN_POLICY active → train_repetition ignored "
                f"(cfg={self.cfg.train_repetition})"
            )
        try:
            cfg_reps = int(self.cfg.train_repetition)
        except Exception:
            cfg_reps = 1
        if cfg_reps < 1:
            print("[evaluate][warn] train_repetition < 1; forcing to 1")
            cfg_reps = 1

        # CSV cache: if we have seen this chromosome before, reuse its fitness.
        if not self.gen_policy:
            cached_row = self.db.get_row(chromo)
            if cached_row is not None:
                cached_rep = cached_row.get("train_repetition", 1)
                try:
                    cached_rep = int(cached_rep)
                except Exception:
                    cached_rep = 1
                if cached_rep != cfg_reps:
                    print(
                        "   ↪ cache-hit skipped (train_repetition mismatch: "
                        f"cached={cached_rep} cfg={cfg_reps})"
                    )
                else:
                    ff_cached = [cached_row[f"ff_{i}"] for i in range(3)]
                    indiv.fitness.values = tuple(ff_cached)
                    indiv.max_p = cached_row.get("max_p", np.nan)
                    indiv.exp_name = cached_row.get("exp_name", None)
                    indiv.train_it = cached_row.get("train_it", self.cfg.train_iters_new)
                    print(
                        f"   ↪ cache-hit uid={getattr(indiv, 'uid', -1)} "
                        f"parent_uid=({getattr(indiv, 'parent_uid_a', -1)}, "
                        f"{getattr(indiv, 'parent_uid_b', -1)}) "
                        f"exp={cached_row.get('exp_name', 'NA')} "
                        f"ff={ff_cached}"
                    )
                    return tuple(ff_cached)
        else:
            print("   ↪ GEN_POLICY → cache bypassed")

        # New chromosome → full train + eval pipeline (or eval-only when gen_policy)
        if self.gen_policy:
            print("   ↪ NEW chromosome → eval-only (gen policy, no training)")
        else:
            print(
                "   ↪ NEW chromosome → training for "
                f"{self.cfg.train_iters_new} iterations "
                f"(or {self.cfg.train_iters_inherit} if inheritance is triggered)."
            )

        parent_info = (
            getattr(indiv, "parent_exp", None),
            getattr(indiv, "parent_ckpt", None),
        )
        env_urdf_dir = os.getenv("URDF_DIR", "").strip()
        urdf_dir = Path(env_urdf_dir).expanduser().resolve() if env_urdf_dir else self.urdf_dir
        cfg = dict(
            TRAIN_ITERS=self.cfg.train_iters_new,
            TRAIN_ITERS_INHERIT=self.cfg.train_iters_inherit,
            TRAIN_REPETITION=cfg_reps,
            TRAIN_ENVS=self.cfg.train_envs,
            EVAL_ENVS=self.cfg.eval_envs,
            VMIN=self.cfg.vmin,
            VMAX=self.cfg.vmax,
            LOG_ROOT=str(self.log_root),
            EXP_PREFIX=self.exp_prefix,
            DEVICE=self.cfg.device,
            BASE_DIR=str(self.base_dir),
            URDF_DIR=str(urdf_dir),
            LOGS_DIR=str(self.logs_dir),
            GENERATION=getattr(self, "_gen", 0),
        )

        if USE_PARALLEL:
            if self.gen_policy:
                print("   ↪ Ray eval-only job launched")
                fut = eval_only_remote.remote(chromo, self.policy_path, self.tag, cfg, True)
            else:
                fut = train_and_eval_remote.remote(chromo, parent_info, self.tag, cfg, True)
            indiv._pending_future = fut
            # Placeholder; real fitness will be set after Ray returns.
            return (0.0, 0.0, 0.0)

        if self.gen_policy:
            print(f"   ↪ GEN_POLICY active → skipping training (policy={self.policy_path})")
            
            ff, meta, extra = _eval_only_custom(
                chromo,
                self.policy_path,
                self.tag,
                cfg,
                return_arrays=True,
            )
            print(f"   ✔ sync-eval ff={ff} max_p={meta['max_p']:.2f}")

            indiv._meta_raw = meta
            indiv._failed = bool(meta.get("failed"))
            indiv.max_p = meta["max_p"]
            indiv.exp_name = meta["exp_name"]
            indiv.train_it = meta["train_it"]
            
            if extra:
                indiv._p_s = extra["p_s"]
                indiv._v_s = extra["v_s"]
                indiv._E_s = extra["E_s"]

            indiv.fitness.values = tuple(ff)
            return tuple(ff)

        ff, meta, extra = _train_and_eval_sync(chromo, parent_info, self.tag, cfg, True)
        print(f"   ✔ sync-train+eval ff={ff} max_p={meta['max_p']:.2f}")

        indiv._meta_raw = meta
        if extra and "rep_payloads" in extra:
            indiv._rep_payloads = extra["rep_payloads"]
            indiv._failed = False
        else:
            indiv._failed = bool(meta.get("failed"))
        indiv.max_p = meta["max_p"]
        indiv.exp_name = meta["exp_name"]
        indiv.train_it = meta["train_it"]
        if extra and "rep_payloads" not in extra:
            indiv._p_s = extra["p_s"]
            indiv._v_s = extra["v_s"]
            indiv._E_s = extra["E_s"]

        indiv.fitness.values = tuple(ff)
        return tuple(ff)

    # ------------------------------------------------------------------ #
    # Fitness post-processing (progress threshold)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pick_triples(
        p_s: np.ndarray,
        v_s: np.ndarray,
        E_s: np.ndarray,
        minimal_p: float,
    ):
        """
        Given smoothed progress (p_s), smoothed mean velocities (v_s),
        and smoothed energy (E_s), aligned on v_cmd,
        extract three representative operating points:

          - vel: max velocity with p >= minimal_p
          - eff: min energy with p >= minimal_p
          - prog: global max progress (unfiltered)
        """
        if len(p_s) == 0:
            zero = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            return zero, zero, zero

        idx_p = int(np.argmax(p_s))
        mask = np.where(p_s >= minimal_p)[0]

        if mask.size == 0:
            vel = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            eff = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
        else:
            idx_v = int(mask[np.argmax(v_s[mask])])
            idx_e = int(mask[np.argmin(E_s[mask])])
            vel = dict(
                mean_v=float(v_s[idx_v]),
                mean_E=float(E_s[idx_v]),
                mean_progress=float(p_s[idx_v]),
            )
            eff = dict(
                mean_v=float(v_s[idx_e]),
                mean_E=float(E_s[idx_e]),
                mean_progress=float(p_s[idx_e]),
            )

        prog = dict(
            mean_v=float(v_s[idx_p]),
            mean_E=float(E_s[idx_p]),
            mean_progress=float(p_s[idx_p]),
        )
        return vel, eff, prog

    def _finalize_and_persist(self, ind: "IndType", minimal_p: float) -> None:
        """
        Compute final fitness for an individual, apply the progress threshold,
        and write a row in the DB.
        """
        uid = getattr(ind, "uid", -1)
        parent_idx_a = getattr(ind, "parent_idx_a", -1)
        parent_idx_b = getattr(ind, "parent_idx_b", -1)
        parent_uid_a = getattr(ind, "parent_uid_a", -1)
        parent_uid_b = getattr(ind, "parent_uid_b", -1)
        parent_gen_a = getattr(ind, "parent_gen_a", -1)
        parent_gen_b = getattr(ind, "parent_gen_b", -1)
        if hasattr(ind, "_rep_payloads"):
            rep_payloads = list(getattr(ind, "_rep_payloads", []))
            if not rep_payloads:
                return

            def _avg_adjust(val: Any, invalid: set[float], replacement: float) -> Any:
                try:
                    val_f = float(val)
                except Exception:
                    return val
                return replacement if val_f in invalid else val_f

            def _accum(acc: Dict[str, List[float]], key: str, val: Any) -> None:
                try:
                    val_f = float(val)
                except Exception:
                    return
                acc.setdefault(key, []).append(val_f)

            rep_ff: List[Tuple[float, float, float]] = []
            rep_exp_names: List[str] = []
            acc: Dict[str, List[float]] = {}
            any_failed = False
            failed_count = 0
            train_repetition = len(rep_payloads)
            avg_failed_ff = (
                float(min(INVALID_V)),
                -1.0,
                float(min(INVALID_P)),
            )
            avg_all_invalid = False

            for payload in rep_payloads:
                rep_meta = dict(payload.get("meta", {}))
                rep_extra = payload.get("extra", {})
                rep_idx = payload.get("rep_idx", None)
                rep_exp_name = rep_meta.get("exp_name", None)
                if rep_exp_name:
                    rep_exp_names.append(rep_exp_name)

                rep_failed = bool(rep_meta.get("failed"))
                if not rep_failed:
                    payload_path = rep_extra.get("payload_path")
                    if payload_path:
                        try:
                            with np.load(payload_path) as data:
                                p_s = np.asarray(data.get("p_s", []))
                                v_s = np.asarray(data.get("v_s", []))
                                E_s = np.asarray(data.get("E_s", []))
                        except Exception as exc:
                            rep_failed = True
                            rep_meta["failed"] = True
                            rep_meta["fail_reason"] = f"payload_load_failed: {exc}"
                            p_s = np.array([])
                            v_s = np.array([])
                            E_s = np.array([])
                    else:
                        p_s = np.asarray(rep_extra.get("p_s", []))
                        v_s = np.asarray(rep_extra.get("v_s", []))
                        E_s = np.asarray(rep_extra.get("E_s", []))
                    if p_s.size == 0 or v_s.size == 0 or E_s.size == 0:
                        rep_failed = True
                        rep_meta["failed"] = True
                        rep_meta["fail_reason"] = rep_meta.get("fail_reason", "empty_eval_arrays")
                    elif not (
                        np.isfinite(p_s).all()
                        and np.isfinite(v_s).all()
                        and np.isfinite(E_s).all()
                    ):
                        rep_failed = True
                        rep_meta["failed"] = True
                        rep_meta["fail_reason"] = rep_meta.get(
                            "fail_reason", "non_finite_eval_arrays"
                        )

                if rep_failed:
                    ff_rep = tuple(_default_fitness())
                    ff_rep_avg = avg_failed_ff
                    any_failed = True
                    failed_count += 1
                    rep_meta.update(
                        dict(
                            vel_v=float(min(INVALID_V)),
                            vel_E=float(min(INVALID_E)),
                            vel_P=float(min(INVALID_P)),
                            eff_v=float(min(INVALID_V)),
                            eff_E=float(min(INVALID_E)),
                            eff_P=float(min(INVALID_P)),
                            prog_v=float(min(INVALID_V)),
                            prog_E=float(min(INVALID_E)),
                            prog_P=float(min(INVALID_P)),
                        )
                    )
                else:
                    vel_d, eff_d, prog_d = self._pick_triples(
                        p_s,
                        v_s,
                        E_s,
                        minimal_p,
                    )
                    ff_rep = (
                        vel_d["mean_v"],
                        -eff_d["mean_E"],
                        prog_d["mean_progress"],
                    )
                    ff_rep_avg = ff_rep
                    rep_meta.update(
                        dict(
                            vel_v=vel_d["mean_v"],
                            vel_E=-vel_d["mean_E"],
                            vel_P=vel_d["mean_progress"],
                            eff_v=eff_d["mean_v"],
                            eff_E=-eff_d["mean_E"],
                            eff_P=eff_d["mean_progress"],
                            prog_v=prog_d["mean_v"],
                            prog_E=-prog_d["mean_E"],
                            prog_P=prog_d["mean_progress"],
                        )
                    )

                rep_meta.update(
                    dict(
                        max_p=rep_meta.get("max_p", np.nan),
                        minimal_p=minimal_p,
                        uid=uid,
                        parent_idx_a=parent_idx_a,
                        parent_idx_b=parent_idx_b,
                        parent_uid_a=parent_uid_a,
                        parent_uid_b=parent_uid_b,
                        parent_gen_a=parent_gen_a,
                        parent_gen_b=parent_gen_b,
                        row_kind="rep",
                        rep_idx=rep_idx,
                        train_repetition=train_repetition,
                        rep_exp_names=rep_meta.get("exp_name", ""),
                    )
                )
                self.db.insert(list(ind), ff_rep, dict(generation=self._gen, **rep_meta))
                rep_ff.append(ff_rep_avg)

                for key in (
                    "vel_v",
                    "vel_E",
                    "vel_P",
                    "eff_v",
                    "eff_E",
                    "eff_P",
                    "prog_v",
                    "prog_E",
                    "prog_P",
                    "max_p",
                    "final_reward",
                    "steps90_pct",
                    "eval_reward_mean",
                ):
                    if key in rep_meta:
                        if key in ("vel_v", "eff_v", "prog_v"):
                            _accum(acc, key, _avg_adjust(rep_meta[key], INVALID_V, 0.0))
                        elif key in ("vel_E", "eff_E", "prog_E"):
                            _accum(acc, key, _avg_adjust(rep_meta[key], INVALID_E, -1.0))
                        elif key in ("vel_P", "eff_P", "prog_P"):
                            _accum(acc, key, _avg_adjust(rep_meta[key], INVALID_P, float(min(INVALID_P))))
                        else:
                            _accum(acc, key, rep_meta[key])
                for i in range(1, 11):
                    key = f"rew_{i * 10}pct"
                    if key in rep_meta:
                        _accum(acc, key, rep_meta[key])

            avg_all_invalid = train_repetition > 0 and failed_count == train_repetition
            if rep_ff and not avg_all_invalid:
                ff_final = tuple(np.nanmean(np.asarray(rep_ff, dtype=float), axis=0))
            else:
                ff_final = tuple(_default_fitness())

            meta = dict(getattr(ind, "_meta_raw", {}))
            meta.update(
                dict(
                    max_p=getattr(ind, "max_p", np.nan),
                    minimal_p=minimal_p,
                    uid=uid,
                    parent_idx_a=parent_idx_a,
                    parent_idx_b=parent_idx_b,
                    parent_uid_a=parent_uid_a,
                    parent_uid_b=parent_uid_b,
                    parent_gen_a=parent_gen_a,
                    parent_gen_b=parent_gen_b,
                    row_kind="agg",
                    rep_idx=-1,
                    train_repetition=train_repetition,
                    rep_exp_names="|".join(rep_exp_names),
                )
            )
            for key, vals in acc.items():
                if vals:
                    meta[key] = float(np.nanmean(vals))
            if avg_all_invalid:
                meta.update(
                    dict(
                        vel_v=float(min(INVALID_V)),
                        vel_E=float(min(INVALID_E)),
                        vel_P=float(min(INVALID_P)),
                        eff_v=float(min(INVALID_V)),
                        eff_E=float(min(INVALID_E)),
                        eff_P=float(min(INVALID_P)),
                        prog_v=float(min(INVALID_V)),
                        prog_E=float(min(INVALID_E)),
                        prog_P=float(min(INVALID_P)),
                    )
                )
            if any_failed:
                meta["failed"] = True
                meta["fail_reason"] = "rep_failed"

            ind.exp_name = meta.get("exp_name", getattr(ind, "exp_name", None))
            ind.train_it = meta.get("train_it", getattr(ind, "train_it", self.cfg.train_iters_new))

            eff_e_pos = -float(meta.get("eff_E", 0.0))
            print(
                f"[finalize][rep-avg] gen={self._gen} uid={uid} "
                f"parent_uid=({parent_uid_a},{parent_uid_b}) "
                f"chr={list(ind)} "
                f"vel={float(meta.get('vel_v', 0.0)):.2f} "
                f"effE={eff_e_pos:.2f} "
                f"prog={float(meta.get('prog_P', 0.0)):.2f} "
                f"reps={train_repetition} failed={failed_count}"
            )

            self.db.insert(list(ind), ff_final, dict(generation=self._gen, **meta))
            ind.fitness.values = ff_final
            return

        if not hasattr(ind, "_p_s") and not getattr(ind, "_failed", False):
            # Cached individuals or failed evals without payload: nothing to do.
            return

        meta = dict(getattr(ind, "_meta_raw", {}))
        meta.update(
            dict(
                max_p=getattr(ind, "max_p", np.nan),
                minimal_p=minimal_p,
                uid=uid,
                parent_idx_a=parent_idx_a,
                parent_idx_b=parent_idx_b,
                parent_uid_a=parent_uid_a,
                parent_uid_b=parent_uid_b,
                parent_gen_a=parent_gen_a,
                parent_gen_b=parent_gen_b,
                row_kind="agg",
                rep_idx=-1,
                train_repetition=int(meta.get("train_repetition", 1) or 1),
                rep_exp_names=meta.get("rep_exp_names", meta.get("exp_name", "")),
            )
        )

        if getattr(ind, "_failed", False):
            ff_final = tuple(_default_fitness())
        else:
            vel_d, eff_d, prog_d = self._pick_triples(
                np.asarray(ind._p_s),
                np.asarray(ind._v_s),
                np.asarray(ind._E_s),
                minimal_p,
            )

            ff_final = (
                vel_d["mean_v"],
                -eff_d["mean_E"],
                prog_d["mean_progress"],
            )

            meta.update(
                dict(
                    vel_v=vel_d["mean_v"],
                    vel_E=-vel_d["mean_E"],
                    vel_P=vel_d["mean_progress"],
                    eff_v=eff_d["mean_v"],
                    eff_E=-eff_d["mean_E"],
                    eff_P=eff_d["mean_progress"],
                    prog_v=prog_d["mean_v"],
                    prog_E=-prog_d["mean_E"],
                    prog_P=prog_d["mean_progress"],
                )
            )

        ind.exp_name = meta.get("exp_name", getattr(ind, "exp_name", None))
        ind.train_it = meta.get("train_it", getattr(ind, "train_it", self.cfg.train_iters_new))

        if getattr(ind, "_failed", False):
            print(
                f"[finalize][failed] gen={self._gen} uid={uid} "
                f"parent_uid=({parent_uid_a},{parent_uid_b}) "
                f"chr={list(ind)} "
                f"reason={meta.get('fail_reason', 'unknown')} ff={ff_final}"
            )
        else:
            print(
                f"[finalize] gen={self._gen} uid={uid} "
                f"parent_uid=({parent_uid_a},{parent_uid_b}) "
                f"chr={list(ind)} "
                f"vel={vel_d['mean_v']:.2f} effE={eff_d['mean_E']:.2f} "
                f"prog={prog_d['mean_progress']:.2f}"
            )

        self.db.insert(list(ind), ff_final, dict(generation=self._gen, **meta))
        ind.fitness.values = ff_final

    # ------------------------------------------------------------------ #
    # Evolution helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cleanup_individual_payloads(ind: "IndType") -> None:
        """
        Drop heavy per-individual payloads and delete temporary rep files.

        This keeps only the information needed for later stages
        (e.g., exp_name/train_it for inheritance and CSV plots).
        """
        if hasattr(ind, "_rep_payloads"):
            try:
                rep_payloads = list(getattr(ind, "_rep_payloads", []))
            except Exception:
                rep_payloads = []
            for payload in rep_payloads:
                rep_extra = payload.get("extra", {}) if isinstance(payload, dict) else {}
                payload_path = rep_extra.get("payload_path") if isinstance(rep_extra, dict) else None
                if payload_path:
                    try:
                        Path(payload_path).unlink(missing_ok=True)
                    except Exception:
                        pass
            try:
                delattr(ind, "_rep_payloads")
            except Exception:
                pass

        for attr in ("_p_s", "_v_s", "_E_s", "_meta_raw"):
            if hasattr(ind, attr):
                try:
                    delattr(ind, attr)
                except Exception:
                    pass

    def _train_eval_population(self, population: List["IndType"]) -> None:
        """
        Train + evaluate all individuals with invalid fitness.
        Handles:
          - launching Ray jobs,
          - waiting for results,
          - computing minimal_p,
          - finalizing and logging fitness values.
        """
        # 1) Launch training/evaluation where needed
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = self.tb.evaluate(ind)

        # 2) Wait for Ray jobs
        if USE_PARALLEL:
            pend = [ind for ind in population if hasattr(ind, "_pending_future")]
            if pend:
                print(f"  ⏳ waiting for {len(pend)} Ray jobs…")
                minimal_p_fixed = float(self.cfg.fixed_p)
                minimal_p_known = not self.cfg.use_dynamic_p
                fail_cfg = dict(
                    TRAIN_ITERS=self.cfg.train_iters_new,
                    TRAIN_REPETITION=self.cfg.train_repetition,
                )
                for ind in pend:
                    try:
                        ff, meta, extra = ray.get(ind._pending_future)
                    except Exception as exc:
                        reason = f"ray_job_failed: {exc}"
                        print(f"[safe_mode] {reason}")
                        ff, meta, extra = _failure_result(reason, fail_cfg)

                    ind._meta_raw = meta
                    if extra and "rep_payloads" in extra:
                        ind._rep_payloads = extra["rep_payloads"]
                        ind._failed = False
                    else:
                        ind._failed = bool(meta.get("failed"))
                    ind.max_p = meta.get("max_p", np.nan)
                    ind.exp_name = meta.get("exp_name", None)
                    ind.train_it = meta.get("train_it", self.cfg.train_iters_new)
                    if extra and "rep_payloads" not in extra:
                        ind._p_s = extra.get("p_s", np.array([]))
                        ind._v_s = extra.get("v_s", np.array([]))
                        ind._E_s = extra.get("E_s", np.array([]))
                    ind.fitness.values = tuple(ff)
                    del ind._pending_future
                    print(
                        f"   ✅ Ray done uid={getattr(ind, 'uid', -1)} "
                        f"parent_uid=({getattr(ind, 'parent_uid_a', -1)}, "
                        f"{getattr(ind, 'parent_uid_b', -1)}) "
                        f"chr={list(ind)} "
                        f"ff={ff} max_p={ind.max_p:.2f}"
                    )

                    # When minimal_p is fixed, we can finalize immediately and
                    # drop all repetition payloads to keep memory bounded.
                    if minimal_p_known:
                        self._finalize_and_persist(ind, minimal_p_fixed)
                        ind._persisted = True
                        self._cleanup_individual_payloads(ind)

        # 3) minimal_p dynamic/fixed
        peaks: List[float] = []
        if self.cfg.use_dynamic_p:
            peaks = [getattr(ind, "max_p", np.nan) for ind in population]
            peaks = [p for p in peaks if not np.isnan(p)]
        if self.cfg.use_dynamic_p and peaks:
            perc = 100.0 - self.cfg.pct_above
            minimal_p = 0.9 * np.percentile(peaks, perc)
        else:
            minimal_p = self.cfg.fixed_p
        print(
            f"[Gen {self._gen}] minimal_p = {minimal_p:.2f} "
            f"(dynamic={self.cfg.use_dynamic_p}, pct_above={self.cfg.pct_above}%)"
        )

        # 4) finalize → CSV
        for ind in population:
            if not hasattr(ind, "_persisted"):
                self._finalize_and_persist(ind, minimal_p)
                ind._persisted = True
            # Regardless of dynamic/fixed minimal_p, once persisted we no
            # longer need per-repetition payloads in memory or on disk.
            if hasattr(ind, "_persisted"):
                self._cleanup_individual_payloads(ind)

    def _apply_variation(
        self,
        offspring: List["IndType"],
        parents: List["IndType"],
        parent_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Crossover, mutation, and optional inheritance **before** training.
        """
        # Clean up custom attributes on offspring
        for ch in offspring:
            for a in (
                "exp_name",
                "parent_exp",
                "parent_ckpt",
                "parent_idx_a",
                "parent_idx_b",
                "parent_uid_a",
                "parent_uid_b",
                "parent_gen_a",
                "parent_gen_b",
                "_pending_future",
                "_meta_raw",
                "_p_s",
                "_v_s",
                "_E_s",
                "_evaluated",
                "_persisted",
                "max_p",
            ):
                if hasattr(ch, a):
                    delattr(ch, a)

        # Apply SBX + polynomial mutation pairwise
        for i in range(0, len(offspring), 2):
            c1, c2 = offspring[i], offspring[i + 1]
            idx_a = parent_indices[i] if parent_indices is not None else -1
            idx_b = parent_indices[i + 1] if parent_indices is not None else -1
            p_a = parents[i] if i < len(parents) else None
            p_b = parents[i + 1] if (i + 1) < len(parents) else None
            p_uid_a = getattr(p_a, "uid", -1) if p_a is not None else -1
            p_uid_b = getattr(p_b, "uid", -1) if p_b is not None else -1
            p_gen = getattr(self, "_gen", 0) - 1
            for child in (c1, c2):
                child.parent_idx_a = idx_a
                child.parent_idx_b = idx_b
                child.parent_uid_a = p_uid_a
                child.parent_uid_b = p_uid_b
                child.parent_gen_a = p_gen
                child.parent_gen_b = p_gen

            # crossover
            if random.random() < self.cx_pb:
                self.tb.mate(c1, c2)
                if hasattr(c1.fitness, "values"):
                    del c1.fitness.values
                if hasattr(c2.fitness, "values"):
                    del c2.fitness.values

            # Keep discrete genes aligned to valid bins after crossover
            c1[:] = Chromosome_Drone.snap_genome_norm(c1)
            c2[:] = Chromosome_Drone.snap_genome_norm(c2)

            # mutation
            if random.random() < self.mut_pb:
                before = list(c1)
                self.tb.mutate(c1)
                del c1.fitness.values
                c1[:] = Chromosome_Drone.apply_discrete_mutation(before, c1)
            if random.random() < self.mut_pb:
                before = list(c2)
                self.tb.mutate(c2)
                del c2.fitness.values
                c2[:] = Chromosome_Drone.apply_discrete_mutation(before, c2)

            # inheritance → assign exp/ckpt BEFORE training
            if self.inherit_policy:
                infos = []
                for p in (parents[i], parents[i + 1]):
                    if hasattr(p, "exp_name"):
                        ck_it = getattr(p, "train_it", self.cfg.train_iters_new)
                        ck = self.log_root / p.exp_name / f"model_{ck_it}.pt"
                        if ck.is_file():
                            infos.append((p.exp_name, ck_it))
                if infos:
                    c1.parent_exp, c1.parent_ckpt = random.choice(infos)
                    c2.parent_exp, c2.parent_ckpt = random.choice(infos)

    def _after_generation(self, pop: List["IndType"]) -> None:
        """Update stats, generate plots, and print generation summary."""
        g = self._gen
        self.stats.record(g, pop)
        if g % 3 == 0 or g == self.n_gen:
            out_dir = self.analysis_dir / f"g{g:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            PostAnalyzer(self.db.path, self.stats).analyze(prefix=str(out_dir / "gen"))

        best_v = np.nanmax(self.stats.V[g])
        best_e = -np.nanmax(self.stats.E[g])
        best_p = np.nanmax(self.stats.M[g])
        print(
            f"--- Gen {g} summary  "
            f"best_vel={best_v:.2f}  best_eff={best_e:.2f}  best_prog={best_p:.2f}"
        )

    # ------------------------------------------------------------------ #
    # Main loop                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> List["IndType"]:
        """
        Run the full NSGA-II evolution and return the final population.
        """
        print(f"[setup] Run directory: {self.base_dir}")
        print(f"[setup] CSV cache: {self.db.path}")

        # GEN 0
        pop = self.tb.pop(self.n_pop)
        self._ensure_uids(pop)
        for ind in pop:
            ind[:] = Chromosome_Drone.snap_genome_norm(ind)
            ind.parent_idx_a = -1
            ind.parent_idx_b = -1
            ind.parent_uid_a = -1
            ind.parent_uid_b = -1
            ind.parent_gen_a = -1
            ind.parent_gen_b = -1
        self._gen = 0
        self._train_eval_population(pop)
        pop = tools.selNSGA2(pop, self.n_pop)
        self._after_generation(pop)

        # GEN ≥ 1
        for g in range(1, self.n_gen + 1):
            self._gen = g
            print(f"\n════════ Generation {g}/{self.n_gen} ════════")
            self._ensure_uids(pop)

            # 1) parent selection (requires crowding_dist)
            parents = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.tb.clone(p) for p in parents]
            for child in offspring:
                self._assign_uid(child)
            parent_idx_map = {id(ind): idx for idx, ind in enumerate(pop)}
            parent_indices = [parent_idx_map.get(id(p), -1) for p in parents]

            # 2) variation (+ inheritance) before training
            self._apply_variation(offspring, parents, parent_indices)

            # 3) train + eval offspring
            self._train_eval_population(offspring)

            # 4) survivor-selection NSGA-II → new population
            pop = tools.selNSGA2(pop + offspring, self.n_pop)

            # 5) logging / plots
            self._after_generation(pop)

        # Save global stats
        with self.stats_path.open("wb") as f:
            import pickle

            pickle.dump(self.stats, f)
        print(f"Statistics saved ✔ → {self.stats_path}")
        return pop


# =============================================================================
#  CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=4, help="Population size.")
    parser.add_argument("--gen", type=int, default=3, help="Number of generations.")
    parser.add_argument(
        "--train_it",
        type=int,
        default=30,
        help="Training iterations for NEW morphologies.",
    )
    parser.add_argument(
        "--train_repetition",
        type=int,
        default=1,
        help="Repeat train+eval N times (gen_policy=0) and average final fitness.",
    )
    parser.add_argument(
        "--inherit",
        action="store_true",
        default=False,
        help="Enable policy inheritance for offspring.",
    )
    parser.add_argument(
        "--dynamic_p",
        type=int,
        choices=(0, 1),
        default=0,
        help="Dynamic minimal_p (1=on, 0=off).",
    )
    parser.add_argument(
        "--no_dynamic_p",
        action="store_true",
        help="Disable dynamic minimal_p and use fixed_p instead.",
    )
    parser.add_argument(
        "--fixed_p",
        type=float,
        default=250.0,
        help="Fixed minimal_p threshold (used if --no_dynamic_p).",
    )
    parser.add_argument(
        "--pct_above",
        type=float,
        default=50.0,
        help="Percentage of individuals above minimal_p when dynamic.",
    )
    parser.add_argument(
        "--gen_policy",
        type=int,
        choices=(0, 1),
        default=0,
        help="0=train+eval, 1=eval-only with a custom pre-trained policy",
    )
    parser.add_argument(
        "--policy_path", type=str, default=None,
        help="Path to a pre-trained policy to use together with --gen_policy"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name (defaults to timestamp).",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Root directory for artifacts (defaults to cfg.base_dir).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for training/eval (e.g., cuda:0 or cpu).",
    )

    args = parser.parse_args()

    # Build GA configuration from defaults + CLI overrides
    cfg = GAConfig()
    cfg.population_size = args.pop
    cfg.num_generations = args.gen
    cfg.train_iters_new = args.train_it
    cfg.train_repetition = args.train_repetition
    cfg.inherit_policy = args.inherit
    cfg.csv_basename = "nsga"
    if args.no_dynamic_p:
        cfg.use_dynamic_p = False
    else:
        cfg.use_dynamic_p = bool(args.dynamic_p)
    cfg.fixed_p = args.fixed_p
    cfg.pct_above = args.pct_above
    cfg.gen_policy = bool(args.gen_policy)
    cfg.policy_path = args.policy_path
    cfg.run_name = args.run_name
    if args.base_dir is not None:
        cfg.base_dir = args.base_dir
    cfg.device = args.device

    ga = CodesignDEAP(cfg)
    ga.run()


if __name__ == "__main__":
    main()
