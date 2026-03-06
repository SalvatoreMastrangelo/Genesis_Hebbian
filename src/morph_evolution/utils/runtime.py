from __future__ import annotations

import builtins
import os
import random
import shutil
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import genesis as gs
import psutil
import torch
from filelock import FileLock

from drone_making import UrdfMaker
from winged_drone_train.defaults import default_mydrone_urdf_dir

_GS_INIT_LOCK = None
_GS_INIT_FILE_LOCK = None
_WORKER_DEBUG_PRINTED = False


def prepare_device_env(device: str) -> str:
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
def pushd(path: Path):
    """Temporarily change working directory."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def set_thread_envs() -> None:
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


def log_worker_context(tag: str, cfg: Optional[Dict[str, object]] = None) -> None:
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


def log_mem(tag: str) -> None:
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


def ensure_gs_initialized() -> None:
    """Initialize Genesis once per process (thread-safe best effort)."""
    global _GS_INIT_LOCK, _GS_INIT_FILE_LOCK
    set_thread_envs()
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


def should_init_gs_for_urdf(urdf_dir: Path) -> bool:
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
    repo_default = default_mydrone_urdf_dir() / "aero_parameters.yaml"
    candidates.append(repo_default)
    return not any(p.is_file() for p in candidates)


def resolve_urdf_dir(default_dir: str | Path) -> Path:
    """Resolve URDF output directory, honoring `URDF_DIR` environment override."""
    env_urdf_dir = os.getenv("URDF_DIR", "").strip()
    if env_urdf_dir:
        return Path(env_urdf_dir).expanduser().resolve()
    return Path(default_dir).expanduser().resolve()


def create_urdf_with_retry(
    phys_genome: Sequence[float],
    urdf_dir: Path,
    max_attempts: int = 6,
    base_sleep: float = 0.2,
) -> Path:
    """Generate a URDF file with retry/backoff on EAGAIN (errno 11)."""
    urdf_dir.mkdir(parents=True, exist_ok=True)
    debug_urdf = os.getenv("DEBUG_URDF", "").strip().lower() in ("1", "true", "yes")
    lock_path = urdf_dir / ".urdf.lock"
    need_gs_init = should_init_gs_for_urdf(urdf_dir)
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            if need_gs_init:
                ensure_gs_initialized()
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
                        for fname in ("aero_parameters.yaml", "actuators.csv"):
                            src_f = urdf_dir / fname
                            if src_f.is_file():
                                shutil.copy2(src_f, mirror_dir / fname)
                        sentinel.write_text("ok")
                except Exception:
                    if debug_urdf:
                        print("[urdf] mirror copy failed")
            return urdf_path
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if (
                not need_gs_init
                and "Genesis hasn't been initialized" in msg
                and attempt < max_attempts
            ):
                # Some aero-config resolution paths still touch Genesis internals.
                need_gs_init = True
                ensure_gs_initialized()
                continue
            if isinstance(exc, OSError):
                if getattr(exc, "errno", None) == 11 and attempt < max_attempts:
                    sleep_s = base_sleep * (2 ** (attempt - 1))
                    time.sleep(sleep_s + random.uniform(0.0, base_sleep))
                    continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("URDF generation failed without exception.")
