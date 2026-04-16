from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from winged_drone_train.noise_config import configure_solver_noise
from winged_drone_train.runtime_random import seed_runtime_randomness

from .ipc import WorkerCommand, WorkerReply


def _safe_reply(conn, ok: bool, payload: Dict[str, Any] | None = None, error: str = "") -> None:
    try:
        conn.send(WorkerReply(ok=ok, payload=payload or {}, error=error))
    except Exception as exc:
        print(f"[worker] _safe_reply failed to send (ok={ok}): {exc}", flush=True)
        # Attempt a lightweight error-only reply so the main process can fail fast
        try:
            conn.send(WorkerReply(ok=False, payload={}, error=f"Reply serialization failed: {exc}"))
        except Exception:
            pass


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", copy=True)


def _bind_process_to_device(device: str) -> str:
    """
    Bind the worker process to a single CUDA device before Genesis init.

    If the worker is assigned `cuda:N`, we mask visibility to only that GPU and
    then use `cuda:0` locally inside the child process.
    """
    dev = str(device).strip().lower()
    if not dev.startswith("cuda"):
        return device

    try:
        _, idx_str = dev.split(":", 1)
        gpu_idx = int(idx_str)
    except Exception:
        gpu_idx = 0

    inherited_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if inherited_visible:
        visible_list = [item.strip() for item in inherited_visible.split(",") if item.strip()]
        if 0 <= gpu_idx < len(visible_list):
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_list[gpu_idx]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return "cuda:0"


def _configure_worker_cache_root() -> Path:
    """
    Force Taichi/genesis cache into a writable shared location for worker processes.
    """
    cache_root = (Path("logs") / ".cache" / "gstaichi").expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    for env_key in ("XDG_CACHE_HOME", "TI_CACHE_DIR", "TAICHI_CACHE_DIR", "GSTAICHI_CACHE_DIR"):
        os.environ[env_key] = str(cache_root)
    mpl_dir = cache_root / "mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    return cache_root


def _configure_worker_cpu_threads(cpu_threads: Optional[int] = None) -> int:
    if cpu_threads is None:
        cpu_threads = int(os.getenv("LOGICAL_SUPER_SCENE_CPU_THREADS", "1"))
    cpu_threads = max(1, int(cpu_threads))
    for env_key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(env_key, str(cpu_threads))
    try:
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    return cpu_threads


def _make_shared_buffers(
    *,
    num_envs: int,
    num_obs: int,
    num_privileged_obs: int,
    num_actions: int,
) -> Dict[str, torch.Tensor]:
    # device="cpu" is mandatory: after gs.init(backend=gs.gpu) Genesis sets the
    # default PyTorch device to CUDA, so bare torch.zeros() creates GPU tensors.
    # GPU tensors use CUDA IPC (_share_cuda_()) which is not supported across
    # independent spawned processes and raises "CUDA driver error: invalid argument".
    cpu = torch.device("cpu")
    return {
        "actions":    torch.zeros((num_envs, num_actions),        dtype=torch.float32, device=cpu).share_memory_(),
        "obs":        torch.zeros((num_envs, num_obs),            dtype=torch.float32, device=cpu).share_memory_(),
        "critic_obs": torch.zeros((num_envs, num_privileged_obs), dtype=torch.float32, device=cpu).share_memory_(),
        "rew":        torch.zeros((num_envs,),                    dtype=torch.float32, device=cpu).share_memory_(),
        "done":       torch.zeros((num_envs,),                    dtype=torch.int64,   device=cpu).share_memory_(),
        "time_outs":  torch.zeros((num_envs,),                    dtype=torch.float32, device=cpu).share_memory_(),
    }


def worker_main(
    conn,
    *,
    urdf_list: List[str],
    num_envs: int,
    env_cfg: Dict,
    obs_cfg: Dict,
    reward_cfg: Dict,
    command_cfg: Dict,
    device: str,
    show_viewer: bool,
    use_shared_memory: bool = True,
    mps_active_thread_percentage: Optional[int] = None,
    cpu_threads: Optional[int] = None,
) -> None:
    """
    Worker process: owns one Gen_Env shard and performs env stepping commands.
    """
    env = None
    shared_buffers: Optional[Dict[str, torch.Tensor]] = None
    try:
        worker_start_t0 = time.perf_counter()
        # Ensure headless rendering path in workers by default.
        os.environ.setdefault("GS_HEADLESS_NO_GL", "1")
        os.environ.setdefault("GS_PARA_LEVEL", "3")
        _configure_worker_cache_root()
        cpu_threads = _configure_worker_cpu_threads(cpu_threads)
        if mps_active_thread_percentage is not None and int(mps_active_thread_percentage) > 0:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(mps_active_thread_percentage))
        local_device = _bind_process_to_device(device)
        seed_runtime_randomness(f"super_scene_worker:{local_device}")

        import genesis as gs
        from general_policy.env_gen import Gen_Env

        gs_init_start = time.perf_counter()
        gs.init(logging_level="error", backend=gs.gpu)
        gs_init_elapsed = time.perf_counter() - gs_init_start

        worker_total_scenes = len(urdf_list)

        def _report_init_progress(payload: Dict[str, Any]) -> None:
            _safe_reply(
                conn,
                ok=True,
                payload={
                    "event": "init_progress",
                    "worker_total_scenes": int(worker_total_scenes),
                    **payload,
                },
            )

        gen_env_build_start = time.perf_counter()
        env = Gen_Env(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            urdf_list=urdf_list,
            max_scenes=None,
            show_viewer=show_viewer,
            eval=False,
            device=local_device,
            progress_callback=_report_init_progress,
        )
        configure_solver_noise(env, env_cfg)
        gen_env_build_elapsed = time.perf_counter() - gen_env_build_start

        initial_reset_start = time.perf_counter()
        obs, info = env.reset()
        critic = info.get("observations", {}).get("critic")
        if critic is None:
            critic = env.privileged_obs_buf
        initial_reset_elapsed = time.perf_counter() - initial_reset_start

        shm_export_elapsed = 0.0
        worker_ready_elapsed = time.perf_counter() - worker_start_t0
        scene_init_total_s = float(getattr(env, "scene_init_total_s", 0.0))
        scene_build_total_s = float(getattr(env, "scene_build_total_s", 0.0))
        slowest_scene_s = float(getattr(env, "slowest_scene_init_s", 0.0))
        slowest_scene_urdf = str(getattr(env, "slowest_scene_urdf", ""))

        if use_shared_memory:
            try:
                shared_buffers = _make_shared_buffers(
                    num_envs=int(env.num_envs),
                    num_obs=int(env.num_obs),
                    num_privileged_obs=int(env.num_privileged_obs),
                    num_actions=int(env.num_actions),
                )
                shared_buffers["obs"].copy_(_to_cpu(obs))
                shared_buffers["critic_obs"].copy_(_to_cpu(critic))
                print("[worker] Shared memory buffers created successfully.", flush=True)
            except Exception as shm_exc:
                print(
                    f"[worker] share_memory_() failed ({shm_exc}); "
                    "falling back to pipe-based tensor transfer.",
                    flush=True,
                )
                shared_buffers = None
                use_shared_memory = False

        _ready_meta = {
            "num_envs": int(env.num_envs),
            "num_obs": int(env.num_obs),
            "num_privileged_obs": int(env.num_privileged_obs),
            "num_actions": int(env.num_actions),
            "max_episode_length": int(env.max_episode_length),
            "dt": float(env.dt),
            "use_shared_memory": use_shared_memory,
        }
        _sent_ready = False
        if use_shared_memory and shared_buffers is not None:
            try:
                conn.send(WorkerReply(ok=True, payload={
                    "event": "ready",
                    "meta": _ready_meta,
                    "obs": None,
                    "critic_obs": None,
                    "shared_buffers": shared_buffers,
                }))
                _sent_ready = True
            except Exception as ser_exc:
                print(
                    f"[worker] shared_buffers serialization failed ({ser_exc}); "
                    "falling back to pipe-based transfer.",
                    flush=True,
                )
                use_shared_memory = False
                shared_buffers = None
                _ready_meta["use_shared_memory"] = False
        if not _sent_ready:
            _safe_reply(
                conn,
                ok=True,
                payload={
                    "event": "ready",
                    "meta": _ready_meta,
                    "obs": _to_cpu(obs),
                    "critic_obs": _to_cpu(critic),
                    "shared_buffers": None,
                },
            )

        while True:
            msg = conn.recv()
            if isinstance(msg, WorkerCommand):
                cmd = msg.cmd
                payload = msg.payload
            elif isinstance(msg, dict):
                cmd = str(msg.get("cmd", ""))
                payload = dict(msg.get("payload", {}))
            else:
                _safe_reply(conn, ok=False, error="Invalid command format")
                continue

            if cmd == "close":
                _safe_reply(conn, ok=True, payload={"event": "closed"})
                break

            if cmd == "reset":
                obs, info = env.reset()
                critic = info.get("observations", {}).get("critic")
                if critic is None:
                    critic = env.privileged_obs_buf
                if use_shared_memory and shared_buffers is not None:
                    shared_buffers["obs"].copy_(_to_cpu(obs))
                    shared_buffers["critic_obs"].copy_(_to_cpu(critic))
                _safe_reply(
                    conn,
                    ok=True,
                    payload={
                        "event": "reset",
                        "obs": _to_cpu(obs) if not use_shared_memory else None,
                        "critic_obs": _to_cpu(critic) if not use_shared_memory else None,
                    },
                )
                continue

            if cmd == "step":
                if use_shared_memory and shared_buffers is not None:
                    actions = shared_buffers["actions"]
                else:
                    actions = payload.get("actions")
                if not torch.is_tensor(actions):
                    _safe_reply(conn, ok=False, error="`actions` must be a tensor")
                    continue
                obs, rew, done, info = env.step(actions.to(env.device))
                critic = info.get("observations", {}).get("critic")
                if critic is None:
                    critic = env.privileged_obs_buf
                time_outs = info.get("time_outs")
                if time_outs is None:
                    time_outs = torch.zeros_like(done, dtype=torch.float32)
                episode = info.get("episode") if isinstance(info, dict) else None

                if use_shared_memory and shared_buffers is not None:
                    shared_buffers["obs"].copy_(_to_cpu(obs))
                    shared_buffers["critic_obs"].copy_(_to_cpu(critic))
                    shared_buffers["rew"].copy_(_to_cpu(rew))
                    shared_buffers["done"].copy_(_to_cpu(done).long())
                    shared_buffers["time_outs"].copy_(_to_cpu(time_outs).float())

                _safe_reply(
                    conn,
                    ok=True,
                    payload={
                        "event": "step",
                        "obs": _to_cpu(obs) if not use_shared_memory else None,
                        "critic_obs": _to_cpu(critic) if not use_shared_memory else None,
                        "rew": _to_cpu(rew) if not use_shared_memory else None,
                        "done": _to_cpu(done) if not use_shared_memory else None,
                        "time_outs": _to_cpu(time_outs) if not use_shared_memory else None,
                        "episode": episode if isinstance(episode, dict) else None,
                    },
                )
                continue

            _safe_reply(conn, ok=False, error=f"Unknown command: {cmd}")

    except Exception as exc:
        tb = traceback.format_exc()
        _safe_reply(conn, ok=False, error=f"Worker crash: {exc}\n{tb}")
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            import genesis as gs

            gs.destroy()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
