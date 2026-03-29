"""
Scene worker subprocess for multi-scene parallel PPO training.
==============================================================

Each worker owns one Genesis scene (``WingedDroneEnv``) compiled in a
spawned subprocess.  It communicates with the main process via a pair of
``multiprocessing.Queue`` objects:

* ``in_q``  — receives commands from the coordinator (main process).
* ``out_q`` — sends results back to the coordinator.

Protocol
--------
Compilation phase (startup):
    Worker compiles and builds the scene, then sends a ``dict`` with
    ``status="READY"`` and the env metadata (num_obs, etc.) to ``out_q``.

Training loop (per PPO step):
    Coordinator puts ``("STEP", actions_cpu_tensor)`` on ``in_q``.
    Worker moves actions to its device, calls ``env.step()``, puts result
    dict on ``out_q``.

Reset:
    Coordinator puts ``("RESET",)`` on ``in_q``.
    Worker calls ``env.reset()`` and puts result dict on ``out_q``.

Shutdown:
    Coordinator puts ``("STOP",)`` on ``in_q``.
    Worker calls ``gs.destroy()`` and exits.

Important implementation notes
--------------------------------
- All tensors sent via Queue are **CPU tensors** to avoid CUDA IPC issues
  with the "spawn" multiprocessing context.
- ``torch.no_grad()`` wraps every env call to prevent gradient-graph
  accumulation (OOM fix from known project issue).
- Environment variables (GS_PARA_LEVEL, TI_NUM_THREADS, cache dirs) are
  set *inside* the worker before any Genesis import, mirroring the pattern
  in ``multi_urdf_utils.orchestrator``.
"""

from __future__ import annotations

import os
import sys
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def worker_main(
    scene_idx: int,
    urdf_paths: List[str],   # list of N URDF paths (N=1 → WingedDroneEnv, N>1 → MultiDroneEnv)
    E: int,
    env_cfg: Dict[str, Any],
    obs_cfg: Dict[str, Any],
    reward_cfg: Dict[str, Any],
    command_cfg: Dict[str, Any],
    device: str,
    src_dir: str,
    in_q: Any,   # mp.Queue
    out_q: Any,  # mp.Queue
    cpu_threads_per_worker: int,
    wp1_cfg: Optional[Any] = None,   # full RunConfig — required for MultiDroneEnv (N>1)
    gpu_id: int = 0,                 # GPU index for this worker (multi-GPU mode)
    num_gpus: int = 1,               # Total number of GPUs available
) -> None:
    """Long-running scene worker.  Runs inside a spawned subprocess.

    Parameters
    ----------
    scene_idx : int
        Zero-based index of this scene (used for logging).
    urdf_paths : list[str]
        Paths to the N URDF files for this scene.  When N==1, a
        ``WingedDroneEnv`` is compiled (same as before).  When N>1, a
        ``MultiDroneEnv`` is compiled with all N morphologies.
    E : int
        Number of parallel environments per drone entity in this scene.
    env_cfg, obs_cfg, reward_cfg, command_cfg : dict
        Legacy config dicts produced by ``RunConfig.to_legacy_cfgs()``.
    device : str
        CUDA device string (e.g. ``"cuda:0"``).
    src_dir : str
        Absolute path to the ``src/`` directory; injected into ``sys.path``
        so that project modules are importable.
    in_q, out_q : multiprocessing.Queue
        Communication queues (parent → worker, worker → parent).
    cpu_threads_per_worker : int
        Number of CPU threads given to the Taichi compiler.
    wp1_cfg : RunConfig, optional
        Full WP1 configuration object.  Required when ``len(urdf_paths) > 1``
        to construct ``MultiDroneEnv``.
    gpu_id : int
        Index of the GPU assigned to this worker (0-based).
    num_gpus : int
        Total number of GPUs available.
    """
    # ------------------------------------------------------------------ #
    # Environment setup — must happen before any Genesis/Taichi import    #
    # ------------------------------------------------------------------ #
    # Multi-GPU: assign this worker to its designated GPU
    if num_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.environ["GS_PARA_LEVEL"] = "4"
    os.environ["TI_NUM_THREADS"] = str(cpu_threads_per_worker)

    # Isolate Taichi/Genesis cache per worker to avoid concurrent write races.
    # Workers with the same scene shape share compiled kernels after the first
    # run because Taichi keys the cache on IR hash, not PID.
    cache_root = (Path("logs") / ".cache" / "gstaichi").expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    for env_key in (
        "XDG_CACHE_HOME",
        "TI_CACHE_DIR",
        "TAICHI_CACHE_DIR",
        "GSTAICHI_CACHE_DIR",
    ):
        os.environ[env_key] = str(cache_root)

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # ------------------------------------------------------------------ #
    # Delayed imports (after sys.path and env var setup)                  #
    # ------------------------------------------------------------------ #
    import torch
    import genesis as gs
    from winged_drone_train.env import WingedDroneEnv
    from winged_drone_train.train import configure_solver_noise

    # ------------------------------------------------------------------ #
    # Compile scene (N=1 → WingedDroneEnv, N>1 → MultiDroneEnv)         #
    # ------------------------------------------------------------------ #
    _compile_start = time.perf_counter()
    print(f"[worker {scene_idx}] Initializing Genesis...", flush=True)
    gs.init(logging_level="error", backend=gs.gpu)

    N = len(urdf_paths)
    print(f"[worker {scene_idx}] Scene has N={N} URDFs, E={E} envs", flush=True)
    for j, urdf_path in enumerate(urdf_paths):
        print(f"[worker {scene_idx}]   URDF {j}: {urdf_path}", flush=True)

    if N == 1:
        # Single-URDF fast path: WingedDroneEnv (same as before)
        print(f"[worker {scene_idx}] Creating WingedDroneEnv...", flush=True)
        env = WingedDroneEnv(
            num_envs=E,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            urdf_file=urdf_paths[0],
            show_viewer=False,
            eval=False,
            device=device,
        )
        configure_solver_noise(env, env_cfg)
        n_envs_total = E            # WingedDroneEnv: E environments per scene
        print(f"[worker {scene_idx}] WingedDroneEnv created: {n_envs_total} envs", flush=True)
    else:
        # Multi-URDF: one MultiDroneEnv with N drone entities × E envs each
        if wp1_cfg is None:
            raise RuntimeError(
                f"[worker {scene_idx}] wp1_cfg must be provided for N>1 URDFs per scene"
            )
        print(f"[worker {scene_idx}] Creating MultiDroneEnv with {N} URDFs...", flush=True)
        from multi_urdf_utils.multi_drone_env import MultiDroneEnv
        env = MultiDroneEnv(
            urdf_paths=urdf_paths,
            num_envs=E,
            wp1_cfg=wp1_cfg,
            device=device,
            training_mode=True,
        )
        n_envs_total = N * E        # MultiDroneEnv: N*E environments per scene
        print(f"[worker {scene_idx}] MultiDroneEnv created: {n_envs_total} envs ({N} URDFs × {E})", flush=True)

    # Signal ready and report env metadata so the coordinator can set up
    # VirtualMultiSceneEnv properties without instantiating an env itself.
    _compile_time_s = time.perf_counter() - _compile_start

    # Collect RAM and VRAM metrics for the benchmark.
    try:
        import psutil
        _ram_mb = psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        _ram_mb = None
    try:
        _vram_alloc_mb = torch.cuda.memory_allocated() / 1024 / 1024
        _vram_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
    except Exception:
        _vram_alloc_mb = None
        _vram_reserved_mb = None

    print(f"[worker {scene_idx}] Sending READY signal...", flush=True)
    out_q.put({
        "status": "READY",
        "num_obs": env.num_obs,
        "num_privileged_obs": env.num_privileged_obs,
        "num_actions": env.num_actions,
        "max_episode_length": env.max_episode_length,
        "N": N,
        "E": E,
        "compile_time_s": _compile_time_s,
        "ram_mb": _ram_mb,
        "vram_allocated_mb": _vram_alloc_mb,
        "vram_reserved_mb": _vram_reserved_mb,
    })

    # ------------------------------------------------------------------ #
    # Command loop                                                         #
    # ------------------------------------------------------------------ #
    print(f"[worker {scene_idx}] Entering command loop...", flush=True)
    step_count = 0
    while True:
        try:
            msg = in_q.get()
            cmd = msg[0]

            if cmd == "RESET":
                try:
                    with torch.no_grad():
                        obs, extras = env.reset()

                    # MultiDroneEnv returns (D, E, obs_dim); flatten to (D*E, obs_dim)
                    if N > 1:
                        obs = obs.reshape(N * E, -1)
                    priv_obs: Optional[torch.Tensor] = (
                        extras.get("observations", {}).get("critic", None)
                    )
                    if priv_obs is not None and N > 1:
                        priv_obs = priv_obs.reshape(N * E, -1)
                    out_q.put({
                        "obs": obs.cpu(),
                        "priv_obs": priv_obs.cpu() if priv_obs is not None else None,
                    })
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    msg_str = str(e)
                    if "out of memory" in msg_str.lower():
                        print(f"[worker {scene_idx}] OOM in RESET: {e}", flush=True)
                        out_q.put({"error": "OOM", "status": "OOM"})
                    else:
                        raise

            elif cmd == "STEP":
                try:
                    step_count += 1
                    actions_flat = msg[1].to(device)   # (N*E, num_actions) from coordinator
                    # MultiDroneEnv expects (N, E, num_actions)
                    if N > 1:
                        actions_in = actions_flat.reshape(N, E, -1)
                    else:
                        actions_in = actions_flat
                    with torch.no_grad():
                        obs, rew, done, extras = env.step(actions_in)

                    # Flatten (D, E, *) → (D*E, *) for multi-drone
                    if N > 1:
                        obs = obs.reshape(N * E, -1)
                        rew = rew.reshape(N * E)
                        done = done.reshape(N * E)

                    priv_obs = extras.get("observations", {}).get("critic", None)
                    if priv_obs is not None and N > 1:
                        priv_obs = priv_obs.reshape(N * E, -1)
                    time_outs = extras.get("time_outs", None)
                    if time_outs is not None and N > 1:
                        time_outs = time_outs.reshape(N * E)

                    # episode dict contains small scalar floats — safe to serialise
                    episode: Dict[str, float] = {}
                    if "episode" in extras:
                        episode = {
                            k: float(v) for k, v in extras["episode"].items()
                        }

                    out_q.put({
                        "obs": obs.cpu(),
                        "rew": rew.cpu(),
                        "done": done.cpu(),
                        "priv_obs": priv_obs.cpu() if priv_obs is not None else None,
                        "time_outs": time_outs.cpu() if time_outs is not None else None,
                        "episode": episode,
                    })
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    msg_str = str(e)
                    if "out of memory" in msg_str.lower():
                        print(f"[worker {scene_idx}] OOM in STEP {step_count}: {e}", flush=True)
                        out_q.put({"error": "OOM", "status": "OOM"})
                    else:
                        raise

            elif cmd == "STOP":
                print(f"[worker {scene_idx}] Shutting down...", flush=True)
                try:
                    gs.destroy()
                except Exception:
                    pass
                del env
                gc.collect()
                break

            else:
                # Unknown command — send error but keep running
                out_q.put({"error": f"Unknown command: {cmd}"})

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg_str = str(e)
            if "out of memory" in msg_str.lower():
                print(f"[worker {scene_idx}] OOM (outer): {e}", flush=True)
                out_q.put({"error": "OOM", "status": "OOM"})
                break
            else:
                raise
        except Exception as e:
            print(f"[worker {scene_idx}] Unexpected error: {e}", flush=True)
            out_q.put({"error": str(e)})
            break
