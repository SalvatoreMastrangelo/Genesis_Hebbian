"""
Parallel scene orchestrator for the multi-drone benchmark.
==========================================================

Compiles and simulates S scenes concurrently, dedicating
``cpu_threads_per_worker`` CPU threads (TI_NUM_THREADS) to each Taichi compiler.
GS_PARA_LEVEL is always set to 4 (maximum loop parallelization) inside workers.

Concurrency model
-----------------
- A ``ProcessPoolExecutor`` (spawn) runs ``num_workers`` scene-worker
  processes in parallel, where ``num_workers = cpu_count // cpu_threads_per_worker``.
- Each worker owns a full gs.init() / compile / simulate / gs.destroy() cycle.
- Workers share the GPU; VRAM per scene is the practical cap on num_workers.
- As soon as a worker slot frees up the next pending scene is dispatched,
  so compilation and simulation of different scenes naturally overlap.

Usage (from benchmark.py)
--------------------------
    from WP2.multi_urdf_utils.orchestrator import run_scenes_parallel
    scene_metrics_list = run_scenes_parallel(scene_specs, cfg, hebb_cfg_dict)
"""

from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #

def _scene_worker(
    scene_idx: int,
    urdf_paths: List[str],
    N: int,
    E: int,
    device: str,
    seed: int,
    num_episodes: int,
    max_steps: int,
    wp1_cfg_dict: Dict[str, Any],
    hebb_cfg_dict: Dict[str, Any],
    checkpoint_model_path: str,
    checkpoint_config_path: str,
    cpu_threads_per_worker: int,
    src_dir: str,
) -> Dict[str, Any]:
    """Full compile + simulate cycle for a single scene.

    Runs inside a spawned subprocess — no Genesis state is inherited from
    the parent.  Returns a plain dict of timing/metric results.
    """
    # ---- environment setup ------------------------------------------------ #
    os.environ["GS_PARA_LEVEL"] = "4"  # always max loop parallelization
    os.environ["TI_NUM_THREADS"] = str(cpu_threads_per_worker)
    # Isolate Taichi cache per worker so concurrent writes don't race.
    # Workers with the same scene shape will share compiled kernels after the
    # first run because Taichi keys the cache on IR hash, not PID.
    cache_root = (Path("logs") / ".cache" / "gstaichi").expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    for env_key in ("XDG_CACHE_HOME", "TI_CACHE_DIR", "TAICHI_CACHE_DIR", "GSTAICHI_CACHE_DIR"):
        os.environ[env_key] = str(cache_root)

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import genesis as gs
    from WP1.config import RunConfig
    from WP2.config import HebbianConfig
    from WP2.multi_urdf_utils.multi_drone_env import MultiDroneEnv
    from WP2.multi_urdf_utils.multi_drone_actor import MultiDroneActorManager, random_hebbian_rules

    # ---- rebuild configs from plain dicts --------------------------------- #
    wp1_cfg = RunConfig._from_dict(wp1_cfg_dict)

    hebb_cfg = HebbianConfig(
        enabled=hebb_cfg_dict["enabled"],
        eta=hebb_cfg_dict["eta"],
        w_max=hebb_cfg_dict["w_max"],
        num_actions=hebb_cfg_dict["num_actions"],
        hidden_dim=hebb_cfg_dict["hidden_dim"],
    )

    # ---- compile ---------------------------------------------------------- #
    t_compile_start = time.time()
    gs.init(logging_level="error", backend=gs.gpu)

    env = MultiDroneEnv(
        urdf_paths=urdf_paths,
        num_envs=E,
        wp1_cfg=wp1_cfg,
        device=device,
        vmin=hebb_cfg_dict.get("vmin", 6.0),
        vmax=hebb_cfg_dict.get("vmax", 30.0),
    )

    hebbian_rules_list = [
        random_hebbian_rules(
            hebb_cfg,
            out_features=hebb_cfg.num_actions,
            in_features=hebb_cfg.hidden_dim,
            seed=seed + i,
            device=device,
        )
        for i in range(N)
    ]

    actor_mgr = MultiDroneActorManager(
        D=N,
        checkpoint_path=checkpoint_model_path,
        checkpoint_config_path=checkpoint_config_path,
        hebbian_rules_list=hebbian_rules_list,
        hebb_cfg=hebb_cfg,
        stochastic=True,
        device=device,
    )

    t_compile = time.time() - t_compile_start

    # ---- memory snapshot after compilation -------------------------------- #
    import psutil
    _proc = psutil.Process()
    ram_after_compile_mb = _proc.memory_info().rss / 1024 ** 2
    dev_idx = int(device.split(":")[-1]) if ":" in device else 0
    vram_after_compile_mb = torch.cuda.memory_allocated(dev_idx) / 1024 ** 2
    vram_reserved_after_compile_mb = torch.cuda.memory_reserved(dev_idx) / 1024 ** 2

    # ---- simulate --------------------------------------------------------- #
    t_sim_start = time.time()
    total_steps = 0
    episode_alive_fracs = []
    episode_sim_times = []

    for ep in range(num_episodes):
        t_ep_start = time.time()
        actor_mgr.reset_episode(E, device)
        obs, _ = env.reset()
        done = torch.zeros(N, E, dtype=torch.bool, device=torch.device(device))
        ep_steps = 0

        for _ in range(max_steps):
            with torch.no_grad():
                actions = actor_mgr.act(obs)
            obs, rew, term, info = env.step(actions)
            done |= term
            ep_steps += 1
            total_steps += 1
            if done.all():
                break

        episode_alive_fracs.append((~done).float().mean().item())
        episode_sim_times.append(time.time() - t_ep_start)

    t_sim = time.time() - t_sim_start

    gs.destroy()
    del env, actor_mgr, hebbian_rules_list
    gc.collect()

    return {
        "scene_idx": scene_idx,
        "compile_time_s": round(t_compile, 3),
        "sim_time_s": round(t_sim, 3),
        "episode_sim_times": episode_sim_times,
        "total_steps": total_steps,
        "mean_alive_frac": float(sum(episode_alive_fracs) / max(len(episode_alive_fracs), 1)),
        "ram_mb": round(ram_after_compile_mb, 1),
        "vram_allocated_mb": round(vram_after_compile_mb, 1),
        "vram_reserved_mb": round(vram_reserved_after_compile_mb, 1),
    }


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #

def run_scenes_parallel(
    urdf_batches: List[List[str]],
    E: int,
    device: str,
    base_seed: int,
    num_episodes: int,
    max_steps: int,
    wp1_cfg_dict: Dict[str, Any],
    hebb_cfg_dict: Dict[str, Any],
    checkpoint_model_path: str,
    checkpoint_config_path: str,
    num_workers: int,
    cpu_threads_per_worker: int = 4,
) -> List[Dict[str, Any]]:
    """Compile and simulate all scenes with up to ``num_workers`` in parallel.

    Parameters
    ----------
    urdf_batches : list of lists
        Each inner list is the URDF paths for one scene.
    E : int
        Environments per scene.
    device : str
        CUDA device string (e.g. ``"cuda:0"``).
    base_seed : int
        Base random seed; each scene gets ``base_seed + scene_idx * N``.
    num_episodes, max_steps : int
        Rollout parameters.
    wp1_cfg_dict : dict
        Serialisable WP1 RunConfig (from ``RunConfig._to_dict()`` / ``asdict``).
    hebb_cfg_dict : dict
        Serialisable Hebbian config including resolved ``num_actions`` /
        ``hidden_dim`` and ``vmin`` / ``vmax``.
    checkpoint_model_path, checkpoint_config_path : str
        Paths to the frozen actor checkpoint.
    num_workers : int
        Max concurrent scene workers.
    cpu_threads_per_worker : int
        CPU threads given to each Taichi compiler (``TI_NUM_THREADS``).

    Returns
    -------
    list of dicts, one per scene, in scene order.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # src/ must be on the path inside workers
    src_dir = str(Path(__file__).resolve().parent.parent.parent)

    S = len(urdf_batches)
    N = len(urdf_batches[0])

    print(f"[orchestrator] {S} scenes  |  {num_workers} workers  |  "
          f"{cpu_threads_per_worker} threads/compiler")

    ctx = mp.get_context("spawn")
    results_by_idx: Dict[int, Dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        futures = {}
        for scene_idx, urdf_paths in enumerate(urdf_batches):
            scene_seed = base_seed + scene_idx * N
            fut = pool.submit(
                _scene_worker,
                scene_idx=scene_idx,
                urdf_paths=urdf_paths,
                N=N,
                E=E,
                device=device,
                seed=scene_seed,
                num_episodes=num_episodes,
                max_steps=max_steps,
                wp1_cfg_dict=wp1_cfg_dict,
                hebb_cfg_dict=hebb_cfg_dict,
                checkpoint_model_path=checkpoint_model_path,
                checkpoint_config_path=checkpoint_config_path,
                cpu_threads_per_worker=cpu_threads_per_worker,
                src_dir=src_dir,
            )
            futures[fut] = scene_idx

        for fut in as_completed(futures):
            scene_idx = futures[fut]
            try:
                metrics = fut.result()
                results_by_idx[scene_idx] = metrics
                print(f"[orchestrator] scene {scene_idx + 1}/{S} done  "
                      f"compile={metrics['compile_time_s']:.1f}s  "
                      f"sim={metrics['sim_time_s']:.1f}s  "
                      f"alive={metrics['mean_alive_frac']:.1%}")
            except Exception as exc:
                print(f"[orchestrator] scene {scene_idx + 1}/{S} FAILED: {exc}")
                results_by_idx[scene_idx] = {
                    "scene_idx": scene_idx,
                    "compile_time_s": 0.0,
                    "sim_time_s": 0.0,
                    "total_steps": 0,
                    "mean_alive_frac": 0.0,
                    "error": str(exc),
                }

    return [results_by_idx[i] for i in range(S)]
