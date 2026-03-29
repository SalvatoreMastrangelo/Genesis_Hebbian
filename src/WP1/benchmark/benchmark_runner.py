"""SNE benchmark runner — executes inside a spawned subprocess.

Each (S, N, E) configuration is isolated in its own subprocess so that
CUDA state, Genesis scenes, and memory are fully cleaned up between runs.

Usage (via grid_search.py):
    mp.get_context("spawn").Process(target=run_sne_config, kwargs=...).start()
"""
from __future__ import annotations

import builtins
import json
import os
import shutil
import statistics
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Public entry point — runs inside spawned subprocess
# ---------------------------------------------------------------------------

def run_sne_config(
    S: int,
    N: int,
    E: int,
    cfg_path: str,
    num_iterations: int,
    catalog_dir_override: Optional[str],
    result_queue,
    num_gpus: int = 1,
) -> None:
    """Benchmark one (S, N, E) configuration and put a result dict in queue.

    This function is designed to be run as the ``target`` of a
    ``multiprocessing.Process`` (spawn context).  It imports all heavy
    dependencies *after* environment variables are configured so that
    Genesis/Taichi see the correct settings from startup.

    Parameters
    ----------
    S : int
        Number of parallel Genesis scene subprocesses.
    N : int
        Number of URDF morphologies per scene.
    E : int
        Number of environments per scene per URDF.
    cfg_path : str
        Path to the base WP1 ``RunConfig`` YAML.
    num_iterations : int
        Number of PPO training iterations to run.
    catalog_dir_override : str or None
        Pre-built catalog directory to use.  ``None`` means no catalog
        (only valid when N == 1).
    result_queue : multiprocessing.Queue
        Queue where the result dict is placed when the run finishes.
    num_gpus : int
        Number of GPUs available. If > 1, scenes are distributed across GPUs.
    """
    # ------------------------------------------------------------------
    # 1. Environment variables — MUST be set before any genesis/torch import
    # ------------------------------------------------------------------
    os.environ["GS_PARA_LEVEL"] = "4"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Thread limits to avoid IZAR nproc exhaustion
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

    # ------------------------------------------------------------------
    # 2. Heavy imports (after env vars)
    # ------------------------------------------------------------------
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from winged_drone_train.rl.A2C_modified import ActorCriticTanh
    from winged_drone_train.train import _configure_cache_root

    from WP1.config import RunConfig
    from WP1.virtual_env import VirtualMultiSceneEnv

    # RSL-RL resolves policy classes by name via builtins
    builtins.ActorCriticTanh = ActorCriticTanh

    # Configure Taichi/Genesis cache root (same as train.py)
    _configure_cache_root()

    wall_start = time.perf_counter()
    device = None  # will be set from cfg
    env = None     # guard for finally block
    tmpdir = None  # guard for finally block

    try:
        # --------------------------------------------------------------
        # 3. Build RunConfig from base YAML
        # --------------------------------------------------------------
        run_cfg = RunConfig.from_yaml(cfg_path)

        # Override multi-scene settings for this (S, N, E) combination
        run_cfg.multi_scene.enabled = True
        run_cfg.multi_scene.S = S
        run_cfg.multi_scene.N = N
        run_cfg.multi_scene.E = E

        device = run_cfg.training.device

        # --------------------------------------------------------------
        # 4. Catalog path
        # --------------------------------------------------------------
        catalog_path: Optional[Path] = None
        if catalog_dir_override is not None:
            catalog_path = Path(catalog_dir_override)
        elif N > 1:
            # N>1 requires a catalog; caller should always pass one
            raise RuntimeError(
                f"[benchmark] N={N} > 1 requires a catalog_dir_override"
            )

        # Also wire the catalog into cfg so that VirtualMultiSceneEnv's
        # URDF assignment reads it correctly
        if catalog_path is not None:
            run_cfg.catalog.catalog_dir = str(catalog_path)

        # --------------------------------------------------------------
        # 5. Create VirtualMultiSceneEnv (this compiles S Genesis scenes)
        # --------------------------------------------------------------
        env_create_start = time.perf_counter()
        env = VirtualMultiSceneEnv(run_cfg, catalog_path=catalog_path, num_gpus=num_gpus)
        env_create_elapsed = time.perf_counter() - env_create_start

        # Collect per-worker compile stats from READY messages
        worker_compile_times: List[float] = []
        worker_ram_mbs: List[float] = []
        worker_vram_alloc_mbs: List[float] = []
        worker_vram_reserved_mbs: List[float] = []

        for meta in env._worker_ready_metas:
            ct = meta.get("compile_time_s")
            if ct is not None:
                worker_compile_times.append(float(ct))
            ram = meta.get("ram_mb")
            if ram is not None:
                worker_ram_mbs.append(float(ram))
            va = meta.get("vram_allocated_mb")
            if va is not None:
                worker_vram_alloc_mbs.append(float(va))
            vr = meta.get("vram_reserved_mb")
            if vr is not None:
                worker_vram_reserved_mbs.append(float(vr))

        compile_wall_time_s: float = env._compile_wall_time_s
        vram_allocated_total_mb: float = env._vram_delta_compile_mb

        # --------------------------------------------------------------
        # 6. Build train_cfg and runner
        # --------------------------------------------------------------
        _, _, _, _, train_cfg = run_cfg.to_legacy_cfgs()

        # Avoid checkpoint writes during benchmark
        train_cfg["save_interval"] = 999999

        tmpdir = tempfile.mkdtemp(prefix="sne_bench_")
        try:
            runner = OnPolicyRunner(env, train_cfg, tmpdir, device=device)
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

        # --------------------------------------------------------------
        # 7. Optionally compile policy with torch.compile
        # --------------------------------------------------------------
        try:
            runner.alg.policy = torch.compile(
                runner.alg.policy, mode="reduce-overhead"
            )
            print(f"[benchmark S={S},N={N},E={E}] torch.compile enabled")
        except Exception as e:
            print(f"[benchmark S={S},N={N},E={E}] torch.compile skipped: {e}")

        # --------------------------------------------------------------
        # 8. Monkey-patch alg.update to measure per-iteration times
        # --------------------------------------------------------------
        num_steps_per_env = run_cfg.training.num_steps_per_env
        total_envs = S * N * E

        iter_times_s: List[float] = []
        steps_per_sec_per_iter: List[float] = []

        _iter_state = {"iter_start": time.perf_counter()}

        _orig_update = runner.alg.update

        def _patched_update(*args, **kwargs):
            result = _orig_update(*args, **kwargs)
            elapsed = time.perf_counter() - _iter_state["iter_start"]
            iter_times_s.append(elapsed)
            # steps collected this iteration = total_envs * num_steps_per_env
            steps = total_envs * num_steps_per_env
            steps_per_sec_per_iter.append(steps / elapsed if elapsed > 0 else 0.0)
            _iter_state["iter_start"] = time.perf_counter()
            return result

        runner.alg.update = _patched_update

        # --------------------------------------------------------------
        # 9. Monkey-patch env.step to measure simulation time per iter
        # --------------------------------------------------------------
        sim_times_s: List[float] = []
        _sim_state = {"sim_accum": 0.0}

        _orig_step = env.step

        def _patched_step(actions):
            t0 = time.perf_counter()
            result = _orig_step(actions)
            _sim_state["sim_accum"] += time.perf_counter() - t0
            return result

        env.step = _patched_step

        # We need to reset the accumulator at each iter boundary.
        # Wrap update a second time to capture sim_accum per iteration.
        _patched_update_inner = runner.alg.update

        def _patched_update_with_sim(*args, **kwargs):
            result = _patched_update_inner(*args, **kwargs)
            sim_times_s.append(_sim_state["sim_accum"])
            _sim_state["sim_accum"] = 0.0
            return result

        runner.alg.update = _patched_update_with_sim

        # --------------------------------------------------------------
        # 10. Run training
        # --------------------------------------------------------------
        _iter_state["iter_start"] = time.perf_counter()
        train_start = time.perf_counter()
        runner.learn(
            num_learning_iterations=num_iterations,
            init_at_random_ep_len=True,
        )
        train_time_total_s = time.perf_counter() - train_start

        # --------------------------------------------------------------
        # 11. Compute aggregate metrics
        # --------------------------------------------------------------
        wall_time_s = time.perf_counter() - wall_start

        def _safe_mean(lst):
            return statistics.mean(lst) if lst else None

        def _safe_min(lst):
            return min(lst) if lst else None

        def _safe_max(lst):
            return max(lst) if lst else None

        result = {
            "status": "OK",
            "S": S,
            "N": N,
            "E": E,
            "total_envs": S * N * E,
            "wall_time_s": wall_time_s,
            "compile_wall_time_s": compile_wall_time_s,
            "sim_time_total_s": sum(sim_times_s),
            "train_time_total_s": train_time_total_s,
            "compile_time_min_s": _safe_min(worker_compile_times),
            "compile_time_mean_s": _safe_mean(worker_compile_times),
            "compile_time_max_s": _safe_max(worker_compile_times),
            "sim_time_per_iter_min_s": _safe_min(sim_times_s),
            "sim_time_per_iter_mean_s": _safe_mean(sim_times_s),
            "sim_time_per_iter_max_s": _safe_max(sim_times_s),
            "steps_per_sec_avg": _safe_mean(steps_per_sec_per_iter),
            "steps_per_sec_per_iter": json.dumps(steps_per_sec_per_iter),
            "iter_times_s": json.dumps(iter_times_s),
            "ram_per_scene_mean_mb": _safe_mean(worker_ram_mbs),
            "ram_per_scene_max_mb": _safe_max(worker_ram_mbs),
            "vram_allocated_per_scene_mean_mb": _safe_mean(worker_vram_alloc_mbs),
            "vram_allocated_per_scene_max_mb": _safe_max(worker_vram_alloc_mbs),
            "vram_allocated_total_mb": vram_allocated_total_mb,
            "vram_reserved_per_scene_mean_mb": _safe_mean(worker_vram_reserved_mbs),
            "vram_reserved_per_scene_max_mb": _safe_max(worker_vram_reserved_mbs),
            "error": "",
        }

    except torch.cuda.OutOfMemoryError as e:
        result = _error_result(S, N, E, "OOM", e)
    except RuntimeError as e:
        # RuntimeError can wrap OOM in some torch versions
        msg = str(e)
        status = "OOM" if "out of memory" in msg.lower() else "ERROR"
        result = _error_result(S, N, E, status, e)
    except Exception as e:
        result = _error_result(S, N, E, "ERROR", e)
    finally:
        # Shut down worker subprocesses cleanly
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        # Clean up RSL-RL temp log dir
        if tmpdir is not None:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    result_queue.put(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_result(
    S: int,
    N: int,
    E: int,
    status: str,
    exc: Exception,
) -> Dict[str, Any]:
    """Build a uniform error-result dict with all numeric fields set to None."""
    tb = traceback.format_exc()
    print(f"[benchmark S={S},N={N},E={E}] {status}: {exc}\n{tb}")
    return {
        "status": status,
        "S": S,
        "N": N,
        "E": E,
        "total_envs": S * N * E,
        "wall_time_s": None,
        "compile_wall_time_s": None,
        "sim_time_total_s": None,
        "train_time_total_s": None,
        "compile_time_min_s": None,
        "compile_time_mean_s": None,
        "compile_time_max_s": None,
        "sim_time_per_iter_min_s": None,
        "sim_time_per_iter_mean_s": None,
        "sim_time_per_iter_max_s": None,
        "steps_per_sec_avg": None,
        "steps_per_sec_per_iter": None,
        "iter_times_s": None,
        "ram_per_scene_mean_mb": None,
        "ram_per_scene_max_mb": None,
        "vram_allocated_per_scene_mean_mb": None,
        "vram_allocated_per_scene_max_mb": None,
        "vram_allocated_total_mb": None,
        "vram_reserved_per_scene_mean_mb": None,
        "vram_reserved_per_scene_max_mb": None,
        "error": str(exc),
    }
