"""
Benchmark — measure compilation and simulation time for multi-drone scenes.
===========================================================================

Times three phases:
1. URDF generation (catalog building)
2. Scene compilation (Genesis scene.build + aero solver init)
3. Simulation (episode rollouts with frozen actor + Hebbian)

Results are saved as CSV for easy aggregation across parameter sweeps.
"""

from __future__ import annotations

import csv
import gc
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from multi_urdf_utils.config import BenchmarkConfig


def _resolve_num_workers(cfg_num_workers: int, cpu_threads_per_worker: int) -> int:
    """Return effective worker count: cfg value or cpu_count // cpu_threads_per_worker."""
    import os
    if cfg_num_workers > 0:
        return cfg_num_workers
    return max(1, (os.cpu_count() or 4) // cpu_threads_per_worker)


def _build_scene_inputs(cfg, urdf_paths_str):
    """Shared setup: resolve hebb_cfg, wp1_cfg, urdf batches.  Returns plain dicts
    suitable for pickling into worker processes."""
    import torch
    from WP1.config import RunConfig
    from WP2_old.config import HebbianConfig
    from dataclasses import asdict

    N = cfg.benchmark.N
    S = cfg.benchmark.S

    # Infer last-layer dims from checkpoint
    hebb_cfg = HebbianConfig(
        enabled=cfg.hebbian.enabled,
        eta=cfg.hebbian.eta,
        w_max=cfg.hebbian.w_max,
    )
    ckpt = torch.load(cfg.checkpoint.model_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if "actor.4.weight" in sd:
        hebb_cfg.num_actions = sd["actor.4.weight"].shape[0]
        hebb_cfg.hidden_dim = sd["actor.4.weight"].shape[1]
    del ckpt, sd

    hebb_cfg_dict = {
        "enabled": hebb_cfg.enabled,
        "eta": hebb_cfg.eta,
        "w_max": hebb_cfg.w_max,
        "num_actions": hebb_cfg.num_actions,
        "hidden_dim": hebb_cfg.hidden_dim,
        "vmin": cfg.env.vmin,
        "vmax": cfg.env.vmax,
    }

    # Build WP1 config and apply benchmark overrides
    wp1_cfg = RunConfig.from_yaml(cfg.checkpoint.config_path)
    wp1_cfg.env.episode_length_s = cfg.env.episode_length_s
    wp1_cfg.env.dens_min = cfg.env.forest_density_min
    wp1_cfg.env.dens_max = cfg.env.forest_density_max
    wp1_cfg.env.growing_forest = cfg.env.growing_forest
    wp1_cfg.env.tree_radius = cfg.env.tree_radius
    wp1_cfg.env.tree_height = cfg.env.tree_height
    wp1_cfg.env.base_init_pos = cfg.env.base_init_pos
    wp1_cfg.env.base_init_quat = cfg.env.base_init_quat
    # Apply aero noise config from benchmark (may differ from training config)
    wp1_cfg.env.aero_noise = cfg.env.aero_noise
    # Apply dt from benchmark config (overrides WP1 training value)
    wp1_cfg.env.dt = cfg.env.dt
    wp1_cfg_dict = asdict(wp1_cfg)

    urdf_batches = [urdf_paths_str[i * N : (i + 1) * N] for i in range(S)]

    return urdf_batches, wp1_cfg_dict, hebb_cfg_dict


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, float]:
    """Execute a multi-scene benchmark run.

    S scenes each contain N URDF entities sharing E environments (N × E
    drone instances per scene, S × N × E total).  Scenes can be compiled
    and simulated in parallel via ``benchmark.num_workers`` worker processes
    (``num_workers=0`` auto-selects ``cpu_count // cpu_threads_per_worker``).

    Parameters
    ----------
    cfg : BenchmarkConfig
        Full benchmark configuration.

    Returns
    -------
    dict
        Timing results and metadata aggregated across all scenes.
    """
    import genesis as gs
    from general_policy.catalog import build_catalog

    N = cfg.benchmark.N
    S = cfg.benchmark.S
    E = cfg.benchmark.E
    D = cfg.total_entities
    device = cfg.device
    num_workers = _resolve_num_workers(cfg.benchmark.num_workers, cfg.benchmark.cpu_threads_per_worker)

    print(f"\n{'='*70}")
    print(f"  WP2.5 Benchmark")
    print(f"{'='*70}")
    print(f"  N={N} URDFs/scene  S={S} scenes  E={E} envs/scene")
    print(f"  Total URDFs={D}  total_instances={cfg.total_instances}")
    print(f"  workers={num_workers}  threads/compiler={cfg.benchmark.cpu_threads_per_worker}")
    print(f"  episodes={cfg.benchmark.num_episodes}  max_steps={cfg.benchmark.max_steps}")
    print(f"  seed={cfg.benchmark.seed}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------ #
    # Phase 1: URDF generation (main process only)                       #
    # ------------------------------------------------------------------ #
    print("[Phase 1] Generating URDFs...")
    t0 = time.time()

    if not gs._initialized:
        gs.init(logging_level="error", backend=gs.gpu)

    catalog_dir = Path(cfg.catalog.catalog_dir)
    urdf_paths = build_catalog(
        catalog_dir=catalog_dir,
        n=D,
        seed=cfg.catalog.urdf_seed,
        include_standard_mydrone=True,
    )
    urdf_paths_str = [str(p) for p in urdf_paths]
    while len(urdf_paths_str) < D:
        urdf_paths_str.append(urdf_paths_str[len(urdf_paths_str) % len(urdf_paths)])

    t_urdf = time.time() - t0
    print(f"  -> {len(urdf_paths_str)} URDFs in {t_urdf:.2f}s\n")

    gs.destroy()

    # ------------------------------------------------------------------ #
    # Shared setup (serialisable for worker processes)                   #
    # ------------------------------------------------------------------ #
    urdf_batches, wp1_cfg_dict, hebb_cfg_dict = _build_scene_inputs(cfg, urdf_paths_str)

    # ------------------------------------------------------------------ #
    # Phases 2 + 3: compile + simulate (parallel or sequential)          #
    # ------------------------------------------------------------------ #
    if num_workers > 1:
        print(f"[Phase 2+3] Parallel mode: {num_workers} workers\n")
        t_phases_start = time.time()

        from multi_urdf_utils.orchestrator import run_scenes_parallel
        scene_results = run_scenes_parallel(
            urdf_batches=urdf_batches,
            E=E,
            device=device,
            base_seed=cfg.benchmark.seed,
            num_episodes=cfg.benchmark.num_episodes,
            max_steps=cfg.benchmark.max_steps,
            wp1_cfg_dict=wp1_cfg_dict,
            hebb_cfg_dict=hebb_cfg_dict,
            checkpoint_model_path=cfg.checkpoint.model_path,
            checkpoint_config_path=cfg.checkpoint.config_path,
            num_workers=num_workers,
            cpu_threads_per_worker=cfg.benchmark.cpu_threads_per_worker,
        )

        t_phases_wall = time.time() - t_phases_start
        # Wall-clock bottleneck (slowest worker); sum would double-count parallel work
        compile_times = [r["compile_time_s"] for r in scene_results]
        sim_times = [r["sim_time_s"] for r in scene_results]
        t_compile_total = max(compile_times)
        t_sim_total = max(sim_times)
        t_compile_min = min(compile_times)
        t_sim_min = min(sim_times)
        t_compile_mean = float(np.mean(compile_times))
        t_sim_mean = float(np.mean(sim_times))
        all_ep_times = [t for r in scene_results for t in r.get("episode_sim_times", [])]
        total_steps = sum(r["total_steps"] for r in scene_results)
        ram_per_scene_mb = [r["ram_mb"] for r in scene_results]
        vram_alloc_per_scene_mb = [r["vram_allocated_mb"] for r in scene_results]
        vram_reserved_per_scene_mb = [r["vram_reserved_mb"] for r in scene_results]

    else:
        print("[Phase 2+3] Sequential mode\n")
        from WP1.config import RunConfig
        from WP2_old.config import HebbianConfig
        from multi_urdf_utils.multi_drone_env import MultiDroneEnv
        from multi_urdf_utils.multi_drone_actor import MultiDroneActorManager, random_hebbian_rules

        # Rebuild objects in main process for sequential path
        from dataclasses import fields as dc_fields
        hebb_cfg = HebbianConfig(
            enabled=hebb_cfg_dict["enabled"],
            eta=hebb_cfg_dict["eta"],
            w_max=hebb_cfg_dict["w_max"],
            num_actions=hebb_cfg_dict["num_actions"],
            hidden_dim=hebb_cfg_dict["hidden_dim"],
        )
        wp1_cfg = RunConfig._from_dict(wp1_cfg_dict)

        t_compile_total = 0.0
        t_sim_total = 0.0
        total_steps = 0
        compile_times = []
        sim_times = []
        all_ep_times = []
        ram_per_scene_mb = []
        vram_alloc_per_scene_mb = []
        vram_reserved_per_scene_mb = []

        for scene_idx, scene_urdf_paths in enumerate(urdf_batches):
            scene_seed = cfg.benchmark.seed + scene_idx * N
            print(f"[Scene {scene_idx + 1}/{S}] Building ({N} URDFs, {E} envs)...")
            t1 = time.time()

            gs.init(logging_level="error", backend=gs.gpu)

            env = MultiDroneEnv(
                urdf_paths=scene_urdf_paths,
                num_envs=E,
                wp1_cfg=wp1_cfg,
                device=device,
                vmin=cfg.env.vmin,
                vmax=cfg.env.vmax,
            )

            use_hebbian = cfg.hebbian.enabled

            if use_hebbian:
                hebbian_rules_list = [
                    random_hebbian_rules(
                        hebb_cfg,
                        out_features=hebb_cfg.num_actions,
                        in_features=hebb_cfg.hidden_dim,
                        seed=scene_seed + i,
                        device=device,
                    )
                    for i in range(N)
                ]
            else:
                hebbian_rules_list = None

            actor_mgr = MultiDroneActorManager(
                D=N,
                checkpoint_path=cfg.checkpoint.model_path,
                checkpoint_config_path=cfg.checkpoint.config_path,
                hebbian_rules_list=hebbian_rules_list,
                hebb_cfg=hebb_cfg,
                stochastic=True,
                device=device,
                use_hebbian=use_hebbian,
            )

            t_compile = time.time() - t1
            t_compile_total += t_compile
            compile_times.append(t_compile)
            print(f"  -> built in {t_compile:.2f}s")

            import psutil as _psutil
            _proc = _psutil.Process()
            ram_per_scene_mb.append(round(_proc.memory_info().rss / 1024 ** 2, 1))
            _dev_idx = int(device.split(":")[-1]) if ":" in device else 0
            vram_alloc_per_scene_mb.append(round(torch.cuda.memory_allocated(_dev_idx) / 1024 ** 2, 1))
            vram_reserved_per_scene_mb.append(round(torch.cuda.memory_reserved(_dev_idx) / 1024 ** 2, 1))

            t2 = time.time()
            for ep in range(cfg.benchmark.num_episodes):
                ep_start = time.time()
                actor_mgr.reset_episode(E, device)
                obs, _ = env.reset()
                done = torch.zeros(N, E, dtype=torch.bool, device=torch.device(device))
                ep_steps = 0

                for _ in range(cfg.benchmark.max_steps):
                    with torch.no_grad():
                        actions = actor_mgr.act(obs)
                    obs, rew, term, info = env.step(actions)
                    done |= term
                    ep_steps += 1
                    total_steps += 1
                    if done.all():
                        break

                ep_time = time.time() - ep_start
                all_ep_times.append(ep_time)
                print(f"  Episode {ep+1}/{cfg.benchmark.num_episodes}: "
                      f"{ep_steps} steps in {ep_time:.2f}s  "
                      f"alive={(~done).float().mean().item():.1%}")

            t_sim_scene = time.time() - t2
            t_sim_total += t_sim_scene
            sim_times.append(t_sim_scene)
            print(f"  -> sim done in {t_sim_scene:.2f}s\n")

            gs.destroy()
            del env, actor_mgr, hebbian_rules_list
            gc.collect()

        t_phases_wall = t_compile_total + t_sim_total  # sequential: wall == cpu
        t_compile_min = min(compile_times)
        t_sim_min = min(sim_times)
        t_compile_mean = float(np.mean(compile_times))
        t_sim_mean = float(np.mean(sim_times))

    # ------------------------------------------------------------------ #
    # Results                                                             #
    # ------------------------------------------------------------------ #
    total_time = t_urdf + t_phases_wall
    steps_per_sec = total_steps / max(t_sim_total, 1e-6)
    instances_steps_per_sec = (N * E * total_steps) / max(t_sim_total, 1e-6)

    results = {
        "N": N,
        "S": S,
        "E": E,
        "num_workers": num_workers,
        "D_total_entities": D,
        "total_instances": cfg.total_instances,
        "seed": cfg.benchmark.seed,
        "num_episodes": cfg.benchmark.num_episodes,
        "max_steps": cfg.benchmark.max_steps,
        "total_steps_executed": total_steps,
        "urdf_gen_time_s": round(t_urdf, 3),
        "compile_time_s": round(t_compile_total, 3),
        "compile_time_mean_s": round(t_compile_mean, 3),
        "compile_time_min_s": round(t_compile_min, 3),
        "sim_time_s": round(t_sim_total, 3),
        "sim_time_mean_s": round(t_sim_mean, 3),
        "sim_time_min_s": round(t_sim_min, 3),
        "sim_time_per_ep_max_s": round(float(np.max(all_ep_times)), 3) if all_ep_times else None,
        "sim_time_per_ep_min_s": round(float(np.min(all_ep_times)), 3) if all_ep_times else None,
        "sim_time_per_ep_mean_s": round(float(np.mean(all_ep_times)), 3) if all_ep_times else None,
        "total_time_s": round(total_time, 3),
        "compile_fraction": round(t_compile_total / max(total_time, 1e-6), 4),
        "steps_per_second": round(steps_per_sec, 2),
        "instance_steps_per_second": round(instances_steps_per_sec, 2),
        "hebbian_enabled": cfg.hebbian.enabled,
        # memory per scene (after compilation, before gs.destroy)
        "ram_per_scene_mean_mb": round(float(np.mean(ram_per_scene_mb)), 1) if ram_per_scene_mb else None,
        "ram_per_scene_max_mb": round(float(np.max(ram_per_scene_mb)), 1) if ram_per_scene_mb else None,
        "vram_allocated_per_scene_mean_mb": round(float(np.mean(vram_alloc_per_scene_mb)), 1) if vram_alloc_per_scene_mb else None,
        "vram_allocated_per_scene_max_mb": round(float(np.max(vram_alloc_per_scene_mb)), 1) if vram_alloc_per_scene_mb else None,
        "vram_reserved_per_scene_mean_mb": round(float(np.mean(vram_reserved_per_scene_mb)), 1) if vram_reserved_per_scene_mb else None,
        "vram_reserved_per_scene_max_mb": round(float(np.max(vram_reserved_per_scene_mb)), 1) if vram_reserved_per_scene_mb else None,
    }

    return results


def save_results(results: Dict[str, float], output_dir: Path) -> Path:
    """Save benchmark results to CSV (append if file exists)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"

    file_exists = csv_path.is_file()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    return csv_path


def format_results_table(results: Dict[str, float]) -> str:
    """Format benchmark results as a pretty-printed string."""
    parallel = results["num_workers"] > 1
    label = " (slowest worker)" if parallel else ""
    lines = [
        f"\n{'='*70}",
        f"  BENCHMARK RESULTS",
        f"{'='*70}",
        f"  Layout:       N={results['N']} URDFs/scene  S={results['S']} scenes  E={results['E']} envs/scene  workers={results['num_workers']}",
        f"  Total URDFs:  {results['D_total_entities']}",
        f"  Instances:    {results['total_instances']}  (S×N×E)",
        f"  Steps:        {results['total_steps_executed']}",
        f"  ─────────────────────────────────────",
        f"  URDF gen:     {results['urdf_gen_time_s']:.3f} s",
        f"  Compilation:  {results['compile_time_s']:.3f} s{label}",
        *(
            [
                f"                {results['compile_time_mean_s']:.3f} s (mean worker)",
                f"                {results['compile_time_min_s']:.3f} s (fastest worker)",
            ]
            if parallel else []
        ),
        f"  Simulation:   {results['sim_time_s']:.3f} s{label}",
        *(
            [
                f"                {results['sim_time_mean_s']:.3f} s (mean worker)",
                f"                {results['sim_time_min_s']:.3f} s (fastest worker)",
            ]
            if parallel else []
        ),
        f"  Sim/episode:  {results['sim_time_per_ep_mean_s']:.3f} s mean  /  "
        f"{results['sim_time_per_ep_min_s']:.3f} s min  /  "
        f"{results['sim_time_per_ep_max_s']:.3f} s max",
        f"  Total (wall): {results['total_time_s']:.3f} s",
        f"  ─────────────────────────────────────",
        f"  Compile frac: {results['compile_fraction']:.1%}",
        f"  Steps/sec:    {results['steps_per_second']:.1f}",
        f"  Inst×steps/s: {results['instance_steps_per_second']:.0f}",
        f"  ─────────────────────────────────────",
        f"  RAM/scene:    {results['ram_per_scene_mean_mb']} MB mean  /  {results['ram_per_scene_max_mb']} MB max",
        f"  VRAM alloc:   {results['vram_allocated_per_scene_mean_mb']} MB mean  /  {results['vram_allocated_per_scene_max_mb']} MB max",
        f"  VRAM reserv:  {results['vram_reserved_per_scene_mean_mb']} MB mean  /  {results['vram_reserved_per_scene_max_mb']} MB max",
        f"{'='*70}\n",
    ]
    return "\n".join(lines)


def print_results(results: Dict[str, float]):
    """Pretty-print benchmark results."""
    print(format_results_table(results))


def save_table(results: Dict[str, float], output_dir: Path) -> Path:
    """Save the pretty-printed results table to a .txt file in output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "benchmark_table.txt"
    with open(txt_path, "w") as f:
        f.write(format_results_table(results))
    return txt_path
