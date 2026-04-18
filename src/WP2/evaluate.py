"""
Evaluation pipeline for WP2 — rollout frozen actor + Hebbian on Genesis.
========================================================================

CMA-ES evaluation path:
  ``evaluate_population_cma_batched`` — evaluates a full CMA-ES generation
  across all catalog URDFs (or a single default URDF).  Returns WP1 reward
  sums as the scalar fitness signal.

Morphology is always fixed.  Each individual's Hebbian rules are decoded
from its genome and applied to independent last-layer weight copies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from WP2.config import HebbianEvolutionConfig
from WP2.frozen_actor import (
    IsolatedPopulationActor,
    build_isolated_population_actor,
    load_frozen_actor,
)
from WP2.utils import decode_hebbian_genes


# ============================================================================
#  Environment construction
# ============================================================================

def _build_env(
    cfg: HebbianEvolutionConfig,
    wp1_cfg,
    device: str,
    num_envs_override: Optional[int] = None,
    base_init_pos: Optional[List[float]] = None,
):
    """Build a WingedDroneEnv using the WP1 default morphology."""
    from morph_evolution.chromosome_drone import Chromosome_Drone
    from winged_drone_train.env import WingedDroneEnv
    from winged_drone_train.noise_config import configure_solver_noise
    from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME, default_mydrone_urdf_path

    phys = list(STANDARD_MYDRONE_GENOME)
    urdf_path = str(default_mydrone_urdf_path())
    naca = Chromosome_Drone.naca_from_physical(phys) or "3416"

    env_cfg = wp1_cfg.to_env_cfg()
    obs_cfg = wp1_cfg.to_obs_cfg()
    reward_cfg = wp1_cfg.to_reward_cfg()
    command_cfg = wp1_cfg.to_command_cfg()

    env_cfg.update(dict(
        visualize_camera=False,
        visualize_target=False,
        naca=naca,
    ))
    command_cfg["min_speed"] = cfg.evaluation.vmin
    command_cfg["max_speed"] = cfg.evaluation.vmax
    if cfg.evaluation.x_upper is not None:
        env_cfg["x_upper"] = cfg.evaluation.x_upper
        env_cfg["forest_x_limit"] = cfg.evaluation.x_upper
    if base_init_pos is not None:
        env_cfg["base_init_pos"] = list(base_init_pos)

    obs_cfg["add_genome_obs_actor"] = False
    obs_cfg["add_genome_obs_critic"] = False

    num_envs = num_envs_override if num_envs_override is not None else cfg.evaluation.num_eval_envs

    env = WingedDroneEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        urdf_file=urdf_path,
        show_viewer=False,
        eval=True,
        device=device,
    )
    configure_solver_noise(env, env_cfg)

    return env, urdf_path


# ============================================================================
#  CMA-ES evaluation
# ============================================================================

@torch.no_grad()
def _rollout_episode_reward_sum(
    env,
    actor: IsolatedPopulationActor,
    device: str,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Run one episode and return WP1 reward sum per environment."""
    B = env.num_envs
    dt = env.dt
    dev = torch.device(device)

    done = torch.zeros(B, dtype=torch.bool, device=dev)
    reward_sum = torch.zeros(B, device=dev)
    t_acc = torch.zeros(B, device=dev)
    dx_acc = torch.zeros(B, device=dev)
    energy_acc = torch.zeros(B, device=dev)
    v_dev_acc = torch.zeros(B, device=dev)
    crashed = torch.zeros(B, dtype=torch.bool, device=dev)

    actor.reset_episode(device=dev)
    obs, _ = env.reset()
    x0 = env.base_pos[:, 0].clone()

    step = 0
    while not done.all():
        actions = actor.act(obs)
        obs, _, term, _ = env.step(actions)
        term = term.bool()
        nan_mask = env.nan_envs.to(torch.bool)

        alive = (~done) & (~term) & (~nan_mask)

        if alive.any():
            reward_sum[alive] += env.last_reward_total[alive]
            t_acc[alive] += dt
            dx_acc[alive] = env.base_pos[alive, 0] - x0[alive]
            energy_acc[alive] += env.power[alive] * dt
            v_dev_acc[alive] += (env.base_lin_vel[alive, 0] - env.commands[alive, 0]).abs() * dt

        just_done = (~done) & term
        if just_done.any():
            for attr in ("pre_collision", "pre_wall_crash", "pre_angle_limit"):
                flag = getattr(env, attr, None)
                if flag is not None:
                    crashed |= just_done & flag.to(torch.bool)

        done |= term | nan_mask
        step += 1

        if verbose and step % 50 == 0:
            n_alive = int((~done).sum().item())
            mean_dx = float(dx_acc.mean().item())
            max_dx = float(dx_acc.max().item())
            mean_r = float(reward_sum.mean().item())
            print(
                f"  step {step:5d} | alive {n_alive:5d}/{B}"
                f" | progress mean {mean_dx:7.1f} m  max {max_dx:7.1f} m"
                f" | reward mean {mean_r:8.2f}",
                flush=True,
            )

    valid = ~env.nan_envs.to(torch.bool)
    nan_np = (~valid).cpu().numpy()

    reward_arr = reward_sum.cpu().numpy()
    t_arr = t_acc.cpu().numpy()
    dx_arr = dx_acc.cpu().numpy()
    energy_arr = energy_acc.cpu().numpy()
    v_dev_arr = v_dev_acc.cpu().numpy()
    crash_arr = crashed.float().cpu().numpy()

    reward_arr[nan_np] = 0.0
    dx_arr[nan_np] = 0.0
    t_arr[nan_np] = 0.0
    energy_arr[nan_np] = 0.0
    v_dev_arr[nan_np] = 0.0

    v_arr = np.where(t_arr > 1e-6, dx_arr / t_arr, 0.0)
    v_dev_arr = np.where(t_arr > 1e-6, v_dev_arr / t_arr, 0.0)

    # Cost of Transport = Energy / (mass * gravity * displacement)
    mg = float(env.nominal_mass) * 9.81
    cot_arr = np.where(dx_arr > 1e-6, energy_arr / (mg * dx_arr), 0.0)

    return {
        "reward_sum": reward_arr,
        "progresses": dx_arr,
        "velocities": v_arr,
        "crash_flags": crash_arr,
        "cots": cot_arr,
        "v_deviations": v_dev_arr,
    }


def _load_catalog(catalog_path: str) -> List[Tuple[str, str]]:
    """Parse a catalog.txt file and return list of (urdf_path, naca) tuples."""
    import ast
    import re
    from morph_evolution.chromosome_drone import Chromosome_Drone

    catalog_path = Path(catalog_path)
    catalog_dir = catalog_path.parent

    entries = []
    with open(catalog_path, "r") as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue

            m = re.match(r'\[(.+)\]\.urdf$', filename)
            if m:
                try:
                    phys = list(ast.literal_eval(f"[{m.group(1)}]"))
                    naca = Chromosome_Drone.naca_from_physical(phys) or "3416"
                except Exception:
                    naca = "3416"
            else:
                naca = "3416"

            urdf_file = str(catalog_dir / filename)
            entries.append((urdf_file, naca))

    return entries


def _build_env_from_urdf(
    urdf_file: str,
    naca: str,
    cfg,
    wp1_cfg,
    device: str,
    num_envs: int,
):
    """Build a WingedDroneEnv from an already-existing URDF file."""
    from winged_drone_train.env import WingedDroneEnv
    from winged_drone_train.noise_config import configure_solver_noise

    env_cfg = wp1_cfg.to_env_cfg()
    obs_cfg = wp1_cfg.to_obs_cfg()
    reward_cfg = wp1_cfg.to_reward_cfg()
    command_cfg = wp1_cfg.to_command_cfg()

    env_cfg.update(dict(
        visualize_camera=False,
        visualize_target=False,
        naca=naca,
    ))
    command_cfg["min_speed"] = cfg.evaluation.vmin
    command_cfg["max_speed"] = cfg.evaluation.vmax
    if cfg.evaluation.x_upper is not None:
        env_cfg["x_upper"] = cfg.evaluation.x_upper
        env_cfg["forest_x_limit"] = cfg.evaluation.x_upper
    obs_cfg["add_genome_obs_actor"] = False
    obs_cfg["add_genome_obs_critic"] = False

    env = WingedDroneEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        urdf_file=urdf_file,
        show_viewer=False,
        eval=True,
        device=device,
    )
    configure_solver_noise(env, env_cfg)
    return env


def evaluate_population_cma_batched(
    solutions: List[np.ndarray],
    cfg: HebbianEvolutionConfig,
    model_and_layer: Tuple,
    wp1_cfg,
    catalog: Optional[List[Tuple[str, str]]] = None,
    existing_env=None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Evaluate a CMA-ES population across all catalog URDFs.

    One ``IsolatedPopulationActor`` is built per generation (new rules each
    generation); it is reused across every URDF in the catalog.  Each env
    slot has its own LSTM + last-layer weights.

    Parameters
    ----------
    solutions : list of np.ndarray
        Raw genomes (each in [0, 1]) from CMA-ES ``ask()``.
    cfg : HebbianEvolutionConfig
    model_and_layer : tuple
        ``(model, last_layer, num_actions, hidden_dim)`` from ``load_frozen_actor``.
    wp1_cfg : RunConfig
    catalog : list of (urdf_path, naca), optional
        If None or empty, uses a single default URDF.
    existing_env : tuple (env, urdf_path), optional
        Pre-built environment to reuse (only when catalog is empty).

    Returns
    -------
    fitnesses : np.ndarray, shape (P,)
        Mean WP1 reward sum per individual, averaged over URDFs and episodes.
    metrics : dict
        Additional per-individual metrics (progresses, velocities, crash_flags).
    """
    import genesis as gs

    P = len(solutions)
    total_envs = cfg.evaluation.num_eval_envs
    S = total_envs // P
    actual_envs = S * P

    if S == 0:
        raise ValueError(
            f"num_eval_envs ({total_envs}) is smaller than population size ({P}). "
            "Increase evaluation.num_eval_envs or reduce population_size."
        )

    if verbose:
        print(
            f"[evaluate_population_cma_batched] P={P} individuals, "
            f"S={S} envs/ind, total={actual_envs} envs"
        )

    na, hd = cfg.hebbian.num_actions, cfg.hebbian.hidden_dim

    # Decode all genomes into Hebbian rules
    hebbian_rules_per_individual: List[Dict[str, torch.Tensor]] = []
    for genome in solutions:
        hebb_part = list(np.clip(genome, 0.0, 1.0))
        rules = decode_hebbian_genes(hebb_part, cfg.hebbian, out_features=na, in_features=hd)
        hebbian_rules_per_individual.append(rules)

    actor = build_isolated_population_actor(
        checkpoint_path=cfg.checkpoint_path,
        wp1_cfg_path=cfg.checkpoint_config_path,
        hebbian_rules_per_individual=hebbian_rules_per_individual,
        cfg=cfg,
        K=P,
        S=S,
        device=cfg.device,
        stochastic=cfg.evaluation.stochastic,
    )

    if not catalog:
        catalog = [(None, None)]
    n_urdfs = len(catalog)
    n_episodes = cfg.catalog.num_episodes

    acc_reward = np.zeros(P)
    acc_progress = np.zeros(P)
    acc_velocity = np.zeros(P)
    acc_crash = np.zeros(P)
    acc_cot = np.zeros(P)
    acc_v_dev = np.zeros(P)

    env_was_reused = False
    env = None
    if (existing_env is not None and n_urdfs == 1 and catalog[0][0] is None):
        existing_env_obj, _ = existing_env
        if existing_env_obj.num_envs == actual_envs:
            env = existing_env_obj
            env_was_reused = True
            if verbose:
                print(
                    f"[evaluate_population_cma_batched] Reusing environment "
                    f"(rules-only, actual_envs={actual_envs})"
                )

    for urdf_idx, (urdf_file, naca) in enumerate(catalog):
        if verbose:
            print(
                f"  [catalog {urdf_idx + 1}/{n_urdfs}] "
                f"{'default URDF' if urdf_file is None else Path(urdf_file).name}"
            )

        if not env_was_reused:
            try:
                if not gs._initialized:
                    gs.init(logging_level="error", backend=gs.gpu)

                if urdf_file is None:
                    env, _ = _build_env(cfg, wp1_cfg, cfg.device, num_envs_override=actual_envs)
                else:
                    env = _build_env_from_urdf(
                        urdf_file, naca, cfg, wp1_cfg, cfg.device, num_envs=actual_envs
                    )
            except Exception as exc:
                print(f"  [catalog {urdf_idx + 1}] env build failed: {exc} — skipping")
                if gs._initialized:
                    try:
                        gs.destroy()
                    except Exception:
                        pass
                continue

        urdf_reward = np.zeros(P)
        urdf_progress = np.zeros(P)
        urdf_velocity = np.zeros(P)
        urdf_crash = np.zeros(P)
        urdf_cot = np.zeros(P)
        urdf_v_dev = np.zeros(P)

        try:
            for ep in range(n_episodes):
                ep_metrics = _rollout_episode_reward_sum(env, actor, cfg.device, verbose=verbose)

                for key, flat_arr in ep_metrics.items():
                    per_ind = flat_arr.reshape(P, S).mean(axis=1)
                    if key == "reward_sum":
                        urdf_reward += per_ind
                    elif key == "progresses":
                        urdf_progress += per_ind
                    elif key == "velocities":
                        urdf_velocity += per_ind
                    elif key == "crash_flags":
                        urdf_crash += per_ind
                    elif key == "cots":
                        urdf_cot += per_ind
                    elif key == "v_deviations":
                        urdf_v_dev += per_ind

        except Exception as exc:
            print(f"  [catalog {urdf_idx + 1}] rollout failed: {exc}")
        finally:
            if not env_was_reused:
                gs.destroy()

        urdf_reward /= n_episodes
        urdf_progress /= n_episodes
        urdf_velocity /= n_episodes
        urdf_crash /= n_episodes
        urdf_cot /= n_episodes
        urdf_v_dev /= n_episodes

        acc_reward += urdf_reward
        acc_progress += urdf_progress
        acc_velocity += urdf_velocity
        acc_crash += urdf_crash
        acc_cot += urdf_cot
        acc_v_dev += urdf_v_dev

        if verbose:
            print(
                f"    best={urdf_reward.max():.4f}  mean={urdf_reward.mean():.4f}  "
                f"crash={urdf_crash.mean() * 100:.1f}%"
            )

    acc_reward /= n_urdfs
    acc_progress /= n_urdfs
    acc_velocity /= n_urdfs
    acc_crash /= n_urdfs
    acc_cot /= n_urdfs
    acc_v_dev /= n_urdfs

    metrics = {
        "reward_sums": acc_reward,
        "progresses": acc_progress,
        "velocities": acc_velocity,
        "crash_flags": acc_crash,
        "cots": acc_cot,
        "v_deviations": acc_v_dev,
    }
    return acc_reward, metrics


# ============================================================================
#  Standalone driver
# ============================================================================

if __name__ == "__main__":
    """Evaluate a single saved Hebbian genome against a Genesis forest env.

    Usage
    -----
    # Evaluate the best genome from a completed run (default forest length):
        python -m WP2.evaluate --run logs/runs_hebbian/2026-xx-xx_my_run

    # Override forest length to 1200 m:
        python -m WP2.evaluate --run logs/runs_hebbian/2026-xx-xx_my_run --x-upper 1200

    # Point at an explicit genome file:
        python -m WP2.evaluate --run logs/runs_hebbian/2026-xx-xx_my_run \\
            --genome path/to/genome.npy --x-upper 800

    # Evaluate a specific generation's best individual:
        python -m WP2.evaluate --run logs/runs_hebbian/2026-xx-xx_my_run \\
            --genome logs/runs_hebbian/2026-xx-xx_my_run/generations/gen_042/solutions.npy \\
            --genome-idx 0

    # Compare Hebbian vs. frozen baseline (zero rules):
        python -m WP2.evaluate --run logs/runs_hebbian/2026-xx-xx_my_run --compare
    """
    import argparse
    import os
    import sys
    from pathlib import Path

    _src_dir = Path(__file__).resolve().parent.parent
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

    parser = argparse.ArgumentParser(
        description="Standalone evaluation of a saved Hebbian genome."
    )
    parser.add_argument(
        "--run", type=str, required=True,
        help="Path to a completed WP2 run directory.",
    )
    parser.add_argument(
        "--genome", type=str, default=None,
        help="Path to a .npy genome file.  Defaults to <run>/best/genome.npy.",
    )
    parser.add_argument(
        "--genome-idx", type=int, default=0,
        help="Row index to use when the .npy file contains multiple genomes (e.g. solutions.npy).",
    )
    parser.add_argument(
        "--x-upper", type=float, default=None,
        help="Override forest corridor length in metres (WP1 env x_upper).  "
             "Default: value from the saved WP1 config (typically 600 m for eval).",
    )
    parser.add_argument(
        "--num-envs", type=int, default=None,
        help="Override evaluation.num_eval_envs.",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of rollout episodes to average over.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device override (e.g. cuda:0).",
    )
    parser.add_argument(
        "--stochastic", action=argparse.BooleanOptionalAction, default=None,
        help="Sample from the policy distribution (--stochastic) or use the mean "
             "(--no-stochastic). Overrides evaluation.stochastic from the run config.",
    )
    parser.add_argument(
        "--base-init-pos", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
        help="Override base_init_pos (drone spawn location). Example: --base-init-pos -50 0 10",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also evaluate the frozen baseline (zero Hebbian rules) and print a comparison.",
    )
    args = parser.parse_args()

    # --- resolve paths ---
    run_dir = Path(args.run)
    cfg_path = run_dir / "reproducibility" / "config.yaml"
    if not cfg_path.is_file():
        sys.exit(f"[ERROR] Config not found: {cfg_path}")

    genome_path = Path(args.genome) if args.genome else run_dir / "best_individual" / "genome.npy"
    if not genome_path.is_file():
        sys.exit(f"[ERROR] Genome file not found: {genome_path}")

    # --- load WP2 config ---
    from WP2.config import HebbianEvolutionConfig
    cfg = HebbianEvolutionConfig.from_yaml(cfg_path)

    if args.device:
        cfg.device = args.device
    if args.num_envs:
        cfg.evaluation.num_eval_envs = args.num_envs
    if args.stochastic is not None:
        cfg.evaluation.stochastic = args.stochastic
    cfg.catalog.num_episodes = args.episodes

    # Infer last-layer dims from checkpoint (same as run.py does)
    import torch
    _ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    _sd = _ckpt.get("model_state_dict", _ckpt) if isinstance(_ckpt, dict) else _ckpt
    if "actor.4.weight" in _sd:
        cfg.hebbian.num_actions = _sd["actor.4.weight"].shape[0]
        cfg.hebbian.hidden_dim = _sd["actor.4.weight"].shape[1]
    del _ckpt, _sd

    # --- load WP1 config and optionally override forest length ---
    from WP1.config import RunConfig
    wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)

    if args.x_upper is not None:
        wp1_cfg.env.x_upper = args.x_upper
        wp1_cfg.env.forest_x_limit = args.x_upper
        cfg.evaluation.x_upper = args.x_upper  # ensure _build_env also uses the override
        print(f"[eval] Forest length overridden to {args.x_upper} m")
    else:
        print(f"[eval] Forest length: {wp1_cfg.env.x_upper} m (from WP1 config)")

    # --- load genome ---
    genome_arr = np.load(genome_path)
    if genome_arr.ndim == 2:
        genome = genome_arr[args.genome_idx]
        print(f"[eval] Loaded genome row {args.genome_idx} from {genome_path.name}  "
              f"(shape {genome_arr.shape})")
    else:
        genome = genome_arr
        print(f"[eval] Loaded genome from {genome_path.name}  (dim={genome.size})")

    # --- load frozen actor ---
    from WP2.frozen_actor import load_frozen_actor
    model_and_layer = load_frozen_actor(
        cfg.checkpoint_path, cfg.checkpoint_config_path, cfg.device
    )

    # --- Genesis init + build env ---
    import genesis as gs
    gs.init(logging_level="error", backend=gs.gpu)

    # Build a single env and reuse it across all evaluations so Genesis is
    # only initialised/destroyed once (existing_env bypasses gs.destroy inside
    # evaluate_population_cma_batched).
    shared_env, urdf_path = _build_env(
        cfg, wp1_cfg, cfg.device, base_init_pos=args.base_init_pos
    )
    existing_env = (shared_env, urdf_path)

    print(
        f"\n[eval] Running evaluation: "
        f"envs={cfg.evaluation.num_eval_envs}  "
        f"episodes={cfg.catalog.num_episodes}  "
        f"stochastic={cfg.evaluation.stochastic}"
    )

    fitnesses, metrics = evaluate_population_cma_batched(
        solutions=[genome],
        cfg=cfg,
        model_and_layer=model_and_layer,
        wp1_cfg=wp1_cfg,
        existing_env=existing_env,
        verbose=True,
    )

    hebb_reward   = fitnesses[0]
    hebb_progress = metrics['progresses'][0]
    hebb_velocity = metrics['velocities'][0]
    hebb_crash    = metrics['crash_flags'][0]
    hebb_cot      = metrics['cots'][0]

    print("\n" + "=" * 50)
    print("  [Hebbian]")
    print(f"  reward   : {hebb_reward:.4f}")
    print(f"  progress : {hebb_progress:.1f} m")
    print(f"  velocity : {hebb_velocity:.2f} m/s")
    print(f"  crash    : {hebb_crash * 100:.1f}%")
    print(f"  cot      : {hebb_cot:.4f}")
    print("=" * 50)

    if args.compare:
        # ------------------------------------------------------------------
        # Build a genome that decodes to all-zero ABCD rules (no plasticity).
        # gene * (hi - lo) + lo = 0  =>  gene = -lo / (hi - lo)
        # ------------------------------------------------------------------
        def _gene_for_zero(lo: float, hi: float) -> float:
            return -lo / (hi - lo) if hi != lo else 0.0

        na, hd = cfg.hebbian.num_actions, cfg.hebbian.hidden_dim
        n_weights = na * hd
        baseline_parts = [
            np.full(n_weights, _gene_for_zero(*cfg.hebbian.A_range)),
            np.full(n_weights, _gene_for_zero(*cfg.hebbian.B_range)),
            np.full(n_weights, _gene_for_zero(*cfg.hebbian.C_range)),
            np.full(n_weights, _gene_for_zero(*cfg.hebbian.D_range)),
        ]
        if cfg.hebbian.evolve_decay:
            lo, hi = cfg.hebbian.decay_range
            baseline_parts.append(np.full(n_weights, (lo + hi) / 2))
        if cfg.hebbian.evolve_eta:
            lo, hi = cfg.hebbian.eta_range
            baseline_parts.append(np.full(n_weights, (lo + hi) / 2))
        baseline_genome = np.concatenate(baseline_parts)

        print(
            f"\n[eval] Running baseline (zero-rules) evaluation: "
            f"envs={cfg.evaluation.num_eval_envs}  "
            f"episodes={cfg.catalog.num_episodes}  "
            f"stochastic={cfg.evaluation.stochastic}"
        )

        base_fitnesses, base_metrics = evaluate_population_cma_batched(
            solutions=[baseline_genome],
            cfg=cfg,
            model_and_layer=model_and_layer,
            wp1_cfg=wp1_cfg,
            existing_env=existing_env,
            verbose=True,
        )

        base_reward   = base_fitnesses[0]
        base_progress = base_metrics['progresses'][0]
        base_velocity = base_metrics['velocities'][0]
        base_crash    = base_metrics['crash_flags'][0]
        base_cot      = base_metrics['cots'][0]

        d_reward   = hebb_reward   - base_reward
        d_progress = hebb_progress - base_progress
        d_velocity = hebb_velocity - base_velocity
        d_crash    = hebb_crash    - base_crash
        d_cot      = hebb_cot      - base_cot

        w = 12
        print("\n" + "=" * 55)
        print(f"  {'Metric':<14}  {'Baseline':>{w}}  {'Hebbian':>{w}}  {'Delta':>{w}}")
        print("  " + "-" * 53)
        print(f"  {'reward':<14}  {base_reward:>{w}.4f}  {hebb_reward:>{w}.4f}  {d_reward:>+{w}.4f}")
        print(f"  {'progress (m)':<14}  {base_progress:>{w}.1f}  {hebb_progress:>{w}.1f}  {d_progress:>+{w}.1f}")
        print(f"  {'velocity (m/s)':<14}  {base_velocity:>{w}.2f}  {hebb_velocity:>{w}.2f}  {d_velocity:>+{w}.2f}")
        print(f"  {'crash (%)':<14}  {base_crash*100:>{w}.1f}  {hebb_crash*100:>{w}.1f}  {d_crash*100:>+{w}.1f}")
        print(f"  {'cot':<14}  {base_cot:>{w}.4f}  {hebb_cot:>{w}.4f}  {d_cot:>+{w}.4f}")
        print("=" * 55)

    gs.destroy()
