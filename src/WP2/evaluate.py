"""
Evaluation pipeline for WP2 — rollout frozen actor + Hebbian on Genesis.
========================================================================

For each individual:
1. Decode genome -> Hebbian rules + morphology
2. Build environment with morphology URDF
3. Load frozen actor, attach HebbianLastLayer
4. Rollout N episodes (reset weights per episode)
5. Collect metrics -> compute multi-objective fitness
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from WP2.config import HebbianEvolutionConfig
from WP2.hebbian import HebbianLastLayer
from WP2.frozen_actor import (
    HebbianActorWrapper,
    attach_hebbian,
    load_frozen_actor,
)
from WP2.objectives import compute_fitness, default_fitness
from WP2.utils import decode_hebbian_genes, split_genome


# ============================================================================
#  Environment construction
# ============================================================================

def _build_env(
    morphology_genome: Optional[List[float]],
    cfg: HebbianEvolutionConfig,
    wp1_cfg,
    device: str,
    num_envs_override: Optional[int] = None,
):
    """Build a WingedDroneEnv for a given morphology.

    Parameters
    ----------
    morphology_genome : list or None
        Normalised [0,1]^15 morphology genome.  If None, uses fixed morphology.
    cfg : HebbianEvolutionConfig
        WP2 config.
    wp1_cfg : RunConfig
        WP1 config (for env/obs/reward/command params).
    device : str
        Target device.
    num_envs_override : int, optional
        If provided, overrides cfg.evaluation.num_eval_envs.

    Returns
    -------
    env : WingedDroneEnv
        Genesis environment instance.
    urdf_path : str
        Path to the generated URDF file.
    """
    from morph_evolution.chromosome_drone import Chromosome_Drone
    from drone_making import UrdfMaker
    from winged_drone_train.env import WingedDroneEnv
    from winged_drone_train.noise_config import configure_solver_noise

    # Determine morphology
    if morphology_genome is not None:
        phys = Chromosome_Drone.to_physical(morphology_genome)
    elif cfg.morphology.fixed_genome is not None:
        phys = Chromosome_Drone.to_physical(cfg.morphology.fixed_genome)
    else:
        from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME
        phys = list(STANDARD_MYDRONE_GENOME)

    # Generate URDF
    urdf_dir = Path("logs") / ".cache" / "wp2_urdfs"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    maker = UrdfMaker(phys, out_dir=str(urdf_dir))
    urdf_path = maker.create_urdf()

    # Get NACA code for env_cfg
    naca = Chromosome_Drone.naca_from_physical(phys) or "3416"

    # Build legacy configs from WP1 RunConfig
    env_cfg = wp1_cfg.to_env_cfg()
    obs_cfg = wp1_cfg.to_obs_cfg()
    reward_cfg = wp1_cfg.to_reward_cfg()
    command_cfg = wp1_cfg.to_command_cfg()

    # Override for evaluation
    env_cfg.update(dict(
        visualize_camera=False,
        visualize_target=False,
        naca=naca,
    ))
    command_cfg["min_speed"] = cfg.evaluation.vmin
    command_cfg["max_speed"] = cfg.evaluation.vmax

    # Disable genome obs for the actor (morphology-blind policy)
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
#  Single-episode rollout
# ============================================================================

@torch.no_grad()
def _rollout_episode(
    env,
    actor_wrapper: HebbianActorWrapper,
    device: str,
    collect_smoothness: bool = False,
) -> Dict[str, np.ndarray]:
    """Run a single episode and collect per-env metrics.

    Returns
    -------
    dict with keys:
        velocities : (num_envs,) mean forward speed per env
        energies : (num_envs,) total energy per env
        progresses : (num_envs,) total forward distance per env
        crash_flags : (num_envs,) 1.0 if crashed, 0.0 otherwise
        action_jerks : (num_envs,) mean action jerk if collected
    """
    B = env.num_envs
    dt = env.dt
    dev = torch.device(device)

    done = torch.zeros(B, dtype=torch.bool, device=dev)
    t_acc = torch.zeros(B, device=dev)
    dx_acc = torch.zeros(B, device=dev)
    E_acc = torch.zeros(B, device=dev)
    crashed = torch.zeros(B, dtype=torch.bool, device=dev)

    prev_actions = None
    prev_prev_actions = None
    jerk_acc = torch.zeros(B, device=dev)
    jerk_count = torch.zeros(B, device=dev)

    # Reset episode
    actor_wrapper.reset_episode(num_envs=B, device=dev)
    obs, _ = env.reset()
    x0 = env.base_pos[:, 0].clone()

    while not done.all():
        actions = actor_wrapper.act(obs)
        obs, _, term, _ = env.step(actions)
        term = term.bool()
        nan_mask = env.nan_envs.to(torch.bool)

        alive = (~done) & (~term) & (~nan_mask)

        if alive.any():
            t_acc[alive] += dt
            dx_acc[alive] = env.base_pos[alive, 0] - x0[alive]
            P = env.power
            E_acc[alive] += P[alive] * dt

            # Action jerk (second derivative)
            if collect_smoothness and prev_actions is not None and prev_prev_actions is not None:
                jerk = torch.abs(actions - 2 * prev_actions + prev_prev_actions)
                jerk_acc[alive] += jerk[alive].mean(dim=-1)
                jerk_count[alive] += 1

        # Track crashes
        just_done = (~done) & term
        if just_done.any():
            for attr in ("pre_collision", "pre_wall_crash", "pre_angle_limit"):
                flag = getattr(env, attr, None)
                if flag is not None:
                    crashed |= just_done & flag.to(torch.bool)

        done |= term | nan_mask

        if collect_smoothness:
            prev_prev_actions = prev_actions
            prev_actions = actions.clone() if actions is not None else None

    # Compute per-env metrics
    valid = ~env.nan_envs.to(torch.bool)
    v_mean = (dx_acc / t_acc.clamp_min(1e-6)).cpu().numpy()
    E_tot = E_acc.cpu().numpy()
    progress = dx_acc.cpu().numpy()
    crash_flags = crashed.float().cpu().numpy()

    # Invalidate NaN envs
    nan_np = (~valid).cpu().numpy()
    v_mean[nan_np] = 0.0
    E_tot[nan_np] = 0.0
    progress[nan_np] = 0.0

    result = {
        "velocities": v_mean,
        "energies": E_tot,
        "progresses": progress,
        "crash_flags": crash_flags,
    }
    if collect_smoothness:
        jerk_mean = (jerk_acc / jerk_count.clamp_min(1)).cpu().numpy()
        jerk_mean[nan_np] = 0.0
        result["action_jerks"] = jerk_mean

    return result


# ============================================================================
#  Multi-episode evaluation
# ============================================================================

def evaluate_individual(
    genome: Sequence[float],
    cfg: HebbianEvolutionConfig,
    model_and_layer=None,
    wp1_cfg=None,
) -> Tuple[List[float], Dict[str, np.ndarray]]:
    """Evaluate a single individual over multiple episodes.

    Parameters
    ----------
    genome : sequence of float
        Full genome (Hebbian + morphology in [0,1]).
    cfg : HebbianEvolutionConfig
        WP2 config.
    model_and_layer : tuple or None
        Pre-loaded (model, last_layer) to avoid repeated loading.
    wp1_cfg : RunConfig or None
        Pre-loaded WP1 config.

    Returns
    -------
    fitness : list of float
        Fitness values matching active objectives.
    aggregated_metrics : dict
        Aggregated metrics across all episodes.
    """
    import genesis as gs
    device = cfg.device

    # Split genome
    hebb_part, morph_part = split_genome(genome, cfg)

    # Load WP1 config if needed
    if wp1_cfg is None:
        from WP1.config import RunConfig
        wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)

    # Load model if needed
    if model_and_layer is None:
        model, last_layer, num_actions, hidden_dim = load_frozen_actor(
            cfg.checkpoint_path, cfg.checkpoint_config_path, device=device
        )
    else:
        model, last_layer, num_actions, hidden_dim = model_and_layer

    # Decode Hebbian rules and attach
    na, hd = cfg.hebbian.num_actions, cfg.hebbian.hidden_dim
    if hebb_part is not None and cfg.hebbian.enabled:
        rules = decode_hebbian_genes(hebb_part, cfg.hebbian, out_features=na, in_features=hd)
        hebbian = attach_hebbian(last_layer, rules, cfg, device=device)
    else:
        # No Hebbian — create a dummy that does nothing
        dummy_rules = {
            "A": torch.zeros(na, hd),
            "B": torch.zeros(na, hd),
            "C": torch.zeros(na, hd),
            "D": torch.zeros(na, hd),
            "lam": torch.zeros(na, hd),
        }
        hebbian = attach_hebbian(last_layer, dummy_rules, cfg, device=device)
        hebbian.eta = 0.0  # disable updates

    actor_wrapper = HebbianActorWrapper(
        model=model,
        hebbian=hebbian,
        stochastic=cfg.evaluation.stochastic,
    )

    # Build environment
    try:
        if not gs._initialized:
            gs.init(logging_level="error", backend=gs.gpu)
        env, urdf_path = _build_env(morph_part, cfg, wp1_cfg, device)
    except Exception as exc:
        print(f"[evaluate] env build failed: {exc}")
        return default_fitness(cfg), {}

    # Run multiple episodes
    collect_smoothness = cfg.objectives.smoothness
    all_metrics = {
        "velocities": [],
        "energies": [],
        "progresses": [],
        "crash_flags": [],
    }
    if collect_smoothness:
        all_metrics["action_jerks"] = []

    try:
        for ep in range(cfg.evaluation.num_eval_episodes):
            # Reset Hebbian weights to checkpoint at episode start
            hebbian.reset_weights()

            ep_metrics = _rollout_episode(
                env, actor_wrapper, device,
                collect_smoothness=collect_smoothness,
            )

            for key in all_metrics:
                if key in ep_metrics:
                    all_metrics[key].append(ep_metrics[key])
    except Exception as exc:
        print(f"[evaluate] rollout failed: {exc}")
        gs.destroy()
        return default_fitness(cfg), {}

    gs.destroy()

    # Aggregate across episodes (mean over episodes, then mean over envs)
    aggregated = {}
    for key, episode_list in all_metrics.items():
        if episode_list:
            # Stack: (num_episodes, num_envs)
            stacked = np.stack(episode_list, axis=0)
            # Mean across episodes per env, then take env mean
            aggregated[key] = np.mean(stacked, axis=0)
        else:
            aggregated[key] = np.array([])

    fitness = compute_fitness(aggregated, cfg)
    return fitness, aggregated


# ============================================================================
#  Batch evaluation (serial)
# ============================================================================

def evaluate_population_serial(
    population: list,
    cfg: HebbianEvolutionConfig,
    model_and_layer=None,
    wp1_cfg=None,
) -> None:
    """Evaluate all individuals sequentially.

    Modifies individuals in-place: sets ``ind.fitness.values``.
    """
    for i, ind in enumerate(population):
        if ind.fitness.valid:
            continue
        fitness, metrics = evaluate_individual(list(ind), cfg, model_and_layer, wp1_cfg)
        ind.fitness.values = tuple(fitness)
        # Store metrics on individual for later analysis
        ind.metrics = {k: v[0] if len(v) > 0 else 0.0 for k, v in metrics.items()}


# ============================================================================
#  Batched population evaluation
# ============================================================================

def evaluate_population_batched(
    population: list,
    cfg: HebbianEvolutionConfig,
    model_and_layer=None,
    wp1_cfg=None,
) -> None:
    """Evaluate all invalid individuals simultaneously via vectorized rollout.

    All individuals evaluate in parallel using shared Genesis environments.
    Modifies individuals in-place: sets ``ind.fitness.values``.

    Parameters
    ----------
    population : list
        DEAP individuals (only invalid ones are evaluated).
    cfg : HebbianEvolutionConfig
        WP2 configuration.
    model_and_layer : tuple, optional
        Pre-loaded (model, last_layer, num_actions, hidden_dim).
    wp1_cfg : RunConfig, optional
        Pre-loaded WP1 config.
    """
    import genesis as gs
    from WP2.hebbian import BatchedHebbianLastLayer
    from WP2.frozen_actor import BatchedHebbianActorWrapper
    from WP1.config import RunConfig

    # Find individuals needing evaluation
    invalid = [ind for ind in population if not ind.fitness.valid]
    P = len(invalid)
    if P == 0:
        return

    print(f"[evaluate_population_batched] Evaluating {P} individuals simultaneously")

    total_envs = cfg.evaluation.num_eval_envs
    S = total_envs // P                          # envs per individual
    actual_envs = S * P                          # may differ from total_envs if not divisible

    print(f"  Total envs: {total_envs}, Population: {P}, Envs/ind: {S}, Actual total: {actual_envs}")

    # Load WP1 config if needed
    if wp1_cfg is None:
        wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)

    # Load model if needed
    if model_and_layer is None:
        model, last_layer, num_actions, hidden_dim = load_frozen_actor(
            cfg.checkpoint_path, cfg.checkpoint_config_path, device=cfg.device
        )
    else:
        model, last_layer, num_actions, hidden_dim = model_and_layer

    # Decode genomes and build batched Hebbian rules
    hebbian_rules_list = []
    morph_parts = []
    for ind in invalid:
        hebb_part, morph_part = split_genome(list(ind), cfg)
        na, hd = cfg.hebbian.num_actions, cfg.hebbian.hidden_dim
        if hebb_part is not None and cfg.hebbian.enabled:
            rules = decode_hebbian_genes(hebb_part, cfg.hebbian, out_features=na, in_features=hd)
        else:
            # Dummy rules for disabled Hebbian
            rules = {
                "A": torch.zeros(na, hd, device=cfg.device),
                "B": torch.zeros(na, hd, device=cfg.device),
                "C": torch.zeros(na, hd, device=cfg.device),
                "D": torch.zeros(na, hd, device=cfg.device),
                "lam": torch.zeros(na, hd, device=cfg.device),
            }
        hebbian_rules_list.append(rules)
        morph_parts.append(morph_part)

    # Check if morphologies differ (if morphology co-evolution is enabled)
    if cfg.morphology.evolve and len(set(str(m) for m in morph_parts)) > 1:
        print("[evaluate_population_batched] Different morphologies detected; falling back to serial evaluation")
        evaluate_population_serial(population, cfg, model_and_layer, wp1_cfg)
        return

    morph_genome = morph_parts[0]

    # Build batched Hebbian controller
    na = cfg.hebbian.num_actions
    hd = cfg.hebbian.hidden_dim
    batched_hebbian = BatchedHebbianLastLayer(
        last_layer,
        hebbian_rules_list,
        eta=cfg.hebbian.eta,
        w_max=cfg.hebbian.w_max,
        device=cfg.device,
        pop_size=P,
        slice_size=S,
    )

    # Create batched actor wrapper
    wrapper = BatchedHebbianActorWrapper(model, batched_hebbian, stochastic=cfg.evaluation.stochastic)

    # Build environment with actual_envs
    try:
        if not gs._initialized:
            gs.init(logging_level="error", backend=gs.gpu)
        env, urdf_path = _build_env(morph_genome, cfg, wp1_cfg, cfg.device, num_envs_override=actual_envs)
    except Exception as exc:
        print(f"[evaluate_population_batched] env build failed: {exc}")
        # Fall back to serial
        evaluate_population_serial(population, cfg, model_and_layer, wp1_cfg)
        return

    # Collect metrics over multiple episodes
    collect_smoothness = cfg.objectives.smoothness
    all_metrics = {
        "velocities": [],
        "energies": [],
        "progresses": [],
        "crash_flags": [],
    }
    if collect_smoothness:
        all_metrics["action_jerks"] = []

    try:
        for ep in range(cfg.evaluation.num_eval_episodes):
            print(f"  Episode {ep + 1}/{cfg.evaluation.num_eval_episodes}")
            wrapper.reset_episode(num_envs=actual_envs, device=cfg.device)

            ep_metrics = _rollout_episode(
                env, wrapper, cfg.device,
                collect_smoothness=collect_smoothness,
            )

            # ep_metrics values are (actual_envs,) flat arrays
            # Reshape to (P, S) and take mean over S to get per-individual metrics
            for key in all_metrics:
                if key in ep_metrics:
                    flat = ep_metrics[key]  # (P*S,)
                    reshaped = flat.reshape(P, S)
                    per_ind = reshaped.mean(axis=1)  # (P,)
                    all_metrics[key].append(per_ind)

    except Exception as exc:
        print(f"[evaluate_population_batched] rollout failed: {exc}")
        gs.destroy()
        # Fall back to serial
        evaluate_population_serial(population, cfg, model_and_layer, wp1_cfg)
        return

    gs.destroy()

    # Aggregate metrics across episodes: (num_episodes, P) -> (P,)
    aggregated_per_ind = {}
    for key, ep_list in all_metrics.items():
        if ep_list:
            stacked = np.stack(ep_list, axis=0)  # (num_episodes, P)
            aggregated_per_ind[key] = np.mean(stacked, axis=0)  # (P,)
        else:
            aggregated_per_ind[key] = np.zeros(P)

    # Assign fitness to each individual
    for i, ind in enumerate(invalid):
        ind_metrics = {
            k: aggregated_per_ind[k][i:i+1] for k in aggregated_per_ind
        }
        try:
            fitness = compute_fitness(ind_metrics, cfg)
            ind.fitness.values = tuple(fitness)
            # Store metrics on individual for later analysis
            ind.metrics = {k: aggregated_per_ind[k][i] for k in aggregated_per_ind}
        except Exception as exc:
            ind.fitness.values = tuple(default_fitness(cfg))
