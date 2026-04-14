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

    # Determine number of environments for per-env weight matrices
    num_envs = cfg.evaluation.num_eval_envs

    # Decode Hebbian rules and attach
    na, hd = cfg.hebbian.num_actions, cfg.hebbian.hidden_dim
    if hebb_part is not None and cfg.hebbian.enabled:
        rules = decode_hebbian_genes(hebb_part, cfg.hebbian, out_features=na, in_features=hd)
        hebbian = attach_hebbian(last_layer, rules, cfg, device=device, num_envs=num_envs)
    else:
        # No Hebbian — create a dummy that does nothing
        dummy_rules = {
            "A": torch.zeros(na, hd),
            "B": torch.zeros(na, hd),
            "C": torch.zeros(na, hd),
            "D": torch.zeros(na, hd),
            "lam": torch.zeros(na, hd),
        }
        hebbian = attach_hebbian(last_layer, dummy_rules, cfg, device=device, num_envs=num_envs)
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
    existing_env=None,
    keep_env_alive=False,
) -> Optional[Tuple]:
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
    existing_env : tuple, optional
        Pre-built (env, urdf_path) to reuse across evaluations.
        If provided and morphologies match, reuses the environment.
    keep_env_alive : bool, optional
        If True, returns (env, urdf_path, morph_genome) instead of destroying.
        If False (default), destroys the environment and returns None.

    Returns
    -------
    tuple or None
        If keep_env_alive=True: (env, urdf_path, morph_genome)
        If keep_env_alive=False: None
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

    # If no environments available per individual, fall back to serial evaluation
    if actual_envs == 0:
        print(f"[evaluate_population_batched] Not enough environments ({total_envs}) for population ({P}); falling back to serial evaluation")
        evaluate_population_serial(population, cfg, model_and_layer, wp1_cfg)
        return

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
        use_oja_coefficient=cfg.hebbian.use_oja_coefficient,
        device=cfg.device,
        pop_size=P,
        slice_size=S,
    )

    # Create batched actor wrapper
    wrapper = BatchedHebbianActorWrapper(model, batched_hebbian, stochastic=cfg.evaluation.stochastic)

    # Determine if we can reuse existing environment
    env = None
    urdf_path = None
    env_was_reused = False

    if existing_env is not None:
        existing_env_obj, existing_urdf_path = existing_env
        # Check if environment dimensions match
        if existing_env_obj.num_envs == actual_envs:
            env = existing_env_obj
            urdf_path = existing_urdf_path
            env_was_reused = True
            print(f"[evaluate_population_batched] Reusing environment (num_envs={actual_envs})")
        else:
            print(f"[evaluate_population_batched] Environment size mismatch (expected {actual_envs}, got {existing_env_obj.num_envs}); rebuilding")

    # Build new environment if needed
    if env is None:
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
        if not keep_env_alive:
            gs.destroy()
        # Fall back to serial
        evaluate_population_serial(population, cfg, model_and_layer, wp1_cfg)
        return

    # Only destroy if not keeping alive
    if not keep_env_alive:
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

    # Return environment if requested
    if keep_env_alive:
        return (env, urdf_path, morph_genome)


# ============================================================================
#  CMA-ES evaluation helpers
# ============================================================================

@torch.no_grad()
def _rollout_episode_reward_sum(
    env,
    actor_wrapper,
    device: str,
) -> Dict[str, np.ndarray]:
    """Run one episode and return the WP1 reward sum per environment.

    Collects ``env.last_reward_total`` at each live step and accumulates it.
    Also gathers the auxiliary metrics (progress, velocity, crash) used for
    diagnostic logging in the CMA-ES summary CSV.

    Returns
    -------
    dict with keys:
        reward_sum  : (num_envs,) total WP1 reward accumulated over the episode
        progresses  : (num_envs,) total forward distance [m]
        velocities  : (num_envs,) mean forward speed [m/s]
        crash_flags : (num_envs,) 1.0 if crashed, 0.0 otherwise
    """
    B = env.num_envs
    dt = env.dt
    dev = torch.device(device)

    done = torch.zeros(B, dtype=torch.bool, device=dev)
    reward_sum = torch.zeros(B, device=dev)
    t_acc = torch.zeros(B, device=dev)
    dx_acc = torch.zeros(B, device=dev)
    crashed = torch.zeros(B, dtype=torch.bool, device=dev)

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
            reward_sum[alive] += env.last_reward_total[alive]
            t_acc[alive] += dt
            dx_acc[alive] = env.base_pos[alive, 0] - x0[alive]

        just_done = (~done) & term
        if just_done.any():
            for attr in ("pre_collision", "pre_wall_crash", "pre_angle_limit"):
                flag = getattr(env, attr, None)
                if flag is not None:
                    crashed |= just_done & flag.to(torch.bool)

        done |= term | nan_mask

    valid = ~env.nan_envs.to(torch.bool)
    nan_np = (~valid).cpu().numpy()

    reward_arr = reward_sum.cpu().numpy()
    t_arr = t_acc.cpu().numpy()
    dx_arr = dx_acc.cpu().numpy()
    crash_arr = crashed.float().cpu().numpy()

    reward_arr[nan_np] = 0.0
    dx_arr[nan_np] = 0.0
    t_arr[nan_np] = 0.0

    v_arr = np.where(t_arr > 1e-6, dx_arr / t_arr, 0.0)

    return {
        "reward_sum": reward_arr,
        "progresses": dx_arr,
        "velocities": v_arr,
        "crash_flags": crash_arr,
    }


def _load_catalog(catalog_path: str) -> List[Tuple[str, str]]:
    """Parse a catalog.txt file and return a list of (urdf_path, naca) tuples.

    Each line in catalog.txt is a URDF filename of the form::

        [p1, p2, ..., p15].urdf

    where the bracketed values are the *physical* drone genome parameters.
    The URDF files are expected to reside in the same directory as catalog.txt.

    Parameters
    ----------
    catalog_path : str
        Absolute or relative path to catalog.txt.

    Returns
    -------
    list of (urdf_path, naca_code)
        ``urdf_path`` is the full path to the URDF file.
        ``naca_code`` is a 4-digit string used for the aerodynamic solver.
    """
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

            # Extract physical genome from filename: "[p1, p2, ...].urdf"
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
    """Build a WingedDroneEnv from an already-existing URDF file.

    Mirrors ``_build_env`` but skips URDF generation and accepts a pre-built
    URDF path directly.  Used by the CMA-ES evaluator to avoid regenerating
    URDFs that already exist in the catalog.

    Parameters
    ----------
    urdf_file : str
        Path to the URDF file.
    naca : str
        4-digit NACA code for the aerodynamic solver configuration.
    cfg : HebbianEvolutionConfig
    wp1_cfg : RunConfig
    device : str
    num_envs : int

    Returns
    -------
    WingedDroneEnv
    """
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
    cfg,
    model_and_layer: Tuple,
    wp1_cfg,
    catalog: Optional[List[Tuple[str, str]]] = None,
    existing_env=None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Evaluate a CMA-ES population across all catalog URDFs.

    Each solution (Hebbian genome in [0,1]^n) is evaluated simultaneously
    with all other solutions using a shared Genesis environment.  If a
    catalog of URDFs is provided, the process repeats for each URDF and
    fitnesses are averaged.

    The ``BatchedHebbianActorWrapper`` is built once per call (shared across
    all URDFs in the catalog), and ``reset_episode()`` is called between
    episodes/URDFs to restore weights to the frozen checkpoint.

    Parameters
    ----------
    solutions : list of np.ndarray
        P genomes in [0,1]^n, as returned by ``cma.CMAEvolutionStrategy.ask()``.
    cfg : HebbianEvolutionConfig
        WP2 configuration.
    model_and_layer : tuple
        Pre-loaded ``(model, last_layer, num_actions, hidden_dim)`` from
        ``load_frozen_actor``.
    wp1_cfg : RunConfig
        Pre-loaded WP1 config.
    catalog : list of (urdf_path, naca) or None
        URDFs to evaluate against.  If None or empty, falls back to the
        default/fixed morphology URDF (``cfg.morphology.fixed_genome``).
    existing_env : tuple, optional
        Pre-built (env, urdf_path) to reuse across evaluations.
        For rules-only evolution, pass this to avoid rebuilding the Genesis
        scene every generation.

    Returns
    -------
    fitnesses : np.ndarray, shape (P,)
        Mean WP1 reward sum per individual, averaged across URDFs and episodes.
    metrics : dict
        Diagnostic metrics per individual (velocities, progresses, crash_flags).
    """
    import genesis as gs
    from WP2.hebbian import BatchedHebbianLastLayer
    from WP2.frozen_actor import BatchedHebbianActorWrapper

    P = len(solutions)
    total_envs = cfg.evaluation.num_eval_envs
    S = total_envs // P
    actual_envs = S * P

    if S == 0:
        raise ValueError(
            f"num_eval_envs ({total_envs}) is smaller than population size ({P}). "
            "Increase evaluation.num_eval_envs or reduce population_size."
        )

    print(f"[evaluate_population_cma_batched] P={P} individuals, "
          f"S={S} envs/ind, total={actual_envs} envs")

    model, last_layer, num_actions, hidden_dim = model_and_layer
    na, hd = cfg.hebbian.num_actions, cfg.hebbian.hidden_dim

    # Decode all genomes into Hebbian rules (once; shared across all URDFs)
    hebbian_rules_list = []
    for genome in solutions:
        hebb_part = list(np.clip(genome, 0.0, 1.0))
        rules = decode_hebbian_genes(hebb_part, cfg.hebbian, out_features=na, in_features=hd)
        hebbian_rules_list.append(rules)

    # Build the batched Hebbian controller (reused across URDFs)
    batched_hebbian = BatchedHebbianLastLayer(
        last_layer,
        hebbian_rules_list,
        eta=cfg.hebbian.eta,
        w_max=cfg.hebbian.w_max,
        use_oja_coefficient=cfg.hebbian.use_oja_coefficient,
        device=cfg.device,
        pop_size=P,
        slice_size=S,
    )
    wrapper = BatchedHebbianActorWrapper(
        model, batched_hebbian, stochastic=cfg.evaluation.stochastic
    )

    # Resolve catalog: empty → single default-morphology entry (None signals _build_env)
    if not catalog:
        catalog = [(None, None)]
    n_urdfs = len(catalog)
    n_episodes = cfg.catalog.num_episodes

    # Accumulators across URDFs (sum, divided at the end)
    acc_reward = np.zeros(P)
    acc_progress = np.zeros(P)
    acc_velocity = np.zeros(P)
    acc_crash = np.zeros(P)

    # Check if we can reuse existing environment (rules-only case: single default URDF)
    env_was_reused = False
    if (existing_env is not None and n_urdfs == 1 and
        catalog[0][0] is None):
        existing_env_obj, _ = existing_env
        if existing_env_obj.num_envs == actual_envs:
            env = existing_env_obj
            env_was_reused = True
            print(f"[evaluate_population_cma_batched] Reusing environment (rules-only, actual_envs={actual_envs})")

    for urdf_idx, (urdf_file, naca) in enumerate(catalog):
        print(f"  [catalog {urdf_idx + 1}/{n_urdfs}] "
              f"{'default URDF' if urdf_file is None else Path(urdf_file).name}")

        # Build environment for this URDF (unless already reused)
        if not env_was_reused:
            try:
                if not gs._initialized:
                    gs.init(logging_level="error", backend=gs.gpu)

                if urdf_file is None:
                    env, _ = _build_env(
                        None, cfg, wp1_cfg, cfg.device,
                        num_envs_override=actual_envs,
                    )
                else:
                    env = _build_env_from_urdf(
                        urdf_file, naca, cfg, wp1_cfg, cfg.device,
                        num_envs=actual_envs,
                    )
            except Exception as exc:
                print(f"  [catalog {urdf_idx + 1}] env build failed: {exc} — skipping")
                # Don't accumulate; adjust denominator at the end
                if gs._initialized:
                    try:
                        gs.destroy()
                    except Exception:
                        pass
                continue

        # Accumulators for this URDF (across episodes)
        urdf_reward = np.zeros(P)
        urdf_progress = np.zeros(P)
        urdf_velocity = np.zeros(P)
        urdf_crash = np.zeros(P)

        try:
            for ep in range(n_episodes):
                ep_metrics = _rollout_episode_reward_sum(env, wrapper, cfg.device)

                # Reshape flat (actual_envs,) → (P, S) → mean over S → (P,)
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

        except Exception as exc:
            print(f"  [catalog {urdf_idx + 1}] rollout failed: {exc}")
        finally:
            # Only destroy if environment was just built (not reused)
            if not env_was_reused:
                gs.destroy()

        # Average over episodes
        urdf_reward /= n_episodes
        urdf_progress /= n_episodes
        urdf_velocity /= n_episodes
        urdf_crash /= n_episodes

        acc_reward += urdf_reward
        acc_progress += urdf_progress
        acc_velocity += urdf_velocity
        acc_crash += urdf_crash

        print(f"    best={urdf_reward.max():.4f}  mean={urdf_reward.mean():.4f}  "
              f"crash={urdf_crash.mean() * 100:.1f}%")

    # Average across URDFs
    acc_reward /= n_urdfs
    acc_progress /= n_urdfs
    acc_velocity /= n_urdfs
    acc_crash /= n_urdfs

    metrics = {
        "reward_sums": acc_reward,
        "progresses": acc_progress,
        "velocities": acc_velocity,
        "crash_flags": acc_crash,
    }
    return acc_reward, metrics
