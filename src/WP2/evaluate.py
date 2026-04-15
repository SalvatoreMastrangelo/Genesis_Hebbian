"""
Evaluation pipeline for WP2 — rollout frozen actor + Hebbian on Genesis.
========================================================================

Every invalid individual is evaluated in the same forward pass using an
``IsolatedPopulationActor`` that owns ``K*S`` fully-isolated environment
slots (K individuals × S environments per individual).  Nothing mutable is
shared across slots, so each individual's fitness is a clean function of
its own genome.
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
    """Build a WingedDroneEnv for a given morphology."""
    from morph_evolution.chromosome_drone import Chromosome_Drone
    from drone_making import UrdfMaker
    from winged_drone_train.env import WingedDroneEnv
    from winged_drone_train.noise_config import configure_solver_noise

    if morphology_genome is not None:
        phys = Chromosome_Drone.to_physical(morphology_genome)
    elif cfg.morphology.fixed_genome is not None:
        phys = Chromosome_Drone.to_physical(cfg.morphology.fixed_genome)
    else:
        from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME
        phys = list(STANDARD_MYDRONE_GENOME)

    urdf_dir = Path("logs") / ".cache" / "wp2_urdfs"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    maker = UrdfMaker(phys, out_dir=str(urdf_dir))
    urdf_path = maker.create_urdf()

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
    actor: IsolatedPopulationActor,
    device: str,
    collect_smoothness: bool = False,
) -> Dict[str, np.ndarray]:
    """Run one episode and collect per-env metrics."""
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

    actor.reset_episode(device=dev)
    obs, _ = env.reset()
    x0 = env.base_pos[:, 0].clone()

    while not done.all():
        actions = actor.act(obs)
        obs, _, term, _ = env.step(actions)
        term = term.bool()
        nan_mask = env.nan_envs.to(torch.bool)

        alive = (~done) & (~term) & (~nan_mask)

        if alive.any():
            t_acc[alive] += dt
            dx_acc[alive] = env.base_pos[alive, 0] - x0[alive]
            P = env.power
            E_acc[alive] += P[alive] * dt

            if collect_smoothness and prev_actions is not None and prev_prev_actions is not None:
                jerk = torch.abs(actions - 2 * prev_actions + prev_prev_actions)
                jerk_acc[alive] += jerk[alive].mean(dim=-1)
                jerk_count[alive] += 1

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

    valid = ~env.nan_envs.to(torch.bool)
    v_mean = (dx_acc / t_acc.clamp_min(1e-6)).cpu().numpy()
    E_tot = E_acc.cpu().numpy()
    progress = dx_acc.cpu().numpy()
    crash_flags = crashed.float().cpu().numpy()

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
#  Batched population evaluation (NSGA-II path)
# ============================================================================

def _decode_rules_for_individual(
    ind,
    cfg: HebbianEvolutionConfig,
) -> Tuple[Dict[str, torch.Tensor], Optional[List[float]]]:
    """Split an individual into (hebbian_rules dict, morphology genome)."""
    hebb_part, morph_part = split_genome(list(ind), cfg)
    na, hd = cfg.hebbian.num_actions, cfg.hebbian.hidden_dim
    if hebb_part is not None and cfg.hebbian.enabled:
        rules = decode_hebbian_genes(hebb_part, cfg.hebbian, out_features=na, in_features=hd)
    else:
        rules = {
            "A": torch.zeros(na, hd, device=cfg.device),
            "B": torch.zeros(na, hd, device=cfg.device),
            "C": torch.zeros(na, hd, device=cfg.device),
            "D": torch.zeros(na, hd, device=cfg.device),
            "lam": torch.zeros(na, hd, device=cfg.device),
        }
    return rules, morph_part


def evaluate_population_batched(
    population: list,
    cfg: HebbianEvolutionConfig,
    model_and_layer=None,
    wp1_cfg=None,
    existing_env=None,
    keep_env_alive: bool = False,
) -> Optional[Tuple]:
    """Evaluate all invalid individuals simultaneously.

    Uses a single ``IsolatedPopulationActor`` spanning all K individuals ×
    S envs/individual.  Each slot owns its own LSTM state and last-layer
    weights — no cross-contamination.  Modifies individuals in-place.

    Returns
    -------
    tuple or None
        If ``keep_env_alive=True``: ``(env, urdf_path, morph_genome)``.
        Otherwise ``None``.
    """
    import genesis as gs
    from WP1.config import RunConfig

    invalid = [ind for ind in population if not ind.fitness.valid]
    K = len(invalid)
    if K == 0:
        return None

    total_envs = cfg.evaluation.num_eval_envs
    S = total_envs // K
    actual_envs = S * K

    print(
        f"[evaluate_population_batched] K={K} individuals, "
        f"S={S} envs/ind, total={actual_envs} envs"
    )

    if S == 0:
        raise ValueError(
            f"num_eval_envs ({total_envs}) is smaller than population size ({K}). "
            "Increase evaluation.num_eval_envs or reduce population_size."
        )

    if wp1_cfg is None:
        wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)

    # Decode genomes
    hebbian_rules_per_individual: List[Dict[str, torch.Tensor]] = []
    morph_parts: List[Optional[List[float]]] = []
    for ind in invalid:
        rules, morph_part = _decode_rules_for_individual(ind, cfg)
        hebbian_rules_per_individual.append(rules)
        morph_parts.append(morph_part)

    if cfg.morphology.evolve and len(set(str(m) for m in morph_parts)) > 1:
        raise RuntimeError(
            "evaluate_population_batched: heterogeneous morphologies in population — "
            "batched evaluation requires a single shared morphology."
        )
    morph_genome = morph_parts[0]

    # Build the isolated actor (deep-copies the model internally)
    actor = build_isolated_population_actor(
        checkpoint_path=cfg.checkpoint_path,
        wp1_cfg_path=cfg.checkpoint_config_path,
        hebbian_rules_per_individual=hebbian_rules_per_individual,
        cfg=cfg,
        K=K,
        S=S,
        device=cfg.device,
        stochastic=cfg.evaluation.stochastic,
    )

    # Resolve environment (reuse if dimensions match)
    env = None
    urdf_path = None
    env_was_reused = False
    if existing_env is not None:
        existing_env_obj, existing_urdf_path = existing_env
        if existing_env_obj.num_envs == actual_envs:
            env = existing_env_obj
            urdf_path = existing_urdf_path
            env_was_reused = True
            print(f"[evaluate_population_batched] Reusing environment (num_envs={actual_envs})")
        else:
            print(
                f"[evaluate_population_batched] Env size mismatch "
                f"(expected {actual_envs}, got {existing_env_obj.num_envs}); rebuilding"
            )

    if env is None:
        if not gs._initialized:
            gs.init(logging_level="error", backend=gs.gpu)
        env, urdf_path = _build_env(
            morph_genome, cfg, wp1_cfg, cfg.device, num_envs_override=actual_envs
        )

    collect_smoothness = cfg.objectives.smoothness
    all_metrics: Dict[str, List[np.ndarray]] = {
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
            ep_metrics = _rollout_episode(
                env, actor, cfg.device, collect_smoothness=collect_smoothness
            )
            for key in all_metrics:
                if key in ep_metrics:
                    flat = ep_metrics[key]  # (K*S,)
                    per_ind = flat.reshape(K, S).mean(axis=1)  # (K,)
                    all_metrics[key].append(per_ind)
    except Exception as exc:
        print(f"[evaluate_population_batched] rollout failed: {exc}")
        if not keep_env_alive:
            gs.destroy()
        raise

    if not keep_env_alive:
        gs.destroy()

    aggregated_per_ind: Dict[str, np.ndarray] = {}
    for key, ep_list in all_metrics.items():
        if ep_list:
            stacked = np.stack(ep_list, axis=0)  # (num_episodes, K)
            aggregated_per_ind[key] = np.mean(stacked, axis=0)
        else:
            aggregated_per_ind[key] = np.zeros(K)

    for i, ind in enumerate(invalid):
        ind_metrics = {k: aggregated_per_ind[k][i:i + 1] for k in aggregated_per_ind}
        try:
            fitness = compute_fitness(ind_metrics, cfg)
            ind.fitness.values = tuple(fitness)
            ind.metrics = {k: aggregated_per_ind[k][i] for k in aggregated_per_ind}
        except Exception:
            ind.fitness.values = tuple(default_fitness(cfg))

    if keep_env_alive:
        return (env, urdf_path, morph_genome)
    return None


# ============================================================================
#  CMA-ES evaluation (catalog of URDFs)
# ============================================================================

@torch.no_grad()
def _rollout_episode_reward_sum(
    env,
    actor: IsolatedPopulationActor,
    device: str,
) -> Dict[str, np.ndarray]:
    """Run one episode and return WP1 reward sum per environment."""
    B = env.num_envs
    dt = env.dt
    dev = torch.device(device)

    done = torch.zeros(B, dtype=torch.bool, device=dev)
    reward_sum = torch.zeros(B, device=dev)
    t_acc = torch.zeros(B, device=dev)
    dx_acc = torch.zeros(B, device=dev)
    crashed = torch.zeros(B, dtype=torch.bool, device=dev)

    actor.reset_episode(device=dev)
    obs, _ = env.reset()
    x0 = env.base_pos[:, 0].clone()

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

    One ``IsolatedPopulationActor`` is built per generation (new rules each
    generation); it is reused across every URDF in the catalog.  Each env
    slot has its own LSTM + last-layer weights.
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

    print(
        f"[evaluate_population_cma_batched] P={P} individuals, "
        f"S={S} envs/ind, total={actual_envs} envs"
    )

    na, hd = cfg.hebbian.num_actions, cfg.hebbian.hidden_dim

    # Decode all genomes into Hebbian rules (shared across all URDFs)
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

    env_was_reused = False
    env = None
    if (existing_env is not None and n_urdfs == 1 and catalog[0][0] is None):
        existing_env_obj, _ = existing_env
        if existing_env_obj.num_envs == actual_envs:
            env = existing_env_obj
            env_was_reused = True
            print(
                f"[evaluate_population_cma_batched] Reusing environment "
                f"(rules-only, actual_envs={actual_envs})"
            )

    for urdf_idx, (urdf_file, naca) in enumerate(catalog):
        print(
            f"  [catalog {urdf_idx + 1}/{n_urdfs}] "
            f"{'default URDF' if urdf_file is None else Path(urdf_file).name}"
        )

        if not env_was_reused:
            try:
                if not gs._initialized:
                    gs.init(logging_level="error", backend=gs.gpu)

                if urdf_file is None:
                    env, _ = _build_env(
                        None, cfg, wp1_cfg, cfg.device, num_envs_override=actual_envs
                    )
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

        try:
            for ep in range(n_episodes):
                ep_metrics = _rollout_episode_reward_sum(env, actor, cfg.device)

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
            if not env_was_reused:
                gs.destroy()

        urdf_reward /= n_episodes
        urdf_progress /= n_episodes
        urdf_velocity /= n_episodes
        urdf_crash /= n_episodes

        acc_reward += urdf_reward
        acc_progress += urdf_progress
        acc_velocity += urdf_velocity
        acc_crash += urdf_crash

        print(
            f"    best={urdf_reward.max():.4f}  mean={urdf_reward.mean():.4f}  "
            f"crash={urdf_crash.mean() * 100:.1f}%"
        )

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
