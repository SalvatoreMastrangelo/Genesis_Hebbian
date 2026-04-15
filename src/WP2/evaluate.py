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
):
    """Build a WingedDroneEnv using the WP1 default morphology."""
    from morph_evolution.chromosome_drone import Chromosome_Drone
    from drone_making import UrdfMaker
    from winged_drone_train.env import WingedDroneEnv
    from winged_drone_train.noise_config import configure_solver_noise
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
#  CMA-ES evaluation
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
    cfg: HebbianEvolutionConfig,
    model_and_layer: Tuple,
    wp1_cfg,
    catalog: Optional[List[Tuple[str, str]]] = None,
    existing_env=None,
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
