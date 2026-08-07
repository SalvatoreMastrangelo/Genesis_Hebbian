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
    last_actor_linear_key,
    load_frozen_actor,
)
from WP2.utils import decode_hebbian_genes


# ============================================================================
#  Environment construction
# ============================================================================

def _apply_noise_toggles(env_cfg: dict, cfg: HebbianEvolutionConfig) -> None:
    """Force-disable per-slot noise sources marked False in ``cfg.evaluation.noise``.

    Mutates ``env_cfg`` in place.  ``True`` toggles leave the WP1 checkpoint
    setting untouched; ``False`` toggles zero out the corresponding magnitude
    or flip the enabling flag.  Only affects sources that are NOT shared
    across individuals (forest layout / commanded speed are deterministic and
    have no toggle).

    When every toggle is ``True`` this is a strict no-op: ``env_cfg`` is not
    touched at all, so the WP1 checkpoint's behavior is preserved byte-for-byte.
    """
    n = cfg.evaluation.noise

    if not n.action_latency:
        env_cfg["simulate_action_latency"] = False

    if not n.aero_noise:
        env_cfg["aero_noise"] = False

    prop_overrides_needed = (
        not n.mass_shift
        or not n.com_shift
        or not n.joint_target_episode_bias
        or not n.joint_target_step_noise
    )
    if prop_overrides_needed:
        prop = dict(env_cfg.get("property_randomization", {}) or {})
        if not n.mass_shift:
            prop["mass_shift_std"] = 0.0
        if not n.com_shift:
            prop["com_shift_std"] = 0.0
        if not n.joint_target_episode_bias:
            prop["joint_target_episode_bias_std"] = 0.0
        if not n.joint_target_step_noise:
            prop["joint_target_step_noise_std"] = 0.0
        env_cfg["property_randomization"] = prop


def _apply_forest_overrides(env_cfg: dict, cfg: HebbianEvolutionConfig) -> None:
    """Apply forest-generation overrides onto ``env_cfg`` in place.

    Precedence (low → high):
      1. WP1 checkpoint config — already present in ``env_cfg``.
      2. ``cfg.evaluation`` legacy fields (``x_upper`` / ``dens_min`` /
         ``dens_max``), kept for old configs.
      3. ``cfg.forest`` — the full forest section (algorithm + every
         parameter).

    A field left at its default (``None``) never overrides the layer below it,
    so an all-null ``forest`` section + null ``evaluation`` overrides reproduce
    the WP1 forest exactly.
    """
    ev = cfg.evaluation

    # --- (2) legacy evaluation.* overrides — preserve historical behaviour ---
    if ev.x_upper is not None:
        env_cfg["x_upper"] = float(ev.x_upper)
        env_cfg["forest_x_limit"] = float(ev.x_upper)
    if ev.dens_min is not None:
        env_cfg["dens_min"] = float(ev.dens_min)
    if ev.dens_max is not None:
        env_cfg["dens_max"] = float(ev.dens_max)

    # --- (3) forest section — wins over both layers above when set ---
    fr = getattr(cfg, "forest", None)
    if fr is None:
        return

    mode = fr.resolved_mode()   # validates; None ⇒ inherit WP1
    if mode is not None:
        env_cfg["forest_mode"] = mode

    float_keys = (
        "x_lower", "x_upper", "y_lower", "y_upper",
        "tree_radius", "tree_height",
        "dens_min", "dens_max", "dens_min_min", "dens_min_max",
        "x_spacing_start", "x_spacing_end", "forest_length",
        "y_spacing_max", "y_spacing_min",
    )
    for k in float_keys:
        v = getattr(fr, k)
        if v is not None:
            env_cfg[k] = float(v)

    for k in ("num_trees", "num_trees_eval"):
        v = getattr(fr, k)
        if v is not None:
            env_cfg[k] = int(v)

    # x_upper also drives forest_x_limit used elsewhere in the env
    if fr.x_upper is not None:
        env_cfg["forest_x_limit"] = float(fr.x_upper)


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
        episode_length_s=60.0,  # 1500 steps at 25 Hz
    ))
    command_cfg["min_speed"] = cfg.evaluation.vmin
    command_cfg["max_speed"] = cfg.evaluation.vmax
    _apply_forest_overrides(env_cfg, cfg)
    if base_init_pos is not None:
        env_cfg["base_init_pos"] = list(base_init_pos)

    obs_cfg["add_genome_obs_actor"] = False
    obs_cfg["add_genome_obs_critic"] = False

    _apply_noise_toggles(env_cfg, cfg)

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

    raw_names = list(getattr(env, "reward_names", []))
    reward_scales = getattr(env, "reward_scales", {}) or {}
    active_idx = [
        i for i, n in enumerate(raw_names)
        if float(reward_scales.get(n, 0.0)) != 0.0
    ]
    reward_names = [raw_names[i] for i in active_idx]
    n_comp = len(reward_names)
    active_idx_t = (
        torch.tensor(active_idx, device=dev, dtype=torch.long) if n_comp else None
    )

    done = torch.zeros(B, dtype=torch.bool, device=dev)
    reward_sum = torch.zeros(B, device=dev)
    reward_comp_sum = torch.zeros(B, n_comp, device=dev) if n_comp else None
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
            if reward_comp_sum is not None:
                reward_comp_sum[alive] += env.last_reward_components[alive][:, active_idx_t]
            t_acc[alive] += dt
            dx_acc[alive] = env.base_pos[alive, 0] - x0[alive]
            energy_acc[alive] += env.power[alive] * dt
            v_dev_acc[alive] += (env.base_lin_vel[alive, 0] - env.commands[alive, 0]).abs() * dt

        # Terminating step: env auto-resets base_pos before step() returns, so
        # we can't trust positional state; but last_reward_{total,components}
        # were filled pre-reset and still carry the crash penalty.
        just_done = (~done) & term & (~nan_mask)
        if just_done.any():
            reward_sum[just_done] += env.last_reward_total[just_done]
            if reward_comp_sum is not None:
                reward_comp_sum[just_done] += env.last_reward_components[just_done][:, active_idx_t]
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

    if reward_comp_sum is not None:
        comp_arr = reward_comp_sum.cpu().numpy()
        comp_arr[nan_np] = 0.0
    else:
        comp_arr = np.zeros((B, 0), dtype=np.float32)

    v_arr = np.where(t_arr > 1e-6, dx_arr / t_arr, 0.0)
    v_dev_arr = np.where(t_arr > 1e-6, v_dev_arr / t_arr, 0.0)

    # Cost of Transport = Energy / (mass * gravity * displacement)
    mg = float(env.nominal_mass) * 9.81
    cot_arr = np.where(dx_arr > 1e-6, energy_arr / (mg * dx_arr), 0.0)

    return {
        "reward_sum": reward_arr,
        "reward_components": comp_arr,
        "reward_names": reward_names,
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
        episode_length_s=60.0,  # 1500 steps at 25 Hz
    ))
    command_cfg["min_speed"] = cfg.evaluation.vmin
    command_cfg["max_speed"] = cfg.evaluation.vmax
    _apply_forest_overrides(env_cfg, cfg)
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


def _resolve_fitness_aggregator(cfg: HebbianEvolutionConfig) -> str:
    """Resolve ``cmaes.fitness_aggregator`` to ``"mean"`` or ``"median"``.

    ``None`` (or a missing field on older configs) and ``"mean"`` map to the
    legacy mean-over-forests fitness; ``"median"`` selects the median. Any
    other value is a config error.
    """
    agg = getattr(getattr(cfg, "cmaes", None), "fitness_aggregator", None)
    if agg is None:
        return "mean"
    agg = str(agg).lower()
    if agg not in ("mean", "median"):
        raise ValueError(
            f"cmaes.fitness_aggregator must be null, 'mean' or 'median' "
            f"(got {agg!r})"
        )
    return agg


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
        WP1 reward sum per individual, aggregated over the S forest slots
        with ``cmaes.fitness_aggregator`` (mean or median) and averaged over
        URDFs and episodes.
    metrics : dict
        Additional per-individual metrics (progresses, velocities, crash_flags).
    """
    import genesis as gs

    P = len(solutions)
    total_envs = cfg.evaluation.num_eval_envs
    S = total_envs // P
    actual_envs = S * P
    fitness_aggregator = _resolve_fitness_aggregator(cfg)

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
    acc_components = None
    reward_names: List[str] = []

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

        # Fix forest assignment so individual k's env i always sees forest i,
        # making fitness comparisons fair across the population.
        if env.cylinders_array is not None and P > 1:
            fixed_ids = torch.arange(S, device=cfg.device, dtype=torch.long).repeat(P)
            env._fixed_forest_ids = fixed_ids
            env.forest_ids[:actual_envs] = fixed_ids
            env._update_cylinders_xy(torch.arange(actual_envs, device=cfg.device, dtype=torch.long))

        # Tile the commanded-velocity grid so env slot i has the same target
        # speed for every individual (linspace over S, repeated P times).
        if P > 1:
            vmin_cmd = float(env.command_cfg.get("min_speed", cfg.evaluation.vmin))
            vmax_cmd = float(env.command_cfg.get("max_speed", cfg.evaluation.vmax))
            per_ind_grid = torch.linspace(vmin_cmd, vmax_cmd, S, device=cfg.device, dtype=torch.float32)
            env._eval_speed_grid = per_ind_grid.repeat(P)

        urdf_reward = np.zeros(P)
        urdf_progress = np.zeros(P)
        urdf_velocity = np.zeros(P)
        urdf_crash = np.zeros(P)
        urdf_cot = np.zeros(P)
        urdf_v_dev = np.zeros(P)
        urdf_components = None

        try:
            for ep in range(n_episodes):
                ep_metrics = _rollout_episode_reward_sum(env, actor, cfg.device, verbose=verbose)

                ep_names = ep_metrics.get("reward_names", [])
                if ep_names and not reward_names:
                    reward_names = list(ep_names)

                ep_comp = ep_metrics.get("reward_components")
                if ep_comp is not None and ep_comp.size:
                    per_ind_comp = ep_comp.reshape(P, S, -1).mean(axis=1)
                    if urdf_components is None:
                        urdf_components = np.zeros_like(per_ind_comp)
                    urdf_components += per_ind_comp

                for key in ("reward_sum", "progresses", "velocities",
                            "crash_flags", "cots", "v_deviations"):
                    flat_arr = ep_metrics[key]
                    per_slot = flat_arr.reshape(P, S)
                    per_ind = per_slot.mean(axis=1)
                    if key == "reward_sum":
                        # The fitness honours cmaes.fitness_aggregator; the
                        # diagnostic metrics below always stay means.
                        if fitness_aggregator == "median":
                            per_ind = np.median(per_slot, axis=1)
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
        if urdf_components is not None:
            urdf_components /= n_episodes

        acc_reward += urdf_reward
        acc_progress += urdf_progress
        acc_velocity += urdf_velocity
        acc_crash += urdf_crash
        acc_cot += urdf_cot
        acc_v_dev += urdf_v_dev
        if urdf_components is not None:
            if acc_components is None:
                acc_components = np.zeros_like(urdf_components)
            acc_components += urdf_components

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
    if acc_components is not None:
        acc_components /= n_urdfs

    metrics = {
        "reward_sums": acc_reward,
        "progresses": acc_progress,
        "velocities": acc_velocity,
        "crash_flags": acc_crash,
        "cots": acc_cot,
        "v_deviations": acc_v_dev,
        "reward_components": acc_components if acc_components is not None
                             else np.zeros((P, 0), dtype=np.float32),
        "reward_names": reward_names,
    }
    return acc_reward, metrics


# ============================================================================
#  Multi-URDF evaluation (single Genesis scene with N different URDFs)
# ============================================================================

def _build_multi_urdf_env(
    urdf_paths: List[str],
    cfg: HebbianEvolutionConfig,
    wp1_cfg,
    device: str,
    num_envs_per_drone: int,
    num_workers: int = 1,
    num_gpus: Optional[int] = None,
):
    """Build a multi-URDF eval env — one Genesis scene per URDF.

    When ``num_workers > 1`` returns a ``ParallelMultiSceneEvalEnv`` that
    dispatches D/N URDFs to N worker processes concurrently, exploiting GPU
    headroom.  Otherwise returns the default sequential ``MultiSceneEvalEnv``.

    We mirror ``WP1.train``'s ``Gen_Env`` pattern: each URDF gets its own
    ``WingedDroneEnv`` (its own rigid solver, aero Taichi state, and
    contact/constraint buffers).  The legacy single-scene multi-URDF env
    shared Taichi state across D URDFs, which caused NaN from one URDF's
    crashed physics to contaminate every env on the next reset → we avoid
    that failure mode entirely by isolating the scenes.

    The WP1 config is translated through ``to_legacy_cfgs()`` — exactly the
    same pipeline ``WP1.train`` uses — and then patched with the same
    eval-time overrides as the legacy single-URDF path (episode length,
    optional ``x_upper``, disabled genome obs).
    """
    from WP2.multi_scene_eval_env import MultiSceneEvalEnv

    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = wp1_cfg.to_legacy_cfgs()

    env_cfg = dict(env_cfg)
    env_cfg["episode_length_s"] = 60.0  # 1500 steps at 25 Hz
    _apply_forest_overrides(env_cfg, cfg)

    obs_cfg = dict(obs_cfg)
    obs_cfg["add_genome_obs_actor"] = False
    obs_cfg["add_genome_obs_critic"] = False

    command_cfg = dict(command_cfg)
    command_cfg["min_speed"] = cfg.evaluation.vmin
    command_cfg["max_speed"] = cfg.evaluation.vmax

    _apply_noise_toggles(env_cfg, cfg)

    env_cls_kwargs = dict(
        urdf_paths=urdf_paths,
        num_envs_per_drone=num_envs_per_drone,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=device,
    )
    if num_workers > 1:
        from WP2.parallel_multi_scene_eval_env import ParallelMultiSceneEvalEnv
        env = ParallelMultiSceneEvalEnv(
            **env_cls_kwargs,
            num_workers=num_workers,
            num_gpus=num_gpus,
        )
    else:
        env = MultiSceneEvalEnv(**env_cls_kwargs)
    # WP2 CRN: share per-slot DR draws across individuals flying the same forest
    # so CMA-ES ranks by Hebbian rules, not by independent randomization luck.
    env._crn_enabled = bool(getattr(cfg.evaluation, "crn", True))
    return env


@torch.no_grad()
def _rollout_episode_multi_urdf(
    env,
    actor,
    device: str,
    P: int,
    F: int,
    verbose: bool = False,
    fitness_aggregator: str = "mean",
) -> Dict[str, np.ndarray]:
    """One rollout across a ``MultiSceneEvalEnv`` holding D URDFs in D scenes.

    Layout
    ------
    - Env slot ``e`` within drone ``d`` encodes ``(individual p, forest f)``
      via ``e = p*F + f``.  The per-slot forest-ids and target speeds are
      broadcast to every sub-env so each (drone, individual) pair sees the
      same F forests and F target speeds.
    - The actor expects a flat ``(P*D*F, obs_dim)`` batch with individual
      as the OUTER index (``slot = p*D*F + d*F + f``) so its per-controller
      forward batches are contiguous.  We achieve this with a
      ``(D, P, F, obs_dim) → (P, D, F, obs_dim)`` permute before ``act()``
      and the inverse after.

    Returns per-individual metric arrays of shape ``(P,)``.
    """
    D = env.D
    E = env.E
    if P * F != E:
        raise ValueError(
            f"_rollout_episode_multi_urdf: P*F ({P}*{F}={P*F}) must equal "
            f"env.E ({E})"
        )

    dt = env.dt
    dev = torch.device(device)

    _ref = env.drones[0]
    raw_names = list(getattr(_ref, "reward_names", []))
    _scales = getattr(_ref, "reward_scales", {}) or {}
    active_idx = [
        i for i, n in enumerate(raw_names)
        if float(_scales.get(n, 0.0)) != 0.0
    ]
    reward_names: List[str] = [raw_names[i] for i in active_idx]
    n_comp = len(reward_names)
    active_idx_t = (
        torch.tensor(active_idx, device=dev, dtype=torch.long) if n_comp else None
    )

    # Per-(drone, env) accumulators — reduced to per-individual at episode end.
    done = torch.zeros(D, E, dtype=torch.bool, device=dev)
    reward_sum = torch.zeros(D, E, device=dev)
    reward_comp_sum = (
        torch.zeros(D, E, n_comp, device=dev) if n_comp else None
    )
    t_acc = torch.zeros(D, E, device=dev)
    dx_acc = torch.zeros(D, E, device=dev)
    energy_acc = torch.zeros(D, E, device=dev)
    v_dev_acc = torch.zeros(D, E, device=dev)
    crashed = torch.zeros(D, E, dtype=torch.bool, device=dev)
    nan_tracker = torch.zeros(D, E, dtype=torch.bool, device=dev)

    actor.reset_episode(device=dev)
    obs, _ = env.reset()  # (D, E, obs_dim)
    # Starting x position per (drone, env)
    x0 = torch.stack([ds.base_pos[:, 0].clone() for ds in env.drones], dim=0)  # (D, E)

    step = 0
    while not done.all():
        # (D, E, obs) → (P, D, F, obs) → (P*D*F, obs)
        obs_pdf = obs.view(D, P, F, -1).permute(1, 0, 2, 3).contiguous()
        actor_obs = obs_pdf.view(P * D * F, -1)

        actor_act = actor.act(actor_obs)  # (P*D*F, num_actions)
        act_pdf = actor_act.view(P, D, F, -1).permute(1, 0, 2, 3).contiguous()
        actions = act_pdf.view(D, E, -1)

        obs, _, _, _ = env.step(actions)  # (D, E, obs), ...

        # ``WingedDroneEnv.step`` has already populated ``ds.last_reward_total``
        # and termination flags for every sub-env; the shim call below is a
        # no-op kept for backward compatibility with the legacy single-scene
        # rollout.
        for d, ds in enumerate(env.drones):
            env._compute_rewards_drone(d, ds)

            nan_d = ds.nan_envs.to(torch.bool)
            term_d = ds.reset_buf.clone()  # this step's termination flag (set in _check_termination)
            alive_d = (~done[d]) & (~term_d) & (~nan_d)

            if alive_d.any():
                reward_sum[d, alive_d] += ds.last_reward_total[alive_d]
                if reward_comp_sum is not None:
                    reward_comp_sum[d, alive_d] += (
                        ds.last_reward_components[alive_d][:, active_idx_t]
                    )
                t_acc[d, alive_d] += dt
                dx_acc[d, alive_d] = ds.base_pos[alive_d, 0] - x0[d, alive_d]
                energy_acc[d, alive_d] += ds.power[alive_d] * dt
                v_dev_acc[d, alive_d] += (
                    ds.base_lin_vel[alive_d, 0] - ds.commands[alive_d, 0]
                ).abs() * dt

            # Terminating step: env auto-resets base_pos before step() returns,
            # so positional state is invalid; reward buffers were filled
            # pre-reset and still carry the crash penalty.
            just_done = (~done[d]) & term_d & (~nan_d)
            if just_done.any():
                reward_sum[d, just_done] += ds.last_reward_total[just_done]
                if reward_comp_sum is not None:
                    reward_comp_sum[d, just_done] += (
                        ds.last_reward_components[just_done][:, active_idx_t]
                    )
                for attr in ("pre_collision", "pre_wall_crash", "pre_angle_limit"):
                    flag = getattr(ds, attr, None)
                    if flag is not None:
                        crashed[d] |= just_done & flag.to(torch.bool)

            nan_tracker[d] |= nan_d & ~done[d]
            done[d] |= term_d | nan_d

        step += 1

        if verbose and step % 50 == 0:
            n_alive = int((~done).sum().item())
            mean_dx = float(dx_acc.mean().item())
            max_dx = float(dx_acc.max().item())
            mean_r = float(reward_sum.mean().item())
            print(
                f"  step {step:5d} | alive {n_alive:5d}/{D * E}"
                f" | progress mean {mean_dx:7.1f} m  max {max_dx:7.1f} m"
                f" | reward mean {mean_r:8.2f}",
                flush=True,
            )

    # Zero contributions from NaN slots, then reduce to per-individual metrics.
    valid = ~nan_tracker  # (D, E)
    reward_sum = reward_sum * valid
    t_acc = t_acc * valid
    dx_acc = dx_acc * valid
    energy_acc = energy_acc * valid
    v_dev_acc = v_dev_acc * valid
    if reward_comp_sum is not None:
        reward_comp_sum = reward_comp_sum * valid.unsqueeze(-1)

    # (D, E=P*F) → (D, P, F) → mean over (D, F) → (P,)
    def _per_ind(metric: torch.Tensor) -> torch.Tensor:
        return metric.view(D, P, F).float().mean(dim=(0, 2))

    # (D, E=P*F) → (D, P, F) → mean over F only → (D, P). Keeps the per-URDF
    # axis so callers (e.g. the NSGA-II outer loop) can read morphology-level
    # objectives from the same rollouts.
    def _per_urdf(metric: torch.Tensor) -> torch.Tensor:
        return metric.view(D, P, F).float().mean(dim=2)

    if fitness_aggregator == "median":
        # Median over the D*F per-(urdf, forest) rollout samples of each
        # individual. torch.quantile interpolates the two middle values for
        # even sample counts, matching np.median. Fitness only — the
        # diagnostic metrics below always stay means.
        samples = reward_sum.view(D, P, F).permute(1, 0, 2).reshape(P, D * F)
        reward_per_ind = torch.quantile(samples.float(), 0.5, dim=1)
    else:
        reward_per_ind = _per_ind(reward_sum)
    t_per_ind = _per_ind(t_acc)
    dx_per_ind = _per_ind(dx_acc)
    energy_per_ind = _per_ind(energy_acc)
    v_dev_per_ind = _per_ind(v_dev_acc)
    crash_per_ind = _per_ind(crashed)

    if reward_comp_sum is not None:
        # (D, E, C) → (D, P, F, C) → mean over (D, F) → (P, C)
        comp_per_ind = reward_comp_sum.view(D, P, F, n_comp).float().mean(dim=(0, 2))
    else:
        comp_per_ind = None

    # Scalar arrays on CPU
    reward_arr = reward_per_ind.cpu().numpy()
    t_arr = t_per_ind.cpu().numpy()
    dx_arr = dx_per_ind.cpu().numpy()
    energy_arr = energy_per_ind.cpu().numpy()
    v_dev_arr = v_dev_per_ind.cpu().numpy()
    crash_arr = crash_per_ind.cpu().numpy()

    v_arr = np.where(t_arr > 1e-6, dx_arr / t_arr, 0.0)
    v_dev_arr = np.where(t_arr > 1e-6, v_dev_arr / t_arr, 0.0)

    # Per-(URDF, individual) matrices, shape (D, P).
    pu_reward = _per_urdf(reward_sum).cpu().numpy()
    pu_t = _per_urdf(t_acc).cpu().numpy()
    pu_dx_t = _per_urdf(dx_acc)  # torch, reused for the CoT ratio below
    pu_dx = pu_dx_t.cpu().numpy()
    pu_crash = _per_urdf(crashed).cpu().numpy()
    pu_velocity = np.where(pu_t > 1e-6, pu_dx / pu_t, 0.0)

    # COT averaged across drones (per drone different nominal mass). Compute per-slot
    # then reduce to per-individual to get the right weighting.
    mg_per_drone = torch.tensor(
        [float(ds.nominal_mass) * 9.81 for ds in env.drones],
        device=dev, dtype=torch.float32,
    ).view(D, 1)  # (D, 1)
    cot_slot = torch.where(
        dx_acc > 1e-6,
        energy_acc / (mg_per_drone * dx_acc.clamp(min=1e-6)),
        torch.zeros_like(dx_acc),
    )
    cot_arr = _per_ind(cot_slot).cpu().numpy()
    # Per-URDF CoT as ratio-of-sums (total energy / total weight·distance)
    # instead of mean-of-ratios: slots that crash immediately have dx≈0 and
    # would otherwise contribute cot=0 — spuriously *good* for a minimize
    # objective. With the ratio, near-zero distance ⇒ large CoT (correctly
    # penalised).
    pu_energy = _per_urdf(energy_acc)  # (D, P)
    pu_cot = (
        pu_energy / (mg_per_drone * pu_dx_t.clamp(min=1e-2))
    ).cpu().numpy()

    # Unreduced per-(URDF, individual, forest) views, shape (D, P, F). Nothing
    # in the inner/outer loops reads these — they exist so offline tools can
    # report a spread across forests (e.g. WP2_Outer_Loop.transfer_eval's
    # standard error), which the (D, P) means above have already averaged out.
    # CoT uses the same distance clamp as the per-URDF ratio, so a slot that
    # crashed at ~zero distance reads as expensive, not free.
    def _per_slot(metric: torch.Tensor) -> np.ndarray:
        return metric.view(D, P, F).float().cpu().numpy()

    ps_t = _per_slot(t_acc)
    ps_dx = _per_slot(dx_acc)
    per_slot = {
        "per_slot_reward":   _per_slot(reward_sum),
        "per_slot_progress": ps_dx,
        "per_slot_velocity": np.where(ps_t > 1e-6, ps_dx / np.maximum(ps_t, 1e-12), 0.0),
        "per_slot_crash":    _per_slot(crashed),
        "per_slot_cot": _per_slot(
            energy_acc / (mg_per_drone * dx_acc.clamp(min=1e-2))
        ),
    }

    if comp_per_ind is not None:
        comp_arr = comp_per_ind.cpu().numpy()
    else:
        comp_arr = np.zeros((P, 0), dtype=np.float32)

    return {
        "reward_sum": reward_arr,
        "reward_components": comp_arr,
        "reward_names": reward_names,
        "progresses": dx_arr,
        "velocities": v_arr,
        "crash_flags": crash_arr,
        "cots": cot_arr,
        "v_deviations": v_dev_arr,
        "per_urdf_reward": pu_reward,
        "per_urdf_progress": pu_dx,
        "per_urdf_velocity": pu_velocity,
        "per_urdf_crash": pu_crash,
        "per_urdf_cot": pu_cot,
        **per_slot,
    }


def evaluate_population_multi_urdf(
    solutions: List[np.ndarray],
    cfg: HebbianEvolutionConfig,
    model_and_layer: Tuple,
    wp1_cfg,
    urdf_paths: List[str],
    existing_env=None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Evaluate a CMA-ES population over a single multi-URDF Genesis scene.

    Sibling of ``evaluate_population_cma_batched`` that supports N≥1 URDFs in
    one scene. The legacy path is left completely unchanged; this function is
    dispatched to by ``HebbianCMAES`` when a catalog (or auto-generated URDFs)
    is active, or when ``cfg.catalog.force_multi_urdf`` is set.

    Parameters
    ----------
    solutions : list[np.ndarray]
        CMA-ES genomes in [0,1]^n (P = len(solutions) individuals).
    cfg : HebbianEvolutionConfig
    model_and_layer : tuple
        ``(model, last_layer, num_actions, hidden_dim)``.
    wp1_cfg : RunConfig
    urdf_paths : list[str]
        The N URDF files held simultaneously in the Genesis scene.
    existing_env : tuple (env, urdf_paths) or None
        Pre-built MultiSceneEvalEnv to reuse (built once by the caller).

    The returned fitness ("reward_sums") aggregates the N*F per-(urdf, forest)
    rollout samples of each individual with ``cmaes.fitness_aggregator``
    (mean by default, median when set); all other metrics are means.
    """
    import genesis as gs

    P = len(solutions)
    N = len(urdf_paths)
    fitness_aggregator = _resolve_fitness_aggregator(cfg)
    if P == 0 or N == 0:
        raise ValueError(
            f"evaluate_population_multi_urdf: need P≥1 and N≥1 "
            f"(got P={P}, N={N})"
        )

    # Determine layout. When an existing env is reusable, derive from it so
    # every call (main eval with P=H and baseline with P=1) maps cleanly to
    # the same scene without rebuilds. Otherwise compute from num_eval_envs.
    existing_ok = False
    if existing_env is not None:
        env_obj, _paths = existing_env
        if env_obj is not None and env_obj.D == N:
            existing_ok = True

    if existing_ok:
        envs_per_drone = existing_env[0].E
        F = envs_per_drone // P
        if F == 0:
            raise ValueError(
                f"Existing env has {envs_per_drone} envs/drone, too small for "
                f"P={P} individuals. Grow num_eval_envs or reduce population."
            )
        if F * P != envs_per_drone:
            raise ValueError(
                f"Existing env has {envs_per_drone} envs/drone which is not "
                f"divisible by P={P} (F*P={F*P}). Pre-build the env with a "
                f"size that is a common multiple of all P values you evaluate."
            )
    else:
        total_envs = cfg.evaluation.num_eval_envs
        F = (total_envs // N) // P
        if F == 0:
            raise ValueError(
                f"num_eval_envs ({total_envs}) too small: need at least "
                f"N*P = {N}*{P} = {N * P} envs so each (urdf, individual) "
                f"pair gets at least 1 forest."
            )
        envs_per_drone = P * F
    actual_total = N * envs_per_drone

    if verbose:
        print(
            f"[evaluate_population_multi_urdf] P={P} individuals, N={N} URDFs, "
            f"F={F} forests/(urdf,ind), envs/drone={envs_per_drone}, "
            f"total_slots={actual_total}"
        )

    na, hd = cfg.hebbian.num_actions, cfg.hebbian.hidden_dim

    hebbian_rules_per_individual: List[Dict[str, torch.Tensor]] = []
    for genome in solutions:
        hebb_part = list(np.clip(genome, 0.0, 1.0))
        rules = decode_hebbian_genes(hebb_part, cfg.hebbian, out_features=na, in_features=hd)
        hebbian_rules_per_individual.append(rules)

    # Build / reuse env
    env = None
    env_was_built_here = False
    if existing_ok and existing_env[0].E == envs_per_drone:
        env = existing_env[0]
    if env is None:
        if not gs._initialized:
            gs.init(logging_level="error", backend=gs.gpu)
        env = _build_multi_urdf_env(
            urdf_paths=urdf_paths,
            cfg=cfg,
            wp1_cfg=wp1_cfg,
            device=cfg.device,
            num_envs_per_drone=envs_per_drone,
            num_workers=int(getattr(cfg.evaluation, "num_eval_workers", 1)),
            num_gpus=int(getattr(cfg.evaluation, "num_gpus", 0)) or None,
        )
        env_was_built_here = True

    # Deterministic forest + velocity grid, shared across all D drones. Each
    # individual's block of F contiguous env slots sees forests [0..F-1] and
    # speeds linspace(vmin, vmax, F), so (urdf, individual) comparisons are fair.
    # Propagated to every sub-env by ``MultiSceneEvalEnv``'s setter; sub-env
    # reset_idx then copies the shared per-slot ids into its own forest_ids
    # and updates cylinders_xy accordingly.
    fixed_ids = torch.arange(F, device=cfg.device, dtype=torch.long).repeat(P)  # (E,)
    v_grid = torch.linspace(
        float(cfg.evaluation.vmin), float(cfg.evaluation.vmax), F,
        device=cfg.device, dtype=torch.float32,
    ).repeat(P)  # (E,)
    env._fixed_forest_ids = fixed_ids
    env._eval_speed_grid = v_grid

    # Actor: K=P individuals, S=N*F env slots per individual.
    actor = build_isolated_population_actor(
        checkpoint_path=cfg.checkpoint_path,
        wp1_cfg_path=cfg.checkpoint_config_path,
        hebbian_rules_per_individual=hebbian_rules_per_individual,
        cfg=cfg,
        K=P,
        S=N * F,
        device=cfg.device,
        stochastic=cfg.evaluation.stochastic,
    )

    n_episodes = cfg.catalog.num_episodes

    acc = {
        "reward_sums":  np.zeros(P),
        "progresses":   np.zeros(P),
        "velocities":   np.zeros(P),
        "crash_flags":  np.zeros(P),
        "cots":         np.zeros(P),
        "v_deviations": np.zeros(P),
        # Per-(URDF, individual) matrices — same rollouts, morphology axis kept.
        "per_urdf_reward":   np.zeros((N, P)),
        "per_urdf_progress": np.zeros((N, P)),
        "per_urdf_velocity": np.zeros((N, P)),
        "per_urdf_crash":    np.zeros((N, P)),
        "per_urdf_cot":      np.zeros((N, P)),
        # Unreduced (URDF, individual, forest) views — see _rollout_episode_multi_urdf.
        "per_slot_reward":   np.zeros((N, P, F)),
        "per_slot_progress": np.zeros((N, P, F)),
        "per_slot_velocity": np.zeros((N, P, F)),
        "per_slot_crash":    np.zeros((N, P, F)),
        "per_slot_cot":      np.zeros((N, P, F)),
    }
    acc_components: Optional[np.ndarray] = None
    reward_names: List[str] = []

    try:
        for _ep in range(n_episodes):
            ep_metrics = _rollout_episode_multi_urdf(
                env, actor, cfg.device, P=P, F=F, verbose=verbose,
                fitness_aggregator=fitness_aggregator,
            )
            acc["reward_sums"]  += ep_metrics["reward_sum"]
            acc["progresses"]   += ep_metrics["progresses"]
            acc["velocities"]   += ep_metrics["velocities"]
            acc["crash_flags"]  += ep_metrics["crash_flags"]
            acc["cots"]         += ep_metrics["cots"]
            acc["v_deviations"] += ep_metrics["v_deviations"]
            for _pu in ("per_urdf_reward", "per_urdf_progress",
                        "per_urdf_velocity", "per_urdf_crash", "per_urdf_cot",
                        "per_slot_reward", "per_slot_progress",
                        "per_slot_velocity", "per_slot_crash", "per_slot_cot"):
                acc[_pu] += ep_metrics[_pu]

            ep_comp = ep_metrics.get("reward_components")
            if ep_comp is not None and ep_comp.size:
                if acc_components is None:
                    acc_components = np.zeros_like(ep_comp)
                acc_components += ep_comp
            ep_names = ep_metrics.get("reward_names", [])
            if ep_names and not reward_names:
                reward_names = list(ep_names)
    finally:
        # Only tear down if we built it in this call.  Otherwise the caller owns
        # the env lifecycle (and will reuse it across generations).
        if env_was_built_here:
            try:
                gs.destroy()
            except Exception:
                pass

    for k in acc:
        acc[k] /= n_episodes
    if acc_components is not None:
        acc_components /= n_episodes

    acc["reward_components"] = (
        acc_components if acc_components is not None
        else np.zeros((P, 0), dtype=np.float32)
    )
    acc["reward_names"] = reward_names

    return acc["reward_sums"], acc


# ============================================================================
#  Standalone driver
# ============================================================================

if __name__ == "__main__":
    """Evaluate a single saved Hebbian genome against a Genesis forest env.

    Usage
    -----
    # Evaluate the best genome by fitness (default):
        python -m WP2.evaluate --run logs/runs_hebbian/2026-xx-xx_my_run

    # Override forest length to 1200 m:
        python -m WP2.evaluate --run logs/runs_hebbian/2026-xx-xx_my_run --x-upper 1200

    # Use the best individual selected by crash_rate:
        python -m WP2.evaluate --run logs/runs_hebbian/2026-xx-xx_my_run \\
            --best crash_rate --x-upper 600

    # Point at an explicit genome file:
        python -m WP2.evaluate --run logs/runs_hebbian/2026-xx-xx_my_run \\
            --genome path/to/genome.npy --x-upper 800

    # Evaluate a specific generation's best individual:
        python -m WP2.evaluate --run logs/runs_hebbian/2026-xx-xx_my_run \\
            --genome logs/runs_hebbian/2026-xx-xx_my_run/generations/gen_042/solutions.npy \\
            --genome-idx 0

    # Compare Hebbian vs. frozen baseline (zero rules):
        python -m WP2.evaluate --run logs/runs_hebbian/2026-xx-xx_my_run --compare \\
            --best crash_rate
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
        help="Path to a .npy genome file.  Defaults to <run>/best_individual/fitness/genome.npy.",
    )
    parser.add_argument(
        "--genome-idx", type=int, default=0,
        help="Row index to use when the .npy file contains multiple genomes (e.g. solutions.npy).",
    )
    parser.add_argument(
        "--best", type=str, default=None,
        help=(
            "Name of the best-individual subfolder to evaluate "
            "(e.g. crash_rate, fitness, progress).  "
            "Must be a directory inside <run>/best_individual/.  "
            "Ignored if --genome is set explicitly."
        ),
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
        "--vmin", type=float, default=None,
        help="Override evaluation.vmin (minimum commanded speed in m/s).",
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Override evaluation.vmax (maximum commanded speed in m/s).",
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

    if args.genome:
        genome_path = Path(args.genome)
    else:
        best_individual_dir = run_dir / "best_individual"
        best_name = args.best or "fitness"
        best_subfolder = best_individual_dir / best_name
        if not best_subfolder.is_dir():
            available = sorted(p.name for p in best_individual_dir.iterdir() if p.is_dir()) if best_individual_dir.is_dir() else []
            avail_str = ", ".join(available) if available else "(none found)"
            sys.exit(
                f"[ERROR] '{best_name}' is not a valid best_individual subfolder.\n"
                f"        Available: {avail_str}"
            )
        genome_path = best_subfolder / "genome.npy"
    if not genome_path.is_file():
        sys.exit(f"[ERROR] Genome file not found: {genome_path}")

    # --- load WP2 config ---
    from WP2.config import HebbianEvolutionConfig
    cfg = HebbianEvolutionConfig.from_yaml(cfg_path)

    if args.device:
        cfg.device = args.device
    if args.num_envs:
        cfg.evaluation.num_eval_envs = args.num_envs
    if args.vmin is not None:
        cfg.evaluation.vmin = args.vmin
    if args.vmax is not None:
        cfg.evaluation.vmax = args.vmax
    if args.stochastic is not None:
        cfg.evaluation.stochastic = args.stochastic
    cfg.catalog.num_episodes = args.episodes

    # Infer last-layer dims from checkpoint (same as run.py does)
    import torch
    _ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    _sd = _ckpt.get("model_state_dict", _ckpt) if isinstance(_ckpt, dict) else _ckpt
    _last_key = last_actor_linear_key(_sd)
    if _last_key is not None:
        cfg.hebbian.num_actions = _sd[_last_key].shape[0]
        cfg.hebbian.hidden_dim = _sd[_last_key].shape[1]
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
        abcd_block = cfg.hebbian.abcd_block_size()  # out×in per-weight, or out per-neuron
        baseline_parts = [
            np.full(abcd_block, _gene_for_zero(*cfg.hebbian.A_range)),
            np.full(abcd_block, _gene_for_zero(*cfg.hebbian.B_range)),
            np.full(abcd_block, _gene_for_zero(*cfg.hebbian.C_range)),
            np.full(abcd_block, _gene_for_zero(*cfg.hebbian.D_range)),
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
