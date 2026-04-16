#!/usr/bin/env python3
"""
Compare run_eval and _rollout_episode over N episodes with the same
morphology, controller, and per-episode forest seed.

Both evaluation loops are run against the same WingedDroneEnv instance.
Per-episode forest assignment is controlled by seeding torch before each
env.reset() call, so both functions see the same tree layouts.

The controller is loaded from a WP1 checkpoint.  By default, zero Hebbian
rules are used for the _rollout_episode path, which makes the two paths
share identical frozen WP1 weights.  Under these conditions (deterministic
inference, same LSTM initial state, same seed) progress and velocity should
match to floating-point precision.  Any divergence indicates a behavioural
difference in one of the evaluation loops.

Usage
-----
    python tests/integration/test_eval_comparison.py \\
        --checkpoint  logs/runs/<run>/tb/model_1999.pt \\
        --wp1-config  logs/runs/<run>/config.yaml \\
        --urdf        src/urdf_generated/standard.urdf \\
        [--n-episodes 10] [--num-envs 32] [--seed 42] [--device cuda:0]

Notes
-----
- Requires a Genesis-capable GPU.
- Genesis is initialised once and destroyed at the end; do not run
  alongside another Genesis process.
- Run from the repository root so that relative paths resolve correctly.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

os.environ["GS_PARA_LEVEL"] = "3"  # evaluation-appropriate parallelism


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _configure_cache_root() -> None:
    cache = (Path("logs") / ".cache" / "gstaichi").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for key in ("XDG_CACHE_HOME", "TI_CACHE_DIR", "TAICHI_CACHE_DIR", "GSTAICHI_CACHE_DIR"):
        os.environ[key] = str(cache)
    mpl = cache / "mpl"
    mpl.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl)


def _build_env(wp1_config_path: str, urdf_path: str, num_envs: int, device: str):
    """Build a WingedDroneEnv with standard evaluation overrides."""
    from WP1.config import RunConfig
    from winged_drone_train.env import WingedDroneEnv
    from winged_drone_train.noise_config import configure_solver_noise

    wp1_cfg = RunConfig.from_yaml(wp1_config_path)
    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = wp1_cfg.to_legacy_cfgs()

    # Mirror the overrides applied by winged_drone_train/eval.py::_apply_eval_env_overrides
    env_cfg.update(dict(
        visualize_camera=False,
        visualize_target=False,
        unique_forests_eval=True,
        growing_forest=True,
        x_upper=600,
        forest_x_limit=600,
        tree_radius=0.75,
        base_init_pos=[-50.0, 0.0, 10.0],
    ))
    obs_cfg["add_genome_obs_actor"] = False
    obs_cfg["add_genome_obs_critic"] = False

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
    return env, wp1_cfg


def _load_wp1_policy(checkpoint_path: str, wp1_config_path: str, env, device: str):
    """Load the WP1 inference policy via OnPolicyRunner."""
    import builtins
    from rsl_rl.runners import OnPolicyRunner
    from WP1.config import RunConfig
    from winged_drone_train.rl.A2C_modified import ActorCriticTanh
    builtins.ActorCriticTanh = ActorCriticTanh  # OnPolicyRunner uses eval() on class_name

    wp1_cfg = RunConfig.from_yaml(wp1_config_path)
    train_cfg = copy.deepcopy(wp1_cfg.to_train_cfg())
    log_dir = str(Path(checkpoint_path).parent)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)
    return policy, runner


def _build_hebbian_actor(
    checkpoint_path: str,
    wp1_config_path: str,
    num_envs: int,
    device: str,
):
    """Build an IsolatedPopulationActor (K=1, S=num_envs) with zero Hebbian rules.

    Zero rules mean the last-layer weights are never updated, so this actor
    is behaviourally equivalent to the plain WP1 policy (same frozen weights,
    same LSTM backbone).
    """
    from WP2.frozen_actor import load_frozen_actor, build_isolated_population_actor

    _, _, num_actions, hidden_dim = load_frozen_actor(
        checkpoint_path, wp1_config_path, device=device
    )

    z = torch.zeros(num_actions, hidden_dim, device=device)
    zero_rules = {"A": z, "B": z, "C": z, "D": z, "lam": z}

    # Minimal config: only the fields accessed by attach_hebbian / HebbianLastLayer
    class _HebbCfg:
        eta = 0.0
        w_max = 1e9          # effectively unbounded (zero rules won't update anyway)
        use_oja_coefficient = False
        decay = 0.0

    class _Cfg:
        hebbian = _HebbCfg()

    return build_isolated_population_actor(
        checkpoint_path=checkpoint_path,
        wp1_cfg_path=wp1_config_path,
        hebbian_rules_per_individual=[zero_rules],
        cfg=_Cfg(),
        K=1,
        S=num_envs,
        device=device,
        stochastic=False,   # deterministic, mirrors act_inference
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_run_eval(env, policy, runner, seed: int) -> Tuple[float, float]:
    """One episode via run_eval.  Returns (mean_progress_m, mean_v_mean_mps)."""
    from winged_drone_train.eval import run_eval

    runner.alg.policy.memory_a.reset()  # zero LSTM hidden state
    torch.manual_seed(seed)             # control forest assignment in env.reset()
    v_mean, _cot, _v_cmd, progress, _reason, _traces, _rew = run_eval(env, policy)
    # run_eval already filters NaN environments; arrays have shape (n_valid,)
    return float(np.mean(progress)), float(np.mean(v_mean))


def _run_one_rollout_episode(env, actor, device: str, seed: int) -> Tuple[float, float]:
    """One episode via _rollout_episode.  Returns (mean_progress_m, mean_v_mean_mps)."""
    from WP2.evaluate import _rollout_episode_reward_sum

    # _rollout_episode_reward_sum calls actor.reset_episode() and env.reset() internally.
    # Seed here controls the forest assignment inside env.reset().
    torch.manual_seed(seed)
    m = _rollout_episode_reward_sum(env, actor, device)
    # NaN-crashed envs have progress=0 and velocities=0; exclude them
    # by using the same mask the function already applied internally.
    prog = m["progresses"]   # shape (B,), NaN envs → 0.0
    vel  = m["velocities"]   # shape (B,), NaN envs → 0.0
    valid = prog > 0          # crude NaN filter (see note in docstring)
    if not valid.any():
        return 0.0, 0.0
    return float(prog[valid].mean()), float(vel[valid].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(args) -> int:
    import genesis as gs
    from winged_drone_train.defaults import default_mydrone_urdf_path

    if args.urdf is None:
        args.urdf = str(default_mydrone_urdf_path())

    _configure_cache_root()
    if not gs._initialized:
        gs.init(logging_level="error", backend=gs.gpu)

    print(f"[test] Building env  ({args.num_envs} envs, urdf={Path(args.urdf).name})")
    env, _ = _build_env(args.wp1_config, args.urdf, args.num_envs, args.device)

    print(f"[test] Loading WP1 policy  ({Path(args.checkpoint).name})")
    wp1_policy, runner = _load_wp1_policy(args.checkpoint, args.wp1_config, env, args.device)

    print(f"[test] Building Hebbian actor  (K=1, S={args.num_envs}, zero rules)")
    hebb_actor = _build_hebbian_actor(args.checkpoint, args.wp1_config, args.num_envs, args.device)

    re_prog, re_vel = [], []
    ro_prog, ro_vel = [], []

    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  {'Episode':>8}  {'seed':>6}  {'function':<22}  {'progress (m)':>12}  {'v_mean (m/s)':>12}")
    print(sep)

    for ep in range(args.n_episodes):
        seed = args.seed + ep

        prog_re, vel_re = _run_one_run_eval(env, wp1_policy, runner, seed)
        prog_ro, vel_ro = _run_one_rollout_episode(env, hebb_actor, args.device, seed)

        delta = abs(prog_re - prog_ro)
        flag  = "  !!" if delta > 1.0 else ""

        print(f"  {ep + 1:>8}  {seed:>6}  {'run_eval':<22}  {prog_re:>12.2f}  {vel_re:>12.3f}")
        print(f"  {'':>8}  {'':>6}  {'_rollout_episode':<22}  {prog_ro:>12.2f}  {vel_ro:>12.3f}  Δ={delta:.3f}{flag}")
        print(sep)

        re_prog.append(prog_re); re_vel.append(vel_re)
        ro_prog.append(prog_ro); ro_vel.append(vel_ro)

    # ── summary table ────────────────────────────────────────────────────
    def _stats(vals):
        a = np.array(vals, dtype=float)
        return dict(min=a.min(), mean=a.mean(), max=a.max(), std=a.std())

    print(f"\n{'=' * 68}")
    print("SUMMARY")
    print(f"{'=' * 68}")
    rows = [
        ("run_eval",         "progress (m)",  re_prog),
        ("run_eval",         "v_mean (m/s)",  re_vel),
        ("_rollout_episode", "progress (m)",  ro_prog),
        ("_rollout_episode", "v_mean (m/s)",  ro_vel),
    ]
    print(f"  {'function':<22}  {'metric':<14}  {'min':>8}  {'mean':>8}  {'max':>8}  {'std':>8}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for fn, metric, vals in rows:
        s = _stats(vals)
        print(f"  {fn:<22}  {metric:<14}  {s['min']:>8.2f}  {s['mean']:>8.2f}  {s['max']:>8.2f}  {s['std']:>8.2f}")

    delta_prog = np.abs(np.array(re_prog) - np.array(ro_prog))
    delta_vel  = np.abs(np.array(re_vel)  - np.array(ro_vel))
    print(f"\n  Max |Δ progress| across {args.n_episodes} episodes: {delta_prog.max():.4f} m")
    print(f"  Max |Δ v_mean|  across {args.n_episodes} episodes: {delta_vel.max():.4f} m/s")

    consistent = delta_prog.max() < 0.1
    if consistent:
        print("\n  ✓  Outputs agree — functions are consistent.")
    else:
        print("\n  ✗  Outputs differ — check LSTM init, Hebbian reset, or termination logic.")

    gs.destroy()
    return 0 if consistent else 1


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint",  required=True,
                    help="Path to WP1 model checkpoint (.pt)")
    ap.add_argument("--wp1-config",  required=True,
                    help="Path to WP1 config.yaml (from the same run folder)")
    ap.add_argument("--urdf",        default=None,
                    help="Path to the URDF file that defines the morphology "
                         "(default: standard mydrone URDF from winged_drone_train.defaults)")
    ap.add_argument("--n-episodes",  type=int,   default=5,
                    help="Number of episodes to run per function (default: 5)")
    ap.add_argument("--num-envs",    type=int,   default=16,
                    help="Number of parallel environments (default: 16)")
    ap.add_argument("--seed",        type=int,   default=42,
                    help="Base seed; episode i uses seed+i (default: 42)")
    ap.add_argument("--device",      default="cuda:0",
                    help="Torch/Genesis device (default: cuda:0)")
    return run_comparison(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
