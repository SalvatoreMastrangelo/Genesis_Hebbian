"""
Prune-vs-original controller evaluation.
========================================

Evaluates a frozen WP1 controller and a *structurally pruned* copy of it
side-by-side, on the same forests, reporting the same metrics as
``src/winged_drone_train/eval.py`` (SUCCESS/OBSTACLES/WALLS/ANGLES, max
velocity, COT, progress, eval reward, mean z at the progress threshold).

The prune targets the **output bottleneck**: the last Linear layer of the
actor reads from ``hidden_dim`` units, but many of those units are dead
(their outgoing weights are ~0). We drop the dead units — slicing the
last Linear's input columns and the preceding Linear's output rows at the
same indices — yielding an ~equivalent but smaller network. For a WP2
Hebbian search this directly shrinks the per-weight ABCD genome
(``4 * num_actions * hidden_dim``).

Everything that matters for a fair A/B is held identical between the two
rollouts: the same pre-built env, and the global RNG re-seeded to the same
value immediately before each ``run_eval`` (forests, commands, DR and noise
are all drawn from the global RNG at ``reset``), so the *only* difference is
the pruned readout.

Inference path is identical to eval.py: ``runner.get_inference_policy()``
(which bakes in empirical observation normalisation), so the ORIGINAL column
here reproduces what eval.py would report for this controller.

Usage
-----
    # Point at a WP2 run dir (uses its reproducibility/ checkpoint + WP1 config):
    PYTHONPATH=src python -m WP2.prune_eval \
        --run logs/runs_hebbian/2026-05-18_11-55-12_intermediate_multi_urdf_actual_controller

    # Or pass an explicit checkpoint + WP1 config:
    PYTHONPATH=src python -m WP2.prune_eval \
        --checkpoint path/to/model_999.pt --config path/to/wp1_config.yaml \
        --envs 8192 --vmin 10 --vmax 20

Prune control (default: relative threshold, auto-inferred K):
    --prune-rel 0.02        keep units with colnorm > 0.02 * max(colnorm)  [default]
    --prune-threshold T     keep units with colnorm > T (absolute override)
    --keep K                force-keep the top-K units by colnorm
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import genesis as gs

from winged_drone_train.env import WingedDroneEnv, _apply_drone_profile_defaults
from winged_drone_train.eval import (
    run_eval,
    _apply_eval_env_overrides,
    _command_speed_range,
)
from winged_drone_train.urdf_resolver import resolve_or_generate_urdf
from winged_drone_train.analysis.eval_plotter import EvaluationPlotter
from rsl_rl.runners import OnPolicyRunner


# ===========================================================================
#  Helpers
# ===========================================================================

def _reseed(seed: int) -> None:
    """Reset every RNG that the env draws from at reset/step.

    Forests, commands, domain-randomisation and per-step noise are all sampled
    from the global torch (and numpy) generators, so re-seeding to the same
    value before each rollout makes both policies face identical scenarios —
    the only divergence is the one caused by the pruned readout itself.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_actor_critic(runner: OnPolicyRunner):
    """Return the actor-critic module from an rsl_rl runner (version-robust)."""
    alg = runner.alg
    ac = getattr(alg, "policy", None)
    if ac is None:
        ac = getattr(alg, "actor_critic", None)
    if ac is None:
        raise RuntimeError("Could not locate actor-critic on runner.alg")
    return ac


def _linear_indices(seq: nn.Sequential) -> List[int]:
    return [i for i, m in enumerate(seq) if isinstance(m, nn.Linear)]


def infer_live_units(
    out_lin: nn.Linear,
    *,
    keep: int | None,
    abs_threshold: float | None,
    rel: float,
) -> Tuple[torch.Tensor, Dict]:
    """Decide which input units of the output Linear to keep.

    The signal is each unit's strongest outgoing weight across the action
    outputs: ``colnorm[j] = max_a |W_out[a, j]|``. Dead units have colnorm ~0.

    Returns ``(live_idx_sorted, info)`` where info documents the decision and
    the natural "cliff" (largest log-gap in the sorted colnorms) for transparency.
    """
    W = out_lin.weight.detach().abs()           # (num_actions, H)
    colnorm = W.max(dim=0).values               # (H,)
    H = colnorm.numel()
    order = torch.argsort(colnorm, descending=True)
    sorted_vals = colnorm[order]

    # Natural cliff: largest ratio gap between consecutive sorted colnorms.
    eps = 1e-12
    ratios = (sorted_vals[:-1] + eps) / (sorted_vals[1:] + eps)
    cliff_pos = int(torch.argmax(ratios).item()) + 1 if H > 1 else H
    cliff_ratio = float(ratios.max().item()) if H > 1 else float("inf")

    if keep is not None:
        k = max(1, min(int(keep), H))
        live = order[:k]
        rule = f"--keep {keep} (top-{k} by colnorm)"
        tol = float(sorted_vals[k - 1].item())
    else:
        tol = float(abs_threshold) if abs_threshold is not None else rel * float(colnorm.max().item())
        live = torch.nonzero(colnorm > tol, as_tuple=False).flatten()
        if live.numel() == 0:                    # never prune everything
            live = order[:1]
        rule = (
            f"--prune-threshold {abs_threshold}" if abs_threshold is not None
            else f"--prune-rel {rel} (tol = {tol:.4g} = {rel}*max)"
        )

    live_sorted, _ = torch.sort(live)
    info = {
        "H": H,
        "kept": int(live_sorted.numel()),
        "pruned": H - int(live_sorted.numel()),
        "tol": tol,
        "rule": rule,
        "colnorm_sorted": [round(float(v), 4) for v in sorted_vals.tolist()],
        "cliff_k": cliff_pos,
        "cliff_ratio": cliff_ratio,
    }
    return live_sorted.to(out_lin.weight.device), info


def build_pruned_actor(seq: nn.Sequential, live_idx: torch.Tensor) -> nn.Sequential:
    """Return a structurally pruned copy of the actor Sequential.

    Slices the last two Linear layers to the ``live_idx`` bottleneck units,
    keeping the intervening (element-wise) activation untouched.
    """
    lins = _linear_indices(seq)
    if len(lins) < 2:
        raise RuntimeError("actor must have >=2 Linear layers to prune a bottleneck")
    bott_i, out_i = lins[-2], lins[-1]
    bott, out = seq[bott_i], seq[out_i]
    dev, dtype = out.weight.device, out.weight.dtype
    k = int(live_idx.numel())

    new_bott = nn.Linear(bott.in_features, k, bias=bott.bias is not None)
    new_bott.weight.data = bott.weight.data[live_idx].clone()
    if bott.bias is not None:
        new_bott.bias.data = bott.bias.data[live_idx].clone()

    new_out = nn.Linear(k, out.out_features, bias=out.bias is not None)
    new_out.weight.data = out.weight.data[:, live_idx].clone()
    if out.bias is not None:
        new_out.bias.data = out.bias.data.clone()

    new_seq = copy.deepcopy(seq)
    new_seq[bott_i] = new_bott
    new_seq[out_i] = new_out
    return new_seq.to(device=dev, dtype=dtype)


# ===========================================================================
#  Metrics (mirror eval.py main())
# ===========================================================================

def compute_metrics(run_out: Tuple, n_envs: int, win_frac: float = 0.05,
                    minimum: float = 250.0) -> Dict:
    (v_mean, COT, v_cmd, progress, final_reason,
     _traces, reward_total, mean_z) = run_out

    n_success = int((final_reason == 0).sum())
    n_obst = int((final_reason == 1).sum())
    n_walls = int((final_reason == 2).sum())
    n_angles = int((final_reason == 3).sum())

    eval_reward_mean = float(np.nanmean(reward_total)) if reward_total.size else float("nan")
    if not np.isfinite(eval_reward_mean):
        eval_reward_mean = 0.0

    # Smoothed peaks aligned on commanded speed (same as eval.py main()).
    top_vel = top_eff = top_prog = 0.0
    max_p = 0.0
    if v_cmd.size:
        _, p_s, _ = EvaluationPlotter.moving_avg(v_cmd, progress, win_frac)
        max_p = float(np.max(p_s)) if len(p_s) else 0.0
        if max_p > minimum:
            idxs = np.where(p_s >= minimum)[0]
            _, v_s, _ = EvaluationPlotter.moving_avg(v_cmd, v_mean, win_frac)
            _, E_s, _ = EvaluationPlotter.moving_avg(v_cmd, COT, win_frac)
            top_vel = float(v_s[idxs][int(np.argmax(v_s[idxs]))])
            top_eff = float(E_s[idxs][int(np.argmin(E_s[idxs]))])
            top_prog = float(p_s[idxs][int(np.argmax(p_s[idxs]))])

    return {
        "n_envs": n_envs,
        "valid": int(v_mean.size),
        "success": n_success,
        "obstacles": n_obst,
        "walls": n_walls,
        "angles": n_angles,
        "crash_rate": 1.0 - n_success / max(1, n_envs),
        # raw aggregates (always meaningful, even when <250 m progress)
        "mean_v": float(np.mean(v_mean)) if v_mean.size else 0.0,
        "mean_progress": float(np.mean(progress)) if progress.size else 0.0,
        "mean_COT": float(np.nanmean(COT)) if COT.size else float("nan"),
        "reward_mean": eval_reward_mean,
        "mean_z_250": float(mean_z),
        # smoothed peaks (eval.py main() headline numbers)
        "top_vel": top_vel,
        "top_eff_COT": top_eff,
        "top_prog": top_prog,
        "max_p": max_p,
    }


def format_report(orig: Dict, pruned: Dict, prune_info: Dict, num_actions: int,
                  meta: Dict) -> str:
    """Build the full human-readable comparison report as a string."""
    H, K = prune_info["H"], prune_info["kept"]
    bar = "=" * 78
    L: List[str] = []
    L.append(bar)
    L.append("  PRUNE-vs-ORIGINAL CONTROLLER EVALUATION")
    L.append(bar)
    L.append(f"  checkpoint : {meta['checkpoint']}")
    L.append(f"  config     : {meta['config']}")
    L.append(f"  urdf       : {meta['urdf']}")
    L.append(f"  envs={meta['envs']}  vmin={meta['vmin']}  vmax={meta['vmax']}  "
             f"seed={meta['seed']}  (same forests for both rollouts)")
    L.append("")
    L.append(bar)
    L.append("  PRUNE SUMMARY")
    L.append(bar)
    L.append(f"  output bottleneck (hidden_dim): {H}  ->  kept {K}  (pruned {prune_info['pruned']})")
    L.append(f"  rule: {prune_info['rule']}")
    L.append(f"  natural cliff: top-{prune_info['cliff_k']} units, "
             f"gap ratio x{prune_info['cliff_ratio']:.1f} to the next unit")
    L.append(f"  Hebbian genome (4 * {num_actions} * hidden): "
             f"{4*num_actions*H}  ->  {4*num_actions*K}")
    L.append(f"  output-unit colnorms (sorted): {prune_info['colnorm_sorted']}")

    rows = [
        ("SUCCESS",            "success",       "{:.0f}",  +1),
        ("OBSTACLES",          "obstacles",     "{:.0f}",  -1),
        ("WALLS",              "walls",         "{:.0f}",  -1),
        ("ANGLES",             "angles",        "{:.0f}",  -1),
        ("crash rate",         "crash_rate",    "{:.3f}",  -1),
        ("mean velocity [m/s]", "mean_v",       "{:.3f}",  +1),
        ("mean progress [m]",  "mean_progress", "{:.2f}",  +1),
        ("mean COT",           "mean_COT",      "{:.3f}",  -1),
        ("eval reward mean",   "reward_mean",   "{:.3f}",  +1),
        ("mean z @250 m [m]",  "mean_z_250",    "{:.2f}",   0),
        ("max velocity (smoothed)", "top_vel",  "{:.2f}",  +1),
        ("COT @best-eff (sm.)", "top_eff_COT",  "{:.2f}",  -1),
        ("progress @best (sm.)", "top_prog",    "{:.2f}",  +1),
    ]
    L.append("")
    L.append(bar)
    L.append(f"  ORIGINAL vs PRUNED   ({orig['n_envs']} envs, "
             f"{orig['valid']}/{pruned['valid']} valid)")
    L.append(bar)
    L.append(f"  {'metric':<26}{'ORIGINAL':>13}{'PRUNED':>13}{'Δ':>13}")
    L.append("  " + "-" * 65)
    for label, key, fmt, direction in rows:
        a, b = orig[key], pruned[key]
        d = b - a
        flag = ""
        if direction != 0 and abs(d) > 1e-9:
            improved = (d > 0) if direction > 0 else (d < 0)
            flag = "  ✓" if improved else "  ✗"
        L.append(f"  {label:<26}{fmt.format(a):>13}{fmt.format(b):>13}"
                 f"{fmt.format(d):>13}{flag}")
    L.append(bar)
    L.append("  (✓ = pruned better/equal-direction, ✗ = worse; relative to the metric's "
             "preferred direction)")
    return "\n".join(L)


def _save_arrays(out_dir: Path, tag: str, run_out: Tuple) -> None:
    (v_mean, COT, v_cmd, progress, final_reason, _traces, reward_total, _z) = run_out
    np.savez(
        out_dir / f"arrays_{tag}.npz",
        v_mean=v_mean, COT=COT, v_cmd=v_cmd, progress=progress,
        final_reason=final_reason, reward_total=reward_total,
    )


def _save_total_plot(out_dir: Path, tag: str, run_out: Tuple,
                     command_speed_range, minimal_progress: float) -> None:
    """eval.py-style velocity / energy / progress vs commanded-speed plot."""
    (v_mean, COT, v_cmd, _progress, _fr, _tr, _rew, _z) = run_out
    progress = run_out[3]
    try:
        EvaluationPlotter().total_plot(
            v_mean, COT, v_cmd, progress,
            win_frac=0.05, minimal_p=minimal_progress,
            out=str(out_dir / f"total_plot_{tag}.png"),
            velocity_range=command_speed_range,
        )
    except Exception as exc:  # plotting must never sink a completed eval
        print(f"[prune_eval][warn] total_plot ({tag}) failed: {exc}")


# ===========================================================================
#  Main
# ===========================================================================

def resolve_inputs(args) -> Tuple[str, str]:
    """Return (checkpoint_path, wp1_config_path) from --run or explicit flags."""
    if args.run:
        repro = Path(args.run).expanduser() / "reproducibility"
        ckpt = repro / "wp1_actor.pt"
        cfg = repro / "wp1_config.yaml"
        if not ckpt.is_file() or not cfg.is_file():
            raise FileNotFoundError(
                f"--run given but {ckpt} or {cfg} missing. Pass --checkpoint/--config instead."
            )
        return str(ckpt), str(cfg)
    if not args.checkpoint or not args.config:
        raise SystemExit("Provide either --run <wp2_run_dir> or both --checkpoint and --config")
    return args.checkpoint, args.config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None,
                   help="WP2 run dir; uses reproducibility/wp1_actor.pt + wp1_config.yaml")
    p.add_argument("--checkpoint", default=None, help="path to frozen WP1 actor .pt")
    p.add_argument("--config", default=None, help="path to WP1 config.yaml")
    p.add_argument("--envs", type=int, default=8192, help="number of eval envs (default 8192)")
    p.add_argument("--vmin", type=float, default=None, help="min commanded speed (default: config)")
    p.add_argument("--vmax", type=float, default=None, help="max commanded speed (default: config)")
    p.add_argument("--urdf", default=None, help="URDF to evaluate on (default: standard drone)")
    p.add_argument("--drone", default=None, help="drone key override")
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (identical for both rollouts)")
    p.add_argument("--out-dir", default=None,
                   help="output folder (default: ./prune_eval_<run_stem> in the cwd)")
    p.add_argument("--minimal-progress", type=float, default=250.0)
    p.add_argument("--dens-min", type=float, default=None)
    p.add_argument("--dens-max", type=float, default=None)
    # prune control
    p.add_argument("--prune-rel", type=float, default=0.02,
                   help="keep units with colnorm > rel*max(colnorm) [default 0.02]")
    p.add_argument("--prune-threshold", type=float, default=None,
                   help="absolute colnorm threshold (overrides --prune-rel)")
    p.add_argument("--keep", type=int, default=None,
                   help="force-keep top-K units by colnorm (overrides thresholds)")
    args = p.parse_args()

    ckpt_path, cfg_path = resolve_inputs(args)
    device = args.gpu if torch.cuda.is_available() else "cpu"

    # Output folder (created in the cwd / repo root by default).
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser()
    else:
        stem = Path(args.run).name if args.run else Path(ckpt_path).stem
        out_dir = Path.cwd() / f"prune_eval_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prune_eval] output folder = {out_dir}")

    # ---------------- load config (WP1) ----------------
    from WP1.config import RunConfig
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = \
        RunConfig.from_yaml(cfg_path).to_legacy_cfgs()

    if not gs._initialized:
        gs.init(logging_level="error", backend=gs.gpu)

    selected_drone = args.drone or env_cfg.get("drone")
    urdf_file = resolve_or_generate_urdf(urdf_file=args.urdf, drone_key=selected_drone)
    if args.drone:
        env_cfg["drone"] = args.drone
    env_cfg = _apply_drone_profile_defaults(env_cfg, urdf_file)

    # command range: CLI overrides, else the training range from the config
    vmin = args.vmin if args.vmin is not None else float(command_cfg.get("min_speed", 10.0))
    vmax = args.vmax if args.vmax is not None else float(command_cfg.get("max_speed", 20.0))
    command_cfg["min_speed"], command_cfg["max_speed"] = vmin, vmax
    command_speed_range = _command_speed_range(command_cfg)

    # eval-mode env tweaks (mirror eval.py main(): no joint-target noise)
    _apply_eval_env_overrides(env_cfg)
    env_cfg.setdefault("property_randomization", {})
    env_cfg["property_randomization"]["joint_target_episode_bias_std"] = 0.0
    env_cfg["property_randomization"]["joint_target_step_noise_std"] = 0.0
    if args.dens_min is not None:
        env_cfg["dens_min"] = float(args.dens_min)
    if args.dens_max is not None:
        env_cfg["dens_max"] = float(args.dens_max)

    print(f"[prune_eval] checkpoint = {ckpt_path}")
    print(f"[prune_eval] config     = {cfg_path}")
    print(f"[prune_eval] urdf       = {Path(urdf_file).name}")
    print(f"[prune_eval] envs={args.envs}  vmin={vmin}  vmax={vmax}  seed={args.seed}  device={device}")

    # ---------------- build env once ----------------
    _reseed(args.seed)
    env = WingedDroneEnv(
        num_envs=args.envs,
        env_cfg=env_cfg,
        obs_cfg=dict(obs_cfg),
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        urdf_file=urdf_file,
        show_viewer=False,
        eval=True,
        device=device,
    )
    env.aero_solver._aero_log = False

    # ---------------- load policy ----------------
    eval_log_dir = str(Path(ckpt_path).resolve().parent)
    runner_cfg = copy.deepcopy(train_cfg)
    runner = OnPolicyRunner(env, runner_cfg, eval_log_dir, device=gs.device)
    runner.load(ckpt_path)

    ac = _get_actor_critic(runner)
    actor_seq = ac.actor
    lins = _linear_indices(actor_seq)
    out_lin = actor_seq[lins[-1]]
    num_actions = out_lin.out_features
    print(f"[prune_eval] actor MLP Linear shapes: "
          f"{[tuple(actor_seq[i].weight.shape) for i in lins]}  (num_actions={num_actions})")

    # =====================================================================
    #  Rollout 1 — ORIGINAL
    # =====================================================================
    print("\n[prune_eval] === ORIGINAL rollout ===")
    _reseed(args.seed)
    policy = runner.get_inference_policy(device=gs.device)
    orig_out = run_eval(env, policy, extra_data=False, minimal_progress=args.minimal_progress)
    orig = compute_metrics(orig_out, args.envs, minimum=args.minimal_progress)

    # =====================================================================
    #  Prune, then Rollout 2 — PRUNED
    # =====================================================================
    live_idx, prune_info = infer_live_units(
        out_lin, keep=args.keep, abs_threshold=args.prune_threshold, rel=args.prune_rel,
    )
    print(f"\n[prune_eval] inferred prune: hidden {prune_info['H']} -> {prune_info['kept']} "
          f"(natural cliff at top-{prune_info['cliff_k']}, x{prune_info['cliff_ratio']:.1f})")
    ac.actor = build_pruned_actor(actor_seq, live_idx)

    print("\n[prune_eval] === PRUNED rollout ===")
    _reseed(args.seed)
    policy_pruned = runner.get_inference_policy(device=gs.device)
    pruned_out = run_eval(env, policy_pruned, extra_data=False, minimal_progress=args.minimal_progress)
    pruned = compute_metrics(pruned_out, args.envs, minimum=args.minimal_progress)

    # total-speed-sweep plots need the env (nominal_mass etc. already baked into
    # the returned arrays), so generate them before destroying the sim.
    _save_total_plot(out_dir, "original", orig_out, command_speed_range, args.minimal_progress)
    _save_total_plot(out_dir, "pruned", pruned_out, command_speed_range, args.minimal_progress)

    gs.destroy()

    # ---------------- report + artefacts ----------------
    meta = {
        "checkpoint": ckpt_path, "config": cfg_path, "urdf": Path(urdf_file).name,
        "envs": args.envs, "vmin": vmin, "vmax": vmax, "seed": args.seed,
        "num_actions": num_actions,
    }
    report = format_report(orig, pruned, prune_info, num_actions, meta)
    print("\n" + report + "\n")

    (out_dir / "comparison.txt").write_text(report + "\n")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"meta": meta, "prune": prune_info,
                   "original": orig, "pruned": pruned}, f, indent=2)
    _save_arrays(out_dir, "original", orig_out)
    _save_arrays(out_dir, "pruned", pruned_out)
    print(f"[prune_eval] wrote: comparison.txt, metrics.json, arrays_*.npz, "
          f"total_plot_*.png  ->  {out_dir}")


if __name__ == "__main__":
    main()
