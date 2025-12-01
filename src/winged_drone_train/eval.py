#!/usr/bin/env python3
# hover_eval_lin_speed.py
# ----------------------------------------------------------------------
# 1. Run evaluation with N environments at linearly spaced commanded speeds
# 2. Collect per-env statistics: mean speed, energy per meter, progress
# 3. Produce summary plots via EvaluationPlotter
# ----------------------------------------------------------------------

from __future__ import annotations

import os
os.environ["GS_PARA_LEVEL"] = "2"
import copy
import math
import pickle
import argparse
import re
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

from winged_drone_train.utils.A2C_modified import ActorCriticTanh
import builtins

builtins.ActorCriticTanh = ActorCriticTanh  # for model loading

import genesis as gs
from winged_drone_train.env import WingedDroneEnv
from rsl_rl.runners import OnPolicyRunner
from tensorboard.backend.event_processing import event_accumulator

from winged_drone_train.utils.eval_plotter import EvaluationPlotter

# ---------------------------------------------------------------------- #
#  Helpers                                                              #
# ---------------------------------------------------------------------- #


def safe_urdf_stem(urdf_file: str | Path, *, already_clean: bool = False) -> str:
    """
    Sanitize a URDF filename for filesystem-safe folder names.

    Replaces brackets/commas/spaces with underscores and collapses
    repeated separators.
    """
    if isinstance(urdf_file, Path):
        stem = urdf_file.stem
    else:
        s = str(urdf_file)
        if already_clean:
            stem = s
        elif os.sep in s or s.endswith(".urdf"):
            stem = Path(s).stem
        else:
            stem = s
    clean = re.sub(r"[\\[\\],\\s]+", "_", stem)
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "urdf"


def _write_placeholder_png(path: Path, reason: str) -> None:
    """Create a minimal placeholder PNG so downstream copy steps never fail silently."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.text(
        0.5,
        0.5,
        f"No data\n{reason}",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"[evaluation][placeholder] wrote {path} ({reason})")


def _configure_cache_root() -> Path:
    """
    Force Taichi/genesis cache into a writable location to avoid ROFS errors.
    """
    cache_root = (Path("logs") / ".cache" / "gstaichi").expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    for env_key in ("XDG_CACHE_HOME", "TI_CACHE_DIR", "TAICHI_CACHE_DIR", "GSTAICHI_CACHE_DIR"):
        os.environ[env_key] = str(cache_root)
    mpl_dir = cache_root / "mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    print(f"[evaluation] cache dir set to {cache_root}")
    return cache_root


# ---------------------------------------------------------------------- #
# 1) Roll-out that returns per-env statistics                            #
# ---------------------------------------------------------------------- #
@torch.no_grad()
def run_eval(env, policy, extra_data: bool = False):
    """
    Lightweight rollout for evaluation.

    Returns
    -------
    v_mean : np.ndarray, shape (N_valid,)
        Mean forward speed (dx / t) for all non-NaN environments.
    E_tot : np.ndarray, shape (N_valid,)
        Total energy per meter for all non-NaN environments.
    v_cmd : np.ndarray, shape (N_valid,)
        Commanded speed of each valid environment.
    progress : np.ndarray, shape (N_envs,)
        Total distance covered along +X for all environments (NaN envs included).
    final_reason : np.ndarray, shape (N_envs,)
        Integer code for termination reason:
        0=success, 1=obstacle, 2=wall, 3=angle, 4=other.
    traces_all : dict
        Per-environment time series, used for plotting:
          - "s": list of 1D arrays of distance.
          - "j_pos": list of 2D arrays (T_i, num_joints).
          - "v_cmd": array of commanded velocities, one per env.
    """
    B, dt, dev = env.num_envs, env.dt, env.device

    done = torch.zeros(B, dtype=torch.bool, device=dev)
    t_acc = torch.zeros(B, device=dev)
    dx_acc = torch.zeros(B, device=dev)
    E_acc = torch.zeros(B, device=dev)
    final_reason = torch.full((B,), 3, dtype=torch.int8, device=dev)

    # reset env and reference x position
    obs, _ = env.reset()
    x0 = env.base_pos[:, 0].clone()

    # traces for ALL envs (minimal: distance + joint positions + v_cmd)
    traces_all = {
        "s": [[] for _ in range(B)],
        "j_pos": [[] for _ in range(B)],
        "v_cmd": env.commands[:, 0].detach().cpu().numpy(),
    }

    while not done.all():
        actions = policy(obs)
        obs, _, term, _ = env.step(actions)
        term = term.bool()
        nan_indices = env.nan_envs.to(torch.bool)  # envs with NaNs

        # only update stats for envs that are still "alive"
        alive = (~done) & (~term) & (~nan_indices)

        if alive.any():
            # time, distance, energy
            t_acc[alive] += dt
            dx_acc[alive] = env.base_pos[alive, 0] - x0[alive]
            P = env.power
            E_acc[alive] += P[alive] * dt

            # store traces needed for heatmaps (distance + joint positions)
            base_x = env.base_pos[:, 0] - x0  # Δx for all envs
            jp = env.joint_position.detach().cpu()  # (B, num_joints)

            alive_ids = alive.nonzero(as_tuple=False).flatten()
            for idx in alive_ids.tolist():
                s_val = base_x[idx].item()
                if extra_data:
                    traces_all["s"][idx].append(s_val)
                    traces_all["j_pos"][idx].append(jp[idx].numpy())

        # update termination reasons
        just_done = (~done) & term
        for j in just_done.nonzero(as_tuple=False).flatten().tolist():
            if getattr(env, "pre_collision", None) is not None and env.pre_collision[j]:
                final_reason[j] = 1
            elif getattr(env, "pre_wall_crash", None) is not None and env.pre_wall_crash[j]:
                final_reason[j] = 2
            elif getattr(env, "pre_angle_limit", None) is not None and env.pre_angle_limit[j]:
                final_reason[j] = 3
            else:
                final_reason[j] = 4

        done |= term | nan_indices

    # ---- global metrics (drop NaN envs for v_mean / E_tot / v_cmd) -------
    nan_indices = env.nan_envs.to(torch.bool)

    v_mean = (dx_acc[~nan_indices] / t_acc[~nan_indices].clamp_min(1e-6)).cpu().numpy()
    E_tot = (E_acc[~nan_indices] / dx_acc[~nan_indices].clamp_min(1e-6)).cpu().numpy()
    mg = env.nominal_mass * 9.81
    COT   = E_tot / mg
    v_cmd = env.commands[~nan_indices, 0].detach().cpu().numpy()
    progress = dx_acc.cpu().numpy()
    final_reason = final_reason.cpu().numpy()

    # ---- compact traces into numpy arrays ---------------------------------
    num_joints = env.joint_position.shape[1]
    if extra_data:
        for i in range(B):
            s_list = traces_all["s"][i]
            jp_list = traces_all["j_pos"][i]

            traces_all["s"][i] = np.asarray(s_list, dtype=float)
            if jp_list:
                traces_all["j_pos"][i] = np.vstack(jp_list)
            else:
                traces_all["j_pos"][i] = np.empty((0, num_joints), dtype=float)

    return v_mean, COT, v_cmd, progress, final_reason, traces_all


# ---------------------------------------------------------------------- #
# 2) Programmatic evaluation function (used by external scripts)         #
# ---------------------------------------------------------------------- #
def evaluation(
    exp_name: str,
    urdf_file: str,
    ckpt: int,
    envs: int,
    vmin: float,
    vmax: float,
    win_frac: float = 0.03,
    return_arrays: bool = False,
    custom_policy_path: str | None = None,
    obs_genome: bool | None = False,
    save_plots: bool = True,
    eval_dir: str | Path | None = None,
):
    """
    Programmatic evaluation entry point.

    Loads a trained policy from logs/ea/<exp_name>, runs evaluation on a
    range of commanded speeds, and returns dictionaries describing the
    best operating points:

        - top_vel:  max mean velocity (within the evaluated range)
        - top_eff:  min energy per meter
        - top_prog: max progress

    Additionally, it computes:
        - max_p:        maximum of the smoothed progress curve
        - final_reward: average final training reward (from TensorBoard)
        - steps90_pct:  percentage of training steps needed to reach
                        ~90% of the final reward (smoothed).

    If `return_arrays` is True, the `extra` dict additionally contains
    raw arrays:
        "p_s": MA(progress), "v_s": v_cmd, "E_s": E_tot
    """
    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    _configure_cache_root()
    gs.init(logging_level="error", backend=gs.gpu)

    log_dir = (Path("logs") / "ea" / exp_name).expanduser().resolve()
    log_dir_str = str(log_dir)
    cfg_path = log_dir / "cfgs.pkl"
    urdf_path = Path(urdf_file).expanduser()
    clean_stem = safe_urdf_stem(urdf_path)

    print(
        f"[evaluation] exp={exp_name} ckpt={ckpt} urdf={urdf_path.name} "
        f"clean={clean_stem} envs={envs} save_plots={save_plots} eval_dir={eval_dir}"
    )

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing cfgs.pkl at {cfg_path}")

    with cfg_path.open("rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    # Command range used in this evaluation
    command_cfg["min_speed"] = float(vmin)
    command_cfg["max_speed"] = float(vmax)

    # Disable observation noise during evaluation
    obs_cfg_eval = dict(obs_cfg)
    obs_cfg_eval["add_noise"] = False
    if obs_genome is not None:
        obs_cfg_eval["add_genome_obs"] = bool(obs_genome)

    # Evaluation-specific environment tweaks
    env_cfg.update(
        dict(
            visualize_camera=False,
            visualize_target=False,
            max_visualize_FPS=15,
            unique_forests_eval=True,
            growing_forest=True,
            x_upper=1000,
            tree_radius=0.75,
        )
    )

    env = WingedDroneEnv(
        num_envs=envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg_eval,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        urdf_file=urdf_file,
        show_viewer=False,
        eval=True,
        device=device,
    )

    runner_cfg = copy.deepcopy(train_cfg)
    runner = OnPolicyRunner(env, runner_cfg, log_dir_str, device=gs.device)

    if custom_policy_path is not None:
        print(f"[evaluation] Using custom policy {custom_policy_path}")
        runner.load(custom_policy_path)
    else:
        runner.load(os.path.join(log_dir, f"model_{ckpt-1}.pt"))
        
    policy = runner.get_inference_policy(device=gs.device)

    env.aero_solver._aero_log = False

    # Use traces when saving plots to enable heatmaps.
    v_mean, COT, v_cmd, progress, final_reason, traces_all = run_eval(
        env,
        policy,
        extra_data=bool(save_plots),
    )
    print(
        f"[evaluation] rollout done | v_mean={len(v_mean)} v_cmd={len(v_cmd)} "
        f"progress_shape={np.shape(progress)} extra_data={bool(save_plots)}"
    )

    gs.destroy()

    # ---------------- TensorBoard training metrics --------------------- #
    final_reward = 0.0
    steps90_pct = 0.0

    final_reward = 0.0
    steps90_pct = 0.0
    try:
        ea = event_accumulator.EventAccumulator(log_dir_str)
        ea.Reload()
        scalar_tags = ea.Tags().get("scalars", [])
        reward_tags = [tag for tag in scalar_tags if tag.startswith("rew_")]
    except Exception as exc:
        print(f"[evaluation][warn] skipping TensorBoard metrics: {exc}")
        reward_tags = []

    if reward_tags:
        total_steps = ckpt
        final_window = max(1, math.ceil(0.05 * total_steps))

        max_step = max(e.step for e in ea.Scalars(reward_tags[0]))
        start_step = max_step - final_window + 1

        total_rewards = []
        for step in range(start_step, max_step + 1):
            values = [
                e.value
                for tag in reward_tags
                for e in ea.Scalars(tag)
                if e.step == step
            ]
            if values:
                total_rewards.append(sum(values))
        if total_rewards:
            final_reward = sum(total_rewards) / float(len(total_rewards))

        # Build series of total reward per step
        steps_count = max_step + 1
        reward_series = [0.0] * steps_count
        for tag in reward_tags:
            for event in ea.Scalars(tag):
                reward_series[event.step] += event.value

        window = max(1, math.ceil(0.03 * total_steps))
        smooth = [0.0] * steps_count
        cum_sum = 0.0
        for i in range(steps_count):
            cum_sum += reward_series[i]
            if i < window:
                smooth[i] = cum_sum / float(i + 1)
            else:
                cum_sum -= reward_series[i - window]
                smooth[i] = cum_sum / float(window)

        threshold = 0.9 * final_reward
        steps_to_90 = 0
        for i, val in enumerate(smooth):
            if val >= threshold:
                steps_to_90 = i + 1
                break

        if total_steps > 0:
            steps90_pct = (steps_to_90 / float(total_steps)) * 100.0

    # ---------------- Peak extraction from evaluation ------------------ #
    v_cmd_m, v_mean_m = v_cmd, v_mean
    E_tot_m, prog_m = COT, progress

    # Smoothed curves over mean velocity
    x_s, p_s, _ = EvaluationPlotter.moving_avg(v_mean_m, prog_m, win_frac)
    idx_p = int(np.argmax(p_s)) if len(p_s) else 0
    max_p = float(p_s[idx_p]) if len(p_s) else 0.0

    idxs = np.arange(len(p_s))

    x_cv, v_s, _ = EvaluationPlotter.moving_avg(v_cmd_m, v_mean_m, win_frac)
    x_e, E_s, _ = EvaluationPlotter.moving_avg(v_mean_m, E_tot_m, win_frac)

    prog_sel = p_s[idxs]
    vel_sel = v_s[idxs]
    energy_sel = E_s[idxs]

    idx_p = int(np.argmax(prog_sel)) if len(prog_sel) else 0
    idx_v = int(np.argmax(vel_sel)) if len(vel_sel) else 0
    idx_e = int(np.argmin(energy_sel)) if len(energy_sel) else 0

    top_vel = {
        "mean_v": float(vel_sel[idx_v]),
        "mean_E": float(energy_sel[idx_v]),
        "mean_progress": float(prog_sel[idx_v]),
    }
    top_eff = {
        "mean_v": float(vel_sel[idx_e]),
        "mean_E": float(energy_sel[idx_e]),
        "mean_progress": float(prog_sel[idx_e]),
    }
    top_prog = {
        "mean_v": float(vel_sel[idx_p]),
        "mean_E": float(energy_sel[idx_p]),
        "mean_progress": float(prog_sel[idx_p]),
    }

    eval_dir_path = None
    plot_paths: Dict[str, str] = {}
    if save_plots:
        print(f"[evaluation] save_plots block starting for {clean_stem}")
        eval_dir_path = Path(eval_dir) if eval_dir is not None else log_dir / f"eval_{clean_stem}"
        eval_dir_path.mkdir(parents=True, exist_ok=True)
        print(
            f"[evaluation] Saving plots for URDF '{urdf_path.name}' "
            f"(clean='{clean_stem}') in: {eval_dir_path} | save_plots={save_plots}"
        )
        plotter = EvaluationPlotter()

        sweep_out = eval_dir_path / "joint_heatmap_sweep.png"
        twist_out = eval_dir_path / "joint_heatmap_twist.png"
        total_out = eval_dir_path / "total_plot.png"

        # Pre-create placeholders so downstream copy always finds something
        for p in (sweep_out, twist_out, total_out):
            if not p.exists():
                _write_placeholder_png(p, "pre-plot placeholder")

        try:
            plotter.plot_joint_diff_heatmap(traces_all, "sweep", out=str(sweep_out))
            print(f"[evaluation] joint_heatmap_sweep → {sweep_out}")
        except Exception as exc:
            print(f"[evaluation][error] joint_heatmap_sweep failed: {exc}")
            _write_placeholder_png(sweep_out, "heatmap_sweep failed")

        try:
            plotter.plot_joint_diff_heatmap(traces_all, "twist", out=str(twist_out))
            print(f"[evaluation] joint_heatmap_twist → {twist_out}")
        except Exception as exc:
            print(f"[evaluation][error] joint_heatmap_twist failed: {exc}")
            _write_placeholder_png(twist_out, "heatmap_twist failed")

        try:
            plotter.total_plot(
                v_mean,
                COT,
                v_cmd,
                progress,
                win_frac=win_frac,
                minimal_p=300.0,
                out=str(total_out),
            )
            print(f"[evaluation] total_plot → {total_out}")
        except Exception as exc:
            print(f"[evaluation][error] total_plot failed: {exc}")
            _write_placeholder_png(total_out, "total_plot failed")

        # Guarantee files exist
        for p in (sweep_out, twist_out, total_out):
            if not p.is_file():
                _write_placeholder_png(p, "missing after plotting")
            else:
                print(f"[evaluation] confirmed plot exists: {p}")

        try:
            contents = sorted([str(p.name) for p in eval_dir_path.iterdir()])
            print(f"[evaluation] eval_dir contents: {eval_dir_path} -> {contents}")
        except Exception as exc:
            print(f"[evaluation][warn] could not list {eval_dir_path}: {exc}")

        plot_paths = {
            "total_plot": str(total_out),
            "joint_heatmap_sweep": str(sweep_out),
            "joint_heatmap_twist": str(twist_out),
        }

    extra = {
        "max_p": max_p,
        "final_reward": final_reward,
        "steps90_pct": steps90_pct,
        "eval_dir": str(eval_dir_path) if eval_dir_path else "",
        "clean_urdf_stem": clean_stem,
        "plot_paths": plot_paths,
    }
    if return_arrays:
        # For compatibility with previous behaviour, we keep these
        # as the raw arrays used inside the evaluation.
        extra.update(
            {
                "p_s": p_s,
                "v_s": v_cmd_m,
                "E_s": E_tot_m,
            }
        )

    if return_arrays:
        return top_vel, top_eff, top_prog, max_p, extra
    else:
        return top_vel, top_eff, top_prog, max_p


# ---------------------------------------------------------------------- #
# 3) CLI entry point                                                    #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", default="drone-forest")
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--vmin", type=float, default=6.0)
    parser.add_argument("--vmax", type=float, default=24.0)
    parser.add_argument("--gpu", default="cuda")
    args = parser.parse_args()

    # ---------------- Load configs ------------------------------------- #
    log_dir = f"logs/{args.exp_name}"
    # Overwrite log_dir if needed coming from cluster
    #log_dir = f"/home/andrea/tb_logs_kuma/ea/{args.exp_name}"
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    eval_log_dir = os.path.join("logs", f"{args.exp_name}_eval")
    os.makedirs(eval_log_dir, exist_ok=True)

    gs.init(logging_level="error")

    # NOTE: same hard-coded URDF path as in the original script
    urdf_file = (
        "/home/andrea/Documents/Genesis/genesis/assets/urdf/mydrone/"
        "[0.7, 3.5, 0.73, 0.38, 0.38, 0.18, 1.3, 0.16, 1.3, 0, 0.25, "
        "2, 2.5, 2, -3].urdf"
    )
    #urdf_file = "/home/andrea/Documents/Genesis/src/urdf_generated/[0.497691, 1.88631, 0.646899, 0.327637, 0.339316, 0.223745, 2.64199, 0.10971, 2.67589, 0, 0.25, 2.4373, 3.45352, 2, -1.30368].urdf"
    #urdf_file = "/home/andrea/Documents/Genesis/src/urdf_generated/[0.651191, 2.23634, 0.488678, 0.363086, 0.372742, 0.264039, 1.8772, 0.198837, 1.20409, -2.91123, 0.25, 2.80622, 2.00658, 2, -3.77787].urdf"
    urdf_file = "/home/andrea/Documents/Genesis/src/urdf_generated/[0.5, 2, 0.5, 0.35, 0.35, 0.15, 1, 0.1, 1, 0, 0.25, 3, 3, 2, -2].urdf"

    command_cfg["min_speed"] = args.vmin
    command_cfg["max_speed"] = args.vmax

    # Disable observation noise during evaluation
    obs_cfg_eval = dict(obs_cfg)
    obs_cfg_eval["add_genome_obs"] = True

    # Print configs for sanity check
    print("\nEnvironment Configuration (eval):")
    print(env_cfg)
    print("\nObservation Configuration (eval):")
    print(obs_cfg_eval)
    print("\nReward Configuration:")
    print(reward_cfg)
    print("\nCommand Configuration:")
    print(command_cfg)

    # Evaluation-specific environment settings
    env_cfg.update(
        dict(
            visualize_camera=False,
            visualize_target=False,
            max_visualize_FPS=25,
            unique_forests_eval=True,
            growing_forest=True,
            x_upper=500,
            forest_x_limit=500,
            tree_radius=0.75,
            base_init_pos=[-50.0, 0.0, 10.0],
        )
    )

    env = WingedDroneEnv(
        num_envs=args.envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg_eval,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        urdf_file=urdf_file,
        show_viewer=False,
        eval=True,
        device=args.gpu,
    )

    # Small amount of solver noise, as in the original script
    env.aero_solver.noise_sigma_mag = 0.03
    env.aero_solver.noise_sigma_dir = 0.03
    env.aero_solver.noise_sigma_param = 0.0

    plotter = EvaluationPlotter()
    plotter.plot_forest(env)

    runner_cfg = copy.deepcopy(train_cfg)
    runner = OnPolicyRunner(env, runner_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # Commanded speeds over all envs (not used for watched envs anymore,
    # but kept if you want to select subsets later).
    v_cmd_all = np.linspace(args.vmin, args.vmax, args.envs)

    # Single evaluation rollout
    v_mean, COT, v_cmd, progress, final_reason, traces_all = run_eval(env, policy, extra_data=True)

    # Reason counts
    n_success = int((final_reason == 0).sum())
    n_obst = int((final_reason == 1).sum())
    n_walls = int((final_reason == 2).sum())
    n_angles = int((final_reason == 3).sum())
    print(
        f"\n►  SUCCESS: {n_success}/{args.envs}   |  "
        f"OBSTACLES: {n_obst}   |  WALLS: {n_walls}   |  ANGLES: {n_angles}"
    )

    # Smoothed peak extraction (same logic as original main)
    v_cmd_m, v_mean_m = v_cmd, v_mean
    E_tot_m, prog_m = COT, progress

    win_frac = 0.05
    x_s, p_s, _ = EvaluationPlotter.moving_avg(v_cmd_m, prog_m, win_frac)
    idx_p = int(np.argmax(p_s)) if len(p_s) else 0
    max_p = float(p_s[idx_p]) if len(p_s) else 0.0
    minimum = 300

    if max_p > minimum:
        idxs = np.where(p_s >= minimum)[0]

        x_cv, v_s, _ = EvaluationPlotter.moving_avg(v_cmd_m, v_mean_m, win_frac)
        x_e, E_s, _ = EvaluationPlotter.moving_avg(v_cmd_m, E_tot_m, win_frac)

        prog_slice = p_s[idxs]
        vel_slice = v_s[idxs]
        energy_slice = E_s[idxs]

        idx_p = int(np.argmax(prog_slice)) if len(prog_slice) else 0
        idx_v = int(np.argmax(vel_slice)) if len(vel_slice) else 0
        idx_e = int(np.argmin(energy_slice)) if len(energy_slice) else 0

        top_vel = {
            "mean_v": float(vel_slice[idx_v]),
            "mean_E": float(energy_slice[idx_v]),
            "mean_progress": float(prog_slice[idx_v]),
        }
        top_eff = {
            "mean_v": float(vel_slice[idx_e]),
            "mean_E": float(energy_slice[idx_e]),
            "mean_progress": float(prog_slice[idx_e]),
        }
        top_prog = {
            "mean_v": float(vel_slice[idx_p]),
            "mean_E": float(energy_slice[idx_p]),
            "mean_progress": float(prog_slice[idx_p]),
        }
    else:
        # Degenerate case: no progress; keep defaults
        top_vel = {"mean_v": 0.0, "mean_E": 0.0, "mean_progress": 0.0}
        top_eff = dict(top_vel)
        top_prog = dict(top_vel)
        idx_v = idx_e = idx_p = 0

    print(
        "\n►  Max velocity: "
        f"{top_vel['mean_v']:.2f} m/s   |  "
        f"COT: {top_eff['mean_E']:.2f} J/Nm   |  "
        f"Progress: {top_prog['mean_progress']:.2f} m"
    )

    # Plots in eval log dir
    plotter.plot_joint_diff_heatmap(
        traces_all,
        "sweep",
        out=f"{eval_log_dir}/{args.exp_name}_{args.ckpt}_joint_behaviour_heatmap_sweep.png",
    )
    plotter.plot_joint_diff_heatmap(
        traces_all,
        "twist",
        out=f"{eval_log_dir}/{args.exp_name}_{args.ckpt}_joint_behaviour_heatmap_twist.png",
    )

    EvaluationPlotter.plot_3d_speed_energy_agility(
        v_mean,
        COT,
        progress,
        v_cmd,
        html_out=f"{eval_log_dir}/{args.exp_name}_{args.ckpt}_3D_speed_energy_progress.html",
    )

    EvaluationPlotter.plot_3d_speed_energy_progress_ma(
        v_mean,
        COT,
        progress,
        v_cmd,
        win_frac=win_frac,
        html_out=f"{eval_log_dir}/{args.exp_name}_{args.ckpt}_3D_speed_energy_progress_MA.html",
    )

    plotter.total_plot(
        v_mean,
        COT,
        v_cmd,
        progress,
        win_frac=win_frac,
        minimal_p=300.0,  # same value as original script
        out=f"{eval_log_dir}/{args.exp_name}_{args.ckpt}_total_plot.png",
    )

    plotter.total_plot_points_instead_of_ma(
        v_mean,
        COT,
        v_cmd,
        progress,
        win_frac=win_frac,
        minimal_p=300.0,
        out=f"{eval_log_dir}/{args.exp_name}_{args.ckpt}_total_plot_points_instead_of_ma.png",
    )
