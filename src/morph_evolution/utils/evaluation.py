from __future__ import annotations

import gc
import os
import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

from morph_evolution.chromosome_drone import Chromosome_Drone
from morph_evolution.utils.reporting import ckpt_idx_from_train_it, parse_fail_reason
from morph_evolution.utils.runtime import (
    create_urdf_with_retry,
    log_mem,
    log_worker_context,
    prepare_device_env,
    pushd,
    resolve_urdf_dir,
    set_thread_envs,
)
from winged_drone_train.eval import evaluation
from winged_drone_train.train import training


def build_run_meta(
    *,
    exp_name: str,
    train_it: int,
    max_p: float,
    eval_reward_mean: float,
    train_repetition: Optional[int] = None,
    rep_exp_names: Optional[str] = None,
    failed: bool = False,
    fail_reason: str = "",
) -> Dict[str, Any]:
    """Build common metadata fields for train/eval bookkeeping."""
    train_it_i = int(train_it)
    meta: Dict[str, Any] = dict(
        exp_name=exp_name,
        train_it=train_it_i,
        ckpt_idx=ckpt_idx_from_train_it(train_it_i),
        max_p=max_p,
        eval_reward_mean=eval_reward_mean,
    )
    if train_repetition is not None:
        meta["train_repetition"] = int(train_repetition)
    if rep_exp_names is not None:
        meta["rep_exp_names"] = rep_exp_names
    if failed:
        meta["failed"] = True
        meta["fail_reason"] = fail_reason
    return meta


def default_fitness(invalid_v: set[float], invalid_e: set[float], invalid_p: set[float]) -> List[float]:
    """Return the sentinel fitness values defined by INVALID_* globals."""
    return [
        float(min(invalid_v)),
        float(min(invalid_e)),
        float(min(invalid_p)),
    ]


def failure_result(
    reason: str,
    cfg: Dict[str, Any],
    invalid_v: set[float],
    invalid_e: set[float],
    invalid_p: set[float],
    exp_name: Optional[str] = None,
    train_it: Optional[int] = None,
) -> Tuple[List[float], Dict[str, Any], Dict[str, np.ndarray]]:
    """Build a safe fallback result for failed train/eval steps."""
    ff = default_fitness(invalid_v, invalid_e, invalid_p)
    fail_category, fail_detail = parse_fail_reason(reason)
    meta = build_run_meta(
        exp_name=exp_name or "failed",
        train_it=int(train_it if train_it is not None else cfg.get("TRAIN_ITERS", 0)),
        train_repetition=int(cfg.get("TRAIN_REPETITION", 1)),
        rep_exp_names=exp_name or "",
        max_p=float("nan"),
        eval_reward_mean=0.0,
        failed=True,
        fail_reason=fail_detail,
    )
    meta["fail_category"] = fail_category
    extra = dict(
        p_s=np.array([]),
        v_s=np.array([]),
        E_s=np.array([]),
    )
    return ff, meta, extra


def all_finite(vals: Sequence[float]) -> bool:
    for v in vals:
        try:
            if not np.isfinite(float(v)):
                return False
        except Exception:
            return False
    return True


def extract_reward_curve(
    log_dir: Path,
    train_iters: int,
    n_points: int = 10,
    win_frac: float = 0.05,
) -> Dict[str, float]:
    """
    Extract reward values at 10%, 20%, …, 100% of training using a moving
    average over a window equal to `win_frac` of total training steps.
    """
    if not log_dir.exists():
        return {}

    ea = event_accumulator.EventAccumulator(str(log_dir))
    try:
        ea.Reload()
    except Exception:
        return {}

    key = "Train/mean_reward"
    if key not in ea.Tags().get("scalars", []):
        return {}

    events = ea.Scalars(key)
    if not events:
        return {}

    steps = np.array([e.step for e in events], dtype=np.int64)
    vals = np.array([e.value for e in events], dtype=np.float64)

    win_steps = max(1, int(round(train_iters * win_frac)))
    half = win_steps // 2

    out: Dict[str, float] = {}
    for frac in range(1, n_points + 1):
        target = int(round(train_iters * frac / n_points))
        lo, hi = target - half, target + half
        mask = (steps >= lo) & (steps <= hi)

        if mask.any():
            out[f"rew_{frac * 10}pct"] = float(np.nanmean(vals[mask]))
        else:
            idx = int(np.argmin(np.abs(steps - target)))
            out[f"rew_{frac * 10}pct"] = float(vals[idx])

    return out


def eval_only_custom(
    genome_norm,
    policy_path,
    tag,
    cfg,
    invalid_v: set[float],
    invalid_e: set[float],
    invalid_p: set[float],
    return_arrays=True,
):
    set_thread_envs()
    log_worker_context("eval_only", cfg)
    try:
        phys_genome = Chromosome_Drone.to_physical(genome_norm)
        urdf_dir = resolve_urdf_dir(cfg["URDF_DIR"])
        urdf_file = create_urdf_with_retry(phys_genome, urdf_dir)
    except Exception as exc:
        traceback.print_exc()
        reason = f"urdf_generation_failed: {exc}"
        print(f"[safe_mode] {reason}")
        return failure_result(reason, cfg, invalid_v, invalid_e, invalid_p, train_it=0)
    exp_name = urdf_file.stem
    ind_uid = cfg.get("IND_UID", None)
    try:
        ind_uid = int(ind_uid) if ind_uid is not None else None
    except Exception:
        ind_uid = None
    if ind_uid is not None and ind_uid >= 0:
        exp_name = f"{exp_name}-u{ind_uid:06d}"
    gen_tag = cfg.get("GENERATION", None)
    try:
        gen_tag = int(gen_tag) if gen_tag is not None else None
    except Exception:
        gen_tag = None
    if gen_tag is not None:
        gen_prefix = f"g{gen_tag:03d}"
        if cfg.get("EXP_PREFIX"):
            exp_name = f"{cfg['EXP_PREFIX']}-{gen_prefix}-{exp_name}"
        else:
            exp_name = f"{gen_prefix}-{exp_name}"
    elif cfg.get("EXP_PREFIX"):
        exp_name = f"{cfg['EXP_PREFIX']}-{exp_name}"

    prepare_device_env(cfg.get("DEVICE", "cuda:0"))
    base_dir = Path(cfg["BASE_DIR"]).expanduser().resolve()
    eval_dir = (Path(cfg["LOGS_DIR"]).expanduser().resolve() / "eval" / exp_name)

    policy_path = Path(policy_path).expanduser().resolve()
    cfg_src = policy_path.parent / "cfgs.pkl"
    cfg_dst = Path(cfg["LOG_ROOT"]).expanduser().resolve() / exp_name / "cfgs.pkl"
    cfg_dst.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_src.is_file():
        raise FileNotFoundError(f"cfgs.pkl not found next to policy: {cfg_src}")
    copied_cfg = False
    if not cfg_dst.is_file():
        shutil.copy2(cfg_src, cfg_dst)
        copied_cfg = True
    print(
        f"[eval_only] exp={exp_name} policy={policy_path} "
        f"cfgs={'copied' if copied_cfg else 'reuse'} eval_dir={eval_dir}"
    )

    with pushd(base_dir):
        print(
            "[eval_only] running evaluation "
            f"(envs={cfg['EVAL_ENVS']} vmin={cfg['VMIN']} vmax={cfg['VMAX']})"
        )
        out = evaluation(
            exp_name=exp_name,
            urdf_file=urdf_file,
            ckpt=None,
            envs=cfg["EVAL_ENVS"],
            vmin=cfg["VMIN"],
            vmax=cfg["VMAX"],
            return_arrays=return_arrays,
            obs_genome=None,
            custom_policy_path=policy_path,
            eval_dir=eval_dir,
        )

    if return_arrays:
        v_dict, e_dict, p_dict, _, extra = out
        max_p = extra["max_p"]
    else:
        v_dict, e_dict, p_dict, _, max_p = out
        extra = None

    eval_reward_mean = float(extra.get("eval_reward_mean", np.nan)) if extra else float("nan")
    if not np.isfinite(eval_reward_mean):
        eval_reward_mean = 0.0
    meta = dict(
        vel_v=v_dict["mean_v"],
        vel_E=-v_dict["mean_E"],
        vel_P=v_dict["mean_progress"],
        eff_v=e_dict["mean_v"],
        eff_E=-e_dict["mean_E"],
        eff_P=e_dict["mean_progress"],
        prog_v=p_dict["mean_v"],
        prog_E=-p_dict["mean_E"],
        prog_P=p_dict["mean_progress"],
        **build_run_meta(
            exp_name=exp_name,
            train_it=0,
            train_repetition=1,
            rep_exp_names=exp_name,
            max_p=max_p,
            eval_reward_mean=eval_reward_mean,
        ),
        **{f"rew_{i*10}pct": 0.0 for i in range(1, 11)},
        final_reward=0.0,
        steps90_pct=0.0,
    )

    ff = [
        v_dict["mean_v"],
        -e_dict["mean_E"],
        p_dict["mean_progress"],
    ]
    print(f"[eval_only] done exp={exp_name} ff={ff} max_p={max_p:.2f}")

    return ff, meta, extra


def train_and_eval_sync(
    genome_norm: Sequence[float],
    parent_info: Tuple[Optional[str], Optional[int]],
    tag: str,
    cfg: Dict[str, Any],
    invalid_v: set[float],
    invalid_e: set[float],
    invalid_p: set[float],
    return_arrays: bool = True,
):
    """
    Synchronous wrapper that:
      1. Maps normalized genome -> physical parameters.
      2. Builds a URDF for the given physical genome.
      3. Trains a control policy (with optional inheritance).
      4. Runs evaluation over a speed range.
      5. Reads training reward curve from TensorBoard.
    """
    parent_exp, parent_ckpt = parent_info

    try:
        set_thread_envs()
        log_mem("train_eval:start")
        log_worker_context("train_and_eval", cfg)
        phys_genome = Chromosome_Drone.to_physical(genome_norm)
        urdf_dir = resolve_urdf_dir(cfg["URDF_DIR"])
        urdf_file = create_urdf_with_retry(phys_genome, urdf_dir)
    except Exception as exc:
        traceback.print_exc()
        reason = f"urdf_generation_failed: {exc}"
        print(f"[safe_mode] {reason}")
        return failure_result(reason, cfg, invalid_v, invalid_e, invalid_p)

    base_exp_name = urdf_file.stem
    ind_uid = cfg.get("IND_UID", None)
    try:
        ind_uid = int(ind_uid) if ind_uid is not None else None
    except Exception:
        ind_uid = None
    if ind_uid is not None and ind_uid >= 0:
        base_exp_name = f"{base_exp_name}-u{ind_uid:06d}"
    if cfg.get("EXP_PREFIX"):
        base_exp_name = f"{cfg['EXP_PREFIX']}-{base_exp_name}"

    device = prepare_device_env(cfg.get("DEVICE", "cuda:0"))
    base_dir = Path(cfg["BASE_DIR"]).expanduser().resolve()

    train_iters = (
        cfg["TRAIN_ITERS_INHERIT"]
        if (parent_exp is not None and parent_ckpt is not None)
        else cfg["TRAIN_ITERS"]
    )

    train_repetition = int(cfg.get("TRAIN_REPETITION", 1) or 1)
    if train_repetition < 1:
        print("[train_eval][warn] TRAIN_REPETITION < 1; forcing to 1")
        train_repetition = 1

    def run_once(run_exp_name: str) -> Tuple[List[float], Dict[str, Any], Dict[str, Any]]:
        try:
            jitter = float(os.getenv("WORKER_START_JITTER", "0") or 0.0)
            if jitter > 0:
                time.sleep(random.uniform(0.0, jitter))
            log_mem(f"train_eval:{run_exp_name}:before_train")
            with pushd(base_dir):
                training(
                    exp_name=run_exp_name,
                    urdf_file=urdf_file,
                    num_envs=cfg["TRAIN_ENVS"],
                    max_iterations=train_iters,
                    parent_exp=parent_exp,
                    parent_ckpt=parent_ckpt,
                    device=device,
                )
        except Exception as exc:
            traceback.print_exc()
            reason = f"training_failed exp={run_exp_name}: {exc}"
            print(f"[safe_mode] {reason}")
            return failure_result(reason, cfg, invalid_v, invalid_e, invalid_p, exp_name=run_exp_name, train_it=train_iters)

        eval_dir = (Path(cfg["LOGS_DIR"]).expanduser().resolve() / "eval" / run_exp_name)

        try:
            log_mem(f"train_eval:{run_exp_name}:before_eval")
            with pushd(base_dir):
                out = evaluation(
                    exp_name=run_exp_name,
                    urdf_file=urdf_file,
                    ckpt=train_iters,
                    envs=cfg["EVAL_ENVS"],
                    vmin=cfg["VMIN"],
                    vmax=cfg["VMAX"],
                    return_arrays=return_arrays,
                    eval_dir=eval_dir,
                )
        except Exception as exc:
            traceback.print_exc()
            reason = f"evaluation_failed exp={run_exp_name}: {exc}"
            print(f"[safe_mode] {reason}")
            return failure_result(reason, cfg, invalid_v, invalid_e, invalid_p, exp_name=run_exp_name, train_it=train_iters)

        try:
            if return_arrays:
                v_dict, e_dict, p_dict, _, extra = out
                max_p = extra["max_p"]
            else:
                v_dict, e_dict, p_dict, _, max_p = out
                extra = None
        except Exception as exc:
            reason = f"eval_output_unpack_failed exp={run_exp_name}: {exc}"
            print(f"[safe_mode] {reason}")
            return failure_result(reason, cfg, invalid_v, invalid_e, invalid_p, exp_name=run_exp_name, train_it=train_iters)

        if not all_finite(
            [
                v_dict.get("mean_v"),
                e_dict.get("mean_E"),
                p_dict.get("mean_progress"),
                max_p,
            ]
        ):
            reason = (
                f"non_finite_metrics exp={run_exp_name} "
                f"v={v_dict.get('mean_v')} E={e_dict.get('mean_E')} "
                f"P={p_dict.get('mean_progress')} max_p={max_p}"
            )
            print(f"[safe_mode] {reason}")
            return failure_result(reason, cfg, invalid_v, invalid_e, invalid_p, exp_name=run_exp_name, train_it=train_iters)

        if return_arrays:
            p_s = np.asarray(extra.get("p_s", []))
            v_s = np.asarray(extra.get("v_s", []))
            E_s = np.asarray(extra.get("E_s", []))
            if p_s.size == 0 or v_s.size == 0 or E_s.size == 0:
                reason = f"empty_eval_arrays exp={run_exp_name}"
                print(f"[safe_mode] {reason}")
                return failure_result(reason, cfg, invalid_v, invalid_e, invalid_p, exp_name=run_exp_name, train_it=train_iters)
            if not (
                np.isfinite(p_s).all()
                and np.isfinite(v_s).all()
                and np.isfinite(E_s).all()
            ):
                reason = f"non_finite_eval_arrays exp={run_exp_name}"
                print(f"[safe_mode] {reason}")
                return failure_result(reason, cfg, invalid_v, invalid_e, invalid_p, exp_name=run_exp_name, train_it=train_iters)

        tb_log_dir = Path(cfg["LOG_ROOT"]) / run_exp_name
        reward_curve = extract_reward_curve(
            tb_log_dir,
            train_iters,
            n_points=10,
            win_frac=0.05,
        )

        eval_reward_mean = float(extra.get("eval_reward_mean", np.nan)) if extra else float("nan")
        if not np.isfinite(eval_reward_mean):
            eval_reward_mean = 0.0

        meta = dict(
            vel_v=v_dict["mean_v"],
            vel_E=-v_dict["mean_E"],
            vel_P=v_dict["mean_progress"],
            eff_v=e_dict["mean_v"],
            eff_E=-e_dict["mean_E"],
            eff_P=e_dict["mean_progress"],
            prog_v=p_dict["mean_v"],
            prog_E=-p_dict["mean_E"],
            prog_P=p_dict["mean_progress"],
            **build_run_meta(
                exp_name=run_exp_name,
                train_it=train_iters,
                max_p=max_p,
                eval_reward_mean=eval_reward_mean,
            ),
            **reward_curve,
        )

        ff = [
            v_dict["mean_v"],
            -e_dict["mean_E"],
            p_dict["mean_progress"],
        ]

        if return_arrays and train_repetition > 1:
            rep_dir = Path(cfg["BASE_DIR"]).expanduser().resolve() / "analysis" / "rep_payloads"
            rep_dir.mkdir(parents=True, exist_ok=True)
            payload_path = rep_dir / f"{run_exp_name}.npz"
            np.savez_compressed(payload_path, p_s=p_s, v_s=v_s, E_s=E_s)
            extra = {"payload_path": str(payload_path)}
            print(f"[train_eval] rep payload saved -> {payload_path}")

        return ff, meta, extra

    def cleanup_after_rep() -> None:
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if train_repetition == 1:
        ff, meta, extra = run_once(base_exp_name)
        meta["train_repetition"] = 1
        meta["rep_exp_names"] = base_exp_name
        return ff, meta, extra

    rep_payloads: List[Dict[str, Any]] = []
    rep_exp_names: List[str] = []
    raw_ffs: List[List[float]] = []
    max_p_vals: List[float] = []

    for rep_idx in range(train_repetition):
        run_exp_name = base_exp_name
        if train_repetition > 1:
            run_exp_name = f"{base_exp_name}_r{rep_idx + 1:02d}"
            print(
                f"[train_eval] rep {rep_idx + 1}/{train_repetition} "
                f"exp={run_exp_name} train_iters={train_iters}"
            )
        ff, meta, extra = run_once(run_exp_name)
        log_mem(f"train_eval:{run_exp_name}:after_eval")
        cleanup_after_rep()
        rep_payloads.append(dict(rep_idx=rep_idx, meta=meta, extra=extra))
        rep_exp_names.append(run_exp_name)

        if meta.get("failed"):
            raw_ffs.append(default_fitness(invalid_v, invalid_e, invalid_p))
        else:
            raw_ffs.append(ff)
        try:
            max_p_vals.append(float(meta.get("max_p", np.nan)))
        except Exception:
            pass

        if train_repetition > 1:
            status = "failed" if meta.get("failed") else "ok"
            print(
                f"[train_eval] rep {rep_idx + 1}/{train_repetition} done "
                f"exp={run_exp_name} status={status}"
            )

    if raw_ffs:
        ff_mean = np.nanmean(np.asarray(raw_ffs, dtype=float), axis=0).tolist()
    else:
        ff_mean = default_fitness(invalid_v, invalid_e, invalid_p)

    max_p_mean = float(np.nanmean(max_p_vals)) if max_p_vals else float("nan")

    meta = dict(
        exp_name=rep_exp_names[0] if rep_exp_names else base_exp_name,
        train_it=train_iters,
        ckpt_idx=ckpt_idx_from_train_it(train_iters),
        train_repetition=train_repetition,
        rep_exp_names="|".join(rep_exp_names),
        max_p=max_p_mean,
    )
    extra = dict(rep_payloads=rep_payloads)

    log_mem("train_eval:end")
    return ff_mean, meta, extra
