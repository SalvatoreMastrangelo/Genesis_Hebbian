#!/usr/bin/env python
"""
WP1 evaluation entry point — thin CLI wrapper around
``winged_drone_train.eval.evaluation``.

Points the wrapped routine at a WP1 run folder (``logs/runs/<ts>_<exp>/``)
instead of the ``logs/ea/<exp>/`` layout used by the standalone trainer.
Everything else — rollout, plotting, TensorBoard reward extraction — is
performed by ``winged_drone_train.eval`` unchanged.

Usage
-----
.. code-block:: bash

    # Auto-pick the latest checkpoint in <run>/tb/
    python -m WP1.eval --run logs/runs/2026-05-11_12-00-00_my-exp

    # Pick a specific checkpoint
    python -m WP1.eval --run logs/runs/.../  --ckpt 500

    # Custom speed sweep and env count
    python -m WP1.eval --run logs/runs/.../ --vmin 4 --vmax 20 --envs 4096

    # Use an arbitrary URDF rather than the one logged at training time
    python -m WP1.eval --run logs/runs/.../ --urdf-file path/to/drone.urdf
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Optional, Tuple

from winged_drone_train.eval import _apply_eval_env_overrides, evaluation


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_MODEL_RE = re.compile(r"model_(\d+)\.pt$")


def _find_latest_checkpoint(tb_dir: Path) -> Tuple[int, Path]:
    """Return ``(ckpt_id, path)`` for the highest-numbered model in *tb_dir*.

    ``ckpt_id`` follows the same convention used by ``winged_drone_train.eval``
    (which loads ``model_{ckpt-1}.pt``): the returned id is ``N+1`` so that
    callers can pass it as ``--ckpt`` and have it resolve back to
    ``model_N.pt``.
    """
    if not tb_dir.is_dir():
        raise FileNotFoundError(f"Expected TensorBoard / checkpoint dir at {tb_dir}")
    matches = [
        (int(m.group(1)), p)
        for p in tb_dir.iterdir()
        for m in [_MODEL_RE.search(p.name)]
        if m is not None
    ]
    if not matches:
        raise FileNotFoundError(f"No model_*.pt checkpoints in {tb_dir}")
    matches.sort(key=lambda kv: kv[0])
    best_n, best_path = matches[-1]
    return best_n + 1, best_path


def _read_urdf_from_cfg(cfg_path: Path) -> Optional[str]:
    """Inspect ``cfgs.pkl`` and return the URDF path recorded at train time."""
    if not cfg_path.is_file():
        return None
    with cfg_path.open("rb") as f:
        env_cfg, *_ = pickle.load(f)
    candidate = env_cfg.get("urdf_file") if isinstance(env_cfg, dict) else None
    return str(candidate) if candidate else None


def _read_env_cfg(cfg_path: Path) -> Dict[str, Any]:
    """Return the env_cfg dict from ``cfgs.pkl`` (empty dict if unreadable)."""
    if not cfg_path.is_file():
        return {}
    with cfg_path.open("rb") as f:
        env_cfg, *_ = pickle.load(f)
    return dict(env_cfg) if isinstance(env_cfg, dict) else {}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a WP1 training run (wraps winged_drone_train.eval.evaluation)."
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to a WP1 run folder (logs/runs/<timestamp>_<exp_name>/).",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=None,
        help=(
            "Checkpoint id (1-based as expected by winged_drone_train.eval). "
            "If omitted, the highest-numbered checkpoint in <run>/tb/ is used."
        ),
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=4096,
        help="Number of parallel environments for the eval rollout.",
    )
    parser.add_argument("--vmin", type=float, default=5.0, help="Min commanded speed (m/s).")
    parser.add_argument("--vmax", type=float, default=25.0, help="Max commanded speed (m/s).")
    parser.add_argument(
        "--minimal-progress",
        type=float,
        default=250.0,
        help="Distance (m) used as the COT integration limit.",
    )
    parser.add_argument(
        "--win-frac",
        type=float,
        default=0.05,
        help="Smoothing window (fraction of curve length) used by EvaluationPlotter.",
    )
    parser.add_argument(
        "--urdf-file",
        type=str,
        default=None,
        help="Override the URDF; defaults to the one recorded in <run>/cfgs.pkl.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot rendering (faster; just returns top_vel/top_eff/top_prog).",
    )
    parser.add_argument(
        "--obs-genome",
        action="store_true",
        help="Force-enable genome observations for the eval env.",
    )
    parser.add_argument(
        "--dens-min",
        dest="dens_min",
        type=float,
        default=None,
        help="Override forest density at x=x_lower [trees/m]. Defaults to the value in cfgs.pkl.",
    )
    parser.add_argument(
        "--dens-max",
        dest="dens_max",
        type=float,
        default=None,
        help="Override forest density at x=x_upper [trees/m]. Defaults to the value in cfgs.pkl.",
    )
    return parser


def evaluate_run(
    run_dir: str | Path,
    *,
    ckpt: Optional[int] = None,
    envs: int = 4096,
    vmin: float = 5.0,
    vmax: float = 25.0,
    minimal_progress: float = 250.0,
    win_frac: float = 0.05,
    urdf_file: Optional[str] = None,
    save_plots: bool = True,
    obs_genome: bool = False,
    dens_min: Optional[float] = None,
    dens_max: Optional[float] = None,
):
    """Programmatic wrapper around ``winged_drone_train.eval.evaluation``
    that points it at a WP1-style run folder.

    Returns whatever ``evaluation`` returns (a 4-tuple by default).
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg_path = run_dir / "cfgs.pkl"
    tb_dir = run_dir / "tb"

    if ckpt is None:
        ckpt, ckpt_path = _find_latest_checkpoint(tb_dir)
    else:
        ckpt_path = tb_dir / f"model_{ckpt - 1}.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    resolved_urdf = urdf_file or _read_urdf_from_cfg(cfg_path)

    eval_dir = run_dir / "eval"

    # Reproduce the env_cfg the underlying evaluation() will use, so we can
    # print the final (post-override) values for x_upper / base_init_pos /
    # densities. The underlying call reloads from cfgs.pkl independently.
    env_cfg_preview = _read_env_cfg(cfg_path)
    _apply_eval_env_overrides(env_cfg_preview)
    if dens_min is not None:
        env_cfg_preview["dens_min"] = float(dens_min)
    if dens_max is not None:
        env_cfg_preview["dens_max"] = float(dens_max)

    print("[WP1.eval] evaluation settings:")
    print(f"  ckpt             = {ckpt}  (model_{ckpt - 1}.pt)")
    print(f"  urdf_file        = {resolved_urdf if resolved_urdf else '<from cfgs.pkl drone_key>'}")
    print(f"  envs             = {envs}")
    print(f"  vmin / vmax      = {vmin} / {vmax} m/s")
    print(f"  dens_min         = {env_cfg_preview.get('dens_min')}"
          f"{' (override)' if dens_min is not None else ''}")
    print(f"  dens_max         = {env_cfg_preview.get('dens_max')}"
          f"{' (override)' if dens_max is not None else ''}")
    print(f"  minimal_progress = {minimal_progress} m")
    print(f"  win_frac         = {win_frac}")
    print(f"  x_lower          = {env_cfg_preview.get('x_lower')}")
    print(f"  x_upper          = {env_cfg_preview.get('x_upper')}")
    print(f"  base_init_pos    = {env_cfg_preview.get('base_init_pos')}")
    print(f"  obs_genome       = {obs_genome}")
    print(f"  save_plots       = {save_plots}")

    return evaluation(
        exp_name=run_dir.name,
        urdf_file=resolved_urdf,
        ckpt=ckpt,
        envs=envs,
        vmin=vmin,
        vmax=vmax,
        win_frac=win_frac,
        minimal_progress=minimal_progress,
        return_arrays=True,
        custom_policy_path=str(ckpt_path),
        cfg_path=cfg_path,
        obs_genome=obs_genome,
        save_plots=save_plots,
        eval_dir=eval_dir,
        dens_min=dens_min,
        dens_max=dens_max,
    )


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = evaluate_run(
        run_dir=args.run,
        ckpt=args.ckpt,
        envs=args.envs,
        vmin=args.vmin,
        vmax=args.vmax,
        minimal_progress=args.minimal_progress,
        win_frac=args.win_frac,
        urdf_file=args.urdf_file,
        save_plots=not args.no_plots,
        obs_genome=args.obs_genome,
        dens_min=args.dens_min,
        dens_max=args.dens_max,
    )
    top_vel, top_eff, top_prog, max_p, extra = result
    print(f"[WP1.eval] top_vel  : {top_vel}")
    print(f"[WP1.eval] top_eff  : {top_eff}")
    print(f"[WP1.eval] top_prog : {top_prog}")
    print(f"[WP1.eval] max_p    : {max_p}")
    print(f"[WP1.eval] mean_progress_all (flat avg over all envs): "
          f"{extra.get('mean_progress_all', float('nan')):.2f} m")


if __name__ == "__main__":
    main()
