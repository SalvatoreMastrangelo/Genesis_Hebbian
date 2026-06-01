"""
WP2 validation: baseline (WP1 actor) vs best Hebbian individual.
================================================================

Re-evaluates a finished WP2 run by replaying, on the **same forests under
common random numbers (CRN)**, two controllers:

  * the frozen WP1 actor (``reproducibility/wp1_actor.pt``) — the no-plasticity
    baseline the run already logs in ``results/baseline_summary.csv``;
  * a chosen ``best_individual`` genome (default: the fitness winner), i.e. the
    same frozen actor plus the evolved last-layer Hebbian rules.

Both rollouts are run through ``src/winged_drone_train/eval.py`` (the
``evaluation`` / ``evaluation_hebbian`` entry points) with an identical ``seed``
and ``crn=True``, so the forest layouts, per-slot forest assignment, commanded
speeds, and episode-reset domain-randomization draws are shared. The two
rollouts therefore differ only in the controller — exactly the comparison
needed to tell whether plasticity helped.

Outputs land in ``<run>/plots/validation/``:

  * ``baseline/`` and ``hebbian/`` — the standard per-controller plot sets
    (``total_plot.png``, joint heatmaps, ...);
  * ``overlay_metrics.png`` / ``overlay_progress.png`` — superimposed curves
    with both controllers on one axis;
  * ``initial_conditions.txt`` — the CRN verification report (forests / commands
    / DR draws compared between the two rollouts);
  * ``summary.txt`` — headline metrics for both controllers.

Usage
-----
    # Only --run is mandatory:
    PYTHONPATH=src python -m WP2.validate --run logs/remote/.../<wp2_run>

    # Pick a different best-individual genome (metric folder under
    # best_individual/, or a direct .npy path):
    PYTHONPATH=src python -m WP2.validate --run <wp2_run> --genome progress

    # Override the course / sampling (all default-as-eval, overridable):
    PYTHONPATH=src python -m WP2.validate --run <wp2_run> \
        --envs 8192 --x-upper 600 --dens-max 5 --vmin 5 --vmax 25

    # Inject a wing-break fault for BOTH controllers (same forests under CRN)
    # to test in-flight adaptation — the pilot's LEFT wing loses half its lift
    # once each env passes 150 m:
    PYTHONPATH=src python -m WP2.validate --run <wp2_run> \
        --break-left-wing 150 --break-left-loss 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Metric sub-folders written under <run>/best_individual/ by the WP2 run.
_GENOME_METRICS = (
    "fitness",
    "progress",
    "crash_rate",
    "cost_of_transport",
    "velocity_deviation",
)

_BASELINE_COLOUR = "#2ca02c"  # green  (matches plot_metrics baseline)
_HEBBIAN_COLOUR = "#1f77b4"   # blue


# ===========================================================================
#  Input resolution
# ===========================================================================
def _resolve_genome(run_dir: Path, genome: str) -> Tuple[Path, str]:
    """Resolve ``--genome`` to a genome ``.npy`` path and a short label.

    ``genome`` is either a ``best_individual/`` metric name (default
    ``fitness``) or a direct path to a ``.npy`` file.
    """
    cand = Path(genome).expanduser()
    if cand.suffix == ".npy":
        if not cand.is_absolute():
            cand = (run_dir / cand) if not cand.is_file() else cand
        if not cand.is_file():
            raise FileNotFoundError(f"--genome path not found: {cand}")
        return cand.resolve(), cand.stem

    npy = run_dir / "best_individual" / genome / "genome.npy"
    if not npy.is_file():
        available = sorted(
            p.name
            for p in (run_dir / "best_individual").glob("*")
            if (p / "genome.npy").is_file()
        ) if (run_dir / "best_individual").is_dir() else []
        raise FileNotFoundError(
            f"--genome '{genome}' not found at {npy}.\n"
            f"Pass a metric name {available or list(_GENOME_METRICS)} "
            f"or a path to a .npy file."
        )
    return npy.resolve(), genome


# ===========================================================================
#  CRN verification
# ===========================================================================
def _compare_initial_conditions(
    base: Optional[Dict[str, Any]],
    hebb: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Compare the two captured initial-condition snapshots.

    Returns ``(ok, report_text)``. ``ok`` requires the scenario-defining fields
    (forest assignment, tree-layout checksum, commanded speeds) and the
    episode-reset DR draws to match bit-for-bit-ish (tight tolerance).
    """
    lines: list[str] = []
    if not base or not hebb:
        return False, "[verify] no initial-condition snapshots were captured."

    # field -> (critical?, comparison)
    #   "exact"   : np.array_equal (integers / ids)
    #   "close"   : np.allclose with tight tol
    #   "scalar"  : float equality within tol
    spec = [
        ("forest_ids", True, "exact"),
        ("cylinders_checksum", True, "scalar"),
        ("commands", True, "close"),
        ("base_pos", True, "close"),
        ("base_lin_vel", True, "close"),
        ("mass_shift", True, "close"),
        ("com_shift", True, "close"),
        ("joint_target_episode_bias", True, "close"),
    ]
    overall_ok = True
    for key, critical, kind in spec:
        if key not in base and key not in hebb:
            continue
        if key not in base or key not in hebb:
            lines.append(f"  [MISS] {key:<26} present in only one rollout")
            if critical:
                overall_ok = False
            continue

        a, b = base[key], hebb[key]
        if kind == "scalar":
            match = bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-6))
            detail = f"baseline={float(a):.6g} hebbian={float(b):.6g}"
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            if a.shape != b.shape:
                lines.append(
                    f"  [FAIL] {key:<26} shape mismatch {a.shape} vs {b.shape}"
                )
                if critical:
                    overall_ok = False
                continue
            if kind == "exact":
                match = bool(np.array_equal(a, b))
                ndiff = int((a != b).sum())
                detail = f"n_differing={ndiff}/{a.size}"
            else:  # close
                match = bool(np.allclose(a, b, rtol=0.0, atol=1e-5))
                maxabs = float(np.max(np.abs(a - b))) if a.size else 0.0
                detail = f"max|Δ|={maxabs:.3e}"

        tag = "OK  " if match else "FAIL"
        lines.append(f"  [{tag}] {key:<26} {detail}")
        if not match and critical:
            overall_ok = False

    header = (
        "[verify] initial conditions MATCH — both controllers flew the same "
        "forests under CRN."
        if overall_ok
        else "[verify] initial conditions DIFFER — see failures below."
    )
    return overall_ok, header + "\n" + "\n".join(lines)


# ===========================================================================
#  Overlay plotting
# ===========================================================================
def _series(extra: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Pull the smoothed v_cmd-aligned arrays out of an evaluation() extra dict."""
    return {
        "v_cmd": np.asarray(extra.get("v_cmd_s", [])),
        "progress": np.asarray(extra.get("p_s", [])),
        "velocity": np.asarray(extra.get("v_s", [])),
        "cot": np.asarray(extra.get("E_s", [])),
        "reward": np.asarray(extra.get("eval_reward_s", [])),
    }


def _plot_overlay(
    base_extra: Dict[str, Any],
    hebb_extra: Dict[str, Any],
    out_dir: Path,
    *,
    title: str,
) -> None:
    """Superimpose baseline vs Hebbian curves (progress / velocity / COT / reward)."""
    b = _series(base_extra)
    h = _series(hebb_extra)

    panels = [
        ("progress", "Progress [m]"),
        ("velocity", "Mean velocity [m/s]"),
        ("cot", "Cost of Transport"),
        ("reward", "Eval reward"),
    ]

    def _draw(ax, key, label):
        if b["v_cmd"].size and b[key].size:
            ax.plot(
                b["v_cmd"], b[key], color=_BASELINE_COLOUR, linewidth=1.8,
                linestyle="--", label="baseline (WP1)",
            )
        if h["v_cmd"].size and h[key].size:
            ax.plot(
                h["v_cmd"], h[key], color=_HEBBIAN_COLOUR, linewidth=1.8,
                label="Hebbian",
            )
        ax.set_xlabel("Commanded speed [m/s]")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    # Combined 2x2.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13)
    for ax, (key, label) in zip(axes.flat, panels):
        _draw(ax, key, label)
    fig.tight_layout()
    out = out_dir / "overlay_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[validate] Saved {out}")

    # Standalone headline: progress vs commanded speed.
    fig_p, ax_p = plt.subplots(figsize=(7, 4.5))
    _draw(ax_p, "progress", "Progress [m]")
    ax_p.set_title(title, fontsize=11)
    fig_p.tight_layout()
    out_p = out_dir / "overlay_progress.png"
    fig_p.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig_p)
    print(f"[validate] Saved {out_p}")


def _fmt_summary(label: str, extra: Dict[str, Any]) -> str:
    return (
        f"{label:<16} "
        f"max_progress={extra.get('max_p', float('nan')):8.2f} m   "
        f"mean_progress={extra.get('mean_progress_all', float('nan')):8.2f} m   "
        f"eval_reward_mean={extra.get('eval_reward_mean', float('nan')):8.3f}"
    )


def _aggregate_extras(extras: list, *, win_frac: float = 0.05) -> Dict[str, Any]:
    """Pool the raw per-env arrays across a multi-URDF batch into one ``extra``.

    Concatenates the (NaN-dropped, mutually aligned) ``raw`` arrays from each
    per-URDF rollout, then re-derives the smoothed v_cmd-aligned curves and the
    headline scalars exactly as :func:`evaluation_hebbian` does for a single
    URDF — so the combined overlay/summary use the same logic as the per-URDF
    plots, just over the pooled population.
    """
    from winged_drone_train.analysis.eval_plotter import EvaluationPlotter

    def _cat(key: str) -> np.ndarray:
        parts = [np.asarray(e["raw"][key]) for e in extras if e.get("raw")]
        return np.concatenate(parts) if parts else np.asarray([])

    v_cmd = _cat("v_cmd")
    v_mean = _cat("v_mean")
    cot = _cat("cot")
    progress = _cat("progress")
    reward = _cat("reward")

    x_s, v_s, _ = EvaluationPlotter.moving_avg(v_cmd, v_mean, win_frac)
    _, p_s, _ = EvaluationPlotter.moving_avg(v_cmd, progress, win_frac)
    _, E_s, _ = EvaluationPlotter.moving_avg(v_cmd, cot, win_frac)
    _, reward_s, _ = EvaluationPlotter.moving_avg(v_cmd, reward, win_frac)

    max_p = float(np.max(p_s)) if len(p_s) else 0.0
    mean_progress_all = float(np.mean(progress)) if progress.size else 0.0
    eval_reward_mean = float(np.mean(reward)) if reward.size else float("nan")
    if not np.isfinite(eval_reward_mean):
        eval_reward_mean = 0.0

    return {
        "max_p": max_p,
        "mean_progress_all": mean_progress_all,
        "eval_reward_mean": eval_reward_mean,
        "v_cmd_s": x_s,
        "p_s": p_s,
        "v_s": v_s,
        "E_s": E_s,
        "eval_reward_s": reward_s,
    }


def _plot_pooled_controller(
    extras: list,
    out_dir: Path,
    *,
    vmin: float,
    vmax: float,
    minimal_progress: float,
    win_frac: float = 0.05,
) -> None:
    """Render a single controller's standard plot set over the POOLED batch.

    Concatenates the per-URDF raw metric arrays and joint-position traces, then
    produces the same ``total_plot`` / ``total_plot_points`` / joint heatmaps as
    a single-URDF rollout — but for the whole multi-URDF population at once.
    """
    from winged_drone_train.analysis.eval_plotter import EvaluationPlotter

    out_dir.mkdir(parents=True, exist_ok=True)

    def _cat(key: str) -> np.ndarray:
        parts = [np.asarray(e["raw"][key]) for e in extras if e.get("raw")]
        return np.concatenate(parts) if parts else np.asarray([])

    v_mean, cot = _cat("v_mean"), _cat("cot")
    v_cmd, progress = _cat("v_cmd"), _cat("progress")

    # Pool the per-env joint traces (lists are concatenated; v_cmd stacked).
    s_all: list = []
    jpos_all: list = []
    vcmd_traces: list = []
    for e in extras:
        tr = e.get("traces")
        if not tr:
            continue
        s_all.extend(list(tr.get("s", [])))
        jpos_all.extend(list(tr.get("j_pos", [])))
        vcmd_traces.append(np.asarray(tr.get("v_cmd", [])))
    traces = {
        "s": s_all,
        "j_pos": jpos_all,
        "v_cmd": np.concatenate(vcmd_traces) if vcmd_traces else np.asarray([]),
    }

    csr = (float(vmin), float(vmax))
    plotter = EvaluationPlotter()
    jobs = [
        ("total_plot.png",
         lambda o: plotter.total_plot(v_mean, cot, v_cmd, progress, win_frac=win_frac,
                                      minimal_p=minimal_progress, out=o, velocity_range=csr)),
        ("total_plot_points_instead_of_ma.png",
         lambda o: plotter.total_plot_points_instead_of_ma(v_mean, cot, v_cmd, progress, win_frac=win_frac,
                                                           minimal_p=minimal_progress, out=o, velocity_range=csr)),
        ("joint_heatmap_sweep.png",
         lambda o: plotter.plot_joint_diff_heatmap(traces, "sweep", out=o, command_speed_range=csr)),
        ("joint_heatmap_twist.png",
         lambda o: plotter.plot_joint_diff_heatmap(traces, "twist", out=o, command_speed_range=csr)),
    ]
    for name, fn in jobs:
        try:
            fn(str(out_dir / name))
        except Exception as exc:  # keep the rest of the set on a single failure
            print(f"[validate][warn] pooled plot {name} failed: {exc}")
    print(f"[validate] Saved pooled plot set -> {out_dir}")


def _safe_stem(urdf_file: Optional[str], fallback: str) -> str:
    """Filesystem-safe short label for a URDF path (or the default drone)."""
    import re

    if not urdf_file:
        return fallback
    stem = Path(urdf_file).stem
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return (clean or fallback)[:60]


def _fault_desc(args: Any) -> str:
    """One-line description of the active fault-injection knobs (for reports)."""
    parts: list[str] = []
    if getattr(args, "freeze_distance", None) is not None:
        parts.append(f"freeze@{args.freeze_distance:g}m")
    if getattr(args, "break_left_distance", None) is not None:
        parts.append(f"break_left@{args.break_left_distance:g}m(-{args.break_left_loss:g}%)")
    if getattr(args, "break_right_distance", None) is not None:
        parts.append(f"break_right@{args.break_right_distance:g}m(-{args.break_right_loss:g}%)")
    return "  ".join(parts) if parts else "none"


# ===========================================================================
#  Main
# ===========================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run", required=True,
        help="WP2 run directory (must contain reproducibility/ and best_individual/).",
    )
    p.add_argument(
        "--genome", default="fitness",
        help="best_individual metric name (default: fitness) or a path to a genome .npy.",
    )
    p.add_argument(
        "--genome-idx", dest="genome_idx", type=int, default=0,
        help="Row index when the genome .npy is 2D (default: 0).",
    )
    # Course / sampling — defaults overridable exactly like eval.py.
    p.add_argument("--envs", type=int, default=8192, help="number of eval envs (default 8192)")
    p.add_argument("--vmin", type=float, default=5.0, help="min commanded speed (default 5)")
    p.add_argument("--vmax", type=float, default=25.0, help="max commanded speed (default 25)")
    p.add_argument(
        "--x-upper", dest="x_upper", type=float, default=600.0,
        help="world / course end along +X in metres (default 600)",
    )
    p.add_argument(
        "--forest-x-limit", dest="forest_x_limit", type=float, default=None,
        help="forest x extent (default: same as --x-upper)",
    )
    p.add_argument("--dens-min", dest="dens_min", type=float, default=None,
                   help="forest density at x_lower [trees/m] (default: config)")
    p.add_argument("--dens-max", dest="dens_max", type=float, default=5.0,
                   help="forest density at x_upper [trees/m] (default 5)")
    p.add_argument("--init-vx", dest="init_vx", type=float, default=None,
                   help="override initial body velocity along +X at reset [m/s]")
    p.add_argument("--minimal-progress", dest="minimal_progress", type=float, default=250.0,
                   help="distance used for the COT / efficiency window [m] (default 250)")
    # ---- Fault injection (mirrors winged_drone_train/eval.py) ----------- #
    # Applied identically to BOTH the baseline and the Hebbian rollout (they
    # share the seed + CRN), so the comparison isolates how each controller
    # copes with the same wing damage / frozen plasticity.
    p.add_argument(
        "--freeze", dest="freeze_distance", type=float, default=None, metavar="DIST",
        help=(
            "Freeze the Hebbian last-layer weights once any env has travelled "
            "DIST metres along +X: no ABCD update and no decay-toward-checkpoint "
            "for the rest of the rollout. No-op on the zero-rule baseline."
        ),
    )
    p.add_argument(
        "--break-left-wing", dest="break_left_distance", type=float, default=None, metavar="DIST",
        help=(
            "Reduce the lift of the wing on the pilot's LEFT (visual orientation) "
            "once each env has travelled DIST metres along +X. The lift lost is "
            "controlled by --break-left-loss (default 50%%). Per-env latched, one-shot."
        ),
    )
    p.add_argument(
        "--break-left-loss", dest="break_left_loss", type=float, default=50.0, metavar="PCT",
        help=(
            "Percentage of left-wing lift to lose when --break-left-wing fires "
            "(default 50). 0 = no effect, 100 = no lift remaining. Clamped to [0, 100]."
        ),
    )
    p.add_argument(
        "--break-right-wing", dest="break_right_distance", type=float, default=None, metavar="DIST",
        help=(
            "Reduce the lift of the wing on the pilot's RIGHT (visual orientation) "
            "once each env has travelled DIST metres along +X. The lift lost is "
            "controlled by --break-right-loss (default 50%%). Per-env latched, one-shot."
        ),
    )
    p.add_argument(
        "--break-right-loss", dest="break_right_loss", type=float, default=50.0, metavar="PCT",
        help=(
            "Percentage of right-wing lift to lose when --break-right-wing fires "
            "(default 50). 0 = no effect, 100 = no lift remaining. Clamped to [0, 100]."
        ),
    )
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed, identical for both rollouts → CRN (default 0)")
    p.add_argument("--no-crn", dest="crn", action="store_false",
                   help="disable CRN sharing of per-slot DR draws (still same seed)")
    p.set_defaults(crn=True)
    p.add_argument("--no-verbose", dest="verbose", action="store_false",
                   help="disable the per-50-step rollout progress log (printed by "
                        "default in validate, mirroring WP2 run.py's first-generation log)")
    p.set_defaults(verbose=True)
    # Multi-URDF robustness sweep.
    p.add_argument(
        "--n-urdf", dest="n_urdf", type=int, default=None,
        help="evaluate over N freshly-sampled random URDFs (excluding the default "
             "drone). --envs is split evenly so each URDF flies envs/N forests, and "
             "ALL URDFs fly the SAME set of forests (shared via forest injection).",
    )
    p.add_argument(
        "--include-default-urdf", dest="include_default_urdf", action="store_true",
        help="replace one of the N sampled URDFs with the default (standard-mydrone) "
             "drone. Batch size stays N.",
    )
    p.add_argument(
        "--urdf-seed", dest="urdf_seed", type=int, default=None,
        help="seed for random URDF sampling (default: --seed). Independent of the "
             "CRN/eval seed.",
    )
    args = p.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    repro = run_dir / "reproducibility"
    wp1_ckpt = repro / "wp1_actor.pt"
    wp1_cfg = repro / "wp1_config.yaml"
    for pth in (run_dir, repro, wp1_ckpt, wp1_cfg):
        if not pth.exists():
            raise FileNotFoundError(f"Required path not found: {pth}")

    genome_path, genome_label = _resolve_genome(run_dir, args.genome)

    out_dir = run_dir / "plots" / "validation"
    baseline_dir = out_dir / "baseline"
    hebbian_dir = out_dir / "hebbian"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[validate] run            = {run_dir}")
    print(f"[validate] genome         = {genome_path}  (label '{genome_label}')")
    print(f"[validate] envs={args.envs}  vmin={args.vmin}  vmax={args.vmax}  "
          f"x_upper={args.x_upper}  dens_max={args.dens_max}  seed={args.seed}  crn={args.crn}")
    print(f"[validate] fault          = {_fault_desc(args)}")
    print(f"[validate] output         = {out_dir}")

    # eval.py lives under src/winged_drone_train; ensure src/ is importable.
    _src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from winged_drone_train.eval import evaluation_hebbian

    # ---- Build the no-plasticity baseline as a zero-rule genome ---------- #
    # The frozen WP1 actor IS the WP2 run's baseline (results/baseline_summary).
    # We express it as a zero-rule genome (A=B=C=D=0 → ΔW=0, decay-to-checkpoint
    # is a no-op on the unchanged weights) so it runs through the SAME
    # actor-only path as the Hebbian genome. This (a) avoids the critic-size
    # mismatch that ``evaluation()``'s full runner.load() hits on
    # privileged-critic checkpoints, and (b) makes env construction identical
    # to the Hebbian rollout, so the forests line up under CRN.
    import torch
    from WP2.config import HebbianEvolutionConfig
    from WP2.utils import create_zero_initialized_genome

    _cfg = HebbianEvolutionConfig.from_yaml(repro / "config.yaml")
    _ck = torch.load(str(wp1_ckpt), map_location="cpu", weights_only=False)
    _sd = _ck.get("model_state_dict", _ck) if isinstance(_ck, dict) else _ck
    if "actor.4.weight" in _sd:
        _cfg.hebbian.num_actions = _sd["actor.4.weight"].shape[0]
        _cfg.hebbian.hidden_dim = _sd["actor.4.weight"].shape[1]
    del _ck, _sd
    zero_genome = np.asarray(create_zero_initialized_genome(_cfg), dtype=np.float64)

    # Sanity: the zero (ABCD-only) genome should match the evolved genome's
    # length when decay/eta are not evolved; fall back to a flat 0.5 vector.
    _real_len = int(np.load(genome_path).reshape(-1).shape[0]) if not _cfg.hebbian.evolve_decay \
        and not _cfg.hebbian.evolve_eta else zero_genome.shape[0]
    if zero_genome.shape[0] != _real_len:
        zero_genome = np.full(_real_len, 0.5, dtype=np.float64)
    baseline_genome = out_dir / "baseline_zero_genome.npy"
    np.save(baseline_genome, zero_genome)

    # Knobs shared by both controllers — identical seed + CRN are what make the
    # comparison apples-to-apples. (``envs`` is supplied per-call so the
    # multi-URDF path can split it across the batch.)
    shared = dict(
        vmin=args.vmin,
        vmax=args.vmax,
        x_upper=args.x_upper,
        forest_x_limit=args.forest_x_limit,
        dens_min=args.dens_min,
        dens_max=args.dens_max,
        init_vx=args.init_vx,
        minimal_progress=args.minimal_progress,
        # Fault injection — applied identically to baseline + Hebbian under CRN.
        freeze_distance=args.freeze_distance,
        break_left_distance=args.break_left_distance,
        break_right_distance=args.break_right_distance,
        break_left_loss_pct=args.break_left_loss,
        break_right_loss_pct=args.break_right_loss,
        seed=args.seed,
        crn=args.crn,
        verbose=args.verbose,
        capture_initial=True,
        return_arrays=True,
        save_plots=True,
    )

    if args.n_urdf is not None:
        _run_multi_urdf(
            run_dir=run_dir,
            baseline_genome=baseline_genome,
            genome_path=genome_path,
            genome_label=genome_label,
            out_dir=out_dir,
            shared=shared,
            args=args,
        )
        return

    # ---- Baseline: frozen WP1 actor (zero-rule genome, no plasticity) ---- #
    print("\n[validate] === Baseline rollout (WP1 actor, zero rules) ===")
    *_, base_extra = evaluation_hebbian(
        hebbian_run=str(run_dir),
        genome_path=str(baseline_genome),
        genome_idx=0,
        eval_dir=str(baseline_dir),
        envs=args.envs,
        **shared,
    )

    # ---- Hebbian: chosen best individual --------------------------------- #
    print("\n[validate] === Hebbian rollout (best individual) ===")
    *_, hebb_extra = evaluation_hebbian(
        hebbian_run=str(run_dir),
        genome_path=str(genome_path),
        genome_idx=args.genome_idx,
        eval_dir=str(hebbian_dir),
        envs=args.envs,
        **shared,
    )

    # ---- CRN verification ------------------------------------------------ #
    ok, report = _compare_initial_conditions(
        base_extra.get("initial_conditions"),
        hebb_extra.get("initial_conditions"),
    )
    print("\n" + report)
    report_path = out_dir / "initial_conditions.txt"
    report_path.write_text(
        f"run: {run_dir}\ngenome: {genome_path} ({genome_label})\n"
        f"seed: {args.seed}  crn: {args.crn}  envs: {args.envs}  "
        f"fault: {_fault_desc(args)}\n\n{report}\n"
    )
    print(f"[validate] Saved {report_path}")

    # ---- Overlay plots --------------------------------------------------- #
    title = f"{run_dir.name} — baseline (WP1) vs Hebbian [{genome_label}]"
    _plot_overlay(base_extra, hebb_extra, out_dir, title=title)

    # ---- Summary --------------------------------------------------------- #
    summary = "\n".join([
        _fmt_summary("baseline (WP1)", base_extra),
        _fmt_summary(f"hebbian[{genome_label}]", hebb_extra),
    ])
    print("\n[validate] Summary:\n" + summary)
    (out_dir / "summary.txt").write_text(
        f"run: {run_dir}\ngenome: {genome_path} ({genome_label})\n"
        f"seed: {args.seed}  crn: {args.crn}  envs: {args.envs}  "
        f"vmin: {args.vmin}  vmax: {args.vmax}  x_upper: {args.x_upper}  "
        f"dens_max: {args.dens_max}  fault: {_fault_desc(args)}\n\n{summary}\n"
    )

    if not ok:
        print(
            "\n[validate][WARNING] CRN check failed: the two rollouts did NOT "
            "fly identical initial conditions. The comparison is still "
            "informative but not strictly variance-reduced."
        )


# ===========================================================================
#  Multi-URDF robustness sweep
# ===========================================================================
def _run_multi_urdf(
    *,
    run_dir: Path,
    baseline_genome: Path,
    genome_path: Path,
    genome_label: str,
    out_dir: Path,
    shared: Dict[str, Any],
    args: Any,
) -> None:
    """Evaluate baseline vs Hebbian over a batch of N freshly-sampled URDFs.

    ``--envs`` is split evenly (``envs_per = envs // N``). The forest pool +
    per-slot assignment is generated once (first rollout) and *injected* into
    every other rollout, so all N URDFs — and both controllers — fly the exact
    same set of forests. Per (URDF, controller) the seed + CRN are identical,
    so baseline and Hebbian see identical noise on every URDF.
    """
    from winged_drone_train.eval import evaluation_hebbian
    from general_policy.catalog import build_catalog

    n = int(args.n_urdf)
    if n < 1:
        raise ValueError("--n-urdf must be >= 1")
    envs_per = args.envs // n
    if envs_per < 1:
        raise ValueError(
            f"--envs {args.envs} too small to split across {n} URDFs "
            f"(envs_per = {envs_per})"
        )
    if args.envs % n:
        print(
            f"[validate][warn] --envs {args.envs} not divisible by --n-urdf {n}; "
            f"using {envs_per} envs/URDF ({envs_per * n} total)."
        )

    # ---- Sample N URDFs (fresh random morphologies) --------------------- #
    # include_standard_mydrone=True puts the default drone as the FIRST URDF
    # and samples the remaining N-1 randomly (so the default replaces one of
    # the N sampled drones, keeping the batch size at N).
    urdf_seed = args.urdf_seed if args.urdf_seed is not None else args.seed
    sampled_dir = out_dir / "sampled_urdfs"
    print(
        f"\n[validate] sampling {n} URDF(s) seed={urdf_seed} "
        f"include_default={args.include_default_urdf} -> {sampled_dir}"
    )
    urdf_paths = [
        str(p) for p in build_catalog(
            catalog_dir=sampled_dir,
            n=n,
            seed=urdf_seed,
            include_standard_mydrone=bool(args.include_default_urdf),
        )
    ]
    if len(urdf_paths) != n:
        print(
            f"[validate][warn] requested {n} URDFs but only {len(urdf_paths)} "
            f"unique morphologies were generated."
        )
    print(f"[validate] envs/URDF = {envs_per}  ({envs_per * len(urdf_paths)} total)")

    # ---- Per-URDF baseline + Hebbian rollouts under a shared forest ----- #
    shared_forest = None  # captured from the very first rollout, then injected
    base_extras: list = []
    hebb_extras: list = []
    crn_reports: list = []
    crn_all_ok = True

    for i, urdf in enumerate(urdf_paths):
        is_default = bool(args.include_default_urdf) and i == 0
        stem = "default_mydrone" if is_default else _safe_stem(urdf, f"urdf_{i}")
        tag = f"urdf_{i:02d}_{stem}"
        base_dir = out_dir / "per_urdf" / tag / "baseline"
        hebb_dir = out_dir / "per_urdf" / tag / "hebbian"
        print(f"\n[validate] ===== URDF {i + 1}/{len(urdf_paths)}: {tag} =====")

        need_capture = shared_forest is None
        print("[validate]   -> baseline rollout")
        *_, base_extra = evaluation_hebbian(
            hebbian_run=str(run_dir),
            genome_path=str(baseline_genome),
            genome_idx=0,
            eval_dir=str(base_dir),
            envs=envs_per,
            urdf_file=urdf,
            inject_forest=shared_forest,
            return_forest=need_capture,
            return_raw=True,
            return_traces=True,
            **shared,
        )
        if need_capture:
            shared_forest = base_extra.get("forest")
            if shared_forest is None or shared_forest[0] is None:
                raise RuntimeError("failed to capture the shared forest pool")

        print("[validate]   -> hebbian rollout")
        *_, hebb_extra = evaluation_hebbian(
            hebbian_run=str(run_dir),
            genome_path=str(genome_path),
            genome_idx=args.genome_idx,
            eval_dir=str(hebb_dir),
            envs=envs_per,
            urdf_file=urdf,
            inject_forest=shared_forest,
            return_forest=False,
            return_raw=True,
            return_traces=True,
            **shared,
        )

        ok, report = _compare_initial_conditions(
            base_extra.get("initial_conditions"),
            hebb_extra.get("initial_conditions"),
        )
        crn_all_ok = crn_all_ok and ok
        crn_reports.append(f"[{tag}]\n{report}")
        base_extras.append(base_extra)
        hebb_extras.append(hebb_extra)

    # ---- Aggregate across the batch ------------------------------------- #
    base_agg = _aggregate_extras(base_extras)
    hebb_agg = _aggregate_extras(hebb_extras)

    # Pooled per-controller plot sets (the same plots as a single-URDF run,
    # but over the whole batch) -> validation/{baseline,hebbian}/.
    print("\n[validate] rendering pooled per-controller plot sets")
    _plot_pooled_controller(
        base_extras, out_dir / "baseline",
        vmin=args.vmin, vmax=args.vmax, minimal_progress=args.minimal_progress,
    )
    _plot_pooled_controller(
        hebb_extras, out_dir / "hebbian",
        vmin=args.vmin, vmax=args.vmax, minimal_progress=args.minimal_progress,
    )

    # Direct overall comparison -> validation/overlay_*.png.
    title = (
        f"{run_dir.name} — baseline (WP1) vs Hebbian [{genome_label}] — "
        f"{len(urdf_paths)} URDFs (envs/URDF={envs_per})"
    )
    _plot_overlay(base_agg, hebb_agg, out_dir, title=title)

    # ---- CRN report ----------------------------------------------------- #
    report_path = out_dir / "initial_conditions.txt"
    report_path.write_text(
        f"run: {run_dir}\ngenome: {genome_path} ({genome_label})\n"
        f"n_urdf: {len(urdf_paths)}  envs/URDF: {envs_per}  seed: {args.seed}  "
        f"crn: {args.crn}  urdf_seed: {urdf_seed}  fault: {_fault_desc(args)}\n"
        f"all_urdf_crn_ok: {crn_all_ok}\n\n" + "\n\n".join(crn_reports) + "\n"
    )
    print(f"\n[validate] Saved {report_path}")

    # ---- Summary (per-URDF + pooled) ------------------------------------ #
    lines: list = []
    for i, (urdf, be, he) in enumerate(zip(urdf_paths, base_extras, hebb_extras)):
        is_default = bool(args.include_default_urdf) and i == 0
        stem = "default_mydrone" if is_default else _safe_stem(urdf, f"urdf_{i}")
        lines.append(f"-- urdf_{i:02d}_{stem} --")
        lines.append(_fmt_summary("  baseline", be))
        lines.append(_fmt_summary(f"  hebbian[{genome_label}]", he))
    lines.append("== POOLED (all URDFs) ==")
    lines.append(_fmt_summary("  baseline", base_agg))
    lines.append(_fmt_summary(f"  hebbian[{genome_label}]", hebb_agg))
    summary = "\n".join(lines)
    print("\n[validate] Summary:\n" + summary)
    (out_dir / "summary.txt").write_text(
        f"run: {run_dir}\ngenome: {genome_path} ({genome_label})\n"
        f"n_urdf: {len(urdf_paths)}  envs/URDF: {envs_per}  seed: {args.seed}  "
        f"crn: {args.crn}  urdf_seed: {urdf_seed}  "
        f"vmin: {args.vmin}  vmax: {args.vmax}  x_upper: {args.x_upper}  "
        f"dens_max: {args.dens_max}  fault: {_fault_desc(args)}\n\n{summary}\n"
    )

    if not crn_all_ok:
        print(
            "\n[validate][WARNING] CRN check failed on at least one URDF: the "
            "baseline and Hebbian rollouts did NOT fly identical initial "
            "conditions everywhere. See initial_conditions.txt."
        )


if __name__ == "__main__":
    main()
