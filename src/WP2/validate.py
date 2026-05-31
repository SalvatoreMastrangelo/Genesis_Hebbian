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
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed, identical for both rollouts → CRN (default 0)")
    p.add_argument("--no-crn", dest="crn", action="store_false",
                   help="disable CRN sharing of per-slot DR draws (still same seed)")
    p.set_defaults(crn=True)
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

    # Knobs shared by both rollouts — identical seed + CRN are what make the
    # comparison apples-to-apples.
    shared = dict(
        envs=args.envs,
        vmin=args.vmin,
        vmax=args.vmax,
        x_upper=args.x_upper,
        forest_x_limit=args.forest_x_limit,
        dens_min=args.dens_min,
        dens_max=args.dens_max,
        init_vx=args.init_vx,
        minimal_progress=args.minimal_progress,
        seed=args.seed,
        crn=args.crn,
        capture_initial=True,
        return_arrays=True,
        save_plots=True,
    )

    # ---- Baseline: frozen WP1 actor (zero-rule genome, no plasticity) ---- #
    print("\n[validate] === Baseline rollout (WP1 actor, zero rules) ===")
    *_, base_extra = evaluation_hebbian(
        hebbian_run=str(run_dir),
        genome_path=str(baseline_genome),
        genome_idx=0,
        eval_dir=str(baseline_dir),
        **shared,
    )

    # ---- Hebbian: chosen best individual --------------------------------- #
    print("\n[validate] === Hebbian rollout (best individual) ===")
    *_, hebb_extra = evaluation_hebbian(
        hebbian_run=str(run_dir),
        genome_path=str(genome_path),
        genome_idx=args.genome_idx,
        eval_dir=str(hebbian_dir),
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
        f"seed: {args.seed}  crn: {args.crn}  envs: {args.envs}\n\n{report}\n"
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
        f"dens_max: {args.dens_max}\n\n{summary}\n"
    )

    if not ok:
        print(
            "\n[validate][WARNING] CRN check failed: the two rollouts did NOT "
            "fly identical initial conditions. The comparison is still "
            "informative but not strictly variance-reduced."
        )


if __name__ == "__main__":
    main()
