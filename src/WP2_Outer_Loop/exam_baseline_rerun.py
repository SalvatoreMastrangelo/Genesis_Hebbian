"""
Offline reconstruction of the exam baseline (the Pareto star).
==============================================================

Runs that predate ``outer.exam_baseline`` finished without
``results/outer_exam_baseline.csv``, so ``pareto_plots`` has no reference
point and draws no star on exam-scored fronts (``_load_exam_baseline``
returns ``None``). This module re-measures that point after the fact.

What it measures is exactly what the in-run hook would have written: the
**standard mydrone** (``validation.validation_catalog`` empty → the URDF
built from ``STANDARD_MYDRONE_GENOME``) flown by the run's own frozen
generalist with **zero Hebbian rules and zero decay**, over forests drawn
from the run's **exam** distribution (``outer.exam_forest`` overrides applied
to a freshly built env). Speed grid, noise toggles, stochastic sampling and
the 60 s episode all come from the run's saved config, so the point is
commensurable with the ``obj_source == "exam"`` rows it is plotted against.

Differences from the in-run hook, both deliberate:

* **Sample size.** The hook reused the live validation env (``n_val_envs``
  slots, one pass per phase). Here the sample is ``--n-forests`` (default
  16384) split into equal passes, each on freshly generated layouts.
* **One row, not one per phase.** The reference does not depend on the outer
  generation — same drone, same controller, same forest distribution — so a
  single aggregate row is written with ``outer_gen = -1`` (sentinel: "whole
  run, not phase-specific"). ``pareto_plots`` averages the CSV for the star
  and draws a constant baseline line on the champion curves.

The first pass after an env build is discarded (``--warmup``): a fresh scene
starts with cold Taichi aero state (the ``_thr_flt`` throttle filter starts
at 0), which biases that rollout pessimistic.

Because the reference depends only on (checkpoint, forest settings), one
measurement can serve several run dirs. Every extra run dir is checked
against the first for eval-relevant equality before it is written to, so a
mismatched run cannot silently inherit someone else's star.

Usage
-----
    PYTHONPATH=src python -m WP2_Outer_Loop.exam_baseline_rerun RUN_DIR [RUN_DIR ...]
        [--n-forests 16384] [--chunk 4096] [--warmup 1] [--device cuda:0]
        [--repo-root PATH] [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from WP2_Outer_Loop.config import OuterNSGA2Config
from WP2_Outer_Loop.nsga_cma import _EXAM_BASELINE_COLS


# Marks the aggregate row as a whole-run reference rather than a phase score.
WHOLE_RUN_GEN = -1

# Provenance columns appended after the metric columns. `_load_exam_baseline`
# and `_load_exam_baseline_series` select columns by name, so extras are inert.
_PROVENANCE_COLS = ("source", "n_passes", "warmup_passes", "exam_overrides")

_SOURCE_TAG = "exam_baseline_rerun"


# ----------------------------------------------------------------------------
#  Config loading
# ----------------------------------------------------------------------------

def _rebase_container_path(path: str, repo_root: Path) -> str:
    """Map a path recorded inside the cluster container onto this checkout.

    Cluster runs bind-mount the repo at ``/workspace/bind``, so their configs
    record e.g. ``/workspace/bind/src/WP2_Outer_Loop/experiments/...``.
    Rewrite anything at or below ``src/`` onto ``repo_root``; leave paths that
    already resolve alone.
    """
    if not path:
        return path
    if os.path.exists(path):
        return path
    marker = "/src/"
    idx = path.find(marker)
    if idx == -1:
        return path
    candidate = repo_root / path[idx + 1:]
    return str(candidate) if candidate.exists() else path


def load_run_config(run_dir: Path, repo_root: Path) -> OuterNSGA2Config:
    """The run's saved config, with container paths rebased onto this repo."""
    cfg_path = run_dir / "reproducibility" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No saved config at {cfg_path}")
    cfg = OuterNSGA2Config.from_yaml(cfg_path)
    cfg.checkpoint_path = _rebase_container_path(cfg.checkpoint_path, repo_root)
    cfg.checkpoint_config_path = _rebase_container_path(
        cfg.checkpoint_config_path, repo_root
    )
    cfg.baseline_checkpoint_path = _rebase_container_path(
        cfg.baseline_checkpoint_path, repo_root
    )
    cfg.baseline_checkpoint_config_path = _rebase_container_path(
        cfg.baseline_checkpoint_config_path, repo_root
    )
    return cfg


def reference_checkpoint(cfg: OuterNSGA2Config) -> Tuple[str, str]:
    """(checkpoint, wp1 config) of the zero-rules reference actor.

    Mirrors ``_evaluate_reference_actor``: the dedicated baseline checkpoint
    when the run configured one, else the run's own frozen actor.
    """
    if cfg.baseline_checkpoint_path:
        return cfg.baseline_checkpoint_path, cfg.baseline_checkpoint_config_path
    return cfg.checkpoint_path, cfg.checkpoint_config_path


def eval_signature(cfg: OuterNSGA2Config) -> Dict:
    """Everything that changes what the reference point measures.

    Two runs sharing this signature would have produced the same star, so one
    measurement may be written to both. Deliberately excludes sizing knobs
    (``num_eval_envs``, worker/GPU counts, population size) and the run's
    seed: those change the sample, not the distribution.
    """
    import hashlib

    ckpt, ckpt_cfg = reference_checkpoint(cfg)
    ckpt_hash = ""
    if ckpt and os.path.exists(ckpt):
        ckpt_hash = hashlib.md5(Path(ckpt).read_bytes()).hexdigest()
    ev = cfg.evaluation
    return {
        "checkpoint_md5": ckpt_hash,
        "wp1_config": Path(ckpt_cfg).read_text() if os.path.exists(ckpt_cfg) else "",
        "forest": asdict(cfg.forest),
        "exam_overrides": exam_overrides(cfg),
        "vmin": ev.vmin,
        "vmax": ev.vmax,
        "stochastic": ev.stochastic,
        "num_eval_episodes": ev.num_eval_episodes,
        "x_upper": ev.x_upper,
        "dens_min": ev.dens_min,
        "dens_max": ev.dens_max,
        "noise": asdict(ev.noise) if hasattr(ev.noise, "__dataclass_fields__")
                 else dict(ev.noise),
        "validation_catalog": (cfg.validation.validation_catalog or "").strip(),
    }


def exam_overrides(cfg: OuterNSGA2Config) -> Dict:
    """The exam's forest overrides ({} when the exam flew nominal forests)."""
    exam_forest = getattr(cfg.outer, "exam_forest", None)
    return exam_forest.overrides() if exam_forest is not None else {}


def _validation_is_standard_drone(cfg: OuterNSGA2Config) -> bool:
    """True when the run's held-out validation env held the standard mydrone.

    Same gate ``pareto_plots._validation_ran_on_standard_drone`` applies before
    drawing the star — measuring a point the plot would refuse to show is a
    waste, and writing one for a custom-catalog run would mislabel the drone.
    """
    if not cfg.validation.enable:
        return False
    vc = (cfg.validation.validation_catalog or "").strip()
    return not vc or vc.lower() == "none"


# ----------------------------------------------------------------------------
#  Measurement
# ----------------------------------------------------------------------------

def reference_genome(cfg: OuterNSGA2Config) -> np.ndarray:
    """Zero-plasticity genome: ABCD = 0 (mid-range on symmetric bounds), and
    an all-zero decay block when decay is evolved. Mirrors
    ``HebbianCMAES._evaluate_reference_actor``.
    """
    genome = np.full(cfg.hebbian_genome_dim(), 0.5)
    if cfg.hebbian.evolve_decay:
        n_weights = cfg.hebbian.num_actions * cfg.hebbian.hidden_dim
        start = 4 * cfg.hebbian.abcd_block_size()
        genome[start: start + n_weights] = 0.0
    return genome


def measure_exam_baseline(
    cfg: OuterNSGA2Config,
    n_forests: int,
    chunk: int,
    warmup: int = 1,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Fly the standard mydrone with zero rules over ``n_forests`` exam forests.

    The env is built once with ``chunk`` slots and re-flown over freshly
    generated layouts, ``warmup`` discarded passes first. Equal pass sizes make
    the pass mean the overall per-forest mean.

    Returns ``(means, per_pass)`` — metric keys are the ``_EXAM_BASELINE_COLS``
    result keys (``fitness``, ``velocity``, ``progress``, ``crash_rate``,
    ``cot``, ``v_deviation``).
    """
    import genesis as gs  # heavy import: only when actually measuring
    import torch
    from WP1.config import RunConfig
    from WP2.evaluate import _build_multi_urdf_env, evaluate_population_multi_urdf
    from WP2.frozen_actor import last_actor_linear_key, load_frozen_actor
    from winged_drone_train.defaults import default_mydrone_urdf_path

    if chunk <= 0 or n_forests <= 0:
        raise ValueError("--n-forests and --chunk must be positive")
    n_passes = max(1, int(round(n_forests / chunk)))
    if n_passes * chunk != n_forests:
        raise ValueError(
            f"--n-forests {n_forests} must be a whole multiple of --chunk "
            f"{chunk} so every pass carries equal weight in the mean"
        )

    ref_cfg = cfg  # caller owns the copy; we mutate the reference-actor knobs
    ref_cfg.hebbian.decay = 0.0
    if device:
        ref_cfg.device = device
    ckpt, ckpt_cfg = reference_checkpoint(ref_cfg)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Reference checkpoint not found: {ckpt}")
    if not os.path.exists(ckpt_cfg):
        raise FileNotFoundError(f"Reference WP1 config not found: {ckpt_cfg}")

    # Re-infer the last-layer shape so the genome length matches this
    # checkpoint's architecture (same as the in-run reference path).
    _ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
    _sd = _ckpt.get("model_state_dict", _ckpt) if isinstance(_ckpt, dict) else _ckpt
    _last_key = last_actor_linear_key(_sd)
    if _last_key is not None:
        ref_cfg.hebbian.num_actions = _sd[_last_key].shape[0]
        ref_cfg.hebbian.hidden_dim = _sd[_last_key].shape[1]
    del _ckpt, _sd

    gs.init(backend=gs.gpu, logging_level="warning")

    wp1_cfg = RunConfig.from_yaml(ckpt_cfg)
    model, last_layer, num_actions, hidden_dim = load_frozen_actor(
        ckpt, ckpt_cfg, device=ref_cfg.device
    )
    model_and_layer = (model, last_layer, num_actions, hidden_dim)

    urdf_paths = [str(default_mydrone_urdf_path())]
    if verbose:
        print(f"[exam_baseline] Standard mydrone: {urdf_paths[0]}")
        print(f"[exam_baseline] Reference actor:  {ckpt}")

    t0 = time.time()
    env = _build_multi_urdf_env(
        urdf_paths, ref_cfg, wp1_cfg, ref_cfg.device,
        num_envs_per_drone=chunk, num_workers=1,
    )
    if verbose:
        print(f"[exam_baseline] Built env with {chunk} slots "
              f"in {time.time() - t0:.1f}s")

    overrides = exam_overrides(ref_cfg)
    if overrides:
        env.apply_forest_overrides(overrides)
        print(f"[exam_baseline] Exam forest overrides: {overrides}")
    else:
        print("[exam_baseline] No exam forest overrides — the exam flew the "
              "inner-loop forest distribution")

    genome = reference_genome(ref_cfg)
    per_pass: List[Dict[str, float]] = []
    try:
        for p in range(warmup + n_passes):
            env.refresh_forests()
            t0 = time.time()
            fitnesses, metrics = evaluate_population_multi_urdf(
                [genome], ref_cfg, model_and_layer, wp1_cfg,
                urdf_paths=urdf_paths, existing_env=(env, urdf_paths),
                verbose=False,
            )
            result = {
                "fitness":     float(fitnesses[0]),
                "velocity":    float(metrics["velocities"][0]),
                "progress":    float(metrics["progresses"][0]),
                "crash_rate":  float(metrics["crash_flags"][0]),
                "cot":         float(metrics["cots"][0]),
                "v_deviation": float(metrics["v_deviations"][0]),
            }
            tag = "warmup (discarded)" if p < warmup else f"pass {p - warmup + 1}/{n_passes}"
            if verbose:
                print(f"[exam_baseline] {tag}: {time.time() - t0:.1f}s  "
                      + "  ".join(f"{k}={v:.4g}" for k, v in result.items()))
            if p >= warmup:
                per_pass.append(result)
    finally:
        try:
            gs.destroy()  # same teardown the run uses (`_cleanup_env`)
        except Exception as exc:
            print(f"[exam_baseline] gs.destroy() failed: {exc}")

    means = {k: float(np.mean([r[k] for r in per_pass])) for k in per_pass[0]}
    return means, per_pass


# ----------------------------------------------------------------------------
#  CSV output
# ----------------------------------------------------------------------------

def write_exam_baseline_csv(
    run_dir: Path,
    means: Dict[str, float],
    per_pass: List[Dict[str, float]],
    n_forests: int,
    warmup: int,
    overrides: Dict,
) -> Tuple[Path, Path]:
    """Write the aggregate row (+ a per-pass sibling) into ``run_dir/results``.

    The aggregate CSV carries the exact header the in-run hook writes, so
    ``pareto_plots`` reads it unchanged; the provenance columns after it mark
    the row as an offline reconstruction.
    """
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [col for col, _key in _EXAM_BASELINE_COLS]

    csv_path = results_dir / "outer_exam_baseline.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["outer_gen", "inner_gen", "n_forests"] + metric_cols
                   + list(_PROVENANCE_COLS))
        w.writerow(
            [WHOLE_RUN_GEN, -1, n_forests]
            + [f"{means[key]:.6g}" for _col, key in _EXAM_BASELINE_COLS]
            + [_SOURCE_TAG, len(per_pass), warmup, json.dumps(overrides,
                                                              sort_keys=True)]
        )

    passes_path = results_dir / "outer_exam_baseline_passes.csv"
    with open(passes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pass", "n_forests"] + metric_cols)
        for i, r in enumerate(per_pass):
            w.writerow([i, n_forests // max(1, len(per_pass))]
                       + [f"{r[key]:.6g}" for _col, key in _EXAM_BASELINE_COLS])
    return csv_path, passes_path


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def _describe(means: Dict[str, float], per_pass: List[Dict[str, float]]) -> str:
    parts = []
    for col, key in _EXAM_BASELINE_COLS:
        spread = float(np.std([r[key] for r in per_pass])) if len(per_pass) > 1 else 0.0
        parts.append(f"{col}={means[key]:.4g}±{spread:.3g}")
    return "  ".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-measure the standard-mydrone exam baseline (Pareto "
                    "star) for runs that predate outer.exam_baseline.")
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="Run directories. The first supplies the measurement "
                         "settings; the rest must match it and receive a copy.")
    ap.add_argument("--n-forests", type=int, default=16384,
                    help="Total forests to fly (default 16384).")
    ap.add_argument("--chunk", type=int, default=4096,
                    help="Env slots per pass; must divide --n-forests.")
    ap.add_argument("--warmup", type=int, default=1,
                    help="Discarded passes before measuring (cold aero state).")
    ap.add_argument("--device", default=None, help="Override cfg.device.")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="Checkout to rebase container paths onto "
                         "(default: this file's repo).")
    ap.add_argument("--force", action="store_true",
                    help="Write to run dirs whose settings differ from the "
                         "first (normally refused).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Check configs and print the plan; measure nothing.")
    args = ap.parse_args(argv)

    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    run_dirs = [Path(d).resolve() for d in args.run_dirs]

    cfgs = [load_run_config(d, repo_root) for d in run_dirs]
    head_sig = eval_signature(cfgs[0])
    mismatched = [
        d for d, c in zip(run_dirs[1:], cfgs[1:])
        if eval_signature(c) != head_sig
    ]
    if mismatched:
        print("[exam_baseline] These run dirs do not share the first run's "
              "eval settings, so one measurement cannot serve them:")
        for d in mismatched:
            print(f"  - {d}")
        if not args.force:
            print("[exam_baseline] Refusing to write. Run them separately, or "
                  "pass --force if you are sure.")
            return 1
        print("[exam_baseline] --force given: writing anyway.")

    not_standard = [d for d, c in zip(run_dirs, cfgs)
                    if not _validation_is_standard_drone(c)]
    if not_standard:
        print("[exam_baseline] These runs did not hold the standard mydrone in "
              "their validation env — pareto_plots would refuse to draw the "
              "star, so they are skipped:")
        for d in not_standard:
            print(f"  - {d}")
        keep = [(d, c) for d, c in zip(run_dirs, cfgs)
                if _validation_is_standard_drone(c)]
        if not keep:
            return 1
        run_dirs = [d for d, _ in keep]
        cfgs = [c for _, c in keep]

    overrides = exam_overrides(cfgs[0])
    print(f"[exam_baseline] {len(run_dirs)} run dir(s), "
          f"{args.n_forests} forests in {args.n_forests // args.chunk} passes "
          f"of {args.chunk} (+{args.warmup} warmup)")
    print(f"[exam_baseline] Exam forest overrides: {overrides or '(none)'}")
    for d in run_dirs:
        print(f"    → {d / 'results' / 'outer_exam_baseline.csv'}")
    if args.dry_run:
        print("[exam_baseline] --dry-run: stopping before measurement.")
        return 0

    means, per_pass = measure_exam_baseline(
        cfgs[0], n_forests=args.n_forests, chunk=args.chunk,
        warmup=args.warmup, device=args.device,
    )
    print(f"[exam_baseline] RESULT over {args.n_forests} forests: "
          f"{_describe(means, per_pass)}")

    for d in run_dirs:
        csv_path, passes_path = write_exam_baseline_csv(
            d, means, per_pass, args.n_forests, args.warmup, overrides
        )
        print(f"[exam_baseline] Wrote {csv_path}")
        print(f"[exam_baseline]   and {passes_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
