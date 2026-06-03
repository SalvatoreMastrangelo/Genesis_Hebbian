"""
diag_chaos_floor_reproducibility — is the σ_rank "chaos floor" reducible?
========================================================================

SNR_ANALYSIS.md §4 measures a noise floor: a population of *bit-identical*
controllers, flown on the *same* forests with deterministic actions and aero
noise off, still spreads out (σ_rank > 0). The doc attributes this to
"GPU floating-point non-determinism (non-associative atomic reductions) amplified
by chaotic flight" and calls it irreducible.

That explanation is plausible but the doc never proves it. There are two
distinct mechanisms that both produce a per-slot spread, and they have opposite
implications for whether the floor can be removed:

  A. RUN-TO-RUN nondeterminism — GPU atomicAdd completion order depends on warp
     scheduling, so each evaluation resolves the non-associative sums in a
     different order. IRREDUCIBLE by CRN/seeding; only averages down as 1/√N.
     Signature: slot i's fitness DIFFERS between two identical re-runs.

  B. DETERMINISTIC per-slot symmetry-breaking — the per-slot spread comes from a
     FIXED effect (memory layout / reduction tiling that assigns slot i different
     lanes than slot j, or an unintended per-slot initial-state difference).
     REPRODUCIBLE run-to-run; potentially removable, and it does NOT average down
     over re-seeds (you'd have to re-randomize the slot↔scenario assignment).
     Signature: slot i's fitness is IDENTICAL between two identical re-runs.

This script settles A vs B. It builds P identical zero-rule genomes, pins the
forests (CRN) + speed grid, disables aero noise, runs deterministic actions, and
then evaluates the population TWICE with the SAME seed. The decisive statistic is
PER-SLOT reproducibility across the two runs:

    within_std        = std over the P slots in one run  (= the measured floor)
    across_run_delta  = |fitness_A[i] - fitness_B[i]|     (per slot, A vs B)

  across_run_delta ≈ 0  →  mechanism B  (deterministic; floor is NOT atomic noise)
  across_run_delta ~ within_std  →  mechanism A  (genuine nondeterminism; irreducible)

NOTE: aero noise MUST be off — its in-kernel Taichi RNG advances between the two
evaluate() calls and would masquerade as run-to-run nondeterminism (a false A).
Stochastic actions are off for the same reason. With both off, the only thing
that can differ between two same-seed runs is GPU scheduling (mechanism A).

This is a fitness-level test (the quantity that actually feeds σ_rank). The
step-by-step trajectory divergence table is in SNR_ANALYSIS.md §4.

Usage (laptop scale; needs a GPU + WP1 checkpoint, like measure_signal.py):

    PYTHONPATH=src .venv/bin/python -u tests/integration/diag_chaos_floor_reproducibility.py \
        --cfg src/WP2/experiments/batch_1/random_period_4/run.yaml \
        --cfg.checkpoint_path <wp1_actor.pt> \
        --cfg.checkpoint_config_path <wp1_config.yaml> \
        --cfg.catalog.num_urdfs 2 --cfg.catalog.refresh_urdfs_every 0 \
        --cfg.evaluation.num_eval_envs 256 --cfg.evaluation.num_eval_workers 1 \
        --cfg.cmaes.population_size 32 --repeats 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("GS_PARA_LEVEL", "4")
_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--repeats", type=int, default=3,
                    help="number of same-seed run-PAIRS (A,B) to average over")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="across-run |Δ| below this (relative to within-run std) ⇒ mechanism B")
    args, remaining = ap.parse_known_args()

    import numpy as np
    import torch
    from WP2.config import HebbianEvolutionConfig
    from WP2.evolve_cma import HebbianCMAES
    from WP2.utils import seed_everything
    from WP2.run import _configure_cache_root
    _configure_cache_root()

    cfg = HebbianEvolutionConfig.from_yaml(args.cfg)
    cfg.apply_cli_overrides(remaining)

    # ---- Pin everything that could differ between two runs, EXCEPT GPU scheduling.
    # Deterministic actions + aero noise OFF + shared forests (CRN). Under perfect
    # determinism + slot symmetry, all P identical genomes would score identically.
    cfg.evaluation.crn = True
    cfg.evaluation.stochastic = False
    cfg.evaluation.noise.aero_noise = False
    # Decode faithfully (matches measure_signal.py); eta is irrelevant for the
    # all-zero genome but we keep the plasticity path identical to production.
    cfg.hebbian.eta = 0.005
    cfg.hebbian.decay = 0.05
    cfg.hebbian.use_oja_coefficient = False
    cfg.hebbian.evolve_eta = False
    cfg.hebbian.evolve_decay = False

    ck = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    if "actor.4.weight" in sd:
        cfg.hebbian.num_actions = sd["actor.4.weight"].shape[0]
        cfg.hebbian.hidden_dim = sd["actor.4.weight"].shape[1]
    del ck, sd

    runner = HebbianCMAES(cfg)
    n = runner.n_genes
    zero = np.full(n, 0.5)  # 0.5 ⇒ zero Hebbian rule after decode (same as measure_signal)

    runner._build_env_once()
    if runner._env is None:
        print("[chaos] ERROR: env build failed"); sys.exit(1)
    env = runner._env

    P = cfg.cmaes.population_size
    # Total env slots: multi-scene env exposes ``E``; legacy single-URDF
    # WingedDroneEnv exposes ``num_envs``.
    E = int(getattr(env, "E", None) or env.num_envs)
    F = E // P
    solutions = [zero.copy() for _ in range(P)]  # P IDENTICAL controllers

    # Pin forests + speed grid so the F scenarios are shared across all P slots.
    fixed = torch.arange(F, device=cfg.device, dtype=torch.long).repeat(P)
    env._fixed_forest_ids = fixed
    env._eval_speed_grid = torch.linspace(
        float(cfg.evaluation.vmin), float(cfg.evaluation.vmax), F,
        device=cfg.device, dtype=torch.float32).repeat(P)
    if hasattr(env, "set_crn_enabled"):
        env.set_crn_enabled(True)
    else:
        env._crn_enabled = True

    # Materialize the pinned forests ONCE. We must NOT refresh between run A and
    # run B (that could regenerate geometry); _uh_reevaluate is designed to reuse
    # the existing forests, so the only thing that can differ A-vs-B is GPU
    # scheduling.
    env.refresh_forests()

    n_urdfs = len(runner._urdf_paths) if runner._urdf_paths else 1
    print(f"\n[chaos] {P} IDENTICAL zero-rule controllers | F={F} forests | "
          f"N={n_urdfs} URDFs | rollouts/indiv={n_urdfs * F}")
    print(f"[chaos] crn=ON  stochastic=OFF  aero_noise=OFF  seed={args.seed}")
    print(f"[chaos] If physics were perfectly deterministic AND slot-symmetric, "
          f"all {P} fitnesses would be identical.\n")

    def _run_once() -> np.ndarray:
        """One full population evaluation, reseeded identically each call.

        Uses the runner's own re-evaluation method (branches single/multi-URDF)
        which reuses the already-materialized forests — so it does NOT refresh
        geometry, and the only thing that can differ between two same-seed calls
        is GPU atomic/scheduling nondeterminism.
        """
        seed_everything(args.seed)
        return np.asarray(runner._uh_reevaluate(solutions), dtype=float)

    hdr = (f"{'pair':>4} | {'within_std_A':>12} {'within_std_B':>12} | "
           f"{'across Δ mean':>13} {'across Δ max':>12} | {'meanΔ/within':>12}")
    print(hdr); print("-" * len(hdr))

    within_stds, across_means, across_maxes, ratios = [], [], [], []
    all_runs = []  # every same-seed evaluation, for the direct per-slot σ-across-runs
    for r in range(max(1, args.repeats)):
        fa = _run_once()
        fb = _run_once()
        all_runs.extend([fa, fb])
        within_a = float(fa.std())
        within_b = float(fb.std())
        across = np.abs(fa - fb)
        across_max = float(across.max())
        across_mean = float(across.mean())
        within_mean = (within_a + within_b) / 2
        # Like-for-like: compare the MEAN run-to-run gap (a per-slot average) to
        # the within-run std (also a per-slot dispersion). Using max over slots
        # vs std would be apples-to-oranges and inflate the ratio.
        ratio = across_mean / within_mean if within_mean > 1e-12 else float("inf")
        within_stds.append(within_mean)
        across_means.append(across_mean)
        across_maxes.append(across_max)
        ratios.append(ratio)
        print(f"{r:>4} | {within_a:12.4f} {within_b:12.4f} | "
              f"{across_mean:13.4e} {across_max:12.4e} | {ratio:12.4f}", flush=True)

    runner._cleanup_env()

    mean_within = float(np.mean(within_stds))
    mean_across = float(np.mean(across_means))
    mean_across_max = float(np.mean(across_maxes))
    mean_ratio = float(np.mean(ratios))

    # Direct σ-across-runs: for each slot, std of its fitness over all (same-seed)
    # runs; then average over slots. This is the per-individual run-to-run noise
    # of the F-rollout-averaged fitness — i.e. the GPU-nondeterminism component of
    # σ_rank at this rollout count, measured with no distributional assumptions.
    runs = np.stack(all_runs, axis=0)              # (n_runs, P)
    per_slot_sigma = runs.std(axis=0, ddof=1)      # (P,)
    sigma_across_runs = float(per_slot_sigma.mean())
    fitness_scale = float(np.abs(runs).mean())

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  n same-seed runs pooled              : {runs.shape[0]}")
    print(f"  mean fitness (scale)                 : {runs.mean():.4f}")
    print(f"  σ ACROSS RUNS per slot (mean over P) : {sigma_across_runs:.4f}"
          f"   ({100.0 * sigma_across_runs / max(fitness_scale, 1e-9):.1f}% of |fitness|)")
    print(f"  mean within-run σ_rank (the floor)   : {mean_within:.4f}")
    print(f"  mean across-run |Δ| (mean over slots): {mean_across:.4e}")
    print(f"  mean across-run |Δ| (max  over slots): {mean_across_max:.4e}")
    print(f"  ratio  meanΔ / within-σ_rank         : {mean_ratio:.4f}")
    print()

    if mean_within < 1e-9:
        print("  → NO FLOOR under these settings: identical controllers scored identically")
        print("    within a run. σ_rank ≈ 0 here, so there is nothing to attribute. (Check")
        print("    that the eval actually ran and forests are non-degenerate.)")
    elif mean_ratio < args.tol:
        print("  → MECHANISM B (deterministic per-slot symmetry-breaking).")
        print("    Each slot is bit-reproducible run-to-run, yet slots differ from each")
        print("    other. The floor is NOT GPU atomic/scheduling nondeterminism — it is a")
        print("    FIXED per-slot effect (memory layout / reduction tiling, or an")
        print("    unintended per-slot initial-state difference).")
        print("    IMPLICATIONS: it will NOT average down over re-seeds; it averages only")
        print("    if you re-randomize the slot↔scenario assignment. It may be removable")
        print("    by fixing the layout. The doc's '1/√N averaging is the only lever' claim")
        print("    would be WRONG for this component — investigate the deterministic source.")
    elif mean_ratio < 0.1:
        print("  → MIXED. A small run-to-run component exists but most of the floor is")
        print("    reproducible per-slot (mechanism B dominates). Investigate the")
        print("    deterministic source before assuming the floor is irreducible.")
    else:
        print("  → MECHANISM A confirmed (genuine run-to-run nondeterminism).")
        print("    Slot fitnesses change between two identical same-seed runs by an amount")
        print("    comparable to the within-run spread. This is GPU atomicAdd/scheduling")
        print("    nondeterminism: CRN, deterministic actions, and aero-off cannot remove")
        print("    it. It only averages down as 1/√(rollouts) — matching SNR_ANALYSIS.md §4.")
    print()
    print("  Mechanism A ⇒ irreducible (more rollouts only).  Mechanism B ⇒ a deterministic")
    print("  bug/artifact worth removing.  Re-run with --repeats higher to tighten the call.")


if __name__ == "__main__":
    main()
