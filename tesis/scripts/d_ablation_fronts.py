"""D ablation on the Pareto fronts of the co-design runs (POSSIBLE_EXPERIMENTS item 1).

For every co-design run given on the command line, the bodies of its last exam front are rebuilt
from their genomes and flown on the exam course (300 m, 1 -> 5.5 trees/m, 480 forests, seed 0:
identical layouts for every body, controller and run) under four controllers at once:
  zero   the frozen generalist (all coefficients at zero),
  best   the run's best set of rules,
  noD    the same rules with D set to zero (activity-dependent part alone),
  onlyD  D alone, with A, B, C set to zero (the static retuning alone).
Same machinery as WP2_Outer_Loop.transfer_eval (one MultiSceneEvalEnv per chunk of bodies), but the
four genomes share every scene, so the scenes are built once instead of four times.

Output per run: logs/remote/outer_nsga/d_ablation/<run>/ablation.csv (one row per body and
controller, mean over the 480 forests and standard error) + summary.json; overall summary in
tesis/images/06_Results_and_Experiments/d_ablation/d_ablation_fronts.json.

Run from the repository root, in the local Genesis image:
    docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD":/workspace/bind -e PYTHONPATH=/workspace/bind/src -w /workspace/bind \
        mygenesis:latest python tesis/scripts/d_ablation_fronts.py <run_dir> [...] \
        [--n-forests 480] [--chunk 8] [--forest-seed 0]
"""
import argparse, copy, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("GS_PARA_LEVEL", "4")
import genesis as gs
from WP2_Outer_Loop.transfer_eval import (
    _METRICS, _best_genome_from_run, _chunks, _fit_genome_to_checkpoint, _gene_columns,
    apply_forest_settings, load_controller_config, select_front, zero_rules_genome,
)
from WP2_Outer_Loop.urdf_population import materialize_urdfs
from WP2.evaluate import _build_multi_urdf_env, evaluate_population_multi_urdf
from WP2.frozen_actor import load_frozen_actor
from WP1.config import RunConfig

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--n-forests", type=int, default=480)
ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--forest-seed", type=int, default=0)
ap.add_argument("--x-upper", type=float, default=300.0)
ap.add_argument("--dens-min", type=float, default=1.0)
ap.add_argument("--dens-max", type=float, default=5.5)
ap.add_argument("--out-root", default="logs/remote/outer_nsga/d_ablation")
ap.add_argument("--summary", default="tesis/images/06_Results_and_Experiments/d_ablation/d_ablation_fronts.json")
args = ap.parse_args()

VARIANTS = ("zero", "best", "noD", "onlyD")
summary_path = Path(args.summary)
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary = json.load(open(summary_path)) if summary_path.is_file() else {}

for run in args.runs:
    run_dir = Path(run)
    if run_dir.name.startswith("outer_") and not (run_dir / "results").is_dir():
        subs = sorted(p for p in run_dir.iterdir() if (p / "results").is_dir())
        run_dir = subs[-1]
    run_name = run_dir.parent.name if run_dir.parent.name.startswith("outer_") else run_dir.name
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    t_run = time.time()

    cfg, kind = load_controller_config(run_dir / "reproducibility" / "config.yaml", REPO, None)
    cfg.device = "cuda:0"
    _fit_genome_to_checkpoint(cfg)
    forest = apply_forest_settings(cfg, {"x_upper": args.x_upper, "dens_min": args.dens_min, "dens_max": args.dens_max})
    front = select_front(run_dir / "results" / "pareto_front.csv", "last", dedup=True)
    best, best_src = _best_genome_from_run(run_dir)
    zero = np.asarray(zero_rules_genome(cfg), dtype=float)
    n = cfg.hebbian.num_actions * cfg.hebbian.hidden_dim
    assert best.shape[0] == 4 * n == zero.shape[0], (best.shape, zero.shape, n)
    noD = best.copy(); noD[3 * n:4 * n] = zero[3 * n:4 * n]
    onlyD = best.copy(); onlyD[0:3 * n] = zero[0:3 * n]
    genomes = [zero, best, noD, onlyD]
    P = len(genomes)
    print(f"[ablation] {run_name}: {len(front)} front bodies (outer gen {int(front['outer_gen'].max())}), "
          f"best rules from {best_src}, forest {json.dumps(forest, sort_keys=True)}", flush=True)

    genes = _gene_columns(front)
    body_genomes = front[genes].to_numpy(dtype=float).tolist()
    if not gs._initialized:
        gs.init(backend=gs.gpu, logging_level="warning")
    urdf_paths = materialize_urdfs(body_genomes, out_dir / "urdfs")
    wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)
    model, last_layer, num_actions, hidden_dim = load_frozen_actor(cfg.checkpoint_path, cfg.checkpoint_config_path, device=cfg.device)
    model_and_layer = (model, last_layer, num_actions, hidden_dim)

    rows = []
    groups = _chunks(list(range(len(urdf_paths))), args.chunk)
    try:
        for ci, idxs in enumerate(groups):
            if not gs._initialized:
                gs.init(backend=gs.gpu, logging_level="warning")
            paths = [urdf_paths[i] for i in idxs]
            chunk_cfg = copy.deepcopy(cfg)
            chunk_cfg.evaluation.num_eval_envs = len(paths) * P * args.n_forests
            chunk_cfg.evaluation.run_baseline = False
            chunk_cfg.evaluation.refresh_forests_per_generation = False
            chunk_cfg.catalog.path = ""
            chunk_cfg.catalog.num_urdfs = len(paths)
            chunk_cfg.catalog.force_multi_urdf = True
            t0 = time.time()
            env = _build_multi_urdf_env(paths, chunk_cfg, wp1_cfg, chunk_cfg.device,
                                        num_envs_per_drone=P * args.n_forests, num_workers=1)
            try:
                torch.manual_seed(args.forest_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.forest_seed)
                env.refresh_forests()
                _fit, metrics = evaluate_population_multi_urdf(
                    genomes, chunk_cfg, model_and_layer, wp1_cfg,
                    urdf_paths=paths, existing_env=(env, paths), verbose=False,
                )
            finally:
                try:
                    gs.destroy()
                except Exception as exc:  # noqa: BLE001
                    print(f"[ablation] gs.destroy() failed: {exc}")
            for local, global_i in enumerate(idxs):
                src = front.iloc[global_i]
                for p, name in enumerate(VARIANTS):
                    row = {"run": run_name, "urdf_idx": int(src["urdf_idx"]), "urdf_file": Path(urdf_paths[global_i]).name,
                           "outer_gen": int(src["outer_gen"]), "variant": name}
                    for col, pu_key, ps_key in _METRICS:
                        row[col] = float(np.asarray(metrics[pu_key])[local, p])
                        slots = np.asarray(metrics[ps_key])[local, p]
                        row[f"{col}_se"] = float(np.std(slots, ddof=1) / np.sqrt(len(slots))) if len(slots) > 1 else 0.0
                    for col in ("progress_m", "cost_of_transport"):
                        if col in src.index:
                            row[f"archived_{col}"] = float(src[col])
                    rows.append(row)
            done = pd.DataFrame(rows[-len(idxs) * P:])
            msg = "  ".join(f"{v}: {done[done.variant == v].progress_m.mean():.1f} m / CoT {done[done.variant == v].cost_of_transport.mean():.3f}" for v in VARIANTS)
            print(f"[ablation] {run_name} chunk {ci + 1}/{len(groups)} ({len(paths)} bodies, {time.time() - t0:.0f} s): {msg}", flush=True)
    finally:
        if gs._initialized:
            try:
                gs.destroy()
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation.csv", index=False)
    res = {"run": run_name, "run_dir": str(run_dir), "n_bodies": int(len(front)), "outer_gen": int(front["outer_gen"].max()),
           "n_forests": args.n_forests, "forest_seed": args.forest_seed, "forest": forest, "best_source": best_src,
           "seconds": round(time.time() - t_run), "variants": {}}
    base = df[df.variant == "zero"].set_index("urdf_idx")
    for v in VARIANTS:
        sub = df[df.variant == v].set_index("urdf_idx").loc[base.index]
        res["variants"][v] = {
            "progress_mean": float(sub.progress_m.mean()), "cot_mean": float(sub.cost_of_transport.mean()),
            "fitness_mean": float(sub.fitness.mean()), "crash_mean": float(sub.crash_rate.mean()), "velocity_mean": float(sub.velocity.mean()),
            "d_progress_vs_zero_mean": float((sub.progress_m - base.progress_m).mean()),
            "d_progress_vs_zero_std": float((sub.progress_m - base.progress_m).std(ddof=1)),
            "d_cot_vs_zero_mean": float((sub.cost_of_transport - base.cost_of_transport).mean()),
            "d_cot_vs_zero_std": float((sub.cost_of_transport - base.cost_of_transport).std(ddof=1)),
            "d_fitness_vs_zero_mean": float((sub.fitness - base.fitness).mean()),
            "bodies_better_progress": int((sub.progress_m > base.progress_m).sum()),
            "bodies_lower_cot": int((sub.cost_of_transport < base.cost_of_transport).sum()),
        }
    json.dump(res, open(out_dir / "summary.json", "w"), indent=1)
    summary[run_name] = res
    json.dump(summary, open(summary_path, "w"), indent=1)
    print(f"[ablation] {run_name} done in {res['seconds']} s: " + "  ".join(
        f"{v}: dP {res['variants'][v]['d_progress_vs_zero_mean']:+.1f} m, dCoT {res['variants'][v]['d_cot_vs_zero_mean']:+.4f}" for v in VARIANTS), flush=True)

print("[ablation] all done")
