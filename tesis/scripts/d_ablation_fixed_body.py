"""D ablation on a fixed body (thesis 6.1.2): which part of the evolved rules carries the gain?

For each inner-loop run given on the command line (rules evolved on the reference drone alone),
the run's best set of rules is decomposed into four controllers that fly the SAME forests:
  zero   all coefficients at zero (the frozen base controller),
  best   the evolved rules as they are,
  best_noD    the evolved A, B, C with D set to zero (the activity-dependent part alone),
  best_onlyD  the evolved D with A, B, C set to zero (the static retuning alone: dW -> eta D / lambda),
and, when the final CMA-ES state can be loaded, the same three for the mean of the search (`mean`,
`mean_noD`, `mean_onlyD`: the un-cursed centre of the distribution, not a lucky sample).
Every controller flies F forests of the run's search course (fresh forests for each of R repeats, the
same forests for all controllers within a repeat), with the noise and the sampled actions of the run.
Output: one JSON per run next to the run's results, and a summary JSON in
tesis/images/06_Results_and_Experiments/inner_loop/d_ablation_fixed_body.json.

Run from the repository root, in the local Genesis image:

    docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
        -v "$PWD":/workspace/bind -e PYTHONPATH=/workspace/bind/src -w /workspace/bind \
        mygenesis:latest python tesis/scripts/d_ablation_fixed_body.py <run_dir> [<run_dir> ...] \
        [--forests 2048] [--repeats 5]
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("GS_PARA_LEVEL", "4")
import genesis as gs
from WP2.config import HebbianEvolutionConfig
from WP2.evaluate import _build_env, evaluate_population_cma_batched
from WP2.frozen_actor import load_frozen_actor
from WP2.utils import create_zero_initialized_genome
from WP1.config import RunConfig

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--forests", type=int, default=2048)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--out", default="tesis/images/06_Results_and_Experiments/inner_loop/d_ablation_fixed_body.json")
args = ap.parse_args()

gs.init(backend=gs.gpu, logging_level="warning")
dev = "cuda:0"
summary = {}
for run in args.runs:
    run = Path(run)
    cfg = HebbianEvolutionConfig.from_yaml(str(run / "reproducibility" / "config.yaml"))
    cfg.device = dev
    wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)
    model, last_layer, _, _ = load_frozen_actor(cfg.checkpoint_path, cfg.checkpoint_config_path, device=dev)
    model_and_layer = (model, last_layer, cfg.hebbian.num_actions, cfg.hebbian.hidden_dim)

    best = np.load(run / "best_individual" / "fitness" / "genome.npy").astype(float)
    n = cfg.hebbian.num_actions * cfg.hebbian.hidden_dim  # 224
    assert best.shape[0] == 4 * n, best.shape
    zero = np.asarray(create_zero_initialized_genome(cfg), dtype=float)
    noD = best.copy(); noD[3 * n:4 * n] = 0.5
    onlyD = best.copy(); onlyD[0:3 * n] = 0.5
    variants = {"zero": zero, "best": best, "best_noD": noD, "best_onlyD": onlyD}
    try:
        with open(run / "cmaes_final_state.pkl", "rb") as f:
            es = pickle.load(f)
        mean = np.asarray(es.mean if hasattr(es, "mean") else es["mean"], dtype=float)
        if mean.shape[0] == 4 * n:
            mean = np.clip(mean, 0.0, 1.0)
            m_noD = mean.copy(); m_noD[3 * n:4 * n] = 0.5
            m_onlyD = mean.copy(); m_onlyD[0:3 * n] = 0.5
            variants.update({"mean": mean, "mean_noD": m_noD, "mean_onlyD": m_onlyD})
    except Exception as exc:  # noqa: BLE001
        print(f"[ablation] no CMA-ES mean for {run.name}: {type(exc).__name__}: {exc}")
    names = list(variants)
    P = len(names)

    F = args.forests
    cfg.evaluation.num_eval_envs = P * F
    t0 = time.time()
    env, urdf_path = _build_env(cfg, wp1_cfg, dev, num_envs_override=P * F)
    print(f"[ablation] {run.name}: env of {P * F} slots built in {time.time() - t0:.0f} s; variants {names}")

    per_rep = {k: [] for k in ("fitness", "progress", "velocity", "crash", "cot")}
    for r in range(args.repeats):
        env.refresh_forests()
        fit, met = evaluate_population_cma_batched(
            [variants[k] for k in names], cfg, model_and_layer, wp1_cfg,
            catalog=None, existing_env=(env, urdf_path), verbose=False,
        )
        per_rep["fitness"].append(np.asarray(fit, dtype=float).tolist())
        per_rep["progress"].append(np.asarray(met["progresses"], dtype=float).tolist())
        per_rep["velocity"].append(np.asarray(met["velocities"], dtype=float).tolist())
        per_rep["crash"].append(np.asarray(met["crash_flags"], dtype=float).tolist())
        per_rep["cot"].append(np.asarray(met["cots"], dtype=float).tolist())
        print(f"[ablation] {run.name} repeat {r}: " + "  ".join(f"{k} {v:.2f}" for k, v in zip(names, fit)), flush=True)

    res = {"run": run.name, "checkpoint": cfg.checkpoint_path, "variants": names, "forests": F, "repeats": args.repeats}
    for key, rows in per_rep.items():
        a = np.asarray(rows)  # (R, P)
        res[key] = {k: {"mean": float(a[:, i].mean()), "std": float(a[:, i].std(ddof=1)) if a.shape[0] > 1 else float("nan")} for i, k in enumerate(names)}
    # paired differences against zero, per repeat
    a = np.asarray(per_rep["fitness"])
    res["gain_vs_zero"] = {k: {"mean": float((a[:, i] - a[:, 0]).mean()), "std": float((a[:, i] - a[:, 0]).std(ddof=1)) if a.shape[0] > 1 else float("nan")} for i, k in enumerate(names)}
    with open(run / "results" / "d_ablation.json", "w") as f:
        json.dump(res, f, indent=1)
    summary[run.name] = res
    print(json.dumps({k: res["gain_vs_zero"][k]["mean"] for k in names}))
    env = None
    torch.cuda.empty_cache()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    json.dump(summary, f, indent=1)
print("[ablation] written", args.out)
