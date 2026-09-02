"""
Controller generality across a co-design run: best-of-generation rules on
random bodies vs the run's own front bodies.
====================================================================

Takes ONE outer-loop run and, for every inner generation (or every k-th),
the best CMA-ES individual of that generation
(``generations/gen_XXX/solutions.npy[argmax fitnesses]``). All those
controllers, plus the zero-rules generalist as reference, then fly

* ``--random-morphs N`` bodies sampled uniformly in [0,1]^15 (the outer
  loop's own gen-0 distribution, so genuinely unseen bodies), and
* the run's final exam-scored Pareto front (``--own-bodies front``) — the
  bodies the late controllers were co-adapted with,

on the SAME forests (``--forests F`` per body, one seed), so every controller
is paired with every other on bodies, forests and speed grid. The question
it answers: does the evolving controller keep the generalist's breadth, or
does it specialise toward the evolved morphology region — which shows as
the random-body curve falling while the front-body curve holds.

Forests default to the run's exam distribution (``outer.exam_forest``,
applied at env build time through ``cfg.forest`` so the DepthSolver is
built against the trees it collides with); ``--nominal`` flies the inner
loop's forests instead, and the ``transfer_eval`` forest flags override
either.

Layout of one measurement
-------------------------
Bodies are materialised as URDFs and flown ``--chunk-morphs`` at a time in
a ``MultiSceneEvalEnv`` (one scene per body) whose per-body slot count is
``controllers_per_pass × F``. ``evaluate_population_multi_urdf`` flies the
P controllers of a pass as one population: slot ``p*F + f`` of every scene
is controller p on forest f, so all controllers share forests 0..F-1 and
the vmin..vmax speed grid. With ``--controllers-per-pass 0`` (default) all
controllers fly in a single pass per chunk; otherwise the env is reused
across passes and the last pass is padded with the zero-rules controller.
The device RNG is re-seeded to ``--forest-seed`` before every
``refresh_forests()``, so every chunk flies bit-identical layouts.

Results are appended to ``generality.csv`` after every chunk, so an
interrupted sweep resumes by skipping the bodies already present.

Usage
-----
    PYTHONPATH=src python -m WP2_Outer_Loop.controller_generality \\
        --run <run_dir> [--every 1] [--forests 64] \\
        [--random-morphs 128 --morph-seed 0] [--own-bodies front|none] \\
        [--nominal | --x-upper .. --dens-min .. --dens-max ..] \\
        [--chunk-morphs 2] [--controllers-per-pass 0] [--warmup 0] \\
        [-o <run_dir>/generality] [--dry-run] [--plot-only]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .exam_baseline_rerun import load_run_config
from .transfer_eval import (
    _METRICS,
    _fit_genome_to_checkpoint,
    _gene_columns,
    _parse_forest_args,
    apply_forest_settings,
    random_morph_rows,
    read_results,
    select_front,
    write_results,
    zero_rules_genome,
)

# ``outer.exam_forest`` field → ``cfg.forest`` (ForestConfig) field.
_EXAM_TO_FOREST = {"forest_mode": "mode"}

_SUMMARY_METRICS: Tuple[str, ...] = (
    "fitness", "progress_m", "cost_of_transport", "crash_rate", "velocity",
)

BODY_SETS = ("random", "front")


# ----------------------------------------------------------------------------
#  Controllers
# ----------------------------------------------------------------------------

def generation_indices(run_dir: Path | str) -> List[int]:
    """Inner generations that still carry both ``solutions.npy`` and
    ``fitnesses.npy`` (pruned or half-synced generations are skipped)."""
    gen_root = Path(run_dir) / "generations"
    if not gen_root.is_dir():
        raise FileNotFoundError(f"No generations/ directory under {run_dir}")
    gens = []
    for d in sorted(gen_root.glob("gen_*")):
        if (d / "solutions.npy").is_file() and (d / "fitnesses.npy").is_file():
            try:
                gens.append(int(d.name[len("gen_"):]))
            except ValueError:
                continue
    if not gens:
        raise FileNotFoundError(f"No complete generations under {gen_root}")
    return sorted(gens)


def select_generations(available: Sequence[int], every: int = 1) -> List[int]:
    """Every ``every``-th generation (by generation number, from 0), always
    including the last available one so the curve ends where the run did."""
    if every < 1:
        raise ValueError(f"--every must be ≥ 1 (got {every})")
    avail = sorted(int(g) for g in available)
    chosen = [g for g in avail if g % every == 0]
    if avail and avail[-1] not in chosen:
        chosen.append(avail[-1])
    return chosen


def best_of_generation(run_dir: Path | str, gen: int) -> Tuple[np.ndarray, int, float]:
    """``(genome, individual index, in-run fitness)`` of the generation's
    argmax-fitness individual."""
    d = Path(run_dir) / "generations" / f"gen_{gen:03d}"
    solutions = np.load(d / "solutions.npy")
    fitnesses = np.load(d / "fitnesses.npy")
    idx = int(np.nanargmax(fitnesses))
    return solutions[idx].astype(np.float64), idx, float(fitnesses[idx])


def build_controllers(
    run_dir: Path | str, gens: Sequence[int], zero_genome: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Controller table + genome matrix. Row 0 is the zero-rules generalist
    (``kind="zero"``, ``gen=-1``); rows 1.. are ``kind="best"`` in the order
    of ``gens``."""
    zero = np.asarray(zero_genome, dtype=np.float64)
    rows = [{"controller_id": 0, "kind": "zero", "gen": -1, "best_idx": -1,
             "in_run_fitness": np.nan}]
    genomes = [zero]
    for g in gens:
        genome, idx, fit = best_of_generation(run_dir, int(g))
        if genome.shape[0] != zero.shape[0]:
            raise ValueError(
                f"gen {g} genome has {genome.shape[0]} genes but the zero-rules "
                f"genome has {zero.shape[0]} — checkpoint/config mismatch"
            )
        rows.append({"controller_id": len(genomes), "kind": "best",
                     "gen": int(g), "best_idx": idx, "in_run_fitness": fit})
        genomes.append(genome)
    return pd.DataFrame(rows), np.stack(genomes)


# ----------------------------------------------------------------------------
#  Forests
# ----------------------------------------------------------------------------

def exam_forest_settings(cfg) -> Dict[str, object]:
    """The run's ``outer.exam_forest`` overrides as ``cfg.forest`` field
    names — empty when the exam flew the nominal inner-loop forests."""
    outer = getattr(cfg, "outer", None)
    exam = getattr(outer, "exam_forest", None)
    if exam is None:
        return {}
    return {_EXAM_TO_FOREST.get(k, k): v for k, v in exam.overrides().items()}


# ----------------------------------------------------------------------------
#  Bodies
# ----------------------------------------------------------------------------

def body_table(
    run_dir: Path | str, random_n: int, morph_seed: int, own: str,
) -> pd.DataFrame:
    """One row per body: ``body_id`` (global), ``body_set``, ``urdf_idx``
    (index within its set), provenance columns and the g* genome. Random
    bodies first, then the run's last exam-scored front (deduplicated)."""
    frames: List[pd.DataFrame] = []
    if random_n > 0:
        rnd = random_morph_rows(int(random_n), int(morph_seed))
        rnd["body_set"] = "random"
        frames.append(rnd)
    if own == "front":
        front = select_front(Path(run_dir) / "results" / "pareto_front.csv", "last")
        front["body_set"] = "front"
        frames.append(front)
    elif own != "none":
        raise ValueError(f"--own-bodies must be 'front' or 'none' (got {own!r})")
    if not frames:
        raise ValueError("No bodies: give --random-morphs N and/or --own-bodies front")

    genes = _gene_columns(frames[0])
    keep = ["body_set", "urdf_idx", "outer_gen", "front_rank"]
    archived = {"progress_m": "archived_progress_m",
                "cost_of_transport": "archived_cost_of_transport"}
    parts = []
    for f in frames:
        part = f.reindex(columns=keep + list(archived) + genes)
        part = part.rename(columns=archived)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "body_id", np.arange(len(out)))
    for col in ("urdf_idx", "outer_gen", "front_rank"):
        out[col] = out[col].fillna(-1).astype(int)
    return out


# ----------------------------------------------------------------------------
#  Passes + reduction
# ----------------------------------------------------------------------------

def controller_passes(n_controllers: int, per_pass: int) -> List[Tuple[List[int], int]]:
    """Split controller ids into passes of exactly ``per_pass`` (the env's
    per-body slot count is ``per_pass × F`` and must not change between
    passes). The last pass is padded with controller 0 (zero rules); the
    second tuple element is how many leading ids are real. ``per_pass ≤ 0``
    or ``≥ n`` ⇒ a single pass holding everything."""
    n = int(n_controllers)
    if per_pass <= 0 or per_pass >= n:
        return [(list(range(n)), n)]
    groups: List[Tuple[List[int], int]] = []
    for start in range(0, n, per_pass):
        ids = list(range(start, min(start + per_pass, n)))
        n_real = len(ids)
        ids += [0] * (per_pass - n_real)
        groups.append((ids, n_real))
    return groups


def reduce_pass(
    metrics: Dict[str, np.ndarray],
    controller_ids: Sequence[int],
    n_real: int,
    bodies: pd.DataFrame,
) -> List[Dict[str, float]]:
    """One result row per (body, real controller) from the ``per_urdf_*``
    ``(N, P)`` means and ``per_slot_*`` ``(N, P, F)`` views of one pass; the
    ``_se`` columns are standard errors across the F forests."""
    rows: List[Dict[str, float]] = []
    for local, (_, b) in enumerate(bodies.iterrows()):
        for pos in range(int(n_real)):
            row: Dict[str, float] = {
                "body_id": int(b["body_id"]),
                "body_set": str(b["body_set"]),
                "urdf_idx": int(b["urdf_idx"]),
                "controller_id": int(controller_ids[pos]),
            }
            for col, pu_key, ps_key in _METRICS:
                row[col] = float(np.asarray(metrics[pu_key])[local, pos])
                slots = np.asarray(metrics[ps_key])[local, pos]
                row[f"{col}_se"] = (
                    float(np.std(slots, ddof=1) / np.sqrt(len(slots)))
                    if len(slots) > 1 else 0.0
                )
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------
#  Resume + summary
# ----------------------------------------------------------------------------

def completed_bodies(csv_path: Path | str) -> Set[int]:
    """``body_id``s already present in a result CSV (empty when absent)."""
    path = Path(csv_path)
    if not path.is_file():
        return set()
    df, _ = read_results(path)
    if df.empty or "body_id" not in df.columns:
        return set()
    return {int(x) for x in df["body_id"].unique()}


def summarize(df: pd.DataFrame, controllers: pd.DataFrame) -> pd.DataFrame:
    """Per (body_set, kind, gen): mean of every metric over bodies, its
    standard error across bodies, and the body count."""
    merged = df.merge(controllers[["controller_id", "kind", "gen"]],
                      on="controller_id", how="inner")
    out = []
    for (body_set, kind, gen), grp in merged.groupby(["body_set", "kind", "gen"],
                                                     sort=True):
        row: Dict[str, object] = {"body_set": body_set, "kind": kind,
                                  "gen": int(gen),
                                  "n_bodies": int(grp["body_id"].nunique())}
        for m in _SUMMARY_METRICS:
            if m not in grp.columns:
                continue
            vals = grp[m].to_numpy(dtype=float)
            row[m] = float(np.nanmean(vals))
            row[f"{m}_se"] = (
                float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1 else 0.0
            )
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
#  Companion series from the run's own CSVs (no GPU)
# ----------------------------------------------------------------------------

def _refresh_every(run_dir: Path | str) -> int:
    import yaml
    cfg_path = Path(run_dir) / "reproducibility" / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    return int((raw.get("catalog") or {}).get("refresh_urdfs_every", 1))


def morph_diversity(run_dir: Path | str) -> pd.DataFrame:
    """Mean pairwise genome distance of the morphology population per outer
    generation, with the inner-generation span each phase covers."""
    pop = pd.read_csv(Path(run_dir) / "results" / "outer_population.csv")
    genes = _gene_columns(pop)
    every = _refresh_every(run_dir)
    rows = []
    for og, grp in pop.groupby("outer_gen", sort=True):
        X = grp[genes].to_numpy(dtype=float)
        if len(X) < 2:
            d = 0.0
        else:
            D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
            d = float(D[np.triu_indices(len(X), 1)].mean())
        og = int(og)
        rows.append({"outer_gen": og, "gen_start": og * every,
                     "gen_end": (og + 1) * every - 1, "diversity": d})
    return pd.DataFrame(rows)


def final_centroid(run_dir: Path | str) -> np.ndarray:
    """Genome-space centroid of the last outer generation's population."""
    pop = pd.read_csv(Path(run_dir) / "results" / "outer_population.csv")
    genes = _gene_columns(pop)
    last = pop[pop["outer_gen"] == pop["outer_gen"].max()]
    return last[genes].mean().to_numpy(dtype=float)


# ----------------------------------------------------------------------------
#  Provenance + resume guard
# ----------------------------------------------------------------------------

# A result CSV may only be appended to by a measurement that agrees on these.
_RESUME_KEYS = ("forest", "forest_seed", "n_forests", "random_morphs",
                "morph_seed", "own_bodies", "gens", "checkpoint_md5",
                "vmin", "vmax", "stochastic")


def build_provenance(cfg, args, forest: Dict, run_dir: Path, gens: Sequence[int],
                     n_controllers: int, n_bodies: int) -> Dict:
    ckpt = Path(cfg.checkpoint_path)
    return {
        "tag": args.tag,
        "run_dir": str(run_dir),
        "forest": {k: v for k, v in sorted(forest.items())},
        "forest_seed": int(args.forest_seed),
        "n_forests": int(args.forests),
        "random_morphs": int(args.random_morphs),
        "morph_seed": int(args.morph_seed),
        "own_bodies": str(args.own_bodies),
        "every": int(args.every),
        "gens": [int(g) for g in gens],
        "n_controllers": int(n_controllers),
        "n_bodies": int(n_bodies),
        "controllers_per_pass": int(args.controllers_per_pass),
        "warmup": int(args.warmup),
        "vmin": float(cfg.evaluation.vmin),
        "vmax": float(cfg.evaluation.vmax),
        "stochastic": bool(cfg.evaluation.stochastic),
        "crn": bool(getattr(cfg.evaluation, "crn", True)),
        "checkpoint": str(ckpt),
        "checkpoint_md5": (hashlib.md5(ckpt.read_bytes()).hexdigest()
                           if ckpt.is_file() else ""),
    }


def resume_conflicts(existing: Dict, new: Dict) -> List[str]:
    """Keys on which an existing result CSV's provenance disagrees with the
    measurement about to append to it — resuming across such a change would
    silently mix two experiments. Empty when there is nothing to compare."""
    if not existing:
        return []
    conflicts = []
    for key in _RESUME_KEYS:
        a = json.dumps(existing.get(key), sort_keys=True)
        b = json.dumps(new.get(key), sort_keys=True)
        if a != b:
            conflicts.append(f"{key}: {a} vs {b}")
    return conflicts


# ----------------------------------------------------------------------------
#  Measurement (GPU)
# ----------------------------------------------------------------------------

def fly_bodies(
    bodies: pd.DataFrame,
    urdf_paths: Sequence[str],
    cfg,
    genomes: np.ndarray,
    n_forests: int,
    chunk_morphs: int,
    controllers_per_pass: int,
    forest_seed: int,
    warmup: int = 0,
    num_workers: int = 1,
    verbose: bool = True,
    on_chunk=None,
) -> List[Dict[str, float]]:
    """Fly every controller in ``genomes`` on every body in ``bodies``.

    ``urdf_paths`` is aligned with ``bodies``' rows. Bodies go
    ``chunk_morphs`` at a time into one ``MultiSceneEvalEnv`` (one scene per
    body, ``per_pass × n_forests`` slots each); controllers go through it in
    the passes of :func:`controller_passes`, all on the forests generated
    once under ``forest_seed``. ``on_chunk(rows)`` is called after every
    chunk so results can be persisted incrementally. ``num_workers > 1``
    builds the chunk's scenes in parallel worker processes
    (``ParallelMultiSceneEvalEnv``) — scene build dominates at large slot
    counts, so this is the main speed knob.
    """
    import genesis as gs
    import torch
    from WP1.config import RunConfig
    from WP2.evaluate import _build_multi_urdf_env, evaluate_population_multi_urdf
    from WP2.frozen_actor import load_frozen_actor

    if len(urdf_paths) != len(bodies):
        raise ValueError("urdf_paths must align with bodies")
    passes = controller_passes(len(genomes), controllers_per_pass)
    per_pass = len(passes[0][0])
    slots_per_body = per_pass * int(n_forests)

    if not gs._initialized:
        gs.init(backend=gs.gpu, logging_level="warning")
    wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)
    model, last_layer, num_actions, hidden_dim = load_frozen_actor(
        cfg.checkpoint_path, cfg.checkpoint_config_path, device=cfg.device
    )
    model_and_layer = (model, last_layer, num_actions, hidden_dim)

    all_rows: List[Dict[str, float]] = []
    starts = list(range(0, len(bodies), max(1, int(chunk_morphs))))
    t_sweep = time.time()
    try:
        for ci, start in enumerate(starts):
            chunk = bodies.iloc[start:start + chunk_morphs]
            paths = [str(urdf_paths[i]) for i in range(start, start + len(chunk))]
            if not gs._initialized:
                gs.init(backend=gs.gpu, logging_level="warning")
            chunk_cfg = copy.deepcopy(cfg)
            chunk_cfg.evaluation.num_eval_envs = len(paths) * slots_per_body
            chunk_cfg.evaluation.run_baseline = False
            chunk_cfg.evaluation.refresh_forests_per_generation = False
            chunk_cfg.catalog.path = ""
            chunk_cfg.catalog.num_urdfs = len(paths)
            chunk_cfg.catalog.force_multi_urdf = True

            if verbose:
                print(f"[generality] Chunk {ci + 1}/{len(starts)}: {len(paths)} "
                      f"bodies × {slots_per_body} slots ({per_pass} controllers "
                      f"× {n_forests} forests), {len(passes)} pass(es)")
            t0 = time.time()
            env = _build_multi_urdf_env(
                paths, chunk_cfg, wp1_cfg, chunk_cfg.device,
                num_envs_per_drone=slots_per_body,
                num_workers=max(1, int(num_workers)),
            )
            if verbose:
                print(f"[generality]   built in {time.time() - t0:.0f}s")
            rows_chunk: List[Dict[str, float]] = []
            try:
                # One seeded pool per chunk ⇒ every chunk (and every pass
                # within it) flies the same forests 0..F-1 per controller.
                torch.manual_seed(int(forest_seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(forest_seed))
                env.refresh_forests()

                for _w in range(int(warmup)):
                    ids, _ = passes[0]
                    evaluate_population_multi_urdf(
                        [genomes[i] for i in ids], chunk_cfg, model_and_layer,
                        wp1_cfg, urdf_paths=paths, existing_env=(env, paths),
                        verbose=False,
                    )
                for pi, (ids, n_real) in enumerate(passes):
                    t1 = time.time()
                    _fit, metrics = evaluate_population_multi_urdf(
                        [genomes[i] for i in ids], chunk_cfg, model_and_layer,
                        wp1_cfg, urdf_paths=paths, existing_env=(env, paths),
                        verbose=False,
                    )
                    rows_chunk += reduce_pass(metrics, ids, n_real, chunk)
                    if verbose:
                        print(f"[generality]   pass {pi + 1}/{len(passes)} "
                              f"in {time.time() - t1:.0f}s")
            finally:
                # Parallel workers own their own Genesis runtimes — ask them to
                # exit before tearing down the main process's.
                shutdown = getattr(env, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception as exc:
                        print(f"[generality] env shutdown failed: {exc}")
                try:
                    gs.destroy()
                except Exception as exc:
                    print(f"[generality] gs.destroy() failed: {exc}")

            all_rows += rows_chunk
            if on_chunk is not None:
                on_chunk(rows_chunk)
            if verbose:
                elapsed = time.time() - t_sweep
                eta = elapsed / (ci + 1) * (len(starts) - ci - 1)
                zero = [r["fitness"] for r in rows_chunk if r["controller_id"] == 0]
                best = [r["fitness"] for r in rows_chunk if r["controller_id"] != 0]
                print(f"[generality]   chunk done: zero-rules fitness "
                      f"{np.mean(zero):.1f}, best-of-gen mean {np.mean(best):.1f} "
                      f"— {elapsed / 60:.1f} min elapsed, ~{eta / 60:.0f} min left")
    finally:
        if gs._initialized:
            try:
                gs.destroy()
            except Exception:
                pass
    return all_rows


# ----------------------------------------------------------------------------
#  Plots
# ----------------------------------------------------------------------------

_SET_COLORS = {"random": "#1f77b4", "front": "#d62728"}
_PANELS = (("fitness", "Fitness (WP1 reward sum)"),
           ("progress_m", "Progress [m]"),
           ("cost_of_transport", "Cost of transport"))


def draw_generality(axes, summary: pd.DataFrame, body_sets: Sequence[str],
                    diversity: Optional[pd.DataFrame] = None) -> None:
    """Fitness / progress / CoT vs generation on three axes: per body set a
    best-of-gen line (± SE across bodies) and the zero-rules generalist as a
    dashed horizontal reference. The morphology population's diversity, when
    given, goes on a twin axis of the first panel."""
    present = [bs for bs in body_sets if bs in set(summary["body_set"])]
    for ax, (metric, ylabel) in zip(axes, _PANELS):
        if metric not in summary.columns:
            ax.set_visible(False)
            continue
        for bs in present:
            color = _SET_COLORS.get(bs, "0.3")
            sub = summary[summary["body_set"] == bs]
            best = sub[sub["kind"] == "best"].sort_values("gen")
            n = int(best["n_bodies"].max()) if len(best) else 0
            if len(best):
                x = best["gen"].to_numpy(float)
                y = best[metric].to_numpy(float)
                se = best.get(f"{metric}_se", pd.Series(0.0, index=best.index)).to_numpy(float)
                ax.plot(x, y, "-", color=color, linewidth=1.6,
                        label=f"best-of-gen on {bs} bodies (n={n})")
                ax.fill_between(x, y - se, y + se, color=color, alpha=0.18,
                                linewidth=0)
            zero = sub[sub["kind"] == "zero"]
            if len(zero):
                z = float(zero[metric].iloc[0])
                zse = float(zero.get(f"{metric}_se", pd.Series([0.0])).iloc[0])
                ax.axhline(z, linestyle="--", color=color, linewidth=1.3,
                           label=f"zero rules on {bs} bodies")
                if zse > 0:
                    ax.axhspan(z - zse, z + zse, color=color, alpha=0.08,
                               linewidth=0)
        ax.set_xlabel("inner generation")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)

    if diversity is not None and len(diversity) and len(axes):
        ax2 = axes[0].twinx()
        xs, ys = [], []
        for _, r in diversity.sort_values("outer_gen").iterrows():
            xs += [float(r["gen_start"]), float(r["gen_end"]) + 1.0]
            ys += [float(r["diversity"])] * 2
        ax2.plot(xs, ys, color="0.45", linewidth=1.0, alpha=0.8,
                 label="morphology diversity (mean pairwise dist)")
        ax2.set_ylabel("morph. population diversity", color="0.45")
        ax2.tick_params(axis="y", colors="0.45")
        h1, l1 = axes[0].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        axes[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    elif len(axes):
        axes[0].legend(fontsize=8, loc="lower left")


def plot_generality(summary: pd.DataFrame, out_path: Path,
                    body_sets: Sequence[str] = BODY_SETS,
                    diversity: Optional[pd.DataFrame] = None,
                    title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    draw_generality(axes, summary, body_sets, diversity)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[generality] Saved {out_path}")
    return out_path


def nearest_generations(available: Sequence[int], wanted: Sequence[float]) -> List[int]:
    """For each wanted generation the closest measured one; duplicates
    collapse, order of first appearance kept."""
    avail = np.asarray(sorted(int(g) for g in available))
    if avail.size == 0:
        return []
    out: List[int] = []
    for w in wanted:
        g = int(avail[np.argmin(np.abs(avail - float(w)))])
        if g not in out:
            out.append(g)
    return out


def plot_delta_vs_distance(df: pd.DataFrame, bodies: pd.DataFrame,
                           controllers: pd.DataFrame, centroid: np.ndarray,
                           gens: Sequence[int], out_path: Path,
                           metric: str = "fitness") -> Path:
    """Per-body (best_g − zero) delta against the body's genome distance to
    the run's final population centroid, one colour per generation — the
    direct test of 'specialised toward the evolved region'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    genes = _gene_columns(bodies)
    rnd = bodies[bodies["body_set"] == "random"] if "random" in set(bodies["body_set"]) else bodies
    dist = pd.Series(
        np.linalg.norm(rnd[genes].to_numpy(float) - np.asarray(centroid, float)[None, :],
                       axis=1),
        index=rnd["body_id"].to_numpy(int),
    )
    zero = df[df["controller_id"] == 0].set_index("body_id")[metric]

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    gens = list(gens)
    for i, g in enumerate(gens):
        ids = controllers[(controllers["kind"] == "best") & (controllers["gen"] == g)]["controller_id"]
        if not len(ids):
            continue
        best = df[df["controller_id"] == int(ids.iloc[0])].set_index("body_id")[metric]
        common = dist.index.intersection(best.index).intersection(zero.index)
        if not len(common):
            continue
        x = dist.loc[common].to_numpy(float)
        y = (best.loc[common] - zero.loc[common]).to_numpy(float)
        color = cmap(i / max(1, len(gens) - 1))
        label = f"gen {g}"
        if len(common) > 2 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            r = float(np.corrcoef(x, y)[0, 1])
            xx = np.linspace(x.min(), x.max(), 20)
            ax.plot(xx, slope * xx + intercept, "-", color=color, linewidth=1.2, alpha=0.9)
            label += f"  (slope {slope:+.2f}/unit, r={r:+.2f})"
        ax.scatter(x, y, s=18, color=color, alpha=0.75, label=label)
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xlabel("body distance to final population centroid (genome space)")
    ax.set_ylabel(f"{metric}: best-of-gen − zero rules")
    ax.set_title("Where does the controller lose generality?")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[generality] Saved {out_path}")
    return out_path


# ----------------------------------------------------------------------------
#  Output bundle
# ----------------------------------------------------------------------------

def write_outputs(out_dir: Path, df: pd.DataFrame, controllers: pd.DataFrame,
                  bodies: pd.DataFrame, run_dir: Path,
                  title: str = "") -> List[Path]:
    """``summary.csv`` + the figures: one per body set present, the overlay
    when both are, and the delta-vs-distance panel when random bodies are."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    summary = summarize(df, controllers)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    written.append(summary_path)

    try:
        diversity = morph_diversity(run_dir)
    except Exception as exc:  # companion series is optional
        print(f"[generality] no diversity series ({exc})")
        diversity = None

    present = [bs for bs in BODY_SETS if bs in set(df["body_set"])]
    base = title or Path(run_dir).name
    for bs in present:
        written.append(plot_generality(
            summary, out_dir / f"generality_{bs}.png", body_sets=(bs,),
            diversity=diversity, title=f"{base} — {bs} bodies"))
    if len(present) > 1:
        written.append(plot_generality(
            summary, out_dir / "generality_all.png", body_sets=tuple(present),
            diversity=diversity, title=f"{base} — random vs front bodies"))

    if "random" in present:
        try:
            centroid = final_centroid(run_dir)
        except Exception as exc:
            print(f"[generality] no final centroid ({exc})")
            centroid = None
        if centroid is not None:
            best_gens = sorted(controllers[controllers["kind"] == "best"]["gen"].astype(int))
            if best_gens:
                wanted = np.linspace(best_gens[0], best_gens[-1], 5)
                gens = nearest_generations(best_gens, wanted)
                written.append(plot_delta_vs_distance(
                    df, bodies, controllers, centroid, gens,
                    out_dir / "delta_vs_distance.png"))
    return written


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def _resolve_run_dir(path: Path) -> Path:
    """Accept the run dir itself or a wrapper holding exactly one run dir
    (the ``logs/remote/outer_nsga/<name>/<timestamp>_<exp>`` layout)."""
    path = Path(path)
    if (path / "reproducibility" / "config.yaml").is_file() or (path / "generations").is_dir():
        return path
    cands = [d for d in path.iterdir() if d.is_dir()
             and ((d / "reproducibility" / "config.yaml").is_file()
                  or (d / "generations").is_dir())] if path.is_dir() else []
    if len(cands) == 1:
        return cands[0]
    raise FileNotFoundError(f"{path} is not a run directory (no reproducibility/config.yaml)")


def _build_parser() -> argparse.ArgumentParser:
    from .transfer_eval import add_forest_arguments

    ap = argparse.ArgumentParser(
        description="Fly every generation's best Hebbian controller of one "
                    "outer run on random and own-front morphologies.")
    ap.add_argument("--run", type=Path, required=True,
                    help="Outer-loop run directory (or its wrapper folder).")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output directory (default <run>/generality).")
    ap.add_argument("--every", type=int, default=1,
                    help="Use every k-th generation (the last is always kept).")
    ap.add_argument("--forests", type=int, default=64,
                    help="Forests (= env slots) per controller per body.")
    ap.add_argument("--random-morphs", type=int, default=128,
                    help="Random bodies sampled uniformly in [0,1]^15 (0 = none).")
    ap.add_argument("--morph-seed", type=int, default=0)
    ap.add_argument("--own-bodies", choices=("front", "none"), default="front",
                    help="Also fly the run's last exam-scored Pareto front.")
    ap.add_argument("--nominal", action="store_true",
                    help="Fly the inner loop's nominal forests instead of the "
                         "run's exam forests.")
    ap.add_argument("--chunk-morphs", type=int, default=2,
                    help="Bodies (scenes) per Genesis env build (VRAM knob).")
    ap.add_argument("--controllers-per-pass", type=int, default=0,
                    help="Controllers per rollout pass (0 = all in one pass); "
                         "lower it if a scene of controllers×forests slots "
                         "does not fit.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel scene-worker processes per chunk (builds "
                         "overlap; each worker holds its own scenes in VRAM).")
    ap.add_argument("--warmup", type=int, default=0,
                    help="Discarded passes per chunk before measuring "
                         "(cold-start aero state).")
    ap.add_argument("--forest-seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--tag", default=None,
                    help="Label recorded in the provenance (default: exam|nominal).")
    ap.add_argument("--force", action="store_true",
                    help="Append to an existing generality.csv even if its "
                         "provenance disagrees (normally refused).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve everything, print the plan, fly nothing.")
    ap.add_argument("--plot-only", action="store_true",
                    help="Rebuild summary + figures from the CSVs in --out.")
    add_forest_arguments(ap)
    return ap


def _plot_only(run_dir: Path, out_dir: Path) -> int:
    csv = out_dir / "generality.csv"
    if not csv.is_file():
        print(f"[generality] {csv} not found — nothing to plot")
        return 1
    df, _prov = read_results(csv)
    controllers = pd.read_csv(out_dir / "controllers.csv")
    bodies = pd.read_csv(out_dir / "bodies.csv")
    write_outputs(out_dir, df, controllers, bodies, run_dir)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = _resolve_run_dir(args.run.resolve())
    out_dir = (args.out or run_dir / "generality").resolve()
    if args.plot_only:
        return _plot_only(run_dir, out_dir)

    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    cfg = load_run_config(run_dir, repo_root)
    if args.device:
        cfg.device = args.device
    _fit_genome_to_checkpoint(cfg)

    forest_settings = {} if args.nominal else exam_forest_settings(cfg)
    forest_settings.update(_parse_forest_args(args))
    forest = apply_forest_settings(cfg, forest_settings)
    if args.tag is None:
        args.tag = "nominal" if args.nominal else "exam"

    gens = select_generations(generation_indices(run_dir), args.every)
    zero = zero_rules_genome(cfg)
    controllers, genomes = build_controllers(run_dir, gens, zero)
    expected = cfg.hebbian_genome_dim()
    if genomes.shape[1] != expected:
        raise ValueError(
            f"Run genomes have {genomes.shape[1]} genes but the checkpoint/config "
            f"implies {expected} ({cfg.hebbian.num_actions}×{cfg.hebbian.hidden_dim})"
        )
    bodies = body_table(run_dir, args.random_morphs, args.morph_seed, args.own_bodies)

    provenance = build_provenance(cfg, args, forest, run_dir, gens,
                                  len(controllers), len(bodies))
    csv = out_dir / "generality.csv"
    df = pd.DataFrame()
    if csv.is_file():
        df, prev = read_results(csv)
        conflicts = resume_conflicts(prev, provenance)
        if conflicts:
            print(f"[generality] {csv} was measured under different settings:")
            for c in conflicts:
                print(f"  - {c}")
            if not args.force:
                print("[generality] Refusing to append. Use another -o, or "
                      "--force to mix them anyway.")
                return 1
    done = completed_bodies(csv)
    todo = bodies[~bodies["body_id"].isin(done)].reset_index(drop=True)

    passes = controller_passes(len(genomes), args.controllers_per_pass)
    per_pass = len(passes[0][0])
    n_chunks = (len(todo) + args.chunk_morphs - 1) // max(1, args.chunk_morphs)
    print(f"[generality] Run:         {run_dir}")
    print(f"[generality] Checkpoint:  {cfg.checkpoint_path}")
    print(f"[generality] Controllers: {len(controllers)} = zero rules + best of "
          f"{len(gens)} gens ({gens[0]}..{gens[-1]}, every {args.every})")
    counts = bodies["body_set"].value_counts().to_dict()
    print(f"[generality] Bodies:      {len(bodies)} {counts}"
          + (f" — {len(done)} already done, {len(todo)} to fly" if done else ""))
    print(f"[generality] Forest:      {json.dumps(forest, sort_keys=True)}")
    print(f"[generality] Plan:        {n_chunks} chunk(s) of ≤{args.chunk_morphs} "
          f"bodies, {len(passes)} pass(es) × {per_pass} controllers × "
          f"{args.forests} forests = {per_pass * args.forests} slots/body, "
          f"seed {args.forest_seed}, warmup {args.warmup}, "
          f"{args.workers} scene worker(s)")
    print(f"[generality] Output:      {csv}")
    if args.dry_run:
        print("[generality] --dry-run: stopping before measurement.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    controllers.to_csv(out_dir / "controllers.csv", index=False)
    bodies.to_csv(out_dir / "bodies.csv", index=False)

    if len(todo):
        import genesis as gs
        from .urdf_population import materialize_urdfs

        gs.init(backend=gs.gpu, logging_level="warning")
        genes = _gene_columns(bodies)
        all_paths = materialize_urdfs(bodies[genes].to_numpy(float).tolist(),
                                      out_dir / "urdfs")
        todo_paths = [all_paths[int(b)] for b in todo["body_id"]]

        state = {"df": df}

        def _persist(rows):
            state["df"] = pd.concat([state["df"], pd.DataFrame(rows)],
                                    ignore_index=True)
            write_results(state["df"], provenance, csv)

        fly_bodies(todo, todo_paths, cfg, genomes, n_forests=args.forests,
                   chunk_morphs=args.chunk_morphs,
                   controllers_per_pass=args.controllers_per_pass,
                   forest_seed=args.forest_seed, warmup=args.warmup,
                   num_workers=args.workers, on_chunk=_persist)
        df = state["df"]

    write_outputs(out_dir, df, controllers, bodies, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
