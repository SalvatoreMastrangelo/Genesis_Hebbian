"""
Re-fly an outer run's Pareto-front morphologies under a chosen controller.
=========================================================================

Takes ``results/pareto_front.csv`` (see ``WP2_Outer_Loop.pareto_fronts``),
regenerates the URDFs of one generation's front from their genomes, and flies
all of them with ONE controller on a forest distribution you choose. The
result is a CoT-vs-progress table and figure for that controller.

The point is transfer: run this once per controller — the run's evolved
Hebbian rules, then the zero-rules generalist — and overlay the two figures
(``--plot A.csv B.csv``) to see how much of the morphologies' evolved
performance survives a controller swap.

Fair overlay
------------
``ForestGenerator.generate()`` draws from the global torch RNG, so
``--forest-seed`` (default 0) is re-seeded immediately before every
``refresh_forests()``. Two separate invocations with the same seed, forest
settings and ``--n-forests`` therefore fly **identical layouts**, and so does
every morph chunk within one invocation — so morphs are comparable to each
other as well as across controllers. ``--plot`` refuses to overlay result
CSVs whose forest settings, seed or forest count disagree.

Forest settings are applied at **env build time** through ``cfg.forest``, not
as runtime generator overrides, so perception-coupled parameters
(``tree_radius``, the y corridor) are safe here — the DepthSolver is built
with the same values it collides against. That is not true of
``outer.exam_forest``, which patches a live generator.

Not corrected for
-----------------
Cold-start aero state (``_thr_flt`` starts at 0 on a fresh env) biases each
chunk's single pass slightly pessimistic, equally for every controller since
each gets its own fresh env. No warmup pass is run, and CRN is left at
whatever the config says.

Usage
-----
    PYTHONPATH=src python -m WP2_Outer_Loop.transfer_eval FRONT_CSV \\
        --config <run>/reproducibility/config.yaml --genome best \\
        --tag hebbian --x-upper 200 --dens-max 8 -o out/

    PYTHONPATH=src python -m WP2_Outer_Loop.transfer_eval FRONT_CSV \\
        --config <run>/reproducibility/config.yaml --genome zero \\
        --tag generalist --x-upper 200 --dens-max 8 -o out/

    PYTHONPATH=src python -m WP2_Outer_Loop.transfer_eval \\
        --plot out/transfer_hebbian.csv out/transfer_generalist.csv \\
        -o out/compare.png
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .pareto_fronts import _nondominated_mask


# Metrics reported per morphology. (output column, per_urdf key, per_slot key)
_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("progress_m",        "per_urdf_progress", "per_slot_progress"),
    ("cost_of_transport", "per_urdf_cot",      "per_slot_cot"),
    ("velocity",          "per_urdf_velocity", "per_slot_velocity"),
    ("crash_rate",        "per_urdf_crash",    "per_slot_crash"),
    ("fitness",           "per_urdf_reward",   "per_slot_reward"),
)

# Provenance keys that must agree before two result CSVs may be overlaid.
_COMPARABLE_KEYS = ("forest", "forest_seed", "n_forests", "vmin", "vmax",
                    "stochastic", "num_eval_episodes",
                    # Two controllers must have flown the SAME bodies: a
                    # different morph seed means a different random population
                    # and the comparison is meaningless.
                    "random_morphs", "morph_seed")

_PROVENANCE_PREFIX = "# provenance: "


# ----------------------------------------------------------------------------
#  Front selection
# ----------------------------------------------------------------------------

def _gene_columns(df: pd.DataFrame) -> List[str]:
    return sorted(
        (c for c in df.columns if len(c) > 1 and c[0] == "g" and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )


def select_front(
    front_csv: Path | str,
    gen: str | int = "last",
    dedup: bool = True,
) -> pd.DataFrame:
    """Rows of one generation's Pareto front, optionally deduplicated.

    ``gen="last"`` takes the highest ``outer_gen`` present. Elites persist
    across generations, so genomes repeat; ``dedup`` collapses rows whose
    genome agrees to 6 decimals (the precision ``outer_population.csv`` is
    written at), keeping the first — flying the same morphology twice costs a
    Genesis scene and tells you nothing.
    """
    df = pd.read_csv(front_csv)
    if df.empty:
        raise ValueError(f"{front_csv} has no rows")
    if gen == "last":
        target = int(df["outer_gen"].max())
    else:
        target = int(gen)
        if target not in set(df["outer_gen"].astype(int)):
            raise ValueError(
                f"outer_gen {target} not in {front_csv} "
                f"(have {sorted(set(df['outer_gen'].astype(int)))})"
            )
    sel = df[df["outer_gen"].astype(int) == target].reset_index(drop=True)

    genes = _gene_columns(sel)
    if not genes:
        raise ValueError(f"{front_csv} has no g* genome columns")
    if dedup:
        before = len(sel)
        key = sel[genes].round(6).astype(str).agg("|".join, axis=1)
        sel = sel[~key.duplicated()].reset_index(drop=True)
        if len(sel) < before:
            print(f"[transfer] Deduplicated {before} → {len(sel)} morphologies")
    print(f"[transfer] outer_gen {target}: {len(sel)} morphologies to fly")
    return sel


def standard_drone_row(n_genes: int) -> pd.DataFrame:
    """A one-row frame for the unevolved standard mydrone.

    Built from ``STANDARD_MYDRONE_GENOME`` through ``from_physical`` — the same
    round trip ``urdf_population.sample_initial_population`` uses to seed slot 0
    — so it is materialized by the identical ``UrdfMaker`` path as every evolved
    morphology, not read from a stale on-disk URDF.

    ``urdf_idx`` / ``front_rank`` are ``-1``: it is a reference point, not a
    member of any front.
    """
    from morph_evolution.chromosome_drone import Chromosome_Drone
    from winged_drone_train.defaults import STANDARD_MYDRONE_GENOME

    norm = Chromosome_Drone.from_physical(list(STANDARD_MYDRONE_GENOME))
    if len(norm) != n_genes:
        raise ValueError(
            f"Standard mydrone genome has {len(norm)} genes but the front CSV "
            f"carries {n_genes} — refusing to fly a mismatched reference"
        )
    row = {"outer_gen": -1, "urdf_idx": -1, "urdf_file": "standard_mydrone.urdf",
           "front_rank": -1, "is_standard": 1}
    row.update({f"g{i}": float(v) for i, v in enumerate(norm)})
    return pd.DataFrame([row])


def random_morph_rows(n: int, seed: int) -> pd.DataFrame:
    """``n`` morphologies sampled uniformly in ``[0,1]^15``.

    Uses ``urdf_population.sample_initial_population`` with the standard-drone
    seeding OFF, i.e. exactly the distribution the outer loop's own generation
    0 draws from — so these are genuinely unseen bodies, not perturbations of
    an evolved front.

    ``outer_gen = -1`` marks "not from any front"; ``urdf_idx`` is the sample
    index, which is reproducible from ``seed`` alone.
    """
    from .urdf_population import sample_initial_population

    pop = sample_initial_population(n, seed, seed_standard_drone=False)
    rows = []
    for i, genome in enumerate(pop):
        row = {"outer_gen": -1, "urdf_idx": i,
               "urdf_file": f"rand_{i:03d}.urdf", "front_rank": i,
               "is_standard": 0}
        row.update({f"g{j}": float(v) for j, v in enumerate(genome)})
        rows.append(row)
    return pd.DataFrame(rows)


def with_standard_drone(front: Optional[pd.DataFrame], n_genes: int = 15) -> pd.DataFrame:
    """``front`` plus the standard mydrone, or just the mydrone when front is
    None. ``is_standard`` marks which rows are the reference."""
    std = standard_drone_row(n_genes)
    if front is None or front.empty:
        return std
    front = front.copy()
    front["is_standard"] = 0
    return pd.concat([front, std], ignore_index=True)


# ----------------------------------------------------------------------------
#  Config + genome resolution
# ----------------------------------------------------------------------------

def load_controller_config(
    config_path: Path, repo_root: Path, checkpoint: Optional[Path] = None,
):
    """Load a WP2 or WP1 config into the ``HebbianEvolutionConfig`` the eval
    machinery expects; returns ``(cfg, kind)``.

    A WP2 config (outer or plain inner) carries its own checkpoint paths. A
    WP1 config describes only the policy/env, so ``--checkpoint`` is required
    and the config path itself becomes ``checkpoint_config_path``.
    """
    import yaml

    from WP2.config import HebbianEvolutionConfig
    from .config import OuterNSGA2Config
    from .exam_baseline_rerun import _rebase_container_path

    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text()) or {}
    is_wp2 = "hebbian" in raw or "checkpoint_path" in raw

    if is_wp2:
        cls = OuterNSGA2Config if "outer" in raw else HebbianEvolutionConfig
        cfg = cls.from_yaml(config_path)
        cfg.checkpoint_path = _rebase_container_path(cfg.checkpoint_path, repo_root)
        cfg.checkpoint_config_path = _rebase_container_path(
            cfg.checkpoint_config_path, repo_root
        )
        if checkpoint:
            cfg.checkpoint_path = str(checkpoint)
        kind = "wp2"
    else:
        if not checkpoint:
            raise ValueError(
                f"{config_path} looks like a WP1 config (no hebbian/checkpoint_path "
                f"section), so --checkpoint is required to say which actor to fly"
            )
        cfg = HebbianEvolutionConfig()
        cfg.checkpoint_path = str(checkpoint)
        cfg.checkpoint_config_path = str(config_path)
        kind = "wp1"

    if not Path(cfg.checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")
    if not Path(cfg.checkpoint_config_path).exists():
        raise FileNotFoundError(
            f"WP1 config not found: {cfg.checkpoint_config_path}"
        )
    return cfg, kind


def zero_rules_genome(cfg) -> np.ndarray:
    """The generalist: ABCD = 0 (mid-range on symmetric bounds) and an all-zero
    decay block when decay is evolved. Flown through the same Hebbian wrapper
    as an evolved genome, so ΔW = 0 and only the rules differ between the two
    controllers — no second code path to diverge."""
    genome = np.full(cfg.hebbian_genome_dim(), 0.5)
    if cfg.hebbian.evolve_decay:
        n_weights = cfg.hebbian.num_actions * cfg.hebbian.hidden_dim
        start = 4 * cfg.hebbian.abcd_block_size()
        genome[start: start + n_weights] = 0.0
    return genome


def _best_genome_from_run(run_dir: Path) -> Tuple[np.ndarray, str]:
    """Highest-fitness genome of a run.

    Prefers ``best_individual/fitness/genome.npy``, which ``_finalize`` writes.
    Interrupted runs (TIMEOUT) never reach it, so fall back to the same
    reconstruction ``WP2.recover_best`` performs: argmax fitness in
    ``results/cma_population.csv`` → ``generations/gen_XXX/solutions.npy``.
    """
    saved = run_dir / "best_individual" / "fitness" / "genome.npy"
    if saved.is_file():
        return np.load(saved).astype(np.float64), str(saved)

    pop_csv = run_dir / "results" / "cma_population.csv"
    if not pop_csv.is_file():
        raise FileNotFoundError(
            f"No {saved} and no {pop_csv} — cannot resolve the best genome. "
            f"Pass --genome PATH explicitly."
        )
    pop = pd.read_csv(pop_csv)
    if pop.empty or "fitness" not in pop.columns:
        raise ValueError(f"{pop_csv} has no usable fitness column")
    row = pop.loc[pop["fitness"].idxmax()]
    gen, idx = int(row["generation"]), int(row["individual_idx"])
    sol_path = run_dir / "generations" / f"gen_{gen:03d}" / "solutions.npy"
    if not sol_path.is_file():
        raise FileNotFoundError(
            f"Best individual is gen {gen} ind {idx}, but {sol_path} is missing "
            f"(pruned or never synced). Pass --genome PATH explicitly."
        )
    solutions = np.load(sol_path)
    if idx >= solutions.shape[0]:
        raise IndexError(
            f"Individual {idx} out of range for gen {gen} "
            f"(population {solutions.shape[0]})"
        )
    return (solutions[idx].astype(np.float64),
            f"{sol_path}[{idx}] (gen {gen}, fitness {float(row['fitness']):.6g})")


def resolve_genome(
    spec: str, cfg, run_dir: Optional[Path],
) -> Tuple[np.ndarray, str]:
    """``(genome, human-readable source)`` for ``--genome zero|best|PATH``."""
    if spec == "zero":
        return zero_rules_genome(cfg), "zero rules (generalist)"
    if spec == "best":
        if run_dir is None:
            raise ValueError(
                "--genome best needs a run directory; pass --run-dir, or point "
                "--config at <run>/reproducibility/config.yaml"
            )
        return _best_genome_from_run(run_dir)
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"--genome {spec}: no such file")
    return np.load(path).astype(np.float64), str(path)


def _fit_genome_to_checkpoint(cfg) -> None:
    """Re-infer the last-layer shape from the checkpoint so the genome length
    matches this actor's architecture (mirrors the in-run reference path)."""
    import torch
    from WP2.frozen_actor import last_actor_linear_key

    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    key = last_actor_linear_key(sd)
    if key is not None:
        cfg.hebbian.num_actions = sd[key].shape[0]
        cfg.hebbian.hidden_dim = sd[key].shape[1]


# ----------------------------------------------------------------------------
#  Forest settings
# ----------------------------------------------------------------------------

def forest_field_names() -> Tuple[str, ...]:
    from WP2.config import ForestConfig
    return tuple(f.name for f in dataclasses.fields(ForestConfig))


def apply_forest_settings(cfg, settings: Dict[str, object]) -> Dict[str, object]:
    """Write ``settings`` onto ``cfg.forest``; return the resolved section.

    Applied at env build time (``WP2.evaluate._apply_forest_overrides`` folds
    ``cfg.forest`` into ``env_cfg``), which is why perception-coupled fields
    are allowed here — the DepthSolver is constructed from the same values.
    """
    valid = forest_field_names()
    for key, val in settings.items():
        if key not in valid:
            raise KeyError(
                f"Unknown forest field {key!r}. Valid: {', '.join(sorted(valid))}"
            )
        setattr(cfg.forest, key, val)
    # evaluation.* legacy overrides sit between the WP1 config and cfg.forest;
    # leaving them set would silently win over an unset cfg.forest field.
    for legacy in ("x_upper", "dens_min", "dens_max"):
        if settings.get(legacy) is not None:
            setattr(cfg.evaluation, legacy, None)
    return {f: getattr(cfg.forest, f) for f in valid
            if getattr(cfg.forest, f) is not None}


def _parse_forest_args(args) -> Dict[str, object]:
    """Explicit flags + repeatable ``--forest KEY=VALUE``, typed per field."""
    from WP2.config import ForestConfig

    settings: Dict[str, object] = {}
    for name in ("x_lower", "x_upper", "y_lower", "y_upper", "dens_min",
                 "dens_max", "num_trees", "tree_radius", "tree_height",
                 "forest_length"):
        val = getattr(args, name, None)
        if val is not None:
            settings[name] = val
    if args.forest_mode is not None:
        settings["mode"] = args.forest_mode

    types = {f.name: f.type for f in dataclasses.fields(ForestConfig)}
    for item in args.forest or []:
        if "=" not in item:
            raise ValueError(f"--forest expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        if key not in types:
            raise KeyError(
                f"Unknown forest field {key!r}. Valid: "
                f"{', '.join(sorted(types))}"
            )
        raw = raw.strip()
        if raw.lower() in ("none", "null", ""):
            settings[key] = None
        elif key in ("mode",):
            settings[key] = raw
        elif "int" in str(types[key]):
            settings[key] = int(raw)
        else:
            settings[key] = float(raw)
    return settings


# ----------------------------------------------------------------------------
#  Measurement
# ----------------------------------------------------------------------------

def _chunks(seq: Sequence, size: int) -> List[Sequence]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _front_generation(df: pd.DataFrame) -> Optional[int]:
    """The outer generation these rows came from, ignoring the standard-drone
    reference (which carries ``outer_gen = -1``). ``None`` when the frame holds
    nothing but the reference."""
    if "outer_gen" not in df.columns:
        return None
    gens = df["outer_gen"]
    if "is_standard" in df.columns:
        gens = gens[df["is_standard"].fillna(0).astype(int) != 1]
    gens = gens[gens >= 0]
    return int(gens.max()) if len(gens) else None


def _default_title(tag: str, df: pd.DataFrame) -> str:
    """Figure title for one controller's results, from the data alone — so a
    replot through ``--plot`` reproduces what the measurement run wrote."""
    gen = _front_generation(df)
    if gen is not None:
        return f"{tag} on gen-{gen} front morphologies"
    n = len(df)
    if "is_standard" in df.columns:
        n = int((df["is_standard"].fillna(0).astype(int) != 1).sum())
    if n == 0:
        return f"{tag} on the Bixler"
    return f"{tag} on {n} random morphologies"


def fly_front(
    front: pd.DataFrame,
    cfg,
    genome: np.ndarray,
    urdf_dir: Path,
    n_forests: int,
    chunk_morphs: int,
    forest_seed: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fly every morphology in ``front`` with one controller.

    Builds one ``MultiSceneEvalEnv`` per chunk of at most ``chunk_morphs``
    URDFs, each with ``n_forests`` slots per drone, re-seeding the device RNG
    to ``forest_seed`` before every ``refresh_forests()`` so all chunks (and
    any other invocation with the same seed) fly identical layouts.

    Returns one row per morphology: each metric's mean over the forests plus
    its standard error across them.
    """
    import genesis as gs
    import torch
    from WP1.config import RunConfig
    from WP2.evaluate import _build_multi_urdf_env, evaluate_population_multi_urdf
    from WP2.frozen_actor import load_frozen_actor
    from .urdf_population import materialize_urdfs

    genes = _gene_columns(front)
    genomes = front[genes].to_numpy(dtype=float).tolist()

    gs.init(backend=gs.gpu, logging_level="warning")
    urdf_paths = materialize_urdfs(genomes, urdf_dir)
    if verbose:
        print(f"[transfer] Materialized {len(urdf_paths)} URDFs → {urdf_dir}")

    wp1_cfg = RunConfig.from_yaml(cfg.checkpoint_config_path)
    model, last_layer, num_actions, hidden_dim = load_frozen_actor(
        cfg.checkpoint_path, cfg.checkpoint_config_path, device=cfg.device
    )
    model_and_layer = (model, last_layer, num_actions, hidden_dim)

    rows: List[Dict[str, float]] = []
    groups = _chunks(list(range(len(urdf_paths))), chunk_morphs)
    try:
        for ci, idxs in enumerate(groups):
            # Each chunk owns a full Genesis lifecycle: the previous chunk's
            # scenes must be gone before the next set is built, or D scenes
            # accumulate and defeat the point of chunking.
            if not gs._initialized:
                gs.init(backend=gs.gpu, logging_level="warning")
            paths = [urdf_paths[i] for i in idxs]
            chunk_cfg = copy.deepcopy(cfg)
            chunk_cfg.evaluation.num_eval_envs = len(paths) * n_forests
            chunk_cfg.evaluation.run_baseline = False
            chunk_cfg.evaluation.refresh_forests_per_generation = False
            chunk_cfg.catalog.path = ""
            chunk_cfg.catalog.num_urdfs = len(paths)
            chunk_cfg.catalog.force_multi_urdf = True

            if verbose:
                print(f"[transfer] Chunk {ci + 1}/{len(groups)}: "
                      f"{len(paths)} morphs × {n_forests} forests")
            env = _build_multi_urdf_env(
                paths, chunk_cfg, wp1_cfg, chunk_cfg.device,
                num_envs_per_drone=n_forests, num_workers=1,
            )
            try:
                # Same seed every chunk ⇒ every morphology, in this run and in
                # any other with this seed, flies the same n_forests layouts.
                torch.manual_seed(forest_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(forest_seed)
                env.refresh_forests()

                _fit, metrics = evaluate_population_multi_urdf(
                    [genome], chunk_cfg, model_and_layer, wp1_cfg,
                    urdf_paths=paths, existing_env=(env, paths), verbose=False,
                )
            finally:
                try:
                    gs.destroy()
                except Exception as exc:
                    print(f"[transfer] gs.destroy() failed: {exc}")

            for local, global_i in enumerate(idxs):
                src = front.iloc[global_i]
                row: Dict[str, float] = {
                    "urdf_idx": int(src["urdf_idx"]),
                    "urdf_file": Path(urdf_paths[global_i]).name,
                    "outer_gen": int(src["outer_gen"]),
                    "front_rank": int(src.get("front_rank", local)),
                    "is_standard": int(src.get("is_standard", 0)),
                }
                for col, pu_key, ps_key in _METRICS:
                    row[col] = float(np.asarray(metrics[pu_key])[local, 0])
                    slots = np.asarray(metrics[ps_key])[local, 0]
                    row[f"{col}_se"] = (
                        float(np.std(slots, ddof=1) / np.sqrt(len(slots)))
                        if len(slots) > 1 else 0.0
                    )
                for col in ("progress_m", "cost_of_transport", "velocity",
                            "crash_rate", "fitness"):
                    if col in src.index:
                        row[f"archived_{col}"] = float(src[col])
                for g in genes:
                    row[g] = float(src[g])
                rows.append(row)
            if verbose:
                done = rows[-len(idxs):]
                print(f"[transfer]   progress {np.mean([r['progress_m'] for r in done]):.1f} m  "
                      f"CoT {np.mean([r['cost_of_transport'] for r in done]):.4f}  "
                      f"crash {np.mean([r['crash_rate'] for r in done]):.3f}")
    finally:
        if gs._initialized:
            try:
                gs.destroy()
            except Exception:
                pass

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
#  Output
# ----------------------------------------------------------------------------

def build_provenance(cfg, args, forest: Dict, genome_source: str,
                     n_morphs: int) -> Dict:
    """Everything that must match before two result CSVs may be overlaid, plus
    non-comparable context (checkpoint, genome) kept for the record."""
    ckpt = Path(cfg.checkpoint_path)
    return {
        "tag": args.tag,
        "forest": {k: v for k, v in sorted(forest.items())},
        "forest_seed": int(args.forest_seed),
        "n_forests": int(args.n_forests),
        "vmin": float(cfg.evaluation.vmin),
        "vmax": float(cfg.evaluation.vmax),
        "stochastic": bool(cfg.evaluation.stochastic),
        "num_eval_episodes": int(cfg.catalog.num_episodes),
        "crn": bool(getattr(cfg.evaluation, "crn", True)),
        "checkpoint": str(ckpt),
        "checkpoint_md5": (hashlib.md5(ckpt.read_bytes()).hexdigest()
                           if ckpt.is_file() else ""),
        "genome_source": genome_source,
        # Absent when only the standard-drone reference was flown.
        "front_csv": (str(Path(args.front_csv).resolve())
                      if args.front_csv else None),
        "outer_gen": args.gen if args.front_csv else None,
        "random_morphs": int(args.random_morphs or 0),
        "morph_seed": int(args.morph_seed),
        "n_morphologies": n_morphs,
    }


def write_results(df: pd.DataFrame, provenance: Dict, out_path: Path) -> Path:
    """Result CSV with the provenance as a leading ``#`` comment line."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        f.write(_PROVENANCE_PREFIX + json.dumps(provenance, sort_keys=True) + "\n")
        df.to_csv(f, index=False)
    print(f"[transfer] Wrote {out_path} — {len(df)} morphologies")
    return out_path


def read_results(path: Path) -> Tuple[pd.DataFrame, Dict]:
    """Inverse of ``write_results``; ``{}`` when the header is absent."""
    path = Path(path)
    first = path.open().readline()
    provenance: Dict = {}
    if first.startswith(_PROVENANCE_PREFIX):
        try:
            provenance = json.loads(first[len(_PROVENANCE_PREFIX):])
        except json.JSONDecodeError:
            provenance = {}
    return pd.read_csv(path, comment="#"), provenance


def comparability_conflicts(provs: Sequence[Dict]) -> List[str]:
    """Keys on which the given provenances disagree — an overlay of results
    measured on different forests, seeds or forest counts is not a controller
    comparison, so ``--plot`` refuses unless forced."""
    conflicts = []
    for key in _COMPARABLE_KEYS:
        vals = [json.dumps(p.get(key), sort_keys=True) for p in provs]
        if len(set(vals)) > 1:
            conflicts.append(
                f"{key}: " + " vs ".join(sorted(set(vals)))
            )
    return conflicts


def plot_transfer(
    results: Sequence[Tuple[str, pd.DataFrame]],
    out_path: Path,
    title: str = "",
) -> Path:
    """CoT-vs-progress scatter per controller, each with its own front line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    def _split_standard(df):
        std = (df.get("is_standard", pd.Series(0, index=df.index))
               .fillna(0).astype(int) == 1).to_numpy()
        return df[~std], df[std]

    # With exactly two controllers, join each morphology's two measurements
    # (matched on urdf_idx) with a dotted segment, under the points, so the
    # per-morph controller shift is readable straight off the overlay.
    if len(results) == 2:
        a, b = _split_standard(results[0][1])[0], _split_standard(results[1][1])[0]
        pair = a.merge(b, on="urdf_idx", suffixes=("_a", "_b"))
        for _, r in pair.iterrows():
            ax.plot([r["progress_m_a"], r["progress_m_b"]],
                    [r["cost_of_transport_a"], r["cost_of_transport_b"]],
                    linestyle=":", color="0.45", linewidth=0.8, alpha=0.65,
                    zorder=1)

    for i, (label, df) in enumerate(results):
        color = palette[i % len(palette)]
        # The unevolved standard mydrone is a reference point, not a front
        # member: it gets a star and is kept out of the scatter and the
        # nondominated line, which describe the evolved population.
        morphs, stds = _split_standard(df)

        x = morphs["progress_m"].to_numpy(dtype=float)
        y = morphs["cost_of_transport"].to_numpy(dtype=float)
        if len(morphs):
            ax.errorbar(
                x, y,
                xerr=morphs.get("progress_m_se"),
                yerr=morphs.get("cost_of_transport_se"),
                fmt="o", color=color, markersize=5, alpha=0.75, elinewidth=0.7,
                capsize=0, label=f"{label}  (n={len(morphs)})",
            )
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum():
            pts = np.column_stack([x[finite], -y[finite]])  # maximization space
            front = pts[_nondominated_mask(pts)]
            front = front[np.argsort(front[:, 0])]
            ax.plot(front[:, 0], -front[:, 1], "-", color=color,
                    linewidth=1.8, alpha=0.9)

        for _, s in stds.iterrows():
            ax.errorbar(
                [s["progress_m"]], [s["cost_of_transport"]],
                xerr=[[s.get("progress_m_se", 0.0)]] * 2,
                yerr=[[s.get("cost_of_transport_se", 0.0)]] * 2,
                fmt="*", color=color, markersize=20, markeredgecolor="black",
                markeredgewidth=0.8, elinewidth=0.9, capsize=0, zorder=5,
                label=f"Bixler — {label}",
            )

    ax.set_xlabel("Progress [m]  (higher better)")
    ax.set_ylabel("Cost of transport  (lower better)")
    ax.set_title(title or "Front morphologies under a swapped controller")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[transfer] Saved {out_path}")
    return out_path


def paired_summary(
    a_label: str, a: pd.DataFrame, b_label: str, b: pd.DataFrame,
) -> str:
    """Per-morphology paired deltas — the statistic that answers 'did the
    controller swap change anything', since both flew the same morphs on the
    same forests."""
    def _morphs(df):
        if "is_standard" not in df.columns:
            return df
        return df[df["is_standard"].fillna(0).astype(int) != 1]

    # The reference drone is not part of the evolved population, so it must not
    # enter the paired mean over front morphologies.
    a, b = _morphs(a), _morphs(b)
    merged = a.merge(b, on="urdf_idx", suffixes=("_a", "_b"))
    if merged.empty:
        return "[transfer] No shared urdf_idx between the two results."
    lines = [f"[transfer] Paired delta ({b_label} − {a_label}) over "
             f"{len(merged)} shared morphologies:"]
    for col in ("progress_m", "cost_of_transport", "crash_rate"):
        d = merged[f"{col}_b"].to_numpy(float) - merged[f"{col}_a"].to_numpy(float)
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        verdict = "" if se and abs(d.mean()) > 2 * se else "   (within 2×SE — not resolved)"
        lines.append(f"    {col:>18}: {d.mean():+.4g} ± {se:.3g}{verdict}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def add_forest_arguments(ap: argparse.ArgumentParser) -> None:
    """The forest-settings flag group, shared with ``controller_generality``
    so both tools accept identical overrides (parsed by
    ``_parse_forest_args``)."""
    g = ap.add_argument_group("forest settings (applied at env build)")
    for name, typ, help_ in (
        ("x-lower", float, "forest start [m]"),
        ("x-upper", float, "forest end + eval success line [m]"),
        ("y-lower", float, "corridor lower bound [m]"),
        ("y-upper", float, "corridor upper bound [m]"),
        ("dens-min", float, "density at x_lower [trees/m]"),
        ("dens-max", float, "density at x_upper [trees/m]"),
        ("num-trees", int, "tree count (uniform mode)"),
        ("tree-radius", float, "tree radius [m]"),
        ("tree-height", float, "tree height [m]"),
        ("forest-length", float, "lattice/latin mode length [m]"),
    ):
        g.add_argument(f"--{name}", type=typ, default=None, help=help_)
    g.add_argument("--forest-mode", default=None,
                   help="uniform | growing | lattice | latin")
    g.add_argument("--forest", action="append", metavar="KEY=VALUE",
                   help="Any other cfg.forest field; repeatable.")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Re-fly an outer run's Pareto-front morphologies under one "
                    "controller, for CoT/progress transfer comparison.")
    ap.add_argument("front_csv", nargs="?", type=Path,
                    help="results/pareto_front.csv from an outer run.")
    ap.add_argument("--config", type=Path,
                    help="WP2 config (carries its own checkpoint) or WP1 "
                         "config (then --checkpoint is required).")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Actor checkpoint; required for a WP1 config, an "
                         "override for a WP2 one.")
    ap.add_argument("--genome", default="best",
                    help="'best' (run's top-fitness genome), 'zero' (the "
                         "generalist: no plasticity), or a path to genome.npy.")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Run directory for --genome best "
                         "(default: the config's grandparent).")
    ap.add_argument("--gen", default="last",
                    help="Which outer generation's front ('last' or an int).")
    ap.add_argument("--no-dedup", action="store_true",
                    help="Keep duplicate genomes instead of collapsing them.")
    ap.add_argument("--random-morphs", type=int, default=None,
                    help="Fly N morphologies sampled uniformly in [0,1]^15 "
                         "(the outer loop's own gen-0 distribution) instead of "
                         "a front. Reproducible from --morph-seed.")
    ap.add_argument("--morph-seed", type=int, default=0,
                    help="RNG seed for --random-morphs (default 0).")
    ap.add_argument("--standard-drone", action="store_true",
                    help="Also fly the unevolved standard mydrone as a "
                         "reference (plotted as a star). Omit FRONT_CSV to fly "
                         "ONLY the reference — cheap, and its forests match "
                         "any run using the same --forest-seed.")
    ap.add_argument("--tag", default=None,
                    help="Name for this controller in outputs "
                         "(default: derived from --genome).")
    ap.add_argument("--n-forests", type=int, default=32,
                    help="Forests (= env slots) per morphology. Also the "
                         "resolution of the vmin..vmax speed grid.")
    ap.add_argument("--chunk-morphs", type=int, default=16,
                    help="Morphologies per Genesis env build (VRAM knob).")
    ap.add_argument("--forest-seed", type=int, default=0,
                    help="Device RNG seed applied before every forest "
                         "refresh; identical seeds ⇒ identical layouts.")
    ap.add_argument("--device", default=None, help="Override cfg.device.")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="Checkout to rebase container paths onto.")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output directory (measure) or PNG path (--plot).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve everything, print the plan, fly nothing.")

    add_forest_arguments(ap)

    ap.add_argument("--plot", nargs="+", type=Path, default=None,
                    help="Overlay mode: result CSVs to plot together.")
    ap.add_argument("--force", action="store_true",
                    help="Overlay result CSVs that disagree on forest "
                         "settings / seed / forest count (normally refused).")
    ap.add_argument("--title", default=None,
                    help="Override the figure title (--plot mode).")
    return ap


def _plot_mode(args) -> int:
    loaded = [read_results(p) for p in args.plot]
    provs = [p for _df, p in loaded]
    conflicts = comparability_conflicts(provs)
    if conflicts:
        print("[transfer] These results were not measured under the same "
              "conditions, so overlaying them would not isolate the controller:")
        for c in conflicts:
            print(f"  - {c}")
        if not args.force:
            print("[transfer] Refusing to plot. Re-measure with matching "
                  "settings, or pass --force if you are sure.")
            return 1
        print("[transfer] --force given: plotting anyway.")

    # Merge same-tag CSVs so a standalone standard-drone measurement lands on
    # its controller's series (and colour) rather than becoming a series of
    # its own.
    labelled: List[Tuple[str, pd.DataFrame]] = []
    order: List[str] = []
    by_tag: Dict[str, List[pd.DataFrame]] = {}
    for (df, p), path in zip(loaded, args.plot):
        tag = p.get("tag") or Path(path).stem
        if tag not in by_tag:
            by_tag[tag] = []
            order.append(tag)
        by_tag[tag].append(df)
    for tag in order:
        labelled.append((tag, pd.concat(by_tag[tag], ignore_index=True)))
    out = args.out or Path(args.plot[0]).parent / "transfer_compare.png"
    if out.suffix.lower() != ".png":
        out = out / "transfer_compare.png"
    # The forest settings are not repeated under the title: they are identical
    # across the overlaid results by construction (comparability_conflicts
    # above refuses otherwise), and each CSV carries them in its provenance.
    title = "Front morphologies under swapped controllers"
    if len(labelled) == 1:
        # Replotting one controller should reproduce the title its measurement
        # run wrote, not the overlay's.
        title = _default_title(*labelled[0])
    plot_transfer(labelled, out, title=args.title or title)
    if len(labelled) == 2:
        print(paired_summary(labelled[0][0], labelled[0][1],
                             labelled[1][0], labelled[1][1]))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.plot:
        return _plot_mode(args)
    if not args.config or not (args.front_csv or args.standard_drone
                               or args.random_morphs):
        print("Usage: transfer_eval FRONT_CSV --config CFG [...]\n"
              "       transfer_eval --config CFG --random-morphs 128 [...]\n"
              "       transfer_eval --config CFG --standard-drone [...]\n"
              "       transfer_eval --plot A.csv B.csv")
        return 1
    if args.front_csv and args.random_morphs:
        print("--random-morphs replaces the front; pass one or the other.")
        return 1

    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    cfg, kind = load_controller_config(args.config, repo_root, args.checkpoint)
    if args.device:
        cfg.device = args.device
    _fit_genome_to_checkpoint(cfg)

    run_dir = args.run_dir
    if run_dir is None and args.config.parent.name == "reproducibility":
        run_dir = args.config.parent.parent
    genome, genome_source = resolve_genome(args.genome, cfg, run_dir)
    expected = cfg.hebbian_genome_dim()
    if genome.shape[0] != expected:
        raise ValueError(
            f"Genome from {genome_source} has {genome.shape[0]} genes but this "
            f"checkpoint/config implies {expected} "
            f"({cfg.hebbian.num_actions}×{cfg.hebbian.hidden_dim} last layer)"
        )

    forest = apply_forest_settings(cfg, _parse_forest_args(args))
    if args.front_csv:
        front = select_front(args.front_csv, args.gen, dedup=not args.no_dedup)
    elif args.random_morphs:
        front = random_morph_rows(args.random_morphs, args.morph_seed)
        print(f"[transfer] {len(front)} random morphologies "
              f"(uniform [0,1]^15, seed {args.morph_seed})")
    else:
        front = None
    if args.standard_drone:
        n_genes = len(_gene_columns(front)) if front is not None else 15
        front = with_standard_drone(front, n_genes)
        print("[transfer] Standard mydrone included among the morphologies")
    tag = args.tag or ("generalist" if args.genome == "zero" else "hebbian")
    # Each source gets its own basename so measurements never clobber one
    # another when they share a tag.
    stem = (f"transfer_{tag}" if args.front_csv
            else f"random_{tag}" if args.random_morphs
            else f"standard_{tag}")
    out_dir = args.out or (Path(args.front_csv).parent / "transfer"
                           if args.front_csv else Path.cwd())

    provenance = build_provenance(cfg, args, forest, genome_source, len(front))
    n_chunks = (len(front) + args.chunk_morphs - 1) // args.chunk_morphs
    print(f"[transfer] Controller: {tag} ({kind} config, {genome_source})")
    print(f"[transfer] Checkpoint: {cfg.checkpoint_path}")
    print(f"[transfer] Forest:     {json.dumps(forest, sort_keys=True)}")
    print(f"[transfer] Plan:       {len(front)} morphs in {n_chunks} chunk(s) "
          f"of ≤{args.chunk_morphs}, {args.n_forests} forests each "
          f"({args.chunk_morphs * args.n_forests} slots/chunk), seed "
          f"{args.forest_seed}")
    print(f"[transfer] Output:     {out_dir / (stem + '.csv')}")
    if args.dry_run:
        print("[transfer] --dry-run: stopping before measurement.")
        return 0

    df = fly_front(
        front, cfg, genome,
        urdf_dir=out_dir / "urdfs" / stem,
        n_forests=args.n_forests,
        chunk_morphs=args.chunk_morphs,
        forest_seed=args.forest_seed,
    )
    write_results(df, provenance, out_dir / f"{stem}.csv")
    plot_transfer([(tag, df)], out_dir / f"{stem}.png",
                  title=args.title or _default_title(tag, df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
