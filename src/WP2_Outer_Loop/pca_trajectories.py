"""
Joint-PCA morphology trajectories across outer-loop runs.
=========================================================

Concatenates the normalized 15D morphology genomes of several runs — every
row of ``results/outer_population.csv``, in appearance order (run, outer
gen, individual) — fits ONE shared PCA on the joint matrix, and plots each
run's per-generation centroid (mean genome) as a path in PC1–PC2 space.
Because both runs are projected into the same basis, the paths show whether
e.g. a morphology-only run and a full co-design run drift toward different
regions of morphology space.

Four figures per invocation:

* ``pca_population.png``   — centroids over ALL evaluated genomes per gen.
* ``pca_pareto_front.png`` — centroids over each gen's Pareto-front members
  only (``results/pareto_front.csv``, backfilled when missing; exam-filtered
  and min-progress-gated like the front plots), with its own PCA fit.
* ``*_centroid_fit.png``   — same two, but the PCA is fit on the per-gen
  centroid genomes instead of all rows: within-gen mutation spread stops
  eating explained variance, so the PCs align with the drift of the mean.

Both figures are written into EVERY included run's ``plots/`` folder, under
a subfolder naming all included runs, e.g.
``<run_dir>/plots/pca_trajectories_morphology_only__codesign/``.

**Condition groups.** With ``--group LABEL COLOR RUN_DIR...`` (repeatable)
runs are coloured by group instead of one colormap per run (light → dark
still encodes generation), each end star is annotated with the run's seed
(``reproducibility/config.yaml``), and two CSVs accompany the figures:
``final_centroid_distances.csv`` (15-D Euclidean distance between the
last-generation centroids of every run pair, tagged *same condition* /
*across conditions, same seed* / *across conditions, different seed*) and
``distance_summary.csv`` (n / mean / min / max per relation), plus
``*_by_seed.png``: the same shared-basis paths split into one panel per seed
(only that seed's runs, common axis limits). That is the seed-paired
question: do a co-design run and a morphology-only run that
started from the same random catalog end nearer each other than runs that
did not? ``--out DIR`` writes everything to one folder instead of every
run's ``plots/``.

Usage
-----
    PYTHONPATH=src python -m WP2_Outer_Loop.pca_trajectories \
        <run_dir_1> <run_dir_2> [...] [--labels NAME1 NAME2 ...]

    PYTHONPATH=src python -m WP2_Outer_Loop.pca_trajectories \
        --group co-design "#c0392b" <run> <run> ... \
        --group morphology-only "#1f5fa8" <run> <run> ... \
        [--labels ...] --out logs/remote/outer_nsga/pca_exam_seed_paired
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

from .pareto_fronts import build_pareto_front_csv_safe

_GENE_RE = re.compile(r"^g\d+$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_")

# One sequential colormap per run (light -> dark encodes outer generation),
# hues in a fixed colorblind-aware order, never cycled per-generation.
_RUN_CMAPS = ("Blues", "Oranges", "Greens", "Purples", "Reds", "Greys",
              "YlOrBr", "RdPu")

_MAX_DIRNAME_LEN = 150


# ----------------------------------------------------------------------------
#  Data loading
# ----------------------------------------------------------------------------

def _gene_columns(df: pd.DataFrame) -> List[str]:
    """The ``g<i>`` genome columns, in file order (g0, g1, ... g14)."""
    return [c for c in df.columns if _GENE_RE.match(c)]


def load_genomes(run_dir: Path | str, source: str = "population") -> pd.DataFrame:
    """``outer_gen`` + gene columns for one run, rows sorted by outer gen.

    ``source`` is ``"population"`` (every evaluated genome, from
    ``outer_population.csv``) or ``"front"`` (Pareto-front members only, from
    ``pareto_front.csv`` — rebuilt from the population CSV when possible, so
    a missing or stale front file self-heals like in ``pareto_plots``).
    """
    run_dir = Path(run_dir)
    if source == "population":
        csv_path = run_dir / "results" / "outer_population.csv"
    elif source == "front":
        build_pareto_front_csv_safe(run_dir)
        csv_path = run_dir / "results" / "pareto_front.csv"
    else:
        raise ValueError(f"Unknown genome source: {source!r}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"No {source} genomes: {csv_path} is missing")

    df = pd.read_csv(csv_path)
    genes = _gene_columns(df)
    if not genes:
        raise ValueError(f"No g<i> genome columns in {csv_path}")
    df = df[["outer_gen"] + genes]
    return df.sort_values("outer_gen", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------
#  PCA + centroid paths
# ----------------------------------------------------------------------------

def load_seed(run_dir: Path | str) -> Optional[int]:
    """Top-level ``seed`` of the run's saved single-file config; ``None``
    when the config or the key is missing."""
    cfg_path = Path(run_dir) / "reproducibility" / "config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        seed = cfg.get("seed")
        return int(seed) if seed is not None else None
    except Exception:
        return None


@dataclass
class RunGroup:
    """Runs of one experimental condition, drawn in one colour."""
    label: str
    color: str
    run_dirs: List[Path]
    names: List[str] = field(default_factory=list)


def _group_of(groups: Sequence[RunGroup]) -> Dict[str, RunGroup]:
    return {name: g for g in groups for name in g.names}


def fit_pca(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """First two principal components of ``X`` (rows = genomes).

    Returns ``(components, explained)``: a (2, n_genes) row-vector basis and
    the two explained-variance ratios. Plain PCA on mean-centered data — the
    genes already share the [0, 1] scale, so no per-gene standardization
    (z-scoring would inflate near-constant genes into noise directions).
    Component signs are fixed so each PC's largest-magnitude loading is
    positive, making outputs reproducible across BLAS/SVD implementations.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError(f"PCA needs a (>=2, >=2) matrix, got {X.shape}")
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    total = float((s ** 2).sum())
    explained = (s[:2] ** 2 / total) if total > 0 else np.zeros(2)
    components = vt[:2].copy()
    for pc in components:
        if pc[np.argmax(np.abs(pc))] < 0:
            pc *= -1.0
    return components, explained


def centroid_paths(
    dfs: Sequence[pd.DataFrame], labels: Sequence[str],
    fit_on: str = "rows",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Per-generation centroids of every run in one shared PC1–PC2 basis.

    ``dfs`` are ``load_genomes`` frames, one per run. Each run's genomes are
    averaged per outer generation — persistent NSGA-II elites appear once
    per generation they survive, which deliberately weights centroids toward
    where the population sits.

    ``fit_on`` picks the matrix the shared PCA is fit on:

    * ``"rows"`` — the concatenation of ALL individual genomes. Explained
      variance includes within-generation mutation spread.
    * ``"centroids"`` — the per-(run, gen) centroid genomes themselves,
      suppressing within-gen noise so the PCs align with the drift of the
      population mean.

    Either way the projected points are the same centroids, just in a
    different basis. Returns ``(paths, explained)``: a frame with columns
    ``run, outer_gen, pc1, pc2`` — runs in input order, generations
    ascending — and the two explained-variance ratios of the fit matrix.
    """
    gene_sets = [_gene_columns(df) for df in dfs]
    if any(g != gene_sets[0] for g in gene_sets[1:]):
        raise ValueError(
            "gene columns differ between runs: "
            + "; ".join(f"{lab}: {len(g)}" for lab, g in zip(labels, gene_sets))
        )
    genes = gene_sets[0]

    meta = []  # (run label, outer_gen) per centroid row
    centroids = []
    for label, df in zip(labels, dfs):
        G = df[genes].to_numpy(dtype=float)
        gens = df["outer_gen"].to_numpy()
        for gen in np.unique(gens):  # np.unique sorts ascending
            meta.append((label, gen))
            centroids.append(G[gens == gen].mean(axis=0))
    C = np.vstack(centroids)

    if fit_on == "rows":
        X = np.vstack([df[genes].to_numpy(dtype=float) for df in dfs])
    elif fit_on == "centroids":
        X = C
    else:
        raise ValueError(f"Unknown fit_on: {fit_on!r} "
                         "(expected 'rows' or 'centroids')")
    components, explained = fit_pca(X)
    # Projecting the centroids == averaging the projected rows (linearity),
    # so the "rows" fit reproduces the mean of per-genome projections.
    proj = (C - X.mean(axis=0)) @ components.T

    paths = pd.DataFrame(
        [(label, gen, p[0], p[1]) for (label, gen), p in zip(meta, proj)],
        columns=["run", "outer_gen", "pc1", "pc2"],
    )
    return paths, explained


# ----------------------------------------------------------------------------
#  Run labels / output naming
# ----------------------------------------------------------------------------

def run_labels(
    run_dirs: Sequence[Path | str], labels: Optional[Sequence[str]] = None,
) -> List[str]:
    """One unique label per run for legends and the output folder name.

    Explicit ``labels`` win. Otherwise the run dir's basename with the
    ``YYYY-MM-DD_HH-MM-SS_`` timestamp stripped; colliding names (e.g. r0/r2
    repeats of the same config) get their parent directory name prefixed,
    and any survivors of that get a positional suffix.
    """
    if labels is not None:
        if len(labels) != len(run_dirs):
            raise ValueError(
                f"{len(labels)} labels for {len(run_dirs)} run dirs"
            )
        return list(labels)

    dirs = [Path(d) for d in run_dirs]
    out = [_TIMESTAMP_RE.sub("", d.name) or d.name for d in dirs]
    dupes = {label for label in out if out.count(label) > 1}
    out = [
        f"{d.parent.name}_{label}" if label in dupes and d.parent.name
        else label
        for label, d in zip(out, dirs)
    ]
    dupes = {label for label in out if out.count(label) > 1}
    return [
        f"{label}_{i}" if label in dupes else label
        for i, label in enumerate(out)
    ]


def _output_dirname(labels: Sequence[str]) -> str:
    """``pca_trajectories_<lab1>__<lab2>...``, hash-truncated when too long
    for one path component."""
    name = "pca_trajectories_" + "__".join(labels)
    if len(name) > _MAX_DIRNAME_LEN:
        digest = hashlib.sha1(name.encode()).hexdigest()[:10]
        name = name[:_MAX_DIRNAME_LEN - 11] + "_" + digest
    return name


# ----------------------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------------------

def _group_cmap(color: str) -> LinearSegmentedColormap:
    """White → group colour → near-black ramp, so light = early generation
    and dark = late, the same convention as the per-run colormaps."""
    rgb = np.array(to_rgb(color))
    return LinearSegmentedColormap.from_list(
        f"grp_{color}", [(1, 1, 1), tuple(rgb), tuple(rgb * 0.45)])


def _draw_run_path(ax, sub: pd.DataFrame, cmap) -> np.ndarray:
    """One run's centroid path on ``ax``: line, generation-shaded dots, an
    open circle at the first gen and a star at the last. Returns the gens."""
    gens = sub["outer_gen"].to_numpy(dtype=float)
    span = gens.max() - gens.min()
    frac = (gens - gens.min()) / span if span > 0 else np.ones_like(gens)
    # Light -> dark encodes generation; start above 0.35 so the earliest
    # centroids stay visible on the white background.
    colors = cmap(0.35 + 0.6 * frac)
    x, y = sub["pc1"].to_numpy(), sub["pc2"].to_numpy()
    ax.plot(x, y, color=cmap(0.7), lw=1.1, alpha=0.5, zorder=1)
    ax.scatter(x, y, c=colors, s=22, edgecolors="none", zorder=2)
    ax.scatter(x[0], y[0], marker="o", s=90, facecolors="none",
               edgecolors=cmap(0.95), lw=1.5, zorder=3)
    ax.scatter(x[-1], y[-1], marker="*", s=180, color=cmap(0.95),
               edgecolors="white", lw=0.5, zorder=3)
    return gens


def seed_panels(
    groups: Sequence[RunGroup], seeds: Dict[str, Optional[int]],
) -> List[Tuple[int, List[str]]]:
    """``[(seed, [run labels...]), ...]`` in ascending seed order, runs in
    group order; runs without a seed are left out (with a message)."""
    by_seed: Dict[int, List[str]] = {}
    for g in groups:
        for name in g.names:
            seed = seeds.get(name)
            if seed is None:
                print(f"[pca] {name}: no seed in its config — not in any "
                      f"by-seed panel")
                continue
            by_seed.setdefault(seed, []).append(name)
    return sorted(by_seed.items())


def _draw_by_seed(
    paths: pd.DataFrame, explained: np.ndarray, labels: Sequence[str],
    title: str, groups: Sequence[RunGroup],
    seeds: Dict[str, Optional[int]], ncols: int = 3,
) -> plt.Figure:
    """One panel per seed, each showing only that seed's runs (one per
    condition in the intended use), all in the SAME shared PC basis and
    with common axis limits, so the panels compare directly."""
    panels = seed_panels(groups, seeds)
    grp = _group_of(groups)
    n = max(1, len(panels))
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    flat = axes.ravel()
    for ax, (seed, names) in zip(flat, panels):
        for name in names:
            sub = paths[paths["run"] == name]
            if sub.empty:
                continue
            _draw_run_path(ax, sub, _group_cmap(grp[name].color))
        ax.set_title(f"seed {seed}")
        ax.grid(alpha=0.3)
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    # Common limits over everything drawn (sharex/sharey then propagate).
    pad = 0.05
    x0, x1 = paths["pc1"].min(), paths["pc1"].max()
    y0, y1 = paths["pc2"].min(), paths["pc2"].max()
    dx, dy = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    flat[0].set_xlim(x0 - pad * dx, x1 + pad * dx)
    flat[0].set_ylim(y0 - pad * dy, y1 + pad * dy)
    for ax in axes[-1, :]:
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    handles = [Line2D([], [], color=_group_cmap(g.color)(0.7), marker="o",
                      markersize=6, label=g.label) for g in groups]
    fig.legend(handles=handles, loc="lower center", ncol=len(groups),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{title}\n(○ = first gen, ★ = last gen; darker = later "
                 f"generation; one shared PCA basis)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def _draw_trajectories(
    paths: pd.DataFrame, explained: np.ndarray, labels: Sequence[str],
    title: str, groups: Optional[Sequence[RunGroup]] = None,
    seeds: Optional[Dict[str, Optional[int]]] = None,
) -> plt.Figure:
    """One centroid path per run. Without ``groups``: one sequential
    colormap per run and a legend entry per run. With ``groups``: one
    colour per group (legend per group), every run's end star annotated
    with its seed (or its label when no seed is known)."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    handles = []
    grp = _group_of(groups) if groups else {}
    seeds = seeds or {}
    for k, label in enumerate(labels):
        sub = paths[paths["run"] == label]
        if sub.empty:
            continue
        if groups:
            cmap = _group_cmap(grp[label].color)
        else:
            cmap = plt.get_cmap(_RUN_CMAPS[k % len(_RUN_CMAPS)])
        gens = _draw_run_path(ax, sub, cmap)
        if groups:
            seed = seeds.get(label)
            x, y = sub["pc1"].to_numpy(), sub["pc2"].to_numpy()
            ax.annotate(str(seed) if seed is not None else label,
                        (x[-1], y[-1]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7,
                        color=cmap(0.95), zorder=4)
        else:
            handles.append(Line2D(
                [], [], color=cmap(0.7), marker="o", markersize=6,
                label=f"{label} (gen {int(gens.min())} → {int(gens.max())})",
            ))
    if groups:
        for g in groups:
            cmap = _group_cmap(g.color)
            handles.append(Line2D(
                [], [], color=cmap(0.7), marker="o", markersize=6,
                label=f"{g.label} ({len(g.names)} runs)"))

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    note = "; end stars labelled by seed" if groups else ""
    ax.set_title(f"{title}\n(○ = first gen, ★ = last gen; "
                 f"darker = later generation{note})")
    ax.grid(alpha=0.3)
    ax.legend(handles=handles, loc="best", fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
#  Seed-paired distances between final centroids
# ----------------------------------------------------------------------------

_SAME = "same condition"
_SAME_SEED = "across conditions, same seed"
_DIFF_SEED = "across conditions, different seed"


def final_centroid_distances(
    groups: Sequence[RunGroup], source: str = "population",
) -> pd.DataFrame:
    """Euclidean distance in the full normalised genome space between the
    last-generation centroids of every pair of runs, tagged by relation.
    Rows: ``run_a, run_b, group_a, group_b, seed_a, seed_b, gen_a, gen_b,
    relation, distance``."""
    rows = []
    for g in groups:
        for name, rd in zip(g.names, g.run_dirs):
            df = load_genomes(rd, source)
            genes = _gene_columns(df)
            last = int(df["outer_gen"].max())
            cen = df.loc[df["outer_gen"] == last, genes].to_numpy(float).mean(axis=0)
            rows.append((name, g.label, load_seed(rd), last, cen))
    out = []
    for (na, ga, sa, la, ca), (nb, gb, sb, lb, cb) in itertools.combinations(rows, 2):
        if ga == gb:
            rel = _SAME
        elif sa is not None and sa == sb:
            rel = _SAME_SEED
        else:
            rel = _DIFF_SEED
        out.append(dict(run_a=na, run_b=nb, group_a=ga, group_b=gb,
                        seed_a=sa, seed_b=sb, gen_a=la, gen_b=lb,
                        relation=rel, distance=float(np.linalg.norm(ca - cb))))
    return pd.DataFrame(out)


def summarize_distances(dist: pd.DataFrame) -> pd.DataFrame:
    """``relation, n, mean, min, max`` in a fixed relation order."""
    order = [_SAME, _SAME_SEED, _DIFF_SEED]
    agg = (dist.groupby("relation")["distance"]
           .agg(n="count", mean="mean", min="min", max="max").reset_index())
    agg["relation"] = pd.Categorical(agg["relation"], order, ordered=True)
    return agg.sort_values("relation").reset_index(drop=True)


_SOURCES = (
    # (load_genomes source, filename stem, figure title)
    ("population", "pca_population",
     "Morphology PCA trajectory — per-gen centroid of all genomes"),
    ("front", "pca_pareto_front",
     "Morphology PCA trajectory — per-gen centroid of Pareto front"),
)

_FITS = (
    # (centroid_paths fit_on, filename suffix, title suffix)
    ("rows", "", ""),
    ("centroids", "_centroid_fit", "\n(PCA fit on per-gen centroids)"),
)


def plot_pca_trajectories(
    run_dirs: Sequence[Path | str], labels: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[RunGroup]] = None,
    out_dir: Optional[Path | str] = None,
) -> List[Path]:
    """Render both PCA-trajectory figures and copy them into every run
    (or only into ``out_dir`` when given).

    With ``groups`` (whose ``run_dirs`` must be exactly ``run_dirs`` in
    order) runs are coloured by condition and the seed-paired distance CSVs
    are written next to the figures.

    Returns the list of written paths. The front figure is skipped (with a
    message) for run sets where no front CSV can be produced.
    """
    dirs = [Path(d) for d in run_dirs]
    names = run_labels(dirs, labels)
    out_dirname = _output_dirname(names)
    seeds = {n: load_seed(d) for n, d in zip(names, dirs)}
    if groups:
        # Bind the (possibly auto-derived) labels to the groups in order.
        it = iter(names)
        for g in groups:
            g.names = [next(it) for _ in g.run_dirs]
    targets = ([Path(out_dir)] if out_dir is not None
               else [d / "plots" / out_dirname for d in dirs])

    written: List[Path] = []
    for source, stem, title in _SOURCES:
        try:
            dfs = [load_genomes(d, source) for d in dirs]
        except FileNotFoundError as exc:
            if source == "population":
                raise
            print(f"[pca] Skipping {stem}*.png: {exc}")
            continue
        for fit_on, fname_suffix, title_suffix in _FITS:
            paths, explained = centroid_paths(dfs, names, fit_on=fit_on)
            fig = _draw_trajectories(
                paths, explained, names, title + title_suffix,
                groups=groups, seeds=seeds)
            for target in targets:
                out = target / f"{stem}{fname_suffix}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out, dpi=150)
                written.append(out)
            plt.close(fig)
            if groups and any(v is not None for v in seeds.values()):
                fig = _draw_by_seed(paths, explained, names,
                                    title + title_suffix, groups, seeds)
                for target in targets:
                    out = target / f"{stem}{fname_suffix}_by_seed.png"
                    fig.savefig(out, dpi=150, bbox_inches="tight")
                    written.append(out)
                plt.close(fig)

    if groups:
        dist = final_centroid_distances(groups)
        summ = summarize_distances(dist)
        for target in targets:
            target.mkdir(parents=True, exist_ok=True)
            dist.to_csv(target / "final_centroid_distances.csv", index=False)
            summ.to_csv(target / "distance_summary.csv", index=False)
            written += [target / "final_centroid_distances.csv",
                        target / "distance_summary.csv"]
        print("[pca] Final-centroid distance (15-D genome space) by relation:")
        print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    for path in written:
        print(f"[pca] Wrote {path}")
    return written


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Joint-PCA morphology trajectories across outer-loop runs"
    )
    parser.add_argument("run_dirs", nargs="*",
                        help="Outer-loop run directories to compare "
                             "(or use --group)")
    parser.add_argument("--group", action="append", nargs="+", metavar="TOKEN",
                        help="LABEL COLOR RUN_DIR [RUN_DIR ...]; repeatable. "
                             "Colours runs by condition and writes the "
                             "seed-paired distance CSVs")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="One legend/folder label per run dir, in "
                             "positional or --group order (default: dir "
                             "names, timestamps stripped)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write everything here instead of into every "
                             "run's plots/ folder")
    args = parser.parse_args(argv)
    groups = None
    run_dirs = list(args.run_dirs)
    if args.group:
        if run_dirs:
            parser.error("give run dirs either positionally or via --group")
        groups = []
        for tokens in args.group:
            if len(tokens) < 3:
                parser.error(f"--group needs LABEL COLOR RUN_DIR..., got {tokens}")
            label, color, *dirs = tokens
            groups.append(RunGroup(label, color, [Path(d) for d in dirs]))
            run_dirs += dirs
    if not run_dirs:
        parser.error("no run directories given")
    plot_pca_trajectories(run_dirs, args.labels, groups=groups, out_dir=args.out)


if __name__ == "__main__":
    main()
