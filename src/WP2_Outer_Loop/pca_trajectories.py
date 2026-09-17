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

**Per-seed PCA (``--per-seed``, group mode).** No joint basis at all: for
every seed shared by the grouped runs a SEPARATE PCA is fit on only that
seed's runs, giving ``seed_<seed>/pca_population[_centroid_fit].png`` and
``seed_<seed>/pca_pareto_front[_centroid_fit].png`` (one figure per seed,
its own axes and explained variance) plus ``*_3d.png`` twins (PC1–PC3 of
a 3-component fit) and ``*_projections.png`` (PC1–PC2 / PC1–PC3 / PC2–PC3
side by side), contact-sheet grids of those independent panels
(``<stem>[_centroid_fit]_seed_pairs[_3d|_pc1_pc3|_pc2_pc3].png``),
``seed_pair_distances.csv`` (15-D last-gen centroid distance between the
runs of each seed) and ``seed_pair_pca_variance.csv``. Axes of different
seeds are not comparable in this mode; use the shared-basis figures above
for cross-seed comparison.

Usage
-----
    PYTHONPATH=src python -m WP2_Outer_Loop.pca_trajectories \
        <run_dir_1> <run_dir_2> [...] [--labels NAME1 NAME2 ...]

    PYTHONPATH=src python -m WP2_Outer_Loop.pca_trajectories \
        --group co-design "#c0392b" <run> <run> ... \
        --group morphology-only "#1f5fa8" <run> <run> ... \
        [--labels ...] --out logs/remote/outer_nsga/pca_exam_seed_paired

    PYTHONPATH=src python -m WP2_Outer_Loop.pca_trajectories --per-seed \
        --group co-design "#c0392b" <run> ... \
        --group morphology-only "#1f5fa8" <run> ... \
        --out logs/remote/outer_nsga/pca_exam_seed_pairs_independent
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
from matplotlib.ticker import MaxNLocator
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


def fit_pca(
    X: np.ndarray, n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """First ``n_components`` principal components of ``X`` (rows = genomes).

    Returns ``(components, explained)``: an (n_components, n_genes)
    row-vector basis and the explained-variance ratios. Plain PCA on
    mean-centered data — the genes already share the [0, 1] scale, so no
    per-gene standardization (z-scoring would inflate near-constant genes
    into noise directions). Component signs are fixed so each PC's
    largest-magnitude loading is positive, making outputs reproducible
    across BLAS/SVD implementations. Fits nest: the first two rows of a
    3-component fit are exactly the 2-component fit.
    """
    X = np.asarray(X, dtype=float)
    k = int(n_components)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < k or k < 1:
        raise ValueError(f"PCA with {k} components needs a (>=2, >={k}) "
                         f"matrix, got {X.shape}")
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    total = float((s ** 2).sum())
    explained = (s[:k] ** 2 / total) if total > 0 else np.zeros(k)
    components = vt[:k].copy()
    for pc in components:
        if pc[np.argmax(np.abs(pc))] < 0:
            pc *= -1.0
    return components, explained


def centroid_paths(
    dfs: Sequence[pd.DataFrame], labels: Sequence[str],
    fit_on: str = "rows",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Per-generation centroids of every run in one shared PC1–PC2 basis.
    See ``centroid_paths_with_basis``; this drops the components."""
    paths, explained, _ = centroid_paths_with_basis(dfs, labels, fit_on)
    return paths, explained


def centroid_paths_with_basis(
    dfs: Sequence[pd.DataFrame], labels: Sequence[str],
    fit_on: str = "rows", n_components: int = 2,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-generation centroids of every run in one shared PC basis
    (``n_components`` axes, PC1–PC2 by default).

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
    different basis. Returns ``(paths, explained, components)``: a frame
    with columns ``run, outer_gen, pc1, pc2[, pc3, ...]`` — runs in input
    order, generations ascending — the explained-variance ratios of the fit
    matrix, and the (n_components, n_genes) PC basis itself.
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
    components, explained = fit_pca(X, n_components)
    # Projecting the centroids == averaging the projected rows (linearity),
    # so the "rows" fit reproduces the mean of per-genome projections.
    proj = (C - X.mean(axis=0)) @ components.T

    pc_cols = [f"pc{i + 1}" for i in range(components.shape[0])]
    paths = pd.DataFrame(
        [(label, gen, *p) for (label, gen), p in zip(meta, proj)],
        columns=["run", "outer_gen"] + pc_cols,
    )
    return paths, explained, components


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


def _gen_shading(sub: pd.DataFrame, cmap) -> Tuple[np.ndarray, np.ndarray]:
    """``(gens, colors)``: light -> dark encodes generation; starts above
    0.35 so the earliest centroids stay visible on the white background."""
    gens = sub["outer_gen"].to_numpy(dtype=float)
    span = gens.max() - gens.min()
    frac = (gens - gens.min()) / span if span > 0 else np.ones_like(gens)
    return gens, cmap(0.35 + 0.6 * frac)


def _draw_run_path_3d(ax, sub: pd.DataFrame, cmap) -> np.ndarray:
    """``_draw_run_path`` on a 3-D axes, using ``pc1, pc2, pc3``."""
    gens, colors = _gen_shading(sub, cmap)
    x, y, z = (sub[c].to_numpy() for c in ("pc1", "pc2", "pc3"))
    ax.plot(x, y, z, color=cmap(0.7), lw=1.1, alpha=0.5)
    ax.scatter(x, y, z, c=colors, s=22, edgecolors="none", depthshade=False)
    ax.scatter([x[0]], [y[0]], [z[0]], marker="o", s=90, facecolors="none",
               edgecolors=cmap(0.95), lw=1.5, depthshade=False)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], marker="*", s=180,
               color=cmap(0.95), edgecolors="white", lw=0.5, depthshade=False)
    return gens


def _draw_run_path(
    ax, sub: pd.DataFrame, cmap, cols: Tuple[str, str] = ("pc1", "pc2"),
) -> np.ndarray:
    """One run's centroid path on ``ax`` (``cols`` = the two projected
    columns): line, generation-shaded dots, an open circle at the first gen
    and a star at the last. Returns the gens."""
    gens, colors = _gen_shading(sub, cmap)
    x, y = sub[cols[0]].to_numpy(), sub[cols[1]].to_numpy()
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
#  Per-seed PCA: one INDEPENDENT fit per seed (only that seed's runs)
# ----------------------------------------------------------------------------

@dataclass
class SeedPCA:
    """One seed's runs projected into a PCA fit on those runs alone."""
    seed: int
    names: List[str]
    paths: pd.DataFrame
    explained: np.ndarray
    components: np.ndarray


def per_seed_pca(
    dfs: Dict[str, pd.DataFrame], panels: Sequence[Tuple[int, Sequence[str]]],
    fit_on: str = "rows", n_components: int = 2,
) -> List[SeedPCA]:
    """Fit a separate PCA for every ``(seed, [run labels])`` panel on ONLY
    that seed's genomes (``dfs`` maps run label → ``load_genomes`` frame)
    and project that seed's centroid paths into it. Unlike the shared-basis
    figures, the axes of two seeds are NOT comparable; each panel answers
    "do this seed's two runs drift apart?" in its own best 2-D (or, with
    ``n_components=3``, 3-D) view."""
    fits = []
    for seed, names in panels:
        names = list(names)
        paths, explained, components = centroid_paths_with_basis(
            [dfs[n] for n in names], names, fit_on=fit_on,
            n_components=n_components)
        fits.append(SeedPCA(seed, names, paths, explained, components))
    return fits


def _pair_legend_handles(
    groups: Sequence[RunGroup], fits: Sequence[SeedPCA],
) -> List[Line2D]:
    """Group entries (with the generation range the shading spans) plus the
    first/last-generation marker glyphs."""
    handles = []
    for g in groups:
        gens = pd.concat([f.paths[f.paths["run"].isin(g.names)]["outer_gen"]
                          for f in fits])
        span = (f", light to dark = gen {int(gens.min())} to "
                f"{int(gens.max())}") if len(gens) else ""
        handles.append(Line2D([], [], color=_group_cmap(g.color)(0.7),
                              marker="o", markersize=6,
                              label=f"{g.label}{span}"))
    handles.append(Line2D([], [], marker="o", markerfacecolor="none",
                          markeredgecolor="0.3", linestyle="", markersize=8,
                          label="first generation"))
    handles.append(Line2D([], [], marker="*", color="0.3", linestyle="",
                          markersize=11, label="last generation"))
    return handles


def _draw_pair_panel(
    ax, fit: SeedPCA, groups: Sequence[RunGroup], fit_on: str,
    pcs: Tuple[int, int] = (1, 2),
) -> None:
    """One seed's runs on ``ax`` in that seed's own PCA basis, projected on
    the 1-based component pair ``pcs`` (PC1–PC2 by default; PC1–PC3 and
    PC2–PC3 need a 3-component fit); axis labels carry the explained
    variance of the two components shown."""
    cols = tuple(f"pc{i}" for i in pcs)
    missing = [c for c in cols if c not in fit.paths.columns]
    if missing:
        raise ValueError(f"projection {pcs} needs component(s) {missing} — "
                         f"fit with n_components >= {max(pcs)}")
    grp = _group_of(groups)
    for name in fit.names:
        sub = fit.paths[fit.paths["run"] == name]
        if sub.empty:
            continue
        _draw_run_path(ax, sub, _group_cmap(grp[name].color), cols)
    tag = ", centroid fit" if fit_on == "centroids" else ""
    ax.set_xlabel(f"PC{pcs[0]} ({fit.explained[pcs[0] - 1] * 100:.1f}% var{tag})")
    ax.set_ylabel(f"PC{pcs[1]} ({fit.explained[pcs[1] - 1] * 100:.1f}% var{tag})")
    ax.set_title(f"seed {fit.seed}")
    ax.grid(alpha=0.3)


_PROJECTIONS = ((1, 2), (1, 3), (2, 3))


def _draw_seed_pair_projections(
    fit: SeedPCA, groups: Sequence[RunGroup], title: str, fit_on: str,
) -> plt.Figure:
    """One seed, 2×2: the three coordinate-plane projections of its
    3-component fit (PC1–PC2, PC1–PC3, PC2–PC3) and the 3-D view."""
    fig = plt.figure(figsize=(11, 10))
    for i, pcs in enumerate(_PROJECTIONS):
        ax = fig.add_subplot(2, 2, i + 1)
        _draw_pair_panel(ax, fit, groups, fit_on, pcs=pcs)
        ax.set_title("")
    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    _draw_pair_panel_3d(ax3, fit, groups, fit_on)
    ax3.set_title("")
    fig.legend(handles=_pair_legend_handles(groups, [fit]),
               loc="lower center", ncol=len(groups) + 2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{title}, seed {fit.seed}", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.95,
                        wspace=0.25, hspace=0.25)
    return fig


def _draw_seed_pair_figure(
    fit: SeedPCA, groups: Sequence[RunGroup], title: str, fit_on: str,
) -> plt.Figure:
    """Stand-alone figure for one seed (its own PCA basis)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_pair_panel(ax, fit, groups, fit_on)
    ax.set_title(f"{title}, seed {fit.seed}")
    ax.legend(handles=_pair_legend_handles(groups, [fit]), loc="best",
              fontsize=9)
    fig.tight_layout()
    return fig


def _draw_seed_pair_grid(
    fits: Sequence[SeedPCA], groups: Sequence[RunGroup], title: str,
    fit_on: str = "rows", ncols: int = 3, pcs: Tuple[int, int] = (1, 2),
) -> plt.Figure:
    """One panel per seed, EACH in its own PCA basis (independent fits, own
    axis limits and explained variance), so the grid is a contact sheet of
    the per-seed figures, not a joint PCA. ``pcs`` picks the projection."""
    n = max(1, len(fits))
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.3 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for ax, fit in zip(flat, fits):
        _draw_pair_panel(ax, fit, groups, fit_on, pcs=pcs)
    for ax in flat[len(fits):]:
        ax.set_visible(False)
    fig.legend(handles=_pair_legend_handles(groups, fits), loc="lower center",
               ncol=len(groups) + 2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{title}, one PCA per seed", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


_VIEW_3D = dict(elev=22, azim=-55)


def _draw_pair_panel_3d(
    ax, fit: SeedPCA, groups: Sequence[RunGroup], fit_on: str,
) -> None:
    """``_draw_pair_panel`` on a 3-D axes (PC1–PC3 of a 3-component fit)."""
    if "pc3" not in fit.paths.columns:
        raise ValueError("3-D panel needs a 3-component fit (pc3 column)")
    grp = _group_of(groups)
    for name in fit.names:
        sub = fit.paths[fit.paths["run"] == name]
        if sub.empty:
            continue
        _draw_run_path_3d(ax, sub, _group_cmap(grp[name].color))
    tag = ", centroid fit" if fit_on == "centroids" else ""
    ax.set_xlabel(f"PC1 ({fit.explained[0] * 100:.1f}% var{tag})", labelpad=6)
    ax.set_ylabel(f"PC2 ({fit.explained[1] * 100:.1f}% var{tag})", labelpad=6)
    ax.set_zlabel(f"PC3 ({fit.explained[2] * 100:.1f}% var{tag})", labelpad=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(5))
    # The 3-D box leaves headroom under the axes' top edge: pull the title
    # down onto it.
    ax.set_title(f"seed {fit.seed}", y=0.94)
    ax.view_init(**_VIEW_3D)
    ax.set_box_aspect((1, 1, 1), zoom=0.88)


def _draw_seed_pair_figure_3d(
    fit: SeedPCA, groups: Sequence[RunGroup], title: str, fit_on: str,
) -> plt.Figure:
    """Stand-alone 3-D figure for one seed (its own 3-component basis)."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    _draw_pair_panel_3d(ax, fit, groups, fit_on)
    # Figure-level title: the 3-D axes' own title area is where the
    # upper-left legend sits.
    ax.set_title("")
    fig.suptitle(f"{title}, seed {fit.seed}", y=0.985, fontsize=12)
    ax.legend(handles=_pair_legend_handles(groups, [fit]), loc="upper left",
              fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.95)
    return fig


def _draw_seed_pair_grid_3d(
    fits: Sequence[SeedPCA], groups: Sequence[RunGroup], title: str,
    fit_on: str = "rows", ncols: int = 3,
) -> plt.Figure:
    """``_draw_seed_pair_grid`` with a 3-D panel per seed."""
    n = max(1, len(fits))
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(5.4 * ncols, 5.0 * nrows))
    for i, fit in enumerate(fits):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        _draw_pair_panel_3d(ax, fit, groups, fit_on)
    fig.legend(handles=_pair_legend_handles(groups, fits), loc="lower center",
               ncol=len(groups) + 2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{title}, one PCA per seed", fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.93,
                        wspace=0.08, hspace=0.12)
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


def seed_pair_distances(
    groups: Sequence[RunGroup], panels: Sequence[Tuple[int, Sequence[str]]],
    source: str = "population",
) -> pd.DataFrame:
    """``final_centroid_distances`` restricted to run pairs that share a
    seed, one block per panel, with a leading ``seed`` column."""
    dir_of = {n: d for g in groups for n, d in zip(g.names, g.run_dirs)}
    blocks = []
    for seed, names in panels:
        sub = [RunGroup(g.label, g.color,
                        [dir_of[n] for n in g.names if n in names],
                        [n for n in g.names if n in names])
               for g in groups]
        sub = [g for g in sub if g.names]
        dist = final_centroid_distances(sub, source)
        dist = dist[dist["group_a"] != dist["group_b"]].copy()
        dist.insert(0, "seed", seed)
        blocks.append(dist)
    cols = ["seed", "run_a", "run_b", "group_a", "group_b", "seed_a",
            "seed_b", "gen_a", "gen_b", "relation", "distance"]
    return (pd.concat(blocks, ignore_index=True) if blocks
            else pd.DataFrame(columns=cols))


_SOURCES = (
    # (load_genomes source, filename stem, figure title)
    ("population", "pca_population",
     "Morphology PCA trajectory — per-gen centroid of all genomes"),
    ("front", "pca_pareto_front",
     "Morphology PCA trajectory — per-gen centroid of Pareto front"),
)

_PAIR_TITLES = {
    # bare figure titles for the per-seed figures (no asides)
    "population": "Morphology PCA trajectory of all genomes",
    "front": "Morphology PCA trajectory of the Pareto front",
}

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


def plot_pca_seed_pairs(
    run_dirs: Sequence[Path | str], labels: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[RunGroup]] = None,
    out_dir: Optional[Path | str] = None,
) -> List[Path]:
    """Per-seed PCA figures: for every seed shared by the grouped runs, ONE
    PCA fit on only that seed's runs (never a joint basis across seeds).

    Writes, per seed, ``seed_<seed>/{pca_population,pca_pareto_front}
    [_centroid_fit].png`` (PC1–PC2) plus a ``*_3d.png`` twin (PC1–PC3 of
    the same 3-component fit) and ``*_projections.png`` (PC1–PC2, PC1–PC3,
    PC2–PC3 side by side); contact-sheet grids of those independent panels,
    ``<stem>[_centroid_fit]_seed_pairs[_3d|_pc1_pc3|_pc2_pc3].png``; and
    two CSVs:
    ``seed_pair_distances.csv`` (15-D last-gen centroid distance between the
    runs of each seed) and ``seed_pair_pca_variance.csv`` (explained
    variance of PC1–PC3 per seed × source × fit). ``groups`` is required (the seed
    pairing is across conditions) and its ``run_dirs`` must be exactly
    ``run_dirs`` in order. Returns the written paths.
    """
    if not groups:
        raise ValueError("per-seed PCA needs condition groups")
    dirs = [Path(d) for d in run_dirs]
    names = run_labels(dirs, labels)
    out_dirname = _output_dirname(names)
    seeds = {n: load_seed(d) for n, d in zip(names, dirs)}
    it = iter(names)
    for g in groups:
        g.names = [next(it) for _ in g.run_dirs]
    panels = seed_panels(groups, seeds)
    if not panels:
        raise ValueError("no run has a seed in reproducibility/config.yaml")
    targets = ([Path(out_dir)] if out_dir is not None
               else [d / "plots" / out_dirname for d in dirs])

    written: List[Path] = []
    var_rows = []
    for source, stem, _ in _SOURCES:
        title = _PAIR_TITLES[source]
        try:
            dfs = {n: load_genomes(d, source) for n, d in zip(names, dirs)}
        except FileNotFoundError as exc:
            if source == "population":
                raise
            print(f"[pca] Skipping {stem}*.png: {exc}")
            continue
        for fit_on, fname_suffix, _ in _FITS:
            # One 3-component fit per seed; the 2-D figures use PC1–PC2 of
            # it (identical to a 2-component fit, PCA nests).
            fits = per_seed_pca(dfs, panels, fit_on=fit_on, n_components=3)
            for fit in fits:
                var_rows.append(dict(
                    seed=fit.seed, source=source, fit_on=fit_on,
                    runs=" | ".join(fit.names),
                    pc1_var=float(fit.explained[0]),
                    pc2_var=float(fit.explained[1]),
                    pc3_var=float(fit.explained[2])))
                for draw, tag in ((_draw_seed_pair_figure, ""),
                                  (_draw_seed_pair_figure_3d, "_3d"),
                                  (_draw_seed_pair_projections,
                                   "_projections")):
                    fig = draw(fit, groups, title, fit_on)
                    for target in targets:
                        out = (target / f"seed_{fit.seed}"
                               / f"{stem}{fname_suffix}{tag}.png")
                        out.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(out, dpi=150)
                        written.append(out)
                    plt.close(fig)
            grids = [(lambda *a: _draw_seed_pair_grid(*a), ""),
                     (_draw_seed_pair_grid_3d, "_3d")]
            grids += [(lambda *a, p=pcs: _draw_seed_pair_grid(*a, pcs=p),
                       f"_pc{pcs[0]}_pc{pcs[1]}") for pcs in _PROJECTIONS[1:]]
            for draw, tag in grids:
                fig = draw(fits, groups, title, fit_on)
                for target in targets:
                    out = target / f"{stem}{fname_suffix}_seed_pairs{tag}.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(out, dpi=150, bbox_inches="tight")
                    written.append(out)
                plt.close(fig)

    dist = seed_pair_distances(groups, panels)
    var = pd.DataFrame(var_rows)
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        dist.to_csv(target / "seed_pair_distances.csv", index=False)
        var.to_csv(target / "seed_pair_pca_variance.csv", index=False)
        written += [target / "seed_pair_distances.csv",
                    target / "seed_pair_pca_variance.csv"]
    print("[pca] Last-gen centroid distance (15-D genome space) per seed:")
    print(dist[["seed", "run_a", "run_b", "gen_a", "gen_b", "distance"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
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
    parser.add_argument("--per-seed", action="store_true",
                        help="Group mode only: instead of one joint PCA, fit "
                             "a separate PCA per seed on only that seed's "
                             "runs (seed_<seed>/ figures + a per-seed grid + "
                             "seed_pair_distances.csv)")
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
    if args.per_seed:
        if not groups:
            parser.error("--per-seed needs --group (seeds pair runs across "
                         "conditions)")
        plot_pca_seed_pairs(run_dirs, args.labels, groups=groups,
                            out_dir=args.out)
        return
    plot_pca_trajectories(run_dirs, args.labels, groups=groups, out_dir=args.out)


if __name__ == "__main__":
    main()
