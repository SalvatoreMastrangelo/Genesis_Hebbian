"""Figures of thesis section 6.2 (full co-design experiments), in the style of
``make_inner_loop_figures.py`` (section 6.1): serif fonts, one palette, no top or
right spines, raw curves thin under a five-generation moving average where a
curve runs over generations, legends below the axes, bare titles.

Writes into ``tesis/images/06_Results_and_Experiments/codesign/`` (pdf + png):

  codesign_fronts        the 16 cumulative exam fronts, the mean attainment curves, the reference drone
  codesign_indicators    per run: CoT at 130..210 m (top row), progress at CoT 0.15..0.35 (bottom row)
  codesign_hv            cumulative hypervolume over the fraction of the run, and at the end of the run
  swapped_fronts         one co-design front under its own rules, one morphology-only front under transplanted rules
  pca_seed_pairs         per seed: the path of the population centroid of the two runs, one PCA per seed
  champion_deltaw        difference of the final weight changes on the two champion bodies of the first run
  generality_fitness     best set of rules of every generation on the front bodies and on random bodies (fitness)

Run from the repository root with the system python (pandas, matplotlib, numpy):
    python3 tesis/images/06_Results_and_Experiments/make_codesign_figures.py
The data are read from logs/remote/outer_nsga/ (run folders, pareto_stats/, pareto_overlay/).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from WP2_Outer_Loop.pareto_overlay import (  # noqa: E402
    FrontGroup, attainment_grid, cumulative_front, mean_attainment, partial_mean_attainment, resolve_run_dir,
)

L = REPO / "logs/remote/outer_nsga"
OUT = REPO / "tesis/images/06_Results_and_Experiments/codesign"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Runs (thesis names: "first co-design run" = exam_4_r0, "second" = exam_4_r1)
# ----------------------------------------------------------------------------
COD = {  # seed -> co-design run folder (phases of four generations)
    67: "outer_exam_4_64_64_300_r0", 68: "outer_exam_4_64_64_300_r1",
    12345: "outer_exam_4_64_64_300_extra_r0", 12346: "outer_exam_4_64_64_300_extra_r1",
    12347: "outer_exam_4_64_64_300_extra_r2", 12348: "outer_exam_4_64_64_300_extra_r3",
}
COD6 = ["outer_exam_6_64_64_300_r0", "outer_exam_6_64_64_300_r1"]   # phases of six generations, seeds 67 / 68
MORPH = {
    67: "outer_morphology_only_exam_r0", 68: "outer_morphology_only_exam_r1",
    69: "outer_morphology_only_exam_r2", 70: "outer_morphology_only_exam_r3",
    12345: "outer_morphology_only_exam_extra_r0", 12346: "outer_morphology_only_exam_extra_r1",
    12347: "outer_morphology_only_exam_extra_r2", 12348: "outer_morphology_only_exam_extra_r3",
}
COD_ALL = list(COD.values()) + COD6
MORPH_ALL = list(MORPH.values())
FIRST_RUN = "outer_exam_4_64_64_300_r0"       # champions, ΔW, generality sweep
SWAP_COD, SWAP_MORPH = "outer_exam_4_64_64_300_r1", "outer_morphology_only_exam_r0"

# ----------------------------------------------------------------------------
# Style (identical to make_inner_loop_figures.py)
# ----------------------------------------------------------------------------
TREE = "#0f8a5f"     # co-design / with rules
ACCENT = "#D55E00"   # second "with rules" series (random bodies)
INK = "#222222"
MUTED = "#8a8a8a"    # morphology-only / frozen generalist
GRID = "#c9c9c9"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.edgecolor": "#555555",
    "pdf.fonttype": 42,
})
W_IN = 6.0
ROLL = 5


def style_axis(ax):
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def rolling(y, w=ROLL):
    return pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()


def legend_below(fig, handles, labels, ncol, bottom=0.12, y=-0.005):
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y), columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(rect=(0, bottom, 1, 1))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", name)


def p_text(p):
    return "p-value < 0.001" if p < 0.001 else f"p-value {p:.2f}"


def nondominated(points):
    """Front (progress up, CoT down) of a (n, 2) array, sorted by progress."""
    pts = np.asarray(points, float)
    order = np.argsort(pts[:, 0])[::-1]
    best = np.inf; keep = []
    for i in order:
        if pts[i, 1] < best:
            keep.append(i); best = pts[i, 1]
    keep = np.array(keep)
    return pts[keep][np.argsort(pts[keep][:, 0])]


star = json.load(open(L / "pareto_stats/summary.json"))["star"]   # (progress, CoT) of the frozen generalist on the reference drone

# ----------------------------------------------------------------------------
# 1. The sixteen fronts and the mean attainment curves
# ----------------------------------------------------------------------------
cod_fronts = [cumulative_front(resolve_run_dir(L / r)) for r in COD_ALL]
morph_fronts = [cumulative_front(resolve_run_dir(L / r)) for r in MORPH_ALL]
groups = [FrontGroup("co-design", TREE, cod_fronts), FrontGroup("morphology-only", MUTED, morph_fronts)]
grid = attainment_grid(groups, step=0.5)

fig, ax = plt.subplots(1, 1, figsize=(W_IN, 3.4))
for g in groups:
    for f in g.fronts:
        ax.step(f[:, 0], f[:, 1], where="post", color=g.color, lw=0.6, alpha=0.35)
for g in groups:
    solid = mean_attainment(g.fronts, grid, min_runs=4)
    part, _n = partial_mean_attainment(g.fronts, grid)
    ax.plot(grid, part, color=g.color, lw=1.6, ls=(0, (2, 1.5)))
    ax.plot(grid, solid, color=g.color, lw=1.8, label=g.label)
ax.plot([star[0]], [star[1]], marker="*", ms=11, color=INK, mec="white", mew=0.6, ls="none",
        label="reference drone, frozen generalist")
ax.set_xlabel("progress in the exam [m]")
ax.set_ylabel("cost of transport")
ax.set_title("cumulative fronts of the sixteen runs")
style_axis(ax)
H, Lb = ax.get_legend_handles_labels()
H = [Line2D([], [], color=TREE, lw=1.8), Line2D([], [], color=MUTED, lw=1.8), H[2]]
legend_below(fig, H, ["co-design, 8 runs", "morphology-only, 8 runs", Lb[2]], ncol=3, bottom=0.1)
save(fig, "codesign_fronts")

# ----------------------------------------------------------------------------
# 2. Indicators per run
# ----------------------------------------------------------------------------
ind = pd.read_csv(L / "pareto_stats/indicators_per_run.csv")
tests = pd.read_csv(L / "pareto_stats/indicator_tests.csv").set_index("key")
rows = [
    ("cost of transport", [(f"cot_at_{p}", f"at {p} m") for p in (130, 150, 170, 190, 210)]),
    ("progress [m]", [(f"prog_at_cot_{c}", f"at CoT {c:.2f}") for c in (0.15, 0.2, 0.25, 0.3, 0.35)]),
]
fig, axes = plt.subplots(2, 5, figsize=(W_IN, 3.7))
for r, (ylab, keys) in enumerate(rows):
    for c, (key, title) in enumerate(keys):
        ax = axes[r, c]
        for j, (grp, col) in enumerate((("co-design", TREE), ("morphology-only", MUTED))):
            y = ind.loc[ind.group == grp, key].to_numpy(float)
            x = j + (np.random.RandomState(7 + j + 10 * c).uniform(-0.12, 0.12, len(y)))
            ax.plot(x, y, "o", ms=3.2, color=col, alpha=0.9, mec="white", mew=0.3)
            ax.hlines(np.median(y), j - 0.28, j + 0.28, color=col, lw=1.6)
        ax.set_title(f"{title}\n{p_text(float(tests.loc[key, 'p_two_sided']))}", fontsize=7)
        ax.set_xticks([0, 1]); ax.set_xticklabels([])
        ax.set_xlim(-0.6, 1.6)
        ax.tick_params(axis="y", labelsize=6.5)
        if c == 0:
            ax.set_ylabel(ylab)
        style_axis(ax)
H = [Line2D([], [], color=TREE, marker="o", ms=4, ls="none"), Line2D([], [], color=MUTED, marker="o", ms=4, ls="none"),
     Line2D([], [], color=INK, lw=1.6)]
legend_below(fig, H, ["co-design, one point per run", "morphology-only, one point per run", "median of the group"], ncol=3, bottom=0.08)
fig.text(0.005, 0.965, "(a)", fontsize=8); fig.text(0.005, 0.51, "(b)", fontsize=8)
save(fig, "codesign_indicators")

# ----------------------------------------------------------------------------
# 3. Hypervolume: over the run and at the end
# ----------------------------------------------------------------------------
hv = pd.read_csv(L / "pareto_stats/hv_trajectories.csv")
fgrid = np.linspace(0, 1, 101)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W_IN, 2.8), gridspec_kw={"width_ratios": [3, 1.6]})
for grp, col in (("co-design", TREE), ("morphology-only", MUTED)):
    curves = []
    for run, d in hv[hv.group == grp].groupby("run"):
        d = d.sort_values("outer_gen")
        frac = d.outer_gen.to_numpy(float) / d.outer_gen.max()
        ax.plot(frac, d.hv_cum, color=col, lw=0.6, alpha=0.35)
        curves.append(np.interp(fgrid, frac, d.hv_cum.to_numpy(float)))
    curves = np.stack(curves)
    m, s = curves.mean(0), curves.std(0, ddof=1)
    ax.fill_between(fgrid, m - s, m + s, color=col, alpha=0.15, lw=0)
    ax.plot(fgrid, m, color=col, lw=1.6, label=grp)
ax.set_xlabel("fraction of the run")
ax.set_ylabel("hypervolume of the cumulative front")
ax.set_title("(a) over the run")
ax.set_xlim(0, 1)
style_axis(ax)
for j, (grp, col) in enumerate((("co-design", TREE), ("morphology-only", MUTED))):
    y = ind.loc[ind.group == grp, "hv"].to_numpy(float)
    x = j + np.random.RandomState(3 + j).uniform(-0.12, 0.12, len(y))
    ax2.plot(x, y, "o", ms=3.4, color=col, mec="white", mew=0.3)
    ax2.hlines(np.median(y), j - 0.28, j + 0.28, color=col, lw=1.6)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["co-design", "morphology-only"])
ax2.set_xlim(-0.6, 1.6)
ax2.set_title(f"(b) at the end of the run\n{p_text(float(tests.loc['hv', 'p_two_sided']))}")
style_axis(ax2)
H = [Line2D([], [], color=TREE, lw=1.6), Line2D([], [], color=MUTED, lw=1.6)]
legend_below(fig, H, ["co-design, 8 runs", "morphology-only, 8 runs"], ncol=2, bottom=0.13)
save(fig, "codesign_hv")

# ----------------------------------------------------------------------------
# 4. Swapped fronts
# ----------------------------------------------------------------------------
def transfer(run):
    d = resolve_run_dir(L / run) / "comparisons_hebbian"
    g = pd.read_csv(d / "transfer_generalist.csv", comment="#")
    h = pd.read_csv(d / "transfer_hebbian.csv", comment="#")
    g = g[g.is_standard == 0].reset_index(drop=True); h = h[h.is_standard == 0].reset_index(drop=True)
    assert (g.urdf_idx.to_numpy() == h.urdf_idx.to_numpy()).all()
    return g, h

fig, axes = plt.subplots(1, 2, figsize=(W_IN, 2.9), sharey=True)
for ax, run, title in ((axes[0], SWAP_COD, "(a) co-design front, its own rules"),
                       (axes[1], SWAP_MORPH, "(b) morphology-only front, transplanted rules")):
    g, h = transfer(run)
    for a, b in zip(g.itertuples(), h.itertuples()):
        ax.plot([a.progress_m, b.progress_m], [a.cost_of_transport, b.cost_of_transport], color=INK, lw=0.4, ls=":", alpha=0.5)
    for d, col, lab in ((g, MUTED, "frozen generalist"), (h, TREE, "with the rules")):
        ax.errorbar(d.progress_m, d.cost_of_transport, xerr=d.progress_m_se, fmt="o", ms=2.8, color=col,
                    ecolor=col, elinewidth=0.5, capsize=0, mec="white", mew=0.3, alpha=0.95, label=lab)
        fr = nondominated(d[["progress_m", "cost_of_transport"]].to_numpy(float))
        ax.step(fr[:, 0], fr[:, 1], where="post", color=col, lw=1.3)
    ax.set_title(title)
    ax.set_xlabel("progress in the exam [m]")
    style_axis(ax)
axes[0].set_ylabel("cost of transport")
H, Lb = axes[0].get_legend_handles_labels()
legend_below(fig, H, Lb, ncol=2, bottom=0.14)
save(fig, "swapped_fronts")

# ----------------------------------------------------------------------------
# 5. PCA of the population centroids, one PCA per seed
# ----------------------------------------------------------------------------
def centroids(run):
    df = pd.read_csv(resolve_run_dir(L / run) / "results/outer_population.csv")
    gcols = [f"g{i}" for i in range(15)]
    return df.groupby("outer_gen")[gcols].mean().sort_index().to_numpy(float)

greens = LinearSegmentedColormap.from_list("greens", ["#9fd8c0", "#0f8a5f", "#053d2a"])
greys = LinearSegmentedColormap.from_list("greys", ["#cfcfcf", "#8a8a8a", "#2a2a2a"])
seeds = [67, 68, 12345, 12346, 12347, 12348]
fig, axes = plt.subplots(2, 3, figsize=(W_IN, 4.1))
for ax, seed in zip(axes.flat, seeds):
    A, B = centroids(COD[seed]), centroids(MORPH[seed])
    X = np.vstack([A, B]); mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = S**2 / (S**2).sum()
    pa, pb = (A - mu) @ Vt[:2].T, (B - mu) @ Vt[:2].T
    for P, cmap in ((pb, greys), (pa, greens)):
        n = len(P); cols = cmap(np.linspace(0, 1, n))
        ax.plot(P[:, 0], P[:, 1], color=cols[n // 2], lw=0.6, alpha=0.6, zorder=1)
        ax.scatter(P[:, 0], P[:, 1], c=cols, s=7, zorder=2, linewidths=0)
        ax.plot(P[-1, 0], P[-1, 1], marker="*", ms=8, color=cols[-1], mec="white", mew=0.5, ls="none", zorder=3)
    ax.plot(pa[0, 0], pa[0, 1], marker="o", ms=6, mfc="none", mec=INK, mew=0.8, ls="none", zorder=3)
    ax.set_title(f"seed {seed}")
    ax.set_xlabel(f"first component, {100 * var[0]:.0f} %", fontsize=7)
    ax.set_ylabel(f"second component, {100 * var[1]:.0f} %", fontsize=7)
    ax.tick_params(labelsize=6.5)
    style_axis(ax)
H = [Line2D([], [], color=TREE, marker="o", ms=4, lw=0.8), Line2D([], [], color=MUTED, marker="o", ms=4, lw=0.8),
     Line2D([], [], color=INK, marker="o", ms=5, mfc="none", ls="none"), Line2D([], [], color=INK, marker="*", ms=7, ls="none")]
legend_below(fig, H, ["co-design, light to dark over the run", "morphology-only, the same",
                      "first phase", "last phase"], ncol=4, bottom=0.09)
save(fig, "pca_seed_pairs")

# ----------------------------------------------------------------------------
# 6. Difference of the final weight changes on the two champion bodies
# ----------------------------------------------------------------------------
dw = np.load(resolve_run_dir(L / FIRST_RUN) / "plots/champion_videos/deltaw/deltaw_final.npz")
diff = dw["diff"]
names = ["throttle", "sweep, left", "sweep, right", "twist, left", "twist, right", "elevator", "rudder"]
div = LinearSegmentedColormap.from_list("div", [ACCENT, "white", TREE])
vmax = float(np.abs(diff).max())
fig, ax = plt.subplots(1, 1, figsize=(W_IN, 2.3))
im = ax.imshow(diff, cmap=div, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
ax.set_yticks(range(7)); ax.set_yticklabels(names)
ax.set_xticks(range(0, 32, 4)); ax.set_xlabel("unit of the last hidden layer")
ax.set_ylabel("command")
ax.set_title("weight changes at the end of the flight, farthest body minus cheapest body")
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.ax.tick_params(labelsize=6.5, length=2)
cb.outline.set_linewidth(0.4)
fig.tight_layout()
save(fig, "champion_deltaw")

# ----------------------------------------------------------------------------
# 7. Generality: fitness of the best set of rules of every generation
# ----------------------------------------------------------------------------
sm = pd.read_csv(resolve_run_dir(L / FIRST_RUN) / "generality/summary.csv")
zero = sm[sm.kind == "zero"].set_index("body_set")
best = sm[sm.kind == "best"]
fig, ax = plt.subplots(1, 1, figsize=(0.8 * W_IN, 3.0))
for bs, col, lab in (("front", TREE, "rules on the bodies of the front"), ("random", ACCENT, "rules on random bodies")):
    d = best[best.body_set == bs].sort_values("gen")
    ax.plot(d.gen, d.fitness, color=col, lw=0.6, alpha=0.3)
    ax.plot(d.gen, rolling(d.fitness), color=col, lw=1.4, label=lab)
n_gen = int(best.gen.max()) + 1
ax.hlines(zero.loc["front", "fitness"], 0, n_gen - 1, color=MUTED, lw=1.2, label="frozen generalist on the bodies of the front")
ax.hlines(zero.loc["random", "fitness"], 0, n_gen - 1, color=MUTED, lw=1.2, ls="--", label="frozen generalist on random bodies")
ax.set_xlim(0, n_gen - 1)
ax.set_xlabel("generation")
ax.set_ylabel("fitness on the exam course")
ax.set_title("the best set of rules of every generation on two sets of bodies")
style_axis(ax)
H, Lb = ax.get_legend_handles_labels()
legend_below(fig, H, Lb, ncol=2, bottom=0.17)
save(fig, "generality_fitness")
