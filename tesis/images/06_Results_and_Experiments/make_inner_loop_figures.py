"""Figures and numbers of thesis section 6.1 (inner-loop experiments).

Reads the CSVs written by ``WP2.evolve_cma`` (``results/cma_summary.csv``,
``baseline_summary.csv``, ``validation_summary.csv``) of the runs listed in RUNS and
writes, into ``tesis/images/06_Results_and_Experiments/inner_loop/``:

  inner_loop_bodies.pdf/png          run A: fitness on the search bodies + held-out reference drone
  inner_loop_bodies_metrics.pdf/png  run A: progress, crash rate, cost of transport, speed (held-out)
  inner_loop_reference.pdf/png       runs G and S: held-out fitness, generalist and specialist, with/without rules
  inner_loop_reference_{generalist,specialist}.pdf/png  the same two panels as separate figures, own scales (the thesis uses these)
  inner_loop_reference_metrics.pdf/png  same four controllers, four metrics
  inner_loop_stats.json              every number quoted in the text of 6.1

Run from the repository root with the system python (pandas + matplotlib):
    python3 tesis/images/06_Results_and_Experiments/make_inner_loop_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tesis/images/06_Results_and_Experiments/inner_loop"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Runs. Paths relative to the repository root.
# ----------------------------------------------------------------------------
RUNS = {
    # 6.1.1: the inner loop on 64 random bodies refreshed by mutation every 4 generations,
    # held-out validation on the reference drone (4 096 forests per generation).
    "A": "logs/remote/wp2_evolution/wp2_mutation_4_bix_validation_lstm_15_r0/"
         "2026-06-13_02-23-30_mutation_4_bix_validation_lstm_15",
    # rules on the GENERALIST, reference drone only; the run's "specialist" columns hold the
    # frozen specialist. Several runs: the first is plotted, all enter the statistics.
    "G": [
        "logs/runs_hebbian/2026-09-24_*_rules_on_generalist_s5535",
        "logs/runs_hebbian/2026-09-2*_rules_on_generalist_s5536",
        "logs/runs_hebbian/2026-07-08_09-59-42_validation_wspecialis_15_no_bix_15_lstm",   # 100 gens
        "logs/runs_hebbian/2026-06-10_10-40-16_validation_wspecialist_no_bix_15_lstm",     # 25-step specialist as reference
    ],
    # rules on the SPECIALIST (r1, r0, r2); the run's "specialist" columns hold the frozen generalist.
    "S": [
        "logs/runs_hebbian/2026-09-24_*_rules_on_specialist_r1_s5535",
        "logs/runs_hebbian/2026-09-2*_rules_on_specialist_r0_s5536",
        "logs/runs_hebbian/2026-09-2*_rules_on_specialist_r2_s5537",
    ],
}

TREE = "#0f8a5f"     # generalist with rules
ACCENT = "#D55E00"   # specialist with rules
INK = "#222222"
MUTED = "#8a8a8a"    # frozen generalist
GRID = "#c9c9c9"
REFRESH = "#b0b0b0"

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
ROLL = 5  # generations of the rolling mean drawn on top of the raw curves


def resolve(pattern):
    hits = sorted(REPO.glob(pattern))
    return hits[-1] if hits else None


def load(run_dir):
    r = Path(run_dir)
    d = {}
    for name in ("cma_summary", "baseline_summary", "specialist_summary", "validation_summary"):
        p = r / "results" / f"{name}.csv"
        if p.is_file():
            df = pd.read_csv(p)
            if len(df):
                d[name] = df
    d["name"] = r.name
    return d


def rolling(y, w=ROLL):
    return pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()


def style_axis(ax):
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_pair(ax, gens, y_rules, y_frozen, c_rules, c_frozen, lab_rules, lab_frozen, ls_frozen="-"):
    ax.plot(gens, y_frozen, color=c_frozen, lw=0.6, alpha=0.3)
    ax.plot(gens, y_rules, color=c_rules, lw=0.6, alpha=0.3)
    ax.plot(gens, rolling(y_frozen), color=c_frozen, lw=1.4, ls=ls_frozen, label=lab_frozen)
    ax.plot(gens, rolling(y_rules), color=c_rules, lw=1.4, label=lab_rules)


def win(df, col, lo, hi):
    sel = df[(df.generation >= lo) & (df.generation <= hi)][col]
    return float(sel.mean()), float(sel.std(ddof=1)) if len(sel) > 1 else float("nan"), int(len(sel))


METRICS = [
    ("progress", "progress [m]"),
    ("crash_rate", "flights ended by a crash"),
    ("cot", "cost of transport"),
    ("velocity", "mean speed [m/s]"),
]

stats = {}

# ----------------------------------------------------------------------------
# Figure 1 + 2: run A
# ----------------------------------------------------------------------------
A = load(REPO / RUNS["A"])
s, b, v = A["cma_summary"], A["baseline_summary"], A["validation_summary"]
gens = s.generation.to_numpy()
n_gen = int(gens.max()) + 1
refresh_every = 4

fig, axes = plt.subplots(1, 2, figsize=(W_IN, 2.5))
ax = axes[0]
for g in range(refresh_every, n_gen, refresh_every):
    ax.axvline(g, color=REFRESH, lw=0.5, ls=":", zorder=0)
pop = pd.read_csv(REPO / RUNS["A"] / "results" / "cma_population.csv")
q = pop.groupby("generation").fitness.quantile([0.25, 0.5, 0.75]).unstack()
ax.fill_between(q.index, q[0.25], q[0.75], color=TREE, alpha=0.15, lw=0, label="population, first to third quartile")
ax.plot(q.index, q[0.5], color=TREE, lw=0.9, ls="--", label="population, median")
ax.plot(gens, s.best_fitness, color=TREE, lw=1.4, label="best set of rules")
ax.plot(b.generation, b.fitness, color=MUTED, lw=1.4, label="frozen generalist")
ax.set_xlabel("generation")
ax.set_ylabel("fitness on the search bodies")
ax.set_title("(a) the 64 bodies of the search")
ax.set_xlim(0, n_gen - 1)
ax.legend(loc="lower right", frameon=False)
style_axis(ax)

ax = axes[1]
for g in range(refresh_every, n_gen, refresh_every):
    ax.axvline(g, color=REFRESH, lw=0.5, ls=":", zorder=0)
draw_pair(ax, v.generation, v.best_fitness, v.baseline_fitness, TREE, MUTED,
          "best set of rules", "frozen generalist")
ax.set_xlabel("generation")
ax.set_ylabel("fitness on the reference drone")
ax.set_title("(b) the reference drone, held out")
ax.set_xlim(0, n_gen - 1)
ax.legend(loc="lower right", frameon=False)
style_axis(ax)
fig.tight_layout(w_pad=1.5)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"inner_loop_bodies.{ext}", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(W_IN, 3.9))
for ax, (col, lab) in zip(axes.flat, METRICS):
    for g in range(refresh_every, n_gen, refresh_every):
        ax.axvline(g, color=REFRESH, lw=0.5, ls=":", zorder=0)
    draw_pair(ax, v.generation, v[f"best_{col}"], v[f"baseline_{col}"], TREE, MUTED,
              "best set of rules", "frozen generalist")
    ax.set_ylabel(lab)
    ax.set_xlim(0, n_gen - 1)
    style_axis(ax)
for ax in axes[1]:
    ax.set_xlabel("generation")
axes[0, 0].legend(loc="lower right", frameon=False)
fig.tight_layout(w_pad=1.5, h_pad=1.0)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"inner_loop_bodies_metrics.{ext}", dpi=200)
plt.close(fig)

last = 10
lo, hi = n_gen - last, n_gen - 1
sa = {"run": A["name"], "generations": n_gen, "window": [lo, hi]}
sa["search_best_first"] = float(s.best_fitness.iloc[0]); sa["search_mean_first"] = float(s.mean_fitness.iloc[0])
sa["search_baseline_first"] = float(b.fitness.iloc[0])
sa["search_best_last"] = win(s, "best_fitness", lo, hi); sa["search_mean_last"] = win(s, "mean_fitness", lo, hi)
sa["search_baseline_last"] = win(b, "fitness", lo, hi)
sa["search_sigma_first_last"] = [float(s.sigma.iloc[0]), float(s.sigma.iloc[-1])]
sa["val_best_first"] = float(v.best_fitness.iloc[0]); sa["val_baseline_first"] = float(v.baseline_fitness.iloc[0])
sa["val_best_last"] = win(v, "best_fitness", lo, hi); sa["val_baseline_last"] = win(v, "baseline_fitness", lo, hi)
d = (v.best_fitness - v.baseline_fitness)
sa["val_gain_last"] = [float(d[v.generation >= lo].mean()), float(d[v.generation >= lo].std(ddof=1))]
sa["val_gain_first_phase"] = float(d[v.generation < 4].mean())
sa["val_baseline_std_over_gens"] = float(v.baseline_fitness.std(ddof=1))
sa["val_gain_by_phase"] = [float(d[(v.generation >= p) & (v.generation < p + 4)].mean()) for p in range(0, n_gen, 4)]
for col, _ in METRICS:
    sa[f"val_{col}_last"] = [win(v, f"best_{col}", lo, hi)[0], win(v, f"baseline_{col}", lo, hi)[0]]
stats["A"] = sa

# ----------------------------------------------------------------------------
# Figure 3 + 4: reference drone, generalist and specialist with and without rules
# ----------------------------------------------------------------------------
G_runs = [load(p) for p in (resolve(x) for x in RUNS["G"]) if p is not None]
S_runs = [load(p) for p in (resolve(x) for x in RUNS["S"]) if p is not None]
G_runs = [r for r in G_runs if "validation_summary" in r]
S_runs = [r for r in S_runs if "validation_summary" in r]
print("G runs:", [r["name"] for r in G_runs])
print("S runs:", [r["name"] for r in S_runs])

if G_runs and S_runs:
    G, S = G_runs[0], S_runs[0]
    vg, vs = G["validation_summary"], S["validation_summary"]
    n_max = max(int(vg.generation.max()), int(vs.generation.max())) + 1

    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 2.8), sharey=True)
    ax = axes[0]
    draw_pair(ax, vg.generation, vg.best_fitness, vg.baseline_fitness, TREE, MUTED,
              "generalist with rules", "frozen generalist")
    if "specialist_fitness" in vg:
        ax.plot(vg.generation, rolling(vg.specialist_fitness), color=INK, lw=1.0, ls="--", label="frozen specialist")
    ax.set_title("(a) rules evolved on the generalist")
    ax.set_ylabel("fitness on the reference drone")
    ax = axes[1]
    draw_pair(ax, vs.generation, vs.best_fitness, vs.baseline_fitness, ACCENT, INK,
              "specialist with rules", "frozen specialist", ls_frozen="--")
    if "specialist_fitness" in vs:
        ax.plot(vs.generation, rolling(vs.specialist_fitness), color=MUTED, lw=1.0, label="frozen generalist")
    ax.set_title("(b) rules evolved on the specialist")
    for ax in axes:
        ax.set_xlabel("generation")
        ax.set_xlim(0, n_max - 1)
        style_axis(ax)
    handles = [h for ax in axes for h in ax.get_legend_handles_labels()[0]]
    labels = [l for ax in axes for l in ax.get_legend_handles_labels()[1]]
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); H.append(h); L.append(l)
    fig.legend(H, L, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005), columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.09, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"inner_loop_reference.{ext}", dpi=200)
    plt.close(fig)

    # The same two panels as separate figures (thesis 2026-09-29: panel (a) belongs to 6.1.1,
    # panel (b) to 6.1.2). Each has its own fitness range (author, 2026-09-29: the generalist
    # figure is larger and its axis stops at 185, since the specialist's rules are not in it).
    def own_ylim(v, top=None):
        cols = [v.best_fitness, v.baseline_fitness] + [v.specialist_fitness] * ("specialist_fitness" in v)
        allv = np.concatenate(cols)
        pad = 0.04 * (allv.max() - allv.min())
        return (float(allv.min() - pad), float(allv.max() + pad) if top is None else float(top))
    for tag, v, c_rules, c_frozen, lab_rules, lab_frozen, ls_frozen, third_c, third_ls, third_lab, title, size, ylim in (
        ("generalist", vg, TREE, MUTED, "generalist with rules", "frozen generalist", "-", INK, "--", "frozen specialist",
         "rules evolved on the generalist", (0.8 * W_IN, 3.0), own_ylim(vg, top=185)),
        ("specialist", vs, ACCENT, INK, "specialist with rules", "frozen specialist", "--", MUTED, "-", "frozen generalist",
         "rules evolved on the specialist", (0.62 * W_IN, 2.6), own_ylim(vs)),
    ):
        fig, ax = plt.subplots(1, 1, figsize=size)
        draw_pair(ax, v.generation, v.best_fitness, v.baseline_fitness, c_rules, c_frozen, lab_rules, lab_frozen,
                  ls_frozen=ls_frozen)
        if "specialist_fitness" in v:
            ax.plot(v.generation, rolling(v.specialist_fitness), color=third_c, lw=1.0, ls=third_ls, label=third_lab)
        ax.set_title(title)
        ax.set_ylabel("fitness on the reference drone")
        ax.set_xlabel("generation")
        ax.set_xlim(0, n_max - 1)
        ax.set_ylim(*ylim)
        style_axis(ax)
        H, L = ax.get_legend_handles_labels()
        fig.legend(H, L, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005),
                   columnspacing=1.2, handlelength=1.8)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"inner_loop_reference_{tag}.{ext}", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(W_IN, 4.1))
    for ax, (col, lab) in zip(axes.flat, METRICS):
        ax.plot(vg.generation, rolling(vg[f"baseline_{col}"]), color=MUTED, lw=1.2, label="frozen generalist")
        ax.plot(vg.generation, rolling(vg[f"best_{col}"]), color=TREE, lw=1.2, label="generalist with rules")
        ax.plot(vs.generation, rolling(vs[f"baseline_{col}"]), color=INK, lw=1.2, ls="--", label="frozen specialist")
        ax.plot(vs.generation, rolling(vs[f"best_{col}"]), color=ACCENT, lw=1.2, label="specialist with rules")
        ax.set_ylabel(lab)
        ax.set_xlim(0, n_max - 1)
        style_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel("generation")
    H, L = axes[0, 0].get_legend_handles_labels()
    fig.legend(H, L, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.5, h_pad=1.0, rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"inner_loop_reference_metrics.{ext}", dpi=200)
    plt.close(fig)


def ref_stats(run, kind):
    v = run["validation_summary"]; s = run["cma_summary"]; b = run["baseline_summary"]
    n = int(v.generation.max()) + 1
    last = 50 if n >= 150 else 25
    lo, hi = n - last, n - 1
    r = {"run": run["name"], "generations": n, "window": [lo, hi], "kind": kind}
    r["val_best_first"] = float(v.best_fitness.iloc[0])
    r["val_base_first"] = float(v.baseline_fitness.iloc[0])
    r["val_best_last"] = win(v, "best_fitness", lo, hi)
    r["val_base_last"] = win(v, "baseline_fitness", lo, hi)
    d = v.best_fitness - v.baseline_fitness
    r["val_gain_last"] = [float(d[v.generation >= lo].mean()), float(d[v.generation >= lo].std(ddof=1))]
    r["val_gain_rel_last"] = r["val_gain_last"][0] / r["val_base_last"][0]
    r["val_gain_gen_0_9"] = float(d[v.generation < 10].mean())
    r["val_gain_gen_40_49"] = float(d[(v.generation >= 40) & (v.generation < 50)].mean())
    r["val_gain_gen_90_99"] = float(d[(v.generation >= 90) & (v.generation < 100)].mean())
    r["val_base_std_over_gens"] = float(v.baseline_fitness.std(ddof=1))
    r["search_best_last"] = win(s, "best_fitness", lo, hi)
    r["search_mean_last"] = win(s, "mean_fitness", lo, hi)
    r["search_base_last"] = win(b, "fitness", lo, hi)
    r["winners_curse_last"] = r["search_best_last"][0] - r["val_best_last"][0]
    r["sigma_first_last"] = [float(s.sigma.iloc[0]), float(s.sigma.iloc[-1])]
    if "specialist_fitness" in v:
        r["val_third_last"] = win(v, "specialist_fitness", lo, hi)
        gap = v.specialist_fitness - v.baseline_fitness
        r["val_gap_third_minus_base_last"] = float(gap[v.generation >= lo].mean())
        r["recovered_share_last"] = float((d[v.generation >= lo] / gap[v.generation >= lo]).mean())
        r["recovered_share_from_means"] = r["val_gain_last"][0] / r["val_gap_third_minus_base_last"]
    for col, _ in METRICS:
        r[f"val_{col}_last"] = {
            "rules": win(v, f"best_{col}", lo, hi)[0],
            "frozen": win(v, f"baseline_{col}", lo, hi)[0],
            "third": win(v, f"specialist_{col}", lo, hi)[0] if f"specialist_{col}" in v else None,
        }
    return r


stats["G"] = [ref_stats(r, "rules on the generalist; third = frozen specialist") for r in G_runs]
stats["S"] = [ref_stats(r, "rules on the specialist; third = frozen generalist") for r in S_runs]

with open(OUT / "inner_loop_stats.json", "w") as f:
    json.dump(stats, f, indent=1)
print(json.dumps(stats, indent=1))
