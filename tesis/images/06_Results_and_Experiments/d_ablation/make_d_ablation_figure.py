"""Figure of the D ablation on the fronts of the eight co-design runs (POSSIBLE_EXPERIMENTS item 1 b).

Reads logs/remote/outer_nsga/d_ablation/<run>/ablation.csv (one row per body and controller) and draws, for
the best rules, D alone and A B C alone, the change against the frozen generalist of every front body in
progress (left) and cost of transport (right): light dots = bodies, dark markers = the mean of each run.
Run from the repository root:  python3 tesis/images/06_Results_and_Experiments/d_ablation/make_d_ablation_figure.py
"""
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.labelsize": 8,
                     "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#555555", "pdf.fonttype": 42})
VARIANTS = [("best", "best rules", "#0f8a5f"), ("onlyD", "$D$ alone", "#D55E00"), ("noD", "$A$, $B$, $C$ alone", "#4477aa")]
frames = []
for f in sorted(glob.glob(str(REPO / "logs/remote/outer_nsga/d_ablation/*/ablation.csv"))):
    frames.append(pd.read_csv(f))
df = pd.concat(frames)
runs = sorted(df.run.unique())
fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7))
rng = np.random.default_rng(0)
for ax, col, lab in ((axes[0], "progress_m", "change in progress [m]"), (axes[1], "cost_of_transport", "change in cost of transport")):
    for i, (v, vlab, color) in enumerate(VARIANTS):
        for j, run in enumerate(runs):
            d = df[df.run == run]
            z = d[d.variant == "zero"].set_index("urdf_idx")[col]
            s = d[d.variant == v].set_index("urdf_idx")[col].loc[z.index]
            delta = (s - z).to_numpy()
            x = i + (j - (len(runs) - 1) / 2) * 0.09
            ax.scatter(np.full(len(delta), x) + rng.normal(0, 0.012, len(delta)), delta, s=3, color=color, alpha=0.18, lw=0)
            ax.scatter([x], [delta.mean()], s=16, color=color, edgecolor="black", lw=0.4, zorder=3)
    ax.axhline(0, color="#8a8a8a", lw=0.8)
    ax.set_xticks(range(len(VARIANTS))); ax.set_xticklabels([v[1] for v in VARIANTS])
    ax.set_ylabel(lab); ax.grid(True, color="#c9c9c9", lw=0.4, alpha=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_title("(a) progress on the exam course")
axes[1].set_title("(b) cost of transport")
fig.tight_layout(w_pad=1.5)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"d_ablation_fronts.{ext}", dpi=200)
print("bodies:", len(df[df.variant == "zero"]), "runs:", len(runs))
