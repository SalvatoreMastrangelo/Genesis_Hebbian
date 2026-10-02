"""Distances between the last populations of the sixteen exam runs (thesis 6.2.3).

Reads results/outer_population.csv of every run under logs/remote/outer_nsga/ and writes
images/06_Results_and_Experiments/codesign/pca_distances_16runs.json with every number of
tab:pca-distances and of the paragraph that follows it: the four kinds of pair, the cluster
of seven control runs, the nearest-control distances and the moves of the six seed pairs.
Centroid = mean genome (g0..g14, unit cube) of the 64 bodies of a phase; distances are
Euclidean in the fifteen genes. Run from the repo root: python3 tesis/scripts/pca_distances_16runs.py
"""
import collections, csv, glob, itertools, json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
L = ROOT / "logs/remote/outer_nsga"
OUT = ROOT / "tesis/images/06_Results_and_Experiments/codesign/pca_distances_16runs.json"
RUNS = {
    "cod4_s67": "outer_exam_4_64_64_300_r0", "cod4_s68": "outer_exam_4_64_64_300_r1",
    "cod6_s67": "outer_exam_6_64_64_300_r0", "cod6_s68": "outer_exam_6_64_64_300_r1",
    "cod4_s12345": "outer_exam_4_64_64_300_extra_r0", "cod4_s12346": "outer_exam_4_64_64_300_extra_r1",
    "cod4_s12347": "outer_exam_4_64_64_300_extra_r2", "cod4_s12348": "outer_exam_4_64_64_300_extra_r3",
    "morph_s67": "outer_morphology_only_exam_r0", "morph_s68": "outer_morphology_only_exam_r1",
    "morph_s69": "outer_morphology_only_exam_r2", "morph_s70": "outer_morphology_only_exam_r3",
    "morph_s12345": "outer_morphology_only_exam_extra_r0", "morph_s12346": "outer_morphology_only_exam_extra_r1",
    "morph_s12347": "outer_morphology_only_exam_extra_r2", "morph_s12348": "outer_morphology_only_exam_extra_r3",
}
OUTLIER = "morph_s70"          # the control run that ends outside the cluster of the other seven


def path(run):
    rows = list(csv.DictReader(open(glob.glob(str(L / run / "*/results/outer_population.csv"))[0])))
    by = collections.defaultdict(list)
    for x in rows:
        by[int(x["outer_gen"])].append([float(x[f"g{i}"]) for i in range(15)])
    return [[st.mean(c) for c in zip(*by[g])] for g in sorted(by)]


paths = {k: path(r) for k, r in RUNS.items()}
fin = {k: p[-1] for k, p in paths.items()}
d = lambda a, b: math.dist(fin[a], fin[b])
seed = lambda k: k.split("_s")[1]
cs = [k for k in RUNS if k.startswith("cod")]
ms = [k for k in RUNS if k.startswith("morph")]
m7 = [m for m in ms if m != OUTLIER]


def row(pairs):
    v = [d(a, b) for a, b in pairs]
    return {"pairs": len(v), "mean": st.mean(v), "std": st.stdev(v), "min": min(v), "max": max(v)}


out = {
    "two_morphology_only": row(itertools.combinations(ms, 2)),
    "two_of_the_seven_clustered": row(itertools.combinations(m7, 2)),
    "outlier_to_the_seven": row([(OUTLIER, m) for m in m7]),
    "two_co_design": row(itertools.combinations(cs, 2)),
    "same_seed": row([(c, m) for c in cs for m in ms if seed(c) == seed(m)]),
    "different_seeds": row([(c, m) for c in cs for m in ms if seed(c) != seed(m)]),
    "nearest_fellow_of_each_of_the_seven": {m: min(d(m, o) for o in m7 if o != m) for m in m7},
    "nearest_control_of_each_co_design": {c: min(d(c, o) for o in ms) for c in cs},
    "seed_partner_is_nearest_control": [c for c in cs if seed(min(ms, key=lambda m: d(c, m))) == seed(c)],
    "seed_pairs": {},
}
for s in ("67", "68", "12345", "12346", "12347", "12348"):
    pc, pm = paths[f"cod4_s{s}"], paths[f"morph_s{s}"]
    dc = [a - b for a, b in zip(pc[-1], pc[0])]
    dm = [a - b for a, b in zip(pm[-1], pm[0])]
    n = lambda v: math.sqrt(sum(x * x for x in v))
    out["seed_pairs"][s] = {"final_distance": math.dist(pc[-1], pm[-1]), "move_co_design": n(dc), "move_control": n(dm),
                            "cosine_of_the_moves": sum(a * b for a, b in zip(dc, dm)) / (n(dc) * n(dm))}
OUT.write_text(json.dumps(out, indent=1))
for k in ("two_morphology_only", "two_of_the_seven_clustered", "outlier_to_the_seven", "two_co_design", "same_seed", "different_seeds"):
    r = out[k]
    print(f"{k:28s} {r['pairs']:3d}  {r['mean']:.2f} ± {r['std']:.2f}  {r['min']:.2f} to {r['max']:.2f}")
print("wrote", OUT.relative_to(ROOT))
