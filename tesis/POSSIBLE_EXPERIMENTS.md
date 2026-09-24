# Possible experiments

Experiments and analyses that are NOT part of the thesis yet and could be run before it is finished, each to be added to the thesis if it is run. This is not the Future Work of chapter 7: an item moves there only if the author decides not to run it. Add an item whenever a discussion or a review ends with "this could be checked"; when an item is run, write the result and where it went in the thesis, do not delete it.

Format of an item: what, why (the question it answers, with the review item if there is one), how (tools, data, commands), cost, where it would land in the thesis, status.

---

## 1. Separate the static part of the rules (D) from the plastic part (A, B, C)

- **Added:** 2026-09-21, on the author's request (A1 of `reviews/2026-09-21_review_4.md`, raised again by the auditor and the examiner of `review_5`: "the sharpest hole the method chapter leaves open").
- **What.** Fly the front bodies of a co-design run again under four controllers, on identical forests: the evolved rules (`best`), the frozen generalist (`zero`), the evolved rules with D set to zero (`noD`), and D alone with A, B and C set to zero (`onlyD`).
- **Why.** By `eq:hebbian-trace` the contribution of $D_{ij}$ to a weight is, after about a second, the constant $\eta D_{ij}/\lambda$, the same on every body: through D the search retunes the output layer, which is not plasticity. The control of the thesis (the frozen controller, η = 0) switches the four coefficients off together, so it cannot say how much of the gain of co-design comes from D and how much from the coefficients that depend on the activity. The expected defence question: "what survives with D at zero?"
- **How.** `src/WP2_Outer_Loop/transfer_eval.py` already does the re-flying and accepts a genome file:
  1. take the run's best genome (the one `--genome best` loads; 896 genes in [0, 1], layout `[A | B | C | D]`, 224 genes each, `src/WP2/utils.py::decode_hebbian_genes`);
  2. `noD`: copy it and set genes 672 to 895 to 0.5 (D = 0); `onlyD`: copy it and set genes 0 to 671 to 0.5 (A = B = C = 0); save both as `.npy`;
  3. for each of `best`, `zero`, `noD.npy`, `onlyD.npy`:
     `PYTHONPATH=src python -m WP2_Outer_Loop.transfer_eval <run>/results/pareto_front.csv --config <run>/reproducibility/config.yaml --genome <...> --tag <...> --x-upper 300 --dens-min 1 --dens-max 5.5 --n-forests 480 --forest-seed 0 -o out/`
     (same seed and forest settings, so the four fly identical layouts; check the exact names of the forest options with `--help`);
  4. overlay with `--plot out/transfer_*.csv`.
  Runs locally in the `mygenesis` docker image (see the memory note on the local docker runtime) or on IZAR. A lighter variant without any flight: compare $\eta D/\lambda$ of the best genome with the ΔW histories already stored in `plots/champion_videos/*/trajectory.pkl` (`champion_deltaw_diff.py` reads them).
- **Cost.** A front of about twenty bodies × 480 forests × 4 controllers: minutes to a few tens of minutes on one GPU. Repeating it on the eight co-design runs: an afternoon. The expensive form (co-design runs with D removed from the genome, four days each) is not proposed.
- **Where it would land.** Chapter 6, next to the comparison between co-design and morphology-only fronts (the "Convergence or Plasticity?" subsection of the outline is the natural place); one sentence in 5.2.3, after "Through D the search can retune the output layer itself", pointing to it.
- **Status.** Not run. The author may run it before finishing the thesis.

---

## Candidates raised by the reviews, not discussed with the author yet

One line each, so that they are not lost; none of them is a decision.

- Mean ΔW per body over the exam flights, to show that the plastic weights differ by body ("adapt to the body" against a memory of 0.8 s; A2 of `review_4`). Needs ΔW logging in the exam, or a re-fly with logging.
- Fraction of flights near 90° of roll next to a trunk (the collision test ignores the wing at large roll; declined as a text change, kept as a prepared answer: B8 of `review_5`).
- Number of bodies removed by the 80 m gate at each exam, from `results/outer_population.csv` (no flight needed).
- Magnitude of the run-to-run divergence with identical seeds (4.1.2 asserts it, no number anywhere).
