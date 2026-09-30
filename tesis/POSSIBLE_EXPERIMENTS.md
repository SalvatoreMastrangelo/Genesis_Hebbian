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
- **Status.** RUN on 2026-09-24 on the author's instruction ("do the D ablation too"), in two parts.
  (a) Fixed body (the reference drone, section 6.1.2): `tesis/scripts/d_ablation_fixed_body.py`, output
  `images/06_Results_and_Experiments/inner_loop/d_ablation_fixed_body.json` and `<run>/results/d_ablation.json`
  for the generalist run (`2026-07-08_09-59-42_validation_wspecialis_15_no_bix_15_lstm`) and the specialist
  run (`2026-09-24_16-47-43_rules_on_specialist_r1_s5535`): seven controllers (zero; best rules, best with D = 0,
  best with A = B = C = 0; the CMA-ES mean and its two ablations) on the same 2 048 forests, 4 repeats.
  Result: the gain is NOT decomposable into a static part and a plastic part. Generalist: best +3.7, D alone
  −0.4, A B C alone −11.2 (crashes 53 → 63 %); mean +4.4, D alone +3.9, A B C alone −0.3. Specialist: best +10.0,
  D alone +1.9, A B C alone +2.9 (crashes 27 → 38 %); mean +10.5, D alone 0.0, A B C alone +4.1. On the
  generalist the centre of the search works mostly through D; on the specialist D alone does nothing and the
  gain needs the activity-dependent terms together with D. Written into 6.1.2 (`tab:d-ablation-fixed`).
  (b) Fronts of the eight co-design runs on the exam course (480 forests, seed 0, four controllers on identical
  forests): `tesis/scripts/d_ablation_fronts.py`, output `logs/remote/outer_nsga/d_ablation/<run>/` and
  `images/06_Results_and_Experiments/d_ablation/d_ablation_fronts.json`; launched 2026-09-24 about 22:55;
  ALL 8 RUNS DONE 2026-09-25 16:52 (the 4 `extra` runs overnight, ~1.4 h each while sharing the GPU; the process was
  stopped during the 5th run because the script leaks host RAM per chunk, 12 GB after 26 chunks; the other 4 were run
  after the queue with `src/WP2/experiments/thesis_inner_loop/run_fronts_ablation_rest.sh`, one process per run, ~30 min
  each). RESULT, mean over the front bodies of the change against the frozen generalist (zero), 328 bodies in 8 fronts,
  exam course, 480 identical forests: best rules progress +17.1 m (per run +9.3 to +25.6; 321 of 328 bodies better),
  CoT −0.006 (262 of 328 lower); D alone +13.7 m (+9.3 to +19.2; 324 of 328 better), CoT −0.004 (237 lower);
  A B C alone −11.7 m (−30.6 to +2.9; 69 of 328 better), CoT −0.004 (230 lower). So on the evolved bodies flown by the
  generalist the constant term D carries about 80 % of the progress gain and the activity terms alone are harmful for
  progress while saving energy; the pair adds ~3 m and more energy saving over D alone. Consistent with the generalist
  half of the fixed-body ablation (the search's centre works mostly through D), unlike the specialists. Per-run rows in
  `logs/remote/outer_nsga/d_ablation/<run>/ablation.csv`; figure `images/06_Results_and_Experiments/d_ablation/d_ablation_fronts.pdf`; first chunk of `extra_r0` (8 bodies): progress zero 198 m, best 210 m, noD 199 m,
  onlyD 207 m (D carries most of the progress gain on evolved bodies, unlike on the reference drone); to be written into 6.2 ("Convergence or Plasticity?") when that section is written.

---

## Candidates raised by the reviews, not discussed with the author yet

One line each, so that they are not lost; none of them is a decision.

- Mean ΔW per body over the exam flights, to show that the plastic weights differ by body ("adapt to the body" against a memory of 0.8 s; A2 of `review_4`). Needs ΔW logging in the exam, or a re-fly with logging.
- Fraction of flights near 90° of roll next to a trunk (the collision test ignores the wing at large roll; declined as a text change, kept as a prepared answer: B8 of `review_5`).
- Number of bodies removed by the 80 m gate at each exam, from `results/outer_population.csv` (no flight needed).
- Magnitude of the run-to-run divergence with identical seeds (4.1.2 asserts it, no number anywhere).
