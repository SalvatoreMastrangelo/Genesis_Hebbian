# Handover: thesis section 6.1 (inner-loop experiments), 2026-09-24

**2026-09-29, author: fill 6.1 from the draft, without the D ablation (kept for a separate paper) and keeping the
"rules adapt the controller to a body" reading. Done: the chapter file is the draft minus `tab:d-ablation-fixed` and its
paragraph, with the question and the closing paragraph of 6.1.2 rewritten; details in `THESIS_DECISIONS.md` §5.**

**2026-09-25 17:10, author: "clear all the chapter 6, it will be rewritten based on my preferences in style and
content, some of the found facts are not necessary". The chapter file is back to headings only (git HEAD); the
withdrawn text is `drafts/06_Results_and_Experiments_draft_2026-09-25.tex`. The runs, figures, numbers and
records below remain valid material for the author's rewrite; nothing below is a decision in force for the text.**

Written by the session that started this work, for the model that takes over. Everything below was
verified in this session against the run folders and the code; nothing is assumed. Read
`tesis/THESIS_DECISIONS.md` (conventions, per-chapter decisions, numbers) and `.claude/CONTEXT.md`
before touching the chapter. The author's brief, verbatim:

> fill up the 6.1 section with the 2 inner loop experiments. in particular, for the 6.1.1 i'd like
> to use the experiment in `logs/remote/wp2_evolution/wp2_mutation_4_bix_validation_lstm_15_r0`,
> while for 6.1.2 maybe a new run might be needed, so it's possible to actually run an hebbian
> adaptation on a specialist with the same settings as
> `logs/runs_hebbian/2026-06-10_10-40-16_validation_wspecialist_no_bix_15_lstm`. decide whether the
> new run is useful, and then consider the usability of [that run] itself too.

Later, same day: "i'm positive for any local run today, so feel free to consider more local runs
if necessary. you have 18 hrs of available computing time" (said at about 18:45 local time).

**Clocks:** the docker container runs in UTC, so run-folder stamps (`2026-09-24_16-47-43_...`) and the
`Elapsed`/`ETA` of the run logs are two hours BEHIND the host clock (CEST); `queue.log` is in host time.
All times below are host (local) time unless they are folder names.

Also, same day (about 17:50): "after each finished simulation give me a brief about how did it go". So: after
EVERY run of the queue ends (an `END <name>` line in `logs/thesis_inner_loop/queue.log`), rerun the figure script
and post a short brief to the author (clean exit? held-out gain of the rules over the frozen base in the last 50
generations, the third line, decomposition into speed / cost of transport / progress / crashes, comparison with the
earlier runs). Background waiters for each END line were armed in the first session; if the session is new, arm
them again (`until grep -q "END   <name>" logs/thesis_inner_loop/queue.log; do sleep 120; done`).

## 1. State of the work (update this section as you go)

- [x] Runs inspected, decision taken: the new run IS useful and cheap (3 h locally). Five local
      runs were queued at 18:47 local (section 3). Run 1 confirmed healthy at generation 2; measured pace
      57 s per generation including the baseline, third-line and validation flights → about 3 h 10 per run;
      run 1 ends about 22:00, then 01:10, 04:20, 07:30, 10:40 on 2026-09-25 (16 h in total, inside the 18 h).
- [x] Figure/stat script written and working:
      `tesis/images/06_Results_and_Experiments/make_inner_loop_figures.py` (system python3, pandas +
      matplotlib; run from the repo root). Output in `tesis/images/06_Results_and_Experiments/inner_loop/`
      (4 figures as pdf+png, `inner_loop_stats.json` with every number for the text). It resolves run
      folders by glob, so it picks up the new runs automatically; rerun it after every run finishes.
- [x] (17:20) Chapter 6 opening written: settings paragraph + `tab:results-runs` (decision A5 of `review_5`);
      `% TODO` comments in the .tex say which rows to update when the queued runs finish.
- [x] (17:20) 6.1.1 written with run A + run C numbers (`% PROVENANCE` / `% TODO` comments inline: switch the
      reference-drone paragraph, caption and table to run 2 when it finishes). Bib entries hansen2019pycma,
      fortin2012deap, paszke2019pytorch added. Thesis builds: 0 errors, 0 em-dashes, chapter 6 starts on page 83.
- [x] (22:40) Run 1 finished cleanly at 22:15 (exit 0, 3 h 28). 6.1.2 WRITTEN from it (window gens 150–199): gain
      +9.6 ± 1.2 (+5.3 %) on specialist r1, progress/crashes unchanged, CoT −20 %, speed −4.5 %; both bases end at the
      same operating point (13.3–13.4 m/s, CoT 0.255–0.257). `% TODO` in the .tex: add specialists r0/r2 when they finish. A `% TODO` block under the heading holds the plan
      and the early data (generation 9: specialist + rules 188 vs frozen specialist 182, slower and cheaper).
- [x] (23:00) D ABLATION, part (a) fixed body, done and written into 6.1.2 as `tab:d-ablation-fixed` + one
      paragraph (author's instruction "do the D ablation too"; see POSSIBLE_EXPERIMENTS.md item 1 for numbers).
- [x] (16:55) D ABLATION part (b) COMPLETE on all 8 fronts (numbers in POSSIBLE_EXPERIMENTS.md item 1 b; figure
      `images/06_Results_and_Experiments/d_ablation/d_ablation_fronts.pdf`). Material for 6.2.6, not written. History:
      the 4 `extra` runs were DONE overnight (JSON had 4 entries). The container was STOPPED by the
      assistant at 05:00 during the 5th run: the script leaks ~0.45 GB of host RAM per chunk (12 GB RSS after 26 chunks,
      3.3 GB RAM left, the harness had reaped the wrapper shell for memory pressure) and the queue's run 3 was at risk.
      The remaining 4 fronts are to be run AFTER the queue, one docker process per run, with
      `src/WP2/experiments/thesis_inner_loop/run_fronts_ablation_rest.sh`. The author said at 13:15 "run the rest after
      the queue": a background chain waits for QUEUE DONE (about 14:55) and runs the script (4 runs, ~1 h each, one
      docker process per run; END lines in `d_ablation_fronts.log`). Brief the author when the 4 are done. Earlier plan: ALL 8 RUNS (author, 23:35: "keep everything running, budget is fine"; the
      watcher that would have stopped it after 4 runs was cancelled). About 1.6 h per run while sharing the GPU
      with the queue; everything (queue + ablation) ends around 14:00 to 18:00 on 2026-09-25. Running in docker (container `thesis_d_ablation_fronts`,
      log `logs/thesis_inner_loop/d_ablation_fronts.log`, ~30 min per run, results
      `logs/remote/outer_nsga/d_ablation/<run>/ablation.csv` + `summary.json`, overall
      `tesis/images/06_Results_and_Experiments/d_ablation/d_ablation_fronts.json`). When done: brief to the
      author (per run: progress and CoT of the front bodies under zero / best / noD / onlyD, deltas vs zero);
      it lands in 6.2 "Convergence or Plasticity?" (not written yet); nothing to add to 6.1.
- [x] (03:30) Run 2 (generalist, 200 gens) finished 03:17, exit 0: window 150–199 rules 177.9 ± 1.1 vs frozen 174.5 ± 0.6
      (+3.4 ± 1.2, +2.0 %), specialist r1 182.2, gap 7.7, recovered 0.44; CoT 0.259/0.285, speed 13.42/13.89; winner's
      curse 4.1. 6.1.1's reference-drone paragraphs, the table row and 6.1.2's cross-references now use it (run C and run B
      are the same-seed repeats). FIXED a wrong number: the specialist's speed on the held-out forests is 14.0 m/s, not
      14.4 (that was the search-env value of the D ablation); the text now says "at much the same speed".
- [x] (07:50) Run 3 (specialist r0) finished 07:46, exit 0: **no gain** (+0.5 ± 1.0); frozen r0 already at 13.3 m/s and
      CoT 0.215 (r1: 13.9, 0.320). 6.1.2 MUST BE RESTRUCTURED once r2 is in (about 11:10): "They add the most" holds for
      r1 only; the reading becomes "the rules move a base to a slower, cheaper flight if it is not there already; the
      gain is the distance of the base from that point; r0 is already there and gains nothing" (r0 + rules ends at
      CoT 0.212, so the "same operating point 0.255–0.259" sentence must be softened). Report per specialist.
      Figure: panel (b) currently shows r1 only; decide whether to add r0/r2 curves (three-panel) or keep r1 + text.
- [x] (11:30) Run 4 (specialist r2) finished 11:12, exit 0: +9.9 ± 0.8, like r1. 6.1.2 RESTRUCTURED with the three
      specialists (`tab:inner-loop-bases`); decisions §5/§6 updated.
- [x] (14:55) Run 5 (generalist seed 5536) finished 14:50, exit 0, QUEUE DONE: +3.6 ± 1.0, same signature. 6.1.1's
      replication sentence and `tab:results-runs` (2 generalist runs) updated; no TODO left in the .tex except the
      6.2 rows of the table. SECTION 6.1 IS COMPLETE.
- [ ] (was: Run 5 brief;) add r0/r2 to 6.1.2 (per specialist +
      spread), add seed 5536 to the replication sentence of 6.1.1 and set `tab:results-runs` to 2 generalist runs;
      rerun the figure script; rebuild.
- [ ] Build (`cd tesis && latexmk -pdf -interaction=nonstopmode thesis.tex`), grep the log for
      errors, `grep -c "—"` must be 0, `pdftoppm` the new pages and look at them.
- [x] (17:40) `THESIS_DECISIONS.md` updated (§4 status, §5 entry "Chapter 6, opening and section 6.1", §6 rows, §7
      labels/keys, §8 mass item), `MANIFEST.md` updated, memory note `project_thesis_ch6_inner_loop.md` + MEMORY.md line
      written. When 6.1.2 is written and the final numbers are in, EXTEND the §5 entry and the §6 rows (do not rewrite).
- [ ] Report to the author (section 7 lists what must be said).

The chapter file is `tesis/chapters/06_Results_and_Experiments/06_Results_and_Experiments.tex`
(currently headings only: `\chapter{Experiments And Results}\label{ch:results}`, `\section{Inner
Loop Experiments}`, `\subsection{Recovering Performance on Generalist Controllers}`,
`\subsection{Improving Performance on Specialist Controllers}`, then `\section{Full Co-Design
Experiments}` with five subsections). Keep the author's headings. 6.2 stays empty (not requested).
The thesis-review hook only fires when a whole chapter is finished, so writing 6.1 alone does not
trigger it.

## 2. The runs and what they contain

All CSVs are in `<run>/results/`: `cma_summary.csv` (per generation: best/mean/worst/std fitness of
the 64 sets of rules on the search flights, sigma), `cma_population.csv` (per individual),
`baseline_summary.csv` (the frozen base controller, zero rules, on the same search flights),
`specialist_summary.csv` (a third controller on the search flights, only where enabled),
`validation_summary.csv` (held-out: `best_*` = the generation's best set of rules, `baseline_*` =
frozen base, `specialist_*` = third controller; all three fly the SAME fresh forests every
generation). Metrics: fitness = mean over flights of the reward summed over the flight (the fitness
of ch. 5.3.2), velocity = mean speed, progress (m, from the release, so 130 m = end of the 100 m
course), crash_rate = fraction of flights ended by a collision, a wall/ground exit or the attitude
limit (`evaluate.py` l. 255: pre_collision | pre_wall_crash | pre_angle_limit; the "crash" of the
reward, ch. 5.1.3), cot = cost of transport, v_deviation.

### Run A (6.1.1): `logs/remote/wp2_evolution/wp2_mutation_4_bix_validation_lstm_15_r0/2026-06-13_02-23-30_mutation_4_bix_validation_lstm_15`
- IZAR, 2026-06-13 02:23 to 06-14 05:27 (about 27 h). Config `reproducibility/config.yaml`; the
  YAML it came from is `src/WP2/experiments/batch_8/mutation_4_bix_validation_lstm_15/run.yaml`.
- Checkpoint = the PRODUCTION generalist (md5 53840f3e… identical to `batch_5/.../checkpoint.pt`
  and to `wp1_no_bix_256_65536_lstm_15_r1/.../tb/model_999.pt`); `wp1_config.yaml` identical to
  batch_5's. Plastic layer as in ch. 5 (eta 0.005, decay 0.05, ranges ±1.5, zero init, per-weight).
- 64 bodies drawn at random (`include_standard_mydrone: false`, reference drone excluded), 50
  generations, CMA-ES pop 64, sigma0 0.075, 245 760 flights per generation (64 × 64 × 60), search
  course (100 m, 0 → 5 trees/m), speeds 10–20, noise on, stochastic actions, crn false, forests
  regenerated every generation, seed 67.
- **Bodies refreshed by MUTATION every 4 generations** (`catalog.mutate: true, mutation_std 0.02,
  refresh_urdfs_every 4`): every gene of every body gets Gaussian noise of std 0.02 in the unit
  cube, clipped, CUMULATIVE (a random walk: after 12 refreshes the drift per gene has std about
  0.07). Refresh dirs `urdfs_gen_004 … urdfs_gen_048` (12 refreshes). No selection of bodies. This is
  the inner loop of the pipeline with the outer loop replaced by a random walk of the bodies.
- Zero-rule baseline every generation on the same bodies/forests; **held-out validation every
  generation: the reference drone (4 096 fresh forests of the search course), best set of rules vs
  frozen generalist** (this is exactly the diagnostic described in ch. 5.3.4).
- Numbers (`inner_loop_stats.json` → "A"; window = last 10 generations, 40–49):
  search bodies: gen 0 population mean 126.1, best 137.6, frozen generalist 140.9 (every
  perturbation of the first generation flies worse than zero rules); gens 40–49: best 142.2 ± 0.8,
  mean 137.1 ± 0.6, frozen 138.2 ± 0.5 → best − frozen = +3.9 (+2.9 %), the population mean is
  still 1.1 below the frozen controller. sigma 0.074 → 0.058.
  held-out reference drone: gen 0 best 172.0 vs frozen 175.3 (−3.4); gain by phase (4 gens):
  −2.5, −0.8, −0.8, −1.0, −0.1, +1.1, +0.4, +2.3, +1.1, +2.7, +2.6, +2.4, +2.7; gens 40–49:
  177.1 ± 0.8 vs 174.6 ± 0.7 → **+2.5 ± 0.6 (+1.4 %)**; the frozen controller's held-out fitness
  varies by 0.6 (std over generations, fresh forests each time).
  metrics gens 40–49 (rules vs frozen): progress 117.2 vs 116.8 m; crash 52.0 % vs 53.4 %;
  CoT 0.273 vs 0.285 (−4.3 %); speed 13.58 vs 13.89 m/s (−2.3 %). i.e. slower and cheaper, same
  progress and collisions.
- Sibling runs exist (not used, author chose mutation_4 lstm_15): `wp2_mutation_{2,6,8}_bix_validation_lstm_15_r0`,
  `wp2_random_4_bix_validation_lstm_15_r0` (random resampling instead of mutation), `_lstm_5`, `_mlp`.

### Run B: `logs/runs_hebbian/2026-06-10_10-40-16_validation_wspecialist_no_bix_15_lstm`
- Local, 3 h 03 (10:42 → 13:45). Rules on the PRODUCTION GENERALIST on the reference drone alone:
  `catalog.num_urdfs 1, include_standard_mydrone true` → legacy single-URDF path, the run COPIES the
  on-disk `genesis/assets/urdf/mydrone/[0.7, 3.5, …].urdf` (see mass finding, section 5).
  64 sets of rules × 200 forests = 12 800 flights per generation, 200 generations, seed 5535,
  search course (the `forest:` section wins over the legacy `evaluation.dens_min 4.0`, see
  `evaluate.py::_apply_forest_overrides`: so the course is 100 m, 0 → 5, NOT 4 → 5).
  Held-out validation 2 048 fresh forests per generation: best rules, frozen generalist, and a
  specialist as third line.
- **Its specialist is NOT the thesis's specialist**: `wp1_intermediate_single_65536_32_010_r2`
  (2026-05-18), same architecture and body, but `num_steps_per_env 25` instead of 15 and seed 3.
  It is a stronger specialist (held-out fitness 191.1 against 182.2 for the production one).
  Usable for the generalist curves (200 gens); its reference line should not be plotted as "the
  specialist" (settings differ; decision A5 says a deviation must be stated where plotted).
- Numbers ("G"[…B]): gens 150–199: rules 179.2 vs frozen 174.8 (+4.7, +2.7 %), specialist 191.1;
  gens 90–99 gain +3.5; winner's curse (search best − held-out best) about 4.5; sigma 0.074 → 0.053.

### Run C: `logs/runs_hebbian/2026-07-08_09-59-42_validation_wspecialis_15_no_bix_15_lstm`
- Local, 1 h 45. Same as B but 100 generations and the PRODUCTION specialist r1
  (`wp1_bix_single_65536_lstm_15_r1`, 2026-07-07, same YAML as the generalist except one body) as the
  third line. Same seed 5535 → same CMA-ES draws as B (the flights diverge: GPU nondeterminism).
- Numbers ("G"[…C], window gens 75–99): rules 177.6 ± 0.9 vs frozen 174.6 ± 0.8 → **+3.0 ± 1.2
  (+1.7 %)**; specialist 182.2 ± 0.6 → gap 7.6 → **recovered share 0.39**; gens 90–99 gain +3.0.
  metrics (rules / frozen / specialist): progress 117.1 / 116.8 / 124.2 m; crash 52.7 / 53.2 /
  27.6 %; CoT 0.249 / 0.285 / 0.321; speed 13.4 / 13.9 / 14.4 m/s. → the specialist's advantage is
  fewer crashes and more progress; the rules' gain is energy (−13 % CoT) at a lower speed; the
  specialist spends MORE per metre than the generalist. Say this in 6.1.1: "recovered 39 % of the
  gap" is true in reward, not in kind.
- B and C agree within noise at gens 90–99 (+3.5 vs +3.0): a replication statement.

### The five new local runs (queued 2026-09-24 18:47 local, sequential, one GPU)
Configs in `src/WP2/experiments/thesis_inner_loop/*.yaml` (derived from run B's config; header
comment in each), queue script `run_queue.sh` there, launched as
`run_queue.sh <5 yamls>` in the background of the first session. Logs:
`logs/thesis_inner_loop/<name>.log` and `queue.log` (START/END lines with exit codes). Run folders
`logs/runs_hebbian/2026-09-24_*_<exp_name>`. Docker: `mygenesis:latest`, `--user $(id -u):$(id -g)`,
`-e HOME=/tmp`, `-w /workspace/bind`, `PYTHONPATH=/workspace/bind/src`, container name
`thesis_<name>` (so `docker ps` shows which one runs; `docker stop thesis_<name>` kills it and the
queue moves on). Measured 57 s per generation (the log's `Iter: 0:38` is the population pass only) →
about 3 h 10 per run; all five ≈ 16 h, i.e. done around 10:40 on 2026-09-25 if nothing fails.
1. `rules_on_specialist_r1_s5535` — rules ON the production specialist r1, seed 5535, third line =
   frozen generalist. Started 18:47 local, run dir `2026-09-24_16-47-43_rules_on_specialist_r1_s5535` (UTC stamp).
   Gen 0 sanity: population best 183.9, mean 152.7, crash 48 %; baseline (specialist r1, zero rules)
   182.0, crash 27.7 %, cot 0.321 (matches run C's specialist line, so the body and the checkpoint
   are the same as in C). GPU 2.2 GB, 84 %.
2. `rules_on_generalist_s5535` — run C repeated for 200 generations (third line = specialist r1).
3. `rules_on_specialist_r0_s5536` — rules on specialist r0, seed 5536, third line = generalist.
4. `rules_on_specialist_r2_s5537` — rules on specialist r2, seed 5537, third line = generalist.
5. `rules_on_generalist_s5536` — generalist, second seed, third line = specialist r0.
**Beware the column naming**: in runs 1, 3, 4 the `specialist_*` columns of `validation_summary.csv`
(and `specialist_summary.csv`) hold the FROZEN GENERALIST (the slot is a generic third line); in
runs 2, 5 they hold the frozen specialist. `make_inner_loop_figures.py` already handles this
("third"). If a run fails, the config's paths are relative to the repo root (the docker `-w`).
Rules-on-specialist runs exist nowhere else; do not look for older ones.

## 3. Plan for the text (what was decided, to be written)

Structure of the chapter file:
```
\chapter{Experiments And Results}\label{ch:results}
<opening: what the chapter delivers (6.1 = the inner loop alone on fixed bodies, 6.2 = co-design vs
morphology-only), then the SETTINGS paragraph + table (decision A5): everything is as in chapter 5
unless stated; libraries pycma 4.4.4 (cite hansen pycma, Zenodo 10.5281/zenodo.2559634 — add a bib
entry, key e.g. hansen2019pycma), DEAP 1.4.3 (Fortin et al. 2012, JMLR 13:2171–2175, key
fortin2012deap), PyTorch 2.10, Genesis 0.3.7 (genesis2024 already in the bib); the runs of the
chapter in a table `tab:results-runs` with: base controller, bodies, flights per generation,
generations, held-out measurement, seeds, number of runs. 6.1 rows now; leave a
`% TODO 6.2 rows: 8 co-design runs (4 of 2026-08-10 + 4 of batch_5) and 8 paired morphology-only
runs, same seeds` comment for whoever writes 6.2.>
\section{Inner Loop Experiments}\label{sec:inner-loop-experiments}
  <intro: the question of 6.1 in ch. 3's terms; on a FIXED body nothing varies between flights but
  the forest, the speed and the noise, so this is the case in which Born to Learn (ch. 5.2) predicts
  the least for plasticity; the outer loop is switched off; the specialist is the yardstick of
  "what the body can do with a controller adapted to it".>
\subsection{Recovering Performance on Generalist Controllers}\label{subsec:results-generalist}
  <(1) run A: the inner loop as in the pipeline but the bodies random-walk instead of being selected;
   fig:inner-loop-bodies (a) search bodies: population band, best, frozen; refresh marks every 4
   gens; the frozen line steps at each refresh because the bodies change (ch. 5.3.4 said the fitness
   is not comparable across updates); (b) held-out reference drone. Numbers above. Point: the first
   generation is worse than zero rules everywhere (start from zero, ch. 5.3.2), the search climbs
   above the frozen controller on the bodies it sees (+2.9 %) and the gain transfers to a body it
   never flew (+1.4 %), but the population mean stays below the frozen controller: a random
   perturbation of the rules still costs more than the selected direction gains.
   fig:inner-loop-bodies-metrics: the gain is slower + cheaper flight, progress and crashes flat.
   (2) the reference drone alone, with the specialist: fig:inner-loop-reference (a) [use run 2 when
   it exists, else run C]; recovered share ~0.39 of the gap in reward; the decomposition shows the
   specialist and the rules gain in different currencies (crashes vs energy); the winner's curse
   (search best vs held-out best, ~4.5): why the pipeline's exam flies the top 8 on new forests.
   (3) replication: B and C (and run 5) agree.>
\subsection{Improving Performance on Specialist Controllers}\label{subsec:results-specialist}
  <runs 1, 3, 4 (+ the generalist third line): the same experiment with the specialist as base.
   fig:inner-loop-reference (b) and fig:inner-loop-reference-metrics. Question: do the rules add as
   much to a controller already trained for the body? If yes: the gain on the generalist is a
   retuning any base accepts (D term, energy), not an adaptation to the body; if less: part of the
   generalist's gain is body-specific. Report per specialist (three) and the spread. Chapter 5.1.5
   promised "no outcome"; 5.2.3 called this "the opposite case". Keep the LSTM line: never say the
   LSTM identifies the body.>
\section{Full Co-Design Experiments} … (unchanged, empty)
```
Register and rules (THESIS_DECISIONS §2): British/Oxford spelling, no em-dashes, `\vect{}`/`\mat{}`
for non-scalars (W_0 etc.), `\autoref` for sections/figures/tables, formulas only when they add
insight (none needed here), numbers as in ch. 5 (`3\,840`), "reference drone" in prose (figures may
say "Bixler" only through the plotting label; the new figures say "reference drone"), "the frozen
generalist", "a set of rules", "held out". Figure titles: no parentheses, "progress" not "P".
Captions carry the method details (forest counts, windows). Every figure must be referenced in the
text. Use `\includegraphics[width=\textwidth]{images/06_Results_and_Experiments/inner_loop/<name>.pdf}`.
Do not write the mass of the reference drone anywhere in 6.1 (section 5).

Suggested caption facts: run A validation = 4 096 forests per generation; B/C/new runs = 2 048;
rolling mean over 5 generations drawn on the raw curves; the same forests for all controllers within
a generation; fresh forests every generation.

## 4. Where the numbers come from

`tesis/images/06_Results_and_Experiments/inner_loop/inner_loop_stats.json`, regenerated by the
script. Windows: run A last 10 generations; 200-generation runs last 50; 100-generation run last 25.
Uncertainty printed as the std over generations of the per-generation paired difference
(best − frozen on the same forests). Also use: std over generations of the frozen controller's
held-out fitness (0.6–0.7) as the forest-to-forest noise floor of a 2 048/4 096-flight mean.

## 5. Finding to report to the author (do not fix in the thesis without his word)

**The reference drone that every 6.1 experiment (and the specialists' training, and very likely the
exam-baseline star of the co-design runs) flies is the on-disk URDF
`genesis/assets/urdf/mydrone/[0.7, 3.5, …].urdf`, which weighs 0.747 kg (old servo masses 40/30/20/20 g,
fuselage shell 20 mm, motor 79 g). Chapter 4 says the reference drone weighs 0.605 kg and describes
the CURRENT generator's masses (servos 25/10/10/10 g, motor 41.5 g, shell 15 mm), which is what
`UrdfMaker` produces today (verified in docker: `build_catalog(include_standard_mydrone=True)` →
0.6048 kg) and what every EVOLVED body of the co-design runs was built with.** Evidence: per-link
masses of run B's copied URDF = the git file (0.7473 kg, 26 links); `evolve_cma.py` l. 555 (single
body → `write_single_urdf_catalog(default_mydrone_urdf_path())`, a copy); WP1 `train.py` →
`resolve_or_generate_urdf()` loads the existing file. The memory note `fix_urdf_mass_drift.md`
claims the file was regenerated to 0.605 kg on 2026-05-11: the file on disk (mtime 2026-05-11
15:37, commit 3e41f7f of 05-12) has the OLD masses, so that regeneration did not survive.
Consequences: (a) within 6.1 everything is consistent (generalist, specialists and rules all fly the
same 0.747 kg body); (b) ch. 4.3.4 "it weighs 0.605 kg" is wrong for the body that was flown; (c) in
6.2 the reference drone's star is a body 0.14 kg heavier than the generator would make of the same
genome, which penalizes the reference against the evolved fronts; THESIS_DECISIONS §8 item (f)
already asked to check this. Options for the author: state 0.747 kg and the older mass set for the
reference drone in ch. 4 (and note that evolved bodies use the lighter set), or regenerate and
re-fly. Not decided.

Second point for the author: there are three production specialists (r0, r1, r2, seeds only); the
existing generalist-vs-specialist run (C) used r1 and the new specialist runs use r1, r0, r2. The
thesis should say which specialist is which.

Third: run B's reference specialist is the 25-step one (above).

## 6. Memory / housekeeping to do at the end

- Memory dir `/home/salvatore/.claude/projects/-home-salvatore-Desktop-code-hebbian-Genesis-Hebbian/memory/`:
  a `project_thesis_ch6_inner_loop.md` note (what 6.1 says, which runs, the mass finding) and a
  MEMORY.md line; correct `fix_urdf_mass_drift.md` (the on-disk file is the old 0.747 kg body).
- `THESIS_DECISIONS.md` §4, §5, §6, §7, §8 as listed in section 1.
- `MANIFEST.md`: add `inner_loop/` (source = the run folders above + the script).
- `.claude/CONTEXT.md`: nothing structural changed; optionally mention `src/WP2/experiments/thesis_inner_loop/`.

## 7. What to tell the author in the final message

Decision on the run (useful, cheap, five queued; when they finish), usability of run B (generalist
curves yes; its specialist is the 25-step one; run C has the right specialist; runs 2 and 5 replace
both), the mass finding (section 5), the three specialists, what 6.1 now says in one line each, the
numbers that matter (gain on bodies +2.9 %, held-out +1.4 %; on the reference drone +1.7 %, 39 % of
the specialist gap, in energy not in crashes; the specialist result when available), what was not
done (6.2, chapter 4's mass, no review run).
