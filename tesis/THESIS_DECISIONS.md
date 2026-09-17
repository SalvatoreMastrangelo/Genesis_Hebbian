# Thesis writing decisions

Running record of the choices made while writing the thesis in this folder, so that any agent (or the author) picking up a later chapter applies the same conventions and knows what earlier chapters promised, deferred, or deliberately left out. Update this file whenever a new decision is taken. Last update: 2026-09-17.

## 1. Build and check

```bash
cd tesis && latexmk -pdf -interaction=nonstopmode thesis.tex   # biber + pdflatex, all passes
grep -c "^! \|Undefined\|undefined" thesis.log                   # must be 0 (grep only AFTER latexmk has exited)
grep -c "—" chapters/*/*.tex chapters/Bibliografia.bib           # must be 0 everywhere (no em-dashes)
pdftoppm -f N -l N -r 70 -png thesis.pdf /tmp/page && <look at the PNG>   # always look at new pages
```

`thesis.log` line 299 `\end{dedication}` is a pre-existing hyperref duplicate-destination warning, not an error. `\nocite{*}` is commented out: uncited bib entries do not print.

## 2. Global conventions

- **Language and spelling.** English (`babel english`). Oxford style as used in chapter 2: `-ize` / `-ization` (optimization, parametrization, normalization) but otherwise British (behaviour, centre, manoeuvrability, labour).
- **No em-dashes anywhere** in thesis prose, captions or bib. Use commas, colons, a parenthesis, or a new sentence.
- **Register.** Flowing prose paragraphs; `\emph{}` on a term at first use; numbered display equations with `\label{eq:...}` referenced by `\eqref{}`; sections, tables and chapters referenced by `\autoref{}` (English names, "section 2.3", "chapter 5"). Numeric citations `\cite{key}`; several keys in one `\cite{}` when citing a group.
- **Formulas and figures only when they add insight** (author's rule, 2026-09-17). A formula that merely restates prose is left out.
- **Cross-chapter pointers: keep to a minimum** (author's rule). Chapter 3 has none. Subsection 2.5.3 has two (to chapters 3 and 5), approved.
- **Bibliography.** `chapters/Bibliografia.bib`, biblatex + biber, `style=numeric, sorting=nyt`. Keep the existing key style `authorYEARkeyword` (e.g. `gupta2021embodied`). Proceedings versions preferred over preprints; arXiv only when nothing else exists (`@misc` with `howpublished = {\url{...}}`).
- **Glossary.** `chapters/XX_Glossario/XX_Glossario.tex` is a manual `itemize` of acronyms in order of first appearance. Add every new acronym there.
- **Figures.** Go in `images/<chapter folder>/` (folders now match chapter names, see §7). Tikz figures use the preamble's shared styles (see the Hebbian locality figure in chapter 2). Figure titles: write "p-value" not "p", "p-value < 0.001" below one per mille, "progress" not "P", bare figure-level titles with no parenthetical details.
- **Reference drone naming.** Plots call it "Bixler (generalist controller)" (`pareto_fronts.BIXLER_LABEL`); chapter 3 prose calls it "the reference drone" / "the reference body". The title page (author's uncommitted edit) now reads "Hebbian Co-Design of Fixed Wing Drones"; prose so far says "winged drone" and "fixed-wing drone" interchangeably, keep "winged drone" as the neutral phrase outside chapter 4.
- **Author-review protocol.** Questions from the author ("don't you think…?", "what do you think about…?") are answered, not acted on; edits wait for an explicit go-ahead. Structural changes are proposed as a brief first.

## 3. Where each kind of content lives

| Content | Chapter | Rule |
|---|---|---|
| General theory (RL, PPO, LSTM, EAs, CMA-ES, NSGA-II, Hebbian rules, ABCD, EPANNs) | 2, sections 2.1 to 2.4 | Theory only. Never "in this work we…". Light forward pointers allowed. |
| Prior work and the gap | 2.5 | Papers with the design choice that differs from this thesis; positioning kept to one short paragraph (2.5.3). |
| The three problems | 3 | Problems and the requirements they impose. **No solutions, no method names** (no "frozen", "Hebbian rules", "output layer", "CMA-ES", "NSGA-II"). **No platform details**: the drone is "a winged drone", its body "a vector of morphological parameters", its controller "a neural network". |
| Drone, genome, forest, objectives, simulator, quantities | 4 | Everything chapter 3 declined to describe (see §5). |
| Method and its justifications | 5 | Includes the two justifications removed from 2.5.3: last-layer plasticity ↔ Raghu et al. 2020 (ANIL) + Finn et al. 2017; "plasticity only pays when something varies between lifetimes" ↔ Soltoggio et al. 2018, with the body as the varying quantity. |
| Results | 6 | Must deliver what chapter 3 sets up (see §5). Figures already copied to `images/06_Results_and_Experiments/`. |

## 4. Chapter status

| File | Status |
|---|---|
| `01_Introduction` | empty |
| `02_Technical_Background` | complete (2.1 RL, 2.2 LSTM, 2.3 EAs, 2.4 Hebbian, 2.5 Related Work) |
| `03_The_Problem` | complete (three sections + intro) |
| `04_Simulation_Environment` | headings only |
| `05_Method` | headings only: Base Controllers (specialist LSTM, generalist LSTM, training), The Hebbian Controller (rules on output layer, weight decay and reset semantics), Inner loop, Outer loop, Full pipeline |
| `06_Results_and_Experiments` | headings only |
| `07_Conclusions_and_Future_Work` | headings only (Meta-learning, real drones, other systems) |
| Abstract, acknowledgements | empty |

## 5. Decisions and promises per chapter

### Chapter 2, section 2.5 Related Work (written 2026-09-17)

- Order: 2.5.1 Co-Design of Morphology and Control → 2.5.2 Evolved and Learned Plasticity → 2.5.3 Positioning of This Thesis. (Co-design is the problem, plasticity the tool, then the gap.)
- 2.5.1 opens with the **bilevel co-design formula** `eq:codesign-bilevel` (outer max over body m, inner max over controller θ) and uses it to compare how each work pays for the inner level. Papers: Sims 1994, Lipson & Pollack 2000, Cheney et al. 2018 (premature convergence, morphological innovation protection), Eiben & Hart 2020, Gupta et al. 2021 (DERL: PPO from scratch per body, Baldwin effect, >1000 CPUs), Muff et al. 2026 (hexacopters: (μ+λ)-ES on 36 body parameters, PPO from scratch per body, <1000 bodies), Bergonti et al. 2024 (same lab, NSGA-II outer, trajectory optimization inner, no learned controller), Gupta et al. 2022 MetaMorph (morphology-conditioned universal policy).
- 2.5.2 opens with Floreano & Urzelai 2000 and the **"Born to Learn" lesson** (plasticity is only selected when something varies between or within lifetimes). Miconi et al. 2018 gets the second formula `eq:diffplast` (fixed weight + α·Hebbian trace, all trained by gradient), Backpropamine 2019 one clause. Najarro & Risi 2020 (ABCD + η per synapse, ES, random weights every episode, static net wins on the intact body), Pedersen & Risi 2021, Ferigo et al. 2023, van Diggelen et al. 2025 (one shared ABCD set, random init per robot, CMA-ES, MARL baselines beaten).
- 2.5.3 = the gap paragraph (enumerate of the three ways co-design works pay for the inner level; the plasticity works all start from random weights or co-train by gradient) + **one** paragraph naming the thesis at a high level, mapping its plastic part onto `eq:diffplast`. The longer version with method details and the Raghu/Soltoggio justifications was **discarded on the author's decision**; those justifications go to chapter 5.
- Notes the author asked for are in place: Gupta = RL-trained controller per body; Muff = EA owns the morphology; Bergonti = trajectory optimization, no controller; Miconi = differentiable not evolved; van Diggelen = converges from random init, same rule set for the whole swarm.

### Chapter 3, Problems (written 2026-09-17)

- Opening: setting in one paragraph (winged drone, forest, distance and energy), then the **three sections as three steps of one argument** ending in the requirement "a controller that adapts to the body it is given, at a cost small enough to sit inside the body search".
- 3.1 Sequential design: fix m, solve θ*(m) only; the body bounds energy per metre (airframe) and manoeuvrability (control authority); ends "a body that was never tried cannot be found". **Left out on purpose:** the reference drone + specialist controller as the sequential-design baseline. There is a `% Possible addition` comment in the source; the author may draw it from the standard-mydrone results later.
- 3.2 Morphology optimization: fix θ_G, search m; **`eq:ranking-flip`** (two bodies ranked one way under the shared controller, the other under their own); the generalist undervalues bodies at the edges of its training distribution, which is where better bodies are; training a specialist after the search cannot recover a discarded body; ends with the requirement "a score that reflects what each body can do with a controller adapted to it".
- 3.3 Feasibility: **`tab:feasibility`** (measured + hypothetical costs, see §6); noise multiplier (60 forests per body); the works that train per body stay <1000 bodies or need >1000 CPUs; ends with the shape any answer must have (controller paid once; adaptation to the scored body from elsewhere, at rollout cost) and the empirical question.
- No figure in chapter 3 (equation + table carry it). No cross-chapter references. No method names. No platform details.

### What later chapters must pick up

- **Chapter 4** must define: the drone (fixed-wing glider with actuated wing joints; 7 actions: throttle, sweep left/right, twist left/right, elevator, rudder), the 15-gene genome (see §6), the forest task, the two objectives (`progress_m` maximize, `cost_of_transport` minimize), the simulator (Genesis, GPU-batched), the stochasticity (random forests, stochastic policy) and evaluation quantities (60 forests per body, 300 m exam corridor, 80 m admission gate).
- **Chapter 5** must state the specialist vs generalist controllers (chapter 3 relies on "a controller trained on that body alone" vs "a shared generalist"), that the actor never receives the morphology, the last-layer Hebbian rules with the ANIL/Finn justification, the Born-to-Learn justification with the body as the varying quantity, weight reset semantics, CMA-ES inner loop, NSGA-II outer loop with phase-end exam.
- **Chapter 6** must deliver the comparisons chapter 3 sets up: search with a fixed generalist vs search with an adapting controller on the same bodies, forests and budget (the "morphology-only" vs "co-design" run groups); evolved fronts vs the reference drone; optionally the reference drone + specialist baseline.
- **Chapter 7** future work: the specialist baseline if still missing; meta-learning (already a heading).

## 6. Numbers and facts to reuse (with provenance)

| Fact | Value | Source |
|---|---|---|
| Generalist (WP1) training | 1 000 PPO iterations, 65 536 parallel envs, 15 steps/env/iteration (~10⁹ steps), 256-body pool, 2 collection GPUs, **29 h** wall-clock | `src/WP2_Outer_Loop/experiments/batch_5/exam_4_64_64_300_extra/wp1_config.yaml`; checkpoint timestamps of `logs/remote/wp1_training/wp1_256_extra_observations_critic_multi_shard_smaller_head_r0/.../2026-04-23_20-58-28_.../tb/` (model_0 21:25 → model_999 next day 02:28) |
| Specialist training on one body | **≈ 1 h** | author's statement, 2026-09-17 |
| Production co-design run | NSGA-II population 64 bodies, CMA-ES population 64, 60 forests per body → 245 760 env slots/generation, 200 CMA generations, URDF refresh every 4 → 50 outer phases, exam corridor 300 m, admission gate 80 m; **94 h** wall-clock (gen_000 2026-08-26 19:40 → gen_199 2026-08-30 17:15) | `logs/remote/outer_nsga/outer_exam_4_64_64_300_extra_r1/2026-08-26_18-19-52_.../reproducibility/config.yaml` and `generations/` mtimes |
| Hypothetical per-body training | 3 200 bodies × 29 h = 92 800 h; × 1 h = 3 200 h (> 4 months vs 4 days, factor > 30) | arithmetic in `tab:feasibility` |
| Actor architecture (production) | obs → LSTM(128) → MLP [128, 32] (ELU) → Linear(32 → 7) → tanh; 224 last-layer weights → 896 ABCD genes | `wp1_config.yaml` above |
| Morphology genome (15 genes, each in [0, 1]) | wing_span 0.4–0.9 m; wing_aspect_ratio 1.5–5; fuselage_length 0.4–0.9 m; cg_x_ratio 0.30–0.60; attach_x_ratio 0.30–0.60; elevator_span 0.2–0.6 m; elevator_aspect_ratio 1.5–4; rudder_span 0.10–0.40 m; rudder_aspect_ratio 1.5–4; dihedral −4…4°; sweep_multiplier 1.5–3.5; twist_multiplier 1.5–3.5; NACA first digit 0–4, second digit 2–5, last two 10–20 (discrete) | `src/morph_evolution/chromosome_drone.py` |
| Objectives | `progress_m` (maximize), `cost_of_transport` (minimize) | `src/WP2_Outer_Loop/configs/outer_nsga_default.yaml` |
| Literature figures used | DERL: >1 000 CPUs (1 152), 5 M interactions/body, learning time halves within 10 generations. Muff: 36 parameters, μ=λ=24, 40 generations (960 bodies), 2.5·10⁸ steps/body. Bergonti: energy −37…74 %, mission time −22…33 % vs Bixler 3; tens of thousands of trajectory optimizations per run. Najarro: intact body static 1604 vs Hebbian 1051; unseen damage Hebbian 452 vs static 68. van Diggelen: 180 weights, 720 coefficients, CMA-ES 30 × 100, sim-to-real drop −0.7 % | fetched abstracts / full texts 2026-09-17 |

## 7. Files, labels, keys

- **Bib keys added 2026-09-17:** `sims1994evolving`, `lipson2000automatic`, `cheney2018scalable`, `eiben2020evolves`, `gupta2021embodied`, `gupta2022metamorph`, `muff2026hexacopters` (EvoApplications 2026, LNCS 16524; arXiv 2505.14129 in `note`), `bergonti2024codesign` (ICRA 2024, pp. 8679–8685), `miconi2019backpropamine`, `salimans2017evolution`, `pedersen2021evolving`, `ferigo2023evolving` (Ferigo, Iacca, Medvet, Pigozzi; IEEE TCDS 15(3)), `vandiggelen2025emergent` (arXiv 2507.11566, 2025), `finn2017maml`, `raghu2020rapid`. Already present and reused: `najarro2020meta`, `floreano2000evolutionary`, `soltoggio2018born`, `soltoggio2008evolutionary`, `miconi2018differentiable`.
- **Currently uncited (waiting for chapter 5):** `finn2017maml`, `raghu2020rapid`.
- **Labels:** chapters `ch:problems`, `ch:environment`, `ch:method`, `ch:results`; sections `sec:ea`, `sec:hebbian`, `sec:related-work`, `sec:problem-sequential`, `sec:problem-morphology`, `sec:problem-feasibility`; subsections `subsec:ppo`, `subsec:cmaes`, `subsec:nsga2`, `subsec:hebb-postulate`, `subsec:abcd`, `subsec:evolved-plasticity`, `subsec:rw-codesign`, `subsec:rw-plasticity`, `subsec:rw-positioning`; equations `eq:hebb-basic`, `eq:hebb-matrix`, `eq:abcd`, `eq:codesign-bilevel`, `eq:diffplast`, `eq:ranking-flip`; table `tab:feasibility`; figure `fig:hebb-locality`.
- **Glossary entries added:** ES, EPANN.
- **Images.** `images/` now has one folder per chapter (`01_Introduction` … `07_Conclusions_and_Future_Work`; the old-scheme empty folders were removed). `images/06_Results_and_Experiments/` holds 212 result figures (PNG + PDF): `pareto_overlay/`, `pareto_stats/` (+ `no_bixler/`), `pca_exam_seed_paired/`, `pca_exam_seed_pairs/`, `pca_exam_seed_pairs_independent/` (+ `seed_*/`), `champion_renders/<run>/` for every exam-scored run. `MANIFEST.md` there maps each file to its source under `logs/remote/outer_nsga/`. Smoke-test run excluded.

## 8. Open items

- Verify whether the Khepera → larger-robot (Koala) transfer claim in 2.5.2 belongs to Floreano & Urzelai 2000 (Neural Networks) or to the 2001 follow-up; adjust the sentence or the citation.
- Muff et al. is cited as the 2026 proceedings version; switch to the 2025 preprint if preferred.
- DERL text says "thousands of morphologies" on purpose (sources disagree on 4 000 vs 40 000).
- Chapter 3: the reference drone + specialist baseline (possible addition, see comment in `03_The_Problem.tex`).
- Chapter 3 has no figure; a small sketch of the ranking flip could be added if wanted.
- Uncommitted working-tree changes as of 2026-09-17: chapters 02, 03, 04 (label), 05 (label), 06 (label), `Bibliografia.bib`, glossary, `images/`, this file, plus the author's own title-page edit.
