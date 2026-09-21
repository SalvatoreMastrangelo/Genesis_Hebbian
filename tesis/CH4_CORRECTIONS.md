# Chapter 4: provisional corrections

Drawn up 2026-09-20 from the external review of chapter 4, after checking every point against the chapter source, the solver code and the production configs. **Nothing here has been applied.** Line numbers refer to `chapters/04_Simulation_Environment/04_Simulation_Environment.tex` as of this date. Draft wording follows the thesis conventions (no em-dashes, Oxford spelling) and is a proposal, to be edited freely.

Legend: **[fact]** the text is wrong or inconsistent; **[gap]** something promised or needed is missing; **[claim]** a statement stronger than its support; **[opt]** optional; **[decide]** needs a decision of the author before any wording.

---

## A. Decisions to take first

### A1. Force rotation, eq. (4.12) `eq:aero-force`, l. 226-230 [decide]
- With the angles of `eq:aero-angles` the relative wind is `(cos β cos α, sin β, cos β sin α)`, which is `R_y(−α) R_z(β) x̂`. The text has `R_z(β) R_y(α)`: opposite sign of α, opposite order. The reviewer derived this from the text alone.
- The equation is a faithful transcription of `genesis/engine/solvers/base_aero_solver.py:432` (`_rot_yz`, rows `[cb·ca, −sb, cb·sa] / [sb·ca, cb, sb·sa] / [−sa, 0, ca]`). It is open item (1) of `THESIS_DECISIONS.md` §8, still in the code today. There is no convention to declare that makes it consistent.
- To do: talk to Andrea. Then one of:
  - (a) he confirms: keep the equation literal and add to "Limits of the model" (l. 264) something like: *"The rotation \eqref{eq:aero-force} is reported as it is implemented. With the angles of \eqref{eq:aero-angles} it does not align the drag exactly with the relative wind, and the resultant is tilted aft by an angle that grows with the incidence, so the model overestimates drag. Every body and every method of this thesis flew in the same model, so the comparisons are not affected, while the absolute values of drag and of cost of transport are pessimistic."*
  - (b) he shows a frame convention that makes it right: state that convention next to `eq:aero-angles`.
- Do B19 (trim table) before or together with this: it tells how large the effect is in absolute terms.

### A2. "which section it uses" in the ranking sentence, l. 264 [decide]
- Open item (a) of `THESIS_DECISIONS.md` §8: airfoil genes are aerodynamically inert after the first body refresh. If that holds for the production runs, the sentence cannot list the section among the things the model ranks. Minimum edit: delete ", which section it uses". Full treatment belongs to the decision on that open item.

---

## B. Corrections

### B1. Definition of progress, l. 301 [fact] (found while checking, not in the review)
- Text: "The coordinate $x$ at which the flight ends is its progress".
- Code: `src/WP2/evaluate.py:829`, `dx = base_pos.x − x0`, with x0 the release point (−30 m). Every progress value in the results (80 m gate, CoT at 190 m, blind ceiling 60 to 72 m = 30 + 26…32…) is a distance from the release point, 30 m more than the coordinate.
- Proposed: *"The distance covered along the corridor, from the point of release to the point where the flight ends, is its \emph{progress}, the first of the two quantities […]. A flight that ends at the first tree therefore has a progress of 30~m."*
- Follow-ups: l. 324 and l. 353 quote blind-flight means (11 m, 26 m) measured from the first tree; add "measured from the first tree" so they are not read as progress. `eq:drone-cot` (l. 457-460) stays correct ("Δx the progress of the flight") and the reviewer's "CoT diverges for Δx ≤ 0" no longer applies (Δx > 0 from release; the code clamps at 1e-2 m). Check that the axes of chapter 6 use the same definition.

### B2. Servo torque limit and gains, l. 76 [fact]
- "a torque limit of 1.5 N m for the wing joints" is true only for the reference gear ratios (0.75 × 2 and 0.6 × 2.5, l. 446).
- Proposed: *"and a torque limit of 1~N\,m for the tail joints and, on the reference drone, 1.5~N\,m for the wing joints, where it scales with the gear ratio of the joint (\autoref{subsec:drone-shape}). The gains are defined at the joint and are the same for every joint and every body."*
- Source: `actuators.csv` (kp 8, kv 2 for both servo types), `env.py:826` sets them per dof with no scaling by k.

### B3. b = 1.4 m is not the wingspan, l. 294, 324, 360, 365 (caption), 378, 491 (caption) [fact] (APPLIED 2026-09-21)
- The collision test uses the sum of the two panel spans, 1.4 m for the reference drone and 2 b_p for every body (`drone_model.py::_wingspan_from_surfaces`, `max(left) + max(right)`, read by `env.py:776` as `aero_solver.tip_to_tip`; `_resolve_wingspan_from_aero_config` at `env.py:64` is only the fallback and gives the same sum); the wing is 1.5 m tip to tip (l. 536).
- Applied as drafted: l. 294 *"$b$ the combined span of the two wing panels (1.4~m for the reference drone, whose wing measures 1.5~m from tip to tip once the fuselage between the panels is counted)"*; after "while it does so" (l. 299) *"The test is an approximation: it ignores the width of the fuselage, the length of the body and the sweep of the panels."*; l. 324 "with $b = 1.4$~m for the reference drone"; l. 360, 365, 378 "narrower than the width $b$ of the reference drone" (caption: "the width $b = 1.4$~m"). Numbers unchanged (11 m, 26 m, 64 % → 39 %): they were computed with 1.4 m.
- History, so that it is not tried again: on 2026-09-21 the section was first switched to the tip-to-tip wingspan, b = 1.5 m (10 m / 25 m, 67 % → 42 %), then reverted the same day. Reason: the b of 4.2 defines the collision test that produced every result, and the code is not going to change before the thesis is handed in; with 1.5 m the equation would not be the test that was run.
- Not touched: the caption of `fig:forest-sensing` (l. 491) still says "trunks and wingspan are to scale", and the legend inside `forest_random_vs_stratified` still says "gap narrower than the wingspan" (both drawn with 1.4 m; 10 cm is invisible at those scales).
- For a future batch only: if the fuselage is to be counted, add w_f where the collision span is read (`env.py:781`, `multi_drone_env.py:396`), NOT in `_wingspan_from_surfaces`, which also scales `k_slip_wing`; apply it to every arm of a comparison at once.

### B4. 30 N force cap, l. 262 [claim]
- "several times the weight of the drone" holds for typical bodies (5 × for the reference drone, 0.605 kg) but not for the heaviest (2.5 kg: two panels carry at most 60 N = 2.4 g).
- Proposed: *"the force on each element is limited to 30~N, about five times the weight of the reference drone […]. On the heaviest bodies that the genome can express the limit is tighter in relative terms (the two wing panels together can carry at most 2.4 times the weight of a body of 2.5~kg), so it also bounds the load factor of those bodies."*

### B5. 245 760 flights, l. 53 (and "a quarter of a million", l. 6, 140) [gap]
- Chapter 3 says only "sixty forests" per body; the reader cannot reconstruct 245 760 = 64 × 64 × 60.
- Proposed, l. 53: *"scores 245\,760 of them (64 bodies, 64 candidate controllers for each body, 60 forests for each pair)"*. No method name needed, no cross-chapter pointer needed.

### B6. Centre-of-pressure noise clip, l. 279 [fact]
- std = σ_cp·s/4 of the chord ≤ 1 % of the chord; the 15 % clip is 15 standard deviations away and never acts. It is in the code (`base_aero_solver.py:602-609`), not a transcription error, but in the text it reads like one.
- Proposed: delete ", limited to 15\% of the chord". Optional: add the physical size, *"at most 1\% of the chord, 2~mm on the wing of the reference drone"*.

### B7. Third level of noise and the unkept pointer, l. 283 [fact + gap]
- Three problems: the offset is never quantified; "the initial conditions are drawn at random for every flight" is true only in training (`env.py:1736`, `if not self.evaluation`); "described with it" points to a description that does not exist in 4.2.
- Values (`wp1_config.yaml` of batch_5): latency 0 or 1 control step, drawn once per flight (`action_latency_random_per_step: false`); joint-target bias constant over the flight, std 0.01; joint-target jitter at every step, std 0.005 (check the unit: normalized command or rad).
- Proposed: *"The third level is the interface between the simulator and the controller. The observations are corrupted by sensor noise (\autoref{tab:drone-sensors}). The commands reach the servomotors with a latency of zero or one control step, drawn once per flight, and each joint target carries an offset that is constant over the flight (standard deviation 0.01) plus a jitter drawn at every step (0.005). The forest is drawn at random for every flight. The initial state is randomized only while the controllers are trained; when a body is evaluated, every flight starts from the same state (\autoref{sec:flight-environment})."*
- l. 290: add the release speed: "heading along $x$ at 15~m/s" (`env.py:917`).
- Training-time randomization of the release (x ± 15 m, y ± 40 m, z −10…+5 m, forward speed 5 to 25 m/s, `env.py:1632-1649`) goes to chapter 5.

### B8. Who sets the commanded forward speed, l. 465 and `tab:drone-sensors` [gap] (found while checking)
- Evaluation: fixed grid `linspace(10, 20 m/s)` over the 60 forests of each body (`evaluate.py:1125`, `run.yaml` vmin/vmax). Training: uniform 5 to 25 m/s. It affects both objectives and is nowhere in the chapter.
- Proposed, after "the forward speed it is asked to keep": *"The commanded speed is set by the experiment, not by the controller, and is constant during a flight. The flights by which a body is evaluated cover a fixed set of speeds between 10 and 20~m/s, so that a body is judged over a range of speeds and not at the one that suits it best."* Training range to chapter 5.

### B9. End of a flight: missing thresholds and the time limit, l. 301 [gap]
- Add the values: ground = altitude below 0.1 m; also an altitude ceiling of 50 m, which the list omits; time limit 60 s when a body is evaluated (`evaluate.py:694`; 100 s in training).
- Proposed: *"The time limit is not what ends a flight in practice: a drone that keeps the lowest commanded speed covers the longest course of this thesis in about half of it, so the progress measures how far a body gets and not how fast."*
- **Verify before writing "never"**: count the time-outs in the exam logs of one production run.

### B10. Definition of Re_nom, l. 163 and l. 219 [gap]
- `src/naca_generation/post_processing.py:566` (`fit_renom_a1`): Re_nom is fitted per airfoil on the XFOIL polars with the model c_l,max(Re)/c_l,max(high Re) ≈ min(Re/Re_nom, 1), robust least squares, range 30 000 to 300 000. NACA 3416: 78 108. Tail section: 50 000.
- Proposed, l. 219: *"$\mathit{Re}_{\mathrm{nom}}$ is fitted for each airfoil on the same polars, as the Reynolds number below which the maximum lift coefficient of the section starts to fall, under the model $c_{\ell,\max} \propto \min(\mathit{Re}/\mathit{Re}_{\mathrm{nom}}, 1)$. It is 78\,000 for the section of the reference drone, whose wing works between $\mathit{Re} = 135\,000$ and $270\,000$ at the speeds commanded in this thesis, so the factor is active only on the narrowest wings that the genome allows."*
- This also answers the reviewer's request for a justification of the linear law, and makes B15 a one-clause matter.

### B11. Value of the slipstream gain, l. 243 [gap]
- "through a gain $k_s$" → "through a gain $k_s = 1$".

### B12. Downwash at the wing vs at the tail, l. 238 [opt]
- The text already says "predicts at the wing". Optional clause: *"(far behind the wing the same theory gives twice this value; the gain $k_{\varepsilon}$ is there to absorb the difference and is left at one)"*.

### B13. Ranking claim in "Limits of the model", l. 264 [claim] (strongest scientific point of the review)
- Problems: (i) "damped only indirectly, through the change of incidence they produce" does not hold for roll, since a roll rate changes no incidence when every surface sees v_cm, so roll has no aerodynamic damping at all; pitch damping ∝ l_t² is absent too; (ii) no section pitching moment (C_m0): the force acts at the centre of pressure of `eq:aero-cp` and the nose-down moment of a cambered section is missing from the trim; (iii) the closing sentence claims the model ranks bodies by "where its surfaces sit with respect to its centre of mass", which is where (i) and (ii) bite.
- Proposed replacement for the ω × r sentence: *"[…] the additional flow that a rotation of the body induces at a surface far from the centre of mass is neglected. The model therefore has no direct aerodynamic damping of the rates of pitch, roll and yaw: a rotation is opposed only once it has changed the incidence of the surfaces, which a pure roll does not do."*
- Add: *"No pitching moment of the section is modelled either: the force acts at the centre of pressure \eqref{eq:aero-cp}, and the nose-down moment of a cambered airfoil is absent from the trim of the aircraft."*
- Proposed replacement for the closing sentence: *"These simplifications leave intact what depends on size and on static balance (how much wing a body has, how it is proportioned, how much it weighs, how its surfaces are placed with respect to its centre of mass for trim and static stability), and the comparisons of this thesis rest on those. What depends on rate damping, such as the benefit of a long tail arm against pitch oscillations, is underestimated, and the absolute values of range and energy are those of the model and would need to be validated against flight data before being read as predictions."* (See A2 for the section.)

### B14. Servomotor torques are idealized, l. 441-446 [claim]
- 0.6 N m before the gear (twist) and 1 N m ungeared (tail) from a 10 g servomotor is 3 to 5 times commercial hardware (about 0.2 N m); 0.75 N m from 25 g is about 2 times.
- Proposed, end of the paragraph: *"These torques are generous for servomotors of this size, several times what commercial units of 10 to 25~g deliver, and should be read as an idealization of the actuators."*

### B15. f_Re also scales the flat-plate coefficients, l. 167 [opt]
- Faithful to the code (`simple_drone.py:1388-1389`; `re_a` = 1 and `w` = 0 in production, so eq. 4.7-4.8 are exact). Physically unfounded post-stall, but inactive except on chords below about 0.1 m (see B10). One clause in "Limits" at most: *"the Reynolds factor also scales the separated-flow coefficients, a choice without physical basis that is inactive for all but the narrowest wings"*. No citation hunt: it is a choice of the solver, documented "as used".

### B16. "The role of chance shrinks", l. 348-353 [claim]
- Computed for the blind straight flight (w = 2.9 m): uniform example mean 10.6 m, sd 10.6 m, sd/mean 1.00; growing example mean 26.0 m, sd 21.4 m, sd/mean 0.83. The reduction is real but small; "flights end in a band rather than anywhere" oversells it.
- Proposed: keep "makes long flights rapidly improbable instead of merely rare" and the "less than once in a hundred" figure; replace "the risk now grows with the distance, and flights end in a band rather than anywhere" with *"the risk now grows with the distance, and the spread of the distance covered, relative to its mean, falls from 1 to 0.83"*.
- Optional and stronger: quote the per-flight spread of progress of a real controller from the existing exam logs (growing forests). No new uniform-forest runs.

### B17. Density profile of the Latin hypercube forest, l. 369 / caption l. 374 [opt]
- Cell side linear in x gives density ∝ 1/side², not linear. One clause: *"(the density, inversely proportional to the area of a cell, then grows faster than linearly: the forest of \autoref{fig:forest-latin} matches that of \autoref{fig:forest-growing} in its number of trees, not in its profile)"*.

### B18. Why not a minimum-distance process, l. 380 [opt, mainly for the defence]
- Rejection at d < 2r removes the 25 % of overlapping trunks but not the 64 % of gaps narrower than the drone, i.e. not the walls the third generator was written against; and two overlapping trunks are just a wider obstacle. Optional sentence saying so; otherwise keep as a prepared answer.

### B19. Trim table of the standard drone, new, in 4.3.4 [gap]
- Rows: stall speed, best L/D of the whole aircraft and its speed, throttle and power in level flight at 10 / 15 / 20 m/s, static margin. One docker run with the standard drone, or a calculation from the model's formulas. The isolated-wing L/D of 18.7 (l. 215) says little, since the fuselage drag is of the same order as the wing's parasitic drag (check the fuselage reference area when filling the table).
- The whole-aircraft numbers include the effect of A1: do this early.

### B20. Table of simulation parameters, new (end of chapter or appendix) [gap]
- Closes the reviewer's list of missing values in one place. Most values are already in `THESIS_DECISIONS.md` §6. Rows to include, with what still has to be looked up or checked:
  - timing: control 25 Hz, physics 100 Hz;
  - servos: k_p 8, k_v 2; torque limits (B2); joint friction / damping before the gear scaling, from `actuators.csv`: sweep servo 0.05 / 0.005, other servos 0.1 / 0.01 (N m, N m s/rad; check against `drone_making.py:449-456`, which scales them by k and k²); armature: Genesis default 0.1 (`genesis/options/morphs.py:791`; check that `urdf_args` does not override it);
  - aerodynamics: ρ 1.225, μ 1.81e-5, e 0.8, m 0.2, k_ε 1, k_s 1, fuselage C_D0 0.35 and its reference area (to look up), force cap 30 N, speed clip 40 m/s, C_ℓα 0.11/deg;
  - propeller: T_max 5.2 N, C_T0 0.093, c_1 0, c_2 2.148, K_V 2300 rpm/V, U 7.4 V, f_c 0.32 Hz, κ 0.01 m, C_P0 0.044, c'_1 0, c'_2 1.104;
  - servo power (R, k_V, k_I): sweep servo (2.80, 1.25, 0.35), others (8.84, 1.39, 0.55) (`power.py:536`). **Caution:** the power model uses fixed gear ratios [2, 2, 2.5, 2.5, 2, 2], not those of the genome (open item (c) of §8); the table will make this visible, decide how to word it;
  - noise: σ_m = σ_d 0.05, σ_cp 0.01, mass 2 %, centre of mass 4 mm, latency 0 to 1 step, target bias 0.01, target jitter 0.005, sensor noise as in `tab:drone-sensors`;
  - task: corridor 100 m, r 0.75 m, release (−30, 0, 15) m at 15 m/s, ground 0.1 m, ceiling 50 m, attitude limits 90° (roll 100° in evaluation), time limit 60 s (evaluation) / 100 s (training), margin τ 1 cm / 20 cm, depth 20 sectors, 80°, 30 m, commanded speed 10 to 20 m/s (evaluation).

---

## C. Structure and style

- **C1.** l. 1: remove "DA RILEGGERE" from the chapter title once the re-read is done (it reaches the table of contents and the running heads).
- **C2. [opt]** 4.1.1: the list of compiler optimizations (l. 15) and `fig:taichi-kernel` are the expendable parts. Keep fields and their binding to kernels, the build, the B axis, the GPU-to-GPU copy: 4.1.2 depends on them. (The reviewer wrote "zero-copy"; the text is right, it is a copy.)
- **C3. [decide]** Credit paragraph for Andrea Vicari (l. 138): the author's decision of 2026-09-18 stands. Optional: ask Andrea how he wants to be cited and add an `@unpublished` / "in preparation" entry. Thanks are repeated in the acknowledgements as already planned.
- **C4.** Style pass during the re-read: about 78 explanatory colons and 27 "is not / does not …" constructions in about 10 000 words. Instances to start from: "None of the numerical work is done by the Python interpreter" (l. 15), "None of it is used here" (l. 133), "this last point is not cosmetic:" (l. 262), "the choice is less innocent than it seems" (l. 303), "It is not a fixed model:" (l. 385), "These numbers are not arbitrary." (l. 555). Keep a few, vary the rest.
- **C5.** Citation [48] (`tao2024genesis`, "two orders of magnitude"): already removed in the working tree. The key stays in the `.bib` uncited (harmless, `\nocite{*}` is off). Update `THESIS_DECISIONS.md` §7, which still lists it as cited in 4.1.

## D. After the edits

- `latexmk`, zero undefined references, zero em-dashes, look at the changed pages (`THESIS_DECISIONS.md` §1).
- Record the decisions on A1, A2, C3 and the progress definition (B1) in `THESIS_DECISIONS.md` §5-§6, and carry B1, B7, B8 into chapter 5 (training release randomization, training speed range, aggregation of CoT).
