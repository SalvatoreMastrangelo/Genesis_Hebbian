# Rewarding Hebbian Plasticity In-Distribution — Diagnosis & Literature Synthesis

> **Session:** 2026-06-01 · **Branch:** `outer_loop`
> **Scope:** Why WP2 CMA-ES evolution of last-layer Hebbian rules washes to the zero-rules
> baseline, why σ won't drop, and the literature-verified, **in-distribution-only** (no OOD)
> levers to make plasticity actually rewarded.
> **Provenance:** empirical analysis of run `2026-06-01_15-32-04_per_output_small_specialist`
> + a deep-research sweep (25 primary sources, 120 claims extracted, 25 adversarially
> verified by 3-vote panels → 16 confirmed / 9 killed).

---

## 0. TL;DR

1. **Your σ is flat because the search has parked on a flat, noisy plateau *at* the
   zero-rules baseline** — not because the optimizer is broken. The population mean fitness
   has converged to the baseline (391 ≈ 391); the "best" individual's edge is fully
   explained by best-of-128 noise (winner's curse).
2. **Forcing σ down would make it worse**, not better: fitness *differences* scale ~linearly
   with σ while the eval noise floor is fixed, so **SNR ∝ σ**. Shrinking σ converges
   prematurely onto a noise point. The algorithm "refusing" to shrink σ is correct
   information: there is no local gradient to descend.
3. **The literature explains the plateau exactly and predicts your eta↑ result.** Plasticity
   pays in-distribution **only** when a hidden, episode-fixed latent variable changes the
   *optimal action* (value-of-information > 0). Your mass/COM/aero domain randomization is
   the "cartpole" case — the optimal action doesn't change, so no plasticity rule can win.
4. **Your frozen recurrent (LSTM) base already does the adaptation job** (RL² in-context
   adaptation), saturating what little pressure exists.
5. **The fix is task + base, not rule magnitude:** (a) a high-VoI hidden latent — *wide
   multi-URDF morphology* is your ready-made one; (b) handicap the LSTM so plasticity has a
   job; (c) optionally a three-factor (neuromodulated) rule. Ranked table in §5.

---

## 1. Empirical diagnosis (live run)

Run: `logs/runs_hebbian/2026-06-01_15-32-04_per_output_small_specialist`
Config: `rules_per_neuron: true`, `num_urdfs: 1`, pop = 128, σ₀ = 0.075, η = 0.005
(evolved, range [0, 0.01]), decay = 0.05 (evolved, range [0, 0.1]), `crn: true`,
`num_eval_envs: 12800` → **F = 12800 / 128 = 100 forests/individual**.

| gen | σ | mean_fit | best_fit | std_fit | cond(C) | **baseline_fit** |
|----:|------:|---------:|---------:|--------:|--------:|------------------:|
| 0   | 0.0714 | 356.6 | 407.5 | 29.7 | 1.00 | **391.5** |
| 25  | 0.0617 | 381.7 | 414.3 | 16.5 | 1.13 | **390.4** |
| 50  | 0.0607 | 384.4 | 414.5 | 15.6 | 1.29 | **389.8** |
| 65  | 0.0604 | 391.1 | 416.7 | 11.8 | 1.40 | — |

**Reading:**
- **σ is ~flat** (0.0714 → 0.0604 over 65 gens, local min 0.0588 @ gen 44, then drifts back up).
- **Mean converged to baseline:** evolved mean 391.1 vs baseline 391.5. On the *real task
  metrics* it is slightly **worse**: progress 247 vs 251, crash 0.87 vs 0.855, velocity
  ~13.0 vs 13.17.
- **best=416.7 is winner's curse, not signal.** Best-of-128 under noise std ≈ 12 sits at
  mean + ~2.6σ ≈ 391 + 31 ≈ **422** from luck alone. The observed best is *below* that — no
  genome edge needed to explain it.
- **cond(C) 1.0 → 1.40, std 30 → 12:** the population is concentrating (real, but onto the
  baseline-equivalent region), not discovering super-baseline rules.

**Genome inspection (gen 69 CMA mean, 252 genes = ABCD 28 + decay 112 + eta 112):**
- ABCD mean |dev from neutral 0.5| = **0.087** (max 0.20) → nonzero but moderate rules.
- decay block mean 0.5005 → ≈ 0.05; eta block mean 0.493 → ≈ 0.0049.
- **Interpretation:** plasticity is *on* (nonzero ABCD, η ≈ 0.005) but **net-neutral** —
  the evolved rules buy nothing over baseline. CMA is wandering a flat manifold of
  rule-sets that all score ≈ baseline.

---

## 2. Why σ won't drop, and why forcing it down backfires

CMA-ES step-size control (CSA) shrinks σ only when consecutive mean-steps **anti-correlate**
(the overshoot signature of descending a real gradient), grows it when they correlate, and
**holds it constant when steps look random** — which is exactly what noise-dominated
selection on a flat landscape produces. Flat σ ⟺ "selection is not producing consistent
directional progress." The algorithm is correctly reporting *no local gradient here*.

**The SNR argument (why a manual σ cap hurts):** for a smooth landscape, fitness differences
between individuals scale ≈ linearly with σ (Δf ≈ ∇f · σ·z), but the eval noise floor is
fixed. So **signal-to-noise ∝ σ**. Shrinking σ → ranking becomes *more* noise-dominated →
the mean random-walks and prematurely converges onto wherever noise points. This is the
classic noisy-ES residual-error result (Arnold & Beyer): below a critical normalized noise
the ES makes *negative* progress. You would lock in a noise-selected point at/just-below
baseline — i.e. the −4.5 % wash already documented in independent CRN validation.

**The only principled way to get σ to fall (and fine-tune for real)** is to lower the noise
floor so signal survives at small steps (`σ_useful_min ≈ noise / |gradient|`): more
rollouts/individual (F↑, floor ∝ 1/√F), a less chaotic eval (sparser/shorter forest → lower
Lyapunov amplification), or a bigger genome signal — **but none of that helps if there is no
gradient to begin with**, which §3 shows is the actual situation.

---

## 3. Why plasticity washes in-distribution (literature, verified)

### 3.1 The governing principle  *(verified 3-0)*
Plasticity yields a measurable in-distribution fitness advantage **only when a hidden,
episode-fixed latent variable changes the *optimal action*** — not merely when the latent
*varies*. Formally: value-of-information > 0 in a Bayes-Adaptive MDP / Hidden-Parameter MDP.
When the latent doesn't flip what to do, the Bayes-optimal adaptive policy **provably
degenerates to a fixed policy** and the adaptation advantage is **exactly zero**.

> Canonical example — **cartpole:** randomizing pole mass is a hidden latent, but "falling
> left → move left" regardless. Optimal action unchanged → adapting net has nothing to do.
> **WP2's mass/COM/aero domain randomization is the cartpole case.** A robust WP1 controller
> handles those with the same action map → flat plateau, unfixable by optimizer tuning or η.

*Doshi-Velez & Konidaris 2013, HiP-MDP (arXiv:1308.3513); Zintgraf et al. 2021, VariBAD
(JMLR v22, 21-0657).*

### 3.2 Why η↑ made convergence *worse*  *(consistent with verified evidence)*
With no differential advantage to find, the fitness surface over rules is flat + noise.
Larger η amplifies the per-step weight perturbation → more trajectory divergence → more
fitness variance → worse rank signal for CMA-ES. You are turning up a mechanism that, absent
a job, only injects instability. Confirmed by the dose data: in-distribution (no adaptation
needed), plastic nets reliably **underperform** static ones (Najarro & Risi Ant: static
**1604 ± 171** vs Hebbian **1051 ± 113**).

### 3.3 Two myths the verification *killed* (these free you up)
- ❌ **"You need within-episode non-stationarity (a mid-episode switch)."** *Refuted 0-3.*
  An **episode-fixed** latent that must be *identified online* is sufficient. **This is the
  key to the no-OOD constraint:** a constant-within-episode hidden parameter is partial
  observability, not distribution shift — no perturbation, no OOD.
- ❌ **"A recurrent base makes plasticity universally redundant."** *Refuted 0-3* (two
  phrasings). The LSTM *competes* for the same job, but does not provably dominate. Your
  setup is *saturated*, not hopeless.

### 3.4 Why your setup is the hardest case — the saturation diagnosis  *(verified 3-0)*
1. **Frozen recurrent base = the job is taken.** In RL², the LSTM hidden state literally
   serves as the inferred task parameter, updated every timestep, never reset within a
   trial — implicit in-context adaptation through the forward pass. Your frozen LSTM is
   already doing the online system-ID last-layer Hebbian is meant to provide.
   *Beck, Vuorio, Zintgraf, Finn, Whiteson 2023, Survey of Meta-RL (arXiv:2301.08028).*
2. **The canonical wins use a *handicapped* base, the opposite of yours.** Najarro & Risi's
   in-distribution CarRacing win (**872 ± 11 vs 711 ± 16**) comes from the static baseline
   committing to one weight set while the Hebbian net **re-derives weights from random init
   every episode** — the random reinit *is* the adaptation pressure. You load a finely-tuned
   frozen checkpoint, so that pressure doesn't exist. Their headline quadruped result is
   **explicitly OOD** (held-out leg damage); on *seen* morphologies the static net **won**.
   *Najarro & Risi 2020 (NeurIPS, arXiv:2007.02686); Palm, Najarro & Risi 2021 (PMLR v148).*

> **Load-bearing unknown:** no paper tests last-layer Hebbian on a *frozen pretrained
> recurrent* policy. The literature robustly validates the **task-design principle**; it
> gives **zero** direct evidence your exact architecture will exploit it. Genuinely novel
> territory — the levers below are principled bets, not guarantees.

---

## 4. The ranked list of levers (in-distribution only)

### Tier 1 — attacks the root cause

**1. Redesign the eval as a high-VoI BAMDP: hidden, episode-fixed latents that flip the
optimal action.**
- *Mechanism:* per episode, draw a latent θ held constant all episode, hidden from the
  actor, that materially changes the optimal control → the only way to score is to identify
  θ online and specialize.
- *Why in-distribution:* θ is from the existing training distribution; nothing is perturbed
  mid-flight, nothing is OOD. Partial observability, not distribution shift.
- *For you, concretely:* **morphology is your ready-made high-VoI latent** — different
  wings/mass genuinely need different control, and the actor is already morphology-blind.
  The current run is `num_urdfs: 1` (zero latent → guaranteed plateau). Go multi-URDF with
  the **widest** morphology spread WP1 was trained on. Mass/aero DR alone won't do it
  (low VoI, the cartpole case).
- *Dose-response is quantified:* advantage *scales* with hidden-latent uncertainty.
  Soltoggio single T-maze (1-of-2) → small edge; double T-maze (1-of-4) → *large* edge
  (modulatory ~97 % of optimal). Push spread to where the inferred latent **repeatedly
  flips** the required behavior.
- *Caveat:* the LSTM may still identify θ first (Tier 2 addresses this).
- *Sources:* Doshi-Velez & Konidaris 2013; Zintgraf et al. 2021; Soltoggio et al. 2008.

**2. Handicap the recurrent route so plasticity has a job.**
- *Mechanism:* shorten/zero the LSTM hidden state at eval (truncate window, periodic reset,
  inject hidden-state noise), or evolve plasticity against a feedforward/weakened readout.
- *Why in-distribution:* you constrain the *policy*, not the *data* — eval distribution
  untouched.
- *Diagnostic value:* the clean test of the saturation hypothesis — if plasticity suddenly
  helps once memory is shortened, the LSTM was the saturator.
- *Caveat:* over-handicapping wrecks base competence; sweep the memory window. **No verified
  quantitative evidence for the magnitude** — treat as unknown.
- *Sources:* implied by RL² saturation (Beck et al. 2023); handicapped-base mechanism
  (Najarro & Risi 2020).

### Tier 2 — better plasticity rule

**3. Upgrade plain ABCD → a three-factor / neuromodulated rule.**
- *Mechanism:* `ẇ = M · H(pre, post)` — a global, network-emitted scalar `M` (reward,
  novelty, surprise) multiplicatively *gates* the local Hebbian term, controlling
  *when/how much* plasticity fires.
- *Why it helps in-distribution:* beats non-modulated plasticity *and* a standard LSTM at
  constant parameter count (PTB language modeling 104.26 → 102.48, Wilcoxon p = 1e-7) —
  value added *on top of* a recurrent base.
- *Critical — plain ABCD is REWARD-BLIND:* the agent never sees the reward, so plain ABCD
  **cannot adapt to anything that only appears in the reward**. The latent must manifest in
  the **observations/activations** the Hebbian term correlates on (morphology does — it
  changes the sensed dynamics), *or* you must switch to a reward-modulated rule.
- *Hard caveat:* Backpropamine's wins came from **end-to-end gradient descent jointly
  co-adapting base + rule** — *not* CMA-ES on a frozen base. Don't expect the published
  magnitudes; you'd evolve the modulator readout with CMA-ES and the differentiable
  co-adaptation that produced those wins is absent.
- *Sources:* Miconi, Rawal, Clune, Stanley 2019, Backpropamine (ICLR, arXiv:2002.10585);
  Frémaux & Gerstner 2016 (Front. Neural Circuits, PMC4717313).

### Tier 3 — search & objective  *(requested but SPECULATIVE — no independently-verified claims this pass)*

**4. Shape the fitness to reward the *slope*, not the *level*.** Score within-episode
improvement rate / area-under-the-learning-curve / regret, so a net that *gets better as it
identifies θ* outscores a flat one — and CMA-ES gets a gradient where raw fitness is flat.
Pairs naturally with the time-resolved diagnostic in §6. *Speculative.*

**5. Swap CMA-ES for Quality-Diversity (MAP-Elites / novelty search).** Illuminate
rule-space by behavioral descriptor instead of a flat scalar; manufacture a search gradient
where fitness gives none. *Speculative; higher cost (replaces the optimizer).*
*Sources:* Mouret & Clune 2015 (MAP-Elites); Lehman & Stanley 2011 (Novelty Search);
Pugh et al. 2016 (QD frontier, Front. Robot. AI).

**6. TTT-style self-supervised plastic signal.** Drive the plastic update from an
in-distribution self-supervised error (e.g. next-obs prediction) rather than reward.
*Caveat: the canonical TTT papers are about distribution **shift** (OOD) — the
in-distribution version is a speculative adaptation, lowest priority given the constraint.*
*Sources:* Sun et al. 2020 (TTT, arXiv:1909.13231); Hansen et al. 2020 (arXiv:2007.04309).

### Cross-cutting warning  *(verified 3-0)*
**Don't naively shrink the ABCD search via joint shared-rule + synapse-assignment (GMM).**
It collapsed to ~zero reward in every config tested (Ant: all GMM configs ≈ 0 vs static
1545). Use **evolve-then-cluster/merge** (Pedersen & Risi 2021) or a small fixed number of
shared rules with *fixed* assignment.
*Source:* Palm, Najarro & Risi 2021 (PMLR v148).

---

## 5. Bottom line — ranked for *your* setup

| # | Lever | Payoff | Cost | Verified? |
|--:|-------|--------|------|-----------|
| 1 | High-VoI episode-fixed latent (wide multi-URDF) | **Highest** — fixes root cause | Low (config) | Principle ✓ |
| 2 | Handicap / shorten the LSTM memory at eval | High + diagnostic | Med (eval code) | Principle ✓, magnitude ✗ |
| 3 | Three-factor neuromodulated rule | Med-High | Med (rule + genes) | ✓ (co-adaptation caveat) |
| 4 | Slope / AULC / regret fitness | Med | Low-Med | Speculative |
| 5 | Quality-Diversity (MAP-Elites / novelty) | Med | High (new optimizer) | Speculative |
| 6 | TTT self-supervised plastic update | Low (mostly OOD) | High | Speculative |

**The single load-bearing sentence:** *your domain randomization doesn't flip the optimal
action, so no plasticity rule can win — and the LSTM has already eaten what little adaptation
pressure exists.* The fix is **task** (high-VoI hidden latent, which multi-URDF gives for
free) plus **base** (stop the LSTM solving it), not rule magnitude.

---

## 6. Recommended first experiment (tests the whole theory at once)

**Wide multi-URDF + an eval-time LSTM-memory handicap + a time-resolved plastic-vs-frozen
curve.**

Diagnostic, not winner's-curse: compare mean-plastic vs frozen on the **same held-out**
in-distribution forests, **time-resolved**. Real adaptation shows the plastic curve
*pulling ahead as the episode progresses* (after θ = morphology is identified). A uniform
gap from t = 0 is just a better static policy; a gap that opens *late* is the actual
plasticity signal. If plasticity pulls ahead late **only once memory is shortened**, you've
confirmed both the latent-design and saturation halves in one run.

---

## 7. Verified claim ledger (audit trail)

### Confirmed (high confidence)
| Claim | Vote | Source(s) |
|-------|------|-----------|
| Plasticity advantage requires hidden episode-fixed latent that flips the *optimal action* (VoI>0); HiP-MDP/BAMDP is the no-OOD recipe; mere variation insufficient (cartpole) | 3-0 (+1-1) | arXiv:1308.3513; JMLR 21-0657 |
| Frozen recurrent (LSTM) base already performs in-episode adaptation (RL²) — supports saturation | 3-0 | arXiv:2301.08028 |
| Najarro & Risi's win derives from a *handicapped* base (random reinit each episode), opposite of a frozen checkpoint | 3-0 (+2-0) | arXiv:2007.02686 |
| Najarro/Palm headline is **OOD** (leg damage); on *seen* morphologies static **beat** plastic | 3-0 | PMLR v148; arXiv:2007.02686 |
| Dose-response: advantage scales with within-episode latent uncertainty (single vs double T-maze; ~97% optimal) | 3-0 | Soltoggio et al. 2008 |
| Neuromodulation (three-factor / Backpropamine) beats plain ABCD *and* LSTM at constant params | 3-0 (+2-1) | arXiv:2002.10585; PMC4717313 |
| Backpropamine wins relied on end-to-end **gradient descent** co-adaptation, **not** CMA-ES on a frozen base | 3-0 | arXiv:2002.10585 |
| Plain ABCD is **reward-blind** → latent must appear in obs/activations, or use a reward-modulated rule | 3-0 | PMLR v148; PMC4717313 |
| Joint shared-rule + assignment (GMM) bottleneck **fails** (~0 reward); prefer evolve-then-merge | 3-0 | PMLR v148 |

### Killed (refuted — do **not** rely on these)
| Refuted claim | Vote |
|---------------|------|
| "Plasticity helps only in non-stationary / variable-reward tasks" | 0-3 |
| "Fixed-weight RNNs can always match plastic nets on adaptive tasks" | 0-3 |
| "Plain two-factor Hebbian is *mathematically incapable* of reward tasks" (it's reward-blind, not incapable) | 0-3 |
| "Differential advantage only on tasks needing outcome discrimination" | 0-3 |
| "Advantage requires the reward location to switch *mid-lifetime*" | 0-3 |
| "Recurrent-memory solutions are harder to evolve than plastic ones" | 1-2 |
| "Always-on Hebbian fails by catastrophic forgetting; gating is required" | 0-3 |
| "RL² with prev action+reward does *all* adaptation, leaving nothing for plasticity" | 0-3 |
| "Adaptation advantage shrinks to zero as reward gets denser" | 0-3 |

### Open questions (the load-bearing unknowns)
1. Will last-layer Hebbian on a **frozen recurrent** base actually exploit episode-fixed
   latent spread, given the hidden state may already absorb the system-ID? *No source tests
   this head-to-head — the central empirical unknown.*
2. How much to handicap the LSTM (window length / zeroing / noise / feedforward swap) to
   free an adaptation job **without** leaving distribution or destroying base competence?
3. Minimum within-distribution latent spread (and identifiability in the obs the ABCD term
   sees, since ABCD is reward-blind) to lift fitness measurably above baseline — where does
   WP2's DR sit on the cartpole↔double-T-maze spectrum?
4. Can slope/AULC/regret fitness or QD manufacture a usable CMA-ES gradient on the flat
   plateau, and can a TTT-style in-distribution self-supervised error drive the plastic
   update? (Threads 4/5/7 — requested but unverified this pass; warrant a dedicated search.)

---

## 8. Sources

**Primary, underpinning verified findings**
- Doshi-Velez & Konidaris 2013 — *Hidden Parameter MDPs* — arXiv:1308.3513
- Zintgraf et al. 2021 — *VariBAD* — JMLR v22 (21-0657)
- Beck, Vuorio, Zintgraf, Finn, Whiteson 2023 — *A Survey of Meta-RL* — arXiv:2301.08028
- Najarro & Risi 2020 — *Meta-Learning through Hebbian Plasticity in Random Networks* — NeurIPS, arXiv:2007.02686
- Palm, Najarro & Risi 2021 — *Testing the Genomic Bottleneck Hypothesis in Hebbian Meta-Learning* — PMLR v148 (palm21a)
- Soltoggio et al. 2008 — *Evolutionary/Computational Advantages of Neuromodulated Plasticity* — ALife XI
- Miconi, Rawal, Clune, Stanley 2019 — *Backpropamine* — ICLR, arXiv:2002.10585
- Frémaux & Gerstner 2016 — *Neuromodulated STDP & Three-Factor Learning Rules* — Front. Neural Circuits, PMC4717313

**Additional sources consulted (claims not independently verified this pass)**
- Duan et al. 2016 — *RL²* — arXiv:1611.02779
- Wang et al. 2016 — *Learning to Reinforcement Learn* — arXiv:1611.05763
- Soltoggio, Stanley, Risi 2017 — *Born to Learn (EPANN review)* — arXiv:1703.10371
- Miconi, Stanley, Clune 2018 — *Differentiable Plasticity* — arXiv:1804.02464
- Mouret & Clune 2015 — *Illuminating Search Spaces by Mapping Elites (MAP-Elites)*
- Lehman & Stanley 2011 — *Abandoning Objectives: Novelty Search*; *Novelty Search + Local Competition*
- Pugh, Soros, Stanley 2016 — *Quality Diversity: A New Frontier* — Front. Robot. AI (frobt.2016.00040)
- Sun et al. 2020 — *Test-Time Training* — arXiv:1909.13231
- Hansen et al. 2020 — *Self-Supervised Policy Adaptation during Deployment* — arXiv:2007.04309
- Ba et al. 2016 — *Using Fast Weights to Attend to the Recent Past* — arXiv:1610.06258
- Additional angle sources: arXiv:2103.06435, arXiv:1806.05865, arXiv:1807.05076,
  journals.sagepub 1059712310379923, ScienceDirect S0893608012001621

**Verification stats:** 5 search angles · 25 sources fetched · 120 claims extracted ·
25 verified (3-vote adversarial, 2/3 to kill) · 16 confirmed / 9 killed · 107 agents.

---

*Companion persistent memories: `project_wp2_plasticity_reward_literature.md`,
`project_wp2_validation_refutes_evolution_gains.md`, `project_wp2_crn_and_ranking_noise.md`,
`project_production_actor_arch.md`.*
