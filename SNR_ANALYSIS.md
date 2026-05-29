# WP2 Signal-to-Noise (SNR) Analysis — CMA-ES Hebbian-Rule Evolution

**Date:** 2026-05-29   **Branch:** `outer_loop`
**Subject run analyzed:** `logs/remote/wp2_evolution/wp2_random_period_4_r0/2026-05-28_01-19-50_random_period_4`
**GPU validation:** local RTX 5060 (8 GB), `.venv/bin/python` (has `gstaichi`).

This report documents why the inner CMA-ES loop stalled, the signal-to-noise (SNR)
framework we used to diagnose it, **exactly how each SNR component is computed
and measured**, the measured numbers, what we changed in the code, and the
conclusions. It deliberately records where the initial diagnosis was *wrong* and
was corrected by direct measurement.

---

## 1. TL;DR

- The inner CMA-ES stalled: step size `sigma` drifted down slowly, the covariance
  matrix never developed structure (`cond(C) ≈ 1`), and the **population mean fitness
  stayed below the zero-rules baseline in 0/25 generations**.
- CMA-ES ranks individuals, so the quantity that governs progress is the
  **ranking SNR = |signal| / σ_rank**, not absolute fitness or absolute noise.
- **Measured at a realistic scale (16 URDFs, 192 rollouts/individual): signal Δ ≈ 3.5,
  σ_rank ≈ 2.5, SNR ≈ 1.2.** SNR ≈ 1 is exactly the marginal-ranking regime that stalls
  CMA-ES.
- **The bottleneck is the signal, not the noise.** The best evolved Hebbian rules beat
  the frozen WP1 controller by only ~2–3 % (the WP1 generalist is already strong, so the
  rules have little headroom at `eta=0.005`).
- The dominant *noise* term is **chaos-amplified GPU floating-point non-determinism**,
  not the explicit domain randomization. It is irreducible by common random numbers (CRN)
  and only averages down as `1/√(rollouts)`.
- CRN (implemented + validated) removes ~20 % of `σ_rank` (the explicit-DR + forest-sharing
  part). It is correct and worth keeping, but it was **not** the dominant lever — an early
  overstatement that direct measurement corrected.

---

## 2. The original symptom

From the `random_period_4` run (`results/cma_summary.csv`):

| metric | gen 0 | gen 24 | reading |
|---|---|---|---|
| `sigma` | 0.0730 | 0.0584 | slow monotonic decay (not convergence) |
| `cond(C)` | 1.0001 | 1.0151 | ≈ identity → **no covariance structure learned** |
| pop mean fitness | 249.5 | 245.7 | below baseline throughout |
| zero-rules baseline | 291.2 | 256.9 | **> pop mean in 0/25 gens** |

The population mean stayed *below* "do nothing" (zero Hebbian rules) the entire run, and
the gap shrank only as `sigma` shrank — i.e. the optimizer's only "progress" was undoing
its own perturbations and collapsing back toward the start point.

Two structural confounds were present and later removed:
- **Full-covariance CMA at genome dim n = 896.** Covariance learning rate
  `c1 + c_mu ≈ 9e-5/gen` ⇒ after 25 gens `cond(C)` cannot move from 1. Full-cov cannot
  adapt at this dimension/budget; **sep-CMA** is required.
- **Non-stationary objective:** `refresh_urdfs_every: 4` swapped in 40 brand-new random
  URDFs every 4 gens; the zero-rules baseline jumped by **±44 fitness points** across
  refreshes — larger than the entire optimization signal. The objective moved faster than
  the search could adapt.

---

## 3. The SNR framework — definitions and how each component is calculated

### 3.1 Why SNR (and not absolute fitness)

CMA-ES never uses raw fitness values; it uses the **ranking** of the population to form a
weighted recombination (the rank-μ update). So what limits progress is whether the ranking
reflects the genome or the noise. The governing quantity is

```
SNR = |signal| / σ_rank
```

where `signal` is the genome-induced fitness difference between candidates and `σ_rank` is
the noise on that difference. SNR ≫ 1 ⇒ reliable ranking; SNR ≈ 1 ⇒ the ranking is roughly
half random and CMA only crawls.

### 3.2 σ_ind — absolute per-individual noise

**Definition.** Evaluate one *fixed* genome many times, each time with completely fresh
randomness (new forests, new domain-randomization draws, new action sampling). `σ_ind` is
the standard deviation of its scalar fitness across those re-evaluations. It answers "how
repeatable is one drone's score."

**Fitness definition.** A genome's fitness is the WP1 reward sum accumulated over an episode,
**averaged over `N · F` rollouts** — `N` URDFs (morphologies) × `F` forest/speed scenarios
per (URDF, individual). So `σ_ind` is the std of that average.

**The averaging law.** If the per-rollout outcome has std `σ_rollout` and rollouts are
independent, then

```
σ_ind  =  σ_rollout / √(N · F)        (rollouts per individual = N · F)
```

This `1/√(rollouts)` law is the single most important scaling relationship in this report.

> **Correction (recorded honestly).** Our first estimate, `σ_ind ≈ 8`, came from scaling the
> *baseline's* cross-generation std (~0.7, measured over `F=1280` forests) by `√(1280/10)`.
> That cross-gen std conflated true sampling noise with **between-generation forest-refresh
> variance**, so it *overestimated* `σ_ind`. Direct measurement (§5) gives the true value.

### 3.3 σ_rank — ranking noise (the quantity that actually matters)

**Definition.** The std of the *difference* in fitness between two individuals evaluated
**head-to-head under common random numbers** (the same scenarios). It is the part of the
noise that does **not** cancel when you compare them — i.e. what corrupts the ranking.

```
non-CRN:   Var(f̂_i − f̂_j) = 2·σ_ind²            (independent draws)
full CRN:  Var(f̂_i − f̂_j) = interaction term     (shared scenarios cancel)
```

Under CRN, most of the per-rollout variance (which forest, which mass/aero draw, which
sensor-noise sequence) is **identical** for `i` and `j` and cancels in the difference; only
the part that depends on the *interaction* between the genome and the scenario survives.
That is why CRN can make `σ_rank ≪ σ_ind` — and CRN is precisely the act of converting
`σ_ind`-level noise into `σ_rank`-level noise.

**How we measure it directly (same-genome-in-all-slots).** Replicate one genome across all
`P` population slots in a single generation eval. Every slot then runs the identical
controller, so the spread of the `P` per-individual fitnesses is pure evaluation noise =
`σ_rank` for that controller:

```
σ_rank  =  std over the P per-individual fitnesses, all individuals = same genome
```

This is cheap (one eval pass) and doubles as the **correctness test for CRN**: with full CRN
+ deterministic actor + no aero noise, identical controllers must collapse to a single value
(`σ_rank → 0`); any residual exposes an unshared source. (See `src/WP2/measure_sigma_rank.py`.)

### 3.4 signal Δ — the genome effect

**Definition.** The noise-free fitness gap between two genuinely different rule-sets.

**How we measure it (co-located, common random numbers).** Build a population whose first
`P/2` individuals are **zero rules** (the base controller) and last `P/2` are an **evolved**
rule-set, then evaluate them on the **same** forests under CRN:

```
signal  =  mean(evolved P/2)  −  mean(zero P/2)        # forest variance cancels
```

Because both halves fly the same scenarios, the forest/DR variance cancels in the difference
and the residual is the genuine effect of the rules. (See `src/WP2/measure_signal.py`.)

### 3.5 SNR

```
SNR  =  |signal| / σ_rank
```

A practical target for clean per-individual ranking is **SNR ≳ 3–4**. Note CMA's rank-μ
recombination averages noise over the top `μ ≈ P/2` individuals, so the *mean-update
direction* has better effective SNR than any single individual — meaning SNR ≈ 1.4 still
yields *slow* progress, not zero. The original run combined SNR ≈ 1 with full-cov-can't-adapt
and a moving objective, which together produced the stall.

### 3.6 Decomposing σ_rank (what the noise is made of)

`σ_rank²` is the quadrature sum of independent noise sources. We isolate each by toggling it
in the same-genome measurement:

```
σ_rank²  ≈  σ_forest²  +  σ_DR²  +  σ_aero_force²  +  σ_actor²  +  σ_chaos²
```

| source | how isolated | CRN-shareable? |
|---|---|---|
| forest layout | `crn` on/off (forest assignment) | yes (forest-id shared) |
| domain randomization (mass/COM/joint/latency/aero-param/obs) | `crn` on/off | yes (per-forest) |
| per-step aero **force** noise (`ti.randn` in Taichi) | `noise.aero_noise` on/off | **no** (in-kernel) |
| stochastic action sampling | `stochastic` on/off | partial (shared ε, not done) |
| **chaos** (GPU float non-determinism × chaotic flight) | identical controllers, deterministic, aero off | **no** (intrinsic) |

Component = `√(σ_with² − σ_without²)` from the relevant toggle pair.

---

## 4. The chaos floor — the dominant, irreducible noise term

The decisive finding. With **64 bit-identical controllers** (zero rules), the **same**
forests, **deterministic** actions, aero noise **off**, and CRN **on**, the only thing that
should differ between individuals is *nothing* → `σ_rank` should be 0. It was not.

Stepping the identical controllers through one episode and measuring the spread of forward
progress across individuals (`src/WP2/measure_signal.py` sibling diagnostic):

| step | progress-x std across identical controllers [m] |
|---|---|
| reset | 0.0 (bit-identical) |
| 1 | 4.4e-06 |
| 40 | 1.1e-03 |
| 70 | 2.0e-02 |
| 100 | 0.34 |
| 150 | 1.08 |
| 200 | **61.5** |
| 300 | 63.3 |

That is ~7 orders of magnitude of **exponential** growth from a bit-identical reset — a
positive Lyapunov exponent. GPU floating-point non-determinism (non-associative atomic
reductions differ per env-slot, ~1e-5 at step 1) is amplified by the chaotic closed-loop
flight through a dense forest into macroscopic, different crash points.

**Consequences:**
- The chaos floor is **per-slot and unshareable** — CRN, deterministic actions, and aero-off
  cannot remove it. It only **averages down as `1/√(rollouts)`**.
- It **dominates** `σ_rank` at low averaging. Stochastic action sampling, by contrast,
  contributes ~0 (stoch on ≈ stoch off in every measurement).
- A less chaotic evaluation (sparser/regular "lattice" forest → lower Lyapunov exponent)
  would lower the floor — a second rationale for the lattice idea, distinct from layout
  variance.

---

## 5. Measured results

All laptop runs: WP1 checkpoint `intermediate_256_65536_32_010/model_999.pt`, `x_upper=150`,
zero rules unless noted. σ_rank = std across replicas of one genome.

### 5.1 σ_rank vs CRN / stochastic — laptop scale (N=2, F=4 → 8 rollouts/indiv)

Aero noise ON (P=32):

| crn | stoch | σ_rank |
|---|---|---|
| OFF | ON | 18.97 |
| OFF | OFF | 18.39 |
| ON | ON | 15.32 |
| ON | OFF | 17.29 |

Aero noise OFF (P=64):

| crn | stoch | σ_rank |
|---|---|---|
| OFF | ON | 17.60 |
| OFF | OFF | 18.29 |
| ON | ON | 15.15 |
| ON | OFF | **14.17** |

**Decomposition (aero-off run, 8 rollouts):** chaos floor (`crn ON, stoch OFF`) = **14.17**;
explicit DR removed by CRN = `√(18.29² − 14.17²) ≈ 11.6`; CRN reduction `18.29→14.17` (−23 %);
stochastic ≈ 0. `crn ON, stoch OFF` did **not** collapse to ~0 ⇒ the residual is chaos
(confirmed by §4), not a CRN bug.

### 5.2 σ_rank vs averaging — the `1/√(rollouts)` law

| config | rollouts/indiv | σ_rank (crn ON, stoch ON) |
|---|---|---|
| laptop (N=2, F=4) | 8 | 15.32 |
| **16 URDFs (N=16, F=12)** | **192** | **2.65** |
| production-like (N=40, F=10) | 400 | ~1.8 (extrapolated) |

`15.32 / 2.65 = 5.8×` reduction for `√(192/8) = √24 ≈ 4.9×` more rollouts — consistent with
`σ_rank ∝ 1/√(rollouts)` (the extra is the harder 16-URDF mix vs 2).

### 5.3 Signal vs noise — the punchline (16 URDFs, 192 rollouts/indiv, eta=0.005)

32 zero-rule + 32 evolved individuals (`best_individual/fitness/genome.npy` from
`2026-05-18_..._actual_controller`), co-located on the same forests:

| crn | stoch | mean_zero | mean_evolved | signal Δ | σ_rank | SNR |
|---|---|---|---|---|---|---|
| OFF | ON | 170.53 | 175.51 | 4.98 | 3.78 | 1.32 |
| OFF | OFF | 176.26 | 179.16 | 2.90 | 3.63 | 0.80 |
| ON | ON | 170.84 | 174.31 | 3.47 | 2.95 | 1.18 |
| ON | OFF | 176.36 | 180.24 | 3.88 | 2.65 | 1.47 |

**Average: signal ≈ 3.8, σ_rank ≈ 3.3, SNR ≈ 1.2.** The best evolved rules beat zero rules by
only ~2–3 % on a base of ~175. **Noise is comparable to the signal — the SNR ≈ 1 marginal
regime.** CRN lowers `σ_rank` ~20 % (3.78→2.95 stoch-on; 3.63→2.65 stoch-off) — real, but it
sharpens a small signal rather than creating one.

> The honest conclusion: the original stall was **not primarily a noise problem**. With
> `eta=0.005` the plasticity barely changes behavior, so the signal is tiny and ≈ the noise.

---

## 6. What was implemented (and validated)

| change | where | status |
|---|---|---|
| **CRN primitive** `crn_share_(buf, ids)` = `buf.copy_(buf[ids])` | `src/winged_drone_train/crn.py` | validated: reset obs-std = 0 |
| Forest-clobber fix: `reset_idx` respects `_fixed_forest_ids` | `env.py` | validated: forests now shared |
| CRN remaps: mass/COM, joint bias+step, latency, aero-param, obs noise | `env.py`, `power.py`, `obs.py`, `simple_drone.py` | gated on `evaluation.crn` |
| CRN propagation through parallel workers | `multi_scene_eval_env.py`, `parallel_multi_scene_eval_env.py` | validated (σ_rank drops on parallel path) |
| `sigma_reinflate` (re-inflate sigma on morphology change) | `cmaes.sigma_reinflate`; outer override `inner_sigma_reinflate` | wired; **inert** without CMA-state carry across outer gens |
| Config: `eta 0.005→0.02`, `decay 0.05→0.02`, `sigma0 0.075→0.15` | `batch_2/*` | new experiments |

**CRN is correct** (the same-genome test confirms identical controllers + full sharing +
deterministic + aero-off → σ_rank ≈ 0 up to the chaos/aero residual) but removes only ~20 %
of `σ_rank`; the rest is the chaos + aero-force floor.

**Forest-clobber bug (root cause of a hidden noise term):** `reset_idx` unconditionally
re-randomized `forest_ids` per slot, silently overwriting the shared assignment that
`refresh_forests` set — so the "same forests across individuals" guarantee was *false* in the
rollout path, while commanded speed *was* shared. Fixed (gated on `crn`).

---

## 7. The levers (ranked) and recommendations

Because `SNR = |signal| / σ_rank`, attack the ratio from both ends:

1. **Raise the signal — `eta` (and `sigma0`).** Highest leverage and *free* (no env-budget
   cost). `eta` caps the achievable plasticity effect (`signal ∝ eta · |ABCD_opt|`); `sigma0`
   sets how much of the rule space CMA explores (0.075 only reached ~30 % of the ABCD range).
   Raised to `eta=0.02`, `sigma0=0.15`. *Unverified bet:* whether more authority converts to
   signal or to crashes — the amplification test answers it.
2. **sep-CMA-ES.** Mandatory at n=896 (full-cov can't adapt `C` in any realistic budget).
3. **More averaging — lower `σ_rank`.** `σ_rank ∝ 1/√(N·F)`. Budget-expensive: to *halve*
   `σ_rank` needs 4× rollouts. Cannot reach SNR≈3–4 from averaging alone within budget
   (it would starve the population), but complements `eta`.
4. **CRN.** Free ~20 % off `σ_rank`. Keep it on.
5. **Stationary inner objective.** Fixed URDF pool (`refresh_urdfs_every: null`) for clean
   inner-loop diagnosis; let the NSGA-II outer loop drive morphology change.

**Population vs averaging trade (fixed `total_envs = N·P·F`):** with sep-CMA, P ≈ 64 is
plenty (full-cov wants P large), so lowering P frees budget for N·F. Do it for **morphology
coverage / generalization**, recognizing the noise gain is sub-multiplicative.

---

## 8. The open question — is there headroom?

The signal measurement implies the deepest issue: **the frozen WP1 controller is already a
strong generalist**, so per-weight Hebbian rules have little to add on *in-distribution*
morphologies (best evolved ≈ +2–3 % over zero rules). If raising `eta`/`sigma0` does not grow
the signal (just adds crashes), the ceiling is fundamental and the answer is to evaluate where
the *fixed* controller fails — **out-of-distribution / damaged morphologies** (broken wings,
frozen weights, extreme morphs already in the repo) — so online adaptation has real headroom.
The amplification test (re-run the evolved genome at `eta=0.02`) and a signal measurement on
OOD morphs are the two decisive next experiments.

---

## 9. How to reproduce

```bash
# σ_rank (noise floor) — replicate one genome across all slots; sweep crn × stochastic
PYTHONPATH=src .venv/bin/python -u src/WP2/measure_sigma_rank.py \
    --cfg src/WP2/experiments/batch_1/random_period_4/run.yaml \
    --cfg.checkpoint_path <wp1_actor.pt> --cfg.checkpoint_config_path <wp1_config.yaml> \
    --cfg.catalog.num_urdfs 16 --cfg.catalog.refresh_urdfs_every 0 \
    --cfg.evaluation.num_eval_envs 12288 --cfg.evaluation.num_eval_workers 2 \
    --cfg.cmaes.population_size 64 --repeats 3
#  -> crn=ON,stoch=OFF,aero=OFF must read ~0 (CRN correctness); residual = chaos+aero.

# signal vs noise — 32 zero + 32 evolved individuals, co-located on the same forests
PYTHONPATH=src .venv/bin/python -u src/WP2/measure_signal.py \
    --cfg src/WP2/experiments/batch_1/random_period_4/run.yaml \
    --genome logs/runs_hebbian/<run>/best_individual/fitness/genome.npy \
    --cfg.checkpoint_path <wp1_actor.pt> --cfg.checkpoint_config_path <wp1_config.yaml> \
    --cfg.catalog.num_urdfs 16 --cfg.evaluation.num_eval_envs 12288 \
    --cfg.evaluation.num_eval_workers 2 --cfg.cmaes.population_size 64 --repeats 3
#  decode the evolved genome with its TRAINING settings (eta=0.005, decay=0.05, use_oja=false).
```

**Eval sizing reference** (`total_envs = N · P · F`, `rollouts/indiv = N · F`,
`σ_rank ∝ 1/√(N·F)`):

| context | N | P | total_envs | F | rollouts/indiv |
|---|---|---|---|---|---|
| old production (`random_period_4`) | 40 | 128 | 51 200 | 10 | 400 |
| laptop σ_rank / signal | 2 / 16 | 32–64 | 256–12 288 | 4 / 12 | 8 / 192 |
| **batch_2** | 40 | 64 | 122 880 | 48 | 1 920 |

At batch_2 sizing, σ_rank ≈ 0.84 (≈ `2.65·√(192/1920)`) ⇒ SNR ≈ 4 against the ~3.5 signal —
clean ranking, so the plasticity (`eta` 0.02 vs 0.005) and optimizer (full vs sep-CMA)
ablation will not be noise-confounded.

---

## 10. Glossary

- **σ_rollout** — std of a single episode's reward sum (high; ~96 % crash rate ⇒ crash-point
  variance dominates).
- **σ_ind** — absolute per-individual noise = `σ_rollout / √(N·F)`.
- **σ_rank** — noise on the *difference* between individuals under CRN; what CMA actually
  consumes. `σ_rank ≤ σ_ind`; CRN drives it toward the chaos+aero floor.
- **signal (Δ)** — noise-free fitness gap between distinct rule-sets, measured co-located on
  shared forests.
- **SNR** — `|Δ| / σ_rank`. ≈ 1 → stall; ≳ 3–4 → clean ranking.
- **chaos floor** — irreducible `σ_rank` from GPU float non-determinism amplified by chaotic
  flight; only `1/√(rollouts)` averaging reduces it.
