# WP2 Outer Loop — Co-evolution of URDFs and Hebbian Rules

## Overview
WP2 currently runs a single CMA-ES loop that evolves Hebbian rules for a fixed
catalog of URDFs over a set of forests. This module adds an **outer NSGA-II
loop** that co-evolves the URDF morphologies themselves, with the existing
CMA-ES loop nested as the inner loop that evolves the Hebbian rules for the
current URDF population.

---

## Pipeline

1. **Gen 0 URDF population.** Sample `P-1` random genomes in `[0,1]^15` (same
   `Chromosome_Drone` used today) plus 1 seed slot holding the standard
   WP1-trained drone genome, giving a starting population of size `P`.
2. **Inner CMA-ES loop.** Run the existing CMA-ES rule-evolution loop on the
   current URDF population for `inner_generations` generations. Each CMA-ES
   generation produces `λ` individuals, and **each individual is itself a
   complete set of Hebbian rules** — so within one inner generation there are
   `λ` distinct candidate rule sets being evaluated, not one. The "sharing
   across URDFs" happens *inside* the evaluation of a single individual: that
   one rule set is applied to every URDF, and the individual's scalar fitness
   is the average performance across all URDFs (as today).
3. **NSGA-II evaluation.** After the inner loop finishes, re-run the same
   environments once more with the **best** Hebbian rules from the inner loop,
   partitioning environments across URDFs (see *NSGA-II Evaluation* below) to
   get per-URDF objective values.
4. **NSGA-II selection + variation.** Apply standard NSGA-II
   (non-dominated sort + crowding distance) with SBX crossover and polynomial
   mutation over `[0,1]^15` to produce the next URDF population of size `P`.
5. **Carry rules forward.** The **best-fitness** Hebbian controller from this
   outer generation's inner loop is carried over as the initialization point
   for the next outer generation's inner loop. The CMA-ES state itself
   (covariance, step size) is **restarted**, not preserved.
6. **Repeat** steps 2–5 for `outer_generations` generations.

At the start of each new inner loop (outer gen ≥ 1), the carried-over rules
are **re-evaluated** on the new URDF population before CMA-ES updates begin,
so the algorithm has an honest fitness baseline on the changed bodies.

---

## Control: Rule Sharing Within an Individual

The inner CMA-ES loop evaluates `λ` individuals per generation, and each
individual **is** a complete candidate rule set. So an inner generation
holds `λ` distinct rule sets, not one.

What is shared is the mapping from a single individual to its fitness:

- For one CMA-ES individual (one rule set `θ`):
  - `θ` is applied identically to every URDF in the current outer-gen
    population (each URDF still runs as its own body, on its own envs).
  - The individual's scalar fitness = **average performance of `θ` across
    all URDFs**.
- CMA-ES then ranks the `λ` individuals by this scalar and updates its
  distribution as usual.

This is what drives generality: an individual only wins if its one rule set
works well across the whole morphology population, not just on one body.

At the end of the inner loop, `best rules` = the single individual (rule
set) with the highest across-URDF mean fitness over the whole inner run.

---

## Morphology: Per-URDF Fitness for NSGA-II

Inner-loop fitness is a single scalar (average across URDFs). NSGA-II needs
**per-URDF** objectives. These are obtained from the dedicated NSGA-II
evaluation step using the best inner-loop rules.

**Objectives (minimization / maximization conventions handled in code):**
- `progress_m` — distance traveled (meters), to maximize.
- `cost_of_transport` — energy per unit distance, to minimize.

(Two objectives is the minimum for NSGA-II to produce a meaningful Pareto
front. Additional objectives can be added later without restructuring the
loop.)

All URDFs are assumed feasible — no feasibility filter or penalty.

---

## NSGA-II Evaluation

After inner loop convergence, one final rollout assigns per-URDF fitness:
- Let `E = num_eval_envs` (total env budget; see *Environment Budget*).
- Let `U_outer` = outer-loop population size `P`.
- Each URDF is evaluated on `E / U_outer` environments (integer division;
  remainder distributed round-robin to the first few URDFs).
- Environments are drawn from the same forest distribution used in the inner
  loop to keep the comparison fair.

The outputs of this step are `P` tuples `(progress_m, cost_of_transport)` fed
to NSGA-II selection.

---

## Baseline

- Baseline is still run in the inner loop at every inner generation, as today.
- Baseline uses the **same URDFs** as the current outer generation's inner
  loop and the **same environments** — so when the URDF population shifts
  across outer generations, the baseline shifts with it. This keeps the
  inner-loop comparison fair but means the baseline curve across outer gens
  is not directly comparable across outer gens (expected; it's a per-gen
  reference, not a global one).

---

## Environment Budget

- A single `num_eval_envs` in the outer-loop config sets the **total** env
  count for any evaluation pass (inner generation rollout, baseline rollout,
  NSGA-II evaluation), matching the existing WP2 convention.
- The inner loop distributes `num_eval_envs` across its URDFs exactly as it
  does today.
- The NSGA-II evaluation distributes `num_eval_envs` across the `P` URDFs
  (`E / P` per URDF).
- Forests are **re-sampled every inner generation** (current behavior
  preserved); this intentionally injects variance to pressure rule
  robustness. NSGA-II evaluation samples one fresh forest set per outer
  generation.

---

## NSGA-II Operators

Standard DEAP NSGA-II:
- **Selection:** non-dominated sorting + crowding distance (`tools.selNSGA2`).
- **Crossover:** Simulated Binary Crossover (SBX) on `[0,1]^15`.
- **Mutation:** polynomial mutation on `[0,1]^15`.
- Crossover/mutation probabilities and SBX/PM `eta` parameters are
  configurable (see *Config*).

---

## Replicability

- One configuration file drives the entire outer-loop experiment, referencing
  the baseline controller and WP1 configs by path (same pattern as current
  WP2).
- A single master seed seeds NumPy, Python `random`, PyTorch, DEAP, CMA-ES,
  and the Genesis environment RNGs deterministically.
- The config pins: outer generations, inner generations, outer population
  size, total num_eval_envs, and NSGA-II crossover/mutation parameters.

---

## Checkpointing & Logging

Checkpoints written at:
- **Every inner generation** (CMA-ES mean, covariance, step size, best rules
  so far, per-URDF fitness — as today).
- **Every outer generation** (URDF population genomes, per-URDF NSGA-II
  objectives, Pareto front, best Hebbian rules carried forward, RNG state).

This allows resuming from either granularity.

---

## Config (sketch)

```yaml
outer_loop:
  outer_generations: 20            # user-configurable, default 20
  inner_generations: 50            # user-configurable, default 50 (matches current WP2 default)
  population_size: 16              # NSGA-II pop; default 16 (divides cleanly into common env counts)
  num_eval_envs: 256               # total env budget per evaluation pass
  seed: 0

  nsga2:
    crossover_prob: 0.9            # standard NSGA-II default
    mutation_prob: 0.0667          # 1 / genome_dim = 1/15
    sbx_eta: 15                    # DEAP default
    pm_eta: 20                     # DEAP default

  objectives:
    - name: progress_m
      direction: maximize
    - name: cost_of_transport
      direction: minimize

  seed_standard_drone: true        # include WP1-trained genome as 1 of P slots at gen 0

  # paths to reused configs (same pattern as current WP2)
  inner_cma_cfg: src/WP2/configs/cma_es_rules_only.yaml
  wp1_cfg: src/WP1/configs/foundation.yaml
  baseline_ckpt: <path>
```

Defaults above are starting suggestions — every one of these is exposed in
the config.

---

## Module Layout

```
src/WP2_Outer_Loop/
    PLAN.md               (this file)
    run.py                # CLI entry: python -m WP2_Outer_Loop.run --cfg ...
    config.py             # dataclass config + YAML loader, seed plumbing
    outer_loop.py         # main NSGA-II loop orchestration
    nsga2.py              # DEAP wiring: operators, selection, variation
    urdf_genome.py        # Chromosome_Drone bridge + per-URDF objective extraction
    evaluation.py         # NSGA-II evaluation pass (distributes E across P URDFs)
    checkpoint.py         # outer-gen checkpoint I/O
    configs/
        outer_loop_default.yaml
```

The inner CMA-ES loop is called as a library from `outer_loop.py` — the
existing `src/WP2/run.py` entry point is not duplicated; its core is
refactored just enough to accept a URDF-population argument and return the
best rules + per-URDF fitness.

---

## Open Work Items (tracked during implementation)

- Refactor inner CMA-ES entry to accept a URDF-population argument and
  return `(best_rules, per_urdf_fitness, cma_log)` — minimal invasive change.
- Extend per-URDF rollout metrics to emit `progress_m` and
  `cost_of_transport` alongside existing fitness signals.
- Wire DEAP NSGA-II with SBX + polynomial mutation over `[0,1]^15`.
- Implement re-evaluation of carried-over rules at start of each non-zero
  outer generation's inner loop.
- Extend checkpointing to outer-gen granularity with RNG state capture.
