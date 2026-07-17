# WP2 Outer Loop — persistent CMA-ES + NSGA-II URDF co-evolution

Multi-objective evolution of drone morphology (NSGA-II over the 15-D
`Chromosome_Drone` genome) wrapped around the WP2 Hebbian-rules CMA-ES,
with **fitness** (WP1 reward sum) and **cost of transport** as objectives.

## How it works

`NSGA2MorphCMAES` (`nsga_cma.py`) subclasses the inner loop's
`HebbianCMAES` and re-purposes its periodic URDF-refresh machinery
(the same hook the `catalog.mutate` experiments use):

```
ONE persistent CMA-ES over Hebbian rules (mean+covariance never reset)
│
├─ every generation: evaluate the CMA population on N URDFs
│    (one Genesis scene each) → per-URDF fitness/CoT harvested for free
│    from the per_urdf_* matrices of evaluate_population_multi_urdf
│
└─ every `catalog.refresh_urdfs_every` generations ("outer generation"):
     1. score each URDF = phase-mean over the top `outer.score_top_frac`
        CMA individuals per generation
     2. archive scores  →  Pareto front / hypervolume tracking
     3. NSGA-II: keep `outer.n_elites` survivors (re-scored next phase),
        breed the rest via binary tournament + SBX + polynomial mutation
     4. materialize new URDFs, rebuild the eval env ONCE,
        re-inflate CMA sigma (`cmaes.sigma_reinflate`)
```

Cost per outer generation = one env rebuild — the same as the existing
inner-loop mutation experiments, and ~2×P Genesis scene builds cheaper
than the legacy nested design.

Notes:
- Per-URDF CoT is a ratio-of-sums (total energy / total weight·distance),
  so morphologies that crash immediately get a *large* CoT instead of a
  spuriously perfect 0.
- URDFs are only compared within a phase (same rules, forests, speeds),
  so the NSGA-II ranking is internally consistent; elites are re-scored
  every phase, so no stale objective survives.

## Run

The config is a SINGLE file: a plain inner-loop `HebbianEvolutionConfig`
YAML (same layout as `src/WP2/configs/cma_es_rules_only.yaml`) plus one
`outer:` section with the NSGA-II knobs. All sizing lives in the standard
inner fields — the run behaves exactly like an inner-loop multi-URDF run
with periodic URDF refresh, except the refresh is an NSGA-II update:

| Outer-loop quantity | Config field |
|---------------------|--------------|
| NSGA-II population size N | `catalog.num_urdfs` |
| CMA generations per phase | `catalog.refresh_urdfs_every` |
| total CMA generations | `evolution.num_generations` |
| env budget (N × H × F) | `evaluation.num_eval_envs` |
| sigma kick on morphology change | `cmaes.sigma_reinflate` |
| elites / scoring / operators / objectives | `outer:` section |

```bash
PYTHONPATH=src python -m WP2_Outer_Loop.run \
    --cfg src/WP2_Outer_Loop/configs/outer_nsga_default.yaml \
    --cfg.checkpoint_path        <WP1 actor .pt> \
    --cfg.checkpoint_config_path <WP1 config.yaml>

# ONE override namespace — inner and outer fields alike:
#   --cfg.cmaes.population_size 16  --cfg.catalog.num_urdfs 4
#   --cfg.outer.n_elites 3          --cfg.outer.score_top_frac 0.25
```

Local demo (4 URDFs, LSTM-15 checkpoint, fits an 8 GB laptop GPU):

```bash
PYTHONPATH=src python -m WP2_Outer_Loop.run \
    --cfg src/WP2_Outer_Loop/configs/outer_nsga_demo_lstm15.yaml
```

Env budget rule: `evaluation.num_eval_envs ≥ catalog.num_urdfs ×
cmaes.population_size` (each (URDF, individual) pair needs ≥ 1 forest).

## Outputs

On top of the standard `HebbianCMAES` run directory:

| Path | Content |
|------|---------|
| `results/outer_per_urdf_per_gen.csv` | per inner gen × URDF diagnostics |
| `results/outer_population.csv` | per outer gen × URDF phase-mean objectives + genomes (Pareto archive) |
| `outer/gen_XXX/{genomes,objectives}.npy`, `outer/pareto_archive.pkl` | phase snapshots |
| `plots/pareto_front_evolution.png` | objective-space scatter + per-gen fronts + cumulative front |
| `plots/pareto_hypervolume.png` | per-gen & cumulative front hypervolume |
| `plots/outer_metrics_evolution.png` | per-URDF metric curves (inner-loop plot style) |

All standard inner-loop plots (`metrics_evolution`, `cma_state`, …) are
generated too — the run *is* a `HebbianCMAES` run.

## Files

| File | Purpose |
|------|---------|
| `nsga_cma.py` | `NSGA2MorphCMAES` + tournament variation |
| `config.py` | `OuterNSGA2Config` (single-file: inner config + `outer:` section) + `OuterLoopConfig` (legacy) |
| `run.py` | entry point |
| `pareto_plots.py` | outer-loop plots |
| `nsga2.py` | DEAP toolbox helpers |
| `urdf_population.py` | genome → URDF materialisation |
| `configs/outer_nsga_default.yaml` | single-file default config (set checkpoint paths) |
| `configs/outer_nsga_demo_lstm15.yaml` | single-file local demo (4 URDFs, LSTM-15, 8 GB GPU) |
| `outer_loop.py`, `evaluation.py`, `legacy_run.py` | **deprecated** legacy nested implementation (see `POSSIBLE_FIXES.md`) |

The full resolved config (including `outer:`) is dumped to
`reproducibility/config.yaml` in the run dir — that file is itself a valid
single-file config to reproduce the run.

Limitations: `--resume` is not supported; multi-URDF path only. Unknown YAML
keys are silently ignored (WP2 config convention) — double-check spelling
when a setting seems to have no effect.
