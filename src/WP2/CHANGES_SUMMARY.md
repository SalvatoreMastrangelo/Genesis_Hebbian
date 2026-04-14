# WP2 CMA-ES Updates — WP1 Reward Integration

## Summary

All fitness metrics have been **replaced with WP1's weighted reward combination**. The evolution now optimizes a single objective: cumulative WP1 reward (progress, crash, energy, smoothness components combined with their weights from `wp1_cfg.reward`).

## What Changed

### 1. **RulesEvolutionEnv** (`src/WP2/rules_evolution.py`)

#### Added reward computation:
- **`_compute_wp1_reward()`** — Calculates WP1 reward components per timestep:
  - Progress: Gaussian reward for forward velocity tracking
  - Energy: Penalty for power consumption
  - Smoothness: Penalty for action changes  
  - Crash: Penalty for collisions
  - Uses the same weights from `wp1_cfg.reward` as WP1 training

#### Updated accumulators:
- Added `_cumulative_reward` to track total reward per individual
- Added `_last_actions` buffer for smoothness penalty computation
- Store `reward_scales` from WP1 config during init

#### Updated `evaluate_population()`:
- **Removed**: `fitness_key` parameter (no longer supports mean_velocity, progress, mean_energy choices)
- **Returns**: Cumulative WP1 reward instead
- **Simplified**: Single metric = WP1's weighted combination

#### Renamed `evaluate_population_full()` → `evaluate_population_with_aux_metrics()`:
- For analysis only (returns auxiliary metrics: velocity, progress, energy)
- Primary fitness is still WP1 reward

### 2. **CMA-ES Integration** (`src/WP2/cmaes_integration.py`)

#### Updated `CMAESConfig`:
- **Removed**: `fitness_key` field

#### Updated `CMAESHebbianPopulationEvaluator`:
- **Removed**: `fitness_key` parameter from `__init__`
- **Simplified**: `evaluate()` method — no longer takes fitness_key, just uses WP1 reward
- Updated docstring to emphasize WP1 reward as the fitness function

#### Updated `run_cmaes_evolution()`:
- **Removed**: `fitness_key` parameter
- Updated logging to state: "fitness: WP1 weighted reward (progress, crash, energy, smoothness)"
- Removed fitness_key from evaluator init

### 3. **CLI** (`src/WP2/run_cmaes.py`)

- **Removed**: `--fitness-key` argument
- Updated logging to show fitness is WP1 weighted reward
- Removed `fitness_key` from `run_cmaes_evolution()` call

### 4. **Documentation** (`src/WP2/README_CMAES.md`)

- Updated quick-start examples (removed `--fitness-key`)
- Clarified fitness computation as WP1's weighted reward
- Updated comparison table to show WP1 reward fitness
- Simplified documentation (no longer explaining multiple metrics)

## Key Design Decisions

### Why WP1 Reward?

1. **Consistency**: Uses the exact same reward function WP1 was trained on
2. **Single objective**: Simpler optimization (CMA-ES excels at 1D)
3. **Aligned incentives**: Hebbian rules now optimize for the same objectives as the base policy

### Reward Components

The fitness function combines:

```
fitness = (progress  * w_progress +
           -crash    * w_crash +
           -energy   * w_energy +
           -smoothness * w_smoothness + ...)
```

Where weights (`w_*`) come from `wp1_cfg.reward` (same as WP1 training).

### Computation Details

- Rewards computed at each physics timestep in `RulesEvolutionEnv.step()`
- Accumulated over the full episode
- Averaged across environment scenarios (multiple forests)
- Averaged across `n_episodes` independent evaluations
- Used directly by CMA-ES as the fitness value

## Testing the Changes

```bash
# Run evolution with WP1's reward function
python -m WP2.run_cmaes \
    --wp1-checkpoint logs/runs/<timestamp>_<exp>/model.pt \
    --wp1-cfg src/WP1/configs/foundation.yaml \
    --pop-size 64 \
    --num-generations 100

# Check output — should show:
# [INFO] Fitness: WP1 weighted reward (progress, crash, energy, smoothness)
# [RulesEvolutionEnv] episode 1/1  mean_reward=<value>  steps_taken=<steps>
```

## Backward Compatibility

- ❌ **Breaking**: Old code calling `evaluate_population(fitness_key="mean_velocity")` will fail
- ⚠️ **Deprecated**: `evaluate_population_full()` renamed to `evaluate_population_with_aux_metrics()`
- ✅ **Preserved**: All reward scale configurations from WP1 are automatically used

## Implementation Notes

### Reward Scaling
- Rewards are scaled by `dt * 50.0` (matching WP1 convention)
- Per-component scaling applied via `reward_scales` from config
- Crash penalty only triggers on episode termination (collision detection)

### Smoothness Penalty
- Computed from action differences: `sum((action_t - action_t-1)^2)`
- Requires tracking `_last_actions` across timesteps
- Reset to None at episode start (no penalty for first action)

### Energy Penalty
- Uses instantaneous power: `ds.power` from drone state
- Accessed via `self._env.drones[d].power`

## Future Extensions

- [ ] Multi-objective CMA-ES using WP1 reward + morphology metrics
- [ ] Adaptive weight scheduling (adjust reward_scales during evolution)
- [ ] Visualization of reward components during evolution
- [ ] Per-generation reward breakdown (how much from progress vs crash penalty)

## Files Modified

| File | Changes |
|------|---------|
| `rules_evolution.py` | Added WP1 reward computation; removed fitness_key parameter |
| `cmaes_integration.py` | Removed fitness_key; simplified to WP1 reward only |
| `run_cmaes.py` | Removed --fitness-key CLI argument |
| `README_CMAES.md` | Updated docs; removed multi-metric explanation |

