# Project Plan: Morphology-Agnostic Flight Control

---

## Vision

Train a single flight controller that:
- Works across a **wide distribution of drone morphologies** (no access to morphology info at inference time)
- Is **fully configurable** via a single config file (architecture, reward, hyperparams, observation space)
- Produces **reproducible, self-documented runs** with rich logging

---

## Constraints & Design Principles

- **No morphology input to the controller**: The actor must not receive genome features. It must learn morphology-invariant flight from state alone (altitude, quaternion, velocity, depth, last actions, speed command).
- **Single config file per run**: Every knob — PPO hyperparams, reward weights, controller architecture, observation flags — must be readable from one YAML/dataclass config. The config is always saved with the checkpoint.
- **Reproducibility**: Every run gets a timestamped folder containing config, checkpoints, logs, and plots. No run should be unreproducible from its folder alone.
- **Modular, non-breaking changes**: Existing `WingedDroneEnv`, `Gen_Env`, and URDF pipeline stay intact. New config layer wraps them.

---

## Task 1 — Infrastructure: Config System & Logging

**Goal**: Make every hyperparameter configurable and every run fully logged.

### Tasks

- [ ] **T1.1 — Unified config dataclass/YAML**
  - Single `RunConfig` covering:
    - PPO hyperparams (currently hardcoded in `get_train_cfg()`)
    - Reward weights (currently hardcoded in `get_cfgs()`)
    - Observation flags (`use_depth`, `use_genome`, `num_sectors`, noise scales)
    - Controller architecture (hidden sizes, LSTM layers/units, action limits)
    - Training settings (`num_envs`, `max_iterations`, `eval_interval`)
    - Catalog settings (`n_urdf`, `catalog_dir`)
  - YAML serialization/deserialization with validation
  - CLI override support (e.g. `--cfg.ppo.lr 3e-4`)

- [ ] **T1.2 — Run folder structure**
  ```
  logs/runs/<YYYY-MM-DD_HH-MM-SS>_<exp_name>/
  ├── config.yaml          # full config snapshot
  ├── catalog.txt          # URDFs used (copy or symlink)
  ├── checkpoints/         # model_*.pt saved every N iters
  ├── tb/                  # TensorBoard event files
  ├── eval/                # per-iteration eval CSVs
  └── plots/               # auto-generated figures
  ```
  - Enforce this layout in both `train.py` and `train_gen.py`
  - On resume, detect existing folder and append `_resumed`

- [ ] **T1.3 — Logging improvements**
  - Per-iteration CSV log: `iter, mean_reward, v_mean, E_tot, progress, crash_rate`
  - Episode termination breakdown (success / wall / angle / timeout / obstacle) as fractions
  - Integrate into existing `RLTrainingLogger` or replace it cleanly

- [ ] **T1.4 — Plotting script**
  - `scripts/plot_run.py <run_folder>` produces:
    - Reward curve (smoothed)
    - v_mean and E_tot over training
    - Termination reason breakdown over time
    - Per-morphology final performance bar chart (if multi-URDF run)
  - All plots saved to `<run_folder>/plots/`

---

## Task 2 — Controller: Morphology-Blind Policy

**Goal**: Ensure the controller has **no access to morphology information** and expose its architecture fully through config.

### Current state
- `ObservationBuilder` optionally appends 15D genome to actor obs (flag: `use_genome`)
- Architecture is hardcoded in `ActorCriticTanh` (2×64 MLP + 1-layer LSTM 64)

### Tasks

- [ ] **T2.1 — Enforce no-genome actor**
  - `use_genome=False` must be the default and enforced for all training runs
  - Critic may optionally still receive genome (it is not deployed at inference)
  - Add assertion in `ObservationBuilder`: if `use_genome=True` and `network=actor`, raise or warn

- [ ] **T2.2 — Configurable architecture**
  - Expose via config:
    - `actor_hidden_sizes: [64, 64]`
    - `critic_hidden_sizes: [64, 64]`
    - `lstm_units: 64`
    - `lstm_layers: 1`
    - `action_limits: {throttle: 1.0, servo: <from_urdf>}`
  - `ActorCriticTanh` reads these from config instead of hardcoding

- [ ] **T2.3 — Asymmetric critic (optional)**
  - Critic can receive privileged info (genome, full state) while actor stays blind
  - Controlled by `critic_privileged: true` flag in config

---

## Task 3 — Pretraining Across Morphology Distribution

**Goal**: Train a single policy on a large catalog of morphologies, optimizing for **robust average performance**.

### Method
- Use existing `Gen_Env` (multi-URDF wrapper) with the no-genome actor (Task 2)
- Sample catalog of N morphologies uniformly from genome space
- Train with PPO for ~4000 iterations

### Tasks

- [ ] **T3.1 — Catalog generation**
  - `scripts/make_catalog.py --n_urdf 64 --out_dir urdf_catalog_64`
  - Saves `catalog.txt` + all URDFs in target dir
  - Seed-controlled for reproducibility

- [ ] **T3.2 — Foundation training run**
  - `scripts/train_foundation.py --cfg configs/foundation.yaml`
  - Uses `Gen_Env` with all catalog URDFs, `use_genome=False`
  - Logs per-morphology reward breakdown each eval interval
  - Saves checkpoint to timestamped run folder (Task 1)

- [ ] **T3.3 — Mid-training evaluation**
  - Every K iterations, evaluate policy on a fixed held-out set of morphologies (not seen during training)
  - Log `v_mean` and `E_tot` per held-out morphology to CSV and TensorBoard

---

## File Change Map

| File | Change |
|------|--------|
| `src/winged_drone_train/train.py` | Read config from YAML, use run folder layout |
| `src/general_policy/train_gen.py` | Same; pass catalog path from config |
| `src/winged_drone_train/perception/obs.py` | Enforce `use_genome=False` for actor by default |
| `src/winged_drone_train/rl/A2C_modified.py` | Accept architecture params from config |
| `src/winged_drone_train/rl/logging.py` | Extend with CSV + termination breakdown |
| `configs/` *(new)* | `train.yaml`, `default.yaml` |
| `scripts/` *(new)* | `make_catalog.py`, `train.py`, `plot_run.py` |

---

## Decisions

- **Asymmetric critic**: controlled by `critic_privileged: bool` in config. When `true`, the critic receives genome features; the actor never does.
- **Catalog size**: set via `catalog.n_urdf` in config. No hardcoded default.
- **Held-out evaluation set**: fixed and seed-controlled — same morphologies across all experiments for fair comparison.
