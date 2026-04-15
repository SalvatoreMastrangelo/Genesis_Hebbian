# WP2 Refactoring Plan — K×S Isolated Environments

## Motivating Bug

**Symptom:** individuals that carry a strong WP1 controller receive very poor fitness
scores during WP2 evaluation.  The degradation occurs even when all Hebbian rules are
zeroed out (A=B=C=D=λ=0), meaning the controller itself is not the problem — something
in the evaluation pipeline is corrupting the forward pass.

**Root cause:** `model.memory_a.hidden_states` is a single `(num_layers, K×S, hidden)`
tensor shared across the entire population batch.  Because all K individuals' S
environments are processed through one `model.memory_a(obs)` call, the LSTM hidden state
is never cleanly owned by a single individual.  Concretely:

- At episode start, `memory_a.reset()` sets `hidden_states = None` for the whole batch,
  which is correct.  But because PyTorch initialises a new hidden-state tensor on the
  first forward call using the actual batch size seen at that moment, any mismatch in
  batch size between episodes (e.g. when `P` changes after elite filtering) silently
  re-shapes the hidden tensor, mixing row assignments.
- Mid-episode auto-resets (done environments stepping again) update
  `hidden_states[..., done_idx, :]` in-place.  If the reset indices span multiple
  individuals' slices, cross-individual contamination occurs.
- The shared `last_layer.weight` buffer is overwritten by `W[0]` at the end of every
  `hebbian_update` call (see `hebbian.py` line 190).  When multiple code paths read
  `last_layer.weight` expecting it to reflect individual k, they silently get
  individual 0's weights instead.

The consequence is that the fitness landscape seen by the genetic algorithm is
corrupted: good genomes look bad (their LSTM context is polluted by neighbours), bad
genomes can look artificially good (their context is boosted by a good neighbour's
carry-over state).  Disabling rules makes the Hebbian weight-corruption disappear but
the LSTM contamination remains, which explains why zeroing rules still yields poor
scores.

This refactoring eliminates all three sources by making every environment's mutable
state (`h`, `c`, `W`) privately owned and unreachable from any other environment.

---

## Answers to open questions (recorded for posterity)
- **Q1 (granularity):** K×S networks — one per environment slot, not one per individual.
- **Q2 (model copy):** deep copy once; only ONE shared frozen backbone (not K×S copies).
- **Q3 (performance vs isolation):** isolation is the priority; each env is independently managed.
- **Q4 (evaluate_individual):** remove entirely; will be rewritten later.
- **Q5 (CMA-ES):** same new implementation; the refactoring is in controllers, not evolution strategies.

---

## The Problem in One Line

`model.memory_a.hidden_states` is ONE tensor `(num_layers, K×S, hidden_size)` owned by
the shared RSL-RL `Memory` module.  Reset it and you reset all K×S environments at once.
Read it and you can accidentally couple individual k's environments to individual k+1's.

---

## What "K×S networks" means concretely

We are NOT making K×S copies of the model weights.  The frozen backbone (LSTM + MLP
layers 1–3) is shared — it is never mutated so sharing is safe and memory-efficient.

Each of the K×S environments owns exactly three things:
1. `h_i, c_i` — LSTM hidden state for environment i (shape per slot: `(num_layers, hidden)`)
2. `W_i` — last-layer weight matrix for environment i (shape: `(out, in)`)
3. `ABCD_i, lam_i` — Hebbian rules for environment i (always equal to individual `⌊i/S⌋`'s rules)

Items 1 and 2 are mutable and must never leak across environments.
Item 3 is read-only after initialisation.

---

## Target Architecture

```
IsolatedPopulationActor
│
├── model: ActorCriticTanh   ← ONE deep copy, all params frozen
│   ├── memory_a.rnn         ← used DIRECTLY (bypassing memory_a.hidden_states)
│   ├── actor[:-1]           ← frozen MLP backbone, shared across all K×S envs
│   └── actor[-1]            ← ONLY used for .bias; .weight is NEVER referenced
│                               (IsolatedPopulationActor uses hebbian.W instead)
│
├── hebbian: HebbianLastLayer(num_envs=K*S)
│   ├── W_checkpoint: (out, in)        ← frozen reference, never modified
│   ├── W:            (K*S, out, in)   ← one matrix per environment
│   ├── A/B/C/D/lam:  (K*S, out, in)  ← pre-expanded from K rule sets (each replicated S times)
│   └── eta: scalar or (K*S, out, in)
│
├── _h: (num_layers, K*S, hidden_size) ← LSTM cell states, owned here NOT in model
└── _c: (num_layers, K*S, hidden_size) ← LSTM hidden states, owned here NOT in model
```

---

## Forward Pass (`IsolatedPopulationActor.act`)

```
obs: (K*S, obs_dim)
│
├── 1. rnn = model.memory_a.rnn   [bypass Memory.hidden_states entirely]
│        rnn_out, (h_new, c_new) = rnn(obs.unsqueeze(0), (_h, _c))
│        inp = rnn_out.squeeze(0)          # (K*S, lstm_out)
│        _h, _c = h_new, c_new             # update owned states
│
├── 2. x = model.actor[:-1](inp)           # (K*S, 64)  shared frozen MLP
│
├── 3. y = einsum("ni,noi->no", x, hebbian.W) + bias  # (K*S, 7)  per-env weights
│
├── 4. hebbian.hebbian_update(x, y)        # updates hebbian.W in-place, no write-back
│
├── 5. sample from N(y, std) or use y      # stochastic flag
│
└── 6. tanh → scale                        # reproduce WP1 pipeline
```

---

## Reset Semantics

| Operation | What resets |
|---|---|
| `reset_episode()` | ALL K×S: zero `_h`, `_c`; copy `W_checkpoint` into every row of `W` |
| `reset_individual(k)` | Envs `[k*S, (k+1)*S)`: zero their `_h`, `_c`, reset their `W` rows |

`reset_individual` enables future mid-generation individual replacement without a full
episode restart.

---

## Changes by File

### `hebbian.py`

**Goal:** one class, no write-back hack, unified batched path.

**Remove:** `BatchedHebbianLastLayer` (merge into `HebbianLastLayer`).

**Modify `HebbianLastLayer`:**

1. Constructor — change `linear_layer: nn.Linear` to `W_checkpoint: Tensor`:
   ```python
   # Old:  HebbianLastLayer(linear_layer, hebbian_rules, ..., num_envs=1)
   # New:  HebbianLastLayer(W_checkpoint, hebbian_rules, ..., num_envs)
   ```
   `attach_hebbian()` in `frozen_actor.py` does the extraction so callers are unchanged.

2. ABCD input — accept either `(out, in)` or `(num_envs, out, in)`:
   - If `(out, in)`, expand to `(num_envs, out, in)` at construction time.
   - This replaces the `pop_size × slice_size` pre-expansion logic that was in
     `BatchedHebbianLastLayer.__init__`.

3. Remove `num_envs == 1` vs `num_envs > 1` branching in `hebbian_update` — always
   use the batched einsum path.  The single-env case is just `num_envs=1`.

4. Remove the `self.layer.weight.data.copy_(self.W[0])` write-back on lines 156 and 190.
   The layer weight is never read back after initialisation; `IsolatedPopulationActor`
   uses `hebbian.W` directly.

5. Add `reset_weights_individual(k: int, slice_size: int) -> None`:
   ```python
   start, end = k * slice_size, (k + 1) * slice_size
   self.W[start:end].copy_(self.W_checkpoint.unsqueeze(0))
   ```

---

### `frozen_actor.py`

**Remove:** `BatchedHebbianActorWrapper` entirely.

**Remove:** `HebbianActorWrapper` entirely (including `_act_single_env` and `_act_multi_env`).

**Keep (unchanged):** `load_frozen_actor`, `_build_actor_critic`.

**Modify `attach_hebbian`:**
- Extract `W_checkpoint` from `last_layer.weight.data` and pass as tensor to
  `HebbianLastLayer` — no longer passes `last_layer` itself.
- Signature stays the same from the caller's perspective.

**Add `IsolatedPopulationActor`:**

```python
class IsolatedPopulationActor:
    """K*S isolated environments, one frozen backbone, explicit LSTM states.

    Parameters
    ----------
    model : nn.Module
        Frozen ActorCriticTanh (deep copy — must not be shared).
    hebbian : HebbianLastLayer
        num_envs=K*S, ABCD pre-expanded from K individual rule sets.
    K : int
        Population size (number of individuals).
    S : int
        Environments per individual.
    stochastic : bool
        Sample from N(y, std) if True; use mean otherwise.
    """

    def __init__(self, model, hebbian, K, S, stochastic=True): ...

    def reset_episode(self, device): ...
        # zero _h, _c; hebbian.reset_weights()

    def reset_individual(self, k, device): ...
        # zero _h/c slice for k; hebbian.reset_weights_individual(k, S)

    @torch.no_grad()
    def act(self, obs): ...  # obs: (K*S, obs_dim) → (K*S, num_actions)
```

**Add `build_isolated_population_actor` factory:**

```python
def build_isolated_population_actor(
    checkpoint_path,
    wp1_cfg_path,
    hebbian_rules_per_individual,   # list of K rule dicts
    cfg,
    K, S,
    device,
    stochastic=True,
) -> IsolatedPopulationActor:
    """Load checkpoint once, deep-copy model, pre-expand ABCD rules, build actor."""
    model, last_layer, num_actions, hidden_dim = load_frozen_actor(
        checkpoint_path, wp1_cfg_path, device
    )
    model = copy.deepcopy(model)  # isolate from the template

    # Pre-expand K rule sets to K*S by replicating each S times
    expanded_rules = {
        key: torch.stack([
            rules[key]
            for rules in hebbian_rules_per_individual
            for _ in range(S)
        ])
        for key in ("A", "B", "C", "D", "lam")
    }

    hebbian = attach_hebbian(last_layer, expanded_rules, cfg, device, num_envs=K*S)
    return IsolatedPopulationActor(model, hebbian, K, S, stochastic)
```

---

### `evaluate.py`

**Remove:** `evaluate_individual`, `evaluate_population_serial`.

**Rewrite `evaluate_population_batched`:**

```python
# Pseudocode for the new evaluate_population_batched

invalid = [ind for ind in population if not ind.fitness.valid]
K = len(invalid)
S = cfg.evaluation.num_eval_envs // K

# Decode genomes
hebbian_rules = [decode_individual_rules(ind, cfg) for ind in invalid]
morph_genomes  = [decode_individual_morph(ind, cfg) for ind in invalid]

# Build actor — ONE model copy, K*S isolated envs
actor = build_isolated_population_actor(
    cfg.checkpoint_path, cfg.checkpoint_config_path,
    hebbian_rules, cfg, K, S, device=cfg.device,
    stochastic=cfg.evaluation.stochastic,
)

# Build environment (PopGenEnv — unchanged)
env = _build_env(morph_genomes, cfg, wp1_cfg, device)

# Episode loop
for ep in range(cfg.evaluation.num_eval_episodes):
    actor.reset_episode(device=cfg.device)
    ep_metrics = _rollout_episode(env, actor, cfg.device)
    # ep_metrics values are (K*S,); reshape to (K, S) and mean over S
    ...

# Assign fitness per individual
for k, ind in enumerate(invalid):
    ind.fitness.values = compute_fitness(per_individual_metrics[k], cfg)
```

**`_rollout_episode` is unchanged** — it only calls `actor.act(obs)` and
`actor.reset_episode()` which are both present on `IsolatedPopulationActor`.

**Rewrite `evaluate_population_cma_batched`** the same way — it currently uses
`BatchedHebbianActorWrapper`; replace the actor construction with
`build_isolated_population_actor`.  The rollout loop and metric collection are the same.

---

## Implementation Order

Each step compiles independently; test before moving to the next.

| Step | File | What changes |
|---|---|---|
| 1 | `hebbian.py` | Unify `HebbianLastLayer` (accept `W_checkpoint` tensor, batched-only path, remove write-back, add `reset_weights_individual`); delete `BatchedHebbianLastLayer` |
| 2 | `frozen_actor.py` | Add `IsolatedPopulationActor` + `build_isolated_population_actor`; update `attach_hebbian` to pass `W_checkpoint` tensor; delete `BatchedHebbianActorWrapper` and `HebbianActorWrapper` |
| 3 | `evaluate.py` | Rewrite `evaluate_population_batched` and `evaluate_population_cma_batched` using new actor; delete `evaluate_individual` and `evaluate_population_serial` |

---

## Memory Budget

| Tensor | Shape | Size (fp32, K=40, S=256) |
|---|---|---|
| Model weights (ONE copy) | ~5 MB | 5 MB |
| `_h` + `_c` (LSTM states) | 2 × (1, K×S, 128) | 2 × 40 × 256 × 128 × 4 B = ~10 MB |
| `hebbian.W` | (K×S, 7, 64) | 40 × 256 × 7 × 64 × 4 B = ~18 MB |
| `hebbian.ABCD+lam` | 5 × (K×S, 7, 64) | ~90 MB |

Total overhead vs current: near zero — the LSTM states and W matrices existed before,
just organised differently.  The ONE deep-copied model is new but costs ~5 MB.

---

## Key Invariants to Verify After Implementation

1. `actor._h[:, k*S:(k+1)*S, :]` after `reset_individual(k)` must be all zeros.
2. `actor.hebbian.W[k*S:(k+1)*S]` after `reset_individual(k)` must equal `W_checkpoint`.
3. An action computed for env `i` must be a function of ONLY `obs[i]`, `_h[:, i, :]`,
   `_c[:, i, :]`, and `hebbian.W[i]` — nothing from env `j ≠ i` should appear.
4. `model.memory_a.hidden_states` is NEVER read or written after model construction
   (bypass confirmed: always call `model.memory_a.rnn(...)` directly).
