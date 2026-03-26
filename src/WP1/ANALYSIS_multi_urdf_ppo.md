# PPO Multi-URDF Analysis: Observation Independence & Batch Size Impact

## ✅ Observation Independence: CORRECT

Each drone within the same Genesis scene gets **truly independent observations**:

### Per-drone observation components:
- **Position** (`base_pos[i]`) - unique per drone per environment
- **Orientation** (`base_quat[i]`) - unique per drone per environment
- **Velocity** (`base_lin_vel[i]`) - unique per drone per environment
- **Depth perception** - computed from drone's own position (line 823-827 in `multi_drone_env.py`)
- **Previous actions** - drone's own action history
- **Command signals** - drone's own target

### Flow:
1. MultiDroneEnv.step() processes each drone independently: `for i, ds in enumerate(self.drones)`
2. Each drone calls `_compute_obs(i, ds)` which builds observations from its own state
3. Observations returned as `(D, E, obs_dim)` = `(10, 512, 36)` per scene
4. Flattened to `(N*E, obs_dim)` = `(5120, 36)` in worker (line 197 `scene_worker_process.py`)
5. No observation mixing - just reshape, observations remain independent

**Result:** Each of the 5,120 "environments" per scene has fully independent observations.

---

## ⚠️  PPO Batch Size & Learning Rate: POTENTIAL ISSUES

### Comparison: Original vs Multi-URDF Setup

| Aspect | Original (foundation.yaml) | Multi-URDF (multi_scene.yaml) |
|--------|---------------------------|-------------------------------|
| **Config** | S=1, N=1, E=5120 | S=4, N=10, E=512 |
| **Reported num_envs** | 5,120 | S×N×E = 20,480 |
| **Actual Genesis instances** | 5,120 | S×E = 2,048 |
| **Morphologies** | 1 | 10 per scene |
| **Batch/iteration** | 5,120 × 25 = 128,000 | 20,480 × 25 = 512,000 |
| **Batch/mini-batch** | 128,000 ÷ 32 = 4,000 | 512,000 ÷ 32 = 16,000 |
| **Learning rate** | 1e-4 | 1e-4 ❌ NOT ADJUSTED |
| **Num epochs** | 2 | 2 |

### The Problem:

**Batch size increased 4x without learning rate adjustment**

- Per-mini-batch sample count: 4,000 → 16,000 samples (4x larger)
- Gradient magnitude scales with batch size
- With same learning rate, steps are now 4x larger relative to gradient magnitude
- Risk: **Overly large policy updates → instability or divergence**

### Is it actually 4x more data?

**No** — there's a subtlety:

The 512,000 "transitions" come from **2,048 independent physics instances**, not 5,120:
- Each physics step produces 10 drone observations (one per morphology)
- Physics is synchronized: all 10 drones step together, share forest obstacles
- The 10 morphologies in one scene are **temporally correlated** (same timestep)
- They are **NOT independent samples** in the statistical sense

Compared to original:
- 128,000 transitions from 5,120 truly independent physics instances
- Each instance is from a different Genesis environment

**Effective independence:**
- Original: 128,000 / 5,120 = 25 transitions per physics instance ✓
- Multi-URDF: 512,000 / 2,048 = 250 transitions per physics instance (per set of 10 drones)

---

## 🎯 Recommendations

### 1. **Adjust Learning Rate**

Current: `learning_rate: 1.0e-4`

**Option A - Conservative (empirical scaling):**
```yaml
learning_rate: 2.5e-4  # Scale by ~2.5x (batch size factor)
```

**Option B - Let adaptive schedule learn it:**
Keep current LR but closely monitor:
- `desired_kl: 0.006` (target KL divergence)
- If KL stays below desired → LR will increase (adaptive schedule)
- If KL overshoots → LR will decrease

### 2. **Monitor These Metrics**

Add to your training logs:
```python
# In your CSV logging or TensorBoard:
- Mean gradient norm per iteration
- KL divergence (check vs desired_kl=0.006)
- Policy loss magnitude
- Value function loss magnitude
- Min/max/mean of policy updates
```

**Expected behavior:**
- With adaptive schedule, KL should oscillate around 0.006
- If KL >> 0.006, LR is being reduced (too large steps)
- If KL << 0.006, LR is being increased (room for larger steps)

### 3. **Consider Batch Composition**

Since 10 morphologies share physics:
```python
# Option: Shuffle morphologies across scenes between iterations
# Currently: morphology assignment is fixed (round-robin)
# Alternative: randomly assign URDFs per scene to increase diversity
```

### 4. **Validate Convergence**

Compare against baseline:
```bash
# Run original foundation.yaml (without multi_scene)
python -m WP1.train --cfg src/WP1/configs/foundation.yaml

# Compare learning curves:
# - Reward per iteration
# - Convergence speed
# - Final performance
# - Stability
```

---

## 📊 What PPO Actually Sees

### Data structure per iteration:
```
Total batch = 512,000 transitions
├── Scene 0 (2,048 steps × 10 morphologies × 25 steps)
├── Scene 1 (2,048 steps × 10 morphologies × 25 steps)
├── Scene 2 (2,048 steps × 10 morphologies × 25 steps)
└── Scene 3 (2,048 steps × 10 morphologies × 25 steps)

PPO mini-batch (after shuffling): 16,000 samples
└── Mix of:
    - Different morphologies
    - Different environments (within each scene)
    - Different scenes
```

### Key insight:
- Morphology diversity is GOOD (policy learns multiple morphologies)
- But reduced environment diversity (2,048 vs 5,120 distinct physics)
- PPO's performance depends on which factor dominates

---

## ✅ Things Working Correctly

1. ✅ Observations are independent per-drone (verified)
2. ✅ Each drone sees its own state, depth, actions
3. ✅ Drones don't see observations from other drones
4. ✅ No observation mixing or cross-contamination
5. ✅ Multi-scene parallelization is sound

## ⚠️ Things to Monitor

1. ⚠️ Learning rate may be undersized (larger batch needs larger LR)
2. ⚠️ Effective environment diversity is lower (2K vs 5K physics instances)
3. ⚠️ Temporal correlation within scenes (10 drones share physics timestep)
4. ⚠️ Convergence speed and stability may differ from original