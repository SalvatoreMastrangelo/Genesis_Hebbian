# Shared-Env Multi-Drone Architecture Implementation Plan

## Summary

Change `MultiDroneEnv` so that N drones **share** the same E environments (same forest per env-slot, static across resets) instead of each drone having independent E environments.

**Config example**: S=4 scenes, N=10 drones per scene, E=512 envs per scene → **2048 physical env-slots** (S×E), each with 10 drones and 1 shared forest.

PPO training batch (`num_envs = S×N×E`) remains unchanged.

---

## Files to Modify

1. **`src/multi_urdf_utils/multi_drone_env.py`** — main changes
2. **`src/WP1/configs/multi_scene.yaml`** — config comments
3. **`src/WP1/config.py`** — docstring for `MultiSceneConfig.E`
4. **`.claude/CONTEXT.md`** — architecture notes

---

## Changes in `multi_drone_env.py`

### 1. Remove per-drone forest fields from `_DroneState.__slots__` (line ~80)

**Before:**
```python
"cylinders_xy", "forest_ids",
```

**After:** Remove these two fields entirely.

---

### 2. Add scene-level forest tracking in `__init__` (after line ~306, the `generate_forests()` call)

**Remove** the per-drone forest init block (lines ~484–489):
```python
# REMOVE THIS BLOCK:
# ds.forest_ids = torch.randint(0, self.total_forests, (self.E,), ...)
# if self.cylinders_array is not None:
#     ds.cylinders_xy = self.cylinders_array[ds.forest_ids, :, :2]
```

**Add** a scene-level block after `generate_forests()` call:
```python
# Scene-level shared forest assignment (all N drones in the same env slot see the same forest)
self.forest_ids = torch.randint(
    0, self.total_forests, (self.E,), device=self.device, dtype=torch.long
)
if self.cylinders_array is not None:
    self.cylinders_xy = self.cylinders_array[self.forest_ids, :, :2]  # (E, T, 2)
else:
    self.cylinders_xy = None
```

Also update the print statement (line ~569) to indicate shared envs:
```python
print(f"[MultiDroneEnv] Building scene: D={self.D} drones × E={self.E} shared envs "
      f"= {self.D * self.E} instances  (all drones in each env slot share the same forest)")
```

---

### 3. Update `_compute_obs()` (line ~821)

**Before:**
```python
if ds.cylinders_xy is not None:
    depth = self.depth_solver.compute_depth(
        base_pos=ds.base_pos,
        base_euler=ds.base_euler,
        cyl_xy_b=ds.cylinders_xy,
    )
```

**After:**
```python
if self.cylinders_xy is not None:
    depth = self.depth_solver.compute_depth(
        base_pos=ds.base_pos,
        base_euler=ds.base_euler,
        cyl_xy_b=self.cylinders_xy,
    )
```

---

### 4. Update `_check_collision()` (line ~990)

**Before:**
```python
if ds.cylinders_xy is None:
    return torch.zeros(self.E, dtype=torch.bool, device=self.device)

# ...
drone_xy = ds.base_pos[:, :2].unsqueeze(1)  # (E, 1, 2)
diff = ds.cylinders_xy - drone_xy            # (E, T, 2)
```

**After:**
```python
if self.cylinders_xy is None:
    return torch.zeros(self.E, dtype=torch.bool, device=self.device)

# ...
drone_xy = ds.base_pos[:, :2].unsqueeze(1)  # (E, 1, 2)
diff = self.cylinders_xy - drone_xy          # (E, T, 2)
```

---

### 5. Update `_reset_drone_idx()` (line ~1036)

**Remove** the entire "Resample forest" block (lines ~1037–1042):
```python
# REMOVE THIS BLOCK:
# new_forest_ids = torch.randint(0, self.total_forests, (n,), ...)
# ds.forest_ids[env_ids] = new_forest_ids
# if self.cylinders_array is not None:
#     ds.cylinders_xy[env_ids] = self.cylinders_array[new_forest_ids, :, :2]
```

The forest now stays **static per env-slot** — it doesn't change on reset. Each drone in a slot always sees the same forest geometry.

Keep the rest of `_reset_drone_idx()` unchanged (pos/vel/episode state randomization).

---

## Changes in `src/WP1/configs/multi_scene.yaml`

Update the comment block (lines ~12–23):

**Before:**
```yaml
# - E: environments per drone per scene.  Total batch = S * N * E.
#   Match S * N * E to your target total num_envs.
```

**After:**
```yaml
# - E: environments per scene (shared by all N drones).  Total batch = S * N * E.
#   Physical Genesis env-slots = S * E  (N drones in each slot, same forest).
#   Match S * N * E to your target total num_envs.
```

---

## Changes in `src/WP1/config.py`

Update the `MultiSceneConfig` dataclass docstring (line ~128):

**Before:**
```python
E: int = 512  # envs per drone per scene
```

**After:**
```python
E: int = 512  # envs per scene (shared by N drones); total batch = S*N*E
```

---

## Changes in `.claude/CONTEXT.md`

### Update the `MultiSceneConfig` section (around line ~121–130):

**Change:**
```
E: int = 512    # envs per drone per scene
```

**To:**
```
E: int = 512    # envs per scene (shared by N drones)
```

### Update the architecture description (around line ~104):

**Change:**
```
Total effective `num_envs = S × N × E`; `training.num_envs` is ignored
```

**To:**
```
Total effective `num_envs = S × N × E`; physical env-slots = S × E
(N drones share each slot, same forest). `training.num_envs` is ignored.
```

### Add a note in the "Key Implementation Notes" section:

Add as item 15:
```
15. **Shared environments in multi-scene**: Each of the N drones in a scene shares
    the same E environment slots. All N drones in env-slot k see the same forest
    (same `cylinders_xy`). Each drone still resets independently (own pos/vel/episode
    state). Forest is static per slot and does not change on reset; diversity comes
    from E different initial forests across slots.
```

---

## Verification Steps

1. **Code inspection**: Confirm `_DroneState` no longer has `cylinders_xy` or `forest_ids`.
   Confirm scene-level `self.forest_ids` and `self.cylinders_xy` are initialized.

2. **Local instantiation test**:
   ```python
   env = MultiDroneEnv(urdf_paths=['path1', 'path2'], num_envs=8, ...)
   print(f"Shared forest_ids shape: {env.forest_ids.shape}")  # should be (8,)
   print(f"Shared cylinders_xy shape: {env.cylinders_xy.shape}")  # should be (8, T, 2)
   ```

3. **Training sanity check**: Run a short training session with:
   ```bash
   python -m WP1.train \
     --cfg src/WP1/configs/foundation.yaml \
     --multi-scene-cfg src/WP1/configs/multi_scene.yaml \
     --cfg.multi_scene.S 2 --cfg.multi_scene.N 2 --cfg.multi_scene.E 16 \
     --cfg.training.num_iterations 2
   ```
   Should complete without crashes or shape mismatches.

4. **Forest consistency check** (optional): After first step, verify that both drones
   in the same env slot have the same `cylinders_xy` array by inspecting env state.

---

## Notes

- PPO runner (`virtual_env.py`) and worker process (`scene_worker_process.py`) need **no changes**.
- Tensor shapes at every interface remain identical (`(S*N*E, *)` for flattened, `(S, N, E, *)` for per-drone).
- Forest diversity is slightly reduced (static per slot instead of re-randomized on reset), but E=512 distinct slots provide plenty of variety.
- If per-reset forest re-randomization is needed in the future, we can add it back with a flag.
