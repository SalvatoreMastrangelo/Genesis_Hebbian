## Quick Installation

Install **PyTorch** first following the official instructions, then install Genesis:
```bash
pip install genesis-world  # Requires Python>=3.10,<3.14
```

Clone this repo and install it in editable mode:
```bash
git clone https://github.com/andrevic1/Genesis.git
cd Genesis
git checkout <your-branch>
pip install -e ".[dev]"
```

## How to fly the winged drone from the keyboard

Run:
```bash
python src/winged_drone_fly.py
```

This script is a complete, ready-to-use example that loads a winged drone URDF, attaches it to the AeroSolver, and lets you fly it in real time with the keyboard.

### What URDF is loaded (and from where)

The URDF is chosen inside `src/winged_drone_fly.py` via `DRONE_NAME`:

- `DRONE_NAME = "mydrone"`  
  URDF path: `genesis/assets/urdf/mydrone/[0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4, 0.2, 2, 0, 2, 2.5, 3, 4, 16].urdf`
- `DRONE_NAME = "lisparrow"`  
  URDF path: `genesis/assets/urdf/lisparrow/lisparrow.urdf`

The script parses the URDF to discover the aerodynamic frames (`aero_frame_*`, `prop_frame_*`) and keeps only the required links when loading the model.

### Which physics is used

The simulation uses **Genesis rigid-body physics** plus the **AeroSolver**:

- Rigid-body integration is handled by Genesis’ rigid solver.
- Aerodynamic forces and torques are computed by a drone-specific AeroSolver and applied each substep.

Solver selection:

- By default, the AeroSolver is `SimpleDroneAeroSolver`.
- If `AERO_SOLVER_KIND` is set to `lisparrow` (or `cpp`/`morphing`), the script switches to `LisparrowAeroSolver`.

### How the aerodynamic physics works (where it gets data, what it calls)

At each substep, Genesis calls the AeroSolver hook:

- `BaseAeroSolver.substep_pre_coupling(...)` → `_aero_step()`

Inside `_aero_step()` the solver:

1. Launches a Taichi kernel (`_aero_compute_kernel`) implemented by the selected solver (`SimpleDroneAeroSolver` or `LisparrowAeroSolver`).  
2. Fills per-surface force and center‑of‑pressure buffers (`force_b`, `cp_b`) in **link-local frames**.  
3. Applies forces/torques to the rigid solver via:
   - `rigid_solver.apply_links_force_at_point_link_frame(...)`
   - `rigid_solver.apply_links_coupling_torque(...)`

**Where the aero data comes from:**

- **URDF geometry**: link names (`aero_frame_*`, `prop_frame_*`) and mesh/box dimensions are parsed to compute area, chord, span, and aspect ratio.  
  (Parsed by `DroneAeroModel` from `genesis/assets/urdf/aero_model.py`, implemented in `genesis/assets/urdf/common/drone_model.py`.)
- **Aero parameters**: base parameters and per-surface parameters are loaded from the solver’s inline config  
  (`SimpleDroneAeroParameters` or `LisparrowAeroParameters`) and merged with the URDF-derived geometry.
- **Actuators**: `actuators.csv` next to the URDF provides thrust limits and servo gains; prop parameters (max thrust, kappa, cutoff) are read here.
- **Optional NACA override**: for `SimpleDroneAeroSolver` only, `apply_naca_wing_override(...)` can replace wing airfoil parameters from `src/naca_generation/naca4.csv`.

**What state it reads at runtime:**

- Link linear/angular velocities and orientations from the rigid solver.
- Joint DOF positions (especially in the Lisparrow solver, which uses sweep/elevator/rudder angles directly).

### What the script does (step by step)

#### **1) Build the scene**

The script creates a Genesis scene with gravity, a ground plane, and a viewer.  
It then loads the URDF as a `RigidEntity` and initializes:

- the initial position, velocity, and orientation,
- servo joint targets and limits read from the URDF,
- PD gains for the servo joints (from `actuators.csv` next to the URDF).

#### **2) Attach to the AeroSolver**

After `scene.build(...)`, the drone is registered with:

```
scene.sim.aero_solver.add_target(drone, drone_model=drone_model)
```

The solver:

- parses the URDF geometry (wing/tail/fuselage/prop),  
- allocates per-surface buffers in Taichi/Torch,  
- computes lift/drag/side forces and centers of pressure in each link frame.

If `DRONE_NAME = "mydrone"` and a NACA code is set, the script calls:

```
scene.sim.aero_solver.apply_naca_wing_override(NACA)
```

to override the wing airfoil parameters from `src/naca_generation/naca4.csv`.

#### **3) Keyboard controls**

A `DroneController` listens to the keyboard and continuously applies:

- **throttle** (sent only via `aero_solver.set_throttle`),  
- **servo target angles** (sent as PD position targets on the joints).

**Key mappings:**

- ↑ / ↓ — increase / decrease throttle  
- w / s — symmetric wing sweep  
- space / shift — symmetric wing twist  
- ← / → — asymmetric wing twist (roll)  
- q / e — elevator up / down (pitch)  
- a / d — rudder left / right (yaw)  
- ESC — quit  

Throttle is the **only** way to generate propeller thrust. The AeroSolver applies both thrust and reaction torque in the correct link frame.

#### **4) Physics + aerodynamics step**

Every simulation step:

1. Servo commands are applied to joint targets.  
2. Throttle is passed to the AeroSolver.  
3. Genesis advances rigid-body physics.  
4. The AeroSolver computes and applies aerodynamic forces/torques on each surface.  
5. If debug is enabled, lift/drag vectors are drawn as arrows in the viewer.

### What you see and where logs go

- The viewer follows the drone in 3D and shows its trajectory.
- If debug is enabled, you’ll see per-surface lift/drag arrows.
- All `print()` output from the script is redirected to `winged_drone_output.txt`.
