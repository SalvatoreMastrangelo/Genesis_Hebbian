## Quick Installation

Install **PyTorch** first following the [official instructions](https://pytorch.org/get-started/locally/).

Then, install Genesis via PyPI:
```bash
pip install genesis-world  # Requires Python>=3.10,<3.14;
```

Then clone the repository and install locally:
```bash
git clone https://github.com/andrevic1/Genesis.git
cd Genesis
git checkout branch name
pip install -e ".[dev]"
```
It is recommended to systematically execute `pip install -e ".[dev]"` after moving HEAD to make sure that all dependencies and entrypoints are up-to-date.

## Documentation

Comprehensive documentation of Genesis is available in [English](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html)

## How to easily fly a drone from the keyboard

The script **`src/winged_drone_fly.py`** provides a complete, ready-to-use example of how to load a fully aerodynamic winged drone in **Genesis**, connect it to the **AeroSolver**, and manually fly it in real time using the keyboard.

### What the script does

#### **Load and build the scene**

The script creates a Genesis scene with gravity, a ground plane, and a 3D viewer.  
Then it loads a parametrized winged-drone URDF and initializes:

- its position, velocity, and orientation,  
- its 6 aerodynamic control-surface joints (sweep, twist, elevator, rudder),  
- PD gains for the servo joints.

#### **Attach the drone to the aerodynamic solver**

After the simulator is built, the drone is registered as an aerodynamic target in the custom **AeroSolver**.  
The solver automatically:

- parses the URDF to extract wing, tail and fuselage geometry,  
- computes aerodynamic coefficients, centers of pressure, stall transitions, wing downwash, propeller induced-velocity, and slipstream effects,  
- allocates per-link aerodynamic force buffers in Taichi/Torch.

#### **Keyboard-based manual flight**

A `DroneController` object continuously listens to key presses.  
It converts these keys into:

- **throttle commands** (sent directly to the AeroSolver),  
- **servo surface commands** (sent as PD position targets).

**Key mappings:**

- ↑ / ↓ — increase / decrease throttle  
- w / s — symmetric wing sweep  
- space / shift — symmetric wing twist  
- ← / → — asymmetric wing twist (roll control)  
- q / e — elevator up / down (pitch control)  
- a / d — rudder left / right (yaw control)  
- ESC — quit  

Throttle is the **only** way to generate propeller thrust: it is passed into the AeroSolver, which applies thrust + propeller reaction torque in the correct link frame.

#### **Aerodynamics + physics stepping**

At every simulation step:

- control-surface targets are applied to the drone joints,  
- throttle is passed into the AeroSolver,  
- Genesis performs rigid-body integration,  
- the AeroSolver computes and applies aerodynamic forces/torques on each surface  
  (fuselage, inner/outer wings, elevators, rudder, prop),  
- if debugging is enabled, lift and drag vectors are drawn as 3D arrows per surface.

#### **Real-time visualization**

The Genesis viewer follows the drone in 3D, showing its trajectory and (optionally) the decomposed aerodynamic lift/drag arrows computed inside the solver.  
The simulation loop runs in a background thread, so the viewer remains responsive.


#### **Haroun Indications**

Things to be changed:

src/winged_drone_fly.py --> make it compatible with the new urdf (new links and new joints)


genesis/asset/urdf/mydrone --> add mesh and urdf file


genesis/engine/solvers/aero_solver.py --> study, adapt for new aerodynamic parameters, adapt for new link. Here you have to do a lot of changes, so I would suggest that you can add also a utilitiy folder with drone specific features, and then adapt the solver on top of it.