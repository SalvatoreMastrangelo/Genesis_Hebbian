"""Editable prescribed Lisparrow trajectory for ``winged_drone_fly.py --trajectory``.

The output of ``command(t)`` deliberately commands exact URDF joint names.
That avoids ambiguity between human control names, mirrored joint axes, and
keyboard sign conventions.

Units:
- all joint targets are radians in the Genesis/URDF joint coordinate
- throttle is normalized in [0, 1]

Lisparrow wing axes are mirrored in the URDF:
- left outer wing axis:  (0, 0, -1)
- right outer wing axis: (0, 0,  1)

Therefore equal left/right joint targets produce symmetric physical wing
sweep, and opposite left/right offsets produce asymmetric sweep.
"""

import math


INITIAL_CONDITIONS = {
    "pos": [0.0, 0.0, 20.0],
    "vel": [6.0, 0.0, 1.0],
    "euler_deg": [0.0, -15.0, 0.0],
    "ang_vel": [0.0, 0.0, 0.0],
    "throttle": 0.15,
    "joint_targets": {
        "joint_left_outer_wing_hinged": -0.2,
        "joint_right_outer_wing_hinged": -0.2,
        "joint_elevator_hinged": -0.0,
        "joint_rudder_hinged": 0.0,
    },
}


CONTROL_CHANNELS = {
    "throttle": {
        "wave": "sin",
        "offset": 0.15,
        "amplitude": 0.0,
        "frequency_hz": 0.20,
        "phase_rad": 0.0,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "sweep_symmetric": {
        "wave": "sin",
        "offset": -1.0,
        "amplitude": 0.0,
        "frequency_hz": 0.1,
        "phase_rad": 0.0,
        "minimum": -1.50,
        "maximum": 1.50,
    },
    "sweep_asymmetric": {
        "wave": "cos",
        "offset": 0.0,
        "amplitude": 0.0,
        "frequency_hz": 0.20,
        "phase_rad": math.pi / 2.0,
        "minimum": -1.50,
        "maximum": 1.50,
    },
    "elevator": {
        "wave": "sin",
        "offset": -0.4,
        "amplitude": 0.0,
        "frequency_hz": 0.2,
        "phase_rad": 0.0,
        "minimum": -0.45,
        "maximum": 0.45,
    },
    "rudder": {
        "wave": "cos",
        "offset": 0.4,
        "amplitude": 0.0,
        "frequency_hz": 0.20,
        "phase_rad": math.pi / 2.0,
        "minimum": -0.45,
        "maximum": 0.45,
    },
}

LISPARROW_JOINT_NAMES = {
    "left_wing": "joint_left_outer_wing_hinged",
    "right_wing": "joint_right_outer_wing_hinged",
    "elevator": "joint_elevator_hinged",
    "rudder": "joint_rudder_hinged",
}


def _wave_value(t: float, cfg: dict) -> float:
    phase = 2.0 * math.pi * float(cfg["frequency_hz"]) * t + float(cfg.get("phase_rad", 0.0))
    if str(cfg.get("wave", "sin")).lower() == "cos":
        raw = math.cos(phase)
    else:
        raw = math.sin(phase)
    value = float(cfg.get("offset", 0.0)) + float(cfg.get("amplitude", 0.0)) * raw
    lo = cfg.get("minimum")
    hi = cfg.get("maximum")
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return value


def control_channels(t: float) -> dict:
    return {name: _wave_value(float(t), cfg) for name, cfg in CONTROL_CHANNELS.items()}


def lisparrow_joint_targets(channels: dict) -> dict:
    sweep_symmetric = float(channels.get("sweep_symmetric", 0.0))
    sweep_asymmetric = float(channels.get("sweep_asymmetric", 0.0))

    return {
        LISPARROW_JOINT_NAMES["left_wing"]: sweep_symmetric + sweep_asymmetric,
        LISPARROW_JOINT_NAMES["right_wing"]: sweep_symmetric - sweep_asymmetric,
        LISPARROW_JOINT_NAMES["elevator"]: float(channels.get("elevator", 0.0)),
        LISPARROW_JOINT_NAMES["rudder"]: float(channels.get("rudder", 0.0)),
    }


def command(t: float) -> dict:
    channels = control_channels(float(t))
    return {
        "throttle": channels["throttle"],
        "joint_targets": lisparrow_joint_targets(channels),
        "channels": channels,
    }
