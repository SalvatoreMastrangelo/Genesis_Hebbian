import copy
from dataclasses import dataclass
import math
import gstaichi as ti
import torch
import numpy as np

from genesis.engine.solvers.base_aero_solver import BaseAeroSolver
from genesis.engine.entities import RigidEntity
from genesis.assets.urdf.aero_model import DroneAeroModel, SurfaceKind
from genesis.utils import geom as gu
from genesis.utils.geom import transform_by_quat


# Numeric codes from DroneAeroModel._build_geom (kept local for Taichi kernels).
K_FUSELAGE = 0
K_WING = 1
K_ELEVATOR = 2
K_RUDDER = 3
K_PROPELLER = 4

LISPARROW_B_OUTER_ZERO = 0.16515
LISPARROW_S_OUTER_ZERO = 0.01629


LISPARROW_SERVO_GAIN_OVERRIDES: dict[str, tuple[float, float]] = {
    # Genesis PD gains are torque/rad and torque/(rad/s).  The actuator CSV
    # keeps catalog coefficients that are not the dyn_casadi actuator model.
    # These Lisparrow-only runtime values keep enough stiffness for roughly
    # 10 m/s aerodynamic loads.  The Lisparrow URDF also carries passive joint
    # damping/friction, so kv is only the active PD damping contribution.
    "joint_left_outer_wing_hinged":  (10.0, 1.5),
    "joint_right_outer_wing_hinged": (10.0, 1.5),
    "joint_elevator_hinged":         (10.0, 1.5),
    "joint_rudder_hinged":           (10.0, 1.5),
}


def lisparrow_servo_gain_override(joint_name: str) -> tuple[float, float] | None:
    return LISPARROW_SERVO_GAIN_OVERRIDES.get(str(joint_name))


@dataclass
class BoundTarget:
    model: DroneAeroModel | None
    B: int
    L: int
    frames: list[str]
    kinds: list[SurfaceKind]
    link_indices: list[int]
    geom: list[tuple[float, float, float, int]]
    base_param_overrides: dict[str, float]
    noise_params: dict[str, float]
    side: list[int]
    slip_code: list[int]
    fus_width: float
    surface_params: list[dict]
    surface_theta_lower: list[float]
    surface_theta_upper: list[float]
    surface_theta_max: list[float]
    surface_axis_mode: list[int]
    swept_wing: list[int]


class LisparrowAeroDefaults:
    """
    Hard-coded baseline parameters for the C++-equivalent Lisparrow solver.

    Values here are safe defaults and are overridden by DroneAeroModel parameters
    when provided. Root-wing geometry entries are set to 0.0 to allow automatic
    derivation from URDF geometry if the YAML does not supply them.
    """

    # Outer wing geometry at theta_sw = 0 deg (from dyn_casadi wing fit).
    B_OUTER_ZERO = LISPARROW_B_OUTER_ZERO
    S_OUTER_ZERO = LISPARROW_S_OUTER_ZERO

    FLUID = {
        "rho": 1.225,
    }

    PROP = {
        "prop_radius": 0.075,
        "max_thrust": 0.9,
        # dyn_casadi motor_tau_inv = 2.054 rad/s.  The existing solver
        # parameter is named as a cutoff frequency, so keep the equivalent Hz.
        "prop_cutoff_hz": 2.054 / (2.0 * math.pi),
        # dyn_casadi applies thrust at pos_prop but does not include a
        # propeller spin reaction torque.
        "kappa_prop": 0.0,
        "throttle_offset": 0.05,
        "motor_omega_map": -0.6713,
        "motor_tau_inv": 2.054,
    }

    SLIPSTREAM = {
        "K_slip_tail": 1.0,
        "K_slip_wing": 1.0,
    }

    EXTENSIONS = {
        "enable_reynolds": 1.0,
        "enable_qs_dyn_lift": 1.0,
        "enable_cos2_beta": 1.0,
        "use_slip_two_segment_sum": 1.0,
        "nu": 1.5e-5,
        "Re_ref": 100000.0,
        "M_Re": 2.5,
    }

    WING = {
        "M_smooth": 0.2,
        "alpha_stall_wing_deg": 14.0,
        "cl_alpha_wing_2D": 2.0 * math.pi,
        "c_d_0_wing": 0.05,
    }

    TAIL = {
        "k_alpha_elev": 0.7,
        "s_hor_tail": 0.021495,
        "b_hor_tail": 0.20,
        "c_hor_tail": 0.10,
        "c_ele": 0.058,
        "alpha_stall_tail_deg": 20.0,
        "cl_alpha_tail_2D": 5.37,
        "c_d_0_tail": 0.2,
    }

    RUDDER = {
        "k_alpha_rudder": 0.7,
        "s_vert_tail": 0.0125625,
        "b_vert_tail": 0.15,
        "c_vert_tail": 0.15,
        "c_rud": 0.096,
    }

    WING_ROOT = {
        "b_center": 0.027,
        "b_root_inner": 0.247,
        "c_root_inner": 0.18,
        "b_root_outer": 0.337,
        "c_root_outer": 0.15,
        "s_one_root": 0.0265,
    }

    REF_POS = {
        "pos_cg_x": -0.085,
        "pos_ele_le_x": -0.339,
        "pos_ele_le_y": 0.0,
        "pos_ele_le_z": 0.0,
        "pos_rud_le_x": -0.344,
        "pos_rud_le_y": 0.0,
        "pos_rud_le_z": 0.075,
        "pos_wing_left_le_x": 0.0,
        "pos_wing_left_le_y": 0.0145,
        "pos_wing_left_le_z": 0.0,
        "pos_wing_right_le_x": 0.0,
        "pos_wing_right_le_y": -0.0145,
        "pos_wing_right_le_z": 0.0,
        "pos_prop_x": 0.18,
        "pos_prop_y": 0.0,
        "pos_prop_z": 0.0,
    }

    ROT_POINTS = {
        "pos_rot_point_wing_left_x": 0.045,
        "pos_rot_point_wing_left_y": 0.174,
        "pos_rot_point_wing_left_z": 0.0,
        "pos_rot_point_wing_right_x": 0.045,
        "pos_rot_point_wing_right_y": -0.174,
        "pos_rot_point_wing_right_z": 0.0,
        "pos_rot_point_ele_x": -0.269,
        "pos_rot_point_ele_y": 0.0,
        "pos_rot_point_ele_z": -0.012,
        "pos_rot_point_rud_x": -0.261,
        "pos_rot_point_rud_y": 0.0,
        "pos_rot_point_rud_z": 0.0,
    }

    LIMITS = {
        "force_cap": 1e9,
    }

    # Randomizable parameters (None means "all params below"). Edit this list to
    # control what `randomize_aero_params` touches.
    RANDOMIZABLE_PARAMS: list[str] | None = None

    # To add or change randomizable parameters, edit the blocks above so the keys
    # are included in base_params()/param_names() and get Taichi fields, then
    # optionally set RANDOMIZABLE_PARAMS to a subset.
    @classmethod
    def base_params(cls) -> dict:
        params = {}
        for block in (
            cls.FLUID,
            cls.PROP,
            cls.SLIPSTREAM,
            cls.EXTENSIONS,
            cls.WING,
            cls.TAIL,
            cls.RUDDER,
            cls.WING_ROOT,
            cls.REF_POS,
            cls.ROT_POINTS,
            cls.LIMITS,
        ):
            params.update(block)
        return params

    @classmethod
    def param_names(cls) -> list[str]:
        names: list[str] = []
        for block in (
            cls.FLUID,
            cls.PROP,
            cls.SLIPSTREAM,
            cls.EXTENSIONS,
            cls.WING,
            cls.TAIL,
            cls.RUDDER,
            cls.WING_ROOT,
            cls.REF_POS,
            cls.ROT_POINTS,
            cls.LIMITS,
        ):
            names.extend(block.keys())
        return names

    @classmethod
    def randomizable_param_names(cls) -> list[str]:
        if cls.RANDOMIZABLE_PARAMS is None:
            return cls.param_names()
        return list(cls.RANDOMIZABLE_PARAMS)


class LisparrowAeroParameters:
    """
    Inline aero configuration for the Lisparrow solver.
    """

    GLOBAL = LisparrowAeroDefaults.base_params()

    TYPES = {
        "fuselage": {
            "cd0": 0.75,
            "k_slip_fus": 0.0,
            "cp_start": 0.25,
            "cp_end": 0.5,
            "cg_to_chord": 0.31,
        },
        "wing": {
            "cl_alpha_2d": LisparrowAeroDefaults.WING["cl_alpha_wing_2D"],
            "alpha0_2d": 0.0,
            "cd0": LisparrowAeroDefaults.WING["c_d_0_wing"],
            "k_slip_wing": LisparrowAeroDefaults.SLIPSTREAM["K_slip_wing"],
            "alpha_stall_deg": LisparrowAeroDefaults.WING["alpha_stall_wing_deg"],
            "m_smooth": LisparrowAeroDefaults.WING["M_smooth"],
            "cp_start": 0.25,
            "cp_end": 0.5,
            "cg_to_chord": 0.31,
            "re_nominal": 100000.0,
            "a": 1.0,
        },
        "elevator": {
            "cl_alpha_2d": LisparrowAeroDefaults.TAIL["cl_alpha_tail_2D"],
            "alpha0_2d": 0.0,
            "cd0": LisparrowAeroDefaults.TAIL["c_d_0_tail"],
            "k_slip_tail": LisparrowAeroDefaults.SLIPSTREAM["K_slip_tail"],
            "k_eps_tail": 1.0,
            "alpha_stall_deg": LisparrowAeroDefaults.TAIL["alpha_stall_tail_deg"],
            "m_smooth": LisparrowAeroDefaults.WING["M_smooth"],
            "cp_start": 0.25,
            "cp_end": 0.5,
            "cg_to_chord": 0.31,
            "re_nominal": 50000.0,
            "a": 1.0,
        },
        "rudder": {
            "cl_alpha_2d": LisparrowAeroDefaults.TAIL["cl_alpha_tail_2D"],
            "alpha0_2d": 0.0,
            "cd0": LisparrowAeroDefaults.TAIL["c_d_0_tail"],
            "alpha_stall_deg": LisparrowAeroDefaults.TAIL["alpha_stall_tail_deg"],
            "m_smooth": LisparrowAeroDefaults.WING["M_smooth"],
            "cp_start": 0.25,
            "cp_end": 0.5,
            "cg_to_chord": 0.31,
            "re_nominal": 50000.0,
            "a": 1.0,
        },
    }

    LINKS = {
        "fuselage": {
            "type": "fuselage",
            "s_folded": 1.0,
        },
        "root_wing_fixed": {
            "type": "wing",
            "s_folded": 1.0,
            "chord": 0.17331144141030075,
            "span": 0.337,
            "S": 0.05786,
        },
        "left_outer_wing_hinged": {
            "type": "wing",
            "actuator_yaw": "X10_servo",
            "actuator_pitch": "X08_servo",
            "s_folded_yaw": 1.0,
            "s_folded_pitch": 1.0,
            "chord": 0.15,
            "span": 0.16515,
            "S": 0.01629,
        },
        "right_outer_wing_hinged": {
            "type": "wing",
            "actuator_yaw": "X10_servo",
            "actuator_pitch": "X08_servo",
            "s_folded_yaw": 1.0,
            "s_folded_pitch": 1.0,
            "chord": 0.15,
            "span": 0.16515,
            "S": 0.01629,
        },
        "elevator_hinged": {
            "type": "elevator",
            "actuator_pitch": "X08_servo",
            "actuator_yaw": None,
            "s_folded_pitch": 1.0,
            "S": 0.021495,
            "chord": 0.10,
            "span": 0.2,
        },
        "rudder_hinged": {
            "type": "rudder",
            "actuator_pitch": None,
            "actuator_yaw": "X08_servo",
            "s_folded_yaw": 1.0,
            "S": 0.0125625,
            "chord": 0.15,
            "span": 0.15,
        },
        "propeller_fixed": {
            "type": "propeller",
            "actuator": "morphing_prop",
            "max_thrust": LisparrowAeroDefaults.PROP["max_thrust"],
            "kappa_prop": LisparrowAeroDefaults.PROP["kappa_prop"],
            "prop_cutoff_hz": LisparrowAeroDefaults.PROP["prop_cutoff_hz"],
            "cp_x": 0.135,
            "cp_y": -1.9995e-05,
            "cp_z": -0.00627,
        },
    }

    NOISE = {
        "sigma_mag": 0.0,
        "sigma_dir": 0.0,
        "sigma_param": 0.0,
        "sigma_cp": 0.0,
    }

    @classmethod
    def as_dict(cls) -> dict:
        return {
            "global": copy.deepcopy(cls.GLOBAL),
            "types": copy.deepcopy(cls.TYPES),
            "links": copy.deepcopy(cls.LINKS),
            "noise": copy.deepcopy(cls.NOISE),
        }


@ti.data_oriented
class LisparrowAeroSolver(BaseAeroSolver):
    """
    C++-equivalent morphing aero solver (forces + CP only).

    The solver matches the original IndMorphingUAVDynamics force model and
    returns per-surface forces and centers of pressure in the link frame.
    """

    # ------------------------------------------------------------------
    # BaseAeroSolver hooks
    # ------------------------------------------------------------------
    def _init_surface_param_names(self):
        # No per-wing/elevator duplicated params are used by this solver.
        self._wing_param_names = []
        self._elevator_param_names = []

    def _const_init(self):
        # Default parameters (overridden by DroneAeroModel when provided).
        self._aero_base = dict(LisparrowAeroDefaults.base_params())
        self._aero_frames: list[str] = []
        self._geom: list[tuple[float, float, float, int]] = []
        self._surface_kinds: list[int] = []
        self._geom_uses_span = False  # Set True if you provide (S, span, chord, kind) manually.
        self._drone_model: DroneAeroModel | None = None
        self._surface_params: list[dict] = []
        self._surface_theta_lower: list[float] = []
        self._surface_theta_upper: list[float] = []
        self._surface_theta_max: list[float] = []
        self._surface_axis_mode: list[int] = []
        self._surface_is_swept_wing: list[int] = []
        self._randomizable_param_names = LisparrowAeroDefaults.randomizable_param_names()
        self.noise_sigma_mag = DroneAeroModel.NOISE_DEFAULTS["sigma_mag"]
        self.noise_sigma_dir = DroneAeroModel.NOISE_DEFAULTS["sigma_dir"]
        self.noise_sigma_param = DroneAeroModel.NOISE_DEFAULTS["sigma_param"]
        self.noise_sigma_cp = DroneAeroModel.NOISE_DEFAULTS["sigma_cp"]
        self.tip_to_tip: float = 0.0
        self.AR_wing: float = 0.0
        self.AR_tail: float = 0.0
        self.S_perp_fus: float = 0.0
        self.cg_fus_local_x: float = 0.0
        self.cz_rudder: float = 0.0
        self.genes: list[float] = []
        self.genes_dict: dict[str, float] = {}
        self._verbose_init: bool = False
        self._verbose_init_done: bool = False
        self._surf_joint_name: list[str] = []

    def add_target(
        self,
        entity: RigidEntity,
        urdf_file: str | None = None,
        drone_model: DroneAeroModel | None = None,
    ):
        self._aero_targets.append(entity)

        # Phase 1: bind model and collect metadata
        bound = self._bind_target_model(entity, urdf_file, drone_model)

        # Phase 2: allocate common + lisparrow-specific fields
        self._alloc_simple_fields(bound.B, bound.L)
        self._alloc_common_fields(bound.B, bound.L)

        # Merge params after geometry is known (to auto-fill wing root values).
        self._aero_base = self._merge_base_params(bound.model)
        self._register_base_params(self._aero_base)

        # Phase 3: populate fields and initialize cached state
        self._init_simple_fields(bound)
        self._map_actuators(entity, bound.model)
        self._verbose_init_done = False

        # Cached Torch buffers for parameters queried each step.
        self._init_param_buffers()

        self._aero_enabled = True

    def _bind_target_model(
        self,
        entity: RigidEntity,
        urdf_file: str | None,
        drone_model: DroneAeroModel | None,
    ) -> BoundTarget:
        """
        Phase 1: resolve the model, validate inputs, and collect binding metadata.
        """
        model = self._resolve_drone_model(urdf_file, drone_model)
        if model is not None:
            self._apply_drone_model(model, entity)
        else:
            if not self._aero_frames:
                raise RuntimeError("Aero frames missing. Provide a DroneAeroModel or URDF.")
            if not self._geom:
                raise RuntimeError("Aerodynamic geometry missing. Provide a valid URDF/model.")
            self._aero_link_idx = [entity.get_link(name).idx for name in self._aero_frames]
            self._surface_kinds = [k for (_, _, _, k) in self._geom]

        if not self._geom:
            raise RuntimeError("Aerodynamic geometry missing. Provide a valid URDF/model.")

        self.L = len(self._geom)
        self._B = getattr(self._rigid_solver.sim, "_B", 1)
        self.n_links_ = max(1, self._rigid_solver.n_links)

        frames = list(self._aero_frames)
        kinds = list(self._surface_kinds) if model is not None else []
        if model is not None and kinds and len(kinds) != len(frames):
            raise RuntimeError("Drone model did not expose per-surface kinds.")

        geom = list(self._geom)
        link_indices = list(self._aero_link_idx)
        base_param_overrides = {}
        noise_params = {}

        side = [0 for _ in range(self.L)]
        slip_code = [0 for _ in range(self.L)]
        fus_width = 0.0
        surface_params = list(self._surface_params) if self._surface_params else [{} for _ in range(self.L)]
        surface_theta_lower = list(self._surface_theta_lower) if self._surface_theta_lower else [0.0 for _ in range(self.L)]
        surface_theta_upper = list(self._surface_theta_upper) if self._surface_theta_upper else [0.0 for _ in range(self.L)]
        surface_theta_max = list(self._surface_theta_max) if self._surface_theta_max else [0.0 for _ in range(self.L)]
        surface_axis_mode = list(self._surface_axis_mode) if self._surface_axis_mode else [0 for _ in range(self.L)]
        swept_wing = list(self._surface_is_swept_wing) if self._surface_is_swept_wing else [0 for _ in range(self.L)]
        for i, (_, _, c, k) in enumerate(geom):
            name = frames[i].lower() if i < len(frames) else ""
            if "left" in name:
                side[i] = +1
            elif "right" in name:
                side[i] = -1
            else:
                side[i] = 0

            if k == K_FUSELAGE:
                slip_code[i] = 0
                fus_width = float(c)
            elif k == K_ELEVATOR:
                slip_code[i] = 1
            elif k == K_WING and "prop" in name:
                slip_code[i] = 2
            else:
                slip_code[i] = 3

        return BoundTarget(
            model=model,
            B=self._B,
            L=self.L,
            frames=frames,
            kinds=kinds,
            link_indices=link_indices,
            geom=geom,
            base_param_overrides=base_param_overrides,
            noise_params=noise_params,
            side=side,
            slip_code=slip_code,
            fus_width=fus_width,
            surface_params=surface_params,
            surface_theta_lower=surface_theta_lower,
            surface_theta_upper=surface_theta_upper,
            surface_theta_max=surface_theta_max,
            surface_axis_mode=surface_axis_mode,
            swept_wing=swept_wing,
        )

    def _resolve_drone_model(self, urdf_file: str | None, drone_model: DroneAeroModel | None):
        if drone_model is not None and urdf_file is not None:
            raise ValueError("Specify either `drone_model` or `urdf_file`, not both.")
        if drone_model is not None:
            return drone_model
        if urdf_file is None:
            return None
        return DroneAeroModel(urdf_file, config_override=LisparrowAeroParameters.as_dict())

    def _apply_drone_model(self, model: DroneAeroModel, entity: RigidEntity):
        self._drone_model = model
        self.tip_to_tip = 0.0
        self.AR_wing = 0.0
        self.AR_tail = 0.0
        self.S_perp_fus = 0.0
        self.cg_fus_local_x = 0.0
        self.cz_rudder = 0.0
        self.genes = list(getattr(model, "genes", []))
        self.genes_dict = dict(getattr(model, "genes_dict", {}))
        frames = [frame for frame in list(model.frames) if frame in LisparrowAeroParameters.LINKS]
        link_idx = model.link_indices(entity)
        frame_to_link_idx = {frame: idx for frame, idx in zip(model.frames, link_idx)}

        geom: list[tuple[float, float, float, int]] = []
        surface_params: list[dict] = []
        surface_kinds: list[int] = []
        wing_span_sum = 0.0
        wing_area_sum = 0.0
        for frame in frames:
            cfg = copy.deepcopy(LisparrowAeroParameters.LINKS[frame])
            kind_name = str(cfg.get("type", "fuselage")).lower()
            kind_code = self._surface_kind_code_from_name(kind_name)
            prm = copy.deepcopy(LisparrowAeroParameters.TYPES.get(kind_name, {}))
            prm.update(cfg)

            area = float(cfg.get("S", 0.0))
            chord = float(cfg.get("chord", 0.0))
            span = float(cfg.get("span", 0.0))
            if kind_code in (K_WING, K_ELEVATOR, K_RUDDER):
                if area <= 0.0 or chord <= 0.0 or span <= 0.0:
                    raise RuntimeError(
                        f"Lisparrow defaults must define S/chord/span for '{frame}', got "
                        f"S={area}, chord={chord}, span={span}."
                    )
                ar = max(1e-6, span / chord)
                if kind_code == K_WING:
                    wing_span_sum += span
                    wing_area_sum += area
            else:
                ar = 0.0

            geom.append((area, ar, chord, kind_code))
            surface_params.append(prm)
            surface_kinds.append(kind_code)

        if wing_span_sum > 0.0:
            self.tip_to_tip = wing_span_sum
        if wing_span_sum > 0.0 and wing_area_sum > 0.0:
            self.AR_wing = (wing_span_sum * wing_span_sum) / wing_area_sum
        s_h_tail = float(LisparrowAeroDefaults.TAIL.get("s_hor_tail", 0.0))
        b_h_tail = float(LisparrowAeroDefaults.TAIL.get("b_hor_tail", 0.0))
        if s_h_tail > 0.0 and b_h_tail > 0.0:
            self.AR_tail = (b_h_tail * b_h_tail) / s_h_tail

        keep = [i for i, _ in enumerate(frames)]

        if not keep:
            raise RuntimeError("Lisparrow solver requires at least one aerodynamic link in the drone model.")

        self._aero_frames = [frames[i] for i in keep]
        self._geom = [geom[i] for i in keep]
        self._surface_kinds = [surface_kinds[i] for i in keep] if surface_kinds else []
        self._aero_link_idx = [frame_to_link_idx[self._aero_frames[i]] for i in range(len(self._aero_frames))]
        self._geom_uses_span = False
        self._surface_params = [dict(surface_params[i]) if i < len(surface_params) else {} for i in keep]

        actuator_map = dict(getattr(model, "actuators", {}) or {})
        self._surface_theta_lower = []
        self._surface_theta_upper = []
        self._surface_theta_max = []
        self._surface_axis_mode = []
        self._surface_is_swept_wing = []
        for out_i, frame in enumerate(self._aero_frames):
            info = actuator_map.get(frame)
            limits = getattr(info, "limits", None) if info is not None else None
            theta_lo = 0.0
            theta_hi = 0.0
            theta_max = 0.0
            if isinstance(limits, (tuple, list)) and len(limits) >= 2:
                lo = float(limits[0])
                hi = float(limits[1])
                if math.isfinite(lo) and math.isfinite(hi):
                    theta_lo = lo
                    theta_hi = hi
                    theta_max = max(abs(lo), abs(hi))
            self._surface_theta_lower.append(theta_lo)
            self._surface_theta_upper.append(theta_hi)
            self._surface_theta_max.append(theta_max)

            kind_code = int(self._surface_kinds[out_i]) if out_i < len(self._surface_kinds) else K_FUSELAGE
            axis_mode = self._axis_mode_from_joint(frame, kind_code, info)
            self._surface_axis_mode.append(axis_mode)

            jname = str(getattr(info, "joint_name", "") or "").lower()
            is_swept = 1 if (kind_code == K_WING and ("sweep" in jname or "outer" in frame.lower())) else 0
            self._surface_is_swept_wing.append(is_swept)

        if hasattr(model, "validate_entity"):
            joint_names = [info.joint_name for info in model.actuators.values() if info.joint_name]
            valid_names: list[str] = []
            for name in joint_names:
                try:
                    joint = entity.get_joint(name)
                except Exception:
                    continue
                idxs = getattr(joint, "dofs_idx_local", None)
                if idxs is None:
                    continue
                if isinstance(idxs, (list, tuple)) and len(idxs) == 0:
                    continue
                valid_names.append(name)
            if valid_names:
                model.validate_entity(entity, servo_joint_names=valid_names)

        self._log_geometry(model)

    @staticmethod
    def _axis_mode_from_joint(frame: str, kind_code: int, info) -> int:
        has_yaw = bool(getattr(info, "yaw_actuator", None)) if info is not None else False
        has_pitch = bool(getattr(info, "pitch_actuator", None)) if info is not None else False
        if has_yaw and not has_pitch:
            return 1
        if has_pitch and not has_yaw:
            return 2

        jname = str(getattr(info, "joint_name", "") or "").lower()
        if any(tok in jname for tok in ("yaw", "sweep", "rudder")):
            return 1
        if any(tok in jname for tok in ("pitch", "elevator", "twist")):
            return 2

        lname = frame.lower()
        if kind_code == K_ELEVATOR or "elevator" in lname:
            return 2
        if kind_code == K_RUDDER or "rudder" in lname:
            return 1
        if kind_code == K_WING:
            return 1
        return 0

    def _log_geometry(self, model: DroneAeroModel):
        dims = getattr(model, "debug_dims", {})
        if not dims:
            return
        print("[LisparrowAeroSolver]: Parsed URDF geometry:")
        for key, val in dims.items():
            print(f"  {key}: {val}")

    def _wing_side_keys(self, name: str) -> tuple[str, str]:
        return (f"{name}_left", f"{name}_right")

    def _elevator_side_keys(self, name: str) -> tuple[str, str]:
        return (f"{name}_left", f"{name}_right")

    def _ensure_wing_param_entries(self):
        return

    def _ensure_elevator_param_entries(self):
        return

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _merge_base_params(self, model: DroneAeroModel | None) -> dict:
        params = dict(LisparrowAeroDefaults.base_params())
        if model is not None:
            model_params = dict(getattr(model, "base_params", {}) or {})
            for name in LisparrowAeroDefaults.param_names():
                if name in model_params:
                    params[name] = float(model_params[name])
        params = {
            name: params[name]
            for name in LisparrowAeroDefaults.param_names()
            if name in params
        }
        return params

    def _derive_wing_root_params(self, params: dict) -> None:
        """
        Fill wing-root parameters from URDF geometry when missing or zero.

        This is an approximation to keep the solver running when the YAML
        does not provide those C++-specific values.
        """
        need_fill = any(params.get(key, 0.0) <= 0.0 for key in (
            "b_center",
            "b_root_inner",
            "c_root_inner",
            "b_root_outer",
            "c_root_outer",
            "s_one_root",
        ))
        if not need_fill:
            return

        wing_indices = [i for i, (_, _, _, k) in enumerate(self._geom) if int(k) == K_WING]
        if not wing_indices:
            return

        s_sum = 0.0
        ar_sum = 0.0
        chord_sum = 0.0
        for i in wing_indices:
            s, ar_or_span, chord, _ = self._geom[i]
            s_sum += float(s)
            ar_sum += float(ar_or_span)
            chord_sum += float(chord)

        wing_count = max(1, len(wing_indices))
        s_per = s_sum / wing_count
        ar_avg = ar_sum / wing_count
        chord_avg = chord_sum / wing_count

        span_single = self._geom_span(s_per, ar_avg)
        span_total = span_single * wing_count
        s_total = s_sum

        b_root_outer = max(0.0, span_total - 2.0 * LisparrowAeroDefaults.B_OUTER_ZERO)
        s_root = max(0.0, s_total - 2.0 * LisparrowAeroDefaults.S_OUTER_ZERO)

        if params.get("b_center", 0.0) <= 0.0:
            params["b_center"] = 0.0
        if params.get("b_root_inner", 0.0) <= 0.0:
            params["b_root_inner"] = 0.0
        if params.get("c_root_inner", 0.0) <= 0.0:
            params["c_root_inner"] = chord_avg
        if params.get("b_root_outer", 0.0) <= 0.0:
            params["b_root_outer"] = b_root_outer
        if params.get("c_root_outer", 0.0) <= 0.0:
            params["c_root_outer"] = chord_avg
        if params.get("s_one_root", 0.0) <= 0.0:
            c_root_inner = params.get("c_root_inner", chord_avg)
            b_center = params.get("b_center", 0.0)
            s_one_root = 0.5 * max(0.0, s_root - b_center * c_root_inner)
            params["s_one_root"] = s_one_root

    def _geom_span(self, s: float, ar_or_span: float) -> float:
        if self._geom_uses_span:
            return float(ar_or_span)
        return math.sqrt(max(1e-9, float(s) * float(ar_or_span)))

    def _alloc_simple_fields(self, B: int, L: int) -> None:
        """
        Phase 2: allocate Lisparrow-specific Taichi fields (naming aligned).
        """
        self._alloc_lisparrow_fields(B, L)

    def _alloc_lisparrow_fields(self, B: int, L: int) -> None:
        # Per-surface metadata.
        self._link_idx = ti.field(ti.i32, shape=(L,))
        self.kind = ti.field(ti.i32, shape=(L,))
        self.side = ti.field(ti.i32, shape=(L,))  # +1 left, -1 right, 0 other
        self.body_link_idx = ti.field(ti.i32, shape=())

        # Nominal geometry (span is derived if geom stores AR).
        self.area0 = ti.field(ti.f32, shape=(L,))
        self.span0 = ti.field(ti.f32, shape=(L,))
        self.chord0 = ti.field(ti.f32, shape=(L,))
        self.seff_ratio_base = ti.field(ti.f32, shape=(L,))
        self.seff_ratio_yaw = ti.field(ti.f32, shape=(L,))
        self.seff_ratio_pitch = ti.field(ti.f32, shape=(L,))
        self.theta_lower_link = ti.field(ti.f32, shape=(L,))
        self.theta_upper_link = ti.field(ti.f32, shape=(L,))
        self.theta_max_link = ti.field(ti.f32, shape=(L,))
        self.surface_axis_mode = ti.field(ti.i32, shape=(L,))
        self.swept_wing = ti.field(ti.i32, shape=(L,))
        self.prop_cp_x = ti.field(ti.f32, shape=(L,))
        self.prop_cp_y = ti.field(ti.f32, shape=(L,))
        self.prop_cp_z = ti.field(ti.f32, shape=(L,))
        # Actuator DOF mapping per surface.
        self._surf_dof = ti.field(ti.i32, shape=(L,))       # primary DOF (fallback)
        self._surf_dof_yaw = ti.field(ti.i32, shape=(L,))   # yaw-like DOF (sweep/rudder)
        self._surf_dof_pitch = ti.field(ti.i32, shape=(L,)) # pitch-like DOF (twist/elevator)
        for i in range(L):
            self._surf_dof[i] = -1
            self._surf_dof_yaw[i] = -1
            self._surf_dof_pitch[i] = -1
            self.prop_cp_x[i] = 0.0
            self.prop_cp_y[i] = 0.0
            self.prop_cp_z[i] = 0.0

        # Per-surface aerodynamic coefficients (placeholders for future tuning).
        self.cd0_link = ti.field(ti.f32, shape=(B, L))
        self.cl_alpha_2d_link = ti.field(ti.f32, shape=(B, L))
        self.alpha_stall_deg_link = ti.field(ti.f32, shape=(B, L))
        self.m_smooth_link = ti.field(ti.f32, shape=(B, L))
        self.k_slip_wing_link = ti.field(ti.f32, shape=(B, L))
        self.k_slip_tail_link = ti.field(ti.f32, shape=(B, L))

        # Stateful prev flow for alpha_dot (dyn aero disabled by default).
        self.flow_prev = ti.Vector.field(3, dtype=ti.f32, shape=(B,))
        self.alpha_dot_prev = ti.field(dtype=ti.f32, shape=(B,))
        self.omega_mot_norm = ti.field(dtype=ti.f32, shape=(B,))

        # ------------------------------------------------------------
        # Debug fields (match SimpleDroneAeroSolver / winged_drone_fly)
        # ------------------------------------------------------------
        # winged_drone_fly.py expects: alpha_dbg, beta_dbg, lift_dbg, drag_dbg, side_force_dbg
        # SimpleDrone uses shape (B, self.n_links_). Keep the same convention here.
        self.alpha_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
        self.beta_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
        self.lift_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
        self.drag_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
        self.side_force_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
        # Optional tail/downwash debug fields (winged_drone_fly reads these with hasattr)
        self.alpha_tail_raw_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
        self.downwash_eps_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
        self.cl_wing_for_tail_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
        self.k_eps_tail_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
        self.flow_dbg = ti.Vector.field(3, ti.f32, shape=(B, self.n_links_))
        self.joint_angle_dbg = ti.field(ti.f32, shape=(B, self.n_links_))

    @ti.func
    def _dbg_decompose_force(self, F: ti.template(), v: ti.template()):
        """
        Decompose aerodynamic force into (drag, lift, side) scalars for debug.
        This is for visualization/logging only (does not affect physics).
        """
        vn = v.norm() + 1e-9
        vhat = v / vn
        # Drag is opposite to velocity direction
        drag = -F.dot(vhat)
        # Remove drag component to get residual; side ~ y component, lift = remaining magnitude
        Fres = F + drag * vhat
        side = Fres.y
        lift = (Fres.norm())
        return drag, lift, side

    def _register_base_params(self, base: dict) -> None:
        self._param = {}
        for name in LisparrowAeroDefaults.param_names():
            val = base.get(name, 0.0)
            f = ti.field(dtype=ti.f32, shape=(self._B,))
            for b in range(self._B):
                f[b] = float(val)
            setattr(self, name, f)
            self._param[name] = f

    def _surface_kind_code(self, kind: SurfaceKind) -> int:
        mapping = {
            SurfaceKind.FUSELAGE: K_FUSELAGE,
            SurfaceKind.WING: K_WING,
            SurfaceKind.ELEVATOR: K_ELEVATOR,
            SurfaceKind.RUDDER: K_RUDDER,
            SurfaceKind.PROPELLER: K_PROPELLER,
        }
        return int(mapping.get(kind, K_FUSELAGE))

    @staticmethod
    def _surface_kind_code_from_name(kind_name: str) -> int:
        mapping = {
            "fuselage": K_FUSELAGE,
            "wing": K_WING,
            "elevator": K_ELEVATOR,
            "rudder": K_RUDDER,
            "propeller": K_PROPELLER,
        }
        return int(mapping.get(str(kind_name).lower(), K_FUSELAGE))

    def _init_simple_fields(self, bound: BoundTarget) -> None:
        """
        Phase 3: populate per-surface metadata and initialize cached state.
        """
        self._surface_params = list(bound.surface_params)
        self._surface_theta_lower = list(bound.surface_theta_lower)
        self._surface_theta_upper = list(bound.surface_theta_upper)
        self._surface_theta_max = list(bound.surface_theta_max)
        self._surface_axis_mode = list(bound.surface_axis_mode)
        self._surface_is_swept_wing = list(bound.swept_wing)
        self._init_surface_metadata()
        self._init_surface_params()
        self._init_prev_states()

    def _init_surface_metadata(self) -> None:
        body_link = self._infer_body_link_idx()
        self.body_link_idx[None] = int(body_link)

        def _clamp01(val, default=1.0):
            try:
                v = float(val)
            except Exception:
                return float(default)
            return max(0.0, min(1.0, v))

        for i, (s, ar_or_span, chord, k) in enumerate(self._geom):
            span = self._geom_span(s, ar_or_span)
            self.area0[i] = float(s)
            self.span0[i] = float(span)
            self.chord0[i] = float(chord)
            self.kind[i] = int(k)
            self._link_idx[i] = int(self._aero_link_idx[i])
            surf_params = self._surface_params[i] if i < len(self._surface_params) else {}
            ratio_base = _clamp01(surf_params.get("s_folded", 1.0), 1.0)
            ratio_yaw = _clamp01(surf_params.get("s_folded_yaw", ratio_base), ratio_base)
            ratio_pitch = _clamp01(surf_params.get("s_folded_pitch", ratio_base), ratio_base)
            self.seff_ratio_base[i] = ratio_base
            self.seff_ratio_yaw[i] = ratio_yaw
            self.seff_ratio_pitch[i] = ratio_pitch
            self.theta_lower_link[i] = float(self._surface_theta_lower[i]) if i < len(self._surface_theta_lower) else 0.0
            self.theta_upper_link[i] = float(self._surface_theta_upper[i]) if i < len(self._surface_theta_upper) else 0.0
            self.theta_max_link[i] = float(self._surface_theta_max[i]) if i < len(self._surface_theta_max) else 0.0
            self.surface_axis_mode[i] = int(self._surface_axis_mode[i]) if i < len(self._surface_axis_mode) else 0
            self.swept_wing[i] = int(self._surface_is_swept_wing[i]) if i < len(self._surface_is_swept_wing) else 0
            self.prop_cp_x[i] = float(surf_params.get("cp_x", 0.0))
            self.prop_cp_y[i] = float(surf_params.get("cp_y", 0.0))
            self.prop_cp_z[i] = float(surf_params.get("cp_z", 0.0))

            name = self._aero_frames[i].lower() if i < len(self._aero_frames) else ""
            if "left" in name:
                self.side[i] = +1
            elif "right" in name:
                self.side[i] = -1
            else:
                self.side[i] = 0

    def _init_surface_params(self) -> None:
        for b in range(self._B):
            for l in range(self.L):
                k = int(self.kind[l])
                if k == K_WING:
                    self.cd0_link[b, l] = self.c_d_0_wing[b]
                    self.cl_alpha_2d_link[b, l] = self.cl_alpha_wing_2D[b]
                    self.alpha_stall_deg_link[b, l] = self.alpha_stall_wing_deg[b]
                    self.m_smooth_link[b, l] = self.M_smooth[b]
                    self.k_slip_wing_link[b, l] = self.K_slip_wing[b]
                    self.k_slip_tail_link[b, l] = self.K_slip_tail[b]
                else:
                    self.cd0_link[b, l] = 0.0
                    self.cl_alpha_2d_link[b, l] = 0.0
                    self.alpha_stall_deg_link[b, l] = 0.0
                    self.m_smooth_link[b, l] = 0.0
                    self.k_slip_wing_link[b, l] = self.K_slip_wing[b]
                    self.k_slip_tail_link[b, l] = self.K_slip_tail[b]

    def _map_actuators(self, entity: RigidEntity, model: DroneAeroModel | None) -> None:
        actuator_map = dict(getattr(model, "actuators", {}) or {}) if model is not None else {}
        self._surf_joint_name = ["" for _ in range(len(self._aero_frames))]

        def _joint_first_dof(joint_name: str) -> int | None:
            try:
                j = entity.get_joint(joint_name)
            except Exception:
                return None
            dofs = getattr(j, "dofs_idx_local", None)
            if dofs is None:
                return None
            if isinstance(dofs, (list, tuple)):
                if len(dofs) == 0:
                    return None
                return int(dofs[0])
            return int(dofs)

        def _axis_from_joint_name(name: str) -> int:
            lname = str(name).lower()
            if any(tok in lname for tok in ("yaw", "rudder", "sweep")):
                return 1
            if any(tok in lname for tok in ("pitch", "elevator", "twist")):
                return 2
            return 0

        def _preferred_motion_joints_for_surface(frame_name: str) -> list[str]:
            lname = frame_name.lower()
            if "left_outer_wing" in lname:
                return ["joint_left_outer_wing_hinged"]
            if "right_outer_wing" in lname:
                return ["joint_right_outer_wing_hinged"]
            if "elevator" in lname:
                return ["joint_elevator_hinged"]
            if "rudder" in lname:
                return ["joint_rudder_hinged"]
            return []

        for si, frame in enumerate(self._aero_frames):
            info = actuator_map.get(frame)
            candidates: list[str] = []
            if info is not None:
                jname = getattr(info, "joint_name", None)
                if jname:
                    candidates.append(str(jname))
            candidates.extend(_preferred_motion_joints_for_surface(frame))

            # Deduplicate preserving order.
            uniq: list[str] = []
            seen = set()
            for name in candidates:
                if name in seen:
                    continue
                seen.add(name)
                uniq.append(name)

            for jname in uniq:
                dof = _joint_first_dof(jname)
                if dof is None:
                    continue
                if self._surf_dof[si] < 0:
                    self._surf_dof[si] = dof
                    self._surf_joint_name[si] = str(jname)
                axis = _axis_from_joint_name(jname)
                if axis == 1 and self._surf_dof_yaw[si] < 0:
                    self._surf_dof_yaw[si] = dof
                elif axis == 2 and self._surf_dof_pitch[si] < 0:
                    self._surf_dof_pitch[si] = dof

            # Final fallback: if axis-specific DOF is missing, use primary.
            if self._surf_dof_yaw[si] < 0:
                self._surf_dof_yaw[si] = self._surf_dof[si]
            if self._surf_dof_pitch[si] < 0:
                self._surf_dof_pitch[si] = self._surf_dof[si]

    def set_verbose_init(self, enabled: bool = True) -> None:
        self._verbose_init = bool(enabled)
        self._verbose_init_done = False
        if enabled:
            # Needed to have alpha/beta and decomposed aerodynamic quantities populated.
            self._aero_log = True

    def _infer_body_link_idx(self) -> int:
        for i, (_, _, _, k) in enumerate(self._geom):
            if int(k) == K_FUSELAGE:
                return int(self._aero_link_idx[i])
        for i, name in enumerate(self._aero_frames):
            if "fuselage" in name.lower():
                return int(self._aero_link_idx[i])
        return int(self._aero_link_idx[0]) if self._aero_link_idx else 0
    # ------------------------------------------------------------------
    # Helpers (DOF + omega + vectors)
    # ------------------------------------------------------------------
    @ti.func
    def _read_surface_angle(self, rigid: ti.template(), b: int, l: int) -> ti.f32:
        dof = self._surf_dof[l] 
        safe_dof = ti.max(dof, 0) 

        q = ti.cast(rigid.dofs_state.pos[safe_dof, b], ti.f32)
        angle = ti.select(dof >= 0, q, 0.0)
        return angle

    @ti.func
    def _read_surface_angle_axis(self, rigid: ti.template(), b: int, l: int, axis_mode: ti.i32) -> ti.f32:
        dof = self._surf_dof[l]
        if axis_mode == 1:
            dof = self._surf_dof_yaw[l]
        elif axis_mode == 2:
            dof = self._surf_dof_pitch[l]
        safe_dof = ti.max(dof, 0)
        q = ti.cast(rigid.dofs_state.pos[safe_dof, b], ti.f32)
        return ti.select(dof >= 0, q, 0.0)

    @ti.func
    def _omega_body(self, rigid: ti.template(), link_idx: int, b: int) -> ti.types.vector(3, ti.f32):
        # Angular velocity is stored in world coordinates; rotate it into link body frame.
        w_world = rigid.links_state.cd_ang[link_idx, b]
        quat = rigid.links_state.quat[link_idx, b]
        return ti.cast(gu.ti_inv_transform_by_quat(w_world, quat), ti.f32)

    @ti.func
    def _base_point_to_link_local(
        self,
        rigid: ti.template(),
        base_link: int,
        link_idx: int,
        b: int,
        p_base: ti.types.vector(3, ti.f32),
    ) -> ti.types.vector(3, ti.f32):
        p_world = rigid.links_state.pos[base_link, b] + gu.ti_transform_by_quat(
            p_base, rigid.links_state.quat[base_link, b]
        )
        rel_world = p_world - rigid.links_state.pos[link_idx, b]
        return ti.cast(gu.ti_inv_transform_by_quat(rel_world, rigid.links_state.quat[link_idx, b]), ti.f32)

    @ti.func
    def _link_point_to_base_local(
        self,
        rigid: ti.template(),
        base_link: int,
        link_idx: int,
        b: int,
        p_link: ti.types.vector(3, ti.f32),
    ) -> ti.types.vector(3, ti.f32):
        p_world = rigid.links_state.pos[link_idx, b] + gu.ti_transform_by_quat(
            p_link, rigid.links_state.quat[link_idx, b]
        )
        rel_world = p_world - rigid.links_state.pos[base_link, b]
        return ti.cast(gu.ti_inv_transform_by_quat(rel_world, rigid.links_state.quat[base_link, b]), ti.f32)

    @ti.func
    def _base_vector_to_link_local(
        self,
        rigid: ti.template(),
        base_link: int,
        link_idx: int,
        b: int,
        v_base: ti.types.vector(3, ti.f32),
    ) -> ti.types.vector(3, ti.f32):
        v_world = gu.ti_transform_by_quat(v_base, rigid.links_state.quat[base_link, b])
        return ti.cast(gu.ti_inv_transform_by_quat(v_world, rigid.links_state.quat[link_idx, b]), ti.f32)

    @ti.func
    def _motor_throttle_to_omega(self, b: int, u_thr: ti.f32) -> ti.f32:
        u = ti.math.clamp(u_thr, self.throttle_offset[b], 1.0)
        c0 = self.motor_omega_map[b]
        c1 = (c0 * (self.throttle_offset[b] * self.throttle_offset[b] - 1.0) + 1.0) / (
            1.0 - self.throttle_offset[b]
        )
        c2 = 1.0 - c0 - c1
        return ti.math.clamp(c0 * u * u + c1 * u + c2, 0.0, 1.0)

    @ti.func
    def _cross(self, a: ti.types.vector(3, ti.f32), b: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
        return ti.Vector([
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        ], dt=ti.f32)

    # ------------------------------------------------------------------
    # C++ math: sigmoid in degrees
    # ------------------------------------------------------------------
    @ti.func
    def _sigmoid(self, x: ti.f32, x_cut: ti.f32, M: ti.f32) -> ti.f32:
        rad2deg = 180.0 / ti.math.pi
        return (1.0 + ti.exp(-M * rad2deg * (x - x_cut)) + ti.exp(M * rad2deg * (x + x_cut))) / (
            (1.0 + ti.exp(-M * rad2deg * (x - x_cut))) * (1.0 + ti.exp(M * rad2deg * (x + x_cut)))
        )

    # ------------------------------------------------------------------
    # Effective surfaces + aero coefficients (aligned with dyn_casadi_mod_3D)
    # ------------------------------------------------------------------
    @ti.func
    def _seff_ratio(self, theta: ti.f32, theta_max: ti.f32, ratio_folded: ti.f32) -> ti.f32:
        ratio_clamped = ti.min(1.0, ti.max(0.0, ratio_folded))
        t = 0.0
        if theta_max > 1e-6:
            t = ti.min(1.0, ti.abs(theta) / theta_max)
        return 1.0 - t * (1.0 - ratio_clamped)

    @ti.func
    def _wing_sweep_scales(self, theta: ti.f32) -> ti.types.vector(2, ti.f32):
        theta_deg = theta * (180.0 / ti.math.pi)
        b_outer = (-0.0201 * theta_deg * theta_deg + 0.0904 * theta_deg + 165.15) / 1000.0
        s_outer = (-142.97 * theta_deg + 16290.0) / 1e6
        span_scale = b_outer / LISPARROW_B_OUTER_ZERO
        area_scale = s_outer / LISPARROW_S_OUTER_ZERO
        span_scale = ti.min(2.0, ti.max(0.05, span_scale))
        area_scale = ti.min(2.0, ti.max(0.05, area_scale))
        return ti.Vector([area_scale, span_scale], dt=ti.f32)

    @ti.func
    def _wing_dyn_ar(self, b: int, theta: ti.f32) -> ti.f32:
        b_wing, s_wing, _, _, _, _ = self._wing_properties(b, ti.abs(theta))
        return (b_wing * b_wing) / ti.max(1e-6, s_wing)

    @ti.func
    def _outer_wing_linear_area_scale(self, theta: ti.f32, theta_lo: ti.f32, theta_hi: ti.f32) -> ti.f32:
        scale = 1.0
        if theta_hi > theta_lo + 1e-6:
            t = (theta - theta_lo) / (theta_hi - theta_lo)
            scale = ti.min(1.0, ti.max(0.0, t))
        return scale

    @ti.func
    def _surface_scales(self, rigid: ti.template(), b: int, l: int) -> ti.types.vector(2, ti.f32):
        theta = self._read_surface_angle(rigid, b, l)
        theta_yaw = self._read_surface_angle_axis(rigid, b, l, 1)
        theta_pitch = self._read_surface_angle_axis(rigid, b, l, 2)
        theta_max = ti.max(0.0, self.theta_max_link[l])

        base_ratio = ti.min(1.0, ti.max(0.0, self.seff_ratio_base[l]))
        ratio_yaw = ti.min(1.0, ti.max(0.0, self.seff_ratio_yaw[l]))
        ratio_pitch = ti.min(1.0, ti.max(0.0, self.seff_ratio_pitch[l]))

        fold_ratio = 1.0
        mode = self.surface_axis_mode[l]
        if mode == 1:
            fold_ratio = self._seff_ratio(theta_yaw, theta_max, ratio_yaw)
        elif mode == 2:
            fold_ratio = self._seff_ratio(theta_pitch, theta_max, ratio_pitch)
        elif mode == 3:
            fold_ratio = self._seff_ratio(theta_yaw, theta_max, ratio_yaw) * self._seff_ratio(theta_pitch, theta_max, ratio_pitch)

        area_scale = base_ratio * fold_ratio
        span_scale = fold_ratio

        if self.kind[l] == K_WING and self.swept_wing[l] == 1:
            sweep_scales = self._wing_sweep_scales(ti.abs(theta_yaw))
            area_scale = base_ratio * sweep_scales.x
            span_scale = sweep_scales.y

        area_scale = ti.max(0.0, area_scale)
        span_scale = ti.max(0.0, span_scale)
        return ti.Vector([area_scale, span_scale], dt=ti.f32)

    @ti.func
    def _effective_surface_area(self, rigid: ti.template(), b: int, l: int) -> ti.f32:
        scales = self._surface_scales(rigid, b, l)
        return ti.max(0.0, self.area0[l] * scales.x)

    @ti.func
    def _effective_surface_span(self, rigid: ti.template(), b: int, l: int) -> ti.f32:
        scales = self._surface_scales(rigid, b, l)
        return ti.max(0.0, self.span0[l] * scales.y)

    @ti.func
    def _reynolds_factor(self, b: int, speed: ti.f32, chord: ti.f32) -> ti.f32:
        nu = ti.max(1e-9, self.nu[b])
        re_ref = ti.max(1e-6, self.Re_ref[b])
        Re = speed * chord / nu
        f_re = 1.0
        if Re < re_ref:
            ratio = (re_ref - Re) / re_ref
            f_re = 1.0 - ti.pow(ti.max(0.0, ratio), self.M_Re[b])
        return ti.min(1.0, ti.max(0.0, f_re))

    @ti.func
    def _qs_dyn_lift_delta(self, omega_y: ti.f32, chord_ref: ti.f32, speed: ti.f32) -> ti.f32:
        vref = ti.max(1e-6, speed)
        return (ti.math.pi * 0.5) * (-omega_y * chord_ref) / vref

    @ti.func
    def _aero_coeff(
        self,
        AR: ti.f32,
        alpha: ti.f32,
        alpha_stall: ti.f32,
        M_smooth: ti.f32,
        cl_alpha_2D: ti.f32,
        c_d_0: ti.f32,
        c_l_dyn_pre_st: ti.f32,
        f_re: ti.f32,
        ac_x: ti.f32,
        geo_x: ti.f32,
    ):
        k_cd = 1.0 - 0.41 * (1.0 - ti.exp(-17.0 / ti.max(1e-6, AR)))

        den = ti.max(1e-6, ti.math.pi - 2.0 * alpha_stall)
        w_pos = ti.cos(ti.math.pi * ((alpha - alpha_stall) / den) - ti.math.pi * 0.5)
        w_neg = ti.cos(ti.math.pi * ((-alpha - alpha_stall) / den) - ti.math.pi * 0.5)
        w = (1.0 / (1.0 + ti.exp(-20.0 * (alpha - alpha_stall)))) * w_pos + (
            1.0 / (1.0 + ti.exp(-20.0 * (-alpha - alpha_stall)))
        ) * w_neg

        sa = ti.sin(alpha)
        ca = ti.cos(alpha)
        c_l_st = f_re * (2.0 * sa * ca * (1.0 - w * (1.0 - k_cd)))
        c_d_st = 2.0 * f_re * (sa * sa) * (1.0 - w * (1.0 - k_cd))

        c_l_lin = f_re * ((cl_alpha_2D * AR) / (2.0 + ti.sqrt(AR * AR + 4.0)) * alpha)
        c_d_quad = c_d_0 + (c_l_lin * c_l_lin) / (ti.math.pi * ti.max(1e-6, AR))

        sig = self._sigmoid(alpha, alpha_stall, M_smooth)
        c_l = (1.0 - sig) * (c_l_lin + c_l_dyn_pre_st) + sig * c_l_st
        c_d = (1.0 - sig) * c_d_quad + sig * c_d_st

        d_x_m = -(ac_x + (2.0 * ti.abs(alpha) / ti.math.pi) * (geo_x - ac_x))
        return c_l, c_d, d_x_m


    # ------------------------------------------------------------------
    # C++ WingProperties(theta_sw)
    # ------------------------------------------------------------------
    @ti.func
    def _wing_root_geom(self, b: int):
        b_center = self.b_center[b]
        b_root_inner = self.b_root_inner[b]
        c_root_inner = self.c_root_inner[b]
        b_root_outer = self.b_root_outer[b]
        c_root_outer = self.c_root_outer[b]
        s_one_root = self.s_one_root[b]

        s_wing_root = (2.0 * s_one_root) + (b_center * c_root_inner)
        mac_root = (1.0 / ti.max(1e-6, s_wing_root)) * (
            b_root_inner * c_root_inner * c_root_inner
            + (b_root_outer - b_root_inner) * c_root_outer * c_root_outer
        )
        geo_ctr_y_root = (1.0 / ti.max(1e-6, s_wing_root)) * (
            b_root_inner * c_root_inner * b_root_inner / 4.0
            + (b_root_outer - b_root_inner) * c_root_outer * (b_root_outer + b_root_inner) / 4.0
        )
        geo_ctr_x_root = (1.0 / ti.max(1e-6, s_wing_root)) * (
            b_root_inner * c_root_inner * c_root_inner / 2.0
            + (b_root_outer - b_root_inner) * c_root_outer * (c_root_outer / 2.0)
        )
        return b_root_outer, s_wing_root, mac_root, geo_ctr_x_root, geo_ctr_y_root

    @ti.func
    def _wing_properties(self, b: int, theta_sw: ti.f32):
        theta_deg = theta_sw * (180.0 / ti.math.pi)

        b_outer = (-0.0201 * theta_deg * theta_deg + 0.0904 * theta_deg + 165.15) / 1000.0
        s_outer = (-142.97 * theta_deg + 16290.0) / 1e6
        ac_x_outer = self.c_root_outer[b] / 6.0
        cpx_outer = ((-0.003 * theta_deg * theta_deg) + 0.7136 * theta_deg + 59.55) / 1000.0
        cpy_outer = (-0.00397 * theta_deg * theta_deg - 0.251 * theta_deg + 62.78) / 1000.0

        b_root_outer, s_root, mac_root, cpx_root, cpy_root = self._wing_root_geom(b)

        b_wing = b_root_outer + 2.0 * b_outer
        s_wing = s_root + 2.0 * s_outer

        ac_x_wing = (((mac_root / 4.0) * s_root) + (ac_x_outer * 2.0 * s_outer)) / ti.max(1e-6, s_wing)
        mac_wing = ((mac_root * s_root) + ((s_outer / ti.max(1e-6, b_outer)) * 2.0 * s_outer)) / ti.max(1e-6, s_wing)

        cpx_wing = ((cpx_root * s_root) + (cpx_outer * 2.0 * s_outer)) / ti.max(1e-6, s_wing)
        cpy_wing = ((cpy_root * s_root) + ((cpy_outer + b_root_outer / 2.0) * 2.0 * s_outer)) / ti.max(1e-6, s_wing)

        return b_wing, s_wing, mac_wing, ac_x_wing, cpx_wing, cpy_wing

    # ------------------------------------------------------------------
    # C++ WingAeroCenter
    # ------------------------------------------------------------------
    @ti.func
    def _wing_aero_center_linkframe(
        self,
        vel_link: ti.types.vector(3, ti.f32),
        chord_link: ti.f32,
        chord_ref_cpp: ti.f32,
        ac_x_cpp: ti.f32,
        cpx_cpp: ti.f32,
        alpha_stall: ti.f32,
        calculate_rot_pt: ti.i32,
    ):
        """
        WingAeroCenter rewritten for the quarter-chord frame convention:
        - returns a point in the LINK frame
        - frame origin is at 25% chord from LE
        - CP/rot point moves ONLY along local X (chord direction)
        - uses Lisparrow ac_x/cpx behavior, but mapped via chord fractions
        """
        # Robust alpha (avoid vel.x sign causing alpha -> 90deg instantly)
        alpha = ti.abs(ti.atan2(vel_link.z, ti.abs(vel_link.x) + 1e-6))
        # Convert C++ distances to fractions using the same chord reference used by the fit.
        cref = ti.max(1e-6, chord_ref_cpp)
        ac_frac = ti.min(1.0, ti.max(0.0, ac_x_cpp / cref))
        cp_frac = ti.min(1.0, ti.max(0.0, cpx_cpp / cref))

        # Ensure ordering: CP limit should be downstream of AC in most cases.
        cp_frac = ti.max(cp_frac, ac_frac)

        # Pick base fraction: rot point uses cp_frac; force point uses ac_frac (pre-stall).
        frac = ti.select(calculate_rot_pt == 1, cp_frac, ac_frac)

        # Post-stall shift for FORCE point only (match original behavior).
        if calculate_rot_pt == 0:
            if alpha > alpha_stall:
                t = (alpha - alpha_stall) / ti.max(1e-6, (ti.math.pi * 0.5 - alpha_stall))
                t = ti.min(1.0, ti.max(0.0, t))
                frac = ac_frac + t * (cp_frac - ac_frac)

        # Convert LE-referenced fraction into quarter-chord-referenced position.
        x = ti.max(1e-6, chord_link) * (0.25 - frac)
        return ti.Vector([x, 0.0, 0.0], dt=ti.f32)

    # ------------------------------------------------------------------
    # Wing / tail / rudder forces (dyn_casadi_mod_3D-consistent)
    # ------------------------------------------------------------------
    @ti.func
    def _wing_force_with_thrust(
        self,
        b: int,
        flow_free: ti.types.vector(3, ti.f32),
        flow_slip: ti.types.vector(3, ti.f32),
        s_w: ti.f32,
        s_slip: ti.f32,
        AR: ti.f32,
        chord_ref: ti.f32,
        ac_x: ti.f32,
        geo_x: ti.f32,
        alpha_stall: ti.f32,
        M_smooth: ti.f32,
        cl_alpha_2D: ti.f32,
        c_d_0: ti.f32,
        omega_y: ti.f32,
    ):
        V = flow_free.norm() + 1e-9
        V_slip = flow_slip.norm() + 1e-9

        alpha = ti.atan2(flow_free.z, -flow_free.x)
        alpha_slip = ti.atan2(flow_slip.z, -flow_slip.x)
        beta = ti.asin(ti.math.clamp(flow_free.y / V, -1.0, 1.0))
        beta_slip = ti.asin(ti.math.clamp(flow_slip.y / V_slip, -1.0, 1.0))

        f_re = self._reynolds_factor(b, V, chord_ref)
        f_re_slip = self._reynolds_factor(b, V_slip, chord_ref)

        c_l_dyn = self._qs_dyn_lift_delta(omega_y, chord_ref, V)
        c_l_dyn_slip = self._qs_dyn_lift_delta(omega_y, chord_ref, V_slip)

        c_l, c_d, d_x = self._aero_coeff(
            AR, alpha, alpha_stall, M_smooth, cl_alpha_2D, c_d_0, c_l_dyn, f_re, ac_x, geo_x
        )
        c_l_s, c_d_s, d_x_s = self._aero_coeff(
            AR, alpha_slip, alpha_stall, M_smooth, cl_alpha_2D, c_d_0, c_l_dyn_slip, f_re_slip, ac_x, geo_x
        )

        S_slip = ti.min(ti.max(0.0, s_slip), ti.max(1e-6, s_w))
        S_free = ti.max(1e-6, s_w - S_slip)

        cosb = ti.cos(beta)
        cosb2 = cosb * cosb
        cosb_s = ti.cos(beta_slip)
        cosb2_s = cosb_s * cosb_s

        q_free = 0.5 * V * V * S_free
        q_slip = 0.5 * V_slip * V_slip * S_slip

        # Casadi formulas assume flow entering from -x. If the local frame has
        # opposite x convention (flow from +x), drag terms must flip sign.
        sgn_free = ti.select(-flow_free.x >= 0.0, 1.0, -1.0)
        sgn_slip = ti.select(-flow_slip.x >= 0.0, 1.0, -1.0)

        fx_free = q_free * cosb2 * (c_l * ti.sin(alpha) - sgn_free * c_d * ti.cos(alpha))
        fz_free = q_free * cosb2 * (c_l * ti.cos(alpha) + sgn_free * c_d * ti.sin(alpha))
        fx_slip = q_slip * cosb2_s * (c_l_s * ti.sin(alpha_slip) - sgn_slip * c_d_s * ti.cos(alpha_slip))
        fz_slip = q_slip * cosb2_s * (c_l_s * ti.cos(alpha_slip) + sgn_slip * c_d_s * ti.sin(alpha_slip))

        F = ti.Vector([fx_free + fx_slip, 0.0, fz_free + fz_slip], dt=ti.f32)
        # Lisparrow wing links are defined with the local origin close to the
        # leading-edge / hinge location, not at quarter chord. Keep CP in the
        # link frame directly so low-alpha forces act near -0.25*c and shift
        # aft toward -0.5*c as alpha grows.
        cp_x = (d_x * S_free + d_x_s * S_slip) / ti.max(1e-6, s_w)
        return F, cp_x, alpha, beta

    @ti.func
    def _hor_tail_force(self, b: int, flow_vel: ti.types.vector(3, ti.f32), s_ref: ti.f32, chord_ref: ti.f32, elevator: ti.f32):
        V = flow_vel.norm() + 1e-9
        alpha_raw = ti.atan2(flow_vel.z, -flow_vel.x)
        alpha_eff = self.k_alpha_elev[b] * elevator + alpha_raw

        sa = ti.sin(alpha_eff)
        ca = ti.cos(alpha_eff)
        c_l = 2.0 * sa * ca
        c_d = self.c_d_0_tail[b] + 2.0 * (sa * sa)

        q = 0.5 * V * V * s_ref
        sgn = ti.select(-flow_vel.x >= 0.0, 1.0, -1.0)
        fx = q * (c_l * ti.sin(alpha_eff) - sgn * c_d * ti.cos(alpha_eff))
        fz = q * (c_l * ti.cos(alpha_eff) + sgn * c_d * ti.sin(alpha_eff))
        d_x_t = self.c_hor_tail[b] * 0.25 + (2.0 * ti.abs(alpha_eff) / ti.math.pi) * (self.c_hor_tail[b] * 0.25)
        cp_body_x = self.pos_ele_le_x[b] - d_x_t
        return ti.Vector([fx, 0.0, fz], dt=ti.f32), cp_body_x, alpha_raw, alpha_eff

    @ti.func
    def _vert_tail_force(self, b: int, flow_vel: ti.types.vector(3, ti.f32), s_ref: ti.f32, chord_ref: ti.f32, rudder: ti.f32):
        V = flow_vel.norm() + 1e-9
        alpha_raw = ti.asin(ti.math.clamp(flow_vel.y / V, -1.0, 1.0))
        alpha_eff = -(self.k_alpha_rudder[b] * rudder) + alpha_raw

        sa = ti.sin(alpha_eff)
        ca = ti.cos(alpha_eff)
        c_l = 2.0 * sa * ca
        c_d = 2.0 * (sa * sa)

        q = 0.5 * V * V * s_ref
        sgn = ti.select(-flow_vel.x >= 0.0, 1.0, -1.0)
        fx = q * (c_l * ti.sin(alpha_eff) - sgn * c_d * ti.cos(alpha_eff))
        fy = q * (c_l * ti.cos(alpha_eff) + sgn * c_d * ti.sin(alpha_eff))
        d_x_t = self.c_vert_tail[b] * 0.25 + (2.0 * ti.abs(alpha_eff) / ti.math.pi) * (self.c_vert_tail[b] * 0.25)
        cp_body_x = self.pos_rud_le_x[b] - d_x_t
        return ti.Vector([fx, fy, 0.0], dt=ti.f32), cp_body_x, alpha_raw, alpha_eff

    @ti.kernel
    def _init_prev_states(self):
        for b in range(self.B):
            self.flow_prev[b] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            self.alpha_dot_prev[b] = 0.0
            self.omega_mot_norm[b] = 0.0
    # ------------------------------------------------------------------
    # Main kernel: compute per surface
    # ------------------------------------------------------------------
    @ti.kernel
    def _aero_compute_kernel(self, rigid: ti.template()):
        # Reset per-surface outputs.
        for b, l in ti.ndrange(self.B, self.L):
            # Ensure debug buffers are always defined (when enabled)
            if ti.static(self._aero_log):
                self.alpha_dbg[b, l] = ti.cast(0.0, ti.f16)
                self.beta_dbg[b, l] = ti.cast(0.0, ti.f16)
                self.drag_dbg[b, l] = ti.cast(0.0, ti.f16)
                self.lift_dbg[b, l] = ti.cast(0.0, ti.f16)
                self.side_force_dbg[b, l] = ti.cast(0.0, ti.f16)
                self.alpha_tail_raw_dbg[b, l] = ti.cast(0.0, ti.f16)
                self.downwash_eps_dbg[b, l] = ti.cast(0.0, ti.f16)
                self.cl_wing_for_tail_dbg[b, l] = ti.cast(0.0, ti.f16)
                self.k_eps_tail_dbg[b, l] = ti.cast(0.0, ti.f16)
            self.force_b[b, l] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            self.cp_b[b, l] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            self.flow_dbg[b, l] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            self.joint_angle_dbg[b, l] = 0.0

        for b in range(self.B):
            base_link = self.body_link_idx[None]

            # Airflow in body frame; by convention flow = -body velocity.
            flow_base = self._get_wind_in_body(rigid, base_link, b)

            # Forward speed along body x-axis.
            u = -flow_base.x

            # dyn_casadi motor model: throttle -> desired normalized omega,
            # first-order motor lag, then thrust proportional to omega^2.
            u_thr = ti.math.clamp(ti.cast(self._thr_raw[b], ti.f32), self.throttle_offset[b], 1.0)
            omega_des = self._motor_throttle_to_omega(b, u_thr)
            motor_tau_inv = ti.max(1e-6, self.motor_tau_inv[b])
            motor_alpha = 1.0 - ti.exp(-self._substep_dt * motor_tau_inv)
            self.omega_mot_norm[b] += motor_alpha * (omega_des - self.omega_mot_norm[b])
            self.omega_mot_norm[b] = ti.math.clamp(self.omega_mot_norm[b], 0.0, 1.0)
            self._thr_flt[b] = self.omega_mot_norm[b]
            thrust_mag = (self.omega_mot_norm[b] * self.omega_mot_norm[b]) * ti.max(0.0, self.max_thrust[b])
            thrust = thrust_mag * self.prop_thrust_sign[b]

            R_prop = ti.max(1e-6, self.prop_radius[b])
            rho = ti.max(1e-6, self.rho[b])

            # Propeller-induced wake velocity (aligned with dyn_casadi_mod_3D).
            disc = u * u + (2.0 * ti.abs(thrust)) / (rho * ti.math.pi * (R_prop * R_prop))
            prop_flow_ind_wake = (-u + ti.sqrt(disc)) * 0.5
            v_slip_wing = self.K_slip_wing[b] * prop_flow_ind_wake
            v_slip_tail = self.K_slip_tail[b] * prop_flow_ind_wake

            # Precompute stall radians.
            alpha_stall_wing = (self.alpha_stall_wing_deg[b] * ti.math.pi) / 180.0
            omega_base = self._omega_body(rigid, base_link, b)

            for l in range(self.L):
                k = self.kind[l]
                link_idx = self._link_idx[l]

                # Wind in THIS surface's link body frame (mydrone convention).
                flow_link = self._get_wind_in_body(rigid, link_idx, b)
                omega_link = self._omega_body(rigid, link_idx, b)
                self.flow_dbg[b, l] = flow_link
                self.joint_angle_dbg[b, l] = self._read_surface_angle(rigid, b, l)
                # ===== WINGS =====
                if k == K_WING:
                    S_eff = self._effective_surface_area(rigid, b, l)
                    span_eff = self._effective_surface_span(rigid, b, l)
                    chord_eff = ti.max(1e-6, self.chord0[l])
                    if S_eff <= 1e-9 or span_eff <= 1e-9:
                        self.force_b[b, l] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                        self.cp_b[b, l] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                        if ti.static(self._aero_log):
                            self.alpha_dbg[b, l] = ti.cast(0.0, ti.f16)
                            self.beta_dbg[b, l] = ti.cast(0.0, ti.f16)
                            self.drag_dbg[b, l] = ti.cast(0.0, ti.f16)
                            self.lift_dbg[b, l] = ti.cast(0.0, ti.f16)
                            self.side_force_dbg[b, l] = ti.cast(0.0, ti.f16)
                        continue
                    theta_yaw = self._read_surface_angle_axis(rigid, b, l, 1)
                    AR_eff = self._wing_dyn_ar(b, theta_yaw)
                    ac_x = 0.25 * chord_eff
                    geo_x = 0.50 * chord_eff
                    chord_ref = chord_eff
                    omega_y = omega_base.y

                    # Dyn-casadi evaluates wing coefficients in the vehicle/body
                    # frame.  Keep that aerodynamic frame here, then transform
                    # the resulting force to the moving physical link frame for
                    # Genesis application.  This avoids swept outer-wing yaw
                    # rotating body sideslip into a false alpha jump.
                    cp_y = 0.0
                    if self.side[l] != 0:
                        cp_y = 0.5 * span_eff * ti.cast(self.side[l], ti.f32)
                    pos_ac_link = ti.Vector([-ac_x, cp_y, 0.0], dt=ti.f32)
                    pos_ac_base = self._link_point_to_base_local(rigid, base_link, link_idx, b, pos_ac_link)
                    flow_rot_base = -self._cross(omega_base, pos_ac_base)
                    flow_free_base = flow_base + flow_rot_base
                    flow_slip_base = flow_free_base + ti.Vector([-v_slip_wing, 0.0, 0.0], dt=ti.f32)

                    s_slip = 0.0
                    if self.side[l] == 0:
                        s_slip = ti.min(S_eff, 2.0 * R_prop * self.c_root_inner[b])
                    F_base, cp_x, alpha_dbg_w, beta_dbg_w = self._wing_force_with_thrust(
                        b,
                        flow_free_base,
                        flow_slip_base,
                        S_eff,
                        s_slip,
                        AR_eff,
                        chord_ref,
                        ac_x,
                        geo_x,
                        alpha_stall_wing,
                        self.M_smooth[b],
                        self.cl_alpha_wing_2D[b],
                        self.c_d_0_wing[b],
                        omega_y,
                    )

                    cap = self.force_cap[b]
                    F_base *= rho
                    F_base *= ti.min(1.0, cap / (F_base.norm() + 1e-9))
                    F = self._base_vector_to_link_local(rigid, base_link, link_idx, b, F_base)
                    self.flow_dbg[b, l] = flow_free_base

                    # Outer wings: use Lisparrow fit outer semi-span b_outer(theta),
                    # not a stale hardcoded fit, to place CP at panel mid-span.
                    pos_force = ti.Vector([cp_x, cp_y, 0.0], dt=ti.f32)

                    self.force_b[b, l] = F
                    self.cp_b[b, l] = pos_force

                    if ti.static(self._aero_log):
                        v_dbg = flow_free_base
                        drag, lift, side = self._dbg_decompose_force(F_base, v_dbg)
                        self.alpha_dbg[b, l] = ti.cast(alpha_dbg_w, ti.f16)
                        self.beta_dbg[b, l] = ti.cast(beta_dbg_w, ti.f16)
                        self.drag_dbg[b, l] = ti.cast(drag, ti.f16)
                        self.lift_dbg[b, l] = ti.cast(lift, ti.f16)
                        self.side_force_dbg[b, l] = ti.cast(side, ti.f16)

                # ===== ELEVATOR / HORIZONTAL TAIL =====
                elif k == K_ELEVATOR:
                    elevator = self._read_surface_angle_axis(rigid, b, l, 2)

                    chord_eff = ti.max(1e-6, self.c_hor_tail[b])
                    S_eff = self.s_hor_tail[b]
                    cp_vel_ele_body = ti.Vector(
                        [self.pos_ele_le_x[b] - 0.5 * self.c_ele[b], self.pos_ele_le_y[b], self.pos_ele_le_z[b]],
                        dt=ti.f32,
                    )
                    flow_rot_ele = -self._cross(omega_base, cp_vel_ele_body)
                    flow_downwash = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                    flow_ele_base = flow_base + flow_rot_ele + flow_downwash + ti.Vector([-v_slip_tail, 0.0, 0.0], dt=ti.f32)

                    F_ele_base, cp_x_ele, alpha_raw_ele, alpha_eff_ele = self._hor_tail_force(
                        b, flow_ele_base, S_eff, chord_eff, elevator
                    )
                    F_ele_base *= rho
                    F_ele = self._base_vector_to_link_local(rigid, base_link, link_idx, b, F_ele_base)
                    cp_ele_body = ti.Vector([cp_x_ele, self.pos_ele_le_y[b], self.pos_ele_le_z[b]], dt=ti.f32)
                    pos_ele = self._base_point_to_link_local(rigid, base_link, link_idx, b, cp_ele_body)

                    cap = self.force_cap[b]
                    F_ele *= ti.min(1.0, cap / (F_ele.norm() + 1e-9))

                    self.force_b[b, l] = F_ele
                    self.cp_b[b, l] = pos_ele

                    if ti.static(self._aero_log):
                        v_dbg = flow_ele_base
                        vmod = v_dbg.norm() + 1e-9
                        beta_dbg = ti.asin(ti.math.clamp(v_dbg.y / vmod, -1.0, 1.0))
                        drag, lift, side = self._dbg_decompose_force(F_ele, v_dbg)
                        self.alpha_tail_raw_dbg[b, l] = ti.cast(alpha_raw_ele, ti.f16)
                        self.alpha_dbg[b, l] = ti.cast(alpha_eff_ele, ti.f16)
                        self.beta_dbg[b, l] = ti.cast(beta_dbg, ti.f16)
                        self.drag_dbg[b, l] = ti.cast(drag, ti.f16)
                        self.lift_dbg[b, l] = ti.cast(lift, ti.f16)
                        self.side_force_dbg[b, l] = ti.cast(side, ti.f16)

                # ===== RUDDER / VERTICAL TAIL =====
                elif k == K_RUDDER:
                    rudder = self._read_surface_angle_axis(rigid, b, l, 1)

                    chord_eff = ti.max(1e-6, self.c_vert_tail[b])
                    S_eff = self.s_vert_tail[b]
                    cp_vel_rud_body = ti.Vector(
                        [self.pos_rud_le_x[b] - 0.5 * self.c_rud[b], self.pos_rud_le_y[b], self.pos_rud_le_z[b]],
                        dt=ti.f32,
                    )
                    flow_rot_rud = -self._cross(omega_base, cp_vel_rud_body)
                    flow_rud_base = flow_base + flow_rot_rud + ti.Vector([-v_slip_tail, 0.0, 0.0], dt=ti.f32)

                    F_rud_base, cp_x_rud, alpha_raw_rud, alpha_eff_rud = self._vert_tail_force(
                        b, flow_rud_base, S_eff, chord_eff, rudder
                    )
                    F_rud_base *= rho
                    F_rud = self._base_vector_to_link_local(rigid, base_link, link_idx, b, F_rud_base)
                    cp_rud_body = ti.Vector([cp_x_rud, self.pos_rud_le_y[b], self.pos_rud_le_z[b]], dt=ti.f32)
                    pos_rud = self._base_point_to_link_local(rigid, base_link, link_idx, b, cp_rud_body)

                    cap = self.force_cap[b]
                    F_rud *= ti.min(1.0, cap / (F_rud.norm() + 1e-9))

                    self.force_b[b, l] = F_rud
                    self.cp_b[b, l] = pos_rud

                    if ti.static(self._aero_log):
                        v_dbg = flow_rud_base
                        vmod = v_dbg.norm() + 1e-9
                        beta_dbg = ti.asin(ti.math.clamp(v_dbg.y / vmod, -1.0, 1.0))
                        drag, lift, side = self._dbg_decompose_force(F_rud, v_dbg)
                        self.alpha_tail_raw_dbg[b, l] = ti.cast(alpha_raw_rud, ti.f16)
                        self.alpha_dbg[b, l] = ti.cast(alpha_eff_rud, ti.f16)
                        self.beta_dbg[b, l] = ti.cast(beta_dbg, ti.f16)
                        self.drag_dbg[b, l] = ti.cast(drag, ti.f16)
                        self.lift_dbg[b, l] = ti.cast(lift, ti.f16)
                        self.side_force_dbg[b, l] = ti.cast(side, ti.f16)

                # ===== PROPELLER (thrust only) =====
                elif k == K_PROPELLER:
                    # Lisparrow's propeller CAD axis is aligned with local +X:
                    # the prop mesh is long in X and mounted at the nose (+X of the fuselage).
                    pos_prop_cg = self._base_point_to_link_local(
                        rigid,
                        base_link,
                        link_idx,
                        b,
                        ti.Vector([self.pos_prop_x[b], self.pos_prop_y[b], self.pos_prop_z[b]], dt=ti.f32),
                    )
                    F_thr = ti.Vector([thrust, 0.0, 0.0], dt=ti.f32)

                    cap = self.force_cap[b]
                    F_thr *= ti.min(1.0, cap / (F_thr.norm() + 1e-9))

                    self.force_b[b, l] = F_thr
                    self.cp_b[b, l] = pos_prop_cg

                    if ti.static(self._aero_log):
                        # For prop, treat thrust as "lift" for visualization purposes
                        self.alpha_dbg[b, l] = ti.cast(0.0, ti.f16)
                        self.beta_dbg[b, l] = ti.cast(0.0, ti.f16)
                        self.drag_dbg[b, l] = ti.cast(0.0, ti.f16)
                        self.lift_dbg[b, l] = ti.cast(F_thr.norm(), ti.f16)
                        self.side_force_dbg[b, l] = ti.cast(0.0, ti.f16)

                else:
                    # Fuselage/other surfaces are not modeled in this C++-equivalent solver.
                    pass

            # Store previous flow if dynamic aero is enabled later.
            self.flow_prev[b] = flow_base
            self.alpha_dot_prev[b] = 0.0

    def _aero_step(self):
        super()._aero_step()
        if self._verbose_init and (not self._verbose_init_done):
            self._print_verbose_init_report()
            self._verbose_init_done = True

    def _print_verbose_init_report(self) -> None:
        if not self._aero_targets:
            return
        if self.L <= 0:
            return
        entity = self._aero_targets[0]
        L = len(self._aero_frames)
        fb = self.force_b.to_torch(device=self._aero_device)[0, :L, :].detach().cpu().numpy()
        cp = self.cp_b.to_torch(device=self._aero_device)[0, :L, :].detach().cpu().numpy()
        alpha = self.alpha_dbg.to_torch(device=self._aero_device)[0, :L].detach().cpu().numpy()
        beta = self.beta_dbg.to_torch(device=self._aero_device)[0, :L].detach().cpu().numpy()
        flow = self.flow_dbg.to_torch(device=self._aero_device)[0, :L, :].detach().cpu().numpy()
        joint_q = self.joint_angle_dbg.to_torch(device=self._aero_device)[0, :L].detach().cpu().numpy()

        print("\n--- Lisparrow Verbose Init (env=0) ---")
        for i, name in enumerate(self._aero_frames):
            link = entity.get_link(name)
            pos_world_t = link.get_pos(envs_idx=0)
            quat_world_t = link.get_quat(envs_idx=0)
            pos_world = pos_world_t.detach().cpu().numpy()
            quat_world = quat_world_t.detach().cpu().numpy()
            if pos_world.ndim > 1:
                pos_world = pos_world[0]
            if quat_world.ndim > 1:
                quat_world = quat_world[0]
            cp_local = cp[i].astype(np.float32)
            cp_world = pos_world + transform_by_quat(
                torch.from_numpy(cp_local[None, :]),
                torch.from_numpy(quat_world.astype(np.float32)[None, :]),
            )[0].detach().cpu().numpy()

            f_local = fb[i]
            fmag = float(np.linalg.norm(f_local))
            jname = self._surf_joint_name[i] if i < len(self._surf_joint_name) else ""
            jinfo = f"{jname}={joint_q[i]:+.4f} rad" if jname else "none"
            print(
                f"[InitVerbose] link={name} | joint={jinfo} | "
                f"pos_w=({pos_world[0]:+.4f},{pos_world[1]:+.4f},{pos_world[2]:+.4f}) | "
                f"cp_l=({cp_local[0]:+.4f},{cp_local[1]:+.4f},{cp_local[2]:+.4f}) | "
                f"cp_w=({cp_world[0]:+.4f},{cp_world[1]:+.4f},{cp_world[2]:+.4f}) | "
                f"F_l=({f_local[0]:+.4f},{f_local[1]:+.4f},{f_local[2]:+.4f}) | |F|={fmag:.4f} N | "
                f"alpha={float(alpha[i]):+.4f} rad | beta={float(beta[i]):+.4f} rad | "
                f"flow_l=({flow[i,0]:+.4f},{flow[i,1]:+.4f},{flow[i,2]:+.4f})"
            )
        print("--- End Lisparrow Verbose Init ---\n")



# Keep the default solver export pattern consistent with other drone solvers.
AeroSolver = LisparrowAeroSolver
