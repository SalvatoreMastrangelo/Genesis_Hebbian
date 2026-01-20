import copy
from dataclasses import dataclass
import math
import gstaichi as ti

from genesis.engine.solvers.base_aero_solver import BaseAeroSolver
from genesis.engine.entities import RigidEntity
from genesis.assets.urdf.mydrone.drone import DroneAeroModel, SurfaceKind
from genesis.utils import geom as gu


# Numeric codes from DroneAeroModel._build_geom (kept local for Taichi kernels).
K_FUSELAGE = 0
K_WING = 1
K_ELEVATOR = 2
K_RUDDER = 3
K_PROPELLER = 4


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


class LisparrowAeroDefaults:
    """
    Hard-coded baseline parameters for the C++-equivalent Lisparrow solver.

    Values here are safe defaults and are overridden by DroneAeroModel parameters
    when provided. Root-wing geometry entries are set to 0.0 to allow automatic
    derivation from URDF geometry if the YAML does not supply them.
    """

    # Outer wing geometry at theta_sw = 0 deg (from the C++ fit).
    B_OUTER_ZERO = 0.165
    S_OUTER_ZERO = 0.016309

    FLUID = {
        "rho": 1.225,
    }

    PROP = {
        "prop_radius": 0.0,
        "max_thrust": 0.0,
        "prop_cutoff_hz": 30.0,
        "kappa_prop": 0.0,
    }

    SLIPSTREAM = {
        "K_slip_tail": 1.0,
        "K_slip_wing": 1.0,
    }

    EXTENSIONS = {
        "enable_reynolds": 0.0,
        "enable_qs_dyn_lift": 0.0,
        "enable_cos2_beta": 0.0,
        "use_slip_two_segment_sum": 0.0,
        "nu": 1.5e-5,
        "Re_ref": 100000.0,
        "M_Re": 2.5,
    }

    WING = {
        "M_smooth": 0.2,
        "alpha_stall_wing_deg": 14.0,
        "cl_alpha_wing_2D": 5.73,
        "c_d_0_wing": 0.12,
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
        "b_center": 0.0,
        "b_root_inner": 0.0,
        "c_root_inner": 0.0,
        "b_root_outer": 0.0,
        "c_root_outer": 0.0,
        "s_one_root": 0.0,
    }

    REF_POS = {
        "pos_cg_x": -0.075,
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
        "aero_frame_fuselage": {
            "type": "fuselage",
            "s_folded": 1.0,
        },
        "aero_frame_left_wing": {
            "type": "wing",
            "actuator_yaw": "X10_servo",
            "actuator_pitch": "X08_servo",
        },
        "aero_frame_right_wing": {
            "type": "wing",
            "actuator_yaw": "X10_servo",
            "actuator_pitch": "X08_servo",
        },
        "aero_frame_elevator_left": {
            "type": "elevator",
            "actuator_pitch": "X08_servo",
            "actuator_yaw": None,
        },
        "aero_frame_elevator_right": {
            "type": "elevator",
            "actuator_pitch": "X08_servo",
            "actuator_yaw": None,
        },
        "aero_frame_rudder": {
            "type": "rudder",
            "actuator_pitch": None,
            "actuator_yaw": "X08_servo",
        },
        "prop_frame_fuselage_0": {
            "type": "propeller",
            "actuator": "morphing_prop",
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
        self._randomizable_param_names = LisparrowAeroDefaults.randomizable_param_names()
        self.noise_sigma_mag = DroneAeroModel.NOISE_DEFAULTS["sigma_mag"]
        self.noise_sigma_dir = DroneAeroModel.NOISE_DEFAULTS["sigma_dir"]
        self.noise_sigma_param = DroneAeroModel.NOISE_DEFAULTS["sigma_param"]
        self.noise_sigma_cp = DroneAeroModel.NOISE_DEFAULTS["sigma_cp"]

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
        base_param_overrides = dict(getattr(model, "base_param_overrides", {}) or {})
        noise_params = dict(getattr(model, "noise_params", {}) or {})

        side = [0 for _ in range(self.L)]
        slip_code = [0 for _ in range(self.L)]
        fus_width = 0.0
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
        frames = list(model.frames)
        geom = list(model.geom)
        surface_kinds = [self._surface_kind_code(k) for k in getattr(model, "surface_kinds", [])]
        link_idx = model.link_indices(entity)

        keep = []
        for i, name in enumerate(frames):
            lname = str(name).lower()
            if lname.startswith("aero_frame_") or lname.startswith("prop_frame_") or lname.startswith("propeller"):
                keep.append(i)

        if not keep:
            raise RuntimeError("Lisparrow solver requires aero_frame_* links (or propeller) in the drone model.")

        self._aero_frames = [frames[i] for i in keep]
        self._geom = [geom[i] for i in keep]
        self._surface_kinds = [surface_kinds[i] for i in keep] if surface_kinds else []
        self._aero_link_idx = [link_idx[i] for i in keep]
        self._geom_uses_span = False  # DroneAeroModel.geom is (S, AR, chord, kind)

        if model.noise_params:
            self.noise_sigma_mag = float(model.noise_params.get("sigma_mag", self.noise_sigma_mag))
            self.noise_sigma_dir = float(model.noise_params.get("sigma_dir", self.noise_sigma_dir))
            self.noise_sigma_param = float(model.noise_params.get("sigma_param", self.noise_sigma_param))
            self.noise_sigma_cp = float(model.noise_params.get("sigma_cp", self.noise_sigma_cp))

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
        params = dict(self._aero_base)
        if model is not None:
            params.update(getattr(model, "base_params", {}) or {})
            params.update(getattr(model, "base_param_overrides", {}) or {})
            if getattr(model, "prop_radius", 0.0) > 0.0 and params.get("prop_radius", 0.0) <= 0.0:
                params["prop_radius"] = float(model.prop_radius)
        params = {
            name: params[name]
            for name in LisparrowAeroDefaults.param_names()
            if name in params
        }
        self._derive_wing_root_params(params)
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

        # Actuator DOF mapping (wing sweep / elevator / rudder).
        self._surf_dof = ti.field(ti.i32, shape=(L,))
        for i in range(L):
            self._surf_dof[i] = -1

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

    def _init_simple_fields(self, bound: BoundTarget) -> None:
        """
        Phase 3: populate per-surface metadata and initialize cached state.
        """
        self._init_surface_metadata()
        self._init_surface_params()
        self._init_prev_states()

    def _init_surface_metadata(self) -> None:
        body_link = self._infer_body_link_idx()
        self.body_link_idx[None] = int(body_link)

        for i, (s, ar_or_span, chord, k) in enumerate(self._geom):
            span = self._geom_span(s, ar_or_span)
            self.area0[i] = float(s)
            self.span0[i] = float(span)
            self.chord0[i] = float(chord)
            self.kind[i] = int(k)
            self._link_idx[i] = int(self._aero_link_idx[i])

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
        if model is None:
            return
        for frame, info in getattr(model, "actuators", {}).items():
            jname = getattr(info, "joint_name", None)
            if not jname:
                continue
            try:
                j = entity.get_joint(jname)
            except Exception:
                continue
            dofs = getattr(j, "dofs_idx_local", None)
            if not dofs:
                continue
            dof0 = int(dofs[0] if isinstance(dofs, (list, tuple)) else dofs)
            try:
                si = self._aero_frames.index(frame)
            except ValueError:
                continue
            self._surf_dof[si] = dof0

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
    def _omega_body(self, rigid: ti.template(), link_idx: int, b: int) -> ti.types.vector(3, ti.f32):
        # Angular velocity is stored in world coordinates; rotate it into link body frame.
        w_world = rigid.links_state.cd_ang[link_idx, b]
        quat = rigid.links_state.quat[link_idx, b]
        return ti.cast(gu.ti_inv_transform_by_quat(w_world, quat), ti.f32)

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
    # Optional extensions: Reynolds degradation, QS dyn lift, cos^2(beta)
    # ------------------------------------------------------------------
    @ti.func
    def _reynolds_factor(self, b: int, speed: ti.f32, chord: ti.f32) -> ti.f32:
        f_re = 1.0
        if self.enable_reynolds[b] > 0.5:
            nu = ti.max(1e-9, self.nu[b])
            re_ref = ti.max(1e-6, self.Re_ref[b])
            Re = speed * chord / nu
            if Re < re_ref:
                ratio = (re_ref - Re) / re_ref
                f_re = 1.0 - ti.pow(ti.max(0.0, ratio), self.M_Re[b])
            else:
                f_re = 1.0
            f_re = ti.min(1.0, ti.max(0.0, f_re))
        return f_re

    @ti.func
    def _qs_dyn_lift_delta(self, b: int, omega_y: ti.f32, chord_ref: ti.f32, speed: ti.f32) -> ti.f32:
        delta = 0.0
        if self.enable_qs_dyn_lift[b] > 0.5:
            vref = ti.max(1e-6, speed)
            delta = (ti.math.pi * 0.5) * (-omega_y * chord_ref) / vref
        return delta

    @ti.func
    def _cos2_beta_factor(self, b: int, flow_vel: ti.types.vector(3, ti.f32)) -> ti.f32:
        factor = 1.0
        if self.enable_cos2_beta[b] > 0.5:
            vnorm = flow_vel.norm() + 1e-9
            sinb = ti.max(-1.0, ti.min(1.0, flow_vel.y / vnorm))
            beta = ti.asin(sinb)
            cosb = ti.cos(beta)
            factor = cosb * cosb
        return factor

    # ------------------------------------------------------------------
    # C++ AeroCoefficients
    # ------------------------------------------------------------------
    @ti.func
    def _aero_coeff(self, AR: ti.f32, alpha: ti.f32, alpha_stall: ti.f32, M_smooth: ti.f32,
                    cl_alpha_2D: ti.f32, c_d_0: ti.f32, delta_cl_lin: ti.f32, f_re: ti.f32):

        # --- NEW: allow signed alpha (SimpleDrone-like) ---
        alpha_abs = ti.abs(alpha)
        alpha = self._wrap_pm_90(alpha)
        k_cd = 1.0 - 0.41 * (1.0 - ti.exp(-17.0 / ti.max(1e-6, AR)))

        # Stall blending weight: depends on |alpha| only
        w = 0.0
        if alpha_abs > alpha_stall and alpha_abs < (ti.math.pi - alpha_stall):
            w = ti.cos(
                ti.math.pi
                * ((alpha_abs - alpha_stall) / ((ti.math.pi - alpha_stall) - alpha_stall))
                - (ti.math.pi * 0.5)
            )

        # Lift uses signed alpha; drag uses |alpha|
        c_l_st = 2.0 * ti.sin(alpha) * ti.cos(alpha) * (1.0 - w * (1.0 - k_cd))
        sa = ti.sin(alpha_abs)
        c_d_st = 2.0 * (sa * sa) * (1.0 - w * (1.0 - k_cd))

        # Linear lift branch: signed
        c_l_lin = (cl_alpha_2D * AR / (2.0 + ti.sqrt(AR * AR + 4.0))) * alpha + delta_cl_lin
        c_d_quad = c_d_0 + (c_l_lin * c_l_lin) / (ti.math.pi * ti.max(1e-6, AR))

        # Blend based on |alpha|
        s = self._sigmoid(alpha_abs, alpha_stall, M_smooth)
        c_l = (1.0 - s) * c_l_lin + s * c_l_st
        c_d = (1.0 - s) * c_d_quad + s * c_d_st

        # Reynolds degradation
        c_l *= f_re
        c_d = c_d_0 + (c_d - c_d_0) * f_re

        return c_l, c_d


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

        b_outer = (-0.0195 * theta_deg * theta_deg + 0.0603 * theta_deg + 165.0) / 1000.0
        s_outer = (-143.58 * theta_deg + 16309.0) / 1e6
        ac_x_outer = 0.15 / 6.0
        cpx_outer = (-0.00359 * theta_deg * theta_deg + 0.754 * theta_deg + 59.04) / 1000.0
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
        WingAeroCenter rewritten for the NEW convention:
        - returns a point in the LINK frame
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

        x = -ti.max(1e-6, chord_link) * frac
        return ti.Vector([x, 0.0, 0.0], dt=ti.f32)

    # ------------------------------------------------------------------
    # C++ WingForce (no slipstream mixing)
    # ------------------------------------------------------------------
    @ti.func
    def _wing_force(self, b: int, n_s: ti.types.vector(3, ti.f32), flow_vel: ti.types.vector(3, ti.f32),
                    b_w: ti.f32, s_w: ti.f32, mac_w: ti.f32, chord_ref: ti.f32,
                    alpha_stall: ti.f32, M_smooth: ti.f32, cl_alpha_2D: ti.f32, c_d_0: ti.f32,
                    omega_y: ti.f32, activate_dyn_aero: ti.i32):
        vnorm_raw = flow_vel.norm()
        vnorm = vnorm_raw + 1e-9
        n_d = flow_vel / vnorm

        sgn_dir = 1.0
        if n_s.dot(flow_vel) < 0.0:
            n_s = -n_s
            sgn_dir = -1.0

        n_l = ti.Vector([1.0, 0.0, 0.0], dt=ti.f32)
        if (n_s - n_d).norm() < 1e-10:
            n_l = ti.Vector([1.0, 0.0, 0.0], dt=ti.f32)
        else:
            n_l = self._cross(self._cross(n_d, n_s), n_d)
        n_l = n_l / (n_l.norm() + 1e-9)

        alpha = ti.atan2(-flow_vel.z, flow_vel.x)

        AR = (b_w * b_w) / ti.max(1e-6, s_w)
        f_re = self._reynolds_factor(b, vnorm_raw, mac_w)
        delta_cl = self._qs_dyn_lift_delta(b, omega_y, chord_ref, vnorm_raw)

        c_l, c_d = self._aero_coeff(AR, alpha, alpha_stall, M_smooth, cl_alpha_2D, c_d_0, delta_cl, f_re)

        eps_downwash = sgn_dir * c_l / (ti.math.pi * ti.max(1e-6, AR))

        # Forces use s/2 (per wing half) exactly like C++.
        vnorm2 = vnorm * vnorm
        f_l = 0.5 * vnorm2 * (s_w * 0.5) * c_l
        f_d = 0.5 * vnorm2 * (s_w * 0.5) * c_d

        # Optional cos^2(beta) attenuation (disabled by default).
        cos2 = self._cos2_beta_factor(b, flow_vel)
        f_l *= cos2
        f_d *= cos2

        F = f_l * n_l + f_d * n_d
        return F, eps_downwash, c_l, c_d

    # ------------------------------------------------------------------
    # C++ WingForcewithThrust (slipstream segment mixing)
    # ------------------------------------------------------------------
    @ti.func
    def _wing_force_with_thrust(self, b: int, n_s, flow_vel, flow_vel_slip, flow_rot,
                                b_w, s_w, mac_w, chord_ref,
                                s_slip,
                                alpha_stall, M_smooth, cl_alpha_2D, c_d_0,
                                omega_y: ti.f32, activate_dyn_aero: ti.i32):
        flow_vel = flow_vel + flow_rot
        flow_vel_slip = flow_vel_slip + flow_rot

        denom = ti.max(1e-6, (s_w * 0.5))
        flow_vel_avg = (flow_vel * (denom - s_slip) + flow_vel_slip * s_slip) / denom

        vnorm_raw = flow_vel.norm()
        vnorm = vnorm_raw + 1e-9
        n_d = flow_vel / vnorm

        sgn_dir = 1.0
        if n_s.dot(flow_vel) < 0.0:
            n_s = -n_s
            sgn_dir = -1.0
        n_l = ti.Vector([1.0, 0.0, 0.0], dt=ti.f32)
        if (n_s - n_d).norm() < 1e-10:
            n_l = ti.Vector([1.0, 0.0, 0.0], dt=ti.f32)
        else:
            n_l = self._cross(self._cross(n_d, n_s), n_d)
        n_l = n_l / (n_l.norm() + 1e-9)

        AR = (b_w * b_w) / ti.max(1e-6, s_w)

        alpha      = ti.atan2(-flow_vel.z,      flow_vel.x)
        alpha_slip = ti.atan2(-flow_vel_slip.z, flow_vel_slip.x)

        vnorm_slip_raw = flow_vel_slip.norm()
        vnorm_slip = vnorm_slip_raw + 1e-9
        beta = ti.atan2(flow_vel.y, flow_vel.x)
        beta_slip = ti.atan2(flow_vel_slip.y, flow_vel_slip.x)

        f_re = self._reynolds_factor(b, vnorm_raw, mac_w)
        f_re_slip = self._reynolds_factor(b, vnorm_slip_raw, mac_w)
        delta_cl = self._qs_dyn_lift_delta(b, omega_y, chord_ref, vnorm_raw)
        delta_cl_slip = self._qs_dyn_lift_delta(b, omega_y, chord_ref, vnorm_slip_raw)

        c_l, c_d = self._aero_coeff(AR, alpha, alpha_stall, M_smooth, cl_alpha_2D, c_d_0, delta_cl, f_re)
        c_l_s, c_d_s = self._aero_coeff(AR, alpha_slip, alpha_stall, M_smooth, cl_alpha_2D, c_d_0, delta_cl_slip, f_re_slip)

        eps_downwash = 0.0  # Exact match to C++ (downwash disabled for slipstream branch).
        F_out = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        c_l_avg = 0.0
        c_d_avg = 0.0
        # Optional two-segment sum: free + slip forces using free-flow directions.
        if self.use_slip_two_segment_sum[b] > 0.5:
            vnorm2 = vnorm * vnorm
            vnorm_slip2 = vnorm_slip * vnorm_slip
            cos2 = self._cos2_beta_factor(b, flow_vel)
            cos2_slip = self._cos2_beta_factor(b, flow_vel_slip)

            f_l_free = 0.5 * vnorm2 * (denom - s_slip) * c_l * cos2
            f_d_free = 0.5 * vnorm2 * (denom - s_slip) * c_d * cos2
            f_l_slip = 0.5 * vnorm_slip2 * s_slip * c_l_s * cos2_slip
            f_d_slip = 0.5 * vnorm_slip2 * s_slip * c_d_s * cos2_slip

            f_l = f_l_free + f_l_slip
            f_d = f_d_free + f_d_slip
            F_out = f_l * n_l + f_d * n_d

            vavg_raw = flow_vel_avg.norm()
            vavg2 = vavg_raw * vavg_raw
            den = vavg2 * denom + 1e-9
            c_l_avg = f_l / (0.5 * den)
            c_d_avg = f_d / (0.5 * den)
        else:

            # Weighted average like C++.
            vnorm2 = vnorm_raw * vnorm_raw
            vnorm_slip2 = vnorm_slip_raw * vnorm_slip_raw
            cosb = ti.cos(beta)
            cosb2 = cosb * cosb
            cosb_slip = ti.cos(beta_slip)
            cosb_slip2 = cosb_slip * cosb_slip
            num = ti.Vector([
                c_l * (vnorm2 * cosb2 * (denom - s_slip)) +
                c_l_s * (vnorm_slip2 * cosb_slip2 * (s_slip)),
                c_d * (vnorm2 * cosb2 * (denom - s_slip)) +
                c_d_s * (vnorm_slip2 * cosb_slip2 * (s_slip)),
                0.0
            ], dt=ti.f32)
            vavg_raw = flow_vel_avg.norm()
            vavg2 = vavg_raw * vavg_raw
            den = vavg2 * denom + 1e-9
            c_l_avg = num.x / den
            c_d_avg = num.y / den

            vavg = vavg_raw + 1e-9
            vavg2 = vavg * vavg
            f_l = 0.5 * vavg2 * denom * c_l_avg
            f_d = 0.5 * vavg2 * denom * c_d_avg

            # Optional cos^2(beta) attenuation using average flow when enabled.
            cos2_avg = self._cos2_beta_factor(b, flow_vel_avg)
            f_l *= cos2_avg
            f_d *= cos2_avg

            F_out = f_l * n_l + f_d * n_d
        return F_out, eps_downwash, c_l_avg, c_d_avg

    # ------------------------------------------------------------------
    # C++ HorTailForce (flat plate) + VertTailForce (flat plate)
    # ------------------------------------------------------------------
    @ti.func
    def _hor_tail_force(self, b: int, n_s, flow_vel, s, chord_ref, alpha_ele_eff, c_d_0):
        vnorm_raw = flow_vel.norm()
        vnorm = vnorm_raw + 1e-9
        n_d = flow_vel / vnorm
        if n_s.dot(flow_vel) < 0.0:
            n_s = -n_s
        n_l = ti.Vector([1.0, 0.0, 0.0], dt=ti.f32)
        if (n_s - n_d).norm() < 1e-10:
            n_l = ti.Vector([1.0, 0.0, 0.0], dt=ti.f32)
        else:
            n_l = self._cross(self._cross(n_d, n_s), n_d)
        n_l = n_l / (n_l.norm() + 1e-9)

        alpha = ti.atan2(-flow_vel.z, flow_vel.x) + alpha_ele_eff

        sa = ti.sin(alpha)
        ca = ti.cos(alpha)
        c_l = 2.0 * sa * ca
        c_d = c_d_0 + 2.0 * (sa * sa)

        # Optional Reynolds degradation (tail uses chord length as reference).
        f_re = self._reynolds_factor(b, vnorm_raw, chord_ref)
        c_l *= f_re
        c_d = c_d_0 + (c_d - c_d_0) * f_re

        vnorm2 = vnorm * vnorm
        f_l = 0.5 * vnorm2 * s * c_l
        f_d = 0.5 * vnorm2 * s * c_d
        return f_l * n_l + f_d * n_d

    @ti.func
    def _vert_tail_force(self, b: int, n_s, flow_vel, s, chord_ref, alpha_rud_eff):
        vnorm_raw = flow_vel.norm()
        vnorm = vnorm_raw + 1e-9
        n_d = flow_vel / vnorm
        if n_s.dot(flow_vel) < 0.0:
            n_s = -n_s
        n_l = ti.Vector([1.0, 0.0, 0.0], dt=ti.f32)
        if (n_s - n_d).norm() < 1e-10:
            n_l = ti.Vector([0.0, 1.0, 0.0], dt=ti.f32)
        else:
            n_l = self._cross(self._cross(n_d, n_s), n_d)
        n_l = n_l / (n_l.norm() + 1e-9)

        beta = ti.atan2(flow_vel.y, flow_vel.x) + alpha_rud_eff

        sb = ti.sin(beta)
        cb = ti.cos(beta)
        c_l = 2.0 * sb * cb
        c_d = 2.0 * (sb * sb)

        # Optional Reynolds degradation (no cd0 baseline for vertical tail).
        f_re = self._reynolds_factor(b, vnorm_raw, chord_ref)
        c_l *= f_re
        c_d *= f_re

        vnorm2 = vnorm * vnorm
        f_l = 0.5 * vnorm2 * s * c_l
        f_d = 0.5 * vnorm2 * s * c_d
        return f_l * n_l + f_d * n_d

    @ti.kernel
    def _init_prev_states(self):
        for b in range(self.B):
            self.flow_prev[b] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            self.alpha_dot_prev[b] = 0.0
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

        for b in range(self.B):
            base_link = self.body_link_idx[None]
            omega = self._omega_body(rigid, base_link, b)

            # Airflow in body frame; by convention flow = -body velocity.
            flow_base = self._get_wind_in_body(rigid, base_link, b)

            # Forward speed along body x-axis.
            u = -flow_base.x

            # Thrust (from BaseAeroSolver throttle fields).
            alpha_lpf = self._substep_dt / (
                self._substep_dt + 1.0 / (2.0 * ti.math.pi * ti.max(1e-3, self.prop_cutoff_hz[b]))
            )
            self._thr_flt[b] += alpha_lpf * (ti.cast(self._thr_raw[b], ti.f32) - ti.cast(self._thr_flt[b], ti.f32))
            thr = ti.max(0.0, ti.min(1.0, ti.cast(self._thr_flt[b], ti.f32)))
            thrust = thr * self.max_thrust[b] * self.prop_thrust_sign[b]

            R_prop = ti.max(1e-6, self.prop_radius[b])
            rho = ti.max(1e-6, self.rho[b])

            # C++ prop induced wake velocity.
            disc = u * u + (2.0 * ti.abs(thrust)) / (rho * ti.math.pi * (R_prop * R_prop))
            prop_flow_ind_wake = (-u + ti.sqrt(disc)) * 0.5

            # Dynamic aero is disabled by default in the C++ implementation.
            activate_dyn_aero = 0

            # Precompute stall radians.
            alpha_stall_wing = (self.alpha_stall_wing_deg[b] * ti.math.pi) / 180.0

            # Root constants.
            c_root_inner = self.c_root_inner[b]
            s_slip = R_prop * c_root_inner

            eps_left = 0.0
            eps_right = 0.0

            for l in range(self.L):
                k = self.kind[l]
                link_idx = self._link_idx[l]

                # Wind in THIS surface's link body frame (mydrone convention).
                flow_link = self._get_wind_in_body(rigid, link_idx, b)
                omega_link = self._omega_body(rigid, link_idx, b)
                # ===== WINGS =====
                if k == K_WING:
                    theta = self._read_surface_angle(rigid, b, l)
                    b_w, s_w, mac_w, ac_x, cpx, cpy = self._wing_properties(b, theta)
                    # QS dyn lift chord reference: use 4*ac_x when available, else mac_w.
                    chord_ref = ti.select(ac_x > 1e-6, 4.0 * ac_x, mac_w)
                    chord_ref = ti.max(1e-6, chord_ref)
                    chord_link = ti.max(1e-6, self.chord0[l])
                    omega_y = omega_link.y
                    pos_rot_link = self._wing_aero_center_linkframe(
                        flow_link, chord_link, chord_ref,
                        ac_x, cpx, alpha_stall_wing, 1
                    )
                    flow_rot = -self._cross(omega_link, pos_rot_link)
                    use_slip = 1
                    F = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                    eps = 0.0
                    cl_avg = 0.0
                    cd_avg = 0.0
                    if use_slip == 1:
                        flow_slip = flow_link + ti.Vector([-prop_flow_ind_wake, 0.0, 0.0], dt=ti.f32)
                        F, eps, cl_avg, cd_avg = self._wing_force_with_thrust(
                            b,
                            ti.Vector([0.0, 0.0, 1.0], dt=ti.f32),
                            flow_link, flow_slip, flow_rot,
                            b_w, s_w, mac_w, chord_ref, s_slip,
                            alpha_stall_wing, self.M_smooth[b], self.cl_alpha_wing_2D[b], self.c_d_0_wing[b],
                            omega_y, activate_dyn_aero
                        )
                    else:
                        F, eps, cl_avg, cd_avg = self._wing_force(
                            b,
                            ti.Vector([0.0, 0.0, 1.0], dt=ti.f32),
                            flow_link + flow_rot,
                            b_w, s_w, mac_w, chord_ref,
                            alpha_stall_wing, self.M_smooth[b], self.cl_alpha_wing_2D[b], self.c_d_0_wing[b],
                            omega_y, activate_dyn_aero
                        )

                    # Multiply by rho to match C++.
                    F *= rho
                    flow_avg = flow_link
                    if use_slip == 1:
                        flow_slip = flow_link + ti.Vector([-prop_flow_ind_wake, 0.0, 0.0], dt=ti.f32)
                        denom = ti.max(1e-6, (s_w * 0.5))
                        flow_avg = (flow_link * (denom - s_slip) + flow_slip * s_slip) / denom
                    else:
                        flow_avg = flow_link

                    pos_force = self._wing_aero_center_linkframe(
                        flow_avg + flow_rot, chord_link, chord_ref,
                        ac_x, cpx, alpha_stall_wing, 0
                    )

                    cap = self.force_cap[b]
                    F *= ti.min(1.0, cap / (F.norm() + 1e-9))

                    self.force_b[b, l] = F
                    self.cp_b[b, l] = pos_force

                    if ti.static(self._aero_log):
                        # Use the same flow used to place the aero center (average flow + rotation)
                        v_dbg = flow_avg + flow_rot
                        alpha_dbg = ti.atan2(v_dbg.z, v_dbg.x)
                        vmod = v_dbg.norm() + 1e-9
                        beta_dbg = ti.asin(ti.math.clamp(v_dbg.y / vmod, -1.0, 1.0))
                        drag, lift, side = self._dbg_decompose_force(F, v_dbg)
                        self.alpha_dbg[b, l] = ti.cast(alpha_dbg, ti.f16)
                        self.beta_dbg[b, l] = ti.cast(beta_dbg, ti.f16)
                        self.drag_dbg[b, l] = ti.cast(drag, ti.f16)
                        self.lift_dbg[b, l] = ti.cast(lift, ti.f16)
                        self.side_force_dbg[b, l] = ti.cast(side, ti.f16)

                    if self.side[l] == +1:
                        eps_left = eps
                    elif self.side[l] == -1:
                        eps_right = eps

                    # Print velocity, forces, alpha, beta and euler angles and cp for debugging
                    print("Wing Debug - B:", b, "L:", l, "Vel:", flow_link, "F:", F, "Alpha:", self.alpha_dbg[b, l], "Beta:", self.beta_dbg[b, l], "CP:", pos_force)

                # ===== ELEVATOR / HORIZONTAL TAIL =====
                elif k == K_ELEVATOR:
                    elevator = self._read_surface_angle(rigid, b, l)

                    c_ele = self.c_ele[b]
                    c_hor = self.c_hor_tail[b]
                    s_hor = self.s_hor_tail[b]
                    b_hor = self.b_hor_tail[b]

                    c_eff = ti.sqrt(c_ele * c_ele + (c_hor - c_ele) ** 2 + 2.0 * c_ele * (c_hor - c_ele) * ti.cos(elevator))
                    ang_eff = self.k_alpha_elev[b] * elevator

                    dir_ele = ti.Vector([-ti.sin(ang_eff), 0.0, ti.cos(ang_eff)], dt=ti.f32)

                    # Apply force in LINK frame at a CP that moves only along local X.
                    chord_link = ti.max(1e-6, self.chord0[l])
                    # Use the same CP logic as wings: quarter-chord pre-stall → half-chord near 90deg.
                    # (if you want, we can expose cp_start/cp_end as params later)
                    ac_x_cpp = 0.25 * chord_link
                    cpx_cpp = 0.50 * chord_link
                    chord_ref = chord_link
                    alpha_stall_tail = (self.alpha_stall_tail_deg[b] * ti.math.pi) / 180.0
                    pos_ele = self._wing_aero_center_linkframe(
                        flow_link, chord_link, chord_ref, ac_x_cpp, cpx_cpp, alpha_stall_tail, 0
                    )

                    flow_rot_ele = -self._cross(omega_link, pos_ele)

                    flow_downwash = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                    if u > 0.0:
                        flow_downwash.z = ti.tan(eps_left + eps_right) * u

                    F_ele = self._hor_tail_force(
                        b,
                        dir_ele, flow_link + flow_rot_ele + flow_downwash,
                        s_hor, c_hor, ang_eff, self.c_d_0_tail[b]
                    )
                    F_ele *= rho

                    cap = self.force_cap[b]
                    F_ele *= ti.min(1.0, cap / (F_ele.norm() + 1e-9))

                    self.force_b[b, l] = F_ele
                    self.cp_b[b, l] = pos_ele

                    if ti.static(self._aero_log):
                        v_dbg = flow_link + flow_rot_ele + flow_downwash
                        alpha_raw = ti.atan2((flow_link + flow_rot_ele).z, (flow_link + flow_rot_ele).x)
                        alpha_eff = ti.atan2(v_dbg.z, v_dbg.x)
                        vmod = v_dbg.norm() + 1e-9
                        beta_dbg = ti.asin(ti.math.clamp(v_dbg.y / vmod, -1.0, 1.0))
                        drag, lift, side = self._dbg_decompose_force(F_ele, v_dbg)
                        self.alpha_tail_raw_dbg[b, l] = ti.cast(alpha_raw, ti.f16)
                        # eps/downwash scalars might exist in your code; if you have them, store them.
                        # If not, keep 0 (already initialized above).
                        self.alpha_dbg[b, l] = ti.cast(alpha_eff, ti.f16)
                        self.beta_dbg[b, l] = ti.cast(beta_dbg, ti.f16)
                        self.drag_dbg[b, l] = ti.cast(drag, ti.f16)
                        self.lift_dbg[b, l] = ti.cast(lift, ti.f16)
                        self.side_force_dbg[b, l] = ti.cast(side, ti.f16)
                    # Print velocity, forces, alpha, beta and euler angles and cp for debugging
                    print("Wing Debug - B:", b, "L:", l, "Vel:", flow_link, "F:", F_ele, "Alpha:", self.alpha_dbg[b, l], "Beta:", self.beta_dbg[b, l], "CP:", pos_ele)

                # ===== RUDDER / VERTICAL TAIL =====
                elif k == K_RUDDER:
                    rudder = self._read_surface_angle(rigid, b, l)

                    c_rud = self.c_rud[b]
                    c_vert = self.c_vert_tail[b]
                    s_vert = self.s_vert_tail[b]
                    b_vert = self.b_vert_tail[b]

                    c_eff = ti.sqrt(c_rud * c_rud + (c_vert - c_rud) ** 2 + 2.0 * c_rud * (c_vert - c_rud) * ti.cos(rudder))
                    ang_eff = self.k_alpha_rudder[b] * rudder

                    dir_rud = ti.Vector([-ti.sin(ang_eff), -ti.cos(ang_eff), 0.0], dt=ti.f32)

                    chord_link = ti.max(1e-6, self.chord0[l])
                    ac_x_cpp = 0.25 * chord_link
                    cpx_cpp = 0.50 * chord_link
                    chord_ref = chord_link
                    alpha_stall_tail = (self.alpha_stall_tail_deg[b] * ti.math.pi) / 180.0
                    pos_rud = self._wing_aero_center_linkframe(
                        flow_link, chord_link, chord_ref, ac_x_cpp, cpx_cpp, alpha_stall_tail, 0
                    )
                    flow_rot_rud = -self._cross(omega_link, pos_rud)

                    F_rud = self._vert_tail_force(
                        b,
                        dir_rud, flow_link + flow_rot_rud,
                        s_vert, c_vert, ang_eff
                    )
                    F_rud *= rho

                    cap = self.force_cap[b]
                    F_rud *= ti.min(1.0, cap / (F_rud.norm() + 1e-9))

                    self.force_b[b, l] = F_rud
                    self.cp_b[b, l] = pos_rud

                    if ti.static(self._aero_log):
                        v_dbg = flow_link + flow_rot_rud
                        alpha_dbg = ti.atan2(v_dbg.z, v_dbg.x)
                        vmod = v_dbg.norm() + 1e-9
                        beta_dbg = ti.asin(ti.math.clamp(v_dbg.y / vmod, -1.0, 1.0))
                        drag, lift, side = self._dbg_decompose_force(F_rud, v_dbg)
                        self.alpha_dbg[b, l] = ti.cast(alpha_dbg, ti.f16)
                        self.beta_dbg[b, l] = ti.cast(beta_dbg, ti.f16)
                        self.drag_dbg[b, l] = ti.cast(drag, ti.f16)
                        self.lift_dbg[b, l] = ti.cast(lift, ti.f16)
                        self.side_force_dbg[b, l] = ti.cast(side, ti.f16)
                    # Print velocity, forces, alpha, beta and euler angles and cp for debugging
                    print("Wing Debug - B:", b, "L:", l, "Vel:", flow_link, "F:", F_rud, "Alpha:", self.alpha_dbg[b, l], "Beta:", self.beta_dbg[b, l], "CP:", pos_rud)

                # ===== PROPELLER (thrust only) =====
                elif k == K_PROPELLER:
                    # Prop thrust axis is +Z in the prop frame (matches DroneAeroModel convention).
                    pos_prop_cg = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                    F_thr = ti.Vector([0.0, 0.0, thrust], dt=ti.f32)

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



# Keep the default solver export pattern consistent with other drone solvers.
AeroSolver = LisparrowAeroSolver
