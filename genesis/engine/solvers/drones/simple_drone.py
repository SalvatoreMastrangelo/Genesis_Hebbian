import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import gstaichi as ti
import numpy as np
import torch

from genesis.engine.solvers.base_aero_solver import BaseAeroSolver
from genesis.engine.entities import RigidEntity  # for get_link()
from genesis.assets.urdf.aero_model import DroneAeroModel, SurfaceKind


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


class SimpleDroneAeroParameters:
    """
    Inline aero configuration for the simple drone solver.
    """

    GLOBAL = {
        "rho": 1.225,
        "force_cap": 30.0,
    }

    TYPES = {
        "fuselage": {
            "cd0": 0.65,
            "k_slip_fus": 0.0,
            "cp_start": 0.0,
            "cp_end": 0.5,
            "cg_to_chord": 0.31,
        },
        "wing": {
            "cl_alpha_2d": 6.283185307179586,
            "alpha0_2d": -0.05235987755982988,
            "cd0": 0.05,
            "oswald_efficiency": 0.8,
            "k_slip_wing": 0.1,
            "alpha_stall_deg": 10.0,
            "m_smooth": 0.2,
            "w": 0.0,
            "cp_start": 0.25,
            "cp_end": 0.5,
            "cg_to_chord": 0.31,
            "re_nominal": 100000.0,
            "a": 1.0,
        },
        "elevator": {
            "cl_alpha_2d": 6.283185307179586,
            "alpha0_2d": 0.0,
            "cd0": 0.013,
            "oswald_efficiency": 0.8,
            "k_slip_tail": 1.0,
            "k_eps_tail": 1.0,
            "alpha_stall_deg": 10.0,
            "m_smooth": 0.2,
            "w": 0.0,
            "cp_start": 0.25,
            "cp_end": 0.5,
            "cg_to_chord": 0.31,
            "re_nominal": 50000.0,
            "a": 1.0,
        },
        "rudder": {
            "cl_alpha_2d": 6.283185307179586,
            "alpha0_2d": 0.0,
            "cd0": 0.013,
            "oswald_efficiency": 0.8,
            "alpha_stall_deg": 10.0,
            "m_smooth": 0.2,
            "w": 0.0,
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
        "sigma_mag": 0.05,
        "sigma_dir": 0.05,
        "sigma_param": 0.15,
        "sigma_cp": 0.05,
        "mass_shift": 0.15,
        "com_shift": 0.01,
    }

    @classmethod
    def as_dict(cls) -> dict:
        return {
            "global": copy.deepcopy(cls.GLOBAL),
            "types": copy.deepcopy(cls.TYPES),
            "links": copy.deepcopy(cls.LINKS),
            "noise": copy.deepcopy(cls.NOISE),
        }

    @classmethod
    def fuselage_cd0(cls) -> float:
        return float(cls.TYPES["fuselage"]["cd0"])


@ti.data_oriented
class SimpleDroneAeroSolver(BaseAeroSolver):
    """Drone-specific aerodynamic solver implementation."""

    # ---------------------------------------------------------------------
    # Initialization utilities
    # ---------------------------------------------------------------------
    def _init_surface_param_names(self):
        # Names of wing-specific duplicated parameters (left/right)
        self._wing_param_names = [
            "cl_alpha_2d",
            "alpha0_2d",
            "cd0",
            "alpha_stall_deg",
            "m_smooth",
            "w",
            "cp_start",
            "cp_end",
            "cg_to_chord",
            "k_slip_wing",
        ]
        self._elevator_param_names = [
            "cl_alpha_2d",
            "alpha0_2d",
            "cd0",
            "alpha_stall_deg",
            "m_smooth",
            "w",
            "cp_start",
            "cp_end",
            "cg_to_chord",
            "k_slip_tail",
            "k_eps_tail",
        ]
        self._randomizable_link_field_names = (
            "cd0_link",
            "oswald_efficiency_link",
            "alpha0_2d_link",
            "cl_alpha_2d_link",
            "alpha_stall_deg_link",
            "m_smooth_link",
            "w_link",
            "cp_start_link",
            "cp_end_link",
            "cg_to_chord_link",
            "k_slip_wing_link",
            "k_slip_tail_link",
            "k_eps_tail_link",
            "k_slip_fus_link",
            "re_nom_link",
            "re_a_link",
        )

    def _const_init(self):
        """
        Initialize base aerodynamic parameters and frame names.

        All values here are *nominal* and stored in `_aero_base`.
        They are later copied into Taichi fields (one scalar field per env)
        in `add_target`, so they can be randomized per environment.
        """
        # Base values (per env, uniform at start)
        self._aero_base = dict(DroneAeroModel.DEFAULT_BASE_PARAMS)
        self._aero_base.setdefault("w", 0.0)
        self._ensure_wing_param_entries()
        self._ensure_elevator_param_entries()
        # Control which parameters are randomized (default: all base params).
        self._randomizable_param_names = list(self._aero_base.keys())

        # Link names used as aerodynamic frames (populated from DroneAeroModel)
        # The last one is assumed to be the propeller frame.
        self._aero_frames: list[str] = list(DroneAeroModel.AERO_FRAMES)

        # Geometry-dependent attributes (filled when a drone model is bound)
        self.tip_to_tip: float = 0.0
        self.AR_wing: float = 0.0
        self.AR_tail: float = 0.0
        self.S_perp_fus: float = 0.0
        self.cg_fus_local_x: float = 0.0
        self.cz_rudder: float = 0.0
        self.prop_radius: float = 0.0
        self._surface_kinds: list[SurfaceKind] = []
        self.genes: list[float] = []
        self.genes_dict: dict[str, float] = {}
        self._drone_model: DroneAeroModel | None = None

        # Stochastic force noise parameters used in `noise_addition`.
        # These are kept as plain Python floats on purpose, so they act as
        # compile-time constants for Taichi kernels and as defaults for
        # `randomize_aero_params`.
        self.noise_sigma_mag: float = DroneAeroModel.NOISE_DEFAULTS["sigma_mag"]  # relative std-dev on |F|
        self.noise_sigma_dir: float = DroneAeroModel.NOISE_DEFAULTS["sigma_dir"]  # std-dev for directional noise
        self.noise_sigma_param: float = DroneAeroModel.NOISE_DEFAULTS["sigma_param"]  # relative std-dev on aero params
        self.noise_sigma_cp: float = DroneAeroModel.NOISE_DEFAULTS["sigma_cp"]  # absolute std-dev on CP location

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
                raise RuntimeError(
                    "AeroSolver.add_target: `_aero_frames` missing. Provide a DroneAeroModel or URDF."
                )
            self._aero_link_idx = [entity.get_link(name).idx for name in self._aero_frames]

        if not self._geom:
            raise RuntimeError(
                "AeroSolver.add_target: empty `_geom`. Provide a valid URDF "
                "before adding the target or pre-populate `self._geom`."
            )

        # Number of aerodynamic surfaces
        self.L = len(self._geom)

        # Batch size (B) and number of global links
        self._B = getattr(self._rigid_solver.sim, "_B", 1)
        self.n_links_ = max(1, self._rigid_solver.n_links)

        frames = list(self._aero_frames)
        kinds = list(getattr(model, "surface_kinds", [])) if model is not None else []
        if model is not None and len(kinds) != len(frames):
            raise RuntimeError("Drone model did not expose per-surface kinds.")

        geom = list(self._geom)
        link_indices = list(self._aero_link_idx)
        base_param_overrides = dict(getattr(model, "base_param_overrides", {}) or {})
        noise_params = dict(getattr(model, "noise_params", {}) or {})
        lower_frames = [frame.lower() for frame in frames]
        side = [0 for _ in range(self.L)]
        slip_code = [0 for _ in range(self.L)]
        fus_width = 0.0
        for i, (_, _, c, k) in enumerate(geom):
            name = lower_frames[i]
            if "left" in name:
                side[i] = -1
            elif "right" in name:
                side[i] = 1
            else:
                side[i] = 0

            if k == 0:
                slip_code[i] = 0
                fus_width = float(c)
            elif k == 2:
                slip_code[i] = 1
            elif k == 1 and "prop" in name:
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

    def _alloc_simple_fields(self, B: int, L: int) -> None:
        """
        Phase 2: allocate SimpleDrone-specific Taichi fields.
        """
        # Per-link parameter fields
        self.cd0_link = ti.field(ti.f32, shape=(B, L))
        self.oswald_efficiency_link = ti.field(ti.f32, shape=(B, L))
        self.alpha0_2d_link = ti.field(ti.f32, shape=(B, L))
        self.cl_alpha_2d_link = ti.field(ti.f32, shape=(B, L))
        self.alpha_stall_deg_link = ti.field(ti.f32, shape=(B, L))
        self.m_smooth_link = ti.field(ti.f32, shape=(B, L))
        self.w_link = ti.field(ti.f32, shape=(B, L))
        self.cp_start_link = ti.field(ti.f32, shape=(B, L))
        self.cp_end_link = ti.field(ti.f32, shape=(B, L))
        self.cg_to_chord_link = ti.field(ti.f32, shape=(B, L))
        self.re_nom_link = ti.field(ti.f32, shape=(B, L))
        self.re_a_link = ti.field(ti.f32, shape=(B, L))

        self.k_slip_wing_link = ti.field(ti.f32, shape=(B, L))
        self.k_slip_tail_link = ti.field(ti.f32, shape=(B, L))
        self.k_eps_tail_link = ti.field(ti.f32, shape=(B, L))
        self.k_slip_fus_link = ti.field(ti.f32, shape=(B, L))

        # Per-surface Reynolds number (computed from local flow speed and chord).
        self.Reynolds = ti.field(ti.f32, shape=(B, L))

        # Debug fields are expensive and only needed for eval/debug traces.
        self._enable_aero_debug_buffers = bool(self._aero_log)
        if self._enable_aero_debug_buffers:
            self.alpha_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
            self.beta_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
            self.lift_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
            self.drag_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
            self.side_force_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
            # Tail/downwash debug (filled only for elevators when _aero_log=True)
            self.alpha_tail_raw_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
            self.downwash_eps_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
            self.cl_wing_for_tail_dbg = ti.field(ti.f16, shape=(B, self.n_links_))
            self.k_eps_tail_dbg = ti.field(ti.f16, shape=(B, self.n_links_))

        # Per-surface constants (area, AR, chord, kind, side, link indices)
        self.area = ti.field(ti.f16, shape=(L,))
        self.AR = ti.field(ti.f16, shape=(L,))
        self.chord = ti.field(ti.f16, shape=(L,))
        self.kind = ti.field(ti.i16, shape=(L,))
        self.side = ti.field(ti.i8, shape=(L,))
        self._link_idx = ti.field(ti.i16, shape=(L,))
        self.slip_code = ti.field(ti.i8, shape=(L,))
        # Cached link index for the propeller (last aerodynamic surface).
        self._prop_link_idx = ti.field(ti.i16, shape=())

        # Base span per surface (total span used in AR = b^2 / S)
        self.span = ti.field(ti.f16, shape=(L,))
        # Fuselage width used for simple wing AR interference correction.
        self.fus_width = ti.field(ti.f16, shape=())

        # Wing CL accumulator (left/right) and induced velocity at the prop
        self.cl_wing_b = ti.field(ti.f16, shape=(B, 2))
        self.v_ind = ti.field(ti.f16, shape=(B,))

    def _init_simple_fields(self, bound: BoundTarget) -> None:
        """
        Phase 3: populate per-link parameters and per-surface constants.
        """
        def require_param(frame: str, params: dict, key: str) -> float:
            if key not in params:
                raise ValueError(f"Surface '{frame}' missing required key '{key}' in aero configuration.")
            return float(params[key])

        model = bound.model
        surface_frames = bound.frames
        surface_kinds = bound.kinds

        for i in range(self.L):
            p = model.per_surface_params[i]
            frame = surface_frames[i]
            kind = surface_kinds[i]

            if kind == SurfaceKind.PROPELLER:
                cd0_val = alpha0_val = cl_alpha_val = 0.0
                oswald_efficiency_val = 1.0
                alpha_stall_val = m_smooth_val = 0.0
                w_val = 0.0
                cp_start_val = cp_end_val = 0.0
                cg_to_chord_val = 0.0
            elif kind == SurfaceKind.FUSELAGE:
                cd0_val = SimpleDroneAeroParameters.fuselage_cd0()
                oswald_efficiency_val = 1.0
                alpha0_val = cl_alpha_val = 0.0
                alpha_stall_val = m_smooth_val = 0.0
                w_val = 0.0
                cp_start_val = require_param(frame, p, "cp_start")
                cp_end_val = require_param(frame, p, "cp_end")
                cg_to_chord_val = require_param(frame, p, "cg_to_chord")
            else:
                cd0_val = require_param(frame, p, "cd0")
                oswald_efficiency_val = float(p.get("oswald_efficiency", 0.8))
                alpha0_val = require_param(frame, p, "alpha0_2d")
                cl_alpha_val = require_param(frame, p, "cl_alpha_2d")
                alpha_stall_val = require_param(frame, p, "alpha_stall_deg")
                m_smooth_val = require_param(frame, p, "m_smooth")
                w_val = float(p.get("w", 0.0))
                cp_start_val = require_param(frame, p, "cp_start")
                cp_end_val = require_param(frame, p, "cp_end")
                cg_to_chord_val = require_param(frame, p, "cg_to_chord")

            k_slip_wing_val = require_param(frame, p, "k_slip_wing") if kind == SurfaceKind.WING else 0.0
            k_slip_tail_val = require_param(frame, p, "k_slip_tail") if kind == SurfaceKind.ELEVATOR else 0.0
            k_eps_tail_val = require_param(frame, p, "k_eps_tail") if kind == SurfaceKind.ELEVATOR else 0.0
            k_slip_fus_val = require_param(frame, p, "k_slip_fus") if kind == SurfaceKind.FUSELAGE else 0.0
            re_nom_val = float(p.get("re_nom", 0.0))
            re_a_val = float(p.get("re_a", 1.0))

            for b in range(self._B):
                self.cd0_link[b, i] = cd0_val
                self.oswald_efficiency_link[b, i] = oswald_efficiency_val
                self.alpha0_2d_link[b, i] = alpha0_val
                self.cl_alpha_2d_link[b, i] = cl_alpha_val
                self.alpha_stall_deg_link[b, i] = alpha_stall_val
                self.m_smooth_link[b, i] = m_smooth_val
                self.w_link[b, i] = w_val
                self.cp_start_link[b, i] = cp_start_val
                self.cp_end_link[b, i] = cp_end_val
                self.cg_to_chord_link[b, i] = cg_to_chord_val

                self.k_slip_wing_link[b, i] = k_slip_wing_val
                self.k_slip_tail_link[b, i] = k_slip_tail_val
                self.k_eps_tail_link[b, i] = k_eps_tail_val
                self.k_slip_fus_link[b, i] = k_slip_fus_val
                self.re_nom_link[b, i] = re_nom_val
                self.re_a_link[b, i] = re_a_val

        # Fill per-surface constant fields from _geom
        for i, (S, AR, c, k) in enumerate(bound.geom):
            self.area[i] = S
            self.AR[i] = AR
            self.chord[i] = c
            self.kind[i] = k
            self._link_idx[i] = bound.link_indices[i]
            # Base span can be derived from S = chord * span.
            if float(c) > 1e-8:
                self.span[i] = float(S) / float(c)
            else:
                self.span[i] = 0.0

            # Side flag is data: -1 left, +1 right, 0 center; param selectors use this single flag.
            self.side[i] = bound.side[i]

            # Slipstream code (0=fuselage,1=tail,2=prop wash wing,3=other)
            self.slip_code[i] = bound.slip_code[i]

        self.fus_width[None] = bound.fus_width
        if self.L > 0:
            # Last aerodynamic surface is the propeller by construction.
            self._prop_link_idx[None] = bound.link_indices[self.L - 1]

    def _sync_side_profile_caches_from_links(self, bound: BoundTarget) -> None:
        """
        Phase 3: sync wing/elevator left/right cache fields from per-link values.
        """
        if not bound.kinds:
            return

        for i, kind in enumerate(bound.kinds):
            side = bound.side[i]
            if side == 0:
                continue

            if kind == SurfaceKind.WING:
                for name in self._wing_param_names:
                    link_field = getattr(self, f"{name}_link", None)
                    if link_field is None:
                        continue
                    left_key, right_key = self._wing_side_keys(name)
                    cache_field = getattr(self, left_key if side < 0 else right_key, None)
                    if cache_field is None:
                        continue
                    for b in range(bound.B):
                        cache_field[b] = float(link_field[b, i])
            elif kind == SurfaceKind.ELEVATOR:
                for name in self._elevator_param_names:
                    link_field = getattr(self, f"{name}_link", None)
                    if link_field is None:
                        continue
                    left_key, right_key = self._elevator_side_keys(name)
                    cache_field = getattr(self, left_key if side < 0 else right_key, None)
                    if cache_field is None:
                        continue
                    for b in range(bound.B):
                        cache_field[b] = float(link_field[b, i])

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def add_target(
        self,
        entity: RigidEntity,
        urdf_file: str | None = None,
        drone_model: DroneAeroModel | None = None,
    ):
        """
        Register a RigidEntity as aerodynamic target and allocate Taichi fields.

        Parameters
        ----------
        entity:
            RigidEntity instance whose links contain the aerodynamic frames.
        urdf_file:
            Optional URDF file path used to infer aerodynamic geometry. If
            provided, a `DroneAeroModel` is built and used to populate the
            aerodynamic metadata. If omitted, supply `drone_model` or ensure
            `_geom` and `_aero_frames` are pre-populated.
        drone_model:
            Pre-parsed aerodynamic model for the drone. When provided it
            overrides `urdf_file`.
        """
        self._aero_targets.append(entity)

        # Phase 1: bind model and collect metadata
        bound = self._bind_target_model(entity, urdf_file, drone_model)

        # Phase 2: allocate common + simple-specific fields
        self._alloc_simple_fields(bound.B, bound.L)
        self._alloc_common_fields(bound.B, bound.L)

        # Phase 3: populate fields and sync caches
        self._init_simple_fields(bound)

        # Register base parameters as per-env Taichi fields
        for name, val in self._aero_base.items():
            f = ti.field(dtype=ti.f32, shape=(self._B,))
            for b in range(self._B):
                f[b] = float(val)
            self._param[name] = f
            setattr(self, name, f)

        self._sync_side_profile_caches_from_links(bound)

        # Cached Torch buffers for parameters queried each step
        self._init_param_buffers()
        self._capture_global_param_nominals()
        self._capture_link_param_nominals()

        # Enable aero once everything is initialized
        self._aero_enabled = True

    def _capture_global_param_nominals(self) -> None:
        """Snapshot nominal per-env aerodynamic fields in Torch buffers."""
        self._param_nominal_torch = {}
        for name, _, fld in self._iter_randomizable_params():
            self._param_nominal_torch[name] = fld.to_torch(device=self._aero_device).clone()

    def _capture_link_param_nominals(self) -> None:
        """Snapshot nominal per-link aerodynamic fields in Torch buffers."""
        self._link_param_nominal_torch = {}
        for name in getattr(self, "_randomizable_link_field_names", ()):
            fld = getattr(self, name, None)
            if fld is None:
                continue
            self._link_param_nominal_torch[name] = fld.to_torch(device=self._aero_device).clone()

    def _ensure_randomization_scratch(self) -> None:
        if not hasattr(self, "_rand_param_scratch_1d"):
            self._rand_param_scratch_1d = {}
        if not hasattr(self, "_rand_param_scratch_2d"):
            self._rand_param_scratch_2d = {}

    def _get_rand_scratch_1d(self, name: str, n_envs: int) -> torch.Tensor:
        self._ensure_randomization_scratch()
        buf = self._rand_param_scratch_1d.get(name)
        if buf is None or buf.shape[0] != n_envs:
            buf = torch.empty((n_envs,), device=self._aero_device, dtype=torch.float32)
            self._rand_param_scratch_1d[name] = buf
        return buf

    def _get_rand_scratch_2d(self, name: str, n_envs: int) -> torch.Tensor:
        self._ensure_randomization_scratch()
        buf = self._rand_param_scratch_2d.get(name)
        if buf is None or buf.shape[0] != n_envs or buf.shape[1] != self.L:
            buf = torch.empty((n_envs, self.L), device=self._aero_device, dtype=torch.float32)
            self._rand_param_scratch_2d[name] = buf
        return buf

    def _set_taichi_field_rows_1d(self, field: ti.Field, env_ids: np.ndarray, values: torch.Tensor) -> None:
        arr = field.to_torch(device=self._aero_device)
        arr[env_ids] = values
        field.from_torch(arr)

    def _set_taichi_field_rows_2d(self, field: ti.Field, env_ids: np.ndarray, values: torch.Tensor) -> None:
        arr = field.to_torch(device=self._aero_device)
        arr[env_ids, :] = values
        field.from_torch(arr)

    def _resolve_drone_model(
        self,
        urdf_file: str | None,
        drone_model: DroneAeroModel | None,
    ) -> DroneAeroModel | None:
        """
        Choose between a provided drone model or build one from the URDF path.
        """
        if drone_model is not None and urdf_file is not None:
            raise ValueError("Specify either `drone_model` or `urdf_file`, not both.")

        if drone_model is not None:
            return drone_model

        if urdf_file is None:
            return None

        return DroneAeroModel(urdf_file, config_override=SimpleDroneAeroParameters.as_dict())

    def _apply_drone_model(self, model: DroneAeroModel, entity: RigidEntity):
        """
        Populate aerodynamic metadata and link indices from a DroneAeroModel.
        """
        self._drone_model = model
        self._aero_frames = list(model.frames)
        self._geom = list(model.geom)
        self._aero_base = dict(model.base_params)
        self._aero_base.setdefault("w", 0.0)
        self._ensure_wing_param_entries()
        self._ensure_elevator_param_entries()
        # Apply any per-link overrides (e.g., left/right wing slip factors)
        if model.base_param_overrides:
            self._aero_base.update(model.base_param_overrides)

        self.tip_to_tip = model.tip_to_tip
        self.AR_wing = model.AR_wing
        self.AR_tail = model.AR_tail
        self.S_perp_fus = model.S_perp_fus
        self.cg_fus_local_x = model.cg_fus_local_x
        self.cz_rudder = model.cz_rudder
        self.prop_radius = model.prop_radius
        self.genes = model.genes
        self.genes_dict = model.genes_dict
        self._surface_kinds = list(getattr(model, "surface_kinds", []))

        self._randomizable_param_names = list(self._aero_base.keys())

        self._aero_link_idx = model.link_indices(entity)

        # Noise overrides from config (optional)
        if model.noise_params:
            self.noise_sigma_mag = float(model.noise_params.get("sigma_mag", self.noise_sigma_mag))
            self.noise_sigma_dir = float(model.noise_params.get("sigma_dir", self.noise_sigma_dir))
            self.noise_sigma_param = float(model.noise_params.get("sigma_param", self.noise_sigma_param))
            self.noise_sigma_cp = float(model.noise_params.get("sigma_cp", self.noise_sigma_cp))

        if self.prop_radius <= 0.0:
            raise RuntimeError("Invalid propeller radius extracted from drone model.")
        if self.AR_wing <= 0.0:
            raise RuntimeError("Invalid wing aspect ratio extracted from drone model.")

        if not self._geom:
            raise RuntimeError("Drone model did not provide aerodynamic geometry.")

        self._log_geometry(model)

    def _log_geometry(self, model: DroneAeroModel):
        dims = getattr(model, "debug_dims", {})
        if not dims:
            return

        print("[AeroSolver]: Parsed URDF geometry:")
        if "c_fus" in dims and "l_fus" in dims:
            print(f"  Fuselage C={dims['c_fus']:.4f} m, L={dims['l_fus']:.4f} m")
        if "c_w" in dims and "l_w" in dims:
            print(f"  Wing C={dims['c_w']:.4f} m, L={dims['l_w']:.4f} m")
        if "c_e" in dims and "l_e" in dims:
            print(f"  Elevator C={dims['c_e']:.4f} m, L={dims['l_e']:.4f} m")
        if "c_r" in dims and "l_r" in dims:
            print(f"  Rudder C={dims['c_r']:.4f} m, L={dims['l_r']:.4f} m")

    def _wing_side_keys(self, name: str) -> tuple[str, str]:
        """Return canonical (left, right) keys for a wing parameter name."""
        suffix = "_wing"
        if name.endswith(suffix):
            base = name[: -len(suffix)]
        else:
            base = name
        return (f"{base}_wing_left", f"{base}_wing_right")

    def _elevator_side_keys(self, name: str) -> tuple[str, str]:
        """Return canonical (left, right) keys for an elevator parameter name."""
        suffix = "_elevator"
        if name.endswith(suffix):
            base = name[: -len(suffix)]
        else:
            base = name
        return (f"{base}_elevator_left", f"{base}_elevator_right")

    def _ensure_wing_param_entries(self):
        """Guarantee left/right wing parameter keys exist in _aero_base."""
        for name in getattr(self, "_wing_param_names", []):
            base_val = self._aero_base.get(name, 0.0)
            left_key, right_key = self._wing_side_keys(name)
            self._aero_base.setdefault(left_key, base_val)
            self._aero_base.setdefault(right_key, base_val)

    def _ensure_elevator_param_entries(self):
        """Guarantee left/right elevator parameter keys exist in _aero_base."""
        for name in getattr(self, "_elevator_param_names", []):
            base_val = self._aero_base.get(name, 0.0)
            left_key, right_key = self._elevator_side_keys(name)
            self._aero_base.setdefault(left_key, base_val)
            self._aero_base.setdefault(right_key, base_val)

    # ------------------------------------------------------------------
    # NACA 4-digit overrides (SimpleDrone only)
    # ------------------------------------------------------------------
    def _normalize_naca_code(self, code: str | int | float | None) -> str | None:
        if code is None:
            return None
        s = str(code).strip().upper()
        if not s:
            return None
        if s.startswith("NACA"):
            s = s[4:].strip()
        if s.isdigit():
            return s.zfill(4)
        return s

    def _find_naca4_csv_path(self) -> Path | None:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "src" / "naca_generation" / "naca4.csv"
            if candidate.exists():
                return candidate
        return None

    def _load_naca4_row(self, naca_code: str, csv_path: Path) -> dict[str, float] | None:
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self._normalize_naca_code(row.get("airfoil")) == naca_code:
                        return {
                            "slope": float(row["slope"]),
                            "alpha_stall": float(row["alpha_stall"]),
                            "alpha0": float(row["alpha0"]),
                            "cd0": float(row["cd0"]),
                            "re_nom": float(row["ReNom"]),
                        }
        except (OSError, KeyError, ValueError, TypeError):
            return None
        return None

    def _set_param_field(self, name: str, value: float) -> None:
        if not hasattr(self, name):
            return
        field = getattr(self, name)
        if not hasattr(self, "B"):
            return
        for b in range(self.B):
            field[b] = float(value)
        if hasattr(self, "_aero_base"):
            self._aero_base[name] = float(value)

    def apply_naca_wing_override(
        self, naca_code: str | int | float | None, csv_path: str | Path | None = None
    ) -> None:
        code = self._normalize_naca_code(naca_code)
        if not code:
            return
        path = Path(csv_path) if csv_path is not None else self._find_naca4_csv_path()
        if path is None or not path.exists():
            return
        entry = self._load_naca4_row(code, path)
        if entry is None:
            return

        def _entry_to_overrides(row: dict[str, float]) -> tuple[dict[str, float], float | None]:
            cl_alpha = row["slope"] * (180.0 / math.pi)  # per-degree -> per-rad
            alpha0 = math.radians(row["alpha0"])
            alpha_stall = row["alpha_stall"]
            cd0 = row["cd0"]
            return (
                {
                    "cl_alpha_2d": cl_alpha,
                    "alpha0_2d": alpha0,
                    "cd0": cd0,
                    "alpha_stall_deg": alpha_stall,
                },
                row.get("re_nom"),
            )

        overrides, re_nom = _entry_to_overrides(entry)

        for key, val in overrides.items():
            self._set_param_field(key, val)
            left_key, right_key = self._wing_side_keys(key)
            self._set_param_field(left_key, val)
            self._set_param_field(right_key, val)

        if re_nom is not None and hasattr(self, "re_nom_link") and hasattr(self, "kind"):
            for b in range(self.B):
                for l in range(self.L):
                    if int(self.kind[l]) == 1:
                        self.re_nom_link[b, l] = float(re_nom)

        tail_entry = self._load_naca4_row("0216", path)
        if tail_entry is None:
            return
        tail_overrides, tail_re_nom = _entry_to_overrides(tail_entry)

        for key, val in tail_overrides.items():
            self._set_param_field(key, val)
            left_key, right_key = self._elevator_side_keys(key)
            self._set_param_field(left_key, val)
            self._set_param_field(right_key, val)

        if hasattr(self, "cl_alpha_2d_link") and hasattr(self, "kind"):
            for b in range(self.B):
                for l in range(self.L):
                    k = int(self.kind[l])
                    if k == 2 or k == 3:
                        self.cl_alpha_2d_link[b, l] = float(tail_overrides["cl_alpha_2d"])
                        self.alpha0_2d_link[b, l] = float(tail_overrides["alpha0_2d"])
                        self.cd0_link[b, l] = float(tail_overrides["cd0"])
                        self.alpha_stall_deg_link[b, l] = float(tail_overrides["alpha_stall_deg"])

        if tail_re_nom is not None and hasattr(self, "re_nom_link") and hasattr(self, "kind"):
            for b in range(self.B):
                for l in range(self.L):
                    k = int(self.kind[l])
                    if k == 2 or k == 3:
                        self.re_nom_link[b, l] = float(tail_re_nom)

        self._capture_global_param_nominals()
        self._capture_link_param_nominals()

    def randomize_aero_params(self, envs_idx, sigma=None):
        """
        Randomize aerodynamic parameters for a subset of environments on-device.
        """
        if sigma is None:
            sigma = getattr(self, "noise_sigma_param", 0.0)
        n_envs = int(len(envs_idx))
        if n_envs == 0:
            return

        env_ids = envs_idx.reshape(-1).to(device=self._aero_device, dtype=torch.long)
        env_ids_np = env_ids.detach().cpu().numpy()
        sigma_f = float(sigma)

        for name, _, fld in self._iter_randomizable_params():
            nominal = getattr(self, "_param_nominal_torch", {}).get(name, None)
            if nominal is None:
                continue
            values = self._get_rand_scratch_1d(name, n_envs)
            values.copy_(nominal.index_select(0, env_ids))
            if sigma_f > 0.0:
                noise = torch.randn_like(values)
                values.mul_(1.0 + sigma_f * noise)
            self._set_taichi_field_rows_1d(fld, env_ids_np, values)

        nominal_map = getattr(self, "_link_param_nominal_torch", {})
        for name in getattr(self, "_randomizable_link_field_names", ()):
            fld = getattr(self, name, None)
            nominal = nominal_map.get(name)
            if fld is None or nominal is None:
                continue
            values = self._get_rand_scratch_2d(name, n_envs)
            values.copy_(nominal.index_select(0, env_ids))
            if sigma_f > 0.0:
                noise = torch.randn_like(values)
                values.mul_(1.0 + sigma_f * noise)
            self._set_taichi_field_rows_2d(fld, env_ids_np, values)

        self._refresh_param_buffers()

    # ---------------------------------------------------------------------
    # Taichi kernels / device-side logic
    # ---------------------------------------------------------------------
    @ti.kernel
    def _aero_compute_kernel(self, rigid: ti.template()):
        """
        Taichi kernel computing aerodynamic forces/torques from the current
        rigid-body state.

        The kernel:
        * filters the throttle to obtain a smooth prop thrust command,
        * computes induced velocity at the propeller,
        * performs a first pass over all surfaces except the elevators,
        * then a second pass over the elevators including wing downwash,
        * and applies stochastic force noise inline (no extra pass).
        """
        # 0) Update filtered throttle and induced velocity at the prop
        for b in range(self.B):
            self._propeller_pass(rigid, b)

        # 1) First pass: all surfaces except tail (elevators)
        for b, l in ti.ndrange(self.B, self.L):
            self._main_surfaces_pass(rigid, b, l)

        # 2) Second pass: elevators (tail), using wing downwash
        for b, l in ti.ndrange(self.B, self.L):
            self._tail_surfaces_pass(rigid, b, l)

        # Noise applied inline: no third pass needed.

    @ti.func
    def _propeller_pass(self, rigid: ti.template(), b: int):
        # Speed clamp
        const_V_MAX = 40.0

        # Simple first-order low-pass on throttle (cutoff = prop_cutoff_hz)
        alpha_lpf = self._substep_dt / (
            self._substep_dt
            + 1.0 / (2.0 * ti.math.pi * self.prop_cutoff_hz[b])
        )
        self._thr_flt[b] += alpha_lpf * (
            float(self._thr_raw[b]) - float(self._thr_flt[b])
        )

        # Max thrust and prop inflow speed
        T_prop = float(self._thr_flt[b]) * self.max_thrust[b]
        prop_idx = self._prop_link_idx[None]

        # Air velocity in prop frame (body frame)
        v_body_prop = self._get_wind_in_body(rigid, prop_idx, b)

        # Axial forward speed along local axis (here aligned with z)
        u = ti.abs(v_body_prop.z)

        # Induced velocity from momentum theory (Selig)
        v_ind_val = (
            -u
            + ti.sqrt(
                u * u
                + 2.0
                * T_prop
                / (self.rho[b] * ti.math.pi * (self.prop_radius * self.prop_radius))
            )
        ) * 0.5
        v_ind_val = ti.math.clamp(v_ind_val, 0.0, const_V_MAX)
        self.v_ind[b] = ti.cast(v_ind_val, ti.f16)

    @ti.func
    def _main_surfaces_pass(self, rigid: ti.template(), b: int, l: int):
        # Speed clamp
        const_V_MAX = 40.0
        const_V_MIN = 0.1
        const_MU_AIR = 1.81e-5  # dynamic viscosity [kg/(m*s)] for Reynolds estimate

        # Reset accumulators (keep f32 precision)
        self.force_b[b, l] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.cp_b[b, l] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.Reynolds[b, l] = 0.0

        kind = ti.cast(self.kind[l], ti.i32)
        if (kind != 2) and (kind != 3):
            side = ti.cast(self.side[l], ti.i32)
            slip_code = ti.cast(self.slip_code[l], ti.i32)
            rho = self.rho[b]

            # Air velocity in link body frame
            v_body = self._get_wind_in_body(rigid, self._link_idx[l], b)

            # Slipstream effect
            slip = 0.0
            if slip_code == 0:      # fuselage
                slip = float(self.k_slip_fus[b])
            elif slip_code == 1:    # tail
                slip = float(self.k_slip_tail[b])
            elif slip_code == 2:    # in prop wash
                slip = float(self._wing_param(self.k_slip_wing, self.k_slip_wing_left, self.k_slip_wing_right, b, side))

            # Reduce x-component for induced velocity
            v_body.x -= slip * ti.cast(self.v_ind[b], ti.f32)

            # Speed and aerodynamic angles
            v_mod = v_body.norm() + 1e-6
            v_mod = ti.math.clamp(v_mod, const_V_MIN, const_V_MAX)
            alpha = ti.atan2(v_body.z, v_body.x)
            beta  = ti.asin(ti.math.clamp(v_body.y / v_mod, -1.0, 1.0))

            # Effective geometry (morphing / folding)
            S_eff = self._compute_eff_S(rigid, b, l)
            AR_eff = self._compute_eff_AR(rigid, b, l, beta)

            # Reynolds number (per-surface)
            c_ref = ti.max(ti.cast(self.chord[l], ti.f32), 1e-6)
            self.Reynolds[b, l] = (rho * v_mod * c_ref) / const_MU_AIR

            # Aero coefficients (lift/drag)
            cl, cd = self._compute_coeff(
                b, l, AR_eff, alpha, beta, kind, side
            )

            # Dynamic pressure * area
            v_mod2 = v_mod * v_mod
            qS = 0.5 * rho * S_eff * v_mod2
            L = qS * cl
            D = qS * cd

            # Force in body frame
            Fb = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            cp = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)

            if kind == 4:
                # Prop: thrust along +z in prop frame
                Fb = (
                    ti.Vector([0.0, 0.0, 1.0], dt=ti.f32)
                    * float(self.prop_thrust_sign[b])
                    * float(self._thr_flt[b])
                    * self.max_thrust[b]
                )
            elif kind == 3:
                # Rudder: forces in YZ plane
                cosa = ti.cos(alpha)
                cosa2 = cosa * cosa
                S = L  # sideforce magnitude tied to CL
                Fb = self._rot_yz(alpha, beta) @ ti.Vector([D * cosa2, S * cosa2, 0.0], dt=ti.f32)
                cp = ti.cast(self._cp_rudder(rigid, b, alpha, beta, l), ti.f32)
            else:
                # Fuselage or wing: forces in XZ plane
                cosb = ti.cos(beta)
                cosb2 = cosb * cosb
                if kind == 1:  # wing
                    Fb = self._rot_yz(alpha, beta) @ ti.Vector([D * cosb2, 0.0, L * cosb2], dt=ti.f32)
                    idx = 0 if side == 1 else 1
                    self.cl_wing_b[b, idx] = ti.cast(cl, ti.f16)  # store CL for tail
                    cp = ti.cast(self._cp_wing(rigid, b, alpha, beta, l), ti.f32)
                else:  # fuselage
                    Fb = self._rot_yz(alpha, beta) @ ti.Vector([D, 0.0, L], dt=ti.f32)
                    cp = ti.cast(self._cp_fus(rigid, b, alpha, beta, l), ti.f32)

            # ---- Inline noise (no extra pass) ---------------------------------
            do_noise = (self.noise_sigma_mag > 0.0) or (self.noise_sigma_dir > 0.0) or (self.noise_sigma_cp > 0.0)
            if do_noise:
                Fb, cp = self._apply_noise(
                    Fb, cp, l, alpha, beta, kind,
                    self.noise_sigma_mag, self.noise_sigma_dir, self.noise_sigma_cp,
                )

            # Force magnitude clamp (after noise)
            Fcap = self.force_cap[b]
            Fn = Fb.norm() + 1e-9
            scale = ti.min(1.0, Fcap / Fn)
            Fb *= scale

            # Store partial results
            self.force_b[b, l] = Fb
            self.cp_b[b, l] = cp

            if ti.static(self._enable_aero_debug_buffers):
                self.alpha_dbg[b, l] = ti.cast(alpha, ti.f16)
                self.beta_dbg[b, l] = ti.cast(beta, ti.f16)
                self.drag_dbg[b, l] = ti.cast(D, ti.f16)
                if kind != 3:
                    self.lift_dbg[b, l] = ti.cast(L, ti.f16)
                    self.side_force_dbg[b, l] = ti.cast(0.0, ti.f16)
                else:
                    self.lift_dbg[b, l] = ti.cast(0.0, ti.f16)
                    self.side_force_dbg[b, l] = ti.cast(L, ti.f16)
                if kind == 4:
                    self.drag_dbg[b, l] = ti.cast(self._thr_flt[b], ti.f16)

    @ti.func
    def _tail_surfaces_pass(self, rigid: ti.template(), b: int, l: int):
        # Speed clamp
        const_V_MAX = 50.0
        const_V_MIN = 0.1
        const_MU_AIR = 1.81e-5  # dynamic viscosity [kg/(m*s)] for Reynolds estimate

        kind = ti.cast(self.kind[l], ti.i32)
        if (kind == 2) or (kind == 3):
            side = ti.cast(self.side[l], ti.i32)
            slip_code = ti.cast(self.slip_code[l], ti.i32)
            rho = self.rho[b]
            # Initialize branch-local values so Taichi always sees them as defined.
            cl_w_mean = 0.0
            k_eps = 0.0
            eps = 0.0
            L_t = 0.0
            D_t = 0.0

            v_body_tail = self._get_wind_in_body(rigid, self._link_idx[l], b)

            slip = 0.0
            if slip_code == 0:
                slip = float(self.k_slip_fus[b])
            elif slip_code == 1:
                slip = float(self._elevator_param(self.k_slip_tail, self.k_slip_tail_elevator_left, self.k_slip_tail_elevator_right, b, side))
            elif slip_code == 2:
                slip = float(self._wing_param(self.k_slip_wing, self.k_slip_wing_left, self.k_slip_wing_right, b, side))

            v_body_tail.x -= slip * ti.cast(self.v_ind[b], ti.f32)

            Vt = v_body_tail.norm() + 1e-6
            Vt = ti.math.clamp(Vt, const_V_MIN, const_V_MAX)

            # Tail angles
            alphat = ti.atan2(v_body_tail.z, v_body_tail.x)
            betat  = ti.asin(ti.math.clamp(v_body_tail.y / Vt, -1.0, 1.0))

            # Reynolds number (per-tail surface)
            c_ref_t = ti.max(ti.cast(self.chord[l], ti.f32), 1e-6)
            self.Reynolds[b, l] = (rho * Vt * c_ref_t) / const_MU_AIR

            S_eff_t = self._compute_eff_S(rigid, b, l)
            AR_eff_t = self._compute_eff_AR(rigid, b, l, betat)

            Fb_t = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            cp_t = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            alpha_eff = alphat

            if kind == 2:
                # Mean wing CL for downwash
                idx = 0 if side == 1 else 1
                cl_w_mean = ti.cast(self.cl_wing_b[b, idx], ti.f32)

                # Effective tail AoA with downwash
                k_eps = self._elevator_param(self.k_eps_tail, self.k_eps_tail_elevator_left, self.k_eps_tail_elevator_right, b, side)
                eps = (k_eps * cl_w_mean) / (ti.math.pi * self.AR_wing)
                alpha_eff = alphat - eps

                # Tail coefficients (kind=2)
                cl_t, cd_t = self._compute_coeff(b, l, AR_eff_t, alpha_eff, betat, 2, side)

                cosb = ti.cos(betat)
                cosb2 = cosb * cosb

                # Tail forces
                Vt2 = Vt * Vt
                qS_t = 0.5 * rho * S_eff_t * Vt2 * cosb2
                L_t = qS_t * cl_t
                D_t = qS_t * cd_t
                Fb_t = self._rot_yz(alpha_eff, betat) @ ti.Vector([D_t, 0.0, L_t], dt=ti.f32)
                # Tail center of pressure
                cp_t = ti.cast(self._cp_elev(rigid, b, alpha_eff, betat, l), ti.f32)
            else:
                # Rudder (kind=3) handled here without downwash
                cl_t, cd_t = self._compute_coeff(b, l, AR_eff_t, alphat, betat, 3, side)
                cosa = ti.cos(alphat)
                cosa2 = cosa * cosa
                Vt2 = Vt * Vt
                qS_t = 0.5 * rho * S_eff_t * Vt2
                L_t = qS_t * cl_t
                D_t = qS_t * cd_t
                Fb_t = self._rot_yz(alphat, betat) @ ti.Vector([D_t * cosa2, L_t * cosa2, 0.0], dt=ti.f32)
                cp_t = ti.cast(self._cp_rudder(rigid, b, alphat, betat, l), ti.f32)
            # ---- Inline noise also for tail -----------------------------------
            do_noise_t = (self.noise_sigma_mag > 0.0) or (self.noise_sigma_dir > 0.0) or (self.noise_sigma_cp > 0.0)
            if do_noise_t:
                Fb_t, cp_t = self._apply_noise(
                    Fb_t, cp_t, l, alpha_eff, betat, 2,
                    self.noise_sigma_mag, self.noise_sigma_dir, self.noise_sigma_cp,
                )

            # Clamp (after noise)
            Fcap = self.force_cap[b]
            Fn_t = Fb_t.norm() + 1e-9
            scale_t = ti.min(1.0, Fcap / Fn_t)
            Fb_t *= scale_t

            # Store
            self.force_b[b, l] = Fb_t
            self.cp_b[b, l] = cp_t

            if ti.static(self._enable_aero_debug_buffers):
                self.alpha_tail_raw_dbg[b, l] = ti.cast(alphat, ti.f16)
                if kind == 2:
                    self.downwash_eps_dbg[b, l] = ti.cast(eps, ti.f16)
                    self.cl_wing_for_tail_dbg[b, l] = ti.cast(cl_w_mean, ti.f16)
                    self.k_eps_tail_dbg[b, l] = ti.cast(k_eps, ti.f16)
                self.alpha_dbg[b, l] = ti.cast(alpha_eff, ti.f16)
                self.beta_dbg[b, l]  = ti.cast(betat, ti.f16)
                self.lift_dbg[b, l]  = ti.cast(L_t, ti.f16)
                self.drag_dbg[b, l]  = ti.cast(D_t, ti.f16)

    @ti.func
    def _seff_ratio(self, theta: ti.f32, theta_max: ti.f32, ratio_folded: ti.f32):
        t = 0.0
        if theta_max > 1e-6:
            t = ti.min(1.0, ti.abs(theta) / theta_max)
        # Linear interpolation: 1 at theta=0, ratio_folded at theta=theta_max
        return 1.0 - t * (1.0 - ratio_folded)

    @ti.func
    def _compute_eff_S(self, rigid: ti.template(), b: int, l: int):
        """
        Compute effective reference area S for a surface.

        The model is intentionally simple:
        - yaw folding scales S linearly from 1.0 to `seff_yaw_ratio`,
        - pitch folding scales S linearly from 1.0 to `seff_pitch_ratio`,
        - the two effects multiply.
        """
        S0 = ti.cast(self.area[l], ti.f32)
        S_eff = S0
        return ti.max(1e-6, S_eff)

    @ti.func
    def _compute_eff_span(self, rigid: ti.template(), b: int, l: int, beta: ti.f32):
        """
        Compute effective span for a surface.
        """
        b0 = ti.cast(self.span[l], ti.f32)
        b_eff = b0 * ti.cos(beta)

        return ti.max(0.0, b_eff)

    @ti.func
    def _compute_eff_AR(self, rigid: ti.template(), b: int, l: int, beta: ti.f32):
        """
        Compute effective aspect ratio for a surface.

        This includes fuselage interference effects.
        """
        k = ti.cast(self.kind[l], ti.i32)
        AR_eff = ti.cast(self.AR[l], ti.f32)

        if (k == 1) or (k == 2):
            b_eff = self._compute_eff_span(rigid, b, l, beta)
            S_eff = self._compute_eff_S(rigid, b, l)

            total_b_eff = 2.0 * b_eff
            if k == 1:
                total_b_eff += ti.cast(self.fus_width[None], ti.f32)

            AR_eff = (total_b_eff * b_eff) / ti.max(S_eff, 1e-6)

        return AR_eff

    # ------------------------------------------------------------------
    # Aerodynamic coefficients and centers of pressure
    # ------------------------------------------------------------------
    @ti.func
    def _compute_coeff(self, b, l, AR, alpha, beta, kind, side):
        """
        Compute lift (cl) and drag (cd) coefficients with a stall model.

        Parameters
        ----------
        b :
            Environment index.
        AR :
            Aspect ratio of the surface.
        alpha, beta :
            Aerodynamic angles in radians.
        kind :
            Surface kind (0=fuselage, 1=wing, 2=elevator, 3=rudder, 4=prop).
        """
        cd0 = self.cd0_link[b, l]
        oswald_e = ti.max(self.oswald_efficiency_link[b, l], 1e-3)
        cut = self.alpha_stall_deg_link[b, l] * ti.math.pi/180
        cl_alpha = self.cl_alpha_2d_link[b, l]
        alpha0 = self.alpha0_2d_link[b, l]
        smooth = self.m_smooth_link[b, l]
        w = self.w_link[b, l]
        # For the rudder (kind=3) use beta as effective alpha
        if kind == 3:
            alpha = beta

        alpha = self._wrap_pm_90(alpha)
        beta  = self._wrap_pm_90(beta)

        # 2*pi-periodic wrapped angle
        two_pi = 2.0 * ti.math.pi
        a_wr = alpha - two_pi * ti.floor((alpha + ti.math.pi) / two_pi)
        a_abs = ti.abs(a_wr)

        # Folded angle in [0, pi/2]
        a_fold = ti.min(a_abs, ti.math.pi - a_abs)

        if kind == 1:  # wing -> allow left/right split
            cl_alpha = self._wing_param(self.cl_alpha_2d, self.cl_alpha_2d_wing_left, self.cl_alpha_2d_wing_right, b, side)
            alpha0 = self._wing_param(self.alpha0_2d, self.alpha0_2d_wing_left, self.alpha0_2d_wing_right, b, side)
            cd0 = self._wing_param(self.cd0, self.cd0_wing_left, self.cd0_wing_right, b, side)
            cut = self._wing_param(self.alpha_stall_deg, self.alpha_stall_deg_wing_left, self.alpha_stall_deg_wing_right, b, side) * ti.math.pi / 180.0
            smooth = self._wing_param(self.m_smooth, self.m_smooth_wing_left, self.m_smooth_wing_right, b, side)
            w = self._wing_param(self.w, self.w_wing_left, self.w_wing_right, b, side)
        elif kind == 2:  # elevator
            cl_alpha = self._elevator_param(self.cl_alpha_2d, self.cl_alpha_2d_elevator_left, self.cl_alpha_2d_elevator_right, b, side)
            alpha0 = self._elevator_param(self.alpha0_2d, self.alpha0_2d_elevator_left, self.alpha0_2d_elevator_right, b, side)
            cd0 = self._elevator_param(self.cd0, self.cd0_elevator_left, self.cd0_elevator_right, b, side)
            cut = self._elevator_param(self.alpha_stall_deg, self.alpha_stall_deg_elevator_left, self.alpha_stall_deg_elevator_right, b, side) * ti.math.pi / 180.0
            smooth = self._elevator_param(self.m_smooth, self.m_smooth_elevator_left, self.m_smooth_elevator_right, b, side)
            w = self._elevator_param(self.w, self.w_elevator_left, self.w_elevator_right, b, side)

        cl_a = cl_alpha * AR / (2.0 + ti.sqrt(AR * AR + 4.0))

        cut_eff = cut
        if (kind == 1) or (kind == 2):   # tipicamente solo wing; se vuoi anche elevator lascia così
            cut_eff = self._alpha_stall_3d(alpha0, cut, AR, cl_a)

        ref = alpha0

        f_re = 1.0
        if (kind == 1) or (kind == 2) or (kind == 3):
            re_nom = self.re_nom_link[b, l]
            if re_nom > 1e-6:
                re_val = ti.max(self.Reynolds[b, l], 0.0)
                r = ti.math.clamp(re_val / re_nom, 0.0, 1.0)
                a = ti.max(self.re_a_link[b, l], 0.0)
                f_re = 1.0 - ti.pow(1.0 - r, a)

        cl_lin = cl_a * (a_wr - ref)
        cl_lin *= f_re
        cd_lin = cd0 + cl_lin * cl_lin / (ti.math.pi * oswald_e * AR)

        # Post-stall flat-plate model (periodic)
        sa, ca = ti.sin(a_wr), ti.cos(a_wr)
        cl_st = 2.0 * sa * ca
        cd_st = 2.0 * sa * sa
        cl_st *= f_re
        cd_st *= f_re
        if (kind == 1) or (kind == 2) or (kind == 3):
            ar_safe = ti.max(AR, 1e-6)
            k_cd = 1.0 - 0.41 * (1.0 - ti.exp(-17.0 / ar_safe))  # flat-plate AR correction
            correction = 1.0 - w * (1.0 - k_cd)  # w scales how much is applied
            cl_st *= correction
            cd_st *= correction

        # Smooth blend between linear and post-stall using m_smooth as width
        width = smooth * cut_eff
        width = ti.max(width, 1e-3)  # avoid extremely sharp transitions / div by zero
        s_arg = (a_fold - cut_eff) / width
        s_blend = 0.5 * (1.0 + ti.tanh(s_arg))
        s_blend = ti.min(ti.max(s_blend, 0.0), 1.0)

        # Interpolated coefficients
        cl = (1.0 - s_blend) * cl_lin + s_blend * cl_st
        cd = (1.0 - s_blend) * cd_lin + s_blend * cd_st

        # Fuselage: only drag (no lift)
        if kind == 0:
            cl = 0.0
            cd = cd0

        return cl, cd

    @ti.func
    def _alpha_stall_3d(self, alpha0: ti.f32, alpha_stall_2d: ti.f32, AR: ti.f32, cl_a_3d: ti.f32) -> ti.f32:
        # delta = how far from zero-lift to stall (2D)
        delta = alpha_stall_2d - alpha0

        denom = cl_a_3d * (1.0 + 2.0 * delta / AR)

        alpha_stall_3d = alpha_stall_2d
        if (denom > 1e-3):
            alpha_stall_3d = alpha0 + (2.0 * ti.math.pi * delta) / denom

        return alpha_stall_3d

    @ti.func
    def _cp_wing(self, rigid: ti.template(), b, alpha, beta, l):
        """
        Wing center of pressure (x-offset in local frame).

        Uses a simple shift between `cp_start` and `cp_end` as a function
        of effective angle of attack.
        """
        # Joint positions are available here if CP models need them.
        two_pi = 2.0 * ti.math.pi
        aa_a = ti.min(
            ti.abs(alpha - ti.math.pi * ti.floor((alpha + ti.math.pi) / two_pi)),
            ti.math.pi / 2,
        )
        aa = ti.min(aa_a, ti.math.pi / 2)
        cp_start = self._wing_param(self.cp_start, self.cp_start_wing_left, self.cp_start_wing_right, b, self.side[l])
        cp_end = self._wing_param(self.cp_end, self.cp_end_wing_left, self.cp_end_wing_right, b, self.side[l])
        cp_frac = aa / (ti.math.pi / 2) * (cp_end - cp_start) + cp_start
        shift = (cp_frac - 0.25) * self.chord[l]
        cx = shift * ti.cos(beta)
        cy = shift * ti.sin(beta)

        return ti.Vector([cx, cy, 0.0])

    @ti.func
    def _cp_fus(self, rigid: ti.template(), b, alpha, beta, l):
        """
        Fuselage center of pressure (x-offset in fuselage frame).
        """
        # Joint positions are available here if CP models need them.
        two_pi = 2.0 * ti.math.pi
        aa_a = ti.min(
            ti.abs(alpha - ti.math.pi * ti.floor((alpha + ti.math.pi) / two_pi)),
            ti.math.pi / 2,
        )
        aa = ti.min(aa_a, ti.math.pi / 2)
        cp_start = self.cp_start_link[b, l]
        cp_end = self.cp_end_link[b, l]
        cp_frac = aa / (ti.math.pi / 2) * (cp_end - cp_start) + cp_start
        shift = (cp_frac - 0.25) * self.chord[l]
        cx = shift
        cy = 0.0
        return ti.Vector([cx, cy, 0.0])

    @ti.func
    def _cp_rudder(self, rigid: ti.template(), b, alpha, beta, l):
        """
        Rudder center of pressure: [x, 0, z] with fixed z offset.
        """
        # Joint positions are available here if CP models need them.
        two_pi = 2.0 * ti.math.pi
        aa_a = ti.min(
            ti.abs(alpha - ti.math.pi * ti.floor((alpha + ti.math.pi) / two_pi)),
            ti.math.pi / 2,
        )
        aa = ti.min(aa_a, ti.math.pi / 2)
        cp_start = self.cp_start_link[b, l]
        cp_end = self.cp_end_link[b, l]
        cp_frac = aa / (ti.math.pi / 2) * (cp_end - cp_start) + cp_start
        cx = (cp_frac - 0.25) * self.chord[l]
        return ti.Vector([cx, 0.0, ti.cast(self.cz_rudder, ti.f32)])

    @ti.func
    def _cp_elev(self, rigid: ti.template(), b, alpha, beta, l):
        """
        Elevator center of pressure [x, y, z] with a small y offset.

        The y offset distinguishes left/right halves of the tailplane so
        that differential elevator deflections can generate rolling moments.
        """
        # Joint positions are available here if CP models need them.
        two_pi = 2.0 * ti.math.pi
        aa_a = ti.min(
            ti.abs(alpha - ti.math.pi * ti.floor((alpha + ti.math.pi) / two_pi)),
            ti.math.pi / 2,
        )
        aa = ti.min(aa_a, ti.math.pi / 2)
        cp_start = self._elevator_param(self.cp_start, self.cp_start_elevator_left, self.cp_start_elevator_right, b, self.side[l])
        cp_end = self._elevator_param(self.cp_end, self.cp_end_elevator_left, self.cp_end_elevator_right, b, self.side[l])
        cp_frac = aa / (ti.math.pi / 2) * (cp_end - cp_start) + cp_start
        cx = (cp_frac - 0.25) * self.chord[l]
        cy = (
            0.25
            * (float(self.area[l]) / float(self.chord[l]))
            * float(self.side[l])
        )
        # CP is expressed in the elevator aero frame; avoid hardcoded offsets so the URDF defines placement.
        return ti.Vector([cx, ti.cast(cy, ti.f32), 0.0])

    # ------------------------------------------------------------------
    # Wing parameter helper (left/right)
    # ------------------------------------------------------------------
    @ti.func
    def _wing_param(self, default_field: ti.template(), left_field: ti.template(), right_field: ti.template(), b: int, side: int):
        """
        Select a parameter for wings allowing left/right split.
        Falls back to the default scalar if side is 0 or fields are missing.
        """
        # A single selector handles left/right using the per-surface side flag.
        val = default_field[b]
        if side < 0:
            val = left_field[b]
        elif side > 0:
            val = right_field[b]
        return val

    @ti.func
    def _elevator_param(self, default_field: ti.template(), left_field: ti.template(), right_field: ti.template(), b: int, side: int):
        """
        Select a parameter for elevators allowing left/right split.
        """
        # A single selector handles left/right using the per-surface side flag.
        val = default_field[b]
        if side < 0:
            val = left_field[b]
        elif side > 0:
            val = right_field[b]
        return val


AeroSolver = SimpleDroneAeroSolver
