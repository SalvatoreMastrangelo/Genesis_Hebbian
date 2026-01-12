import torch
import gstaichi as ti

from genesis.engine.solvers.base_aero_solver import BaseAeroSolver
from genesis.engine.entities import RigidEntity  # for get_link()
from genesis.assets.urdf.mydrone.drone import DroneAeroModel, SurfaceKind


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
            "cp_start",
            "cp_end",
            "cg_to_chord",
            "k_slip_tail",
            "k_eps_tail",
        ]

    def _const_init(self):
        """
        Initialize base aerodynamic parameters and frame names.

        All values here are *nominal* and stored in `_aero_base`.
        They are later copied into Taichi fields (one scalar field per env)
        in `add_target`, so they can be randomized per environment.
        """
        # Base values (per env, uniform at start)
        self._aero_base = dict(DroneAeroModel.DEFAULT_BASE_PARAMS)
        self._ensure_wing_param_entries()
        self._ensure_elevator_param_entries()

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

        # Now that _B and L are known, allocate per-link parameter fields
        self.cd0_link         = ti.field(ti.f32, shape=(self._B, self.L))
        self.alpha0_2d_link    = ti.field(ti.f32, shape=(self._B, self.L))
        self.cl_alpha_2d_link  = ti.field(ti.f32, shape=(self._B, self.L))
        self.alpha_stall_deg_link = ti.field(ti.f32, shape=(self._B, self.L))
        self.m_smooth_link     = ti.field(ti.f32, shape=(self._B, self.L))
        self.cp_start_link     = ti.field(ti.f32, shape=(self._B, self.L))
        self.cp_end_link       = ti.field(ti.f32, shape=(self._B, self.L))
        self.cg_to_chord_link  = ti.field(ti.f32, shape=(self._B, self.L))
        self.re_nom_link       = ti.field(ti.f32, shape=(self._B, self.L))
        self.re_a_link         = ti.field(ti.f32, shape=(self._B, self.L))

        self.k_slip_wing_link  = ti.field(ti.f32, shape=(self._B, self.L))
        self.k_slip_tail_link  = ti.field(ti.f32, shape=(self._B, self.L))
        self.k_eps_tail_link   = ti.field(ti.f32, shape=(self._B, self.L))
        self.k_slip_fus_link   = ti.field(ti.f32, shape=(self._B, self.L))

        surface_frames = list(model.frames)
        surface_kinds = list(getattr(model, "surface_kinds", []))
        if len(surface_kinds) != len(surface_frames):
            raise RuntimeError("Drone model did not expose per-surface kinds.")

        def require_param(frame: str, params: dict, key: str) -> float:
            if key not in params:
                raise ValueError(f"Surface '{frame}' missing required key '{key}' in aero_parameters.yaml.")
            return float(params[key])

        for i in range(self.L):
            p = model.per_surface_params[i]
            frame = surface_frames[i]
            kind = surface_kinds[i]

            if kind == SurfaceKind.PROPELLER:
                cd0_val = alpha0_val = cl_alpha_val = 0.0
                alpha_stall_val = m_smooth_val = 0.0
                cp_start_val = cp_end_val = 0.0
                cg_to_chord_val = 0.0
            elif kind == SurfaceKind.FUSELAGE:
                cd0_val = require_param(frame, p, "cd0")
                alpha0_val = cl_alpha_val = 0.0
                alpha_stall_val = m_smooth_val = 0.0
                cp_start_val = require_param(frame, p, "cp_start")
                cp_end_val = require_param(frame, p, "cp_end")
                cg_to_chord_val = require_param(frame, p, "cg_to_chord")
            else:
                cd0_val = require_param(frame, p, "cd0")
                alpha0_val = require_param(frame, p, "alpha0_2d")
                cl_alpha_val = require_param(frame, p, "cl_alpha_2d")
                alpha_stall_val = require_param(frame, p, "alpha_stall_deg")
                m_smooth_val = require_param(frame, p, "m_smooth")
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
                self.alpha0_2d_link[b, i] = alpha0_val
                self.cl_alpha_2d_link[b, i] = cl_alpha_val
                self.alpha_stall_deg_link[b, i] = alpha_stall_val
                self.m_smooth_link[b, i] = m_smooth_val
                self.cp_start_link[b, i] = cp_start_val
                self.cp_end_link[b, i] = cp_end_val
                self.cg_to_chord_link[b, i] = cg_to_chord_val

                self.k_slip_wing_link[b, i] = k_slip_wing_val
                self.k_slip_tail_link[b, i] = k_slip_tail_val
                self.k_eps_tail_link[b, i] = k_eps_tail_val
                self.k_slip_fus_link[b, i] = k_slip_fus_val
                self.re_nom_link[b, i] = re_nom_val
                self.re_a_link[b, i] = re_a_val


        # Taichi fields: forces, application points, throttle
        self.force_b = ti.Vector.field(3, ti.f32, shape=(self._B, self.n_links_))
        self.cp_b = ti.Vector.field(3, ti.f32, shape=(self._B, self.n_links_))
        self._thr_raw = ti.field(ti.f32, shape=(self._B,))
        self._thr_flt = ti.field(ti.f32, shape=(self._B,))
        # Per-env sign for prop thrust direction (+1 or -1), set from Python after binding a target.
        self.prop_thrust_sign = ti.field(ti.f32, shape=(self._B,))
        for b in range(self._B):
            self.prop_thrust_sign[b] = 1.0
        self.B = self._B  # alias used inside kernels

        # Debug fields (angles and forces per surface)
        self.alpha_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        self.beta_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        self.lift_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        self.drag_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        self.side_force_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        # Per-surface Reynolds number (computed from local flow speed and chord).
        self.Reynolds = ti.field(ti.f32, shape=(self._B, self.L))
        # Tail/downwash debug (filled only for elevators when _aero_log=True)
        self.alpha_tail_raw_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        self.downwash_eps_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        self.cl_wing_for_tail_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        self.k_eps_tail_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))

        # Aerodynamic force center and global CoM per env
        self._aero_CF_world_b = ti.Vector.field(3, ti.f32, shape=(self._B,))
        self._CoM_world_b = ti.Vector.field(3, ti.f32, shape=(self._B,))

        # Per-surface constants (area, AR, chord, kind, side, link indices)
        self.area = ti.field(ti.f16, shape=(self.L,))
        self.AR = ti.field(ti.f16, shape=(self.L,))
        self.chord = ti.field(ti.f16, shape=(self.L,))
        self.kind = ti.field(ti.i16, shape=(self.L,))
        self.side = ti.field(ti.i8, shape=(self.L,))
        self._link_idx = ti.field(ti.i16, shape=(self.L,))
        self.slip_code = ti.field(ti.i8, shape=(self.L,))

        # Base span per surface (total span used in AR = b^2 / S)
        self.span = ti.field(ti.f16, shape=(self.L,))
        # Fuselage width used for simple wing AR interference correction.
        self.fus_width = ti.field(ti.f16, shape=())

        # Wing CL accumulator (left/right) and induced velocity at the prop
        self.cl_wing_b = ti.field(ti.f16, shape=(self._B, 2))
        self.v_ind = ti.field(ti.f16, shape=(self._B,))
        # Torch buffers for zero-copy fetch (filled by _copy_force_cp)
        self._force_buf = torch.empty((self._B, self.L, 3), device=self._aero_device, dtype=torch.float32)
        self._cp_buf = torch.empty_like(self._force_buf)

        # Fill per-surface constant fields from _geom
        fus_width = 0.0
        for i, (S, AR, c, k) in enumerate(self._geom):
            self.area[i] = S
            self.AR[i] = AR
            self.chord[i] = c
            self.kind[i] = k
            self._link_idx[i] = self._aero_link_idx[i]
            # Base span can be derived from S = chord * span.
            if float(c) > 1e-8:
                self.span[i] = float(S) / float(c)
            else:
                self.span[i] = 0.0
            if k == 0:
                fus_width = float(c)

            # Side flag is data: -1 left, +1 right, 0 center; param selectors use this single flag.
            name = self._aero_frames[i].lower()
            if "left" in name:
                self.side[i] = -1
            elif "right" in name:
                self.side[i] = 1
            else:
                self.side[i] = 0

            # Slipstream code (0=fuselage,1=tail,2=prop wash wing,3=other)
            if k == 0:
                self.slip_code[i] = 0
            elif k == 2:
                self.slip_code[i] = 1
            elif k == 1 and "prop" in name:
                self.slip_code[i] = 2
            else:
                self.slip_code[i] = 3

        self.fus_width[None] = fus_width

        # Register base parameters as per-env Taichi fields
        for name, val in self._aero_base.items():
            f = ti.field(dtype=ti.f32, shape=(self._B,))
            for b in range(self._B):
                f[b] = float(val)
            self._param[name] = f
            setattr(self, name, f)

        # Cached Torch buffers for parameters queried each step
        self._init_param_buffers()

        # Enable aero once everything is initialized
        self._aero_enabled = True

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

        return DroneAeroModel(urdf_file)

    def _apply_drone_model(self, model: DroneAeroModel, entity: RigidEntity):
        """
        Populate aerodynamic metadata and link indices from a DroneAeroModel.
        """
        self._drone_model = model
        self._aero_frames = list(model.frames)
        self._geom = list(model.geom)
        self._aero_base = dict(model.base_params)
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

        if model.base_param_overrides:
            self._aero_base.update(model.base_param_overrides)

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
        const_V_MAX = 50.0

        # Simple first-order low-pass on throttle (cutoff = prop_cutoff_hz)
        ti.cast(0, ti.f32)  # ensure float
        alpha_lpf = self._substep_dt / (
            self._substep_dt
            + 1.0 / (2.0 * ti.math.pi * self.prop_cutoff_hz[b])
        )
        self._thr_flt[b] += alpha_lpf * (
            float(self._thr_raw[b]) - float(self._thr_flt[b])
        )

        # Max thrust and prop inflow speed
        T_prop = float(self._thr_flt[b]) * self.max_thrust[b]
        prop_idx = self._link_idx[self.L - 1]  # last surface is the prop

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
                / (self.rho[b] * ti.math.pi * (self.prop_radius ** 2))
            )
        ) * 0.5
        v_ind_val = ti.math.clamp(v_ind_val, 0.0, const_V_MAX)
        self.v_ind[b] = ti.cast(v_ind_val, ti.f16)

    @ti.func
    def _main_surfaces_pass(self, rigid: ti.template(), b: int, l: int):
        # Speed clamp
        const_V_MAX = 50.0
        const_V_MIN = 0.1
        const_MU_AIR = 1.81e-5  # dynamic viscosity [kg/(m*s)] for Reynolds estimate

        # Reset accumulators (keep f32 precision)
        self.force_b[b, l] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.cp_b[b, l] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.Reynolds[b, l] = 0.0

        if self.kind[l] != 2:
            # Air velocity in link body frame
            v_body = self._get_wind_in_body(rigid, self._link_idx[l], b)

            # Slipstream effect
            slip = 0.0
            if self.slip_code[l] == 0:      # fuselage
                slip = float(self.k_slip_fus[b])
            elif self.slip_code[l] == 1:    # tail
                slip = float(self.k_slip_tail[b])
            elif self.slip_code[l] == 2:    # in prop wash
                slip = float(self._wing_param(self.k_slip_wing, self.k_slip_wing_left, self.k_slip_wing_right, b, self.side[l]))

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
            self.Reynolds[b, l] = (self.rho[b] * v_mod * c_ref) / const_MU_AIR

            # Aero coefficients (lift/drag)
            cl, cd = self._compute_coeff(
                b, l, AR_eff, alpha, beta, int(self.kind[l]), self.side[l]
            )

            # Trig helpers
            cosb2 = ti.cos(beta) ** 2
            cosa2 = ti.cos(alpha) ** 2

            # Dynamic pressure * area
            ti_05 = 0.5
            qS = ti_05 * self.rho[b] * S_eff * v_mod ** 2
            L = qS * cl
            D = qS * cd
            S = qS * cl  # here sideforce magnitude is tied to CL

            # Force in body frame
            Fb = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            cp = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)

            if self.kind[l] == 4:
                # Prop: thrust along +z in prop frame
                Fb = (
                    ti.Vector([0.0, 0.0, 1.0], dt=ti.f32)
                    * float(self.prop_thrust_sign[b])
                    * float(self._thr_flt[b])
                    * self.max_thrust[b]
                )
            elif self.kind[l] == 3:
                # Rudder: forces in YZ plane
                Fb = self._rot_yz(alpha, beta) @ ti.Vector([D * cosa2, S * cosa2, 0.0], dt=ti.f32)
                cp = ti.cast(self._cp_rudder(rigid, b, alpha, beta, l), ti.f32)
            else:
                # Fuselage or wing: forces in XZ plane
                Fb = self._rot_yz(alpha, beta) @ ti.Vector([D * cosb2, 0.0, L * cosb2], dt=ti.f32)
                if self.kind[l] == 1:  # wing
                    idx = 0 if self.side[l] == 1 else 1
                    self.cl_wing_b[b, idx] = ti.cast(cl, ti.f16)  # store CL for tail
                    cp = ti.cast(self._cp_wing(rigid, b, alpha, beta, l), ti.f32)
                else:  # fuselage
                    cp = ti.cast(self._cp_fus(rigid, b, alpha, beta, l), ti.f32)

            # ---- Inline noise (no extra pass) ---------------------------------
            do_noise = (self.noise_sigma_mag > 0.0) or (self.noise_sigma_dir > 0.0) or (self.noise_sigma_cp > 0.0)
            if do_noise:
                Fb, cp = self._apply_noise(
                    Fb, cp, l, alpha, beta, int(self.kind[l]),
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

            if ti.static(self._aero_log):
                self.alpha_dbg[b, l] = ti.cast(alpha, ti.f16)
                self.beta_dbg[b, l] = ti.cast(beta, ti.f16)
                self.drag_dbg[b, l] = ti.cast(D, ti.f16)
                if self.kind[l] != 3:
                    self.lift_dbg[b, l] = ti.cast(L, ti.f16)
                    self.side_force_dbg[b, l] = ti.cast(0.0, ti.f16)
                else:
                    self.lift_dbg[b, l] = ti.cast(0.0, ti.f16)
                    self.side_force_dbg[b, l] = ti.cast(S, ti.f16)
                if self.kind[l] == 4:
                    self.drag_dbg[b, l] = ti.cast(self._thr_flt[b], ti.f16)

    @ti.func
    def _tail_surfaces_pass(self, rigid: ti.template(), b: int, l: int):
        # Speed clamp
        const_V_MAX = 50.0
        const_V_MIN = 0.1
        const_MU_AIR = 1.81e-5  # dynamic viscosity [kg/(m*s)] for Reynolds estimate

        if self.kind[l] == 2:
            v_body_tail = self._get_wind_in_body(rigid, self._link_idx[l], b)

            slip = 0.0
            if self.slip_code[l] == 0:
                slip = float(self.k_slip_fus[b])
            elif self.slip_code[l] == 1:
                slip = float(self._elevator_param(self.k_slip_tail, self.k_slip_tail_elevator_left, self.k_slip_tail_elevator_right, b, self.side[l]))
            elif self.slip_code[l] == 2:
                slip = float(self._wing_param(self.k_slip_wing, self.k_slip_wing_left, self.k_slip_wing_right, b, self.side[l]))

            v_body_tail.x -= slip * ti.cast(self.v_ind[b], ti.f32)

            Vt = v_body_tail.norm() + 1e-6
            Vt = ti.math.clamp(Vt, const_V_MIN, const_V_MAX)

            # Tail angles
            alphat = ti.atan2(v_body_tail.z, v_body_tail.x)
            betat  = ti.asin(ti.math.clamp(v_body_tail.y / Vt, -1.0, 1.0))

            # Mean wing CL for downwash
            idx = 0 if self.side[l] == 1 else 1
            cl_w_mean = ti.cast(self.cl_wing_b[b, idx], ti.f32)

            # Effective tail AoA with downwash
            k_eps = self._elevator_param(self.k_eps_tail, self.k_eps_tail_elevator_left, self.k_eps_tail_elevator_right, b, self.side[l])
            eps = (k_eps * cl_w_mean) / (ti.math.pi * self.AR_wing)
            alpha_eff = alphat - eps

            # Reynolds number (per-tail surface)
            c_ref_t = ti.max(ti.cast(self.chord[l], ti.f32), 1e-6)
            self.Reynolds[b, l] = (self.rho[b] * Vt * c_ref_t) / const_MU_AIR
            # Tail coefficients (kind=2)
            S_eff_t = self._compute_eff_S(rigid, b, l)
            AR_eff_t = self._compute_eff_AR(rigid, b, l, betat)
            cl_t, cd_t = self._compute_coeff(b, l, AR_eff_t, alpha_eff, betat, 2, self.side[l])

            cosb2 = ti.cos(betat) ** 2

            # Tail forces
            qS_t = 0.5 * self.rho[b] * S_eff_t * Vt ** 2 * cosb2
            L_t = qS_t * cl_t
            D_t = qS_t * cd_t
            Fb_t = self._rot_yz(alpha_eff, betat) @ ti.Vector([D_t, 0.0, L_t], dt=ti.f32)
            # Tail center of pressure
            cp_t = ti.cast(self._cp_elev(rigid, b, alpha_eff, betat, l), ti.f32)
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

            if ti.static(self._aero_log):
                self.alpha_tail_raw_dbg[b, l] = ti.cast(alphat, ti.f16)
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
        cut = self.alpha_stall_deg_link[b, l] * ti.math.pi/180
        cl_alpha = self.cl_alpha_2d_link[b, l]
        alpha0 = self.alpha0_2d_link[b, l]
        smooth = self.m_smooth_link[b, l]
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
        elif kind == 2:  # elevator
            cl_alpha = self._elevator_param(self.cl_alpha_2d, self.cl_alpha_2d_elevator_left, self.cl_alpha_2d_elevator_right, b, side)
            alpha0 = self._elevator_param(self.alpha0_2d, self.alpha0_2d_elevator_left, self.alpha0_2d_elevator_right, b, side)
            cd0 = self._elevator_param(self.cd0, self.cd0_elevator_left, self.cd0_elevator_right, b, side)
            cut = self._elevator_param(self.alpha_stall_deg, self.alpha_stall_deg_elevator_left, self.alpha_stall_deg_elevator_right, b, side) * ti.math.pi / 180.0
            smooth = self._elevator_param(self.m_smooth, self.m_smooth_elevator_left, self.m_smooth_elevator_right, b, side)

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
        cd_lin = cd0 + cl_lin * cl_lin / (ti.math.pi * AR)

        # Post-stall flat-plate model (periodic)
        sa, ca = ti.sin(a_wr), ti.cos(a_wr)
        cl_st = 2.0 * sa * ca
        cd_st = 2.0 * sa * sa
        cl_st *= f_re
        cd_st *= f_re

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
