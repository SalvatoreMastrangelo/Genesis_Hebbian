import math
import numpy as np
import torch
import gstaichi as ti

from .base_solver import Solver
from genesis.utils import geom as gu
from genesis.engine.entities import RigidEntity  # for get_link()
from genesis.assets.urdf.mydrone.drone import DroneAeroModel, SurfaceKind
from genesis.engine.solvers.drones.simple_drone import SimpleDroneAeroParameters


@ti.data_oriented
class AeroSolver(Solver):
    """
    Aerodynamic solver that couples a simple aero model with Genesis' RigidSolver.

    Notes
    -----
    * The solver is intentionally "stateless" from the simulator point of view:
      all physical state lives inside the underlying RigidSolver.
    * Aerodynamic parameters are stored as Taichi fields per environment.
    * Geometry is inferred from the URDF and mapped to a fixed set of aero frames.
    """

    def __init__(self, scene, sim, options=None):
        super().__init__(scene, sim, options)

        # Reference to underlying rigid solver and timing
        self._rigid_solver = sim.rigid_solver
        self._substep_dt = sim._substep_dt

        # Target rigid entities that carry aerodynamics
        self._aero_targets: list[RigidEntity] = []
        self._aero_enabled: bool = False

        # Get Aerodynamic parameters for each link from a CSV
        self._params_from_csv: bool = False
        csv_file: str | None = None

        # Debug / logging flags
        self._aero_log: bool = False  # heavy prints are gated elsewhere

        # Torch device used when exporting Taichi buffers
        self._aero_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Aerodynamic geometry list: tuples of (area S, aspect ratio AR, chord, kind)
        # kind: 0=fuselage, 1=wing, 2=elevator, 3=rudder, 4=prop
        self._geom: list[tuple[float, float, float, int]] = []
        self.L: int = 0  # number of aerodynamic surfaces

        # Base aero constants and frames (overridden by DroneAeroModel/config)
        self._const_init()

        # Override base params from CSV if requested
        if self._params_from_csv and csv_file is not None:
            self._get_params_from_csv(csv_file)

        # Per-parameter Taichi fields (allocated in add_target)
        self._param: dict[str, ti.Field] = {}

    # ---------------------------------------------------------------------
    # Initialization utilities
    # ---------------------------------------------------------------------
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

    def _get_params_from_csv(self, csv_file: str):
        """
        Load aerodynamic parameters from a CSV file.

        The CSV must contain columns matching the keys in `self._aero_base`.
        Loaded values override the nominal ones in `_aero_base`.

        Parameters
        ----------
        csv_file:
            Path to the CSV file.
        """
        import pandas as pd

        df = pd.read_csv(csv_file)
        for key in self._aero_base.keys():
            if key in df.columns:
                val = float(df[key].iloc[0])
                self._aero_base[key] = val

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
                raise ValueError(f"Surface '{frame}' missing required key '{key}' in aero configuration.")
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

        # ---- Seff metadata (per surface) ----
        self.seff_yaw_enabled     = ti.field(ti.i8,  shape=(self.L,))
        self.seff_pitch_enabled   = ti.field(ti.i8,  shape=(self.L,))
        self.seff_yaw_ratio       = ti.field(ti.f16, shape=(self.L,))
        self.seff_pitch_ratio     = ti.field(ti.f16, shape=(self.L,))
        self.seff_yaw_theta_max   = ti.field(ti.f16, shape=(self.L,))
        self.seff_pitch_theta_max = ti.field(ti.f16, shape=(self.L,))
        self.seff_yaw_dof_idx     = ti.field(ti.i32, shape=(self.L,))
        self.seff_pitch_dof_idx   = ti.field(ti.i32, shape=(self.L,))

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

            # Side flag (left/right) inferred from name; fuselage/prop default to 0
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

        # ---- Seff metadata (optional, aligned with model.frames) ----
        for i in range(self.L):
            self.seff_yaw_enabled[i] = 0
            self.seff_pitch_enabled[i] = 0
            self.seff_yaw_ratio[i] = 1.0
            self.seff_pitch_ratio[i] = 1.0
            self.seff_yaw_theta_max[i] = 0.0
            self.seff_pitch_theta_max[i] = 0.0
            self.seff_yaw_dof_idx[i] = -1
            self.seff_pitch_dof_idx[i] = -1

        if model is not None and hasattr(model, "seff_yaw_enabled"):
            # Ensure DOF indices are resolved (requires scene.build()).
            if hasattr(model, "resolve_seff_dof_indices"):
                try:
                    model.resolve_seff_dof_indices(entity)
                except Exception:
                    pass

            yaw_local = getattr(model, "seff_yaw_dof_idx_local", [-1 for _ in range(self.L)])
            pitch_local = getattr(model, "seff_pitch_dof_idx_local", [-1 for _ in range(self.L)])
            dof_start = int(getattr(entity, "dof_start", 0))

            for i in range(self.L):
                ey = bool(model.seff_yaw_enabled[i]) if i < len(model.seff_yaw_enabled) else False
                ep = bool(model.seff_pitch_enabled[i]) if i < len(model.seff_pitch_enabled) else False
                ry = float(model.seff_yaw_ratio[i]) if i < len(model.seff_yaw_ratio) else 1.0
                rp = float(model.seff_pitch_ratio[i]) if i < len(model.seff_pitch_ratio) else 1.0
                ty = float(model.seff_yaw_theta_max[i]) if i < len(model.seff_yaw_theta_max) else 0.0
                tp = float(model.seff_pitch_theta_max[i]) if i < len(model.seff_pitch_theta_max) else 0.0

                yi = int(yaw_local[i]) if i < len(yaw_local) else -1
                pi = int(pitch_local[i]) if i < len(pitch_local) else -1

                if ey and yi >= 0:
                    self.seff_yaw_enabled[i] = 1
                    self.seff_yaw_dof_idx[i] = dof_start + yi
                else:
                    self.seff_yaw_enabled[i] = 0
                    self.seff_yaw_dof_idx[i] = -1

                if ep and pi >= 0:
                    self.seff_pitch_enabled[i] = 1
                    self.seff_pitch_dof_idx[i] = dof_start + pi
                else:
                    self.seff_pitch_enabled[i] = 0
                    self.seff_pitch_dof_idx[i] = -1

                self.seff_yaw_ratio[i] = ry
                self.seff_pitch_ratio[i] = rp
                self.seff_yaw_theta_max[i] = ty
                self.seff_pitch_theta_max[i] = tp

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

        return DroneAeroModel(urdf_file, config_override=SimpleDroneAeroParameters.as_dict())

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

    def _init_param_buffers(self):
        """Allocate Torch buffers and prime them from Taichi fields."""
        self._kappa_buf = torch.empty((self._B,), device=self._aero_device, dtype=torch.float32)
        self._fcap_buf = torch.empty((self._B, 1, 1), device=self._aero_device, dtype=torch.float32)
        self._refresh_param_buffers()

    def _refresh_param_buffers(self):
        """Refresh cached Torch views of Taichi parameter fields."""
        if not hasattr(self, "_kappa_buf"):
            return
        self._kappa_buf.copy_(self.kappa_prop.to_torch(device=self._aero_device))
        cap = self.force_cap.to_torch(device=self._aero_device)
        if self._fcap_buf.shape[0] != cap.shape[0]:
            self._fcap_buf = torch.empty((cap.shape[0], 1, 1), device=self._aero_device, dtype=torch.float32)
        self._fcap_buf.copy_(cap.view(-1, 1, 1))

    def set_throttle(self, thr: torch.Tensor | float):
        """
        Set throttle command for propellers (scalar or torch tensor).

        Parameters
        ----------
        thr:
            Either a scalar in [0, 1] or a tensor broadcastable to the batch
            dimension B. Values are NOT clamped here.
        """
        if torch.is_tensor(thr):
            # Avoid per-step cpu().numpy(); write directly from torch
            t = thr.reshape(-1).to(self._aero_device, dtype=torch.float32)
            if not hasattr(self, "_thr_buf") or self._thr_buf.shape[0] != self.B:
                self._thr_buf = torch.empty((self.B,), device=self._aero_device, dtype=torch.float32)
            if t.numel() == 1:
                self._thr_buf.fill_(t.item())
            else:
                self._thr_buf.copy_(t)
            self._thr_raw.from_torch(self._thr_buf)
        else:
            if not hasattr(self, "_thr_buf") or self._thr_buf.shape[0] != self.B:
                self._thr_buf = torch.empty((self.B,), device=self._aero_device, dtype=torch.float32)
            self._thr_buf.fill_(float(thr))
            self._thr_raw.from_torch(self._thr_buf)

    def substep_pre_coupling(self, f: int):
        """
        Genesis hook called once per physics substep, before coupling.

        The aerodynamic solver computes forces/torques on the rigid bodies
        and applies them to the underlying RigidSolver.
        """
        self._aero_step()

    # ---------------------------------------------------------------------
    # Main aerodynamic step (Python side)
    # ---------------------------------------------------------------------
    def _aero_step(self):
        """
        Compute aerodynamic forces/torques and apply them to the RigidSolver.

        This function:
        1. Launches the Taichi kernel `_aero_compute_kernel` to populate
           per-link forces (`force_b`) and application points (`cp_b`).
        2. Converts those fields to torch tensors.
        3. Computes the propeller reaction torque.
        4. Clamps forces/torques for numerical stability.
        5. Applies them to the RigidSolver in link-local coordinates.
        """
        if not self._aero_targets:
            return

        # 1) compute forces/torques in Taichi
        self._aero_compute_kernel(self._rigid_solver)

        # 2) fetch Fb and cp into persistent torch buffers (link frame)
        self._copy_force_cp(self._force_buf, self._cp_buf)
        fb = self._force_buf[:, : len(self._aero_link_idx), :]  # (B, L, 3)
        cp = self._cp_buf[:, : len(self._aero_link_idx), :]     # (B, L, 3)

        # 3) prop reaction torque (about prop z-axis, link frame)
        tq = torch.zeros_like(fb)
        thrust = fb[:, -1, 2]  # z-component of last surface (prop) in its frame
        tq[:, -1, 2] = -self._kappa_buf * thrust
        # 4) clamp and clean numerical issues
        cap = self._fcap_buf

        fb = torch.nan_to_num(fb).clamp(min=-cap, max=cap)
        cp = torch.nan_to_num(cp)
        tq = torch.nan_to_num(tq).clamp(min=-cap, max=cap)

        # 5) apply forces and torques to Genesis' rigid solver
        self._rigid_solver.apply_links_force_at_point_link_frame(
            pos=cp,
            force=fb,
            links_idx=self._aero_link_idx,
        )
        self._rigid_solver.apply_links_coupling_torque(
            torque=tq,
            links_idx=self._aero_link_idx,
            ref="link_origin",
            local=True,
        )

    @ti.kernel
    def _copy_force_cp(
        self,
        out_force: ti.types.ndarray(dtype=ti.f32, ndim=3),
        out_cp: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for b, l in ti.ndrange(self.B, self.L):
            out_force[b, l, 0] = self.force_b[b, l][0]
            out_force[b, l, 1] = self.force_b[b, l][1]
            out_force[b, l, 2] = self.force_b[b, l][2]
            out_cp[b, l, 0] = self.cp_b[b, l][0]
            out_cp[b, l, 1] = self.cp_b[b, l][1]
            out_cp[b, l, 2] = self.cp_b[b, l][2]
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
        # Speed clamp
        const_V_MAX = 50.0
        const_V_MIN = 0.1
        const_MU_AIR = 1.81e-5  # dynamic viscosity [kg/(m*s)] for Reynolds estimate

        # 0) Update filtered throttle and induced velocity at the prop
        for b in range(self.B):
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

        # 1) First pass: all surfaces except tail (elevators)
        for b, l in ti.ndrange(self.B, self.L):
            # Reset accumulators (keep f32 precision)
            self.force_b[b, l] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            self.cp_b[b, l] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            self.Reynolds[b, l] = 0.0

            if self.kind[l] == 2:  # skip elevators here
                continue

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
            AR_eff = self._compute_eff_AR(rigid, b, l)

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
                cp = ti.cast(self._cp_rudder(b, alpha, beta, l), ti.f32)
            else:
                # Fuselage or wing: forces in XZ plane
                Fb = self._rot_yz(alpha, beta) @ ti.Vector([D * cosb2, 0.0, L * cosb2], dt=ti.f32)
                if self.kind[l] == 1:  # wing
                    idx = 0 if self.side[l] == 1 else 1
                    self.cl_wing_b[b, idx] = ti.cast(cl, ti.f16)  # store CL for tail
                    cp = ti.cast(self._cp_wing(b, alpha, beta, l), ti.f32)
                else:  # fuselage
                    cp = ti.cast(self._cp_fus(b, alpha, beta, l), ti.f32)

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

        # 2) Second pass: elevators (tail), using wing downwash
        for b, l in ti.ndrange(self.B, self.L):
            if self.kind[l] != 2:
                continue

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

            # Tail coefficients (kind=2)
            S_eff_t = self._compute_eff_S(rigid, b, l)
            AR_eff_t = self._compute_eff_AR(rigid, b, l)
            cl_t, cd_t = self._compute_coeff(b, l, AR_eff_t, alpha_eff, betat, 2, self.side[l])

            cosb2 = ti.cos(betat) ** 2

            # Reynolds number (per-tail surface)
            c_ref_t = ti.max(ti.cast(self.chord[l], ti.f32), 1e-6)
            self.Reynolds[b, l] = (self.rho[b] * Vt * c_ref_t) / const_MU_AIR

            # Tail forces
            qS_t = 0.5 * self.rho[b] * S_eff_t * Vt ** 2 * cosb2
            L_t = qS_t * cl_t
            D_t = qS_t * cd_t
            Fb_t = self._rot_yz(alpha_eff, betat) @ ti.Vector([D_t, 0.0, L_t], dt=ti.f32)
            # Tail center of pressure
            cp_t = ti.cast(self._cp_elev(b, alpha_eff, betat, l), ti.f32)
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
        for b, l in ti.ndrange(self.B, self.L):
            print(f"Reynolds link[{l}] batch[{b}]: ", self.Reynolds[b, l])
        # Noise applied inline: no third pass needed.

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
        r_y = 1.0
        r_p = 1.0

        if self.seff_yaw_enabled[l] != 0 and self.seff_yaw_dof_idx[l] >= 0:
            theta = ti.cast(rigid.dofs_state.pos[self.seff_yaw_dof_idx[l], b], ti.f32)
            r_y = self._seff_ratio(
                theta,
                ti.cast(self.seff_yaw_theta_max[l], ti.f32),
                ti.cast(self.seff_yaw_ratio[l], ti.f32),
            )

        if self.seff_pitch_enabled[l] != 0 and self.seff_pitch_dof_idx[l] >= 0:
            theta = ti.cast(rigid.dofs_state.pos[self.seff_pitch_dof_idx[l], b], ti.f32)
            r_p = self._seff_ratio(
                theta,
                ti.cast(self.seff_pitch_theta_max[l], ti.f32),
                ti.cast(self.seff_pitch_ratio[l], ti.f32),
            )

        return ti.max(1e-6, S0 * r_y * r_p)
    
    @ti.func
    def _compute_eff_span(self, rigid: ti.template(), b: int, l: int):
        """
        Compute effective span for a surface.
        """
        b0 = ti.cast(self.span[l], ti.f32)
        b_eff = b0

        # Only apply the cos(sweep) projection to wings (sweep joints rotate about Z).
        if int(self.kind[l]) == 1 and self.seff_yaw_enabled[l] != 0 and self.seff_yaw_dof_idx[l] >= 0:
            theta = ti.cast(rigid.dofs_state.pos[self.seff_yaw_dof_idx[l], b], ti.f32)
            b_eff = b0 * ti.cos(ti.abs(theta))

        return ti.max(0.0, b_eff)
    
    @ti.func
    def _compute_eff_AR(self, rigid: ti.template(), b: int, l: int):
        """
        Compute effective aspect ratio for a surface.

        This includes fuselage interference effects.
        """
        k = ti.cast(self.kind[l], ti.i32)
        AR_eff = ti.cast(self.AR[l], ti.f32)

        if (k == 1) or (k == 2):
            b_eff = self._compute_eff_span(rigid, b, l)
            S_eff = self._compute_eff_S(rigid, b, l)

            total_b_eff = 2.0 * b_eff
            if k == 1:
                total_b_eff += ti.cast(self.fus_width[None], ti.f32)

            AR_eff = (total_b_eff * b_eff) / ti.max(S_eff, 1e-6)

        return AR_eff

    @ti.func
    def _get_wind_in_body(self, rigid: ti.template(), link_idx: int, b: int):
        """
        Compute the local air velocity (wind) in the link body frame.

        The air is assumed still in the world frame, so the wind is simply
        the negative of the link COM velocity, rotated into the body frame.
        """
        # 1) link COM velocity in world frame
        v_link_world = rigid.links_state.cd_vel[link_idx, b]

        # 2) air velocity = negative body velocity
        v_world = -v_link_world

        # 3) rotate world velocity into body frame using quaternion
        quat = rigid.links_state.quat[link_idx, b]
        return gu.ti_inv_transform_by_quat(v_world, quat)

    @ti.func
    def _rot_yz(self, alpha, beta):
        """
        Rotation matrix in body frame: Ry(alpha) then Rz(beta).

        It is used to transform forces expressed in aero-aligned axes
        back to the body frame.
        """
        ca, sa = ti.cos(alpha), ti.sin(alpha)
        cb, sb = ti.cos(beta), ti.sin(beta)
        # Compose Rz(beta) @ Ry(alpha) without forming two matrices
        return ti.Matrix(
            [
                [cb * ca, -sb, cb * sa],
                [sb * ca, cb, sb * sa],
                [-sa, 0.0, ca],
            ]
        )

    @ti.func
    def _wrap_pm_90(self, angle):
        """
        Wrap angle to [-pi/2, pi/2] using periodic symmetry.

        This exploits the symmetry of lift/drag curves to keep angles
        inside a compact range and improve numerical robustness.
        """
        two_pi = 2.0 * ti.math.pi

        # Wrap to (-pi, pi]
        a_wr = angle - two_pi * ti.floor((angle + ti.math.pi) / two_pi)
        a_abs = ti.abs(a_wr)

        # Fold to [0, pi/2]
        a_fold = ti.min(a_abs, ti.math.pi - a_abs)

        # Restore sign
        sign = 1.0
        if a_wr < 0.0:
            sign = -1.0
        return sign * a_fold

    # ------------------------------------------------------------------
    # Parameter randomization (Python side)
    # ------------------------------------------------------------------
    def randomize_aero_params(self, envs_idx, sigma=None):
        """
        Randomize aerodynamic parameters for a subset of environments.

        Parameters
        ----------
        envs_idx :
            1-D torch.Tensor (dtype=int64) with the indices of the envs.
        sigma :
            Standard deviation of the multiplicative noise. If None, uses
            `self.noise_sigma_mag` as default.
        """
        if sigma is None:
            sigma = self.noise_sigma_param
        if len(envs_idx) == 0:
            return
            

        envs_np = envs_idx.cpu().numpy()

        # Loop over aerodynamic constants
        for k, base_val in self._aero_base.items():
            fld = self._param[k]  # ScalarField (shape = (B,))
            arr = fld.to_numpy()  # host copy
            arr[envs_np] = base_val  # reset to nominal value

            if sigma > 0:
                arr[envs_np] *= 1.0 + sigma * np.random.randn(len(envs_np))

            fld.from_numpy(arr)
        # Refresh cached Torch buffers used in the Python step
        self._refresh_param_buffers()

    # ------------------------------------------------------------------
    # Force noise model
    # ------------------------------------------------------------------
    @ti.func
    def _sigma_scale(self, alpha, beta):
        """
        Angular-dependent scale factor for noise intensity.

        Returns a scalar in roughly [1, 3], increasing with the magnitude
        of the combined angle (alpha, beta). This concentrates higher
        noise near large angles of attack/sideslip.
        """
        a2 = alpha * alpha + beta * beta
        a_eff = ti.min(ti.sqrt(a2), ti.math.pi)
        t = a_eff / (ti.math.pi)
        return 1.0 + 3.0 * t * t

    @ti.func
    def _apply_noise(
        self,
        F_in,                 # ti.Vector(3, f32)
        cp_in,                # ti.Vector(3, f32)
        l: ti.i32,
        alpha: ti.f32,
        beta: ti.f32,
        kind: ti.i32,
        sig_mag_base: ti.f32,
        sig_dir_base: ti.f32,
        sig_cp_base: ti.f32,
    ):
        """
        Apply stochastic noise directly to an already computed aerodynamic force F_in.
        (Taichi-safe: single return at the end, no early returns inside non-static branches)
        """
        # Cast to f32 for arithmetic; start with passthrough
        F = ti.cast(F_in, ti.f32)
        cp = ti.cast(cp_in, ti.f32)
        out_F = F
        out_cp = cp

        # Precompute magnitude
        mag = F.norm()

        # Runtime guard (no early return!)
        do_noise = ((sig_mag_base > 0.0) or (sig_dir_base > 0.0)) and (kind != 4) and (mag > 0.0)

        if do_noise:
            # Rudder: treat sideslip as effective alpha
            a = alpha
            b = beta
            if kind == 3:
                a = beta
                b = alpha

            # Angle-dependent scaling
            s = self._sigma_scale(a, b)
            sig_mag = sig_mag_base * s
            sig_dir = sig_dir_base * s

            # --- Magnitude noise ---
            mag_n = mag * (1.0 + sig_mag * ti.randn(ti.f32))

            # --- Direction noise ---
            g = ti.Vector([ti.randn(ti.f32), ti.randn(ti.f32), ti.randn(ti.f32)])
            inv_mag2 = 1.0 / ti.max(mag * mag, 1e-3)
            g -= (g.dot(F) * inv_mag2) * F  # orthogonalize to F
            g2 = g.dot(g) + 1e-6
            g *= ti.rsqrt(g2)               # fast normalize

            dir_vec = F / mag + sig_dir * g
            d2 = dir_vec.dot(dir_vec) + 1e-6
            dir_vec *= ti.rsqrt(d2)         # fast normalize

            out = dir_vec * mag_n

            out_F = out

            # --- CP noise (small random shift, scales with chord and angle) ---
            chord = ti.cast(self.chord[l], ti.f32)
            sig_cp = 0.25 * chord * (sig_cp_base) * s
            dx = sig_cp * ti.randn(ti.f32)
            dy = sig_cp * ti.randn(ti.f32)
            dz = 0.2 * sig_cp * ti.randn(ti.f32)
            lim = 0.15 * chord
            dx = ti.math.clamp(dx, -lim, lim)
            dy = ti.math.clamp(dy, -lim, lim)
            dz = ti.math.clamp(dz, -lim, lim)
            out_cp = cp + ti.Vector([dx, dy, dz], dt=ti.f32)

        return out_F, out_cp

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

        if kind == 1:  # wing → allow left/right split
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

        ref = alpha0

        cl_lin = cl_a * (a_wr - ref)
        cd_lin = cd0 + cl_lin * cl_lin / (ti.math.pi * AR)

        # Post-stall flat-plate model (periodic)
        sa, ca = ti.sin(a_wr), ti.cos(a_wr)
        cl_st = 2.0 * sa * ca
        cd_st = 2.0 * sa * sa

        # Smooth blend between linear and post-stall using m_smooth as width
        width = smooth * cut
        width = ti.max(width, 1e-3)  # avoid extremely sharp transitions / div by zero
        s_arg = (a_fold - cut) / width
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
    def _cp_wing(self, b, alpha, beta, l):
        """
        Wing center of pressure (x-offset in local frame).

        Uses a simple shift between `cp_start` and `cp_end` as a function
        of effective angle of attack.
        """
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
    def _cp_fus(self, b, alpha, beta, l):
        """
        Fuselage center of pressure (x-offset in fuselage frame).
        """
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
    def _cp_rudder(self, b, alpha, beta, l):
        """
        Rudder center of pressure: [x, 0, z] with fixed z offset.
        """
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
    def _cp_elev(self, b, alpha, beta, l):
        """
        Elevator center of pressure [x, y, z] with a small y offset.

        The y offset distinguishes left/right halves of the tailplane so
        that differential elevator deflections can generate rolling moments.
        """
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
        val = default_field[b]
        if side < 0:
            val = left_field[b]
        elif side > 0:
            val = right_field[b]
        return val

    # ------------------------------------------------------------------
    # Solver interface required by Genesis
    # ------------------------------------------------------------------
    @property
    def n_entities(self) -> int:
        """
        AeroSolver doesn't own entities; it just acts on the RigidSolver.

        Keeping this at 0 prevents Simulator.reset() from calling set_state()
        on AeroSolver, which would be meaningless.
        """
        return 0

    def build(self):
        """
        Called once when the simulator is built.

        We mainly need to:
        * Let the base Solver allocate gravity fields if needed.
        * Cache the batch size B from the simulator.
        """
        super().build()
        self.B = self._B

        # Optionally auto-enable aero when targets exist
        if len(self._aero_targets) > 0:
            self._aero_enabled = True

    def get_state(self, f: int):
        """
        AeroSolver has no independent state to store.

        Returning None is fine; SimState will just carry a None entry
        for this solver.
        """
        return None

    def set_state(self, f: int, state, envs_idx=None):
        """
        No state to restore. All relevant state is in the RigidSolver.

        This method is called only if solver.n_entities > 0, which for
        AeroSolver is always 0.
        """
        return

    def process_input(self, in_backward: bool = False):
        """
        Called once per *step* before substeps.

        RigidSolver uses this to process high-level control targets.
        AeroSolver doesn't need anything here because throttle / control
        inputs are set explicitly through `set_throttle` from user code.
        """
        return

    def process_input_grad(self):
        """
        Gradient counterpart of process_input. Currently unused.
        """
        return

    def substep_pre_coupling_grad(self, f: int):
        """
        Gradient counterpart of substep_pre_coupling. Not implemented.
        """
        return

    def substep_post_coupling(self, f: int):
        """
        Called after coupling step. AeroSolver doesn't need a post stage.
        """
        return

    def substep_post_coupling_grad(self, f: int):
        """
        Gradient counterpart of substep_post_coupling. Not implemented.
        """
        return

    def add_grad_from_state(self, state):
        """
        Used in reverse mode to accumulate gradients from SimState.

        AeroSolver has no state, so nothing to do.
        """
        return

    def collect_output_grads(self):
        """
        Called at simulator level to let each solver pull grads out of
        entities' queried states. AeroSolver doesn't store any, so noop.
        """
        return

    def reset_grad(self):
        """
        Reset any internal gradient buffers. Nothing to reset here.
        """
        return

    def save_ckpt(self, ckpt_name: str):
        """
        Save local solver-specific checkpoint for gradient checkpointing.

        AeroSolver doesn't maintain local history; noop.
        """
        return

    def load_ckpt(self, ckpt_name: str):
        """
        Load solver-specific checkpoint. Not needed for AeroSolver.
        """
        return

    def is_active(self):
        """
        Keep the same convention as other solvers: a "truthy" method
        used like a property in the Simulator.

        We consider AeroSolver active only if:
        * aerodynamic targets have been registered, and
        * a RigidSolver exists and is active (has links).
        """
        return (
            self._aero_enabled
            and self._rigid_solver is not None
            and self._rigid_solver.is_active
        )
