import math
import numpy as np
import torch
import gstaichi as ti

from .base_solver import Solver
from genesis.utils import geom as gu
from genesis.engine.entities import RigidEntity  # for get_link()


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

        # Aerodynamic geometry list: tuples of (area S, aspect ratio AR, chord, kind)
        # kind: 0=fuselage, 1=wing, 2=elevator, 3=rudder, 4=prop
        self._geom: list[tuple[float, float, float, int]] = []
        self.L: int = 0  # number of aerodynamic surfaces

        # Base aero constants and frames
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
        self._aero_base = dict(
            rho=1.225,
            cl_alpha_2d=2 * math.pi,
            alpha_stall_deg=12.0,
            alpha0_2d=-3.0 * math.pi / 180.0,
            alpha0_2d_fus=0.0,
            cd0=0.05,
            cd0_fus=0.75,
            m_smooth=0.2,
            prop_cutoff_hz=1.0,
            k_eps_tail=1.0,
            k_slip_fus=0.0,
            k_slip_tail=1.0,
            k_slip_wing=1.0,
            kappa_prop=0.01,
            max_thrust=5.0,
            force_cap=40.0,
            cp_start=0.25,
            cp_end=0.50,
            cg_to_chord=0.31,
        )

        # Link names used as aerodynamic frames (must exist in the URDF)
        # The last one is assumed to be the propeller frame.
        self._aero_frames = [
            "aero_frame_fuselage",
            "aero_frame_left_wing_prop",
            "aero_frame_left_wing_free",
            "aero_frame_right_wing_prop",
            "aero_frame_right_wing_free",
            "aero_frame_elevator_left",
            "aero_frame_elevator_right",
            "aero_frame_rudder",
            "prop_frame_fuselage_0",
        ]

        # Stochastic force noise parameters used in `noise_addition`.
        # These are kept as plain Python floats on purpose, so they act as
        # compile-time constants for Taichi kernels and as defaults for
        # `randomize_aero_params`.
        self.noise_sigma_mag: float = 0.1  # relative std-dev on |F|
        self.noise_sigma_dir: float = 0.1  # std-dev for directional noise
        self.noise_sigma_param: float = 0.1  # relative std-dev on aero params

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
    def add_target(self, entity: RigidEntity, urdf_file: str | None = None):
        """
        Register a RigidEntity as aerodynamic target and allocate Taichi fields.

        Parameters
        ----------
        entity:
            RigidEntity instance whose links contain the aerodynamic frames.
        urdf_file:
            Optional URDF file path used to infer aerodynamic geometry. If
            provided, `_parse_urdf` fills `self._geom`. If it is omitted,
            `_geom` must already be populated before calling `add_target`.
        """
        self._aero_targets.append(entity)

        # Global link indices for aerodynamic frames
        self._aero_link_idx = [entity.get_link(name).idx for name in self._aero_frames]

        # Extract geometry from URDF if provided
        if urdf_file is not None:
            self._parse_urdf(urdf_file)

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

        # Taichi fields: forces, application points, throttle
        self.force_b = ti.Vector.field(3, ti.f32, shape=(self._B, self.n_links_))
        self.cp_b = ti.Vector.field(3, ti.f32, shape=(self._B, self.n_links_))
        self._thr_raw = ti.field(ti.f32, shape=(self._B,))
        self._thr_flt = ti.field(ti.f32, shape=(self._B,))
        self.B = self._B  # alias used inside kernels

        # Debug fields (angles and forces per surface)
        self.alpha_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        self.beta_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        self.lift_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        self.drag_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))
        self.side_force_dbg = ti.field(ti.f16, shape=(self._B, self.n_links_))

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

        # Wing CL accumulator (left/right) and induced velocity at the prop
        self.cl_wing_b = ti.field(ti.f16, shape=(self._B, 2))
        self.v_ind = ti.field(ti.f16, shape=(self._B,))
        # Torch buffers for zero-copy fetch (filled by _copy_force_cp)
        self._force_buf = torch.empty((self._B, self.L, 3), device=self._aero_device, dtype=torch.float32)
        self._cp_buf = torch.empty_like(self._force_buf)

        # Fill per-surface constant fields from _geom
        for i, (S, AR, c, k) in enumerate(self._geom):
            self.area[i] = S
            self.AR[i] = AR
            self.chord[i] = c
            self.kind[i] = k
            self._link_idx[i] = self._aero_link_idx[i]

            # Side flag (left/right)
            name = self._aero_frames[i]
            if name.endswith("_elevator_left"):
                self.side[i] = -1
            elif name.endswith("_elevator_right"):
                self.side[i] = 1
            elif name.endswith("_left_wing_free"):
                self.side[i] = -1
            elif name.endswith("_right_wing_free"):
                self.side[i] = 1
            elif name.endswith("_left_wing_prop"):
                self.side[i] = -1
            elif name.endswith("_right_wing_prop"):
                self.side[i] = 1
            else:
                # Fuselage/propeller: no side
                self.side[i] = 0

            # Slipstream code
            if k == 0:  # fuselage
                self.slip_code[i] = 0
            elif k == 2:  # tail planes
                self.slip_code[i] = 1
            elif k == 1 and name.endswith("_wing_prop"):
                self.slip_code[i] = 2  # wing section inside prop wash
            else:
                self.slip_code[i] = 3  # outer wings, rudder, etc.

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

    def _parse_urdf(self, urdf_file: str):
        """
        Parse URDF to extract aerodynamic surface dimensions and metadata.

        Populates:
        * `self._geom` with a list of surfaces (S, AR, chord, kind)
        * several helper attributes used by CP and downwash models.
        """
        import xml.etree.ElementTree as ET
        from pathlib import Path

        self._urdf_path = Path(urdf_file)
        root = ET.parse(self._urdf_path).getroot()

        # Optional "genes" metadata used to override some base parameters
        meta = root.find(".//metadata[@type='genes']")
        if meta is not None:
            order = meta.get("order", "").split()
            vals = list(map(float, meta.get("value", "").split()))
            self.genes = vals
            self.genes_dict = dict(zip(order, vals))
        else:
            self.genes = []
            self.genes_dict = {}

        # Override some base params from genes if present
        for key in ("cl_alpha_2d", "alpha0_2d"):
            if key in self.genes_dict:
                self._aero_base[key] = self.genes_dict[key]

        # Helpers to read geometry from the URDF
        def box(name: str) -> list[float]:
            size = root.find(f".//link[@name='{name}']/collision/geometry/box").get("size")
            return list(map(float, size.split()))

        def xyz(name: str) -> list[float]:
            origin = root.find(f".//link[@name='{name}']/inertial/origin").get("xyz")
            return list(map(float, origin.split()))

        # Fuselage frontal section (sx, sz)
        sx, _, sz = box("fuselage")
        S_fus = sx * sz * 4 * 4
        sp_fus = sx * 4
        c_fus = sz * 4

        # Equivalent circular frontal area
        self.S_perp_fus = sx * sx * 4 * 4 * math.pi / 4

        # Fuselage CG x-position in its local frame
        cg_x = xyz("fuselage")[0]
        self.cg_fus_local_x = cg_x

        # Wings (inner "prop" and outer "free") – boxes are (thickness, chord, span)
        _, c_wp, sp_wp = box("right_wing_prop")
        _, c_wf, sp_wf = box("right_wing_free")

        S_lw_prop = c_wp * sp_wp
        AR_lw_prop = sp_wp / c_wp
        S_lw_free = c_wf * sp_wf
        AR_lw_free = sp_wf / c_wf

        # Right wing (same dims as left)
        S_rw_prop, AR_rw_prop = S_lw_prop, AR_lw_prop
        S_rw_free, AR_rw_free = S_lw_free, AR_lw_free

        # Elevators (horizontal tail, two symmetric halves) – boxes are (chord, span, thickness)
        c_el, sp_el, _ = box("elevator_left")
        print(f"Elevator box: c={c_el}, sp={sp_el}")
        S_el = c_el * sp_el
        AR_el = sp_el / c_el
        S_er, AR_er = S_el, AR_el

        # Rudder (vertical tail) – box is (chord, thickness, span)
        c_r, _, sp_r = box("rudder")
        S_r = c_r * sp_r
        AR_r = sp_r / c_r
        cz_r = xyz("rudder")[2] + 0.5 * S_r / c_r  # z offset for rudder aero center
        self.cz_rudder = cz_r

        # Prop disk diameter
        diam_prop = box("prop_frame_fuselage_0")[0]
        self.prop_radius = diam_prop / 2.0

        # Global AR (wing and tail)
        self.tip_to_tip = 2 * (sp_wp + sp_wf) + sp_fus
        self.AR_wing = self.tip_to_tip / c_wp
        self.AR_tail = 2 * sp_el / c_el

        # Surface list: (S, AR, chord, kind)
        # kind: 0=fuselage, 1=wing, 2=elevator, 3=rudder, 4=prop
        self._geom = [
            (self.S_perp_fus, S_fus / (c_fus * c_fus), c_fus, 0),  # fuselage
            (S_lw_prop, self.AR_wing, c_wp, 1),  # left inner wing (prop)
            (S_lw_free, self.AR_wing, c_wf, 1),  # left outer wing
            (S_rw_prop, self.AR_wing, c_wp, 1),  # right inner wing
            (S_rw_free, self.AR_wing, c_wf, 1),  # right outer wing
            (S_el, self.AR_tail, c_el, 2),  # left elevator
            (S_er, self.AR_tail, c_el, 2),  # right elevator
            (S_r, AR_r, c_r, 3),  # rudder
            (0.0, 1.0, 0.0, 4),  # prop (placeholder, S and c unused)
        ]

        # Print major geometry info
        print("[AeroSolver]: Parsed URDF geometry:")
        print(f"  Fuselage C={c_fus:.4f} m, L={sz:.4f} m")
        print(f"  Wing C={c_wf:.4f} m, L={sp_wp+sp_wf:.4f} m")
        print(f"  Elevator C={c_el:.4f} m, L={2*sp_el:.4f} m")
        print(f"  Rudder C={c_r:.4f} m, L={sp_r:.4f} m")

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
                slip = float(self.k_slip_wing[b])

            # Reduce x-component for induced velocity
            v_body.x -= slip * ti.cast(self.v_ind[b], ti.f32)

            # Speed and aerodynamic angles
            v_mod = v_body.norm() + 1e-6
            v_mod = ti.math.clamp(v_mod, const_V_MIN, const_V_MAX)
            alpha = ti.atan2(v_body.z, v_body.x)
            beta  = ti.asin(ti.math.clamp(v_body.y / v_mod, -1.0, 1.0))

            # Wrap aerodynamic angles to [-pi/2, pi/2]
            alpha = self._wrap_pm_90(alpha)
            beta  = self._wrap_pm_90(beta)

            # Aero coefficients (lift/drag)
            cl, cd = self._compute_coeff(
                b, self.AR[l], alpha, beta, int(self.kind[l])
            )

            # Trig helpers
            cosb2 = ti.cos(beta) ** 2
            cosa2 = ti.cos(alpha) ** 2

            # Dynamic pressure * area
            ti_05 = 0.5
            qS = ti_05 * self.rho[b] * self.area[l] * v_mod ** 2
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
                    * float(self._thr_flt[b])
                    * self.max_thrust[b]
                )
            elif self.kind[l] == 3:
                # Rudder: forces in YZ plane
                Fb = self._rot_yz(alpha, beta) @ ti.Vector([D * cosa2, S * cosa2, 0.0], dt=ti.f32)
                cp = ti.cast(self._cp_rudder(b, beta, l), ti.f32)
            else:
                # Fuselage or wing: forces in XZ plane
                Fb = self._rot_yz(alpha, beta) @ ti.Vector([D * cosb2, 0.0, L * cosb2], dt=ti.f32)
                if self.kind[l] == 1:  # wing
                    idx = 0 if self.side[l] == 1 else 1
                    self.cl_wing_b[b, idx] = ti.cast(cl, ti.f16)  # store CL for tail
                    cp = ti.cast(self._cp_wing(b, alpha, l), ti.f32)
                else:  # fuselage
                    cp = ti.cast(self._cp_fus(b, alpha, l), ti.f32)

            # ---- Inline noise (no extra pass) ---------------------------------
            do_noise = (self.noise_sigma_mag > 0.0) or (self.noise_sigma_dir > 0.0)
            if do_noise:
                Fb = self._apply_noise(
                    Fb, alpha, beta, int(self.kind[l]),
                    self.noise_sigma_mag, self.noise_sigma_dir
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
                self.lift_dbg[b, l] = ti.cast(L, ti.f16)
                self.drag_dbg[b, l] = ti.cast(D, ti.f16)
                self.side_force_dbg[b, l] = ti.cast(S, ti.f16)

        # 2) Second pass: elevators (tail), using wing downwash
        for b, l in ti.ndrange(self.B, self.L):
            if self.kind[l] != 2:
                continue

            v_body_tail = self._get_wind_in_body(rigid, self._link_idx[l], b)

            slip = 0.0
            if self.slip_code[l] == 0:
                slip = float(self.k_slip_fus[b])
            elif self.slip_code[l] == 1:
                slip = float(self.k_slip_tail[b])
            elif self.slip_code[l] == 2:
                slip = float(self.k_slip_wing[b])

            v_body_tail.x -= slip * ti.cast(self.v_ind[b], ti.f32)

            Vt = v_body_tail.norm() + 1e-6
            Vt = ti.math.clamp(Vt, const_V_MIN, const_V_MAX)

            # Tail angles
            alphat = ti.atan2(v_body_tail.z, v_body_tail.x)
            betat  = ti.asin(ti.math.clamp(v_body_tail.y / Vt, -1.0, 1.0))

            # Wrap to [-pi/2, pi/2]
            alphat = self._wrap_pm_90(alphat)
            betat  = self._wrap_pm_90(betat)

            # Mean wing CL for downwash
            idx = 0 if self.side[l] == 1 else 1
            cl_w_mean = ti.cast(self.cl_wing_b[b, idx], ti.f32)

            # Effective tail AoA with downwash
            alpha_eff = alphat - (self.k_eps_tail[b] * cl_w_mean) / (ti.math.pi * self.AR_wing)
            alpha_eff = self._wrap_pm_90(alpha_eff)

            # Tail coefficients (kind=2)
            cl_t, cd_t = self._compute_coeff(b, self.AR[l], alpha_eff, betat, 2)

            cosb2 = ti.cos(betat) ** 2

            # Tail forces
            qS_t = 0.5 * self.rho[b] * self.area[l] * Vt ** 2 * cosb2
            L_t = qS_t * cl_t
            D_t = qS_t * cd_t
            Fb_t = self._rot_yz(alpha_eff, betat) @ ti.Vector([D_t, 0.0, L_t], dt=ti.f32)

            # ---- Inline noise also for tail -----------------------------------
            do_noise_t = (self.noise_sigma_mag > 0.0) or (self.noise_sigma_dir > 0.0)
            if do_noise_t:
                Fb_t = self._apply_noise(
                    Fb_t, alpha_eff, betat, 2,
                    self.noise_sigma_mag, self.noise_sigma_dir
                )

            # Clamp (after noise)
            Fcap = self.force_cap[b]
            Fn_t = Fb_t.norm() + 1e-9
            scale_t = ti.min(1.0, Fcap / Fn_t)
            Fb_t *= scale_t

            # Tail center of pressure
            cp_t = ti.cast(self._cp_elev(b, alpha_eff, l), ti.f32)

            # Store
            self.force_b[b, l] = Fb_t
            self.cp_b[b, l] = cp_t

        # Noise applied inline: no third pass needed.

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
        alpha: ti.f32,
        beta: ti.f32,
        kind: ti.i32,
        sig_mag_base: ti.f32,
        sig_dir_base: ti.f32,
    ):
        """
        Apply stochastic noise directly to an already computed aerodynamic force F_in.
        (Taichi-safe: single return at the end, no early returns inside non-static branches)
        """
        # Cast to f32 for arithmetic; start with passthrough
        F = ti.cast(F_in, ti.f32)
        out = F  # default: no change

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

        return out

    # ------------------------------------------------------------------
    # Aerodynamic coefficients and centers of pressure
    # ------------------------------------------------------------------
    @ti.func
    def _compute_coeff(self, b, AR, alpha, beta, kind):
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
        # For the rudder (kind=3) use beta as effective alpha
        if kind == 3:
            alpha = beta

        # Base parasitic drag
        cd0 = self.cd0[b]

        # Stall threshold (in radians)
        cut = self.alpha_stall_deg[b] * ti.math.pi / 180.0

        # 2*pi-periodic wrapped angle
        two_pi = 2.0 * ti.math.pi
        a_wr = alpha - two_pi * ti.floor((alpha + ti.math.pi) / two_pi)
        a_abs = ti.abs(a_wr)

        # Folded angle in [0, pi/2]
        a_fold = ti.min(a_abs, ti.math.pi - a_abs)

        # Finite-AR linear lift slope (Prandtl lifting-line)
        cl_a = self.cl_alpha_2d[b] * AR / (2.0 + ti.sqrt(AR * AR + 4.0))

        # Zero-lift angle: fuselage + tail use alpha0_2d_fus, wings + rudder alpha0_2d
        ref = (
            self.alpha0_2d_fus[b]
            if (kind == 0 or kind == 2 or kind == 3)
            else self.alpha0_2d[b]
        )

        cl_lin = cl_a * (a_wr - ref)
        cd_lin = cd0 + cl_lin * cl_lin / (ti.math.pi * AR)

        # Post-stall flat-plate model (periodic)
        sa, ca = ti.sin(a_wr), ti.cos(a_wr)
        cl_st = 2.0 * sa * ca
        cd_st = 2.0 * sa * sa

        # Smooth blend between linear and post-stall using m_smooth as width
        width = self.m_smooth[b] * cut
        width = ti.max(width, 1e-3)  # avoid extremely sharp transitions / div by zero
        s_arg = (a_fold - cut) / width
        s_blend = 0.5 * (1.0 + ti.tanh(s_arg))
        s_blend = ti.min(ti.max(s_blend, 0.0), 1.0)

        # Interpolated coefficients
        cl = (1.0 - s_blend) * cl_lin + s_blend * cl_st
        cd = (1.0 - s_blend) * cd_lin + s_blend * cd_st

        # Fuselage: only drag with fixed cd
        if kind == 0:
            cl = 0.0
            cd = self.cd0_fus[b]

        return cl, cd

    @ti.func
    def _cp_wing(self, b, alpha, l):
        """
        Wing center of pressure (x-offset in local frame).

        Uses a simple shift between `cp_start` and `cp_end` as a function
        of effective angle of attack.
        """
        aa = ti.min(
            ti.abs(
                alpha
                - ti.math.pi * ti.floor((alpha + ti.math.pi) / (2 * ti.math.pi))
            ),
            ti.math.pi / 2,
        )
        cx = (
            -self.cg_to_chord[b]
            + aa / (ti.math.pi / 2) * (self.cp_end[b] - self.cp_start[b])
            + self.cp_start[b]
        ) * self.chord[l]
        return ti.Vector([cx, 0.0, 0.0])

    @ti.func
    def _cp_fus(self, b, alpha, l):
        """
        Fuselage center of pressure (x-offset in fuselage frame).
        """
        aa = ti.min(
            ti.abs(
                alpha
                - ti.math.pi * ti.floor((alpha + ti.math.pi) / (2 * ti.math.pi))
            ),
            ti.math.pi / 2,
        )
        cx = self.cg_fus_local_x + (
            aa / (ti.math.pi / 2) * (self.cp_end[b] - self.cp_start[b])
            + self.cp_start[b]
        ) * self.chord[l]
        return ti.Vector([cx, 0.0, 0.0])

    @ti.func
    def _cp_rudder(self, b, beta, l):
        """
        Rudder center of pressure: [x, 0, z] with fixed z offset.
        """
        aa = ti.min(
            ti.abs(
                beta
                - ti.math.pi * ti.floor((beta + ti.math.pi) / (2 * ti.math.pi))
            ),
            ti.math.pi / 2,
        )
        cx = (
            -self.cg_to_chord[b]
            + aa / (ti.math.pi / 2) * (self.cp_end[b] - self.cp_start[b])
            + self.cp_start[b]
        ) * self.chord[l]
        return ti.Vector([cx, 0.0, ti.cast(self.cz_rudder, ti.f32)])

    @ti.func
    def _cp_elev(self, b, alpha, l):
        """
        Elevator center of pressure [x, y, z] with a small y offset.

        The y offset distinguishes left/right halves of the tailplane so
        that differential elevator deflections can generate rolling moments.
        """
        aa = ti.min(
            ti.abs(
                alpha
                - ti.math.pi * ti.floor((alpha + ti.math.pi) / (2 * ti.math.pi))
            ),
            ti.math.pi / 2,
        )
        cx = (
            -self.cg_to_chord[b]
            + aa / (ti.math.pi / 2) * (self.cp_end[b] - self.cp_start[b])
            + self.cp_start[b]
        ) * self.chord[l]
        cy = (
            0.25
            * (float(self.area[l]) / float(self.chord[l]))
            * float(self.side[l])
        )
        return ti.Vector([cx, ti.cast(cy, ti.f32), 0.02])

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
