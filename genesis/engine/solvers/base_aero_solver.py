import csv
import math
from pathlib import Path

import numpy as np
import torch
import gstaichi as ti

from .base_solver import Solver
from genesis.utils import geom as gu
from genesis.engine.entities import RigidEntity  # for get_link()
from genesis.assets.urdf.mydrone.drone import DroneAeroModel, SurfaceKind


@ti.data_oriented
class BaseAeroSolver(Solver):
    """
    Aerodynamic solver base template that couples a simple aero model with Genesis' RigidSolver.

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

        # Names of wing/elevator duplicated parameters (left/right), set by subclass
        self._wing_param_names: list[str] = []
        self._elevator_param_names: list[str] = []
        self._init_surface_param_names()

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

    def _raise_not_implemented(self, name: str):
        raise NotImplementedError(f"{self.__class__.__name__}.{name} must be implemented by a subclass.")

    # ---------------------------------------------------------------------
    # Surface-specific hooks (stubs; subclasses must override)
    # ---------------------------------------------------------------------
    def _init_surface_param_names(self):
        """Populate per-surface parameter name lists in the subclass."""
        self._raise_not_implemented("_init_surface_param_names")

    def _const_init(self):
        """Surface-specific initialization (override in subclass)."""
        self._raise_not_implemented("_const_init")

    def add_target(
        self,
        entity: RigidEntity,
        urdf_file: str | None = None,
        drone_model: DroneAeroModel | None = None,
    ):
        """Register a target and allocate fields (override in subclass)."""
        self._raise_not_implemented("add_target")

    def _resolve_drone_model(
        self,
        urdf_file: str | None,
        drone_model: DroneAeroModel | None,
    ) -> DroneAeroModel | None:
        """Resolve a drone model from URDF or provided instance (override in subclass)."""
        self._raise_not_implemented("_resolve_drone_model")

    def _apply_drone_model(self, model: DroneAeroModel, entity: RigidEntity):
        """Populate aerodynamic metadata from a drone model (override in subclass)."""
        self._raise_not_implemented("_apply_drone_model")

    def _log_geometry(self, model: DroneAeroModel):
        """Log geometry details for debugging (override in subclass)."""
        self._raise_not_implemented("_log_geometry")

    def _wing_side_keys(self, name: str) -> tuple[str, str]:
        """Return canonical (left, right) keys for a wing parameter name."""
        self._raise_not_implemented("_wing_side_keys")

    def _elevator_side_keys(self, name: str) -> tuple[str, str]:
        """Return canonical (left, right) keys for an elevator parameter name."""
        self._raise_not_implemented("_elevator_side_keys")

    def _ensure_wing_param_entries(self):
        """Guarantee left/right wing parameter keys exist in _aero_base."""
        self._raise_not_implemented("_ensure_wing_param_entries")

    def _ensure_elevator_param_entries(self):
        """Guarantee left/right elevator parameter keys exist in _aero_base."""
        self._raise_not_implemented("_ensure_elevator_param_entries")

    def _aero_compute_kernel(self, rigid: ti.template()):
        """Surface-specific Taichi kernel (override in subclass)."""
        self._raise_not_implemented("_aero_compute_kernel")

    def _propeller_pass(self, rigid: ti.template(), b: int):
        """Propeller pass stub (override in subclass)."""
        self._raise_not_implemented("_propeller_pass")

    def _main_surfaces_pass(self, rigid: ti.template(), b: int, l: int):
        """Main-surface pass stub (override in subclass)."""
        self._raise_not_implemented("_main_surfaces_pass")

    def _tail_surfaces_pass(self, rigid: ti.template(), b: int, l: int):
        """Tail-surface pass stub (override in subclass)."""
        self._raise_not_implemented("_tail_surfaces_pass")

    def _seff_ratio(self, theta: ti.f32, theta_max: ti.f32, ratio_folded: ti.f32):
        """Effective surface ratio stub (override in subclass)."""
        self._raise_not_implemented("_seff_ratio")

    def _compute_eff_S(self, rigid: ti.template(), b: int, l: int):
        """Effective area stub (override in subclass)."""
        self._raise_not_implemented("_compute_eff_S")

    def _compute_eff_span(self, rigid: ti.template(), b: int, l: int):
        """Effective span stub (override in subclass)."""
        self._raise_not_implemented("_compute_eff_span")

    def _compute_eff_AR(self, rigid: ti.template(), b: int, l: int):
        """Effective aspect ratio stub (override in subclass)."""
        self._raise_not_implemented("_compute_eff_AR")

    def _compute_coeff(self, b, l, AR, alpha, beta, kind, side):
        """Aerodynamic coefficients stub (override in subclass)."""
        self._raise_not_implemented("_compute_coeff")

    def _cp_wing(self, rigid: ti.template(), b, alpha, beta, l):
        """Wing center of pressure stub (override in subclass)."""
        self._raise_not_implemented("_cp_wing")

    def _cp_fus(self, rigid: ti.template(), b, alpha, beta, l):
        """Fuselage center of pressure stub (override in subclass)."""
        self._raise_not_implemented("_cp_fus")

    def _cp_rudder(self, rigid: ti.template(), b, alpha, beta, l):
        """Rudder center of pressure stub (override in subclass)."""
        self._raise_not_implemented("_cp_rudder")

    def _cp_elev(self, rigid: ti.template(), b, alpha, beta, l):
        """Elevator center of pressure stub (override in subclass)."""
        self._raise_not_implemented("_cp_elev")

    def _wing_param(self, default_field: ti.template(), left_field: ti.template(), right_field: ti.template(), b: int, side: int):
        """Wing parameter selector stub (override in subclass)."""
        self._raise_not_implemented("_wing_param")

    def _elevator_param(self, default_field: ti.template(), left_field: ti.template(), right_field: ti.template(), b: int, side: int):
        """Elevator parameter selector stub (override in subclass)."""
        self._raise_not_implemented("_elevator_param")

    # ---------------------------------------------------------------------
    # Initialization utilities
    # ---------------------------------------------------------------------
    def _alloc_common_fields(self, B: int, L: int) -> None:
        """
        Allocate Taichi and Torch buffers that are shared across aero solvers.
        """
        self.force_b = ti.Vector.field(3, ti.f32, shape=(B, self.n_links_))
        self.cp_b = ti.Vector.field(3, ti.f32, shape=(B, self.n_links_))
        self._thr_raw = ti.field(ti.f32, shape=(B,))
        self._thr_flt = ti.field(ti.f32, shape=(B,))
        # Per-env sign for prop thrust direction (+1 or -1), set after binding a target.
        self.prop_thrust_sign = ti.field(ti.f32, shape=(B,))
        for b in range(B):
            self.prop_thrust_sign[b] = 1.0
        self.B = B  # alias used inside kernels

        # Aerodynamic force center and global CoM per env
        self._aero_CF_world_b = ti.Vector.field(3, ti.f32, shape=(B,))
        self._CoM_world_b = ti.Vector.field(3, ti.f32, shape=(B,))

        # Torch buffers for zero-copy fetch (filled by _copy_force_cp)
        self._force_buf = torch.empty((B, L, 3), device=self._aero_device, dtype=torch.float32)
        self._cp_buf = torch.empty_like(self._force_buf)

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

    def apply_naca_wing_override(self, naca_code: str | int | float | None, csv_path: str | Path | None = None) -> None:
        code = self._normalize_naca_code(naca_code)
        if not code:
            return
        path = Path(csv_path) if csv_path is not None else self._find_naca4_csv_path()
        if path is None or not path.exists():
            return
        entry = self._load_naca4_row(code, path)
        if entry is None:
            return

        cl_alpha = entry["slope"] * (180.0 / math.pi)  # per-degree -> per-rad
        alpha0 = math.radians(entry["alpha0"])
        alpha_stall = entry["alpha_stall"]
        cd0 = entry["cd0"]

        overrides = {
            "cl_alpha_2d": cl_alpha,
            "alpha0_2d": alpha0,
            "cd0": cd0,
            "alpha_stall_deg": alpha_stall,
        }

        for key, val in overrides.items():
            self._set_param_field(key, val)
            if hasattr(self, "_wing_side_keys"):
                left_key, right_key = self._wing_side_keys(key)
                self._set_param_field(left_key, val)
                self._set_param_field(right_key, val)
            else:
                self._set_param_field(f"{key}_wing_left", val)
                self._set_param_field(f"{key}_wing_right", val)
        re_nom = entry.get("re_nom")
        if re_nom is not None and hasattr(self, "re_nom_link") and hasattr(self, "kind"):
            for b in range(self.B):
                for l in range(self.L):
                    if int(self.kind[l]) == 1:
                        self.re_nom_link[b, l] = float(re_nom)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
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
    # Taichi kernels / device-side logic (generic helpers)
    # ---------------------------------------------------------------------
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
    def _iter_randomizable_params(self):
        names = getattr(self, "_randomizable_param_names", None)
        if not names:
            names = self._aero_base.keys()
        seen = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            if name not in self._aero_base:
                continue
            fld = self._param.get(name)
            if fld is None:
                continue
            yield name, self._aero_base[name], fld

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
            sigma = getattr(self, "noise_sigma_param", 0.0)
        if len(envs_idx) == 0:
            return

        envs_np = envs_idx.cpu().numpy()

        # Loop over aerodynamic constants
        for k, base_val, fld in self._iter_randomizable_params():
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


AeroSolver = BaseAeroSolver
