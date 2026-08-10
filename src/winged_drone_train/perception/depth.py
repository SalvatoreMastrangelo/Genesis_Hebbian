# utils/depth.py
# -----------------------------------------------------------------------------
# Ultra-fast batched depth sensing for Genesis drone envs (GPU with gstaichi).
#
# Rays: S fan-shaped, horizontal in body frame (+X forward, +Y left).
# Intersections with:
#   - vertical cylinders (tree trunks, radius = tree_radius)
#   - lateral walls at y = [y_lower, y_upper]
# FOV shrinks with roll: half_eff = half_nom * cos(roll).
#
# Implementation notes:
#   - No kernel arguments (gstaichi compatibility). Kernels read/write class fields.
#   - Single kernel over (B, S); inner loop over trees (T).
#   - Inputs are Taichi fields filled from torch via .from_torch().
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Optional

import torch
import gstaichi as ti


@ti.data_oriented
class DepthSolver:
    """Batched depth sensing with Taichi kernels."""

    def __init__(
        self,
        num_sectors: int,
        cone_angle_deg: float,
        *,
        max_distance: float = 30.0,
        short_range: float = 0.0,
        tree_radius: float = 1.0,
        y_lower: float = -50.0,
        y_upper: float = 50.0,
        torch_device: torch.device | str = "cuda",
        backend: str = "torch",
    ) -> None:
        if num_sectors <= 0:
            raise ValueError("num_sectors must be positive")

        self.S = int(num_sectors)
        self.cone_angle_deg = float(cone_angle_deg)
        self._torch_device = torch.device(torch_device)
        self.backend = str(backend).strip().lower()
        if self.backend not in {"torch", "taichi"}:
            raise ValueError("DepthSolver backend must be either 'torch' or 'taichi'")
        self._max_distance_value = float(max_distance)
        self._short_range_value = float(short_range)
        self._tree_radius_value = float(tree_radius)
        self._y_lower_value = float(y_lower)
        self._y_upper_value = float(y_upper)
        self._half_cone_nom_value = 0.5 * math.radians(self.cone_angle_deg)

        # Scalar params (kernel-friendly 0D fields)
        self.max_distance = ti.field(dtype=ti.f32, shape=())
        self.short_range = ti.field(dtype=ti.f32, shape=())
        self.tree_radius = ti.field(dtype=ti.f32, shape=())
        self.y_lower = ti.field(dtype=ti.f32, shape=())
        self.y_upper = ti.field(dtype=ti.f32, shape=())
        self.half_cone_nom = ti.field(dtype=ti.f32, shape=())
        self.T_f = ti.field(dtype=ti.i32, shape=())  # active tree count for kernels

        self.max_distance[None] = float(max_distance)
        self.short_range[None] = float(short_range)
        self.tree_radius[None] = float(tree_radius)
        self.y_lower[None] = float(y_lower)
        self.y_upper[None] = float(y_upper)
        self.half_cone_nom[None] = 0.5 * math.radians(self.cone_angle_deg)
        self.T_f[None] = 0

        # Precomputed body-frame ray directions (cos θ, sin θ)
        self.rays_body = ti.Vector.field(2, dtype=ti.f32, shape=(self.S,))
        self._init_ray_dirs()
        angles = torch.linspace(
            -self._half_cone_nom_value,
            self._half_cone_nom_value,
            self.S,
            device=self._torch_device,
            dtype=torch.float32,
        )
        self._ray_dx_t = torch.cos(angles)
        self._ray_dy_t = torch.sin(angles)
        self._sector_idx_t = torch.arange(self.S, device=self._torch_device, dtype=torch.long).view(1, 1, self.S)

        # Input fields (allocated per (B, T))
        self.pos_x_f = None   # (B,)
        self.pos_y_f = None   # (B,)
        self.roll_f = None    # (B,)
        self.yaw_f = None     # (B,)
        self.cyl_f = None     # (B, max(T,1)) of vec2
        # Torch staging buffers reused across steps to reduce allocation churn.
        self._pos_x_t = None
        self._pos_y_t = None
        self._roll_t = None
        self._yaw_t = None
        self._cyl_t = None
        self._noise_t = None

        # Output field (B, S), allocated per B
        self._B_alloc = 0
        self.depth = None
        self._depth_t = None

        # Taichi kernels bind the fields they reference at FIRST compilation
        # and keep reading those exact fields forever — reallocating an input
        # field afterwards would leave the compiled kernel on stale data.
        # Record the shapes the kernels were compiled against so we can
        # refuse (and let callers rebuild) instead of silently going blind.
        self._kernels_launched = False
        self._bound_B = 0
        self._bound_T = 0

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_depth(
        self,
        base_pos: torch.Tensor,       # (B, 3)
        base_euler: torch.Tensor,     # (B, 3) roll, pitch, yaw
        cyl_xy_b: Optional[torch.Tensor] = None,  # (B, T, 2) or None
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        if base_pos.ndim != 2 or base_pos.shape[1] != 3:
            raise ValueError("base_pos must be (B, 3)")
        if base_euler.ndim != 2 or base_euler.shape[1] != 3:
            raise ValueError("base_euler must be (B, 3)")

        if self.backend == "torch":
            return self._compute_depth_torch(base_pos, base_euler, cyl_xy_b=cyl_xy_b, noise_std=noise_std)

        B = int(base_pos.shape[0])
        T = 0 if cyl_xy_b is None else int(cyl_xy_b.shape[1])

        if self.needs_rebuild(B, T):
            raise RuntimeError(
                f"DepthSolver(taichi): input shape changed after kernel "
                f"compilation (B {self._bound_B}->{B}, "
                f"T {self._bound_T}->{max(T, 1)}). Compiled gstaichi kernels "
                f"stay bound to the fields captured at first launch, so "
                f"continuing would silently read stale tree data (the "
                f"blind-exam bug). Construct a fresh DepthSolver instead "
                f"(see WingedDroneEnv._depth_solver_for)."
            )

        self._ensure_buffers(B)
        self._ensure_input_buffers(B, T)
        self._ensure_torch_staging(B, T)

        # Upload inputs via reusable staging buffers.
        self._copy_column_to_staging(self._pos_x_t, base_pos, 0)
        self._copy_column_to_staging(self._pos_y_t, base_pos, 1)
        self._copy_column_to_staging(self._roll_t, base_euler, 0)
        self._copy_column_to_staging(self._yaw_t, base_euler, 2)

        self.pos_x_f.from_torch(self._pos_x_t)
        self.pos_y_f.from_torch(self._pos_y_t)
        self.roll_f.from_torch(self._roll_t)
        self.yaw_f.from_torch(self._yaw_t)

        if T > 0:
            if cyl_xy_b.ndim != 3 or cyl_xy_b.shape[0] != B or cyl_xy_b.shape[2] != 2:
                raise ValueError("cyl_xy_b must be (B, T, 2)")
            if (
                cyl_xy_b.device == self._torch_device
                and cyl_xy_b.dtype == torch.float32
                and cyl_xy_b.is_contiguous()
            ):
                self._cyl_t.copy_(cyl_xy_b)
            else:
                self._cyl_t.copy_(
                    cyl_xy_b.contiguous().to(self._torch_device, dtype=torch.float32)
                )
            self.cyl_f.from_torch(self._cyl_t)
        self.T_f[None] = int(T)

        # Clear and compute
        self._k_clear()
        self._kernel_depth()
        if not self._kernels_launched:
            self._kernels_launched = True
            self._bound_B = B
            self._bound_T = max(T, 1)

        depth = self.depth.to_torch(device=str(self._torch_device))
        self._depth_t.copy_(depth)
        depth = self._depth_t
        if noise_std > 0.0:
            self._ensure_noise_buffer(B)
            self._noise_t.normal_()
            depth.add_(self._noise_t, alpha=float(noise_std))
            depth.clamp_(min=0.0, max=float(self.max_distance[None]))
        return depth

    @torch.no_grad()
    def _compute_depth_torch(
        self,
        base_pos: torch.Tensor,
        base_euler: torch.Tensor,
        *,
        cyl_xy_b: Optional[torch.Tensor],
        noise_std: float,
    ) -> torch.Tensor:
        device = self._torch_device
        base_pos = base_pos if base_pos.device == device else base_pos.to(device=device)
        base_euler = base_euler if base_euler.device == device else base_euler.to(device=device)
        if base_pos.dtype != torch.float32:
            base_pos = base_pos.to(dtype=torch.float32)
        if base_euler.dtype != torch.float32:
            base_euler = base_euler.to(dtype=torch.float32)

        B = int(base_pos.shape[0])
        self._ensure_buffers(B)

        pos_x = base_pos[:, 0]
        pos_y = base_pos[:, 1]
        roll = base_euler[:, 0]
        yaw = base_euler[:, 2]

        dx_b = self._ray_dx_t.view(1, self.S)
        dy_b = self._ray_dy_t.view(1, self.S)

        cy = torch.cos(yaw).unsqueeze(1)
        sy = torch.sin(yaw).unsqueeze(1)
        dx_w = cy * dx_b - sy * dy_b
        dy_w = sy * dx_b + cy * dy_b
        dy_w = dy_w.clamp(min=-0.999, max=0.999)

        max_d = self._max_distance_value
        py = pos_y.unsqueeze(1)
        d_wall = torch.full((B, self.S), max_d, device=device, dtype=torch.float32)
        left_mask = dy_w < 0.0
        right_mask = dy_w > 0.0
        if left_mask.any():
            t_left = (self._y_lower_value - py) / dy_w
            d_wall = torch.minimum(d_wall, torch.where(left_mask, torch.clamp_min(t_left, 0.0), d_wall))
        if right_mask.any():
            t_right = (self._y_upper_value - py) / dy_w
            d_wall = torch.minimum(d_wall, torch.where(right_mask, torch.clamp_min(t_right, 0.0), d_wall))
        depth = d_wall.clamp(max=max_d)

        T = 0 if cyl_xy_b is None else int(cyl_xy_b.shape[1])
        if T > 0:
            if cyl_xy_b.ndim != 3 or cyl_xy_b.shape[0] != B or cyl_xy_b.shape[2] != 2:
                raise ValueError("cyl_xy_b must be (B, T, 2)")
            if cyl_xy_b.device != device or cyl_xy_b.dtype != torch.float32:
                cyl_xy = cyl_xy_b.to(device=device, dtype=torch.float32)
            else:
                cyl_xy = cyl_xy_b

            wx = cyl_xy[:, :, 0]
            wy = cyl_xy[:, :, 1]
            dx = wx - pos_x.unsqueeze(1)
            dy = wy - pos_y.unsqueeze(1)
            cy_tree = torch.cos(yaw).unsqueeze(1)
            sy_tree = torch.sin(yaw).unsqueeze(1)
            x_loc = cy_tree * dx + sy_tree * dy
            y_loc = -sy_tree * dx + cy_tree * dy

            r = torch.sqrt(x_loc * x_loc + y_loc * y_loc) + 1e-9
            r_surf = torch.clamp_min(r - self._tree_radius_value, 0.0)
            theta = torch.atan2(y_loc, x_loc)
            ratio = torch.clamp(self._tree_radius_value / r, max=1.0)
            delta = torch.asin(ratio)

            croll = torch.clamp(torch.cos(roll), min=1e-3).unsqueeze(1)
            half_eff = self._half_cone_nom_value * croll
            in_cone = (theta >= -half_eff) & (theta <= half_eff) & (r <= max_d)
            near = (r <= self._short_range_value) & (x_loc >= 0.0)
            active = in_cone | near
            if active.any():
                sw = (2.0 * half_eff / float(self.S)).clamp(min=1e-6)
                theta_l = theta - delta
                theta_r = theta + delta
                idx_l = torch.floor((theta_l + half_eff) / sw).to(torch.long).clamp_(0, self.S - 1)
                idx_r = torch.floor((theta_r + half_eff) / sw).to(torch.long).clamp_(0, self.S - 1)

                sector_mask = (self._sector_idx_t >= idx_l.unsqueeze(-1)) & (self._sector_idx_t <= idx_r.unsqueeze(-1))
                update_mask = active.unsqueeze(-1) & sector_mask
                tree_depth = torch.where(
                    update_mask,
                    r_surf.unsqueeze(-1),
                    torch.full((1,), max_d, device=device, dtype=torch.float32),
                ).amin(dim=1)
                depth = torch.minimum(depth, tree_depth)

        if noise_std > 0.0:
            self._ensure_noise_buffer(B)
            self._noise_t.normal_()
            depth.add_(self._noise_t, alpha=float(noise_std))
            depth.clamp_(min=0.0, max=max_d)

        self._depth_t.copy_(depth)
        return self._depth_t

    def needs_rebuild(self, B: int, T: int) -> bool:
        """True when this solver can no longer serve inputs of shape (B, T).

        The taichi backend compiles its kernels against the concrete fields
        allocated for the first (B, T) it sees; a later shape change would
        make ``compute_depth`` raise. Callers should then construct a fresh
        ``DepthSolver`` (fresh instance ⇒ fresh kernel bindings). The torch
        backend is shape-dynamic and never needs a rebuild.
        """
        if self.backend != "taichi" or not self._kernels_launched:
            return False
        return int(B) != self._bound_B or max(int(T), 1) != self._bound_T

    # ------------------------------------------------------------------ #
    # Setup / housekeeping                                               #
    # ------------------------------------------------------------------ #
    def _init_ray_dirs(self) -> None:
        """rays_body[s] = (cos θ, sin θ), θ ∈ [-cone/2, +cone/2]."""
        half = 0.5 * math.radians(self.cone_angle_deg)
        if self.S == 1:
            self.rays_body[0] = ti.Vector([1.0, 0.0])
            return
        for s in range(self.S):
            t = -half + (2.0 * half) * (s / (self.S - 1))
            self.rays_body[s] = ti.Vector([math.cos(t), math.sin(t)])

    def _ensure_buffers(self, B: int) -> None:
        """Allocate output depth field for batch size B."""
        if B <= 0:
            raise ValueError("batch B must be > 0")
        if self.depth is not None and self._B_alloc == B:
            return
        self.depth = ti.field(dtype=ti.f32, shape=(B, self.S))
        self._depth_t = torch.empty((B, self.S), device=self._torch_device, dtype=torch.float32)
        self._B_alloc = B

    def _ensure_input_buffers(self, B: int, T: int) -> None:
        """Allocate input fields for current (B, T)."""
        def need(field, shape):
            return (field is None) or (tuple(field.shape) != tuple(shape))

        if need(self.pos_x_f, (B,)):
            self.pos_x_f = ti.field(dtype=ti.f32, shape=(B,))
        if need(self.pos_y_f, (B,)):
            self.pos_y_f = ti.field(dtype=ti.f32, shape=(B,))
        if need(self.roll_f, (B,)):
            self.roll_f = ti.field(dtype=ti.f32, shape=(B,))
        if need(self.yaw_f, (B,)):
            self.yaw_f = ti.field(dtype=ti.f32, shape=(B,))

        TT = max(T, 1)  # keep a valid shape when no trees
        if (self.cyl_f is None) or (self.cyl_f.shape[0] != B) or (self.cyl_f.shape[1] != TT):
            self.cyl_f = ti.Vector.field(2, dtype=ti.f32, shape=(B, TT))

    def _ensure_torch_staging(self, B: int, T: int) -> None:
        """Allocate torch staging buffers for current (B, T)."""
        if self._pos_x_t is None or self._pos_x_t.shape[0] != B:
            self._pos_x_t = torch.empty((B,), device=self._torch_device, dtype=torch.float32)
            self._pos_y_t = torch.empty((B,), device=self._torch_device, dtype=torch.float32)
            self._roll_t = torch.empty((B,), device=self._torch_device, dtype=torch.float32)
            self._yaw_t = torch.empty((B,), device=self._torch_device, dtype=torch.float32)
        TT = max(T, 1)
        if self._cyl_t is None or self._cyl_t.shape[0] != B or self._cyl_t.shape[1] != TT:
            self._cyl_t = torch.empty((B, TT, 2), device=self._torch_device, dtype=torch.float32)

    def _copy_column_to_staging(self, dst: torch.Tensor, src2d: torch.Tensor, col: int) -> None:
        """Copy one column from a (B, C) tensor to reusable staging buffer."""
        col_view = src2d[:, col]
        if col_view.device == self._torch_device and col_view.dtype == torch.float32:
            dst.copy_(col_view)
        else:
            dst.copy_(col_view.to(self._torch_device, dtype=torch.float32))

    def _ensure_noise_buffer(self, B: int) -> None:
        if self._noise_t is None or self._noise_t.shape != (B, self.S):
            self._noise_t = torch.empty((B, self.S), device=self._torch_device, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Kernels (no arguments; iterate over field shapes)                  #
    # ------------------------------------------------------------------ #
    @ti.kernel
    def _k_clear(self):
        max_d = self.max_distance[None]
        for b, s in self.depth:
            self.depth[b, s] = max_d

    @ti.kernel
    def _kernel_depth(self):
        max_d = self.max_distance[None]
        r_tree = self.tree_radius[None]
        yL = self.y_lower[None]
        yU = self.y_upper[None]
        halfNom = self.half_cone_nom[None]
        T = self.T_f[None]

        # Walls: initialize depth with nearest lateral wall
        for b, s in self.depth:
            dx_b = self.rays_body[s][0]
            dy_b = self.rays_body[s][1]

            cy = ti.cos(self.yaw_f[b])
            sy = ti.sin(self.yaw_f[b])
            dx_w = cy * dx_b - sy * dy_b
            dy_w = sy * dx_b + cy * dy_b

            if dy_w > 0.999:
                dy_w = 0.999
            if dy_w < -0.999:
                dy_w = -0.999

            py = self.pos_y_f[b]
            tL = max_d
            if dy_w < 0.0:
                tL = (yL - py) / dy_w
            tR = max_d
            if dy_w > 0.0:
                tR = (yU - py) / dy_w

            d_wall = ti.min(ti.max(tL, 0.0), ti.max(tR, 0.0))
            if d_wall > max_d:
                d_wall = max_d

            self.depth[b, s] = d_wall


        # Trees: reduce with nearest cylinder surface
        for b, s in self.depth:
            croll = ti.cos(self.roll_f[b])
            if croll < 1e-3:
                croll = 1e-3
            halfEff = halfNom * croll
            sw = 2.0 * halfEff / ti.cast(self.S, ti.f32)

            cy = ti.cos(self.yaw_f[b])
            sy = ti.sin(self.yaw_f[b])

            best = self.depth[b, s]

            for t in range(T):
                wx = self.cyl_f[b, t][0]
                wy = self.cyl_f[b, t][1]

                dx = wx - self.pos_x_f[b]
                dy = wy - self.pos_y_f[b]
                x_loc = cy * dx + sy * dy
                y_loc = -sy * dx + cy * dy

                r = ti.sqrt(x_loc * x_loc + y_loc * y_loc) + 1e-9
                r_surf = r - r_tree
                if r_surf < 0.0:
                    r_surf = 0.0

                theta = ti.atan2(y_loc, x_loc)
                ratio = r_tree / r
                if ratio > 1.0:
                    ratio = 1.0
                delta = ti.asin(ratio)

                in_cone = (-halfEff <= theta) and (theta <= halfEff) and (r <= max_d)
                near = (r <= self.short_range[None]) and (x_loc >= 0.0)
                if not (in_cone or near):
                    continue

                theta_l = theta - delta
                theta_r = theta + delta
                idx_l = ti.i32(ti.floor((theta_l + halfEff) / sw))
                idx_r = ti.i32(ti.floor((theta_r + halfEff) / sw))
                if idx_l < 0:
                    idx_l = 0
                if idx_r < 0:
                    idx_r = 0
                if idx_l > self.S - 1:
                    idx_l = self.S - 1
                if idx_r > self.S - 1:
                    idx_r = self.S - 1

                if (idx_l <= s) and (s <= idx_r):
                    if r_surf < best:
                        best = r_surf

            self.depth[b, s] = best
