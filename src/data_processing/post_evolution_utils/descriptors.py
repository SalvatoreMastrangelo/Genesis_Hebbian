from __future__ import annotations

import ast
import atexit
import math
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .common import (
    AIR_DENSITY_KG_M3,
    AIR_DYNAMIC_VISCOSITY_KG_M_S,
    _analysis_csv_path,
)

NACA4_CSV_PATH = Path(__file__).resolve().parents[2] / "naca_generation" / "naca4.csv"
NACA4_FULL_POLARS_CSV_PATH = Path(__file__).resolve().parents[2] / "naca_generation" / "naca4_full_polars.csv"
NACA4_PER_RE_CSV_PATH = Path(__file__).resolve().parents[2] / "naca_generation" / "naca4_per_re_metrics_long.csv"
TAIL_SOLVER_AIRFOIL_CODE = "0216"
_TEMP_URDF_DIR_CTX: tempfile.TemporaryDirectory[str] | None = None
_TEMP_URDF_DIR_PATH: Path | None = None


def _normalize_airfoil_code(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("naca"):
        text = text[4:].strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(4) if digits else ""


def _cleanup_temp_urdf_dir() -> None:
    global _TEMP_URDF_DIR_CTX, _TEMP_URDF_DIR_PATH
    if _TEMP_URDF_DIR_CTX is not None:
        _TEMP_URDF_DIR_CTX.cleanup()
        _TEMP_URDF_DIR_CTX = None
        _TEMP_URDF_DIR_PATH = None


def _get_temp_urdf_dir() -> Path:
    global _TEMP_URDF_DIR_CTX, _TEMP_URDF_DIR_PATH
    if _TEMP_URDF_DIR_PATH is None:
        _TEMP_URDF_DIR_CTX = tempfile.TemporaryDirectory(prefix="post_evolution_urdf_")
        _TEMP_URDF_DIR_PATH = Path(_TEMP_URDF_DIR_CTX.name)
    return _TEMP_URDF_DIR_PATH


atexit.register(_cleanup_temp_urdf_dir)


def _load_chromosome_drone_class():
    try:
        from morph_evolution.chromosome_drone import Chromosome_Drone
    except Exception:
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from morph_evolution.chromosome_drone import Chromosome_Drone
    return Chromosome_Drone


@lru_cache(maxsize=1)
def _load_urdf_maker_class():
    try:
        from drone_making import UrdfMaker
    except Exception:
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from drone_making import UrdfMaker
    return UrdfMaker


@lru_cache(maxsize=1)
def _load_naca_metrics_table() -> pd.DataFrame:
    naca_df = pd.read_csv(NACA4_CSV_PATH)
    naca_df["airfoil_code"] = naca_df["airfoil"].map(_normalize_airfoil_code)
    return naca_df.drop_duplicates("airfoil_code").copy()


@lru_cache(maxsize=1)
def _load_naca_per_re_table() -> pd.DataFrame:
    per_re_df = pd.read_csv(NACA4_PER_RE_CSV_PATH)
    per_re_df["airfoil_code"] = per_re_df["airfoil"].map(_normalize_airfoil_code)
    return per_re_df.copy()


@lru_cache(maxsize=1)
def _load_naca_full_polar_summary_table() -> pd.DataFrame:
    summary = _load_naca_metrics_table().set_index("airfoil_code")
    full = pd.read_csv(NACA4_FULL_POLARS_CSV_PATH)
    full["airfoil_code"] = full["NACA"].map(_normalize_airfoil_code)

    rows: list[dict[str, float | str]] = []
    for airfoil_code, group in full.groupby("airfoil_code", sort=True):
        if not airfoil_code:
            continue
        group = group.copy()
        target_re = float(summary.loc[airfoil_code, "ReNom"]) if airfoil_code in summary.index else float(np.nanmedian(pd.to_numeric(group["Re"], errors="coerce")))
        re_values = np.sort(pd.to_numeric(group["Re"], errors="coerce").dropna().unique())
        if re_values.size == 0:
            continue
        chosen_re = float(re_values[np.argmin(np.abs(re_values - target_re))])
        subset = group.loc[pd.to_numeric(group["Re"], errors="coerce") == chosen_re].copy()
        subset["alpha"] = pd.to_numeric(subset["alpha"], errors="coerce")
        subset["cl"] = pd.to_numeric(subset["cl"], errors="coerce")
        subset["cd"] = pd.to_numeric(subset["cd"], errors="coerce")
        subset["cm"] = pd.to_numeric(subset["cm"], errors="coerce")
        subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=["alpha", "cl", "cd"])
        if subset.empty:
            continue
        subset = subset.sort_values("alpha", kind="mergesort")

        cl_peak_idx = int(subset["cl"].idxmax())
        cd_peak_idx = int(subset["cd"].idxmax())
        cd_min_idx = int(subset["cd"].idxmin())
        ld_candidates = subset.loc[(subset["cl"] > 0.0) & (subset["cd"] > 0.0)].copy()
        if ld_candidates.empty:
            ld_max = float("nan")
            alpha_ld_max = float("nan")
            cl_ld_max = float("nan")
            cd_ld_max = float("nan")
            cm_ld_max = float("nan")
        else:
            ld_values = ld_candidates["cl"] / ld_candidates["cd"]
            best_idx = int(ld_values.idxmax())
            ld_max = float(ld_values.loc[best_idx])
            alpha_ld_max = float(ld_candidates.loc[best_idx, "alpha"])
            cl_ld_max = float(ld_candidates.loc[best_idx, "cl"])
            cd_ld_max = float(ld_candidates.loc[best_idx, "cd"])
            cm_ld_max = float(ld_candidates.loc[best_idx, "cm"]) if "cm" in ld_candidates else float("nan")

        cl_peak = float(subset.loc[cl_peak_idx, "cl"])
        cd_at_cl_peak = float(subset.loc[cl_peak_idx, "cd"])
        cd_peak = float(subset.loc[cd_peak_idx, "cd"])
        rows.append(
            {
                "airfoil_code": airfoil_code,
                "polar_re_used": chosen_re,
                "cl_peak_2d": cl_peak,
                "alpha_cl_peak_deg_2d": float(subset.loc[cl_peak_idx, "alpha"]),
                "cd_at_cl_peak_2d": cd_at_cl_peak,
                "cm_at_cl_peak_2d": float(subset.loc[cl_peak_idx, "cm"]) if "cm" in subset else float("nan"),
                "cd_peak_2d": cd_peak,
                "alpha_cd_peak_deg_2d": float(subset.loc[cd_peak_idx, "alpha"]),
                "cd_min_2d": float(subset.loc[cd_min_idx, "cd"]),
                "alpha_cd_min_deg_2d": float(subset.loc[cd_min_idx, "alpha"]),
                "cl_peak_over_cd_peak_2d": cl_peak / cd_peak if abs(cd_peak) > 1e-12 else float("nan"),
                "clcd_max_2d": ld_max,
                "alpha_clcd_max_deg_2d": alpha_ld_max,
                "cl_at_clcd_max_2d": cl_ld_max,
                "cd_at_clcd_max_2d": cd_ld_max,
                "cm_at_clcd_max_2d": cm_ld_max,
            }
        )
    return pd.DataFrame(rows)


def _lookup_airfoil_series(airfoil_code: str) -> pd.Series:
    metrics = _load_naca_metrics_table().set_index("airfoil_code")
    if airfoil_code in metrics.index:
        return metrics.loc[airfoil_code]
    return pd.Series({"ReNom": np.nan, "slope": np.nan, "alpha_stall": np.nan, "alpha0": np.nan, "cd0": np.nan})


def _lookup_airfoil_polar_series(airfoil_code: str) -> pd.Series:
    polar = _load_naca_full_polar_summary_table().set_index("airfoil_code")
    if airfoil_code in polar.index:
        return polar.loc[airfoil_code]
    return pd.Series(
        {
            "polar_re_used": np.nan,
            "cl_peak_2d": np.nan,
            "alpha_cl_peak_deg_2d": np.nan,
            "cd_at_cl_peak_2d": np.nan,
            "cm_at_cl_peak_2d": np.nan,
            "cd_peak_2d": np.nan,
            "alpha_cd_peak_deg_2d": np.nan,
            "cd_min_2d": np.nan,
            "alpha_cd_min_deg_2d": np.nan,
            "cl_peak_over_cd_peak_2d": np.nan,
            "clcd_max_2d": np.nan,
            "alpha_clcd_max_deg_2d": np.nan,
            "cl_at_clcd_max_2d": np.nan,
            "cd_at_clcd_max_2d": np.nan,
            "cm_at_clcd_max_2d": np.nan,
        }
    )


def _parse_xyz(origin_elem) -> np.ndarray:
    if origin_elem is None:
        return np.zeros(3, dtype=float)
    raw = origin_elem.get("xyz", "0 0 0")
    return np.asarray([float(tok) for tok in raw.split()], dtype=float)


def _parse_rpy(origin_elem) -> np.ndarray:
    if origin_elem is None:
        return np.zeros(3, dtype=float)
    raw = origin_elem.get("rpy", "0 0 0")
    return np.asarray([float(tok) for tok in raw.split()], dtype=float)


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    r, p, y = [float(v) for v in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _origin_transform(origin_elem) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rpy_matrix(_parse_rpy(origin_elem))
    transform[:3, 3] = _parse_xyz(origin_elem)
    return transform


def _build_link_world_transforms(root) -> dict[str, np.ndarray]:
    transforms: dict[str, np.ndarray] = {"root_link": np.eye(4, dtype=float)}
    children_by_parent: dict[str, list[tuple[str, np.ndarray]]] = {}
    all_children: set[str] = set()
    for joint in root.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is None or child_elem is None:
            continue
        parent = parent_elem.get("link")
        child = child_elem.get("link")
        if not parent or not child:
            continue
        children_by_parent.setdefault(parent, []).append((child, _origin_transform(joint.find("origin"))))
        all_children.add(child)
    root_candidates = [link.get("name") for link in root.findall("link") if link.get("name") not in all_children]
    start = "root_link" if "root_link" in {link.get("name") for link in root.findall("link")} else (root_candidates[0] if root_candidates else None)
    if start is None:
        return transforms
    transforms = {start: np.eye(4, dtype=float)}
    stack = [start]
    while stack:
        parent = stack.pop()
        parent_transform = transforms[parent]
        for child, local_transform in children_by_parent.get(parent, []):
            transforms[child] = parent_transform @ local_transform
            stack.append(child)
    return transforms


def _extract_total_mass_and_cg(root, transforms: dict[str, np.ndarray]) -> tuple[float, np.ndarray]:
    total_mass = 0.0
    weighted_sum = np.zeros(3, dtype=float)
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if not name or inertial is None or name not in transforms:
            continue
        mass_elem = inertial.find("mass")
        if mass_elem is None:
            continue
        mass = float(mass_elem.get("value", 0.0))
        if not math.isfinite(mass) or mass <= 0.0:
            continue
        origin = inertial.find("origin")
        com_local = _parse_xyz(origin)
        com_world = transforms[name] @ np.array([com_local[0], com_local[1], com_local[2], 1.0], dtype=float)
        total_mass += mass
        weighted_sum += mass * com_world[:3]
    if total_mass <= 0.0:
        return 0.0, np.full(3, np.nan, dtype=float)
    return total_mass, weighted_sum / total_mass


def _mean_x(transforms: dict[str, np.ndarray], names: Sequence[str]) -> float:
    values = [float(transforms[name][0, 3]) for name in names if name in transforms]
    return float(np.mean(values)) if values else float("nan")


def _get_exact_geometry_row(phys_key: tuple[float, ...]) -> dict[str, float | str]:
    phys = np.asarray(phys_key, dtype=float)
    wing_semispan = float(phys[0])
    wing_ar_half = float(phys[1])
    fuselage_length = float(phys[2])
    cg_ratio = float(phys[3])
    attach_ratio = float(phys[4])
    elevator_span_total = float(phys[5])
    elevator_ar = float(phys[6])
    rudder_span = float(phys[7])
    rudder_ar = float(phys[8])
    dihedral_deg = float(phys[9])
    sweep_multiplier = float(phys[10])
    twist_multiplier = float(phys[11])
    naca_d1 = int(round(float(phys[12])))
    naca_d2 = int(round(float(phys[13])))
    naca_last2 = int(round(float(phys[14])))

    wing_chord = wing_semispan / max(wing_ar_half, 1e-12)
    elevator_chord = elevator_span_total / max(elevator_ar, 1e-12)
    rudder_chord = rudder_span / max(rudder_ar, 1e-12)
    wing_total_span = 2.0 * wing_semispan
    wing_area_total = wing_total_span * wing_chord
    elevator_area_total = elevator_span_total * elevator_chord
    rudder_area = rudder_span * rudder_chord
    wing_thickness = wing_chord * (naca_last2 / 100.0)
    wing_aspect_ratio_total = wing_total_span / max(wing_chord, 1e-12)

    airfoil_code = f"{naca_d1}{naca_d2}{naca_last2:02d}"
    wing_airfoil = _lookup_airfoil_series(airfoil_code)
    tail_airfoil = _lookup_airfoil_series(TAIL_SOLVER_AIRFOIL_CODE)
    wing_polar = _lookup_airfoil_polar_series(airfoil_code)
    tail_polar = _lookup_airfoil_polar_series(TAIL_SOLVER_AIRFOIL_CODE)

    UrdfMaker = _load_urdf_maker_class()
    temp_urdf_dir = _get_temp_urdf_dir()
    temp_urdf_dir.mkdir(parents=True, exist_ok=True)
    maker = UrdfMaker(list(phys), out_dir=temp_urdf_dir, aero_solver_kind="simple")
    root = maker.build_tree().getroot()
    transforms = _build_link_world_transforms(root)
    total_mass, cg = _extract_total_mass_and_cg(root, transforms)

    wing_ac_x = _mean_x(transforms, ["aero_frame_left_wing", "aero_frame_right_wing"])
    tail_ac_x = _mean_x(transforms, ["aero_frame_elevator_left", "aero_frame_elevator_right"])
    rudder_ac_x = _mean_x(transforms, ["aero_frame_rudder"])
    tail_hinge_x = _mean_x(transforms, ["elevator_hinge"])
    wing_attach_x = -attach_ratio * fuselage_length

    wing_le_x = wing_ac_x + 0.25 * wing_chord
    wing_mac = wing_chord
    h_cg = (wing_le_x - cg[0]) / max(wing_mac, 1e-12)
    h_ac_w = (wing_le_x - wing_ac_x) / max(wing_mac, 1e-12)
    tail_arm_from_wing_ac = wing_ac_x - tail_ac_x
    tail_arm_from_cg = cg[0] - tail_ac_x
    rudder_arm_from_cg = cg[0] - rudder_ac_x

    horizontal_tail_volume_coeff = (elevator_area_total * tail_arm_from_wing_ac) / max(wing_area_total * wing_mac, 1e-12)
    vertical_tail_volume_coeff = (rudder_area * rudder_arm_from_cg) / max(wing_area_total * wing_total_span, 1e-12)

    wing_slope_2d = float(wing_airfoil.get("slope", np.nan)) * (180.0 / math.pi)
    tail_slope_2d = float(tail_airfoil.get("slope", np.nan)) * (180.0 / math.pi)
    wing_lift_curve_slope_3d = wing_slope_2d / (1.0 + wing_slope_2d / (math.pi * max(wing_aspect_ratio_total, 1e-12))) if math.isfinite(wing_slope_2d) else float("nan")
    tail_lift_curve_slope_3d = tail_slope_2d / (1.0 + tail_slope_2d / (math.pi * max(elevator_ar, 1e-12))) if math.isfinite(tail_slope_2d) else float("nan")
    downwash_gradient = 2.0 * wing_lift_curve_slope_3d / (math.pi * max(wing_aspect_ratio_total, 1e-12)) if math.isfinite(wing_lift_curve_slope_3d) else float("nan")
    downwash_gradient = float(np.clip(downwash_gradient, 0.0, 0.95)) if math.isfinite(downwash_gradient) else float("nan")

    fus_width = float(getattr(UrdfMaker, "_REF", {}).get("tfus", np.nan))
    fus_height = float(getattr(UrdfMaker, "_REF", {}).get("hfus", np.nan))
    fuselage_volume_est = math.pi * 0.25 * fus_width * fus_height * fuselage_length if math.isfinite(fus_width) and math.isfinite(fus_height) else float("nan")
    fuselage_cm_alpha_est = 2.0 * fuselage_volume_est / max(wing_area_total * wing_mac, 1e-12) if math.isfinite(fuselage_volume_est) else float("nan")

    if math.isfinite(wing_lift_curve_slope_3d) and abs(wing_lift_curve_slope_3d) > 1e-12:
        neutral_point_pct_mac = 100.0 * (
            h_ac_w
            - fuselage_cm_alpha_est / wing_lift_curve_slope_3d
            + horizontal_tail_volume_coeff * (tail_lift_curve_slope_3d / wing_lift_curve_slope_3d) * (1.0 - downwash_gradient)
        )
    else:
        neutral_point_pct_mac = float("nan")
    cg_pct_mac = 100.0 * h_cg if math.isfinite(h_cg) else float("nan")
    static_margin_pct_mac = neutral_point_pct_mac - cg_pct_mac if math.isfinite(neutral_point_pct_mac) and math.isfinite(cg_pct_mac) else float("nan")
    neutral_point_x = wing_le_x - (neutral_point_pct_mac / 100.0) * wing_mac if math.isfinite(neutral_point_pct_mac) else float("nan")

    fuselage_cd0 = 0.75
    fuselage_ref_area = math.pi * 0.25 * fus_width * fus_height if math.isfinite(fus_width) and math.isfinite(fus_height) else float("nan")
    wing_cd0 = float(wing_airfoil.get("cd0", np.nan))
    tail_cd0 = float(tail_airfoil.get("cd0", np.nan))
    aircraft_cd0_est = (
        wing_cd0
        + tail_cd0 * (elevator_area_total + rudder_area) / max(wing_area_total, 1e-12)
        + fuselage_cd0 * fuselage_ref_area / max(wing_area_total, 1e-12)
    )
    oswald_efficiency_est = 1.78 * (1.0 - 0.045 * max(wing_aspect_ratio_total, 0.0) ** 0.68) - 0.64
    oswald_efficiency_est = float(np.clip(oswald_efficiency_est, 0.3, 0.98))
    induced_drag_factor_est = 1.0 / (math.pi * max(wing_aspect_ratio_total, 1e-12) * max(oswald_efficiency_est, 1e-12))
    aircraft_cl_at_ld_max_est = math.sqrt(max(aircraft_cd0_est, 0.0) / induced_drag_factor_est) if math.isfinite(aircraft_cd0_est) and math.isfinite(induced_drag_factor_est) and induced_drag_factor_est > 0.0 else float("nan")
    aircraft_ld_max_est = 1.0 / (2.0 * math.sqrt(max(aircraft_cd0_est, 0.0) * induced_drag_factor_est)) if math.isfinite(aircraft_cd0_est) and math.isfinite(induced_drag_factor_est) and aircraft_cd0_est >= 0.0 and induced_drag_factor_est > 0.0 else float("nan")

    sweep_limit_rad = 0.7 / max(sweep_multiplier, 0.5)
    twist_range_rad = 0.5 / max(twist_multiplier, 0.5)
    wing_ac_sweep_shift = abs(math.sin(sweep_limit_rad) * wing_semispan)
    cg_minus_wing_ac = float(cg[0] - wing_ac_x)
    cg_minus_wing_ac_fus_pct = 100.0 * cg_minus_wing_ac / max(fuselage_length, 1e-12)
    wing_ac_sweep_shift_fus_pct = 100.0 * wing_ac_sweep_shift / max(fuselage_length, 1e-12)

    wing_loading_n_m2 = total_mass * 9.81 / max(wing_area_total, 1e-12)

    row: dict[str, float | str] = {
        "wing_span": wing_semispan,
        "wing_semispan": wing_semispan,
        "wing_total_span": wing_total_span,
        "wing_aspect_ratio": wing_ar_half,
        "wing_aspect_ratio_total": wing_aspect_ratio_total,
        "wing_chord": wing_chord,
        "wing_area_one_side": wing_semispan * wing_chord,
        "wing_area_total": wing_area_total,
        "wing_thickness": wing_thickness,
        "fuselage_length": fuselage_length,
        "mass_total_kg": total_mass,
        "cg_x": float(cg[0]),
        "cg_y": float(cg[1]),
        "cg_z": float(cg[2]),
        "wing_attach_x": wing_attach_x,
        "wing_ac_x": wing_ac_x,
        "wing_le_x": wing_le_x,
        "neutral_point_x": neutral_point_x,
        "cg_minus_wing_ac": cg_minus_wing_ac,
        "cg_minus_wing_ac_fus_pct": cg_minus_wing_ac_fus_pct,
        "cg_minus_wing_ac_fus_pct_min": cg_minus_wing_ac_fus_pct - wing_ac_sweep_shift_fus_pct,
        "cg_minus_wing_ac_fus_pct_max": cg_minus_wing_ac_fus_pct + wing_ac_sweep_shift_fus_pct,
        "wing_ac_sweep_shift": wing_ac_sweep_shift,
        "wing_ac_sweep_shift_fus_pct": wing_ac_sweep_shift_fus_pct,
        "tail_hinge_x": tail_hinge_x,
        "tail_arm": tail_arm_from_cg,
        "tail_arm_from_wing_ac": tail_arm_from_wing_ac,
        "tail_arm_from_cg": tail_arm_from_cg,
        "elevator_span": elevator_span_total,
        "elevator_aspect_ratio": elevator_ar,
        "elevator_chord": elevator_chord,
        "elevator_area": elevator_area_total,
        "rudder_span": rudder_span,
        "rudder_aspect_ratio": rudder_ar,
        "rudder_chord": rudder_chord,
        "rudder_area": rudder_area,
        "horizontal_tail_volume": horizontal_tail_volume_coeff,
        "horizontal_tail_volume_coeff": horizontal_tail_volume_coeff,
        "horizontal_tail_surface_lever": elevator_area_total * tail_arm_from_cg,
        "lateral_tail_surface_lever": rudder_area * rudder_arm_from_cg,
        "vertical_tail_volume": vertical_tail_volume_coeff,
        "vertical_tail_volume_coeff": vertical_tail_volume_coeff,
        "dihedral_deg": dihedral_deg,
        "sweep_multiplier": sweep_multiplier,
        "twist_multiplier": twist_multiplier,
        "sweep_range_rad": sweep_limit_rad,
        "twist_range_rad": twist_range_rad,
        "naca_d1": float(naca_d1),
        "naca_d2": float(naca_d2),
        "naca_last2": float(naca_last2),
        "wing_airfoil_code": airfoil_code,
        "solver_wing_airfoil_code": airfoil_code,
        "solver_tail_airfoil_code": TAIL_SOLVER_AIRFOIL_CODE,
        "wing_airfoil_camber_pct": float(naca_d1),
        "wing_airfoil_camber_pos_tenths": float(naca_d2),
        "wing_airfoil_thickness_pct": float(naca_last2),
        "wing_re_nom_a1": float(wing_airfoil.get("ReNom", np.nan)),
        "wing_slope_a1": float(wing_airfoil.get("slope", np.nan)),
        "wing_alpha_stall_deg_a1": float(wing_airfoil.get("alpha_stall", np.nan)),
        "wing_alpha0_deg_a1": float(wing_airfoil.get("alpha0", np.nan)),
        "wing_cd0_a1": wing_cd0,
        "tail_re_nom_solver": float(tail_airfoil.get("ReNom", np.nan)),
        "tail_slope_solver": float(tail_airfoil.get("slope", np.nan)),
        "tail_alpha_stall_deg_solver": float(tail_airfoil.get("alpha_stall", np.nan)),
        "tail_alpha0_deg_solver": float(tail_airfoil.get("alpha0", np.nan)),
        "tail_cd0_solver": tail_cd0,
        "wing_lift_curve_slope_3d": wing_lift_curve_slope_3d,
        "tail_lift_curve_slope_3d": tail_lift_curve_slope_3d,
        "downwash_gradient": downwash_gradient,
        "fuselage_cm_alpha_est": fuselage_cm_alpha_est,
        "cg_pct_mac": cg_pct_mac,
        "neutral_point_pct_mac": neutral_point_pct_mac,
        "static_margin": static_margin_pct_mac / 100.0 if math.isfinite(static_margin_pct_mac) else float("nan"),
        "static_margin_pct_mac": static_margin_pct_mac,
        "wing_loading_n_m2": wing_loading_n_m2,
        "aircraft_cd0_est": aircraft_cd0_est,
        "oswald_efficiency_est": oswald_efficiency_est,
        "induced_drag_factor_est": induced_drag_factor_est,
        "aircraft_cl_at_ld_max_est": aircraft_cl_at_ld_max_est,
        "aircraft_ld_max_est": aircraft_ld_max_est,
    }
    for key, value in wing_polar.items():
        row[f"wing_{key}"] = float(value) if isinstance(value, (int, float, np.floating)) else value
    for key, value in tail_polar.items():
        row[f"tail_{key}"] = float(value) if isinstance(value, (int, float, np.floating)) else value
    return row


@lru_cache(maxsize=4096)
def _exact_geometry_row_cached(phys_key: tuple[float, ...]) -> dict[str, float | str]:
    return _get_exact_geometry_row(phys_key)


def _parse_chromosome_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    if "chromosome" not in df.columns:
        raise ValueError("Missing chromosome column for genome PCA.")

    parsed = []
    indices = []
    for idx, val in df["chromosome"].items():
        if pd.isna(val):
            continue
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val, dtype=float)
        elif isinstance(val, str):
            try:
                arr = np.asarray(ast.literal_eval(val), dtype=float)
            except (SyntaxError, ValueError):
                continue
        else:
            continue
        if arr.ndim != 1:
            continue
        parsed.append(arr)
        indices.append(idx)

    if not parsed:
        raise ValueError("No valid genomes found in chromosome column.")

    lengths = np.array([len(arr) for arr in parsed], dtype=int)
    target_len = int(np.bincount(lengths).argmax())
    keep_mask = lengths == target_len
    if not np.any(keep_mask):
        raise ValueError("No genomes with consistent length for PCA.")

    matrix = np.vstack([arr for arr, keep in zip(parsed, keep_mask) if keep])
    idx_keep = [idx for idx, keep in zip(indices, keep_mask) if keep]
    return matrix, df.loc[idx_keep].copy()


def _aero_descriptor_frame_from_physical(
    phys: np.ndarray,
    *,
    aligned_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if phys.ndim != 2 or phys.shape[1] < 15:
        raise ValueError("Expected physical genome matrix with shape (N, 15).")

    rows = []
    for row in phys:
        key = tuple(float(v) for v in row.tolist())
        rows.append(_exact_geometry_row_cached(key))

    desc_df = pd.DataFrame(rows).reset_index(drop=True)
    if aligned_df is not None:
        base_df = aligned_df.copy().reset_index(drop=True)
        overlapping = [col for col in desc_df.columns if col in base_df.columns]
        if overlapping:
            base_df = base_df.drop(columns=overlapping)
    else:
        base_df = pd.DataFrame(index=np.arange(len(desc_df)))
    out = pd.concat([base_df, desc_df], axis=1)
    if "ff_1" in out.columns:
        out["cost_of_transport"] = -pd.to_numeric(out["ff_1"], errors="coerce")
    return out


def _aero_descriptor_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    matrix, aligned_df = _parse_chromosome_matrix(df)
    Chromosome_Drone = _load_chromosome_drone_class()
    phys = np.asarray([Chromosome_Drone.to_physical(row) for row in matrix], dtype=float)
    return _aero_descriptor_frame_from_physical(phys, aligned_df=aligned_df)


def _reward_curve_matrix(df: pd.DataFrame) -> np.ndarray:
    reward_cols = [f"rew_{pct}pct" for pct in range(10, 101, 10)]
    reward_df = pd.DataFrame(index=df.index)
    for col in reward_cols:
        if col in df.columns:
            reward_df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            reward_df[col] = np.nan
    return reward_df.to_numpy(dtype=float)


def _interpolated_reward_curve(
    reward_values: np.ndarray,
    *,
    n_points: int = 201,
) -> tuple[np.ndarray, np.ndarray] | None:
    reward_values = np.asarray(reward_values, dtype=float)
    if reward_values.ndim != 1 or reward_values.size == 0:
        return None
    x_knots = np.linspace(0.1, 1.0, reward_values.size, dtype=float)
    valid_mask = np.isfinite(reward_values)
    if np.sum(valid_mask) < 2:
        return None
    x_valid = x_knots[valid_mask]
    y_valid = reward_values[valid_mask]
    x_dense = np.linspace(float(x_valid[0]), float(x_valid[-1]), n_points, dtype=float)
    y_dense = np.interp(x_dense, x_valid, y_valid)
    return x_dense, y_dense


def _nanquantile_or_nan(values: pd.Series | np.ndarray, q: float) -> float:
    arr = pd.to_numeric(values, errors="coerce")
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanquantile(arr, q))


def _row_learning_proxy_frame(df: pd.DataFrame) -> pd.DataFrame:
    from .common import _compute_steps90_ratio_from_row

    reward_mat = _reward_curve_matrix(df)
    early = np.full(len(df), np.nan, dtype=float)
    final = np.full(len(df), np.nan, dtype=float)
    best = np.full(len(df), np.nan, dtype=float)
    gain = np.full(len(df), np.nan, dtype=float)
    steps90_ratio = np.array([_compute_steps90_ratio_from_row(row) for _, row in df.iterrows()], dtype=float)
    burnin = np.full(len(df), np.nan, dtype=float)
    volatility = np.full(len(df), np.nan, dtype=float)
    for idx, curve_raw in enumerate(reward_mat):
        interp = _interpolated_reward_curve(curve_raw)
        if interp is None:
            continue
        x_dense, y_dense = interp
        early[idx] = float(np.interp(0.1, x_dense, y_dense))
        final[idx] = float(y_dense[-1])
        best[idx] = float(np.nanmax(y_dense))
        gain[idx] = final[idx] - early[idx]
        gain_i = gain[idx]
        if not np.isfinite(gain_i):
            continue
        if abs(gain_i) > 1e-9:
            target = early[idx] + 0.2 * gain_i
            reached = np.where(y_dense >= target)[0]
            if reached.size:
                burnin[idx] = float(x_dense[int(reached[0])])
            volatility[idx] = float(np.sum(np.abs(np.diff(y_dense))) / max(abs(gain_i), 1e-9))

    out = df.copy()
    out["early_reward_proxy"] = early
    out["final_reward_proxy"] = final
    out["best_reward_proxy"] = best
    out["reward_gain_proxy"] = gain
    out["steps90_ratio"] = steps90_ratio
    out["burnin_ratio_proxy"] = burnin
    out["volatility_proxy"] = volatility
    return out


def _wing_reynolds_dataframe(df: pd.DataFrame, velocity_col: str = "eff_v") -> pd.DataFrame:
    desc_df = _aero_descriptor_dataframe(df)
    chord = pd.to_numeric(desc_df["wing_chord"], errors="coerce").to_numpy(dtype=float)
    ref_v = pd.to_numeric(desc_df[velocity_col], errors="coerce").to_numpy(dtype=float)
    reynolds = (AIR_DENSITY_KG_M3 * ref_v * chord) / AIR_DYNAMIC_VISCOSITY_KG_M_S
    wing_re_nom = pd.to_numeric(desc_df["wing_re_nom_a1"], errors="coerce").to_numpy(dtype=float)
    wing_re_delta = reynolds - wing_re_nom
    wing_re_ratio_pct = np.divide(
        100.0 * reynolds,
        wing_re_nom,
        out=np.full_like(reynolds, np.nan, dtype=float),
        where=np.isfinite(wing_re_nom) & (np.abs(wing_re_nom) > 1e-12),
    )

    out = desc_df.copy()
    out[f"wing_reynolds_{velocity_col}"] = reynolds
    out["wing_reynolds_minus_nom_a1"] = wing_re_delta
    out["wing_reynolds_over_nom_pct_a1"] = wing_re_ratio_pct
    return out


def compute_learning_proxy_summary_by_generation(run_dir: Path) -> pd.DataFrame:
    eval_df = pd.read_csv(_analysis_csv_path(run_dir, "generation_summary.csv"))
    proxy_df = _row_learning_proxy_frame(eval_df)
    proxy_df["generation"] = pd.to_numeric(proxy_df["generation"], errors="coerce")
    proxy_df = proxy_df[np.isfinite(proxy_df["generation"])].copy()
    grouped = proxy_df.groupby("generation", sort=True)
    summary = grouped.agg(
        early_reward_median=("early_reward_proxy", "median"),
        early_reward_q25=("early_reward_proxy", lambda s: _nanquantile_or_nan(s, 0.25)),
        early_reward_q75=("early_reward_proxy", lambda s: _nanquantile_or_nan(s, 0.75)),
        final_reward_median=("final_reward_proxy", "median"),
        final_reward_q25=("final_reward_proxy", lambda s: _nanquantile_or_nan(s, 0.25)),
        final_reward_q75=("final_reward_proxy", lambda s: _nanquantile_or_nan(s, 0.75)),
        reward_gain_median=("reward_gain_proxy", "median"),
        reward_gain_q25=("reward_gain_proxy", lambda s: _nanquantile_or_nan(s, 0.25)),
        reward_gain_q75=("reward_gain_proxy", lambda s: _nanquantile_or_nan(s, 0.75)),
        steps90_ratio_median=("steps90_ratio", "median"),
        steps90_ratio_q25=("steps90_ratio", lambda s: _nanquantile_or_nan(s, 0.25)),
        steps90_ratio_q75=("steps90_ratio", lambda s: _nanquantile_or_nan(s, 0.75)),
        burnin_ratio_median=("burnin_ratio_proxy", "median"),
        burnin_ratio_q25=("burnin_ratio_proxy", lambda s: _nanquantile_or_nan(s, 0.25)),
        burnin_ratio_q75=("burnin_ratio_proxy", lambda s: _nanquantile_or_nan(s, 0.75)),
        volatility_median=("volatility_proxy", "median"),
        volatility_q25=("volatility_proxy", lambda s: _nanquantile_or_nan(s, 0.25)),
        volatility_q75=("volatility_proxy", lambda s: _nanquantile_or_nan(s, 0.75)),
    ).reset_index()
    return summary
__all__ = [name for name in globals() if not name.startswith("__")]

