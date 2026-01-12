#!/usr/bin/env python3
"""
Post-processing for NACA4 polars from `naca4_full_polars.csv`.

- Reconstruct per-(airfoil, Re) polars (alpha, cl, cm)
- Compute per-Re metrics (including Cd0 and data-driven alpha_cl0) and data-quality flags
- Fit Renom using cl_max-based fRe with a=1
- Aggregate metrics for Re >= Renom
- Write long and summary CSVs
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Config (centralized constants)
# -----------------------------
DEFAULT_FULL_POLARS_CSV = "naca4_full_polars.csv"
DEFAULT_LONG_OUTPUT_CSV = "naca4_per_re_metrics_long.csv"
DEFAULT_WIDE_OUTPUT_CSV = "naca4_nominal_Re100k_with_stability.csv"
DEFAULT_NACA4_OUTPUT_CSV = "naca4.csv"

# Fixed lift-curve slope (Cl per degree). This is a modeling choice applied globally.
LIFT_CURVE_SLOPE = 0.11

RE_SWEEP: List[int] = [30_000, 40_000] + list(range(50_000, 150_000 + 1, 10_000)) + list(
    range(175_000, 300_000 + 1, 25_000)
)
RE_SWEEP_SET = set(RE_SWEEP)

# Linear window for slope
ALPHA_LINEAR_MIN = 2.0
ALPHA_LINEAR_MAX = 4.0
MIN_LINEAR_POINTS = 3
LINEAR_OUTLIER_SIGMA = 2.0
LINEAR_OUTLIER_MIN_POINTS = 8

# Zero-lift (alpha_cl0) detection
CL0_ABS_TOL = 1e-6

# Cd0 fit window (Cd = Cd0 + k*Cl^2) on small angles
CD0_ALPHA_MIN = -2.0
CD0_ALPHA_MAX = 2.0
CD0_MIN_POINTS = 4
CD0_OUTLIER_SIGMA = 2.5
CD0_OUTLIER_MIN_POINTS = 8

# Truncated polar handling for Cl_max
SATURATION_TAIL_POINTS = 8
SATURATION_PLATEAU_ABS = 0.1
SATURATION_PLATEAU_REL = 0.1

# Expected alpha sweep (heuristic for truncation flag)
ALPHA_SWEEP_MIN_EXPECTED = -5.0
ALPHA_SWEEP_MAX_EXPECTED = 25.0
ALPHA_TRUNC_EPS = 1e-9

# Hard data-quality gate
MIN_VALID_POLARS = 5

# Renom fit (a=1)
RENOM_MIN = 30_000
RENOM_MAX = 300_000
RENOM_GRID_SIZE = 1000
RENOM_MIN_POINTS = 4
RENOM_HUBER_DELTA = 0.15
HIGH_RE_TOP_K = 3
Y_CLIP_MIN = 0.0
Y_CLIP_MAX = 1.2

REQUIRED_BASE_COLUMNS = {"re", "alpha", "cl"}
AIRFOIL_COLUMN_CANDIDATES = ("airfoil", "naca")
ALLOWED_STATUS = {
    "ok",
    "xfoil_failed",
    "timeout",
    "empty_polar",
    "parse_error",
}


# ============================================================
# Input loading
# ============================================================
def normalize_airfoil(value: str) -> str:
    """
    Normalize airfoil identifiers as zero-padded strings (e.g., 12 -> 0012).
    """
    text = str(value).strip()
    if text.isdigit():
        if len(text) > 4:
            raise ValueError(f"Unexpected airfoil id length: {text}")
        return text.zfill(4)
    return text


def load_inputs(full_polars_csv: str) -> Tuple[
    Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    List[str],
    Dict[str, object],
]:
    """
    Load coarse polars from CSV, enforcing schema and status consistency.

    Returns:
      polars[(airfoil, Re)] = (alpha_array, cl_array, cd_array, cm_array) sorted by alpha
      airfoil_list = sorted list of distinct airfoil names
    """
    path = Path(full_polars_csv)
    if not path.exists():
        raise FileNotFoundError(str(path))

    tmp: Dict[Tuple[str, int], List[Tuple[float, float, float, float]]] = {}
    airfoils: set[str] = set()
    re_seen: set[int] = set()

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}")

        field_map = {name.lower(): name for name in reader.fieldnames}
        missing_cols = sorted(col for col in REQUIRED_BASE_COLUMNS if col not in field_map)
        airfoil_col = next((col for col in AIRFOIL_COLUMN_CANDIDATES if col in field_map), None)
        if missing_cols or airfoil_col is None:
            missing = missing_cols[:]
            if airfoil_col is None:
                missing.append("airfoil/naca")
            raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")

        has_status = "status" in field_map
        has_sweep = "sweep" in field_map
        has_cm = "cm" in field_map
        has_cd = "cd" in field_map

        def get_field(row: Dict[str, str], key: str, default: Optional[str] = None) -> Optional[str]:
            col = field_map.get(key)
            if col is None:
                return default
            return row.get(col, default)

        for row in reader:
            try:
                airfoil_raw = get_field(row, airfoil_col)
                airfoil = normalize_airfoil(airfoil_raw if airfoil_raw is not None else "")
                Re = int(float(get_field(row, "re", "nan")))
            except Exception as exc:
                raise ValueError(f"Failed to parse airfoil/Re in {path}: {exc}") from exc

            if Re not in RE_SWEEP_SET:
                raise ValueError(f"Unexpected Re={Re} in {path}; expected values in RE_SWEEP.")

            if not airfoil:
                raise ValueError(f"Empty airfoil id in {path} for Re={Re}")

            re_seen.add(Re)

            status = "ok"
            if has_status:
                status = (get_field(row, "status") or "").strip().lower()
                if status not in ALLOWED_STATUS:
                    raise ValueError(f"Unexpected status='{status}' for {airfoil}, Re={Re} in {path}")

            if has_sweep:
                sweep = (get_field(row, "sweep") or "").strip().lower()
                if sweep and sweep != "coarse":
                    continue

            if status != "ok":
                continue

            try:
                alpha = float(get_field(row, "alpha", "nan"))
                cl = float(get_field(row, "cl", "nan"))
            except Exception as exc:
                raise ValueError(
                    f"Invalid alpha/cl for {airfoil}, Re={Re} with status=ok in {path}: {exc}"
                ) from exc

            try:
                cd = float(get_field(row, "cd", "nan")) if has_cd else float("nan")
            except Exception:
                cd = float("nan")

            try:
                cm = float(get_field(row, "cm", "nan")) if has_cm else float("nan")
            except Exception:
                cm = float("nan")

            tmp.setdefault((airfoil, Re), []).append((alpha, cl, cd, cm))
            airfoils.add(airfoil)

    polars: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, pairs in tmp.items():
        pairs.sort(key=lambda x: x[0])
        polars[key] = (
            np.array([p[0] for p in pairs], dtype=float),
            np.array([p[1] for p in pairs], dtype=float),
            np.array([p[2] for p in pairs], dtype=float),
            np.array([p[3] for p in pairs], dtype=float),
        )

    airfoil_list = sorted(airfoils)
    meta = {
        "re_seen": re_seen,
        "has_cd": has_cd,
    }
    return polars, airfoil_list, meta


def validate_re_sweep(re_seen: set[int]) -> str:
    """
    Validate that the Re sweep matches the expected generation sweep.
    Returns a note for missing Re values; unexpected Re values raise.
    """
    extra = sorted(re_seen.difference(RE_SWEEP_SET))
    if extra:
        extra_str = ", ".join(str(x) for x in extra)
        raise ValueError(f"Unexpected Re values in input: {extra_str}")

    missing = sorted(RE_SWEEP_SET.difference(re_seen))
    if missing:
        missing_str = ", ".join(str(x) for x in missing)
        return f"re_sweep_incomplete: missing {missing_str}"
    return ""


# ============================================================
# Truncated polar handling
# ============================================================
def _has_saturation_in_tail(alpha: np.ndarray, cl: np.ndarray) -> bool:
    """
    Detect whether the tail of the polar has started to saturate.
    This is a conservative check used only when the sweep is truncated.
    """
    a = np.asarray(alpha, dtype=float)
    c = np.asarray(cl, dtype=float)
    mask = np.isfinite(a) & np.isfinite(c)
    if mask.sum() < SATURATION_TAIL_POINTS:
        return False

    idx = np.argsort(a[mask])
    a = a[mask][idx]
    c = c[mask][idx]

    cl_max = float(np.max(c))
    if not np.isfinite(cl_max):
        return False

    plateau_tol = max(SATURATION_PLATEAU_ABS, SATURATION_PLATEAU_REL * max(1.0, abs(cl_max)))
    # Saturation is accepted if any consecutive run of SATURATION_TAIL_POINTS
    # lies within a tight plateau around Cl_max, even if later points drop.
    within = np.abs(c - cl_max) <= plateau_tol
    run = 0
    for ok in within:
        if ok:
            run += 1
            if run >= SATURATION_TAIL_POINTS:
                return True
        else:
            run = 0

    return False


# ============================================================
# Linear fit: slope on [ALPHA_LINEAR_MIN, ALPHA_LINEAR_MAX]
# ============================================================
def compute_slope(alpha: np.ndarray, cl: np.ndarray) -> Tuple[float, float, bool]:
    """
    Fit Cl = m*alpha + q on a small linear window and return:
      slope (m), r2, has_linear_window.
    """
    a = np.asarray(alpha, dtype=float)
    c = np.asarray(cl, dtype=float)

    finite_mask = np.isfinite(a) & np.isfinite(c)
    a = a[finite_mask]
    c = c[finite_mask]

    window = (a >= ALPHA_LINEAR_MIN - 1e-12) & (a <= ALPHA_LINEAR_MAX + 1e-12)
    has_linear_window = bool(window.sum() >= MIN_LINEAR_POINTS)
    if not has_linear_window:
        return (np.nan, np.nan, False)

    x = a[window]
    y = c[window]

    if np.ptp(x) < 1e-12:
        return (np.nan, np.nan, True)

    def fit_line(xx: np.ndarray, yy: np.ndarray) -> Tuple[float, float, float]:
        m, q = np.polyfit(xx, yy, 1)
        yhat = m * xx + q
        ss_res = float(np.sum((yy - yhat) ** 2))
        ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0
        return float(m), float(q), float(r2)

    m, q, r2 = fit_line(x, y)

    # One-pass outlier rejection in the linear window.
    resid = y - (m * x + q)
    sigma = float(np.std(resid))
    if sigma > 1e-12 and len(x) >= LINEAR_OUTLIER_MIN_POINTS:
        keep = np.abs(resid) <= (LINEAR_OUTLIER_SIGMA * sigma)
        if keep.sum() >= MIN_LINEAR_POINTS:
            m, q, r2 = fit_line(x[keep], y[keep])

    if not np.isfinite(m) or m <= 0.0:
        return (np.nan, np.nan, True)

    return (float(m), float(r2), True)


def compute_alpha_cl0_data_driven(alpha: np.ndarray, cl: np.ndarray) -> Tuple[float, str, float, float]:
    """
    Estimate alpha_cl0 by bracketing Cl=0 using the closest points in the polar.
    Returns (alpha_cl0, method, cl_min_abs, alpha_at_cl_min_abs).
    """
    a = np.asarray(alpha, dtype=float)
    c = np.asarray(cl, dtype=float)

    finite_mask = np.isfinite(a) & np.isfinite(c)
    a = a[finite_mask]
    c = c[finite_mask]

    if a.size == 0:
        return (np.nan, "no_bracket", np.nan, np.nan)

    idx = np.argsort(a)
    a = a[idx]
    c = c[idx]

    abs_c = np.abs(c)
    i0 = int(np.argmin(abs_c))
    cl_min_abs = float(abs_c[i0])
    alpha_at_cl_min_abs = float(a[i0])

    if np.any(abs_c <= CL0_ABS_TOL):
        i_zero = int(np.argmin(abs_c))
        return (float(a[i_zero]), "exact_zero", cl_min_abs, alpha_at_cl_min_abs)

    def is_bracket(c1: float, c2: float) -> bool:
        return (c1 * c2) < 0.0

    def interp_alpha(a1: float, c1: float, a2: float, c2: float) -> float:
        denom = (c2 - c1)
        if abs(denom) < 1e-12:
            return np.nan
        return float(a1 + (0.0 - c1) * (a2 - a1) / denom)

    if i0 - 1 >= 0 and is_bracket(c[i0 - 1], c[i0]):
        alpha_cl0 = interp_alpha(a[i0 - 1], c[i0 - 1], a[i0], c[i0])
        return (alpha_cl0, "local_bracket", cl_min_abs, alpha_at_cl_min_abs)

    if i0 + 1 < a.size and is_bracket(c[i0], c[i0 + 1]):
        alpha_cl0 = interp_alpha(a[i0], c[i0], a[i0 + 1], c[i0 + 1])
        return (alpha_cl0, "local_bracket", cl_min_abs, alpha_at_cl_min_abs)

    best_i: Optional[int] = None
    best_score = np.inf
    for i in range(a.size - 1):
        if is_bracket(c[i], c[i + 1]):
            score = min(abs_c[i], abs_c[i + 1])
            if score < best_score:
                best_score = score
                best_i = i

    if best_i is not None:
        alpha_cl0 = interp_alpha(a[best_i], c[best_i], a[best_i + 1], c[best_i + 1])
        return (alpha_cl0, "global_bracket", cl_min_abs, alpha_at_cl_min_abs)

    return (np.nan, "no_bracket", cl_min_abs, alpha_at_cl_min_abs)


def compute_cd0(alpha: np.ndarray, cl: np.ndarray, cd: np.ndarray) -> Tuple[float, bool]:
    """
    Fit Cd = Cd0 + k*Cl^2 in a small alpha window and return (Cd0, has_window).
    """
    a = np.asarray(alpha, dtype=float)
    c_l = np.asarray(cl, dtype=float)
    c_d = np.asarray(cd, dtype=float)

    finite_mask = np.isfinite(a) & np.isfinite(c_l) & np.isfinite(c_d)
    a = a[finite_mask]
    c_l = c_l[finite_mask]
    c_d = c_d[finite_mask]

    window = (a >= CD0_ALPHA_MIN - 1e-12) & (a <= CD0_ALPHA_MAX + 1e-12)
    has_window = bool(window.sum() >= CD0_MIN_POINTS)
    if not has_window:
        return (np.nan, False)

    x = c_l[window] ** 2
    y = c_d[window]

    if np.ptp(x) < 1e-12:
        return (np.nan, True)

    def fit_line(xx: np.ndarray, yy: np.ndarray) -> Tuple[float, float]:
        m, q = np.polyfit(xx, yy, 1)
        return float(m), float(q)

    m, q = fit_line(x, y)

    resid = y - (m * x + q)
    sigma = float(np.std(resid))
    if sigma > 1e-12 and len(x) >= CD0_OUTLIER_MIN_POINTS:
        keep = np.abs(resid) <= (CD0_OUTLIER_SIGMA * sigma)
        if keep.sum() >= CD0_MIN_POINTS:
            _m, q = fit_line(x[keep], y[keep])

    return (float(q), True)


def compute_metrics_per_re(
    polars: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    airfoil_list: List[str],
    has_cd_column: bool,
) -> Dict[Tuple[str, int], Dict[str, object]]:
    """
    Compute per-Re metrics and data-quality flags for every (airfoil, Re).
    Derived aerodynamics metrics are based only on cl_max, cd0, and alpha_cl0.
    """
    per_re: Dict[Tuple[str, int], Dict[str, object]] = {}

    for airfoil in airfoil_list:
        for Re in RE_SWEEP:
            key = (airfoil, Re)
            if key not in polars:
                per_re[key] = {
                    "cl_max": np.nan,
                    "alpha_stall": np.nan,
                    "slope": np.nan,
                    "alpha_cl0": np.nan,
                    "alpha_cl0_method": "no_bracket",
                    "cl_min_abs": np.nan,
                    "alpha_at_cl_min_abs": np.nan,
                    "cd0": np.nan,
                    "n_points": 0,
                    "alpha_min": np.nan,
                    "alpha_max": np.nan,
                    "is_truncated": np.nan,
                    "has_linear_window": False,
                    "has_cd0_window": False,
                    "stall_detected": False,
                    "notes": "missing_polar",
                }
                continue

            alpha, cl, cd, _cm = polars[key]

            mask = np.isfinite(alpha) & np.isfinite(cl)
            n_points = int(mask.sum())
            if n_points > 0:
                alpha_min = float(np.min(alpha[mask]))
                alpha_max = float(np.max(alpha[mask]))
                is_truncated = bool(
                    (alpha_min > ALPHA_SWEEP_MIN_EXPECTED + ALPHA_TRUNC_EPS)
                    or (alpha_max < ALPHA_SWEEP_MAX_EXPECTED - ALPHA_TRUNC_EPS)
                )
            else:
                alpha_min = np.nan
                alpha_max = np.nan
                is_truncated = np.nan

            cl_max_raw = float(np.nanmax(cl[mask])) if n_points > 0 else np.nan
            has_saturation = False
            if n_points > 0 and is_truncated is True:
                # Truncated polars cannot safely provide Cl_max unless the tail
                # has already flattened. This avoids systematic Cl_max
                # underestimation and prevents corruption of Re trends.
                has_saturation = _has_saturation_in_tail(alpha[mask], cl[mask])
                cl_max = cl_max_raw if has_saturation else np.nan
            else:
                cl_max = cl_max_raw

            # Diagnostic linear fit; final slope output is fixed globally.
            slope_fit, _r2, has_linear_window = compute_slope(alpha, cl)
            alpha_cl0, alpha_cl0_method, cl_min_abs, alpha_at_cl_min_abs = compute_alpha_cl0_data_driven(
                alpha, cl
            )
            cd0, has_cd0_window = compute_cd0(alpha, cl, cd)

            # Derived aerodynamics metrics use a fixed slope and an alpha_stall
            # proxy from alpha_cl0/cl_max, with NaNs preserved when inputs are missing.
            slope = float(LIFT_CURVE_SLOPE) if n_points > 0 else np.nan
            if np.isfinite(alpha_cl0) and np.isfinite(cl_max):
                # Operational alpha_stall proxy: 95% of Cl_max under a fixed slope.
                alpha_stall = float(alpha_cl0 + (0.95 * cl_max) / LIFT_CURVE_SLOPE)
            else:
                alpha_stall = np.nan

            notes_parts: List[str] = []
            if n_points == 0:
                notes_parts.append("empty_polar")
            if is_truncated is True:
                notes_parts.append("truncated_polar")
                if not has_saturation and n_points > 0:
                    notes_parts.append("truncated_no_saturation")
            if not has_linear_window:
                notes_parts.append("no_linear_window")
            if has_linear_window and not np.isfinite(slope_fit):
                notes_parts.append("invalid_slope")
            if alpha_cl0_method == "no_bracket" and n_points > 0:
                notes_parts.append("no_cl0_bracket")
            if not has_cd_column:
                notes_parts.append("missing_cd_column")
            if not has_cd0_window and has_cd_column:
                notes_parts.append("no_cd0_window")
            if has_cd0_window and not np.isfinite(cd0):
                notes_parts.append("invalid_cd0")
            notes = ";".join(notes_parts)

            per_re[key] = {
                "cl_max": cl_max,
                "alpha_stall": alpha_stall,
                "slope": float(slope),
                "alpha_cl0": float(alpha_cl0),
                "alpha_cl0_method": alpha_cl0_method,
                "cl_min_abs": float(cl_min_abs),
                "alpha_at_cl_min_abs": float(alpha_at_cl_min_abs),
                "cd0": float(cd0),
                "n_points": n_points,
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
                "is_truncated": is_truncated,
                "has_linear_window": has_linear_window,
                "has_cd0_window": has_cd0_window,
                "stall_detected": False,
                "notes": notes,
            }

    return per_re


# ============================================================
# Renom fit (a=1)
# ============================================================
def _huber_loss(residual: np.ndarray, delta: float) -> np.ndarray:
    abs_r = np.abs(residual)
    quad = np.minimum(abs_r, delta)
    return 0.5 * quad**2 + delta * (abs_r - quad)


def _re_spacing_weights(re_vals: List[float]) -> np.ndarray:
    if not re_vals:
        return np.array([], dtype=float)
    if len(re_vals) == 1:
        return np.array([1.0], dtype=float)
    re_arr = np.array(re_vals, dtype=float)
    deltas = np.diff(re_arr)
    weights = np.empty_like(re_arr)
    weights[0] = deltas[0]
    weights[-1] = deltas[-1]
    if re_arr.size > 2:
        weights[1:-1] = 0.5 * (deltas[:-1] + deltas[1:])
    return weights


def fit_renom_a1(
    re_list: List[int],
    cl_max_by_re: Dict[int, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Fit Renom in [RENOM_MIN, RENOM_MAX] using y = cl_max/cl_max_high and
    y ~= min(Re/Renom, 1) with robust least squares (Huber).
    """
    re_vals: List[float] = []
    cl_max_vals: List[float] = []
    for Re in re_list:
        cl_max = cl_max_by_re.get(Re, np.nan)
        if np.isfinite(cl_max):
            re_vals.append(float(Re))
            cl_max_vals.append(float(cl_max))

    diag = {
        "cl_max_high": np.nan,
        "n_points": float(len(re_vals)),
        "objective": np.nan,
    }
    if len(re_vals) < RENOM_MIN_POINTS:
        return np.nan, diag

    pairs = sorted(zip(re_vals, cl_max_vals), key=lambda x: x[0])
    top_k = [v for _, v in pairs[-HIGH_RE_TOP_K:]] if pairs else []
    if not top_k:
        return np.nan, diag

    cl_max_high = float(np.median(top_k))
    if not np.isfinite(cl_max_high) or cl_max_high <= 0.0:
        return np.nan, diag

    re_arr = np.array([p[0] for p in pairs], dtype=float)
    cl_arr = np.array([p[1] for p in pairs], dtype=float)
    y = cl_arr / cl_max_high
    y = np.clip(y, Y_CLIP_MIN, Y_CLIP_MAX)
    # Weight by Re spacing to avoid bias from non-uniform grids.
    weights = _re_spacing_weights(re_arr.tolist())

    candidates = np.linspace(RENOM_MIN, RENOM_MAX, RENOM_GRID_SIZE)
    model = np.minimum(re_arr[:, None] / candidates[None, :], 1.0)
    resid = y[:, None] - model
    if weights.size != re_arr.size:
        weights = np.ones_like(re_arr)
    obj = np.sum(_huber_loss(resid, RENOM_HUBER_DELTA) * weights[:, None], axis=0)
    obj = obj / float(np.sum(weights))
    best_idx = int(np.argmin(obj))
    renom = float(candidates[best_idx])

    diag["cl_max_high"] = cl_max_high
    diag["objective"] = float(obj[best_idx])
    diag["n_points"] = float(len(re_arr))
    return renom, diag


def compute_high_metric(
    re_list: List[int],
    metric_by_re: Dict[int, float],
    top_k: int = HIGH_RE_TOP_K,
) -> float:
    pairs = [
        (float(Re), float(metric_by_re.get(Re, np.nan)))
        for Re in re_list
        if np.isfinite(metric_by_re.get(Re, np.nan))
    ]
    if not pairs:
        return np.nan
    pairs.sort(key=lambda x: x[0])
    top_vals = [v for _, v in pairs[-top_k:]]
    if not top_vals:
        return np.nan
    return float(np.median(top_vals))


# ============================================================
# Aggregation for Re >= Renom
# ============================================================
def _weighted_mean_or_nan(values: List[float], weights: List[float]) -> float:
    # NaN-handling philosophy: ignore non-finite values and return NaN if none remain.
    v = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def aggregate_ge_renom(
    airfoil: str,
    renom: float,
    per_re: Dict[Tuple[str, int], Dict[str, object]],
) -> Dict[str, float]:
    """
    Aggregate metrics for Re >= Renom. Means use finite values only.
    """
    if not np.isfinite(renom):
        return {
            "slope_mean": np.nan,
            "alpha_stall_mean": np.nan,
            "alpha_cl0_mean": np.nan,
            "n_used": np.nan,
        }

    re_ge = [Re for Re in RE_SWEEP if Re >= renom]
    slope_vals = [per_re[(airfoil, Re)]["slope"] for Re in re_ge]
    alpha_stall_vals = [per_re[(airfoil, Re)]["alpha_stall"] for Re in re_ge]
    alpha_cl0_vals = [per_re[(airfoil, Re)]["alpha_cl0"] for Re in re_ge]
    cd0_vals = [per_re[(airfoil, Re)]["cd0"] for Re in re_ge]
    # Weight by spacing so irregular Re sweeps do not bias means.
    weights = _re_spacing_weights([float(Re) for Re in re_ge]).tolist()

    n_used = int(np.sum(np.isfinite(np.array(slope_vals, dtype=float))))
    return {
        "slope_mean": _weighted_mean_or_nan(slope_vals, weights),
        "alpha_stall_mean": _weighted_mean_or_nan(alpha_stall_vals, weights),
        "alpha_cl0_mean": _weighted_mean_or_nan(alpha_cl0_vals, weights),
        "cd0_mean": _weighted_mean_or_nan(cd0_vals, weights),
        "n_used": float(n_used),
    }


# ============================================================
# Outputs
# ============================================================
def write_outputs(
    output_long_csv: str,
    output_summary_csv: str,
    airfoil_list: List[str],
    per_re: Dict[Tuple[str, int], Dict[str, object]],
    airfoil_stats: Dict[str, Dict[str, object]],
    summary_rows: Dict[str, Dict[str, object]],
) -> None:
    """
    Write long and summary CSVs. Per-Re outputs are nulled for airfoils that
    fail the MIN_VALID_POLARS gate.
    """
    long_header = [
        "airfoil",
        "Re",
        "cl_max",
        "alpha_stall",
        "slope",
        "alpha_cl0",
        "alpha_cl0_method",
        "cl_min_abs",
        "alpha_at_cl_min_abs",
        "cd0",
        "n_points",
        "alpha_min",
        "alpha_max",
        "is_truncated",
        "has_linear_window",
        "has_cd0_window",
        "stall_detected",
        "Renom_a1",
        "Renom_a1_obj",
        "Renom_a1_n_points",
        "Renom_a1_cl_max_high",
        "Renom_a1_slope_high",
        "insufficient_polars",
        "notes",
    ]

    with open(output_long_csv, "w", newline="") as f_long:
        wr = csv.writer(f_long)
        wr.writerow(long_header)
        for airfoil in airfoil_list:
            stats = airfoil_stats[airfoil]
            insufficient = bool(stats["insufficient"])
            for Re in RE_SWEEP:
                row = per_re[(airfoil, Re)].copy()
                notes_parts = [row.get("notes", "")] if row.get("notes") else []
                if insufficient:
                    notes_parts.append("insufficient_polars")
                elif stats.get("renom_note"):
                    notes_parts.append(stats["renom_note"])

                if insufficient:
                    for field in [
                        "cl_max",
                        "alpha_stall",
                        "slope",
                        "alpha_cl0",
                        "alpha_cl0_method",
                        "cl_min_abs",
                        "alpha_at_cl_min_abs",
                        "cd0",
                        "n_points",
                        "alpha_min",
                        "alpha_max",
                        "is_truncated",
                        "has_linear_window",
                        "has_cd0_window",
                        "stall_detected",
                    ]:
                        row[field] = np.nan
                    row["alpha_cl0_method"] = ""

                renom = stats["renom"] if not insufficient else np.nan
                renom_obj = stats["renom_obj"] if not insufficient else np.nan
                renom_n = stats["renom_n"] if not insufficient else np.nan
                renom_cl_max_high = stats["renom_cl_max_high"] if not insufficient else np.nan
                renom_slope_high = stats["renom_slope_high"] if not insufficient else np.nan

                wr.writerow(
                    [
                        airfoil,
                        Re,
                        row["cl_max"],
                        row["alpha_stall"],
                        row["slope"],
                        row["alpha_cl0"],
                        row["alpha_cl0_method"],
                        row["cl_min_abs"],
                        row["alpha_at_cl_min_abs"],
                        row["cd0"],
                        row["n_points"],
                        row["alpha_min"],
                        row["alpha_max"],
                        row["is_truncated"],
                        row["has_linear_window"],
                        row["has_cd0_window"],
                        row["stall_detected"],
                        renom,
                        renom_obj,
                        renom_n,
                        renom_cl_max_high,
                        renom_slope_high,
                        insufficient,
                        ";".join([p for p in notes_parts if p]),
                    ]
                )

    summary_header = [
        "airfoil",
        "Renom_a1",
        "slope_ge_Renom_mean",
        "alpha_stall_ge_Renom_mean",
        "alpha_cl0_ge_Renom_mean",
        "cd0_ge_Renom_mean",
        "n_Re_total",
        "n_Re_valid",
        "n_Re_ge_Renom_used",
        "insufficient_polars",
        "summary_ok",
        "notes",
    ]

    with open(output_summary_csv, "w", newline="") as f_summary:
        wr = csv.writer(f_summary)
        wr.writerow(summary_header)
        for airfoil in airfoil_list:
            summary = summary_rows[airfoil]
            wr.writerow(
                [
                    airfoil,
                    summary["Renom_a1"],
                    summary["slope_ge_Renom_mean"],
                    summary["alpha_stall_ge_Renom_mean"],
                    summary["alpha_cl0_ge_Renom_mean"],
                    summary["cd0_ge_Renom_mean"],
                    summary["n_Re_total"],
                    summary["n_Re_valid"],
                    summary["n_Re_ge_Renom_used"],
                    summary["insufficient_polars"],
                    summary["summary_ok"],
                    summary["notes"],
                ]
            )


def write_naca4_csv(
    output_naca4_csv: str,
    airfoil_list: List[str],
    summary_rows: Dict[str, Dict[str, object]],
) -> int:
    """
    Write `naca4.csv`: a compact, one-row-per-airfoil export of final metrics.
    The file is intentionally minimal (no per-Re rows or extra columns) for
    downstream consumers that only need the aggregated values.
    A "valid airfoil" has a non-empty id and finite ReNom, slope, alpha_stall,
    alpha0, and cd0; rows that fail any requirement are dropped.
    """
    header = ["airfoil", "ReNom", "slope", "alpha_stall", "alpha0", "cd0"]
    valid_rows: List[List[object]] = []
    for airfoil in sorted(airfoil_list):
        if not airfoil:
            continue
        summary = summary_rows[airfoil]
        renom = summary.get("Renom_a1", np.nan)
        slope = summary.get("slope_ge_Renom_mean", np.nan)
        alpha_stall = summary.get("alpha_stall_ge_Renom_mean", np.nan)
        alpha0 = summary.get("alpha_cl0_ge_Renom_mean", np.nan)
        cd0 = summary.get("cd0_ge_Renom_mean", np.nan)
        if not all(np.isfinite([renom, slope, alpha_stall, alpha0, cd0])):
            continue
        valid_rows.append([airfoil, renom, slope, alpha_stall, alpha0, cd0])

    with open(output_naca4_csv, "w", newline="") as f_out:
        wr = csv.writer(f_out)
        wr.writerow(header)
        wr.writerows(valid_rows)

    return len(valid_rows)


# ============================================================
# Main
# ============================================================
def main() -> None:
    input_csv = DEFAULT_FULL_POLARS_CSV
    output_long_csv = DEFAULT_LONG_OUTPUT_CSV
    output_summary_csv = DEFAULT_WIDE_OUTPUT_CSV
    output_naca4_csv = str(Path(output_summary_csv).with_name(DEFAULT_NACA4_OUTPUT_CSV))

    polars, airfoil_list, meta = load_inputs(input_csv)
    re_sweep_note = validate_re_sweep(meta.get("re_seen", set()))
    if re_sweep_note:
        print(f"[WARN] {re_sweep_note}")

    per_re = compute_metrics_per_re(polars, airfoil_list, bool(meta.get("has_cd", False)))

    airfoil_stats: Dict[str, Dict[str, object]] = {}
    summary_rows: Dict[str, Dict[str, object]] = {}

    for airfoil in airfoil_list:
        cl_max_by_re = {Re: per_re[(airfoil, Re)]["cl_max"] for Re in RE_SWEEP}
        # NaN-safe: only finite Cl_max values are eligible for the Renom fit.
        cl_max_by_re_finite = {
            Re: val for Re, val in cl_max_by_re.items() if np.isfinite(float(val))
        }
        slope_by_re = {Re: per_re[(airfoil, Re)]["slope"] for Re in RE_SWEEP}
        n_valid = int(len(cl_max_by_re_finite))
        insufficient = n_valid < MIN_VALID_POLARS

        renom_note = ""
        if insufficient:
            renom = np.nan
            diag = {"cl_max_high": np.nan, "n_points": np.nan, "objective": np.nan}
        else:
            renom, diag = fit_renom_a1(RE_SWEEP, cl_max_by_re_finite)
            if not np.isfinite(renom):
                renom_note = "renom_fit_failed"
        slope_high = compute_high_metric(RE_SWEEP, slope_by_re)

        airfoil_stats[airfoil] = {
            "n_total": len(RE_SWEEP),
            "n_valid": n_valid,
            "insufficient": insufficient,
            "renom": renom,
            "renom_obj": diag["objective"],
            "renom_n": diag["n_points"],
            "renom_cl_max_high": diag["cl_max_high"],
            "renom_slope_high": slope_high,
            "renom_note": renom_note,
        }

    for airfoil in airfoil_list:
        stats = airfoil_stats[airfoil]
        insufficient = bool(stats["insufficient"])
        notes_parts: List[str] = []
        if re_sweep_note:
            notes_parts.append(re_sweep_note)

        if insufficient:
            insuff_notes = ["insufficient_polars"]
            insuff_notes.extend(notes_parts)
            summary = {
                "Renom_a1": np.nan,
                "slope_ge_Renom_mean": np.nan,
                "alpha_stall_ge_Renom_mean": np.nan,
                "alpha_cl0_ge_Renom_mean": np.nan,
                "cd0_ge_Renom_mean": np.nan,
                "n_Re_total": np.nan,
                "n_Re_valid": np.nan,
                "n_Re_ge_Renom_used": np.nan,
                "insufficient_polars": True,
                "summary_ok": False,
                "notes": ";".join([p for p in insuff_notes if p]),
            }
            summary_rows[airfoil] = summary
            continue

        renom = stats["renom"]
        agg = aggregate_ge_renom(airfoil, float(renom), per_re)

        if not np.isfinite(renom):
            notes_parts.append("renom_fit_failed")

        if np.isfinite(agg["n_used"]) and agg["n_used"] == 0:
            notes_parts.append("no_ge_renom_points")

        summary_ok = (
            np.isfinite(renom)
            and np.isfinite(agg["slope_mean"])
            and np.isfinite(agg["alpha_stall_mean"])
            and np.isfinite(agg["alpha_cl0_mean"])
            and np.isfinite(agg["n_used"])
            and agg["n_used"] > 0
        )

        summary = {
            "Renom_a1": renom if np.isfinite(renom) else np.nan,
            "slope_ge_Renom_mean": agg["slope_mean"] if np.isfinite(renom) else np.nan,
            "alpha_stall_ge_Renom_mean": agg["alpha_stall_mean"] if np.isfinite(renom) else np.nan,
            "alpha_cl0_ge_Renom_mean": agg["alpha_cl0_mean"] if np.isfinite(renom) else np.nan,
            "cd0_ge_Renom_mean": agg["cd0_mean"] if np.isfinite(renom) else np.nan,
            "n_Re_total": float(stats["n_total"]),
            "n_Re_valid": float(stats["n_valid"]),
            "n_Re_ge_Renom_used": agg["n_used"] if np.isfinite(renom) else np.nan,
            "insufficient_polars": False,
            "summary_ok": bool(summary_ok),
            "notes": ";".join(notes_parts),
        }
        summary_rows[airfoil] = summary

    write_outputs(output_long_csv, output_summary_csv, airfoil_list, per_re, airfoil_stats, summary_rows)
    n_valid = write_naca4_csv(output_naca4_csv, airfoil_list, summary_rows)

    print(f"[DONE] Wrote: {output_long_csv}")
    print(f"[DONE] Wrote: {output_summary_csv}")
    print(f"[DONE] Wrote: {output_naca4_csv} with {n_valid} valid airfoils")


if __name__ == "__main__":
    main()
