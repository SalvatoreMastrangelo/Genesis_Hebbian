#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def load_apc_dat(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float, float]] = []
    rpm: float | None = None
    for line in path.read_text(errors="ignore").splitlines():
        match = re.search(r"PROP RPM\s*=\s*(\d+)", line)
        if match:
            rpm = float(match.group(1))
            continue

        text = line.strip()
        if not text or rpm is None:
            continue

        parts = text.split()
        if len(parts) < 15:
            continue

        try:
            _v = float(parts[0])
            j = float(parts[1])
            _pe = float(parts[2])
            ct = float(parts[3])
            cp = float(parts[4])
        except ValueError:
            continue

        rows.append((rpm, j, ct, cp))

    if not rows:
        raise ValueError(f"No APC data rows parsed from {path}.")

    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def quadratic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    a2, a1, a0 = np.polyfit(x, y, 2)
    return float(a0), float(a1), float(a2)


def normalized_ct_from_poly(a0: float, a1: float, a2: float) -> tuple[float, float, float]:
    return a0, -a1 / a0, -a2 / a0


def fit_ct_shape_with_fixed_ct0(x: np.ndarray, y: np.ndarray, ct0_fixed: float) -> tuple[float, float]:
    # Fit y ≈ ct0_fixed * (1 - ct1*x - ct2*x^2) with ct1 >= 0 and ct2 >= 0.
    # A simple grid search on ct1 plus closed-form LS for ct2 is stable and
    # makes the solver-form constraints explicit.
    best: tuple[float, float, float] | None = None
    z = -ct0_fixed * (x**2)
    denom = float(np.dot(z, z))
    if denom < 1e-12:
        return 0.0, 0.0

    for ct1 in np.linspace(0.0, 2.0, 2001):
        rhs = y - ct0_fixed * (1.0 - ct1 * x)
        ct2 = float(np.dot(z, rhs) / denom)
        if ct2 < 0.0:
            ct2 = 0.0
        yhat = ct0_fixed * (1.0 - ct1 * x - ct2 * x * x)
        mse = float(np.mean((yhat - y) ** 2))
        if best is None or mse < best[0]:
            best = (mse, float(ct1), ct2)

    assert best is not None
    _, ct1, ct2 = best
    return ct1, ct2


def monotone_cp_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    best: tuple[float, float, float, float] | None = None
    c0_min = max(0.001, float(y.min()))
    c0_max = float(y.max()) * 1.1
    for c0 in np.linspace(c0_min, c0_max, 400):
        for c1 in np.linspace(0.0, 1.0, 400):
            z = -c0 * (x**2)
            rhs = y - c0 * (1.0 - c1 * x)
            denom = float(np.dot(z, z))
            if denom < 1e-12:
                continue
            c2 = float(np.dot(z, rhs) / denom)
            if c2 < 0.0:
                c2 = 0.0
            yhat = c0 * (1.0 - c1 * x - c2 * x * x)
            err = float(np.mean((yhat - y) ** 2))
            if best is None or err < best[0]:
                best = (err, c0, c1, c2)
    assert best is not None
    _, c0, c1, c2 = best
    return c0, c1, c2


def load_current_csv_coeffs(path: Path) -> dict[str, float]:
    with path.open(newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["name"] == "morphing_prop")
    return {
        "prop_ct0": float(row["prop_ct0"]),
        "prop_ct1": float(row["prop_ct1"]),
        "prop_ct2": float(row["prop_ct2"]),
        "prop_cp0": float(row["prop_cp0"]),
        "prop_cp1": float(row["prop_cp1"]),
        "prop_cp2": float(row["prop_cp2"]),
    }


def print_fit_report(label: str, rpm: np.ndarray, j: np.ndarray, ct: np.ndarray, cp: np.ndarray) -> None:
    ct_a0, ct_a1, ct_a2 = quadratic_fit(j, ct)
    cp_a0, cp_a1, cp_a2 = quadratic_fit(j, cp)
    ct0, ct1, ct2 = normalized_ct_from_poly(ct_a0, ct_a1, ct_a2)
    cp0_mono, cp1_mono, cp2_mono = monotone_cp_fit(j, cp)

    print(label)
    print(f"  rows: {len(j)}")
    print(f"  rpm range: {int(rpm.min())} .. {int(rpm.max())}")
    print(f"  J range: {j.min():.4f} .. {j.max():.4f}")
    print(f"  Ct direct: Ct(J) = {ct_a0:.6f} {ct_a1:+.6f} J {ct_a2:+.6f} J^2")
    print(f"  Ct solver free: Ct(J) = {ct0:.6f} * (1 - {ct1:.6f} J - {ct2:.6f} J^2)")
    print(f"  Cp direct: Cp(J) = {cp_a0:.6f} {cp_a1:+.6f} J {cp_a2:+.6f} J^2")
    print(
        "  Cp solver monotone: "
        f"Cp(J) = {cp0_mono:.6f} * (1 - {cp1_mono:.6f} J - {cp2_mono:.6f} J^2)"
    )
    print()


def describe_operating_point(rpm_all: np.ndarray, ct_all: np.ndarray, cp_all: np.ndarray) -> None:
    unique_rpm = np.unique(rpm_all)
    print("Ct(0) / Cp(0) near each APC RPM block")
    for rpm_value in unique_rpm:
        mask = rpm_all == rpm_value
        ct0 = ct_all[mask][0]
        cp0 = cp_all[mask][0]
        print(f"  RPM {int(rpm_value):5d}: Ct(0) = {ct0:.4f}, Cp(0) = {cp0:.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Ct/Cp from an official APC .dat file.")
    parser.add_argument(
        "--dat",
        type=Path,
        default=Path(__file__).resolve().parent / "PER3_8x4E.dat",
        help="Path to the APC .dat file.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "genesis" / "assets" / "urdf" / "mydrone" / "actuators.csv",
        help="Current actuator CSV to compare against.",
    )
    parser.add_argument("--j-max", type=float, default=0.35, help="Maximum advance ratio J for the operating fit.")
    parser.add_argument("--rpm-min", type=float, default=8000.0, help="Minimum APC RPM block for the operating fit.")
    parser.add_argument("--rpm-max", type=float, default=12000.0, help="Maximum APC RPM block for the operating fit.")
    parser.add_argument("--rho", type=float, default=1.225, help="Air density used to infer fixed ct0 from static thrust.")
    parser.add_argument("--diameter", type=float, default=0.2, help="Propeller diameter [m].")
    parser.add_argument("--static-thrust", type=float, default=5.2, help="Static thrust target [N].")
    parser.add_argument("--rpm-ref", type=float, default=10000.0, help="Reference loaded RPM used to fix ct0.")
    parser.add_argument("--ct0-fixed", type=float, default=None, help="If provided, override ct0 directly instead of deriving it from rpm_ref.")
    args = parser.parse_args()

    rpm_all, j_all, ct_all, cp_all = load_apc_dat(args.dat)
    current = load_current_csv_coeffs(args.csv)

    print(f"Loaded APC dataset: {args.dat}")
    print(f"Total parsed rows: {len(j_all)}")
    print(f"RPM blocks: {int(rpm_all.min())} .. {int(rpm_all.max())}")
    print()

    describe_operating_point(rpm_all, ct_all, cp_all)

    mask_all = j_all <= args.j_max
    print_fit_report(
        f"Global fit over all RPM blocks with J <= {args.j_max:.2f}",
        rpm_all[mask_all],
        j_all[mask_all],
        ct_all[mask_all],
        cp_all[mask_all],
    )

    mask_oper = (rpm_all >= args.rpm_min) & (rpm_all <= args.rpm_max) & (j_all <= args.j_max)
    print_fit_report(
        f"Operating-range fit with {int(args.rpm_min)} <= RPM <= {int(args.rpm_max)} and J <= {args.j_max:.2f}",
        rpm_all[mask_oper],
        j_all[mask_oper],
        ct_all[mask_oper],
        cp_all[mask_oper],
    )

    if args.ct0_fixed is not None:
        ct0_fixed = float(args.ct0_fixed)
        ct0_source = f"ct0 fixed directly = {ct0_fixed:.6f}"
    else:
        n_ref = args.rpm_ref / 60.0
        ct0_fixed = args.static_thrust / (args.rho * (n_ref**2) * (args.diameter**4))
        ct0_source = f"ct0 fixed from static thrust {args.static_thrust:.3f} N at rpm_ref = {args.rpm_ref:.1f}"
    ct1_fixed, ct2_fixed = fit_ct_shape_with_fixed_ct0(j_all[mask_oper], ct_all[mask_oper], ct0_fixed)
    cp0, cp1, cp2 = monotone_cp_fit(j_all[mask_oper], cp_all[mask_oper])

    print("Fixed-ct0 fit used to compare with the current CSV")
    print(f"  {ct0_source}")
    print(f"  prop_ct0 = {ct0_fixed:.6f}")
    print(f"  prop_ct1 = {ct1_fixed:.6f}")
    print(f"  prop_ct2 = {ct2_fixed:.6f}")
    print(f"  prop_cp0 = {cp0:.6f}")
    print(f"  prop_cp1 = {cp1:.6f}")
    print(f"  prop_cp2 = {cp2:.6f}")
    print()

    print("Rounded values")
    print(f"  prop_ct0 = {ct0_fixed:.3f}")
    print(f"  prop_ct1 = {ct1_fixed:.3f}")
    print(f"  prop_ct2 = {ct2_fixed:.3f}")
    print(f"  prop_cp0 = {cp0:.3f}")
    print(f"  prop_cp1 = {cp1:.3f}")
    print(f"  prop_cp2 = {cp2:.3f}")
    print()

    print("Current CSV values")
    for key, value in current.items():
        print(f"  {key} = {value:.6f}")
    print()

    print("Difference between rounded fixed-ct0 fit and current CSV")
    rounded = {
        "prop_ct0": round(ct0_fixed, 3),
        "prop_ct1": round(ct1_fixed, 3),
        "prop_ct2": round(ct2_fixed, 3),
        "prop_cp0": round(cp0, 3),
        "prop_cp1": round(cp1, 3),
        "prop_cp2": round(cp2, 3),
    }
    for key, value in rounded.items():
        delta = value - current[key]
        print(f"  {key}: fit={value:.3f}, csv={current[key]:.3f}, delta={delta:+.3f}")


if __name__ == "__main__":
    main()
