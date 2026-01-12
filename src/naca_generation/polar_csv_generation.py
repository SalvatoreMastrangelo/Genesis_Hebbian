#!/usr/bin/env python3
import csv
import numpy as np
import subprocess
import tempfile
from pathlib import Path
# Percorso eseguibile xfoil
XFOIL_CMD = "xfoil"

# Cartella in cui salviamo tutti i polari
POLAR_DIR = Path("polars")
POLAR_DIR.mkdir(exist_ok=True)

# ============================================================


# ============================================================
# Lettura del file di polare XFOIL
# ============================================================
def load_polar(polar_path: Path):
    alphas = []
    cls = []

    if not polar_path.exists():
        return None, None

    with open(polar_path, "r") as f:
        data_section = False
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.lower().startswith("alpha"):
                data_section = True
                continue

            if not data_section:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                a = float(parts[0])
                cl = float(parts[1])
            except ValueError:
                continue

            alphas.append(a)
            cls.append(cl)

    if len(alphas) == 0:
        return None, None

    alphas = np.array(alphas)
    cls = np.array(cls)

    idx = np.argsort(alphas)
    return alphas[idx], cls[idx]


def load_polar_full(polar_path: Path):
    """
    Legge il polar XFOIL in modo completo.
    Ritorna un dict di numpy array con chiavi:
      alpha, cl, cd, cdp, cm, top_xtr, bot_xtr
    Se una colonna non esiste nel file, viene riempita con NaN.
    """
    if not polar_path.exists():
        return None

    rows = []
    header_cols = None
    data_section = False

    with open(polar_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.lower().startswith("alpha"):
                header_cols = s.lower().split()
                data_section = True
                continue

            if not data_section:
                continue

            parts = s.split()
            if header_cols is None or len(parts) < 2:
                continue

            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue

            rows.append(vals)

    if not rows or header_cols is None:
        return None

    col_to_idx = {name: i for i, name in enumerate(header_cols)}

    def get_col(name: str):
        if name not in col_to_idx:
            return np.full(len(rows), np.nan, dtype=float)
        j = col_to_idx[name]
        out = np.full(len(rows), np.nan, dtype=float)
        for i, r in enumerate(rows):
            if j < len(r):
                out[i] = r[j]
        return out

    alpha = get_col("alpha")
    cl    = get_col("cl")
    cd    = get_col("cd")
    cdp   = get_col("cdp")
    cm    = get_col("cm")

    top_xtr = get_col("top_xtr") if "top_xtr" in col_to_idx else get_col("top_xtr.")
    bot_xtr = get_col("bot_xtr") if "bot_xtr" in col_to_idx else get_col("bot_xtr.")

    idx = np.argsort(alpha)
    data = {
        "alpha": alpha[idx],
        "cl": cl[idx],
        "cd": cd[idx],
        "cdp": cdp[idx],
        "cm": cm[idx],
        "top_xtr": top_xtr[idx],
        "bot_xtr": bot_xtr[idx],
    }
    return data


# ============================================================
# Chiamata XFOIL
# ============================================================
def run_xfoil_aseq(
    naca: str,
    Re: int,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    iter_limit: int = 1000,
    timeout_sec: int = 3,
    sweep_tag: str = "coarse",
):
    for attempt in range(1, 3 + 1):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            polar_file = tmpdir / "polar.dat"
            cmd_file = tmpdir / "commands.inp"

            script = f"""NACA {naca}
PANE
OPER
VISC {Re}
N 7
ITER {iter_limit}
PACC
{polar_file}

ASEQ {alpha_start} {alpha_end} {alpha_step}
PACC
QUIT
"""
            cmd_file.write_text(script)

            try:
                subprocess.run(
                    [XFOIL_CMD],
                    stdin=open(cmd_file, "r"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=tmpdir,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                print(f"[TIMEOUT] XFOIL timeout for {naca}, Re={Re}, attempt {attempt}")
                continue

            if not polar_file.exists():
                print(f"[WARN] No polar file produced for {naca}, Re={Re}, attempt {attempt}")
                continue

            dst_name = f"polar_{naca}_Re{Re}_{sweep_tag}_a{alpha_start}-{alpha_end}_s{alpha_step}_try{attempt}.dat"
            dst_path = POLAR_DIR / dst_name
            polar_file.replace(dst_path)

            alphas, cls = load_polar(dst_path)
            if alphas is None:
                print(f"[WARN] polar file empty for {naca}, Re={Re}, attempt {attempt}")
                continue

            full = load_polar_full(dst_path)
            return alphas, cls, full, dst_path

    print(f"[ERROR] XFOIL failed for {naca}, Re={Re} after {3} attempts.")
    return None, None, None, None


# ============================================================
# Sweep: coarse
# ============================================================
def coarse_polars_for_re(naca: str, Re: int):
    all_points = []
    print(f"[INFO] Stage 1 coarse sweep for {naca}, Re={Re}")
    alpha_c, cl_c, full_c, _ = run_xfoil_aseq(
        naca=naca,
        Re=Re,
        alpha_start=-5.0,
        alpha_end=25.0,
        alpha_step=0.25,
        iter_limit=500,
        timeout_sec=10,
        sweep_tag="coarse",
    )

    if alpha_c is None:
        print(f"[ERROR] No coarse polar for {naca}, Re={Re}")
        return None, None, []

    if full_c is not None:
        for i in range(len(full_c["alpha"])):
            all_points.append({
                "NACA": naca,
                "Re": Re,
                "sweep": "coarse",
                "alpha": full_c["alpha"][i],
                "cl": full_c["cl"][i],
                "cd": full_c["cd"][i],
                "cdp": full_c["cdp"][i],
                "cm": full_c["cm"][i],
                "top_xtr": full_c["top_xtr"][i],
                "bot_xtr": full_c["bot_xtr"][i],
            })

    return alpha_c, cl_c, all_points


# ============================================================
# MAIN
# ============================================================
def main():
    reynolds_list = [30_000, 40_000] + list(range(50_000, 150_000 + 1, 10_000)) + list(
        range(175_000, 300_000 + 1, 25_000)
    )
    output_csv_long = "naca4_full_polars.csv"

    with open(output_csv_long, "w", newline="") as longfile:
        long_writer = csv.DictWriter(
            longfile,
            fieldnames=["NACA", "Re", "sweep", "alpha", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr"]
        )
        long_writer.writeheader()

        for m in range(0, 5, 1):
            for p in range(2, 6, 1):
                for t in range(8, 23, 1):
                    naca = f"{m}{p}{t:02d}"
                    print("\n====================================")
                    print(f"[INFO] Processing {naca}")
                    print("====================================")

                    for Re in reynolds_list:
                        _, _, points = coarse_polars_for_re(naca, Re)

                        for row in points:
                            long_writer.writerow(row)

    print(f"[DONE] Full polars written to {output_csv_long}")


if __name__ == "__main__":
    main()
