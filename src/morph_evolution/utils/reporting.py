from __future__ import annotations

import datetime
import os
import random
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from filelock import FileLock

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _safe_row_nanmax(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    mask = np.isfinite(arr).any(axis=1)
    if np.any(mask):
        out[mask] = np.nanmax(arr[mask], axis=1)
    return out


def ckpt_idx_from_train_it(train_it: Any) -> int:
    """Map a training-iteration count to the saved checkpoint suffix."""
    train_it_i = int(train_it)
    return int(train_it_i - 1) if train_it_i > 0 else -1


def parse_fail_reason(reason: str) -> Tuple[str, str]:
    """
    Split a free-form failure reason into:
      - fail_category: stable machine-friendly label
      - fail_detail: original message
    """
    txt = str(reason or "").strip()
    if not txt:
        return "unknown", ""
    head, _, _rest = txt.partition(":")
    category = head.strip().replace(" ", "_").lower()
    if not category:
        category = "unknown"
    return category, txt


class PostAnalyzer:
    """
    Helper to analyze the CSV produced by the GA and generate plots.

    It can:
      - Clean invalid sentinel values.
      - Plot best fronts per generation (velocity, energy, progress).
      - Plot final reward vs generation and learning speed.
    """

    def __init__(
        self,
        csv_path: str,
        stats_obj: Optional["Stats"] = None,
        pkl_path: Optional[str] = None,
        invalid_v: Optional[set[float]] = None,
        invalid_e: Optional[set[float]] = None,
        invalid_p: Optional[set[float]] = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.invalid_v = {0.0} if invalid_v is None else set(invalid_v)
        self.invalid_e = {-10.0} if invalid_e is None else set(invalid_e)
        self.invalid_p = {0.0} if invalid_p is None else set(invalid_p)

        if "row_kind" in self.df.columns:
            row_kind = self.df["row_kind"].fillna("agg")
            self.df = self.df[row_kind != "rep"].copy()

        for col in ("vel_v", "eff_v", "prog_v"):
            self.df.loc[self.df[col].isin(self.invalid_v), col] = np.nan
        for col in ("vel_E", "eff_E", "prog_E"):
            self.df.loc[self.df[col].isin(self.invalid_e), col] = np.nan
        for col in ("vel_P", "eff_P", "prog_P"):
            self.df.loc[self.df[col].isin(self.invalid_p), col] = np.nan

        self.stats = stats_obj
        if self.stats is None and pkl_path and Path(pkl_path).is_file():
            import pickle

            with open(pkl_path, "rb") as f:
                self.stats = pickle.load(f)

        self.vel = self.df[["vel_v", "vel_E", "vel_P"]].rename(
            columns={"vel_v": "v", "vel_E": "E", "vel_P": "P"}
        )
        self.eff = self.df[["eff_v", "eff_E", "eff_P"]].rename(
            columns={"eff_v": "v", "eff_E": "E", "eff_P": "P"}
        )
        self.prog = self.df[["prog_v", "prog_E", "prog_P"]].rename(
            columns={"prog_v": "v", "prog_E": "E", "prog_P": "P"}
        )

    def final_reward_steps(self, out: str = "final_reward_steps.png") -> None:
        if "final_reward" not in self.df.columns:
            print("final_reward missing")
            return

        df_valid = self.df.dropna(subset=["final_reward", "steps90_pct"])
        if df_valid.empty:
            print("No valid reward data – skipping plot.")
            return

        idx = self.df.groupby("generation")["final_reward"].idxmax()
        gens = self.df.loc[idx, "generation"].to_numpy(dtype=float)
        final_vals = self.df.loc[idx, "final_reward"].to_numpy(dtype=float)
        steps_vals = self.df.loc[idx, "steps90_pct"].to_numpy(dtype=float)

        order = np.argsort(gens)
        gens = gens[order]
        final_vals = final_vals[order]
        steps_vals = steps_vals[order]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(gens, final_vals, label="Final reward")
        ax2.plot(gens, steps_vals, label="Steps to 90% (pct)")

        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Final reward (avg last 5%)")
        ax2.set_ylabel("Steps to 90% final reward [%]")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.title("Evolution of final reward and learning speed")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ final reward / learning speed plot →", out)

    def fronts_progress_V(self, out: str = "best_vel_per_gen.png") -> None:
        if self.stats is None:
            print("No stats – skipping velocity progress plot.")
            return

        V = np.where(np.isin(self.stats.V, list(self.invalid_v)), np.nan, self.stats.V)
        if not np.isfinite(V).any():
            print("No valid velocity stats – skipping velocity progress plot.")
            return
        plt.plot(_safe_row_nanmax(V), label="velocity ↑")
        plt.xlabel("Generation")
        plt.ylabel("Best value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ velocity progress plot →", out)

    def fronts_progress_P(self, out: str = "best_prog_per_gen.png") -> None:
        if self.stats is None:
            print("No stats – skipping progress plot.")
            return

        M = np.where(np.isin(self.stats.M, list(self.invalid_p)), np.nan, self.stats.M)
        if not np.isfinite(M).any():
            print("No valid progress stats – skipping progress plot.")
            return
        best_prog = _safe_row_nanmax(M)

        plt.figure()
        plt.plot(best_prog, label="best progress ↑")

        if "minimal_p" in self.df.columns:
            min_p_ser = self.df.groupby("generation")["minimal_p"].first()
            min_p_curve = min_p_ser.reindex(range(len(best_prog)))
            plt.plot(min_p_curve, "--", label="minimal_p threshold")

        plt.xlabel("Generation")
        plt.ylabel("Meters")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ progress plot →", out)

    def fronts_progress_E(self, out: str = "best_eff_per_gen.png") -> None:
        if self.stats is None:
            print("No stats – skipping energy plot.")
            return

        E = np.where(np.isin(self.stats.E, list(self.invalid_e)), np.nan, self.stats.E)
        if not np.isfinite(E).any():
            print("No valid energy stats – skipping energy progress plot.")
            return
        plt.plot(_safe_row_nanmax(E), label="-energy ↑")
        plt.xlabel("Generation")
        plt.ylabel("Best value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ energy progress plot →", out)

    def analyze(self, prefix: str = "analysis") -> None:
        self.fronts_progress_V(f"{prefix}_progress_vel.png")
        self.fronts_progress_P(f"{prefix}_progress_prog.png")
        self.fronts_progress_E(f"{prefix}_progress_eff.png")
        self.final_reward_steps(f"{prefix}_reward_speed.png")


class FitnessDB:
    """
    Simple CSV-backed cache mapping chromosome → fitness + metadata.

    This avoids retraining individuals that have already been evaluated.
    """

    def __init__(self, name: str, n_obj: int, root: Path) -> None:
        self.n_obj = n_obj
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{name}.csv"
        self.df = pd.read_csv(self.path) if self.path.exists() else self._blank()
        if self.path.exists():
            if self._ensure_columns():
                lock = FileLock(str(self.path) + ".lock")
                with lock:
                    self.df.to_csv(self.path, index=False)
        else:
            self.df.to_csv(self.path, index=False)

    def lookup_fitness(self, chromo: Sequence[float]) -> Optional[List[float]]:
        df = self._filter_agg_rows(self.df)
        row = df[df.chromosome == str(list(chromo))]
        if row.empty:
            return None
        return [row[f"ff_{i}"].min() for i in range(self.n_obj)]

    def insert(self, chromo: Sequence[float], ff: Sequence[float], meta: Dict[str, Any]) -> None:
        print(f"   ↘ writing CSV  gen={meta.get('generation')}  ff={ff}")
        row_dict: Dict[str, Any] = {c: np.nan for c in self.df.columns}

        row_dict["timestamp"] = time.time()
        row_dict["chromosome"] = str(list(chromo))
        for i, val in enumerate(ff):
            row_dict[f"ff_{i}"] = val
        for k, v in meta.items():
            if k in row_dict:
                row_dict[k] = v
            else:
                print(f"Warning: meta key {k} not in columns")

        row_df = pd.DataFrame([row_dict])

        lock = FileLock(str(self.path) + ".lock")
        with lock:
            if self.df.empty:
                self.df = row_df.copy()
            else:
                self.df = pd.concat([self.df, row_df], ignore_index=True)
            row_df.to_csv(self.path, mode="a", header=False, index=False)

    def _blank(self) -> pd.DataFrame:
        cols = (
            ["timestamp", "chromosome"]
            + [f"ff_{i}" for i in range(self.n_obj)]
            + [
                "generation",
                "uid",
                "parent_idx_a",
                "parent_idx_b",
                "parent_uid_a",
                "parent_uid_b",
                "parent_gen_a",
                "parent_gen_b",
                "row_kind",
                "rep_idx",
                "exp_name",
                "rep_exp_names",
                "train_it",
                "ckpt_idx",
                "train_repetition",
                "max_p",
                "minimal_p",
                "vel_v",
                "vel_E",
                "vel_P",
                "eff_v",
                "eff_E",
                "eff_P",
                "prog_v",
                "prog_E",
                "prog_P",
                "failed",
                "fail_category",
                "fail_reason",
                "rew_10pct",
                "rew_20pct",
                "rew_30pct",
                "rew_40pct",
                "rew_50pct",
                "rew_60pct",
                "rew_70pct",
                "rew_80pct",
                "rew_90pct",
                "rew_100pct",
                "final_reward",
                "steps90_pct",
                "eval_reward_mean",
            ]
        )
        return pd.DataFrame(columns=cols)

    @staticmethod
    def _filter_agg_rows(df: pd.DataFrame) -> pd.DataFrame:
        if "row_kind" not in df.columns:
            return df
        row_kind = df["row_kind"].fillna("agg")
        return df[row_kind != "rep"]

    def _ensure_columns(self) -> bool:
        expected = list(self._blank().columns)
        extra = [c for c in self.df.columns if c not in expected]
        missing = [c for c in expected if c not in self.df.columns]
        changed = False
        for col in missing:
            self.df[col] = np.nan
            changed = True
        new_cols = expected + extra
        if list(self.df.columns) != new_cols:
            self.df = self.df[new_cols]
            changed = True
        return changed

    def get_row(self, chromo: Sequence[float]) -> Optional[pd.Series]:
        df = self._filter_agg_rows(self.df)
        row = df[df.chromosome == str(list(chromo))]
        return None if row.empty else row.iloc[0]


class Stats:
    """Container for per-generation distributions of the three objectives."""

    def __init__(self, n_pop: int, n_gen: int, n_obj: int) -> None:
        self.arr = np.zeros((n_obj, n_gen + 1, n_pop))
        self.arr.fill(np.nan)

    def record(
        self,
        gen: int,
        pop: Sequence[Any],
        invalid_v: set[float],
        invalid_e: set[float],
        invalid_p: set[float],
    ) -> None:
        invalid_sets = (invalid_v, invalid_e, invalid_p)
        for j in range(self.arr.shape[0]):
            vals = [ind.fitness.values[j] for ind in pop]
            vals = [np.nan if v in invalid_sets[j] else v for v in vals]
            self.arr[j, gen] = vals

    @property
    def V(self) -> np.ndarray:
        return self.arr[0]

    @property
    def E(self) -> np.ndarray:
        return self.arr[1]

    @property
    def M(self) -> np.ndarray:
        return self.arr[2]


def init_report_csvs(
    population_history_path: Path,
    pareto_history_path: Path,
    generation_summary_path: Path,
) -> None:
    if not population_history_path.exists():
        pd.DataFrame(
            columns=[
                "generation",
                "uid",
                "parent_uid_a",
                "parent_uid_b",
                "parent_gen_a",
                "parent_gen_b",
                "pareto_rank",
                "is_pareto",
                "chromosome",
                "ff_0",
                "ff_1",
                "ff_2",
                "max_p",
                "exp_name",
                "train_it",
                "ckpt_idx",
                "failed",
                "fail_category",
                "fail_reason",
            ]
        ).to_csv(population_history_path, index=False)

    if not pareto_history_path.exists():
        pd.DataFrame(
            columns=[
                "generation",
                "uid",
                "parent_uid_a",
                "parent_uid_b",
                "parent_gen_a",
                "parent_gen_b",
                "chromosome",
                "ff_0",
                "ff_1",
                "ff_2",
                "max_p",
                "exp_name",
                "train_it",
                "ckpt_idx",
                "failed",
                "fail_category",
                "fail_reason",
            ]
        ).to_csv(pareto_history_path, index=False)

    if not generation_summary_path.exists():
        pd.DataFrame(
            columns=[
                "generation",
                "population_size",
                "valid_count",
                "invalid_count",
                "failure_count",
                "failure_rate",
                "pareto_size",
                "minimal_p",
                "best_vel",
                "best_eff",
                "best_prog",
                "median_vel",
                "median_eff",
                "median_prog",
                "q1_vel",
                "q1_eff",
                "q1_prog",
                "q3_vel",
                "q3_eff",
                "q3_prog",
            ]
        ).to_csv(generation_summary_path, index=False)


def write_run_manifest(
    run_manifest_path: Path,
    run_name: str,
    base_dir: Path,
    analysis_dir: Path,
    logs_dir: Path,
    urdf_dir: Path,
    cfg_dict: Dict[str, Any],
    use_parallel: bool,
) -> None:
    now = datetime.datetime.now().isoformat()
    try:
        py_state0 = int(random.getstate()[1][0])
    except Exception:
        py_state0 = -1
    try:
        np_state0 = int(np.random.get_state()[1][0])
    except Exception:
        np_state0 = -1
    try:
        torch_seed = int(torch.initial_seed())
    except Exception:
        torch_seed = -1

    row = {
        "created_at": now,
        "run_name": run_name,
        "base_dir": str(base_dir),
        "analysis_dir": str(analysis_dir),
        "logs_dir": str(logs_dir),
        "urdf_dir": str(urdf_dir),
        "ga_parallel": int(bool(use_parallel)),
        "ga_parallel_env": os.getenv("GA_PARALLEL", "auto"),
        "python_random_state0": py_state0,
        "numpy_random_state0": np_state0,
        "torch_initial_seed": torch_seed,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_available": int(bool(torch.cuda.is_available())),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
    }
    row.update({f"cfg_{k}": v for k, v in cfg_dict.items()})
    pd.DataFrame([row]).to_csv(run_manifest_path, index=False)


def invalid_objective_masks(
    pop: Sequence[Any],
    invalid_v: set[float],
    invalid_e: set[float],
    invalid_p: set[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray([list(ind.fitness.values) for ind in pop], dtype=float)
    v = arr[:, 0].copy()
    e = arr[:, 1].copy()
    p = arr[:, 2].copy()
    v[np.isin(v, list(invalid_v))] = np.nan
    e[np.isin(e, list(invalid_e))] = np.nan
    p[np.isin(p, list(invalid_p))] = np.nan
    return v, e, p


def append_population_history(
    population_history_path: Path,
    generation: int,
    pop: Sequence[Any],
    fronts: Sequence[Sequence[Any]],
) -> None:
    rank_map: Dict[int, int] = {}
    for ridx, front in enumerate(fronts):
        for ind in front:
            rank_map[id(ind)] = ridx

    rows: List[Dict[str, Any]] = []
    for ind in pop:
        ff = list(ind.fitness.values)
        train_it_raw = getattr(ind, "train_it", np.nan)
        try:
            ckpt_idx = ckpt_idx_from_train_it(train_it_raw)
        except Exception:
            ckpt_idx = np.nan
        fail_reason = getattr(ind, "fail_reason", "")
        fail_category = getattr(ind, "fail_category", "")
        if not fail_category and fail_reason:
            fail_category, fail_reason = parse_fail_reason(fail_reason)
        rows.append(
            dict(
                generation=generation,
                uid=getattr(ind, "uid", -1),
                parent_uid_a=getattr(ind, "parent_uid_a", -1),
                parent_uid_b=getattr(ind, "parent_uid_b", -1),
                parent_gen_a=getattr(ind, "parent_gen_a", -1),
                parent_gen_b=getattr(ind, "parent_gen_b", -1),
                pareto_rank=rank_map.get(id(ind), -1),
                is_pareto=int(rank_map.get(id(ind), -1) == 0),
                chromosome=str(list(ind)),
                ff_0=ff[0],
                ff_1=ff[1],
                ff_2=ff[2],
                max_p=getattr(ind, "max_p", np.nan),
                exp_name=getattr(ind, "exp_name", ""),
                train_it=train_it_raw,
                ckpt_idx=ckpt_idx,
                failed=int(bool(getattr(ind, "_failed", False) or getattr(ind, "failed", False))),
                fail_category=fail_category,
                fail_reason=fail_reason,
            )
        )

    pd.DataFrame(rows).to_csv(
        population_history_path,
        mode="a",
        header=False,
        index=False,
    )


def append_pareto_history(
    pareto_history_path: Path,
    generation: int,
    front0: Sequence[Any],
) -> None:
    rows: List[Dict[str, Any]] = []
    for ind in front0:
        ff = list(ind.fitness.values)
        train_it_raw = getattr(ind, "train_it", np.nan)
        try:
            ckpt_idx = ckpt_idx_from_train_it(train_it_raw)
        except Exception:
            ckpt_idx = np.nan
        fail_reason = getattr(ind, "fail_reason", "")
        fail_category = getattr(ind, "fail_category", "")
        if not fail_category and fail_reason:
            fail_category, fail_reason = parse_fail_reason(fail_reason)
        rows.append(
            dict(
                generation=generation,
                uid=getattr(ind, "uid", -1),
                parent_uid_a=getattr(ind, "parent_uid_a", -1),
                parent_uid_b=getattr(ind, "parent_uid_b", -1),
                parent_gen_a=getattr(ind, "parent_gen_a", -1),
                parent_gen_b=getattr(ind, "parent_gen_b", -1),
                chromosome=str(list(ind)),
                ff_0=ff[0],
                ff_1=ff[1],
                ff_2=ff[2],
                max_p=getattr(ind, "max_p", np.nan),
                exp_name=getattr(ind, "exp_name", ""),
                train_it=train_it_raw,
                ckpt_idx=ckpt_idx,
                failed=int(bool(getattr(ind, "_failed", False) or getattr(ind, "failed", False))),
                fail_category=fail_category,
                fail_reason=fail_reason,
            )
        )
    if rows:
        pd.DataFrame(rows).to_csv(
            pareto_history_path,
            mode="a",
            header=False,
            index=False,
        )


def append_generation_summary(
    generation_summary_path: Path,
    generation: int,
    pop: Sequence[Any],
    pareto_size: int,
    minimal_p: float,
    invalid_v: set[float],
    invalid_e: set[float],
    invalid_p: set[float],
) -> None:
    v, e, p = invalid_objective_masks(pop, invalid_v, invalid_e, invalid_p)
    n = len(pop)
    valid_mask = np.isfinite(v) & np.isfinite(e) & np.isfinite(p)
    valid_count = int(np.sum(valid_mask))
    invalid_count = int(n - valid_count)
    failure_count = int(
        np.sum(
            [
                bool(getattr(ind, "_failed", False) or getattr(ind, "failed", False))
                for ind in pop
            ]
        )
    )
    failure_rate = (failure_count / n) if n > 0 else 0.0

    def nan_stat(arr: np.ndarray, fn, default=np.nan):
        arrv = arr[np.isfinite(arr)]
        if arrv.size == 0:
            return default
        return float(fn(arrv))

    row = dict(
        generation=generation,
        population_size=n,
        valid_count=valid_count,
        invalid_count=invalid_count,
        failure_count=failure_count,
        failure_rate=float(failure_rate),
        pareto_size=int(pareto_size),
        minimal_p=float(minimal_p),
        best_vel=nan_stat(v, np.max),
        best_eff=nan_stat(e, np.max),
        best_prog=nan_stat(p, np.max),
        median_vel=nan_stat(v, np.median),
        median_eff=nan_stat(e, np.median),
        median_prog=nan_stat(p, np.median),
        q1_vel=nan_stat(v, lambda x: np.percentile(x, 25)),
        q1_eff=nan_stat(e, lambda x: np.percentile(x, 25)),
        q1_prog=nan_stat(p, lambda x: np.percentile(x, 25)),
        q3_vel=nan_stat(v, lambda x: np.percentile(x, 75)),
        q3_eff=nan_stat(e, lambda x: np.percentile(x, 75)),
        q3_prog=nan_stat(p, lambda x: np.percentile(x, 75)),
    )
    pd.DataFrame([row]).to_csv(
        generation_summary_path,
        mode="a",
        header=False,
        index=False,
    )
