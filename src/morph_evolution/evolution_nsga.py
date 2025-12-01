#!/usr/bin/env python3
"""
evolution_nsga.py

NSGA-II co-design of drone morphology + control policy in a continuous
genome space, with:

  - Simulated Binary Crossover (SBX) in [0, 1]^D.
  - Polynomial mutation (bounded).
  - Policy inheritance between generations.
  - Fitness caching in a CSV "database".
  - Optional multi-GPU parallelism via Ray.
  - Post-hoc analysis and plotting utilities.

The genome lives in [0, 1]^D and is mapped to physical parameters by
`Chromosome_Drone.to_physical()`, which in turn drives `UrdfMaker`.
"""

from __future__ import annotations

import argparse
import datetime
import os
import time
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
from deap import base, creator, tools
from filelock import FileLock
from tensorboard.backend.event_processing import event_accumulator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Project imports (adapt to your package layout)
from drone_making import UrdfMaker
from chromosome_drone import Chromosome_Drone
from winged_drone_train.train import training
from winged_drone_train.eval import evaluation


# =============================================================================
#  GLOBAL GA CONFIGURATION (single place for all hyper-parameters)
# =============================================================================


@dataclass
class GAConfig:
    """
    All GA / NSGA-II hyper-parameters in one place.

    Edit this dataclass (or override fields from the CLI) to configure
    the search behaviour.
    """

    # --- Population & evolutionary budget ---------------------------------
    population_size: int = 4       # number of individuals per generation
    num_generations: int = 3       # number of generations to run

    # --- NSGA-II operators (continuous) -----------------------------------
    crossover_probability: float = 0.9   # probability of SBX crossover
    mutation_probability: float = 0.3    # probability of applying mutation
    eta_c: float = 15.0                  # SBX "spread" parameter (higher = more local)
    eta_m: float = 20.0                  # polynomial mutation parameter

    # --- RL training / evaluation -----------------------------------------
    gen_policy: bool = False
    policy_path: Optional[str] = None  # path to initial policy checkpoint
    train_iters_new: int = 50       # iterations for NEW morphologies
    train_iters_inherit: int = 200  # iterations when inheriting from a parent
    train_envs: int = 256           # number of envs during training
    eval_envs: int = 256            # number of envs during evaluation
    vmin: float = 6.0               # min commanded speed in evaluation
    vmax: float = 18.0              # max commanded speed in evaluation

    # --- Fitness shaping / invalid individuals ----------------------------
    fail_value: float = 1e6         # not used directly, kept for completeness
    weights: Tuple[float, float, float] = (1.0, -1.0, 1.0)  # (+vel, -energy, +progress)

    # --- Progress threshold (minimal_p) -----------------------------------
    use_dynamic_p: bool = True      # if True: percentile-based threshold
    fixed_p: float = 200.0          # fallback / fixed threshold [m]
    pct_above: float = 50.0         # fraction of individuals above minimal_p

    # --- Policy inheritance -----------------------------------------------
    inherit_policy: bool = False    # if True: offspring can inherit parent policy

    # --- Output / logging -------------------------------------------------
    csv_basename: str = "nsga"       # CSV filename (".csv" added automatically)
    run_name: Optional[str] = None   # Optional custom run name (defaults to timestamp)
    base_dir: str = "nsga"           # Root directory for all artifacts
    device: str = "cuda:0"           # Device passed to training/evaluation


# Default config used when no custom config is provided
DEFAULT_GA_CONFIG = GAConfig()


# =============================================================================
#  PARALLELISM SELECTION (Ray or serial)
# =============================================================================


def _want_parallel() -> bool:
    """
    Decide whether to run in parallel (Ray) or serial.

    Rules
    -----
    - GA_PARALLEL=0 / "false" / "no"  → force serial.
    - GA_PARALLEL=1 / "true" / "yes"  → force parallel.
    - GA_PARALLEL unset / "auto"      → parallel only if ≥ 2 GPUs.
    """
    flag = os.getenv("GA_PARALLEL", "auto").lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    return torch.cuda.device_count() > 1


USE_PARALLEL = _want_parallel()

if USE_PARALLEL:
    import ray  # type: ignore[import]

    ray.init(log_to_driver=False)


def _prepare_device_env(device: str) -> str:
    """
    Normalize a device string and set CUDA visibility accordingly.

    This keeps training/eval consistent between processes and makes the
    requested GPU explicit (useful on clusters).
    """
    dev = device.strip()
    low = dev.lower()
    if low.startswith("cuda:"):
        _, _, idx = low.partition(":")
        if idx:
            os.environ["CUDA_VISIBLE_DEVICES"] = idx
            return f"cuda:{idx}"
        return "cuda"
    if low == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    return dev


@contextmanager
def _pushd(path: Path):
    """Temporarily change working directory."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# =============================================================================
#  SENTINEL VALUES FOR INVALID INDIVIDUALS
# =============================================================================

INVALID_V = {0.0}       # invalid average velocity
INVALID_E = {-100.0}    # negative energy sentinel (equivalent to +100 before flip)
INVALID_P = {0.0}       # invalid progress / maneuverability


# =============================================================================
#  TRAIN + EVAL (single GPU, programmatic)
# =============================================================================


def _extract_reward_curve(
    log_dir: Path,
    train_iters: int,
    n_points: int = 10,
    win_frac: float = 0.05,
) -> Dict[str, float]:
    """
    Extract reward values at 10%, 20%, …, 100% of training using a moving
    average over a window equal to `win_frac` of total training steps.

    Assumptions
    -----------
    - TensorBoard scalar tag is exactly `"Train/mean_reward"`.
    - Window is defined in *training steps* (± half-window around target step).
    - If the window has no events (sparse logging), we fall back to the
      nearest event.

    Returns
    -------
    dict
        Keys are "rew_10pct", "rew_20pct", ..., "rew_100pct".
        If the curve cannot be extracted, an empty dict is returned.
    """
    if not log_dir.exists():
        return {}

    ea = event_accumulator.EventAccumulator(str(log_dir))
    try:
        ea.Reload()
    except Exception:
        return {}

    key = "Train/mean_reward"
    if key not in ea.Tags().get("scalars", []):
        return {}

    events = ea.Scalars(key)
    if not events:
        return {}

    steps = np.array([e.step for e in events], dtype=np.int64)
    vals = np.array([e.value for e in events], dtype=np.float64)

    win_steps = max(1, int(round(train_iters * win_frac)))
    half = win_steps // 2

    out: Dict[str, float] = {}
    for frac in range(1, n_points + 1):
        target = int(round(train_iters * frac / n_points))
        lo, hi = target - half, target + half
        mask = (steps >= lo) & (steps <= hi)

        if mask.any():
            out[f"rew_{frac * 10}pct"] = float(np.nanmean(vals[mask]))
        else:
            idx = int(np.argmin(np.abs(steps - target)))
            out[f"rew_{frac * 10}pct"] = float(vals[idx])

    return out

def _eval_only_custom(
    genome_norm,
    policy_path,
    tag,
    cfg,
    return_arrays=True,
):
    phys_genome = Chromosome_Drone.to_physical(genome_norm)
    urdf_dir = Path(cfg["URDF_DIR"]).expanduser().resolve()
    urdf_file = Path(UrdfMaker(phys_genome, out_dir=urdf_dir).create_urdf()).resolve()
    exp_name = urdf_file.stem
    if cfg.get("EXP_PREFIX"):
        exp_name = f"{cfg['EXP_PREFIX']}-{exp_name}"

    _prepare_device_env(cfg.get("DEVICE", "cuda:0"))
    base_dir = Path(cfg["BASE_DIR"]).expanduser().resolve()
    eval_dir = (Path(cfg["LOGS_DIR"]).expanduser().resolve() / "eval" / exp_name)

    # Usa evaluation ma caricando la policy custom
    with _pushd(base_dir):
        out = evaluation(
            exp_name=exp_name,
            urdf_file=urdf_file,
            ckpt=None,  # Ignorato
            envs=cfg["EVAL_ENVS"],
            vmin=cfg["VMIN"],
            vmax=cfg["VMAX"],
            return_arrays=return_arrays,
            custom_policy_path=policy_path,  # << PATCH IN eval.py
            eval_dir=eval_dir,
        )

    if return_arrays:
        v_dict, e_dict, p_dict, _, extra = out
        max_p = extra["max_p"]
    else:
        v_dict, e_dict, p_dict, _, max_p = out
        extra = None

    meta = dict(
        vel_v=v_dict["mean_v"],
        vel_E=-v_dict["mean_E"],
        vel_P=v_dict["mean_progress"],
        eff_v=e_dict["mean_v"],
        eff_E=-e_dict["mean_E"],
        eff_P=e_dict["mean_progress"],
        prog_v=p_dict["mean_v"],
        prog_E=-p_dict["mean_E"],
        prog_P=p_dict["mean_progress"],
        train_it=0,
        exp_name=exp_name,
        max_p=max_p,
        # reward curve zerata
        **{f"rew_{i*10}pct": 0.0 for i in range(1, 11)},
        final_reward=0.0,
        steps90_pct=0.0,
    )

    ff = [
        v_dict["mean_v"],
        -e_dict["mean_E"],
        p_dict["mean_progress"],
    ]

    return ff, meta, extra


def _train_and_eval_sync(
    genome_norm: Sequence[float],
    parent_info: Tuple[Optional[str], Optional[int]],
    tag: str,
    cfg: Dict[str, Any],
    return_arrays: bool = True,
):
    """
    Synchronous wrapper that:

      1. Maps normalized genome → physical parameters.
      2. Builds a URDF for the given physical genome.
      3. Trains a control policy (with optional inheritance).
      4. Runs evaluation over a speed range.
      5. Reads training reward curve from TensorBoard.

    Parameters
    ----------
    genome_norm : sequence of float
        Normalized genome in [0, 1]^D.
    parent_info : (parent_exp, parent_ckpt)
        If provided, we inherit policy weights from this experiment.
    tag : str
        Run tag for naming/logging (e.g. timestamp).
    cfg : dict
        Training / evaluation config.
    return_arrays : bool
        If True, also return raw arrays from evaluation.

    Returns
    -------
    ff : list[float]
        Final 3D fitness (velocity, -energy, progress).
    meta : dict
        Detailed metadata for CSV logging.
    extra : Optional[dict]
        When `return_arrays` is True, contains raw arrays:
            "p_s": smoothed progress,
            "v_s": commanded speeds,
            "E_s": energy per meter.
    """
    parent_exp, parent_ckpt = parent_info

    # 1) Map genome to physical parameters and generate URDF
    phys_genome = Chromosome_Drone.to_physical(genome_norm)
    urdf_dir = Path(cfg["URDF_DIR"]).expanduser().resolve()
    urdf_file = Path(UrdfMaker(phys_genome, out_dir=urdf_dir).create_urdf()).resolve()
    exp_name = urdf_file.stem
    if cfg.get("EXP_PREFIX"):
        exp_name = f"{cfg['EXP_PREFIX']}-{exp_name}"

    device = _prepare_device_env(cfg.get("DEVICE", "cuda:0"))
    base_dir = Path(cfg["BASE_DIR"]).expanduser().resolve()

    # 2) Training iterations: shorter if inheriting a parent policy
    train_iters = (
        cfg["TRAIN_ITERS_INHERIT"]
        if (parent_exp is not None and parent_ckpt is not None)
        else cfg["TRAIN_ITERS"]
    )

    # 3) Train policy for this morphology
    with _pushd(base_dir):
        training(
            exp_name=exp_name,
            urdf_file=urdf_file,
            num_envs=cfg["TRAIN_ENVS"],
            max_iterations=train_iters,
            parent_exp=parent_exp,
            parent_ckpt=parent_ckpt,
            device=device,
        )

    # 4) Evaluate policy at multiple commanded speeds
    eval_dir = (Path(cfg["LOGS_DIR"]).expanduser().resolve() / "eval" / exp_name)

    with _pushd(base_dir):
        out = evaluation(
            exp_name=exp_name,
            urdf_file=urdf_file,
            ckpt=train_iters,
            envs=cfg["EVAL_ENVS"],
            vmin=cfg["VMIN"],
            vmax=cfg["VMAX"],
            return_arrays=return_arrays,
            eval_dir=eval_dir,
        )

    if return_arrays:
        v_dict, e_dict, p_dict, _, extra = out
        max_p = extra["max_p"]
    else:
        v_dict, e_dict, p_dict, _, max_p = out
        extra = None

    # 5) Read TensorBoard logs and compute a smoothed reward curve
    tb_log_dir = Path(cfg["LOG_ROOT"]) / exp_name
    reward_curve = _extract_reward_curve(
        tb_log_dir,
        train_iters,
        n_points=10,
        win_frac=0.05,
    )

    # 6) Build metadata dict for logging
    meta = dict(
        vel_v=v_dict["mean_v"],
        vel_E=-v_dict["mean_E"],
        vel_P=v_dict["mean_progress"],
        eff_v=e_dict["mean_v"],
        eff_E=-e_dict["mean_E"],
        eff_P=e_dict["mean_progress"],
        prog_v=p_dict["mean_v"],
        prog_E=-p_dict["mean_E"],
        prog_P=p_dict["mean_progress"],
        train_it=train_iters,
        exp_name=exp_name,
        max_p=max_p,
        **reward_curve,
    )

    ff = [
        v_dict["mean_v"],           # +velocity
        -e_dict["mean_E"],          # +(-energy)
        p_dict["mean_progress"],    # +progress
    ]

    return ff, meta, extra


# Parallel / serial dispatch wrapper
if USE_PARALLEL:

    @ray.remote(num_gpus=1)
    def train_and_eval_remote(*args, **kwargs):
        return _train_and_eval_sync(*args, **kwargs)

else:

    def train_and_eval_remote(*args, **kwargs):
        return _train_and_eval_sync(*args, **kwargs)


# =============================================================================
#  POST-HOC ANALYSIS (plots)
# =============================================================================


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
        csv_path: str = "deap_temp.csv",
        stats_obj: Optional["Stats"] = None,
        pkl_path: Optional[str] = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)

        # Replace sentinel values with NaN so Matplotlib ignores them.
        for col in ("vel_v", "eff_v", "prog_v"):
            self.df.loc[self.df[col].isin(INVALID_V), col] = np.nan
        for col in ("vel_E", "eff_E", "prog_E"):
            self.df.loc[self.df[col].isin(INVALID_E), col] = np.nan
        for col in ("vel_P", "eff_P", "prog_P"):
            self.df.loc[self.df[col].isin(INVALID_P), col] = np.nan

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
        """
        Plot, for the best individual in each generation:
          - Final episodic reward (left y-axis).
          - Percentage of steps needed to reach 90% reward (right y-axis).
        """
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
        """Plot best velocity per generation."""
        if self.stats is None:
            print("No stats – skipping velocity progress plot.")
            return

        V = np.where(np.isin(self.stats.V, list(INVALID_V)), np.nan, self.stats.V)
        plt.plot(np.nanmax(V, axis=1), label="velocity ↑")
        plt.xlabel("Generation")
        plt.ylabel("Best value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ velocity progress plot →", out)

    def fronts_progress_P(self, out: str = "best_prog_per_gen.png") -> None:
        """Plot best progress per generation (and minimal_p if available)."""
        if self.stats is None:
            print("No stats – skipping progress plot.")
            return

        M = np.where(np.isin(self.stats.M, list(INVALID_P)), np.nan, self.stats.M)
        best_prog = np.nanmax(M, axis=1)

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
        """Plot best (-energy) per generation."""
        if self.stats is None:
            print("No stats – skipping energy plot.")
            return

        E = np.where(np.isin(self.stats.E, list(INVALID_E)), np.nan, self.stats.E)
        plt.plot(np.nanmax(E, axis=1), label="-energy ↑")
        plt.xlabel("Generation")
        plt.ylabel("Best value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("✓ energy progress plot →", out)

    def analyze(self, prefix: str = "analysis") -> None:
        """Run all standard plots."""
        self.fronts_progress_V(f"{prefix}_progress_vel.png")
        self.fronts_progress_P(f"{prefix}_progress_prog.png")
        self.fronts_progress_E(f"{prefix}_progress_eff.png")
        self.final_reward_steps(f"{prefix}_reward_speed.png")


# =============================================================================
#  FITNESS DATABASE (CSV cache)
# =============================================================================


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
        if not self.path.exists():
            self.df.to_csv(self.path, index=False)

    def lookup_fitness(self, chromo: Sequence[float]) -> Optional[List[float]]:
        """Return cached fitness values for the chromosome, if present."""
        row = self.df[self.df.chromosome == str(list(chromo))]
        if row.empty:
            return None
        return [row[f"ff_{i}"].min() for i in range(self.n_obj)]

    def insert(self, chromo: Sequence[float], ff: Sequence[float], meta: Dict[str, Any]) -> None:
        """Append a new row (chromo, fitness, metadata) to the CSV."""
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
            self.df = pd.concat([self.df, row_df], ignore_index=True)
            row_df.to_csv(self.path, mode="a", header=False, index=False)

    def _blank(self) -> pd.DataFrame:
        cols = (
            ["timestamp", "chromosome"]
            + [f"ff_{i}" for i in range(self.n_obj)]
            + [
                "generation",
                "exp_name",
                "train_it",
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
            ]
        )
        return pd.DataFrame(columns=cols)

    def get_row(self, chromo: Sequence[float]) -> Optional[pd.Series]:
        """Return the entire row for the chromosome, or None if absent."""
        row = self.df[self.df.chromosome == str(list(chromo))]
        return None if row.empty else row.iloc[0]


# =============================================================================
#  STATS CONTAINER
# =============================================================================


class Stats:
    """
    Container for per-generation distributions of the three objectives:
    velocity, energy, and progress.
    """

    def __init__(self, n_pop: int, n_gen: int, n_obj: int) -> None:
        self.arr = np.zeros((n_obj, n_gen + 1, n_pop))
        self.arr.fill(np.nan)

    def record(self, gen: int, pop: List["IndType"]) -> None:
        invalid_sets = (INVALID_V, INVALID_E, INVALID_P)
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


IndType = TypeVar("IndType")


# =============================================================================
#  NSGA-II MAIN CLASS
# =============================================================================


class CodesignDEAP:
    """
    NSGA-II loop for morphology + controller co-design.

    Usage (programmatic)
    --------------------
        cfg = GAConfig()
        cfg.population_size = 40
        cfg.num_generations = 30
        ...
        ga = CodesignDEAP(cfg)
        final_pop = ga.run()
    """

    def __init__(self, config: GAConfig = DEFAULT_GA_CONFIG) -> None:
        self.cfg = config

        # Basic checks
        if self.cfg.population_size % 4 != 0:
            raise ValueError("population_size must be a multiple of 4 for tournamentDCD.")

        self.n_pop = self.cfg.population_size
        self.n_gen = self.cfg.num_generations
        self.cx_pb = self.cfg.crossover_probability
        self.mut_pb = self.cfg.mutation_probability
        self.inherit_policy = self.cfg.inherit_policy

        self.gen_policy = self.cfg.gen_policy
        self.policy_path = self.cfg.policy_path

        self.tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = self.cfg.run_name or self.tag
        self.base_dir = (Path(self.cfg.base_dir) / self.run_name).expanduser().resolve()
        self.urdf_dir = self.base_dir / "urdf_generated"
        self.logs_dir = self.base_dir / "logs"
        self.analysis_dir = self.base_dir / "analysis"
        for d in (self.urdf_dir, self.logs_dir, self.analysis_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.log_root = (self.logs_dir / "ea").resolve()
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.exp_prefix = self.run_name
        self.stats_path = (self.analysis_dir / "stats.pkl").resolve()

        self.db = FitnessDB(self.cfg.csv_basename, 3, root=self.analysis_dir)
        self.stats = Stats(self.n_pop, self.n_gen, 3)

        # Create DEAP fitness and individual types (only once)
        if "FitMulti" not in creator.__dict__:
            creator.create("FitMulti", base.Fitness, weights=self.cfg.weights)
        if "Chrom" not in creator.__dict__:
            creator.create("Chrom", list, fitness=creator.FitMulti)

        self.IndType = creator.Chrom

        # Bounds in normalized space (all genes ∈ [0, 1])
        low, up = Chromosome_Drone.get_bounds()
        self._low = low
        self._up = up
        n_genes = Chromosome_Drone.num_genes()

        # DEAP toolbox
        self.tb = base.Toolbox()
        self.tb.register("attr_float", random.random)
        self.tb.register("ind", tools.initRepeat, self.IndType, self.tb.attr_float, n=n_genes)
        self.tb.register("pop", tools.initRepeat, list, self.tb.ind)

        # Typical NSGA-II operators for continuous decision variables
        self.tb.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            low=self._low,
            up=self._up,
            eta=self.cfg.eta_c,
        )
        self.tb.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=self._low,
            up=self._up,
            eta=self.cfg.eta_m,
            indpb=1.0 / n_genes,
        )
        self.tb.register("select", tools.selNSGA2)
        self.tb.register("evaluate", self._evaluate)

    # ------------------------------------------------------------------ #
    # Fitness evaluation                                                 #
    # ------------------------------------------------------------------ #

    def _evaluate(self, indiv: "IndType") -> Tuple[float, float, float]:
        """
        DEAP evaluation hook – possibly spawns Ray jobs.

        The individual is a list of floats in [0, 1] (normalized genome).
        """
        chromo = list(indiv)
        print(f"[evaluate] gen={getattr(self, '_gen', 0)} chr={chromo}")

        # CSV cache: if we have seen this chromosome before, reuse its fitness.
        cached_row = self.db.get_row(chromo)
        if cached_row is not None:
            ff_cached = [cached_row[f"ff_{i}"] for i in range(3)]
            indiv.fitness.values = tuple(ff_cached)
            indiv.max_p = cached_row.get("max_p", np.nan)
            indiv.exp_name = cached_row.get("exp_name", None)
            indiv.train_it = cached_row.get("train_it", self.cfg.train_iters_new)
            print(f"   ↪ cache-hit exp={cached_row.get('exp_name', 'NA')} ff={ff_cached}")
            return tuple(ff_cached)

        # New chromosome → full train + eval pipeline
        print(
            "   ↪ NEW chromosome → training for "
            f"{self.cfg.train_iters_new} iterations "
            f"(or {self.cfg.train_iters_inherit} if inheritance is triggered)."
        )

        parent_info = (
            getattr(indiv, "parent_exp", None),
            getattr(indiv, "parent_ckpt", None),
        )
        cfg = dict(
            TRAIN_ITERS=self.cfg.train_iters_new,
            TRAIN_ITERS_INHERIT=self.cfg.train_iters_inherit,
            TRAIN_ENVS=self.cfg.train_envs,
            EVAL_ENVS=self.cfg.eval_envs,
            VMIN=self.cfg.vmin,
            VMAX=self.cfg.vmax,
            LOG_ROOT=str(self.log_root),
            EXP_PREFIX=self.exp_prefix,
            DEVICE=self.cfg.device,
            BASE_DIR=str(self.base_dir),
            URDF_DIR=str(self.urdf_dir),
            LOGS_DIR=str(self.logs_dir),
        )

        if USE_PARALLEL:
            fut = train_and_eval_remote.remote(chromo, parent_info, self.tag, cfg, True)
            indiv._pending_future = fut
            # Placeholder; real fitness will be set after Ray returns.
            return (0.0, 0.0, 0.0)

        if self.gen_policy:
            print("   ↪ GEN_POLICY active → skipping training")
            
            ff, meta, extra = _eval_only_custom(
                chromo,
                self.policy_path,
                self.tag,
                cfg,
                return_arrays=True,
            )
            print(f"   ✔ sync-eval ff={ff} max_p={meta['max_p']:.2f}")

            indiv._meta_raw = meta
            indiv.max_p = meta["max_p"]
            indiv.exp_name = meta["exp_name"]
            indiv.train_it = meta["train_it"]
            
            if extra:
                indiv._p_s = extra["p_s"]
                indiv._v_s = extra["v_s"]
                indiv._E_s = extra["E_s"]

            indiv.fitness.values = tuple(ff)
            return tuple(ff)

        ff, meta, extra = _train_and_eval_sync(chromo, parent_info, self.tag, cfg, True)
        print(f"   ✔ sync-train+eval ff={ff} max_p={meta['max_p']:.2f}")

        indiv._meta_raw = meta
        indiv.max_p = meta["max_p"]
        indiv.exp_name = meta["exp_name"]
        indiv.train_it = meta["train_it"]
        if extra:
            indiv._p_s = extra["p_s"]
            indiv._v_s = extra["v_s"]
            indiv._E_s = extra["E_s"]

        indiv.fitness.values = tuple(ff)
        return tuple(ff)

    # ------------------------------------------------------------------ #
    # Fitness post-processing (progress threshold)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pick_triples(
        p_s: np.ndarray,
        v_s: np.ndarray,
        E_s: np.ndarray,
        minimal_p: float,
    ):
        """
        Given smoothed progress (p_s), commanded speeds (v_s) and energy (E_s),
        extract three representative operating points:

          - vel: max velocity with p >= minimal_p
          - eff: min energy with p >= minimal_p
          - prog: global max progress (unfiltered)
        """
        if len(p_s) == 0:
            zero = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            return zero, zero, zero

        idx_p = int(np.argmax(p_s))
        mask = np.where(p_s >= minimal_p)[0]

        if mask.size == 0:
            vel = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
            eff = dict(mean_v=0.0, mean_E=100.0, mean_progress=0.0)
        else:
            idx_v = int(mask[np.argmax(v_s[mask])])
            idx_e = int(mask[np.argmin(E_s[mask])])
            vel = dict(
                mean_v=float(v_s[idx_v]),
                mean_E=float(E_s[idx_v]),
                mean_progress=float(p_s[idx_v]),
            )
            eff = dict(
                mean_v=float(v_s[idx_e]),
                mean_E=float(E_s[idx_e]),
                mean_progress=float(p_s[idx_e]),
            )

        prog = dict(
            mean_v=float(v_s[idx_p]),
            mean_E=float(E_s[idx_p]),
            mean_progress=float(p_s[idx_p]),
        )
        return vel, eff, prog

    def _finalize_and_persist(self, ind: "IndType", minimal_p: float) -> None:
        """
        Compute final fitness for an individual, apply the progress threshold,
        and write a row in the DB.
        """
        if not hasattr(ind, "_p_s"):
            # Cached individuals or failed evals: nothing to do.
            return

        vel_d, eff_d, prog_d = self._pick_triples(
            np.asarray(ind._p_s),
            np.asarray(ind._v_s),
            np.asarray(ind._E_s),
            minimal_p,
        )

        ff_final = (
            vel_d["mean_v"],
            -eff_d["mean_E"],
            prog_d["mean_progress"],
        )

        meta = dict(getattr(ind, "_meta_raw", {}))
        meta.update(
            dict(
                max_p=ind.max_p,
                minimal_p=minimal_p,
                vel_v=vel_d["mean_v"],
                vel_E=-vel_d["mean_E"],
                vel_P=vel_d["mean_progress"],
                eff_v=eff_d["mean_v"],
                eff_E=-eff_d["mean_E"],
                eff_P=eff_d["mean_progress"],
                prog_v=prog_d["mean_v"],
                prog_E=-prog_d["mean_E"],
                prog_P=prog_d["mean_progress"],
            )
        )

        ind.exp_name = meta.get("exp_name", getattr(ind, "exp_name", None))
        ind.train_it = meta.get("train_it", getattr(ind, "train_it", self.cfg.train_iters_new))

        print(
            f"[finalize] gen={self._gen} chr={list(ind)} "
            f"vel={vel_d['mean_v']:.2f} effE={eff_d['mean_E']:.2f} "
            f"prog={prog_d['mean_progress']:.2f}"
        )

        self.db.insert(list(ind), ff_final, dict(generation=self._gen, **meta))
        ind.fitness.values = ff_final

    # ------------------------------------------------------------------ #
    # Evolution helpers                                                  #
    # ------------------------------------------------------------------ #

    def _train_eval_population(self, population: List["IndType"]) -> None:
        """
        Train + evaluate all individuals with invalid fitness.
        Handles:
          - launching Ray jobs,
          - waiting for results,
          - computing minimal_p,
          - finalizing and logging fitness values.
        """
        # 1) Launch training/evaluation where needed
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = self.tb.evaluate(ind)

        # 2) Wait for Ray jobs
        if USE_PARALLEL:
            pend = [ind for ind in population if hasattr(ind, "_pending_future")]
            if pend:
                print(f"  ⏳ waiting for {len(pend)} Ray jobs…")
                results = ray.get([ind._pending_future for ind in pend])
                for ind, (ff, meta, extra) in zip(pend, results):
                    ind._meta_raw = meta
                    ind.max_p = meta["max_p"]
                    ind.exp_name = meta["exp_name"]
                    ind.train_it = meta["train_it"]
                    if extra:
                        ind._p_s = extra["p_s"]
                        ind._v_s = extra["v_s"]
                        ind._E_s = extra["E_s"]
                    ind.fitness.values = tuple(ff)
                    del ind._pending_future
                    print(
                        f"   ✅ Ray done chr={list(ind)} "
                        f"ff={ff} max_p={meta['max_p']:.2f}"
                    )

        # 3) minimal_p dynamic/fixed
        peaks = [getattr(ind, "max_p", np.nan) for ind in population]
        peaks = [p for p in peaks if not np.isnan(p)]
        if self.cfg.use_dynamic_p and peaks:
            perc = 100.0 - self.cfg.pct_above
            minimal_p = 0.9 * np.percentile(peaks, perc)
        else:
            minimal_p = self.cfg.fixed_p
        print(
            f"[Gen {self._gen}] minimal_p = {minimal_p:.2f} "
            f"(dynamic={self.cfg.use_dynamic_p}, pct_above={self.cfg.pct_above}%)"
        )

        # 4) finalize → CSV
        for ind in population:
            if not hasattr(ind, "_persisted"):
                self._finalize_and_persist(ind, minimal_p)
                ind._persisted = True

    def _apply_variation(self, offspring: List["IndType"], parents: List["IndType"]) -> None:
        """
        Crossover, mutation, and optional inheritance **before** training.
        """
        # Clean up custom attributes on offspring
        for ch in offspring:
            for a in (
                "exp_name",
                "parent_exp",
                "parent_ckpt",
                "_pending_future",
                "_meta_raw",
                "_p_s",
                "_v_s",
                "_E_s",
                "_evaluated",
                "_persisted",
                "max_p",
            ):
                if hasattr(ch, a):
                    delattr(ch, a)

        # Apply SBX + polynomial mutation pairwise
        for i in range(0, len(offspring), 2):
            c1, c2 = offspring[i], offspring[i + 1]

            # crossover
            if random.random() < self.cx_pb:
                self.tb.mate(c1, c2)
                if hasattr(c1.fitness, "values"):
                    del c1.fitness.values
                if hasattr(c2.fitness, "values"):
                    del c2.fitness.values

            # mutation
            if random.random() < self.mut_pb:
                self.tb.mutate(c1)
                del c1.fitness.values
            if random.random() < self.mut_pb:
                self.tb.mutate(c2)
                del c2.fitness.values

            # inheritance → assign exp/ckpt BEFORE training
            if self.inherit_policy:
                infos = []
                for p in (parents[i], parents[i + 1]):
                    if hasattr(p, "exp_name"):
                        ck_it = getattr(p, "train_it", self.cfg.train_iters_new)
                        ck = self.log_root / p.exp_name / f"model_{ck_it}.pt"
                        if ck.is_file():
                            infos.append((p.exp_name, ck_it))
                if infos:
                    c1.parent_exp, c1.parent_ckpt = random.choice(infos)
                    c2.parent_exp, c2.parent_ckpt = random.choice(infos)

    def _after_generation(self, pop: List["IndType"]) -> None:
        """Update stats, generate plots, and print generation summary."""
        g = self._gen
        self.stats.record(g, pop)
        if g % 3 == 0 or g == self.n_gen:
            out_dir = self.analysis_dir / f"g{g:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            PostAnalyzer(self.db.path, self.stats).analyze(prefix=str(out_dir / "gen"))

        best_v = np.nanmax(self.stats.V[g])
        best_e = -np.nanmin(self.stats.E[g])
        best_p = np.nanmax(self.stats.M[g])
        print(
            f"--- Gen {g} summary  "
            f"best_vel={best_v:.2f}  best_eff={best_e:.2f}  best_prog={best_p:.2f}"
        )

    # ------------------------------------------------------------------ #
    # Main loop                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> List["IndType"]:
        """
        Run the full NSGA-II evolution and return the final population.
        """
        print(f"[setup] Run directory: {self.base_dir}")
        print(f"[setup] CSV cache: {self.db.path}")

        # GEN 0
        pop = self.tb.pop(self.n_pop)
        self._gen = 0
        self._train_eval_population(pop)
        pop = tools.selNSGA2(pop, self.n_pop)
        self._after_generation(pop)

        # GEN ≥ 1
        for g in range(1, self.n_gen + 1):
            self._gen = g
            print(f"\n════════ Generation {g}/{self.n_gen} ════════")

            # 1) parent selection (requires crowding_dist)
            parents = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.tb.clone(p) for p in parents]

            # 2) variation (+ inheritance) before training
            self._apply_variation(offspring, parents)

            # 3) train + eval offspring
            self._train_eval_population(offspring)

            # 4) survivor-selection NSGA-II → new population
            pop = tools.selNSGA2(pop + offspring, self.n_pop)

            # 5) logging / plots
            self._after_generation(pop)

        # Save global stats
        with self.stats_path.open("wb") as f:
            import pickle

            pickle.dump(self.stats, f)
        print(f"Statistics saved ✔ → {self.stats_path}")
        return pop


# =============================================================================
#  CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=4, help="Population size.")
    parser.add_argument("--gen", type=int, default=3, help="Number of generations.")
    parser.add_argument(
        "--train_it",
        type=int,
        default=30,
        help="Training iterations for NEW morphologies.",
    )
    parser.add_argument(
        "--inherit",
        action="store_true",
        default=False,
        help="Enable policy inheritance for offspring.",
    )
    parser.add_argument(
        "--no_dynamic_p",
        action="store_true",
        help="Disable dynamic minimal_p and use fixed_p instead.",
    )
    parser.add_argument(
        "--fixed_p",
        type=float,
        default=200.0,
        help="Fixed minimal_p threshold (used if --no_dynamic_p).",
    )
    parser.add_argument(
        "--pct_above",
        type=float,
        default=50.0,
        help="Percentage of individuals above minimal_p when dynamic.",
    )
    parser.add_argument(
        "--gen_policy", action="store_true", default=False,
        help="Skip training and evaluate using a custom pre-trained policy"
    )
    parser.add_argument(
        "--policy_path", type=str, default=None,
        help="Path to a pre-trained policy to use together with --gen_policy"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name (defaults to timestamp).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for training/eval (e.g., cuda:0 or cpu).",
    )

    args = parser.parse_args()

    # Build GA configuration from defaults + CLI overrides
    cfg = GAConfig()
    cfg.population_size = args.pop
    cfg.num_generations = args.gen
    cfg.train_iters_new = args.train_it
    cfg.inherit_policy = args.inherit
    cfg.csv_basename = "nsga"
    cfg.use_dynamic_p = not args.no_dynamic_p
    cfg.fixed_p = args.fixed_p
    cfg.pct_above = args.pct_above
    cfg.gen_policy = args.gen_policy
    cfg.policy_path = args.policy_path
    cfg.run_name = args.run_name
    cfg.device = args.device

    ga = CodesignDEAP(cfg)
    ga.run()


if __name__ == "__main__":
    main()
