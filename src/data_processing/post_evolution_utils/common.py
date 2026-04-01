from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None


FITNESS_COLUMNS = ["ff_0", "ff_1", "ff_2"]
EVAL_REWARD_COLUMN = "eval_reward_mean"
FITNESS_LABELS = {
    "ff_0": "Speed",
    "ff_1": "Cost Of Transport",
    "ff_2": "Progress",
    EVAL_REWARD_COLUMN: "Reward accumulated during Evaluation",
}

# Custom reference point (BIX3) for Pareto plots.
DEFAULT_BIX3_POINT = np.array([18.2, 0.33, 371.0], dtype=float)

# Sentinel values defined in src/morph_evolution/evolution_nsga.py
INVALID_V = {0.0}
INVALID_E = {-10.0}
INVALID_P = {0.0}
INVALID_PROGRESS_THRESHOLD = 250.0
AIR_DENSITY_KG_M3 = 1.225
AIR_DYNAMIC_VISCOSITY_KG_M_S = 1.81e-5
NACA4_CSV_PATH = Path(__file__).resolve().parents[2] / "naca_generation" / "naca4.csv"
DEFAULT_RUN_NAMES = ("nsga_GP", "nsga_SP")
COMPARISON_DIR_NAME = "evolution comparison"
EVOLUTION_COLORS = {
    "nsga_GP": "#1f77b4",
    "nsga_SP": "#ff7f0e",
}
STEPS90_REWARD_TARGET = 0.9
STEPS90_REWARD_OFFSET = 5.0
TARGET_BIX3_URDF_STEM = "[0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4, 0.2, 2, 0, 2, 2.5, 3, 4, 16]"
TARGET_BIX3_URDF_PARAMS = "[0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4.0, 0.2, 2.0, 0.0, 2.0, 2.5, 3.0, 4.0, 16.0]"
SP_EVAL_RESULTS_PATH = Path(__file__).resolve().parents[1] / "evaluation_results_SP.csv"
GP_EVAL_RESULTS_PATH = Path(__file__).resolve().parents[1] / "evaluation_results_GP1.csv"
REWARD_CHECKPOINT_COLUMNS = [f"rew_{pct}pct" for pct in range(10, 101, 10)]
PLOT_COLOR_CYCLE = {
    "speed": "#1b9e77",
    "efficiency": "#1f78b4",
    "progress": "#d95f02",
    "reward": "#6a3d9a",
    "baseline": "#6d2e46",
    "count": "#4c566a",
    "takeover": "#0f766e",
    "diversity": "#8b5cf6",
    "innovation": "#c2410c",
    "selected": "#1d4ed8",
    "rejected": "#94a3b8",
}
PAPER_LINKS = {
    "derl": "https://www.nature.com/articles/s41467-021-25874-z",
    "hexa": "https://arxiv.org/abs/2505.14129",
    "controller_learning": "https://www.cs.vu.nl/~gusz/papers/2020_Evolving-Controllers%20versus%20learning-controllers%20for%20morphologically%20evolvable%20robots.pdf",
    "inheritance": "https://eprints.whiterose.ac.uk/id/eprint/182721/1/IEEE_TCDS_2021_Revised.pdf",
}

def load_nsga_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"generation", *FITNESS_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns in {csv_path}: {missing_list}")
    return df

def _resolve_analysis_dir(input_path: Path) -> Path | None:
    path = input_path.expanduser().resolve()
    if path.is_dir():
        if (path / "population_history.csv").exists():
            return path
        if (path / "analysis" / "population_history.csv").exists():
            return (path / "analysis").resolve()
        return None

    parent = path.parent
    if path.name in {"population_history.csv", "pareto_history.csv", "generation_summary.csv", "nsga.csv"}:
        return parent if (parent / "population_history.csv").exists() else None
    if parent.name == "analysis" and (parent / "population_history.csv").exists():
        return parent.resolve()
    if (parent / "analysis" / "population_history.csv").exists():
        return (parent / "analysis").resolve()
    return None

def _resolve_run_dir(input_path: Path, analysis_dir: Path | None) -> Path:
    path = input_path.expanduser().resolve()
    if path.is_dir():
        if analysis_dir is not None and path == analysis_dir:
            return analysis_dir.parent.resolve()
        return path
    if analysis_dir is not None:
        return analysis_dir.parent.resolve()
    return path.parent.resolve()

def _find_bix3_line(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("agg,-1,"):
                continue
            if TARGET_BIX3_URDF_STEM in line and TARGET_BIX3_URDF_PARAMS in line:
                return line
    raise ValueError(f"Could not find BIX3 reference line in {csv_path}")

def _bix3_suffix_tokens(line: str) -> list[str]:
    marker = f"{TARGET_BIX3_URDF_PARAMS},"
    marker_idx = line.find(marker)
    if marker_idx < 0:
        raise ValueError("Could not locate BIX3 parameter block in evaluation row.")
    suffix = line[marker_idx + len(marker):]
    return [token.strip() for token in suffix.split(",")]

def _parse_float_token(token: str) -> float:
    token = token.strip()
    if token.lower() == "nan":
        return float("nan")
    return float(token)

def _load_bix3_point_from_sp() -> np.ndarray:
    line = _find_bix3_line(SP_EVAL_RESULTS_PATH)
    tokens = _bix3_suffix_tokens(line)
    if len(tokens) < 4:
        raise ValueError("Unexpected SP BIX3 row format.")
    point = np.array([_parse_float_token(token) for token in tokens[-4:-1]], dtype=float)
    point[1] = -point[1]
    return point

def _load_bix3_point_from_gp() -> np.ndarray:
    line = _find_bix3_line(GP_EVAL_RESULTS_PATH)
    tokens = _bix3_suffix_tokens(line)
    if len(tokens) < 16:
        raise ValueError("Unexpected GP BIX3 row format.")
    baseline_tokens = tokens[-16:-4]
    baseline_values = np.array([_parse_float_token(token) for token in baseline_tokens], dtype=float).reshape(3, 4)
    point = baseline_values[:, :3].mean(axis=0)
    point[1] = -point[1]
    return point

def resolve_bix3_point(run_dir: Path) -> np.ndarray:
    run_name = run_dir.name
    if run_name == "nsga_SP":
        return _load_bix3_point_from_sp()
    if run_name == "nsga_GP":
        return _load_bix3_point_from_gp()
    return DEFAULT_BIX3_POINT.copy()

def load_run_tables(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, Path, Path]:
    analysis_dir = _resolve_analysis_dir(input_path)
    run_dir = _resolve_run_dir(input_path, analysis_dir)
    if analysis_dir is not None:
        population_path = analysis_dir / "population_history.csv"
        summary_path = analysis_dir / "generation_summary.csv"
        pop_df = load_nsga_csv(population_path)
        summary_df = pd.read_csv(summary_path) if summary_path.exists() else None
        return pop_df, summary_df, population_path, run_dir

    csv_path = input_path.expanduser().resolve()
    pop_df = load_nsga_csv(csv_path)
    return pop_df, None, csv_path, run_dir

def _extract_genome_name_series(df: pd.DataFrame) -> pd.Series:
    if "exp_name" not in df.columns and "rep_exp_names" not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype="object")

    def _extract(row: pd.Series) -> str | None:
        for col in ("exp_name", "rep_exp_names"):
            if col not in row:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            name = str(val).split("|")[0]
            match = re.search(r"\[[^\]]+\]", name)
            if match:
                return match.group(0)
        return None

    return df.apply(_extract, axis=1)

def _invalid_row_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["ff_0"].isin(INVALID_V)
        | df["ff_1"].isin(INVALID_E)
        | df["ff_2"].isin(INVALID_P)
    )

def invalid_repetition_mask(
    df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame | None = None,
) -> pd.Series:
    ref_df = reference_df if reference_df is not None else df
    invalid_row_ref = _invalid_row_mask(ref_df)
    genome_series_ref = _extract_genome_name_series(ref_df)
    if genome_series_ref.notna().any():
        invalid_genomes = set(genome_series_ref[invalid_row_ref].dropna().unique())
        genome_series = _extract_genome_name_series(df)
        invalid_by_genome = genome_series.isin(invalid_genomes)
        return _invalid_row_mask(df) | invalid_by_genome
    return _invalid_row_mask(df)

def filter_sentinels(df: pd.DataFrame, *, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    invalid_mask = invalid_repetition_mask(df, reference_df=reference_df)
    return df.loc[~invalid_mask].copy()

def apply_invalid_repetition_values(
    df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()
    invalid_mask = invalid_repetition_mask(df, reference_df=reference_df)
    df.loc[invalid_mask, "ff_0"] = 0.0
    df.loc[invalid_mask, "ff_1"] = float(min(INVALID_E))
    return df

def filter_to_agg(df: pd.DataFrame) -> pd.DataFrame:
    if "row_kind" not in df.columns:
        return df.copy()
    row_kind = df["row_kind"].astype(str).str.lower()
    return df.loc[row_kind == "agg"].copy()

def _scaled_ylim(values: np.ndarray, padding_ratio: float = 0.08) -> tuple[float, float]:
    if values.size == 0:
        return (0.0, 1.0)
    lower = np.nanmin(values)
    upper = np.nanmax(values)
    if lower == upper:
        padding = 1.0 if lower == 0 else abs(lower) * 0.1
        return (lower - padding, upper + padding)
    padding = (upper - lower) * padding_ratio
    return (lower - padding, upper + padding)

def _scaled_ylim_from_series(values: list[np.ndarray], padding_ratio: float = 0.1) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    combined = np.concatenate([v for v in values if v.size])
    return _scaled_ylim(combined, padding_ratio=padding_ratio)

def _fitness_trend_ylim(
    median_values: np.ndarray,
    best_values: np.ndarray,
    *,
    lower_from_best: bool = False,
    margin_ratio: float = 0.05,
) -> tuple[float, float]:
    med = np.asarray(median_values, dtype=float)
    med = med[np.isfinite(med)]
    best = np.asarray(best_values, dtype=float)
    best = best[np.isfinite(best)]

    if med.size == 0 and best.size == 0:
        return (0.0, 1.0)

    if med.size == 0:
        med = best.copy()
    if best.size == 0:
        best = med.copy()

    if lower_from_best:
        lower_ref = float(np.min(best))
        upper_ref = float(np.max(med))
    else:
        lower_ref = float(np.min(med))
        upper_ref = float(np.max(best))
    lower_margin = abs(lower_ref) * float(margin_ratio)
    upper_margin = abs(upper_ref) * float(margin_ratio)

    lower = lower_ref - lower_margin
    upper = upper_ref + upper_margin
    if np.isclose(lower, upper):
        pad = max(abs(lower) * float(margin_ratio), 1e-6)
        return (lower - pad, upper + pad)
    return (lower, upper)

def _apply_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "axes.facecolor": "#fbfbfa",
        "axes.edgecolor": "#d4d4d4",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.alpha": 0.28,
        "grid.linewidth": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#d4d4d4",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })

def _analysis_csv_path(run_dir: Path, filename: str) -> Path:
    run_dir = run_dir.expanduser().resolve()
    candidates = [
        run_dir / "analysis" / filename,
        run_dir / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} under {run_dir}")

def _write_plot_note(
    output_path: Path,
    *,
    title: str,
    importance: str,
    prior_work: list[str],
    what_to_read: list[str],
    takeaways: list[str],
    caveats: list[str] | None = None,
) -> None:
    return

def _write_standard_run_output_notes(output_dir: Path) -> None:
    note_specs = {
        "pareto_fronts.png": dict(
            title="Pareto fronts and BIX3 reference",
            importance=(
                "This is the main trade-off figure of the run: it shows where discovered solutions lie in objective space, "
                "which points define the current non-dominated frontier, and how far that frontier sits from the BIX3 baseline."
            ),
            prior_work=[
                f"Unconventional Hexacopters via Evolution and Learning uses Pareto-style performance comparisons to show that unusual morphologies can outperform conventional ones ({PAPER_LINKS['hexa']}).",
                f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots compares how controller regime changes the quality and position of discovered morphologies ({PAPER_LINKS['controller_learning']}).",
            ],
            what_to_read=[
                "Look for frontier expansion toward higher speed and progress and lower cost of transport.",
                "The distance between the frontier and the BIX3 star tells you whether improvement is only internal to the search or externally meaningful.",
                "A curved frontier with a visible knee suggests a real family of trade-offs rather than one dominant design.",
            ],
            takeaways=[
                "This is the clearest summary of what objective compromises evolution found.",
                "It also tells you whether your search is uncovering unconventional but competitive designs against the reference drone.",
            ],
        ),
        "pareto_front_3d.html": dict(
            title="3D Pareto front",
            importance=(
                "The 3D Pareto view helps when 2D slices hide structure. It makes visible whether one niche dominates all three objectives or whether distinct specialist families occupy different parts of the volume."
            ),
            prior_work=[
                f"Unconventional Hexacopters via Evolution and Learning motivates interpreting whole performance trade-off structures, not only single-objective winners ({PAPER_LINKS['hexa']}).",
                f"Embodied intelligence via learning and evolution studies families of morphologies over time rather than isolated endpoints ({PAPER_LINKS['derl']}).",
            ],
            what_to_read=[
                "Look for separated clouds or ridges, which indicate distinct morphological niches.",
                "Compare where BIX3 sits relative to the full front, not just one 2D projection.",
                "If the front is thin and sheet-like, the trade-off is structured; if it is diffuse, search may still be noisy.",
            ],
            takeaways=[
                "This plot is useful when you want to argue that the search discovered multiple classes of solution, not one accidental winner.",
            ],
        ),
        "cot_vs_wing_reynolds.png": dict(
            title="Cost of transport vs operating Reynolds ratio",
            importance=(
                "This figure links energetic performance to the Reynolds regime actually experienced by the wing, normalized by the nominal Reynolds value attached to the selected NACA profile in the aero solver."
            ),
            prior_work=[
                f"Unconventional Hexacopters via Evolution and Learning interprets evolved designs through physically meaningful morphology descriptors, not only objective scores ({PAPER_LINKS['hexa']}).",
                f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots emphasizes that different search regimes prefer different body parameters ({PAPER_LINKS['controller_learning']}).",
            ],
            what_to_read=[
                "If low-CoT designs cluster near 100%, the search may be aligning geometry and speed with the airfoil's nominal Reynolds regime.",
                "If efficient points sit systematically above or below 100%, that indicates a mismatch or a deliberate operating shift.",
                "Coloring by effective speed tells you whether Reynolds effects are mostly geometric, mostly kinematic, or both.",
            ],
            takeaways=[
                "This is one of the strongest aerodynamic interpretation plots because it connects the genome to the solver parameters you actually use.",
                "It can reveal whether performance is coming from airfoil choice, chord sizing, operating speed, or their interaction.",
            ],
        ),
        "cot_vs_wing_chord.png": dict(
            title="Cost of transport vs wing chord",
            importance=(
                "Chord is a compact bridge between geometry and aerodynamics: it directly affects wing area, thickness, and Reynolds number. This plot checks whether energetic efficiency is associated with a characteristic chord regime."
            ),
            prior_work=[
                f"Unconventional Hexacopters via Evolution and Learning motivates interpreting unusual morphologies through body descriptors rather than raw parameters ({PAPER_LINKS['hexa']}).",
                f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots studies which body parameters become favored under different controller optimization schemes ({PAPER_LINKS['controller_learning']}).",
            ],
            what_to_read=[
                "A narrow low-CoT band indicates the search prefers a specific chord scale.",
                "A broad cloud means chord alone is not enough and must be read together with span, airfoil and speed.",
                "Color by effective speed helps separate purely geometric from operating-point effects.",
            ],
            takeaways=[
                "This plot gives a simple geometric interpretation for energetic winners.",
            ],
        ),
        "evolutionary_metrics.png": dict(
            title="Evolutionary search metrics",
            importance=(
                "Objective trends alone do not tell you whether the search is converging, diversifying, or collapsing. Hypervolume, diversity, spacing, entropy and distance metrics summarize the health of the evolutionary process itself."
            ),
            prior_work=[
                f"Embodied intelligence via learning and evolution analyzes evolutionary progress over time rather than only the final best design ({PAPER_LINKS['derl']}).",
                f"Morpho-evolution with learning using a controller archive studies how support mechanisms affect exploration and convergence dynamics ({PAPER_LINKS['inheritance']}).",
            ],
            what_to_read=[
                "Hypervolume up usually means a better frontier.",
                "GD and IGD down mean the current generations are approaching the best frontier found over the whole run.",
                "Diversity and entropy tell you whether improvement is happening with continued exploration or with collapse into a few niches.",
            ],
            takeaways=[
                "This is the main process-level diagnostic for whether the run improved in a healthy way.",
                "It is especially useful to compare GP and SP beyond raw objective values.",
            ],
        ),
        "steps90_by_generation.png": dict(
            title="Steps to 90% reward by generation",
            importance=(
                "This is the simplest learnability plot: it measures how much normalized training budget individuals need before reaching 90% of their own best reward."
            ),
            prior_work=[
                f"Embodied intelligence via learning and evolution studies whether evolution discovers bodies that are easier to adapt and learn with ({PAPER_LINKS['derl']}).",
                f"Unconventional Hexacopters via Evolution and Learning reports learning-speed descriptors to explain why some morphologies are controller-friendly ({PAPER_LINKS['hexa']}).",
            ],
            what_to_read=[
                "Lower is better: it means useful learning happens earlier.",
                "If the mean decreases and the spread decreases, the whole population is becoming easier and more consistent to train.",
                "A plateau means the search stopped improving learnability even if final reward may still move.",
            ],
            takeaways=[
                "This figure is one of the cleanest signatures of embodied intelligence in your data.",
            ],
        ),
        "training_speed90_by_generation.png": dict(
            title="Training speed to 90% reward by generation",
            importance=(
                "This plot combines reward scale and learning speed into one proxy: higher values mean individuals both reach strong rewards and do so quickly."
            ),
            prior_work=[
                f"Unconventional Hexacopters via Evolution and Learning uses learning descriptors richer than final reward alone ({PAPER_LINKS['hexa']}).",
                f"Embodied intelligence via learning and evolution motivates looking for bodies that improve adaptation efficiency, not only asymptotic score ({PAPER_LINKS['derl']}).",
            ],
            what_to_read=[
                "Higher is better because it rewards fast arrival at strong behavior.",
                "If this rises while steps90 falls, morphologies are becoming both better and easier to train.",
                "If this rises only because final reward rises, then learning efficiency itself may not be improving.",
            ],
            takeaways=[
                "This is a compact summary figure for trainability under SP.",
            ],
        ),
        "dominance_solutions.png": dict(
            title="Global Pareto front growth",
            importance=(
                "This figure tracks how many genuinely new non-dominated solutions the run discovers and how many of those still matter at the end."
            ),
            prior_work=[
                f"Embodied intelligence via learning and evolution tracks the spread and persistence of successful morphological families over evolutionary time ({PAPER_LINKS['derl']}).",
                f"Morpho-evolution with learning using a controller archive studies how search support affects useful innovation discovery ({PAPER_LINKS['inheritance']}).",
            ],
            what_to_read=[
                "The red curve is the size of the accumulated global front.",
                "Blue bars show new global-front discoveries; green bars show how many of those survive into the final global front.",
                "Late green bars mean the run kept finding lasting breakthroughs rather than only early ones.",
            ],
            takeaways=[
                "This plot distinguishes temporary novelty from durable progress.",
            ],
        ),
        "pareto_rank_distribution.png": dict(
            title="Pareto rank distribution by generation",
            importance=(
                "Pareto ranks summarize selection pressure. This plot shows whether the population is concentrating into front 1 or still spreading across weaker fronts."
            ),
            prior_work=[
                f"Unconventional Hexacopters via Evolution and Learning motivates reading evolutionary progress in terms of how many competitive designs are being maintained ({PAPER_LINKS['hexa']}).",
                f"Morpho-evolution with learning using a controller archive studies how learning support changes the efficiency of evolutionary search ({PAPER_LINKS['inheritance']}).",
            ],
            what_to_read=[
                "More mass in front 1 means stronger concentration of good solutions.",
                "Persistent mass in deeper fronts indicates continued exploration or weaker selection.",
                "A sudden collapse into front 1 can mean convergence, but can also mean loss of diversity.",
            ],
            takeaways=[
                "This is a clean selection-pressure plot complementary to hypervolume and diversity.",
            ],
        ),
        "pair_generation_rank_distribution.png": dict(
            title="Previous vs current generation rank distribution",
            importance=(
                "This plot checks how the combined pool of previous and current generation individuals is sorted by Pareto rank, which is close to the actual NSGA-II selection event."
            ),
            prior_work=[
                f"Morpho-evolution with learning using a controller archive studies search dynamics at the level of selection and inheritance events ({PAPER_LINKS['inheritance']}).",
                f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots is directly relevant because controller regime changes which bodies survive selection ({PAPER_LINKS['controller_learning']}).",
            ],
            what_to_read=[
                "If current-generation individuals dominate front 1, innovation is entering the elite set.",
                "If previous-generation individuals keep most of front 1, search is exploiting incumbents.",
                "The rank breakdown tells you where newcomers are competitive and where they are filtered out.",
            ],
            takeaways=[
                "This figure is useful to understand whether new offspring are genuinely displacing older elites.",
            ],
        ),
        "newcomer_rank_distribution.png": dict(
            title="Newcomer rank distribution",
            importance=(
                "This plot isolates only newly generated individuals and asks how good they are immediately when inserted into the selection pool."
            ),
            prior_work=[
                f"Unconventional Hexacopters via Evolution and Learning is relevant because it argues that unusual new morphologies can create useful breakthroughs ({PAPER_LINKS['hexa']}).",
                f"Morpho-evolution with learning using a controller archive studies how often innovation is productive enough to survive selection ({PAPER_LINKS['inheritance']}).",
            ],
            what_to_read=[
                "Many newcomers in front 1 means mutation and recombination are generating strong designs.",
                "If newcomers are mostly in deep fronts, the search is relying on refinement of incumbents.",
                "Generation-to-generation changes show whether innovation quality is improving over time.",
            ],
            takeaways=[
                "This is the most direct quality-control plot for newly created morphologies.",
            ],
        ),
        "phylogenetic_tree_like.png": dict(
            title="Phylogenetic tree of top surviving lineages",
            importance=(
                "This plot turns logged ancestry into a paper-style phylogeny so you can see which founding families expand, split, and survive over time."
            ),
            prior_work=[
                f"Embodied intelligence via learning and evolution explicitly uses phylogenetic analysis to interpret how morphology families spread ({PAPER_LINKS['derl']}).",
                f"Unconventional Hexacopters via Evolution and Learning is relevant because it also interprets families of unconventional designs rather than isolated winners ({PAPER_LINKS['hexa']}).",
            ],
            what_to_read=[
                "Long dense branches indicate persistent successful lineages.",
                "Large dark nodes point to individuals with many descendants and strong scalar-fitness proxy.",
                "Late branch splits suggest continuing innovation inside already successful families.",
            ],
            takeaways=[
                "This is the most intuitive genealogy figure for a reader coming from the DERL-style literature.",
            ],
        ),
        "two_parent_genealogy.png": dict(
            title="Two-parent genealogy of top surviving lineages",
            importance=(
                "Unlike the tree-style approximation, this figure shows the actual two-parent structure created by crossover, so it exposes recombination between successful families."
            ),
            prior_work=[
                f"Morpho-evolution with learning using a controller archive is relevant because inheritance mechanisms affect how lineages combine and survive ({PAPER_LINKS['inheritance']}).",
                f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots motivates comparing which morphology families survive under different optimization regimes ({PAPER_LINKS['controller_learning']}).",
            ],
            what_to_read=[
                "Cross-family connections indicate recombination between successful lineages.",
                "Dense local ancestry with few cross-links indicates mostly within-family refinement.",
                "Large nodes with many descendants are hubs of future search.",
            ],
            takeaways=[
                "This is the honest genealogy plot for your actual algorithm, because it does not hide two-parent reproduction.",
            ],
        ),
        "lineage_muller_fractional.png": dict(
            title="Muller diagram of top surviving lineages",
            importance=(
                "This figure compresses genealogy into lineage population share over time, which makes takeover and coexistence immediately visible."
            ),
            prior_work=[
                f"Embodied intelligence via learning and evolution uses Muller-style lineage analysis to show how successful families spread ({PAPER_LINKS['derl']}).",
                f"Unconventional Hexacopters via Evolution and Learning is relevant because it frames the search in terms of families of unconventional designs, not isolated points ({PAPER_LINKS['hexa']}).",
            ],
            what_to_read=[
                "A band that expands and stays thick corresponds to a successful lineage takeover.",
                "Several thick bands coexisting indicate sustained diversity among strong families.",
                "Late appearance of a new thick band is evidence of a delayed breakthrough.",
            ],
            takeaways=[
                "This is the clearest visual summary of lineage competition across generations.",
            ],
        ),
        "genome_pca_generations.png": dict(
            title="Genome PCA across generations",
            importance=(
                "PCA gives a coarse morphospace view: it shows whether the population drifts toward a new region of genome space, contracts into a narrow basin, or keeps exploring broadly."
            ),
            prior_work=[
                f"Embodied intelligence via learning and evolution studies how body families occupy and move through morphology space ({PAPER_LINKS['derl']}).",
                f"Unconventional Hexacopters via Evolution and Learning motivates looking for shifts toward unconventional morphology regions ({PAPER_LINKS['hexa']}).",
            ],
            what_to_read=[
                "Cloud drift means the search is moving through genome space, not only refining in place.",
                "Cloud contraction means convergence.",
                "Separated clusters suggest multiple morphological niches surviving at the same time.",
            ],
            takeaways=[
                "This plot is a compact visual summary of exploration versus convergence in morphospace.",
            ],
        ),
        "genome_fitness_correlation.png": dict(
            title="Genome-fitness correlation map",
            importance=(
                "This plot asks which genes are consistently associated with speed, efficiency and progress, giving a first interpretation of what the optimizer is actually selecting."
            ),
            prior_work=[
                f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots is directly relevant because it studies how different optimization regimes select different body parameters ({PAPER_LINKS['controller_learning']}).",
                f"Unconventional Hexacopters via Evolution and Learning interprets performance through morphology descriptors and design choices ({PAPER_LINKS['hexa']}).",
            ],
            what_to_read=[
                "A gene correlated with one objective and anti-correlated with another is part of the trade-off structure.",
                "Consistent correlation signs across objectives indicate broadly favorable directions.",
                "Weak correlations suggest the phenotype is strongly interaction-driven rather than one-gene-at-a-time.",
            ],
            takeaways=[
                "This is the first pass for turning black-box search into interpretable parameter effects.",
            ],
        ),
        "genome_top5pct_gene_means.png": dict(
            title="Top-5% genome means by objective",
            importance=(
                "Instead of correlations over the full population, this plot asks what the elite tails of each objective actually look like in genome space."
            ),
            prior_work=[
                f"Unconventional Hexacopters via Evolution and Learning motivates comparing morphology archetypes attached to different performance niches ({PAPER_LINKS['hexa']}).",
                f"Evolving-Controllers Versus Learning-Controllers for Morphologically Evolvable Robots is relevant because elite bodies differ depending on the optimization regime ({PAPER_LINKS['controller_learning']}).",
            ],
            what_to_read=[
                "Genes where elite means separate strongly across objectives are the most niche-defining coordinates.",
                "Overlap between elite means indicates parameters that are good across several objectives.",
                "Large error bars mean the elite set admits multiple design strategies even for one objective.",
            ],
            takeaways=[
                "This plot is a simple bridge between objectives and elite genotype structure before moving to derived aero descriptors.",
            ],
        ),
    }

    for filename, spec in note_specs.items():
        output_path = output_dir / filename
        if not output_path.exists():
            continue
        _write_plot_note(output_path, **spec)

def _fitness_columns_for_plots(df: pd.DataFrame) -> list[str]:
    cols = [col for col in FITNESS_COLUMNS if col in df.columns]
    if EVAL_REWARD_COLUMN in df.columns:
        cols.append(EVAL_REWARD_COLUMN)
    return cols

def _fitness_frame_for_plot(
    df: pd.DataFrame,
    fitness: str,
    *,
    is_raw: bool = False,
) -> pd.DataFrame:
    temp = df[["generation", fitness]].copy()
    if is_raw and fitness == "ff_0":
        temp.loc[temp[fitness].isin(INVALID_V), fitness] = 0.0
    if is_raw and fitness == "ff_1":
        temp.loc[temp[fitness].isin(INVALID_E), fitness] = float(min(INVALID_E))
    if fitness == "ff_1":
        temp[fitness] = -temp[fitness]
    return temp

def _fitness_points_for_plot(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    points = df[cols].to_numpy(dtype=float)
    for idx, col in enumerate(cols):
        if col == "ff_1":
            points[:, idx] = -points[:, idx]
    return points

def _pareto_points_from_plot(points: np.ndarray, cols: list[str]) -> np.ndarray:
    pareto_points = points.copy()
    for idx, col in enumerate(cols):
        if col == "ff_1":
            pareto_points[:, idx] = -pareto_points[:, idx]
    return pareto_points

def _pca_fit_transform_unique(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.size == 0:
        raise ValueError("No points available for PCA.")
    unique_points = np.unique(points, axis=0)
    mean = np.mean(unique_points, axis=0)
    centered = unique_points - mean
    if centered.shape[0] < 2:
        raise ValueError("Need at least two unique genomes for PCA.")
    _, singular_vals, vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = (singular_vals**2) / max(centered.shape[0] - 1, 1)
    total_var = float(np.sum(explained_var))
    explained_ratio = explained_var / total_var if total_var > 0 else np.zeros_like(explained_var)
    return mean, vt, explained_ratio

def _pc_label(explained_ratio: np.ndarray, idx: int) -> str:
    ratio = float(explained_ratio[idx]) if explained_ratio.size > idx else 0.0
    return f"PC{idx + 1} ({ratio * 100:.2f}%)"

def _select_equally_spaced_generations(df: pd.DataFrame, n_generations: int = 6) -> list[int]:
    unique_generations = np.sort(pd.to_numeric(df["generation"], errors="coerce").dropna().unique().astype(int))
    if unique_generations.size == 0:
        raise ValueError("No valid generations available for genome PCA plot.")
    if unique_generations.size <= n_generations:
        return unique_generations.tolist()

    targets = np.linspace(unique_generations[0], unique_generations[-1], n_generations)
    selected: list[int] = []
    used: set[int] = set()

    for idx, target in enumerate(targets):
        if idx == n_generations - 1:
            gen = int(unique_generations[-1])
        else:
            order = np.argsort(np.abs(unique_generations - target))
            gen = None
            for candidate_idx in order:
                candidate = int(unique_generations[candidate_idx])
                if candidate not in used:
                    gen = candidate
                    break
            if gen is None:
                gen = int(unique_generations[order[0]])
        selected.append(gen)
        used.add(gen)

    return selected

def compute_mean_runtime_per_individual_by_generation(run_dir: Path) -> pd.DataFrame:
    nsga_path = run_dir / "analysis" / "nsga.csv"
    if not nsga_path.exists():
        raise FileNotFoundError(f"NSGA log not found: {nsga_path}")

    usecols = ["generation", "uid", "row_kind", "train_duration_s", "eval_duration_s"]
    df = pd.read_csv(nsga_path, usecols=usecols)
    agg_df = df[df["row_kind"].astype(str) == "agg"].copy()

    if agg_df.empty:
        agg_df = (
            df.groupby(["generation", "uid"], as_index=False)[["train_duration_s", "eval_duration_s"]]
            .mean()
            .copy()
        )
    else:
        agg_df = agg_df[["generation", "uid", "train_duration_s", "eval_duration_s"]].copy()

    for col in ["generation", "train_duration_s", "eval_duration_s"]:
        agg_df[col] = pd.to_numeric(agg_df[col], errors="coerce")

    agg_df["runtime_total_s"] = agg_df["train_duration_s"].fillna(0.0) + agg_df["eval_duration_s"].fillna(0.0)
    return (
        agg_df.groupby("generation", as_index=False)
        .agg(
            n_individuals=("uid", "nunique"),
            mean_train_duration_s=("train_duration_s", "mean"),
            std_train_duration_s=("train_duration_s", "std"),
            mean_eval_duration_s=("eval_duration_s", "mean"),
            std_eval_duration_s=("eval_duration_s", "std"),
            mean_runtime_total_s=("runtime_total_s", "mean"),
            std_runtime_total_s=("runtime_total_s", "std"),
        )
        .sort_values("generation")
    )

def _compute_steps90_ratio_from_row(row: pd.Series, prefix: str = "rew_") -> float:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)pct$")
    pct_cols: list[tuple[int, str]] = []
    for col in row.index:
        match = pattern.match(str(col))
        if match:
            pct_cols.append((int(match.group(1)), str(col)))
    if not pct_cols:
        return float("nan")

    pct_cols.sort(key=lambda item: item[0])
    max_pct = pct_cols[-1][0]
    if max_pct <= 0:
        return float("nan")

    values = []
    for _, col in pct_cols:
        values.append(pd.to_numeric(row.get(col), errors="coerce"))
    values = np.asarray(values, dtype=float) + float(STEPS90_REWARD_OFFSET)

    valid = np.isfinite(values)
    if not np.any(valid):
        return float("nan")

    max_reward = np.nanmax(values)
    if not np.isfinite(max_reward) or max_reward <= 0:
        return float("nan")

    threshold = float(STEPS90_REWARD_TARGET) * max_reward
    pcts = np.asarray([pct for pct, _ in pct_cols], dtype=float)
    valid_mask = np.isfinite(values) & np.isfinite(pcts)
    pcts = pcts[valid_mask]
    vals = values[valid_mask]
    if len(pcts) == 0:
        return float("nan")

    for i in range(1, len(pcts)):
        x0, x1 = pcts[i - 1], pcts[i]
        y0, y1 = vals[i - 1], vals[i]
        crossed = (y0 < threshold <= y1) or (y0 > threshold >= y1)
        if not crossed:
            continue
        if np.isclose(y1, y0):
            x_cross = x1
        else:
            alpha = float(np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0))
            x_cross = x0 + alpha * (x1 - x0)
        return float(x_cross / max_pct)

    above_idx = np.where(vals >= threshold)[0]
    if len(above_idx) > 0:
        return float(pcts[int(above_idx[0])] / max_pct)
    return float("nan")

def _compute_best_reward_from_row(row: pd.Series, prefix: str = "rew_") -> float:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)pct$")
    reward_cols: list[str] = []
    for col in row.index:
        if pattern.match(str(col)):
            reward_cols.append(str(col))
    if not reward_cols:
        return float("nan")

    values = pd.to_numeric(row[reward_cols], errors="coerce").to_numpy(dtype=float)
    values = values + float(STEPS90_REWARD_OFFSET)
    if not np.isfinite(values).any():
        return float("nan")
    best_reward = float(np.nanmax(values))
    if not np.isfinite(best_reward) or best_reward <= 0.0:
        return float("nan")
    return best_reward

def _load_bix3_eval_row(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation results CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        header = next(f).strip().split(",")
    if "urdf_params" not in header:
        raise ValueError(f"Missing 'urdf_params' in evaluation header: {csv_path}")

    line = _find_bix3_line(csv_path)
    suffix_tokens = _bix3_suffix_tokens(line)
    start_idx = header.index("urdf_params") + 1
    suffix_cols = header[start_idx:]
    n = min(len(suffix_cols), len(suffix_tokens))
    data = {col: token for col, token in zip(suffix_cols[:n], suffix_tokens[:n])}
    return pd.Series(data, dtype="object")

def _baseline_drone_steps90_ratio_for_run(run_dir: Path) -> float:
    run_name = run_dir.name
    if run_name == "nsga_SP":
        csv_path = SP_EVAL_RESULTS_PATH
    else:
        csv_path = GP_EVAL_RESULTS_PATH
    bix3_row = _load_bix3_eval_row(csv_path)
    return _compute_steps90_ratio_from_row(bix3_row)

def compute_steps90_summary_by_generation(run_dir: Path) -> pd.DataFrame:
    nsga_path = run_dir / "analysis" / "nsga.csv"
    if not nsga_path.exists():
        raise FileNotFoundError(f"NSGA log not found: {nsga_path}")

    nsga_df = pd.read_csv(nsga_path)
    row_kind = nsga_df["row_kind"].astype(str) if "row_kind" in nsga_df.columns else pd.Series("", index=nsga_df.index)
    steps_df = nsga_df[row_kind == "agg"].copy()
    if steps_df.empty:
        steps_df = nsga_df[row_kind == "rep"].copy()
    if steps_df.empty:
        steps_df = nsga_df.copy()

    steps_df["generation"] = pd.to_numeric(steps_df["generation"], errors="coerce")
    steps_df["steps90_ratio"] = steps_df.apply(_compute_steps90_ratio_from_row, axis=1)
    steps_df = steps_df[np.isfinite(steps_df["generation"])].copy()

    summary_df = (
        steps_df.groupby("generation", as_index=False)
        .agg(
            n_samples=("steps90_ratio", lambda s: int(np.sum(np.isfinite(pd.to_numeric(s, errors="coerce"))))),
            mean_steps90_ratio=("steps90_ratio", "mean"),
            std_steps90_ratio=("steps90_ratio", "std"),
        )
        .sort_values("generation")
    )
    summary_df["std_steps90_ratio"] = summary_df["std_steps90_ratio"].fillna(0.0)
    summary_df["mean_steps90_pct"] = summary_df["mean_steps90_ratio"] * 100.0
    summary_df["std_steps90_pct"] = summary_df["std_steps90_ratio"] * 100.0
    return summary_df

def compute_training_speed90_summary_by_generation(run_dir: Path) -> pd.DataFrame:
    nsga_path = run_dir / "analysis" / "nsga.csv"
    if not nsga_path.exists():
        raise FileNotFoundError(f"NSGA log not found: {nsga_path}")

    nsga_df = pd.read_csv(nsga_path)
    row_kind = nsga_df["row_kind"].astype(str) if "row_kind" in nsga_df.columns else pd.Series("", index=nsga_df.index)
    speed_df = nsga_df[row_kind == "agg"].copy()
    if speed_df.empty:
        speed_df = nsga_df[row_kind == "rep"].copy()
    if speed_df.empty:
        speed_df = nsga_df.copy()

    speed_df["generation"] = pd.to_numeric(speed_df["generation"], errors="coerce")
    speed_df["steps90_ratio"] = speed_df.apply(_compute_steps90_ratio_from_row, axis=1)
    speed_df["best_reward"] = speed_df.apply(_compute_best_reward_from_row, axis=1)
    speed_df["training_speed90"] = np.where(
        np.isfinite(speed_df["steps90_ratio"]) &
        (speed_df["steps90_ratio"] > 0.0) &
        np.isfinite(speed_df["best_reward"]) &
        (speed_df["best_reward"] > 0.0),
        float(STEPS90_REWARD_TARGET) * speed_df["best_reward"] / speed_df["steps90_ratio"],
        np.nan,
    )
    speed_df = speed_df[np.isfinite(speed_df["generation"])].copy()

    summary_df = (
        speed_df.groupby("generation", as_index=False)
        .agg(
            n_samples=("training_speed90", lambda s: int(np.sum(np.isfinite(pd.to_numeric(s, errors="coerce"))))),
            mean_training_speed90=("training_speed90", "mean"),
            std_training_speed90=("training_speed90", "std"),
        )
        .sort_values("generation")
    )
    summary_df["std_training_speed90"] = summary_df["std_training_speed90"].fillna(0.0)
    return summary_df

def _descriptor_summary_by_generation(df: pd.DataFrame, descriptor_cols: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    grouped = df.groupby("generation", sort=True)
    for col in descriptor_cols:
        out[col] = grouped[col].agg(
            median="median",
            q1=lambda s: float(np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.25)),
            q3=lambda s: float(np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.75)),
        ).reset_index()
    return out

def _covariance_regularized(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if points.ndim != 2:
        raise ValueError("Expected a 2D array for covariance computation.")
    n_features = points.shape[1]
    if points.shape[0] <= 1:
        return np.eye(n_features, dtype=float) * eps
    cov = np.cov(points, rowvar=False)
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if cov.shape != (n_features, n_features):
        cov = cov.reshape(n_features, n_features)
    cov = 0.5 * (cov + cov.T)
    cov += np.eye(n_features, dtype=float) * eps
    return cov

def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

def _frechet_distance_gaussians(points_a: np.ndarray, points_b: np.ndarray) -> float:
    mean_a = np.mean(points_a, axis=0)
    mean_b = np.mean(points_b, axis=0)
    cov_a = _covariance_regularized(points_a)
    cov_b = _covariance_regularized(points_b)
    mean_term = float(np.sum((mean_a - mean_b) ** 2))
    cov_a_sqrt = _sqrtm_psd(cov_a)
    middle = cov_a_sqrt @ cov_b @ cov_a_sqrt
    cov_term = float(np.trace(cov_a + cov_b - 2.0 * _sqrtm_psd(middle)))
    return max(0.0, mean_term + cov_term)

def _weighted_centroid_distance(points_a: np.ndarray, points_b: np.ndarray, weights: np.ndarray) -> float:
    mean_a = np.mean(points_a, axis=0)
    mean_b = np.mean(points_b, axis=0)
    delta = mean_a - mean_b
    return float(np.sqrt(np.sum(weights * (delta**2))))

def compute_joint_pca_cluster_metrics(
    run_dfs: dict[str, pd.DataFrame],
    *,
    n_slots: int = 12,
) -> pd.DataFrame:
    if set(run_dfs.keys()) != set(DEFAULT_RUN_NAMES):
        raise ValueError("Joint cluster metrics currently expect nsga_GP and nsga_SP.")

    from .descriptors import _parse_chromosome_matrix

    parsed_runs: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for run_name, df in run_dfs.items():
        parsed_runs[run_name] = _parse_chromosome_matrix(df)

    combined_matrix = np.vstack([matrix for matrix, _ in parsed_runs.values()])
    mean, vt, explained_ratio = _pca_fit_transform_unique(combined_matrix)
    projected_runs: dict[str, tuple[np.ndarray, pd.DataFrame]] = {
        run_name: ((matrix - mean) @ vt.T, aligned_df)
        for run_name, (matrix, aligned_df) in parsed_runs.items()
    }

    gp_pcs, gp_df = projected_runs["nsga_GP"]
    sp_pcs, sp_df = projected_runs["nsga_SP"]
    gp_gens = np.sort(pd.to_numeric(gp_df["generation"], errors="coerce").dropna().unique().astype(int))
    sp_gens = np.sort(pd.to_numeric(sp_df["generation"], errors="coerce").dropna().unique().astype(int))
    progress_grid = np.linspace(0.0, 1.0, n_slots)

    rows = []
    for progress in progress_grid:
        gp_target = int(round(progress * gp_gens[-1])) if gp_gens.size else 0
        sp_target = int(round(progress * sp_gens[-1])) if sp_gens.size else 0
        gp_gen = int(gp_gens[np.argmin(np.abs(gp_gens - gp_target))])
        sp_gen = int(sp_gens[np.argmin(np.abs(sp_gens - sp_target))])
        gp_points = gp_pcs[gp_df["generation"].to_numpy() == gp_gen]
        sp_points = sp_pcs[sp_df["generation"].to_numpy() == sp_gen]
        if gp_points.size == 0 or sp_points.size == 0:
            continue
        n_components = min(gp_points.shape[1], sp_points.shape[1], vt.shape[0])
        gp_all = gp_points[:, :n_components]
        sp_all = sp_points[:, :n_components]
        component_weights = explained_ratio[:n_components]
        rows.append(
            {
                "progress": float(progress),
                "generation_nsga_GP": gp_gen,
                "generation_nsga_SP": sp_gen,
                "n_nsga_GP": int(gp_all.shape[0]),
                "n_nsga_SP": int(sp_all.shape[0]),
                "frechet_all_pcs": _frechet_distance_gaussians(gp_all, sp_all),
                "centroid_distance_all_pcs_weighted": _weighted_centroid_distance(gp_all, sp_all, component_weights),
                "explained_pc1": float(explained_ratio[0]) if explained_ratio.size > 0 else 0.0,
                "explained_pc2": float(explained_ratio[1]) if explained_ratio.size > 1 else 0.0,
                "explained_pc3": float(explained_ratio[2]) if explained_ratio.size > 2 else 0.0,
                "explained_pc4": float(explained_ratio[3]) if explained_ratio.size > 3 else 0.0,
            }
        )

    return pd.DataFrame(rows)

def _comparison_run_label(run_name: str) -> str:
    labels = {
        "nsga_GP": "Platform\nIndependent",
        "nsga_SP": "Platform\nDependent",
    }
    return labels.get(run_name, run_name)

def _comparison_subset(df: pd.DataFrame, subset_mode: str) -> pd.DataFrame:
    work_df = filter_sentinels(filter_to_agg(df), reference_df=df)
    if work_df.empty:
        raise ValueError("No valid rows available for comparative subset.")

    generations = pd.to_numeric(work_df["generation"], errors="coerce")
    work_df = work_df.loc[np.isfinite(generations)].copy()
    if work_df.empty:
        raise ValueError("No valid generations available for comparative subset.")

    if subset_mode == "all_generations":
        return work_df.copy()

    if subset_mode == "global_pareto_front":
        fitness_df = work_df.reindex(columns=FITNESS_COLUMNS).apply(pd.to_numeric, errors="coerce")
        points = fitness_df.to_numpy(dtype=float)
        pareto_full = pareto_mask_finite(points)
        if np.any(pareto_full):
            return work_df.loc[pareto_full].copy()
        return work_df.copy()

    final_generation = int(generations.max())
    final_df = work_df.loc[generations == final_generation].copy()
    if final_df.empty:
        raise ValueError("No final-generation rows available for comparative subset.")

    if subset_mode == "final_generation_all":
        return final_df

    if subset_mode == "final_generation_pareto":
        if "is_pareto" in final_df.columns:
            pareto_mask = pd.to_numeric(final_df["is_pareto"], errors="coerce").fillna(0.0) > 0.0
            if pareto_mask.any():
                return final_df.loc[pareto_mask].copy()
        return final_df

    raise ValueError(f"Unknown comparison subset mode: {subset_mode}")

def _comparison_subset_by_run(run_dfs: dict[str, pd.DataFrame], subset_mode: str) -> dict[str, pd.DataFrame]:
    return {run_name: _comparison_subset(df, subset_mode) for run_name, df in run_dfs.items()}

def _comparison_subset_title(subset_mode: str) -> str:
    titles = {
        "all_generations": "All valid generations",
        "global_pareto_front": "Global Pareto Front",
        "final_generation_all": "Final generation, all valid individuals",
        "final_generation_pareto": "Final Pareto Fronts",
    }
    if subset_mode not in titles:
        raise ValueError(f"Unknown comparison subset mode: {subset_mode}")
    return titles[subset_mode]

def _pooled_scale(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    var_a = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    denom = max(a.size + b.size - 2, 1)
    pooled_var = (((a.size - 1) * var_a) + ((b.size - 1) * var_b)) / denom
    if not np.isfinite(pooled_var) or pooled_var <= 1e-12:
        scale = float(np.nanstd(np.concatenate([a, b]), ddof=1)) if (a.size + b.size) > 2 else 0.0
        return scale if scale > 1e-12 else float("nan")
    return float(np.sqrt(pooled_var))


def _two_sample_distribution_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float | int | str]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    pooled = _pooled_scale(a, b)
    median_a = float(np.nanmedian(a)) if a.size else float("nan")
    median_b = float(np.nanmedian(b)) if b.size else float("nan")
    median_delta = median_b - median_a if np.isfinite(median_a) and np.isfinite(median_b) else float("nan")
    effect = median_delta / pooled if np.isfinite(median_delta) and np.isfinite(pooled) and abs(pooled) > 1e-12 else float("nan")
    p_value = float("nan")
    test_name = "mannwhitneyu_two_sided"
    if a.size and b.size:
        if np.array_equal(a, b) or np.allclose(np.nanmedian(a), np.nanmedian(b), equal_nan=True) and np.nanstd(np.concatenate([a, b])) <= 1e-12:
            p_value = 1.0
        elif mannwhitneyu is not None:
            try:
                result = mannwhitneyu(a, b, alternative="two-sided", method="auto")
                p_value = float(result.pvalue)
            except ValueError:
                p_value = 1.0
    return {
        "n_nsga_GP": int(a.size),
        "n_nsga_SP": int(b.size),
        "median_nsga_GP": median_a,
        "median_nsga_SP": median_b,
        "median_delta_sp_minus_gp": median_delta,
        "standardized_delta": effect,
        "p_value": p_value,
        "test_name": test_name,
    }


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)
    if not np.any(finite_mask):
        return q_values
    finite_vals = p_values[finite_mask]
    order = np.argsort(finite_vals)
    ranked = finite_vals[order]
    m = float(ranked.size)
    adjusted = np.empty_like(ranked)
    prev = 1.0
    for idx in range(ranked.size - 1, -1, -1):
        rank = idx + 1.0
        candidate = min(prev, ranked[idx] * m / rank)
        adjusted[idx] = candidate
        prev = candidate
    unsorted = np.empty_like(adjusted)
    unsorted[order] = adjusted
    q_values[finite_mask] = unsorted
    return q_values


def _format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    return f"{p_value:.4f}"

def _get_gene_names(n_genes: int) -> list[str]:
    gene_cols = [f"g{i}" for i in range(n_genes)]
    try:
        from morph_evolution.chromosome_drone import Chromosome_Drone
    except Exception:
        import sys
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from morph_evolution.chromosome_drone import Chromosome_Drone

    param_names = [param.name for param in Chromosome_Drone.PARAMS]
    if len(param_names) == n_genes:
        gene_cols = param_names
    return gene_cols

def pareto_mask(points: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    mask = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not mask[i]:
            continue
        point = points[i]
        dominates = np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        if np.any(dominates):
            mask[i] = False
            continue
        dominated = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        mask[dominated] = False
        mask[i] = True
    return mask

def pareto_mask_finite(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    finite_mask = np.isfinite(points).all(axis=1)
    if not np.any(finite_mask):
        return np.zeros(points.shape[0], dtype=bool)
    mask = np.zeros(points.shape[0], dtype=bool)
    mask[finite_mask] = pareto_mask(points[finite_mask])
    return mask

def _non_dominated_sort_ranks(points: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    if n_points == 0:
        return np.array([], dtype=int)

    domination_counts = np.zeros(n_points, dtype=int)
    dominates_list = [[] for _ in range(n_points)]
    fronts: list[list[int]] = [[]]

    for i in range(n_points):
        for j in range(i + 1, n_points):
            i_dominates_j = np.all(points[i] >= points[j]) and np.any(points[i] > points[j])
            j_dominates_i = np.all(points[j] >= points[i]) and np.any(points[j] > points[i])

            if i_dominates_j:
                dominates_list[i].append(j)
                domination_counts[j] += 1
            elif j_dominates_i:
                dominates_list[j].append(i)
                domination_counts[i] += 1

    for i in range(n_points):
        if domination_counts[i] == 0:
            fronts[0].append(i)

    ranks = np.zeros(n_points, dtype=int)
    current_rank = 1
    current_front = fronts[0]
    while current_front:
        next_front: list[int] = []
        for idx in current_front:
            ranks[idx] = current_rank
            for dominated_idx in dominates_list[idx]:
                domination_counts[dominated_idx] -= 1
                if domination_counts[dominated_idx] == 0:
                    next_front.append(dominated_idx)
        current_front = next_front
        current_rank += 1

    return ranks

def _genealogy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"generation", "uid", "parent_uid_a", "parent_uid_b"}
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns for genealogy plots: {missing_list}")

    work_df = df.copy()
    finite_mask = np.isfinite(work_df[FITNESS_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    work_df = work_df.loc[finite_mask].copy()
    work_df = work_df.sort_values(["generation", "uid"]).drop_duplicates(subset="uid", keep="first")
    if work_df.empty:
        raise ValueError("No finite rows available for genealogy plots.")

    work_df["uid"] = work_df["uid"].astype(int)
    work_df["generation"] = work_df["generation"].astype(int)
    work_df["parent_uid_a"] = work_df["parent_uid_a"].fillna(-1).astype(int)
    work_df["parent_uid_b"] = work_df["parent_uid_b"].fillna(-1).astype(int)
    if "pareto_rank" in work_df.columns:
        work_df["pareto_rank"] = pd.to_numeric(work_df["pareto_rank"], errors="coerce").fillna(0).astype(int)
    else:
        work_df["pareto_rank"] = 0
    return work_df

def _founder_contribution_table(genealogy_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    founders = genealogy_df.loc[
        (genealogy_df["generation"] == int(genealogy_df["generation"].min()))
        | ((genealogy_df["parent_uid_a"] < 0) & (genealogy_df["parent_uid_b"] < 0)),
        "uid",
    ].drop_duplicates().astype(int).tolist()
    if not founders:
        raise ValueError("No founder individuals available for genealogy plots.")

    founder_index = {uid: idx for idx, uid in enumerate(founders)}
    uid_to_vec: dict[int, np.ndarray] = {}
    for row in genealogy_df.itertuples(index=False):
        uid = int(row.uid)
        parent_ids = [int(row.parent_uid_a), int(row.parent_uid_b)]
        valid_parents = [pid for pid in parent_ids if pid >= 0 and pid in uid_to_vec]
        if not valid_parents:
            vec = np.zeros(len(founders), dtype=float)
            vec[founder_index[uid]] = 1.0
            uid_to_vec[uid] = vec
            continue
        vec = np.zeros(len(founders), dtype=float)
        weight = 1.0 / float(len(valid_parents))
        for pid in valid_parents:
            vec += uid_to_vec[pid] * weight
        total = float(np.sum(vec))
        if total > 0.0:
            vec /= total
        uid_to_vec[uid] = vec

    contrib_columns = [f"founder_uid_{uid}" for uid in founders]
    contrib_df = pd.DataFrame.from_dict(uid_to_vec, orient="index", columns=contrib_columns)
    contrib_df.index.name = "uid"
    return contrib_df, founders

def _primary_parent_forest(genealogy_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, list[int]], dict[int, int]]:
    parent_map: dict[int, int] = {}
    for row in genealogy_df.itertuples(index=False):
        uid = int(row.uid)
        parent_uid = int(row.parent_uid_a) if int(row.parent_uid_a) >= 0 else int(row.parent_uid_b)
        parent_map[uid] = parent_uid if parent_uid >= 0 else -1

    founder_map: dict[int, int] = {}

    def founder_of(uid: int) -> int:
        if uid in founder_map:
            return founder_map[uid]
        parent_uid = parent_map.get(uid, -1)
        if parent_uid < 0:
            founder_map[uid] = uid
            return uid
        founder_map[uid] = founder_of(parent_uid)
        return founder_map[uid]

    for uid in parent_map:
        founder_of(uid)

    children_map: dict[int, list[int]] = {uid: [] for uid in parent_map}
    for uid, parent_uid in parent_map.items():
        if parent_uid >= 0 and parent_uid in children_map:
            children_map[parent_uid].append(uid)
    for uid in children_map:
        children_map[uid] = sorted(children_map[uid])

    return parent_map, children_map, founder_map

def _descendant_counts_from_children(children_map: dict[int, list[int]]) -> pd.Series:
    memo: dict[int, set[int]] = {}

    def collect_descendants(uid: int) -> set[int]:
        if uid in memo:
            return memo[uid]
        descendants: set[int] = set()
        for child_uid in children_map.get(uid, []):
            descendants.add(child_uid)
            descendants |= collect_descendants(child_uid)
        memo[uid] = descendants
        return descendants

    counts = {uid: len(collect_descendants(int(uid))) for uid in children_map}
    return pd.Series(counts, name="descendant_count", dtype=float)

def _scalar_fitness_proxy(df: pd.DataFrame) -> pd.Series:
    points = df[FITNESS_COLUMNS].to_numpy(dtype=float).copy()
    points[:, 1] = -points[:, 1]
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    ranges = np.where((maxs - mins) == 0.0, 1.0, maxs - mins)
    norm = np.clip((points - mins) / ranges, 0.0, 1.0)
    return pd.Series(np.mean(norm, axis=1), index=df.index, dtype=float, name="fitness_proxy")

def _normalize_points(points: np.ndarray, mins: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    safe_ranges = np.where(ranges == 0.0, 1.0, ranges)
    norm = (points - mins) / safe_ranges
    return np.clip(norm, 0.0, 1.0)

def _crowding_distances(points: np.ndarray) -> np.ndarray:
    n_points, n_dims = points.shape
    if n_points == 0:
        return np.array([])
    distances = np.zeros(n_points, dtype=float)
    for dim in range(n_dims):
        order = np.argsort(points[:, dim])
        distances[order[0]] += 0.0
        distances[order[-1]] += 0.0
        for i in range(1, n_points - 1):
            prev_val = points[order[i - 1], dim]
            next_val = points[order[i + 1], dim]
            distances[order[i]] += next_val - prev_val
    return distances

def _nearest_neighbor_stats(points: np.ndarray) -> tuple[float, float]:
    n_points = points.shape[0]
    if n_points < 2:
        return 0.0, 0.0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    mean_nn = float(np.mean(nearest))
    spacing = float(np.sqrt(np.mean((nearest - mean_nn) ** 2)))
    max_gap = float(np.max(nearest))
    return spacing, max_gap

def _mean_pairwise_distance(points: np.ndarray) -> float:
    n_points = points.shape[0]
    if n_points < 2:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    upper = dist[np.triu_indices(n_points, k=1)]
    return float(np.mean(upper)) if upper.size else 0.0

def _entropy_simpson(points: np.ndarray, bins: int = 8) -> tuple[float, float]:
    n_points = points.shape[0]
    if n_points == 0:
        return 0.0, 0.0
    hist, _ = np.histogramdd(points, bins=bins, range=[(0, 1)] * points.shape[1])
    counts = hist.flatten()
    total = np.sum(counts)
    if total == 0:
        return 0.0, 0.0
    probs = counts[counts > 0] / total
    entropy = -np.sum(probs * np.log2(probs))
    simpson = 1.0 - np.sum(probs ** 2)
    return float(entropy), float(max(simpson, 0.0))

def _min_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.array([])
    diff = a[:, None, :] - b[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return np.min(dist, axis=1)

def _hypervolume(points: np.ndarray, grid_res: int = 25) -> float:
    n_points = points.shape[0]
    if n_points == 0:
        return 0.0
    grid = np.linspace(0.0, 1.0, grid_res)
    mesh = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    covered = np.zeros(mesh.shape[0], dtype=bool)
    for point in points:
        covered |= np.all(mesh <= point, axis=1)
    return float(np.mean(covered))

def compute_evolutionary_metrics(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    all_points = df[FITNESS_COLUMNS].to_numpy()
    if all_points.size == 0:
        raise ValueError("No points available to compute evolutionary metrics.")

    all_points = all_points[np.isfinite(all_points).all(axis=1)]
    if all_points.size == 0:
        raise ValueError("No finite points available to compute evolutionary metrics.")
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    ranges = maxs - mins

    global_front = all_points[pareto_mask(all_points)]
    global_front_norm = _normalize_points(global_front, mins, ranges)

    rows = []
    for generation in sorted(df["generation"].unique()):
        gen_df = df[df["generation"] == generation]
        points = gen_df[FITNESS_COLUMNS].to_numpy()
        points = points[np.isfinite(points).all(axis=1)]
        n_points = int(points.shape[0])
        if n_points == 0:
            rows.append(
                dict(
                    generation=generation,
                    hypervolume=0.0,
                    spacing=0.0,
                    max_gap=0.0,
                    crowding_mean=0.0,
                    diversity=0.0,
                    entropy=0.0,
                    simpson=0.0,
                    gd=0.0,
                    igd=0.0,
                    n_points=0,
                    n_front_points=0,
                )
            )
            continue

        front_points = points[pareto_mask(points)]
        front_norm = _normalize_points(front_points, mins, ranges)

        crowding = _crowding_distances(front_norm)
        crowding_mean = float(np.mean(crowding)) if crowding.size else 0.0
        spacing, max_gap = _nearest_neighbor_stats(front_norm)
        diversity = _mean_pairwise_distance(front_norm)
        entropy, simpson = _entropy_simpson(front_norm)

        gd_vals = _min_distances(front_norm, global_front_norm)
        igd_vals = _min_distances(global_front_norm, front_norm)
        gd = float(np.mean(gd_vals)) if gd_vals.size else 0.0
        igd = float(np.mean(igd_vals)) if igd_vals.size else 0.0

        hypervolume_norm = _hypervolume(front_norm)
        hypervolume = float(hypervolume_norm * np.prod(np.where(ranges == 0.0, 1.0, ranges)))

        rows.append(
            dict(
                generation=generation,
                hypervolume=hypervolume,
                spacing=spacing,
                max_gap=max_gap,
                crowding_mean=crowding_mean,
                diversity=diversity,
                entropy=entropy,
                simpson=simpson,
                gd=gd,
                igd=igd,
                n_points=n_points,
                n_front_points=int(front_points.shape[0]),
            )
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df

def report_duplicate_stats(df: pd.DataFrame) -> None:
    genome_series = _extract_genome_name_series(df)
    if genome_series.isna().all():
        raise ValueError("Missing 'exp_name'/'rep_exp_names' columns for genome parsing.")
    missing_genome = int(genome_series.isna().sum())
    if missing_genome:
        print(f"Rows without genome in exp_name: {missing_genome}")

    df = df.copy()
    df["genome_name"] = genome_series
    df = df[df["genome_name"].notna()]
    dup_mask = df.duplicated(subset=["genome_name"], keep=False)
    dup_count = int(dup_mask.sum())
    print(f"Duplicate individuals (by genome filename): {dup_count}")
    if dup_count == 0:
        return
    dup_df = df.loc[dup_mask, ["genome_name", "generation", *FITNESS_COLUMNS]].copy()
    grouped = dup_df.groupby("genome_name", sort=False)
    counts = grouped.size().rename("n_evals")
    means = grouped[FITNESS_COLUMNS].mean().add_prefix("mean_")
    stds = grouped[FITNESS_COLUMNS].std().fillna(0.0).add_prefix("std_")
    generations = grouped["generation"].apply(
        lambda series: ",".join(str(val) for val in sorted(set(series)))
    ).rename("generations")
    summary = pd.concat([counts, means, stds, generations], axis=1)
    summary = summary.loc[summary["n_evals"] > 1]
    print("Duplicate genomes (evaluated >1):")
    print(summary.to_string(float_format="%.6f"))

def _extract_urdf_name(df: pd.DataFrame) -> pd.Series:
    if "exp_name" not in df.columns and "rep_exp_names" not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype="object")

    def _extract(row: pd.Series) -> str | None:
        for col in ("exp_name", "rep_exp_names"):
            if col not in row:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            name = str(val).split("|")[0]
            match = re.search(r"\[[^\]]+\]", name)
            if match:
                return match.group(0)
            if ".urdf" in name:
                return Path(name).stem
            if name:
                return name
        return None

    return df.apply(_extract, axis=1)

def export_pareto_csv(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    points = df[FITNESS_COLUMNS].to_numpy()
    if points.size == 0:
        raise ValueError("No points available to compute Pareto front.")
    pareto_mask_full = pareto_mask_finite(points)
    if not np.any(pareto_mask_full):
        raise ValueError("No finite points available to compute Pareto front.")
    pareto_df = df.loc[pareto_mask_full].copy()
    pareto_df["urdf_name"] = _extract_urdf_name(pareto_df)
    cols = ["urdf_name", "exp_name", "generation", *FITNESS_COLUMNS]
    existing_cols = [c for c in cols if c in pareto_df.columns]
    pareto_df = pareto_df[existing_cols].sort_values(["generation", "urdf_name"])
    pareto_df.to_csv(output_path, index=False)
    return pareto_df

def export_objective_rankings(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        raise ValueError("No points available to export objective rankings.")

    export_df = df.copy()
    export_df["urdf_name"] = _extract_urdf_name(export_df)
    export_df["speed"] = pd.to_numeric(export_df["ff_0"], errors="coerce")
    export_df["cost_of_transport"] = -pd.to_numeric(export_df["ff_1"], errors="coerce")
    export_df["progress"] = pd.to_numeric(export_df["ff_2"], errors="coerce")

    preferred_cols = [
        "urdf_name",
        "exp_name",
        "generation",
        "speed",
        "cost_of_transport",
        "progress",
        *FITNESS_COLUMNS,
    ]
    extra_cols = [col for col in export_df.columns if col not in preferred_cols]
    ordered_cols = [col for col in preferred_cols if col in export_df.columns] + extra_cols
    export_df = export_df[ordered_cols]

    ranking_specs = [
        ("speed", "drones_by_speed.csv", False),
        ("ff_1", "drones_by_efficiency.csv", False),
        ("progress", "drones_by_progress.csv", False),
    ]
    for sort_col, filename, ascending in ranking_specs:
        ranked_df = export_df.sort_values(
            by=[sort_col, "generation", "urdf_name"],
            ascending=[ascending, True, True],
            na_position="last",
        ).reset_index(drop=True)
        ranked_df.insert(0, "rank", np.arange(1, len(ranked_df) + 1, dtype=int))
        (output_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        ranked_df.to_csv(output_dir / filename, index=False)
__all__ = [name for name in globals() if not name.startswith("__")]

