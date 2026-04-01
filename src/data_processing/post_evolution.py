from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from post_evolution_utils.common import (
    COMPARISON_DIR_NAME,
    DEFAULT_RUN_NAMES,
    _analysis_csv_path,
    _baseline_drone_steps90_ratio_for_run,
    apply_invalid_repetition_values,
    compute_evolutionary_metrics,
    compute_joint_pca_cluster_metrics,
    compute_mean_runtime_per_individual_by_generation,
    compute_steps90_summary_by_generation,
    compute_training_speed90_summary_by_generation,
    export_objective_rankings,
    export_pareto_csv,
    filter_sentinels,
    filter_to_agg,
    load_nsga_csv,
    load_run_tables,
    report_duplicate_stats,
    resolve_bix3_point,
)
from post_evolution_utils.plots_aero import (
    plot_aero_archetype_heatmap,
    plot_aero_parameter_evolution,
    plot_aero_parameter_top10pct_by_fitness,
    plot_comparative_dihedral_stability_margin,
    plot_comparative_top10pct_aero_distributions,
    plot_nose_cg_minus_nose_wing_ac_top10_by_fitness,
)
from post_evolution_utils.plots_comparison import (
    plot_comparative_evolutionary_metrics,
    plot_comparative_gene_distributions,
    plot_comparative_pareto_fronts_2d,
    plot_comparative_runtime_per_individual,
    plot_comparative_steps90_by_generation,
    plot_genome_pca_generations,
    plot_joint_genome_pca_generations,
    plot_joint_pca_cluster_metrics,
)
from post_evolution_utils.plots_fronts import (
    plot_dominance_solutions,
    plot_evolutionary_metrics,
    plot_fitness_trends,
    plot_fractional_lineage_muller,
    plot_genome_fitness_correlation,
    plot_newcomer_rank_distribution,
    plot_pareto_front_3d,
    plot_pareto_fronts,
    plot_pareto_rank_distribution,
    plot_phylogenetic_tree_like,
    plot_top5pct_gene_means,
    plot_two_parent_genealogy,
)
from post_evolution_utils.plots_learning import (
    plot_bix3_beating_curves,
    plot_energy_vs_learnability,
    plot_innovation_payoff,
    plot_lineage_takeover_metrics,
    plot_steps90_by_generation,
    plot_training_speed90_by_generation,
)

SINGLE_RUN_SUBDIRS = {
    'overview': 'overview',
    'pareto': 'pareto',
    'metrics': 'metrics',
    'learning': 'learning',
    'aero': 'aero',
    'selection': 'selection',
    'lineage': 'lineage',
    'genome': 'genome',
    'tables': 'tables',
}

COMPARISON_SUBDIRS = {
    'metrics': 'metrics',
    'pareto': 'pareto',
    'genome': 'genome',
    'aero': 'aero',
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate post-evolution plots from legacy nsga.csv or a new NSGA run directory.'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data_processing/nsga_GP'),
        help=(
            'Path to a legacy nsga.csv, to population_history.csv, to an analysis directory, '
            'or to the root folder of a new NSGA run.'
        ),
    )
    parser.add_argument('--run-dir', dest='input', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--csv', dest='input', type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        '--tag',
        type=str,
        default=None,
        help='Optional tag to name the output folder as post_evolution_plots_<tag>.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Optional output directory. Default: <data-processing>/outputs/evolution GP|SP[/<tag>].',
    )
    parser.add_argument(
        '--generation',
        type=int,
        default=None,
        help='Generation to use for Pareto fronts (default: last generation).',
    )
    return parser.parse_args(argv)


def _default_data_processing_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_outputs_dir() -> Path:
    return _default_data_processing_dir() / 'outputs'


def _default_output_run_name(run_name: str) -> str:
    mapping = {
        'nsga_GP': 'evolution GP',
        'nsga_SP': 'evolution SP',
    }
    return mapping.get(run_name, run_name)


def _default_run_dir(run_name: str) -> Path:
    return _default_data_processing_dir() / run_name


def _comparison_output_dir() -> Path:
    return _default_outputs_dir() / COMPARISON_DIR_NAME


def _prepare_output_tree(base_dir: Path, subdirs: dict[str, str]) -> dict[str, Path]:
    tree: dict[str, Path] = {}
    for key, name in subdirs.items():
        path = base_dir / name
        path.mkdir(parents=True, exist_ok=True)
        tree[key] = path
    return tree


def _cleanup_generated_output_files(base_dir: Path) -> None:
    if not base_dir.exists():
        return
    for path in sorted(base_dir.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.csv', '.txt'}:
            path.unlink()



def process_single_run(
    input_path: Path,
    *,
    tag: str | None = None,
    output_dir: Path | None = None,
    generation: int | None = None,
) -> pd.DataFrame:
    df_raw, summary_df, source_csv, run_dir = load_run_tables(input_path)
    bix3_point = resolve_bix3_point(run_dir)
    df_agg_raw = filter_to_agg(df_raw)
    df = filter_sentinels(df_agg_raw, reference_df=df_raw)
    trends_raw = apply_invalid_repetition_values(df_agg_raw, reference_df=df_raw)
    nsga_generated_df = df_agg_raw
    nsga_generated_valid = df
    nsga_csv_path = source_csv.parent / 'nsga.csv'
    if nsga_csv_path.exists():
        nsga_full_df = load_nsga_csv(nsga_csv_path)
        nsga_generated_df = filter_to_agg(nsga_full_df)
        nsga_generated_valid = filter_sentinels(nsga_generated_df, reference_df=nsga_full_df)

    if output_dir is not None:
        output_dir = output_dir.expanduser().resolve()
    else:
        tag = tag.strip() if isinstance(tag, str) and tag.strip() else ''
        safe_tag = re.sub(r'[^A-Za-z0-9._-]+', '_', tag).strip('_')
        output_dir = _default_outputs_dir() / _default_output_run_name(run_dir.name)
        if safe_tag:
            output_dir = output_dir / safe_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    out = _prepare_output_tree(output_dir, SINGLE_RUN_SUBDIRS)

    print(f'Loaded evolution data from: {source_csv}')
    print(f'Using run directory: {run_dir}')
    report_duplicate_stats(df)
    plot_fitness_trends(
        df,
        out['overview'] / 'fitness_trends_overview.png',
        raw_df=trends_raw,
        summary_df=summary_df,
        generated_valid_df=nsga_generated_valid,
        bix3_point=bix3_point,
    )

    if generation is None:
        generation = int(df['generation'].max())
    plot_pareto_fronts(nsga_generated_valid, out['pareto'] / 'pareto_fronts_2d.png', generation, bix3_point)
    plot_pareto_front_3d(nsga_generated_valid, out['pareto'] / 'pareto_front_3d.html', bix3_point)
    export_pareto_csv(nsga_generated_valid, out['tables'] / 'pareto_front.csv')
    export_objective_rankings(nsga_generated_valid, out['tables'] / 'objective_rankings')
    metrics_df = compute_evolutionary_metrics(df, out['metrics'] / 'evolutionary_metrics.csv')
    plot_evolutionary_metrics(metrics_df, out['metrics'] / 'evolutionary_metrics_over_time.png')
    steps90_df = compute_steps90_summary_by_generation(run_dir)
    baseline_steps90_ratio = _baseline_drone_steps90_ratio_for_run(run_dir)
    plot_steps90_by_generation(
        steps90_df,
        out['learning'] / 'steps_to_90pct_reward_by_generation.png',
        title='Steps to 90% Reward by Generation',
        baseline_ratio=baseline_steps90_ratio,
    )
    if run_dir.name == 'nsga_SP':
        training_speed90_df = compute_training_speed90_summary_by_generation(run_dir)
        plot_training_speed90_by_generation(
            training_speed90_df,
            out['learning'] / 'training_speed_to_90pct_reward_by_generation.png',
            title='Training Speed to 90% Reward by Generation',
        )
    plot_bix3_beating_curves(df, out['learning'] / 'bix3_beating_curves.png', bix3_point)
    plot_energy_vs_learnability(nsga_generated_valid, out['learning'] / 'energy_vs_learnability.png')
    plot_lineage_takeover_metrics(df, out['lineage'] / 'lineage_takeover_metrics.png')
    selection_pool_df = pd.read_csv(_analysis_csv_path(run_dir, 'selection_pool_history.csv'))
    plot_innovation_payoff(selection_pool_df, out['learning'] / 'innovation_payoff.png')
    plot_aero_parameter_evolution(df, out['aero'] / 'aero_parameter_evolution.png')
    plot_aero_parameter_top10pct_by_fitness(df, out['aero'] / 'aero_parameter_top10pct_by_fitness.png')
    plot_nose_cg_minus_nose_wing_ac_top10_by_fitness(
        df,
        out['aero'] / 'nose_cg_minus_nose_wing_ac_top10_by_fitness.png',
    )
    plot_aero_archetype_heatmap(df, out['aero'] / 'aero_archetype_heatmap.png')
    plot_dominance_solutions(nsga_generated_valid, out['selection'] / 'dominance_solutions.png', summary_df=summary_df)
    plot_pareto_rank_distribution(df_agg_raw, out['selection'] / 'pareto_rank_distribution.png')
    plot_newcomer_rank_distribution(nsga_generated_valid, out['selection'] / 'newcomer_rank_distribution.png')
    plot_phylogenetic_tree_like(nsga_generated_valid, out['lineage'] / 'phylogenetic_tree_like.png')
    plot_two_parent_genealogy(nsga_generated_valid, out['lineage'] / 'two_parent_genealogy.png')
    plot_fractional_lineage_muller(nsga_generated_valid, out['lineage'] / 'lineage_muller_fractional.png')
    plot_genome_pca_generations(df, out['genome'] / 'genome_pca_generations_pc1_pc2.png')
    plot_genome_fitness_correlation(df, out['genome'] / 'genome_fitness_correlation.png')
    plot_top5pct_gene_means(df, out['genome'] / 'top5pct_gene_means.png')
    _cleanup_generated_output_files(output_dir)
    return df


def run_default_batch(generation: int | None = None) -> None:
    comparison_dir = _comparison_output_dir()
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_out = _prepare_output_tree(comparison_dir, COMPARISON_SUBDIRS)
    run_dfs: dict[str, pd.DataFrame] = {}
    metrics_by_run: dict[str, pd.DataFrame] = {}
    runtime_by_run: dict[str, pd.DataFrame] = {}
    steps90_by_run: dict[str, pd.DataFrame] = {}
    bix3_points_by_run: dict[str, np.ndarray] = {}
    for run_name in DEFAULT_RUN_NAMES:
        run_dir = _default_run_dir(run_name)
        if not run_dir.exists():
            raise FileNotFoundError(f'Default run directory not found: {run_dir}')
        print(f'\n=== Processing {run_name} ===')
        bix3_points_by_run[run_name] = resolve_bix3_point(run_dir)
        run_dfs[run_name] = process_single_run(run_dir, generation=generation)
        metrics_by_run[run_name] = compute_evolutionary_metrics(
            run_dfs[run_name],
            comparison_dir / f'._tmp_{run_name}_metrics.csv',
        )
        runtime_by_run[run_name] = compute_mean_runtime_per_individual_by_generation(run_dir)
        steps90_by_run[run_name] = compute_steps90_summary_by_generation(run_dir)

    plot_joint_genome_pca_generations(
        run_dfs,
        comparison_out['genome'] / 'joint_genome_pca_generations_pc1_pc2.png',
    )
    plot_joint_genome_pca_generations(
        run_dfs,
        comparison_out['genome'] / 'joint_genome_pca_generations_pc3_pc4.png',
        pc_x=2,
        pc_y=3,
    )
    cluster_metrics_df = compute_joint_pca_cluster_metrics(run_dfs)
    cluster_metrics_df.to_csv(comparison_out['genome'] / 'joint_pca_cluster_metrics.csv', index=False)
    plot_joint_pca_cluster_metrics(cluster_metrics_df, comparison_out['genome'] / 'joint_pca_cluster_metrics.png')
    plot_comparative_evolutionary_metrics(
        metrics_by_run,
        comparison_out['metrics'] / 'comparative_evolutionary_metrics.png',
    )
    plot_comparative_runtime_per_individual(
        runtime_by_run,
        comparison_out['metrics'] / 'comparative_runtime_per_individual.png',
    )
    plot_comparative_steps90_by_generation(
        steps90_by_run,
        comparison_out['metrics'] / 'comparative_steps90_by_generation.png',
    )
    plot_comparative_pareto_fronts_2d(
        run_dfs,
        bix3_points_by_run,
        comparison_out['pareto'] / 'comparative_pareto_fronts_2d.png',
    )
    plot_comparative_gene_distributions(
        run_dfs,
        comparison_out['genome'] / 'comparative_gene_distributions_global_pareto_front.png',
        subset_mode='global_pareto_front',
    )
    plot_comparative_top10pct_aero_distributions(
        run_dfs,
        comparison_out['aero'] / 'comparative_top10pct_aero_distributions_global_pareto_front.png',
        subset_mode='global_pareto_front',
    )
    plot_comparative_dihedral_stability_margin(
        run_dfs,
        comparison_out['aero'] / 'comparative_dihedral_stability_margin_global_pareto_front.png',
        subset_mode='global_pareto_front',
    )
    for run_name in metrics_by_run:
        tmp_path = comparison_dir / f'._tmp_{run_name}_metrics.csv'
        if tmp_path.exists():
            tmp_path.unlink()
    _cleanup_generated_output_files(comparison_dir)
    print(f'Comparison outputs saved to: {comparison_dir}')


def main(argv: list[str] | None = None) -> None:
    cli_args = sys.argv[1:] if argv is None else argv
    if len(cli_args) == 0:
        run_default_batch()
        return

    args = parse_args(cli_args)
    explicit_input = any(flag in cli_args for flag in ('--input', '--run-dir', '--csv'))
    if not explicit_input and args.output_dir is None and args.tag is None:
        run_default_batch(generation=args.generation)
        return

    process_single_run(
        args.input,
        tag=args.tag,
        output_dir=args.output_dir,
        generation=args.generation,
    )


if __name__ == '__main__':
    main()
