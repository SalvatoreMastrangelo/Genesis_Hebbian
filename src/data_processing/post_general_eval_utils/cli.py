from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

from .plotter import URDFHistogramPlotter


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_outputs_dir(data_dir: str) -> Path:
    return Path(data_dir).resolve() / 'outputs' / 'general evaluation'


def _cleanup_generated_output_files(base_dir: Path) -> None:
    if not base_dir.exists():
        return
    for path in sorted(base_dir.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.csv', '.txt'}:
            path.unlink()


def _apply_plot_style() -> None:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'axes.facecolor': '#fbfbfa',
        'axes.edgecolor': '#d4d4d4',
        'axes.linewidth': 0.9,
        'axes.grid': True,
        'grid.color': '#d9d9d9',
        'grid.alpha': 0.28,
        'grid.linewidth': 0.8,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#d4d4d4',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titleweight': 'bold',
    })



def main() -> None:
    parser = argparse.ArgumentParser(description='Post-processing and plotting for GP/SP evaluation.')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/home/andrea/Documents/Genesis/src/data_processing',
        help='Directory containing evaluation_results_*.csv files.',
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: <data-dir>/outputs/general evaluation).',
    )
    parser.add_argument(
        '--gp-idx',
        type=int,
        default=2,
        help='General policy index used by selected-vs-SP plots.',
    )
    parser.add_argument(
        '--drone-idx',
        type=int,
        default=None,
        help='1-based URDF index. If set, generate only the drone picture and exit.',
    )
    parser.add_argument(
        '--urdf-idxs',
        type=int,
        nargs='*',
        default=None,
        help=(
            'Optional 1-based URDF indices for per-URDF outputs. '
            'If omitted, no per-URDF plots or drone pictures are generated.'
        ),
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    csv_path = os.path.join(data_dir, 'evaluation_general_SP.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_dir, 'evaluation_results_SP.csv')
    gp_csv_paths = sorted(
        [
            os.path.join(data_dir, name)
            for name in os.listdir(data_dir)
            if re.match(r'^evaluation_results_GP.*\.csv$', name)
        ]
    )
    plotter = URDFHistogramPlotter(csv_path, gp_csv_paths=gp_csv_paths)
    requested_general_policy_idx = int(args.gp_idx)
    general_policy_idx = plotter._resolve_policy_index(
        requested_general_policy_idx,
        plotter.general_policy_count,
        'General',
    )
    save_dir = Path(
        os.path.abspath(args.save_dir)
        if args.save_dir is not None
        else str(_default_outputs_dir(data_dir))
    )
    _ensure_dir(save_dir)
    _apply_plot_style()

    overview_dir = _ensure_dir(save_dir / 'overview')
    means_dir = _ensure_dir(save_dir / 'means')
    distributions_dir = _ensure_dir(save_dir / 'distributions')
    correlations_dir = _ensure_dir(save_dir / 'correlations')
    urdf_dir = _ensure_dir(save_dir / 'per_urdf')

    if args.drone_idx is not None:
        plotter.render_single_drone_picture(drone_idx=args.drone_idx, save_dir=str(urdf_dir))
        raise SystemExit(0)

    plotter.plot(
        general_policy_idx=general_policy_idx,
        save_path=str(overview_dir / 'gp_vs_sp_summary.png'),
        show=False,
    )
    plotter.plot_bix3_vs_mean(
        save_path=str(overview_dir / 'bix3_vs_population_mean.png'),
        general_policy_idx=general_policy_idx,
        show=False,
    )
    selected_urdfs = plotter._normalize_urdf_indices(args.urdf_idxs)
    if selected_urdfs is not None:
        plotter.plot_per_urdf_policies(
            save_dir=str(urdf_dir),
            show=False,
            urdf_indices=selected_urdfs,
        )
        plotter.plot_reward_evolution_per_urdf(
            save_dir=str(urdf_dir),
            show=False,
            urdf_indices=selected_urdfs,
        )
        for urdf_idx in sorted(selected_urdfs):
            plotter.render_single_drone_picture(drone_idx=urdf_idx, save_dir=str(urdf_dir))
    plotter.plot_mean_policies(
        save_path=str(means_dir / 'mean_metrics_gp_vs_sp.png'),
        show=False,
    )
    plotter.plot_mean_policies(
        save_path=str(means_dir / 'mean_metrics_gp_vs_sp_above_min_progress.png'),
        show=False,
        only_above_minimal_progress=True,
    )
    plotter.plot_mean_policies(
        save_path=str(means_dir / 'mean_metrics_gp_vs_sp_above_300m_progress.png'),
        show=False,
        only_above_minimal_progress=True,
        progress_threshold_m=300.0,
    )
    plotter.plot_selected_gp_vs_sp_distribution(
        save_path=str(distributions_dir / f'gp{general_policy_idx}_vs_sp_mean_distribution.png'),
        general_policy_idx=general_policy_idx,
        show=False,
    )
    plotter.plot_selected_gp_vs_sp_distribution(
        save_path=str(distributions_dir / f'gp{general_policy_idx}_vs_sp_all_seeds_distribution.png'),
        general_policy_idx=general_policy_idx,
        show=False,
        sp_use_all_repetitions=True,
    )
    plotter.plot_mean_sp_policies(
        save_path=str(means_dir / 'mean_metrics_sp_only.png'),
        show=False,
    )
    plotter.plot_mean_delta_vs_trained(
        save_path=str(means_dir / 'gp_vs_sp_mean_relative_delta.png'),
        show=False,
    )
    genome_delta_dir = _ensure_dir(correlations_dir / f'genome_delta_gp{general_policy_idx}')
    plotter.plot_genome_deviation_correlation(
        general_policy_idx=general_policy_idx,
        save_path=str(genome_delta_dir / f'genome_vs_gp{general_policy_idx}_delta_correlation.png'),
        save_dir=str(genome_delta_dir),
        show=False,
    )
    plotter.plot_genome_specialized_correlation(
        save_path=str(correlations_dir / 'genome_vs_sp_correlation.png'),
        show=False,
    )
    plotter.plot_genome_general_policy_correlation(
        general_policy_idx=general_policy_idx,
        save_path=str(correlations_dir / f'genome_vs_gp{general_policy_idx}_correlation.png'),
        show=False,
    )
    plotter.plot_policy_reward_metric_correlation(
        save_path=str(correlations_dir / 'policy_reward_metric_correlation.png'),
        show=False,
    )
    plotter.plot_selected_gp_sp_mean_reward_metric_correlation(
        general_policy_idx=general_policy_idx,
        save_path=str(correlations_dir / f'gp{general_policy_idx}_vs_sp_mean_reward_metric_correlation.png'),
        show=False,
    )
    plotter.plot_sp_metric_vs_steps_correlation_table(
        save_path=str(correlations_dir / 'sp_metric_vs_steps_correlation_table.png'),
        show=False,
    )
    gp_steps_dir = _ensure_dir(correlations_dir / f'gp{general_policy_idx}_vs_sp_steps')
    plotter.plot_gp1_sp_mismatch_vs_sp_steps(
        general_policy_idx=general_policy_idx,
        save_dir=str(gp_steps_dir),
        show=False,
    )
    _cleanup_generated_output_files(save_dir)


if __name__ == '__main__':
    main()
