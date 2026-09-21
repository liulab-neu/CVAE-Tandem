import argparse
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FixedLocator, MaxNLocator

from dual_peak_utils import (
    BASE_DIR,
    bounded_double_lorentzian,
    build_cvae,
    build_tandem_runtime,
    case_tag,
    detect_two_resonances,
    load_wavelength_grid,
    peak_error_metrics,
    resonance_quality_metrics,
    seed_everything,
    select_cvae_assisted_design,
    sorted_peak_pair,
)


matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.0,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})

FIG_DPI = 600
SINGLE_COLUMN_SIZE = (3.5, 2.7)
PPT_BLUE = '#2E75B6'
PPT_ORANGE = '#ED7D31'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Run direct Tandem and/or CVAE-assisted Tandem for a two-'
            'resonance target and save paper-ready plots, data, results, '
            'and structures.'
        )
    )
    parser.add_argument('--peak1', type=float, default=430.0)
    parser.add_argument('--peak2', type=float, default=620.0)
    parser.add_argument(
        '--method',
        choices=('both', 'tandem', 'cvae'),
        default='both',
    )
    parser.add_argument('--gamma', type=float, default=15.0)
    parser.add_argument(
        '--n-candidates',
        type=int,
        default=20,
        help=(
            'Total CVAE candidates, all decoded from the wavelength-sorted '
            'condition the CVAE was trained on (default: 20).'
        ),
    )
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument(
        '--position-slack-nm',
        type=float,
        default=2.5,
        help=(
            'Slack used only to report the diagnostic position window; '
            'it does not affect candidate ranking (default: 2.5 nm).'
        ),
    )
    parser.add_argument(
        '--maximum-valley-transmittance',
        type=float,
        default=0.35,
    )
    parser.add_argument('--minimum-prominence', type=float, default=0.30)
    parser.add_argument(
        '--maximum-valley-imbalance',
        type=float,
        default=0.10,
    )
    parser.add_argument('--minimum-width-ratio', type=float, default=0.50)
    parser.add_argument('--maximum-width-ratio', type=float, default=1.50)
    parser.add_argument('--output-root', default=BASE_DIR)
    args = parser.parse_args(argv)
    if args.gamma <= 0:
        parser.error('--gamma must be positive')
    if args.n_candidates < 1:
        parser.error('--n-candidates must be a positive integer')
    if args.position_slack_nm < 0:
        parser.error('--position-slack-nm must be non-negative')
    if not 0 <= args.maximum_valley_transmittance <= 1:
        parser.error('--maximum-valley-transmittance must be in [0, 1]')
    if args.minimum_prominence <= 0:
        parser.error('--minimum-prominence must be positive')
    if args.maximum_valley_imbalance < 0:
        parser.error('--maximum-valley-imbalance must be non-negative')
    if args.minimum_width_ratio <= 0:
        parser.error('--minimum-width-ratio must be positive')
    if args.maximum_width_ratio <= args.minimum_width_ratio:
        parser.error(
            '--maximum-width-ratio must exceed --minimum-width-ratio'
        )
    sorted_peak_pair(args.peak1, args.peak2)
    return args


def setup_figure():
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN_SIZE)
    ax.tick_params(
        axis='both',
        which='major',
        length=3,
        width=0.7,
        direction='in',
        top=True,
        right=True,
        pad=2,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.xaxis.set_major_locator(FixedLocator([400, 500, 600, 700, 800]))
    return fig, ax


def set_adaptive_y_limits(ax, *curves):
    finite_values = np.concatenate([
        np.asarray(curve, dtype=float).reshape(-1)
        for curve in curves
    ])
    finite_values = finite_values[np.isfinite(finite_values)]
    if not len(finite_values):
        ax.set_ylim(0, 1)
        return

    data_min = float(np.min(finite_values))
    data_max = float(np.max(finite_values))
    data_range = max(data_max - data_min, 0.05)
    lower = max(-0.02, data_min - 0.07 * data_range)
    upper = min(1.02, data_max + 0.07 * data_range)
    step = 0.05
    lower = np.floor(lower / step) * step
    upper = np.ceil(upper / step) * step
    if upper - lower < 0.20:
        midpoint = 0.5 * (upper + lower)
        lower = max(-0.02, midpoint - 0.10)
        upper = min(1.02, midpoint + 0.10)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def save_figure(fig, path):
    fig.tight_layout(pad=0.45)
    fig.savefig(
        path,
        dpi=FIG_DPI,
        bbox_inches='tight',
        pad_inches=0.04,
        facecolor='white',
    )
    plt.close(fig)


def annotate_pair(ax, detection, color):
    if not detection.valid:
        return
    offsets = [(-14, 12), (14, 12)]
    alignments = ['right', 'left']
    for index, (x_value, y_value) in enumerate(
        zip(detection.wavelengths, detection.values)
    ):
        ax.scatter(
            x_value,
            y_value,
            s=17,
            color=color,
            edgecolors='white',
            linewidths=0.5,
            zorder=6,
        )
        ax.annotate(
            f'{x_value:.1f}',
            xy=(x_value, y_value),
            xytext=offsets[index],
            textcoords='offset points',
            ha=alignments[index],
            va='bottom',
            fontsize=6.7,
            color='#333333',
            arrowprops=dict(
                arrowstyle='->',
                color='#555555',
                linewidth=0.55,
                shrinkA=2,
                shrinkB=2,
                mutation_scale=6,
            ),
            annotation_clip=True,
        )


def plot_case(
    wavelengths,
    reference,
    final_response,
    final_detection,
    path,
    reference_label,
):
    fig, ax = setup_figure()
    ax.plot(
        wavelengths,
        reference,
        color=PPT_ORANGE,
        linewidth=1.25,
        linestyle=(0, (4, 2.4)),
        dash_capstyle='round',
        label=reference_label,
        zorder=2,
    )
    ax.plot(
        wavelengths,
        final_response,
        color=PPT_BLUE,
        linewidth=1.4,
        solid_capstyle='round',
        label='Final Tandem response',
        zorder=3,
    )
    ax.set_xlim(380, 800)
    set_adaptive_y_limits(ax, reference, final_response)
    annotate_pair(ax, final_detection, PPT_BLUE)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmittance')
    ax.legend(
        loc='upper right',
        frameon=False,
        handlelength=2.5,
        handletextpad=0.5,
    )
    save_figure(fig, path)


def save_structure(path, structure):
    np.savetxt(
        path,
        np.asarray(structure, dtype=float).reshape(1, 3),
        delimiter='\t',
        fmt='%.10f',
        header='R\tn_host1\tn_host2',
        comments='',
    )


def write_pair_metrics(result_file, prefix, detection, metrics):
    result_file.write(
        f'{prefix}_two_resonances_detected\t{int(detection.valid)}\n'
    )
    result_file.write(
        f'{prefix}_significant_minima_count\t'
        f'{len(detection.all_indices)}\n'
    )
    for peak_index in range(2):
        result_file.write(
            f'{prefix}_peak{peak_index + 1}_min_index\t'
            f'{int(detection.indices[peak_index])}\n'
        )
        result_file.write(
            f'{prefix}_peak{peak_index + 1}_wavelength_nm\t'
            f'{detection.wavelengths[peak_index]:.10f}\n'
        )
        result_file.write(
            f'{prefix}_peak{peak_index + 1}_transmittance\t'
            f'{detection.values[peak_index]:.10e}\n'
        )
        result_file.write(
            f'{prefix}_peak{peak_index + 1}_signed_error_nm\t'
            f'{metrics["signed_errors_nm"][peak_index]:.10f}\n'
        )
        result_file.write(
            f'{prefix}_peak{peak_index + 1}_absolute_error_nm\t'
            f'{metrics["absolute_errors_nm"][peak_index]:.10f}\n'
        )
    for metric_name in (
        'max_absolute_error_nm',
        'pair_rmse_nm',
        'midpoint_error_nm',
        'separation_error_nm',
    ):
        result_file.write(
            f'{prefix}_{metric_name}\t{metrics[metric_name]:.10e}\n'
        )


def write_quality_metrics(result_file, prefix, quality):
    result_file.write(
        f'{prefix}_quality_valid\t{int(quality.valid)}\n'
    )
    for peak_index in range(2):
        result_file.write(
            f'{prefix}_peak{peak_index + 1}_prominence\t'
            f'{quality.prominences[peak_index]:.10e}\n'
        )
        result_file.write(
            f'{prefix}_peak{peak_index + 1}_fwhm_nm\t'
            f'{quality.fwhm_nm[peak_index]:.10f}\n'
        )
        result_file.write(
            f'{prefix}_peak{peak_index + 1}_quality_factor\t'
            f'{quality.quality_factors[peak_index]:.10e}\n'
        )
        result_file.write(
            f'{prefix}_peak{peak_index + 1}_contrast_quality_fom\t'
            f'{quality.contrast_quality_fom[peak_index]:.10e}\n'
        )
    result_file.write(
        f'{prefix}_worst_valley_transmittance\t'
        f'{quality.worst_valley_transmittance:.10e}\n'
    )
    result_file.write(
        f'{prefix}_widest_fwhm_nm\t'
        f'{quality.widest_fwhm_nm:.10f}\n'
    )
    result_file.write(
        f'{prefix}_minimum_prominence\t'
        f'{quality.minimum_prominence:.10e}\n'
    )


def save_candidate_diagnostics(path, diagnostics):
    columns = (
        'candidate_index',
        'condition_order_code',
        'valid',
        'candidate_spectrum_valid',
        'candidate_spectrum_maximum_absolute_error_nm',
        'candidate_spectrum_summed_absolute_error_nm',
        'peak1_wavelength_nm',
        'peak2_wavelength_nm',
        'peak1_signed_error_nm',
        'peak2_signed_error_nm',
        'maximum_absolute_error_nm',
        'summed_absolute_error_nm',
        'mean_absolute_error_nm',
        'peak1_transmittance',
        'peak2_transmittance',
        'worst_valley_transmittance',
        'peak1_fwhm_nm',
        'peak2_fwhm_nm',
        'widest_fwhm_nm',
        'peak1_prominence',
        'peak2_prominence',
        'minimum_prominence',
        'valley_transmittance_imbalance',
        'peak1_fwhm_to_ideal_ratio',
        'peak2_fwhm_to_ideal_ratio',
        'shape_gate_passed',
        'shape_minimax_score',
        'inside_position_window',
        'selected',
    )
    data = np.asarray([
        [row[column] for column in columns]
        for row in diagnostics
    ], dtype=float)
    np.savetxt(
        path,
        data,
        delimiter='\t',
        fmt='%.10e',
        header='\t'.join(columns),
        comments='',
    )


def save_direct_case(
    output_dir,
    wavelengths,
    targets,
    ideal_target,
    structure,
    final_response,
    gamma,
):
    os.makedirs(output_dir, exist_ok=True)
    detection = detect_two_resonances(final_response, wavelengths)
    metrics = peak_error_metrics(detection, targets)
    quality = resonance_quality_metrics(
        final_response,
        wavelengths,
        detection,
    )
    mse = float(np.mean((final_response - ideal_target) ** 2))
    rmse = float(np.sqrt(mse))

    plot_case(
        wavelengths,
        ideal_target,
        final_response,
        detection,
        os.path.join(output_dir, 'tandem_plot.png'),
        'Ideal target',
    )
    np.savetxt(
        os.path.join(output_dir, 'tandem_data.txt'),
        np.column_stack((wavelengths, ideal_target, final_response)),
        delimiter='\t',
        fmt='%.10e',
        header=(
            'wavelength_nm\tideal_target_transmittance\t'
            'tandem_response_transmittance'
        ),
        comments='',
    )
    save_structure(
        os.path.join(output_dir, 'tandem_structure.txt'),
        structure,
    )
    with open(
        os.path.join(output_dir, 'tandem_results.txt'),
        'w',
        encoding='utf-8',
    ) as result_file:
        result_file.write(f'target_peak1_nm\t{targets[0]:.10f}\n')
        result_file.write(f'target_peak2_nm\t{targets[1]:.10f}\n')
        result_file.write('target_spectrum\tbounded_double_lorentzian\n')
        result_file.write(f'lorentzian_gamma_nm\t{gamma:.10f}\n')
        result_file.write(f'predicted_R\t{structure[0]:.10f}\n')
        result_file.write(f'predicted_n_host1\t{structure[1]:.10f}\n')
        result_file.write(f'predicted_n_host2\t{structure[2]:.10f}\n')
        write_pair_metrics(result_file, 'final', detection, metrics)
        write_quality_metrics(result_file, 'final', quality)
        result_file.write(f'ideal_to_final_mse\t{mse:.10e}\n')
        result_file.write(f'ideal_to_final_rmse\t{rmse:.10e}\n')

    return detection, metrics


def save_cvae_case(
    output_dir,
    wavelengths,
    targets,
    ideal_target,
    selected,
    gamma,
    seed,
    n_candidates,
):
    os.makedirs(output_dir, exist_ok=True)
    candidate_detection = selected['candidate_detection']
    final_detection = selected['final_detection']
    candidate_metrics = peak_error_metrics(candidate_detection, targets)
    final_metrics = peak_error_metrics(final_detection, targets)
    candidate_quality = resonance_quality_metrics(
        selected['candidate_spectrum'],
        wavelengths,
        candidate_detection,
    )
    final_quality = selected['final_quality']

    plot_case(
        wavelengths,
        selected['candidate_spectrum'],
        selected['final_response'],
        final_detection,
        os.path.join(output_dir, 'cvae_plot.png'),
        'Selected CVAE spectrum',
    )
    np.savetxt(
        os.path.join(output_dir, 'cvae_data.txt'),
        np.column_stack((
            wavelengths,
            ideal_target,
            selected['candidate_spectrum'],
            selected['final_response'],
        )),
        delimiter='\t',
        fmt='%.10e',
        header=(
            'wavelength_nm\tideal_target_transmittance\t'
            'cvae_target_transmittance\t'
            'tandem_response_transmittance'
        ),
        comments='',
    )
    save_structure(
        os.path.join(output_dir, 'cvae_structure.txt'),
        selected['structure'],
    )
    save_candidate_diagnostics(
        os.path.join(output_dir, 'cvae_candidate_scores.txt'),
        selected['candidate_diagnostics'],
    )
    # Only the wavelength-sorted condition is used: CVAE_2p.lowest_wavelength
    # sorts the two dips ascending before forming the training condition, so
    # the reversed pair is an input the model has never seen.
    order_name = 'low_wavelength_then_high_wavelength'
    with open(
        os.path.join(output_dir, 'cvae_results.txt'),
        'w',
        encoding='utf-8',
    ) as result_file:
        result_file.write(f'target_peak1_nm\t{targets[0]:.10f}\n')
        result_file.write(f'target_peak2_nm\t{targets[1]:.10f}\n')
        result_file.write('target_spectrum\tbounded_double_lorentzian\n')
        result_file.write(f'lorentzian_gamma_nm\t{gamma:.10f}\n')
        result_file.write(f'random_seed\t{seed}\n')
        result_file.write(f'cvae_candidates_total\t{n_candidates}\n')
        result_file.write('cvae_condition_orders\t1\n')
        result_file.write(
            f'cvae_candidates_per_condition_order\t{n_candidates}\n'
        )
        result_file.write(
            'selection_metric\tscreen_cts_consistency_and_response_'
            'validity_then_rank_realized_wavelength_then_notch_'
            'transmittance_then_cts_wavelength_then_spectrum_mse\n'
        )
        result_file.write(
            'shape_score\tminimax_of_valley_depth_valley_balance_'
            'relative_fwhm_error_and_minimum_prominence\n'
        )
        for name, value in selected['quality_thresholds'].items():
            result_file.write(
                f'quality_threshold_{name}\t{value:.10f}\n'
            )
        result_file.write(
            f'best_available_position_error_nm\t'
            f'{selected["best_available_position_error_nm"]:.10f}\n'
        )
        result_file.write(
            f'position_quality_window_nm\t'
            f'{selected["position_quality_window_nm"]:.10f}\n'
        )
        result_file.write('position_window_used_for_selection\t0\n')
        result_file.write(
            f'selected_shape_gate_passed\t'
            f'{selected["selected_shape_gate_passed"]}\n'
        )
        result_file.write(
            f'selected_shape_minimax_score\t'
            f'{selected["selected_shape_minimax_score"]:.10e}\n'
        )
        result_file.write(
            f'selected_candidate_index\t'
            f'{selected["selected_index"]}\n'
        )
        result_file.write(
            f'selected_condition_order\t{order_name}\n'
        )
        result_file.write(
            f'candidate_valid_count\t'
            f'{selected["candidate_valid_count"]}\n'
        )
        result_file.write(
            f'final_response_valid_count\t'
            f'{selected["final_valid_count"]}\n'
        )
        result_file.write(
            f'candidate_maximum_index_error\t'
            f'{selected["candidate_maximum_index_error"]:.10f}\n'
        )
        result_file.write(
            f'candidate_summed_index_error\t'
            f'{selected["candidate_summed_index_error"]:.10f}\n'
        )
        result_file.write(
            f'candidate_maximum_wavelength_error_nm\t'
            f'{selected["candidate_maximum_wavelength_error"]:.10f}\n'
        )
        result_file.write(
            f'candidate_summed_wavelength_error_nm\t'
            f'{selected["candidate_summed_wavelength_error"]:.10f}\n'
        )
        result_file.write(
            f'final_maximum_index_error\t'
            f'{selected["final_maximum_index_error"]:.10f}\n'
        )
        result_file.write(
            f'final_summed_index_error\t'
            f'{selected["final_summed_index_error"]:.10f}\n'
        )
        result_file.write(
            f'final_maximum_wavelength_error_nm\t'
            f'{selected["final_maximum_wavelength_error"]:.10f}\n'
        )
        result_file.write(
            f'final_summed_wavelength_error_nm\t'
            f'{selected["final_summed_wavelength_error"]:.10f}\n'
        )
        result_file.write(
            f'final_mean_absolute_wavelength_error_nm\t'
            f'{0.5 * selected["final_summed_wavelength_error"]:.10f}\n'
        )
        result_file.write(
            f'predicted_R\t{selected["structure"][0]:.10f}\n'
        )
        result_file.write(
            f'predicted_n_host1\t{selected["structure"][1]:.10f}\n'
        )
        result_file.write(
            f'predicted_n_host2\t{selected["structure"][2]:.10f}\n'
        )
        write_pair_metrics(
            result_file,
            'cvae_target',
            candidate_detection,
            candidate_metrics,
        )
        write_quality_metrics(
            result_file,
            'cvae_target',
            candidate_quality,
        )
        write_pair_metrics(
            result_file,
            'final',
            final_detection,
            final_metrics,
        )
        write_quality_metrics(
            result_file,
            'final',
            final_quality,
        )
        write_quality_metrics(
            result_file,
            'ideal_target',
            selected['ideal_quality'],
        )
        result_file.write(
            f'ideal_to_cvae_mse\t'
            f'{selected["ideal_to_candidate_mse"]:.10e}\n'
        )
        result_file.write(
            f'ideal_to_final_mse\t'
            f'{selected["ideal_to_final_mse"]:.10e}\n'
        )
        result_file.write(
            f'cvae_to_final_mse\t'
            f'{selected["candidate_to_final_mse"]:.10e}\n'
        )

    return candidate_detection, final_detection, final_metrics


def main(argv=None):
    args = parse_args(argv)
    targets = sorted_peak_pair(args.peak1, args.peak2)
    output_root = os.path.abspath(args.output_root)
    case_dir = os.path.join(
        output_root,
        case_tag(*targets),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Target peaks: {targets[0]:.4f}, {targets[1]:.4f} nm')
    print(f'Output directory: {case_dir}')

    wavelengths = load_wavelength_grid()
    ideal_target = bounded_double_lorentzian(
        targets,
        wavelengths,
        gamma=args.gamma,
    )
    tandem = build_tandem_runtime(device)

    if args.method in ('both', 'tandem'):
        direct_structures, direct_responses = tandem.inverse_forward(
            ideal_target
        )
        _, direct_metrics = save_direct_case(
            os.path.join(case_dir, 'tandem'),
            wavelengths,
            targets,
            ideal_target,
            direct_structures[0],
            direct_responses[0],
            args.gamma,
        )
        print(
            'Direct Tandem: '
            f'peaks={detect_two_resonances(direct_responses[0], wavelengths).wavelengths}, '
            f'Emax={direct_metrics["max_absolute_error_nm"]:.4f} nm'
        )

    if args.method in ('both', 'cvae'):
        seed_everything(args.random_seed)
        cvae = build_cvae(device)
        selected = select_cvae_assisted_design(
            model=cvae,
            tandem=tandem,
            target_centers=targets,
            ideal_target=ideal_target,
            wavelengths=wavelengths,
            n_candidates=args.n_candidates,
            device=device,
            position_slack_nm=args.position_slack_nm,
            maximum_valley_transmittance=(
                args.maximum_valley_transmittance
            ),
            minimum_prominence=args.minimum_prominence,
            maximum_valley_imbalance=args.maximum_valley_imbalance,
            minimum_width_ratio=args.minimum_width_ratio,
            maximum_width_ratio=args.maximum_width_ratio,
        )
        _, final_detection, final_metrics = save_cvae_case(
            os.path.join(case_dir, 'cvae'),
            wavelengths,
            targets,
            ideal_target,
            selected,
            args.gamma,
            args.random_seed,
            args.n_candidates,
        )
        print(
            'CVAE-assisted Tandem: '
            f'candidate={selected["selected_index"]}, '
            f'peaks={final_detection.wavelengths}, '
            f'Emax={final_metrics["max_absolute_error_nm"]:.4f} nm'
        )


if __name__ == '__main__':
    main()
