#!/usr/bin/env python3
"""Compare Direct Tandem and CVAE-assisted Tandem on 30 dual-peak cases.

The target set is a deterministic 6 x 5 factorial design:

    midpoint  = [470, 492, 514, 536, 558, 580] nm
    separation = [40, 60, 80, 100, 120] nm

All outputs are written into one statistics directory.  No per-case
directories are created and no CST simulations are run.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from dual_peak_utils import (
    BASE_DIR,
    ENCODER_SIZE,
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
)


MIDPOINTS_NM = np.array([470, 492, 514, 536, 558, 580], dtype=float)
SEPARATIONS_NM = np.array([40, 60, 80, 100, 120], dtype=float)
DEFAULT_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    'statistics_pair_sweep_410_640',
)

FIG_DPI = 600
SINGLE_COLUMN_SIZE = (3.5, 2.65)
DOUBLE_COLUMN_SIZE = (7.0, 3.0)
DIRECT_COLOR = '#2E75B6'
CVAE_COLOR = '#ED7D31'


matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate Direct Tandem and CVAE-assisted Tandem on a '
            'deterministic 30-case dual-resonance grid. The script does '
            'not run CST and stores every result in one output folder.'
        )
    )
    parser.add_argument(
        '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help=(
            'Single output directory for all tables and figures '
            f'(default: {DEFAULT_OUTPUT_DIR})'
        ),
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=15.0,
        help='Lorentzian half-width parameter in nm (default: 15)',
    )
    parser.add_argument(
        '--n-candidates',
        type=int,
        default=20,
        help=(
            'Total CVAE candidates per case, split equally between the '
            'two condition orders (default: 20)'
        ),
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed used to create the shared latent set (default: 42)',
    )
    parser.add_argument(
        '--prominence',
        type=float,
        default=0.01,
        help='Minimum prominence for resonance detection (default: 0.01)',
    )
    parser.add_argument(
        '--minimum-distance-nm',
        type=float,
        default=20.0,
        help=(
            'Minimum distance between detected minima in nm '
            '(default: 20)'
        ),
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cpu', 'cuda'),
        default='auto',
        help='Inference device (default: auto)',
    )
    args = parser.parse_args(argv)

    if args.gamma <= 0:
        parser.error('--gamma must be positive')
    if args.n_candidates < 2 or args.n_candidates % 2:
        parser.error('--n-candidates must be an even integer of at least 2')
    if args.prominence <= 0:
        parser.error('--prominence must be positive')
    if args.minimum_distance_nm <= 0:
        parser.error('--minimum-distance-nm must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('--device cuda was requested but CUDA is unavailable')
    return args


def resolve_device(device_name):
    if device_name == 'auto':
        return torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    return torch.device(device_name)


def build_target_cases():
    cases = []
    for midpoint in MIDPOINTS_NM:
        for separation in SEPARATIONS_NM:
            low = midpoint - separation / 2.0
            high = midpoint + separation / 2.0
            cases.append({
                'midpoint_nm': float(midpoint),
                'separation_nm': float(separation),
                'target_low_nm': float(low),
                'target_high_nm': float(high),
                'case_tag': case_tag(low, high),
            })

    pairs = {
        (case['target_low_nm'], case['target_high_nm'])
        for case in cases
    }
    if len(cases) != 30 or len(pairs) != 30:
        raise RuntimeError('The target design must contain 30 unique pairs.')
    bounds = np.array([
        value
        for case in cases
        for value in (case['target_low_nm'], case['target_high_nm'])
    ])
    if np.min(bounds) < 410 or np.max(bounds) > 640:
        raise RuntimeError('A target wavelength lies outside 410-640 nm.')
    return cases


def method_result(
    prefix,
    detection,
    metrics,
    response,
    ideal_target,
    structure,
    wavelengths,
):
    quality = resonance_quality_metrics(
        response,
        wavelengths,
        detection,
    )
    values = {
        f'{prefix}_detected': int(detection.valid),
        f'{prefix}_significant_minima_count': len(
            detection.all_indices
        ),
        f'{prefix}_extra_minima_count': max(
            0,
            len(detection.all_indices) - 2,
        ),
        f'{prefix}_predicted_low_nm': detection.wavelengths[0],
        f'{prefix}_predicted_high_nm': detection.wavelengths[1],
        f'{prefix}_low_signed_error_nm': (
            metrics['signed_errors_nm'][0]
        ),
        f'{prefix}_high_signed_error_nm': (
            metrics['signed_errors_nm'][1]
        ),
        f'{prefix}_low_absolute_error_nm': (
            metrics['absolute_errors_nm'][0]
        ),
        f'{prefix}_high_absolute_error_nm': (
            metrics['absolute_errors_nm'][1]
        ),
        f'{prefix}_emax_nm': metrics['max_absolute_error_nm'],
        f'{prefix}_pair_rmse_nm': metrics['pair_rmse_nm'],
        f'{prefix}_midpoint_error_nm': metrics['midpoint_error_nm'],
        f'{prefix}_separation_error_nm': (
            metrics['separation_error_nm']
        ),
        f'{prefix}_ideal_spectrum_mse': float(
            np.mean(
                (
                    np.asarray(response, dtype=float)
                    - np.asarray(ideal_target, dtype=float)
                ) ** 2
            )
        ),
        f'{prefix}_peak1_transmittance': float(
            detection.values[0]
        ),
        f'{prefix}_peak2_transmittance': float(
            detection.values[1]
        ),
        f'{prefix}_worst_valley_transmittance': float(
            quality.worst_valley_transmittance
        ),
        f'{prefix}_peak1_fwhm_nm': float(quality.fwhm_nm[0]),
        f'{prefix}_peak2_fwhm_nm': float(quality.fwhm_nm[1]),
        f'{prefix}_widest_fwhm_nm': float(quality.widest_fwhm_nm),
        f'{prefix}_minimum_prominence': float(
            quality.minimum_prominence
        ),
        f'{prefix}_R': float(structure[0]),
        f'{prefix}_n_host1': float(structure[1]),
        f'{prefix}_n_host2': float(structure[2]),
    }
    return values


def evaluate_cases(args, device):
    wavelengths = load_wavelength_grid()
    cases = build_target_cases()

    seed_everything(args.random_seed)
    tandem = build_tandem_runtime(device)
    cvae = build_cvae(device)

    # Reset after model construction so the saved checkpoints' constructor
    # initialization cannot change the latent set.  This exact tensor is
    # reused for every one of the 30 target pairs.
    seed_everything(args.random_seed)
    # One latent per candidate. This was n_candidates // 2 back when
    # make_condition_orders returned the reversed pair as well and each latent
    # was decoded twice; that order is gone (the CVAE never saw it in
    # training), so halving here now under-fills the batch and
    # generate_cvae_candidates rejects the shape outright.
    base_latents = torch.randn(
        args.n_candidates,
        ENCODER_SIZE[-1],
        dtype=torch.float32,
        device=device,
    )

    rows = []
    candidate_score_rows = []
    spectrum_columns = [wavelengths]
    spectrum_headers = ['wavelength_nm']
    direct_grid = np.full(
        (len(MIDPOINTS_NM), len(SEPARATIONS_NM)),
        np.nan,
    )
    cvae_grid = np.full_like(direct_grid, np.nan)

    for case_index, case in enumerate(cases, start=1):
        targets = np.array([
            case['target_low_nm'],
            case['target_high_nm'],
        ])
        ideal_target = bounded_double_lorentzian(
            targets,
            wavelengths,
            gamma=args.gamma,
        )

        direct_structures, direct_responses = tandem.inverse_forward(
            ideal_target
        )
        direct_structure = direct_structures[0]
        direct_response = direct_responses[0]
        direct_detection = detect_two_resonances(
            direct_response,
            wavelengths,
            prominence=args.prominence,
            minimum_distance_nm=args.minimum_distance_nm,
        )
        direct_metrics = peak_error_metrics(
            direct_detection,
            targets,
        )

        selected = select_cvae_assisted_design(
            model=cvae,
            tandem=tandem,
            target_centers=targets,
            ideal_target=ideal_target,
            wavelengths=wavelengths,
            n_candidates=args.n_candidates,
            device=device,
            base_latents=base_latents,
            prominence=args.prominence,
            minimum_distance_nm=args.minimum_distance_nm,
        )
        for diagnostic in selected['candidate_diagnostics']:
            candidate_score_rows.append({
                'case_index': case_index,
                'case_tag': case['case_tag'],
                'target_low_nm': case['target_low_nm'],
                'target_high_nm': case['target_high_nm'],
                **diagnostic,
            })
        cvae_response = selected['final_response']
        cvae_structure = selected['structure']
        cvae_detection = selected['final_detection']
        cvae_metrics = peak_error_metrics(cvae_detection, targets)

        row = {
            'case_index': case_index,
            **case,
        }
        row.update(method_result(
            'direct',
            direct_detection,
            direct_metrics,
            direct_response,
            ideal_target,
            direct_structure,
            wavelengths,
        ))
        row.update(method_result(
            'cvae',
            cvae_detection,
            cvae_metrics,
            cvae_response,
            ideal_target,
            cvae_structure,
            wavelengths,
        ))
        row.update({
            'cvae_selected_candidate_index': selected[
                'selected_index'
            ],
            'cvae_condition_order_code': selected[
                'condition_order_code'
            ],
            'cvae_candidate_valid_count': selected[
                'candidate_valid_count'
            ],
            'cvae_final_response_valid_count': selected[
                'final_valid_count'
            ],
            'cvae_selected_shape_gate_passed': selected[
                'selected_shape_gate_passed'
            ],
            'cvae_selected_shape_minimax_score': selected[
                'selected_shape_minimax_score'
            ],
            'cvae_best_available_position_error_nm': selected[
                'best_available_position_error_nm'
            ],
            'cvae_position_quality_window_nm': selected[
                'position_quality_window_nm'
            ],
            'cvae_candidate_significant_minima_count': len(
                selected['candidate_detection'].all_indices
            ),
            'cvae_candidate_extra_minima_count': max(
                0,
                len(selected['candidate_detection'].all_indices) - 2,
            ),
            'cvae_candidate_emax_nm': peak_error_metrics(
                selected['candidate_detection'],
                targets,
            )['max_absolute_error_nm'],
            'cvae_ideal_to_candidate_mse': selected[
                'ideal_to_candidate_mse'
            ],
            'cvae_candidate_to_final_mse': selected[
                'candidate_to_final_mse'
            ],
        })
        if (
            np.isfinite(row['direct_emax_nm'])
            and np.isfinite(row['cvae_emax_nm'])
        ):
            row['cvae_emax_improvement_nm'] = (
                row['direct_emax_nm'] - row['cvae_emax_nm']
            )
        else:
            row['cvae_emax_improvement_nm'] = np.nan

        midpoint_index = int(np.where(
            MIDPOINTS_NM == case['midpoint_nm']
        )[0][0])
        separation_index = int(np.where(
            SEPARATIONS_NM == case['separation_nm']
        )[0][0])
        direct_grid[midpoint_index, separation_index] = row[
            'direct_emax_nm'
        ]
        cvae_grid[midpoint_index, separation_index] = row[
            'cvae_emax_nm'
        ]
        rows.append(row)

        tag = case['case_tag']
        spectrum_headers.extend([
            f'{tag}_ideal_target',
            f'{tag}_direct_tandem',
            f'{tag}_cvae_assisted_tandem',
        ])
        spectrum_columns.extend([
            ideal_target,
            direct_response,
            cvae_response,
        ])

        direct_text = (
            f'{row["direct_emax_nm"]:.3f}'
            if np.isfinite(row['direct_emax_nm'])
            else 'not detected'
        )
        cvae_text = (
            f'{row["cvae_emax_nm"]:.3f}'
            if np.isfinite(row['cvae_emax_nm'])
            else 'not detected'
        )
        print(
            f'[{case_index:02d}/30] {tag}: '
            f'Direct Emax={direct_text} nm, '
            f'CVAE Emax={cvae_text} nm',
            flush=True,
        )

    return {
        'wavelengths': wavelengths,
        'rows': rows,
        'spectrum_headers': spectrum_headers,
        'spectrum_data': np.column_stack(spectrum_columns),
        'direct_grid': direct_grid,
        'cvae_grid': cvae_grid,
        'base_latents': base_latents.detach().cpu().numpy(),
        'candidate_score_rows': candidate_score_rows,
    }


def table_fieldnames():
    fields = [
        'case_index',
        'case_tag',
        'midpoint_nm',
        'separation_nm',
        'target_low_nm',
        'target_high_nm',
    ]
    per_method = [
        'detected',
        'significant_minima_count',
        'extra_minima_count',
        'predicted_low_nm',
        'predicted_high_nm',
        'low_signed_error_nm',
        'high_signed_error_nm',
        'low_absolute_error_nm',
        'high_absolute_error_nm',
        'emax_nm',
        'pair_rmse_nm',
        'midpoint_error_nm',
        'separation_error_nm',
        'ideal_spectrum_mse',
        'peak1_transmittance',
        'peak2_transmittance',
        'worst_valley_transmittance',
        'peak1_fwhm_nm',
        'peak2_fwhm_nm',
        'widest_fwhm_nm',
        'minimum_prominence',
        'R',
        'n_host1',
        'n_host2',
    ]
    fields.extend(f'direct_{name}' for name in per_method)
    fields.extend(f'cvae_{name}' for name in per_method)
    fields.extend([
        'cvae_emax_improvement_nm',
        'cvae_selected_candidate_index',
        'cvae_condition_order_code',
        'cvae_candidate_valid_count',
        'cvae_final_response_valid_count',
        'cvae_selected_shape_gate_passed',
        'cvae_selected_shape_minimax_score',
        'cvae_best_available_position_error_nm',
        'cvae_position_quality_window_nm',
        'cvae_candidate_significant_minima_count',
        'cvae_candidate_extra_minima_count',
        'cvae_candidate_emax_nm',
        'cvae_ideal_to_candidate_mse',
        'cvae_candidate_to_final_mse',
    ])
    return fields


def format_table_value(value):
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return 'nan'
        return f'{float(value):.10e}'
    return str(value)


def save_detailed_table(output_dir, rows):
    path = os.path.join(output_dir, 'dual_peak_30_cases.tsv')
    fieldnames = table_fieldnames()
    with open(path, 'w', encoding='utf-8', newline='') as table_file:
        writer = csv.DictWriter(
            table_file,
            fieldnames=fieldnames,
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: format_table_value(row[field])
                for field in fieldnames
            })
    return path


def save_spectra(output_dir, spectrum_headers, spectrum_data):
    path = os.path.join(output_dir, 'dual_peak_30_case_spectra.txt')
    np.savetxt(
        path,
        spectrum_data,
        delimiter='\t',
        fmt='%.10e',
        header='\t'.join(spectrum_headers),
        comments='',
    )
    return path


def finite_values(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=float)
    return values[np.isfinite(values)]


def method_summary(rows, prefix):
    detected = np.asarray(
        [bool(row[f'{prefix}_detected']) for row in rows],
        dtype=bool,
    )
    emax_all = np.asarray(
        [row[f'{prefix}_emax_nm'] for row in rows],
        dtype=float,
    )
    emax = emax_all[np.isfinite(emax_all)]
    pair_rmse = finite_values(rows, f'{prefix}_pair_rmse_nm')
    extra = np.asarray(
        [row[f'{prefix}_extra_minima_count'] for row in rows],
        dtype=float,
    )
    worst_valley = finite_values(
        rows,
        f'{prefix}_worst_valley_transmittance',
    )
    widest_fwhm = finite_values(rows, f'{prefix}_widest_fwhm_nm')
    minimum_prominences = finite_values(
        rows,
        f'{prefix}_minimum_prominence',
    )
    total = len(rows)

    if len(emax):
        error_stats = {
            'emax_mean_nm': float(np.mean(emax)),
            'emax_median_nm': float(np.median(emax)),
            'emax_rms_nm': float(np.sqrt(np.mean(emax ** 2))),
            'emax_q75_nm': float(np.quantile(emax, 0.75)),
            'emax_q90_nm': float(np.quantile(emax, 0.90)),
            'emax_q95_nm': float(np.quantile(emax, 0.95)),
            'emax_max_nm': float(np.max(emax)),
            'pair_rmse_mean_nm': float(np.mean(pair_rmse)),
        }
    else:
        error_stats = {
            key: np.nan
            for key in (
                'emax_mean_nm',
                'emax_median_nm',
                'emax_rms_nm',
                'emax_q75_nm',
                'emax_q90_nm',
                'emax_q95_nm',
                'emax_max_nm',
                'pair_rmse_mean_nm',
            )
        }

    summary = {
        'detected_count': int(np.sum(detected)),
        'detection_success_rate': float(np.mean(detected)),
        **error_stats,
        'extra_minima_mean': float(np.mean(extra)),
        'extra_minima_total': int(np.sum(extra)),
        'cases_with_extra_minima': int(np.sum(extra > 0)),
        'mean_worst_valley_transmittance': (
            float(np.mean(worst_valley))
            if len(worst_valley)
            else np.nan
        ),
        'median_worst_valley_transmittance': (
            float(np.median(worst_valley))
            if len(worst_valley)
            else np.nan
        ),
        'mean_widest_fwhm_nm': (
            float(np.mean(widest_fwhm))
            if len(widest_fwhm)
            else np.nan
        ),
        'median_widest_fwhm_nm': (
            float(np.median(widest_fwhm))
            if len(widest_fwhm)
            else np.nan
        ),
        'mean_minimum_prominence': (
            float(np.mean(minimum_prominences))
            if len(minimum_prominences)
            else np.nan
        ),
    }
    for threshold in (1.0, 2.0, 3.0):
        # Invalid detections remain failures because NaN <= threshold is
        # False.  The denominator is always all 30 test cases.
        passed = np.isfinite(emax_all) & (emax_all <= threshold)
        summary[f'within_{int(threshold)}nm_count'] = int(
            np.sum(passed)
        )
        summary[f'within_{int(threshold)}nm_rate'] = float(
            np.mean(passed)
        )
    return summary


def improvement_summary(rows):
    direct = np.asarray(
        [row['direct_emax_nm'] for row in rows],
        dtype=float,
    )
    cvae = np.asarray(
        [row['cvae_emax_nm'] for row in rows],
        dtype=float,
    )
    paired = np.isfinite(direct) & np.isfinite(cvae)
    differences = direct[paired] - cvae[paired]
    tolerance = 1e-9

    return {
        'paired_valid_count': int(np.sum(paired)),
        'mean_emax_improvement_nm': (
            float(np.mean(differences)) if len(differences) else np.nan
        ),
        'median_emax_improvement_nm': (
            float(np.median(differences))
            if len(differences)
            else np.nan
        ),
        'cvae_better_count': int(np.sum(differences > tolerance)),
        'tied_count': int(np.sum(np.abs(differences) <= tolerance)),
        'direct_better_count': int(
            np.sum(differences < -tolerance)
        ),
        'cvae_better_rate_among_paired': (
            float(np.mean(differences > tolerance))
            if len(differences)
            else np.nan
        ),
        'cvae_rescued_detection_count': int(
            np.sum(~np.isfinite(direct) & np.isfinite(cvae))
        ),
        'cvae_lost_detection_count': int(
            np.sum(np.isfinite(direct) & ~np.isfinite(cvae))
        ),
    }


def write_summary(output_dir, args, device, rows):
    direct = method_summary(rows, 'direct')
    cvae = method_summary(rows, 'cvae')
    improvement = improvement_summary(rows)
    path = os.path.join(output_dir, 'summary.txt')

    with open(path, 'w', encoding='utf-8') as summary_file:
        summary_file.write(
            'Dual-peak 30-case Direct Tandem vs CVAE-assisted Tandem\n'
        )
        summary_file.write(
            '========================================================\n\n'
        )
        summary_file.write(
            'target_design\t6_midpoints_x_5_separations\n'
        )
        summary_file.write(
            'midpoints_nm\t'
            + ','.join(f'{value:g}' for value in MIDPOINTS_NM)
            + '\n'
        )
        summary_file.write(
            'separations_nm\t'
            + ','.join(f'{value:g}' for value in SEPARATIONS_NM)
            + '\n'
        )
        summary_file.write('target_range_nm\t410-640\n')
        summary_file.write(f'case_count\t{len(rows)}\n')
        summary_file.write(f'gamma_nm\t{args.gamma:.10g}\n')
        summary_file.write(f'random_seed\t{args.random_seed}\n')
        summary_file.write(
            f'cvae_candidates_per_case\t{args.n_candidates}\n'
        )
        summary_file.write(
            'cvae_condition_orders\t1\n'
        )
        summary_file.write(
            'cvae_candidates_per_condition_order\t'
            f'{args.n_candidates}\n'
        )
        summary_file.write(
            'shared_latent_set_across_all_cases\t1\n'
        )
        summary_file.write(
            'cvae_quality_selection\tposition_error_within_best_plus_'
            '2.5nm_then_shape_gate_then_minimax_of_depth_balance_'
            'fwhm_and_prominence\n'
        )
        summary_file.write(
            'cvae_shape_gate_passed_count\t'
            f'{sum(row["cvae_selected_shape_gate_passed"] for row in rows)}\n'
        )
        summary_file.write(f'inference_device\t{device}\n')
        summary_file.write(
            f'resonance_prominence\t{args.prominence:.10g}\n'
        )
        summary_file.write(
            'minimum_resonance_distance_nm\t'
            f'{args.minimum_distance_nm:.10g}\n'
        )
        summary_file.write(
            'peak_pairing\tpredicted_minima_sorted_by_wavelength_'
            'and_paired_low_to_low_high_to_high\n'
        )
        summary_file.write(
            'threshold_denominator\tall_30_cases_invalid_detection_'
            'counts_as_failure\n'
        )
        summary_file.write(
            'error_distribution_denominator\tsuccessfully_detected_'
            'two_peak_cases_only\n'
        )

        for title, result in (
            ('Direct Tandem', direct),
            ('CVAE-assisted Tandem', cvae),
        ):
            summary_file.write(f'\n[{title}]\n')
            for key, value in result.items():
                summary_file.write(
                    f'{key}\t{format_table_value(value)}\n'
                )

        summary_file.write('\n[CVAE improvement]\n')
        summary_file.write(
            'definition\tDirect_Emax_minus_CVAE_Emax_'
            'positive_means_CVAE_is_better\n'
        )
        for key, value in improvement.items():
            summary_file.write(
                f'{key}\t{format_table_value(value)}\n'
            )
    return path


def save_ecdf_data(output_dir, rows):
    path = os.path.join(output_dir, 'emax_ecdf_data.txt')
    with open(path, 'w', encoding='utf-8', newline='') as data_file:
        writer = csv.writer(
            data_file,
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writerow([
            'method',
            'rank',
            'valid_case_count',
            'emax_nm',
            'empirical_cumulative_probability',
        ])
        for method, key in (
            ('Direct_Tandem', 'direct_emax_nm'),
            ('CVAE_assisted_Tandem', 'cvae_emax_nm'),
        ):
            values = np.sort(finite_values(rows, key))
            for rank, value in enumerate(values, start=1):
                writer.writerow([
                    method,
                    rank,
                    len(values),
                    f'{value:.10e}',
                    f'{rank / len(values):.10e}',
                ])
    return path


def style_axes(ax):
    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        top=True,
        right=True,
        length=3,
        width=0.7,
        pad=2,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_ecdf_figure(output_dir, rows):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN_SIZE)
    style_axes(ax)

    plotted = []
    for label, key, color, linestyle in (
        ('Direct Tandem', 'direct_emax_nm', DIRECT_COLOR, '-'),
        (
            'CVAE-assisted Tandem',
            'cvae_emax_nm',
            CVAE_COLOR,
            (0, (4, 2.2)),
        ),
    ):
        values = np.sort(finite_values(rows, key))
        if not len(values):
            continue
        cumulative = np.arange(1, len(values) + 1) / len(values)
        ax.step(
            values,
            cumulative,
            where='post',
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            label=f'{label} ($n={len(values)}$)',
        )
        plotted.extend(values.tolist())

    for threshold in (1, 2, 3):
        ax.axvline(
            threshold,
            color='#B8B8B8',
            linewidth=0.55,
            linestyle=(0, (2, 2.4)),
            zorder=0,
        )
    ax.set_xlabel(r'Maximum paired error, $E_{\mathrm{max}}$ (nm)')
    ax.set_ylabel('Empirical cumulative probability')
    ax.set_ylim(0, 1.02)
    if plotted:
        upper = max(3.2, max(plotted) * 1.05)
        ax.set_xlim(0, upper)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(
        axis='y',
        color='#D9D9D9',
        linewidth=0.45,
        alpha=0.65,
    )
    ax.legend(
        loc='lower right',
        frameon=False,
        handlelength=2.5,
        handletextpad=0.5,
    )
    fig.tight_layout(pad=0.45)
    path = os.path.join(output_dir, 'emax_ecdf.png')
    fig.savefig(
        path,
        dpi=FIG_DPI,
        bbox_inches='tight',
        pad_inches=0.04,
        facecolor='white',
    )
    plt.close(fig)
    return path


def save_heatmap_data(output_dir, rows):
    path = os.path.join(output_dir, 'emax_heatmap_data.txt')
    with open(path, 'w', encoding='utf-8', newline='') as data_file:
        writer = csv.writer(
            data_file,
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writerow([
            'midpoint_nm',
            'separation_nm',
            'target_low_nm',
            'target_high_nm',
            'direct_emax_nm',
            'cvae_emax_nm',
        ])
        for row in rows:
            writer.writerow([
                format_table_value(row['midpoint_nm']),
                format_table_value(row['separation_nm']),
                format_table_value(row['target_low_nm']),
                format_table_value(row['target_high_nm']),
                format_table_value(row['direct_emax_nm']),
                format_table_value(row['cvae_emax_nm']),
            ])
    return path


def save_heatmap_figure(output_dir, direct_grid, cvae_grid):
    finite = np.concatenate((
        direct_grid[np.isfinite(direct_grid)],
        cvae_grid[np.isfinite(cvae_grid)],
    ))
    maximum = float(np.max(finite)) if len(finite) else 1.0
    color_maximum = max(1.0, np.ceil(maximum * 2.0) / 2.0)
    colormap = matplotlib.colormaps.get_cmap('viridis').copy()
    colormap.set_bad('#E5E5E5')

    fig = plt.figure(figsize=DOUBLE_COLUMN_SIZE)
    grid_spec = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.055),
        left=0.085,
        right=0.945,
        bottom=0.18,
        top=0.88,
        wspace=0.14,
    )
    axes = (
        fig.add_subplot(grid_spec[0, 0]),
        fig.add_subplot(grid_spec[0, 1]),
    )
    colorbar_axis = fig.add_subplot(grid_spec[0, 2])
    images = []
    for ax, grid, title in (
        (axes[0], direct_grid, 'Direct Tandem'),
        (axes[1], cvae_grid, 'CVAE-assisted Tandem'),
    ):
        image = ax.imshow(
            np.ma.masked_invalid(grid),
            origin='lower',
            aspect='auto',
            interpolation='nearest',
            cmap=colormap,
            vmin=0,
            vmax=color_maximum,
        )
        images.append(image)
        ax.set_title(title, pad=4)
        ax.set_xticks(np.arange(len(SEPARATIONS_NM)))
        ax.set_xticklabels([
            f'{value:g}' for value in SEPARATIONS_NM
        ])
        ax.set_yticks(np.arange(len(MIDPOINTS_NM)))
        ax.set_yticklabels([f'{value:g}' for value in MIDPOINTS_NM])
        ax.set_xlabel('Peak separation (nm)')
        style_axes(ax)

        for row_index in range(grid.shape[0]):
            for column_index in range(grid.shape[1]):
                value = grid[row_index, column_index]
                if np.isfinite(value):
                    text = f'{value:.1f}'
                    text_color = (
                        'white'
                        if value / color_maximum < 0.57
                        else '#171717'
                    )
                else:
                    text = '–'
                    text_color = '#555555'
                ax.text(
                    column_index,
                    row_index,
                    text,
                    ha='center',
                    va='center',
                    fontsize=6.8,
                    color=text_color,
                )

    axes[0].set_ylabel('Pair midpoint (nm)')
    colorbar = fig.colorbar(
        images[-1],
        cax=colorbar_axis,
    )
    colorbar.set_label(r'$E_{\mathrm{max}}$ (nm)')
    colorbar.ax.tick_params(
        direction='in',
        length=2.5,
        width=0.7,
        labelsize=8,
    )
    colorbar.outline.set_linewidth(0.8)
    path = os.path.join(output_dir, 'emax_heatmaps.png')
    fig.savefig(
        path,
        dpi=FIG_DPI,
        bbox_inches='tight',
        pad_inches=0.04,
        facecolor='white',
    )
    plt.close(fig)
    return path


def save_shared_latents(output_dir, latents):
    path = os.path.join(output_dir, 'shared_cvae_latents.txt')
    np.savetxt(
        path,
        latents,
        delimiter='\t',
        fmt='%.10e',
        header='\t'.join(
            f'z{index + 1}' for index in range(latents.shape[1])
        ),
        comments='',
    )
    return path


def save_candidate_scores(output_dir, rows):
    path = os.path.join(output_dir, 'candidate_scores.txt')
    if not rows:
        raise ValueError('No candidate-score rows were generated.')
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as score_file:
        writer = csv.DictWriter(
            score_file,
            fieldnames=fieldnames,
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: format_table_value(row[field])
                for field in fieldnames
            })
    return path


def main(argv=None):
    args = parse_args(argv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = resolve_device(args.device)

    print(f'Device: {device}', flush=True)
    print(f'Output directory: {output_dir}', flush=True)
    print(
        'Target design: 6 midpoints x 5 separations = 30 unique pairs',
        flush=True,
    )
    evaluated = evaluate_cases(args, device)

    saved_paths = [
        save_detailed_table(output_dir, evaluated['rows']),
        save_spectra(
            output_dir,
            evaluated['spectrum_headers'],
            evaluated['spectrum_data'],
        ),
        write_summary(
            output_dir,
            args,
            device,
            evaluated['rows'],
        ),
        save_ecdf_data(output_dir, evaluated['rows']),
        save_ecdf_figure(output_dir, evaluated['rows']),
        save_heatmap_data(output_dir, evaluated['rows']),
        save_heatmap_figure(
            output_dir,
            evaluated['direct_grid'],
            evaluated['cvae_grid'],
        ),
        save_shared_latents(output_dir, evaluated['base_latents']),
        save_candidate_scores(
            output_dir,
            evaluated['candidate_score_rows'],
        ),
    ]
    print('Saved outputs:', flush=True)
    for path in saved_paths:
        print(f'  {path}', flush=True)


if __name__ == '__main__':
    main()
