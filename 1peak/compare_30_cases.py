import argparse
import os

import numpy as np
import torch

import CVAE
from lorentz_vs_cvae_compare import (
    DEFAULT_FWHM_REFERENCE_NM,
    DEFAULT_N_CANDIDATES,
    DEFAULT_VALLEY_REFERENCE_TRANSMITTANCE,
    DEFAULT_WAVELENGTH_TOLERANCE_NM,
    MODEL_PATH,
    build_tandem_network,
    lorentzian_trans,
)
from notch_quality import measure_notch, select_single_notch_candidate
from plot_30_case_comparison import create_comparison_plots


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'statistics_410_650')

ENCODER_SIZE = [1001, 1024, 2048, 1024, 512, 256, 128, 20]
DECODER_SIZE = [20, 128, 256, 512, 1024, 2048, 1024, 1001]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Compare direct Tandem and CVAE-to-Tandem inverse design over an '
            'evenly spaced wavelength sweep. All cases are written to one '
            'summary directory.'
        )
    )
    parser.add_argument('--start-wavelength', type=float, default=410.0)
    parser.add_argument('--stop-wavelength', type=float, default=650.0)
    parser.add_argument('--n-cases', type=int, default=30)
    parser.add_argument(
        '--n-candidates',
        type=int,
        default=DEFAULT_N_CANDIDATES,
    )
    parser.add_argument(
        '--wavelength-tolerance-nm',
        type=float,
        default=DEFAULT_WAVELENGTH_TOLERANCE_NM,
    )
    parser.add_argument(
        '--valley-reference-transmittance',
        type=float,
        default=DEFAULT_VALLEY_REFERENCE_TRANSMITTANCE,
    )
    parser.add_argument(
        '--fwhm-reference-nm',
        type=float,
        default=DEFAULT_FWHM_REFERENCE_NM,
    )
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=15.0)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    if args.n_cases < 2:
        parser.error('--n-cases must be at least 2')
    if args.n_candidates < 1:
        parser.error('--n-candidates must be at least 1')
    if args.stop_wavelength <= args.start_wavelength:
        parser.error('--stop-wavelength must exceed --start-wavelength')
    if args.gamma <= 0:
        parser.error('--gamma must be positive')
    if args.wavelength_tolerance_nm < 0:
        parser.error('--wavelength-tolerance-nm must be non-negative')
    if args.valley_reference_transmittance <= 0:
        parser.error('--valley-reference-transmittance must be positive')
    if args.fwhm_reference_nm <= 0:
        parser.error('--fwhm-reference-nm must be positive')
    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cvae(device):
    model = CVAE.CVAE(
        encoder_size=ENCODER_SIZE,
        decoder_size=DECODER_SIZE,
        c_dim=1,
    ).to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Missing model checkpoint: {MODEL_PATH}')
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def calculate_stats(signed_errors, spectral_mse):
    signed_errors = np.asarray(signed_errors, dtype=float)
    absolute_errors = np.abs(signed_errors)
    spectral_mse = np.asarray(spectral_mse, dtype=float)
    return {
        'mean_absolute_error_nm': float(np.mean(absolute_errors)),
        'median_absolute_error_nm': float(np.median(absolute_errors)),
        'max_absolute_error_nm': float(np.max(absolute_errors)),
        'wavelength_rmse_nm': float(np.sqrt(np.mean(signed_errors ** 2))),
        'mean_signed_error_nm': float(np.mean(signed_errors)),
        'mean_spectral_mse': float(np.mean(spectral_mse)),
    }


def write_summary(
    path,
    args,
    design_wavelengths,
    direct_stats,
    cvae_stats,
    direct_absolute_errors,
    cvae_absolute_errors,
    direct_min_transmittance,
    direct_fwhms,
    cvae_min_transmittance,
    cvae_fwhms,
    selection_fallback_flags,
):
    cvae_wins = int(np.sum(cvae_absolute_errors < direct_absolute_errors))
    direct_wins = int(np.sum(direct_absolute_errors < cvae_absolute_errors))
    ties = int(len(design_wavelengths) - cvae_wins - direct_wins)
    improvement = direct_absolute_errors - cvae_absolute_errors

    with open(path, 'w', encoding='utf-8') as result_file:
        result_file.write(f'n_cases\t{len(design_wavelengths)}\n')
        result_file.write(
            f'design_wavelength_start_nm\t{design_wavelengths[0]:.10f}\n'
        )
        result_file.write(
            f'design_wavelength_stop_nm\t{design_wavelengths[-1]:.10f}\n'
        )
        result_file.write('case_spacing\tevenly_spaced_in_wavelength\n')
        result_file.write('target_spectrum\tlorentzian_dip\n')
        result_file.write(f'lorentzian_gamma_nm\t{args.gamma:.10f}\n')
        result_file.write(f'random_seed\t{args.random_seed}\n')
        result_file.write(f'cvae_candidates_per_case\t{args.n_candidates}\n')
        result_file.write(
            'cvae_selection_metric\t'
            'final_wavelength_tolerance_then_equal_weight_penalty_of_'
            'near_zero_valley_and_narrow_fwhm\n'
        )
        result_file.write(
            f'cvae_requested_wavelength_tolerance_nm\t'
            f'{args.wavelength_tolerance_nm:.10f}\n'
        )
        result_file.write(
            'cvae_valley_reference_transmittance\t'
            f'{args.valley_reference_transmittance:.10f}\n'
        )
        result_file.write(
            f'cvae_fwhm_reference_nm\t{args.fwhm_reference_nm:.10f}\n'
        )
        result_file.write(
            'cvae_latent_sampling\t'
            'same_seeded_candidate_latents_for_every_design_wavelength\n'
        )
        for key, value in direct_stats.items():
            result_file.write(f'direct_tandem_{key}\t{value:.10e}\n')
        for key, value in cvae_stats.items():
            result_file.write(f'cvae_tandem_{key}\t{value:.10e}\n')
        result_file.write(f'cvae_lower_absolute_error_cases\t{cvae_wins}\n')
        result_file.write(f'direct_lower_absolute_error_cases\t{direct_wins}\n')
        result_file.write(f'equal_absolute_error_cases\t{ties}\n')
        result_file.write(
            'mean_absolute_error_reduction_direct_minus_cvae_nm\t'
            f'{np.mean(improvement):.10e}\n'
        )
        result_file.write(
            'direct_tandem_mean_min_transmittance\t'
            f'{np.mean(direct_min_transmittance):.10e}\n'
        )
        result_file.write(
            f'direct_tandem_mean_fwhm_nm\t{np.mean(direct_fwhms):.10e}\n'
        )
        result_file.write(
            'cvae_tandem_mean_min_transmittance\t'
            f'{np.mean(cvae_min_transmittance):.10e}\n'
        )
        result_file.write(
            f'cvae_tandem_mean_fwhm_nm\t{np.mean(cvae_fwhms):.10e}\n'
        )
        result_file.write(
            'cvae_selection_fallback_cases\t'
            f'{int(np.sum(selection_fallback_flags))}\n'
        )


def main(argv=None):
    args = parse_args(argv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    design_wavelengths = np.linspace(
        args.start_wavelength,
        args.stop_wavelength,
        args.n_cases,
        dtype=float,
    )

    print(f'Device: {device}')
    print(
        f'Running {args.n_cases} cases from '
        f'{design_wavelengths[0]:.4f} to {design_wavelengths[-1]:.4f} nm'
    )

    seed_everything(args.random_seed)
    tandem_net = build_tandem_network(device)
    cvae_model = build_cvae(device)
    wavelengths = (
        tandem_net.wavelengths.detach().cpu().numpy().astype(float)
    )
    n_wavelengths = len(wavelengths)
    if n_wavelengths != ENCODER_SIZE[0]:
        raise ValueError(
            f'Expected {ENCODER_SIZE[0]} wavelength samples, '
            f'found {n_wavelengths}.'
        )

    target_indices = np.array(
        [
            int(np.argmin(np.abs(wavelengths - center_nm)))
            for center_nm in design_wavelengths
        ],
        dtype=int,
    )
    condition_values = target_indices / (n_wavelengths - 1)
    ideal_targets = np.stack(
        [
            lorentzian_trans(
                center_nm,
                gamma=args.gamma,
                wavelengths=wavelengths,
            )[1]
            for center_nm in design_wavelengths
        ],
        axis=0,
    ).astype(np.float32)

    print('Direct Tandem batch inference...')
    ideal_tensor = torch.as_tensor(
        ideal_targets,
        dtype=torch.float32,
        device=device,
    )
    direct_structures, direct_responses = tandem_net.test(
        torch.empty(0, device=device),
        ideal_tensor,
        'tandem',
    )

    print(
        f'CVAE candidate generation and Tandem evaluation '
        f'({args.n_candidates} candidates/case)...'
    )
    seed_everything(args.random_seed)
    base_latents = torch.randn(
        args.n_candidates,
        ENCODER_SIZE[-1],
        device=device,
    )
    selected_indices = np.empty(args.n_cases, dtype=int)
    minimum_peak_index_errors = np.empty(args.n_cases, dtype=int)
    selected_cvae_targets = np.empty_like(ideal_targets)
    selected_cvae_structures = np.empty((args.n_cases, 2), dtype=float)
    selected_cvae_responses = np.empty_like(ideal_targets)
    selected_quality_scores = np.empty(args.n_cases, dtype=float)
    selected_final_fwhms = np.empty(args.n_cases, dtype=float)
    selected_cvae_fwhms = np.empty(args.n_cases, dtype=float)
    effective_wavelength_limits = np.empty(args.n_cases, dtype=float)
    selection_fallback_flags = np.empty(args.n_cases, dtype=int)
    candidate_score_rows = []

    for case_index in range(args.n_cases):
        condition = torch.tensor(
            [[condition_values[case_index]]],
            dtype=torch.float32,
            device=device,
        ).expand(args.n_candidates, -1)
        with torch.no_grad():
            candidate_tensor = cvae_model.decoder(base_latents, condition)
        candidate_structures, candidate_responses = tandem_net.test(
            torch.empty(0, device=device),
            candidate_tensor,
            'tandem',
        )
        candidates = candidate_tensor.detach().cpu().numpy()
        candidate_min_indices = np.argmin(candidates, axis=1)
        peak_index_errors = np.abs(
            candidate_min_indices - target_indices[case_index]
        )
        minimum_peak_index_error = int(np.min(peak_index_errors))
        peak_matched = np.flatnonzero(
            peak_index_errors == minimum_peak_index_error
        )
        response_target_mse = np.mean(
            (
                candidate_responses
                - ideal_targets[case_index][None, :]
            ) ** 2,
            axis=1,
        )
        selection = select_single_notch_candidate(
            wavelengths,
            candidate_responses,
            design_wavelengths[case_index],
            wavelength_tolerance_nm=args.wavelength_tolerance_nm,
            valley_weight=0.5,
            fwhm_weight=0.5,
            valley_reference_transmittance=(
                args.valley_reference_transmittance
            ),
            fwhm_reference_nm=args.fwhm_reference_nm,
            tie_breaker=response_target_mse,
        )
        selected_index = selection.selected_index
        cvae_metrics = tuple(
            measure_notch(
                wavelengths,
                candidate,
                design_wavelengths[case_index],
            )
            for candidate in candidates
        )
        selected_indices[case_index] = selected_index
        minimum_peak_index_errors[case_index] = minimum_peak_index_error
        selected_cvae_targets[case_index] = candidates[selected_index]
        selected_cvae_structures[case_index] = candidate_structures[
            selected_index
        ]
        selected_cvae_responses[case_index] = candidate_responses[
            selected_index
        ]
        selected_quality_scores[case_index] = selection.quality_scores[
            selected_index
        ]
        selected_final_fwhms[case_index] = selection.metrics[
            selected_index
        ].fwhm_nm
        selected_cvae_fwhms[case_index] = cvae_metrics[
            selected_index
        ].fwhm_nm
        effective_wavelength_limits[case_index] = (
            selection.effective_wavelength_limit_nm
        )
        selection_fallback_flags[case_index] = int(
            selection.selection_mode == 'fallback_best_wavelength_band'
        )
        for candidate_index, (cvae_metric, final_metric) in enumerate(
            zip(cvae_metrics, selection.metrics)
        ):
            candidate_score_rows.append(
                [
                    case_index + 1,
                    design_wavelengths[case_index],
                    candidate_index,
                    int(candidate_index == selected_index),
                    int(selection.eligible_mask[candidate_index]),
                    final_metric.min_wavelength_nm,
                    final_metric.absolute_wavelength_error_nm,
                    final_metric.min_transmittance,
                    final_metric.fwhm_nm,
                    selection.valley_scores[candidate_index],
                    selection.fwhm_scores[candidate_index],
                    selection.quality_scores[candidate_index],
                    cvae_metric.min_wavelength_nm,
                    cvae_metric.min_transmittance,
                    cvae_metric.fwhm_nm,
                    response_target_mse[candidate_index],
                    candidate_structures[candidate_index][0],
                    candidate_structures[candidate_index][1],
                ]
            )
        selected_final_index = int(
            np.argmin(selected_cvae_responses[case_index])
        )
        print(
            f'  [{case_index + 1:02d}/{args.n_cases}] '
            f'target={design_wavelengths[case_index]:.4f} nm, '
            f'candidate={selected_index}, '
            f'final={wavelengths[selected_final_index]:.4f} nm'
        )

    direct_min_indices = np.argmin(direct_responses, axis=1)
    cvae_target_min_indices = np.argmin(selected_cvae_targets, axis=1)
    cvae_final_min_indices = np.argmin(selected_cvae_responses, axis=1)
    row_indices = np.arange(args.n_cases)

    direct_min_wavelengths = wavelengths[direct_min_indices]
    cvae_target_min_wavelengths = wavelengths[cvae_target_min_indices]
    cvae_final_min_wavelengths = wavelengths[cvae_final_min_indices]
    direct_signed_errors = direct_min_wavelengths - design_wavelengths
    cvae_signed_errors = cvae_final_min_wavelengths - design_wavelengths
    direct_absolute_errors = np.abs(direct_signed_errors)
    cvae_absolute_errors = np.abs(cvae_signed_errors)
    direct_spectral_mse = np.mean(
        (direct_responses - ideal_targets) ** 2,
        axis=1,
    )
    cvae_spectral_mse = np.mean(
        (selected_cvae_responses - ideal_targets) ** 2,
        axis=1,
    )
    cvae_to_tandem_mse = np.mean(
        (selected_cvae_responses - selected_cvae_targets) ** 2,
        axis=1,
    )
    direct_min_transmittance = direct_responses[
        row_indices, direct_min_indices
    ]
    cvae_target_min_transmittance = selected_cvae_targets[
        row_indices, cvae_target_min_indices
    ]
    cvae_final_min_transmittance = selected_cvae_responses[
        row_indices, cvae_final_min_indices
    ]
    direct_fwhms = np.asarray(
        [
            measure_notch(
                wavelengths,
                direct_responses[case_index],
                design_wavelengths[case_index],
            ).fwhm_nm
            for case_index in range(args.n_cases)
        ],
        dtype=float,
    )

    comparison_columns = np.column_stack((
        row_indices + 1,
        design_wavelengths,
        target_indices,
        wavelengths[target_indices],
        direct_structures[:, 0],
        direct_structures[:, 1],
        direct_min_indices,
        direct_min_wavelengths,
        direct_signed_errors,
        direct_absolute_errors,
        direct_min_transmittance,
        direct_spectral_mse,
        selected_indices,
        minimum_peak_index_errors,
        cvae_target_min_indices,
        cvae_target_min_wavelengths,
        cvae_target_min_transmittance,
        selected_cvae_structures[:, 0],
        selected_cvae_structures[:, 1],
        cvae_final_min_indices,
        cvae_final_min_wavelengths,
        cvae_signed_errors,
        cvae_absolute_errors,
        cvae_final_min_transmittance,
        cvae_spectral_mse,
        cvae_to_tandem_mse,
        direct_absolute_errors - cvae_absolute_errors,
        direct_fwhms,
        selected_cvae_fwhms,
        selected_final_fwhms,
        selected_quality_scores,
        effective_wavelength_limits,
        selection_fallback_flags,
    ))
    data_path = os.path.join(output_dir, 'comparison_data.txt')
    np.savetxt(
        data_path,
        comparison_columns,
        delimiter='\t',
        fmt='%.10e',
        header=(
            'case_index\tdesign_wavelength_nm\ttarget_grid_index\t'
            'target_grid_wavelength_nm\tdirect_R\tdirect_n_host\t'
            'direct_final_min_index\tdirect_final_min_wavelength_nm\t'
            'direct_signed_wavelength_error_nm\t'
            'direct_absolute_wavelength_error_nm\t'
            'direct_final_min_transmittance\t'
            'direct_ideal_to_tandem_mse\t'
            'cvae_selected_candidate_index\t'
            'cvae_minimum_peak_index_error\tcvae_target_min_index\t'
            'cvae_target_min_wavelength_nm\t'
            'cvae_target_min_transmittance\tcvae_R\tcvae_n_host\t'
            'cvae_final_min_index\tcvae_final_min_wavelength_nm\t'
            'cvae_signed_wavelength_error_nm\t'
            'cvae_absolute_wavelength_error_nm\t'
            'cvae_final_min_transmittance\t'
            'cvae_ideal_to_tandem_mse\tcvae_to_tandem_mse\t'
            'absolute_error_reduction_direct_minus_cvae_nm\t'
            'direct_final_fwhm_nm\tcvae_target_fwhm_nm\t'
            'cvae_final_fwhm_nm\tcvae_selected_quality_score\t'
            'cvae_effective_wavelength_limit_nm\t'
            'cvae_selection_used_fallback'
        ),
        comments='',
    )
    candidate_score_path = os.path.join(
        output_dir,
        'candidate_scores.txt',
    )
    np.savetxt(
        candidate_score_path,
        np.asarray(candidate_score_rows, dtype=float),
        delimiter='\t',
        fmt='%.10e',
        header=(
            'case_index\tdesign_wavelength_nm\tcandidate_index\tselected\t'
            'wavelength_eligible\tfinal_min_wavelength_nm\t'
            'final_abs_wavelength_error_nm\tfinal_min_transmittance\t'
            'final_fwhm_nm\tvalley_penalty_normalized\t'
            'fwhm_penalty_normalized\tquality_score\t'
            'cvae_min_wavelength_nm\tcvae_min_transmittance\t'
            'cvae_fwhm_nm\tideal_to_final_mse\tR\tn_host'
        ),
        comments='',
    )

    direct_stats = calculate_stats(
        direct_signed_errors,
        direct_spectral_mse,
    )
    cvae_stats = calculate_stats(
        cvae_signed_errors,
        cvae_spectral_mse,
    )
    result_path = os.path.join(output_dir, 'comparison_results.txt')
    write_summary(
        result_path,
        args,
        design_wavelengths,
        direct_stats,
        cvae_stats,
        direct_absolute_errors,
        cvae_absolute_errors,
        direct_min_transmittance,
        direct_fwhms,
        cvae_final_min_transmittance,
        selected_final_fwhms,
        selection_fallback_flags,
    )

    plot_path, wavelength_plot_path = create_comparison_plots(
        design_wavelengths,
        direct_absolute_errors,
        cvae_absolute_errors,
        output_dir,
    )

    print(f'Comparison data: {data_path}')
    print(f'Candidate scores: {candidate_score_path}')
    print(f'Summary results: {result_path}')
    print(f'Main ECDF plot: {plot_path}')
    print(f'Wavelength-resolved plot: {wavelength_plot_path}')
    print(
        'Mean absolute wavelength error: '
        f'direct={direct_stats["mean_absolute_error_nm"]:.4f} nm, '
        f'CVAE-to-Tandem={cvae_stats["mean_absolute_error_nm"]:.4f} nm'
    )


if __name__ == '__main__':
    main()
