import argparse
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

import CVAE
import tandem
from notch_quality import measure_notch, select_single_notch_candidate


# -----------------------------------------------------------------------------
# Matplotlib style (match ./1peak/input_no_theta.py)
# -----------------------------------------------------------------------------
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

SHOW_PLOT = False
SAVE_PLOT = True
SAVE_PDF = False  # 子图阶段只保存 PNG，整合多面板图时再统一导出
FIG_DPI = 600
SINGLE_COLUMN_SIZE = (3.5, 2.7)  # 约 89 mm，适合期刊单栏
DEFAULT_TARGET_WAVELENGTH = 410
DEFAULT_RANDOM_SEED = 42
DEFAULT_N_CANDIDATES = 20
DEFAULT_WAVELENGTH_TOLERANCE_NM = 2.0
DEFAULT_MAXIMUM_FINAL_MIN_TRANSMITTANCE = 0.1
DEFAULT_VALLEY_REFERENCE_TRANSMITTANCE = 0.1
DEFAULT_FWHM_REFERENCE_NM = 30.0

PPT_BLUE = '#2E75B6'
PPT_ORANGE = '#ED7D31'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'cvae_best_mean.pth')
TANDEM_INN_PATH = os.path.join(BASE_DIR, 'model', 'DNN_tandem_INN_label.ckpt')
TANDEM_FNN_PATH = os.path.join(BASE_DIR, 'model', 'DNN_tandem_FNN_label.ckpt')


def setup_figure(figsize=SINGLE_COLUMN_SIZE):
    fig, ax = plt.subplots(figsize=figsize)
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
    ax.xaxis.set_major_locator(FixedLocator([400, 500, 600, 700, 800]))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.set_xlim(380, 800)
    ax.set_ylim(-0.025, 1.025)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return fig, ax


def save_fig(fig, basepath, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(os.path.join(output_dir, basepath + '.png'), dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.04)
    if SAVE_PDF:
        fig.savefig(os.path.join(output_dir, basepath + '.pdf'), bbox_inches='tight', pad_inches=0.04)


def _autoscale_axes(ax, x, ys, *, xpad_frac=0.02, ypad_frac=0.06, force_01=False):
    x = np.asarray(x, dtype=float)
    if x.size:
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        xpad = (xmax - xmin) * xpad_frac if xmax > xmin else 1.0
        ax.set_xlim(xmin - xpad, xmax + xpad)

    y_min = np.inf
    y_max = -np.inf
    for y in ys:
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            continue
        y_min = min(y_min, np.nanmin(y))
        y_max = max(y_max, np.nanmax(y))

    if np.isfinite(y_min) and np.isfinite(y_max):
        if force_01 and y_min >= -0.05 and y_max <= 1.05:
            ax.set_ylim(0.0, 1.0)
        else:
            ypad = (y_max - y_min) * ypad_frac if y_max > y_min else 0.1
            ax.set_ylim(y_min - ypad, y_max + ypad)


def lorentzian_trans(
    center_nm,
    wlmin=380,
    wlmax=800,
    num=1001,
    gamma=15,
    wavelengths=None,
):
    if wavelengths is None:
        wl = np.linspace(wlmin, wlmax, num)
    else:
        wl = np.asarray(wavelengths, dtype=float)
    spe = 1 / (1 + ((wl - center_nm) / gamma) ** 2)
    # convert to "dip" transmittance: 0 at center, -> 1 far away
    spe = (spe - 1) * -1
    return wl, spe


def cvae_generate_spectrum(
    center_nm,
    *,
    n_samples=DEFAULT_N_CANDIDATES,
    wlmin=380,
    wlmax=800,
    condition_value=None,
    random_seed=None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CVAE hyper-params must match training
    encoder_size = [1001, 1024, 2048, 1024, 512, 256, 128, 20]
    decoder_size = [20, 128, 256, 512, 1024, 2048, 1024, 1001]
    c_dim = 1

    model = CVAE.CVAE(encoder_size=encoder_size, decoder_size=decoder_size, c_dim=c_dim).to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model ckpt: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    # Training used min_index / (n_points - 1), so inference must use
    # the same index-based condition rather than a nominal wavelength formula.
    if condition_value is None:
        condition_value = (center_nm - wlmin) / (wlmax - wlmin)
    c = torch.tensor([[condition_value]], dtype=torch.float32, device=device)

    # sample latent z and decode multiple spectra
    z = torch.randn(n_samples, encoder_size[-1], device=device)
    with torch.no_grad():
        conditions = c.expand(n_samples, -1)
        spectra = model.decoder(z, conditions)
    return spectra


def build_tandem_network(device):
    n_input = 2
    n_classes = 1001
    fnn_size = [n_input, 64, 128, 256, 512, 1024, 2048, 1024, n_classes]
    inn_size = [n_classes, 1024, 2048, 1024, 512, 256, 128, 64, n_input]
    network = tandem.tandem_network(
        INN_size=inn_size,
        FNN_size=fnn_size,
        training=False,
        structure_min=torch.tensor([2.5, 1.5]),
        structure_max=torch.tensor([6.5, 3.05]),
    ).to(device)
    network.restore_INN(TANDEM_INN_PATH)
    network.restore_FNN(TANDEM_FNN_PATH)
    return network


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Generate a CVAE target spectrum, reconstruct it through the '
            'Tandem network, and save the plot and numeric results.'
        )
    )
    parser.add_argument(
        '--target-wavelength',
        type=float,
        default=DEFAULT_TARGET_WAVELENGTH,
        help=f'target dip wavelength in nm (default: {DEFAULT_TARGET_WAVELENGTH})',
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f'random seed for CVAE sampling (default: {DEFAULT_RANDOM_SEED})',
    )
    parser.add_argument(
        '--n-candidates',
        type=int,
        default=DEFAULT_N_CANDIDATES,
        help=f'number of CVAE candidates to evaluate (default: {DEFAULT_N_CANDIDATES})',
    )
    parser.add_argument(
        '--wavelength-tolerance-nm',
        type=float,
        default=DEFAULT_WAVELENGTH_TOLERANCE_NM,
        help=(
            'maximum final-response resonance error allowed before depth/FWHM '
            f'ranking (default: {DEFAULT_WAVELENGTH_TOLERANCE_NM:g} nm)'
        ),
    )
    parser.add_argument(
        '--maximum-final-min-transmittance',
        type=float,
        default=DEFAULT_MAXIMUM_FINAL_MIN_TRANSMITTANCE,
        help=(
            'maximum final-response notch minimum allowed before '
            'wavelength ranking '
            f'(default: {DEFAULT_MAXIMUM_FINAL_MIN_TRANSMITTANCE:g})'
        ),
    )
    parser.add_argument(
        '--valley-reference-transmittance',
        type=float,
        default=DEFAULT_VALLEY_REFERENCE_TRANSMITTANCE,
        help=(
            'normalization reference for the near-zero valley penalty '
            f'(default: {DEFAULT_VALLEY_REFERENCE_TRANSMITTANCE:g})'
        ),
    )
    parser.add_argument(
        '--fwhm-reference-nm',
        type=float,
        default=DEFAULT_FWHM_REFERENCE_NM,
        help=(
            'normalization reference for the narrow-FWHM penalty '
            f'(default: {DEFAULT_FWHM_REFERENCE_NM:g} nm)'
        ),
    )
    args = parser.parse_args(argv)
    if args.n_candidates < 1:
        parser.error('--n-candidates must be at least 1')
    if args.wavelength_tolerance_nm < 0:
        parser.error('--wavelength-tolerance-nm must be non-negative')
    if args.maximum_final_min_transmittance < 0:
        parser.error(
            '--maximum-final-min-transmittance must be non-negative'
        )
    if args.valley_reference_transmittance <= 0:
        parser.error('--valley-reference-transmittance must be positive')
    if args.fwhm_reference_nm <= 0:
        parser.error('--fwhm-reference-nm must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    wlmin = 380
    wlmax = 800
    gamma = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    center_nm = float(args.target_wavelength)
    random_seed = args.random_seed
    n_candidates = args.n_candidates
    output_dir = os.path.join(BASE_DIR, f'{center_nm:g}', 'cvae')
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    tandem_net = build_tandem_network(device)
    wl = tandem_net.wavelengths.detach().cpu().numpy()
    target_grid_index = int(np.argmin(np.abs(wl - center_nm)))
    condition_value = target_grid_index / (len(wl) - 1)
    _, ideal_target = lorentzian_trans(
        center_nm,
        wlmin=wlmin,
        wlmax=wlmax,
        gamma=gamma,
        wavelengths=wl,
    )

    # Generate multiple CVAE candidates and pass all through Tandem.  Candidate
    # quality is judged on the *final Tandem response*: first require a finite
    # FWHM and a sufficiently deep notch, then enforce resonance-wavelength
    # tolerance, and finally minimize an equal-weight, dimensionless penalty
    # for a near-zero valley and narrow half-depth FWHM.
    # Reset the seed after network construction so candidate z values do not
    # depend on random parameter initialization that is immediately restored.
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    cvae_spectra = cvae_generate_spectrum(
        center_nm,
        n_samples=n_candidates,
        wlmin=wlmin,
        wlmax=wlmax,
        condition_value=condition_value,
        random_seed=random_seed,
    ).to(device)
    structures, tandem_responses = tandem_net.test(
        torch.empty(0, device=device),
        cvae_spectra,
        'tandem',
    )
    cvae_candidates = cvae_spectra.detach().cpu().numpy()
    response_target_mse = np.mean(
        (tandem_responses - ideal_target[None, :]) ** 2,
        axis=1,
    )
    # Allow half a wavelength-grid cell when gating the decoded CVAE target.
    # This avoids rejecting a physically equivalent discrete minimum solely
    # because the requested wavelength lies between two sampled wavelengths.
    cvae_grid_slack_nm = 0.5 * float(np.median(np.diff(wl)))
    selection = select_single_notch_candidate(
        wl,
        tandem_responses,
        center_nm,
        wavelength_tolerance_nm=args.wavelength_tolerance_nm,
        valley_weight=0.5,
        fwhm_weight=0.5,
        valley_reference_transmittance=(
            args.valley_reference_transmittance
        ),
        fwhm_reference_nm=args.fwhm_reference_nm,
        maximum_final_min_transmittance=(
            args.maximum_final_min_transmittance
        ),
        tie_breaker=response_target_mse,
        generated_responses=cvae_candidates,
        generated_wavelength_tolerance_nm=args.wavelength_tolerance_nm,
        generated_grid_slack_nm=cvae_grid_slack_nm,
    )
    selected_index = selection.selected_index
    cvae_target = cvae_candidates[selected_index]
    tandem_response = tandem_responses[selected_index]
    predicted_structure = structures[selected_index]

    ideal_idx = int(np.argmin(ideal_target))
    cvae_idx = int(np.argmin(cvae_target))
    tandem_idx = int(np.argmin(tandem_response))

    fig, ax = setup_figure()
    ax.plot(
        wl,
        cvae_target,
        color=PPT_ORANGE,
        linewidth=1.35,
        linestyle=(0, (4, 2.4)),
        dash_capstyle='round',
        zorder=2,
    )
    ax.plot(
        wl,
        tandem_response,
        color=PPT_BLUE,
        linewidth=1.5,
        solid_capstyle='round',
        zorder=3,
    )
    ax.scatter(
        wl[cvae_idx],
        cvae_target[cvae_idx],
        marker='D',
        facecolors='white',
        edgecolors=PPT_ORANGE,
        s=16,
        zorder=5,
        linewidths=0.8,
    )
    ax.scatter(
        wl[tandem_idx],
        tandem_response[tandem_idx],
        color=PPT_BLUE,
        s=15,
        zorder=6,
        edgecolors='white',
        linewidths=0.5,
    )
    ax.annotate(
        f'{wl[cvae_idx]:.1f}',
        xy=(wl[cvae_idx], cvae_target[cvae_idx]),
        xytext=(20, 23),
        textcoords='offset points',
        fontsize=6.7,
        color='#333333',
        ha='left',
        va='bottom',
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
    ax.annotate(
        f'{wl[tandem_idx]:.1f}',
        xy=(wl[tandem_idx], tandem_response[tandem_idx]),
        xytext=(44, 9),
        textcoords='offset points',
        fontsize=6.7,
        color='#333333',
        ha='left',
        va='bottom',
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
    ax.set_xlim(380, 800)
    _autoscale_axes(
        ax,
        wl,
        (cvae_target, tandem_response),
        xpad_frac=0.0,
        ypad_frac=0.04,
        force_01=False,
    )
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmittance')

    base = 'cvae_plot'
    if SAVE_PLOT:
        save_fig(fig, base, output_dir)
    if SHOW_PLOT:
        plt.show()
    plt.close(fig)

    data_path = os.path.join(output_dir, 'cvae_data.txt')
    np.savetxt(
        data_path,
        np.column_stack((wl, ideal_target, cvae_target, tandem_response)),
        delimiter='\t',
        fmt='%.10e',
        header=(
            'wavelength_nm\tideal_target_transmittance\t'
            'cvae_target_transmittance\ttandem_response_transmittance'
        ),
        comments='',
    )

    ideal_cvae_mse = float(np.mean((cvae_target - ideal_target) ** 2))
    ideal_tandem_mse = float(np.mean((tandem_response - ideal_target) ** 2))
    cvae_tandem_mse = float(np.mean((tandem_response - cvae_target) ** 2))
    if selection.generated_metrics is None:
        raise RuntimeError('CVAE metrics missing from candidate selection')
    cvae_metrics = selection.generated_metrics
    selected_cvae_metrics = cvae_metrics[selected_index]
    selected_tandem_metrics = selection.metrics[selected_index]

    candidate_score_path = os.path.join(
        output_dir,
        'cvae_candidate_scores.txt',
    )
    score_rows = []
    for candidate_index, (cvae_metric, final_metric) in enumerate(
        zip(cvae_metrics, selection.metrics)
    ):
        score_rows.append(
            [
                candidate_index,
                int(candidate_index == selected_index),
                int(selection.final_notch_valid_mask[candidate_index]),
                int(selection.final_eligible_mask[candidate_index]),
                int(selection.generated_eligible_mask[candidate_index]),
                int(selection.eligible_mask[candidate_index]),
                cvae_metric.min_wavelength_nm,
                cvae_metric.absolute_wavelength_error_nm,
                cvae_metric.min_transmittance,
                cvae_metric.fwhm_nm,
                final_metric.min_wavelength_nm,
                final_metric.wavelength_error_nm,
                final_metric.absolute_wavelength_error_nm,
                final_metric.min_transmittance,
                final_metric.baseline_transmittance,
                final_metric.notch_depth,
                final_metric.fwhm_nm,
                selection.valley_scores[candidate_index],
                selection.fwhm_scores[candidate_index],
                selection.quality_scores[candidate_index],
                response_target_mse[candidate_index],
                float(
                    np.mean(
                        (
                            tandem_responses[candidate_index]
                            - cvae_candidates[candidate_index]
                        )
                        ** 2
                    )
                ),
                structures[candidate_index][0],
                structures[candidate_index][1],
            ]
        )
    np.savetxt(
        candidate_score_path,
        np.asarray(score_rows, dtype=float),
        delimiter='\t',
        fmt='%.10e',
        header=(
            'candidate_index\tselected\tfinal_notch_valid\t'
            'final_wavelength_eligible\t'
            'cvae_wavelength_eligible\tjointly_eligible\t'
            'cvae_min_wavelength_nm\tcvae_abs_wavelength_error_nm\t'
            'cvae_min_transmittance\tcvae_fwhm_nm\t'
            'final_min_wavelength_nm\tfinal_wavelength_error_nm\t'
            'final_abs_wavelength_error_nm\tfinal_min_transmittance\t'
            'final_baseline_transmittance\tfinal_notch_depth\t'
            'final_fwhm_nm\tvalley_penalty_normalized\t'
            'fwhm_penalty_normalized\tquality_score\t'
            'ideal_to_final_mse\tcvae_to_final_mse\tR\tn_host'
        ),
        comments='',
    )

    result_path = os.path.join(output_dir, 'cvae_results.txt')
    with open(result_path, 'w', encoding='utf-8') as result_file:
        result_file.write(f'random_seed\t{random_seed}\n')
        result_file.write(f'n_candidates\t{n_candidates}\n')
        result_file.write(f'selected_candidate_index\t{selected_index}\n')
        result_file.write(
            'selection_metric\t'
            'final_notch_depth_and_fwhm_gate_then_joint_final_and_cvae_'
            'wavelength_gate_then_minimum_final_wavelength_error_then_'
            'final_depth_fwhm_quality\n'
        )
        result_file.write(
            f'selection_priority\t{selection.ranking_rule}\n'
        )
        result_file.write(
            f'selection_mode\t{selection.selection_mode}\n'
        )
        result_file.write(
            'requested_wavelength_tolerance_nm\t'
            f'{selection.wavelength_tolerance_nm:.10f}\n'
        )
        result_file.write(
            'effective_wavelength_limit_nm\t'
            f'{selection.effective_wavelength_limit_nm:.10f}\n'
        )
        result_file.write(
            'requested_maximum_final_min_transmittance\t'
            f'{selection.maximum_final_min_transmittance:.10f}\n'
        )
        result_file.write(
            'effective_maximum_final_min_transmittance\t'
            f'{selection.effective_maximum_final_min_transmittance:.10f}\n'
        )
        result_file.write(
            'final_depth_gate_fallback_used\t'
            f'{int(selection.final_depth_gate_fallback_used)}\n'
        )
        result_file.write(
            'cvae_requested_wavelength_tolerance_nm\t'
            f'{selection.generated_wavelength_tolerance_nm:.10f}\n'
        )
        result_file.write(
            'cvae_grid_quantization_slack_nm\t'
            f'{selection.generated_grid_slack_nm:.10f}\n'
        )
        result_file.write(
            'cvae_effective_wavelength_limit_nm\t'
            f'{selection.generated_effective_wavelength_limit_nm:.10f}\n'
        )
        result_file.write(
            f'n_wavelength_eligible_candidates\t'
            f'{int(np.sum(selection.eligible_mask))}\n'
        )
        result_file.write(
            'n_final_notch_valid_candidates\t'
            f'{int(np.sum(selection.final_notch_valid_mask))}\n'
        )
        result_file.write(
            'n_final_wavelength_eligible_candidates\t'
            f'{int(np.sum(selection.final_eligible_mask))}\n'
        )
        result_file.write(
            'n_cvae_wavelength_eligible_candidates\t'
            f'{int(np.sum(selection.generated_eligible_mask))}\n'
        )
        result_file.write(
            'n_jointly_eligible_candidates\t'
            f'{int(np.sum(selection.eligible_mask))}\n'
        )
        result_file.write('valley_quality_weight\t0.5000000000\n')
        result_file.write('fwhm_quality_weight\t0.5000000000\n')
        result_file.write(
            'valley_reference_transmittance\t'
            f'{selection.valley_reference_transmittance:.10f}\n'
        )
        result_file.write(
            f'fwhm_reference_nm\t{selection.fwhm_reference_nm:.10f}\n'
        )
        result_file.write(
            'selected_quality_score\t'
            f'{selection.quality_scores[selected_index]:.10e}\n'
        )
        result_file.write(
            f'condition_index_normalized\t{condition_value:.10f}\n'
        )
        result_file.write(f'target_wavelength_nm\t{center_nm:.10f}\n')
        result_file.write(f'ideal_target_min_index\t{ideal_idx}\n')
        result_file.write(
            f'ideal_target_min_wavelength_nm\t{wl[ideal_idx]:.10f}\n'
        )
        result_file.write(f'cvae_min_index\t{cvae_idx}\n')
        result_file.write(f'cvae_min_wavelength_nm\t{wl[cvae_idx]:.10f}\n')
        result_file.write(
            'cvae_wavelength_error_nm\t'
            f'{selected_cvae_metrics.wavelength_error_nm:.10f}\n'
        )
        result_file.write(
            'cvae_abs_wavelength_error_nm\t'
            f'{selected_cvae_metrics.absolute_wavelength_error_nm:.10f}\n'
        )
        result_file.write(
            f'cvae_min_transmittance\t{cvae_target[cvae_idx]:.10e}\n'
        )
        result_file.write(
            f'cvae_fwhm_nm\t{selected_cvae_metrics.fwhm_nm:.10f}\n'
        )
        result_file.write(f'tandem_min_index\t{tandem_idx}\n')
        result_file.write(
            f'tandem_min_wavelength_nm\t{wl[tandem_idx]:.10f}\n'
        )
        result_file.write(
            f'tandem_min_transmittance\t{tandem_response[tandem_idx]:.10e}\n'
        )
        result_file.write(
            'tandem_baseline_transmittance\t'
            f'{selected_tandem_metrics.baseline_transmittance:.10e}\n'
        )
        result_file.write(
            f'tandem_notch_depth\t{selected_tandem_metrics.notch_depth:.10e}\n'
        )
        result_file.write(
            f'tandem_fwhm_nm\t{selected_tandem_metrics.fwhm_nm:.10f}\n'
        )
        result_file.write(
            f'tandem_wavelength_error_nm\t{wl[tandem_idx] - center_nm:.10f}\n'
        )
        result_file.write(
            'tandem_abs_wavelength_error_nm\t'
            f'{selected_tandem_metrics.absolute_wavelength_error_nm:.10f}\n'
        )
        result_file.write(f'predicted_R\t{predicted_structure[0]:.10f}\n')
        result_file.write(
            f'predicted_n_host\t{predicted_structure[1]:.10f}\n'
        )
        result_file.write(f'ideal_to_cvae_mse\t{ideal_cvae_mse:.10e}\n')
        result_file.write(f'ideal_to_tandem_mse\t{ideal_tandem_mse:.10e}\n')
        result_file.write(f'cvae_to_tandem_mse\t{cvae_tandem_mse:.10e}\n')
        result_file.write(
            f'cvae_to_tandem_rmse\t{np.sqrt(cvae_tandem_mse):.10e}\n'
        )

    structure_path = os.path.join(output_dir, 'cvae_structure.txt')
    np.savetxt(
        structure_path,
        np.asarray(predicted_structure[:2], dtype=float).reshape(1, -1),
        delimiter='\t',
        fmt='%.10f',
        header='R\tn_host',
        comments='',
    )

    print(
        f'Selected CVAE candidate: {selected_index}/{n_candidates - 1}, '
        f'structure={predicted_structure}'
    )
    print(
        f'CVAE valley: {wl[cvae_idx]:.4f} nm, '
        f'Tandem valley: {wl[tandem_idx]:.4f} nm'
    )
    print(
        f'CVAE-to-Tandem MSE: {cvae_tandem_mse:.10e}; '
        f'outputs saved to: {output_dir}'
    )
    print(
        f'Final notch quality: Tmin={selected_tandem_metrics.min_transmittance:.5f}, '
        f'FWHM={selected_tandem_metrics.fwhm_nm:.3f} nm, '
        f'selection={selection.selection_mode}'
    )


if __name__ == '__main__':
    main()
