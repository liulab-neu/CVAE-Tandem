import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator

# -----------------------------------------------------------------------------
# Style (match ./1peak/input_no_theta.py)
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

PPT_BLUE = '#2E75B6'   # 0 degree
PPT_ORANGE = '#ED7D31' # 30 degree

SHOW_PLOT = False
SAVE_PLOT = True
SAVE_PDF = False  # 子图阶段只保存 PNG，整合多面板图时再统一导出
FIG_DPI = 600
SINGLE_COLUMN_SIZE = (3.5, 2.7)  # 约 89 mm，适合期刊单栏
DEFAULT_DESIGN_WAVELENGTH = 410.0
DEFAULT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_WINDOW_HALF_WIDTH_NM = 30.0


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
    ax.set_xlim(380, 800)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
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
    lower = np.floor(lower / 0.05) * 0.05
    upper = np.ceil(upper / 0.05) * 0.05
    if upper - lower < 0.20:
        midpoint = 0.5 * (upper + lower)
        lower = max(-0.02, midpoint - 0.10)
        upper = min(1.02, midpoint + 0.10)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _autoscale_axes(ax, x, ys, *, force_01=True):
    x = np.asarray(x, dtype=float)
    if x.size:
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        xpad = (xmax - xmin) * 0.02 if xmax > xmin else 1.0
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
            ypad = (y_max - y_min) * 0.06 if y_max > y_min else 0.1
            ax.set_ylim(y_min - ypad, y_max + ypad)


def save_fig(fig, basepath, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout(pad=0.45)
    fig.savefig(os.path.join(output_dir, basepath + '.png'), dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.04)
    if SAVE_PDF:
        fig.savefig(os.path.join(output_dir, basepath + '.pdf'), bbox_inches='tight', pad_inches=0.04)


def load_cst_metadata(path):
    metadata = {}
    with open(path, 'r', encoding='utf-8') as metadata_file:
        for line in metadata_file:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
    return metadata


def validate_cst_sources(source_files):
    metadata = [
        load_cst_metadata(os.path.splitext(source_file)[0] + '.a')
        for source_file in source_files
    ]
    expected_angles = (0.0, 30.0, 60.0)
    actual_angles = tuple(float(item['theta']) for item in metadata)
    if not np.allclose(actual_angles, expected_angles):
        raise ValueError(
            f'CST source angles are {actual_angles}, expected {expected_angles}. '
            'Please regenerate a coherent 0/30/60-degree data set.'
        )

    structure_keys = ('R', 'n_host1', 'n_host2')
    reference_structure = np.array(
        [float(metadata[0][key]) for key in structure_keys]
    )
    for source_index, item in enumerate(metadata[1:], start=2):
        structure = np.array([float(item[key]) for key in structure_keys])
        if not np.allclose(structure, reference_structure):
            raise ValueError(
                f'CST source 0000{source_index} uses structure {structure.tolist()}, '
                f'but 00001 uses {reference_structure.tolist()}. '
                'Refusing to save mixed CST data.'
            )
    return dict(zip(structure_keys, reference_structure))


def load_cvae_columns(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'CVAE plotting data not found: {path}')

    with open(path, 'r', encoding='utf-8') as data_file:
        column_names = data_file.readline().strip().split('\t')
    required_columns = (
        'wavelength_nm',
        'tandem_response_transmittance',
    )
    missing = set(required_columns) - set(column_names)
    if missing:
        raise ValueError(
            f'Missing CVAE data columns {sorted(missing)} in {path}.'
        )

    data = np.loadtxt(path, delimiter='\t', skiprows=1, ndmin=2)
    if data.shape[1] != len(column_names):
        raise ValueError(
            f'CVAE header has {len(column_names)} columns but data has '
            f'{data.shape[1]} columns in {path}.'
        )
    columns = {
        name: data[:, column_names.index(name)]
        for name in required_columns
    }
    if not all(np.all(np.isfinite(values)) for values in columns.values()):
        raise ValueError(f'Non-finite value found in {path}.')
    if not np.all(np.diff(columns['wavelength_nm']) > 0):
        raise ValueError(f'CVAE wavelength grid is not strictly increasing: {path}')
    return columns


def comparison_metrics(reference, prediction):
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    error = prediction - reference
    mse = float(np.mean(error ** 2))
    total_variance = float(np.sum((reference - np.mean(reference)) ** 2))
    r_squared = (
        float(1.0 - np.sum(error ** 2) / total_variance)
        if total_variance > 0
        else float('nan')
    )
    return {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'mae': float(np.mean(np.abs(error))),
        'max_abs_error': float(np.max(np.abs(error))),
        'r_squared': r_squared,
        'correlation': float(np.corrcoef(reference, prediction)[0, 1]),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Compare the final Tandem response with the equal-weight average '
            'of the 0/30/60-degree CST spectra.'
        )
    )
    parser.add_argument(
        '--design-wavelength',
        type=float,
        default=DEFAULT_DESIGN_WAVELENGTH,
        help='Design wavelength in nm (default: 410).',
    )
    parser.add_argument(
        '--base-dir',
        default=None,
        help=(
            'Directory containing <wavelength>/cst and <wavelength>/cvae. '
            'Defaults to the directory containing this script.'
        ),
    )
    return parser.parse_args()


def main(design_wavelength=DEFAULT_DESIGN_WAVELENGTH, base_dir=None):
    design_wavelength = float(design_wavelength)
    if not np.isfinite(design_wavelength) or design_wavelength <= 0:
        raise ValueError('design_wavelength must be a finite positive number.')

    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    base_dir = os.path.abspath(os.path.expanduser(base_dir))
    wavelength_dir_name = f'{design_wavelength:g}'
    output_dir = os.path.join(base_dir, wavelength_dir_name, 'cst')
    cst_source_data_dir = output_dir
    cvae_data_file = os.path.join(
        base_dir,
        wavelength_dir_name,
        'cvae',
        'cvae_data.txt',
    )
    cvae_structure_file = os.path.join(
        base_dir,
        wavelength_dir_name,
        'cvae',
        'cvae_structure.txt',
    )

    # 00001.b / 00002.b / 00003.b correspond to 0/30/60 degrees.
    # Their equal-weight arithmetic mean is the spectrum used for comparison
    # with the final CVAE-designed structure's Tandem/FNN response.
    f0 = os.path.join(cst_source_data_dir, '00001.b')
    f30 = os.path.join(cst_source_data_dir, '00002.b')
    f60 = os.path.join(cst_source_data_dir, '00003.b')
    structure = validate_cst_sources((f0, f30, f60))
    if not os.path.isfile(cvae_structure_file):
        raise FileNotFoundError(
            f'CVAE structure not found: {cvae_structure_file}'
        )
    selected_structure = np.loadtxt(
        cvae_structure_file,
        skiprows=1,
        ndmin=2,
    )
    if selected_structure.shape != (1, 2):
        raise ValueError(
            f'Expected one R,n_host row in {cvae_structure_file}.'
        )
    expected_cst_structure = np.array([
        selected_structure[0, 0],
        selected_structure[0, 1],
        selected_structure[0, 1],
    ])
    actual_cst_structure = np.array([
        structure['R'],
        structure['n_host1'],
        structure['n_host2'],
    ])
    if not np.allclose(
        actual_cst_structure,
        expected_cst_structure,
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError(
            f'CST structure {actual_cst_structure.tolist()} does not match '
            f'the selected CVAE structure '
            f'{expected_cst_structure.tolist()}. Regenerate CST first.'
        )

    spec0 = np.loadtxt(f0, delimiter='\t')
    spec30 = np.loadtxt(f30, delimiter='\t')
    spec60 = np.loadtxt(f60, delimiter='\t')

    wl0, t0 = spec0[:, 0], spec0[:, 1]
    wl30, t30 = spec30[:, 0], spec30[:, 1]
    wl60, t60 = spec60[:, 0], spec60[:, 1]
    if not (
        np.allclose(wl0, wl30, rtol=0.0, atol=1e-8)
        and np.allclose(wl0, wl60, rtol=0.0, atol=1e-8)
    ):
        raise ValueError('The 0/30/60-degree CST wavelength grids do not match.')
    if not all(
        np.all(np.isfinite(values))
        for values in (wl0, t0, t30, t60)
    ):
        raise ValueError('Non-finite value found in CST source data.')

    cvae_columns = load_cvae_columns(cvae_data_file)
    cvae_wavelength = cvae_columns['wavelength_nm']
    if (
        cvae_wavelength[0] < wl0[0]
        or cvae_wavelength[-1] > wl0[-1]
    ):
        raise ValueError(
            'The CVAE wavelength range is not contained in the CST range; '
            'comparison would require extrapolation.'
        )

    # Use the native CVAE wavelength grid for the quantitative comparison.
    # It lies completely inside the CST range, so only interpolation is used.
    cst_0_aligned = np.interp(cvae_wavelength, wl0, t0)
    cst_30_aligned = np.interp(cvae_wavelength, wl0, t30)
    cst_60_aligned = np.interp(cvae_wavelength, wl0, t60)
    cst_angle_stack = np.vstack(
        (cst_0_aligned, cst_30_aligned, cst_60_aligned)
    )
    cst_average = np.mean(cst_angle_stack, axis=0)
    final_response = cvae_columns['tandem_response_transmittance']

    # Final paper panel: final predicted response versus angle-averaged CST.
    fig, ax = setup_figure()
    ax.plot(
        cvae_wavelength,
        final_response,
        color=PPT_BLUE,
        linewidth=1.5,
        solid_capstyle='round',
        zorder=3,
    )
    ax.plot(
        cvae_wavelength,
        cst_average,
        color=PPT_ORANGE,
        linewidth=1.35,
        linestyle=(0, (4, 2.4)),
        dash_capstyle='round',
        zorder=2,
    )
    ax.set_xlim(380, 800)
    set_adaptive_y_limits(ax, final_response, cst_average)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmittance')
    if SAVE_PLOT:
        save_fig(fig, 'cst_plot', output_dir)
    if SHOW_PLOT:
        plt.show()
    plt.close(fig)

    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, 'cst_data.txt')
    np.savetxt(
        data_path,
        np.column_stack(
            (
                cvae_wavelength,
                final_response,
                cst_average,
            )
        ),
        delimiter='\t',
        fmt='%.10e',
        header=(
            'wavelength_nm\tfinal_result_transmittance\t'
            'cst_average_transmittance'
        ),
        comments='',
    )

    result_path = os.path.join(output_dir, 'cst_results.txt')
    comparison = comparison_metrics(final_response, cst_average)
    cst_average_min_index = int(np.argmin(cst_average))
    final_min_index = int(np.argmin(final_response))
    local_window_min = design_wavelength - LOCAL_WINDOW_HALF_WIDTH_NM
    local_window_max = design_wavelength + LOCAL_WINDOW_HALF_WIDTH_NM
    local_mask = (
        (cvae_wavelength >= local_window_min)
        & (cvae_wavelength <= local_window_max)
    )
    if not np.any(local_mask):
        raise ValueError(
            f'No wavelength samples fall inside the local comparison window '
            f'[{local_window_min:g}, {local_window_max:g}] nm.'
        )
    local_rmse = float(
        np.sqrt(
            np.mean(
                (
                    cst_average[local_mask]
                    - final_response[local_mask]
                ) ** 2
            )
        )
    )
    with open(result_path, 'w', encoding='utf-8') as result_file:
        result_file.write(
            'validation_type\texact_structure_CST_simulation\n'
        )
        result_file.write('exact_structure_CST_simulation\t1\n')
        result_file.write(
            f'source_structure_file\t{cvae_structure_file}\n'
        )
        result_file.write(f'cst_source_0deg_file\t{f0}\n')
        result_file.write(f'cst_source_30deg_file\t{f30}\n')
        result_file.write(f'cst_source_60deg_file\t{f60}\n')
        result_file.write(
            f'design_wavelength_nm\t{design_wavelength:.10f}\n'
        )
        result_file.write(f'R\t{structure["R"]:.10f}\n')
        result_file.write(f'n_host1\t{structure["n_host1"]:.10f}\n')
        result_file.write(f'n_host2\t{structure["n_host2"]:.10f}\n')
        result_file.write(f'final_result_source_file\t{cvae_data_file}\n')
        result_file.write(
            'final_result_source_column\t'
            'tandem_response_transmittance\n'
        )
        result_file.write('cst_average_angles_deg\t0,30,60\n')
        result_file.write(
            'cst_average_formula\t'
            '(T_0deg + T_30deg + T_60deg) / 3\n'
        )
        result_file.write(
            f'cst_average_min_index\t{cst_average_min_index}\n'
        )
        result_file.write(
            'cst_average_min_wavelength_nm\t'
            f'{cvae_wavelength[cst_average_min_index]:.10f}\n'
        )
        result_file.write(
            'cst_average_min_transmittance\t'
            f'{cst_average[cst_average_min_index]:.10e}\n'
        )
        result_file.write(
            f'final_result_min_index\t{final_min_index}\n'
        )
        result_file.write(
            'final_result_min_wavelength_nm\t'
            f'{cvae_wavelength[final_min_index]:.10f}\n'
        )
        result_file.write(
            'final_result_min_transmittance\t'
            f'{final_response[final_min_index]:.10e}\n'
        )
        result_file.write(
            'cst_average_minus_final_min_wavelength_nm\t'
            f'{cvae_wavelength[cst_average_min_index] - cvae_wavelength[final_min_index]:.10f}\n'
        )
        result_file.write(
            'cst_average_minus_final_min_transmittance\t'
            f'{cst_average[cst_average_min_index] - final_response[final_min_index]:.10e}\n'
        )
        for metric_name, metric_value in comparison.items():
            result_file.write(
                f'cst_average_vs_final_{metric_name}\t'
                f'{metric_value:.10e}\n'
            )
        result_file.write(
            f'cst_average_vs_final_local_{local_window_min:g}_'
            f'{local_window_max:g}nm_rmse\t'
            f'{local_rmse:.10e}\n'
        )

    print(
        'CST average valley: '
        f'{cvae_wavelength[cst_average_min_index]:.4f} nm; '
        'final result valley: '
        f'{cvae_wavelength[final_min_index]:.4f} nm'
    )
    print(
        'CST average vs final result: '
        f'RMSE={comparison["rmse"]:.10e}, '
        f'correlation={comparison["correlation"]:.10f}'
    )
    print(f'Final-result/CST comparison outputs saved to: {output_dir}')


if __name__ == '__main__':
    args = parse_args()
    main(
        design_wavelength=args.design_wavelength,
        base_dir=args.base_dir,
    )
