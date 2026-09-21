import argparse
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter


matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.2,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'statistics_410_650')
DEFAULT_DATA_FILE = os.path.join(DEFAULT_OUTPUT_DIR, 'comparison_data.txt')
FIG_DPI = 600
SINGLE_COLUMN_SIZE = (3.5, 2.65)
PPT_BLUE = '#2E75B6'
PPT_ORANGE = '#ED7D31'
LIGHT_GRID = '#E5E5E5'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Create publication-ready comparison figures from the saved '
            '30-case Tandem and CVAE-to-Tandem results.'
        )
    )
    parser.add_argument('--data-file', default=DEFAULT_DATA_FILE)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


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
    ax.set_axisbelow(True)
    return fig, ax


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


def empirical_cdf(values):
    sorted_values = np.sort(np.asarray(values, dtype=float))
    cumulative_fraction = (
        np.arange(1, len(sorted_values) + 1, dtype=float)
        / len(sorted_values)
    )
    return (
        np.concatenate(([0.0], sorted_values)),
        np.concatenate(([0.0], cumulative_fraction)),
    )


def create_comparison_plots(
    design_wavelengths,
    direct_absolute_errors,
    cvae_absolute_errors,
    output_dir,
):
    design_wavelengths = np.asarray(design_wavelengths, dtype=float)
    direct_absolute_errors = np.asarray(
        direct_absolute_errors,
        dtype=float,
    )
    cvae_absolute_errors = np.asarray(cvae_absolute_errors, dtype=float)

    if not (
        design_wavelengths.ndim
        == direct_absolute_errors.ndim
        == cvae_absolute_errors.ndim
        == 1
    ):
        raise ValueError('All plotting inputs must be one-dimensional.')
    if not (
        len(design_wavelengths)
        == len(direct_absolute_errors)
        == len(cvae_absolute_errors)
    ):
        raise ValueError('Plotting inputs must have equal lengths.')
    if len(design_wavelengths) < 2:
        raise ValueError('At least two cases are required.')
    if not np.all(
        np.isfinite(
            np.concatenate((
                design_wavelengths,
                direct_absolute_errors,
                cvae_absolute_errors,
            ))
        )
    ):
        raise ValueError('Plotting inputs contain non-finite values.')

    os.makedirs(output_dir, exist_ok=True)
    direct_mae = float(np.mean(direct_absolute_errors))
    cvae_mae = float(np.mean(cvae_absolute_errors))

    # Main-text figure: ECDF clearly communicates the full error distribution
    # without implying that the jagged case-to-case fluctuations are smooth.
    direct_x, direct_y = empirical_cdf(direct_absolute_errors)
    cvae_x, cvae_y = empirical_cdf(cvae_absolute_errors)
    fig, ax = setup_figure()
    ax.step(
        direct_x,
        direct_y,
        where='post',
        color=PPT_BLUE,
        linewidth=1.35,
        label=f'Direct Tandem (MAE = {direct_mae:.2f} nm)',
        zorder=3,
    )
    ax.step(
        cvae_x,
        cvae_y,
        where='post',
        color=PPT_ORANGE,
        linewidth=1.45,
        linestyle=(0, (4, 2.4)),
        dash_capstyle='round',
        label=f'CVAE-assisted Tandem (MAE = {cvae_mae:.2f} nm)',
        zorder=4,
    )
    max_error = float(
        max(np.max(direct_absolute_errors), np.max(cvae_absolute_errors))
    )
    ax.set_xlim(0.0, max_error * 1.04)
    ax.set_ylim(0.0, 1.015)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis='y', color=LIGHT_GRID, linewidth=0.45)
    ax.set_xlabel('Absolute wavelength error (nm)')
    ax.set_ylabel('Cumulative fraction of cases')
    ax.legend(
        loc='lower right',
        frameon=False,
        handlelength=2.7,
        handletextpad=0.6,
        borderaxespad=0.5,
    )
    ecdf_path = os.path.join(output_dir, 'method_comparison_plot.png')
    save_figure(fig, ecdf_path)

    # Auxiliary figure: retain the wavelength-resolved result requested for
    # later analysis, but reduce line/marker weight and make ties visible.
    fig, ax = setup_figure()
    ax.plot(
        design_wavelengths,
        direct_absolute_errors,
        color=PPT_BLUE,
        linewidth=0.85,
        marker='o',
        markersize=3.2,
        markeredgewidth=0,
        solid_capstyle='round',
        label='Direct Tandem',
        zorder=3,
    )
    ax.plot(
        design_wavelengths,
        cvae_absolute_errors,
        color=PPT_ORANGE,
        linewidth=0.95,
        linestyle=(0, (4, 2.4)),
        marker='s',
        markersize=1.9,
        markeredgewidth=0,
        dash_capstyle='round',
        label='CVAE-assisted Tandem',
        zorder=4,
    )
    x_padding = 0.015 * (
        design_wavelengths[-1] - design_wavelengths[0]
    )
    ax.set_xlim(
        design_wavelengths[0] - x_padding,
        design_wavelengths[-1] + x_padding,
    )
    ax.set_ylim(0.0, max(1.0, max_error * 1.08))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis='y', color=LIGHT_GRID, linewidth=0.45)
    ax.set_xlabel('Design wavelength (nm)')
    ax.set_ylabel(r'$|\Delta\lambda|$ (nm)')
    ax.legend(
        loc='upper left',
        frameon=False,
        handlelength=2.6,
        handletextpad=0.6,
        borderaxespad=0.5,
    )
    wavelength_path = os.path.join(
        output_dir,
        'method_comparison_by_wavelength.png',
    )
    save_figure(fig, wavelength_path)

    return ecdf_path, wavelength_path


def load_saved_comparison(path):
    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 23:
        raise ValueError(
            f'Expected at least 23 columns in {path}, found {data.shape}.'
        )
    return data[:, 1], data[:, 9], data[:, 22]


def main(argv=None):
    args = parse_args(argv)
    data_file = os.path.abspath(args.data_file)
    output_dir = os.path.abspath(args.output_dir)
    design_wavelengths, direct_errors, cvae_errors = (
        load_saved_comparison(data_file)
    )
    ecdf_path, wavelength_path = create_comparison_plots(
        design_wavelengths,
        direct_errors,
        cvae_errors,
        output_dir,
    )
    print(f'Main ECDF figure: {ecdf_path}')
    print(f'Wavelength-resolved figure: {wavelength_path}')


if __name__ == '__main__':
    main()
