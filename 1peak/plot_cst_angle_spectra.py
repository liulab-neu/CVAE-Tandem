#!/usr/bin/env python3
"""Plot the 0/30/60-degree CST spectra of a single-notch design.

Reads the per-angle exports written by Filter_CST_DataGen_Share/main.py
(00001.b = 0 deg, 00002.b = 30 deg, 00003.b = 60 deg) and draws them on one
axis, styled to match the other single-column figures in the paper.
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator

# Font sizes and line weights are copied from
# plot_spectral_comparison_3x3.py so this figure matches the 3x3 panels when
# both are placed in the manuscript at the same scale.
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Tinos', 'DejaVu Serif', 'serif'],
    'font.size': 16.0,
    'axes.labelsize': 17.0,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 16.0,
    'ytick.labelsize': 16.0,
    'legend.fontsize': 16.0,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DPI = 600
# The 3x3 figure shares one axis label across three stacked panels, so a
# single panel's 2.05 in axes box would leave a 15.5 pt label almost as tall
# as the plot.  Scale the axes box up by ~1.5 so the label length relative to
# the axis matches what the 3x3 shows, keeping the same font sizes.
FIGSIZE = (5.35, 3.95)
AXES_BOX = (4.30, 3.05)
ANGLES = ((1, 0, '#2E75B6', 'solid'),
          (2, 30, '#ED7D31', (0, (4, 2.2))),
          (3, 60, '#57A84B', 'solid'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot the 0/30/60-degree CST spectra for one design.'
    )
    parser.add_argument(
        '--design-wavelength',
        type=float,
        default=610.0,
        help='Design wavelength in nm; selects 1peak/<wavelength>/cst.',
    )
    parser.add_argument(
        '--sans',
        action='store_true',
        help='Use a sans-serif face, to match PowerPoint-drawn schematics.',
    )
    parser.add_argument(
        '--legend-loc',
        default='lower right',
        help='Matplotlib legend location (default: lower right).',
    )
    parser.add_argument(
        '--font-scale',
        type=float,
        default=1.0,
        help='Multiplier on every font size (default: 1.0).',
    )
    parser.add_argument(
        '--base-dir',
        default=BASE_DIR,
        help='Directory holding <wavelength>/cst (default: this file\'s dir).',
    )
    return parser.parse_args()


def load_angles(cst_dir):
    curves = []
    for index, angle, color, style in ANGLES:
        path = os.path.join(cst_dir, f'{index:05d}.b')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing CST export: {path}')
        data = np.loadtxt(path)
        curves.append((angle, data[:, 0], data[:, 1], color, style))
    grid = curves[0][1]
    for angle, wavelengths, _, _, _ in curves[1:]:
        if not np.allclose(wavelengths, grid, rtol=0.0, atol=5.0e-4):
            raise ValueError(
                f'The {angle}-degree export uses a different wavelength grid.'
            )
    return curves


def main():
    args = parse_args()
    if args.sans:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [
            'Aptos', 'Calibri', 'Carlito', 'Arial', 'Liberation Sans',
            'DejaVu Sans',
        ]
    scale = args.font_scale
    if scale != 1.0:
        for key in ('font.size', 'axes.labelsize', 'xtick.labelsize',
                    'ytick.labelsize', 'legend.fontsize'):
            matplotlib.rcParams[key] = matplotlib.rcParams[key] * scale
    # Strokes have to grow with the text, otherwise the curves and the frame
    # look spindly once the panel is shrunk into a composite figure.
    curve_width = 1.35 * scale
    frame_width = 0.8 * scale
    tick_width = 0.7 * scale
    tick_length = 3.0 * scale
    case = f'{args.design_wavelength:g}'
    cst_dir = os.path.join(os.path.abspath(args.base_dir), case, 'cst')
    curves = load_angles(cst_dir)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    box_w, box_h = AXES_BOX
    left, bottom = 0.90 / FIGSIZE[0], 0.78 / FIGSIZE[1]
    fig.subplots_adjust(
        left=left,
        right=left + box_w / FIGSIZE[0],
        bottom=bottom,
        top=bottom + box_h / FIGSIZE[1],
    )
    for angle, wavelengths, transmittance, color, style in curves:
        ax.plot(
            wavelengths,
            transmittance,
            color=color,
            linewidth=curve_width,
            linestyle=style,
            solid_capstyle='round',
            dash_capstyle='round',
            label=f'{angle}$^\\circ$',
        )
        index = int(np.argmin(transmittance))
        print(
            f'  {angle:>2} deg: dip {wavelengths[index]:.2f} nm, '
            f'T = {transmittance[index]:.4f}'
        )

    ax.set_xlim(380, 800)
    ax.set_ylim(-0.025, 1.025)
    ax.xaxis.set_major_locator(FixedLocator([400, 500, 600, 700, 800]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(
        axis='both',
        which='major',
        length=tick_length,
        width=tick_width,
        direction='in',
        top=True,
        right=True,
        pad=2,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(frame_width)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmittance')
    ax.legend(
        loc=args.legend_loc,
        frameon=False,
        handlelength=1.9,
        handletextpad=0.55,
        labelspacing=0.30,
        borderaxespad=0.7,
    )

    output_dir = os.path.join(os.path.abspath(args.base_dir), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, f'cst_angle_spectra_{case}' + ('_sans' if args.sans else ''))
    for extension in ('tiff', 'pdf', 'svg', 'png'):
        fig.savefig(
            f'{stem}.{extension}',
            dpi=FIG_DPI,
            bbox_inches='tight',
            pad_inches=0.04,
            facecolor='white',
        )
        print(f'Saved: {stem}.{extension}')
    plt.close(fig)


if __name__ == '__main__':
    main()
