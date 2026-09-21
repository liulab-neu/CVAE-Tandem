#!/usr/bin/env python3
"""Ideal target spectrum (ITS) versus the CVAE-generated target spectrum (CTS).

Replaces the inline block at the end of test_cvae.py, which drew the same
comparison but sampled the latent vector without a seed and evaluated the
ideal Lorentzian on np.linspace(380, 800, 1001) while plotting it against the
CST wavelength grid.  Both are fixed here.
"""

import argparse
import os

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, BASE_DIR)

import CVAE
from data_loader import load_data

# Matches plot_spectral_comparison_3x3.py and plot_cst_angle_spectra.py.
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

BLUE = '#2E75B6'
ORANGE = '#ED7D31'
FIG_DPI = 600
FIGSIZE = (5.35, 3.95)
AXES_BOX = (4.30, 3.05)
ENCODER_SIZE = [1001, 1024, 2048, 1024, 512, 256, 128, 20]
DECODER_SIZE = [20, 128, 256, 512, 1024, 2048, 1024, 1001]
DATA_ROOT = '/home/yuxiao/Yuxiao Li/batch3'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot the ideal target spectrum against the CVAE target.'
    )
    parser.add_argument('--center', type=float, default=650.0,
                        help='Target notch wavelength in nm (default: 650).')
    parser.add_argument('--gamma', type=float, default=15.0,
                        help='Lorentzian half-width in nm (default: 15).')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Seed for the CVAE latent vector (default: 42).')
    parser.add_argument('--font-scale', type=float, default=1.0,
                        help='Multiplier on font sizes and stroke widths.')
    parser.add_argument('--sans', action='store_true',
                        help='Use a sans-serif face, matching PowerPoint art.')
    parser.add_argument('--legend-loc', default='lower right',
                        help='Legend location (default: lower right).')
    parser.add_argument('--output-dir',
                        default=os.path.join(BASE_DIR, 'figures'))
    return parser.parse_args()


def ideal_target(center, wavelengths, gamma):
    dip = 1.0 / (1.0 + ((wavelengths - center) / gamma) ** 2)
    return 1.0 - dip


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
    # Strokes grow with the text so the curves do not look spindly once the
    # panel is shrunk into a composite figure.
    curve_width = 1.35 * scale
    frame_width = 0.8 * scale
    tick_width = 0.7 * scale
    tick_length = 3.0 * scale
    _, data_b_list = load_data(DATA_ROOT)
    wavelengths = np.array(data_b_list[0])[:, 0]

    model = CVAE.CVAE(
        encoder_size=ENCODER_SIZE, decoder_size=DECODER_SIZE, c_dim=1
    )
    model.load_state_dict(
        torch.load(os.path.join(BASE_DIR, 'model', 'cvae_best_mean.pth'),
                   map_location='cpu')
    )
    model.eval()

    torch.manual_seed(args.random_seed)
    condition = torch.tensor(
        [[(args.center - wavelengths[0])
          / (wavelengths[-1] - wavelengths[0])]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        latent = torch.randn(1, ENCODER_SIZE[-1])
        cvae_target = model.decoder(latent, condition).squeeze(0).numpy()

    its = ideal_target(args.center, wavelengths, args.gamma)

    for name, curve in (('ITS', its), ('CTS', cvae_target)):
        index = int(np.argmin(curve))
        print(f'  {name}: dip {wavelengths[index]:.2f} nm, '
              f'T = {curve[index]:.4f}')

    fig, ax = plt.subplots(figsize=FIGSIZE)
    left, bottom = 0.90 / FIGSIZE[0], 0.78 / FIGSIZE[1]
    fig.subplots_adjust(
        left=left,
        right=left + AXES_BOX[0] / FIGSIZE[0],
        bottom=bottom,
        top=bottom + AXES_BOX[1] / FIGSIZE[1],
    )
    ax.plot(wavelengths, its, color=ORANGE, linewidth=curve_width,
            linestyle=(0, (4, 2.2)), dash_capstyle='round', label='ITS')
    ax.plot(wavelengths, cvae_target, color=BLUE, linewidth=curve_width,
            solid_capstyle='round', label='CTS')

    ax.set_xlim(380, 800)
    ax.set_ylim(-0.025, 1.025)
    ax.xaxis.set_major_locator(FixedLocator([400, 500, 600, 700, 800]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis='both', which='major', length=tick_length,
                   width=tick_width, direction='in', top=True, right=True,
                   pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(frame_width)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmittance')
    ax.legend(loc=args.legend_loc, frameon=False, handlelength=1.9,
              handletextpad=0.55, labelspacing=0.30, borderaxespad=0.7)

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir,
                        f'ideal_vs_cvae_target_{args.center:g}'
                        + ('_sans' if args.sans else ''))
    for extension in ('tiff', 'pdf', 'svg', 'png'):
        fig.savefig(f'{stem}.{extension}', dpi=FIG_DPI, bbox_inches='tight',
                    pad_inches=0.04, facecolor='white')
        print(f'Saved: {stem}.{extension}')
    plt.close(fig)


if __name__ == '__main__':
    main()
