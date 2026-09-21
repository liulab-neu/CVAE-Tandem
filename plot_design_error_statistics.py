#!/usr/bin/env python3
"""Statistics over the whole reachable target range, not hand-picked cases.

Every target the structure family can actually reach is designed twice: once
by feeding the ideal Lorentzian target straight to the inverse network (ITS)
and once through a CVAE-generated realisable target (CTS).  The empirical CDFs
and the per-target scatter together show that the CVAE-assisted route wins
across the band rather than at a few favourable wavelengths.

Inputs are the sweep dumps produced by sweep_single.py / sweep_pairs.py.
"""

import argparse
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tandem-mpl-cache')

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BLUE = '#0072B2'
ORANGE = '#D55E00'
DARK = '#202020'
GRAY = '#5A5A5A'
# Kept compact on purpose: the type is set large for print, so a bigger canvas
# would only shrink it again once the figure is scaled to the column width.
FIGSIZE = (8.6, 7.4)
FIG_DPI = 600


matplotlib.rcParams.update({
    'font.family': 'serif',
    # Times New Roman is not installed here; Nimbus Roman is the
    # Times-metric clone that is, so it has to precede DejaVu or the
    # figure renders in a visibly different face from its neighbours.
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'DejaVu Serif', 'serif'],
    'font.size': 16.0,
    'axes.labelsize': 17.0,
    'axes.titlesize': 17.0,
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


def ecdf(values):
    """Step coordinates for the empirical CDF, starting at zero."""
    ordered = np.sort(np.asarray(values, dtype=float))
    fraction = np.arange(1, ordered.size + 1) / ordered.size
    return np.concatenate([[0.0], ordered]), np.concatenate([[0.0], fraction])


def style_axes(axes):
    for spine in axes.spines.values():
        spine.set_linewidth(0.8)
    axes.tick_params(width=0.8, length=3.4, direction='out')


def draw_ecdf(axes, direct, assisted, xlabel, xmax):
    for values, colour, dashes in ((direct, ORANGE, (0, (4, 2.2))),
                                   (assisted, BLUE, None)):
        x, y = ecdf(values)
        axes.step(x, y, where='post', color=colour, linewidth=1.55,
                  linestyle=dashes if dashes else '-')
    axes.set_xlim(0.0, xmax)
    axes.set_ylim(0.0, 1.02)
    axes.set_xlabel(xlabel)
    axes.set_ylabel('Cumulative fraction')
    axes.grid(True, color='#DDDDDD', linewidth=0.55)
    axes.set_axisbelow(True)
    style_axes(axes)


def draw_scatter(axes, positions, direct, assisted, xlabel, ylabel):
    # A vertical stem per target makes the paired improvement legible even
    # where the two markers nearly overlap.
    for x, lo, hi in zip(positions, assisted, direct):
        axes.plot([x, x], [lo, hi], color=GRAY, linewidth=0.6, zorder=1)
    axes.plot(positions, direct, linestyle='none', marker='o', markersize=4.2,
              markerfacecolor='none', markeredgecolor=ORANGE,
              markeredgewidth=1.1, zorder=3)
    axes.plot(positions, assisted, linestyle='none', marker='D', markersize=3.6,
              color=BLUE, zorder=4)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_ylim(bottom=0.0)
    axes.grid(True, color='#DDDDDD', linewidth=0.55)
    axes.set_axisbelow(True)
    style_axes(axes)


def annotate(axes, direct, assisted):
    # Coloured to match the curves; a shared box would not say which is which.
    for value, colour, height in ((np.median(direct), ORANGE, 0.175),
                                  (np.median(assisted), BLUE, 0.065)):
        axes.text(0.955, height, f'median {value:.2f} nm', ha='right',
                  va='bottom', transform=axes.transAxes, color=colour,
                  fontsize=16.0,
                  bbox=dict(boxstyle='round,pad=0.26', facecolor='white',
                            edgecolor='none', alpha=0.85))


def draw_paired(axes, direct, assisted, label, limit):
    """Each target once: below the diagonal the CVAE-assisted route wins.

    With 55 pairs the stem layout used for the single-notch case turns into a
    thicket, and the count of wins is what the panel has to convey.
    """
    axes.plot([0, limit], [0, limit], color=DARK, linewidth=0.9,
              linestyle=(0, (3.4, 2.4)), zorder=2)
    crowded = len(direct) > 80
    axes.plot(direct, assisted, linestyle='none', marker='o',
              markersize=3.0 if crowded else 4.4, markerfacecolor=BLUE,
              markeredgecolor='white', markeredgewidth=0.35 if crowded else 0.5,
              alpha=0.65 if crowded else 0.85, zorder=3)
    wins = int(np.sum(np.asarray(assisted) < np.asarray(direct)))
    axes.text(0.955, 0.075, f'{wins}/{len(direct)} below', ha='right',
              va='bottom', transform=axes.transAxes, color=DARK, fontsize=16.0,
              bbox=dict(boxstyle='round,pad=0.32', facecolor='white',
                        edgecolor='#BBBBBB', linewidth=0.6))
    axes.set_xlim(0.0, limit)
    axes.set_ylim(0.0, limit)
    axes.set_aspect('equal', adjustable='box')
    # The quantity is already named in the companion ECDF panel, so short
    # labels here keep the equal-aspect box from crowding the row above.
    axes.set_xlabel(f'Direct {label}')
    axes.set_ylabel(f'CVAE-assisted {label}')
    axes.xaxis.labelpad = 2.0
    axes.grid(True, color='#DDDDDD', linewidth=0.55)
    axes.set_axisbelow(True)
    style_axes(axes)


def panel_label(axes, letter):
    # A left-aligned title never collides with the y-axis tick labels.
    axes.set_title(f'({letter})', loc='left', pad=6.0, color=DARK)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', required=True)
    parser.add_argument('--dual', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    single = json.load(open(args.single))
    dual = [row for row in json.load(open(args.dual))
            if row['its'] and row['cts']]

    centers = np.array([row['center'] for row in single], dtype=float)
    single_direct = np.array([row['its']['error'] for row in single])
    single_assisted = np.array([row['cts']['error'] for row in single])

    dual_direct = np.array([row['its']['error_sum'] for row in dual])
    dual_assisted = np.array([row['cts']['error_sum'] for row in dual])

    figure, grid = plt.subplots(2, 2, figsize=FIGSIZE)
    figure.subplots_adjust(left=0.125, right=0.985, bottom=0.150, top=0.935,
                           wspace=0.38, hspace=0.60)

    draw_ecdf(grid[0, 0], single_direct, single_assisted,
              r'$|\Delta\lambda|$ (nm)', 8.0)
    annotate(grid[0, 0], single_direct, single_assisted)
    draw_scatter(grid[0, 1], centers, single_direct, single_assisted,
                 'Target wavelength (nm)', r'$|\Delta\lambda|$ (nm)')

    draw_ecdf(grid[1, 0], dual_direct, dual_assisted,
              r'$\Sigma|\Delta\lambda|$ (nm)', 20.0)
    annotate(grid[1, 0], dual_direct, dual_assisted)
    draw_paired(grid[1, 1], dual_direct, dual_assisted, '(nm)', 20.0)

    for axes, letter in zip(grid.ravel(), 'abcd'):
        panel_label(axes, letter)

    handles = [
        Line2D([], [], color=ORANGE, linewidth=1.55, linestyle=(0, (4, 2.2)),
               marker='o', markersize=4.2, markerfacecolor='none',
               markeredgecolor=ORANGE, markeredgewidth=1.1),
        Line2D([], [], color=BLUE, linewidth=1.55, marker='D', markersize=3.6),
    ]
    figure.legend(handles, ['Direct inverse design',
                            'CVAE-assisted inverse design'],
                  loc='lower center', ncol=2, frameon=False,
                  bbox_to_anchor=(0.5, 0.005), handlelength=2.9,
                  columnspacing=2.1)

    stem = os.path.splitext(args.output)[0]
    # Same set the other manuscript figures are shipped in.
    for suffix, options in (('.tiff', {'dpi': FIG_DPI,
                                       'pil_kwargs': {'compression': 'tiff_lzw'}}),
                            ('.pdf', {}), ('.svg', {}),
                            ('.png', {'dpi': FIG_DPI})):
        figure.savefig(stem + suffix, bbox_inches='tight', pad_inches=0.04,
                       facecolor='white', **options)
    print(f'单峰 {len(single)} 个目标, 双峰 {len(dual)} 对')
    print(f'Saved: {stem}.{{tiff,pdf,svg,png}}')


if __name__ == '__main__':
    main()
