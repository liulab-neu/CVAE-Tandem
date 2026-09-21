#!/usr/bin/env python3
"""Error over the joint grid of the two physics-loss weights.

The manuscript adds the same two terms to three networks, so all three are
shown: forward, tandem inverse and CVAE, for both cases.  Varying the weights
jointly rather than one at a time keeps an interaction between them visible.

--metric picks what the cells report.  The dip position is what the terms are
meant to fix, but a weight that buys dip accuracy by distorting the rest of the
spectrum is not a good trade, so the depth error and the whole-spectrum MSE are
drawn from the same dumps on the same grid.

Every tandem cell is trained on the same forward checkpoint, so its row is the
effect of the loss on the inverse network alone.

Input is the grid dump written by each ablation evaluator.
"""

import argparse
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tandem-mpl-cache')

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


DARK = '#202020'
FIGSIZE = (7.4, 3.4)
FIG_DPI = 600

# What the cells can report, with the scaling and the number format each one
# needs: the MSE runs from 3e-5 to 7e-2, which is unreadable printed raw.
# `keys` lists the candidate fields in preference order, because the two
# evaluators gave the same name to different statistics: in the forward dump
# `dip_nm` and `depth` are means, while in the tandem and CVAE dumps they are
# medians and the means sit under `dip_mean` and `depth_mean`.  Preferring the
# explicit mean fields makes every panel of a row the same statistic.
METRICS = {
    'dip': {'label': r'Dip $|\Delta\lambda|$ (nm)', 'scale': 1.0,
            'format': '{:.2f}', 'keys': ('dip_mean', 'dip_nm')},
    # Transmittance is a fraction, so its error in points of a percent is both
    # the natural unit and the one that prints in three characters.
    'depth': {'label': r'Dip depth $|\Delta T|$ (%)', 'scale': 100.0,
              'format': '{:.2f}', 'keys': ('depth_mean', 'depth')},
    # 1e4, not 1e3: three significant digits of the smallest cell then fit the
    # same five characters as the largest, and the cells stop touching.
    'mse': {'label': r'Spectral MSE ($\times 10^{-4}$)', 'scale': 1e4,
            'format': '{:.3g}', 'keys': ('mse',)},
}

matplotlib.rcParams.update({
    'font.family': 'serif',
    # Times New Roman is not installed here; without the clone matplotlib
    # silently falls back to DejaVu Serif.
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif', 'serif'],
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


def matrix(rows, key='value'):
    """Cells are keyed on the two weights; the value key differs per dump."""
    pws = sorted({r['w_pw'] for r in rows})
    pis = sorted({r['w_pi'] for r in rows})
    grid = np.full((len(pws), len(pis)), np.nan)
    for i, pw in enumerate(pws):
        for j, pi in enumerate(pis):
            values = [r[key] for r in rows
                      if r['w_pw'] == pw and r['w_pi'] == pi]
            if values:
                grid[i, j] = np.mean(values)
    return pws, pis, grid


def draw(axes, rows, title, norm, number):
    pws, pis, grid = matrix(rows)
    image = axes.imshow(grid, cmap='YlOrRd', norm=norm, aspect='auto')
    axes.set_xticks(range(len(pis)))
    axes.set_xticklabels([f'{p:g}' for p in pis])
    axes.set_yticks(range(len(pws)))
    axes.set_yticklabels([f'{p:g}' for p in pws])
    # The symbol first, the name after: the text refers to these weights only
    # as gamma and delta, so an axis labelled w_peak-I alone leaves the reader
    # to guess which of the two the figure is varying.
    axes.set_xlabel(r'$\delta$  ($w_{\mathrm{peak}\text{-}I}$)')
    axes.set_ylabel(r'$\gamma$  ($w_{\mathrm{peak}\text{-}\lambda}$)')
    if title:
        axes.set_title(title, pad=6.0, color=DARK)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]):
                continue
            # White on the dark end of the map, black on the light end.
            shade = 'white' if norm(grid[i, j]) > 0.55 else DARK
            axes.text(j, i, number.format(grid[i, j]), ha='center',
                      va='center', color=shade, fontsize=16.0)
    for spine in axes.spines.values():
        spine.set_linewidth(0.8)
    axes.tick_params(width=0.8, length=3.4)
    return image


def load(path, keys, scale):
    rows = json.load(open(path))
    key = next((k for k in keys if k in rows[0]), None)
    if key is None:
        raise KeyError(f'{path}: none of {keys} present')
    # A dump that reports `dip_mean` came from the stage evaluator, which stores
    # the depth as a median unless it has been re-scored.  Falling back to
    # `depth` there would put a median next to the forward panel's mean, which
    # is the mismatch this preference list exists to remove -- so say so instead
    # of drawing it.
    if key == 'depth' and 'dip_mean' in rows[0]:
        raise KeyError(f'{path}: stage dump without depth_mean; re-score it '
                       f'before drawing the depth row')
    return [{'w_pw': r['w_pw'], 'w_pi': r['w_pi'], 'value': r[key] * scale}
            for r in rows]


def save(figure, stem):
    for suffix, options in (('.tiff', {'dpi': FIG_DPI,
                                       'pil_kwargs': {'compression': 'tiff_lzw'}}),
                            ('.pdf', {}), ('.svg', {}),
                            ('.png', {'dpi': FIG_DPI})):
        figure.savefig(stem + suffix, bbox_inches='tight', pad_inches=0.04,
                       facecolor='white', **options)
    print(f'Saved: {stem}.{{tiff,pdf,svg,png}}')


def by_metric(entries, metric, stem):
    """One metric per figure: rows are the cases, columns the networks."""
    panels = [(load(path, metric['keys'], metric['scale']), label)
              for label, path in entries]
    everything = [r['value'] for rows, _ in panels for r in rows]
    # One shared logarithmic scale: the values span two orders of magnitude and
    # a linear map would flatten every cell except the worst row.
    norm = LogNorm(vmin=min(everything), vmax=max(everything))

    columns = 3
    rows_count = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows_count, columns,
                                figsize=(3.5 * columns, 3.2 * rows_count))
    axes = np.atleast_2d(axes)
    figure.subplots_adjust(left=0.075, right=0.885, bottom=0.10, top=0.925,
                           wspace=0.34, hspace=0.42)
    flat = axes.ravel()
    for panel, (rows, title) in zip(flat, panels):
        image = draw(panel, rows, title, norm, metric['format'])
    for panel in flat[len(panels):]:
        panel.axis('off')

    bar = figure.add_axes([0.905, 0.10, 0.018, 0.825])
    figure.colorbar(image, cax=bar).set_label(metric['label'])
    save(figure, stem)


def by_case(entries, stem):
    """One case per figure: rows are the metrics, columns the networks.

    The three metrics carry different units, so they cannot share a colour bar
    anyway -- making them the rows gives each its own scale and puts all three
    verdicts on one network directly above one another, which is the comparison
    the figure exists to support.  Both figures are built from the same call so
    a row can share its scale across the two cases.
    """
    # Labels arrive as "network, case"; the case decides the file, the network
    # the column.
    networks, cases, cells = [], [], {}
    for label, path in entries:
        network, case = (part.strip() for part in label.split(','))
        networks.append(network) if network not in networks else None
        cases.append(case) if case not in cases else None
        cells[(network, case)] = path

    loaded, norms = {}, {}
    for name, metric in METRICS.items():
        for (network, case), path in cells.items():
            loaded[(name, network, case)] = load(
                path, metric['keys'], metric['scale'])
        values = [r['value'] for (n, _, _), rows in loaded.items()
                  if n == name for r in rows]
        norms[name] = LogNorm(vmin=min(values), vmax=max(values))

    for case in cases:
        figure, axes = plt.subplots(len(METRICS), len(networks),
                                    figsize=(11.0, 2.95 * len(METRICS)))
        axes = np.atleast_2d(axes)
        figure.subplots_adjust(left=0.075, right=0.855, bottom=0.075,
                               top=0.905, wspace=0.30, hspace=0.26)
        for row, (name, metric) in enumerate(METRICS.items()):
            for column, network in enumerate(networks):
                image = draw(axes[row, column],
                             loaded[(name, network, case)],
                             network if row == 0 else '', norms[name],
                             metric['format'])
                # Every panel carries the same two axes, so only the outer ones
                # are named; the tick values stay on all nine.
                if row < len(METRICS) - 1:
                    axes[row, column].set_xlabel('')
                if column > 0:
                    axes[row, column].set_ylabel('')
            # A bar per row, spanning exactly that row's panels.
            position = axes[row, 0].get_position()
            bar = figure.add_axes([0.875, position.y0, 0.016, position.height])
            figure.colorbar(image, cax=bar).set_label(metric['label'])
        figure.suptitle(f'{case.capitalize()}-notch system', y=0.965,
                        fontweight='bold', color=DARK)
        save(figure, f'{stem}_{case}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps', nargs='+', required=True,
                        help='six "label=path" entries, row-major, where the '
                             'label is "network, case"')
    parser.add_argument('--layout', choices=('case', 'metric'), default='case',
                        help='case: one figure per case, metrics as rows; '
                             'metric: one figure per metric, cases as rows')
    parser.add_argument('--metric', choices=tuple(METRICS), default='dip',
                        help='which metric --layout metric draws')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    entries = [tuple(entry.split('=')) for entry in args.dumps]
    stem = os.path.splitext(args.output)[0]
    if args.layout == 'metric':
        by_metric(entries, METRICS[args.metric], stem)
    else:
        by_case(entries, stem)


if __name__ == '__main__':
    main()
