#!/usr/bin/env python3
"""Create a publication-style 2x3 training-loss figure."""

import os
import re

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BLUE = '#0072B2'
ORANGE = '#D55E00'
DARK = '#202020'
SPINE = '#4B4B4B'
FIGSIZE = (11.0, 6.6)
FIG_DPI = 600


matplotlib.rcParams.update({
    'font.family': 'serif',
    # Times New Roman is not installed here; without the clone matplotlib falls
    # back to DejaVu Serif and the figure stops matching the others.
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
    'svg.fonttype': 'path',
})


PIPE_PATTERN = re.compile(
    r'Epoch\s+(\d+)\s*\|\s*'
    r'Train Loss:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\|\s*'
    r'Val Loss:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
    re.IGNORECASE,
)
COLON_PATTERN = re.compile(
    r'Epoch:\s*(\d+)\s*,\s*'
    r'Train loss:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*'
    r'Val loss:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
    re.IGNORECASE,
)


def parse_loss_log(path):
    epochs = []
    train_losses = []
    validation_losses = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as log_file:
        for line in log_file:
            match = PIPE_PATTERN.search(line) or COLON_PATTERN.search(line)
            if match is None:
                continue
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            validation_losses.append(float(match.group(3)))

    if not epochs:
        raise ValueError(f'No training-loss records found in {path}')

    return (
        np.asarray(epochs, dtype=int),
        np.asarray(train_losses, dtype=float),
        np.asarray(validation_losses, dtype=float),
    )


def style_axis(ax):
    # Logarithmic: the first few epochs are one to two orders of magnitude
    # above the converged level, so a linear axis squashes the entire
    # convergence into a flat line at the bottom of the panel.
    ax.set_yscale('log')
    ax.grid(
        axis='y',
        color='#B8B8B8',
        linestyle='-',
        linewidth=0.45,
        alpha=0.18,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        top=True,
        right=True,
        length=3.2,
        width=0.7,
        pad=3.0,
    )
    ax.tick_params(
        axis='both',
        which='minor',
        direction='in',
        top=True,
        right=True,
        length=1.8,
        width=0.5,
    )
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)


def save_figure(fig, output_dir, output_stem):
    os.makedirs(output_dir, exist_ok=True)
    tiff_path = os.path.join(output_dir, output_stem + '.tiff')
    pdf_path = os.path.join(output_dir, output_stem + '.pdf')
    svg_path = os.path.join(output_dir, output_stem + '.svg')
    # The PNG was missing from this list, so the copy on disk was whatever
    # somebody last exported by hand rather than the current figure.
    png_path = os.path.join(output_dir, output_stem + '.png')

    fig.savefig(
        tiff_path,
        dpi=FIG_DPI,
        bbox_inches='tight',
        pad_inches=0.04,
        facecolor='white',
        pil_kwargs={'compression': 'tiff_lzw'},
    )
    fig.savefig(
        pdf_path,
        bbox_inches='tight',
        pad_inches=0.04,
        facecolor='white',
    )
    fig.savefig(
        svg_path,
        bbox_inches='tight',
        pad_inches=0.04,
        facecolor='white',
    )
    fig.savefig(
        png_path,
        dpi=FIG_DPI,
        bbox_inches='tight',
        pad_inches=0.04,
        facecolor='white',
    )
    return tiff_path, pdf_path, svg_path, png_path


def create_loss_figure():
    titles = ('Conditional VAE', 'Forward network', 'Tandem inverse network')
    row_configs = (
        (
            '1peak',
            ('CVAE_loss.txt', 'DNN_FNN-t.txt', 'DNN_tandem-t.txt'),
        ),
        (
            '2peak',
            ('CVAE_2p-t.txt', 'DNN_FNN-t.txt', 'DNN_tandem-t.txt'),
        ),
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=FIGSIZE,
        sharex=False,
        sharey=False,
    )

    for row, (peak_kind, log_names) in enumerate(row_configs):
        # The deployed checkpoints were each taken from the best cell of the
        # weight sweep, and the three networks did not win at the same cell, so
        # their training runs are three different ones.  The logs sitting
        # directly under 1peak/ and 2peak/ belong to a separate run and do not
        # match the models the paper reports; logs_selected/ holds the runs
        # that produced the checkpoints in model/.
        base_dir = os.path.join(ROOT_DIR, peak_kind, 'logs_selected')
        for column, (title, log_name) in enumerate(zip(titles, log_names)):
            ax = axes[row, column]
            log_path = os.path.join(base_dir, log_name)
            if not os.path.isfile(log_path):
                raise FileNotFoundError(
                    f'Required loss log not found: {log_path}'
                )

            epochs, train_loss, validation_loss = parse_loss_log(log_path)
            ax.plot(
                epochs,
                train_loss,
                color=BLUE,
                linewidth=1.35,
                solid_capstyle='round',
                label='Training',
                zorder=3,
            )
            ax.plot(
                epochs,
                validation_loss,
                color=ORANGE,
                linewidth=1.25,
                solid_capstyle='round',
                label='Validation',
                zorder=2,
            )
            if row == 0:
                ax.set_title(title, pad=8, fontweight='bold', color=DARK)
            ax.set_xlim(0, int(epochs.max()))
            style_axis(ax)

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.13,
        top=0.845,
        wspace=0.30,
        hspace=0.34,
    )
    fig.supxlabel(
        'Epoch',
        x=0.53,
        y=0.025,
        fontsize=16.0,
        color=DARK,
    )
    fig.supylabel(
        'Loss',
        x=0.052,
        y=0.48,
        fontsize=16.0,
        color=DARK,
    )
    fig.text(
        0.014,
        0.675,
        'Single peak',
        ha='center',
        va='center',
        rotation=90,
        fontsize=16.0,
        fontweight='bold',
        color=DARK,
    )
    fig.text(
        0.014,
        0.285,
        'Dual peak',
        ha='center',
        va='center',
        rotation=90,
        fontsize=16.0,
        fontweight='bold',
        color=DARK,
    )

    train_handle = Line2D([], [], color=BLUE, linewidth=1.35)
    validation_handle = Line2D([], [], color=ORANGE, linewidth=1.25)
    fig.legend(
        (train_handle, validation_handle),
        ('Training', 'Validation'),
        loc='upper center',
        bbox_to_anchor=(0.53, 0.965),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.45,
        columnspacing=1.6,
        borderaxespad=0,
    )

    output_dir = os.path.join(ROOT_DIR, 'figures')
    paths = save_figure(fig, output_dir, 'training_loss_curves_2x3')
    plt.close(fig)
    return paths


def main():
    for path in create_loss_figure():
        print(f'Saved: {path}')


if __name__ == '__main__':
    main()
