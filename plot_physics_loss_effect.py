#!/usr/bin/env python3
"""What the physics loss changes, stage by stage.

The manuscript adds L_peak-lambda and L_peak-I to three networks -- forward,
tandem inverse and CVAE -- so the ablation has to cover all three rather than
just the forward model.  Each panel compares training with those terms against
training on the spectral MSE alone, at the weights the grid search selected
(single notch 100/1, double notch 1/1).

Dips are located with find_peaks and the same prominence gate the loss uses,
so the metric and the objective agree; for the double-notch case both
resonances are scored, since a single argmin only ever sees the deeper one.
"""

import argparse
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tandem-mpl-cache')

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.abspath(__file__))
SP = ('/tmp/claude-1002/-home-yuxiao-Yuxiao-Li-tandem/'
      '49704235-96ad-4344-bde8-d754f725ff04/scratchpad')
BLUE = '#0072B2'
ORANGE = '#D55E00'
DARK = '#202020'
FIGSIZE = (7.4, 3.3)
FIG_DPI = 600
ENCODER = [1001, 1024, 2048, 1024, 512, 256, 128, 20]
DECODER = [20, 128, 256, 512, 1024, 2048, 1024, 1001]

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


# 20 nm is part of the definition of "two notches" and is what
# dual_peak_utils.detect_two_resonances enforces; without it a broad notch can
# return two of its own samples as the pair.
MINIMUM_DIP_SEPARATION_NM = 20.0


def dips(row, dual, distance_samples=1):
    if not dual:
        return np.array([int(np.argmin(row))])
    peaks, _ = find_peaks(-row, prominence=0.01, distance=distance_samples)
    if len(peaks) < 2:
        return None
    return np.sort(peaks[np.argsort(row[peaks])[:2]])


def score(predicted, target, grid, dual):
    predicted = predicted.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    step = float(np.median(np.diff(np.asarray(grid, dtype=float))))
    distance_samples = max(1, int(round(MINIMUM_DIP_SEPARATION_NM / step)))
    positions, depths = [], []
    for a, b in zip(predicted, target):
        pa = dips(a, dual, distance_samples)
        pb = dips(b, dual, distance_samples)
        if pa is None or pb is None:
            continue
        positions.append(np.abs(grid[pa] - grid[pb]).mean())
        depths.append(np.abs(a[pa] - b[pb]).mean())
    return np.array(positions), np.array(depths)


def measure(kind):
    """Both variants of all three networks, on the held-out test tenth."""
    sys.path.insert(0, f'{ROOT}/{kind}')
    os.chdir(f'{ROOT}/{kind}')
    if kind == '1peak':
        import train_networks as trainer
        import CVAE as cvae_module
        dual, c_dim, name = False, 1, 'cvae_best_mean.pth'
    else:
        import train_networks_2p as trainer
        import CVAE_2p as cvae_module
        import dual_peak_utils as dpu
        dual, c_dim, name = True, 2, 'cvae_best.pth'
    device = trainer.DEVICE
    structures, spectra, _, _, test_idx = trainer.load_split(42)
    x = structures[test_idx].to(device)
    y = spectra[test_idx].to(device)

    weights = ((100.0, 1.0) if kind == '1peak' else (1.0, 1.0))
    variants = {'physics': (f'{ROOT}/{kind}/model', weights),
                'plain': (f'{SP}/nophys/{kind}/model', (0.0, 0.0))}
    out = {}
    for label, (directory, (pw, pi)) in variants.items():
        net = trainer.build_tandem(42, argparse.Namespace(
            w_peak_wavelength=pw, w_peak_intensity=pi))
        net.restore_FNN(f'{directory}/DNN_tandem_FNN_label.ckpt')
        net.restore_INN(f'{directory}/DNN_tandem_INN_label.ckpt')
        net.fnn.eval()
        net.inn.eval()
        grid = (dpu.load_wavelength_grid() if dual else
                net.wavelengths.detach().cpu().numpy().reshape(-1))
        with torch.no_grad():
            forward = net.fnn(x)
            _, tandem = net.forward(y, 'tandem')

        model = cvae_module.CVAE(encoder_size=ENCODER, decoder_size=DECODER,
                                 c_dim=c_dim).to(device)
        state = torch.load(f'{directory}/{name}', map_location=device)
        model.load_state_dict(state.get('model', state)
                              if isinstance(state, dict) else state)
        model.eval()
        with torch.no_grad():
            _, condition = model.lowest_wavelength(y)
            condition = torch.as_tensor(condition, dtype=torch.float32,
                                        device=device).reshape(len(y), -1)
            output = model(y, condition)
            cvae = output[0] if isinstance(output, tuple) else output

        stage = {}
        for key, prediction in (('Forward', forward), ('Tandem', tandem),
                                ('CVAE', cvae)):
            positions, depths = score(prediction, y, grid, dual)
            stage[key] = {
                'dip_median': float(np.median(positions)),
                'dip_mean': float(positions.mean()),
                'depth_median': float(np.median(depths)),
                'mse': float(torch.nn.functional.mse_loss(prediction, y)),
            }
            print(f'  {kind} {label:<8} {key:<8} 谷位中位 '
                  f'{stage[key]["dip_median"]:6.3f} nm   谷深中位 '
                  f'{stage[key]["depth_median"]:.4f}   MSE '
                  f'{stage[key]["mse"]:.3e}', flush=True)
        out[label] = stage
    sys.path.pop(0)
    return out


def draw(axes, results, title):
    stages = ['Forward', 'Tandem', 'CVAE']
    plain = [results['plain'][s]['dip_median'] for s in stages]
    physics = [results['physics'][s]['dip_median'] for s in stages]
    positions = np.arange(len(stages))
    width = 0.36
    axes.bar(positions - width / 2, plain, width, color=ORANGE,
             edgecolor='white', linewidth=0.6, label='Spectral MSE only')
    axes.bar(positions + width / 2, physics, width, color=BLUE,
             edgecolor='white', linewidth=0.6, label='+ physics terms')
    for x, (a, b) in enumerate(zip(plain, physics)):
        for offset, value in ((-width / 2, a), (width / 2, b)):
            axes.annotate(f'{value:.2f}', xy=(x + offset, value),
                          xytext=(0, 2), textcoords='offset points',
                          ha='center', va='bottom', fontsize=16.0, color=DARK)
    axes.set_xticks(positions)
    axes.set_xticklabels(stages)
    axes.set_ylabel(r'Dip $|\Delta\lambda|$ (nm)')
    axes.set_title(title, pad=6.0, color=DARK)
    axes.set_ylim(0, max(plain + physics) * 1.28)
    axes.grid(True, axis='y', color='#DDDDDD', linewidth=0.55)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_linewidth(0.8)
    axes.tick_params(width=0.8, length=3.4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results = {kind: measure(kind) for kind in ('1peak', '2peak')}
    json.dump(results, open(f'{SP}/physics_effect.json', 'w'), indent=2)

    figure, panels = plt.subplots(1, 2, figsize=FIGSIZE)
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.30, top=0.885,
                           wspace=0.28)
    draw(panels[0], results['1peak'], 'Single notch')
    draw(panels[1], results['2peak'], 'Double notch')
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
                  bbox_to_anchor=(0.5, 0.005))

    stem = os.path.splitext(os.path.join(ROOT, args.output)
                            if not os.path.isabs(args.output)
                            else args.output)[0]
    for suffix, options in (('.tiff', {'dpi': FIG_DPI,
                                       'pil_kwargs': {'compression': 'tiff_lzw'}}),
                            ('.pdf', {}), ('.svg', {}),
                            ('.png', {'dpi': FIG_DPI})):
        figure.savefig(stem + suffix, bbox_inches='tight', pad_inches=0.04,
                       facecolor='white', **options)
    print(f'\nSaved: {stem}.{{tiff,pdf,svg,png}}')


if __name__ == '__main__':
    main()
