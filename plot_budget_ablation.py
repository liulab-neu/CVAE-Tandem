#!/usr/bin/env python3
"""Is the CVAE-assisted gain the generative model or just the search?

The two routes in the main text do not get the same resources: the direct route
receives one analytic target, the CVAE-assisted route fifty candidates and a
selection rule.  This gives the direct route the same budget -- fifty analytic
targets perturbed over the width, depth and baseline the training set actually
spans -- and puts both through the same admissibility conditions and ordering.

The fifty CVAE candidates are drawn as a strip so the distribution the selection
works from is visible next to the design it keeps: the gap between the strip and
the marker is the search, and the gap between the strips of the two routes is
the generative model.
"""

import argparse
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tandem-mpl-cache')

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = os.path.dirname(os.path.abspath(__file__))
BLUE = '#0072B2'
ORANGE = '#D55E00'
GREEN = '#009E73'
DARK = '#202020'
GRAY = '#8A8A8A'
FIG_DPI = 600
FIG_WIDTH = 11.0
BASE_FONT = 16.0

# The twelve-target sweep of Fig. S4 rather than the three cases of the main
# text: three points cannot settle whether the advantage is systematic.
TARGETS = tuple(range(420, 651, 20))
# Matched to the pipeline that produced the designs of the main text
# (lorentz_vs_cvae_compare.DEFAULT_N_CANDIDATES), so the CVAE column
# reproduces the reported design rather than a differently tuned one.
N_CANDIDATES = 20
# Both systems use the budget of the case pipelines that produced the designs
# reported in the paper.
DUAL_CANDIDATES = 20
SEED = 42
# Three targets drawn in full, spread across the band, so the comparison cannot
# be read as one favourable case.
SHOWCASE = (440, 520, 620)

# Named so draw() can re-assert it: 1peak's lorentz_vs_cvae_compare runs
# its own rcParams.update at import time and drops font.size to 8, which
# silently shrank every tick label on a full (non-cached) run.
FIGURE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif', 'serif'],
    'font.size': BASE_FONT,
    'axes.labelsize': BASE_FONT + 1.0,
    'axes.titlesize': BASE_FONT + 1.0,
    'axes.labelcolor': DARK,
    'axes.edgecolor': DARK,
    'axes.linewidth': 0.9,
    'xtick.labelsize': BASE_FONT - 1.0,
    'ytick.labelsize': BASE_FONT - 1.0,
    'xtick.color': DARK,
    'ytick.color': DARK,
    'legend.fontsize': BASE_FONT - 1.0,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
}

matplotlib.rcParams.update(FIGURE_STYLE)


def lorentz(wavelengths, centre, gamma, depth, baseline):
    """Notch of adjustable width, depth and baseline; the target used in the
    main text is this with depth = baseline = 1."""
    shape = 1.0 / (1.0 + ((wavelengths - centre) / gamma) ** 2)
    return baseline - depth * shape


def compute():
    sys.path.insert(0, f'{ROOT}/1peak')
    os.chdir(f'{ROOT}/1peak')

    import torch
    import train_networks as trainer
    import lorentz_vs_cvae_compare as lvc
    from notch_quality import measure_notch, select_single_notch_candidate

    net = lvc.build_tandem_network(trainer.DEVICE)
    net.eval_mode('tandem')
    wavelengths = net.wavelengths.detach().cpu().numpy().reshape(-1)
    step = float(np.median(np.diff(wavelengths)))

    # The perturbation ranges are the ones the data reaches, not invented ones,
    # so the baseline cannot be dismissed as a straw target.
    _, spectra, train_idx, _, _ = trainer.load_split(42)
    depths, widths, baselines = [], [], []
    for row in spectra[train_idx].numpy():
        metrics = measure_notch(wavelengths, row,
                               float(wavelengths[int(np.argmin(row))]))
        if np.isfinite(metrics.fwhm_nm):
            depths.append(1.0 - metrics.min_transmittance)
            widths.append(metrics.fwhm_nm)
            baselines.append(float(np.quantile(row, 0.95)))
    ranges = np.array([
        [np.quantile(widths, 0.1) / 2, np.quantile(widths, 0.9) / 2],
        [np.quantile(depths, 0.1), np.quantile(depths, 0.9)],
        [np.quantile(baselines, 0.1), np.quantile(baselines, 0.9)]])
    print('扰动范围  gamma {:.1f}-{:.1f} nm   depth {:.3f}-{:.3f}   '
          'baseline {:.3f}-{:.3f}'.format(*ranges.ravel()), flush=True)

    rng = np.random.default_rng(SEED)

    def responses_of(targets):
        tensor = torch.as_tensor(np.asarray(targets, dtype=np.float32)
                                 ).reshape(len(targets), -1).to(trainer.DEVICE)
        with torch.no_grad():
            _, out = net.test(tensor, tensor, 'tandem')
        return np.asarray(out, dtype=float)

    def errors_of(responses, centre):
        return np.array([abs(wavelengths[int(r.argmin())] - centre)
                         for r in responses])

    def pick(targets, responses, reference, centre):
        selection = select_single_notch_candidate(
            wavelengths, responses, float(centre),
            wavelength_tolerance_nm=2.0,
            tie_breaker=np.mean((responses - reference[None, :]) ** 2, axis=1))
        return int(selection.selected_index)

    cvae_all, cvae_pick, its_all, its_pick, analytic = [], [], [], [], []
    showcase = {}
    for centre in TARGETS:
        index = int(np.argmin(np.abs(wavelengths - centre)))
        reference = lorentz(wavelengths, centre, 15.0, 1.0, 1.0)

        generated = np.asarray(lvc.cvae_generate_spectrum(
            centre, n_samples=N_CANDIDATES,
            condition_value=index / (len(wavelengths) - 1),
            random_seed=SEED).detach().cpu())
        out = responses_of(generated)
        errors = errors_of(out, centre)
        chosen = pick(generated, out, reference, centre)
        cvae_all.append(errors)
        cvae_pick.append(errors[chosen])
        if centre in SHOWCASE:
            showcase[f'cvae_targets_{centre}'] = generated
            showcase[f'cvae_errors_{centre}'] = errors
            showcase[f'cvae_index_{centre}'] = chosen

        perturbed = np.stack([
            lorentz(wavelengths, centre, *(rng.uniform(lo, hi)
                                           for lo, hi in ranges))
            for _ in range(N_CANDIDATES)])
        out = responses_of(perturbed)
        errors = errors_of(out, centre)
        chosen = pick(perturbed, out, reference, centre)
        its_all.append(errors)
        its_pick.append(errors[chosen])
        if centre in SHOWCASE:
            showcase[f'its_targets_{centre}'] = perturbed
            showcase[f'its_errors_{centre}'] = errors
            showcase[f'its_index_{centre}'] = chosen
            showcase[f'reference_{centre}'] = reference
            showcase['wavelengths'] = wavelengths

        analytic.append(errors_of(responses_of(reference[None, :]), centre)[0])
        print(f'  {centre} nm  CVAE 未筛选 {cvae_all[-1].mean():5.2f} '
              f'选中 {cvae_pick[-1]:5.2f} | 扰动ITS 未筛选 '
              f'{its_all[-1].mean():5.2f} 选中 {its_pick[-1]:5.2f} | '
              f'原基线 {analytic[-1]:5.2f}', flush=True)

    return dict(targets=np.array(TARGETS, dtype=float),
                cvae_all=np.stack(cvae_all), cvae_pick=np.array(cvae_pick),
                its_all=np.stack(its_all), its_pick=np.array(its_pick),
                analytic=np.array(analytic), ranges=ranges, **showcase)


def compute_dual():
    """The same ablation on the double-notch system.

    The perturbed target is the product of two notches, each given its own
    width and depth from the training distribution, so the direct route is
    offered the same kind of variation the CVAE samples rather than a single
    rigid pair.
    """
    sys.path.insert(0, f'{ROOT}/2peak')
    os.chdir(f'{ROOT}/2peak')

    import train_networks_2p as trainer
    import dual_peak_utils as dpu

    wavelengths = dpu.load_wavelength_grid()
    tandem = dpu.build_tandem_runtime(trainer.DEVICE)
    cvae = dpu.build_cvae(trainer.DEVICE)
    pairs = dpu.supplement_pair_grid()

    _, spectra, train_idx, _, _ = trainer.load_split(42)
    widths, depths = [], []
    for row in spectra[train_idx].numpy()[:2000]:
        detection = dpu.detect_two_resonances(row, wavelengths)
        if not detection.valid:
            continue
        quality = dpu.resonance_quality_metrics(row, wavelengths, detection)
        if quality.valid and np.all(np.isfinite(quality.fwhm_nm)):
            widths.extend(quality.fwhm_nm.tolist())
            depths.extend((1.0 - detection.values).tolist())
    gamma_range = (np.quantile(widths, 0.1) / 2, np.quantile(widths, 0.9) / 2)
    depth_range = (np.quantile(depths, 0.1), np.quantile(depths, 0.9))
    print('双峰扰动范围  gamma {:.1f}-{:.1f} nm   depth {:.3f}-{:.3f}'.format(
        *gamma_range, *depth_range), flush=True)

    def two_dips(row):
        # dual_peak_utils' own detector rather than a bare find_peaks call: it
        # carries the 20 nm separation that keeps two samples of one broad
        # notch from being returned as the pair.
        detection = dpu.detect_two_resonances(
            np.asarray(row, dtype=float).reshape(-1), wavelengths,
            prominence=0.01, minimum_distance_nm=20.0)
        if not detection.valid:
            return None
        return np.asarray(detection.indices, dtype=int)

    def error(row, centres):
        found = two_dips(row)
        if found is None:
            return float('nan')
        return float(np.abs(wavelengths[found] - np.array(centres)).sum())

    rng = np.random.default_rng(SEED)
    analytic, its_pick, cvae_pick = [], [], []
    for low, high in pairs:
        ideal = dpu.bounded_double_lorentzian((low, high), wavelengths, 15.0)
        _, responses = tandem.inverse_forward(ideal)
        analytic.append(error(np.asarray(responses[0], dtype=float),
                              (low, high)))

        perturbed = []
        for _ in range(DUAL_CANDIDATES):
            spectrum = np.ones_like(wavelengths)
            for centre in (low, high):
                gamma = rng.uniform(*gamma_range)
                depth = rng.uniform(*depth_range)
                shape = 1.0 / (1.0 + ((wavelengths - centre) / gamma) ** 2)
                spectrum = spectrum * (1.0 - depth * shape)
            perturbed.append(spectrum)
        for label, override, store in (('its', np.stack(perturbed), its_pick),
                                       ('cvae', None, cvae_pick)):
            dpu.seed_everything(SEED)
            try:
                selected = dpu.select_cvae_assisted_design(
                    model=cvae, tandem=tandem, target_centers=(low, high),
                    ideal_target=ideal, wavelengths=wavelengths,
                    n_candidates=DUAL_CANDIDATES, device=trainer.DEVICE,
                    candidate_override=override)
                store.append(error(np.asarray(selected['final_response'],
                                              dtype=float).reshape(-1),
                                   (low, high)))
            except (ValueError, KeyError):
                store.append(float('nan'))
        print(f'  {low:g}/{high:g}: 原基线 {analytic[-1]:6.2f}  扰动ITS '
              f'{its_pick[-1]:6.2f}  CVAE {cvae_pick[-1]:6.2f}', flush=True)

    return dict(dual_labels=np.array([f'{a:g}/{b:g}' for a, b in pairs]),
                dual_analytic=np.array(analytic),
                dual_its_pick=np.array(its_pick),
                dual_cvae_pick=np.array(cvae_pick))


def draw(data):
    # After the model imports, not before: see FIGURE_STYLE.
    matplotlib.rcParams.update(FIGURE_STYLE)
    """Two panels, single-notch above double-notch.

    The two systems differ by an order of magnitude in error, so a shared axis
    would flatten the single-notch comparison; they get a panel each.
    """
    figure, panels = plt.subplots(2, 1, figsize=(FIG_WIDTH, 7.6),
                                  layout='constrained')
    blocks = (
        (panels[0], data['targets'], data['analytic'], data['its_pick'],
         data['cvae_pick'], 'Single-notch targets', 'nm', N_CANDIDATES,
         r'$|\Delta\lambda|$ (nm)'),
        (panels[1], data['dual_labels'], data['dual_analytic'],
         data['dual_its_pick'], data['dual_cvae_pick'],
         'Double-notch target pairs', '', DUAL_CANDIDATES,
         # The two panels do not plot the same quantity: the double-notch bar
         # is the sum over both resonances, so one shared "Dip error" label
         # understated it by about a factor of two.
         r'$\Sigma|\Delta\lambda|$ (nm)'))

    for axes, labels, analytic, its, cvae, title, unit, count, ylabel in blocks:
        series = ((analytic, DARK, 'Direct, 1 analytic target'),
                  (its, ORANGE,
                   f'Direct, {count} targets + selection'),
                  (cvae, BLUE,
                   f'CVAE, {count} candidates + selection'))
        positions = np.arange(len(labels), dtype=float)
        width = 0.26
        for index, (values, colour, label) in enumerate(series):
            axes.bar(positions + (index - 1) * width, values, width,
                     color=colour, alpha=0.85, edgecolor=DARK, linewidth=0.7,
                     zorder=3,
                     label=f'{label}  (mean {np.nanmean(values):.2f} nm)')
        axes.set_xticks(positions)
        axes.set_xticklabels([f'{v:g}' if unit else str(v) for v in labels])
        axes.set_xlim(-0.55, len(labels) - 0.45)
        # Headroom for the legend, which sits inside the axes: at 1.1 the tallest
        # bars ran through it.
        axes.set_ylim(0, np.nanmax([v.max() for v, _, _ in series]) * 1.42)
        axes.set_title(title, pad=6.0, color=DARK, fontsize=BASE_FONT)
        axes.set_ylabel(ylabel)
        axes.tick_params(axis='x', length=0.0, pad=6.0)
        axes.tick_params(axis='y', which='major', direction='in', length=4.3,
                         right=True)
        for side in ('top', 'right'):
            axes.spines[side].set_visible(False)
        axes.grid(axis='y', color='#E6E6E6', linewidth=0.6)
        axes.set_axisbelow(True)
        axes.legend(loc='upper left', frameon=False, handlelength=1.3,
                    handletextpad=0.5, labelspacing=0.25,
                    fontsize=BASE_FONT - 3.0, labelcolor=DARK)

    panels[0].set_xlabel('Requested wavelength (nm)')
    panels[1].set_xlabel('Requested wavelength pair (nm)')
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-cache', action='store_true')
    parser.add_argument('--output', default='figures/budget_ablation.png')
    args = parser.parse_args()

    stem = os.path.splitext(os.path.join(ROOT, args.output))[0]
    cache = stem + '_data.npz'
    if args.from_cache:
        data = dict(np.load(cache))
    else:
        data = compute()
        data.update(compute_dual())
        np.savez(cache, **data)

    figure = draw(data)
    for suffix, options in (('.tiff', {'dpi': FIG_DPI,
                                       'pil_kwargs': {'compression': 'tiff_lzw'}}),
                            ('.pdf', {}), ('.svg', {}),
                            ('.png', {'dpi': FIG_DPI})):
        figure.savefig(stem + suffix, facecolor='white', **options)

    print(f"\n单峰均值  原基线 {data['analytic'].mean():.2f}   "
          f"扰动ITS 未筛选 {data['its_all'].mean():.2f} "
          f"选中 {data['its_pick'].mean():.2f}   "
          f"CVAE 未筛选 {data['cvae_all'].mean():.2f} "
          f"选中 {data['cvae_pick'].mean():.2f} nm")
    print(f'Saved: {stem}.{{tiff,pdf,svg,png}}')


if __name__ == '__main__':
    main()
