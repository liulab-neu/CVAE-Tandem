#!/usr/bin/env python3
"""Many designs at once, for the supplement.

The 3x3 figures in the main text show three cases each; a reviewer will
reasonably ask whether those three are representative.  This lays out a wide
span of targets under the same two routes, so the reader can see the direct
route's miss and the CVAE-assisted correction across the whole reachable band
rather than at three hand-picked points.

Responses are computed in memory: writing a full design directory per case
would litter the repository with dozens of runs that nothing else reads.  The
curves are cached next to the figure, so the layout can be retuned with
--from-cache without paying for the sweep again.
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
from matplotlib.ticker import MultipleLocator
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.abspath(__file__))
BLUE = '#0072B2'
ORANGE = '#D55E00'
GREEN = '#009E73'
DARK = '#202020'
GRAY = '#5A5A5A'
GUIDE = '#C9C9C9'
FIG_DPI = 600

# Geometry in inches.  The type size only means something relative to the
# canvas, so these two numbers are pinned to what the rest of the paper's
# figures use -- 16 pt on an 11 in canvas -- and every rule, pad and line width
# below is set in the same units.  Drawing this one at true print size instead
# left its labels visibly smaller than the neighbouring figures.
FIG_WIDTH = 11.0
PANEL_HEIGHT = 2.05
# 19, not the 16 the single-panel figures use: this one is 11 in wide with a
# 4x3 grid on it, so at the same point size its labels print noticeably smaller
# than those of the figures beside it in the supplement.
BASE_FONT = 19.0
TITLE_PAD = 24.0 * BASE_FONT / 16.0

# Named and re-applied before drawing: importing 1peak's
# lorentz_vs_cvae_compare runs its own rcParams.update at import time
# and drops font.size from 19 to 8, which is why this figure's tick
# labels and axis names came out visibly smaller than the double-notch
# one it is meant to sit beside.
FIGURE_STYLE = {
    'font.family': 'serif',
    # Nimbus Roman is the Times clone that is actually installed here; without
    # it matplotlib silently falls back to DejaVu Serif, which is what made the
    # earlier version of this figure look nothing like the rest of the paper.
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
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'xtick.minor.width': 0.7,
    'ytick.minor.width': 0.7,
    # The legend keeps the smaller size: three spelled-out entries at the
    # panel size would be wider than the canvas.
    'legend.fontsize': BASE_FONT - 3.0,
    'lines.solid_capstyle': 'round',
    'lines.dash_capstyle': 'round',
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
}

matplotlib.rcParams.update(FIGURE_STYLE)


def apply_style(font_scale=1.0):
    """Re-assert the figure's own style, scaled, over anything an import set."""
    matplotlib.rcParams.update(FIGURE_STYLE)
    if font_scale != 1.0:
        for key in ('font.size', 'axes.labelsize', 'axes.titlesize',
                    'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize'):
            matplotlib.rcParams[key] = FIGURE_STYLE[key] * font_scale


def style(axes, wavelengths):
    """Ticks in, on all four sides, with minors -- the house style of the
    optics journals this figure is written for."""
    axes.set_xlim(float(wavelengths[0]), float(wavelengths[-1]))
    axes.set_ylim(-0.04, 1.08)
    axes.set_yticks([0.0, 0.5, 1.0])
    axes.set_xticks([400, 500, 600, 700, 800])
    axes.xaxis.set_minor_locator(MultipleLocator(50))
    axes.yaxis.set_minor_locator(MultipleLocator(0.25))
    axes.tick_params(which='major', direction='in', length=4.3, pad=3.5,
                     top=True, right=True)
    axes.tick_params(which='minor', direction='in', length=2.3,
                     top=True, right=True)
    axes.set_axisbelow(True)


def single_cases(net, lvc, wavelengths, targets, candidates, device):
    import torch
    from notch_quality import (select_single_notch_candidate,
                               single_notch_verdict)
    out = []
    for centre in targets:
        index = int(np.argmin(np.abs(wavelengths - centre)))
        _, ideal = lvc.lorentzian_trans(centre, wlmin=380, wlmax=800,
                                        gamma=15, wavelengths=wavelengths)
        tensor = torch.as_tensor(np.asarray(ideal, dtype=np.float32)
                                 ).reshape(1, -1).to(device)
        with torch.no_grad():
            _, direct = net.test(tensor, tensor, 'tandem')
        direct = np.asarray(direct, dtype=float).reshape(-1)

        generated = lvc.cvae_generate_spectrum(
            centre, n_samples=candidates,
            condition_value=index / (len(wavelengths) - 1), random_seed=42)
        with torch.no_grad():
            _, responses = net.test(generated, generated, 'tandem')
        responses = np.asarray(responses, dtype=float)
        selection = select_single_notch_candidate(
            wavelengths, responses, float(centre),
            wavelength_tolerance_nm=0.5,
            tie_breaker=np.mean((responses - np.asarray(ideal, float)) ** 2,
                                axis=1),
            generated_responses=np.asarray(generated.detach().cpu()),
            generated_wavelength_tolerance_nm=0.5,
            generated_grid_slack_nm=0.5 * float(np.median(np.diff(wavelengths))))
        assisted = responses[selection.selected_index]
        # The same treatment the double-notch panels get: the notch's depth
        # and width are checked, and a response that lands on the target
        # while being far too broad is marked rather than silently counted as
        # a success.
        ideal_array = np.asarray(ideal, dtype=float)
        direct_ok, _, direct_measured = single_notch_verdict(
            wavelengths, direct, ideal_array, float(centre))
        assisted_ok, _, assisted_measured = single_notch_verdict(
            wavelengths, assisted, ideal_array, float(centre))
        out.append({'label': f'{centre:g} nm',
                    'centres': (float(centre),),
                    'ideal': ideal_array,
                    'direct': direct, 'assisted': assisted,
                    'direct_error': direct_measured['error'],
                    'assisted_error': assisted_measured['error'],
                    'direct_gate': direct_ok,
                    'assisted_gate': assisted_ok})
        print(f'  {centre:g} nm: 直接 {direct_measured["error"]:5.2f}'
              f'{"" if direct_ok else " 不合格"}  '
              f'CVAE {assisted_measured["error"]:5.2f}'
              f'{"" if assisted_ok else " 不合格"}'
              f'   (宽度比 {direct_measured["width_ratio"]:.2f} / '
              f'{assisted_measured["width_ratio"]:.2f})', flush=True)
    return out


def dual_cases(dpu, tandem, cvae, wavelengths, pairs, candidates, device):
    # dual_peak_utils.shape_gate_verdict, not a local reimplementation: it
    # carries all four of the manuscript's admissibility conditions, and an
    # earlier copy here silently dropped the notch-width one. The verdict is
    # reported alongside the error, never in place of it -- the inverse
    # network always returns a structure, so "inadmissible" describes a design
    # that exists rather than the absence of one, and for the direct route,
    # which has no selection step, its failures are its result.
    def graded(row, centres, ideal):
        passed, _failed, measured = dpu.shape_gate_verdict(
            np.asarray(row, dtype=float).reshape(-1), wavelengths, ideal)
        if not measured:
            return float('nan'), False
        positions = np.asarray(measured['peaks'], dtype=float)
        error = float(np.abs(positions - np.asarray(centres, float)).sum())
        return error, bool(passed)

    out = []
    for low, high in pairs:
        ideal = dpu.bounded_double_lorentzian((low, high), wavelengths, 15.0)
        _, responses = tandem.inverse_forward(ideal)
        direct = np.asarray(responses[0], dtype=float)
        try:
            dpu.seed_everything(42)
            selected = dpu.select_cvae_assisted_design(
                model=cvae, tandem=tandem, target_centers=(low, high),
                ideal_target=ideal, wavelengths=wavelengths,
                n_candidates=candidates, device=device)
            assisted = np.asarray(selected['final_response'], dtype=float
                                  ).reshape(-1)
        except (ValueError, KeyError):
            continue

        direct_error, direct_gate = graded(direct, (low, high), ideal)
        assisted_error, assisted_gate = graded(assisted, (low, high), ideal)

        # %g, not the bare value: the shared pair grid hands out floats and
        # "400.0 / 450.0 nm" is not how a target is written.
        out.append({'label': f'{low:g} / {high:g} nm',
                    'centres': (float(low), float(high)),
                    'ideal': ideal,
                    'direct': direct, 'assisted': assisted,
                    'direct_error': direct_error,
                    'assisted_error': assisted_error,
                    'direct_gate': direct_gate,
                    'assisted_gate': assisted_gate})
        def shown(value, passed):
            return f'{value:6.2f}' + ('' if passed else ' 不合格')
        print(f'  {low:g}/{high:g}: 直接 {shown(direct_error, direct_gate)}  '
              f'CVAE {shown(assisted_error, assisted_gate)}', flush=True)
    return out


def save_cases(path, wavelengths, cases):
    width = max(len(case['centres']) for case in cases)
    centres = np.full((len(cases), width), np.nan)
    for row, case in enumerate(cases):
        centres[row, :len(case['centres'])] = case['centres']
    np.savez(path, wavelengths=wavelengths,
             labels=np.array([case['label'] for case in cases]),
             centres=centres,
             ideal=np.stack([case['ideal'] for case in cases]),
             direct=np.stack([case['direct'] for case in cases]),
             assisted=np.stack([case['assisted'] for case in cases]),
             direct_error=np.array([case['direct_error'] for case in cases]),
             assisted_error=np.array([case['assisted_error']
                                      for case in cases]),
             # Single-notch cases have no double-notch shape gate, so they
             # store True and the flag simply carries no information there.
             direct_gate=np.array([case.get('direct_gate', True)
                                   for case in cases]),
             assisted_gate=np.array([case.get('assisted_gate', True)
                                     for case in cases]))


def load_cases(path):
    stored = np.load(path, allow_pickle=False)
    cases = [{'label': str(stored['labels'][i]),
              'centres': tuple(stored['centres'][i]
                               [~np.isnan(stored['centres'][i])]),
              'ideal': stored['ideal'][i],
              'direct': stored['direct'][i],
              'assisted': stored['assisted'][i],
              'direct_error': float(stored['direct_error'][i]),
              'assisted_error': float(stored['assisted_error'][i]),
              'direct_gate': bool(stored['direct_gate'][i])
              if 'direct_gate' in stored else True,
              'assisted_gate': bool(stored['assisted_gate'][i])
              if 'assisted_gate' in stored else True}
             for i in range(len(stored['labels']))]
    return stored['wavelengths'], cases


def draw(wavelengths, cases, columns, error_symbol):
    rows = (len(cases) + columns - 1) // columns
    height = rows * (PANEL_HEIGHT + 0.58 * BASE_FONT / 16.0) + 1.0
    figure, panels = plt.subplots(rows, columns,
                                  figsize=(FIG_WIDTH, height),
                                  sharex=True, sharey=True,
                                  layout='constrained')
    figure.get_layout_engine().set(w_pad=0.03, h_pad=0.03,
                                   wspace=0.035, hspace=0.075)
    panels = np.atleast_2d(panels)
    flat = panels.ravel()
    for axes, case in zip(flat, cases):
        for centre in case['centres']:
            # A hairline at the requested wavelength lets the reader see the
            # miss directly instead of inferring it from the number above.
            axes.axvline(centre, color=GUIDE, linewidth=0.8,
                         linestyle=(0, (1.4, 2.2)), zorder=0)
        axes.plot(wavelengths, case['ideal'], color=ORANGE, linewidth=1.5,
                  linestyle=(0, (4.4, 2.6)), zorder=2)
        axes.plot(wavelengths, case['direct'], color=GREEN, linewidth=1.5,
                  zorder=3)
        axes.plot(wavelengths, case['assisted'], color=BLUE, linewidth=1.8,
                  zorder=4)
        # Target in the title, the two errors on a second, quieter line below
        # it -- one long title per panel was what made the old version run into
        # its neighbour.  The pad has to clear the whole error line, not just
        # its baseline, or the two run together.  Panel letters are added
        # during layout, not here.
        axes.set_title(case['label'], loc='center', pad=TITLE_PAD, color=DARK)
        # A design that misses the admissibility conditions still has dip
        # positions and still has an error; the dagger says the response is
        # not a usable double notch, which is a different statement from
        # having produced nothing. "n/a" is kept only for the case where no
        # two notches resolve at all and there is genuinely nothing to report.
        def error_label(value, admissible):
            if not np.isfinite(value):
                return 'n/a'
            return f'{value:.1f}' + ('' if admissible else '$^{\\dagger}$')

        axes.text(0.5, 1.015,
                  f"{error_symbol} "
                  f"{error_label(case['direct_error'], case['direct_gate'])} "
                  f"$\\rightarrow$ "
                  f"{error_label(case['assisted_error'], case['assisted_gate'])}"
                  f" nm",
                  transform=axes.transAxes, ha='center', va='bottom',
                  fontsize=BASE_FONT - 1.5, color=GRAY)
        style(axes, wavelengths)
    for axes in flat[len(cases):]:
        axes.axis('off')
    # One label for the whole grid rather than one per edge panel: every panel
    # shares both axes, so repeating the names adds ink without information.
    # The x label goes on the middle panel of the bottom row rather than on the
    # figure, because an outside legend already occupies the strip a figure-level
    # label would be placed in.
    panels[-1, columns // 2].set_xlabel('Wavelength (nm)')
    figure.supylabel('Transmittance', color=DARK)

    handles = [Line2D([], [], color=ORANGE, linewidth=1.5,
                      linestyle=(0, (4.4, 2.6))),
               Line2D([], [], color=GREEN, linewidth=1.5),
               Line2D([], [], color=BLUE, linewidth=1.8)]
    # Same wording as the 3x3 comparison figures: both curves are responses of
    # a designed structure, not spectra the networks were asked to match.
    figure.legend(handles, ['Ideal target spectrum', 'Direct Tandem response',
                            'CVAE-assisted response'],
                  loc='outside lower center', ncol=3, frameon=False,
                  handlelength=1.8, handletextpad=0.5, columnspacing=1.5,
                  borderaxespad=0.0, labelcolor=DARK)
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('1peak', '2peak'), required=True)
    parser.add_argument('--columns', type=int, default=3)
    # 20, matching the case pipeline that produced the designs the tables and
# the CST runs report; raising it to 100 changes the mean dip error by less
# than the 0.42 nm wavelength sampling step.
    parser.add_argument('--candidates', type=int, default=20)
    parser.add_argument('--limit', type=int,
                        help='keep this many cases, so the grid stays full')
    parser.add_argument('--font-scale', type=float, default=1.0,
                        help='scale every text size, for a figure that has to '
                             'sit at something other than full text width')
    parser.add_argument('--from-cache', action='store_true',
                        help='replot the stored curves instead of rerunning '
                             'the networks')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    kind = args.kind

    stem = os.path.splitext(os.path.join(ROOT, args.output)
                            if not os.path.isabs(args.output)
                            else args.output)[0]
    cache = stem + '_curves.npz'
    if args.from_cache:
        wavelengths, cases = load_cases(cache)
    else:
        sys.path.insert(0, f'{ROOT}/{kind}')
        os.chdir(f'{ROOT}/{kind}')
        if kind == '1peak':
            import train_networks as trainer
            import lorentz_vs_cvae_compare as lvc
            net = lvc.build_tandem_network(trainer.DEVICE)
            net.eval_mode('tandem')
            wavelengths = net.wavelengths.detach().cpu().numpy().reshape(-1)
            targets = list(range(420, 651, 20))
            cases = single_cases(net, lvc, wavelengths, targets,
                                 args.candidates, trainer.DEVICE)
        else:
            import train_networks_2p as trainer
            import dual_peak_utils as dpu
            wavelengths = dpu.load_wavelength_grid()
            tandem = dpu.build_tandem_runtime(trainer.DEVICE)
            cvae = dpu.build_cvae(trainer.DEVICE)
            pairs = dpu.supplement_pair_grid()
            cases = dual_cases(dpu, tandem, cvae, wavelengths, pairs,
                               args.candidates, trainer.DEVICE)
        save_cases(cache, wavelengths, cases)

    if args.limit:
        # A trailing part-filled row reads as a mistake; trimming to a multiple
        # of the column count keeps the block rectangular.
        cases = cases[:args.limit]
    # The single-notch error is one distance; the dual-notch error sums the two,
    # so the two figures must not label their numbers the same way.  \Sigma, not
    # \sum: the big-operator glyph is taller than the line it sits on and pushes
    # into the title above.
    symbol = ('$|\\Delta\\lambda|$' if max(len(case['centres'])
                                           for case in cases) == 1
              else '$\\Sigma|\\Delta\\lambda|$')
    # After the model imports, not before: see FIGURE_STYLE.
    apply_style(args.font_scale)
    figure = draw(wavelengths, cases, args.columns, symbol)

    for suffix, options in (('.tiff', {'dpi': FIG_DPI,
                                       'pil_kwargs': {'compression': 'tiff_lzw'}}),
                            ('.pdf', {}), ('.svg', {}),
                            ('.png', {'dpi': FIG_DPI})):
        figure.savefig(stem + suffix, facecolor='white', **options)
    print(f'\n{len(cases)} 个案例  Saved: {stem}.{{tiff,pdf,svg,png}}')


if __name__ == '__main__':
    main()
