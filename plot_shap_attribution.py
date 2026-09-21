#!/usr/bin/env python3
"""SHAP attribution of the forward network, in the manuscript's figure style.

Which geometry parameter sets which spectral feature: one row per explained
quantity.  The left panel ranks the inputs by mean |SHAP|; each remaining panel
shows how one input's attribution varies with its own value, coloured by the
input it interacts with most, so a reader can see both what matters and where
it stops mattering.  The dual-notch case explains four quantities (both dip
positions and both dip depths) against three parameters; the single-notch case
explains two against two.

The attributions are cached next to the figure, so the layout can be retuned
with --from-cache instead of re-running the explainer.
"""

import argparse
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tandem-mpl-cache')

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.abspath(__file__))
DARK = '#202020'
GRAY = '#5A5A5A'
ZERO = '#9A9A9A'
TREND = '#101010'
# One colour per input, held fixed across the rows so a reader can follow a
# parameter down the figure without re-reading the axis.
FEATURE_COLOURS = ('#0072B2', '#D55E00', '#009E73')
CMAP = LinearSegmentedColormap.from_list(
    'shap_coolwarm', ('#2166AC', '#F2F2F2', '#B2182B'), N=256)
FIG_DPI = 600

# Both figures are drawn on the same 11 in canvas at the same type size as the
# rest of the paper's figures, so the dual-notch grid simply packs narrower
# panels rather than shrinking its labels.
FIG_WIDTH = 11.0
BASE_FONT = 16.0
# Wide and shallow: these panels carry a trend, not a shape, and the extra
# height only pushed the rows apart.
PANEL_ASPECT = 0.60

matplotlib.rcParams.update({
    'font.family': 'serif',
    # Times New Roman is not installed here; without the clone matplotlib falls
    # back to DejaVu Serif and the figure stops matching the others.
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif', 'serif'],
    'font.size': BASE_FONT,
    # 15, below what the tightest panel of either figure can hold, so the
    # axis labels are one size across both and never get shrunk at draw time.
    # They used to be BASE_FONT + 1 and were then shrunk per label to fit,
    # which is how 17 pt and 12.6 pt labels ended up in the same figure and
    # why the single- and double-notch figures did not match each other.
    'axes.labelsize': 15.0,
    'axes.titlesize': BASE_FONT + 1.0,
    'axes.labelcolor': DARK,
    'axes.edgecolor': DARK,
    'axes.linewidth': 0.9,
    'xtick.labelsize': BASE_FONT - 2.0,
    'ytick.labelsize': BASE_FONT - 2.0,
    'xtick.color': DARK,
    'ytick.color': DARK,
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'legend.fontsize': BASE_FONT,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


def style(axes):
    axes.tick_params(which='major', direction='out', length=3.6, pad=3.0)
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    axes.set_axisbelow(True)


def smooth_trend(x, y, points=240):
    """Gaussian-kernel local mean -- the black guide through the cloud.

    A running mean rather than a fit: the dependence curves bend and saturate,
    and a polynomial of any degree that follows them also invents structure
    where the samples run out.
    """
    span = float(np.ptp(x))
    if span == 0.0:
        return None, None
    bandwidth = 0.05 * span
    grid = np.linspace(x.min(), x.max(), points)
    weights = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bandwidth) ** 2)
    total = weights.sum(axis=1)
    # Where the samples are too sparse for the kernel there is nothing to say.
    keep = total > 1e-8
    return grid[keep], (weights[keep] @ y) / total[keep]


def interaction_partner(values, features, index):
    """Which other input this one's attribution depends on.

    Scored on what the input's own trend leaves behind: if the residual still
    tracks another parameter, that parameter is the interaction, and colouring
    the panel by it shows the reader the same thing the number does.
    """
    x, y = features[:, index], values[:, index]
    grid, trend = smooth_trend(x, y)
    residual = y if grid is None else y - np.interp(x, grid, trend)
    best, score = None, -1.0
    for other in range(features.shape[1]):
        if other == index or np.ptp(features[:, other]) == 0:
            continue
        if np.ptp(residual) == 0:
            correlation = 0.0
        else:
            correlation = abs(float(np.corrcoef(residual,
                                                features[:, other])[0, 1]))
        if not np.isfinite(correlation):
            correlation = 0.0
        if correlation > score:
            best, score = other, correlation
    return best


def quantity(symbol, unit):
    """The unit of the quantity a SHAP axis explains, if it has one.

    The row heading already names the quantity ("First-notch wavelength"),
    so repeating the symbol on both axes of every panel was redundant -- and
    it was what made the labels outgrow the narrow panels of the dual-notch
    grid, forcing a per-label shrink that left 17 pt and 12.6 pt labels side
    by side. The unit alone still separates a wavelength attribution in nm
    from a dimensionless transmittance one, which is the distinction that
    matters between the rows.
    """
    return f'({unit})' if unit else ''


def importance(axes, values, names, symbol, unit):
    """Mean |SHAP| per input: a lollipop rather than a bar, so the value label
    has somewhere to sit without a heavy block of ink behind it."""
    mean = np.abs(values).mean(axis=0)
    order = np.argsort(mean)
    for row, index in enumerate(order):
        colour = FEATURE_COLOURS[index % len(FEATURE_COLOURS)]
        axes.plot([0.0, mean[index]], [row, row], color=colour, linewidth=3.2,
                  solid_capstyle='round', zorder=2)
        axes.plot([mean[index]], [row], marker='o', color=colour,
                  markersize=7.0, markeredgecolor='white',
                  markeredgewidth=0.8, zorder=3)
        axes.annotate(f'{mean[index]:.3g}', xy=(mean[index], row),
                      xytext=(8, 0), textcoords='offset points',
                      va='center', ha='left', color=DARK,
                      fontsize=BASE_FONT - 1.0)
    axes.set_yticks(range(len(order)))
    axes.set_yticklabels([names[i] for i in order])
    axes.set_ylim(-0.65, len(order) - 0.35)
    # Room on the right for the value labels, which sit outside the lollipop.
    axes.set_xlim(0.0, float(mean.max()) * 1.42 if mean.max() else 1.0)
    axes.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
    axes.set_xlabel(f'Mean |SHAP| {quantity(symbol, unit)}'.strip())
    axes.tick_params(axis='y', length=0.0)
    axes.spines['left'].set_visible(False)
    style(axes)


def dependence(axes, values, features, index, partner):
    x, y = features[:, index], values[:, index]
    axes.axhline(0.0, color=ZERO, linewidth=0.9, linestyle=(0, (1.4, 2.2)),
                 zorder=1)
    points = axes.scatter(x, y, c=features[:, partner], cmap=CMAP, s=17.0,
                          linewidths=0.0, alpha=0.9, rasterized=True, zorder=2)
    grid, trend = smooth_trend(x, y)
    if grid is not None:
        axes.plot(grid, trend, color=TREND, linewidth=1.9, zorder=4)
    axes.xaxis.set_major_locator(MaxNLocator(4))
    axes.yaxis.set_major_locator(MaxNLocator(5))
    style(axes)
    return points


def colourbar(figure, axes, points, label, panel_height):
    """Horizontal scale sitting above its own panel.

    Positioned in inches off the panel top rather than in axes fractions: the
    two figures have different panel heights, and a fraction that clears the
    tick labels in one of them collides in the other.
    """
    def fraction(inches):
        return inches / panel_height

    cax = axes.inset_axes([0.0, 1.0 + fraction(0.36), 1.0, fraction(0.10)])
    bar = figure.colorbar(points, cax=cax, orientation='horizontal')
    bar.outline.set_linewidth(0.7)
    bar.outline.set_edgecolor(DARK)
    cax.xaxis.set_ticks_position('bottom')
    cax.xaxis.set_label_position('top')
    values = points.get_array()
    bar.set_ticks([float(np.min(values)), float(np.max(values))])
    cax.set_xticklabels([f'{np.min(values):.2f}', f'{np.max(values):.2f}'])
    cax.tick_params(length=0.0, pad=2.0, labelsize=BASE_FONT - 3.0)
    bar.set_label(label, labelpad=3.0, fontsize=BASE_FONT - 1.0, color=DARK)
    return cax


class Explained:
    """Scalar spectral feature of the forward network, for SHAP to attribute.

    The dips come from find_peaks with the same prominence gate the loss and
    the evaluation use.  The original script took the two smallest samples of
    the spectrum instead, and on 99% of the test set both of those land inside
    the same notch -- so its "two dips" were two neighbouring points of one
    resonance, 89 nm from the real pair.

    The 20 nm minimum separation matters for the same reason: a prominence
    gate alone still admits two local minima of one broad notch, which would
    make lambda1 and lambda2 attributions of the same resonance.  This is the
    separation dual_peak_utils.detect_two_resonances applies.
    """

    MINIMUM_DIP_SEPARATION_NM = 20.0

    def __init__(self, net, mode, dual, grid):
        self.net, self.mode, self.dual, self.grid = net, mode, dual, grid
        self.distance_samples = 1
        if dual and grid is not None:
            step = float(np.median(np.diff(np.asarray(grid, dtype=float))))
            self.distance_samples = max(
                1, int(round(self.MINIMUM_DIP_SEPARATION_NM / step)))

    def __call__(self, x):
        import torch
        tensor = torch.as_tensor(np.asarray(x, dtype=np.float32),
                                 device=next(self.net.parameters()).device)
        with torch.no_grad():
            spectra = self.net(tensor).cpu().numpy()
        out = []
        for row in spectra:
            if self.dual:
                peaks, _ = find_peaks(-row, prominence=0.01,
                                      distance=self.distance_samples)
                if len(peaks) >= 2:
                    chosen = np.sort(peaks[np.argsort(row[peaks])[:2]])
                else:
                    fallback = int(np.argmin(row))
                    chosen = np.array([fallback, fallback])
            else:
                chosen = np.array([int(np.argmin(row))] * 2)
            if self.mode == 'lambda1':
                out.append(self.grid[chosen[0]])
            elif self.mode == 'lambda2':
                out.append(self.grid[chosen[1]])
            elif self.mode == 'val1':
                out.append(row[chosen[0]])
            else:
                out.append(row[chosen[1]])
        return np.asarray(out, dtype=float)


def build_model(kind):
    import argparse as _argparse

    sys.path.insert(0, f'{ROOT}/{kind}')
    os.chdir(f'{ROOT}/{kind}')
    if kind == '1peak':
        import train_networks as trainer
        grid = None
    else:
        import train_networks_2p as trainer
        import dual_peak_utils as dpu
        grid = dpu.load_wavelength_grid()
    net = trainer.build_tandem(42, _argparse.Namespace(
        w_peak_wavelength=0.0, w_peak_intensity=0.0))
    net.restore_FNN(f'{ROOT}/{kind}/model/DNN_tandem_FNN_label.ckpt')
    net.fnn.eval()
    if grid is None:
        grid = net.wavelengths.detach().cpu().numpy().reshape(-1)
    return trainer, net, grid


def case_layout(kind):
    """Names, units and headings -- short enough to fit a narrow panel, with
    the long form carried by the column heading instead of the axis label."""
    if kind == '2peak':
        names = ['$R$', '$n_{\\mathrm{host,1}}$', '$n_{\\mathrm{host,2}}$']
        headers = ['$R$ dependence', '$n_{\\mathrm{host,1}}$ dependence',
                   '$n_{\\mathrm{host,2}}$ dependence']
        axis_labels = ['$R$ (nm)', '$n_{\\mathrm{host,1}}$',
                       '$n_{\\mathrm{host,2}}$']
        colour_labels = ['$R$ (nm)', '$n_{\\mathrm{host,1}}$',
                         '$n_{\\mathrm{host,2}}$']
        modes = [('lambda1', 'nm', 'First-notch wavelength', '$\\lambda_1$'),
                 ('lambda2', 'nm', 'Second-notch wavelength', '$\\lambda_2$'),
                 ('val1', '', 'First-notch transmittance', '$T_{\\min,1}$'),
                 ('val2', '', 'Second-notch transmittance', '$T_{\\min,2}$')]
    else:
        names = ['$R$', '$n_{\\mathrm{host}}$']
        headers = ['$R$ dependence', '$n_{\\mathrm{host}}$ dependence']
        axis_labels = ['Particle radius, $R$ (nm)',
                       'Host refractive index, $n_{\\mathrm{host}}$']
        colour_labels = ['$R$ (nm)', '$n_{\\mathrm{host}}$']
        modes = [('lambda1', 'nm', 'Wavelength at minimum transmittance',
                  '$\\lambda_{\\min}$'),
                 ('val1', '', 'Minimum transmittance', '$T_{\\min}$')]
    return names, headers, axis_labels, colour_labels, modes


def compute(kind, samples, background):
    import shap

    trainer, net, grid = build_model(kind)
    structures, _, train_idx, _, test_idx = trainer.load_split(42)
    reference = structures[train_idx[:background]].numpy()
    features = structures[test_idx[:samples]].numpy()
    names, _, _, _, modes = case_layout(kind)

    stack = []
    for mode, *_ in modes:
        wrapped = Explained(net.fnn, mode, kind == '2peak', grid)
        explainer = shap.Explainer(wrapped, reference)
        values = explainer(features).values
        stack.append(np.asarray(values, dtype=float))
        print(f'  {mode}: |SHAP| 均值 ' +
              '  '.join(f'{n}={v:.3g}' for n, v
                        in zip(names, np.abs(values).mean(axis=0))), flush=True)
    return features, np.stack(stack)


def draw(kind, features, stack):
    names, headers, axis_labels, colour_labels, modes = case_layout(kind)
    rows, columns = len(modes), 1 + len(names)
    # One partner per column, scored over every row: the feature values are the
    # same in all rows, so a single scale serves the column and the colour bar
    # only has to be drawn once, at the top.
    partners = []
    for index in range(len(names)):
        votes = [interaction_partner(stack[row], features, index)
                 for row in range(rows)]
        votes = [v for v in votes if v is not None]
        partners.append(max(set(votes), key=votes.count) if votes else 0)

    panel_width = (FIG_WIDTH - 1.15) / columns
    panel_height = panel_width * PANEL_ASPECT
    # Nothing is titled, so the space each heading needs has to be reserved by
    # hand: a strip at the top of the canvas for the column headings and the
    # first row's colour bars, and a gap between the rows for the row heading.
    top_strip, row_gap = 1.11, 0.40
    height = (top_strip + rows * (panel_height + 0.62)
              + (rows - 1) * row_gap + 0.10)
    figure, panels = plt.subplots(rows, columns,
                                  figsize=(FIG_WIDTH, height),
                                  layout='constrained')
    figure.get_layout_engine().set(
        w_pad=0.04, h_pad=0.04, wspace=0.045,
        hspace=row_gap / panel_height,
        rect=(0.0, 0.0, 1.0, 1.0 - top_strip / height))
    panels = np.atleast_2d(panels)

    bars = []
    for row, (_, unit, _, symbol) in enumerate(modes):
        values = stack[row]
        importance(panels[row, 0], values, names, symbol, unit)
        for column in range(len(names)):
            axes = panels[row, column + 1]
            points = dependence(axes, values, features, column,
                                partners[column])
            axes.set_xlabel(axis_labels[column])
            # On every panel, not just the first: the columns of a row differ
            # by two orders of magnitude, so a single label on the left would
            # read as a shared scale.
            axes.set_ylabel(f'SHAP {quantity(symbol, unit)}'.strip())
            if row == 0:
                bars.append(colourbar(figure, axes, points,
                                      colour_labels[partners[column]],
                                      panel_height))
        panels[row, 0].set_ylabel('Input feature')

    # The headings go on once the layout has settled, so they can be hung off
    # the letters' real positions instead of a guessed offset.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    inverse = figure.transFigure.inverted()

    def box(artist):
        return inverse.transform_bbox(artist.get_window_extent(renderer))

    def fit(artist, limit, vertical=False):
        """Shrink one label until it stops overrunning the space given.

        Used for the row headings, which are independent of one another --
        the axis labels below are sized as a group instead, so that a long one
        does not end up visibly smaller than a short one in the same figure.
        """
        extent = box(artist)
        length = extent.height if vertical else extent.width
        if length > limit > 0:
            artist.set_fontsize(max(artist.get_fontsize() * limit / length,
                                    11.5))

    def needed_scale(artist, limit, vertical=False):
        """How much this label must shrink to stop overrunning its panel.

        The layout is frozen by this point, so a label that outgrows its panel
        has nowhere to push -- it just reaches across into the neighbour.
        """
        extent = box(artist)
        length = extent.height if vertical else extent.width
        if length > limit > 0:
            return limit / length
        return 1.0

    # One size for every axis label in the figure, set by whichever label
    # needs the most shrinking. Shrinking each label on its own put 17 pt and
    # 12.6 pt labels side by side in the same figure -- the long y labels of
    # the single-notch case were squeezed while the short ones were not, which
    # read as two different type sizes rather than one considered choice.
    scale = 1.0
    for axes in panels.ravel():
        position = axes.get_position()
        scale = min(scale,
                    needed_scale(axes.xaxis.label, position.width),
                    needed_scale(axes.yaxis.label, position.height,
                                 vertical=True))
    if scale < 1.0:
        for axes in panels.ravel():
            for label in (axes.xaxis.label, axes.yaxis.label):
                label.set_fontsize(
                    max(label.get_fontsize() * scale, 11.5))
        print(f'  axis labels scaled to {scale:.2f} of '
              f'{BASE_FONT + 1.0:.0f} pt uniformly', flush=True)

    # The heading sits on its own panel, just above it: in the first row the
    # colour bars occupy the band overhead, so a heading hung from the top of
    # the strip would float half an inch clear of the panel it names.
    gap = 0.10 / height
    for row, (_, _, heading, _) in enumerate(modes):
        position = panels[row, 0].get_position()
        text = figure.text(0.5 * (position.x0 + position.x1),
                           position.y1 + gap, heading, ha='center',
                           va='bottom', fontweight='bold', color=DARK,
                           fontsize=BASE_FONT - 1.0)
        # A heading wider than its own panel would reach into the next column's
        # y label.  The margin to the left of the first panel is empty at this
        # height, so let the heading spend that too, and shrink it only if the
        # whole column block is still not wide enough.
        limit = box(panels[row, 1].yaxis.label).x0 - 0.008
        fit(text, limit - 0.006)
        overhang = box(text).x1 - limit
        if overhang > 0:
            text.set_x(max(text.get_position()[0] - overhang,
                           0.5 * box(text).width + 0.004))
    top = max(box(bar.xaxis.label).y1 for bar in bars)
    for column, heading in enumerate(['Global importance'] + headers):
        position = panels[0, column].get_position()
        figure.text(0.5 * (position.x0 + position.x1), top + 0.006, heading,
                    ha='center', va='bottom', fontweight='bold', color=DARK,
                    fontsize=BASE_FONT + 2.0)
    # Freeze, or the engine reflows the axes on save and the headings, which are
    # placed in figure coordinates, drift off the columns they belong to.
    figure.set_layout_engine('none')
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('1peak', '2peak'), default='2peak')
    parser.add_argument('--samples', type=int, default=400,
                        help='test spectra to explain')
    parser.add_argument('--background', type=int, default=100)
    parser.add_argument('--from-cache', action='store_true',
                        help='redraw the stored attributions instead of '
                             're-running the explainer')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # build_model() chdir's into the case directory, so a relative --output
    # would land there rather than where the caller meant.
    stem = os.path.splitext(os.path.abspath(
        os.path.join(ROOT, args.output)
        if not os.path.isabs(args.output) else args.output))[0]
    cache = stem + '_values.npz'
    if args.from_cache:
        stored = np.load(cache)
        features, stack = stored['features'], stored['values']
    else:
        features, stack = compute(args.kind, args.samples, args.background)
        np.savez(cache, features=features, values=stack)

    figure = draw(args.kind, features, stack)
    for suffix, options in (('.tiff', {'dpi': FIG_DPI,
                                       'pil_kwargs': {'compression': 'tiff_lzw'}}),
                            ('.pdf', {}), ('.svg', {}),
                            ('.png', {'dpi': FIG_DPI})):
        # Safe now that the engine is frozen: this only crops the slack the
        # reserved strips left over, it does not reflow the panels.
        figure.savefig(stem + suffix, bbox_inches='tight', pad_inches=0.04,
                       facecolor='white', **options)
    print(f'Saved: {stem}.{{tiff,pdf,svg,png}}')


if __name__ == '__main__':
    main()
