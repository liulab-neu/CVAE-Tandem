#!/usr/bin/env python3
"""Create publication-style 3x3 spectral comparison figures."""

import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tandem-mpl-cache')

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from scipy.signal import find_peaks, peak_prominences


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BLUE = '#0072B2'
ORANGE = '#D55E00'
DARK = '#202020'
GRAY = '#5A5A5A'
FIGSIZE = (11.0, 8.8)
FIG_DPI = 600


matplotlib.rcParams.update({
    'font.family': 'serif',
    # Times New Roman is not installed here; Nimbus Roman is the Times-metric
    # clone that is, so it has to precede the DejaVu fallback or the figure
    # silently renders in a visibly different face from its neighbours.
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif', 'serif'],
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
    # Keep labels as real SVG text so PowerPoint and browser renderers retain
    # word spacing and the figure remains editable.
    'svg.fonttype': 'none',
})


# Hand-tuned label positions in data coordinates.  Each entry corresponds
# to the detected resonances from low to high wavelength.  The layouts are
# deliberately panel-specific so labels avoid axes, curves, and one another.
ANNOTATION_LAYOUTS = {
    # The two labels of the 410 nm panels stack vertically, so the upper one
    # sits far enough right that its leader passes above the lower label rather
    # than across the top of it.
    ('1peak', '410', 'tandem', 'model'): [(455, 0.07, 'left')],
    ('1peak', '410', 'tandem', 'reference'): [(505, 0.33, 'left')],
    ('1peak', '410', 'cvae', 'model'): [(455, 0.07, 'left')],
    ('1peak', '410', 'cvae', 'reference'): [(505, 0.33, 'left')],
    ('1peak', '530', 'tandem', 'model'): [(495, 0.11, 'right')],
    ('1peak', '530', 'tandem', 'reference'): [(560, 0.22, 'left')],
    ('1peak', '530', 'cvae', 'model'): [(500, 0.11, 'right')],
    ('1peak', '530', 'cvae', 'reference'): [(565, 0.22, 'left')],
    ('1peak', '610', 'tandem', 'model'): [(580, 0.17, 'right')],
    ('1peak', '610', 'tandem', 'reference'): [(650, 0.27, 'left')],
    ('1peak', '610', 'cvae', 'model'): [(575, 0.12, 'right')],
    ('1peak', '610', 'cvae', 'reference'): [(650, 0.24, 'left')],
    # Main-text single-notch cases, chosen from the 25-target sweep for a
    # visible direct-route miss that the CVAE route closes.
    ('1peak', '470', 'tandem', 'model'): [(520, 0.10, 'left')],
    ('1peak', '470', 'tandem', 'reference'): [(560, 0.28, 'left')],
    ('1peak', '470', 'cvae', 'model'): [(520, 0.10, 'left')],
    ('1peak', '470', 'cvae', 'reference'): [(560, 0.28, 'left')],
    ('1peak', '550', 'tandem', 'model'): [(500, 0.11, 'right')],
    ('1peak', '550', 'tandem', 'reference'): [(590, 0.24, 'left')],
    ('1peak', '550', 'cvae', 'model'): [(500, 0.11, 'right')],
    ('1peak', '550', 'cvae', 'reference'): [(590, 0.24, 'left')],
    ('1peak', '630', 'tandem', 'model'): [(590, 0.14, 'right')],
    ('1peak', '630', 'tandem', 'reference'): [(670, 0.26, 'left')],
    ('1peak', '630', 'cvae', 'model'): [(590, 0.14, 'right')],
    ('1peak', '630', 'cvae', 'reference'): [(670, 0.26, 'left')],
    ('2peak', '430_620', 'tandem', 'model'): [
        (450, 0.08, 'left'),
        (600, 0.13, 'right'),
    ],
    ('2peak', '430_620', 'tandem', 'reference'): [
        (490, 0.31, 'left'),
        (645, 0.28, 'left'),
    ],
    ('2peak', '430_620', 'cvae', 'model'): [
        (470, 0.08, 'left'),
        (620, 0.12, 'right'),
    ],
    ('2peak', '430_620', 'cvae', 'reference'): [
        (505, 0.28, 'left'),
        (655, 0.25, 'left'),
    ],
    ('2peak', '450_500', 'tandem', 'model'): [
        (435, 0.32, 'left'),
        (545, 0.09, 'left'),
    ],
    ('2peak', '450_500', 'tandem', 'reference'): [
        (415, 0.13, 'left'),
        (585, 0.30, 'left'),
    ],
    ('2peak', '450_500', 'cvae', 'model'): [
        (560, 0.32, 'left'),
        (660, 0.10, 'left'),
    ],
    ('2peak', '450_500', 'cvae', 'reference'): [
        (560, 0.48, 'left'),
        (660, 0.26, 'left'),
    ],
    ('2peak', '500_600', 'tandem', 'model'): [
        (470, 0.22, 'right'),
        (580, 0.20, 'right'),
    ],
    ('2peak', '500_600', 'tandem', 'reference'): [
        (525, 0.35, 'left'),
        (635, 0.34, 'left'),
    ],
    ('2peak', '500_600', 'cvae', 'model'): [
        (475, 0.12, 'right'),
        (575, 0.12, 'right'),
    ],
    ('2peak', '500_600', 'cvae', 'reference'): [
        (530, 0.28, 'left'),
        (635, 0.28, 'left'),
    ],
    # Both dips sit within 100 nm of the left edge, so the labels go in the
    # empty region to the right of 500 nm rather than beside the resonances.
    ('2peak', '415_480', 'tandem', 'model'): [
        (525, 0.30, 'left'),
        (525, 0.12, 'left'),
    ],
    ('2peak', '415_480', 'tandem', 'reference'): [
        (655, 0.30, 'left'),
        (655, 0.12, 'left'),
    ],
    ('2peak', '415_480', 'cvae', 'model'): [
        (525, 0.30, 'left'),
        (525, 0.12, 'left'),
    ],
    ('2peak', '415_480', 'cvae', 'reference'): [
        (655, 0.30, 'left'),
        (655, 0.12, 'left'),
    ],
    ('2peak', '450_575', 'tandem', 'model'): [
        (455, 0.10, 'right'),
        (595, 0.12, 'left'),
    ],
    ('2peak', '450_575', 'tandem', 'reference'): [
        (430, 0.32, 'left'),
        (620, 0.30, 'left'),
    ],
    ('2peak', '450_575', 'cvae', 'model'): [
        (455, 0.10, 'right'),
        (595, 0.12, 'left'),
    ],
    ('2peak', '450_575', 'cvae', 'reference'): [
        (430, 0.32, 'left'),
        (620, 0.30, 'left'),
    ],
    ('2peak', '475_630', 'tandem', 'model'): [
        (480, 0.10, 'right'),
        (650, 0.12, 'left'),
    ],
    ('2peak', '475_630', 'tandem', 'reference'): [
        (455, 0.32, 'left'),
        (675, 0.30, 'left'),
    ],
    ('2peak', '475_630', 'cvae', 'model'): [
        (480, 0.10, 'right'),
        (665, 0.10, 'left'),
    ],
    ('2peak', '475_630', 'cvae', 'reference'): [
        (455, 0.34, 'left'),
        (665, 0.34, 'left'),
    ],
    # 435/490: the dips are 55 nm apart near the left edge, so the labels
    # stack just to their right instead of reaching across the panel.
    ('2peak', '435_490', 'tandem', 'model'): [
        (455, 0.30, 'left'),
        (515, 0.12, 'left'),
    ],
    ('2peak', '435_490', 'tandem', 'reference'): [
        (560, 0.42, 'left'),
        (620, 0.24, 'left'),
    ],
    ('2peak', '435_490', 'cvae', 'model'): [
        (455, 0.30, 'left'),
        (515, 0.12, 'left'),
    ],
    ('2peak', '435_490', 'cvae', 'reference'): [
        (560, 0.42, 'left'),
        (620, 0.24, 'left'),
    ],
    # 405/580: the dips are far apart, so each label sits beside its own dip.
    ('2peak', '405_580', 'tandem', 'model'): [
        (425, 0.12, 'left'),
        (600, 0.12, 'left'),
    ],
    ('2peak', '405_580', 'tandem', 'reference'): [
        (425, 0.32, 'left'),
        (620, 0.32, 'left'),
    ],
    ('2peak', '405_580', 'cvae', 'model'): [
        (425, 0.12, 'left'),
        (600, 0.12, 'left'),
    ],
    ('2peak', '405_580', 'cvae', 'reference'): [
        (425, 0.32, 'left'),
        (620, 0.32, 'left'),
    ],
    # 550/625: both dips sit right of centre, so labels go left of 520 nm.
    ('2peak', '550_625', 'tandem', 'model'): [
        (505, 0.12, 'right'),
        (665, 0.12, 'left'),
    ],
    ('2peak', '550_625', 'tandem', 'reference'): [
        (505, 0.32, 'right'),
        (665, 0.32, 'left'),
    ],
    ('2peak', '550_625', 'cvae', 'model'): [
        (505, 0.12, 'right'),
        (665, 0.12, 'left'),
    ],
    ('2peak', '550_625', 'cvae', 'reference'): [
        (505, 0.32, 'right'),
        (665, 0.32, 'left'),
    ],
    # 475/540: the dips are 65 nm apart mid-band, so the labels stack to their
    # right where the spectrum has recovered.
    ('2peak', '475_540', 'tandem', 'model'): [
        (585, 0.30, 'left'),
        (585, 0.12, 'left'),
    ],
    ('2peak', '475_540', 'tandem', 'reference'): [
        (690, 0.30, 'left'),
        (690, 0.12, 'left'),
    ],
    ('2peak', '475_540', 'cvae', 'model'): [
        (585, 0.30, 'left'),
        (585, 0.12, 'left'),
    ],
    ('2peak', '475_540', 'cvae', 'reference'): [
        (690, 0.30, 'left'),
        (690, 0.12, 'left'),
    ],
    # 405/590: widely separated, so each label sits beside its own dip.
    ('2peak', '405_590', 'tandem', 'model'): [
        (425, 0.12, 'left'),
        (615, 0.12, 'left'),
    ],
    ('2peak', '405_590', 'tandem', 'reference'): [
        (425, 0.32, 'left'),
        (640, 0.32, 'left'),
    ],
    ('2peak', '405_590', 'cvae', 'model'): [
        (425, 0.12, 'left'),
        (615, 0.12, 'left'),
    ],
    ('2peak', '405_590', 'cvae', 'reference'): [
        (425, 0.32, 'left'),
        (640, 0.32, 'left'),
    ],
    # 500/550: the dips are only 50 nm apart mid-band, so all four labels stack
    # to their right where the spectrum has recovered.
    ('2peak', '500_550', 'tandem', 'model'): [
        (592, 0.30, 'left'),
        (592, 0.12, 'left'),
    ],
    ('2peak', '500_550', 'tandem', 'reference'): [
        (697, 0.30, 'left'),
        (697, 0.12, 'left'),
    ],
    ('2peak', '500_550', 'cvae', 'model'): [
        (592, 0.30, 'left'),
        (592, 0.12, 'left'),
    ],
    ('2peak', '500_550', 'cvae', 'reference'): [
        (697, 0.30, 'left'),
        (697, 0.12, 'left'),
    ],
    # 450/550: 100 nm apart, so each dip keeps its own pair of labels -- the
    # short-wavelength one to its left, in the band below the rising edge.
    ('2peak', '450_550', 'tandem', 'model'): [
        (505, 0.30, 'left'),
        (612, 0.12, 'left'),
    ],
    ('2peak', '450_550', 'tandem', 'reference'): [
        (400, 0.22, 'left'),
        (700, 0.28, 'left'),
    ],
    ('2peak', '450_550', 'cvae', 'model'): [
        (505, 0.30, 'left'),
        (612, 0.12, 'left'),
    ],
    ('2peak', '450_550', 'cvae', 'reference'): [
        (400, 0.22, 'left'),
        (700, 0.28, 'left'),
    ],
    # 500/650: widely separated with a flat plateau between them, so both
    # labels sit to the left of their own dip inside that plateau.
    ('2peak', '500_650', 'tandem', 'model'): [
        (462, 0.14, 'right'),
        (612, 0.12, 'right'),
    ],
    ('2peak', '500_650', 'tandem', 'reference'): [
        (437, 0.34, 'right'),
        (587, 0.32, 'right'),
    ],
    ('2peak', '500_650', 'cvae', 'model'): [
        (462, 0.14, 'right'),
        (612, 0.12, 'right'),
    ],
    ('2peak', '500_650', 'cvae', 'reference'): [
        (437, 0.34, 'right'),
        (587, 0.32, 'right'),
    ],
}


def load_columns(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Required plotting data not found: {path}')
    with open(path, 'r', encoding='utf-8') as data_file:
        names = data_file.readline().strip().split('\t')
    values = np.loadtxt(path, delimiter='\t', skiprows=1, ndmin=2)
    if values.shape[1] != len(names):
        raise ValueError(
            f'Header/data column mismatch in {path}: '
            f'{len(names)} names, {values.shape[1]} columns.'
        )
    return {name: values[:, index] for index, name in enumerate(names)}


def ideal_reference_for_grid(base_dir, wavelengths):
    """Load the canonical ideal target on the case's actual CST grid."""
    columns = load_columns(os.path.join(
        base_dir, 'cvae', 'cvae_data.txt'
    ))
    reference_wavelengths = columns['wavelength_nm']
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.shape != reference_wavelengths.shape or not np.allclose(
        wavelengths,
        reference_wavelengths,
        rtol=0.0,
        atol=5.0e-4,
    ):
        raise ValueError(
            f'Ideal-target wavelength grid does not match the response grid '
            f'in {base_dir}.'
        )
    return columns['ideal_target_transmittance']


def panel_data(peak_kind, case, method):
    base_dir = os.path.join(ROOT_DIR, peak_kind, case)
    if peak_kind == '1peak':
        if method == 'tandem':
            columns = load_columns(os.path.join(
                base_dir, 'tandem', 'tandem_pred_0_data.txt'
            ))
            wavelengths = columns['wavelength_nm']
            return (
                wavelengths,
                ideal_reference_for_grid(base_dir, wavelengths),
                columns['predicted_transmittance'],
            )
        if method == 'cvae':
            columns = load_columns(os.path.join(
                base_dir, 'cvae', 'cvae_data.txt'
            ))
            return (
                columns['wavelength_nm'],
                columns['cvae_target_transmittance'],
                columns['tandem_response_transmittance'],
            )
        columns = load_columns(os.path.join(
            base_dir, 'cst', 'cst_data.txt'
        ))
        return (
            columns['wavelength_nm'],
            columns['final_result_transmittance'],
            columns['cst_average_transmittance'],
        )

    if method == 'tandem':
        columns = load_columns(os.path.join(
            base_dir, 'tandem', 'tandem_data.txt'
        ))
        wavelengths = columns['wavelength_nm']
        return (
            wavelengths,
            ideal_reference_for_grid(base_dir, wavelengths),
            columns['tandem_response_transmittance'],
        )
    if method == 'cvae':
        columns = load_columns(os.path.join(
            base_dir, 'cvae', 'cvae_data.txt'
        ))
        return (
            columns['wavelength_nm'],
            columns['cvae_target_transmittance'],
            columns['tandem_response_transmittance'],
        )
    columns = load_columns(os.path.join(
        base_dir, 'cst', 'cst_data.txt'
    ))
    # The double-notch column used to be the nearest dataset sample, which was
    # a stand-in while CST could not be launched. It is now the mean of the
    # 0/30/60 degree runs of the selected design, matching both the single-notch
    # column and the angle average the networks are trained on.
    return (
        columns['wavelength_nm'],
        columns['final_network_transmittance'],
        columns['cst_average_transmittance'],
    )


def resonance_indices(wavelengths, response, count):
    wavelengths = np.asarray(wavelengths, dtype=float)
    response = np.asarray(response, dtype=float)
    if count == 1:
        return np.array([int(np.argmin(response))], dtype=int)

    spacing = float(np.median(np.diff(wavelengths)))
    minimum_distance = max(1, int(round(20.0 / spacing)))
    candidates, _ = find_peaks(
        -response,
        prominence=0.01,
        distance=minimum_distance,
    )
    if len(candidates) < count:
        candidates, _ = find_peaks(-response, distance=minimum_distance)
    if len(candidates) < count:
        return np.argsort(response)[:count]
    prominences = peak_prominences(-response, candidates)[0]
    selected = candidates[np.argsort(prominences)[-count:]]
    return np.sort(selected)


def annotate_resonances(
    ax,
    wavelengths,
    response,
    count,
    color,
    positions,
    display_values=None,
):
    indices = resonance_indices(wavelengths, response, count)
    if len(positions) != len(indices):
        raise ValueError(
            f'Expected {len(indices)} annotation positions, got '
            f'{len(positions)}.'
        )
    if display_values is not None and len(display_values) != len(indices):
        raise ValueError(
            f'Expected {len(indices)} annotation values, got '
            f'{len(display_values)}.'
        )
    for peak_index, sample_index in enumerate(indices):
        x_value = float(wavelengths[sample_index])
        y_value = float(response[sample_index])
        label = (
            f'{x_value:.1f}'
            if display_values is None
            else f'{float(display_values[peak_index]):g}'
        )
        ax.scatter(
            x_value,
            y_value,
            s=16,
            color=color,
            edgecolors='white',
            linewidths=0.55,
            zorder=5,
        )
        label_x, label_y, horizontal_alignment = positions[peak_index]
        ax.annotate(
            label,
            xy=(x_value, y_value),
            xytext=(label_x, label_y),
            textcoords='data',
            ha=horizontal_alignment,
            va='center',
            color=GRAY,
            fontsize=16.0,
            fontweight='medium',
            bbox=dict(
                boxstyle='square,pad=0.04',
                facecolor='white',
                edgecolor='none',
                alpha=0.96,
            ),
            arrowprops=dict(
                arrowstyle='-',
                color=GRAY,
                linewidth=0.55,
                shrinkA=1.5,
                shrinkB=2.0,
            ),
            annotation_clip=True,
            zorder=6,
        )


def style_axis(ax):
    ax.set_xlim(380, 800)
    ax.set_ylim(-0.025, 1.025)
    ax.xaxis.set_major_locator(FixedLocator([400, 500, 600, 700, 800]))
    ax.yaxis.set_major_locator(
        FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        top=True,
        right=True,
        length=3.0,
        width=0.65,
        pad=2.0,
    )
    ax.tick_params(
        axis='both',
        which='minor',
        direction='in',
        top=True,
        right=True,
        length=1.7,
        width=0.5,
    )
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_color('#4B4B4B')
        spine.set_linewidth(0.8)
    ax.tick_params(labelbottom=True, labelleft=True)


def create_figure(peak_kind):
    if peak_kind == '1peak':
        # 410/530/610 rather than 470/550/630: only these three carry the CST
        # simulation the third column needs.
        cases = ('410', '530', '610')
        row_labels = ('410 nm', '530 nm', '610 nm')
        output_stem = 'single_peak_spectral_comparison_3x3'
        resonance_count = 1
    else:
        # Picked from the 30-pair sweep for a direct-route miss the CVAE route
        # closes, at round target wavelengths, and restricted to the pairs that
        # carry the full-wave run the third column needs.
        cases = ('500_550', '450_550', '500_650')
        row_labels = ('500 / 550 nm', '450 / 550 nm', '500 / 650 nm')
        output_stem = 'dual_peak_spectral_comparison_3x3'
        resonance_count = 2

    methods = ('tandem', 'cvae', 'cst')
    column_titles = (
        'Direct Tandem',
        'CVAE-assisted Tandem',
        'CST validation',
    )
    reference_labels = ('Ideal target spectrum', 'CVAE generated spectrum',
                        'CST simulated spectrum')

    fig, axes = plt.subplots(
        3,
        3,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
    )
    for row, (case, row_label) in enumerate(zip(cases, row_labels)):
        requested_resonances = tuple(
            float(value) for value in case.split('_')
        )
        for column, method in enumerate(methods):
            ax = axes[row, column]
            wavelengths, reference, response = panel_data(
                peak_kind, case, method
            )
            if method == 'cst':
                first_curve = reference
                second_curve = response
            else:
                first_curve = response
                second_curve = reference

            ax.plot(
                wavelengths,
                first_curve,
                color=BLUE,
                linewidth=1.35,
                solid_capstyle='round',
                label='Model response',
                zorder=3,
            )
            ax.plot(
                wavelengths,
                second_curve,
                color=ORANGE,
                linewidth=1.15,
                linestyle=(0, (4, 2.2)),
                dash_capstyle='round',
                label=reference_labels[column],
                zorder=2,
            )
            style_axis(ax)
            if method != 'cst':
                model_layout = ANNOTATION_LAYOUTS[
                    (peak_kind, case, method, 'model')
                ]
                reference_layout = ANNOTATION_LAYOUTS[
                    (peak_kind, case, method, 'reference')
                ]
                annotate_resonances(
                    ax,
                    wavelengths,
                    second_curve,
                    resonance_count,
                    ORANGE,
                    reference_layout,
                    # The Direct-Tandem reference is the analytic ideal target,
                    # so it carries the requested design wavelengths.  The CVAE
                    # reference is a generated, physically realisable spectrum,
                    # so it carries its own resonance positions.
                    display_values=(
                        requested_resonances if method == 'tandem' else None
                    ),
                )
                annotate_resonances(
                    ax,
                    wavelengths,
                    first_curve,
                    resonance_count,
                    BLUE,
                    model_layout,
                )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.875,
        wspace=0.24,
        hspace=0.22,
    )
    fig.supxlabel(
        'Wavelength (nm)',
        x=0.53,
        y=0.012,
        fontsize=16.0,
        color=DARK,
    )
    fig.supylabel(
        'Transmittance',
        x=0.012,
        y=0.475,
        fontsize=16.0,
        color=DARK,
    )

    column_legend_labels = (
        ('Ideal target spectrum', 'Direct Tandem response'),
        ('CVAE generated spectrum', 'CVAE-assisted response'),
        ('CST simulated spectrum', 'Final network response'),
    )
    orange_handle = Line2D(
        [], [], color=ORANGE, linewidth=1.15, linestyle=(0, (4, 2.2))
    )
    blue_handle = Line2D([], [], color=BLUE, linewidth=1.35)
    for column, labels in enumerate(column_legend_labels):
        position = axes[0, column].get_position()
        center_x = 0.5 * (position.x0 + position.x1)
        fig.text(
            center_x,
            0.975,
            column_titles[column],
            ha='center',
            va='top',
            fontsize=16.0,
            fontweight='bold',
            color=DARK,
        )
        fig.legend(
            (orange_handle, blue_handle),
            labels,
            loc='upper center',
            bbox_to_anchor=(center_x, 0.945),
            ncol=1,
            frameon=False,
            # The legend is centred on its column, so its width has to stay
            # inside the column: the spelled-out labels are wide enough that a
            # full-length handle at the body size pushes the entry out past the
            # panel's left edge.
            handlelength=1.5,
            handletextpad=0.4,
            labelspacing=0.20,
            borderaxespad=0,
            fontsize=14.0,
        )

    output_dir = os.path.join(ROOT_DIR, peak_kind, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    tiff_path = os.path.join(output_dir, output_stem + '.tiff')
    pdf_path = os.path.join(output_dir, output_stem + '.pdf')
    svg_path = os.path.join(output_dir, output_stem + '.svg')
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
    # The PNG was missing from this list, so the copy on disk kept whatever
    # labels it was last exported with while the other three formats moved on.
    fig.savefig(
        os.path.join(output_dir, output_stem + '.png'),
        dpi=FIG_DPI,
        bbox_inches='tight',
        pad_inches=0.04,
        facecolor='white',
    )
    plt.close(fig)
    return (tiff_path, pdf_path, svg_path,
            os.path.join(output_dir, output_stem + '.png'))


def main():
    for peak_kind in ('1peak', '2peak'):
        for path in create_figure(peak_kind):
            print(f'Saved: {path}')


if __name__ == '__main__':
    main()
