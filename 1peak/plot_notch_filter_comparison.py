#!/usr/bin/env python3
"""Plot current single- and double-notch wavelength-error comparisons."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tandem-mpl-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
FIGURE_DIR = SCRIPT_DIR / "figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
DARK = "#202020"
SPINE = "#4B4B4B"
GRID = "#AFAFAF"
# 16 pt on an 11 in canvas, the ratio the rest of the paper's figures use; at
# the old 7.1 in the same type came out half again too large for the panels.
FIGSIZE = (11.0, 4.40)
FIG_DPI = 600
BAR_LABEL_SIZE = 14.0


matplotlib.rcParams.update(
    {
        "font.family": "serif",
        # Neither Times New Roman nor Tinos is installed here; without the
        # clone matplotlib falls back to DejaVu Serif and the figure stops
        # matching the others.
        "font.serif": ["Times New Roman", "Nimbus Roman", "Tinos",
                       "Liberation Serif", "DejaVu Serif", "serif"],
        "font.size": 16.0,
        "axes.labelsize": 17.0,
        "axes.titlesize": 17.0,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 16.0,
        "ytick.labelsize": 16.0,
        "legend.fontsize": 16.0,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Keep labels as real SVG text so PowerPoint and browser renderers
        # retain word spacing and the figure remains editable.
        "svg.fonttype": "none",
    }
)


def read_key_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for raw_line in stream:
            fields = raw_line.rstrip("\n").split("\t", 1)
            if len(fields) == 2:
                values[fields[0].strip()] = fields[1].strip()
    return values


def require_float(values: dict[str, str], key: str, path: Path) -> float:
    if key not in values:
        raise KeyError(f"Missing {key!r} in {path}")
    return float(values[key])


def load_current_metrics():
    single_cases = ("410", "530", "610")
    single_direct = []
    single_cvae = []
    provenance_rows = []

    for case in single_cases:
        direct_path = (
            ROOT_DIR
            / "1peak"
            / case
            / "tandem"
            / "tandem_pred_0_results.txt"
        )
        cvae_path = (
            ROOT_DIR / "1peak" / case / "cvae" / "cvae_results.txt"
        )
        direct_values = read_key_values(direct_path)
        cvae_values = read_key_values(cvae_path)
        direct_target = require_float(
            direct_values, "target_wavelength_nm", direct_path
        )
        direct_prediction = require_float(
            direct_values, "predicted_min_wavelength_nm", direct_path
        )
        cvae_target = require_float(
            cvae_values, "target_wavelength_nm", cvae_path
        )
        cvae_prediction = require_float(
            cvae_values, "tandem_min_wavelength_nm", cvae_path
        )
        direct_error = abs(direct_prediction - direct_target)
        cvae_error = abs(cvae_prediction - cvae_target)
        single_direct.append(direct_error)
        single_cvae.append(cvae_error)
        provenance_rows.extend(
            (
                {
                    "filter_type": "single",
                    "target_wavelength_or_pair_nm": case,
                    "method": "Direct Tandem",
                    "target_peak1_nm": direct_target,
                    "target_peak2_nm": np.nan,
                    "predicted_peak1_nm": direct_prediction,
                    "predicted_peak2_nm": np.nan,
                    "peak1_absolute_error_nm": direct_error,
                    "peak2_absolute_error_nm": np.nan,
                    "nominal_target_total_error_nm": direct_error,
                    "particle_radius_R_nm": require_float(
                        direct_values, "predicted_R", direct_path
                    ),
                    "n_host": require_float(
                        direct_values, "predicted_n_host", direct_path
                    ),
                    "n_host1": np.nan,
                    "n_host2": np.nan,
                    "source_file": str(direct_path.relative_to(ROOT_DIR)),
                },
                {
                    "filter_type": "single",
                    "target_wavelength_or_pair_nm": case,
                    "method": "CVAE-assisted Tandem",
                    "target_peak1_nm": cvae_target,
                    "target_peak2_nm": np.nan,
                    "predicted_peak1_nm": cvae_prediction,
                    "predicted_peak2_nm": np.nan,
                    "peak1_absolute_error_nm": cvae_error,
                    "peak2_absolute_error_nm": np.nan,
                    "nominal_target_total_error_nm": cvae_error,
                    "particle_radius_R_nm": require_float(
                        cvae_values, "predicted_R", cvae_path
                    ),
                    "n_host": require_float(
                        cvae_values, "predicted_n_host", cvae_path
                    ),
                    "n_host1": np.nan,
                    "n_host2": np.nan,
                    "source_file": str(cvae_path.relative_to(ROOT_DIR)),
                },
            )
        )

    # Same three pairs as the dual 3x3 figure: picked for a direct-route miss
    # the CVAE route closes, at round target wavelengths, and restricted to the
    # pairs that carry a full-wave run.
    dual_cases = ("500_550", "450_550", "500_650")
    dual_labels = tuple(case.replace("_", " / ") for case in dual_cases)
    dual_direct = []
    dual_cvae = []
    for case, label in zip(dual_cases, dual_labels):
        direct_path = (
            ROOT_DIR / "2peak" / case / "tandem" / "tandem_results.txt"
        )
        cvae_path = ROOT_DIR / "2peak" / case / "cvae" / "cvae_results.txt"
        direct_values = read_key_values(direct_path)
        cvae_values = read_key_values(cvae_path)

        direct_targets = (
            require_float(direct_values, "target_peak1_nm", direct_path),
            require_float(direct_values, "target_peak2_nm", direct_path),
        )
        cvae_targets = (
            require_float(cvae_values, "target_peak1_nm", cvae_path),
            require_float(cvae_values, "target_peak2_nm", cvae_path),
        )
        direct_predictions = (
            require_float(
                direct_values, "final_peak1_wavelength_nm", direct_path
            ),
            require_float(
                direct_values, "final_peak2_wavelength_nm", direct_path
            ),
        )
        cvae_predictions = (
            require_float(
                cvae_values, "final_peak1_wavelength_nm", cvae_path
            ),
            require_float(
                cvae_values, "final_peak2_wavelength_nm", cvae_path
            ),
        )
        direct_peak_errors = tuple(
            abs(prediction - target)
            for prediction, target in zip(direct_predictions, direct_targets)
        )
        cvae_peak_errors = tuple(
            abs(prediction - target)
            for prediction, target in zip(cvae_predictions, cvae_targets)
        )
        direct_total = float(np.sum(direct_peak_errors))
        cvae_total = float(np.sum(cvae_peak_errors))
        dual_direct.append(direct_total)
        dual_cvae.append(cvae_total)
        provenance_rows.extend(
            (
                {
                    "filter_type": "double",
                    "target_wavelength_or_pair_nm": label,
                    "method": "Direct Tandem",
                    "target_peak1_nm": direct_targets[0],
                    "target_peak2_nm": direct_targets[1],
                    "predicted_peak1_nm": direct_predictions[0],
                    "predicted_peak2_nm": direct_predictions[1],
                    "peak1_absolute_error_nm": direct_peak_errors[0],
                    "peak2_absolute_error_nm": direct_peak_errors[1],
                    "nominal_target_total_error_nm": direct_total,
                    "particle_radius_R_nm": require_float(
                        direct_values, "predicted_R", direct_path
                    ),
                    "n_host": np.nan,
                    "n_host1": require_float(
                        direct_values, "predicted_n_host1", direct_path
                    ),
                    "n_host2": require_float(
                        direct_values, "predicted_n_host2", direct_path
                    ),
                    "source_file": str(direct_path.relative_to(ROOT_DIR)),
                },
                {
                    "filter_type": "double",
                    "target_wavelength_or_pair_nm": label,
                    "method": "CVAE-assisted Tandem",
                    "target_peak1_nm": cvae_targets[0],
                    "target_peak2_nm": cvae_targets[1],
                    "predicted_peak1_nm": cvae_predictions[0],
                    "predicted_peak2_nm": cvae_predictions[1],
                    "peak1_absolute_error_nm": cvae_peak_errors[0],
                    "peak2_absolute_error_nm": cvae_peak_errors[1],
                    "nominal_target_total_error_nm": cvae_total,
                    "particle_radius_R_nm": require_float(
                        cvae_values, "predicted_R", cvae_path
                    ),
                    "n_host": np.nan,
                    "n_host1": require_float(
                        cvae_values, "predicted_n_host1", cvae_path
                    ),
                    "n_host2": require_float(
                        cvae_values, "predicted_n_host2", cvae_path
                    ),
                    "source_file": str(cvae_path.relative_to(ROOT_DIR)),
                },
            )
        )

    return {
        "single_labels": single_cases,
        "single_direct": np.asarray(single_direct),
        "single_cvae": np.asarray(single_cvae),
        "dual_labels": dual_labels,
        "dual_direct": np.asarray(dual_direct),
        "dual_cvae": np.asarray(dual_cvae),
        "provenance_rows": provenance_rows,
    }


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.45,
        linestyle="-",
        alpha=0.20,
        zorder=0,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=3.2,
        width=0.7,
        pad=3.0,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=1.8,
        width=0.5,
    )
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)


def value_label(value: float) -> str:
    return f"{value:.1f}"


def plot_grouped_bars(
    ax: plt.Axes,
    labels,
    direct_values: np.ndarray,
    cvae_values: np.ndarray,
) -> None:
    positions = np.arange(len(labels), dtype=float)
    width = 0.32
    direct_bars = ax.bar(
        positions - width / 2,
        direct_values,
        width,
        color=BLUE,
        edgecolor=SPINE,
        linewidth=0.45,
        zorder=3,
    )
    cvae_bars = ax.bar(
        positions + width / 2,
        cvae_values,
        width,
        color=ORANGE,
        edgecolor=SPINE,
        linewidth=0.45,
        zorder=3,
    )
    ax.set_xticks(positions, labels=labels)
    maximum = float(max(np.max(direct_values), np.max(cvae_values)))
    ax.set_ylim(0, maximum * 1.24)
    ax.bar_label(
        direct_bars,
        labels=[value_label(value) for value in direct_values],
        padding=2,
        fontsize=BAR_LABEL_SIZE,
        color=DARK,
    )
    ax.bar_label(
        cvae_bars,
        labels=[value_label(value) for value in cvae_values],
        padding=2,
        fontsize=BAR_LABEL_SIZE,
        color=DARK,
    )
    style_axis(ax)


def write_plot_data(metrics, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "notch_filter_comparison_data.tsv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        column_names = (
            "filter_type",
            "target_wavelength_or_pair_nm",
            "method",
            "target_peak1_nm",
            "target_peak2_nm",
            "predicted_peak1_nm",
            "predicted_peak2_nm",
            "peak1_absolute_error_nm",
            "peak2_absolute_error_nm",
            "nominal_target_total_error_nm",
            "particle_radius_R_nm",
            "n_host",
            "n_host1",
            "n_host2",
            "source_file",
        )
        writer.writerow(column_names)
        for row in metrics["provenance_rows"]:
            formatted_row = []
            for name in column_names:
                value = row[name]
                if isinstance(value, float):
                    formatted_row.append(
                        "" if np.isnan(value) else f"{value:.10f}"
                    )
                else:
                    formatted_row.append(value)
            writer.writerow(formatted_row)
    return output_path


def save_figure(fig: plt.Figure, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "notch_filter_comparison"
    tiff_path = stem.with_suffix(".tiff")
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    # The PNG was missing from this list, so the one on disk stayed at whatever
    # the script produced the last time somebody exported it by hand -- three
    # days older than the other three formats, and showing different numbers.
    png_path = stem.with_suffix(".png")
    fig.savefig(
        tiff_path,
        dpi=FIG_DPI,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
    )
    fig.savefig(
        svg_path,
        facecolor="white",
    )
    fig.savefig(
        png_path,
        dpi=FIG_DPI,
        facecolor="white",
    )
    return tiff_path, pdf_path, svg_path, png_path


def create_figure(metrics):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=False)
    fig.subplots_adjust(
        left=0.070,
        right=0.988,
        bottom=0.135,
        top=0.845,
        wspace=0.20,
    )

    plot_grouped_bars(
        axes[0],
        metrics["single_labels"],
        metrics["single_direct"],
        metrics["single_cvae"],
    )
    plot_grouped_bars(
        axes[1],
        metrics["dual_labels"],
        metrics["dual_direct"],
        metrics["dual_cvae"],
    )

    axes[0].set_title("Single-notch filter", pad=6, fontweight="bold", color=DARK)
    axes[1].set_title("Double-notch filter", pad=6, fontweight="bold", color=DARK)
    axes[0].set_xlabel("Nominal target wavelength (nm)")
    axes[1].set_xlabel("Nominal target wavelength pair (nm)")
    error_label = "Total nominal-target error (nm)"
    axes[0].set_ylabel(error_label)
    axes[1].set_ylabel(error_label)

    legend_handles = (
        Patch(facecolor=BLUE, edgecolor=SPINE, linewidth=0.45),
        Patch(facecolor=ORANGE, edgecolor=SPINE, linewidth=0.45),
    )
    fig.legend(
        legend_handles,
        ("Direct Tandem", "CVAE-assisted Tandem"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        handlelength=1.15,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0,
    )
    return fig


def main() -> None:
    metrics = load_current_metrics()
    data_path = write_plot_data(metrics, FIGURE_DIR)
    figure = create_figure(metrics)
    figure_paths = save_figure(figure, FIGURE_DIR)
    plt.close(figure)

    print("Single-notch Direct Tandem:", metrics["single_direct"])
    print("Single-notch CVAE-assisted:", metrics["single_cvae"])
    print("Double-notch Direct Tandem:", metrics["dual_direct"])
    print("Double-notch CVAE-assisted:", metrics["dual_cvae"])
    print(f"Saved: {data_path}")
    for path in figure_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
