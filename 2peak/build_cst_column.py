#!/usr/bin/env python3
"""Write the CST-validation column of a double-notch case from real simulations.

``plot_cst_dual_peak.py`` fills that column with the nearest sample in the
dataset, which is what had to be done while CST could not start. With CST
working, this builds the column from the actual full-wave run instead.

The networks are trained on the mean of the 0, 30 and 60 degree spectra, so the
validation column is the same mean of the three exported angles -- comparing a
normal-incidence simulation against an angle-averaged prediction would be
measuring the averaging, not the design.

Writes ``<case>/cst/cst_data.txt`` with ``wavelength_nm``,
``final_network_transmittance`` and ``cst_average_transmittance``, matching the
single-notch naming, plus a short ``cst_results.txt`` summary.
"""

import argparse
import os

import numpy as np
from scipy.signal import find_peaks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', help='Case directory name, e.g. 500_550')
    parser.add_argument(
        '--angles',
        type=float,
        nargs='+',
        default=(0.0, 30.0, 60.0),
        help='Angles that were exported, in the order the run used them.',
    )
    return parser.parse_args(argv)


def load_angle(path):
    """One exported spectrum as (wavelength, transmittance).

    ``export1D`` writes ``x, Re, Im`` and CST puts the magnitude in the Re
    column for this result, so the magnitude is |Re + i Im| either way.
    """
    table = np.loadtxt(path)
    if table.shape[1] < 3:
        raise ValueError(f'Unexpected export shape {table.shape} in {path}')
    return table[:, 0], np.hypot(table[:, 1], table[:, 2])


def dips(wavelengths, spectrum, count=2):
    peaks, _ = find_peaks(-spectrum, prominence=0.05)
    if len(peaks) < count:
        return np.array([]), np.array([])
    chosen = np.sort(peaks[np.argsort(spectrum[peaks])[:count]])
    return wavelengths[chosen], spectrum[chosen]


def main(argv=None):
    args = parse_args(argv)
    case_dir = os.path.join(BASE_DIR, args.case)
    cst_dir = os.path.join(case_dir, 'cst')

    grid = None
    spectra = []
    for index, angle in enumerate(args.angles, start=1):
        path = os.path.join(cst_dir, f'{index:05d}.b')
        if not os.path.isfile(path):
            raise SystemExit(f'Missing CST export for {angle:g} deg: {path}')
        wavelengths, transmittance = load_angle(path)
        if grid is None:
            grid = wavelengths
        elif not np.allclose(wavelengths, grid, atol=1e-3):
            transmittance = np.interp(grid, wavelengths, transmittance)
        spectra.append(transmittance)
    averaged = np.mean(spectra, axis=0)

    # The network's own prediction for the selected design.
    network = np.loadtxt(
        os.path.join(case_dir, 'cvae', 'cvae_data.txt'), skiprows=1)
    net_grid, net_response = network[:, 0], network[:, 3]
    if not np.allclose(net_grid, grid, atol=1e-3):
        averaged = np.interp(net_grid, grid, averaged)
        grid = net_grid

    np.savetxt(
        os.path.join(cst_dir, 'cst_data.txt'),
        np.column_stack((grid, net_response, averaged)),
        delimiter='\t',
        fmt='%.10e',
        header='wavelength_nm\tfinal_network_transmittance\t'
               'cst_average_transmittance',
        comments='',
    )

    net_dips, net_depths = dips(grid, net_response)
    cst_dips, cst_depths = dips(grid, averaged)
    rmse = float(np.sqrt(np.mean((net_response - averaged) ** 2)))
    with open(os.path.join(cst_dir, 'cst_results.txt'), 'w',
              encoding='utf-8') as handle:
        handle.write('comparison\tfinal_network_vs_angle_averaged_cst\n')
        handle.write('angles_deg\t'
                     + ','.join(f'{a:g}' for a in args.angles) + '\n')
        handle.write(f'spectrum_rmse\t{rmse:.10e}\n')
        for label, positions, depths in (('network', net_dips, net_depths),
                                         ('cst', cst_dips, cst_depths)):
            for order, (position, depth) in enumerate(
                    zip(positions, depths), start=1):
                handle.write(f'{label}_notch{order}_wavelength_nm\t'
                             f'{position:.10f}\n')
                handle.write(f'{label}_notch{order}_transmittance\t'
                             f'{depth:.10e}\n')
        if len(net_dips) == len(cst_dips) == 2:
            for order in range(2):
                handle.write(f'notch{order + 1}_shift_nm\t'
                             f'{cst_dips[order] - net_dips[order]:.10f}\n')

    print(f'{args.case}: rmse {rmse:.3e}', end='')
    if len(net_dips) == len(cst_dips) == 2:
        print(f'  network {np.round(net_dips, 2)}  cst {np.round(cst_dips, 2)}'
              f'  shift {np.round(cst_dips - net_dips, 2)} nm')
    else:
        print(f'  network dips {len(net_dips)}, cst dips {len(cst_dips)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
