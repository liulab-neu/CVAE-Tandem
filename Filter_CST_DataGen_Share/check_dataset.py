"""Validate a generated 2peak dataset directory against the grid it claims.

Checks every sample, not a sample of them:

* the id -> (theta, R, n_host1, n_host2) mapping matches the grid the sweep
  was told to produce, so nothing is mislabelled;
* one shared, strictly increasing 1001-point wavelength grid;
* spectra are finite and transmittance stays within [0, 1];
* ids are complete with no gaps or duplicates.

Grid flags must match the ones the sweep ran with, since the ids are a pure
function of the grid. Uses os.scandir rather than a glob because a full sweep
has tens of thousands of files, which is enough to hit ARG_MAX in a shell.
"""

import argparse
import os
import sys

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Dataset sample directory.')
    parser.add_argument('--r-start', type=float, default=2.5)
    parser.add_argument('--r-stop', type=float, default=6.5)
    parser.add_argument('--r-step', type=float, default=0.1)
    parser.add_argument('--n-start', type=float, default=1.5)
    parser.add_argument('--n-stop', type=float, default=3.05)
    parser.add_argument('--n-step', type=float, default=0.1)
    parser.add_argument('--theta-list', default='0 30 60')
    parser.add_argument(
        '--max-report',
        type=int,
        default=5,
        help='How many examples of each problem to print.',
    )
    return parser.parse_args(argv)


def axis(start, stop, step):
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    values = np.round(start + step * np.arange(count), 10)
    if values[-1] < stop - 1e-9:
        values = np.append(values, round(stop, 10))
    return values


def build_grid(args):
    r_values = axis(args.r_start, args.r_stop, args.r_step)
    n_values = axis(args.n_start, args.n_stop, args.n_step)
    thetas = [float(v) for v in args.theta_list.replace(',', ' ').split()]
    return [
        (theta, float(r), float(n1), float(n2))
        for theta in thetas
        for r in r_values
        for n1 in n_values
        for n2 in n_values
    ], r_values, n_values, thetas


def main(argv=None):
    args = parse_args(argv)
    source = os.path.abspath(args.source)
    grid, r_values, n_values, thetas = build_grid(args)

    ids = sorted(
        int(entry.name[:5])
        for entry in os.scandir(source)
        if entry.name.endswith('.npz') and entry.name[:5].isdigit()
    )
    print(f'Source       : {source}')
    print(f'Grid         : {len(thetas)} theta x {len(r_values)} R x '
          f'{len(n_values)}^2 n = {len(grid)}')
    print(f'Samples found: {len(ids)}')

    problems = {name: [] for name in (
        'duplicate id', 'id out of range', 'missing id', 'unreadable',
        'parameter mismatch', 'wavelength mismatch', 'bad shape',
        'non-finite', 'transmittance out of range',
    )}

    seen = set()
    for sample_id in ids:
        if sample_id in seen:
            problems['duplicate id'].append(sample_id)
        seen.add(sample_id)
        if not 1 <= sample_id <= len(grid):
            problems['id out of range'].append(sample_id)
    problems['missing id'] = sorted(set(range(1, len(grid) + 1)) - seen)

    reference_grid = None
    for position, sample_id in enumerate(ids):
        if not 1 <= sample_id <= len(grid):
            continue
        path = os.path.join(source, f'{sample_id:05d}.npz')
        try:
            with np.load(path) as sample:
                theta = float(sample['theta_deg'])
                values = (
                    theta,
                    float(sample['R']),
                    float(sample['n_host1']),
                    float(sample['n_host2']),
                )
                wavelengths = sample['wavelength_nm']
                transmission = np.hypot(sample['t_real'], sample['t_imag'])
        except Exception as error:
            problems['unreadable'].append(f'{sample_id}: {error}')
            continue

        if not np.allclose(values, grid[sample_id - 1], atol=1e-5):
            problems['parameter mismatch'].append(
                f'{sample_id}: got {values} expected {grid[sample_id - 1]}'
            )
        if wavelengths.shape != (1001,) or transmission.shape != (1001,):
            problems['bad shape'].append(
                f'{sample_id}: {wavelengths.shape} / {transmission.shape}'
            )
            continue
        if reference_grid is None:
            reference_grid = wavelengths
            if not np.all(np.diff(reference_grid) > 0):
                problems['wavelength mismatch'].append(
                    'reference grid is not strictly increasing'
                )
        elif not np.allclose(wavelengths, reference_grid, atol=1e-3):
            problems['wavelength mismatch'].append(str(sample_id))
        if not np.all(np.isfinite(transmission)):
            problems['non-finite'].append(str(sample_id))
        if transmission.min() < -1e-6 or transmission.max() > 1.0001:
            problems['transmittance out of range'].append(
                f'{sample_id}: {transmission.min():.4f}..'
                f'{transmission.max():.4f}'
            )

        if position % 5000 == 0:
            print(f'  checked {position}/{len(ids)}', flush=True)

    print()
    failed = False
    for name, found in problems.items():
        print(f'{name:<28} {len(found)}')
        if found:
            failed = True
            for item in found[:args.max_report]:
                print(f'    {item}')
            if len(found) > args.max_report:
                print(f'    ... {len(found) - args.max_report} more')

    if reference_grid is not None:
        print(f'\nwavelength grid  {reference_grid[0]:.2f} .. '
              f'{reference_grid[-1]:.2f} nm, {len(reference_grid)} points, '
              f'step {np.mean(np.diff(reference_grid)):.3f} nm')
    print('\nRESULT: ' + ('PROBLEMS FOUND' if failed else 'all checks passed'))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
