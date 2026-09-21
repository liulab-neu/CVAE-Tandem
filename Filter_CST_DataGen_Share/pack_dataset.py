"""Pack a CST dataset directory into one compressed ``.npz``.

Handles both layouts:

* the archived text layout (``<id>.a`` parameters, ``<id>.b`` transmission,
  ``<id>.c`` reflection, plus preview PNGs), and
* the per-sample ``<id>.npz`` layout written by ``gen_2peak_dataset.py``.

The result is a single array bundle -- ``ids``, ``R``, ``n_host1``,
``n_host2``, ``wavelength_nm``, ``t_real``, ``t_imag`` and, when present,
``r_real`` / ``r_imag``. Spectra are float32, which is ample for values in
[0, 1] and about 10x smaller than the text export.

The source directory is never modified. Verify the output, then delete or
archive the originals yourself.
"""

import argparse
import os
import re
import sys

import numpy as np


SAMPLE_PATTERN = re.compile(r'^(\d{5})\.(a|b|npz)$')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'source',
        help='Dataset sample directory, e.g. 2peak/data/cstdata_10496/data',
    )
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help='Output .npz; defaults to <source parent>.npz next to it.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Only pack the first N samples; useful for a quick check.',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Re-read a few packed samples and compare against the source.',
    )
    parser.add_argument(
        '--average-theta',
        action='store_true',
        help=(
            'Collapse the incidence angles into one angle-averaged spectrum '
            'per structure. theta is the outermost grid axis, so the samples '
            'form N_theta equal, identically ordered blocks and the average '
            'is a plain mean across them. The bundle then carries one row per '
            '(R, n_host1, n_host2) plus an averaged_theta_deg list.'
        ),
    )
    return parser.parse_args(argv)


def load_parameters(path):
    """Read an ``<id>.a`` table, evaluating its expression-valued entries."""
    values = {}
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            name, expression = line.split(':', 1)
            values[name.strip()] = eval(expression.strip(), None, values)
    return values


def discover_samples(source):
    """Map sample id -> 'text' or 'npz', preferring npz when both exist."""
    samples = {}
    for entry in sorted(os.listdir(source)):
        match = SAMPLE_PATTERN.match(entry)
        if not match:
            continue
        sample_id, kind = match.group(1), match.group(2)
        if kind == 'npz':
            samples[sample_id] = 'npz'
        elif sample_id not in samples:
            samples[sample_id] = 'text'
    return samples


def read_text_sample(source, sample_id):
    transmission = np.loadtxt(os.path.join(source, f'{sample_id}.b'))
    parameters = load_parameters(os.path.join(source, f'{sample_id}.a'))
    record = {
        'wavelength_nm': transmission[:, 0],
        't_real': transmission[:, 1],
        't_imag': transmission[:, 2],
        'R': parameters['R'],
        'n_host1': parameters['n_host1'],
        'n_host2': parameters['n_host2'],
    }
    if 'theta' in parameters:
        record['theta_deg'] = parameters['theta']
    reflection_path = os.path.join(source, f'{sample_id}.c')
    if os.path.isfile(reflection_path):
        reflection = np.loadtxt(reflection_path)
        record['r_real'] = reflection[:, 1]
        record['r_imag'] = reflection[:, 2]
    return record


def read_npz_sample(source, sample_id):
    with np.load(os.path.join(source, f'{sample_id}.npz')) as bundle:
        return {name: bundle[name] for name in bundle.files}


def average_over_theta(bundle, scalar_names, spectra):
    """Collapse the theta blocks of a packed bundle into their mean.

    theta is the outermost grid axis, so the samples are N_theta equal blocks
    that repeat the same (R, n_host1, n_host2) sequence. That is asserted
    here rather than assumed: the structure columns of every block must match
    the first one, otherwise the layout is not what this function expects and
    averaging would silently mix different structures.
    """
    if 'theta_deg' not in bundle:
        raise SystemExit(
            '--average-theta needs a theta_deg column; this dataset was '
            'generated without an angle axis.'
        )

    angles = np.unique(bundle['theta_deg'])
    count = len(bundle['ids'])
    if count % len(angles):
        raise SystemExit(
            f'{count} samples do not split evenly into {len(angles)} angle '
            'blocks; the sweep is probably incomplete.'
        )
    block = count // len(angles)

    reference = np.stack([bundle[name][:block] for name in scalar_names])
    for index in range(1, len(angles)):
        chunk = slice(index * block, (index + 1) * block)
        other = np.stack([bundle[name][chunk] for name in scalar_names])
        if not np.array_equal(reference, other):
            raise SystemExit(
                f'Angle block {index} does not repeat the same structures as '
                'block 0; refusing to average.'
            )

    averaged = {
        'ids': bundle['ids'][:block],
        'wavelength_nm': bundle['wavelength_nm'],
        'averaged_theta_deg': np.asarray(
            sorted(float(a) for a in angles),
            dtype=np.float32,
        ),
    }
    for name in scalar_names:
        averaged[name] = bundle[name][:block]
    for name in spectra:
        stacked = bundle[name].reshape(len(angles), block, -1)
        averaged[name] = stacked.mean(axis=0).astype(np.float32)

    print(
        f'Averaged {len(angles)} angles '
        f'({", ".join(f"{a:g}" for a in averaged["averaged_theta_deg"])} deg) '
        f'-> {block} structures'
    )
    return averaged


def main(argv=None):
    args = parse_args(argv)
    source = os.path.abspath(args.source)
    if not os.path.isdir(source):
        raise SystemExit(f'Not a directory: {source}')

    samples = discover_samples(source)
    if not samples:
        raise SystemExit(f'No <id>.a / <id>.npz samples found in {source}')

    sample_ids = sorted(samples)
    if args.limit:
        sample_ids = sample_ids[:args.limit]

    output = args.output
    if output is None:
        parent = os.path.dirname(source.rstrip(os.sep))
        output = parent + '.npz'
    output = os.path.abspath(output)

    print(f'Source        : {source}')
    print(f'Samples found : {len(samples)}')
    print(f'Packing       : {len(sample_ids)}')
    print(f'Output        : {output}')

    wavelengths = None
    scalar_names = ('R', 'n_host1', 'n_host2')
    # theta varies across blocks, so it is tracked separately from the
    # structure columns that --average-theta requires to repeat.
    optional_names = ('theta_deg',)
    spectrum_names = ('t_real', 't_imag', 'r_real', 'r_imag')
    scalars = {name: [] for name in scalar_names}
    optionals = {}
    spectra = {}
    packed_ids = []

    for position, sample_id in enumerate(sample_ids):
        reader = (
            read_npz_sample
            if samples[sample_id] == 'npz'
            else read_text_sample
        )
        record = reader(source, sample_id)

        grid = np.asarray(record['wavelength_nm'], dtype=np.float64)
        if wavelengths is None:
            wavelengths = grid
        elif not np.allclose(grid, wavelengths, rtol=0, atol=1e-6):
            raise SystemExit(
                f'Sample {sample_id} has a different wavelength grid; '
                'these samples cannot share one bundle.'
            )

        packed_ids.append(int(sample_id))
        for name in scalar_names:
            scalars[name].append(float(record[name]))
        for name in optional_names:
            if name in record:
                optionals.setdefault(name, []).append(float(record[name]))
        for name in spectrum_names:
            if name in record:
                spectra.setdefault(name, []).append(
                    np.asarray(record[name], dtype=np.float32)
                )

        if position % 500 == 0 or position + 1 == len(sample_ids):
            print(
                f'  [{position + 1}/{len(sample_ids)}] {sample_id}',
                flush=True,
            )

    for name, values in spectra.items():
        if len(values) != len(packed_ids):
            raise SystemExit(
                f'{name} is present for only {len(values)} of '
                f'{len(packed_ids)} samples; refusing to write a ragged bundle.'
            )

    bundle = {
        'ids': np.asarray(packed_ids, dtype=np.int32),
        'wavelength_nm': wavelengths.astype(np.float32),
    }
    for name in scalar_names:
        bundle[name] = np.asarray(scalars[name], dtype=np.float32)
    for name, values in optionals.items():
        if len(values) == len(packed_ids):
            bundle[name] = np.asarray(values, dtype=np.float32)
    for name, values in spectra.items():
        bundle[name] = np.stack(values)

    if args.average_theta:
        bundle = average_over_theta(bundle, scalar_names, spectra)

    np.savez_compressed(output, **bundle)
    size_mb = os.path.getsize(output) / 1e6
    print(f'Wrote {size_mb:.1f} MB')
    for name, array in bundle.items():
        print(f'  {name:<14} {array.shape} {array.dtype}')

    if args.verify:
        print('Verifying a sample of the bundle against the source ...')
        with np.load(output) as packed:
            rows = len(packed['ids'])
            checks = np.linspace(0, rows - 1, num=min(5, rows), dtype=int)
            if args.average_theta:
                # A packed row is the mean of its angle block, so compare it
                # with that mean recomputed from the source, not with one raw
                # sample -- doing the latter always "fails".
                angles = len(bundle['averaged_theta_deg'])
                for index in checks:
                    block_ids = [
                        sample_ids[index + step * rows]
                        for step in range(angles)
                    ]
                    stacks = {}
                    for block_id in block_ids:
                        record = (
                            read_npz_sample
                            if samples[block_id] == 'npz'
                            else read_text_sample
                        )(source, block_id)
                        for name in ('t_real', 't_imag'):
                            stacks.setdefault(name, []).append(
                                np.asarray(record[name], dtype=np.float32)
                            )
                    for name, values in stacks.items():
                        expected = np.stack(values).mean(axis=0).astype(
                            np.float32
                        )
                        if not np.array_equal(expected, packed[name][index]):
                            raise SystemExit(
                                f'Mismatch for averaged row {index} '
                                f'({block_ids}) in {name}.'
                            )
                    print(f'  row {index} ({"+".join(block_ids)}) ok')
                print('Verification passed.')
                return 0
            for index in checks:
                sample_id = sample_ids[index]
                reader = (
                    read_npz_sample
                    if samples[sample_id] == 'npz'
                    else read_text_sample
                )
                record = reader(source, sample_id)
                for name in ('t_real', 't_imag'):
                    original = np.asarray(record[name], dtype=np.float32)
                    if not np.array_equal(original, packed[name][index]):
                        raise SystemExit(
                            f'Mismatch for sample {sample_id} in {name}.'
                        )
                for name in scalar_names:
                    if not np.float32(record[name]) == packed[name][index]:
                        raise SystemExit(
                            f'Mismatch for sample {sample_id} in {name}.'
                        )
                print(f'  {sample_id} ok')
        print('Verification passed.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
