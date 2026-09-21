#!/usr/bin/env python3
"""Design every reachable double-notch target twice and dump the errors.

``plot_design_error_statistics.py`` needs one row per target pair, carrying the
result of the direct route (ideal target straight into the inverse network) and
of the CVAE-assisted route (a generated realisable target, then selection).
The script that produced the archived ``figures/pair_bn.json`` is no longer in
the repository, so this rebuilds it from the current models.

The pair list is read from an existing dump by default, so a rerun covers
exactly the targets the archived sweep covered and the two are comparable
row by row. ``--grid`` regenerates the list instead.

Rows where either route fails to produce two resolvable dips carry ``null`` for
that route and a short ``note``; the plotting script drops them.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
# The detection prominence is deliberately loose -- it only has to find the
# candidate valleys; the admissibility thresholds below are what decide whether
# a response counts as a double notch. All four values are the ones
# dual_peak_utils.select_cvae_assisted_design uses for the manuscript's Screen,
# so a response that counts here is one the Screen would also accept.
DIP_PROMINENCE = 0.01
MINIMUM_DISTANCE_NM = 20.0
MAXIMUM_VALLEY_TRANSMITTANCE = 0.35
MINIMUM_PROMINENCE = 0.30
MAXIMUM_VALLEY_IMBALANCE = 0.10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pairs-from',
        default=os.path.join(ROOT, 'figures', 'pair_bn.json'),
        help='Existing dump to take the target pairs from.',
    )
    parser.add_argument(
        '--grid',
        action='store_true',
        help='Generate the pair grid instead of reading --pairs-from.',
    )
    # The archived sweep ran low 405-575 with separations 55-185 in steps of
    # 10, so every separation ended in 5 and no round separation -- 50, 100,
    # 150 -- was ever swept; with the second notch also capped at 640 nm, none
    # of the three main-text cases fell inside the band the statistics cover.
    # These bounds fix both: separations are multiples of 10 from 50, and the
    # cap moves to where the structure family actually runs out (only 0.1% of
    # the training spectra put their second notch past 660 nm, against 21.5%
    # past 640).
    parser.add_argument('--low-start', type=float, default=400.0)
    parser.add_argument('--low-stop', type=float, default=600.0)
    parser.add_argument('--low-step', type=float, default=5.0)
    parser.add_argument('--separation-start', type=float, default=50.0)
    parser.add_argument('--separation-stop', type=float, default=200.0)
    parser.add_argument('--separation-step', type=float, default=10.0)
    parser.add_argument('--high-min', type=float, default=450.0)
    parser.add_argument('--high-max', type=float, default=660.0)
    parser.add_argument(
        '--candidates',
        type=int,
        default=20,
        help='CVAE candidates per target, matching the case pipeline.',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0,
                        help='Only sweep the first N pairs, for a smoke test.')
    parser.add_argument('--output', required=True)
    return parser.parse_args(argv)


def pairs_from_dump(path):
    with open(path, 'r', encoding='utf-8') as handle:
        rows = json.load(handle)
    return [(float(row['low']), float(row['high'])) for row in rows]


def pairs_from_grid(args):
    lows = np.arange(
        args.low_start, args.low_stop + 0.5 * args.low_step, args.low_step)
    separations = np.arange(
        args.separation_start,
        args.separation_stop + 0.5 * args.separation_step,
        args.separation_step,
    )
    out = []
    for low in lows:
        for separation in separations:
            high = low + separation
            if args.high_min <= high <= args.high_max:
                out.append((float(low), float(high)))
    return out


def score(dpu, wavelengths, response, ideal_target, centres):
    """Locate the two notches and grade them, always reporting the result.

    The admissibility conditions say whether a response is a usable double
    notch. They do not say whether a design exists -- the inverse network
    always returns a structure -- so a failing design keeps its dip positions
    and its wavelength error, and is marked instead of dropped. Dropping them
    would have hidden the direct route's failures, which is precisely the
    comparison the paper is making: the direct route has no selection step, so
    its failures are its result.

    Only a response with no two resolvable notches has no positions to report;
    that returns None.
    """
    response = np.asarray(response, dtype=float).reshape(-1)
    passed, failed, measured = dpu.shape_gate_verdict(
        response, wavelengths, ideal_target)
    if not measured:
        return None

    positions = np.asarray(measured['peaks'], dtype=float)
    errors = np.abs(positions - np.asarray(centres, dtype=float))
    measured.update({
        'error_sum': float(errors.sum()),
        'error_max': float(errors.max()),
        'shape_gate_passed': bool(passed),
        'failed_conditions': failed,
    })
    return measured


def main(argv=None):
    args = parse_args(argv)
    pairs = (pairs_from_grid(args) if args.grid
             else pairs_from_dump(args.pairs_from))
    if args.limit:
        pairs = pairs[:args.limit]

    output = (args.output if os.path.isabs(args.output)
              else os.path.join(ROOT, args.output))

    sys.path.insert(0, os.path.join(ROOT, '2peak'))
    os.chdir(os.path.join(ROOT, '2peak'))
    import train_networks_2p as trainer
    import dual_peak_utils as dpu

    wavelengths = dpu.load_wavelength_grid()
    tandem = dpu.build_tandem_runtime(trainer.DEVICE)
    cvae = dpu.build_cvae(trainer.DEVICE)

    print(f'pairs      : {len(pairs)}')
    print(f'candidates : {args.candidates}')
    print(f'output     : {output}', flush=True)

    rows = []
    started = time.time()
    for position, (low, high) in enumerate(pairs, start=1):
        ideal = dpu.bounded_double_lorentzian((low, high), wavelengths, 15.0)
        _, responses = tandem.inverse_forward(ideal)
        direct = score(dpu, wavelengths, responses[0], ideal, (low, high))

        assisted = None
        note = ''
        try:
            dpu.seed_everything(args.seed)
            selected = dpu.select_cvae_assisted_design(
                model=cvae, tandem=tandem, target_centers=(low, high),
                ideal_target=ideal, wavelengths=wavelengths,
                n_candidates=args.candidates, device=trainer.DEVICE)
            assisted = score(dpu, wavelengths,
                             selected['final_response'], ideal, (low, high))
        except (ValueError, KeyError) as error:
            note = f'cvae selection failed: {error}'

        row = {
            'low': low,
            'high': high,
            'separation': high - low,
            'its': direct,
            'cts': assisted,
        }
        notes = [note] if note else []
        for key, label in (('its', 'direct'), ('cts', 'cvae')):
            result = row[key]
            if result is None:
                notes.append(f'{label} route: fewer than two resolvable dips')
            elif not result['shape_gate_passed']:
                notes.append(f'{label} route inadmissible: '
                             + ', '.join(result['failed_conditions']))
        row['note'] = '; '.join(notes)
        rows.append(row)

        if position % 20 == 0 or position == len(pairs):
            elapsed = time.time() - started
            rate = elapsed / position
            print(f'  [{position}/{len(pairs)}] {elapsed / 60:.1f} min '
                  f'({rate:.2f} s/pair, '
                  f'{(len(pairs) - position) * rate / 60:.1f} min left)',
                  flush=True)

    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=1)

    usable = [row for row in rows if row['its'] and row['cts']]
    direct_errors = np.array([row['its']['error_sum'] for row in usable])
    assisted_errors = np.array([row['cts']['error_sum'] for row in usable])
    print(f'\nwrote {output}')
    print(f'\nall designs ({len(usable)}/{len(rows)} pairs where both routes '
          f'resolve two notches)')
    print(f'   mean sum|dlambda|   direct {direct_errors.mean():6.2f} nm   '
          f'CVAE-assisted {assisted_errors.mean():6.2f} nm')
    print(f'   median              direct {np.median(direct_errors):6.2f} nm   '
          f'CVAE-assisted {np.median(assisted_errors):6.2f} nm')
    print(f'   CVAE better in      '
          f'{int((assisted_errors < direct_errors).sum())}/{len(usable)}')

    # Admissibility is reported next to the error, never in place of it: the
    # design exists either way, and for the direct route -- which has no
    # selection step -- its failures are its result.
    print('\nadmissibility (the manuscript\'s four conditions):')
    for key, label in (('its', 'direct'), ('cts', 'CVAE-assisted')):
        resolved = [row[key] for row in rows if row[key]]
        admissible = [r for r in resolved if r['shape_gate_passed']]
        errors = np.array([r['error_sum'] for r in admissible])
        print(f'   {label}:')
        print(f'      two notches resolved   {len(resolved)}/{len(rows)}')
        print(f'      admissible             {len(admissible)}/{len(rows)}'
              + (f'   mean over those {errors.mean():.2f} nm'
                 if len(admissible) else ''))
        counts = {}
        for entry in resolved:
            for condition in entry['failed_conditions']:
                counts[condition] = counts.get(condition, 0) + 1
        for condition, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f'         failed {condition:<16} {count}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
