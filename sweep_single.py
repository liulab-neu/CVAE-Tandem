#!/usr/bin/env python3
"""Design every reachable single-notch target twice and dump the errors.

The single-notch half of what ``plot_design_error_statistics.py`` consumes:
one row per target wavelength carrying the direct route (the ideal Lorentzian
straight into the inverse network) and the CVAE-assisted route (a generated
realisable target, then selection).

Like ``sweep_pairs.py`` this is a reconstruction -- the script that wrote the
archived ``figures/single_bn.json`` is no longer in the repository, so that
dump could not be regenerated and the figure depended on a file nothing could
reproduce. The routes here are the ones ``plot_design_gallery.single_cases``
uses, which is the same pair of routes the case pipeline runs.

The single notch has no double-notch shape gate: ``select_single_notch_candidate``
applies its own admissibility rule and reports which one fired in ``mode``, so
that is recorded rather than re-derived.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=float, default=410.0)
    parser.add_argument('--stop', type=float, default=650.0)
    parser.add_argument('--step', type=float, default=10.0)
    parser.add_argument(
        '--candidates',
        type=int,
        default=20,
        help='CVAE candidates per target, matching the case pipeline.',
    )
    parser.add_argument('--gamma', type=float, default=15.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = (args.output if os.path.isabs(args.output)
              else os.path.join(ROOT, args.output))
    targets = np.arange(args.start, args.stop + 0.5 * args.step, args.step)

    sys.path.insert(0, os.path.join(ROOT, '1peak'))
    os.chdir(os.path.join(ROOT, '1peak'))
    import torch
    import train_networks as trainer
    import lorentz_vs_cvae_compare as lvc
    from notch_quality import (select_single_notch_candidate,
                               single_notch_verdict)

    net = lvc.build_tandem_network(trainer.DEVICE)
    net.eval_mode('tandem')
    wavelengths = net.wavelengths.detach().cpu().numpy().reshape(-1)
    step = float(np.median(np.diff(wavelengths)))

    print(f'targets    : {len(targets)}')
    print(f'candidates : {args.candidates}')
    print(f'output     : {output}', flush=True)

    rows = []
    for centre in targets:
        index = int(np.argmin(np.abs(wavelengths - centre)))
        _, ideal = lvc.lorentzian_trans(
            centre, wlmin=380, wlmax=800, gamma=args.gamma,
            wavelengths=wavelengths)
        ideal = np.asarray(ideal, dtype=float)
        tensor = torch.as_tensor(
            ideal.astype(np.float32)).reshape(1, -1).to(trainer.DEVICE)
        with torch.no_grad():
            structures, direct = net.test(tensor, tensor, 'tandem')
        direct = np.asarray(direct, dtype=float).reshape(-1)
        # net.test returns numpy for the structure branch and a tensor for the
        # spectrum, so this has to cope with either.
        structure = np.asarray(
            structures.detach().cpu() if hasattr(structures, 'detach')
            else structures,
            dtype=float).reshape(-1).tolist()

        generated = lvc.cvae_generate_spectrum(
            centre, n_samples=args.candidates,
            condition_value=index / (len(wavelengths) - 1),
            random_seed=args.seed)
        with torch.no_grad():
            _, responses = net.test(generated, generated, 'tandem')
        responses = np.asarray(responses, dtype=float)
        selection = select_single_notch_candidate(
            wavelengths, responses, float(centre),
            wavelength_tolerance_nm=0.5,
            tie_breaker=np.mean((responses - ideal) ** 2, axis=1),
            generated_responses=np.asarray(generated.detach().cpu()),
            generated_wavelength_tolerance_nm=0.5,
            generated_grid_slack_nm=0.5 * step)
        assisted = responses[selection.selected_index]

        def record(row, extra):
            # The verdict travels with the error rather than replacing it:
            # the design exists either way, and the direct route has no
            # selection step, so its failures are its result.
            passed, failed, measured = single_notch_verdict(
                wavelengths, row, ideal, float(centre))
            measured.update({
                'shape_gate_passed': bool(passed),
                'failed_conditions': failed,
            })
            measured.update(extra)
            return measured

        rows.append({
            'center': float(centre),
            'its': record(direct, {'structure': structure}),
            'cts': record(assisted, {'mode': str(selection.selection_mode)}),
        })
        print(f'  {centre:.0f} nm: 直接 {rows[-1]["its"]["error"]:5.2f}  '
              f'CVAE {rows[-1]["cts"]["error"]:5.2f} nm  '
              f'({rows[-1]["cts"]["mode"]})', flush=True)

    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=1)

    direct_errors = np.array([row['its']['error'] for row in rows])
    assisted_errors = np.array([row['cts']['error'] for row in rows])
    print(f'\nwrote {output}')
    print(f'targets                {len(rows)}')
    print(f'mean abs(dlambda)      direct {direct_errors.mean():.2f} nm   '
          f'CVAE-assisted {assisted_errors.mean():.2f} nm')
    print(f'median                 direct {np.median(direct_errors):.2f} nm   '
          f'CVAE-assisted {np.median(assisted_errors):.2f} nm')
    print(f'CVAE better in         '
          f'{int((assisted_errors < direct_errors).sum())}/{len(rows)}')

    print('\nadmissibility (the double-notch conditions that apply to one '
          'resonance):')
    for key, label in (('its', 'direct'), ('cts', 'CVAE-assisted')):
        passed = [row[key] for row in rows if row[key]['shape_gate_passed']]
        ratios = np.array([row[key]['width_ratio'] for row in rows])
        print(f'   {label}:')
        print(f'      admissible        {len(passed)}/{len(rows)}')
        print(f'      FWHM / ideal      {ratios.min():.2f} - {ratios.max():.2f}'
              f'   (median {np.median(ratios):.2f})')
        counts = {}
        for row in rows:
            for condition in row[key]['failed_conditions']:
                counts[condition] = counts.get(condition, 0) + 1
        for condition, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f'         failed {condition:<16} {count}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
