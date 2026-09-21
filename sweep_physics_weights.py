#!/usr/bin/env python3
"""Physics-loss weight grid for either the single- or double-notch system.

Retrains the forward, tandem and CVAE networks once per
(w_peak-lambda, w_peak-I) cell and scores all three on the held-out test
tenth, emitting the JSON dumps ``plot_weight_grid.py`` consumes:
``grid16_<system>_{forward,tandem,cvae}.json``.

The scoring matches ``plot_physics_loss_effect.py`` so the grid and the
stage-ablation figure measure the same thing. The single-notch dip is the
spectrum's argmin; the double-notch dips come from ``find_peaks`` at
prominence 0.01 with a 20 nm minimum separation -- the same rule as
``dual_peak_utils.detect_two_resonances`` -- keeping the two deepest sorted by
wavelength, both scored, because a single argmin only ever sees the deeper
one.

Each cell is one full training run, so the sweep is resumable: a cell whose
checkpoints already exist is scored without retraining, and a cell whose
training diverges is left out rather than taking the sweep down -- the
published grid has an n/a cell for exactly that reason.

The dumps behind the published figures were written by an ad-hoc script into
a /tmp scratchpad that has since been cleared, so this is a reconstruction of
that pipeline from the surviving plotting code. Cell-by-cell agreement with
the old figures is therefore unverified; the dumps this writes live in the
repository so they cannot be lost the same way.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = (0.0, 1.0, 10.0, 100.0)

SYSTEMS = {
    '1peak': {
        'trainer': 'train_networks',
        'cvae_module': 'CVAE',
        'cvae_checkpoint': 'cvae_best_mean.pth',
        'c_dim': 1,
        'dual': False,
        'takes_data_flag': False,
    },
    '2peak': {
        'trainer': 'train_networks_2p',
        'cvae_module': 'CVAE_2p',
        'cvae_checkpoint': 'cvae_best.pth',
        'c_dim': 2,
        'dual': True,
        'takes_data_flag': True,
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=tuple(SYSTEMS), required=True)
    parser.add_argument(
        '--data',
        default=None,
        help=(
            'Dataset for the 2peak system (default: its packed 35547 bundle). '
            'Ignored for 1peak, whose trainer has a fixed data root.'
        ),
    )
    parser.add_argument('--work-dir', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-epochs', type=int, default=2000)
    parser.add_argument(
        '--seed-cell-from',
        default=None,
        help='Copy an existing model directory in as --seed-cell.',
    )
    parser.add_argument(
        '--seed-cell', nargs=2, type=float, default=(1.0, 1.0),
        metavar=('W_PW', 'W_PI'),
    )
    parser.add_argument(
        '--only', nargs='+', default=None,
        help='Restrict to cells given as w_pw,w_pi (e.g. 0,0 1,1).',
    )
    parser.add_argument(
        '--stage',
        choices=('all', 'tandem-on-best-forward'),
        default='all',
        help=(
            'all: one full fnn+tandem+cvae run per cell. '
            'tandem-on-best-forward: keep the forward fixed at the cell the '
            'forward grid selected and sweep only the inverse network\'s '
            'weights, which is how the published ablation was run -- the '
            'tandem is always trained against the best forward, not against '
            'a forward retrained at its own cell.'
        ),
    )
    parser.add_argument(
        '--forward-from',
        default=None,
        help=(
            'FNN checkpoint the tandem sweep trains against. Defaults to the '
            'best cell of the forward dump written by the --stage all run.'
        ),
    )
    parser.add_argument(
        '--split',
        choices=('val', 'test'),
        default='test',
        help=(
            'Which split the grid is scored on. test matches the original '
            'implementation -- plot_physics_loss_effect.measure() scores the '
            'held-out test tenth -- and is what the published grids report. '
            'The manuscript text says the weights were selected on '
            'validation, so the write-up and the code disagree; the code is '
            'what produced the published numbers.'
        ),
    )
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    base = os.path.join(ROOT, args.system)
    if args.work_dir is None:
        args.work_dir = os.path.join(base, 'weight_grid_runs')
    if args.output is None:
        args.output = os.path.join(
            args.work_dir, f'grid16_{args.system}.json')
    if args.system == '2peak' and args.data is None:
        args.data = os.path.join('data', 'cstdata_35547.npz')
    return args


def cell_name(w_pw, w_pi):
    return f'w{w_pw:g}_{w_pi:g}'


def dips(row, dual, distance_samples=1):
    if not dual:
        return np.array([int(np.argmin(row))])
    # The 20 nm separation is part of the definition of "two notches", and
    # dual_peak_utils.detect_two_resonances enforces it. Without it a broad
    # notch can return two of its own samples as the pair, which reports a
    # dip-position error against a resonance that was never there.
    peaks, _ = find_peaks(-row, prominence=0.01, distance=distance_samples)
    if len(peaks) < 2:
        return None
    return np.sort(peaks[np.argsort(row[peaks])[:2]])


# The admissibility conditions of the manuscript's Screen are deliberately not
# applied here: this grid measures how accurately each network reproduces the
# dips of the held-out CST spectra, which is a model-accuracy question. Those
# conditions decide whether a *design* is usable, and are applied where designs
# are judged (sweep_pairs.py, plot_design_gallery.py).
MINIMUM_DIP_SEPARATION_NM = 20.0


def score(predicted, target, grid, dual):
    predicted = predicted.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    step = float(np.median(np.diff(np.asarray(grid, dtype=float))))
    distance_samples = max(1, int(round(MINIMUM_DIP_SEPARATION_NM / step)))
    positions, depths = [], []
    for a, b in zip(predicted, target):
        pa = dips(a, dual, distance_samples)
        pb = dips(b, dual, distance_samples)
        if pa is None or pb is None:
            continue
        positions.append(np.abs(grid[pa] - grid[pb]).mean())
        depths.append(np.abs(a[pa] - b[pb]).mean())
    return np.asarray(positions), np.asarray(depths)


def best_forward_cell(args):
    """The (w_pw, w_pi) the forward grid selected, by mean dip error."""
    stem = os.path.splitext(args.output)[0]
    rows = json.load(open(f'{stem}_forward.json'))
    best = min(rows, key=lambda r: r['dip_mean'])
    return (best['w_pw'], best['w_pi']), best['dip_mean']


def cell_bounds(model_dir):
    """Normalisation bounds the checkpoints in this directory were trained with.

    build_tandem falls back to its module constants otherwise, and those still
    say 3.0 while the 3.05 dataset's checkpoints were trained with 3.05. The
    inverse network denormalises with these, so the wrong pair silently shifts
    every structure it predicts -- the forward network is unaffected because it
    takes raw structure values.
    """
    path = os.path.join(model_dir, 'normalisation.json')
    if not os.path.isfile(path):
        return None, None
    with open(path, 'r', encoding='utf-8') as handle:
        saved = json.load(handle)
    return (torch.tensor(saved['structure_min'], dtype=torch.float32),
            torch.tensor(saved['structure_max'], dtype=torch.float32))


def build_with_bounds(trainer, args, w_pw, w_pi, model_dir):
    namespace = argparse.Namespace(
        w_peak_wavelength=w_pw, w_peak_intensity=w_pi)
    lo, hi = cell_bounds(model_dir)
    if lo is None:
        return trainer.build_tandem(args.seed, namespace)
    return trainer.build_tandem(args.seed, namespace, lo, hi)


def train_cell(args, spec, w_pw, w_pi, model_dir, log_dir, stage='all'):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    base = os.path.join(ROOT, args.system)
    command = [
        sys.executable, '-u', os.path.join(base, f'{spec["trainer"]}.py'),
        '--model-dir', model_dir,
        '--log-dir', log_dir,
        '--seed', str(args.seed),
        '--max-epochs', str(args.max_epochs),
        '--w-peak-wavelength', f'{w_pw:g}',
        '--w-peak-intensity', f'{w_pi:g}',
    ]
    if stage != 'all':
        command += ['--stage', stage]
    if spec['takes_data_flag']:
        command += ['--data', args.data]
    with open(os.path.join(log_dir, 'train.log'), 'w', encoding='utf-8') as log:
        subprocess.run(command, cwd=base, stdout=log, stderr=log, check=True)


def evaluate_cell(args, spec, modules, w_pw, w_pi, model_dir,
                  test_x, test_y, grid):
    trainer, cvae_module = modules
    device = trainer.DEVICE
    # 1peak's build_tandem takes no bounds and its module constants are
    # already right for its unchanged dataset, so only pass them when a cell
    # actually recorded a pair.
    net = build_with_bounds(trainer, args, w_pw, w_pi, model_dir)
    net.restore_FNN(os.path.join(model_dir, 'DNN_tandem_FNN_label.ckpt'))
    net.restore_INN(os.path.join(model_dir, 'DNN_tandem_INN_label.ckpt'))
    net.fnn.eval()
    net.inn.eval()
    with torch.no_grad():
        forward = net.fnn(test_x)
        _, tandem = net.forward(test_y, 'tandem')

    model = cvae_module.CVAE(
        encoder_size=trainer.ENCODER_SIZE,
        decoder_size=trainer.DECODER_SIZE,
        c_dim=spec['c_dim'],
    ).to(device)
    state = torch.load(
        os.path.join(model_dir, spec['cvae_checkpoint']),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state.get('model', state)
                          if isinstance(state, dict) else state)
    model.eval()
    with torch.no_grad():
        _, condition = model.lowest_wavelength(test_y)
        condition = torch.as_tensor(
            condition, dtype=torch.float32, device=device,
        ).reshape(len(test_y), -1)
        output = model(test_y, condition)
        cvae = output[0] if isinstance(output, tuple) else output

    rows = {}
    for stage, prediction in (('forward', forward), ('tandem', tandem),
                              ('cvae', cvae)):
        positions, depths = score(prediction, test_y, grid, spec['dual'])
        if not len(positions):
            rows[stage] = None
            continue
        rows[stage] = {
            'w_pw': w_pw,
            'w_pi': w_pi,
            # plot_weight_grid prefers the *_mean keys and refuses a dump that
            # offers dip_mean without depth_mean, so both are written.
            'dip_mean': float(positions.mean()),
            'dip_median': float(np.median(positions)),
            'depth_mean': float(depths.mean()),
            'depth_median': float(np.median(depths)),
            'mse': float(torch.nn.functional.mse_loss(prediction, test_y)),
            'scored_cases': int(len(positions)),
        }
    return rows


def sweep_tandem(args, spec, modules, cells, forward_ckpt,
                 test_x, test_y, grid, stem):
    """Sweep only the inverse network, against one fixed forward.

    This is the protocol the published ablation used: the forward is chosen
    once, then every tandem cell is trained on that same frozen forward.
    Training each cell's tandem on a forward retrained at its own weights
    instead confounds the two networks' contributions.
    """
    import shutil

    trainer, _ = modules
    work = args.work_dir + '_tandem'
    os.makedirs(work, exist_ok=True)
    rows_out = []
    for index, (w_pw, w_pi) in enumerate(cells, start=1):
        name = cell_name(w_pw, w_pi)
        cell_dir = os.path.join(work, name)
        model_dir = os.path.join(cell_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        target = os.path.join(model_dir, 'DNN_tandem_FNN_label.ckpt')
        if not os.path.isfile(target):
            shutil.copy(forward_ckpt, target)
        started = time.time()
        if not os.path.isfile(os.path.join(
                model_dir, 'DNN_tandem_INN_label.ckpt')):
            print(f'[{index}/{len(cells)}] {name}: training tandem',
                  flush=True)
            try:
                train_cell(args, spec, w_pw, w_pi, model_dir, cell_dir,
                           stage='tandem')
            except subprocess.CalledProcessError:
                print(f'[{index}/{len(cells)}] {name}: TRAINING FAILED, left '
                      f'as n/a (see {cell_dir}/train.log)', flush=True)
                continue
        else:
            print(f'[{index}/{len(cells)}] {name}: already trained',
                  flush=True)

        net = build_with_bounds(trainer, args, w_pw, w_pi, model_dir)
        net.restore_FNN(target)
        net.restore_INN(os.path.join(model_dir, 'DNN_tandem_INN_label.ckpt'))
        net.fnn.eval()
        net.inn.eval()
        with torch.no_grad():
            _, tandem = net.forward(test_y, 'tandem')
        positions, depths = score(tandem, test_y, grid, spec['dual'])
        if not len(positions):
            print(f'[{index}/{len(cells)}] {name}: no scorable case',
                  flush=True)
            continue
        rows_out.append({
            'w_pw': w_pw,
            'w_pi': w_pi,
            'dip_mean': float(positions.mean()),
            'dip_median': float(np.median(positions)),
            'depth_mean': float(depths.mean()),
            'depth_median': float(np.median(depths)),
            'mse': float(torch.nn.functional.mse_loss(tandem, test_y)),
            'scored_cases': int(len(positions)),
            'forward_checkpoint': forward_ckpt,
        })
        print(f'    dip_mean {rows_out[-1]["dip_mean"]:.3f} nm  '
              f'depth_mean {rows_out[-1]["depth_mean"] * 100:.2f} %  '
              f'mse {rows_out[-1]["mse"] * 1e4:.3g}e-4  '
              f'({time.time() - started:.0f} s)', flush=True)
        with open(f'{stem}_tandem.json', 'w', encoding='utf-8') as handle:
            json.dump(rows_out, handle, indent=2)
    print(f'\nWrote: {stem}_tandem.json  ({len(rows_out)} cells)')
    return 0


def main(argv=None):
    args = parse_args(argv)
    spec = SYSTEMS[args.system]
    base = os.path.join(ROOT, args.system)
    sys.path.insert(0, base)
    os.chdir(base)

    trainer = __import__(spec['trainer'])
    cvae_module = __import__(spec['cvae_module'])
    os.makedirs(args.work_dir, exist_ok=True)

    cells = [(pw, pi) for pw in WEIGHTS for pi in WEIGHTS]
    if args.only:
        wanted = {tuple(float(v) for v in item.split(','))
                  for item in args.only}
        cells = [c for c in cells if c in wanted]

    if args.seed_cell_from:
        target = os.path.join(
            args.work_dir, cell_name(*args.seed_cell), 'model')
        if not os.path.isdir(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            subprocess.run(['cp', '-r', args.seed_cell_from, target],
                           check=True)
            print(f'Seeded cell {cell_name(*args.seed_cell)} from '
                  f'{args.seed_cell_from}')

    print(f'System     : {args.system} (dual={spec["dual"]})')
    print(f'Cells      : {len(cells)}')
    print(f'Data       : {args.data or "trainer default"}')
    print(f'Work dir   : {args.work_dir}')
    if args.dry_run:
        for pw, pi in cells:
            done = os.path.isfile(os.path.join(
                args.work_dir, cell_name(pw, pi), 'model',
                spec['cvae_checkpoint']))
            print(f'  {cell_name(pw, pi):<10} '
                  f'{"trained" if done else "needs training"}')
        return 0

    if spec['takes_data_flag']:
        split = trainer.load_split(args.seed, args.data, 'auto')
    else:
        split = trainer.load_split(args.seed)
    # Loss-weight selection scores the VALIDATION split. The manuscript keeps
    # the test set untouched until the final configuration is fixed, so
    # scoring the grid on test would both contaminate it and misreport the
    # selection criterion.
    structures, spectra, _, val_idx, test_idx = split
    eval_idx = val_idx if args.split == 'val' else test_idx
    test_x = structures[eval_idx].to(trainer.DEVICE)
    test_y = spectra[eval_idx].to(trainer.DEVICE)

    if spec['dual']:
        import dual_peak_utils as dpu
        grid = dpu.load_wavelength_grid()
    else:
        reference = trainer.build_tandem(
            args.seed,
            argparse.Namespace(w_peak_wavelength=1.0, w_peak_intensity=1.0),
        )
        grid = reference.wavelengths.detach().cpu().numpy().reshape(-1)
    print(f'Scoring on : {args.split} split, {len(eval_idx)} cases   '
          f'grid {len(grid)} points')

    stem = os.path.splitext(args.output)[0]
    if args.stage == 'tandem-on-best-forward':
        forward_ckpt = args.forward_from
        if forward_ckpt is None:
            cell, dip = best_forward_cell(args)
            forward_ckpt = os.path.join(
                args.work_dir, cell_name(*cell), 'model',
                'DNN_tandem_FNN_label.ckpt')
            print(f'Best forward: cell {cell_name(*cell)} '
                  f'(dip_mean {dip:.3f} nm) -> {forward_ckpt}')
        if not os.path.isfile(forward_ckpt):
            raise SystemExit(f'No forward checkpoint at {forward_ckpt}')
        return sweep_tandem(args, spec, (trainer, cvae_module), cells,
                            forward_ckpt, test_x, test_y, grid, stem)

    dumps = {'forward': [], 'tandem': [], 'cvae': []}
    for index, (w_pw, w_pi) in enumerate(cells, start=1):
        name = cell_name(w_pw, w_pi)
        cell_dir = os.path.join(args.work_dir, name)
        model_dir = os.path.join(cell_dir, 'model')
        started = time.time()
        if not os.path.isfile(os.path.join(
                model_dir, spec['cvae_checkpoint'])):
            print(f'[{index}/{len(cells)}] {name}: training', flush=True)
            try:
                train_cell(args, spec, w_pw, w_pi, model_dir, cell_dir)
            except subprocess.CalledProcessError:
                print(f'[{index}/{len(cells)}] {name}: TRAINING FAILED, left '
                      f'as n/a (see {cell_dir}/train.log)', flush=True)
                continue
        else:
            print(f'[{index}/{len(cells)}] {name}: already trained',
                  flush=True)

        try:
            rows = evaluate_cell(args, spec, (trainer, cvae_module),
                                 w_pw, w_pi, model_dir, test_x, test_y, grid)
        except Exception as error:
            print(f'[{index}/{len(cells)}] {name}: SCORING FAILED ({error})',
                  flush=True)
            continue
        for stage, row in rows.items():
            if row is not None:
                dumps[stage].append(row)
        summary = rows.get('tandem') or rows.get('forward')
        if summary:
            print(f'    dip_mean {summary["dip_mean"]:.3f} nm  '
                  f'depth_mean {summary["depth_mean"] * 100:.2f} %  '
                  f'mse {summary["mse"] * 1e4:.3g}e-4  '
                  f'({time.time() - started:.0f} s)', flush=True)
        for stage, rows_out in dumps.items():
            with open(f'{stem}_{stage}.json', 'w', encoding='utf-8') as handle:
                json.dump(rows_out, handle, indent=2)

    print('\nWrote:')
    for stage in dumps:
        print(f'  {stem}_{stage}.json  ({len(dumps[stage])} cells)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
