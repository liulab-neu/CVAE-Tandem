#!/usr/bin/env python3
"""Reproducible training for the dual-notch forward, tandem and CVAE networks.

Dual-peak counterpart of 1peak/train_networks.py.  The dual-peak split was
already seeded (train_test_split(random_state=42)); what is new here is early
stopping on the validation loss and keeping the best-validation weights rather
than the final epoch's.  Here the split and the initialisation are
seeded, every stage stops when the validation loss has not improved for
``--patience`` epochs, and the checkpoint written is the best-validation one.

Stages, in order:
  fnn     structures -> spectrum
  tandem  spectrum -> structures -> spectrum, with the forward network frozen
  cvae    conditional generator of realisable target spectra

Logs keep the "Epoch N | Train Loss: x | Val Loss: y" format so the existing
loss-curve scripts still parse them.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import CVAE_2p as cvae_module
import tandem_2p as tandem_module
from data_loader_2P import load_angle_averaged, load_arrays, load_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# The current dataset: 3.05 n axis, angles 0/30/60 collapsed to their mean.
DATA_ROOT = os.path.join(BASE_DIR, 'data', 'cstdata_35547.npz')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
# Written next to the checkpoints so inference uses the same normalisation the
# weights were trained with; see dual_peak_utils.load_structure_bounds.
NORMALISATION_FILE = 'normalisation.json'

N_INPUT = 3
N_CLASSES = 1001
FNN_SIZE = [N_INPUT, 64, 128, 256, 512, 1024, 2048, 1024, N_CLASSES]
INN_SIZE = [N_CLASSES, 1024, 2048, 1024, 512, 256, 128, 64, N_INPUT]
def _deployed_bounds():
    """Bounds the checkpoints in MODEL_DIR were trained with.

    build_tandem's callers may omit bounds -- plot_physics_loss_effect and
    plot_shap_attribution both do -- and a stale hard-coded pair here silently
    mis-denormalises everything the inverse network predicts. Reading the
    sidecar keeps those callers correct without each having to know about it.
    load_split still derives the bounds from its own training set, which is
    what a fresh run writes into the sidecar.
    """
    path = os.path.join(MODEL_DIR, NORMALISATION_FILE)
    # See load_structure_bounds: sidecars make this path rare, and 3.0
    # was the value that silently mis-denormalised the inverse network.
    minimum, maximum = [2.5, 1.5, 1.5], [6.5, 3.05, 3.05]
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as handle:
            saved = json.load(handle)
        minimum = saved.get('structure_min', minimum)
        maximum = saved.get('structure_max', maximum)
    return (torch.tensor(minimum, dtype=torch.float32),
            torch.tensor(maximum, dtype=torch.float32))


STRUCTURE_MIN, STRUCTURE_MAX = _deployed_bounds()
ENCODER_SIZE = [1001, 1024, 2048, 1024, 512, 256, 128, 20]
DECODER_SIZE = [20, 128, 256, 512, 1024, 2048, 1024, 1001]

# The original dual-notch settings, confirmed by the supplementary weight
# sweep: the dip position is normalised here, so 1 is already the right scale
# and raising it to 100 wrecks the design accuracy.
W_PEAK_WAVELENGTH = 1.0
W_PEAK_INTENSITY = 1.0
W_GENERATION = 300.0         # prior-sample condition term; 0 disables it


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('fnn', 'tandem', 'cvae', 'all'),
                   default='all')
    p.add_argument('--seed', type=int, default=42)
    # The original script ran a fixed 2000 epochs; early stopping is available
    # but off by default so a plain run reproduces that schedule.
    p.add_argument('--max-epochs', type=int, default=2000)
    p.add_argument('--patience', type=int, default=2000,
                   help='Stop after this many epochs without a new best.')
    p.add_argument('--w-peak-wavelength', type=float,
                   default=W_PEAK_WAVELENGTH)
    p.add_argument('--w-peak-intensity', type=float, default=W_PEAK_INTENSITY)
    p.add_argument('--w-generation', type=float, default=W_GENERATION,
                   help='Weight of the CVAE prior-sample condition term.')
    p.add_argument('--norm', choices=('bn', 'gn'), default='bn',
                   help='Normalisation inside the dense stacks. gn keeps no '
                        'running statistics, so train and eval behave '
                        'identically -- see the note in train_fnn about the '
                        'BatchNorm statistics being frozen after epoch 0.')
    p.add_argument('--data', default=DATA_ROOT,
                   help='Dataset directory or packed .npz bundle.')
    p.add_argument('--average-theta', choices=('auto', 'yes', 'no'),
                   default='auto',
                   help='Collapse incidence angles into one mean spectrum '
                        'per structure. auto averages when the dataset '
                        'carries an angle axis.')
    p.add_argument('--model-dir', default=MODEL_DIR)
    p.add_argument('--log-dir', default=BASE_DIR)
    return p.parse_args()


def dataset_has_angles(source):
    if os.path.isdir(source):
        names = sorted(f for f in os.listdir(source) if f.endswith('.npz'))
        if not names:
            return False
        with np.load(os.path.join(source, names[0])) as sample:
            return 'theta_deg' in sample.files
    if source.endswith('.npz'):
        with np.load(source) as bundle:
            return 'theta_deg' in bundle.files
    return False


def load_split(seed, source=DATA_ROOT, average_theta='auto'):
    if source.endswith('.npz') or (
        os.path.isdir(source)
        and not any(f.endswith('.a') for f in os.listdir(source))
    ):
        should_average = (
            average_theta == 'yes'
            or (average_theta == 'auto' and dataset_has_angles(source))
        )
        loader = load_angle_averaged if should_average else load_arrays
        parameters, spectra_array, _ = loader(source)
        if should_average:
            print('Averaging the incidence angles into one spectrum per '
                  'structure.')
        structures = torch.tensor(parameters, dtype=torch.float32)
        spectra = torch.tensor(spectra_array, dtype=torch.float32)
    else:
        structures_raw, spectra_raw = load_data(source)
        structures = torch.tensor(
            [[d['R'], d['n_host1'], d['n_host2']] for d in structures_raw],
            dtype=torch.float32,
        )
        spectra = torch.tensor(
            np.array([np.array(v)[:, 1].reshape(-1) for v in spectra_raw]),
            dtype=torch.float32,
        )
    rng = np.random.default_rng(seed)
    order = np.arange(len(spectra))
    rng.shuffle(order)
    n_train = int(len(spectra) * 0.8)
    n_val = int(len(spectra) * 0.1)
    return (structures, spectra, order[:n_train],
            order[n_train:n_train + n_val], order[n_train + n_val:])


def structure_bounds(structures):
    """Normalisation bounds taken from the data rather than hard-coded.

    Keeping these in sync with the dataset by hand is what would silently
    mis-denormalise the INN when the n axis was extended to 3.05.
    """
    return structures.min(dim=0).values, structures.max(dim=0).values


def save_normalisation(model_dir, minimum, maximum, norm='bn'):
    path = os.path.join(model_dir, NORMALISATION_FILE)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump({'structure_min': [float(v) for v in minimum],
                   'structure_max': [float(v) for v in maximum],
                   'norm': norm},
                  handle, indent=2)
    return path


def build_tandem(seed, args, minimum=None, maximum=None):
    torch.manual_seed(seed)
    net = tandem_module.tandem_network(
        INN_size=INN_SIZE, FNN_size=FNN_SIZE, starter_learning_rate=0.001,
        decay_step=1200, decay_rate=0.97,
        norm=getattr(args, 'norm', 'bn'),
        structure_min=STRUCTURE_MIN if minimum is None else minimum,
        structure_max=STRUCTURE_MAX if maximum is None else maximum,
    ).to(DEVICE)
    net.w_peak_wavelength = args.w_peak_wavelength
    net.w_peak_intensity = args.w_peak_intensity
    return net


def target_dip_cache(spectra, net, batch=512):
    """Dip indices for every target spectrum, found once.

    The peak search dominates the dual-notch loss, and the targets never
    change during training, so recomputing them every step was most of the
    cost -- caching cuts the loss's peak searches in half.
    """
    located = []
    with torch.no_grad():
        for start in range(0, len(spectra), batch):
            chunk = spectra[start:start + batch].to(DEVICE)
            located.append(net.lowest_wavelength(chunk)[0])
    return torch.cat(located)


def tandem_losses(net, pred, target, target_index=None):
    spectral = net.loss_fn(pred, target)
    index_pred, position_pred = net.lowest_wavelength(pred)
    if target_index is None:
        _, position_true = net.lowest_wavelength(target)
    else:
        position_true = net.interpolate_positions(target, target_index)
    peak_wavelength = net.loss_fn(position_pred, position_true)
    # Both resonances, at the same indices the position term uses -- a single
    # argmin would leave the shallower notch's depth unconstrained.
    rows = torch.arange(pred.shape[0], device=pred.device).reshape(-1, 1)
    peak_intensity = net.loss_fn(pred[rows, index_pred],
                                 target[rows, index_pred])
    total = (net.w_spectral * spectral
             + net.w_peak_wavelength * peak_wavelength
             + net.w_peak_intensity * peak_intensity)
    return total, spectral


def run_stage(name, net, data, args, step_fn, eval_fn, log_path, save_fn):
    best = float('inf')
    best_epoch = -1
    since = 0
    started = time.time()
    with open(log_path, 'w', encoding='utf-8') as log:
        for epoch in range(args.max_epochs):
            train_loss = step_fn(epoch)
            val_loss = eval_fn()
            log.write(f'Epoch {epoch} | Train Loss: {train_loss:.6f} '
                      f'| Val Loss: {val_loss:.6f}\n')
            log.flush()
            if val_loss < best - 1e-12:
                best, best_epoch, since = val_loss, epoch, 0
                save_fn()
            else:
                since += 1
                if since >= args.patience:
                    break
            if (epoch + 1) % 200 == 0:
                print(f'  [{name}] epoch {epoch + 1} train {train_loss:.3e} '
                      f'val {val_loss:.3e} best {best:.3e}@{best_epoch}',
                      flush=True)
    print(f'  [{name}] stopped at epoch {epoch}; best val {best:.6e} at '
          f'epoch {best_epoch}; {time.time() - started:.0f} s', flush=True)
    return {'best_val': best, 'best_epoch': best_epoch, 'epochs': epoch + 1}


def train_fnn(args, structures, spectra, train_idx, val_idx,
              bounds=(None, None)):
    net = build_tandem(args.seed, args, *bounds)
    x_train = structures[train_idx].to(DEVICE)
    y_train = spectra[train_idx].to(DEVICE)
    x_val = structures[val_idx].to(DEVICE)
    y_val = spectra[val_idx].to(DEVICE)
    batch = 300
    generator = np.random.default_rng(args.seed)
    train_dips = target_dip_cache(y_train, net)

    def step(epoch):
        # Reproduces tandem_2p.py's training behaviour: the first epoch
        # populates the BatchNorm running statistics and every epoch after it
        # runs in eval mode with those statistics frozen.
        if args.norm == 'gn':
            net.fnn.train()
        else:
            net.fnn.train() if epoch == 0 else net.fnn.eval()
        order = generator.permutation(len(x_train))
        total = 0.0
        n = max(1, len(x_train) // batch)
        for i in range(n):
            sel = torch.as_tensor(order[i * batch:(i + 1) * batch],
                                  device=DEVICE)
            net.optimizer_fnn.zero_grad()
            pred = net.fnn(x_train[sel])
            loss, spectral = tandem_losses(net, pred, y_train[sel],
                                           train_dips[sel])
            loss.backward()
            net.optimizer_fnn.step()
            net.scheduler_fnn.step()
            total += float(spectral.detach())
        return total / n

    def evaluate():
        net.fnn.eval()
        with torch.no_grad():
            return float(net.loss_fn(net.fnn(x_val), y_val))

    path = os.path.join(args.model_dir, 'DNN_tandem_FNN_label.ckpt')
    return net, run_stage('fnn', net, None, args, step, evaluate,
                          os.path.join(args.log_dir, 'DNN_FNN-t.txt'),
                          lambda: net.save_FNN(path))


def train_tandem(args, structures, spectra, train_idx, val_idx,
                 bounds=(None, None)):
    net = build_tandem(args.seed + 1, args, *bounds)
    net.restore_FNN(os.path.join(args.model_dir, 'DNN_tandem_FNN_label.ckpt'))
    for parameter in net.fnn.parameters():
        parameter.requires_grad_(False)
    y_train = spectra[train_idx].to(DEVICE)
    y_val = spectra[val_idx].to(DEVICE)
    batch = 300
    generator = np.random.default_rng(args.seed + 1)
    train_dips = target_dip_cache(y_train, net)

    def step(epoch):
        # Frozen BatchNorm statistics after the first epoch, as in train_fnn.
        if args.norm == 'gn':
            net.inn.train()
        else:
            net.inn.train() if epoch == 0 else net.inn.eval()
        net.fnn.eval()
        order = generator.permutation(len(y_train))
        total = 0.0
        n = max(1, len(y_train) // batch)
        for i in range(n):
            sel = torch.as_tensor(order[i * batch:(i + 1) * batch],
                                  device=DEVICE)
            target = y_train[sel]
            net.optimizer_inn.zero_grad()
            _, pred = net.forward(target, 'tandem')
            loss, spectral = tandem_losses(net, pred, target,
                                           train_dips[sel])
            loss.backward()
            net.optimizer_inn.step()
            net.scheduler_inn.step()
            total += float(spectral.detach())
        return total / n

    def evaluate():
        net.inn.eval()
        net.fnn.eval()
        with torch.no_grad():
            _, pred = net.forward(y_val, 'tandem')
            return float(net.loss_fn(pred, y_val))

    path = os.path.join(args.model_dir, 'DNN_tandem_INN_label.ckpt')
    return net, run_stage('tandem', net, None, args, step, evaluate,
                          os.path.join(args.log_dir, 'DNN_tandem-t.txt'),
                          lambda: net.save_INN(path))


def train_cvae(args, spectra, train_idx, val_idx):
    torch.manual_seed(args.seed + 2)
    model = cvae_module.CVAE(encoder_size=ENCODER_SIZE,
                             decoder_size=DECODER_SIZE, c_dim=2).to(DEVICE)
    model.w_peak_wavelength = args.w_peak_wavelength
    model.w_peak_intensity = args.w_peak_intensity
    model.w_generation = args.w_generation
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1200,
                                                gamma=0.97)
    y_train = spectra[train_idx].to(DEVICE)
    y_val = spectra[val_idx].to(DEVICE)

    def condition(batch_spectra):
        _, position = model.lowest_wavelength(batch_spectra)
        return position.detach()

    batch = 100
    generator = np.random.default_rng(args.seed + 2)

    def step(_):
        model.train()
        order = generator.permutation(len(y_train))
        total = 0.0
        n = max(1, len(y_train) // batch)
        for i in range(n):
            sel = torch.as_tensor(order[i * batch:(i + 1) * batch],
                                  device=DEVICE)
            data = y_train[sel]
            optimiser.zero_grad()
            c = condition(data)
            recon, mu, log_var, _ = model(data, c)
            loss = model.loss_function(recon, data, mu, log_var, c)
            if model.w_generation:
                loss = loss + model.w_generation * model.generation_loss(c)
            loss.backward()
            optimiser.step()
            scheduler.step()
            total += float(loss.detach())
        return total / n

    def evaluate():
        model.eval()
        with torch.no_grad():
            c = condition(y_val)
            recon, mu, log_var, _ = model(y_val, c)
            total = model.loss_function(recon, y_val, mu, log_var, c)
            if model.w_generation:
                total = total + model.w_generation * model.generation_loss(c)
            return float(total)

    path = os.path.join(args.model_dir, 'cvae_best.pth')

    def save():
        torch.save(model.state_dict(), path)

    return model, run_stage('cvae', model, None, args, step, evaluate,
                            os.path.join(args.log_dir, 'CVAE_2p-t.txt'), save)


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    np.random.seed(args.seed)
    structures, spectra, train_idx, val_idx, test_idx = load_split(
        args.seed, args.data, args.average_theta)
    minimum, maximum = structure_bounds(structures)
    print(f'device {DEVICE}  train {len(train_idx)}  val {len(val_idx)}  '
          f'test {len(test_idx)}  w_peak_wavelength {args.w_peak_wavelength}')
    print(f'data {args.data}')
    print(f'structure_min {[round(float(v), 4) for v in minimum]}  '
          f'structure_max {[round(float(v), 4) for v in maximum]}')
    saved = save_normalisation(args.model_dir, minimum, maximum, args.norm)
    print(f'Saved: {saved}')
    bounds = (minimum, maximum)

    summary = {}
    if args.stage in ('fnn', 'all'):
        print('== forward network ==')
        _, summary['fnn'] = train_fnn(args, structures, spectra,
                                      train_idx, val_idx, bounds)
    if args.stage in ('tandem', 'all'):
        print('== tandem network ==')
        _, summary['tandem'] = train_tandem(args, structures, spectra,
                                            train_idx, val_idx, bounds)
    if args.stage in ('cvae', 'all'):
        print('== CVAE ==')
        _, summary['cvae'] = train_cvae(args, spectra, train_idx, val_idx)

    path = os.path.join(args.log_dir, 'training_summary.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump({'seed': args.seed, 'data': args.data, 'norm': args.norm,
                   'structure_min': [float(v) for v in minimum],
                   'structure_max': [float(v) for v in maximum],
                   'stages': summary,
                   'w_peak_wavelength': args.w_peak_wavelength,
                   'w_peak_intensity': args.w_peak_intensity,
                   'w_generation': args.w_generation}, handle,
                  indent=2)
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
