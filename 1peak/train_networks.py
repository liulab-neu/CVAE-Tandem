#!/usr/bin/env python3
"""Reproducible training for the single-notch forward, tandem and CVAE networks.

Replaces the ad-hoc flow in train_no_theta.py / train_CVAE.py, which shuffled
the split without a seed, always kept the final epoch's weights, and had to be
re-run by hand to reach convergence.  Here the split and the initialisation are
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

import CVAE as cvae_module
import tandem as tandem_module
from data_loader import load_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = '/home/yuxiao/Yuxiao Li/batch3'
MODEL_DIR = os.path.join(BASE_DIR, 'model')

N_INPUT = 2
N_CLASSES = 1001
FNN_SIZE = [N_INPUT, 64, 128, 256, 512, 1024, 2048, 1024, N_CLASSES]
INN_SIZE = [N_CLASSES, 1024, 2048, 1024, 512, 256, 128, 64, N_INPUT]
STRUCTURE_MIN = torch.tensor([2.5, 1.5])
STRUCTURE_MAX = torch.tensor([6.5, 3.05])
ENCODER_SIZE = [1001, 1024, 2048, 1024, 512, 256, 128, 20]
DECODER_SIZE = [20, 128, 256, 512, 1024, 2048, 1024, 1001]

W_PEAK_WAVELENGTH = 100.0   # chosen by the sweep in the supplementary ablation
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
    p.add_argument('--model-dir', default=MODEL_DIR)
    p.add_argument('--log-dir', default=BASE_DIR)
    return p.parse_args()


def load_split(seed):
    structures_raw, _ = load_data(DATA_ROOT)
    structures = torch.tensor(
        [[d['R'], d['n_host']] for d in structures_raw], dtype=torch.float32
    )
    spectra = torch.mean(
        torch.load(os.path.join(BASE_DIR, 'data', 'y_related.pt')), dim=1
    )
    rng = np.random.default_rng(seed)
    order = np.arange(len(spectra))
    rng.shuffle(order)
    n_train = int(len(spectra) * 0.8)
    n_val = int(len(spectra) * 0.1)
    return (structures, spectra, order[:n_train],
            order[n_train:n_train + n_val], order[n_train + n_val:])


def build_tandem(seed, args):
    torch.manual_seed(seed)
    net = tandem_module.tandem_network(
        INN_size=INN_SIZE, FNN_size=FNN_SIZE, starter_learning_rate=0.001,
        decay_step=1200, decay_rate=0.97,
        structure_min=STRUCTURE_MIN, structure_max=STRUCTURE_MAX,
    ).to(DEVICE)
    net.w_peak_wavelength = args.w_peak_wavelength
    net.w_peak_intensity = args.w_peak_intensity
    return net


def tandem_losses(net, pred, target):
    spectral = net.loss_fn(pred, target)
    _, position_pred = net.lowest_wavelength(pred)
    _, position_true = net.lowest_wavelength(target)
    peak_wavelength = net.loss_fn(position_pred, position_true)
    index = torch.argmin(pred, dim=1)
    rows = torch.arange(pred.shape[0], device=pred.device)
    peak_intensity = net.loss_fn(pred[rows, index], target[rows, index])
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


def train_fnn(args, structures, spectra, train_idx, val_idx):
    net = build_tandem(args.seed, args)
    x_train = structures[train_idx].to(DEVICE)
    y_train = spectra[train_idx].to(DEVICE)
    x_val = structures[val_idx].to(DEVICE)
    y_val = spectra[val_idx].to(DEVICE)
    batch = 64
    generator = np.random.default_rng(args.seed)

    def step(epoch):
        # Reproduces tandem.py's training behaviour: the modules start in train
        # mode, so the first epoch populates the BatchNorm running statistics,
        # and every epoch after that runs in eval mode with those statistics
        # frozen.  Updating them throughout instead costs roughly 27x in
        # validation loss on this dataset.
        net.fnn.train() if epoch == 0 else net.fnn.eval()
        order = generator.permutation(len(x_train))
        total = 0.0
        n = max(1, len(x_train) // batch)
        for i in range(n):
            sel = torch.as_tensor(order[i * batch:(i + 1) * batch],
                                  device=DEVICE)
            net.optimizer_fnn.zero_grad()
            pred = net.fnn(x_train[sel])
            loss, spectral = tandem_losses(net, pred, y_train[sel])
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


def train_tandem(args, structures, spectra, train_idx, val_idx):
    net = build_tandem(args.seed + 1, args)
    net.restore_FNN(os.path.join(args.model_dir, 'DNN_tandem_FNN_label.ckpt'))
    for parameter in net.fnn.parameters():
        parameter.requires_grad_(False)
    y_train = spectra[train_idx].to(DEVICE)
    y_val = spectra[val_idx].to(DEVICE)
    batch = 128
    generator = np.random.default_rng(args.seed + 1)

    def step(epoch):
        # Frozen BatchNorm statistics after the first epoch, as in train_fnn.
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
            loss, spectral = tandem_losses(net, pred, target)
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
                             decoder_size=DECODER_SIZE, c_dim=1).to(DEVICE)
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

    path = os.path.join(args.model_dir, 'cvae_best_mean.pth')

    def save():
        torch.save(model.state_dict(), path)

    return model, run_stage('cvae', model, None, args, step, evaluate,
                            os.path.join(args.log_dir, 'CVAE_loss.txt'), save)


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    np.random.seed(args.seed)
    structures, spectra, train_idx, val_idx, test_idx = load_split(args.seed)
    print(f'device {DEVICE}  train {len(train_idx)}  val {len(val_idx)}  '
          f'test {len(test_idx)}  w_peak_wavelength {args.w_peak_wavelength}')

    summary = {}
    if args.stage in ('fnn', 'all'):
        print('== forward network ==')
        _, summary['fnn'] = train_fnn(args, structures, spectra,
                                      train_idx, val_idx)
    if args.stage in ('tandem', 'all'):
        print('== tandem network ==')
        _, summary['tandem'] = train_tandem(args, structures, spectra,
                                            train_idx, val_idx)
    if args.stage in ('cvae', 'all'):
        print('== CVAE ==')
        _, summary['cvae'] = train_cvae(args, spectra, train_idx, val_idx)

    path = os.path.join(args.log_dir, 'training_summary.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump({'seed': args.seed, 'stages': summary,
                   'w_peak_wavelength': args.w_peak_wavelength,
                   'w_peak_intensity': args.w_peak_intensity,
                   'w_generation': args.w_generation}, handle,
                  indent=2)
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
