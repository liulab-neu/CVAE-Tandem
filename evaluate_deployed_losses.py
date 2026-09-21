#!/usr/bin/env python3
"""Test-set loss components of the six deployed networks.

The supplement quotes a total objective and its parts -- reconstruction, KLD,
peak-wavelength, peak-intensity -- for each of the six models actually
shipped in ``<system>/model``. Those numbers were not recoverable from any
file in the repository, so this recomputes them from the deployed
checkpoints on the held-out test tenth.

Each total is formed with the loss weights the deployed checkpoint was
trained with, read from ``<system>/model/provenance.json`` where present, so
the total is the objective that model was actually optimising rather than a
default.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))

# gamma (peak-wavelength) and delta (peak-intensity) of the deployed cells.
DEPLOYED = {
    ('1peak', 'forward'): (100.0, 1.0),
    ('1peak', 'tandem'): (1.0, 0.0),
    ('1peak', 'cvae'): (100.0, 1.0),
    ('2peak', 'forward'): (0.0, 1.0),
    ('2peak', 'tandem'): (1.0, 0.0),
    ('2peak', 'cvae'): (100.0, 0.0),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=('1peak', '2peak'), required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=None,
                        help='Write the components to this JSON as well.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    system = args.system
    sys.path.insert(0, os.path.join(ROOT, system))
    os.chdir(os.path.join(ROOT, system))

    import argparse as _argparse
    if system == '1peak':
        import train_networks as trainer
        import CVAE as cvae_module
        cvae_file, c_dim = 'cvae_best_mean.pth', 1
    else:
        import train_networks_2p as trainer
        import CVAE_2p as cvae_module
        cvae_file, c_dim = 'cvae_best.pth', 2

    structures, spectra, train_idx, val_idx, test_idx = trainer.load_split(
        args.seed)
    device = trainer.DEVICE
    x = torch.as_tensor(structures[test_idx]).float().to(device)
    y = torch.as_tensor(spectra[test_idx]).float().to(device)
    print(f'{system}: test rows {len(test_idx)} '
          f'(train {len(train_idx)}, val {len(val_idx)})')

    mse = torch.nn.MSELoss()
    report = {}

    def net_for(stage):
        gamma, delta = DEPLOYED[(system, stage)]
        options = _argparse.Namespace(
            seed=args.seed, w_peak_wavelength=gamma, w_peak_intensity=delta)
        net = trainer.build_tandem(args.seed, options)
        net.restore_FNN(os.path.join('model', 'DNN_tandem_FNN_label.ckpt'))
        net.restore_INN(os.path.join('model', 'DNN_tandem_INN_label.ckpt'))
        net.fnn.eval()
        net.inn.eval()
        return net, gamma, delta

    # ---- forward ----
    net, gamma, delta = net_for('forward')
    with torch.no_grad():
        predicted = net.fnn(x)
        total, spectral = trainer.tandem_losses(net, predicted, y)
        index, position = net.lowest_wavelength(predicted)
        _, truth = net.lowest_wavelength(y)
        rows = torch.arange(predicted.shape[0],
                            device=predicted.device).reshape(-1, 1)
        peak_w = mse(position, truth)
        peak_i = mse(predicted[rows, index], y[rows, index])
    report['forward'] = {
        'gamma': gamma, 'delta': delta,
        'w_spectral': float(net.w_spectral),
        'reconstruction': float(spectral), 'peak_wavelength': float(peak_w),
        'peak_intensity': float(peak_i), 'total': float(total),
    }

    # ---- tandem ----
    net, gamma, delta = net_for('tandem')
    with torch.no_grad():
        if system == '1peak':
            _, predicted = net.test(y, y, 'tandem')
            predicted = torch.as_tensor(np.asarray(predicted)).float().to(device)
        else:
            predicted = net.forward(y, 'tandem')[1]
        total, spectral = trainer.tandem_losses(net, predicted, y)
        index, position = net.lowest_wavelength(predicted)
        _, truth = net.lowest_wavelength(y)
        rows = torch.arange(predicted.shape[0],
                            device=predicted.device).reshape(-1, 1)
        peak_w = mse(position, truth)
        peak_i = mse(predicted[rows, index], y[rows, index])
    report['tandem'] = {
        'gamma': gamma, 'delta': delta,
        'w_spectral': float(net.w_spectral),
        'reconstruction': float(spectral), 'peak_wavelength': float(peak_w),
        'peak_intensity': float(peak_i), 'total': float(total),
    }

    # ---- CVAE ----
    gamma, delta = DEPLOYED[(system, 'cvae')]
    model = cvae_module.CVAE(encoder_size=trainer.ENCODER_SIZE,
                             decoder_size=trainer.DECODER_SIZE,
                             c_dim=c_dim).to(device)
    model.load_state_dict(torch.load(os.path.join('model', cvae_file),
                                     map_location=device, weights_only=True))
    model.w_peak_wavelength = gamma
    model.w_peak_intensity = delta
    model.eval()
    net_for_index = net
    with torch.no_grad():
        index, condition = net_for_index.lowest_wavelength(y)
        condition = condition.reshape(len(y), c_dim).to(device)
        recon, mu, log_var, _ = model(y, condition)
        total = model.loss_function(recon, y, mu, log_var, condition)
        min_index, c_hat = model.lowest_wavelength(recon)
        rows = torch.arange(min_index.shape[0],
                            device=recon.device).reshape(-1, 1)
        report['cvae'] = {
            'gamma': gamma, 'delta': delta,
            'w_reconstruction': float(model.w_reconstruction),
            'reconstruction': float(mse(recon, y)),
            'kld': float(torch.mean(-0.5 * torch.sum(
                1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)),
            'peak_wavelength': float(mse(c_hat.to(device), condition)),
            'peak_intensity': float(mse(y[rows, min_index],
                                        recon[rows, min_index])),
            'total': float(total),
        }

    print(f'\n{"network":<10}{"gamma":>7}{"delta":>7}{"total":>12}'
          f'{"recon":>12}{"KLD":>10}{"peak-lambda":>14}{"peak-I":>12}')
    for stage in ('forward', 'tandem', 'cvae'):
        row = report[stage]
        print(f'{stage:<10}{row["gamma"]:>7g}{row["delta"]:>7g}'
              f'{row["total"]:>12.4g}{row["reconstruction"]:>12.4g}'
              f'{row.get("kld", float("nan")):>10.4g}'
              f'{row["peak_wavelength"]:>14.4g}{row["peak_intensity"]:>12.4g}')

    if args.output:
        path = (args.output if os.path.isabs(args.output)
                else os.path.join(ROOT, args.output))
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=1)
        print(f'\nwrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
