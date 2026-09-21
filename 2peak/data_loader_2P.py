# data_loader.py
import os

import numpy as np
import pandas as pd

def load_a(filename):
    """
    Load data from a .a file.
    """
    values = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            var, expr = line.split(":")
            var = var.strip()
            expr = expr.strip()
            values[var] = eval(expr, None, values)
    return values

def load_b(filename):
    """
    Load data from a .b file.
    """
    return pd.read_csv(filename, sep='\t', header=None, skiprows=1, usecols=[0, 1])

def load_arrays(source):
    """
    Load a packed dataset as arrays: (parameters, spectra, wavelengths).

    ``source`` is either a bundle written by pack_dataset.py or a directory of
    per-sample .npz files from gen_2peak_dataset.py. Parameters come back as
    (N, 3) in R, n_host1, n_host2 order and spectra as (N, 1001) transmittance,
    matching what load_data returns column-wise but without the per-sample
    Python objects.
    """
    if os.path.isdir(source):
        names = sorted(f for f in os.listdir(source) if f.endswith('.npz'))
        if not names:
            raise FileNotFoundError(f'No .npz samples in {source}')
        parameters, spectra, wavelengths = [], [], None
        for name in names:
            with np.load(os.path.join(source, name)) as sample:
                if wavelengths is None:
                    wavelengths = sample['wavelength_nm'].astype(float)
                parameters.append([
                    float(sample['R']),
                    float(sample['n_host1']),
                    float(sample['n_host2']),
                ])
                spectra.append(
                    np.hypot(sample['t_real'], sample['t_imag'])
                )
        return (
            np.asarray(parameters, dtype=np.float32),
            np.stack(spectra).astype(np.float32),
            wavelengths,
        )

    with np.load(source) as bundle:
        parameters = np.stack(
            [bundle['R'], bundle['n_host1'], bundle['n_host2']],
            axis=1,
        ).astype(np.float32)
        spectra = np.hypot(bundle['t_real'], bundle['t_imag'])
        return (
            parameters,
            spectra.astype(np.float32),
            bundle['wavelength_nm'].astype(float),
        )


def load_angle_averaged(source):
    """Load a multi-angle dataset as one angle-averaged spectrum per structure.

    theta is the outermost grid axis of the sweep, so the samples are N_theta
    equally sized blocks repeating the same (R, n_host1, n_host2) sequence and
    the average is a plain mean across blocks. Returns the same
    (parameters, spectra, wavelengths) triple as load_arrays, with one row per
    structure.
    """
    parameters, spectra, wavelengths = load_arrays(source)

    thetas = None
    if not os.path.isdir(source):
        with np.load(source) as bundle:
            if 'averaged_theta_deg' in bundle.files:
                # Already collapsed by pack_dataset.py --average-theta.
                return parameters, spectra, wavelengths
            if 'theta_deg' in bundle.files:
                thetas = bundle['theta_deg']
    else:
        thetas = []
        for name in sorted(f for f in os.listdir(source) if f.endswith('.npz')):
            with np.load(os.path.join(source, name)) as sample:
                thetas.append(float(sample['theta_deg']))
        thetas = np.asarray(thetas)

    if thetas is None:
        raise ValueError(f'{source} carries no theta column to average over')

    angles = np.unique(thetas)
    count = len(parameters)
    if count % len(angles):
        raise ValueError(
            f'{count} samples do not split evenly into {len(angles)} angle '
            'blocks; the sweep is incomplete.'
        )
    block = count // len(angles)
    for index in range(1, len(angles)):
        chunk = parameters[index * block:(index + 1) * block]
        if not np.array_equal(chunk, parameters[:block]):
            raise ValueError(
                f'Angle block {index} does not repeat the structures of '
                'block 0; refusing to average.'
            )

    averaged = spectra.reshape(len(angles), block, -1).mean(axis=0)
    return parameters[:block], averaged.astype(np.float32), wavelengths


def load_data(directory):
    """
    Load data from .a and .b files in the 'datasets' directory.
    Returns a tuple containing two lists: data_a_list and data_b_list.

    A packed .npz bundle or a directory of per-sample .npz files is also
    accepted and is reshaped into the same two lists, so existing callers do
    not need to change.
    """
    if directory.endswith('.npz') or (
        os.path.isdir(directory)
        and not any(f.endswith('.a') for f in os.listdir(directory))
    ):
        parameters, spectra, wavelengths = load_arrays(directory)
        data_a_list = [
            {'R': row[0], 'n_host1': row[1], 'n_host2': row[2]}
            for row in parameters
        ]
        data_b_list = [
            pd.DataFrame({0: wavelengths, 1: spectrum})
            for spectrum in spectra
        ]
        return data_a_list, data_b_list

    files = os.listdir(directory)
    a_files = sorted([f for f in files if f.endswith('.a')])
    b_files = sorted([f for f in files if f.endswith('.b')])

    data_a_list = []
    data_b_list = []

    for a_file, b_file in zip(a_files, b_files):
        #data_a = load_a(os.path.join('C:/Users/liyux/Desktop/traindata/batch1', a_file))
        #data_b = load_b(os.path.join('C:/Users/liyux/Desktop/traindata/batch1', b_file))
        data_a = load_a(os.path.join(directory, a_file))
        #print(data_a['R'])
        data_b = load_b(os.path.join(directory, b_file))
        data_a_list.append(data_a)
        # print("\n")
        # print(f'list:{data_a_list}')
        data_b_list.append(data_b)

    return data_a_list, data_b_list

def display_first_three_groups():
    """
    Display the first three groups of files.
    """
    data_a_list, data_b_list = load_data('C:/Users/liyux/Desktop/traindata/batch1')

    if len(data_a_list) < 3 or len(data_b_list) < 3:
        print("Not enough data groups to display.")
        return

    print("First three groups of files:")
    for i in range(3):
        print(f"Group {i + 1}")
        print("Data from .a file:")
        print(data_a_list[i])
        print("Data from .b file:")
        print(data_b_list[i])
        print("------------------------")

if __name__ == "__main__":
    # Execute code only if the file is run as the main module
    display_first_three_groups()
