"""Generate the 2peak CST training dataset.

The grid is theta (outermost), then R, then n_host1, then n_host2. Within one
theta the R/n1/n2 ordering reproduces the archived ``cstdata_10496`` dataset
(00001 = R 2.5 / n1 1.5 / n2 1.5, 00017 = R 2.5 / n1 1.6 / n2 1.5), and
putting theta outermost means the theta=0 block keeps exactly the ids a
theta-free sweep produces -- so adding angles later never invalidates work
already on disk. Sample ids are a pure function of the grid, which keeps
``--resume`` and sharding consistent across restarts.

Each sample is written as a compressed ``<id>.npz`` by default; ``--format
text`` reproduces the archived ``<id>.a`` / ``<id>.b`` / ``<id>.c`` layout.
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np

import utils as cst_utils
from user_setting import cst_file, lib_path as configured_lib_path, parameter_values
from utils import (
    creatresultFolder,
    cst_parameter_sweep_and_export,
    delete_files_in_result_folder,
    save_parameters,
    update_value_in_list_of_dicts,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r-start', type=float, default=2.5)
    parser.add_argument('--r-stop', type=float, default=6.5)
    parser.add_argument('--r-step', type=float, default=0.1)
    parser.add_argument('--n-start', type=float, default=1.5)
    parser.add_argument(
        '--n-stop',
        type=float,
        default=3.05,
        help='Inclusive upper bound of n_host1/n_host2 (1peak uses 3.05).',
    )
    parser.add_argument(
        '--n-step',
        type=float,
        default=0.05,
        help='n_host1/n_host2 increment (1peak uses 0.05, archived 2peak 0.1).',
    )
    parser.add_argument(
        '--n-host',
        type=float,
        default=2.3,
        help='Legacy single-host parameter, held fixed as in the archive.',
    )
    parser.add_argument('--theta', type=float, default=0.0)
    parser.add_argument(
        '--theta-list',
        default=None,
        help=(
            'Incidence angles in degrees, e.g. "0 30 60" or "0,30,60". theta '
            'is the OUTERMOST grid axis, so the theta=0 block keeps exactly '
            'the ids a theta-free sweep produced and already-solved samples '
            'stay valid. Overrides --theta.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Sample directory; defaults to 2peak/data/cstdata_<count>/data.',
    )
    parser.add_argument(
        '--cst-file',
        default=None,
        help='CST project to drive; use a private copy per parallel worker.',
    )
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument(
        '--num-shards',
        type=int,
        default=1,
        help='Workers stride the global id list: shard i takes i, i+N, i+2N.',
    )
    parser.add_argument('--start-id', type=int, default=1)
    parser.add_argument(
        '--end-id',
        type=int,
        default=0,
        help='Inclusive last global id; 0 means the last grid point.',
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Re-solve samples whose .b export already exists.',
    )
    parser.add_argument(
        '--reopen-every',
        type=int,
        default=1,
        help=(
            'Close, clear results and reopen the project every N samples. '
            '1 matches the validated main.py behaviour; larger values trade '
            'safety for throughput and should be checked against known '
            'samples first.'
        ),
    )
    parser.add_argument(
        '--format',
        choices=('npz', 'text'),
        default='npz',
        help=(
            'npz (default) keeps one compressed <id>.npz per sample and drops '
            'the CST text exports and preview PNGs, which is ~10x smaller and '
            'avoids ~2 matplotlib renders per sample. text reproduces the '
            'archived .a/.b/.c layout.'
        ),
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help=(
            'Exit cleanly after this many samples so the driver can restart '
            'the process and reclaim memory. CST result readers leak across '
            'exports -- a shard left running grew past 25 GB and starved the '
            'solver into "Not enough memory". 0 disables the bound. The exit '
            'status is 75 when work remains, 0 when the shard is finished.'
        ),
    )
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    if args.r_step <= 0 or args.n_step <= 0:
        parser.error('--r-step and --n-step must be positive')
    if args.r_stop < args.r_start or args.n_stop < args.n_start:
        parser.error('grid stop values must not be below the start values')
    if args.num_shards < 1:
        parser.error('--num-shards must be at least 1')
    if not 0 <= args.shard < args.num_shards:
        parser.error('--shard must satisfy 0 <= shard < num-shards')
    if args.start_id < 1:
        parser.error('--start-id is 1-based')
    if args.reopen_every < 0:
        parser.error('--reopen-every must be non-negative')
    if args.max_samples < 0:
        parser.error('--max-samples must be non-negative')
    return args


def axis(start, stop, step):
    """Inclusive axis that tolerates floating-point step accumulation.

    ``stop`` is always the last value.  When it is not a whole number of
    steps above ``start`` it is appended, so ``1.5 .. 3.05`` with step 0.1
    yields the 0.1 grid plus the 3.05 endpoint rather than silently stopping
    at 3.0.
    """
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    values = np.round(start + step * np.arange(count), 10)
    if values[-1] < stop - 1e-9:
        values = np.append(values, round(stop, 10))
    return values


def theta_axis(args):
    if not args.theta_list:
        return [float(args.theta)]
    raw = args.theta_list.replace(',', ' ').split()
    values = [float(value) for value in raw]
    if not values:
        raise ValueError('--theta-list is empty')
    return values


def build_grid(args):
    r_values = axis(args.r_start, args.r_stop, args.r_step)
    n_values = axis(args.n_start, args.n_stop, args.n_step)
    theta_values = theta_axis(args)
    # theta outermost: the first len(r)*len(n)**2 ids are exactly the ids a
    # theta-free sweep produces, so earlier theta=0 output stays resumable.
    grid = [
        (float(theta), float(r), float(n1), float(n2))
        for theta in theta_values
        for r in r_values
        for n1 in n_values
        for n2 in n_values
    ]
    return r_values, n_values, theta_values, grid


def default_output_dir(total):
    return os.path.join(
        REPO_ROOT,
        '2peak',
        'data',
        f'cstdata_{total}',
        'data',
    )


def resolve_cst_project_file(override):
    project_name = override or cst_file
    if os.path.isabs(project_name):
        return project_name
    return os.path.join(SCRIPT_DIR, project_name)


def pack_sample_npz(output_dir, fid, structure, remove_text=True):
    """Fold one sample's CST text exports into a single compressed ``.npz``.

    ``<id>.b`` and ``<id>.c`` are three-column ``x, Re, Im`` tables written by
    ``export1D``.  Both are stored as float32 — ample for spectra in [0, 1] —
    together with the structure parameters, then the text and PNG files are
    dropped.  Returns the path written.
    """
    transmission_path = os.path.join(output_dir, f'{fid}.b')
    reflection_path = os.path.join(output_dir, f'{fid}.c')
    transmission = np.loadtxt(transmission_path)
    if transmission.shape != (1001, 3):
        raise ValueError(
            f'Unexpected transmission export shape {transmission.shape} '
            f'in {transmission_path}.'
        )

    arrays = {
        'wavelength_nm': transmission[:, 0].astype(np.float32),
        't_real': transmission[:, 1].astype(np.float32),
        't_imag': transmission[:, 2].astype(np.float32),
        'R': np.float32(structure[0]),
        'n_host1': np.float32(structure[1]),
        'n_host2': np.float32(structure[2]),
        'theta_deg': np.float32(structure[3]),
    }
    if os.path.isfile(reflection_path):
        reflection = np.loadtxt(reflection_path)
        arrays['r_real'] = reflection[:, 1].astype(np.float32)
        arrays['r_imag'] = reflection[:, 2].astype(np.float32)

    npz_path = os.path.join(output_dir, f'{fid}.npz')
    # Write to a temporary name first so an interrupted run never leaves a
    # truncated .npz that --resume would mistake for a finished sample.
    partial_path = npz_path + '.partial'
    np.savez_compressed(partial_path, **arrays)
    os.replace(partial_path + '.npz', npz_path)

    if remove_text:
        for suffix in ('.a', '.b', '.c', '.b.png', '.c.png'):
            stale = os.path.join(output_dir, f'{fid}{suffix}')
            if os.path.isfile(stale):
                os.remove(stale)
    return npz_path


def clear_solver_temp(project_file):
    """Drop the project's ``Temp`` directory between samples.

    ``delete_files_in_result_folder`` only clears ``Result``.  ``Temp`` is
    pure solver scratch and is never read back, but it grows without bound
    across a sweep — the shared project had accumulated 35 GB — which would
    fill the disk long before an 11k-sample run finishes.  This is called at
    the same point as the result clear, with the project closed.
    """
    temp_dir = os.path.join(os.path.splitext(project_file)[0], 'Temp')
    if not os.path.isdir(temp_dir):
        return
    for entry in os.listdir(temp_dir):
        path = os.path.join(temp_dir, entry)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError:
            # A file still held by the exiting modeler is dropped next round.
            pass


def set_periodic_scan_direction(current_project, direction='inward'):
    vba_command = '\n'.join((
        'Sub Main',
        'With Boundary',
        f'.SetPeriodicBoundaryAnglesDirection "{direction}"',
        'End With',
        'End Sub',
    ))
    current_project.schematic.execute_vba_code(vba_command)


def open_project_with_retry(
    design_environment,
    project_file,
    *,
    attempts=4,
    retry_delay_seconds=5.0,
):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return design_environment.open_project(project_file)
        except Exception as error:
            last_error = error
            if attempt == attempts:
                break
            print(
                f'CST project open attempt {attempt}/{attempts} failed: '
                f'{error}. Retrying in {retry_delay_seconds:g} s.',
                flush=True,
            )
            time.sleep(retry_delay_seconds)
    raise RuntimeError(
        f'Failed to open CST project after {attempts} attempts: {project_file}'
    ) from last_error


def pending_samples(args, grid, output_dir):
    """Global ids this shard still has to solve, in ascending order."""
    end_id = args.end_id or len(grid)
    selected = []
    for index in range(len(grid)):
        sample_id = index + 1
        if sample_id < args.start_id or sample_id > end_id:
            continue
        if index % args.num_shards != args.shard:
            continue
        if not args.no_resume:
            suffix = '.npz' if args.format == 'npz' else '.b'
            exported = os.path.join(output_dir, f'{sample_id:05d}{suffix}')
            if os.path.isfile(exported):
                continue
        selected.append(sample_id)
    return selected


def main(argv=None):
    args = parse_args(argv)
    r_values, n_values, theta_values, grid = build_grid(args)
    total = len(grid)
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else default_output_dir(total)
    )
    project_file = resolve_cst_project_file(args.cst_file)
    plot_dir = os.path.join(os.path.dirname(output_dir), 'txt')

    print(f'R axis        : {len(r_values)} values '
          f'{r_values[0]:g} .. {r_values[-1]:g} step {args.r_step:g}')
    print(f'n axis        : {len(n_values)} values '
          f'{n_values[0]:g} .. {n_values[-1]:g} step {args.n_step:g}')
    print(f'Grid points   : {total}')
    print(f'Output dir    : {output_dir}')
    print(f'CST project   : {project_file}')
    print(f'Shard         : {args.shard + 1}/{args.num_shards}')
    print(f'theta axis    : {len(theta_values)} values '
          + ', '.join(f'{value:g}' for value in theta_values)
          + f' deg (outermost); n_host fixed at {args.n_host:g}')

    todo = pending_samples(args, grid, output_dir)
    print(f'Samples to run: {len(todo)}')
    if todo:
        print(f'First / last  : {todo[0]:05d} / {todo[-1]:05d}')

    if args.dry_run:
        for sample_id in todo[:5]:
            theta_value, r_value, n1_value, n2_value = grid[sample_id - 1]
            print(
                f'  {sample_id:05d} theta={theta_value:g} R={r_value:g} '
                f'n_host1={n1_value:g} n_host2={n2_value:g}'
            )
        if len(todo) > 5:
            print(f'  ... {len(todo) - 5} more')
        print('Dry run complete; CST was not opened and no files were written.')
        return

    if not todo:
        print('Nothing to do.')
        return

    creatresultFolder(output_dir)

    if args.format == 'npz':
        # The text exports and preview PNGs are folded into <id>.npz right
        # after each solve, so do not spend time producing either.
        cst_utils.make_preview_plots = False
        cst_utils.save_txt_for_plot = False

    active_lib_path = configured_lib_path
    if active_lib_path is None:
        active_lib_path = (
            r'C:\Program Files (x86)\CST Studio Suite 2023\AMD64'
            r'\python_cst_libraries'
        )
    cst_utils.lib_path = active_lib_path
    sys.path.append(active_lib_path)

    import cst
    import cst.interface
    import cst.results

    design_environment = cst.interface.DesignEnvironment()
    design_environment.print_version()
    current_project = None
    started = time.time()
    stopped_early = False

    update_value_in_list_of_dicts(parameter_values, 'n_host', args.n_host)

    try:
        try:
            current_project = design_environment.get_open_project(project_file)
        except Exception:
            current_project = open_project_with_retry(
                design_environment,
                project_file,
            )
        set_periodic_scan_direction(current_project, 'inward')

        for position, sample_id in enumerate(todo):
            if args.max_samples and position >= args.max_samples:
                stopped_early = True
                print(
                    f'Reached --max-samples {args.max_samples}; exiting so '
                    f'the driver can restart this shard '
                    f'({len(todo) - position} left in its queue).',
                    flush=True,
                )
                break
            theta_value, r_value, n1_value, n2_value = grid[sample_id - 1]
            update_value_in_list_of_dicts(parameter_values, 'theta', theta_value)
            update_value_in_list_of_dicts(parameter_values, 'R', r_value)
            update_value_in_list_of_dicts(parameter_values, 'n_host1', n1_value)
            update_value_in_list_of_dicts(parameter_values, 'n_host2', n2_value)

            fid = f'{sample_id:05d}'
            elapsed = time.time() - started
            rate = elapsed / position if position else 0.0
            remaining = rate * (len(todo) - position)
            print(
                f'[{position + 1}/{len(todo)}] id={fid} theta={theta_value:g} '
                f'R={r_value:g} n_host1={n1_value:g} n_host2={n2_value:g} '
                f'elapsed={elapsed / 60:.1f} min '
                f'eta={remaining / 60:.1f} min',
                flush=True,
            )
            save_parameters(
                fid,
                parameter_values,
                data_dir=output_dir,
                plot_dir=plot_dir,
            )

            if args.reopen_every and position % args.reopen_every == 0:
                current_project.close()
                current_project = None
                # ``close`` is asynchronous in the CST Linux interface; give
                # the modeler time to exit before clearing and reopening.
                time.sleep(3.0)
                delete_files_in_result_folder(project_file)
                clear_solver_temp(project_file)
                current_project = open_project_with_retry(
                    design_environment,
                    project_file,
                )
                set_periodic_scan_direction(current_project, 'inward')

            cst_parameter_sweep_and_export(
                current_project,
                fid,
                parameter_values,
                data_dir=output_dir,
                plot_dir=plot_dir,
                project_file=project_file,
            )

            if args.format == 'npz':
                pack_sample_npz(
                    output_dir,
                    fid,
                    (r_value, n1_value, n2_value, theta_value),
                )
    finally:
        if current_project is not None:
            try:
                current_project.close()
            except Exception:
                pass
        design_environment.close()

    print(
        f'Shard {args.shard + 1}/{args.num_shards} finished '
        f'{len(todo)} samples in {(time.time() - started) / 3600:.2f} h -> '
        f'{output_dir}'
    )
    if stopped_early:
        # Distinct from both success and a crash: the driver restarts on this.
        return 75
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
