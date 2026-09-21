"""Run CST validation from a saved Tandem or CVAE+Tandem structure."""

import argparse
import os
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
    parser = argparse.ArgumentParser(
        description=(
            'Load a saved inverse-design structure, run CST, and export the '
            'validation spectra into the matching wavelength directory.'
        )
    )
    parser.add_argument(
        '--workflow',
        choices=('1peak', '2peak'),
        default='1peak',
        help=(
            'Structure family to validate. The 2peak workflow preserves the '
            'independent n_host1 and n_host2 parameters.'
        ),
    )
    parser.add_argument(
        '--target-wavelength',
        type=float,
        default=410.0,
        help='Requested wavelength in nm; used to resolve 1peak/<wavelength>.',
    )
    parser.add_argument(
        '--peak-wavelengths',
        type=float,
        nargs=2,
        metavar=('PEAK1', 'PEAK2'),
        default=(430.0, 620.0),
        help=(
            'Requested two-peak wavelengths in nm; used to resolve '
            '2peak/<low>_<high> when --workflow 2peak.'
        ),
    )
    parser.add_argument(
        '--method',
        choices=('cvae', 'tandem'),
        default='cvae',
        help='Structure source when --structure-file is omitted.',
    )
    parser.add_argument(
        '--structure-file',
        default=None,
        help=(
            'Optional explicit structure TXT. 1peak expects R,n_host; '
            '2peak expects R,n_host1,n_host2.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help=(
            'Optional CST output directory; otherwise derived from the '
            'selected workflow and target wavelength(s).'
        ),
    )
    parser.add_argument(
        '--angles',
        type=float,
        nargs='+',
        default=[0.0],
        help='Incidence angles in degrees, for example: --angles 0 30 60.',
    )
    parser.add_argument(
        '--resume-id',
        type=int,
        default=0,
        help='Skip run IDs less than or equal to this value.',
    )
    parser.add_argument(
        '--cst-file',
        default=None,
        help=(
            'Optional CST project file, overriding the workflow default. '
            'Use a private copy per process when running cases in parallel, '
            'because each run clears the project result folder.'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate and print resolved inputs without opening CST or writing files.',
    )
    return parser.parse_args(argv)


def wavelength_dir_name(wavelength):
    return f'{wavelength:g}'


def peak_pair(args):
    peaks = np.sort(np.asarray(args.peak_wavelengths, dtype=float))
    if (
        peaks.shape != (2,)
        or not np.all(np.isfinite(peaks))
        or peaks[0] <= 0
        or peaks[1] <= peaks[0]
    ):
        raise ValueError(
            '--peak-wavelengths must contain two distinct positive values.'
        )
    return tuple(float(value) for value in peaks)


def pair_dir_name(peaks):
    return '_'.join(wavelength_dir_name(value) for value in peaks)


def resolve_structure_file(args):
    if args.structure_file:
        return os.path.abspath(args.structure_file)

    if args.workflow == '2peak':
        target_dir = os.path.join(
            REPO_ROOT,
            '2peak',
            pair_dir_name(peak_pair(args)),
        )
    else:
        target_dir = os.path.join(
            REPO_ROOT,
            '1peak',
            wavelength_dir_name(args.target_wavelength),
        )
    if args.method == 'cvae':
        filename = 'cvae_structure.txt'
    elif args.workflow == '2peak':
        filename = 'tandem_structure.txt'
    else:
        filename = 'tandem_pred_0_structure.txt'
    return os.path.join(target_dir, args.method, filename)


def resolve_output_dir(args):
    if args.output_dir:
        return os.path.abspath(args.output_dir)
    if args.workflow == '2peak':
        return os.path.join(
            REPO_ROOT,
            '2peak',
            pair_dir_name(peak_pair(args)),
            'cst',
        )
    return os.path.join(
        REPO_ROOT,
        '1peak',
        wavelength_dir_name(args.target_wavelength),
        'cst',
    )


def resolve_cst_project_file(workflow, override=None):
    if override:
        return os.path.abspath(override)
    if workflow == '1peak':
        project_name = 'model2021.cst'
    else:
        project_name = cst_file
    if os.path.isabs(project_name):
        return project_name
    return os.path.join(SCRIPT_DIR, project_name)


def load_structure_file(path, workflow):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Structure file not found: {path}')

    with open(path, 'r', encoding='utf-8') as structure_file:
        column_names = structure_file.readline().strip().split()
    values = np.loadtxt(path, skiprows=1, ndmin=2)
    if values.shape[0] != 1:
        raise ValueError(
            f'Expected exactly one structure row in {path}, got {values.shape[0]}.'
        )

    structure = dict(zip(column_names, values[0]))
    required = (
        {'R', 'n_host1', 'n_host2'}
        if workflow == '2peak'
        else {'R', 'n_host'}
    )
    missing = required - structure.keys()
    if missing:
        raise ValueError(
            f'Missing required structure columns {sorted(missing)} in {path}.'
        )

    result = {
        name: float(structure[name])
        for name in sorted(required)
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError(f'Non-finite structure value in {path}.')
    return result


def configure_cst_parameters(structure, workflow):
    update_value_in_list_of_dicts(parameter_values, 'R', structure['R'])
    if workflow == '2peak':
        # The archived 2peak sweep kept the legacy single-host parameter at
        # 2.3 while independently sweeping the two physical host layers.
        update_value_in_list_of_dicts(parameter_values, 'n_host', 2.3)
        update_value_in_list_of_dicts(
            parameter_values,
            'n_host1',
            structure['n_host1'],
        )
        update_value_in_list_of_dicts(
            parameter_values,
            'n_host2',
            structure['n_host2'],
        )
    else:
        refractive_index = structure['n_host']
        update_value_in_list_of_dicts(
            parameter_values,
            'n_host',
            refractive_index,
        )
        update_value_in_list_of_dicts(
            parameter_values,
            'n_host1',
            refractive_index,
        )
        update_value_in_list_of_dicts(
            parameter_values,
            'n_host2',
            refractive_index,
        )


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
    """Open a CST project, allowing the previous modeler time to exit.

    CST occasionally returns from ``project.close()`` before its modeler and
    VBA window have fully shut down.  An immediate reopen can then fail with
    ``Could not create VBA edit window``.  Retrying the same, already-closed
    project after a short bounded delay avoids mixing or skipping angle runs.
    """
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
        f'Failed to open CST project after {attempts} attempts: '
        f'{project_file}'
    ) from last_error


def save_run_configuration(
    output_dir,
    structure_file,
    workflow,
    method,
    target_wavelength,
    peak_wavelengths,
    structure,
    angles,
):
    creatresultFolder(output_dir)
    structure_copy = os.path.join(output_dir, 'cst_input_structure.txt')
    if workflow == '2peak':
        structure_names = ('R', 'n_host1', 'n_host2')
    else:
        structure_names = ('R', 'n_host')
    np.savetxt(
        structure_copy,
        np.array(
            [[structure[name] for name in structure_names]],
            dtype=float,
        ),
        delimiter='\t',
        fmt='%.10f',
        header='\t'.join(structure_names),
        comments='',
    )

    manifest_path = os.path.join(output_dir, 'cst_run_config.txt')
    with open(manifest_path, 'w', encoding='utf-8') as manifest:
        manifest.write(f'workflow\t{workflow}\n')
        if workflow == '2peak':
            manifest.write(
                'target_peak_wavelengths_nm\t'
                + ','.join(
                    f'{value:.10f}' for value in peak_wavelengths
                )
                + '\n'
            )
            manifest.write('training_reference_angle_deg\t0\n')
            manifest.write('periodic_scan_direction\tinward\n')
        else:
            manifest.write(
                f'target_wavelength_nm\t{target_wavelength:.10f}\n'
            )
        manifest.write(f'source_method\t{method}\n')
        manifest.write(f'source_structure_file\t{structure_file}\n')
        for name in structure_names:
            manifest.write(f'{name}\t{structure[name]:.10f}\n')
        manifest.write(
            'angles_deg\t' + ','.join(f'{angle:g}' for angle in angles) + '\n'
        )


def main(argv=None):
    args = parse_args(argv)
    structure_file = resolve_structure_file(args)
    output_dir = resolve_output_dir(args)
    project_file = resolve_cst_project_file(args.workflow, args.cst_file)
    peaks = peak_pair(args) if args.workflow == '2peak' else ()
    structure = load_structure_file(structure_file, args.workflow)
    angles = tuple(float(angle) for angle in args.angles)
    if not angles or not all(np.isfinite(angle) for angle in angles):
        raise ValueError('At least one finite incidence angle is required.')

    configure_cst_parameters(structure, args.workflow)

    print(f'Structure file: {structure_file}')
    print(f'CST project: {project_file}')
    print(f'Output directory: {output_dir}')
    print(
        'Structure: '
        + ', '.join(
            f'{name}={value:.10f}'
            for name, value in structure.items()
        )
    )
    if args.workflow == '2peak':
        print(
            '2peak validation uses independent n_host1/n_host2; '
            'the primary training reference is theta=0 deg.'
        )
    print(f'Angles: {angles}')

    if args.dry_run:
        print('Dry run complete; CST was not opened and no files were written.')
        return

    save_run_configuration(
        output_dir=output_dir,
        structure_file=structure_file,
        workflow=args.workflow,
        method=args.method,
        target_wavelength=args.target_wavelength,
        peak_wavelengths=peaks,
        structure=structure,
        angles=angles,
    )
    plot_dir = os.path.join(output_dir, 'txt')

    active_lib_path = configured_lib_path
    if active_lib_path is None:
        active_lib_path = (
            r'C:\Program Files (x86)\CST Studio Suite 2023\AMD64'
            r'\python_cst_libraries'
        )
    print(f'CST Python library: {active_lib_path}')
    cst_utils.lib_path = active_lib_path
    sys.path.append(active_lib_path)

    import cst
    import cst.interface
    import cst.results

    print(cst.__file__)
    project = cst.interface.DesignEnvironment()
    project.print_version()
    current_project = None

    try:
        try:
            current_project = project.get_open_project(project_file)
        except Exception:
            current_project = open_project_with_retry(project, project_file)

        for run_id, angle in enumerate(angles, start=1):
            if run_id <= args.resume_id:
                continue

            update_value_in_list_of_dicts(parameter_values, 'theta', angle)
            fid = f'{run_id:05d}'
            print(
                f'id={fid} R={structure["R"]:.10f} '
                f'n_host1='
                f'{next(item["n_host1"] for item in parameter_values if "n_host1" in item):.10f} '
                f'n_host2='
                f'{next(item["n_host2"] for item in parameter_values if "n_host2" in item):.10f} '
                f'theta={angle:g}'
            )
            save_parameters(
                fid,
                parameter_values,
                data_dir=output_dir,
                plot_dir=plot_dir,
            )

            current_project.close()
            current_project = None
            # ``close`` is asynchronous in the CST Linux interface.  Give the
            # old modeler/VBA window time to exit before removing results and
            # reopening the shared project for the next angle.
            time.sleep(3.0)
            delete_files_in_result_folder(project_file)
            current_project = open_project_with_retry(project, project_file)
            if args.workflow == '2peak':
                set_periodic_scan_direction(current_project, 'inward')
            cst_parameter_sweep_and_export(
                current_project,
                fid,
                parameter_values,
                data_dir=output_dir,
                plot_dir=plot_dir,
                project_file=project_file,
            )
    finally:
        if current_project is not None:
            try:
                current_project.close()
            except Exception:
                pass
        project.close()

    print(f'CST validation outputs saved to: {output_dir}')


if __name__ == '__main__':
    main()
