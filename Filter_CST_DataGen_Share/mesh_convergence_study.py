"""Mesh-convergence study for one 2peak structure.

Background
----------
The archived 2peak dataset and a fresh CST 2026 re-run of the *same* structure
disagree by ~1.3-1.7 nm in resonance position. A global StepsPerWave sweep on
06883 showed the two transmission dips drift monotonically and never converge
(431.8 -> 428.8 nm for dip1 across 5/2 -> 24/12), so the dataset itself is an
unconverged, version-specific result that cannot be reproduced by tuning alone.

The nanoparticles carry a *local* tetrahedral mesh (group "meshgroup4",
Size = R/5 ~ 1 nm), pinned independently of the global StepsPerWave. That local
mesh is the suspected convergence bottleneck. This script sweeps it (and/or the
global steps) at a fixed counterpart and reports where the dips settle.

Each level forces a genuinely fresh solve: close project, wipe the cached Result
folder, reopen, re-apply mesh + parameters. Without that CST returns the previous
cached spectrum (a ~2 s "solve") and every level looks identical.

Modes
-----
  --mode global : sweep global StepsPerWave (near,far) at fixed local Size.
  --mode local  : sweep local particle Size = R/<divisor> at fixed global.
  --mode custom : explicit per-level specs "near,far,divisor;..." .

Run (from Filter_CST_DataGen_Share, needs xvfb + a CST-capable python):
  MPLBACKEND=Agg xvfb-run -a python -u mesh_convergence_study.py \
      --mode local --global-fixed 16,8 --local-divisors 5;10;16;24
"""

import argparse
import os
import sys
import time

import numpy as np

import utils as cst_utils
from user_setting import cst_file, lib_path as configured_lib_path, parameter_values, treeItems
from utils import (
    creatresultFolder,
    delete_files_in_result_folder,
    export1D,
    generate_command,
    update_value_in_list_of_dicts,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Local particle mesh group as defined in the model history.
PARTICLE_MESH_GROUP = 'meshgroup4'

# Dataset reference dip positions for 06883 (nm), for the printout only.
DATASET_REFERENCE = {'06883': (430.1, 623.2)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--structure-file',
        default=os.path.join(
            REPO_ROOT, '2peak', 'cst_repro_dataset', '06883_structure.txt'
        ),
        help='2peak structure TXT with columns R, n_host1, n_host2.',
    )
    p.add_argument('--output-dir', default=None,
                   help='Per-level CST exports. Defaults next to the structure.')
    p.add_argument('--mode', choices=('global', 'local', 'custom'),
                   default='local')
    p.add_argument('--global-fixed', default='16,8',
                   help='near,far used when --mode local.')
    p.add_argument('--local-fixed', type=int, default=5,
                   help='Size divisor R/<d> used when --mode global.')
    p.add_argument('--global-levels', default='5,2;10,5;16,8;24,12',
                   help='Semicolon near,far list when --mode global.')
    p.add_argument('--local-divisors', default='5;10;16;24',
                   help='Semicolon divisor list when --mode local.')
    p.add_argument('--custom-specs', default=None,
                   help='"near,far,divisor;..." when --mode custom.')
    p.add_argument('--angle', type=float, default=0.0,
                   help='Incidence angle in degrees (training reference is 0).')
    p.add_argument('--solver', choices=('auto', 'direct', 'iterative'),
                   default='auto',
                   help=(
                       'FD linear solver. The iterative solver stalls on the '
                       'ill-conditioned fine particle mesh; use "direct" for a '
                       'trustworthy converged reference.'
                   ))
    return p.parse_args(argv)


def build_specs(args):
    """Return list of (near, far, divisor) triples, coarse -> fine."""
    if args.mode == 'global':
        near_far = [tuple(int(v) for v in c.split(','))
                    for c in args.global_levels.split(';') if c.strip()]
        return [(n, f, args.local_fixed) for n, f in near_far]
    if args.mode == 'local':
        near, far = (int(v) for v in args.global_fixed.split(','))
        divisors = [int(v) for v in args.local_divisors.split(';') if v.strip()]
        return [(near, far, d) for d in divisors]
    # custom
    specs = []
    for chunk in (args.custom_specs or '').split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        n, f, d = (int(v) for v in chunk.split(','))
        specs.append((n, f, d))
    if not specs:
        raise ValueError('No specs parsed for --mode custom.')
    return specs


def load_structure(path):
    with open(path, 'r', encoding='utf-8') as fh:
        names = fh.readline().split()
    values = np.loadtxt(path, skiprows=1, ndmin=2)
    if values.shape[0] != 1:
        raise ValueError(f'Expected one structure row in {path}.')
    row = dict(zip(names, values[0]))
    for key in ('R', 'n_host1', 'n_host2'):
        if key not in row:
            raise ValueError(f'Structure {path} is missing column {key}.')
    return {k: float(row[k]) for k in ('R', 'n_host1', 'n_host2')}


def configure_parameters(structure, angle):
    update_value_in_list_of_dicts(parameter_values, 'R', structure['R'])
    update_value_in_list_of_dicts(parameter_values, 'n_host', 2.3)
    update_value_in_list_of_dicts(parameter_values, 'n_host1', structure['n_host1'])
    update_value_in_list_of_dicts(parameter_values, 'n_host2', structure['n_host2'])
    update_value_in_list_of_dicts(parameter_values, 'theta', angle)


def global_mesh_vba(near, far):
    return '\n'.join((
        'Sub Main',
        'With Mesh',
        '.MeshType "Tetrahedral"',
        '.SetCreator "High Frequency"',
        'End With',
        'With MeshSettings',
        '.SetMeshType "Tet"',
        f'.Set "StepsPerWaveNear", "{near}"',
        f'.Set "StepsPerWaveFar", "{far}"',
        f'.Set "StepsPerBoxNear", "{near}"',
        f'.Set "StepsPerBoxFar", "{far}"',
        'End With',
        'End Sub',
    ))


def local_mesh_vba(divisor, group=PARTICLE_MESH_GROUP):
    """Set the local max cell size on the nanoparticle mesh group."""
    return '\n'.join((
        'Sub Main',
        'With MeshSettings',
        f'With .ItemMeshSettings ("group${group}")',
        '.SetMeshType "Tet"',
        f'.Set "Size", "R/{divisor}"',
        'End With',
        'End With',
        'End Sub',
    ))


def solver_type_vba(solver):
    """Force the FD linear solver type: Auto | Direct | Iterative."""
    mapping = {'auto': 'Auto', 'direct': 'Direct', 'iterative': 'Iterative'}
    return '\n'.join((
        'Sub Main',
        'With FDSolver',
        f'.Type "{mapping[solver]}"',
        'End With',
        'End Sub',
    ))


def scan_direction_vba(direction='inward'):
    return '\n'.join((
        'Sub Main',
        'With Boundary',
        f'.SetPeriodicBoundaryAnglesDirection "{direction}"',
        'End With',
        'End Sub',
    ))


def find_dip(lam, mag, lo, hi):
    """Minimum of a transmission magnitude within [lo, hi] nm."""
    mask = (lam >= lo) & (lam <= hi)
    sub_lam, sub_mag = lam[mask], mag[mask]
    i = sub_mag.argmin()
    return float(sub_lam[i]), float(sub_mag[i])


def main(argv=None):
    args = parse_args(argv)
    structure_file = os.path.abspath(args.structure_file)
    structure = load_structure(structure_file)
    specs = build_specs(args)

    stem = os.path.basename(structure_file).split('_')[0]
    reference = DATASET_REFERENCE.get(stem)

    output_root = args.output_dir or os.path.join(
        os.path.dirname(structure_file), f'{stem}_meshconv_{args.mode}'
    )
    creatresultFolder(output_root)

    project_file = os.path.join(SCRIPT_DIR, cst_file)
    configure_parameters(structure, args.angle)

    print(f'Structure file : {structure_file}')
    print(f'Structure      : R={structure["R"]:.4f} '
          f'n_host1={structure["n_host1"]:.4f} n_host2={structure["n_host2"]:.4f}')
    print(f'CST project    : {project_file}')
    print(f'Output root    : {output_root}')
    print(f'Mode           : {args.mode}')
    print(f'Solver         : {args.solver}')
    print(f'Specs (near,far,R/div): {specs}', flush=True)
    if reference:
        print(f'Dataset dips   : {reference[0]:.1f} / {reference[1]:.1f} nm')

    active_lib_path = configured_lib_path or (
        r'C:\Program Files (x86)\CST Studio Suite 2023\AMD64\python_cst_libraries'
    )
    cst_utils.lib_path = active_lib_path
    sys.path.append(active_lib_path)

    import cst
    import cst.interface  # noqa: F401
    import cst.results  # noqa: F401

    env = cst.interface.DesignEnvironment()
    env.print_version()

    project = None
    results = []
    try:
        params_vba = generate_command(parameter_values)

        for near, far, divisor in specs:
            tag = f'g{near}_{far}_L{divisor}'
            level_dir = os.path.join(output_root, tag)
            plot_dir = os.path.join(level_dir, 'txt')
            creatresultFolder(level_dir)

            print(f'\n=== global {near}/{far}  local R/{divisor} ===', flush=True)
            if project is not None:
                try:
                    project.close()
                except Exception:
                    pass
            delete_files_in_result_folder(project_file)
            project = env.open_project(project_file)
            project.schematic.execute_vba_code(scan_direction_vba('inward'))
            project.schematic.execute_vba_code(solver_type_vba(args.solver))
            project.schematic.execute_vba_code(global_mesh_vba(near, far))
            project.schematic.execute_vba_code(local_mesh_vba(divisor))
            project.schematic.execute_vba_code(params_vba)

            t0 = time.time()
            project.modeler.run_solver()
            solve_s = time.time() - t0
            print(f'  solve wall-time: {solve_s:.1f} s', flush=True)

            t_item = treeItems[0]
            data_file = os.path.join(level_dir, '00001.b')
            txt_file = os.path.join(
                plot_dir, '00001_b_' + cst_utils.safe_filename(
                    t_item[t_item.rfind('\\') + 1:]) + '.txt')
            creatresultFolder(plot_dir)
            ok = export1D(cst_file=project_file, treeItem=t_item,
                          data_file=data_file, txt_file=txt_file)
            if not ok:
                print('  transmission result missing; skipping level')
                continue

            spectrum = np.loadtxt(data_file, comments='#')
            lam = spectrum[:, 0]
            mag = np.hypot(spectrum[:, 1], spectrum[:, 2])
            d1, t1 = find_dip(lam, mag, 410, 445)
            d2, t2 = find_dip(lam, mag, 600, 640)
            results.append((near, far, divisor, d1, t1, d2, t2, solve_s))
            print(f'  dip1 {d1:.2f} nm (T={t1:.3f})  dip2 {d2:.2f} nm (T={t2:.3f})',
                  flush=True)
    finally:
        if project is not None:
            try:
                project.close()
            except Exception:
                pass
        env.close()

    print('\n================ convergence summary ================')
    print(f'{"near":>4} {"far":>4} {"R/d":>4} {"dip1":>8} {"dip2":>8}'
          f' {"d(dip1)":>8} {"d(dip2)":>8} {"solve_s":>8}')
    prev = None
    for near, far, div, d1, t1, d2, t2, s in results:
        dd1 = dd2 = float('nan')
        if prev is not None:
            dd1, dd2 = d1 - prev[0], d2 - prev[1]
        print(f'{near:4d} {far:4d} {div:4d} {d1:8.2f} {d2:8.2f}'
              f' {dd1:8.3f} {dd2:8.3f} {s:8.1f}')
        prev = (d1, d2)
    if reference and results:
        d1, d2 = results[-1][3], results[-1][5]
        print(f'\nfinest vs dataset: dip1 {d1:.2f} vs {reference[0]:.1f} '
              f'({d1 - reference[0]:+.2f} nm), '
              f'dip2 {d2:.2f} vs {reference[1]:.1f} ({d2 - reference[1]:+.2f} nm)')
    print('Converged when successive d(dip) rows both fall below your tolerance '
          '(e.g. < 0.3 nm).')


if __name__ == '__main__':
    main()
