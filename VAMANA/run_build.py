"""Print the original build commands; --execute explicitly runs them.

Use a new output directory. Existing graphs are never overwritten by this entry.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='HNSW: HDF5; NSG/NSSG: base fvecs; VAMANA: base fbin')
    parser.add_argument('--output', type=Path, required=True, help='New build output directory')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    params = json.loads((ROOT / 'build_parameters.json').read_text())
    index = json.loads((ROOT / 'experiment.json').read_text())['index']
    output = args.output.resolve()
    data = args.data.resolve()
    binary = ROOT / 'build/bin'
    if output.exists():
        parser.error('Output directory already exists; choose a new directory.')
    if not data.is_file():
        parser.error('Input data file does not exist.')
    if index == 'HNSW':
        commands = [[binary / 'ann_build', '--dataset', data,
                     '--index', output / 'hnsw-1m.bin', '--result', output / 'build.json',
                     '--m', params['M'], '--ef-construction', params['ef_construction'],
                     '--seed', params['seed'], '--build-threads', params['threads']]]
    elif index in ('NSG', 'NSSG'):
        knn = output / 'sift_200nn.graph'
        commands = [[binary / 'ann_knn', data, knn, params['K'], params['L'],
                     params['iterations'], params['S'], params['R']]]
        if index == 'NSG':
            commands.append([binary / 'ann_build', data, knn, params['graph_L'],
                             params['graph_R'], params['graph_C'], output / 'sift.nsg'])
        else:
            commands.append([binary / 'ann_build', data, knn, params['graph_L'],
                             params['graph_R'], params['angle'], output / 'sift.ssg', params['seed']])
    else:
        commands = [[binary / 'ann_build', '--data_type', 'float', '--dist_fn', 'l2',
                     '--data_path', data, '--index_path_prefix', output / 'sift1m_r50_l70_a2',
                     '-R', params['R'], '-L', params['L'], '--alpha', params['alpha'],
                     '-T', params['threads'], '--build_PQ_bytes', 0]]
    commands = [[str(x) for x in command] for command in commands]
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.execute:
        output.mkdir(parents=True, exist_ok=False)
        environment = dict(os.environ, OMP_NUM_THREADS=str(params['threads']),
                           OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', LC_ALL='C')
        for command in commands:
            subprocess.run(command, env=environment, check=True)


if __name__ == '__main__':
    main()
