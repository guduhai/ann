"""统一 Linux/Android 构图入口。默认展示命令，--execute 才执行。"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from src.execution import Runtime, add_execution_arguments, uint

ROOT = Path(__file__).resolve().parent


def build_commands(runtime, data, output, params, count):
    for key in ('K', 'L', 'iterations', 'S', 'R', 'graph_R', 'graph_L', 'graph_C', 'threads'):
        uint(params[key], key, 1)
    if params.get('seed', 20261014) != 20261014:
        raise ValueError('当前构图使用固定 seed=20261014')
    if not (count > 100 and params['K'] <= params['L'] < count
            and 2 <= params['S'] <= params['L'] and 2 * params['S'] < count
            and params['graph_R'] <= params['graph_L'] < count
            and params['graph_C'] >= params['graph_R']):
        raise ValueError('构图参数与实际底库规模不兼容')
    binary = runtime.binary_dir()
    return [
        [binary / 'ann_knn', data, output / 'candidate.knn', params['K'], params['L'],
         params['iterations'], params['S'], params['R'], count, params['threads']],
        [binary / 'ann_build', data, output / 'candidate.knn', output / 'final.graph',
         params['graph_R'], params['graph_L'], params['graph_C'], count, params['threads']],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='128D fvecs 或 EGF32001 底库')
    parser.add_argument('--output', required=True, help='尚不存在的输出目录')
    parser.add_argument('--config', type=Path, default=ROOT / 'build_parameters.json')
    parser.add_argument('--vectors', type=int, default=0, help='0=全部，否则使用前 N 条')
    parser.add_argument('--threads', type=int, help='覆盖配置文件的构图线程数，默认 4')
    parser.add_argument('--execute', action='store_true')
    add_execution_arguments(parser)
    args = parser.parse_args()
    try:
        runtime = Runtime(ROOT, args)
        params = json.loads(args.config.read_text(encoding='utf-8'))
        if args.threads is not None:
            params['threads'] = args.threads
        data, output = runtime.path(args.data), runtime.path(args.output)
        runtime.check_build(data, output)
        available = runtime.vector_rows(data)
        count = uint(args.vectors, 'vectors') or available
        if count > available:
            raise ValueError('vectors 超过底库实际条数')
        commands = build_commands(runtime, data, output, params, count)
        for command in commands:
            print(shlex.join(runtime.wrapped(command, params['threads'])), flush=True)
        if args.execute:
            runtime.execute(commands, output, params['threads'], build=True)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
