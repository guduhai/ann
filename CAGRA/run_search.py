"""统一 Linux/Android Top-10 无重排查询入口。--output 为输出前缀。"""
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
METHODS = {'exact': 'float32', 'float32': 'float32', 'int8': 'int8',
           'pca80': 'pca', 'pca': 'pca', 'pq64': 'pq', 'pq': 'pq', 'local': 'local'}


def make_command(args, runtime):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['index'] != 'CAGRA' or config.get('k', 10) != 10 or config.get('threads', 1) != 1 or config.get('rerank', False):
        raise ValueError('CAGRA 查询要求 Top-10、单线程、无重排')
    kind = METHODS[args.method]
    if kind == 'float32' and not args.base:
        raise ValueError('float32/exact 需要 --base')
    if kind != 'float32' and (not args.codes or not args.model):
        raise ValueError('量化方法需要 --codes 和 --model')
    query, graph, truth = map(runtime.path, (args.query, args.graph, args.truth))
    base = runtime.path(args.base) if kind == 'float32' else 'NO_FLOAT_DATABASE'
    codes = runtime.path(args.codes) if kind != 'float32' else 'unused'
    model = runtime.path(args.model) if kind != 'float32' else 'unused'
    available = runtime.vector_rows(base) if kind == 'float32' else runtime.code_rows(codes / 'codes.npy')
    n = uint(args.vectors, 'vectors') or available
    if n > available or (kind != 'float32' and n != available):
        raise ValueError('底库数量不匹配；量化编码的行数必须与图节点数一致')
    q_available = runtime.vector_rows(query)
    requested = args.queries if args.queries is not None else config.get('queries', 0)
    count = uint(requested, 'queries') or q_available
    warm = uint(args.warmup if args.warmup is not None else min(config.get('warmup', 200), count), 'warmup')
    repeats = uint(args.repeats if args.repeats is not None else config.get('repeats', 3), 'repeats', 1)
    width = uint(args.width if args.width is not None else config['W'], 'W', 10)
    if count > q_available or warm > count or width > n or repeats > 100:
        raise ValueError('查询数、预热数、搜索宽度或重复次数超出范围')
    size = 0
    if kind == 'pq':
        size = uint(config.get('pq_m', 64), 'pq_m', 1)
        if size > 128 or 128 % size:
            raise ValueError('pq_m 必须整除 128')
    elif kind == 'pca':
        size = uint(config.get('pca_dimension', 80), 'pca_dimension', 1)
        if size > 128:
            raise ValueError('PCA 维度不得超过 128')
    if (args.method == 'pq64' and size != 64) or (args.method == 'pca80' and size != 80):
        raise ValueError('方法名称与配置维度不一致；其他参数请使用 pq/pca 名称')
    return [runtime.binary_dir() / 'ann_query', graph, base, query, truth, n, count, kind,
            codes, model, width, warm, repeats, runtime.output(args.output), size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=list(METHODS), required=True)
    parser.add_argument('--graph', required=True, help='原生邻接图 final.graph')
    parser.add_argument('--query', required=True)
    parser.add_argument('--truth', required=True, help='与底库匹配的 ivecs 或 EGI32001 真值')
    parser.add_argument('--base', help='仅 float32 基线需要，量化模式不访问此路径')
    parser.add_argument('--codes', help='含 codes.npy 的目录')
    parser.add_argument('--model', help='含原生模型 .bin 的目录')
    parser.add_argument('--vectors', type=int, default=0, help='0=从底库或编码推导节点数')
    parser.add_argument('--queries', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--output', required=True, help='新输出前缀，生成 .json/.labels/.stdout/.time')
    parser.add_argument('--config', type=Path, default=ROOT / 'experiment.json')
    parser.add_argument('--execute', action='store_true')
    add_execution_arguments(parser)
    args = parser.parse_args()
    try:
        runtime = Runtime(ROOT, args)
        command = make_command(args, runtime)
        print(shlex.join(runtime.wrapped(command, 1)), flush=True)
        if args.execute:
            runtime.execute([command], runtime.output(args.output), 1)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
