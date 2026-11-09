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
from src.metrics import process_metrics

ROOT = Path(__file__).resolve().parent
METHODS = {'exact': 'float32', 'float32': 'float32', 'int8': 'int8',
           'pq48': 'pq', 'pca80': 'pca', 'pca': 'pca', 'pq64': 'pq', 'pq': 'pq', 'local': 'local'}


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
    q_available, dimension = runtime.vector_shape(query)
    if kind == "float32" and runtime.vector_shape(base)[1] != dimension:
        raise ValueError("底库与查询维度不一致")
    requested = args.queries if args.queries is not None else config.get('queries', 0)
    count = uint(requested, 'queries') or q_available
    warm = uint(args.warmup if args.warmup is not None else min(config.get('warmup', 200), count), 'warmup')
    repeats = uint(args.repeats if args.repeats is not None else config.get('repeats', 3), 'repeats', 1)
    width = uint(args.width if args.width is not None else config['W'], 'W', 10)
    if count > q_available or warm > count or width > n or repeats > 100:
        raise ValueError('查询数、预热数、搜索宽度或重复次数超出范围')
    size = 0
    if kind == 'pq':
        size = uint(args.code_size if args.code_size is not None else (48 if args.method == 'pq48' else 64 if args.method == 'pq64' else config.get('pq_m_by_dimension', {}).get(str(dimension), 48 if dimension == 96 else 64)), 'pq_m', 1)
        if size > dimension or dimension % size:
            raise ValueError('pq_m 必须整除实际向量维度')
    elif kind == 'pca':
        size = uint(args.code_size if args.code_size is not None else config.get('pca_dimension', 80), 'pca_dimension', 1)
        if size > dimension:
            raise ValueError('PCA 维度不得超过原始维度')
    if (args.method == 'pq48' and size != 48) or (args.method == 'pq64' and size != 64) or (args.method == 'pca80' and size != 80):
        raise ValueError('方法名称与配置维度不一致；其他参数请使用 pq/pca 名称')
    return [runtime.binary_dir() / ('ann_query96' if dimension == 96 else 'ann_query'), graph, base, query, truth, n, count, kind,
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
    parser.add_argument('--width', '--W', dest='width', type=int)
    parser.add_argument('--code-size', type=int, help='PQ 子空间数或 PCA 维度')
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
            if not runtime.android:
                prefix = runtime.output(args.output)
                path = Path(str(prefix) + '.json')
                result = json.loads(path.read_text())
                resource = process_metrics(str(prefix) + '.time')
                result['vmhwm_peak_rss_mib'] = result['peak_rss_mib']
                result['peak_rss_kib'] = resource['peak_rss_kib']
                result['peak_rss_mib'] = resource['peak_rss_kib'] / 1024
                result['process_seconds'] = resource['seconds']
                result['memory_metric'] = 'GNU time maximum resident set size, matching DEEP workers'
                result['median_qpm'] = result['qpm']
                path.write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps(result, indent=2), flush=True)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
