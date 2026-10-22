"""本机数据准备与共享量化辅助流程；正式构图/查询复用统一入口。"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from ann_codecs import make_codec, save_codec

ROOT = Path(__file__).resolve().parent
KINDS = ('float32', 'int8', 'pca', 'pq', 'local')


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def writable(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'results') or path == ROOT / 'results':
        raise ValueError('输出必须位于 CAGRA/results 内')
    return path


def matrix(path, values, integer=False):
    values = np.asarray(values, dtype='<u4' if integer else '<f4', order='C')
    with Path(path).open('xb') as stream:
        stream.write(struct.pack('<8sII', b'EGI32001' if integer else b'EGF32001', *values.shape))
        values.tofile(stream)


def load_matrix(path):
    with Path(path).open('rb') as stream:
        magic, n, d = struct.unpack('<8sII', stream.read(16))
    if magic != b'EGF32001' or d != 128 or Path(path).stat().st_size != 16+n*d*4:
        raise ValueError('需要完整的 128 维 EGF32001 文件')
    return np.memmap(path, mode='r', offset=16, shape=(n, d), dtype='<f4')


def prepare(args):
    out = writable(args.output)
    if args.dataset:
        import h5py
        with h5py.File(args.dataset, 'r') as dataset:
            total = len(dataset['train'])
            n = total if args.vectors == 0 else args.vectors
            if n > total or args.queries > len(dataset['test']):
                raise ValueError('数据集规模不足')
            x = np.asarray(dataset['train'][:n], dtype=np.float32)
            q = np.asarray(dataset['test'][:args.queries], dtype=np.float32)
            truth = np.asarray(dataset['neighbors'][:len(q), :10], dtype=np.uint32) if n == total else None
        source = str(args.dataset.resolve())
    else:
        n = args.vectors
        if n <= 100:
            raise ValueError('合成底库必须超过 100 条')
        rng = np.random.default_rng(20261014)
        x = rng.normal(size=(n, 128)).astype(np.float32)
        q = rng.normal(size=(args.queries, 128)).astype(np.float32)
        truth, source = None, 'synthetic normal seed=20261014'
    if x.shape[1] != 128 or q.shape[1] != 128 or not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError('需要有限的 128 维输入')
    recomputed = truth is None
    if recomputed:
        truth = np.empty((len(q), 10), dtype=np.uint32)
        for i, row in enumerate(q):
            delta = x-row
            dist = np.einsum('ij,ij->i', delta, delta)
            top = np.argpartition(dist, 9)[:10]
            truth[i] = top[np.lexsort((top, dist[top]))]
    out.mkdir(parents=True, exist_ok=False)
    matrix(out/'base.f32', x); matrix(out/'queries.f32', q); matrix(out/'truth.i32', truth, True)
    save(out/'dataset.json', dict(source=source, vectors=len(x), dimensions=128, queries=len(q),
         subset_groundtruth_recomputed=recomputed, base_sha256=digest(out/'base.f32'),
         queries_sha256=digest(out/'queries.f32'), truth_sha256=digest(out/'truth.i32')))
    print(f'已准备 {len(x)} 条底库、{len(q)} 条查询', flush=True)


def execute(command, log):
    with Path(log).open('x', encoding='utf-8') as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)


def build(args):
    out = writable(args.output)
    meta = json.loads((out/'dataset.json').read_text(encoding='utf-8'))
    if digest(out/'base.f32') != meta['base_sha256']:
        raise ValueError('底库已改变')
    params = dict(K=args.gk, L=args.knn_l, iterations=args.iterations, S=args.sample, R=args.reverse,
                  graph_R=args.degree, graph_L=args.build_l, graph_C=args.build_c,
                  threads=args.threads, seed=20261014)
    config = out/'build_parameters.json'
    if config.exists():
        raise ValueError('构图参数文件已存在，请选择新实验目录')
    save(config, params)
    started = time.perf_counter()
    execute([sys.executable, '-B', ROOT/'run_build.py', '--data', out/'base.f32',
             '--output', out/'graph', '--binary-dir', args.bin, '--config', config, '--execute'], out/'build-launch.log')
    save(out/'build.json', dict(build_threads=args.threads, parameters=params,
         total_process_seconds=time.perf_counter()-started, graph_format='native adjacency',
         graph_sha256=digest(out/'graph/final.graph')))


def encode(args):
    out = writable(args.output)
    meta = json.loads((out/'dataset.json').read_text(encoding='utf-8'))
    if digest(out/'base.f32') != meta['base_sha256']:
        raise ValueError('底库已改变')
    x = load_matrix(out/'base.f32'); n, d = x.shape
    rng = np.random.default_rng(20261014)
    sample = np.array(x[rng.choice(n, min(n, args.train), replace=False)])
    directory = out/'codecs'; directory.mkdir(exist_ok=False)
    graph_hash = digest(out/'graph/final.graph')
    reports = {}
    for kind in KINDS[1:]:
        started = time.perf_counter()
        options = {}
        if kind == 'pca': options = dict(components=args.pca_dim)
        if kind == 'pq': options = dict(subspaces=args.pq_m, centroids=256, iterations=args.codec_iterations)
        if kind == 'local': options = dict(clusters=32, iterations=args.codec_iterations)
        codec = make_codec(kind, d, **options); codec.fit(sample)
        prepared = directory/kind; save_codec(codec, prepared)
        codes = np.lib.format.open_memmap(prepared/'codes.npy', mode='w+',
            dtype=codec.code_dtype(), shape=(n,)+codec.code_shape())
        for begin in range(0, n, 4096):
            codes[begin:begin+4096] = codec.encode(x[begin:begin+4096])
        codes.flush(); del codes
        native = prepared/'native'; native.mkdir()
        for name, array in codec.arrays().items():
            np.asarray(array, dtype='<f4', order='C').tofile(native/(name+'.bin'))
        report = dict(kind=kind, vectors=n, dimension=d, graph_sha256=graph_hash,
            codes_bytes=n*np.dtype(codec.code_dtype()).itemsize*int(np.prod(codec.code_shape(), dtype=int)),
            codes_sha256=digest(prepared/'codes.npy'), model_config=codec.config(),
            model_files={p.name: digest(p) for p in native.iterdir()},
            seconds=time.perf_counter()-started, search_only=True, graph_rebuilt=False)
        save(prepared/'manifest.json', report); reports[kind] = report
        print('已编码', kind, flush=True)
    save(directory/'manifest.json', dict(base_sha256=meta['base_sha256'], graph_sha256=graph_hash, methods=reports))


def query(args):
    out = writable(args.output)
    manifest = json.loads((out/'codecs/manifest.json').read_text(encoding='utf-8'))
    meta = json.loads((out/'dataset.json').read_text(encoding='utf-8'))
    graph = out/'graph/final.graph'
    if digest(graph) != manifest['graph_sha256']:
        raise ValueError('图哈希不匹配')
    for filename, key in (('queries.f32', 'queries_sha256'), ('truth.i32', 'truth_sha256')):
        if digest(out/filename) != meta[key]: raise ValueError('查询或真值已改变')
    config = json.loads((ROOT/'experiment.json').read_text(encoding='utf-8'))
    config.update(pq_m=manifest['methods']['pq']['model_config']['subspaces'],
                  pca_dimension=manifest['methods']['pca']['model_config']['components'])
    result = out/args.run_name; result.mkdir(exist_ok=False)
    save(result/'experiment.json', config)
    runs = []
    for kind in args.methods.split(','):
        if kind not in KINDS: raise ValueError('未知方法')
        extra = ['--base', out/'base.f32']
        if kind != 'float32':
            prepared = out/'codecs'/kind; recorded = manifest['methods'][kind]
            if digest(prepared/'codes.npy') != recorded['codes_sha256']:
                raise ValueError('量化编码已改变')
            for filename, expected in recorded['model_files'].items():
                if digest(prepared/'native'/filename) != expected: raise ValueError('量化模型已改变')
            extra = ['--codes', prepared, '--model', prepared/'native']
        elif digest(out/'base.f32') != meta['base_sha256']:
            raise ValueError('float32 底库已改变')
        for width in map(int, args.search_l.split(',')):
            prefix = result/f'{kind}-l{width}'
            execute([sys.executable, '-B', ROOT/'run_search.py', '--method', kind, '--graph', graph,
                     '--query', out/'queries.f32', '--truth', out/'truth.i32', '--output', prefix,
                     '--config', result/'experiment.json', '--binary-dir', args.bin,
                     '--width', width, '--warmup', args.warmup, '--repeats', args.repeats, '--execute', *extra],
                    Path(str(prefix)+'.launch.log'))
            runs.append(json.loads(Path(str(prefix)+'.json').read_text(encoding='utf-8')))
    save(result/'summary.json', dict(platform=platform.platform(), machine=platform.machine(),
         graph_sha256=manifest['graph_sha256'], query_binary_sha256=digest(args.bin/'ann_query'), runs=runs))
    report = ['# CAGRA 本机查询结果', '', '| 方法 | L | Recall@10 | QPM | 峰值 RSS（MiB） |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in runs:
        report.append(f"| {row['method']} | {row['search_l']} | {row['recall_at_10']:.3%} | {row['qpm']:,.0f} | {row['peak_rss_mib']:.2f} |")
    (result/'RESULTS_CN.md').write_text('\n'.join(report)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'build', 'encode', 'query'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bin', type=Path, default=ROOT/'build/bin')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--vectors', type=int, default=20000)
    parser.add_argument('--queries', type=int, default=500)
    parser.add_argument('--dim', type=int, choices=[128], default=128)
    for flag, default in [('gk',64), ('knn-l',80), ('iterations',8), ('sample',10), ('reverse',100),
                          ('degree',32), ('build-l',64), ('build-c',132), ('threads',4),
                          ('train',10000), ('pca-dim',80), ('pq-m',64), ('codec-iterations',10)]:
        parser.add_argument('--'+flag, type=int, default=default)
    parser.add_argument('--search-l', default='50')
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--methods', default=','.join(KINDS))
    parser.add_argument('--run-name', default='query-results')
    args = parser.parse_args(); args.bin = args.bin.resolve()
    if args.run_name in ('', '.', '..') or '/' in args.run_name or '\\' in args.run_name:
        parser.error('run-name 必须是单个新目录名')
    if args.vectors < 0 or args.queries < 1 or args.train < 256 or args.warmup < 0 or not 1 <= args.repeats <= 100:
        parser.error('无效的数据规模、训练数、预热数或重复数')
    globals()[args.phase](args)


if __name__ == '__main__':
    main()
