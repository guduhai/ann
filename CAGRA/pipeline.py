"""Standalone native NSG+CAGRA experiment. No Faiss imports or index files."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import json
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
from ann_codecs import make_codec, save_codec

ROOT = Path(__file__).resolve().parent
KINDS = ('float32', 'int8', 'pca', 'pq', 'local')

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def writable(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Output must be inside CAGRA/')
    return path

def matrix(path, a, integer=False):
    a = np.asarray(a, dtype='<u4' if integer else '<f4', order='C')
    with Path(path).open('xb') as f:
        f.write(struct.pack('<8sII', b'EGI32001' if integer else b'EGF32001', *a.shape))
        a.tofile(f)

def load_matrix(path):
    with Path(path).open('rb') as f:
        magic,n,d=struct.unpack('<8sII',f.read(16))
    if magic != b'EGF32001' or Path(path).stat().st_size != 16+n*d*4:
        raise ValueError('Invalid matrix')
    return np.memmap(path, mode='r', offset=16, shape=(n,d), dtype='<f4')

def prepare(args):
    out=writable(args.output);out.mkdir(parents=True,exist_ok=False)
    if args.dataset:
        import h5py
        with h5py.File(args.dataset,'r') as f:
            n=len(f['train']) if args.vectors==0 else args.vectors
            if n>len(f['train']) or args.queries>len(f['test']): raise ValueError('Dataset too small')
            x=np.asarray(f['train'][:n],dtype=np.float32)
            q=np.asarray(f['test'][:args.queries],dtype=np.float32)
            truth=np.asarray(f['neighbors'][:len(q),:10],dtype=np.uint32) if n==len(f['train']) else None
        source=str(Path(args.dataset).resolve())
    else:
        n=args.vectors
        if n<=100: raise ValueError('Synthetic N must exceed 100')
        rng=np.random.default_rng(20261014)
        x=rng.normal(size=(n,args.dim)).astype(np.float32)
        q=rng.normal(size=(args.queries,args.dim)).astype(np.float32)
        truth=None;source='synthetic normal seed=20261014'
    if not np.isfinite(x).all() or not np.isfinite(q).all(): raise ValueError('Nonfinite input')
    recomputed=truth is None
    if recomputed:
        # Ground truth must be recalculated for a subset, never reuse SIFT1M IDs.
        truth=np.empty((len(q),10),dtype=np.uint32)
        for i,row in enumerate(q):
            delta=x-row
            dist=np.einsum('ij,ij->i',delta,delta)
            top=np.argpartition(dist,9)[:10]
            truth[i]=top[np.lexsort((top,dist[top]))]
        del delta,dist
    matrix(out/'base.f32',x);matrix(out/'queries.f32',q);matrix(out/'truth.i32',truth,True)
    save(out/'dataset.json',dict(source=source,vectors=len(x),dimensions=x.shape[1],queries=len(q),
        subset_groundtruth_recomputed=recomputed,
        base_sha256=digest(out/'base.f32'),queries_sha256=digest(out/'queries.f32'),truth_sha256=digest(out/'truth.i32')))
    print('Prepared',len(x),'vectors,',len(q),'queries',flush=True)

def execute(command, log):
    print('Running:', ' '.join(map(str,command)),flush=True)
    t=time.perf_counter()
    with Path(log).open('x') as f:
        p=subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError(Path(log).read_text(errors='replace')[-6000:])
    return time.perf_counter()-t

def convert_graph(path, output, n, d, target_degree):
    with Path(path).open('rb') as f:
        width,entry=struct.unpack('<II',f.read(8))
        if entry>=n or width>=n: raise ValueError('Invalid graph header')
        offsets=np.zeros(n+1,dtype='<u8')
        # Two passes avoid a million Python lists for SIFT1M.
        for i in range(n):
            size=struct.unpack('<I',f.read(4))[0]
            if not 0<size<=width: raise ValueError('Invalid degree')
            offsets[i+1]=offsets[i]+size;f.seek(size*4,1)
        if f.read(1): raise ValueError('Unexpected nodes')
        if f.tell()!=Path(path).stat().st_size: raise ValueError('Truncated graph')
        ids=np.empty(int(offsets[-1]),dtype='<u4');f.seek(8)
        for i in range(n):
            size=struct.unpack('<I',f.read(4))[0];row=np.frombuffer(f.read(size*4),dtype='<u4')
            if len(row)!=size or np.any(row>=n) or np.any(row==i) or len(np.unique(row))!=size:
                raise ValueError('Invalid/self/duplicate graph edge')
            ids[int(offsets[i]):int(offsets[i+1])]=row
    seen=np.zeros(n,dtype=np.bool_);seen[entry]=True;stack=[entry]
    while stack:
        v=stack.pop()
        for w in ids[int(offsets[v]):int(offsets[v+1])]:
            if not seen[w]: seen[w]=True;stack.append(int(w))
    if not seen.all(): raise ValueError('Graph is not reachable from entry')
    with Path(output).open('xb') as f:
        f.write(struct.pack('<8sIIIIQ',b'EGCSR001',n,d,entry,0,len(ids)));offsets.tofile(f);ids.tofile(f)
    degrees=np.diff(offsets)
    return dict(nodes=n,edges=len(ids),entry=entry,reachable_nodes=int(seen.sum()),
                requested_degree=target_degree,min_degree=int(degrees.min()),max_degree=int(degrees.max()),
                mean_degree=float(degrees.mean()),nodes_above_requested_degree=int((degrees>target_degree).sum()),
                graph_bytes=Path(output).stat().st_size,graph_sha256=digest(output))

def build(args):
    out=writable(args.output);meta=json.loads((out/'dataset.json').read_text())
    if digest(out/'base.f32')!=meta['base_sha256']: raise ValueError('Base changed')
    t=time.perf_counter()
    knn_s=execute([args.bin/'ann_knn',out/'base.f32',out/'candidate.knn',args.gk,args.knn_l,args.iterations,args.sample,args.reverse],out/'knn.log')
    graph_s=execute([args.bin/'ann_build',out/'base.f32',out/'candidate.knn',out/'raw.nsg',args.degree,args.build_l,args.build_c],out/'build.log')
    report=convert_graph(out/'raw.nsg',out/'index.graph',meta['vectors'],meta['dimensions'],args.degree)
    report.update(knn_process_seconds=knn_s,nsg_process_seconds=graph_s,total_prepare_graph_seconds=time.perf_counter()-t,
                  algorithm='native NSG + CAGRA-style detour pruning',faiss_dependency=False,build_threads=1,
                  parameters=dict(gk=args.gk,knn_l=args.knn_l,iterations=args.iterations,sample=args.sample,reverse=args.reverse,
                                  degree=args.degree,build_l=args.build_l,build_c=args.build_c))
    save(out/'build.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

def encode(args):
    out=writable(args.output);directory=out/'codecs';directory.mkdir(exist_ok=False)
    x=load_matrix(out/'base.f32');n,d=x.shape
    rng=np.random.default_rng(20261014);sample=np.array(x[rng.choice(n,min(n,args.train),replace=False)])
    meta=json.loads((out/'dataset.json').read_text())
    if digest(out/'base.f32')!=meta['base_sha256']: raise ValueError('Base changed')
    graph_hash=digest(out/'index.graph');reports={}
    for mode,kind in enumerate(KINDS):
        started=time.perf_counter();width=centroids=clusters=0
        if kind=='float32': codec=None;arrays={};stride=d*4
        else:
            opts={}
            if kind=='pca': width=min(args.pca_dim,d);opts=dict(components=width)
            if kind=='pq': width=args.pq_m;centroids=args.pq_k;opts=dict(subspaces=width,centroids=centroids,iterations=args.codec_iterations)
            if kind=='local': clusters=args.clusters;opts=dict(clusters=clusters,iterations=args.codec_iterations)
            codec=make_codec(kind,d,**opts);codec.fit(sample);arrays=codec.arrays()
            save_codec(codec,directory/kind)
            stride=np.dtype(codec.code_dtype()).itemsize*int(np.prod(codec.code_shape(),dtype=int))
        path=directory/(kind+'.bin')
        with path.open('xb') as f:
            f.write(struct.pack('<8s6IQ',b'EGCODE01',n,d,mode,width,centroids,clusters,n*stride))
            for start in range(0,n,4096):
                chunk=x[start:start+4096];codes=chunk if codec is None else codec.encode(chunk)
                np.ascontiguousarray(codes).tofile(f)
            for key in ('scale','mean','basis','codebooks','centers'):
                if key in arrays: np.asarray(arrays[key],dtype='<f4',order='C').tofile(f)
        reports[kind]=dict(seconds=time.perf_counter()-started,file_bytes=path.stat().st_size,sha256=digest(path))
        print('Encoded',kind,reports[kind],flush=True)
    save(directory/'manifest.json',dict(base_sha256=meta['base_sha256'],graph_sha256=graph_hash,methods=reports))

def verify_inputs(out):
    meta=json.loads((out/'dataset.json').read_text());manifest=json.loads((out/'codecs/manifest.json').read_text())
    if digest(out/'index.graph')!=manifest['graph_sha256']: raise ValueError('Graph hash mismatch')
    for name in ('queries','truth'):
        if digest(out/('queries.f32' if name=='queries' else 'truth.i32'))!=meta[name+'_sha256']: raise ValueError('Input changed')
    return manifest

def query(args):
    out=writable(args.output);manifest=verify_inputs(out)
    result=out/args.run_name;result.mkdir(exist_ok=False)
    runs=[]
    for kind in args.methods.split(','):
        if kind not in KINDS: raise ValueError('Unknown method')
        codec=out/'codecs'/(kind+'.bin')
        if digest(codec)!=manifest['methods'][kind]['sha256']: raise ValueError('Codec hash mismatch')
        for L in map(int,args.search_l.split(',')):
            prefix=result/f'{kind}-l{L}'
            execute([args.bin/'ann_query',out/'index.graph',codec,out/'queries.f32',out/'truth.i32',prefix,L,args.warmup,args.repeats],prefix.with_suffix('.log'))
            row=json.loads(prefix.with_suffix('.json').read_text());row['method']=kind;runs.append(row)
            print(kind,'L',L,'Recall',row['recall_at_10'],'RSS MiB',row['peak_rss_bytes']/2**20,flush=True)
    save(result/'summary.json',dict(platform=platform.platform(),machine=platform.machine(),query_threads=1,
        graph_sha256=manifest['graph_sha256'],query_binary_sha256=digest(args.bin/'ann_query'),runs=runs))
    report=['# 本地独立 C++ 无重排查询结果','',
        '本轮建图使用原生 NSG/EFANNA，剪枝替换为 CAGRA 风格 detour-count；不依赖 Faiss。',
        '量化查询不加载全精度底库；每个方法、每个 L 使用独立 C++ 子进程。',
        '内存为 Linux /proc/self/status 的 VmHWM，覆盖本次可执行程序的加载及查询；编码完整读入内存。',
        '另外记录 ru_maxrss 供诊断，它可能包含 exec 前继承的 Python 父进程内存峰值，因此不用于主表。',
        'QPM 使用重复完整查询耗时的中位数；包含查询投影/查表，不包含文件加载、预热、召回计算及写盘。','',
        '| 方法 | L | Recall@10 | QPM | 峰值 RSS（MiB） | 图+编码载荷（MiB） |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in runs:
        report.append(f"| {r['method']} | {r['search_l']} | {r['recall_at_10']:.3%} | {r['qpm']:,.0f} | {r['peak_rss_bytes']/2**20:.2f} | {(r['graph_payload_bytes']+r['codec_payload_bytes'])/2**20:.2f} |")
    report+=['','这里只是当前本机数据规模下的结果，不代表树莓派或 Android 性能。',
             '同一个 L 并不代表相同召回率；取消重排后的精度损失必须单独报告。',
             '尚未测量冷缓存磁盘 I/O，也没有宣称磁盘 I/O 优化。']
    (result/'RESULTS_CN.md').write_text('\n'.join(report)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','build','encode','query'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bin',type=Path,default=ROOT/'build')
    p.add_argument('--dataset',type=Path);p.add_argument('--vectors',type=int,default=20000)
    p.add_argument('--queries',type=int,default=200);p.add_argument('--dim',type=int,default=128)
    p.add_argument('--gk',type=int,default=64);p.add_argument('--knn-l',type=int,default=80)
    p.add_argument('--iterations',type=int,default=8);p.add_argument('--sample',type=int,default=10);p.add_argument('--reverse',type=int,default=100)
    p.add_argument('--degree',type=int,default=32);p.add_argument('--build-l',type=int,default=64);p.add_argument('--build-c',type=int,default=132)
    p.add_argument('--train',type=int,default=10000);p.add_argument('--pca-dim',type=int,default=80)
    p.add_argument('--pq-m',type=int,default=64);p.add_argument('--pq-k',type=int,default=256)
    p.add_argument('--clusters',type=int,default=32);p.add_argument('--codec-iterations',type=int,default=10)
    p.add_argument('--search-l',default='20,50,100,200');p.add_argument('--warmup',type=int,default=20);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--methods',default=','.join(KINDS));p.add_argument('--run-name',default='query-results')
    args=p.parse_args();args.bin=args.bin.resolve()
    if Path(args.run_name).name!=args.run_name: p.error('run-name must be one directory name')
    if args.queries<1 or args.train<2 or args.dim<1 or args.warmup<1 or args.repeats<1: p.error('Invalid count')
    globals()[args.phase](args)

if __name__=='__main__': main()
