"""Controlled PQ recall audit on an existing small case. Production inputs read-only.

All extra files are confined to a NEW standalone/results output directory.
The probe includes production query.cpp, but emits candidates and exhaustive ADC
scores; exact reranking and independent decoded-vector validation happen here.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from ann_codecs import ProductQuantization,load_codec,save_codec
from pipeline import load_matrix,digest,save,writable

def recall(ids,truth):
    return float(np.any(ids[:,:,None]==truth[:,None,:],axis=2).sum()/truth.size)

def read_pq(path):
    with path.open('rb') as f:
        magic,n,d,mode,m,k,clusters,nb=struct.unpack('<8s6IQ',f.read(40))
        assert magic==b'EGCODE01' and mode==3 and clusters==0 and nb==n*m and d%m==0
        codes=np.fromfile(f,dtype='u1',count=n*m).reshape(n,m)
        books=np.fromfile(f,dtype='<f4',count=k*d).reshape(m,k,d//m)
        assert f.read()==b''
    return codes,books

def decode(codes,books):
    return np.concatenate([books[s,codes[:,s]] for s in range(codes.shape[1])],axis=1)

def write_pq(path,codes,books):
    n,m=codes.shape;_,k,w=books.shape
    with path.open('xb') as f:
        f.write(struct.pack('<8s6IQ',b'EGCODE01',n,m*w,3,m,k,0,n*m))
        codes.tofile(f);books.astype('<f4').tofile(f)

def top_rows(dist,k=10):
    # Full lexicographic sort avoids arbitrary membership at a tied cutoff.
    ids=np.arange(dist.shape[1])
    return np.stack([np.lexsort((ids,row))[:k] for row in dist]).astype(np.uint32)

def original_audit(case,out,probe):
    x=np.asarray(load_matrix(case/'base.f32'));q=np.asarray(load_matrix(case/'queries.f32'))
    truth=np.fromfile(case/'truth.i32',dtype='<u4',offset=16).reshape(len(q),10)
    if len(x)>100000:raise ValueError('This diagnostic is for small cases')
    inputs=[case/name for name in ['base.f32','queries.f32','truth.i32','index.graph','codecs/pq.bin',
                                 'codecs/pq/codec.json','codecs/pq/codec.npz']]
    before={str(p.relative_to(case)):digest(p) for p in inputs}
    prefix=out/'original'
    if not prefix.with_suffix('.scores').exists():
        subprocess.run([str(probe),str(case/'index.graph'),str(case/'codecs/pq.bin'),str(case/'queries.f32'),
                        str(prefix),'20,50,100,200','1'],check=True)
    scores=np.memmap(str(prefix)+'.scores',mode='r',dtype='<f4',shape=(len(q),len(x)))
    codes,books=read_pq(case/'codecs/pq.bin');model=load_codec(case/'codecs/pq')
    assert np.array_equal(books,model.codebooks)
    reencoded=np.concatenate([model.encode(x[i:i+1024]) for i in range(0,len(x),1024)])
    assert np.array_equal(codes,reencoded)
    print('Binary codebook/model and all stored codes match',flush=True)
    # Independent encoding reference uses explicit squared differences in float64,
    # not the norm/dot-product identity used by _nearest.
    violations=ties=0;max_gap=0.
    for s in range(codes.shape[1]):
        w=books.shape[2];centers=books[s].astype(np.float64)
        for i in range(0,len(x),256):
            block=x[i:i+256,s*w:(s+1)*w].astype(np.float64)
            dd=((block[:,None,:]-centers[None,:,:])**2).sum(axis=2)
            assigned=dd[np.arange(len(block)),codes[i:i+len(block),s]]
            gap=assigned-dd.min(axis=1);max_gap=max(max_gap,float(gap.max()))
            violations+=int(np.count_nonzero(gap>0.05))
            ties+=int(np.count_nonzero(codes[i:i+len(block),s]!=dd.argmin(axis=1)))
    assert violations==0,'Encoding did not select a nearest codeword'
    print('Direct float64 nearest-codeword check complete',flush=True)
    reconstructed=decode(codes,books).astype(np.float64);x64=x.astype(np.float64)
    exact=np.empty((len(q),len(x)),dtype=np.float64)
    decoded_top=np.empty_like(truth);max_abs=max_rel=0.;truth_invalid=0;boundary_ties=0
    ids=np.arange(len(x));relative_rank_changes=[]
    for i,query in enumerate(q):
        original_dist=((x64-query.astype(np.float64))**2).sum(axis=1)
        exact[i]=original_dist
        cutoff=np.partition(original_dist,9)[9]
        truth_invalid+=int(np.count_nonzero(original_dist[truth[i]]>cutoff))
        boundary_ties+=int(np.count_nonzero(original_dist==cutoff)>1)
        decoded_dist=((reconstructed-query.astype(np.float64))**2).sum(axis=1)
        diff=np.abs(scores[i].astype(np.float64)-decoded_dist)
        max_abs=max(max_abs,float(diff.max()));max_rel=max(max_rel,float((diff/np.maximum(decoded_dist,1)).max()))
        decoded_top[i]=np.lexsort((ids,decoded_dist))[:10]
    assert truth_invalid==0 and max_rel<3e-6
    # SIFT integer inputs make float64 exact ground truth directly verifiable.
    print('10 million C++ ADC scores checked against float64 decoded-vector L2',flush=True)
    pq_top=top_rows(scores);all_rows=[]
    for L in [20,50,100,200]:
        pool=np.fromfile(str(prefix)+f'.l{L}.pool',dtype='<u4').reshape(len(q),L)
        old=np.fromfile(case/f'query-results/pq-l{L}.labels',dtype='<u4').reshape(len(q),10)
        assert np.array_equal(pool[:,:10],old),'Diagnostic search differs from recorded production search'
        reranked=np.stack([row[np.lexsort((row,exact[i,row]))[:10]] for i,row in enumerate(pool)])
        # Compare to exact top-L under ADC: isolate graph traversal misses.
        exhaustive_pool=top_rows(scores,L)
        all_rows.append(dict(L=L,no_rerank_recall=recall(pool[:,:10],truth),
            exact_rerank_same_pool_recall=recall(reranked,truth),
            true_top10_candidate_coverage=recall(pool,truth),
            graph_top10_overlap_with_exhaustive_pq=recall(pool[:,:10],pq_top),
            exhaustive_pq_topL_true_top10_coverage=recall(exhaustive_pool,truth)))
    mse=float(np.mean((x64-reconstructed)**2))
    result=dict(vectors=len(x),queries=len(q),dimension=x.shape[1],subspaces=codes.shape[1],
        bytes_per_vector=codes.shape[1],k=books.shape[1],training_vectors=10000,iterations=10,
        stored_codebooks_equal_model=True,all_stored_codes_equal_reencoded=True,
        nearest_codeword_nonminimal_count=violations,nearest_argmin_difference_count=ties,
        nearest_max_squared_distance_gap=max_gap,checked_cpp_distances=int(scores.size),
        cpp_vs_decoded_l2_max_absolute_error=max_abs,cpp_vs_decoded_l2_max_relative_error=max_rel,
        groundtruth_invalid_ids=truth_invalid,queries_with_gt_boundary_ties=boundary_ties,
        exhaustive_pq_recall_at_10=recall(pq_top,truth),decoded_float64_recall_at_10=recall(decoded_top,truth),
        cpp_vs_float64_decoded_top10_overlap=recall(pq_top,decoded_top),reconstruction_mse=mse,
        graph_results=all_rows,inputs_before=before)
    after={str(p.relative_to(case)):digest(p) for p in inputs};assert before==after
    result['inputs_unchanged']=True
    result['source_hashes']={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'src/query.cpp',ROOT/'include/distance_kernels.h',ROOT/'ann_codecs.py',ROOT/'tests/pq_probe.cpp']}
    save(out/'original-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs_before','source_hashes']},indent=2),flush=True)
    np.save(out/'exact-distances.npy',exact)
    return x,q,truth

def variants(case,out,probe,x,q,truth):
    # One factor at a time, then increase code capacity. Same graph/query set.
    configs=[('m16-train10k-it50',16,10000,50),('m16-train20k-it50',16,20000,50),
             ('m32-train20k-it50',32,20000,50)]
    reports=[]
    for name,m,train,it in configs:
        destination=out/name;destination.mkdir(exist_ok=False)
        sample=x[np.random.default_rng(20261014).choice(len(x),min(train,len(x)),replace=False)]
        model=ProductQuantization(x.shape[1],m,256,it)
        t=time.perf_counter();model.fit(sample)
        codes=np.concatenate([model.encode(x[i:i+1024]) for i in range(0,len(x),1024)])
        fit_encode_s=time.perf_counter()-t
        save_codec(model,destination/'model');write_pq(destination/'pq.bin',codes,model.codebooks)
        prefix=destination/'probe'
        subprocess.run([str(probe),str(case/'index.graph'),str(destination/'pq.bin'),str(case/'queries.f32'),
                        str(prefix),'50,200','1'],check=True)
        scores=np.memmap(str(prefix)+'.scores',mode='r',dtype='<f4',shape=(len(q),len(x)))
        pq_top=top_rows(scores)
        r=dict(name=name,subspaces=m,bytes_per_vector=m,train=len(sample),iterations=it,
            fit_encode_seconds=fit_encode_s,exhaustive_recall_at_10=recall(pq_top,truth),
            reconstruction_mse=float(np.mean((x.astype(np.float64)-decode(codes,model.codebooks).astype(np.float64))**2)),
            graph_results=[])
        for L in [50,200]:
            pool=np.fromfile(str(prefix)+f'.l{L}.pool',dtype='<u4').reshape(len(q),L)
            r['graph_results'].append(dict(L=L,no_rerank_recall=recall(pool[:,:10],truth),
                overlap_exhaustive_pq=recall(pool[:,:10],pq_top)))
        save(destination/'summary.json',r);reports.append(r);print(json.dumps(r,indent=2),flush=True)
    save(out/'variants.json',reports)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,default=ROOT/'results/sift20k')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variants',action='store_true')
    a=p.parse_args();case=writable(a.case);out=writable(a.output)
    out.mkdir(exist_ok=True,parents=True)
    if (out/'original-audit.json').exists():raise ValueError('Audit already exists; choose new output')
    probe=out/'pq_probe'
    if not probe.exists():
        subprocess.run(['/usr/bin/g++','-O3','-std=c++17','-ffp-contract=off','-I'+str(ROOT/'include'),
                        str(ROOT/'tests/pq_probe.cpp'),'-o',str(probe)],check=True)
    x,q,truth=original_audit(case,out,probe)
    if a.variants:variants(case,out,probe,x,q,truth)

if __name__=='__main__':main()
