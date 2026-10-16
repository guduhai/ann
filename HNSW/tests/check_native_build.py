"""Small functional checks only: official Python parity and parallel validity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import hnswlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hnsw_graph import HnswGraph


def topology(path):
    graph = HnswGraph(path)
    digest = hashlib.sha256()
    digest.update(np.array([graph.count, graph.entry, graph.max_level], dtype='<u8').tobytes())
    for row in range(graph.count):
        digest.update(np.array([graph.labels[row], graph.counts[row]],dtype='<u8').tobytes())
        digest.update(graph.links[row,:graph.counts[row]].tobytes())
        if graph.upper_offsets[row] >= 0:
            upper_bytes=int(np.frombuffer(graph.raw,dtype='<u4',count=1,offset=int(graph.upper_offsets[row])-4)[0])
            for level in range(1,upper_bytes//graph.upper_stride+1):
                links=graph.upper_links(row,level)
                digest.update(np.array([level,len(links)],dtype='<u8').tobytes())
                digest.update(links.tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=ROOT/'build/bin/ann_build')
    args=parser.parse_args()
    results=[]
    (ROOT/'build').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='native-build-check-',dir=ROOT/'build') as temporary:
        tmp=Path(temporary)
        x=np.random.default_rng(20261016).normal(size=(1024,128)).astype('<f4')
        records=np.empty((len(x),129),dtype='<f4');records.view('<i4')[:,0]=128;records[:,1:]=x
        data=tmp/'sample.fvecs';records.tofile(data)
        reference=hnswlib.Index(space='l2',dim=128)
        reference.init_index(max_elements=len(x),M=16,ef_construction=200,random_seed=100)
        reference.add_items(x,np.arange(len(x),dtype=np.int32),num_threads=1)
        reference.save_index(str(tmp/'python.bin'))
        for threads in [1,4]:
            output=tmp/f'native-{threads}.bin'
            cmd=[str(args.binary),'--dataset',str(data),'--index',str(output),'--build-threads',str(threads)]
            p=subprocess.run(cmd,capture_output=True,text=True,check=True)
            report=json.loads(p.stdout)
            assert report['vectors']==1024 and report['dimensions']==128
            assert (report['m'],report['ef_construction'],report['seed'])==(16,200,100)
            actual=hnswlib.Index(space='l2',dim=128);actual.load_index(str(output));actual.set_ef(50)
            ids=np.arange(len(x),dtype=np.int32)
            assert actual.get_current_count()==len(x)
            assert np.array_equal(actual.get_items(ids),x)
            labels,_=actual.knn_query(x[:32],k=10,num_threads=1)
            assert np.array_equal(labels[:,0],ids[:32])
            entry={'threads':threads,'loaded_by_official_hnswlib':True,'all_vectors_match':True,'self_queries_pass':True}
            if threads==1:
                entry['same_topology_as_python']=topology(output)==topology(tmp/'python.bin')
                assert entry['same_topology_as_python'],'Single-thread graph differs from official Python build'
            before=hashlib.sha256(output.read_bytes()).hexdigest()
            rejected=subprocess.run(cmd,capture_output=True,text=True)
            assert rejected.returncode!=0 and hashlib.sha256(output.read_bytes()).hexdigest()==before
            entry['overwrite_rejected']=True
            results.append(entry)
        bad=tmp/'bad.fvecs';bad.write_bytes(data.read_bytes()[:-1])
        p=subprocess.run([str(args.binary),'--dataset',str(bad),'--index',str(tmp/'bad.bin')],capture_output=True,text=True)
        assert p.returncode!=0 and not (tmp/'bad.bin').exists()
    print(json.dumps({'vectors':1024,'dimensions':128,'checks':results,'truncated_input_rejected':True}))


if __name__=='__main__':
    main()
