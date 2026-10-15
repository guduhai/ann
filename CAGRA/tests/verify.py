"""Cross-check distances, exhaustive search, malformed inputs, and no-base operation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from ann_codecs import load_codec
from pipeline import KINDS,load_matrix,save,writable,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--bin',type=Path,default=ROOT/'build')
    p.add_argument('--output-name',default='verification')
    a=p.parse_args();case=writable(a.case);binary=a.bin.resolve()/'ann_query'
    out=case/a.output_name;out.mkdir(exist_ok=False)
    env=os.environ.copy()
    for k in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH'):env.pop(k,None)
    env['ASAN_OPTIONS']='detect_leaks=1'
    x=load_matrix(case/'base.f32');q=load_matrix(case/'queries.f32');n,d=x.shape
    if n>4096:raise ValueError('Exhaustive validation limited to <=4096 vectors')
    report={'binary_sha256':digest(binary),'distance_checks':{},'exhaustive_checks':{},'no_base_checks':{}}
    def run(args,expect=True):
        r=subprocess.run([str(binary),*map(str,args)],capture_output=True,env=env)
        if expect and r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
        if not expect and not r.returncode:raise RuntimeError('Malformed input unexpectedly accepted')
        return r
    for kind in KINDS:
        codec_path=case/'codecs'/(kind+'.bin')
        result=run(['scores',codec_path,case/'queries.f32'])
        cpp=np.frombuffer(result.stdout,dtype='<f4')
        if kind=='float32':
            delta=x-q[0];ref=np.einsum('ij,ij->i',delta,delta)
        else:
            codec=load_codec(case/'codecs'/kind);codes=codec.encode(x)
            ref=codec.distances(codec.query_state(q[0]),codes)
        np.testing.assert_allclose(cpp,ref,rtol=3e-5,atol=3e-4)
        report['distance_checks'][kind]={'max_relative_error':float(np.max(np.abs(cpp-ref)/np.maximum(1,np.abs(ref))))}
        prefix=out/(kind+'-exhaustive')
        run([case/'index.graph',codec_path,case/'queries.f32',case/'truth.i32',prefix,n,1,2])
        labels=np.fromfile(prefix.with_suffix('.labels'),dtype='<u4').reshape(len(q),10)
        # L=N must return exact nearest neighbors under the selected CODEC metric.
        expected=np.lexsort((np.arange(n),cpp))[:10]
        np.testing.assert_array_equal(labels[0],expected)
        if np.any(labels>=n) or any(len(np.unique(row))!=10 for row in labels):raise AssertionError('Invalid output IDs')
        metrics=json.loads(prefix.with_suffix('.json').read_text())
        if metrics['full_precision_base_loaded']!=(kind=='float32') or metrics['rerank']:raise AssertionError('Wrong query mode')
        report['exhaustive_checks'][kind]=True
    # Quantized subprocesses cannot open either of the standalone float32 base files.
    del x
    moved=[]
    try:
        for original in [case/'base.f32',case/'codecs/float32.bin']:
            hidden=original.with_suffix(original.suffix+'.verification-hidden')
            if hidden.exists():raise RuntimeError('Temporary target already exists')
            original.rename(hidden);moved.append((original,hidden))
        for kind in KINDS[1:]:
            prefix=out/(kind+'-no-base')
            run([case/'index.graph',case/'codecs'/(kind+'.bin'),case/'queries.f32',case/'truth.i32',prefix,50,1,2])
            report['no_base_checks'][kind]=True
    finally:
        for original,hidden in reversed(moved):hidden.rename(original)
    bad=out/'truncated.bin';bad.write_bytes((case/'codecs/int8.bin').read_bytes()[:45])
    run([case/'index.graph',bad,case/'queries.f32',case/'truth.i32',out/'bad-code',50,1,1],expect=False)
    bad_graph=out/'bad.graph';content=bytearray((case/'index.graph').read_bytes());content[32:40]=(1).to_bytes(8,'little');bad_graph.write_bytes(content)
    run([bad_graph,case/'codecs/int8.bin',case/'queries.f32',case/'truth.i32',out/'bad-graph',50,1,1],expect=False)
    report['malformed_inputs_rejected']=True
    deps=subprocess.run(['ldd',str(binary)],capture_output=True,text=True,env=env,check=True).stdout
    (out/'linked-libraries.txt').write_text(deps)
    if 'faiss' in deps.lower() or 'cuvs' in deps.lower():raise AssertionError('Unexpected dependency')
    report['no_faiss_or_cuvs_dynamic_dependency']=True
    save(out/'verification.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
