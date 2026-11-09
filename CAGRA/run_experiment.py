"""Run a new DEEP1M CAGRA experiment; --execute starts the work.

Uses the same fitted PQ/PCA and INT8/Local sampling as the completed four-index
DEEP experiment. Only one temporary representation is retained at a time.
"""
import sys
sys.dont_write_bytecode=True
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parent
ANN=ROOT.parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=ANN/'data/deep1M')
    parser.add_argument('--reference-dir',type=Path,default=ANN/'HNSW/experiments/deep1m-20261109')
    parser.add_argument('--base-sha256',default='f25e9ba07d05c5f18c9a4fbe23bd2ee8b8e31747bcddbb77cc3811e768fb7bb2',
                        help='Expected base identity; a subset requires its own base hash and exact truth')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--W',type=int,default=50)
    parser.add_argument('--warmup',type=int,default=200)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve()
    allowed=(ROOT/'results').resolve()
    if not out.is_relative_to(allowed) or out==allowed or out.exists():
        parser.error('Use a new output directory inside CAGRA/results')
    if min(args.threads,args.W,args.repeats)<1 or args.W<10 or args.warmup<0:
        parser.error('Invalid thread/search parameters')
    base=args.data_dir/'deep_base.fvecs'
    query=args.data_dir/'deep_query.fvecs'
    truth=args.data_dir/'deep_groundtruth.ivecs'
    reference=args.reference_dir/'deep_hnsw.bin'
    required=[base,query,truth,reference,args.reference_dir/'truth.json']
    for method in ('pq48','pca80'):
        required += [args.reference_dir/'codes'/method/'codec.json',args.reference_dir/'codes'/method/'codec.npz']
    for path in required:
        if not path.is_file(): parser.error('Missing input: '+str(path))
    print(json.dumps({'output':str(out),'base':str(base),'query':str(query),'truth':str(truth),
                      'methods':['exact','int8','pq48','pca80','local'],
                      'threads':args.threads,'W':args.W,'repeats':args.repeats,
                      'temporary_codes':'one representation at a time; removed after query'},indent=2),flush=True)
    if not args.execute:
        return
    from prepare_codes import digest, fvecs
    shape=fvecs(base).shape
    query_shape=fvecs(query).shape
    if shape[1]!=96 or query_shape[1]!=96 or shape[0]<256 or query_shape[0]<1:
        raise ValueError('This comparison entry requires 96-D base/query data and at least 256 base rows')
    expected_truth=json.loads((args.reference_dir/'truth.json').read_text())['sha256']
    if digest(truth)!=expected_truth:
        raise ValueError('Ground truth differs from the four-index DEEP experiment')
    expected_base=args.base_sha256
    if digest(base)!=expected_base:
        raise ValueError('Base differs from the four-index DEEP experiment')
    # Native graph + KNN output, or graph + largest temporary PCA representation,
    # plus 256 MiB headroom. No copied float database or float32 codec file.
    required_space=max(shape[0]*(65+33)*4,shape[0]*(33*4+80*4))+256*1024**2
    if shutil.disk_usage(ROOT).free<required_space:
        raise ValueError('Need approximately 700 MiB free for this staged workflow')
    out.mkdir(parents=True,exist_ok=False)
    def run(script,*arguments):
        command=[sys.executable,'-B',str(ROOT/script),*map(str,arguments)]
        print('RUN',command,flush=True)
        subprocess.run(command,check=True)
    run('run_build.py','--data',base,'--output',out/'graph','--threads',args.threads,'--execute')
    summary={'dataset':'DEEP1M' if shape[0]==1000000 else 'DEEP subset','dimensions':96,
             'vectors':shape[0],'queries':query_shape[0],'name':'CAGRA',
             'build':json.loads((out/'graph/build.json').read_text()),'searches':{}}
    for method,size in (('exact',0),('int8',96),('pq',48),('pca',80),('local',97)):
        with tempfile.TemporaryDirectory(prefix='temporary-'+method+'-',dir=out) as temporary:
            codes=Path(temporary)/'codes'
            arguments=['--method',method,'--graph',out/'graph/final.graph','--query',query,
                       '--truth',truth,'--W',args.W,'--warmup',args.warmup,
                       '--repeats',args.repeats,'--output',out/'queries'/method,'--execute']
            if method=='exact':
                arguments += ['--base',base]
            else:
                preparation=['--base',base,'--method',method,'--output',codes]
                if method in ('pq','pca'):
                    preparation += ['--model-source',args.reference_dir/'codes'/f'{method}{size}','--code-size',size]
                else:
                    preparation += ['--reference-hnsw',reference]
                run('prepare_codes.py',*preparation)
                provenance=json.loads((codes/'provenance.json').read_text())
                (out/(method+'-encoding.json')).write_text(json.dumps(provenance,indent=2)+'\n')
                arguments += ['--codes',codes,'--model',codes/'native']
                if method in ('pq','pca'): arguments += ['--code-size',size]
            run('run_search.py',*arguments)
            summary['searches'][method]=json.loads((out/'queries'/f'{method}.json').read_text())
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('COMPLETE',out/'summary.json',flush=True)

if __name__=='__main__':
    main()
