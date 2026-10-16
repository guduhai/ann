"""Prepare the four no-rerank representations using the unchanged shared codecs.

Training uses the original HNSW internal vector order, as in the size study.
Write both internal-ID codes and original-dataset-ID codes for the other graphs.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ann_quantization import make_codec, save_codec
from hnsw_graph import HnswGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; existing prepared models will not be overwritten.')
    graph = HnswGraph(args.index)
    if graph.dimension != 128 or graph.count < 256:
        parser.error('Expected at least 256 vectors of dimension 128.')
    labels = graph.labels.astype(np.int64)
    if not np.array_equal(np.sort(labels), np.arange(graph.count)):
        raise ValueError('Graph labels must be a permutation of original dataset IDs.')
    with args.index.open('rb') as stream:
        graph_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    plans = [('int8','int8',100000,{}), ('pq64','pq',8192,{'subspaces':64,'centroids':256,'iterations':10}),
             ('pca80','pca',30000,{'components':128}), ('local','local',20000,{})]
    args.output.mkdir(parents=True)
    for name,kind,sample_count,options in plans:
        codec = make_codec(kind,128,**options)
        sample_count = min(sample_count, graph.count)
        ids = np.linspace(0,graph.count-1,sample_count,dtype=np.int32)
        codec.fit(np.asarray(graph.vectors[ids]))
        if kind == 'pca':
            codec.components = 80
            codec.basis = np.ascontiguousarray(codec.basis[:,:80])
        directory = args.output/name
        save_codec(codec,directory)
        internal_dir=args.output/'hnsw-order'/name
        internal_dir.mkdir(parents=True)
        shape=(graph.count,)+codec.code_shape()
        ordered=np.lib.format.open_memmap(directory/'codes.npy',mode='w+',dtype=codec.code_dtype(),shape=shape)
        internal=np.lib.format.open_memmap(internal_dir/'codes.npy',mode='w+',dtype=codec.code_dtype(),shape=shape)
        for start in range(0,graph.count,8192):
            end=min(start+8192,graph.count)
            batch=codec.encode(np.asarray(graph.vectors[start:end]))
            internal[start:end]=batch
            ordered[labels[start:end]]=batch
        ordered.flush();internal.flush()
        native=directory/'native';native.mkdir()
        for key,array in codec.arrays().items():
            np.asarray(array,dtype='<f4',order='C').tofile(native/(key+'.bin'))
        print(json.dumps({'name':name,'training_graph_sha256':graph_hash,'sample_count':sample_count,
                          'config':codec.config(),'bytes':int(ordered.nbytes),'rerank':False}),flush=True)
        del ordered,internal


if __name__ == '__main__':
    main()
