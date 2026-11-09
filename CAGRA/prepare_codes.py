"""Write shared codes.npy/native models in original dataset ID order.

For the existing DEEP comparison, pass --model-source for PQ/PCA and
--reference-hnsw for INT8/Local. Reference files are opened read-only.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from ann_quantization.codecs import make_codec, load_codec, save_codec


def fvecs(path):
    with path.open('rb') as stream:
        dimension = struct.unpack('<I', stream.read(4))[0]
    if dimension not in (96, 128) or path.stat().st_size % ((dimension+1)*4):
        raise ValueError('Expected complete 96/128-D fvecs')
    records = np.memmap(path, dtype='<f4', mode='r').reshape(-1, dimension+1)
    if not np.all(records.view('<u4')[:, 0] == dimension):
        raise ValueError('Inconsistent fvecs dimensions')
    return records[:, 1:]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(base_path, method, output, model_source=None, reference_hnsw=None, code_size=None):
    base = fvecs(base_path)
    n, dimension = base.shape
    output = output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT or output.exists():
        raise ValueError('Output must be a new directory inside CAGRA')
    if model_source:
        codec = load_codec(model_source)
        if codec.dimension != dimension or codec.kind != method:
            raise ValueError('Reference model method/dimension mismatch')
        if code_size is not None and codec.code_shape()[0] != code_size:
            raise ValueError('Reference model code size mismatch')
        provenance = {'model_source': str(model_source.resolve()),
                      'model_sha256': digest(model_source/'codec.npz')}
    else:
        options = {}
        if method == 'pq':
            options['subspaces'] = code_size or (48 if dimension == 96 else 64)
        if method == 'pca':
            options['components'] = code_size or 80
        codec = make_codec(method, dimension, **options)
        samples = min(n, 100000 if method == 'int8' else 20000 if method == 'local' else 10000)
        vectors = base
        if reference_hnsw:
            sys.path.insert(0, str(ROOT.parent/'HNSW/src'))
            from hnsw_graph import HnswGraph
            reference = HnswGraph(reference_hnsw, dimension=dimension)
            if reference.count != n:
                raise ValueError('Reference HNSW and base row counts differ')
            labels = reference.labels
            if not np.array_equal(np.sort(labels), np.arange(n)):
                raise ValueError('Reference HNSW labels are not a dataset permutation')
            for start in range(0, n, 8192):
                end = min(n, start+8192)
                if not np.array_equal(reference.vectors[start:end], base[labels[start:end]]):
                    raise ValueError('Reference HNSW vectors differ from the requested base')
            vectors = reference.vectors
        codec.fit(np.asarray(vectors[np.linspace(0, n-1, samples, dtype=np.int32)]))
        provenance = {'training_samples': samples, 'sampling': 'linspace in HNSW internal order' if reference_hnsw else 'linspace in dataset order',
                      'reference_hnsw': str(reference_hnsw) if reference_hnsw else None}
    required = n * int(np.prod(codec.code_shape())) * np.dtype(codec.code_dtype()).itemsize
    if shutil.disk_usage(ROOT).free < required + 64*1024**2:
        raise ValueError('Insufficient disk space for this one representation')
    output.mkdir(parents=True, exist_ok=False)
    save_codec(codec, output)
    native = output/'native'
    native.mkdir()
    for key, value in codec.arrays().items():
        np.asarray(value, dtype='<f4', order='C').tofile(native/(key+'.bin'))
    codes = np.lib.format.open_memmap(output/'codes.npy', mode='w+', dtype=codec.code_dtype(),
                                    shape=(n,)+codec.code_shape())
    for start in range(0, n, 8192):
        end = min(n, start+8192)
        codes[start:end] = codec.encode(np.asarray(base[start:end]))
    codes.flush()
    del codes
    record = dict(method=method, vectors=n, dimensions=dimension, order='original dataset ID',
                  base_sha256=digest(base_path), codes_sha256=digest(output/'codes.npy'), **provenance)
    (output/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--method', choices=('int8','pq','pca','local'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-source', type=Path)
    parser.add_argument('--reference-hnsw', type=Path)
    parser.add_argument('--code-size', type=int)
    args=parser.parse_args()
    print(json.dumps(prepare(args.base, args.method, args.output, args.model_source,
                             args.reference_hnsw, args.code_size), indent=2))

if __name__ == '__main__':
    main()
