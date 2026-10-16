# CPU graph search: NSG with CAGRA-style pruning

[中文完整说明](README_CN.md) | [Repository overview](../README.md)

This experiment uses native NSG/EFANNA graph construction with CAGRA-style
detour-count pruning. A standalone C++ query program supports float32, INT8, PCA,
PQ and local adaptive INT8. Quantized search does not rerank with full precision
vectors. Training and encoding use the repository's shared `ann_quantization`
package and NumPy.

This is a CPU research implementation, not NVIDIA CUDA/cuVS CAGRA. Faiss, CUDA,
cuVS are not required by CAGRA. The repository-wide dependency list is maintained in the root README.

See the [repository root guide](../README.md) for shared dependencies, environment setup and builds for all five indexes.

## Repository layout

Clone the whole repository and preserve this layout:

```text
ann/
├── ann_quantization/
│   ├── __init__.py
│   └── codecs.py
└── CAGRA/
    ├── ann_codecs.py
    ├── pipeline.py
    ├── python_pi.sh
    ├── CMakeLists.txt
    ├── include/
    ├── src/
    ├── tests/
    └── vendor/
```

`ann_codecs.py` locates the sibling package relative to its own file, so the
repository can have any name and location. Downloading only `CAGRA/` is insufficient.

The commands below start in `CAGRA/`, after setup and compilation using the [repository root guide](../README.md). `python_pi.sh` uses the repository-root `.venv/bin/python`, clears external environment paths and changes into `CAGRA/`. Relative arguments are interpreted from `CAGRA/`.

## Synthetic example: no dataset download

```bash
OUT=results/synthetic-1024
bash python_pi.sh pipeline.py prepare --output "$OUT" \
  --vectors 1024 --queries 32 --dim 128
bash python_pi.sh pipeline.py build --output "$OUT"
bash python_pi.sh pipeline.py encode --output "$OUT" \
  --train 1024 --pca-dim 80 --pq-m 64 --pq-k 256 \
  --clusters 32 --codec-iterations 10
bash python_pi.sh pipeline.py query --output "$OUT" \
  --search-l 20,50,100,200 --warmup 20 --repeats 3
```

Without `--dataset`, preparation generates normal random vectors with seed
`20261014` and computes exact Top-10 ground truth. This example exercises the
pipeline; its accuracy and speed do not represent SIFT1M performance.

For an automatic build and run, the repository-root virtual environment is used by default. This workflow also runs compilation and tests:

```bash
DATASET= \
OUTPUT=results/synthetic-next VECTORS=1024 QUERIES=32 WARMUP=20 REPEATS=3 \
bash run_local.sh
```

`run_local.sh` uses the repository-root virtual environment unless `PYTHON` is explicitly set. Outputs must stay inside `CAGRA/`.
Use a new output directory for each experiment; existing experiment, encoding
and query outputs are not overwritten.

## Download SIFT1M and run the full experiment

Use the [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks)
[SIFT-128 Euclidean HDF5 dataset](https://ann-benchmarks.com/sift-128-euclidean.hdf5),
with 1,000,000 base vectors, 128 dimensions and 10,000 queries. Download it outside
the repository:

```bash
DATASET="${XDG_CACHE_HOME:-$HOME/.cache}/ann-datasets/sift-128-euclidean.hdf5"
mkdir -p "$(dirname "$DATASET")"
curl --fail --location --output "$DATASET" \
  https://ann-benchmarks.com/sift-128-euclidean.hdf5
```

The HDF5 reader expects:

| Key | Shape and meaning |
| --- | --- |
| `train` | `(N, D)` base vectors, converted to float32 on input |
| `test` | `(Q, D)` query vectors with matching dimensions |
| `neighbors` | `(Q, K)`, `K >= 10`; exact Euclidean neighbors in distance order, as zero-based row IDs into `train` |

Vectors must contain finite values. `distances` and other additional datasets
are not read. Full-data preparation uses the first ten `neighbors` columns;
subsets recompute their own exact Top-10 ground truth.

After installation, compilation and download, run in the same Bash session:

```bash
OUT=results/sift1m-m64-pca80
bash python_pi.sh pipeline.py prepare --output "$OUT" \
  --dataset "$DATASET" --vectors 0 --queries 10000
bash python_pi.sh pipeline.py build --output "$OUT" \
  --gk 64 --knn-l 80 --iterations 8 --sample 10 --reverse 100 \
  --degree 32 --build-l 64 --build-c 132
bash python_pi.sh pipeline.py encode --output "$OUT" \
  --train 10000 --pca-dim 80 --pq-m 64 --pq-k 256 \
  --clusters 32 --codec-iterations 10
bash python_pi.sh pipeline.py query --output "$OUT" \
  --methods float32,int8,pca,pq,local \
  --search-l 20,50,100,200 --warmup 200 --repeats 3
```

`--vectors 0` selects all training vectors. Graph construction currently uses one
thread. Prepared data, intermediate graphs and representations require several
GB of disk space; construction and training also need additional RAM.

For a smaller SIFT experiment, choose a new output directory and change the
preparation counts to `--vectors 20000 --queries 500`. PQ requires the dimension
to be divisible by its subspace count: M64 works for 128 dimensions; use an
appropriate value such as `--pq-m 32` for 96 dimensions. Training samples must
cover at least the PQ centroid count and the Local cluster count.

Query an existing graph and representations again with a new result name:

```bash
bash python_pi.sh pipeline.py query --output "$OUT" \
  --run-name query-again --search-l 20,50,100,200 --warmup 200 --repeats 3
```

## Tests

Users can run the pruning test and small-case verification with:

```bash
ctest --test-dir build --output-on-failure
bash python_pi.sh tests/verify.py \
  --case results/synthetic-1024 --bin build --output-name verification
```

The synthetic case must already have a graph and all five representations.
Verification compares C++ distances with NumPy, checks exhaustive search,
quantized search without full precision base files, and malformed input rejection.
It is limited to 4,096 base vectors. Choose a new `--output-name` when repeating it.

For AddressSanitizer and UndefinedBehaviorSanitizer, build a separate directory
and use `--bin build-asan` in the verification command:

```bash
cmake -S . -B build-asan -DANN_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++
cmake --build build-asan -j2
```

## Results and interpretation

An experiment directory contains:

- `dataset.json`: source, input sizes and hashes.
- `index.graph` and `build.json`: CSR graph, parameters, timings and graph checks.
- `codecs/` and `codecs/manifest.json`: representations, models and identity hashes.
- `query-results/summary.json`: platform, query binary hash and individual metrics.
- `query-results/RESULTS_CN.md`: generated result table.
- `query-results/*.labels`: Top-10 IDs for each query configuration.

Each method and search L runs in a separate C++ process. Query timing includes
projection, lookup-table preparation and traversal, but excludes loading, warmup,
recall computation and file output. Reported speed uses the median repeated query
time. Peak memory uses Linux `VmHWM`; `ru_maxrss` is diagnostic only. Compare
speed together with recall: equal L does not imply equal recall.

Quantized search loads one representation into RAM and does not open `base.f32`
or `codecs/float32.bin`. It does not use mmap or full precision reranking. Memory
savings apply to search, not necessarily construction. Cold-cache disk I/O has
not been measured.

Historical Raspberry Pi artifacts may exist locally under
`results/sift1m-m64-pca80-run1/`; generated files are ignored by Git and are not
part of a source checkout. For old reports with generic platform wording, use
the `machine` and `platform` fields in `summary.json`.

## Implementation and attribution

EFANNA NN-Descent produces candidates. NSG collects candidates and handles reverse
edges and connectivity repair. Detour-count pruning orders candidates by distance
and prioritizes edges with fewer two-hop detours. The original NSG `tree_grow`
repair remains, so some degrees can exceed the requested R; `build.json` reports
the actual graph statistics.

The source layout, default representations and version 1 little-endian binary
formats are described in the [Chinese reference](README_CN.md).
See [third-party notices](THIRD_PARTY_NOTICES.md),
[upstream revisions](UPSTREAM.json), [vendored files](VENDORED_FILES.json),
and [source changes](vendor_changes.patch). Preserve the upstream licenses and
author notices under `vendor/`.

Do not commit `.venv/`, `build*/`, `results/`, `logs/`, Python caches or datasets.
Reproduction uses source, dependencies, parameters and input data rather than
machine-specific environments or executables.
