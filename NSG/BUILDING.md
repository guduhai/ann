# NSG: Linux / Android CPU builds

The existing `src/`, `vendor/`, build parameters and query configuration are shared.
Both targets configure from this directory's CMakeLists.txt. Platform build settings
are in `cmake/linux.cmake` and `cmake/android.cmake`. Build artifacts are not source.

Linux (run from the repository root; use linux-arm64 on a Raspberry Pi):

```bash
cmake -S NSG -B NSG/build/linux-arm64 -DCMAKE_BUILD_TYPE=Release
cmake --build NSG/build/linux-arm64 -j2
```

The Linux native build requires the same system dependencies as the original
GitHub project. VAMANA additionally needs AIO, LAPACKE, OpenBLAS and Boost
program_options; ARM uses SIMDe-compatible headers. Native CPU flags are disabled
when cross-compiling; ANN_NATIVE_CPU can also be disabled explicitly.

Android (host-side NDK r26c, API 28 or higher; tested device is Android 14):

```bash
cmake -S NSG -B NSG/build/android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DANDROID_STL=c++_static \
  -DANN_BOOST_INCLUDE_DIR="/path/to/boost-1.71/include" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build NSG/build/android-arm64 -j2
```

Dependency paths above are placeholders, not machine-specific paths to commit.
Boost 1.71 headers/program_options and SIMDe 0.8.2 are the versions used in the
Android port. Dependency source packages may be stored in ignored `.deps/` or
outside the repository. Linux libraries cannot be reused as Android libraries.
VAMANA's Android build supports the existing float memory graph build
(`--build_PQ_bytes 0`); it does not enable upstream disk/PQ/OPQ construction.
Its separate four-method quantized query program is retained.

Programs are under `build/<target>/bin/`. The common names are `ann_build`,
`ann_query` and, where applicable, `ann_knn`, `ann_query_official`, `fvecs_to_bin`.
`nsg_search` remains available. NSG/NSSG/VAMANA also retain their former
Android executable names as copies, so deployment need not rename old programs.

## Existing launchers

`run_build.py` and `run_search.py` remain the public entries. They print the
command by default; only `--execute` runs the algorithm. Local execution remains
the default, with the old `build/bin` path. Select a new local build using
`--binary-dir NSG/build/linux-arm64/bin`.

For Android, use `--target android --device SERIAL`. Python runs on the host,
not on the phone. `--binary-dir` then means an absolute device path; by default it
is `/data/local/tmp/ann/bin/nsg`. Building does not deploy files: push
the selected programs explicitly with ADB before requesting execution.

Android search requires explicit absolute device `--graph` and `--query` paths,
plus `--base` for the NSG/NSSG/VAMANA float baseline, or `--codes` and `--model`
for quantized methods. Query output must be a new file under
`/data/local/tmp/ann/results/nsg/`; build output must be a new directory
under `/data/local/tmp/ann/indexes/nsg/`. Override the runtime root with
`--device-root`. A dry run may read device file headers but does not run search.
Local search output retains the original requirement to stay in this index's
`results/`. Existing outputs are never intentionally overwritten. Failed runs
can leave partial outputs; inspect them and choose a new path before retrying.

Data/model/graph paths are not renamed by this source reorganization. The shared
`codes/sift1m/quant` and `models/sift1m/quant` remain reusable for NSG/NSSG/VAMANA;
HNSW compact search must use encoding in that graph's internal node order.
Server/phone historical datasets, models, indexes and results are preserved.

The old date-specific Android experiment scripts are retained in the migration
backup, not installed as a second public workflow. Existing HNSW prepare_codes
and export_compact_graph tools remain in src/. Standard launchers generate no
`*_command.txt` files and do not automatically run a full benchmark campaign.

Python dependencies are maintained in the repository-root requirements.txt.
From the repository root, install them with `python -m pip install -r requirements.txt`
in the active virtual environment. The obsolete per-index manifests have been removed.
