#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python="${PYTHON:-python3}"
out="${OUTPUT:-results/local-$(date +%Y%m%d-%H%M%S)}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"${JOBS:-2}"
ctest --test-dir build --output-on-failure
dataset_args=()
if [[ -n "${DATASET:-}" ]]; then dataset_args=(--dataset "$DATASET"); fi
"$python" -B pipeline.py prepare --output "$out" "${dataset_args[@]}" --vectors "${VECTORS:-20000}" --queries "${QUERIES:-500}"
"$python" -B pipeline.py build --output "$out"
"$python" -B pipeline.py encode --output "$out"
"$python" -B pipeline.py query --output "$out" --warmup "${WARMUP:-100}" --repeats "${REPEATS:-3}"
