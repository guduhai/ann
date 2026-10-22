#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
python="${PYTHON:-$root/../.venv/bin/python}"
if ! command -v "$python" >/dev/null 2>&1; then
  echo "找不到 Python 解释器：$python。请按根目录 README.md 配置环境，或用 PYTHON 指定解释器路径。" >&2
  exit 1
fi
out="${OUTPUT:-results/local-$(date +%Y%m%d-%H%M%S)}"
build_dir="${BUILD_DIR:-build/linux-arm64}"
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" -j"${JOBS:-2}"
dataset_args=()
if [[ -n "${DATASET:-}" ]]; then dataset_args=(--dataset "$DATASET"); fi
"$python" -B pipeline.py prepare --output "$out" "${dataset_args[@]}" --vectors "${VECTORS:-20000}" --queries "${QUERIES:-500}"
"$python" -B pipeline.py build --output "$out" --bin "$build_dir/bin" --threads "${BUILD_THREADS:-4}"
"$python" -B pipeline.py encode --output "$out"
"$python" -B pipeline.py query --output "$out" --bin "$build_dir/bin" --warmup "${WARMUP:-100}" --repeats "${REPEATS:-3}"
