#!/usr/bin/env bash
set -euo pipefail

# 根据脚本位置找到项目，不依赖当前终端目录。
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
venv_python="$root/.venv/bin/python"

if [[ ! -x "$venv_python" ]]; then
  echo "缺少项目虚拟环境，请先在 CAGRA 目录创建 .venv。" >&2
  exit 1
fi

# 清除外部 Python 路径，避免混入系统或其他环境的包。
unset PYTHONPATH PYTHONHOME
unset LD_LIBRARY_PATH LD_PRELOAD
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd "$root"
export TMPDIR="$root/build/tmp"
mkdir -p "$TMPDIR"

exec "$venv_python" -B "$@"
