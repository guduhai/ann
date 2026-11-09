#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$root"
exec bash python_pi.sh run_experiment.py "$@"
