#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${1:?pass completed NNUE scratch root}" && pwd)
routes="$here/../atlas/horizon/kings/routes"
export PYTHONPATH="$routes:$routes/..:$routes/../..:$routes/../../..:$here/../circuits:$here/..:$root/pychess${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1
work="$root/manuscript/cycles"
mkdir -p "$work" "$here/results"
g++ -O3 -std=c++17 "$here/cycles.cpp" -o "$work/scan"
seq 25 | xargs -P "${CYCLE_JOBS:-5}" -I@ bash -c '
    a=$((($1 - 1) / 5 + 1))
    b=$((($1 - 1) % 5 + 1))
    "$3/scan" "$2/atlas" "$a" "$b" "$3"
' sh @ "$root" "$work"
python "$here/cycle_analysis.py" "$root" "$here/results"
