#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${1:?pass completed NNUE scratch root}" && pwd)
routes="$here/../atlas/horizon/kings/routes"
export PYTHONPATH="$here:$routes:$routes/..:$routes/../..:$routes/../../..:$here/../circuits:$here/..:$root/pychess${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1
work="$root/manuscript/motifs"
mkdir -p "$work" "$here/results"
g++ -O3 -std=c++17 "$here/motif_rectangles.cpp" -o "$work/scan"
for a in 2 3 4 5; do
    "$work/scan" "$root/atlas" "$a" 1 "$work/$a-1.txt"
done
"$work/scan" "$root/atlas" 3 4 "$work/3-4.txt"
python "$here/motif_rectangles.py" "$root" "$here/results"
python "$here/motif_trace.py" "$root" "$here/results"
python "$here/motif_audit.py" "$root" "$here/results"
python "$here/board_figure.py"
