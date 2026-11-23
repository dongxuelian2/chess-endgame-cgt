#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${1:?pass completed NNUE scratch root}" && pwd)
routes="$here/../atlas/horizon/kings/routes"
export PYTHONPATH="$here:$routes:$routes/..:$routes/../..:$routes/../../..:$here/../circuits:$here/..:$root/pychess${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1
work="$root/manuscript/rewriting"
mkdir -p "$work" "$here/results"
g++ -O3 -std=c++17 "$routes/screen.cpp" -o "$work/screen"
for a in 2 3 4 5; do
    "$work/screen" "$root/atlas" "$a" 4 "$work/open-$a.txt" open 56801
done
for analysis in promotion_prediction promotion_census promotion_residuals promotion_extension promotion_stress skewer_controls; do
    python "$here/$analysis.py" "$root" "$here/results"
done
