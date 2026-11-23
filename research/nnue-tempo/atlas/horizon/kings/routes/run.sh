#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${1:?pass the completed king study scratch root}" && pwd)
atlas="$root/atlas"
work="$atlas/horizon/kings/routes"
out="$here/results"
mkdir -p "$work" "$out"
export PYTHONPATH="$here/..:$here/../..:$here/../../..:$here/../../../../circuits:$here/../../../..:$root/pychess${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1
g++ -O3 -std=c++17 "$here/screen.cpp" -o "$work/screen"
for a in 3 4; do
    for b in 1 2 3 4 5; do
        "$work/screen" "$atlas" "$a" "$b" "$work/$a-$b.pairs" strict
    done
    "$work/screen" "$atlas" "$a" 1 "$work/relaxed-$a.pairs" relaxed
done
for a in 2 3 4 5; do
    "$work/screen" "$atlas" "$a" 1 "$work/solo-$a.pairs" solo
    "$work/screen" "$atlas" "$a" 1 "$work/promote-$a.txt" promotion
done
python "$here/panels.py" "$root" "$out"
python "$here/study.py" "$root" "$out"
for analysis in interventions switches rank_shift readout pawn_channels promotion_geometry blockers tactics destinations promotion_trial games audit; do
    python "$here/$analysis.py" "$root" "$out"
done
