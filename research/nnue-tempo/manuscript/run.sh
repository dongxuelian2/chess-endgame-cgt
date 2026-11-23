#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${1:?pass completed NNUE scratch root}" && pwd)
routes="$here/../atlas/horizon/kings/routes"
export PYTHONPATH="$routes:$routes/..:$routes/../..:$routes/../../..:$here/../circuits:$here/..:$root/pychess${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1
work="$root/manuscript"
mkdir -p "$work" "$here/results"
g++ -O3 -std=c++17 "$routes/screen.cpp" -o "$work/screen"
for a in 2 3 4 5; do
    "$work/screen" "$root/atlas" "$a" 1 "$work/alternate-$a-1.txt" promotion 728903
done
for pair in '3 4' '4 2'; do
    read -r a b <<< "$pair"
    "$work/screen" "$root/atlas" "$a" "$b" "$work/alternate-$a-$b.txt" strict 728903
done
for pair in '3 4' '5 4' '4 2' '5 2'; do
    read -r a b <<< "$pair"
    "$work/screen" "$root/atlas" "$a" "$b" "$work/solo-$a-$b.txt" solo 31027
done
python "$here/transfer.py" "$root" "$here/results"
python "$here/cohorts.py" "$root" "$here/results"
python "$here/promotion.py" "$root" "$here/results"
python "$here/promotion_errors.py" "$root" "$here/results"
python "$here/perspectives.py" "$root" "$here/results"
python "$here/king_crossing.py" "$root" "$here/results"
python "$here/tactical.py" "$root" "$here/results"
python "$here/geometry.py" "$root" "$here/results"
python "$here/stability.py" "$root" "$here/results"
python "$here/capture_regimes.py" "$root" "$here/results"
python "$here/specificity.py" "$root" "$here/results"
python "$here/figures.py" "$root"
bash "$here/cycles.sh" "$root"
bash "$here/motifs.sh" "$root"
bash "$here/prediction.sh" "$root"
python "$here/paper_figures.py" "$root"
typst compile "$here/paper.typ" "$here/paper.pdf"
