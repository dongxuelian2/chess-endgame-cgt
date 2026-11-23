#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${1:?pass the completed horizon scratch root}" && pwd)
atlas="$root/atlas"
work="$atlas/horizon/kings"
out="$here/results"
mkdir -p "$work" "$out"
export PYTHONPATH="$here/..:$here/../..:$root/pychess${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1
python "$here/build.py" "$root"
engine="$work/engine/src/stockfish"
tb="$root/pychess/data/syzygy/regular"
g++ -std=c++17 -O3 -march=native "$here/../race.cpp" -o "$work/race"
for partition in pilot remaining; do
    "$work/race" "$atlas" 3 "$partition" "$work/race-$partition" "$work/exceptions-$partition"
done
g++ -std=c++17 -O3 "$here/rank_counts.cpp" -o "$work/rank_counts"
{
    echo white_man,family,total,incorrect,tied,correct,keep67_quarters,drop67_quarters
    for a in 2 3 4 5; do
        "$engine" rank3 "$tb" 0 "$a" 1 "$atlas/$a-1.bin" "$work/rank-$a.bin" >&2
        "$work/rank_counts" "$a" "$atlas/$a-1.bin" "$work/rank-$a.bin" "$work/../head.txt" "$work/../3.raw" "$work/stratified-$a.txt"
    done
} > "$out/rank-three.csv"
python "$here/bishop.py" "$root" "$out"
python "$here/rank-check.py" "$root" "$out"
python "$here/panel.py" "$root" "$out"
python "$here/replay.py" "$root" "$out"
for a in 4 5; do
    for input in "$work"/pairs-"$a"-1-*.in; do
        "$engine" third "$tb" 0 "$a" 1 "$input" "${input%.in}.third"
    done
done
python "$here/third.py" "$root" "$out"
g++ -std=c++17 -O3 "$here/substitutions.cpp" -o "$work/substitutions"
"$work/substitutions" "$atlas" "$work/substitutions.txt" "$out/substitution-screen.csv"
cut -d' ' -f1-5 "$work/substitutions.txt" > "$work/substitutions.in"
for a in 4 5; do
    "$engine" third "$tb" 0 "$a" 1 "$work/substitutions.in" "$work/substitutions-$a.third"
done
python "$here/paths.py" "$root" "$out"
