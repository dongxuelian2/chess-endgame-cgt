#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${1:?pass the completed atlas scratch directory}" && pwd)
scratch="$root/atlas/horizon"
out="$here/results"
mkdir -p "$scratch" "$out"
export PYTHONPATH="$here/..:$root/pychess${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1
for m in 10 11 12; do
    archive="$root/atlas/games-$m.zst"
    if [[ ! -f "$archive" ]]; then
        curl -fLsS --retry 3 "https://database.lichess.org/standard/lichess_db_standard_rated_2013-$m.pgn.zst" -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
done
(cd "$root/atlas" && sha256sum -c "$here/games.sha256")
python "$here/fresh.py" "$root"
python "$here/build.py" "$root"
python "$here/sample.py" "$root" --depth 5
python "$here/sample.py" "$root" --depth 7 --games
python "$here/sample.py" "$root" --depth 5 --all-games
python "$here/results.py" "$root" "$out"
python "$here/depth.py" "$root" "$out" --population games7
python "$here/depth.py" "$root" "$out" --population all5
python "$here/dominance.py" "$root" "$out"
python "$here/versions.py" "$root" "$out"
python "$here/policy.py" "$root" "$out"
python "$here/bottleneck.py" "$root" "$out"
for a in 2 3; do
    "$scratch/engine/src/stockfish" traces none 0 "$a" 1 "$root/atlas/$a-1.bin" "$scratch/$a.raw"
done
python "$here/compress.py" "$root" "$out"
python "$here/figure.py" "$root" "$out"
g++ -O3 -march=native -std=c++17 "$here/race.cpp" -o "$scratch/race"
g++ -O3 -march=native -std=c++17 "$here/pins.cpp" -o "$scratch/pins"
for a in 2 3; do
    "$scratch/race" "$root/atlas" "$a" tables "$scratch/race-$a.bin"
    for f in pilot remaining; do
        "$scratch/race" "$root/atlas" "$a" "$f" "$scratch/race-$a-$f.txt"
    done
done
for a in 3 4 5; do
    for f in pilot remaining; do
        "$scratch/pins" "$root/atlas" "$a" "$f" "$scratch/pin-$a-$f.txt"
    done
done
python "$here/race_results.py" "$root" "$out"
python "$here/race_audit.py" "$root" "$out"
python "$here/pins.py" "$root" "$out"
python "$here/audit.py" "$root" "$out"
python "$here/example.py" "$root" "$out"
