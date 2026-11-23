#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${1:?pass the prepared scratch directory}" && pwd)
out="$here/results"
mkdir -p "$root/atlas" "$out"
export OPENBLAS_NUM_THREADS=1 PYTHONPATH="$root/pychess${PYTHONPATH:+:$PYTHONPATH}"
for month in 04 05 06 07 08 09; do
  file="$root/atlas/games-$month.zst"
  if [[ ! -f "$file" ]]; then
    curl -fsSL --retry 3 "https://database.lichess.org/standard/lichess_db_standard_rated_2013-$month.pgn.zst" -o "$file.part"
    mv "$file.part" "$file"
  fi
done
(cd "$root/atlas" && sha256sum -c "$here/game-archives.sha256")
python "$here/instrument.py" "$root"
python "$here/instrument18.py" "$root"
python "$here/run.py" "$root"
python "$here/run.py" "$root" --replies
python "$here/run.py" "$root" --remaining
python "$here/analyze.py" "$root"
python "$here/analyze.py" "$root" --remaining
python "$here/games.py" "$root"
python "$here/validate.py" "$root"
python "$here/validate.py" "$root" --games
python "$here/game_decisions.py" "$root"
python "$here/panels.py" "$root"
python "$here/panels.py" "$root" --remaining
python "$here/weights.py" "$root"
python "$here/analyze.py" "$root" --decisions
python "$here/analyze.py" "$root" --remaining --decisions
python "$here/summarize.py" "$root" "$out"
python "$here/finish.py" "$root" "$out"
python "$here/deletion.py" "$root" "$out"
python "$here/audit_choices.py" "$root" "$out"
python "$here/symmetry.py" "$root" "$out"
python "$here/release.py" "$root" "$out"
python "$here/residual.py" "$root" "$out"
python "$here/sparse.py" "$root" "$out"
python "$here/increment.py" "$root" "$out"
python "$here/prerequisite.py" "$root" "$out"
python "$here/analyze.py" "$root" --crossover
python "$here/analyze.py" "$root" --remaining --crossover
python "$here/rectangles.py" "$root" "$out"
python "$here/promotion_trap.py" "$root" "$out"
python "$here/catalog.py"
