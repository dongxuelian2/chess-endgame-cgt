#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${1:?pass the prepared scratch directory}" && pwd)
out=${2:?pass the output directory}
mkdir -p "$out" "$root/pawn-geometry"
export PYTHONPATH="$(dirname "$here"):$(dirname "$here")/circuits:$root/pychess${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
for mat in KRRvKP KRBvKP KRNvKP; do
  file="$root/pawn-geometry/$mat.rtbw"
  if [[ ! -f "$file" ]]; then
    curl -fLsS --retry 3 "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/$mat.rtbw" -o "$file.part"
    mv "$file.part" "$file"
  fi
done
(cd "$root/pawn-geometry" && sha256sum -c "$here/pin-tables.sha256")
python "$here/seeds.py" "$root" "$out"
python "$here/boards.py" "$out"
python "$here/transfer.py" "$root"
python "$here/transfer.py" "$root" --unconditioned
python "$here/measure.py" "$root" "$out"
python "$here/audit.py" "$root" "$out"
python "$here/king_sweep.py" "$root" "$out"
python "$here/king_neighbors.py" "$root" "$out"
python "$here/channels.py" "$root" "$out"
python "$here/file_control.py" "$root" "$out"
python "$here/two_rules.py" "$root" "$out"
python "$here/pins.py" "$root" "$out"
python "$here/played.py" "$root" "$out"
