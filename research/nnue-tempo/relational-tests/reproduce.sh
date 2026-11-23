#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
parent=$(dirname "$here")
root=$(cd "${1:?pass the prepared scratch directory}" && pwd)
dest=${2:?pass the results directory}
mkdir -p "$dest" "$root/relational-tests"
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$parent:$parent/circuits:$parent/pawn-geometry:$root/pychess${PYTHONPATH:+:$PYTHONPATH}"
python "$here/model.py" "$root"
python "$here/certificate.py"
python "$here/signed.py"
for job in census additive replies search_games permute refinement confirmation branches race_test; do
  python "$here/$job.py" "$root" "$dest"
done
python "$here/intervals.py" "$dest"
python "$here/audit.py" "$root" "$dest"
