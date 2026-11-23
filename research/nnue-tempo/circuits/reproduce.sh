#!/usr/bin/env bash
# Run on a scratch directory prepared with setup.sh, collect.py, and game_pairs.py.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
study=$(dirname "$here")
data=$(cd "${1:?pass the prepared study scratch directory}" && pwd)
out=${2:?pass an output directory for the aggregate tables}
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$study:$data/pychess${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$data/circuits"
for m in 02 03; do
  pgn="$data/lichess-2013-$m.pgn.zst"
  if [[ ! -f "$pgn" ]]; then
    curl -fLsS --retry 3 "https://database.lichess.org/standard/lichess_db_standard_rated_2013-$m.pgn.zst" -o "$pgn.part"
    mv "$pgn.part" "$pgn"
  fi
done
(cd "$data" && sha256sum -c "$here/archives.sha256")
if [[ ! -x "$data/circuits/probe/src/stockfish" ]]; then
  git -c advice.detachedHead=false clone -q "file://$data/clean-sf141" "$data/circuits/probe"
  cp "$data/sf141/src/nn-13406b1dcbe0.nnue" "$data/circuits/probe/src/"
  python "$here/ft_export.py" "$data/circuits/probe"
  make -C "$data/circuits/probe/src" -j"${JOBS:-8}" build ARCH="${ARCH:-x86-64-avx2}"
fi
python "$study/game_pairs.py" "$data" --month 2013-02 --output circuits/feb.jsonl.gz
python "$here/siblings.py" "$data"
for month in 2013-01 2013-02 2013-03; do
  python "$here/contacts.py" "$data" "$month"
  if [[ "$month" != 2013-03 ]]; then python "$here/contacts.py" "$data" "$month" --forks; fi
done
python "$here/spectra.py" "$data"
python "$here/spectra.py" "$data" --contacts
python "$here/spectra.py" "$data" --forks
python "$here/certificates.py" "$data"
python "$here/feature_routes.py" "$data"
python "$here/edge_transfer.py" "$data"
python "$here/threat_reversal.py" "$data" --month 2013-01
python "$here/wdl_reversal.py" "$data"
for cohort in 2013-02 2013-03 wdl; do
  if [[ "$cohort" != wdl ]]; then python "$here/threat_reversal.py" "$data" --month "$cohort"; fi
  python "$here/interaction_layers.py" "$data" --month "$cohort"
  python "$here/thresholds.py" "$data" --month "$cohort"
  python "$here/audit.py" "$data" "$cohort"
done
python "$here/classical_terms.py" "$data"
python "$here/classical_terms.py" "$data" --full
python "$here/cutoffs.py" "$data"
python "$here/release.py" "$data" "$out"
python "$here/figure.py" "$out"
