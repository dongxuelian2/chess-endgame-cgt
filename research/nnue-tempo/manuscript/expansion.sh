#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "${1:?pass completed NNUE scratch root}" && pwd)
routes="$here/../atlas/horizon/kings/routes"
export PYTHONPATH="$here:$routes:$routes/..:$routes/../..:$routes/../../..:$here/../circuits:$here/..:$root/pychess${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1
work="$root/manuscript/expansion"
fetch() {
  local url=$1 output=$2
  if [[ ! -s "$output" ]]; then
    curl -fLsS --retry 3 --retry-delay 20 "$url" -o "$output.part"
    if [[ "$output" = *.zst ]]; then zstd -tq "$output.part"; fi
    mv "$output.part" "$output"
  fi
}
inputs() {
  local month material
  local tb=https://tablebase.lichess.ovh/tables/standard
  mkdir -p "$work/archives" "$work/tables"
  for month in 01 02 03 04 05 06; do
    fetch "https://database.lichess.org/standard/lichess_db_standard_rated_2014-$month.pgn.zst" \
      "$work/archives/2014-$month.zst"
  done
  for material in KNPPvKP KBPPvKP KRPPvKP KQPPvKP; do
    fetch "$tb/6-wdl/$material.rtbw" "$work/tables/$material.rtbw"
  done
  for material in KQPvKP KPPvKQ KPPvKR KPPvKB KPPvKN KNPPvK KBPPvK KRPPvK KQPPvK; do
    fetch "$tb/3-4-5-wdl/$material.rtbw" "$work/tables/$material.rtbw"
  done
}
step=${2:-all}
case "$step" in
  inputs)
    inputs
    ;;
  prepare)
    python "$here/expansion_classical.py" "$root"
    ;;
  games|synthetic|static)
    python "$here/expansion.py" "$root" "$here/results" "$step" "${@:3}"
    ;;
  donors)
    python "$here/expansion_donors.py" "$root" "$here/results"
    ;;
  matching)
    mkdir -p "$work/archives"
    for month in 07 08 09 10 11 12; do
      fetch "https://database.lichess.org/standard/lichess_db_standard_rated_2014-$month.pgn.zst" \
        "$work/archives/2014-$month.zst"
      if [[ ! -s "$work/2014-$month.jsonl.gz" ]]; then
        python "$here/expansion.py" "$root" "$here/results" games --archive "2014-$month"
      fi
    done
    python "$here/promotion_match.py" "$root" "$here/results"
    ;;
  search)
    python "$here/expansion_search.py" "$root" "$here/results"
    ;;
  root)
    python "$here/expansion_root.py" "$root" "$here/results"
    ;;
  modern)
    python "$here/expansion_modern.py" "$root" "$here/results"
    ;;
  all)
    inputs
    python "$here/expansion_classical.py" "$root"
    for pieces in 4 5 6; do
      python "$here/expansion.py" "$root" "$here/results" synthetic --pieces "$pieces"
    done
    for month in 01 02 03 04 05 06; do
      python "$here/expansion.py" "$root" "$here/results" games --archive "2014-$month"
    done
    python "$here/expansion.py" "$root" "$here/results" static
    python "$here/expansion_search.py" "$root" "$here/results"
    python "$here/expansion_donors.py" "$root" "$here/results"
    python "$here/expansion_modern.py" "$root" "$here/results"
    python "$here/expansion_root.py" "$root" "$here/results"
    python -c 'from pathlib import Path; from paper_figures import search_depths; import sys; search_depths(Path(sys.argv[1]))' "$here"
    ;;
  *)
    echo "Expected inputs, prepare, synthetic, games, static, donors, matching, search, root, modern, or all." >&2
    exit 2
    ;;
esac
