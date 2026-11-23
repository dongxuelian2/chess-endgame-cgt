#!/usr/bin/env bash
# Prepare clean upstream checkouts, frozen networks, tablebases and the game corpus.
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
dest=${1:?provide a fresh working directory}
mkdir -p "$dest"
cd "$dest"
git clone -q https://github.com/niklasf/python-chess pychess
git -C pychess checkout -q 6e58bd46d99d40e6a75f2ab7b8f6e8c3f819ba2a
for entry in 'sf12 sf_12' 'sf141 sf_14.1' 'sf18 sf_18'; do
  read -r tree tag <<< "$entry"
  git -c advice.detachedHead=false clone -q --depth 1 -b "$tag" https://github.com/official-stockfish/Stockfish "$tree"
  git -c advice.detachedHead=false clone -q "file://$PWD/$tree" "clean-$tree"
  case "$tree" in
    sf12) nets=(82215d0fd0df);;
    sf141) nets=(13406b1dcbe0);;
    sf18) nets=(c288c895ea92 37f18f62d772);;
  esac
  for id in "${nets[@]}"; do
    file="nn-$id.nnue"
    digest=""
    for source in https://raw.githubusercontent.com/official-stockfish/networks/master \
                  https://media.githubusercontent.com/media/official-stockfish/networks/master \
                  https://tests.stockfishchess.org/api/nn; do
      if curl -fsSL --retry 2 "$source/$file" -o "$tree/src/$file"; then
        digest=$(sha256sum "$tree/src/$file")
        if [[ "$digest" == "$id"* ]]; then break; fi
      fi
    done
    [[ "$digest" == "$id"* ]]
    cp "$tree/src/$file" "clean-$tree/src/$file"
  done
  if [[ "$tree" = sf18 ]]; then python "$study/instrument18.py" "$tree"
  else python "$study/instrument.py" "$tree"; fi
  for checkout in "$tree" "clean-$tree"; do
    make -C "$checkout/src" -j"${JOBS:-6}" build ARCH="${ARCH:-x86-64-avx2}"
  done
done
mkdir -p tb5
for material in KRPvKR KNPvKP KBPvKP KPPvKP KQPvKQ KRPvKP KRPPvK KRNvKR; do
  curl -fsSL --retry 2 "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/$material.rtbw" -o "tb5/$material.rtbw.part"
  mv "tb5/$material.rtbw.part" "tb5/$material.rtbw"
done
curl -fsSL --retry 2 https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst -o lichess-2013-01.pgn.zst
echo 'aa40b3671fa3cf1072eb182892cd90b0e1e003a4a5943492f64b77e7f3fd1635  lichess-2013-01.pgn.zst' | sha256sum -c -
