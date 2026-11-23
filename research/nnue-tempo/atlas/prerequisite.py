"""Separate a possible reply from the pre-existing board condition that makes it possible."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import chess
import numpy as np

from summarize import MATS, score, write
from validate import RULES, board, sign


def predicates(a, b, wk, bk, y, x):
    q = board(a, b, wk, bk, y, x)
    attacked = q.is_attacked_by(True, y)
    pinned = q.is_pinned(False, y)
    loose = chess.square_distance(bk, y) > 1
    wg = chess.square_distance(wk, x) == 1
    return {
        "unpin": pinned,
        "stay_on_pin_line": pinned,
        "escape_attack": attacked,
        "counterattack": attacked,
        "king_rescue": attacked and loose,
        "king_attack_guarded": wg,
        "king_attack_loose": not wg,
        "pin": not q.is_pinned(True, x),
    }


def records(root, mode):
    if mode == "replication":
        for a, b in MATS:
            p = root / "atlas/remaining/metrics" / f"{a}-{b}.npz"
            if p.exists():
                for r in np.load(p)["panel"]:
                    i = int(r[0])
                    yield (
                        a,
                        b,
                        i >> 12,
                        i >> 6 & 63,
                        i & 63,
                        int(r[-2]),
                        int(r[-1]),
                        int(r[2]),
                        int(r[3]),
                        int(r[6]),
                    )
    else:
        with gzip.open(root / "atlas/game-results.gz", "rt") as f:
            for _, _, fen, u, v, a, b, ((hi, hm), (lo, lm)) in map(json.loads, f):
                if hi[1] != lo[1] or sign(hi[-1]) == sign(lo[-1]):
                    continue
                if hi[-1] < lo[-1]:
                    hi, lo, hm, lm, u, v = lo, hi, lm, hm, v, u
                q = chess.Board(fen)
                y = next(iter(chess.SquareSet(q.occupied_co[False] & ~q.kings)))
                yield (
                    a,
                    b,
                    q.king(True),
                    q.king(False),
                    y,
                    chess.parse_square(u[2:4]),
                    chess.parse_square(v[2:4]),
                    hm[0],
                    lm[0],
                    hi[2] - lo[2],
                )


def main(root, out):
    with (out / "incremental-rules.csv").open() as f:
        signs = {
            r["rule"]: int(r["direction"])
            for r in csv.DictReader(f)
            if r["panel"] == "replication"
        }
    result = []
    for mode in ("replication", "played"):
        agg = {}
        for a, b, w, k, y, x, z, h, l, nn in records(root, mode):
            if (h ^ l) & 511:
                continue
            ph = predicates(a, b, w, k, y, x)
            pl = predicates(a, b, w, k, y, z)
            for t, high_pre in ph.items():
                j = RULES.index(t)
                hd = bool(h >> j & 1)
                ld = bool(l >> j & 1)
                low_pre = pl[t]
                assert (not hd or high_pre) and (not ld or low_pre)
                if hd == ld:
                    continue
                key = (
                    t,
                    "both_eligible"
                    if high_pre == low_pre
                    else "different_prerequisite",
                )
                a1 = agg.setdefault(key, [0, 0, 0])
                a1[0] += 1
                a1[1] += float(score(signs[t] * (int(hd) - int(ld))))
                a1[2] += float(score(nn))
        for (t, k), (n, correct, neural) in sorted(agg.items()):
            result.append(
                {
                    "panel": mode,
                    "rule": t,
                    "geometry": k,
                    "n": n,
                    "rule_accuracy": correct / n,
                    "sf141": neural / n,
                }
            )
    write(out / "prerequisites.csv", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
