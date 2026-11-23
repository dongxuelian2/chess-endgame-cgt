"""Measure identified channels with each king's orientation explicit, including played moves."""

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network, credit
from promotion_geometry import relation
from promotion_trial import restrict
from summarize import write

from effects import batches, delta, patch


def channels(out):
    with (out / "promotion.csv").open() as f:
        return {
            ord(r["file"]) - 97: list(map(int, r["ft"].split()))
            for r in csv.DictReader(f)
            if r["intervention"] == "first_eight"
        }


def census(root, out, net, ranked):
    records = []
    for f in [1, 6, 7]:
        y = (8 + f) ^ 7
        for u in sorted(set(ranked[f][:4] + ([1869] if f == 1 else []))):
            black = u < 1024
            i = u % 1024
            sums = defaultdict(lambda: np.zeros(2))
            for a in range(2, 6):
                v.state(a, 1, 0, 0, 0, 0)
                for w in range(64):
                    if w % 8 >= 4:
                        continue
                    k, x = np.where(v.maps[a, 1][w, :, y, :]["v"] > 0)
                    king = k if black else np.full(len(k), w)
                    flip = 56 * int(black) + 7 * (king % 8 < 4)
                    canon = king ^ flip
                    bucket = 31 - 4 * (canon // 8) - (canon % 8 - 4)
                    T = net.w[:, :, :, i]
                    values = (
                        net.bias[i]
                        + T[bucket, 10, w ^ flip]
                        + T[bucket, 10, k ^ flip]
                        + T[bucket, int(not black), y ^ flip]
                        + T[bucket, 2 * (a - 1) + int(black), x ^ flip]
                    ).clip(0, 127)
                    rel = relation(a, x, y - 8, w, k)
                    for sf in np.unique((y ^ flip) % 8):
                        for g in (2, 3, 4):
                            s = values[(((y ^ flip) % 8) == sf) & (rel == g)]
                            sums[a, sf, g] += [len(s), s.sum()]
            W = net.w[:, int(not black), 8:56, i]
            wins = W.argmax(1) + 8
            for (a, sf, g), (n, total) in sorted(sums.items()):
                if n:
                    square = (48 if black else 8) + sf
                    records.append(
                        {
                            "file": chr(97 + f),
                            "ft": u,
                            "perspective": "black" if black else "white",
                            "white": a,
                            "own_file": chr(97 + sf),
                            "pawn_argmax_buckets": int((wins == square).sum()),
                            "relation": {2: "off", 3: "blocked", 4: "clear"}[g],
                            "boards": int(n),
                            "mean_byte": total / n,
                        }
                    )
    write(out / "perspective-census.csv", records)


def played_groups(root):
    for period, archive in [
        ("Apr-Sep", "atlas/played.gz"),
        ("Oct-Dec", "atlas/horizon/played.gz"),
    ]:
        groups = defaultdict(list)
        with gzip.open(root / archive, "rt") as stream:
            for _, game, fen, m, n, *_ in map(json.loads, stream):
                q = chess.Board(fen)
                w, k = q.king(True), q.king(False)
                s = chess.parse_square(m[:2])
                a = q.piece_type_at(s)
                pawn = q.pieces(chess.PAWN, False)
                if len(pawn) != 1 or a not in (2, 3, 4, 5):
                    continue
                y = next(iter(pawn))
                f = (y ^ (7 if w % 8 < 4 else 0)) % 8
                if y // 8 != 1:
                    continue
                x, z = [chess.parse_square(move[2:4]) for move in (m, n)]
                h, l = [v.state(a, 1, w, k, y, d) for d in (x, z)]
                if h["f"] != l["f"] or np.sign(h["w"]) == np.sign(l["w"]):
                    continue
                if h["w"] < l["w"]:
                    x, z = z, x
                r = [w, k, y, x, z, s]
                if restrict((a, *r)):
                    groups[a, f].append((r, game))
        yield period, groups


def played(root, out, net, ranked):
    records = []
    for period, groups in played_groups(root):
        stats = defaultdict(list)
        for (a, f), items in groups.items():
            rs, _ids = zip(*items)
            for r in batches(np.array(rs)):
                raw, p, _ = net.encode(r, a, 1)
                t = net.forward(raw, p)
                x = raw.clip(0, 127)
                native = credit(delta(t["score"]))
                stats[f].append(
                    np.c_[
                        native, *[patch(net, x, t, ranked[f][:j])[0] for j in (1, 4, 8)]
                    ]
                )
        for f, data in sorted(stats.items()):
            z = np.concatenate(data)
            records.append(
                {
                    "period": period,
                    "file": chr(97 + f),
                    "n": len(z),
                    "native": z[:, 0].mean(),
                    "drop_one": z[:, 1].mean(),
                    "drop_four": z[:, 2].mean(),
                    "drop_eight": z[:, 3].mean(),
                }
            )
    write(out / "played.csv", records)


if __name__ == "__main__":
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    v.init(root)
    net = Network(root)
    ranked = channels(out)
    census(root, out, net, ranked)
    played(root, out, net, ranked)
