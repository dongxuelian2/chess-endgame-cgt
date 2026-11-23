"""One-reply interventions on the comparisons untouched by the original two rules."""

import argparse
import gzip
import json
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from measure import table

TYPES = ("king_takes_actor", "king_takes_any", "pawn_push_without_promotion")


def classify(root, r, tb):
    parent = chess.Board(r["parent"])
    vals = []
    tags = []
    for u in r["moves"]:
        b = parent.copy()
        move = chess.Move.from_uci(u)
        b.push(move)
        moves = list(b.legal_moves)
        reply = []
        for m in moves:
            king = b.piece_type_at(m.from_square) == chess.KING
            cap = b.is_capture(m)
            flags = [
                king and m.to_square == move.to_square,
                king and cap,
                not king and not cap and not m.promotion,
            ]
            b.push(m)
            try:
                v = -int(np.sign(tb.probe_wdl(b)))
            except chess.syzygy.MissingTableError:
                v = None
            b.pop()
            reply.append((flags, v))
        native = int(np.sign(tb.probe_wdl(b)))
        rest = []
        for t in range(3):
            permitted = [v for f, v in reply if not f[t]]
            known = [v for v in permitted if v is not None]
            if not permitted or (None in permitted and native not in known):
                rest.append(None)
            else:
                rest.append(-max(known))
        vals.append(rest)
        tags.append([int(any(f[t] for f, v in reply)) for t in range(3)])
    return vals, tags


def main(root, dest):
    with gzip.open(root / "pawn-geometry/played.gz") as f:
        rs = [
            r
            for r in map(json.loads, f)
            if r["labels"][0] != r["labels"][1] and r["tags"][0] == r["tags"][1]
        ]
    assert len(rs) == 333
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        tb.add_directory(str(root / "tb5"))
        for r in rs:
            r["restricted"], r["geometry"] = classify(root, r, tb)
    with gzip.open(root / "relational-tests/residual.gz", "wt") as f:
        for r in rs:
            print(json.dumps(r), file=f)
    out = []
    for j, t in enumerate(TYPES):
        groups = {
            k: []
            for k in [
                "both",
                "either",
                "neither",
                "any",
                "necessary",
                "unresolved",
                "other_resolved",
            ]
        }
        for r in rs:
            v = [x[j] for x in r["restricted"]]
            g = [x[j] for x in r["geometry"]]
            side = int(np.sign(r["labels"][0] - r["labels"][1]))
            groups["any"].append(r)
            groups[["neither", "either", "both"][sum(g)]].append(r)
            key = (
                "unresolved"
                if None in v
                else "necessary"
                if side * (v[0] - v[1]) <= 0
                else "other_resolved"
            )
            groups[key].append(r)
        for name, data in groups.items():
            if not data:
                continue
            gain = np.array(
                [
                    [
                        np.sign(r["labels"][0] - r["labels"][1]) * s
                        for s in r["margins"][:2]
                    ]
                    for r in data
                ]
            )
            rates = ((gain > 0) + 0.5 * (gain == 0)).mean(0)
            out.append(
                {
                    "restriction": t,
                    "subset": name,
                    "n": len(data),
                    "hce": rates[0],
                    "nnue": rates[1],
                }
            )
    table(dest / "residual.csv", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    main(a.root, a.dest)
