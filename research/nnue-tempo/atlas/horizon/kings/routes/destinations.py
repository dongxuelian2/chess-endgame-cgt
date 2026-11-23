"""Relate the matched unit signals to actual rook and queen destinations."""

import sys
from functools import lru_cache
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network, credit, retain
from study import panels
from summarize import write


@lru_cache(250000)
def geometry(a, w, k, y, x):
    q = v.board(a, 1, w, k, y, x)
    target = y % 8
    att = q.attacks(x)
    flags = (
        x % 8 == y % 8,
        x // 8 == 0,
        x % 8 == y % 8 or x // 8 == 0,
        y in att,
        target in att,
        q.is_pinned(False, y),
    )
    return np.array(flags)


def main(root, out):
    net = Network(root)
    result = []
    examples = []
    for panel, pieces, b, rank, rule, r in panels(root / "atlas/horizon/kings/routes"):
        if rule != "escort" or len(r) == 0 or panel.startswith("relaxed"):
            continue
        for a in pieces:
            flags = np.array(
                [
                    [geometry(a, *map(int, row[:3]), int(row[i])) for i in (3, 4)]
                    for row in r
                ]
            )
            stats = []
            for start in range(0, len(r), 256):
                rows = r[start : start + 256]
                raw, p, _ = net.encode(rows, a, 1)
                t = net.forward(raw, p)
                score = t["score"][:, 0] - t["score"][:, 1]
                H = t["h1"]
                gaps = abs(H[:, 0] - H[:, 1])
                stats.extend(
                    np.c_[
                        credit(score),
                        retain(net, t, [2]),
                        retain(net, t, [6, 7]),
                        gaps[:, 2],
                        gaps[:, 6],
                        gaps[:, 7],
                    ]
                )
            stats = np.asarray(stats)
            for i, tag in enumerate(
                (
                    "pawn_file",
                    "last_rank",
                    "file_or_last_rank",
                    "attacks_pawn",
                    "attacks_promotion",
                    "pins_pawn",
                )
            ):
                for name, mask in [
                    ("high_only", flags[:, 0, i] & ~flags[:, 1, i]),
                    ("low_only", ~flags[:, 0, i] & flags[:, 1, i]),
                    ("both", flags[:, 0, i] & flags[:, 1, i]),
                    ("neither", ~flags[:, 0, i] & ~flags[:, 1, i]),
                ]:
                    n = mask.sum()
                    if n:
                        result.append(
                            dict(
                                panel=panel,
                                piece=a,
                                rank=rank,
                                tag=tag,
                                condition=name,
                                n=int(n),
                                **dict(
                                    zip(
                                        (
                                            "native",
                                            "keep2",
                                            "keep67",
                                            "gap2",
                                            "gap6",
                                            "gap7",
                                        ),
                                        stats[mask].mean(0),
                                    )
                                ),
                            )
                        )
        if panel == "4-1":
            measures = []
            right = np.ones(len(r), bool)
            for a in pieces:
                raw, p, _ = net.encode(r, a, 1)
                t = net.forward(raw, p)
                measures.append(abs(t["h1"][:, 0] - t["h1"][:, 1])[:, [2, 6, 7]])
                right &= t["score"][:, 0] > t["score"][:, 1]
            joined = np.concatenate(measures, axis=1)
            loss = (
                ((joined - joined.mean(0)) / np.maximum(joined.std(0), 1)) ** 2
            ).sum(1)
            chosen = np.flatnonzero(right)[np.argmin(loss[right])]
            for label, idx in [("first", 0), ("near_mean", chosen)]:
                w, k, y, h, l, s = map(int, r[idx])
                for a in pieces:
                    q = v.board(a, 1, w, k, y, s)
                    q.turn = True
                    raw, p, _ = net.encode(r[idx : idx + 1], a, 1)
                    t = net.forward(raw, p)
                    examples.append(
                        {
                            "selection": label,
                            "piece": a,
                            "fen": q.fen(),
                            "high": q.san(chess.Move(s, h)),
                            "low": q.san(chess.Move(s, l)),
                            "score_high": int(t["score"][0, 0]),
                            "score_low": int(t["score"][0, 1]),
                            "h1_high": " ".join(map(str, t["h1"][0, 0])),
                            "h1_low": " ".join(map(str, t["h1"][0, 1])),
                        }
                    )
    write(out / "destinations.csv", result)
    write(out / "escort-example.csv", examples)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
