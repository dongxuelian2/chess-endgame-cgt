"""Apply the fixed square-to-channel map to independent played parents and legal alternatives."""

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network, credit
from promotion_trial import CHANNELS, donors, restrict
from summarize import write


def main(root, out):
    net = Network(root)
    v.init(root)
    rows = []
    flow = []
    examples = []
    for period, file in [
        ("Apr-Sep", root / "atlas/played.gz"),
        ("Oct-Dec", root / "atlas/horizon/played.gz"),
    ]:
        counts = Counter()
        groups = defaultdict(list)
        with gzip.open(file, "rt") as games:
            for _, game, fen, m, n, *_ in map(json.loads, games):
                counts["all_parents"] += 1
                q = chess.Board(fen)
                w = q.king(True)
                k = q.king(False)
                bb = q.pieces(chess.PAWN, False)
                if len(bb) != 1:
                    continue
                y = next(iter(bb))
                s = chess.parse_square(m[:2])
                a = q.piece_type_at(s)
                x, z = [chess.parse_square(u[2:4]) for u in (m, n)]
                if a not in (2, 3, 4, 5) or y // 8 != 1:
                    continue
                counts["penultimate_pawn"] += 1
                square = y ^ (7 if w % 8 < 4 else 0)
                if square not in CHANNELS:
                    continue
                counts["five_files"] += 1
                entries = [v.state(a, 1, w, k, y, u) for u in (x, z)]
                assert all(e["v"] for e in entries)
                if entries[0]["f"] != entries[1]["f"]:
                    continue
                counts["classical_tie"] += 1
                order = int(np.sign(entries[0]["w"])) - int(np.sign(entries[1]["w"]))
                if not order:
                    continue
                counts["strict_wdl"] += 1
                h, l = (x, z) if order > 0 else (z, x)
                r = [w, k, y, h, l, s]
                if not restrict((a, *r)):
                    continue
                counts["promotion_necessary"] += 1
                groups[a, square].append((r, game))
        pooled = []
        for (a, square), items in sorted(groups.items()):
            rr, ids = zip(*items)
            r = np.array(rr)
            raw, p, _ = net.encode(r, a, 1)
            t = net.forward(raw, p)
            x = raw.clip(0, 127)
            ft = CHANNELS[square]
            without, effect = donors(ft, net, x, t["z"], p, t["score"])
            native = credit(t["score"][:, 0] - t["score"][:, 1])
            delta = x[:, 0, ft] - x[:, 1, ft]
            record = np.c_[native, without, effect, abs(effect), credit(delta)]
            pooled.extend(record)
            rows.append(
                dict(
                    period=period,
                    white=a,
                    ft=ft,
                    n=len(items),
                    **dict(
                        zip(
                            [
                                "native",
                                "without",
                                "margin_contribution",
                                "abs_contribution",
                                "channel_order",
                            ],
                            record.mean(0),
                        )
                    ),
                )
            )
            for gid, rv, sc in zip(ids, rr, record):
                examples.append(
                    {
                        "period": period,
                        "game": gid,
                        "white": a,
                        "ft": ft,
                        "wk": rv[0],
                        "bk": rv[1],
                        "pawn": rv[2],
                        "high": rv[3],
                        "low": rv[4],
                        "source": rv[5],
                        "native": sc[0],
                        "without": sc[1],
                    }
                )
        if pooled:
            rows.append(
                dict(
                    period=period,
                    white="all",
                    ft="selected",
                    n=len(pooled),
                    **dict(
                        zip(
                            [
                                "native",
                                "without",
                                "margin_contribution",
                                "abs_contribution",
                                "channel_order",
                            ],
                            np.mean(pooled, axis=0),
                        )
                    ),
                )
            )
        flow.append(dict(period=period, **counts))
    write(out / "game-screen.csv", flow)
    write(out / "game-transfer.csv", rows)
    write(out / "game-cases.csv", examples)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
