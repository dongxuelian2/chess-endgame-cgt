"""Separate the geometry of checks from forced captures of the white piece."""

import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network, credit, retain
from summarize import write

LABELS = (
    "distant_check",
    "knight_fork",
    "rook_skewer",
    "forced_capture",
    "forced_capture_distant",
    "safe_knight_fork",
)


@lru_cache(16384)
def tree(a, b, w, k, y, x):
    q = v.board(a, b, w, k, y, x)
    base = v.sign(v.tb.probe_wdl(q))
    moves = []
    for m in list(q.legal_moves):
        src, dst = m.from_square, m.to_square
        q.push(m)
        val = -v.sign(v.tb.probe_wdl(q))
        tags = set()
        if src == y and q.is_check():
            if chess.square_distance(dst, w) >= 3:
                tags.add("distant_check")
            if b == 2 and x in q.attacks(dst):
                tags.add("knight_fork")
            if (
                b == 4
                and int(chess.between(dst, x)) & chess.BB_SQUARES[w]
                and (dst % 8 == x % 8 or dst // 8 == x // 8)
            ):
                tags.add("rook_skewer")
            replies = list(q.legal_moves)
            if "knight_fork" in tags and all(z.to_square != dst for z in replies):
                tags.add("safe_knight_fork")
            forced = bool(replies)
            for z in replies:
                if not forced:
                    break
                q.push(z)
                target = z.to_square if z.from_square == x else x
                forced = (
                    q.piece_at(dst) == chess.Piece(b, False)
                    and chess.Move(dst, target) in q.legal_moves
                )
                q.pop()
            if forced:
                tags.add("forced_capture")
                if "distant_check" in tags:
                    tags.add("forced_capture_distant")
        q.pop()
        moves.append((val, tags, m))
    assert max(z[0] for z in moves) == base
    values = [
        max((val for val, tags, _ in moves if t not in tags), default=base)
        for t in LABELS
    ]
    avail = [any(t in tags for _, tags, _ in moves) for t in LABELS]
    return base, values, avail


def classify(job):
    a, b, w, k, y, h, l = job
    high = tree(a, b, w, k, y, h)
    low = tree(a, b, w, k, y, l)
    assert high[0] > low[0]
    return [x <= z for x, z in zip(high[1], low[1])] + [
        x and not z for x, z in zip(high[2], low[2])
    ]


def main(root, out):
    root = root.resolve()
    net = Network(root)
    work = root / "atlas/horizon/kings/routes"
    report = []
    witness = []
    with ProcessPoolExecutor(
        8, mp_context=get_context("fork"), initializer=v.init, initargs=(root,)
    ) as pool:
        for a, b in ((4, 2), (3, 4), (4, 4)):
            d = np.load(work / f"{a}-{b}.npz")
            r = d["rows"]
            masks = []
            for piece in (a, 5):
                m = np.array(
                    list(
                        pool.map(
                            classify,
                            [(piece, b, *map(int, row[:5])) for row in r],
                            chunksize=180,
                        )
                    )
                )
                reference = d[str(piece)][:, 2]
                idx = 9 if b == 2 else 4
                col = 1 if b == 2 else 0
                assert np.array_equal(m[:, col], (reference >> idx & 1).astype(bool))
                masks.append(m)
            both = masks[0] & masks[1]
            for i, t in enumerate(LABELS):
                for condition, mask in [
                    ("necessary", both[:, i]),
                    ("necessary_and_exclusive", both[:, i] & both[:, i + len(LABELS)]),
                ]:
                    group = r[mask]
                    n = len(group)
                    if n and condition == "necessary":
                        w, k, y, h, l, s = map(int, group[0])
                        q = v.board(a, b, w, k, y, s)
                        q.turn = True
                        witness.append(
                            {
                                "base": a,
                                "black": b,
                                "rule": t,
                                "parent": q.fen(),
                                "high": q.san(chess.Move(s, h)),
                                "low": q.san(chess.Move(s, l)),
                            }
                        )
                    for piece in (a, 5):
                        stats = np.zeros(8)
                        for j in range(0, n, 192):
                            raw, p, _ = net.encode(group[j : j + 192], piece, b)
                            trace = net.forward(raw, p)
                            orig = credit(trace["score"][:, 0] - trace["score"][:, 1])
                            second = trace["h2"]
                            # Test first-layer complement and a common pair of second-layer output units.
                            vals = [
                                orig,
                                retain(net, trace, [0, 1, 4]),
                                retain(net, trace, [1, 3, 7]),
                                retain(net, trace, [0, 2, 3, 5, 6, 7]),
                                retain(net, trace, [6, 7]),
                                retain(net, trace, [0, 1, 2, 4, 5, 6, 7]),
                            ]
                            for u in (2, 13):
                                dpos = (second[:, 0, u] - second[:, 1, u]) * net.out[u]
                                left = (
                                    np.trunc((p[:, 0] + trace["pos"][:, 0] - dpos) / 16)
                                    - trace["score"][:, 1]
                                )
                                right = trace["score"][:, 0] - np.trunc(
                                    (p[:, 1] + trace["pos"][:, 1] + dpos) / 16
                                )
                                vals.append((credit(left) + credit(right)) / 2)
                            stats += np.array(vals).sum(axis=1)
                        report.append(
                            dict(
                                base=a,
                                white=piece,
                                black=b,
                                rule=t,
                                selection=condition,
                                n=n,
                                **dict(
                                    zip(
                                        [
                                            "native",
                                            "keep014",
                                            "keep137",
                                            "drop14",
                                            "keep67",
                                            "drop3",
                                            "without_h2_2",
                                            "without_h2_13",
                                        ],
                                        stats / n if n else [None] * 8,
                                    )
                                ),
                            )
                        )
    write(out / "tactical-controls.csv", report)
    write(out / "tactical-examples.csv", witness)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
