"""Check the new motif cases against independent Stockfish traces."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
import validate as v
from bottleneck import read_batch
from feature_routes import get_features
from network import Network
from summarize import write

from motif_trace import rows_for


def audit(root, net, rows, a, b):
    raw, ps, _ = net.encode(rows, a, b)
    t = net.forward(raw, ps)
    positions = []
    keys = []
    for w, k, y, h, l, s in rows.tolist():
        p = v.board(a, b, w, k, y, s)
        p.turn = chess.WHITE
        children = []
        for destination in (h, l):
            q = p.copy()
            move = chess.Move(s, destination)
            assert move in q.legal_moves
            q.push(move)
            assert q.is_valid() and not q.is_check()
            children.append(q.fen())
            k0 = (a, b, w, k, y, destination)
            assert v.tb.probe_wdl(q) == v.state(*k0)["w"]
            keys.append(k0)
        positions.append({"children": children})
    ft, h1logit, acc = get_features(positions, root, raw=True)
    native = read_batch(root, keys)
    upper = np.array([native[key] for key in keys]).reshape(len(rows), 2, -1)
    for actual, expected in [
        (raw, acc),
        (raw.clip(0, 127), ft),
        (t["z"], h1logit),
        (t["ps"], upper[:, :, 0]),
        (t["pos"], upper[:, :, 1]),
        (t["h1"], upper[:, :, 2:10]),
        (t["h2"], upper[:, :, 10:]),
    ]:
        assert np.array_equal(actual, expected)
    return len(keys)


def main(root, out):
    v.init(root)
    net = Network(root)
    rng = np.random.default_rng(72610)
    report = []
    # The witness order fixes plus and minus and preserves both original parent moves.
    groups = defaultdict(list)
    with (out / "motif-witness-positions.csv").open() as f:
        for r in csv.DictReader(f):
            groups[r["case"], int(r["white"]), int(r["arc"])].append(r)
    witness = defaultdict(list)
    for (key, a, _), pairs in groups.items():
        p = chess.Board(pairs[0]["parent"])
        b = 4 if key == "skewer_vs_neutral" else 1
        y = next(iter(p.pieces(b, chess.BLACK)))
        moves = [p.parse_san(r["move"]) for r in pairs]
        witness[a, b].append(
            [
                p.king(True),
                p.king(False),
                y,
                moves[0].to_square,
                moves[1].to_square,
                moves[0].from_square,
            ]
        )
    for (a, b), rows in sorted(witness.items()):
        report.append(
            {
                "sample": "witness",
                "white": a,
                "black": b,
                "boards": audit(root, net, np.array(rows), a, b),
            }
        )
    work = root / "manuscript/motifs"
    for a, b in [(i, 1) for i in range(2, 6)] + [(3, 4), (5, 4)]:
        d = np.load(work / f"panel-{3 if b == 4 else a}-{b}.npz")
        available = d["rows"][d["validation"]]
        rows = available[rng.choice(len(available), 16, replace=False)]
        report.append(
            {
                "sample": "validation",
                "white": a,
                "black": b,
                "boards": audit(root, net, rows_for(rows).reshape(-1, 6), a, b),
            }
        )
    write(out / "motif-native-audit.csv", report)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
