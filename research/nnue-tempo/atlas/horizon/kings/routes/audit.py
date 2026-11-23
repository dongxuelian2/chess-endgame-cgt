"""Compare serialized weights, native layer traces, legal moves and Syzygy restrictions."""

import sys
from pathlib import Path

import chess
import numpy as np
import validate as v
from bottleneck import read_batch
from feature_routes import get_features
from network import Network
from promotion_geometry import relation
from summarize import write
from third import GUARD, restrict, tree


def main(root, out):
    root = root.resolve()
    work = root / "atlas/horizon/kings/routes"
    net = Network(root)
    v.init(root)
    rng = np.random.default_rng(831016)
    result = []
    for f in sorted(work.glob("*.npz")):
        d = np.load(f)
        rows = d["rows"]
        key = f.stem
        b = 1 if key.startswith(("solo", "relaxed")) else int(key[-1])
        pieces = [int(k) for k in d.files[1:]]
        ix = rng.choice(len(rows), min(32, len(rows)), replace=False)
        r = rows[ix]
        for a in pieces:
            children = []
            states = []
            for w, k, y, h, l, s in r.tolist():
                q = v.board(a, b, w, k, y, s)
                q.turn = True
                assert q.is_valid() and not q.is_check()
                kids = []
                for t in (h, l):
                    board = q.copy()
                    move = chess.Move(s, t)
                    assert move in board.legal_moves
                    board.push(move)
                    assert (
                        board.is_valid()
                        and not board.is_check()
                        and not any(
                            board.is_capture(m) or m.promotion
                            for m in board.legal_moves
                        )
                    )
                    kids.append(board.fen())
                    states.append((a, b, w, k, y, t))
                children.append({"children": kids})
            raw, ps, _ = net.encode(r, a, b)
            tr = net.forward(raw, ps)
            ft, logits, acc = get_features(children, root, raw=True)
            units = read_batch(root, states)
            native = np.array([units[s] for s in states]).reshape(len(r), 2, -1)
            assert np.array_equal(raw, acc) and np.array_equal(raw.clip(0, 127), ft)
            assert np.array_equal(tr["z"], logits)
            assert np.array_equal(tr["h1"], native[:, :, 2:10]) and np.array_equal(
                tr["h2"], native[:, :, 10:]
            )
            assert np.array_equal(ps, native[:, :, 0]) and np.array_equal(
                tr["pos"], native[:, :, 1]
            )
            guards = 0
            if b == 1:
                for j, row in zip(ix[:12], r[:12]):
                    w, k, y, h, l, s = map(int, row)
                    trees = [tree(v.board(a, b, w, k, y, x)) for x in (h, l)]
                    decisive = (
                        restrict(trees[0], GUARD)[0] <= restrict(trees[1], GUARD)[0]
                    )
                    assert decisive == bool(int(d[str(a)][j, 0]) >> 22 & 1)
                    guards += 1
            result.append(
                {
                    "panel": key,
                    "white": a,
                    "boards": len(states),
                    "layer_scalars_per_board": 2048 * 2 + 8 * 2 + 32 + 2,
                    "guard_comparisons": guards,
                }
            )
    strict = np.load(work / "4-1.npz")
    relaxed = np.load(work / "relaxed-4.npz")
    for field in strict.files:
        assert np.array_equal(strict[field], relaxed[field])
    # The previous rank-three set is exactly reproduced, coordinate by coordinate.
    d = np.load(work / "4-1.npz")
    mask = ((d["4"][:, 0] & d["5"][:, 0]) >> 22 & 1).astype(bool)
    before = [
        np.loadtxt(work.parent / f"substitutions-{a}.third", dtype="i8") for a in (4, 5)
    ]
    old = ((before[0][:, -1] & before[1][:, -1]) >> 22 & 1).astype(bool)
    assert set(map(tuple, d["rows"][mask, :5])) == set(map(tuple, before[0][old, :5]))
    # Compare promotion-ray categories to direct attacks with and without the kings.
    for a in (2, 3, 4, 5):
        for _ in range(1500):
            w, k, y, x = map(int, rng.choice(64, 4, replace=False))
            y = 8 + y % 8
            t = y - 8
            if len({w, k, y, x}) < 4:
                continue
            q = v.board(a, 1, w, k, y, x)
            q.remove_piece_at(y)
            observed = t in q.attacks(x)
            q.remove_piece_at(w)
            q.remove_piece_at(k)
            expected = (
                0
                if x == t
                else 1
                if t in (w, k)
                else 2
                if t not in q.attacks(x)
                else 4
                if observed
                else 3
            )
            assert relation(a, x, t, w, k) == expected
    write(out / "native-audit.csv", result)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
