"""Cross-check finite game values using scalar recursion and python-chess move generation."""

import argparse
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from results import values
from summarize import write
from validate import board, sign


def exact(q, depth, mask, tb):
    choices = list(q.legal_moves)
    if not choices:
        v = -int(q.is_check())
        return (v, v, v) if mask & 8 else (-1, 0, 1)
    if not depth:
        return (-1, 0, 1)
    best = [-1, -1, -1]
    for m in choices:
        capture = q.is_capture(m)
        q.push(m)
        if capture or m.promotion:
            active = 4 if m.promotion else 2 if q.pawns else 1
            if active & mask:
                v = sign(tb.probe_wdl(q))
                other = (v, v, v)
            else:
                other = (-1, 0, 1)
        else:
            other = exact(q, depth - 1, mask, tb)
        q.pop()
        best = [max(x, -y) for x, y in zip(best, reversed(other))]
    return tuple(best)


def main(root, out):
    tb = chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular")
    rows = []
    rng = np.random.default_rng(1408)
    for f in sorted((root / "atlas/horizon/all5").glob("*.out")):
        a, b = map(int, f.stem.split("-"))
        arr = np.loadtxt(f, dtype=int, ndmin=2)
        n = min(24, len(arr))
        tested = 0
        for ix, row in enumerate(arr[rng.choice(len(arr), n, replace=False)]):
            q = board(a, b, *map(int, row[:4]))
            d = 3 if ix < 4 else 2
            for mask in {15, int(rng.integers(1, 15))}:
                ref = exact(q, d, mask, tb)
                v = values(row[4:].reshape(-1, 3, 2)[d], mask)
                assert tuple(v) == ref, (q.fen(), d, mask, ref, v)
                tested += 1
        rows.append({"white": a, "black": b, "boards": n, "scalar_triples": tested})
    write(out / "independent-audit.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
