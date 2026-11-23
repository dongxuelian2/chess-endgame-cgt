"""Independently enumerate the actual sibling choices in deterministic king/pawn blocks."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import validate
from summarize import write


def main(root, out):
    validate.init(root)
    home = root / "atlas/horizon"
    rows = []
    for a in (2, 3):
        tables = np.fromfile(home / f"race-{a}.bin", dtype="<u8").reshape(4097, 64)
        count = pairs = 0
        for f in sorted(home.glob(f"race-{a}-*.txt.audit")):
            for row in np.loadtxt(f, dtype="i8", ndmin=2):
                idx = int(row[0])
                w, k, y = idx >> 12, idx >> 6 & 63, idx & 63
                children = {}
                siblings = set()
                for x in range(64):
                    if x in (w, k, y):
                        continue
                    q = validate.board(a, 1, w, k, y, x)
                    if not q.is_valid() or q.is_check() or not any(q.legal_moves):
                        continue
                    moves = list(q.legal_moves)
                    if not any(m.promotion or q.is_capture(m) for m in moves):
                        st = validate.state(a, 1, w, k, y, x)
                        v = validate.sign(validate.tb.probe_wdl(q))
                        assert v == validate.sign(st["w"])
                        children[x] = (int(st["f"]), v)
                    q.turn = True
                    to = [
                        m.to_square
                        for m in q.legal_moves
                        if m.from_square == x
                        and not q.is_capture(m)
                        and not q.gives_check(m)
                    ]
                    siblings.update(combinations(sorted(to), 2))
                hist = np.zeros((2, 3, 3), dtype="i8")
                for j, i in siblings:
                    if i not in children or j not in children:
                        continue
                    hi, vi = children[i]
                    hj, vj = children[j]
                    if hi != hj:
                        continue
                    pairs += 1
                    for s, block in enumerate((0, w * 64 + k + 1)):
                        m = int(tables[block, y])
                        r = ((m >> i) & 1) - ((m >> j) & 1)
                        hist[s, 1 + validate.sign(vi - vj), 1 + r] += 1
                assert np.array_equal(hist.ravel(), row[1:]), (a, w, k, y)
                count += 1
        rows.append({"white": a, "blocks": count, "sibling_pairs": pairs})
    write(out / "race-pair-audit.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
