"""Verify R/Q substitutions and measure routing contrasts on the common support."""

from collections import defaultdict

import chess
import numpy as np
import validate as v
from panel import TAGS
from summarize import write
from third import GUARD, restrict, tree


def pairs(root):
    v.init(root)
    work = root / "atlas/horizon/kings"
    rows = np.loadtxt(work / "substitutions.txt", dtype="i8", ndmin=2)
    selected = np.ones(len(rows), bool)
    for a in (4, 5):
        data = np.loadtxt(work / f"substitutions-{a}.third", dtype="i8", ndmin=2)
        assert np.array_equal(data[:, :5], rows[:, :5])
        selected &= ((data[:, 5] >> TAGS.index(GUARD)) & 1).astype(bool)
    result = {tuple(map(int, r[:5])): int(r[5]) for r in rows[selected]}
    rng = np.random.default_rng(715323)
    for j in rng.choice(len(rows), min(96, len(rows)), replace=False):
        w, k, y, h, l, s = map(int, rows[j])
        guards = []
        labels = []
        for a in (4, 5):
            parent = v.board(a, 1, w, k, y, s)
            parent.turn = True
            assert parent.is_valid() and not parent.is_check()
            roots = []
            for x in (h, l):
                assert chess.Move(s, x) in parent.legal_moves
                b = parent.copy()
                b.push(chess.Move(s, x))
                assert not b.is_check() and not any(
                    b.is_capture(m) or m.promotion for m in b.legal_moves
                )
                roots.append(tree(b))
            guards.append(restrict(roots[0], GUARD)[0] <= restrict(roots[1], GUARD)[0])
            labels.append(tuple(max(b.value for b in r) for r in roots))
        assert labels[0] == labels[1] and labels[0][0] > labels[0][1]
        assert all(guards) == selected[j]
    return result


def summarize(rows, out):
    metrics = [
        "native",
        "keep67",
        "drop67",
        "without_2",
        "without_7",
        "phi2",
        "phi6",
        "phi7",
    ]
    cells = defaultdict(list)
    for r in rows:
        cells[r["white_man"], r["wk"], r["bk"], r["pawn"]].append(r)
    contexts = []
    for (a, w, k, y), rs in sorted(cells.items()):
        contexts.append(
            dict(
                white_man=a,
                wk=w,
                bk=k,
                pawn=y,
                n=len(rs),
                **{m: np.mean([r[m] for r in rs]) for m in metrics},
            )
        )
    write(out / "matched-escort.csv", contexts)
    summary = []
    for a in (4, 5):
        rc = [r for r in contexts if r["white_man"] == a]
        for name, weights in [
            ("pairs", [r["n"] for r in rc]),
            ("contexts", [1] * len(rc)),
        ]:
            summary.append(
                dict(
                    white_man=a,
                    weighting=name,
                    n_pairs=sum(r["n"] for r in rc),
                    contexts=len(rc),
                    **{
                        m: np.average([r[m] for r in rc], weights=weights)
                        for m in metrics
                    },
                )
            )
    write(out / "material-substitution.csv", summary)
