"""Inspect Black's second turn after a pawn advance in the rook and queen panels."""

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import validate as v
from panel import TAGS, samples
from replay import action, remaining
from summarize import write

GUARD = "king_guard_own_pawn_next"


def tree(q):
    result = []
    for m in list(q.legal_moves):
        b = action(q, m)
        if "push_pawn" in b.tags:
            for u in list(q.legal_moves):
                w = action(q, u)
                for t in list(q.legal_moves):
                    w.replies.append(action(q, t))
                    q.pop()
                assert max((z.value for z in w.replies), default=w.value) == w.value
                b.replies.append(w)
                q.pop()
            assert min((w.value for w in b.replies), default=b.value) == b.value
        result.append(b)
        q.pop()
    return result


def restrict(rows, tag):
    lines = []
    branches = []
    for b in rows:
        values = [
            max((t.value for t in remaining(w.replies, tag)), default=w.value)
            for w in b.replies
        ]
        new = min(values, default=b.value)
        branches.append(new)
        if new < b.value:
            for w, val in zip(b.replies, values):
                if val == new:
                    opts = sorted(t.san for t in w.replies if t.value > val)
                    assert opts
                    lines.append(f"{b.san} {w.san} / {' '.join(opts)}")
    return max(branches), " | ".join(sorted(lines))


def main(root, out):
    v.init(root)
    work = root / "atlas/horizon/kings"
    needed = samples(root)
    census = Counter()
    cases = []
    rng = np.random.default_rng(91701)
    residual = {r["game"] for r in csv.DictReader((out / "residual-28.csv").open())}
    found = set()
    for a in (4, 5):
        for f in sorted(work.glob(f"pairs-{a}-1-*.in")):
            data = np.loadtxt(f.with_suffix(".third"), dtype="i8", ndmin=2)
            assert np.array_equal(data[:, :5], np.loadtxt(f, dtype="i8", ndmin=2))
            audit = set(rng.choice(len(data), min(5, len(data)), replace=False))
            for j, row in enumerate(data):
                w, k, y, h, l, mask = map(int, row)
                key = (a, 1, w, k, y, h, l)
                assert key not in found
                found.add(key)
                tags = {t for i, t in enumerate(TAGS) if mask >> i & 1}
                groups = needed[key]
                ids = [
                    gid
                    for name, gid, *_ in groups
                    if name == "fresh" and gid in residual
                ]
                if ids or j in audit:
                    trees = [tree(v.board(a, 1, w, k, y, x)) for x in (h, l)]
                    assert tags == {
                        t
                        for t in TAGS
                        if restrict(trees[0], t)[0] <= restrict(trees[1], t)[0]
                    }
                    for gid in ids:
                        hv, line = restrict(trees[0], GUARD)
                        lv, _ = restrict(trees[1], GUARD)
                        cases.append(
                            {
                                "game": gid,
                                "action_classes": " ".join(sorted(tags)),
                                "without_guard_high": hv,
                                "without_guard_low": lv,
                                "high_continuations": line,
                            }
                        )
                for panel, _, opaque, nn, *_ in groups:
                    for t in tags | {"all_pairs"}:
                        census[a, panel, t, "n"] += 1
                        census[a, panel, t, "neural"] += (v.sign(nn) + 1) / 2
    assert found == {key for key in needed if key[:2] in ((4, 1), (5, 1))}
    write(
        out / "post-push-third.csv",
        [
            {
                "white_man": a,
                "panel": p,
                "third_action": t,
                "n": n,
                "neural_points": census[a, p, t, "neural"],
            }
            for (a, p, t, key), n in sorted(census.items())
            if key == "n"
        ],
    )
    write(out / "played-third.csv", sorted(cases, key=lambda r: r["game"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
