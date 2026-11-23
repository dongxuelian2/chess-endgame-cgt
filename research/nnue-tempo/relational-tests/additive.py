"""Four-corner lower bounds and independent legal-move enumeration for maximal examples."""

import argparse
import gzip
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from measure import table
from siblings import trace_positions

from model import Model, board_for


def scores_for_sources(r, model):
    wk, bk, a, p = (r[k] for k in ("wk", "bk", "actor", "pawn"))
    men = (6, -6, a, -1)
    all_pairs = []
    for source in range(64):
        both = set(range(64))
        for pawn in (p, p - 8):
            if len({wk, bk, source, pawn}) < 4:
                both.clear()
                break
            b = board_for([wk, bk, source, pawn], men, True)
            if not b.is_valid() or b.is_check():
                both.clear()
                break
            ok = set()
            for m in list(b.legal_moves):
                if m.from_square != source or b.is_capture(m) or b.gives_check(m):
                    continue
                b.push(m)
                if any(b.legal_moves):
                    ok.add(m.to_square)
                b.pop()
            both &= ok
        all_pairs.extend(combinations(sorted(both), 2))
    pairs = sorted(set(all_pairs))
    assert len(pairs) == r["pairs"]
    values = {}
    for pawn in (p, p - 8):
        s = sorted({x for pair in pairs for x in pair})
        x, ps = model.read([[wk, bk, actor, pawn] for actor in s], men)
        scores, *_ = model.score(x, ps, 0)
        values.update({(pawn, actor): int(v) for actor, v in zip(s, scores)})
    diff = lambda actor: values[p - 8, actor] - values[p, actor]
    assert max(abs(diff(i) - diff(j)) for i, j in pairs) == r["maximum"]
    return [values[q, s] for q in (p, p - 8) for s in (r["A"], r["B"])]


def main(root, dest):
    groups = defaultdict(list)
    with gzip.open(root / "relational-tests/census.gz") as f:
        for line in f:
            r = json.loads(line)
            if r["target"] == "engine":
                groups[r["actor"]].append(r)
    model = Model(root)
    tb = chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular")
    result = []
    for a, rows in groups.items():
        r = max(rows, key=lambda x: x["maximum"])
        corners = scores_for_sources(r, model)
        margin = corners[2] - corners[3] - corners[0] + corners[1]
        assert margin == r["maximum"]
        children = [
            board_for([r["wk"], r["bk"], r[dest], q], [6, -6, a, -1])
            for q in (r["pawn"], r["pawn"] - 8)
            for dest in ("A", "B")
        ]
        native = [v for _, v in trace_positions([b.fen() for b in children], root)]
        assert [v[1] for v in native] == corners
        classical = " ".join(str(v[0]) for v in native)
        wdl = " ".join(str(int(np.sign(tb.probe_wdl(b)))) for b in children)
        result.append(
            {
                "actor": chess.piece_symbol(a),
                "blocks": len(rows),
                "median_bound": float(np.median([x["maximum"] / 4 for x in rows])),
                "largest_bound": margin / 4,
                "parent": board_for(
                    [r["wk"], r["bk"], r["source"], r["pawn"]], [6, -6, a, -1], True
                ).fen(),
                "pawn_before": chess.square_name(r["pawn"]),
                "pawn_after": chess.square_name(r["pawn"] - 8),
                "A": chess.square_name(r["A"]),
                "B": chess.square_name(r["B"]),
                "before_A": corners[0],
                "before_B": corners[1],
                "after_A": corners[2],
                "after_B": corners[3],
                "classical_black": classical,
                "wdl_black": wdl,
            }
        )
    table(dest / "additive.csv", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    main(a.root, a.dest)
