"""Validate stratified census matches and compare three networks on identical boards."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
import validate as v
from panels import readouts
from replay import branches
from summarize import write


def run(root, out):
    v.init(root)
    home = root / "atlas/horizon/kings"
    rows = []
    seeds = set()
    for r in csv.DictReader((out / "bishop-295.csv").open()):
        q = chess.Board(r["parent_fen"])
        key = (
            3,
            q.king(True),
            q.king(False),
            next(iter(q.pieces(1, False))),
            chess.parse_square(r["high"]),
            chess.parse_square(r["low"]),
        )
        seeds.add(key)
        rows.append(("reversal", r["family"], key))
    for a in (2, 3, 4, 5):
        for row in (home / f"stratified-{a}.txt").read_text().splitlines():
            family, *rest = row.split()
            key = (a, *map(int, rest))
            if key in seeds:
                continue
            rows.append(("outside_reversal", family, key))
    fens = []
    native = []
    stats = defaultdict(list)
    for panel, fam, (a, w, k, y, h, l) in rows:
        qs = [v.board(a, 1, w, k, y, t) for t in (h, l)]
        fens.extend(q.fen() for q in qs)
        if panel == "outside_reversal":
            trees = [branches(q) for q in qs]
            assert [max(z.value for z in b) for b in trees] == [1, 0]
            if fam == "royal_fork":
                hi = trees[0]
                assert all(
                    "king_attack_with_promotion_guard" in m.tags
                    for m in hi
                    if m.value == 1
                )
                assert any("king_attack_with_promotion_guard" not in m.tags for m in hi)
                assert all(
                    next(m.value for m in b if "push_pawn" in m.tags) == 0
                    for b in trees
                )
            else:
                assert [m.tags for m in trees[0] if m.value == 1] and all(
                    "push_pawn" in m.tags for m in trees[0] if m.value == 1
                )
                push = next(m for m in trees[1] if "push_pawn" in m.tags)
                assert push.value == 0

                def category(u):
                    if "uncover_check" in u.tags:
                        return "discovered"
                    if "check" in u.tags:
                        return "direct_check"
                    if "pin_man" in u.tags:
                        return "quiet_pin"
                    return "other"

                types = {category(u) for u in push.replies if u.value == 0}
                assert types == (
                    {"direct_check", "quiet_pin"} if fam == "pin_or_check" else {fam}
                )
                assert any(category(u) not in types for u in push.replies)
        sh, sl = [v.state(a, 1, w, k, y, t) for t in (h, l)]
        assert sh["f"] == sl["f"]
        native.append(int(sh["n"]) - int(sl["n"]))
    t, old = readouts(root, fens)
    modern = t[:, 0, :] - t[:, 1, :]
    old = old[:, 0] - old[:, 1]
    for (p, fam, (a, *_)), pred in zip(
        rows, np.c_[old, native, modern[:, 15], modern[:, 0]]
    ):
        stats[p, a, fam].append(pred)
    report = []
    for (p, a, f), arr in sorted(stats.items()):
        points = (np.sign(arr) + 1).sum(0) / 2
        report.append(
            dict(
                panel=p,
                white_man=a,
                family=f,
                n=len(arr),
                **dict(
                    zip(
                        ("sf12", "sf141", "sf18_large", "sf18_large_no_direct_edges"),
                        points,
                    )
                ),
            )
        )
    write(out / "rank-network-controls.csv", report)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
