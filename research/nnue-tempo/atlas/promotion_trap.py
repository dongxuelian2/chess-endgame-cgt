"""Count rook-pawn reversals where promotion either wins or allows mate in one."""

import argparse
from collections import Counter
from functools import cache
from itertools import combinations
from pathlib import Path

import chess
import numpy as np

import validate
from summarize import write


@cache
def classify(w, k, y, x):
    q = validate.board(chess.ROOK, chess.PAWN, w, k, y, x)
    legal = list(q.legal_moves)
    best = []
    goal = validate.sign(validate.state(4, 1, w, k, y, x)["w"])
    queen_mate = False
    for m in legal:
        q.push(m)
        value = -validate.sign(validate.tb.probe_wdl(q))
        if value == goal:
            best.append(m)
        if m.promotion == chess.QUEEN and value == -1:
            for response in list(q.legal_moves):
                q.push(response)
                queen_mate = q.is_checkmate()
                q.pop()
                if queen_mate:
                    break
        q.pop()
    if len(best) != 1:
        return 0
    if goal == 1 and best[0].promotion == chess.QUEEN:
        return 1
    if (
        goal == 0
        and best[0].promotion == chess.KNIGHT
        and q.gives_check(best[0])
        and queen_mate
    ):
        return -1
    return 0


def columns(w, k, y, x, tags):
    candidates = {}
    for src in np.flatnonzero(x["v"]):
        q = validate.board(4, 1, w, k, y, int(src))
        q.turn = True
        dest = []
        for m in q.legal_moves:
            t = m.to_square
            if m.from_square == src and not q.is_capture(m) and x[t]["v"]:
                dest.append(t)
        for i, j in combinations(sorted(dest), 2):
            if x[i]["f"] != x[j]["f"] or {int(x[i]["w"]), int(x[j]["w"])} != {0, 2}:
                continue
            hi, lo = int(tags[i]), int(tags[j])
            if hi != lo:
                continue
            key = (i, j, int(x[i]["f"]), hi)
            candidates[key] = candidates.get(key, 0) | (1 << int(src))
    return candidates


def main(root, out):
    validate.init(root)
    dense = np.memmap(
        root / "atlas/4-1.bin", mode="r", dtype=validate.DT, shape=(64, 64, 64, 64)
    )
    counts = Counter(
        rectangles=0, forced_promotion_traps=0, nnue_both=0, positional_both=0
    )
    ks = set()
    children = set()
    for folder in (root / "atlas", root / "atlas/remaining"):
        rr = np.memmap(folder / "4-1.replies", mode="r", dtype=validate.REPLY)
        ids = {int(v): i for i, v in enumerate(rr["index"]) if (int(v) & 63) // 8 == 1}
        for kings in sorted({i >> 6 for i in ids}):
            w, k = divmod(kings, 64)
            pawn_files = [
                f
                for f in range(8)
                if chess.BB_KNIGHT_ATTACKS[f] & (1 << w) and kings * 64 + 8 + f in ids
            ]
            for c, d in combinations(pawn_files, 2):
                y, z = 8 + c, 8 + d
                left = columns(
                    w, k, y, dense[w, k, y], rr["mask"][ids[kings * 64 + y], :, 0]
                )
                right = columns(
                    w, k, z, dense[w, k, z], rr["mask"][ids[kings * 64 + z], :, 0]
                )
                for key in left.keys() & right.keys():
                    if not left[key] & right[key]:
                        continue
                    i, j, _, _ = key
                    slots = [(pawn, rook) for pawn in (y, z) for rook in (i, j)]
                    st = [validate.state(4, 1, w, k, p, r) for p, r in slots]
                    if st[0]["w"] == st[2]["w"]:
                        continue
                    counts["rectangles"] += 1
                    polar = [classify(w, k, p, r) for p, r in slots]
                    if polar not in ([1, -1, -1, 1], [-1, 1, 1, -1]):
                        continue
                    counts["forced_promotion_traps"] += 1
                    ks.add(kings)
                    children.update((w, k, p, r) for p, r in slots)
                    for col, name in [("n", "nnue_both"), ("d", "positional_both")]:
                        counts[name] += int(
                            all(
                                (int(st[t][col]) - int(st[t + 1][col])) * polar[t] > 0
                                for t in (0, 2)
                            )
                        )
    write(
        out / "rook-promotion-traps.csv",
        [{**counts, "king_placements": len(ks), "distinct_children": len(children)}],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
