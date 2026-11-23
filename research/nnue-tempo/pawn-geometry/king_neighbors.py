"""Hold exact four-board WDL fixed while displacing a king by one square."""

import argparse
import gzip
import json
from pathlib import Path

import chess
from measure import table


def summarize(path):
    positions = {}
    with gzip.open(path) as f:
        for wk, bk, wdl, n, h in map(json.loads, f):
            if wdl[0] <= wdl[1] or wdl[2] >= wdl[3]:
                continue
            a, b = n[1] - n[0], n[3] - n[2]
            positions[wk, bk] = tuple(wdl), a > 0 > b, a > b
    out = {
        "template": path.stem,
        "n": len(positions),
        "correct_both": 0,
        "right_direction_only": 0,
        "wrong_or_flat": 0,
        "white_neighbors": 0,
        "white_changes": 0,
        "black_neighbors": 0,
        "black_changes": 0,
    }
    for (wk, bk), (target, correct, direction) in positions.items():
        out[
            "correct_both"
            if correct
            else ("right_direction_only" if direction else "wrong_or_flat")
        ] += 1
        for side, loc in [("white", wk), ("black", bk)]:
            for dst in chess.scan_forward(chess.BB_KING_ATTACKS[loc]):
                if dst <= loc:
                    continue
                key = (dst, bk) if side == "white" else (wk, dst)
                other = positions.get(key)
                if other and other[0] == target:
                    out[side + "_neighbors"] += 1
                    out[side + "_changes"] += correct != other[1]
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    table(
        a.dest / "neighbor-kings.csv",
        [summarize(f) for f in sorted((a.root / "pawn-geometry").glob("king-*.gz"))],
    )
