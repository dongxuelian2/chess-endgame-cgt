"""Compare frozen tie refinements with the stock network-selection and scaling policy."""

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np

from panels import raw
from summarize import MATS, ci, group, score, write
from validate import board, sign


def generated(job):
    root, a, b = job
    p = root / "atlas/remaining/metrics" / f"{a}-{b}.npz"
    if not p.exists():
        return []
    data = np.load(p)
    rows = data["panel"]
    fs = []
    for r in rows:
        idx = int(r[0])
        wk, bk, y = idx >> 12, idx >> 6 & 63, idx & 63
        for x in r[-2:]:
            fs.append(board(a, b, wk, bk, y, int(x)).fen())
    v = raw(root / "atlas/engine18/src/stockfish", fs, "release18").reshape(-1, 2)
    return [
        (group(int(i) >> 12, int(i) >> 6 & 63), int(t - u), int(z))
        for i, (t, u), z in zip(rows[:, 0], v, data["s18"][:, 15], strict=True)
    ]


def main(root, dest):
    os.environ["ATLAS_RELEASE_ONLY"] = "1"
    with ProcessPoolExecutor(5) as ex:
        gen = [
            r
            for batch in ex.map(generated, [(root, a, b) for a, b in MATS])
            for r in batch
        ]
    with gzip.open(root / "atlas/game-results.gz", "rt") as f:
        cases = list(map(json.loads, f))
    original = np.load(root / "atlas/game-scores.npz")["s18"][:, 15]
    fs = []
    pairs = []
    for row, z in zip(cases, original, strict=True):
        _, _, fen, u, v, _, _, ((s1, _), (s2, _)) = row
        if s1[1] != s2[1] or sign(s1[-1]) == sign(s2[-1]):
            continue
        q = chess.Board(fen)
        g = group(q.king(True), q.king(False))
        direction = 1 if s1[-1] > s2[-1] else -1
        for m in (u, v) if direction == 1 else (v, u):
            q.push_uci(m)
            q.halfmove_clock = 0
            fs.append(q.fen())
            q.pop()
        pairs.append((g, z * direction))
    value = raw(root / "atlas/engine18/src/stockfish", fs, "release18").reshape(-1, 2)
    played = [(g, t - u, z) for (g, z), (t, u) in zip(pairs, value, strict=True)]
    rows = []
    for name, cases in [("replication", gen), ("played", played)]:
        g, n, z = np.array(cases).T
        diff = score(n) - score(z)
        lo, hi = ci(diff, g)
        rows.append(
            {
                "panel": name,
                "n": len(g),
                "released": score(n).mean(),
                "raw_big": score(z).mean(),
                "paired_delta": diff.mean(),
                "lo": lo,
                "hi": hi,
            }
        )
    write(dest / "release-policy.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    arg = p.parse_args()
    main(arg.root, arg.out)
