"""Cross-version static readouts and exact FullThreats factorial counterfactuals."""

import argparse
import gzip
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import chess
import numpy as np

from validate import DOMAIN, REPLY, board


def raw(exe, fens, tag, knockout=False):
    env = os.environ.copy()
    env.pop("SF18_ZERO_THREATS", None)
    if knockout:
        env["SF18_ZERO_THREATS"] = "1"
    cmd = "".join("position fen " + fen + "\neval\n" for fen in fens) + "quit\n"
    proc = subprocess.run(
        [exe], input=cmd, env=env, text=True, capture_output=True, check=True
    )
    rows = [
        list(map(int, s.split()[1:]))
        for s in proc.stdout.splitlines()
        if s.startswith(tag + " ")
    ]
    assert len(rows) == len(fens)
    return np.array(rows)


def readouts(root, fs):
    t = raw(root / "atlas/engine18/src/stockfish", fs, "atlas18")
    sample = fs[:24]
    for flag, mask in ((False, 15), (True, 0)):
        u = raw(root / "sf18/src/stockfish", sample, "raw18", flag)
        assert np.array_equal(u[:, :2].sum(1), t[: len(sample), mask])
    s12 = raw(root / "sf12/src/stockfish", fs, "scores")[:, 1]
    return t.reshape(-1, 2, 16), s12.reshape(-1, 2)


def material(job):
    root, a, b, remaining = job
    prefix = root / "atlas"
    prefix = prefix / "remaining" if remaining else prefix
    stem = f"{a}-{b}"
    rr = np.memmap(prefix / (stem + ".replies"), mode="r", dtype=REPLY)
    if not (prefix / "metrics" / (stem + ".pairs")).stat().st_size:
        return
    rows = np.loadtxt(prefix / "metrics" / (stem + ".pairs"), dtype="i8", ndmin=2)
    if remaining:
        rng = np.random.default_rng(20261122 + 31 * a + b)
        rows = rows[rng.integers(0, 8, size=len(rows)) == 0]
    if rows.size == 0:
        return
    indices = rows[:, 0]
    idx = np.searchsorted(rr["index"], indices)
    assert np.array_equal(rr["index"][idx], indices)
    hi, lo = rows[:, -2:].T
    rm = rr["mask"]
    rh = rm[idx, hi].copy()
    rl = rm[idx, lo].copy()
    erased = (
        rh[:, 1]
        & rl[:, 1]
        & ~((rh[:, 3] & ~rl[:, 3]) | (rh[:, 2] & ~rl[:, 2]))
        & np.uint64(DOMAIN)
    )
    fens = [
        board(a, b, int(i >> 12), int(i >> 6 & 63), int(i & 63), int(s)).fen()
        for i, pair in zip(indices, rows[:, -2:])
        for s in pair
    ]
    t, s12 = readouts(root, fens)
    np.savez_compressed(
        prefix / "metrics" / (stem + ".npz"),
        panel=rows,
        erased=erased,
        s18=t[:, 0] - t[:, 1],
        s12=s12[:, 0] - s12[:, 1],
    )


def evaluate_games(root):
    with gzip.open(root / "atlas/game-results.gz", "rt") as f:
        ds = list(map(json.loads, f))
    fs = []
    for _, _, fen, u, v, *_ in ds:
        q = chess.Board(fen)
        for m in (u, v):
            q.push_uci(m)
            fs.append(q.fen())
            q.pop()
    t, s12 = readouts(root, fs)
    np.savez_compressed(
        root / "atlas/game-scores.npz", s18=t[:, 0] - t[:, 1], s12=s12[:, 0] - s12[:, 1]
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--remaining", action="store_true")
    args = p.parse_args()
    with ProcessPoolExecutor(6) as pool:
        list(
            pool.map(
                material,
                [
                    (args.root, a, b, args.remaining)
                    for a, b in product(range(1, 6), repeat=2)
                ],
            )
        )
    if not args.remaining:
        evaluate_games(args.root)
