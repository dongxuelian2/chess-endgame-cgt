"""Exact random-deletion comparator matched to each chess rule's deletion size in both children."""

import argparse
import gzip
import json
import math
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import batched
from pathlib import Path

import chess
import numpy as np

from summarize import MATS, write
from validate import DOMAIN, RULES, init, probe, sign


@lru_cache(8192)
def cdf(n, low, draw):
    # Entry zero is unused because that restriction leaves no legal replies.
    k = np.arange(n + 1)
    return np.array(
        [
            [math.comb(c, int(j)) / math.comb(n, int(j)) for j in k]
            for c in (low, low + draw)
        ]
    )


def child(a, b, wk, bk, y, x):
    _, moves = probe(a, b, wk, bk, y, x, True)
    scores = np.array([z[1] for z in moves])
    m = len(moves)
    tags = np.array([z[2] for z in moves], dtype="<u8")
    indicator = np.unpackbits(
        tags.view("u1").reshape(-1, 8), axis=1, bitorder="little"
    )[:, : len(RULES)]
    retained = m - indicator.sum(0).astype(int)
    restricted = np.max(np.where(indicator != 0, -2, scores[:, None]), axis=0)
    f = cdf(m, int(sum(scores == -1)), int(sum(scores == 0)))[:, retained]
    return retained, m, restricted, f


def batch_run(rows):
    out = np.zeros((2, len(RULES), 3))
    for a, b, wk, bk, y, hi, lo, erased in rows:
        h, mh, vh, F = child(a, b, wk, bk, y, hi)
        l, ml, vl, G = child(a, b, wk, bk, y, lo)
        eligible = (h > 0) & (l > 0) & ((h < mh) | (l < ml))
        event = (vh <= vl) & eligible
        actual = sum(1 << int(t) for t in np.flatnonzero(event))
        assert actual == erased, (a, b, wk, bk, y, hi, lo, actual, erased)
        random = F[0] * G[0] + F[1] * (G[1] - G[0]) + 1 - G[1]
        obs = np.column_stack((eligible, event, np.clip(random, 0, 1) * eligible))
        out[0] += obs
        primary = [0, 1, 4, 5]
        if np.all(h[primary] == mh) and np.all(l[primary] == ml):
            out[1] += obs
    return out


def samples(root, tag):
    if tag == "generated":
        for a, b in MATS:
            p = root / "atlas/remaining/metrics" / f"{a}-{b}.npz"
            if not p.exists():
                continue
            s = np.load(p)
            for r, e in zip(s["panel"], s["erased"], strict=True):
                i = int(r[0])
                yield a, b, i >> 12, i >> 6 & 63, i & 63, int(r[-2]), int(r[-1]), int(e)
    else:
        with gzip.open(root / "atlas/game-results.gz", "rt") as f:
            for month, gid, fen, u, v, a, b, opts in map(json.loads, f):
                (z, r), (w, s) = opts
                if z[1] != w[1] or sign(z[-1]) == sign(w[-1]):
                    continue
                if z[-1] < w[-1]:
                    u, v, r, s = v, u, s, r
                q = chess.Board(fen)
                wk, bk = q.king(True), q.king(False)
                y = next(iter(chess.SquareSet(q.occupied_co[False] & ~q.kings)))
                e = r[1] & s[1] & ~((r[3] & ~s[3]) | (r[2] & ~s[2])) & DOMAIN
                yield (
                    a,
                    b,
                    wk,
                    bk,
                    y,
                    chess.parse_square(u[2:4]),
                    chess.parse_square(v[2:4]),
                    e,
                )


def main(root, dest):
    rows = []
    for panel in ("played", "generated"):
        tot = np.zeros((2, len(RULES), 3))
        with ProcessPoolExecutor(12, initializer=init, initargs=(root,)) as ex:
            for z in ex.map(batch_run, batched(samples(root, panel), 64), chunksize=1):
                tot += z
        for i, scope in enumerate(("all", "no_immediate_capture_or_promotion")):
            for rule, (n, event, null) in zip(RULES, tot[i], strict=True):
                rows.append(
                    {
                        "panel": panel,
                        "scope": scope,
                        "rule": rule,
                        "eligible": int(n),
                        "erased": int(event),
                        "expected_random": null,
                        "excess_fraction": (event - null) / n if n else None,
                        "ratio": event / null if null > 1e-10 else None,
                    }
                )
        write(dest / "size-matched-deletions.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    main(a.root, a.dest)
