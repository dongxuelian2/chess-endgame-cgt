"""Replay chess action tests and resolve motif-tagged rectangles."""

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import validate as v
from promotion_geometry import relation
from summarize import write
from tactics import LABELS, classify

from cohorts import kings, orbit
from promotion_errors import labels


@lru_cache(maxsize=32768)
def promotion_actions(a, w, k, y, h, l):
    return labels((a, (w, k, y, h, l, 0)))


def qualifies(job):
    a, b, rows = job
    for row in rows:
        w, k, y, z, h, l, _ = map(int, row)
        if b == 1:
            cap, straight, _ = promotion_actions(a, w, k, y, h, l)
            if not cap or straight:
                continue
            cap, straight, _ = promotion_actions(a, w, k, z, l, h)
            if not straight or cap:
                continue
        else:
            i = LABELS.index("rook_skewer")
            if not all(
                (m := classify((piece, b, w, k, y, h, l)))[i] and m[i + len(LABELS)]
                for piece in (a, 5)
            ):
                continue
        return row
    return None


def main(root, out):
    groups = sorted({orbit(w, k) for w in range(64) for k in range(64)})
    validation = np.random.default_rng(96164).permutation(groups)[::2]
    flow = []
    with ProcessPoolExecutor(
        8, mp_context=get_context("fork"), initializer=v.init, initargs=(root,)
    ) as pool:
        for a, b in [(i, 1) for i in range(2, 6)] + [(3, 4)]:
            d = root / "manuscript/motifs"
            rows = np.loadtxt(d / f"{a}-{b}.txt", dtype="i8", ndmin=2)
            n = len(rows)
            if b == 1:
                w, k, y, _, h, l, _ = rows.T
                (y ^ 7) % 8
                # Keep both the known harmful regime and the other coverage regimes.
                hazard = (relation(a, h, y - 8, w, k) == 4) & (
                    relation(a, l, y - 8, w, k) == 2
                )
            else:
                np.zeros(n, dtype=int)
                hazard = np.zeros(n, dtype=bool)
            rng = np.random.default_rng(48712)
            order = np.lexsort(
                (
                    rng.permutation(n),
                    hazard,
                    rows[:, 3],
                    rows[:, 2],
                    rows[:, 1],
                    rows[:, 0],
                )
            )
            key = np.column_stack([rows[:, :4], hazard])[order]
            split = np.r_[0, np.where(np.any(key[1:] != key[:-1], axis=1))[0] + 1, n]
            batches = [
                rows[order[i : min(i + 16, j)]] for i, j in itertools.pairwise(split)
            ]
            result = list(
                pool.map(qualifies, [(a, b, r) for r in batches], chunksize=20)
            )
            chosen = np.array([r for r in result if r is not None], dtype="i8").reshape(
                -1, 7
            )
            held = np.isin(kings(chosen), validation)
            np.savez_compressed(d / f"panel-{a}-{b}.npz", rows=chosen, validation=held)
            flow.append(
                {
                    "white": a,
                    "black": b,
                    "geometric_candidates": n,
                    "strata": len(batches),
                    "shortlist": sum(map(len, batches)),
                    "qualified": len(chosen),
                    "discovery": int((~held).sum()),
                    "validation": int(held.sum()),
                    "validation_king_orbits": len(set(kings(chosen[held]))),
                }
            )
    write(out / "motif-census.csv", flow)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
