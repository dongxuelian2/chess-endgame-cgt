"""Separate king-layout orbits and relabel alternate destinations by legal move deletion."""

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import validate as v
from promotion_trial import restrict
from summarize import group, write
from tactics import LABELS, classify


def orbit(w, k):
    def transpose(x):
        return (x % 8) * 8 + x // 8

    return min(group(w, k), group(transpose(w), transpose(k)))


def kings(rows):
    return np.array([orbit(int(w), int(k)) for w, k in rows[:, :2]])


def tactic(job):
    a, b, *r = job
    i = LABELS.index("rook_skewer" if b == 4 else "safe_knight_fork")
    for p in (a, 5):
        m = classify((p, b, *r[:5]))
        if not m[i] or (b == 2 and not m[i + len(LABELS)]):
            return False
    return True


def main(root, out):
    work = root / "manuscript"
    routes = root / "atlas/horizon/kings/routes"
    groups = sorted({orbit(w, k) for w in range(64) for k in range(64)})
    test = np.random.default_rng(96164).permutation(groups)[::2]
    records = []
    with ProcessPoolExecutor(
        8, mp_context=get_context("fork"), initializer=v.init, initargs=(root,)
    ) as pool:
        for a, b in [(i, 1) for i in range(2, 6)] + [(3, 4), (4, 2)]:
            path = routes / (f"promote-{a}.txt" if b == 1 else f"{a}-{b}.pairs")
            base = np.loadtxt(path, dtype=int, ndmin=2)
            alt = np.loadtxt(work / f"alternate-{a}-{b}.txt", dtype=int, ndmin=2)
            assert np.array_equal(base[:, :3], alt[:, :3])
            changed = np.any(base[:, 3:5] != alt[:, 3:5], axis=1)
            held = np.isin(kings(base), test)
            for sample, rows in [
                ("discovery", base[~held]),
                ("validation", alt[held & changed]),
            ]:
                jobs = [(a, *r) if b == 1 else (a, b, *r) for r in rows.tolist()]
                mask = np.array(
                    list(pool.map(restrict if b == 1 else tactic, jobs, chunksize=100)),
                    bool,
                )
                np.savez_compressed(
                    work / f"{sample}-{a}-{b}.npz", rows=rows, mask=mask
                )
                records.append(
                    {
                        "sample": sample,
                        "white": a,
                        "black": b,
                        "eligible": len(base),
                        "split_layouts": int(
                            (held if sample == "validation" else ~held).sum()
                        ),
                        "tested": len(rows),
                        "necessary": int(mask.sum()),
                        "king_orbits": len(set(kings(rows[mask]))),
                    }
                )
    write(out / "cohorts.csv", records)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
