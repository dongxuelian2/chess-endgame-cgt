"""Sample fixed-material tactical tests on king/attacker triples outside the substitution panel."""

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import validate as v
from summarize import write
from tactics import LABELS, classify


def motif(job):
    a, b, *r = job
    m = classify((a, b, *r[:5]))
    i = LABELS.index("rook_skewer" if b == 4 else "safe_knight_fork")
    return m[i] and (b == 4 or m[i + len(LABELS)])


def main(root, out):
    work = root / "manuscript"
    counts = []
    with ProcessPoolExecutor(
        8, mp_context=get_context("fork"), initializer=v.init, initargs=(root,)
    ) as pool:
        for a, b in [(3, 4), (4, 2)]:
            base = np.loadtxt(work / f"alternate-{a}-{b}.txt", dtype=int)
            orig = set(map(tuple, base[:, :3]))
            for piece in (a, 5):
                rows = np.loadtxt(work / f"solo-{piece}-{b}.txt", dtype=int)
                new = rows[[tuple(r) not in orig for r in rows[:, :3]]]
                jobs = [(piece, b, *r) for r in new.tolist()]
                selected = np.array(list(pool.map(motif, jobs, chunksize=100)), bool)
                np.savez_compressed(
                    work / f"transfer-{piece}-{b}.npz", rows=new, mask=selected
                )
                counts.append(
                    {
                        "white": piece,
                        "black": b,
                        "solo_candidates": len(rows),
                        "outside_matched": len(new),
                        "necessary": int(selected.sum()),
                    }
                )
    write(out / "outside-layouts.csv", counts)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
