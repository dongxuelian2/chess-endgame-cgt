"""Check the full reflected partner of every canonical four-man board."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from summarize import MATS, mat, write
from validate import DT


def census(job):
    root, a, b = job
    m = np.memmap(
        root / "atlas" / f"{a}-{b}.bin", mode="r", dtype=DT, shape=(64, 64, 64, 64)
    )
    ix = np.arange(64) ^ 7
    counts = np.zeros(6, dtype="int64")
    peaks = counts.copy()
    n = 0
    for wk in range(64):
        if wk % 8 > 3:
            continue
        left = m[wk]
        right = m[wk ^ 7][np.ix_(ix, ix, ix)]
        valid = left["v"].astype(bool)
        assert np.array_equal(left["v"], right["v"])
        n += valid.sum()
        for j, key in enumerate(DT.names[:-1]):
            d = (left[key].astype("i4") - right[key])[valid]
            counts[j] += np.count_nonzero(d)
            peaks[j] = max(peaks[j], np.max(abs(d), initial=0))
    assert counts[-1] == 0
    return dict(
        material=mat(a, b),
        pairs=int(n),
        **{f"{k}_unequal": int(v) for k, v in zip(DT.names, counts)},
        **{f"{k}_max_abs": int(v) for k, v in zip(DT.names, peaks)},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    arg = p.parse_args()
    with ThreadPoolExecutor(5) as pool:
        write(
            arg.out / "reflections.csv",
            list(pool.map(census, [(arg.root, a, b) for a, b in MATS])),
        )
