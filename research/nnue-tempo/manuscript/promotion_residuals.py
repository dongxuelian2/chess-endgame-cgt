"""Audit strict errors outside the surviving-capture correction's support."""

import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import validate as v
from summarize import write

from promotion_errors import labels
from promotion_prediction import board_features


def case(job):
    a, row, required = job
    w, k, y, h, l, _ = row
    high = board_features(a, w, k, y, h)[2]
    low = board_features(a, w, k, y, l)[2]
    if high != low:
        return "capture_term_differs"
    if not required:
        return "promotion_deletion_preserves_order"
    cap, push, _minor = labels((a, row))
    if push and cap:
        return "each_promotion_deletion_erases_order"
    if push:
        return "straight_deletion_erases_order"
    if cap:
        return "capture_deletion_erases_order"
    return "only_combined_deletion_erases_order"


def main(root, out):
    v.init(root)
    records = []
    with ProcessPoolExecutor(
        8, mp_context=get_context("fork"), initializer=v.init, initargs=(root,)
    ) as pool:
        for a in range(2, 6):
            panel = np.load(root / "manuscript" / f"validation-{a}-1.npz")
            r = panel["rows"]
            w, k, y, h, l, _ = r.T
            m = (
                v.state(a, 1, w, k, y, h)["n"].astype(int)
                - v.state(a, 1, w, k, y, l)["n"]
            )
            wrong = m < 0
            rows = r[wrong].tolist()
            counts = Counter(
                pool.map(
                    case,
                    [(a, row, p) for row, p in zip(rows, panel["mask"][wrong])],
                    chunksize=48,
                )
            )
            for k, n in sorted(counts.items()):
                records.append({"white": a, "condition": k, "strict_errors": n})
    write(out / "prediction-promotion-residuals.csv", records)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
