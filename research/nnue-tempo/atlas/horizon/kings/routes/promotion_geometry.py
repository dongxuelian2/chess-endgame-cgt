"""Test whether penultimate-rank pawn channels also distinguish the defender's promotion rays."""

import sys
from pathlib import Path

import numpy as np
import validate as v
from network import Network
from summarize import write


def relation(a, x, target, w, k):
    x = np.asarray(x)
    dx = target % 8 - x % 8
    dy = target // 8 - x // 8
    ray = ((a in (4, 5)) & ((dx == 0) | (dy == 0))) | (
        (a in (3, 5)) & (abs(dx) == abs(dy))
    )
    if a == 2:
        ray = abs(dx * dy) == 2

    def blocks(z):
        return (
            (a != 2)
            & ray
            & ((z % 8 - x % 8) * dy == (z // 8 - x // 8) * dx)
            & (
                (z % 8 - x % 8) * (z % 8 - target % 8)
                + (z // 8 - x // 8) * (z // 8 - target // 8)
                < 0
            )
        )

    blocked = (blocks(w) | blocks(k)) if w is not None else np.zeros_like(ray)
    return np.select(
        [x == target, np.equal(w, target) | np.equal(k, target), ~ray, blocked],
        [0, 1, 2, 3],
        default=4,
    )


def main(root, out):
    v.init(root)
    net = Network(root)
    vals = net.w[:, 1, 8:56].astype("i4")
    idx = vals.argmax(1)
    gap = vals.max(1) - np.partition(vals, -2, axis=1)[:, -2]
    ids = np.flatnonzero((idx[0] < 8) & (idx == idx[:1]).all(0) & (gap.min(0) > 0))
    report = []
    weight_rows = []
    labels = [
        "defender_on_promotion",
        "king_on_promotion",
        "off_ray",
        "king_blocks_ray",
        "clear_ray",
    ]
    for a in (2, 3, 4, 5):
        v.state(a, 1, 0, 0, 0, 0)
        for i in ids:
            square = int(8 + idx[0, i])
            target = (square ^ 7) - 8
            counts = np.zeros((5, 4), dtype="i8")
            for w in range(64):
                if w % 8 >= 4:
                    continue
                k, x = np.where(v.maps[a, 1][w, :, square ^ 7, :]["v"] > 0)
                wk = w ^ 7
                bucket = 31 - 4 * (wk // 8) - (wk % 8 - 4)
                W = net.w[bucket, :, :, i]
                r = (
                    net.bias[i]
                    + W[10, wk]
                    + W[10, k ^ 7]
                    + W[1, square]
                    + W[2 * (a - 1), x ^ 7]
                ).clip(0, 127)
                flag = relation(a, x, target, w, k)
                for g in range(5):
                    sub = r[flag == g]
                    counts[g] += [
                        len(sub),
                        np.count_nonzero(sub),
                        sub.sum(),
                        np.count_nonzero(sub == 127),
                    ]
            for label, (n, active, s, c) in zip(labels, counts):
                if n:
                    report.append(
                        {
                            "white": a,
                            "ft": 1024 + int(i),
                            "oriented_pawn": square,
                            "coverage": label,
                            "n": n,
                            "fires": active,
                            "mean_byte": s / n,
                            "ceiling": c,
                        }
                    )
            squares = np.arange(64)
            t = square - 8
            g = relation(a, squares, t, None, None)
            off = g == 2
            on = (g == 4) & (squares != square)
            if not np.any(on):
                continue
            wgt = net.w[:, 2 * (a - 1), :, i]
            contrast = wgt[:, off].mean(1) - wgt[:, on].mean(1)
            weight_rows.append(
                {
                    "white": a,
                    "ft": 1024 + int(i),
                    "oriented_pawn": square,
                    "mean_off_minus_on": contrast.mean(),
                    "minimum": contrast.min(),
                    "maximum": contrast.max(),
                    "positive_buckets": int((contrast > 0).sum()),
                }
            )
    write(out / "promotion-coverage.csv", report)
    write(out / "promotion-weights.csv", weight_rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
