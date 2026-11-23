"""Recheck the behavioral and channel results after excluding every immediate capture and promotion."""

import argparse
from pathlib import Path

import numpy as np

from summarize import MATS, bits, game_panel, mean, score, write
from validate import RULES


def main(root, dest):
    ds = []
    for a, b in MATS:
        f = root / "atlas/remaining/metrics" / f"{a}-{b}.npz"
        if f.exists():
            z = np.load(f)
            r = z["panel"]
            ds.append([r[:, 2], r[:, 3], r[:, 6], z["s18"], z["erased"]])
    g, _ = game_panel(root)
    weights = np.loadtxt(root / "atlas/weights.txt", dtype=int)[:, 0]
    out = []
    for name, (hi, lo, n, s18, er) in [
        ("replication", list(map(np.concatenate, zip(*ds)))),
        ("played", [g[k] for k in ("hh", "ll", "nn", "modern", "erased")]),
    ]:
        mask = ((hi | lo) & (1 | 2 | 16 | 32)) == 0
        X = bits(hi) - bits(lo)
        for j, rule in enumerate(["ALL", *RULES]):
            keep = mask if j == 0 else mask & ((er >> (j - 1) & 1) != 0)
            if not keep.any():
                continue
            out.append(
                {
                    "panel": name,
                    "rule": rule,
                    "n": int(sum(keep)),
                    "sf141": mean(score(n[keep])),
                    "sf18": mean(score(s18[keep, 15])),
                    "sf18_without": mean(score(s18[keep, 0])),
                    "fitted": mean(score((X @ weights)[keep])),
                }
            )
    write(dest / "quiet-residual.csv", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
