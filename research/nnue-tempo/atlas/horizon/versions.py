"""Compare three frozen native networks on the same new-game children and proof strata."""

import argparse
from pathlib import Path

import numpy as np
from panels import readouts
from results import load
from summarize import mat, score, write
from validate import board


def main(root, out):
    panel, _, data, _refs = load(root / "atlas/horizon/games7")
    proof = np.load(root / "atlas/horizon/games7/proofs.npz")["proof"]
    query = [
        board(*map(int, row)).fen()
        for row in zip(
            data["a"], data["b"], data["wk"], data["bk"], data["y"], data["x"]
        )
    ]
    other = [
        board(*map(int, row)).fen()
        for row in zip(
            data["a"], data["b"], data["wk"], data["bk"], data["y"], data["z"]
        )
    ]
    fens = [f for pair in zip(query, other) for f in pair]
    t, s12 = readouts(root, fens)
    modern = t[:, 0] - t[:, 1]
    old = s12[:, 0] - s12[:, 1]
    p = proof <= 13
    lo = p[:, :, :2].sum(2) - 1
    hi = 1 - p[:, :, 2:].sum(2)
    unresolved = (lo[0] < hi[1]) & (lo[1] < hi[0])
    rows = []
    for name in np.unique(panel):
        t = panel == name
        groups = [("all", t), ("interval_undecided_13", t & unresolved)]
        groups.extend(
            (mat(a, b), t & (data["a"] == a) & (data["b"] == b))
            for a, b in sorted(set(zip(data["a"][t], data["b"][t])))
        )
        for label, mask in groups:
            if not mask.any():
                continue
            rows.append(
                {
                    "panel": name,
                    "condition": label,
                    "n": int(mask.sum()),
                    "sf12": score(old[mask]).mean(),
                    "sf141": score(data["nn"][mask]).mean(),
                    "sf18_raw_big": score(modern[mask, 15]).mean(),
                    "sf18_direct_edges_off": score(modern[mask, 0]).mean(),
                }
            )
    write(out / "game-generations.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
