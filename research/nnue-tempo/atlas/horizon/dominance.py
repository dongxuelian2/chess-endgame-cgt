"""Measure every safe two-action preference certified by the interval endpoints."""

import argparse
from pathlib import Path

import numpy as np
from results import load, values
from summarize import ci, group, score, write


def main(root, out):
    rows = []
    for name in ("all5", "games7", "unselected5"):
        panel, _, ds, r = load(root / "atlas/horizon" / name)
        g = np.array([group(k, w) for k, w in zip(ds["wk"], ds["bk"])])
        base = score(ds["nn"])
        gap = ds["wh"] - ds["wl"]
        for mask in (9, 15):
            v = values(r, mask)
            for h in (3, 5, 7, 9, 11, 13):
                if name == "games7" and h <= 5:
                    continue
                if h >= v.shape[2]:
                    if mask != 15 or name == "unselected5":
                        continue
                    proof = np.load(root / "atlas/horizon" / name / "proofs.npz")[
                        "proof"
                    ]
                    t = proof <= h
                    L = t[:, :, :2].sum(2) - 1
                    U = 1 - t[:, :, 2:].sum(2)
                    neutral = np.clip(0, L, U)
                else:
                    L, neutral, U = (v[:, :, h, i] for i in range(3))
                    assert np.array_equal(neutral, np.clip(0, L, U))
                weak = (L[0] >= U[1]).astype("i1") - (L[1] >= U[0]).astype("i1")
                point = neutral[0] - neutral[1]
                assert np.all(weak * gap >= 0) and np.all(
                    weak[point != 0] == np.sign(point[point != 0])
                )
                for pop in np.unique(panel):
                    if name == "all5" and pop != "enumerated" and h > 5:
                        continue
                    m = panel == pop
                    hybrid = np.where(weak != 0, score(weak), base)
                    gain = (hybrid - base) * gap
                    lo, hi = ci(gain[m], g[m])
                    rows.append(
                        {
                            "panel": pop,
                            "selection": name,
                            "mask": mask,
                            "depth": h,
                            "n": int(m.sum()),
                            "true_unequal": int((m & (gap != 0)).sum()),
                            "weak_choice": int((m & (weak != 0)).sum()),
                            "true_unequal_weak_choice": int(
                                (m & (weak != 0) & (gap != 0)).sum()
                            ),
                            "new_over_neutral": int(
                                (m & (weak != 0) & (point == 0) & (gap != 0)).sum()
                            ),
                            "strict_interval": int(
                                (m & ((L[0] > U[1]) | (L[1] > U[0]))).sum()
                            ),
                            "nnue_on_undecided": base[
                                m & (weak == 0) & (gap != 0)
                            ].mean()
                            if (m & (weak == 0) & (gap != 0)).any()
                            else None,
                            "hybrid_accuracy_on_unequal": hybrid[m & (gap != 0)].mean(),
                            "mean_wdl_gain": gain[m].mean(),
                            "lo": lo,
                            "hi": hi,
                        }
                    )
    write(out / "interval-decisions.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
