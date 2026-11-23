"""Evaluate first-choice utility without selecting games by the true outcome gap."""

import argparse
from pathlib import Path

import numpy as np
from results import load, values
from summarize import ci, group, score, write


def main(root, out):
    panel, _, d, refs = load(root / "atlas/horizon/unselected5")
    gr = np.array([group(w, k) for w, k in zip(d["wk"], d["bk"])])
    # A positive score difference favors the first child for Black. White chooses the lower score.
    gap = d["wh"] - d["wl"]
    base = score(d["nn"])
    w = np.loadtxt(root / "atlas/weights.txt", dtype=int)[:, 0]
    fitted = score(
        sum((((d["h"] >> j) & 1) - ((d["l"] >> j) & 1)) * c for j, c in enumerate(w))
    )
    rows = []
    for mask in (0, 8, 9, 11, 15):
        v = values(refs, mask)
        for h in (3, 5):
            l, m, u = (v[:, :, h, i] for i in range(3))
            assert np.array_equal(m, np.clip(0, l, u))
            diff = m[0] - m[1]
            strict = diff != 0
            assert (diff * gap >= 0).all()
            for name in np.unique(panel):
                keep = panel == name
                for policy, secondary in [
                    ("classical_tie", np.repeat(0.5, len(base))),
                    ("nnue", base),
                    ("rules48", fitted),
                ]:
                    pred = np.where(strict, score(diff), secondary)
                    gain = (pred - secondary) * gap
                    assert np.all(gain >= 0)
                    lo, hi = ci(gain[keep], gr[keep])
                    rows.append(
                        {
                            "panel": name,
                            "depth": h,
                            "mask": mask,
                            "secondary": policy,
                            "n": int(keep.sum()),
                            "true_unequal": int((keep & (gap != 0)).sum()),
                            "strict_true_equal": int(
                                (keep & strict & (gap == 0)).sum()
                            ),
                            "strict_true_unequal": int(
                                (keep & strict & (gap != 0)).sum()
                            ),
                            "changed_wdl_up": int((keep & (gain > 0)).sum()),
                            "changed_wdl_down": int((keep & (gain < 0)).sum()),
                            "mean_wdl_gain": gain[keep].mean(),
                            "lo": lo,
                            "hi": hi,
                        }
                    )
    write(out / "unselected-policy.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
