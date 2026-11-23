"""Forward rule selection on discovery alone, followed by fixed-size held-out readouts."""

import argparse
from pathlib import Path

import numpy as np

from summarize import MATS, bits, ci, fit, game_panel, read_panels, score, write
from validate import RULES


def main(root, out):
    d = read_panels(root)
    ix = d["fold"] == 0
    X = bits(d["hh"][ix]) - bits(d["ll"][ix])
    hs = []
    ls = []
    gs = []
    from summarize import group

    for a, b in MATS:
        p = root / "atlas/remaining/metrics" / f"{a}-{b}.npz"
        if p.exists():
            rows = np.load(p)["panel"]
            hs.extend(rows[:, 2])
            ls.extend(rows[:, 3])
            gs.extend(group(int(i >> 12), int(i >> 6 & 63)) for i in rows[:, 0])
    hh, ll = np.array(hs), np.array(ls)
    g, _ = game_panel(root)
    panels = [
        ("replication", hh, ll, np.array(gs)),
        ("played", g["hh"], g["ll"], g["groups"]),
    ]
    frozen = []
    trajectory = []
    coef = np.zeros(0)
    signal = np.zeros(len(X))
    remaining = list(range(len(RULES)))
    moments = (X * X).mean(0)
    outrows = []
    for step in range(1, len(RULES) + 1):
        residual = 1 / (1 + np.exp(np.clip(signal, -40, 40)))
        derivative = abs(X.T @ residual / len(X)) / np.sqrt(moments + 0.01)
        j = max(remaining, key=lambda j: derivative[j])
        remaining.remove(j)
        frozen.append(j)
        coef = fit(X[:, frozen])
        signal = X[:, frozen] @ coef
        trajectory.append(
            {"step": step, "added": RULES[j], "training_score": score(signal).mean()}
        )
        if step not in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
            continue
        for name, h, l, clusters in panels:
            Z = bits(h) - bits(l)
            quiet = ((h | l) & 51) == 0
            # Fixed rounding gives identical scores to identical feature signatures.
            pred = score(Z[:, frozen] @ np.rint(1e6 * coef))
            for scope, mask in [("all", np.ones(len(h), bool)), ("quiet", quiet)]:
                lo, hi = ci(pred[mask], clusters[mask])
                outrows.append(
                    {
                        "panel": name,
                        "scope": scope,
                        "k": step,
                        "n": int(mask.sum()),
                        "accuracy": pred[mask].mean(),
                        "lo": lo,
                        "hi": hi,
                    }
                )
    write(out / "rule-selection.csv", trajectory)
    write(out / "rule-budget.csv", outrows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
