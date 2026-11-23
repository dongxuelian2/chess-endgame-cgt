"""Measure additional predictive content with immediate capture and promotion opportunities held fixed."""

import argparse
from pathlib import Path

import numpy as np

from summarize import MATS, bits, game_panel, group, read_panels, score, write
from validate import RULES

# All capture and promotion subtype flags among the first nine rules.
BASIC_MASK = (1 << 9) - 1


def select(p):
    h, l = p["hh"], p["ll"]
    return ((h ^ l) & BASIC_MASK) == 0


def main(root, out):
    d = read_panels(root)
    train = (d["fold"] == 0) & select(d)
    direction = np.sign((bits(d["hh"][train]) - bits(d["ll"][train])).sum(0))
    hh = []
    ll = []
    gg = []
    nn = []
    for a, b in MATS:
        path = root / "atlas/remaining/metrics" / f"{a}-{b}.npz"
        if not path.exists():
            continue
        rows = np.load(path)["panel"]
        hh.extend(rows[:, 2])
        ll.extend(rows[:, 3])
        gg.extend(group(int(i >> 12), int(i >> 6 & 63)) for i in rows[:, 0])
        nn.extend(rows[:, 6])
    rep = {
        "hh": np.array(hh),
        "ll": np.array(ll),
        "groups": np.array(gg),
        "nn": np.array(nn),
    }
    game, _ = game_panel(root)
    rows = []
    for name, p in [("replication", rep), ("played", game)]:
        use = select(p)
        x = (bits(p["hh"][use]) - bits(p["ll"][use])) * direction
        groups = p["groups"][use]
        n = p["nn"][use]
        rng = np.random.default_rng(9456)
        # For each rule measure mean accuracy on its strict predictions, conditional on the basic signature.
        key, inv = np.unique(groups, return_inverse=True)
        cnt = np.zeros((len(key), len(RULES)))
        success = cnt.copy()
        np.add.at(cnt, inv, x != 0)
        np.add.at(success, inv, x > 0)
        totals = cnt.sum(0)
        rates = success.sum(0) / np.maximum(totals, 1)
        influence = (success - cnt * rates) / np.maximum(totals, 1)
        se = np.sqrt(np.sum(influence**2, axis=0) * len(key) / (len(key) - 1))
        ok = (totals >= 30) & ((cnt > 0).sum(0) >= 10) & (se > 0)
        q = (
            np.quantile(
                np.max(
                    abs(
                        rng.choice([-1.0, 1.0], (4999, len(key)))
                        @ (influence[:, ok] / se[ok])
                    ),
                    axis=1,
                ),
                0.95,
            )
            if ok.any()
            else 0
        )
        for j, t in enumerate(RULES):
            active = x[:, j] != 0
            rows.append(
                {
                    "panel": name,
                    "rule": t,
                    "basic_signature_ties": int(use.sum()),
                    "direction": int(direction[j]),
                    "strict_secondary": int(totals[j]),
                    "groups": int((cnt[:, j] > 0).sum()),
                    "rule_accuracy": rates[j] if totals[j] else None,
                    "lo": rates[j] - q * se[j] if ok[j] else None,
                    "hi": rates[j] + q * se[j] if ok[j] else None,
                    "mean_gain_all": float(x[:, j].mean() / 2),
                    "native_on_same": float(score(n[active]).mean())
                    if active.any()
                    else None,
                }
            )
    write(out / "incremental-rules.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
