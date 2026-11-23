"""Verify the native traces and measure orientation, ranking, and unit ablations."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from collect import EXTENDED, MATERIAL


def stack_weights(root, bucket):
    net = (root / "sf141/src/nn-13406b1dcbe0.nnue").read_bytes()
    # Eight serialized stacks, each prefixed by a 4-byte layer hash.
    size = 4 + 8 * 4 + 8 * 2048 + 32 * 4 + 32 * 32 + 4 + 32
    start = len(net) - (8 - bucket) * size + 4 + 8 * 4 + 8 * 2048

    def take(dtype, n):
        nonlocal start
        v = np.frombuffer(net, dtype=dtype, count=n, offset=start).astype(np.int32)
        start += np.dtype(dtype).itemsize * n
        return v

    return (
        take("<i4", 32),
        take("i1", 32 * 32).reshape(32, 32)[:, :8],
        take("<i4", 1),
        take("i1", 32),
    )


def auc_parts(y, score):
    a, b = score[y], score[~y]
    if min(len(a), len(b)) == 0:
        return np.nan, np.zeros(len(y))
    a0, b0 = np.sort(a), np.sort(b)
    pos = (b0.searchsorted(a, "left") + b0.searchsorted(a, "right")) / (2 * len(b))
    neg = 1 - (a0.searchsorted(b, "left") + a0.searchsorted(b, "right")) / (2 * len(a))
    value = pos.mean()
    inf = np.empty(len(y))
    inf[y], inf[~y] = (pos - value) / len(a), (neg - value) / len(b)
    return value, inf


def auc(y, score):
    return auc_parts(y, score)[0]


def gap(y, neural, classical, groups):
    a, ia = auc_parts(y, neural)
    b, ib = auc_parts(y, classical)
    _, ids = np.unique(groups, return_inverse=True)
    z = np.bincount(ids, weights=ia - ib)
    se = np.sqrt(np.sum(z * z) * len(z) / (len(z) - 1))
    return [float(x) for x in (a - b, a - b - 1.96 * se, a - b + 1.96 * se)]


def material_stats(rows, weights):
    b2, w2, b3, w3 = weights
    u = np.array([r["units"] for r in rows], dtype=np.int32)
    wdl = np.array([r["wdl"] for r in rows])
    cursed = int(np.isin(wdl, [-1, 1]).sum())
    wdl[abs(wdl) == 1] = 0
    delta = wdl.sum(axis=1)
    eligible = delta >= 0
    positive = delta > 0
    fold = np.array([r["fold"] for r in rows])
    fit, test = (fold == 0) & eligible, (fold == 1) & eligible
    groups = np.array([64 * r["group"][0] + r["group"][1] for r in rows])
    assert not set(groups[fit]).intersection(groups[test])
    h1, h2 = u[:, :, 2:10], u[:, :, 10:]

    def forward(x):
        return np.clip((x @ w2.T + b2) >> 6, 0, 127)

    assert np.array_equal(forward(h1), h2)
    assert np.array_equal(h2 @ w3 + b3, u[:, :, 1])
    assert np.all(u[:, 0, 0] == -u[:, 1, 0])
    native = np.array([r["sf141"] for r in rows], dtype=int)
    assert np.array_equal(np.trunc(u[:, :, :2].sum(2) / 16), native[:, :, 1])
    scores = {}
    for e in ("sf141", "sf12"):
        engine = np.array([r[e] for r in rows]).sum(axis=1)
        scores[e + "h"], scores[e + "n"] = engine.T
    caps = np.array([r["capture"] for r in rows]).sum(1)
    promos = np.array([r["promotion"] for r in rows]).sum(1)
    scores["forcing"] = caps + promos
    quiet = scores["forcing"] == 0
    scores["unrounded"] = u[:, :, 1].sum(1)

    def group_stats(mask):
        y = positive[mask]
        if not (0 < y.sum() < len(y)):
            return {
                "n": int(mask.sum()),
                "pos": int(y.sum()),
                "auc": {k: None for k in scores},
                "gap": [None] * 3,
            }
        return {
            "n": int(mask.sum()),
            "pos": int(y.sum()),
            "auc": {k: auc(y, s[mask]) for k, s in scores.items()},
            "gap": gap(y, scores["sf141n"][mask], scores["sf141h"][mask], groups[mask]),
        }

    result = {
        "cursed_or_blessed": cursed,
        "held": group_stats(test),
        "quiet": group_stats(test & quiet),
        "negative": int((delta < 0).sum()),
        "discovery": int(fit.sum()),
        "test_groups": len(set(groups[test])),
    }
    # Conventional position ranking gives each unequal WDL pair one vote within a turn.
    ordinary = {}
    for e in ("sf141", "sf12"):
        engine = np.array([r[e] for r in rows])
        for col, name in enumerate(("h", "n")):
            concordant = pairs = 0
            for turn in (0, 1):
                labels = wdl[fold == 1, turn]
                s = engine[fold == 1, turn, col]
                for hi, lo in ((2, 0), (0, -2), (2, -2)):
                    keep = (labels == hi) | (labels == lo)
                    y = labels[keep] == hi
                    n = int(y.sum()) * int((~y).sum())
                    if n:
                        concordant += n * auc(y, s[keep])
                        pairs += n
            ordinary[e + name] = concordant / pairs
    result["ordinary_concordance"] = ordinary
    # Freeze one coordinate at its discovery-set mean for both turns.
    # Measure the downstream change in ranking with all other weights fixed.
    lesions = {}
    for layer, x in ((1, h1), (2, h2)):
        mu = np.rint(x[fit].mean(axis=(0, 1))).astype(int)
        base = scores["unrounded"]
        loss = {}
        lesioned = []
        for j in range(x.shape[-1]):
            if layer == 1:
                changed = h1.copy()
                changed[:, :, j] = mu[j]
                out = (forward(changed) @ w3).sum(1)
            else:
                out = base + (2 * mu[j] - h2[:, :, j].sum(1)) * w3[j]
            lesioned.append(out)
            loss[j] = auc(positive[fit], base[fit]) - auc(positive[fit], out[fit])
        j = max(loss, key=loss.get)
        measured = [auc(positive[test], p[test]) for p in lesioned]
        lesions[layer] = {
            "unit": j,
            "discovery_drop": loss[j],
            "test_auc": measured[j],
            "all_test_auc": measured,
            "means": mu.tolist(),
            "test_drop": auc(positive[test], base[test]) - measured[j],
        }
    result["lesions"] = lesions
    examples = {}
    for kind, condition in [
        (
            "quiet_success",
            test & positive & quiet & (scores["sf141h"] <= 0) & (scores["sf141n"] > 0),
        ),
        ("quiet_failure", test & positive & quiet & (scores["sf141n"] < 0)),
    ]:
        ids = np.flatnonzero(condition)
        if ids.size:
            i = ids[0]
            examples[kind] = {
                k: rows[i][k] for k in ("board", "wdl", "sf141", "capture", "promotion")
            }
    result["examples"] = examples
    return result


def main(root, extended):
    w = stack_weights(root, int(extended))
    results = {}
    for m in EXTENDED if extended else MATERIAL:
        with gzip.open(root / (m + ".jsonl.gz"), "rt") as f:
            results[m] = material_stats([json.loads(line) for line in f], w)
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    p.add_argument("--extended", action="store_true")
    args = p.parse_args()
    main(args.scratch, args.extended)
