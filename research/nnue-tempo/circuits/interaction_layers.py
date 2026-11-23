"""Decompose an observed move-by-attacker interaction along the three NNUE transforms."""

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
from analyze import stack_weights
from feature_routes import ft_weights, get_features, slope
from spectra import load


def endpoints(rows, root, weights):
    b1, w1, b2, w2, _w3 = weights
    x, pre = get_features(rows, root)
    x = x.astype(float)
    u = np.array([r["units"] for r in rows])
    assert np.array_equal(x @ w1.T + b1, pre)
    p2 = u[:, :, 2:10] @ w2.T + b2
    assert np.array_equal((pre >> 6).clip(0, 127), u[:, :, 2:10])
    assert np.array_equal((p2 >> 6).clip(0, 127), u[:, :, 10:])
    delta = lambda v: v[:, 1] - v[:, 0]

    def gates(post, p):
        d = delta(p)
        sec = slope(delta(post), d)
        # A zero-length secant has no effect on either endpoint prediction.
        interior = (p[:, 0] > 0) & (p[:, 0] < 127 * 64)
        return sec, np.where((d == 0) & interior, 1 / 64, sec)

    return (
        delta(x),
        gates(u[:, :, 2:10], pre),
        gates(u[:, :, 10:], p2),
        delta(u[:, :, 1]),
        x,
        u[:, :, 0],
    )


def decompose(o, e, w1, w2, w3, convention):
    vals = {}
    for m in range(8):
        dx = (o if m & 1 else e)[0]
        g1 = (o if m & 2 else e)[1][convention]
        g2 = (o if m & 4 else e)[2][convention]
        vals[m] = (((dx @ w1.T) * g1) @ w2.T * g2) @ w3
    assert np.allclose(vals[7], o[3]) and np.allclose(vals[0], e[3])
    credited = np.zeros((len(o[0]), 3))
    for order in permutations(range(3)):
        m = 0
        for j in order:
            k = m | 1 << j
            credited[:, j] += (vals[k] - vals[m]) / 6
            m = k
    assert np.allclose(credited.sum(1), o[3] - e[3])
    return credited


def equalize(x):
    # Closest common move increment for the two contexts, then fit each pair
    # around its native midpoint under the network's physical byte bounds.
    increments = x[:, :, 1] - x[:, :, 0]
    common = np.rint(increments.mean(1))
    first = np.rint((x[:, :, 0] + x[:, :, 1] - common[:, None, :]) / 2)
    first = np.clip(
        first,
        np.maximum(0, -common)[:, None, :],
        np.minimum(127, 127 - common)[:, None, :],
    )
    adjusted = np.stack([first, first + common[:, None, :]], axis=2)
    assert np.all((adjusted >= 0) & (adjusted <= 127))
    assert np.all(
        (adjusted[:, 0, 1] - adjusted[:, 0, 0])
        == (adjusted[:, 1, 1] - adjusted[:, 1, 0])
    )
    unchanged = increments[:, 0] == increments[:, 1]
    assert not np.any((adjusted != x) & unchanged[:, None, None, :])
    return adjusted


def patch_interactions(o, e, weights, rank):
    b1, w1, b2, w2, w3 = weights
    x = np.stack([o[4], e[4]], axis=1)
    additive = equalize(x)
    base = additive @ w1.T + b1
    delta = x - additive
    psqt = np.stack([o[5], e[5]], axis=1)
    options = {"equalized": [], "intact": np.arange(2048)}
    random = np.random.default_rng(9873).permutation(2048)
    for k in [64, 256, 1024]:
        options[f"jan{k}"] = rank[:k]
        options[f"random{k}"] = random[:k]
    metrics = {}
    true = o[3] - e[3]
    for name, idx in options.items():
        p1 = base + (delta[:, :, :, idx] @ w1[:, idx].T if len(idx) else 0)
        h1 = np.clip(np.floor(p1 / 64), 0, 127)
        h2 = np.clip(np.floor((h1 @ w2.T + b2) / 64), 0, 127)
        y = h2 @ w3
        margins = (y + psqt)[:, :, 1] - (y + psqt)[:, :, 0]
        interaction = (y[:, 0, 1] - y[:, 0, 0]) - (y[:, 1, 1] - y[:, 1, 0])
        if name == "intact":
            assert np.array_equal(interaction, true)
        both = (margins[:, 0] > 0) & (margins[:, 1] < 0)
        metrics[name] = [
            len(x),
            float(interaction @ true),
            float(true @ true),
            float(np.sum((true - interaction) ** 2)),
            int(both.sum()),
        ]
    return metrics


def patch_summary(records):
    result = {}
    for k in records[0]:
        n, alignment, power, error, both = np.sum([r[k] for r in records], axis=0)
        result[k] = {
            "n": int(n),
            "interaction_projection": alignment / power,
            "nmse": error / power,
            "both_rate": both / n,
        }
    return result


def summarize(v):
    interaction = v.sum(1)
    power = interaction @ interaction
    return {
        "n": len(v),
        "mean_credits": v.mean(0).tolist(),
        "share_of_interaction_projection": ((v.T @ interaction) / power).tolist()
        if power
        else None,
        "mean_absolute_credits": np.abs(v).mean(0).tolist(),
    }


def main(root, month):
    directory = root / "circuits"
    rows = load(directory / f"reversals-{month}.gz")
    out = {}
    pooled = [[], []]
    rankdata = json.loads((directory / "feature-routes.json").read_text())
    all_patches = []
    for stage in range(8):
        data = [r for r in rows if r["stack"] == stage]
        if not data:
            continue
        b1, w1 = ft_weights(root, stage)
        b2, w2, _, w3 = stack_weights(root, stage)
        measures = [[], []]
        patches = []
        for start in range(0, len(data), 256):
            rs = data[start : start + 256]
            ed = [
                dict(r, children=r["edited_children"], units=r["edited_units"])
                for r in rs
            ]
            o, e = [
                endpoints(sample, root, (b1, w1, b2, w2, w3)) for sample in (rs, ed)
            ]
            patches.append(
                patch_interactions(
                    o,
                    e,
                    (b1, w1, b2, w2, w3),
                    np.array(rankdata[str(stage)]["ranking"]),
                )
            )
            for i in range(2):
                measures[i].append(decompose(o, e, w1, w2, w3, i))
        measures = [np.concatenate(v) for v in measures]
        out[stage] = [summarize(v) for v in measures]
        out[f"patch_{stage}"] = patch_summary(patches)
        all_patches.extend(patches)
        for dst, src in zip(pooled, measures):
            dst.append(src)
    out["patch_all"] = patch_summary(all_patches)
    out["all"] = [summarize(np.concatenate(v)) for v in pooled]
    (directory / f"interaction-layers-{month}.json").write_text(
        json.dumps(out, allow_nan=False)
    )
    print(out["all"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--month", default="2013-02")
    args = p.parse_args()
    main(args.root, args.month)
