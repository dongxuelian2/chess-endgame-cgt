"""Resolve the exact feature interaction into lower and upper clipping thresholds."""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze import stack_weights
from feature_routes import ft_weights, get_features, slope
from spectra import load


def mixed(x):
    return (x[:, 0, 1] - x[:, 0, 0]) - (x[:, 1, 1] - x[:, 1, 0])


def batch(rows, root, stage):
    b1, w1 = ft_weights(root, stage)
    b2, w2, _, w3 = stack_weights(root, stage)
    edits = [dict(r, children=r["edited_children"]) for r in rows]
    x, z, raw = get_features(rows + edits, root, raw=True)

    def arrange(v):
        a, b = np.split(v, 2)
        return np.stack([a, b], axis=1).astype(np.int32)

    x, z, raw = map(arrange, (x, z, raw))
    assert np.array_equal(x @ w1.T + b1, z)
    assert np.all(mixed(raw) == 0)
    low = mixed(np.maximum(-raw, 0))
    high = -mixed(np.maximum(raw - 127, 0))
    mx = mixed(x)
    assert np.array_equal(mx, low + high)
    h = (z >> 6).clip(0, 127)
    p = h @ w2.T + b2
    h2 = (p >> 6).clip(0, 127)
    delta = lambda v: v[:, :, 1] - v[:, :, 0]
    g1 = slope(delta(h), delta(z))
    g2 = slope(delta(h2), delta(p)) * w3
    gain = np.zeros_like(mx, dtype=float)
    for i, j, prob in [(0, 0, 1 / 3), (1, 1, 1 / 3), (0, 1, 1 / 6), (1, 0, 1 / 6)]:
        gain += prob * ((g1[:, i] * (g2[:, j] @ w2)) @ w1)
    interaction = mixed(h2) @ w3
    learned = low * gain, high * gain
    credited = np.stack([v.sum(1) for v in learned], axis=1)
    return (
        len(x),
        interaction @ interaction.astype(float),
        credited.T @ interaction,
        np.bincount((mx != 0).sum(1), minlength=2049),
        np.abs(learned[0] + learned[1]).sum(0),
        np.count_nonzero(low),
        np.count_nonzero(high),
    )


def measure(s):
    n, power, projection, histogram, weight_sum, nlo, nhi = s
    order = np.argsort(-weight_sum)
    normalized = np.cumsum(weight_sum[order]) / weight_sum.sum()
    return {
        "n": int(n),
        "threshold_projection": (projection / power).tolist(),
        "crossing_fraction": [nlo / (2048 * n), nhi / (2048 * n)],
        "interacting_slots_q10_median_q90": np.searchsorted(
            np.cumsum(histogram), np.array([0.1, 0.5, 0.9]) * n
        ).tolist(),
        "slots_for_half_and_ninety_percent_abs_credit": (
            np.searchsorted(normalized, [0.5, 0.9]) + 1
        ).tolist(),
        "slot_abs_credit": (weight_sum / n).tolist(),
    }


def main(root, month):
    directory = root / "circuits"
    rows = load(directory / f"reversals-{month}.gz")
    ref = json.loads((directory / f"interaction-layers-{month}.json").read_text())
    out = {}
    totals = []
    for stage in range(8):
        data = [r for r in rows if r["stack"] == stage]
        if not data:
            continue
        chunks = [
            batch(data[i : i + 128], root, stage) for i in range(0, len(data), 128)
        ]
        sums = [np.sum(v, axis=0) for v in zip(*chunks)]
        out[stage] = measure(sums)
        assert np.isclose(
            sum(out[stage]["threshold_projection"]),
            ref[str(stage)][0]["share_of_interaction_projection"][0],
        )
        totals.append(sums)
    pooled = [np.sum(v, axis=0) for v in zip(*totals)]
    out["all"] = measure(pooled)
    (directory / f"thresholds-{month}.json").write_text(
        json.dumps(out, allow_nan=False)
    )
    print(month, {k: v for k, v in out["all"].items() if k != "slot_abs_credit"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--month", default="2013-02")
    args = p.parse_args()
    main(args.root, args.month)
