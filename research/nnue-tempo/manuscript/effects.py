"""Endpoint accounting, paired donor tests and grouped uncertainty estimates."""

import numpy as np
from network import credit


def delta(a):
    return a[:, 0] - a[:, 1]


def secant(values, logits):
    n, d = delta(values), delta(logits)
    return np.divide(n, d, where=d != 0, out=np.zeros_like(d, dtype=float))


def ft_flow(net, x, t, units=None):
    """Allocate the positional margin in score units via both quantizers."""
    to_h1 = (secant(t["h2"], t["z2"]) * net.out) @ net.w2
    if units is not None:
        keep = np.zeros(8)
        keep[units] = 1
        to_h1 *= keep
    flow = delta(x) * ((to_h1 * secant(t["h1"], t["z"])) @ net.w1) / 16
    if units is None:
        assert np.allclose(flow.sum(1), delta(t["pos"]) / 16)
    return flow


def donor_margins(net, x, t, ids):
    change = delta(x)[:, ids] @ net.w1[:, ids].T
    scores = net.upper(t["z"] + np.stack([-change, change], axis=1), t["ps"])["score"]
    native = t["score"]
    return scores[:, 0] - native[:, 1], native[:, 0] - scores[:, 1]


def patch(net, x, t, ids):
    left, right = donor_margins(net, x, t, ids)
    return (credit(left) + credit(right)) / 2, delta(t["score"]) - (left + right) / 2


def ci(keys, effects):
    ids, inverse = np.unique(keys, return_inverse=True)
    mass = np.bincount(inverse)
    sums = np.bincount(inverse, weights=effects)
    draws = np.random.default_rng(61031).integers(len(ids), size=(2000, len(ids)))
    return np.percentile(sums[draws].sum(1) / mass[draws].sum(1), [2.5, 97.5])


def batches(rows, size=128):
    return (rows[i : i + size] for i in range(0, len(rows), size))
