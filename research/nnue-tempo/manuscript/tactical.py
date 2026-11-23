"""Trace material substitutions and knight forks from both accumulators to the score."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import validate as v
from network import Network, credit, trunc
from study import check_native
from summarize import write

from cohorts import kings
from effects import batches, delta, ft_flow

FORK = [0, 1, 4]


def load(root, sample, a, b):
    data = np.load(root / "manuscript" / f"{sample}-{a}-{b}.npz")
    return data["rows"][data["mask"]]


def top(net, root, a, b):
    total = np.zeros(2048)
    for r in batches(load(root, "discovery", a, b)):
        raw, p, _ = net.encode(r, a, b)
        x = raw.clip(0, 127)
        t = net.forward(raw, p)
        if b == 2:
            total += ft_flow(net, x, t, FORK).sum(0)
        else:
            rq, pq, _ = net.encode(r, 5, b)
            tq = net.forward(rq, pq)
            dz = tq["z"] - t["z"]
            sec = np.divide(
                tq["h1"] - t["h1"], dz, where=dz != 0, out=np.zeros_like(dz)
            )
            term = (rq.clip(0, 127) - x) * ((sec * net.w2[2]) @ net.w1)
            assert np.allclose(term.sum(2), tq["z2"][:, :, 2] - t["z2"][:, :, 2])
            total -= term.sum((0, 1))
    return np.argsort(-total, kind="stable")[:8].tolist()


def drop_h2(net, t, i):
    change = delta(t["h2"])[:, i] * net.out[i]
    base = t["pos"] + t["ps"]
    left = trunc(base[:, 0] - change, 16) - t["score"][:, 1]
    right = t["score"][:, 0] - trunc(base[:, 1] + change, 16)
    return (credit(left) + credit(right)) / 2


def metrics(net, t):
    h = t["h2"][:, :, 2]
    m = t["score"]
    return np.c_[
        credit(delta(m)),
        drop_h2(net, t, 2),
        t["z2"][:, :, 2].mean(1) / 64,
        h.mean(1),
        (h == 0).mean(1),
        abs(delta(h)),
        delta(h) * net.out[2] / 16,
    ]


def replace_h1(net, t, h):
    z = h @ net.w2.T + net.b2
    h2 = (z >> 6).clip(0, 127)
    pos = h2 @ net.out + net.offset
    return dict(t, h1=h, z2=z, h2=h2, pos=pos, score=trunc(pos + t["ps"], 16))


def main(root, out):
    net = Network(root)
    v.init(root)
    summaries = []
    layers = []
    channels = []
    subs = []
    for a, b in [(3, 4), (4, 2)]:
        ids = top(net, root, a, b)
        validation = load(root, "validation", a, b)
        for sample, piece, rows in [("matched", p, validation) for p in (a, 5)] + [
            ("outside", p, load(root, "transfer", p, b)) for p in (a, 5)
        ]:
            sums = defaultdict(float)
            changes = defaultdict(float)
            neurons = defaultdict(float)
            for r in batches(rows):
                raw, ps, bs = net.encode(r, piece, b)
                x = raw.clip(0, 127)
                t = net.forward(raw, ps)
                check_native(piece, b, r, t)

                def add(name, tt, sums=sums):
                    sums[name] += metrics(net, tt).sum(0)

                add("native", t)
                for i in range(32):
                    h = t["h2"][:, :, i]
                    neurons["H2", i] += np.array(
                        [
                            h.mean(),
                            (h == 0).mean(),
                            (h == 127).mean(),
                            abs(delta(h)).mean(),
                            (delta(h) * net.out[i] / 16).mean(),
                            drop_h2(net, t, i).mean(),
                        ]
                    ) * len(r)
                for i in range(8):
                    h = t["h1"][:, :, i]
                    from network import retain

                    neurons["H1", i] += np.array(
                        [
                            h.mean(),
                            (h == 0).mean(),
                            (h == 127).mean(),
                            abs(delta(h)).mean(),
                            (h * net.w2[2, i] / 64).mean(),
                            retain(net, t, [j for j in range(8) if j != i]).mean(),
                        ]
                    ) * len(r)
                if b == 2:
                    for label, target in [
                        ("one", ids[:1]),
                        ("four", ids[:4]),
                        ("eight", ids),
                        (
                            "random_eight",
                            np.random.default_rng(138).choice(
                                np.setdiff1d(np.arange(2048), ids), 8, replace=False
                            ),
                        ),
                    ]:
                        dz = delta(x)[:, target] @ net.w1[:, target].T
                        for route, indices in [
                            ("all", list(range(8))),
                            ("fork014", FORK),
                            ("complement", [2, 3, 5, 6, 7]),
                        ]:
                            change = np.zeros_like(dz)
                            change[:, indices] = dz[:, indices]
                            alt = net.upper(
                                t["z"] + np.stack([-change, change], axis=1), ps
                            )
                            left = alt["score"][:, 0] - t["score"][:, 1]
                            right = t["score"][:, 0] - alt["score"][:, 1]
                            changes[label, route] += np.array(
                                [
                                    ((credit(left) + credit(right)) / 2).sum(),
                                    (delta(t["score"]) - (left + right) / 2).sum(),
                                ]
                            )
                if sample == "matched":
                    other = 5 if piece == a else a
                    rq, pq, _ = net.encode(r, other, b)
                    tq = net.forward(rq, pq)
                    donor = rq.clip(0, 127)
                    for label, target in [
                        ("black", slice(None, 1024)),
                        ("white", slice(1024, None)),
                        ("one", ids[:1]),
                        ("four", ids[:4]),
                        ("eight", ids),
                    ]:
                        xx = x.copy()
                        xx[:, :, target] = donor[:, :, target]
                        add(label, net.forward(xx, ps))
                    offset = net.offset_change(bs, a) * (1 if piece == a else -1)
                    for label, xx in [
                        ("identity_mean", raw + offset),
                        ("square_residual", rq - offset),
                    ]:
                        add(label, net.forward(xx, ps))
                    for label, units in [
                        ("h1_13", [1, 3]),
                        ("h1_7", [7]),
                        ("h1_0134", [0, 1, 3, 4]),
                        ("h1_all", list(range(8))),
                    ] + [(f"h1_unit_{i}", [i]) for i in range(8)]:
                        h = t["h1"].copy()
                        h[:, :, units] = tq["h1"][:, :, units]
                        add(label, replace_h1(net, t, h))
            n = len(rows)
            meta = {
                "sample": sample,
                "white": piece,
                "black": b,
                "n": n,
                "king_orbits": len(set(kings(rows))),
            }
            for name, z in sums.items():
                summaries.append(
                    dict(
                        **meta,
                        exchange=name,
                        **dict(
                            zip(
                                [
                                    "native_or_hybrid",
                                    "without_h2_2",
                                    "h2_2_mean_prebyte",
                                    "h2_2_mean",
                                    "h2_2_floor",
                                    "h2_2_abs_gap",
                                    "h2_2_signed_score",
                                ],
                                z / n,
                            )
                        ),
                    )
                )
            for (label, route), z in changes.items():
                subs.append(
                    dict(
                        **meta,
                        ft_set=label,
                        through=route,
                        patched=z[0] / n,
                        score_removed=z[1] / n,
                    )
                )
            for (layer, i), z in neurons.items():
                layers.append(
                    dict(
                        **meta,
                        layer=layer,
                        unit=i,
                        mean=z[0] / n,
                        floor=z[1] / n,
                        ceiling=z[2] / n,
                        abs_gap=z[3] / n,
                        mean_term=z[4] / n,
                        without=z[5] / n,
                    )
                )
        for rank, u in enumerate(ids, 1):
            channels.append(
                {
                    "white": a,
                    "black": b,
                    "rank": rank,
                    "ft": u,
                    "h1_0": net.w1[0, u],
                    "h1_1": net.w1[1, u],
                    "h1_3": net.w1[3, u],
                    "h1_4": net.w1[4, u],
                    "h1_7": net.w1[7, u],
                }
            )
    write(out / "tactical-switch.csv", summaries)
    write(out / "tactical-layers.csv", layers)
    write(out / "tactical-inputs.csv", channels)
    write(out / "fork-path.csv", subs)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
