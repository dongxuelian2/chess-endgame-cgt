"""Test candidate square channels on naturally occurring promotion-dependent choices."""

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import validate as v
from network import Network, credit
from summarize import write

CHANNELS = {8: 2018, 10: 1424, 11: 1666, 12: 1268, 13: 1196}


def restrict(job):
    a, w, k, y, h, l, _s = job

    def node(x):
        q = v.board(a, 1, w, k, y, x)
        truth = v.sign(v.tb.probe_wdl(q))
        rest = []
        for m in list(q.legal_moves):
            if not m.promotion:
                q.push(m)
                rest.append(-v.sign(v.tb.probe_wdl(q)))
                q.pop()
        return truth, max(rest, default=truth)

    h0, h1 = node(h)
    l0, l1 = node(l)
    assert h0 > l0
    return h1 <= l1


def donors(i, net, x, z, p, native):
    dz = (x[:, 0, i] - x[:, 1, i])[:, None] * net.w1[:, i]
    logit = np.stack([z[:, 0] - dz, z[:, 1] + dz], axis=1)
    alt = net.upper(logit, p)["score"]
    left = alt[:, 0] - native[:, 1]
    right = native[:, 0] - alt[:, 1]
    delta = native[:, 0] - native[:, 1]
    return (credit(left) + credit(right)) / 2, delta - (left + right) / 2


def matched_activity(net, a, square, ft, rows):
    from summarize import group

    keys = np.array([group(int(w), int(k)) for w, k in rows[:, :2]])
    unique = np.unique(keys)
    folds = np.isin(keys, np.random.default_rng(19228).permutation(unique)[::2])
    records = []
    for fold in (False, True):
        train, test = rows[folds != fold], rows[folds == fold]
        if not len(train) or not len(test):
            continue
        scale = np.zeros(1024)
        for j in range(0, len(train), 192):
            raw, _, _ = net.encode(train[j : j + 192], a, 1)
            vals = raw[:, :, 1024:].clip(0, 127)
            scale += abs(vals[:, 0] - vals[:, 1]).sum(0)
        order = np.argsort(abs(scale - scale[ft - 1024]), kind="stable") + 1024
        selected = np.r_[ft, order[order != ft][:8]]
        size = np.zeros(len(selected))
        effect = size.copy()
        for j in range(0, len(test), 192):
            raw, p, _ = net.encode(test[j : j + 192], a, 1)
            t = net.forward(raw, p)
            x = raw.clip(0, 127)
            for idx, i in enumerate(selected):
                impact = donors(i, net, x, t["z"], p, t["score"])[1]
                effect[idx] += abs(impact).sum()
                size[idx] += abs(x[:, 0, i] - x[:, 1, i]).sum()
        for i, g, e in zip(selected, size, effect):
            records.append(
                {
                    "white": a,
                    "oriented_pawn": square,
                    "target": ft,
                    "channel": i,
                    "fold": int(fold),
                    "n": len(test),
                    "training_mean_gap": scale[i - 1024] / len(train),
                    "test_mean_gap": g / len(test),
                    "test_mean_absolute_score_change": e / len(test),
                }
            )
    return records


def main(root, out):
    root = root.resolve()
    work = root / "atlas/horizon/kings/routes"
    net = Network(root)
    records = []
    activity_controls = []
    rng = np.random.default_rng(72031)
    random_units = rng.choice(
        np.setdiff1d(np.arange(1024, 2048), list(CHANNELS.values())), 32, replace=False
    )
    with ProcessPoolExecutor(
        max_workers=8,
        mp_context=get_context("fork"),
        initializer=v.init,
        initargs=(root,),
    ) as pool:
        for a in (2, 3, 4, 5):
            rows = np.loadtxt(work / f"promote-{a}.txt", dtype="i8", ndmin=2)
            erase = np.array(
                list(
                    pool.map(restrict, [(a, *r) for r in rows.tolist()], chunksize=200)
                )
            )
            for square, ft in CHANNELS.items():
                mask = (rows[:, 2] ^ 7) == square
                for label, m in [("necessary", erase), ("other", ~erase)]:
                    group = rows[mask & m]
                    sums = np.zeros(7)
                    random_sum = np.zeros(len(random_units))
                    for start in range(0, len(group), 192):
                        r = group[start : start + 192]
                        raw, p, _ = net.encode(r, a, 1)
                        t = net.forward(raw, p)
                        z = t["z"]
                        x = raw.clip(0, 127)
                        native = t["score"]

                        without, contribution = donors(ft, net, x, z, p, native)
                        nativecredit = credit(native[:, 0] - native[:, 1])
                        sums += np.array(
                            [
                                nativecredit,
                                without,
                                contribution,
                                abs(contribution),
                                x[:, 0, ft] - x[:, 1, ft],
                                credit(x[:, 0, ft] - x[:, 1, ft]),
                                (x[:, 0, ft] != x[:, 1, ft]).astype(float),
                            ]
                        ).sum(1)
                        for j, rnd in enumerate(random_units):
                            random_sum[j] += abs(
                                donors(rnd, net, x, z, p, native)[1]
                            ).sum()
                    if label == "necessary" and len(group):
                        activity_controls.extend(
                            matched_activity(net, a, square, ft, group)
                        )
                    n = len(group)
                    records.append(
                        dict(
                            white=a,
                            ft=ft,
                            oriented_pawn=square,
                            selection=label,
                            n=n,
                            **dict(
                                zip(
                                    [
                                        "native",
                                        "without",
                                        "mean_margin_contribution",
                                        "mean_abs_margin_contribution",
                                        "mean_byte_delta",
                                        "channel_order",
                                        "changes",
                                    ],
                                    sums / n if n else [None] * 7,
                                )
                            ),
                            random_median_abs=float(np.median(random_sum) / n)
                            if n
                            else None,
                            random_max_abs=float(random_sum.max() / n) if n else None,
                        )
                    )
    write(out / "promotion-decisions.csv", records)
    write(out / "activity-controls.csv", activity_controls)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
