"""Compare the same hidden readout under competing legal Black actions."""

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network, credit
from study import check_native
from summarize import write

from cohorts import kings, orbit
from effects import batches, ci, delta
from tactical import drop_h2


def actions(a, w, k, y, x):
    q = v.board(a, 4, w, k, y, x)
    values = []
    for m in list(q.legal_moves):
        direct = m.to_square == x
        skewer = (
            m.from_square == y
            and bool(chess.between(m.to_square, x) & chess.BB_SQUARES[w])
            and q.gives_check(m)
        )
        q.push(m)
        values.append((-v.sign(v.tb.probe_wdl(q)), direct, skewer))
        q.pop()
    ordinary = max(z[0] for z in values)
    rest = [max((z[0] for z in values if not z[i]), default=ordinary) for i in (1, 2)]
    present = [any(z[i] for z in values) for i in (1, 2)]
    return ordinary, np.array(rest), np.array(present)


def decision(job):
    a, w, k, y, h, l = job
    hb, hr, ha = actions(a, w, k, y, h)
    lb, lr, la = actions(a, w, k, y, l)
    assert hb > lb
    return np.r_[hr <= lr, ha, la]


def labels(flags):
    direct, skewer, hc, hs, lc, ls = flags.T
    return {
        "all": np.ones(len(flags), bool),
        "direct_capture": direct & ~skewer,
        "skewer": skewer & ~direct,
        "skewer_quiet": skewer & ~direct & ~hc & ~lc,
        "quiet_neither": ~hc & ~lc & ~hs & ~ls,
        "both_deletions": direct & skewer,
        "other_or_multiple": ~direct & ~skewer & (hc | lc | hs | ls),
    }


def readout(root, out):
    v.init(root)
    held = np.random.default_rng(96164).permutation(
        sorted({orbit(w, k) for w in range(64) for k in range(64)})
    )[::2]
    net = Network(root)
    report = []
    with ProcessPoolExecutor(
        8, mp_context=get_context("fork"), initializer=v.init, initargs=(root,)
    ) as pool:
        for a in range(2, 6):
            rows = np.loadtxt(
                root / "manuscript/rewriting" / f"open-{a}.txt", dtype="i8"
            )
            available = rows[np.isin(kings(rows), held)]
            rng = np.random.default_rng(2334 + a)
            r = available[
                rng.choice(len(available), min(18000, len(available)), replace=False)
            ]
            flags = np.array(
                list(
                    pool.map(
                        decision, [(a, *rr[:5]) for rr in r.tolist()], chunksize=120
                    )
                ),
                bool,
            )
            results = []
            for batch in batches(r):
                raw, ps, _ = net.encode(batch, a, 4)
                t = net.forward(raw, ps)
                check_native(a, 4, batch, t)
                margin = delta(t["score"])
                patches = [drop_h2(net, t, j) for j in (2, 6, 13)]
                h2 = delta(t["h2"])[:, 2] * net.out[2] / 16
                results.append(
                    np.c_[
                        kings(batch),
                        margin,
                        credit(margin),
                        *patches,
                        h2,
                        (t["h2"][:, :, 2] == 0).all(1),
                    ]
                )
            z = np.concatenate(results)
            for label, mask in labels(flags).items():
                for band, use in [
                    ("all", mask),
                    (
                        "abs_native_1_31",
                        mask & (abs(z[:, 1]) >= 1) & (abs(z[:, 1]) <= 31),
                    ),
                ]:
                    selected = z[use]
                    if not len(selected):
                        continue
                    for j, unit in enumerate((2, 6, 13)):
                        native, without = selected[:, 2], selected[:, 3 + j]
                        lo, hi = ci(selected[:, 0].astype(int), native - without)
                        report.append(
                            {
                                "white": a,
                                "stratum": label,
                                "margin_band": band,
                                "unit": unit,
                                "eligible": len(available),
                                "tested": len(r),
                                "n": len(selected),
                                "king_orbits": len(set(selected[:, 0])),
                                "native": native.mean(),
                                "without": without.mean(),
                                "lower": lo,
                                "upper": hi,
                                "native_helped": int((native > without).sum()),
                                "native_harmed": int((native < without).sum()),
                                "h2_2_mean_term": selected[:, 6].mean(),
                                "h2_2_both_floor": selected[:, 7].mean(),
                            }
                        )
    write(out / "prediction-skewer-controls.csv", report)


if __name__ == "__main__":
    readout(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
