"""Separate a king-conditioned piece identity offset from its square-dependent residual."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from network import Network, credit, retain
from study import panels
from summarize import write


def primary(work):
    p = np.load(work / "4-1.npz")
    mask = (p["4"][:, 0] & p["5"][:, 0]) >> 22 & 1
    r = p["rows"][mask.astype(bool)]
    assert len(r) == 2830
    return r


def endpoint_gap(z):
    h = np.floor(z / 64).clip(0, 127)
    return abs(h[:, 0] - h[:, 1])


def main(root, out):
    net = Network(root)
    work = root / "atlas/horizon/kings/routes"
    interventions = []
    gates = []
    features = []
    for panel, a, b, rank, rule, rows in panels(work):
        if not (
            (panel == "4-1" and rule == "escort")
            or (panel in ("3-4", "4-4") and rule == "check_from_far")
            or (panel == "4-2" and rule == "check_and_attack_man")
        ) or not len(rows):
            continue
        a = a[0]
        name = {"panel": panel, "rule": rule, "n": len(rows)}
        totals = defaultdict(lambda: np.zeros(12))
        gt = np.zeros((8, 6))
        select = np.zeros((8, 2048))
        for j in range(0, len(rows), 192):
            r = rows[j : j + 192]
            base, p, buckets = net.encode(r, a, b)
            queen, _q, _ = net.encode(r, 5, b)
            offset = net.offset_change(buckets, a)
            specific = queen - base - offset
            half = np.arange(2048) < 1024
            designs = {
                "base": base,
                "identity_offset": base + offset,
                "square_residual": base + specific,
                "queen": queen,
                "black_king_view_only": np.where(half, queen, base),
                "white_king_view_only": np.where(~half, queen, base),
            }
            traces = {}
            for mode, x in designs.items():
                # A common direct term isolates transformations on the positional pathway.
                t = net.forward(x, p)
                traces[mode] = t
                delta = endpoint_gap(t["z"])
                res = [
                    *delta.sum(0),
                    retain(net, t, [6, 7]).sum(),
                    retain(net, t, [2]).sum(),
                    (t["pos"][:, 0] - t["pos"][:, 1]).sum() / 16,
                    credit(t["score"][:, 0] - t["score"][:, 1]).sum(),
                ]
                totals[mode] += res
            z1, z2 = traces["base"]["z"], traces["queen"]["z"]
            mid = lambda z: z.mean(1, keepdims=True)
            # Work in half-logit units. The native quantizer is defined on this grid.
            g0, g1 = endpoint_gap(z1), endpoint_gap(z2)
            mab = endpoint_gap(mid(z2) + z1 - mid(z1))
            mba = endpoint_gap(mid(z1) + z2 - mid(z2))
            dm = (mab - g0 + g1 - mba) / 2
            dd = (mba - g0 + g1 - mab) / 2
            assert np.array_equal(dm + dd, g1 - g0)
            for i, v in enumerate((g0, g1, mab, mba, dm, dd)):
                gt[:, i] += v.sum(0)
            # Exact affine attribution of the queen-minus-base change in the midpoint.
            change = (queen.clip(0, 127) - base.clip(0, 127)).mean(1)
            select += net.w1 * change.sum(0)[None, :] / 64
        for mode, total in totals.items():
            vals = total / len(rows)
            interventions.append(
                dict(
                    **name,
                    variant=mode,
                    **{f"gap{i}": vals[i] for i in range(8)},
                    keep67=vals[8],
                    only2=vals[9],
                    positional_margin=vals[10],
                    accuracy=vals[11],
                )
            )
        for i, x in enumerate(gt / len(rows)):
            gates.append(
                dict(
                    **name,
                    unit=i,
                    base_gap=x[0],
                    queen_gap=x[1],
                    queen_mid_base_delta=x[2],
                    base_mid_queen_delta=x[3],
                    midpoint_contribution=x[4],
                    increment_contribution=x[5],
                )
            )
        for unit, v in enumerate(select / len(rows)):
            for idx in np.argsort(-abs(v), kind="stable")[:12]:
                features.append(
                    dict(
                        **name,
                        unit=unit,
                        ft=idx,
                        side="black" if idx < 1024 else "white",
                        midpoint_shift=v[idx],
                        total_shift=sum(v),
                    )
                )
    write(out / "input-exchange.csv", interventions)
    write(out / "gate-exchange.csv", gates)
    write(out / "feature-columns.csv", features)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
