"""Test all king-bucket-invariant pawn-square weight maxima on legal positions."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import validate as v
from interventions import primary
from network import Network, relocate
from summarize import write


def main(root, out):
    v.init(root)
    net = Network(root)
    V = net.w[:, 1, 8:56, :].astype("i4")
    arg = V.argmax(1)
    gaps = V.max(1) - np.partition(V, -2, axis=1)[:, -2]
    ids = np.flatnonzero((arg == arg[:1]).all(0) & (gaps.min(0) > 0))
    target = arg[0, ids] + 8
    weights = net.w[:, :, :, ids].astype("i4")
    rows = []
    for a in (2, 3, 4, 5):
        v.state(a, 1, 0, 0, 0, 0)
        ct = np.zeros((len(ids), 5), dtype="i8")
        for w in range(64):
            if w % 8 >= 4:
                continue
            ks, ys, xs = np.where(v.maps[a, 1][w]["v"] > 0)
            flip = 7
            ok = w ^ flip
            bucket = 31 - 4 * (ok // 8) - (ok % 8 - 4)
            for start in range(0, len(xs), 12000):
                k, y, x = [m[start : start + 12000] ^ flip for m in (ks, ys, xs)]
                raw = (
                    net.bias[ids]
                    + weights[bucket, 10, ok]
                    + weights[bucket, 10, k]
                    + weights[bucket, 1, y]
                    + weights[bucket, 2 * (a - 1), x]
                )
                active = raw > 0
                hit = y[:, None] == target[None, :]
                ct[:, 0] += len(k)
                ct[:, 1] += hit.sum(0)
                ct[:, 2] += active.sum(0)
                ct[:, 3] += (hit & active).sum(0)
                ct[:, 4] += (hit & (raw >= 127)).sum(0)
        for n, i in enumerate(ids):
            size, targets, fires, tp, sat = map(int, ct[n])
            rows.append(
                {
                    "white": a,
                    "ft": int(i + 1024),
                    "oriented_square": int(target[n]),
                    "min_weight_gap": int(gaps[:, i].min()),
                    "boards": size,
                    "target": targets,
                    "fires": fires,
                    "true_fires": tp,
                    "saturated_on_target": sat,
                    "precision": tp / fires if fires else None,
                    "recall": tp / targets,
                }
            )
    write(out / "pawn-square-channels.csv", rows)
    # On common legal moves compare transfer and deletion of each penultimate-rank column.
    orig = primary(root / "atlas/horizon/kings/routes")
    orig, moved = relocate(orig, 2)
    rows = []
    for a in (4, 5):
        for oriented in (10, 13):
            keep = (moved[:, 2] ^ 7) == oriented
            totals = defaultdict(lambda: np.zeros(4))
            for start in range(0, keep.sum(), 192):
                pair = [
                    net.encode(rr[keep][start : start + 192], a, 1)
                    for rr in (orig, moved)
                ]
                t, u = [net.forward(raw, p) for raw, p, _ in pair]
                delta = pair[1][0].clip(0, 127) - pair[0][0].clip(0, 127)
                for label, indices in [
                    ("1424", [1424]),
                    ("1196", [1196]),
                    ("both", [1424, 1196]),
                ]:
                    dz = delta[:, :, indices].astype("f8") @ net.w1[:, indices].T
                    for variant, z in [
                        ("before", t["z"]),
                        ("after", u["z"]),
                        ("transfer", t["z"] + dz),
                        ("freeze", u["z"] - dz),
                    ]:
                        H = net.upper(z, t["ps"])["h1"]
                        totals[label, variant] += abs(H[:, 0] - H[:, 1])[
                            :, [2, 4, 6, 7]
                        ].sum(0)
            for (label, variant), total in totals.items():
                rows.append(
                    dict(
                        piece=a,
                        oriented_pawn=oriented,
                        n=int(keep.sum()),
                        channels=label,
                        variant=variant,
                        **dict(
                            zip(["gap2", "gap4", "gap6", "gap7"], total / keep.sum())
                        ),
                    )
                )
    write(out / "pawn-channel-exchange.csv", rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
