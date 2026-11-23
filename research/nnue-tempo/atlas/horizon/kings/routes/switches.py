"""Exhaustive front-end switch tests and physical pawn relocations on the original geometries."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import validate as v
from interventions import primary
from network import Network, relocate, retain
from summarize import write


def add(groups, key, cols, mask):
    if mask.any():
        groups[key] += np.array([mask.sum(), *[col[mask].sum() for col in cols]])


def census(root, out, net):
    # All placements legal and nonchecking in both materials, one horizontal representative.
    units = np.array([333, 625, 961])
    weights = net.w[:, :, :, units].astype("i4")
    counts = defaultdict(lambda: np.zeros(6, dtype="i8"))
    for b in (1, 2, 3, 4, 5):
        for a in (4, 5):
            v.state(a, b, 0, 0, 0, 0)
        for w in range(64):
            if w % 8 >= 4:
                continue
            k, y, x = np.where(
                (v.maps[4, b][w]["v"] != 0) & (v.maps[5, b][w]["v"] != 0)
            )
            flip = 56 + 7 * (k % 8 < 4)
            ok = k ^ flip
            bucket = 31 - 4 * (ok // 8) - (ok % 8 - 4)
            bg = (
                net.bias[units]
                + weights[bucket, 10, k ^ flip]
                + weights[bucket, 10, w ^ flip]
                + weights[bucket, 2 * (b - 1), y ^ flip]
            )
            r = (bg + weights[bucket, 7, x ^ flip]).clip(0, 127)
            q = (bg + weights[bucket, 9, x ^ flip]).clip(0, 127)
            c1 = (r[:, 0] == 0) & (q[:, 0] > 0)
            c2 = (r[:, 1] > 0) & (q[:, 1] == 0)
            c3 = (r[:, 2] == 0) & (q[:, 2] == 127)
            cols = [c1, c2, c3, c1 & c2 & c3, (q[:, 2] - r[:, 2])]
            add(counts, (b, "all", 0), cols, np.ones(len(k), bool))
            if b == 1:
                for label, values in [
                    ("rank", 1 + y // 8),
                    ("file_from_edge", np.minimum(y % 8, 7 - y % 8)),
                    ("black_king_rank", 1 + k // 8),
                    ("promotion_distance", np.maximum(abs(k % 8 - y % 8), k // 8)),
                ]:
                    for g in np.unique(values):
                        add(counts, (b, label, int(g)), cols, values == g)
    write(
        out / "switch-census.csv",
        [
            {
                "black": b,
                "group": group,
                "value": value,
                "boards": n,
                "ft333_opens": s1,
                "ft625_closes": s2,
                "ft961_full_switch": s3,
                "all_three": all3,
                "mean_ft961_difference": d / n,
            }
            for (b, group, value), (n, s1, s2, s3, all3, d) in sorted(counts.items())
        ],
    )


def displace(root, out, net):
    # Each source, both kings, two destinations and pawn file remain fixed.
    original = primary(root / "atlas/horizon/kings/routes")
    data = []
    for rank in range(2, 8):
        _, r = relocate(original, rank)
        w, k, y, h, l, _ = r.T
        for a in (4, 5):
            st = v.maps[a, 1]
            vh = np.sign(st[w, k, y, h]["w"])
            vl = np.sign(st[w, k, y, l]["w"])
            rs = []
            for start in range(0, len(r), 256):
                chunk = r[start : start + 256]
                raw, p, _ = net.encode(chunk, a, 1)
                t = net.forward(raw, p)
                gap = abs(t["h1"][:, 0] - t["h1"][:, 1])
                rs.extend(gap[:, [2, 4, 6, 7]])
            means = np.mean(rs, axis=0) if rs else [None] * 4
            data.append(
                {
                    "white": a,
                    "rank": rank,
                    "legal_moves": len(r),
                    "contexts": len(set(map(tuple, r[:, :3]))),
                    "original_order": int((vh > vl).sum()),
                    "equal_wdl": int((vh == vl).sum()),
                    "reversed": int((vh < vl).sum()),
                    "gap2": means[0],
                    "gap4": means[1],
                    "gap6": means[2],
                    "gap7": means[3],
                }
            )
    write(out / "pawn-relocations.csv", data)


def reroute(root, out, net):
    work = root / "atlas/horizon/kings/routes"
    selected = primary(work)
    d = np.load(work / "4-4.npz")
    far = ((d["4"][:, 2] & d["5"][:, 2]) >> 4 & 1).astype(bool)
    levels = {
        "original": [],
        "625": [625],
        "961": [961],
        "333": [333],
        "three": [333, 625, 961],
    }
    records = []
    rng = np.random.default_rng(92087)
    candidates = list(
        rng.choice(
            np.setdiff1d(np.arange(1024), [333, 625, 961]), size=(24, 3), replace=False
        )
    )
    for family, b, rows in [
        ("escort", 1, selected),
        ("distant_rook_check", 4, d["rows"][far]),
    ]:
        stats = defaultdict(lambda: np.zeros(7))
        for start in range(0, len(rows), 192):
            r = rows[start : start + 192]
            rr, pr, _ = net.encode(r, 4, b)
            rq, pq, _ = net.encode(r, 5, b)
            for a, src, dst, p in ((4, rr, rq, pr), (5, rq, rr, pq)):
                t = net.forward(src, p)
                z = t["z"]
                for key, indices in list(levels.items()) + [
                    (f"random{i}", l) for i, l in enumerate(candidates)
                ]:
                    shift = (
                        dst[:, :, indices].clip(0, 127)
                        - src[:, :, indices].clip(0, 127)
                    ).astype("f8") @ net.w1[:, indices].T
                    t = net.upper(z + shift, p)
                    gaps = abs(t["h1"][:, 0] - t["h1"][:, 1])
                    stats[a, key] += np.r_[
                        gaps[:, [2, 4, 6, 7]].sum(0),
                        retain(net, t, [6, 7]).sum(),
                        retain(net, t, [2]).sum(),
                        (np.sign(t["score"][:, 0] - t["score"][:, 1]) + 1).sum() / 2,
                    ]
        for (a, key), s in sorted(stats.items()):
            records.append(
                dict(
                    family=family,
                    recipient=a,
                    donated=key,
                    n=len(rows),
                    **dict(
                        zip(
                            [
                                "gap2",
                                "gap4",
                                "gap6",
                                "gap7",
                                "keep67",
                                "only2",
                                "native_order",
                            ],
                            s / len(rows),
                        )
                    ),
                )
            )
    write(out / "channel-interchange.csv", records)


if __name__ == "__main__":
    root, out = map(Path, sys.argv[1:])
    v.init(root)
    net = Network(root)
    census(root, out, net)
    displace(root, out, net)
    reroute(root, out, net)
