"""Partition signed native margins over the eight coordinates on the residual and transfer sets."""

import csv
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from math import comb
from pathlib import Path

import numpy as np
import validate as v
from bottleneck import read_batch, stack_weights
from matched import pairs, summarize
from panel import TAGS, samples
from summarize import write


def run(root, out):
    root = root.resolve()
    needed = samples(root)
    work = root / "atlas/horizon/kings"
    residual = {r["game"] for r in csv.DictReader((out / "residual-28.csv").open())}
    targets = {}
    for key, groups in needed.items():
        game = next(
            (gid for name, gid, *_ in groups if name == "fresh" and gid in residual),
            None,
        )
        ext = key[:2] in ((4, 1), (5, 1)) and any(g[0] == "extension" for g in groups)
        if game or ext:
            targets[key] = (game, ext, groups[0][3])
    guards = set()
    for a in (4, 5):
        for f in work.glob(f"pairs-{a}-1-*.third"):
            for *coords, m in np.loadtxt(f, dtype="i8", ndmin=2):
                if int(m) >> TAGS.index("king_guard_own_pawn_next") & 1:
                    guards.add((a, 1, *map(int, coords)))
    matched = pairs(root)
    for coords in matched:
        for a in (4, 5):
            key = (a, 1, *coords)
            h, l = [v.state(a, 1, *coords[:3], t) for t in coords[-2:]]
            targets.setdefault(key, (None, False, int(h["n"]) - int(l["n"])))
    matched_rows = []
    keys = sorted(targets)
    states = sorted({(*key[:5], x) for key in keys for x in key[-2:]})
    data = {}
    with ThreadPoolExecutor(4) as pool:
        for r in pool.map(lambda b: read_batch(root, b), batched(states, 512)):
            data.update(r)
    bias, coef, offset, last = stack_weights(root, 0)
    mask = np.arange(256)
    bits = (mask[:, None] >> np.arange(8) & 1).astype(bool)
    scale = lambda x: np.trunc(x / 16).astype("i4")
    played = []
    agg = defaultdict(list)
    for key in keys:
        game, ext, native = targets[key]
        h, l = [np.asarray(data[(*key[:5], x)], dtype="i4") for x in key[-2:]]
        mix = np.where(bits, h[2:10], l[2:10])
        hidden = ((mix @ coef.T + bias) >> 6).clip(0, 127)
        assert np.array_equal(hidden[[0, 255]], np.stack([l[10:], h[10:]]))
        pos = hidden @ last + offset
        left = scale(h[0] + pos) - scale(l[0] + pos[0])
        right = scale(h[0] + pos[-1]) - scale(l[0] + pos[::-1])
        margin = (left + right) / 2
        credit = (2 + np.sign(left) + np.sign(right)) / 4
        assert left[-1] == right[-1] == native
        phi = []
        for u in range(8):
            m = mask[~bits[:, u]]
            dif = margin[m | 1 << u] - margin[m]
            weight = np.array([1 / (8 * comb(7, int(v).bit_count())) for v in m])
            phi.append(dif @ weight)
        assert np.isclose(margin[0] + sum(phi), native)
        values = {
            label: credit[m]
            for label, m in [
                ("direct", 0),
                ("keep67", 192),
                ("drop67", 63),
                ("native", 255),
            ]
        }
        values.update({f"without_{u}": credit[255 ^ (1 << u)] for u in range(8)})
        values["direct_margin"] = margin[0]
        values.update({f"phi{u}": v for u, v in enumerate(phi)})
        if key[2:] in matched and key[:2] in ((4, 1), (5, 1)):
            w, k, y, h, l = key[2:]
            matched_rows.append(
                dict(
                    white_man=key[0],
                    wk=w,
                    bk=k,
                    pawn=y,
                    source=matched[key[2:]],
                    high=h,
                    low=l,
                    margin=native,
                    **values,
                )
            )
        if game:
            played.append(dict(game=game, margin=native, **values))
        if ext:
            agg[key[0], key in guards].append(values)
    summarize(matched_rows, out)
    write(out / "played-paths.csv", sorted(played, key=lambda r: r["game"]))
    write(
        out / "escort-paths.csv",
        [
            dict(
                white_man=a,
                guard_erases=int(g),
                n=len(records),
                **{k: np.mean([r[k] for r in records]) for k in records[0]},
            )
            for (a, g), records in sorted(agg.items())
        ],
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
