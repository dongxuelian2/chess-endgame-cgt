"""Score paired bounds, selective predictions, and comparisons against static networks."""

import argparse
import csv
from pathlib import Path

import numpy as np
from summarize import ci, group, mat, score, write


def load(path):
    with (path / "panel.csv").open() as f:
        raw = list(csv.DictReader(f))
    panel = np.array([r.pop("panel") for r in raw])
    gids = [r.pop("gid") for r in raw]
    data = {key: np.array([int(r[key]) for r in raw]) for key in raw[0]}
    path_bits = {}
    for file in path.glob("*.out"):
        a, b = map(int, file.stem.split("-"))
        for row in np.loadtxt(file, dtype="i8", ndmin=2):
            path_bits[(a, b, *row[:4])] = row[4:].reshape(-1, 3, 2)
    refs = np.array(
        [
            [
                path_bits[tuple(r)]
                for r in zip(*[data[k] for k in ("a", "b", "wk", "bk", "y", dst)])
            ]
            for dst in ("x", "z")
        ]
    )
    return panel, gids, data, refs


def values(refs, scope):
    return ((refs >> scope) & 1).sum(-1).astype("i1") - 1


def main(root, out):
    rows = []
    conditional = []
    detailed = []
    materials = []
    coefs = np.loadtxt(root / "atlas/weights.txt", dtype=int)[:, 0]
    for run in ("all5", "games7"):
        panel, gid, data, refs = load(root / "atlas/horizon" / run)
        gr = np.array([group(w, k) for w, k in zip(data["wk"], data["bk"])])
        nn = score(data["nn"])
        ps = score(data["ps"])
        hce48 = score(
            sum(
                (((data["h"] >> j) & 1) - ((data["l"] >> j) & 1)) * c
                for j, c in enumerate(coefs)
            )
        )
        truth = np.stack((data["wh"], data["wl"]))[:, :, None]
        for s in range(16):
            v = values(refs, s)
            assert np.array_equal(v[..., 1], np.clip(0, v[..., 0], v[..., 2]))
            assert np.all(v[:, :, :, 0] <= truth) and np.all(v[:, :, :, 2] >= truth)
            assert np.all(np.diff(v[:, :, :, 0], axis=2) >= 0) and np.all(
                np.diff(v[:, :, :, 2], axis=2) <= 0
            )
            assert np.all(v[:, :, :, 0] <= v[:, :, :, 1]) and np.all(
                v[:, :, :, 1] <= v[:, :, :, 2]
            )
            for side in range(4):
                if s >> side & 1:
                    continue
                enriched = values(refs, s | (1 << side))
                assert np.all(enriched[:, :, :, 0] >= v[:, :, :, 0]) and np.all(
                    enriched[:, :, :, 2] <= v[:, :, :, 2]
                )
            cert = v[0, :, :, 0] > v[1, :, :, 2]
            point = v[0, :, :, 1] - v[1, :, :, 1]
            for name in np.unique(panel):
                take = panel == name
                for h in range(1, v.shape[2]):
                    if run == "games7" and h <= 5:
                        continue
                    d = point[take, h]
                    c = cert[take, h]
                    rows.append(
                        {
                            "panel": name,
                            "depth": h,
                            "mask": s,
                            "n": int(sum(take)),
                            "certified": int(sum(c)),
                            "point_right": int(sum(d > 0)),
                            "point_wrong": int(sum(d < 0)),
                            "neutral_accuracy": score(d).mean(),
                            "nnue": nn[take].mean(),
                            "nnue_tiebreak": np.where(
                                d == 0, nn[take], score(d)
                            ).mean(),
                        }
                    )
                    if s != 15:
                        continue
                    masks = [
                        ("all", take),
                        ("certified", take & cert[:, h]),
                        ("overlap", take & ~cert[:, h]),
                        ("neutral_tie", take & (point[:, h] == 0)),
                        ("neutral_wrong", take & (point[:, h] < 0)),
                        ("flags_identical", take & (data["h"] == data["l"])),
                    ]
                    for kind, m in masks:
                        if not m.any():
                            continue
                        advantage = nn[m] - hce48[m]
                        lo, hi = ci(advantage, gr[m])
                        conditional.append(
                            {
                                "panel": name,
                                "depth": h,
                                "condition": kind,
                                "n": int(m.sum()),
                                "sf141": nn[m].mean(),
                                "direct_psqt": ps[m].mean(),
                                "rules48": hce48[m].mean(),
                                "gap_lo": lo,
                                "gap_hi": hi,
                            }
                        )
                    if h == v.shape[2] - 1:
                        for a, b in sorted(set(zip(data["a"][take], data["b"][take]))):
                            m = take & (data["a"] == a) & (data["b"] == b)
                            materials.append(
                                {
                                    "panel": name,
                                    "depth": h,
                                    "material": mat(a, b),
                                    "n": int(m.sum()),
                                    "certified": int(cert[m, h].sum()),
                                    "nnue": nn[m].mean(),
                                    "rules48": hce48[m].mean(),
                                    "neutral_accuracy": score(point[m, h]).mean(),
                                }
                            )
                        for i in np.flatnonzero(take):
                            if name == "enumerated":
                                continue
                            detailed.append(
                                {
                                    "panel": name,
                                    "game": gid[i],
                                    "depth": h,
                                    "material": mat(data["a"][i], data["b"][i]),
                                    "wk": data["wk"][i],
                                    "bk": data["bk"][i],
                                    "y": data["y"][i],
                                    "better": data["x"][i],
                                    "worse": data["z"][i],
                                    "black_wdl_high": data["wh"][i],
                                    "black_wdl_low": data["wl"][i],
                                    "nnue_delta": data["nn"][i],
                                    "certified": int(cert[i, h]),
                                    "neutral_delta": point[i, h],
                                    "minimum_depth": next(
                                        (j for j in range(h + 1) if cert[i, j]), ""
                                    ),
                                }
                            )
    write(out / "horizons.csv", rows)
    write(out / "conditional.csv", conditional)
    write(out / "material.csv", materials)
    write(out / "game-comparisons.csv", detailed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    main(args.root, args.out)
