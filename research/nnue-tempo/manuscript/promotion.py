"""Discover sparse promotion routes on one set of kings and test fresh alternatives."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import validate as v
from network import Network, credit
from promotion_geometry import relation
from promotion_trial import CHANNELS
from study import check_native
from summarize import write

from cohorts import kings
from effects import batches, ci, delta, ft_flow, patch


def load(root, sample):
    result = defaultdict(list)
    for a in range(2, 6):
        p = np.load(root / "manuscript" / f"{sample}-{a}-1.npz")
        rows = p["rows"][p["mask"]]
        for f in range(8):
            r = rows[(rows[:, 2] ^ 7) == f + 8]
            for chunk in batches(r):
                result[f].append((a, chunk))
    return result


def best_channels(net, discovery):
    ranked = {}
    controls = {}
    for f, parts in discovery.items():
        flow = np.zeros(2048)
        activity = np.zeros(2048)
        for a, r in parts:
            raw, p, _ = net.encode(r, a, 1)
            t = net.forward(raw, p)
            x = raw.clip(0, 127)
            flow += ft_flow(net, x, t).sum(0)
            activity += abs(delta(x)).sum(0)
        ranked[f] = np.argsort(-flow, kind="stable")[:8].tolist()
        used = set(ranked[f])
        matched = []
        for u in ranked[f][:4]:
            pool = np.array(
                [v for v in range(2048) if v // 1024 == u // 1024 and v not in used]
            )
            v = int(pool[np.argmin(abs(activity[pool] - activity[u]))])
            matched.append(v)
            used.add(v)
        controls[f] = matched
    return ranked, controls


def screen(net, out):
    old = out.parents[1] / "atlas/horizon/kings/routes/results"
    with (old / "promotion-coverage.csv").open() as s:
        obs = {
            (int(z["ft"]), int(z["white"]), z["coverage"]): float(z["mean_byte"])
            for z in csv.DictReader(s)
        }
    with (old / "promotion-weights.csv").open() as s:
        weights = {
            (int(z["ft"]), int(z["white"])): int(z["positive_buckets"])
            for z in csv.DictReader(s)
        }
    W = net.w[:, 1, 8:56, :]
    lead = W.max(1) - np.partition(W, -2, axis=1)[:, -2]
    rows = []
    for f in range(8):
        wins = (W.argmax(1) == f) & (lead > 0)
        for i in np.flatnonzero(wins.sum(0) >= 24):
            u = 1024 + int(i)
            all32 = wins[:, i].all()
            for a in range(2, 6):
                diff = (
                    (obs[u, a, "off_ray"] - obs[u, a, "clear_ray"]) if all32 else None
                )
                rows.append(
                    {
                        "file": chr(97 + f),
                        "ft": u,
                        "white": a,
                        "unique_buckets": int(wins[:, i].sum()),
                        "off_minus_clear": diff,
                        "positive_weight_buckets": weights.get((u, a))
                        if all32
                        else None,
                    }
                )
    write(out / "screen.csv", rows)


def evaluate(net, data, ranked, controls, out):
    report = []
    errors = []
    semantics = []
    for f, parts in sorted(data.items()):
        ids = ranked[f]
        rng = np.random.default_rng(1920 + f)
        control = rng.choice(np.setdiff1d(np.arange(2048), ids), 8, replace=False)
        sets = {
            "first": ids[:1],
            "next_three": ids[1:4],
            "first_four": ids[:4],
            "first_eight": ids,
            "random_eight": control,
            "other_file_four": ranked[(f + 1) % 8][:4],
        }
        sets["activity_four"] = controls[f]
        if f == 1:
            sets["near_miss"] = [1869]
        if 8 + f in CHANNELS:
            sets["previous"] = [CHANNELS[8 + f]]
        vals = defaultdict(list)
        geo = defaultdict(list)
        error = defaultdict(list)
        for a, r in parts:
            raw, p, _ = net.encode(r, a, 1)
            x = raw.clip(0, 127)
            t = net.forward(raw, p)
            check_native(a, 1, r, t)
            m = delta(t["score"])
            native = credit(m)
            key = kings(r)
            groups = np.stack(
                [relation(a, r[:, j], (f ^ 7), r[:, 0], r[:, 1]) for j in (3, 4)],
                axis=1,
            )
            for g, matches in [
                ("all", np.ones(len(r), bool)),
                ("high_off_low_clear", (groups[:, 0] == 2) & (groups[:, 1] == 4)),
                ("same_coverage", groups[:, 0] == groups[:, 1]),
                (
                    "other",
                    (groups[:, 0] != groups[:, 1])
                    & ~((groups[:, 0] == 2) & (groups[:, 1] == 4)),
                ),
            ]:
                if matches.any():
                    for piece in (a, 0):
                        error[piece, g].extend(m[matches].tolist())
            for u in ids[:4]:
                for g in (2, 3, 4):
                    geo[u, g].extend(x[:, :, u][groups == g].tolist())
            for name, units in sets.items():
                without, effect = patch(net, x, t, units)
                vals[name].append(np.c_[key, native, without, effect])
        for (piece, g), ms in sorted(error.items()):
            m = np.array(ms)
            errors.append(
                {
                    "file": chr(97 + f),
                    "white": piece,
                    "relation": g,
                    "n": len(m),
                    "correct": int((m > 0).sum()),
                    "ties": int((m == 0).sum()),
                    "wrong": int((m < 0).sum()),
                    "accuracy": credit(m).mean(),
                }
            )
        for name, chunks in vals.items():
            z = np.concatenate(chunks)
            dif = z[:, 1] - z[:, 2]
            lo, hi = ci(z[:, 0], dif)
            report.append(
                {
                    "file": chr(97 + f),
                    "intervention": name,
                    "ft": " ".join(map(str, sets[name])),
                    "n": len(z),
                    "king_orbits": len(set(z[:, 0])),
                    "native": z[:, 1].mean(),
                    "patched": z[:, 2].mean(),
                    "drop": dif.mean(),
                    "lower": lo,
                    "upper": hi,
                    "mean_margin_removed": z[:, 3].mean(),
                }
            )
        for (u, g), values in geo.items():
            if values:
                semantics.append(
                    {
                        "file": chr(97 + f),
                        "ft": u,
                        "geometry": {2: "off", 3: "blocked", 4: "clear"}[g],
                        "endpoints": len(values),
                        "mean_byte": np.mean(values),
                    }
                )
    write(out / "promotion.csv", report)
    write(out / "promotion-errors.csv", errors)
    write(out / "promotion-geometry.csv", semantics)


if __name__ == "__main__":
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    v.init(root)
    net = Network(root)
    screen(net, out)
    ranked, controls = best_channels(net, load(root, "discovery"))
    evaluate(net, load(root, "validation"), ranked, controls, out)
