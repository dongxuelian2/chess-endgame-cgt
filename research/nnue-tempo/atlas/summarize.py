"""Aggregate exact finite counts, legal game witnesses and held-out policy comparisons."""

import argparse
import csv
import gzip
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np

from validate import DOMAIN, RULES, sign

MATS = list(itertools.product(range(1, 6), repeat=2))
SYMBOL = " PNBRQ"


def mat(a, b):
    return "K" + SYMBOL[a] + "vK" + SYMBOL[b]


def write(path, ds):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=ds[0], lineterminator="\n")
        w.writeheader()
        w.writerows(ds)


def mean(x):
    return float(np.mean(x)) if len(x) else None


def score(x):
    return (np.sign(x) + 1) * 0.5


def frac(x, n):
    return x / n if n else None


def group(w, b):
    return min(min(w ^ m, b ^ m) * 64 + max(w ^ m, b ^ m) for m in (0, 7, 56, 63))


def bits(arr):
    return (
        (
            np.asarray(arr, dtype="uint64")[:, None]
            >> np.arange(len(RULES), dtype="uint64")
        )
        & 1
    ).astype(float)


def ci(values, clusters, seed=3646):
    keys, inv = np.unique(clusters, return_inverse=True)
    if len(keys) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    tot = np.bincount(inv, weights=values)
    n = np.bincount(inv)
    res = []
    for _ in range(32):
        sel = rng.integers(len(keys), size=(64, len(keys)))
        res.extend(tot[sel].sum(1) / n[sel].sum(1))
    return tuple(np.quantile(res, [0.025, 0.975]))


def fit(z):
    w = np.zeros(z.shape[1])
    ridge = 0.01
    for _ in range(20):
        p = 1 / (1 + np.exp(np.clip(z @ w, -40, 40)))
        grad = ridge * w - z.T @ p / len(p)
        hess = (z.T * (p * (1 - p))) @ z / len(p) + ridge * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if max(abs(step)) < 1e-8:
            break
    return w


def compact(root, out):
    stats = []
    census = []
    over = defaultdict(int)
    witness = []
    base = root / "atlas"
    metrics = [
        "available",
        "required",
        "compulsory",
        "all_strict",
        "erased",
        "n",
        "h",
        "p",
        "d",
        "sole",
        "availability_right",
        "availability_wrong",
        "equal_outcome_changed",
        "all_equal",
        "king_pairs",
        "example",
        "hi",
        "lo",
        "source",
    ]
    for a, b in MATS:
        stem = f"{a}-{b}"
        m = mat(a, b)
        lines = (base / "metrics" / (stem + ".metrics")).read_text().splitlines()
        ns = list(map(int, lines[0].split()[1:]))
        v, win, loss, err, en, amb, pairs, nn, hh, pp, dd = ns
        c1, c2 = map(int, (base / (stem + ".reply-count")).read_text().split()[-2:])
        census.append(
            {
                "material": m,
                "positions": v,
                "black_win": win,
                "draw": v - win - loss,
                "black_loss": loss,
                "ambiguous_positions": amb,
                "min_errors_H": err,
                "min_errors_HN": en,
                "unequal_wdl_tied_H_pairs": pairs,
                "nnue_points": nn / 2,
                "stock_classical_points": hh / 2,
                "psqt_points": pp / 2,
                "deep_points": dd / 2,
                "reply_nodes": c1,
                "legal_replies": c2,
            }
        )
        for line in lines[1:]:
            f, tag, *rest = line.split()
            r = dict(zip(metrics, map(int, rest), strict=True))
            if r["example"] >= 0:
                idx, x, y, s = r["example"], r["hi"], r["lo"], r["source"]
                from validate import board

                p = board(a, b, idx >> 12, idx >> 6 & 63, idx & 63, s)
                p.turn = True
                moves = [chess.Move(s, x), chess.Move(s, y)]
                assert all(
                    mv in p.legal_moves
                    and not p.is_capture(mv)
                    and not p.gives_check(mv)
                    for mv in moves
                )
                witness.append(
                    {
                        "rule": tag,
                        "fold": int(f),
                        "material": m,
                        "parent": p.fen(),
                        "inferior": moves[0].uci(),
                        "superior": moves[1].uci(),
                        "count": r["erased"],
                    }
                )
            for k in ("example", "hi", "lo", "source"):
                del r[k]
            stats.append(dict(material=m, fold=int(f), rule=tag, **r))
        for line in (base / "metrics" / (stem + ".overlap")).read_text().splitlines():
            f, x, y, n = line.split()
            over[int(f), x, y] += int(n)
    write(out / "census.csv", census)
    write(out / "by-material.csv", stats)
    primary = []
    for t in RULES:
        for f in (0, 1):
            rs = [r for r in stats if r["fold"] == f and r["rule"] == t]
            o = {k: sum(r[k] for r in rs) for k in metrics[:-4]}
            n = o["erased"]
            na = o["availability_right"] + o["availability_wrong"]
            primary.append(
                {
                    "rule": t,
                    "fold": f,
                    "positions_requiring": o["required"],
                    "compulsory": o["compulsory"],
                    "position_exposure": o["available"],
                    "erased_pairs": n,
                    "material_families": sum(r["erased"] > 0 for r in rs),
                    "nnue": frac(o["n"], 2 * n),
                    "released_H": frac(o["h"], 2 * n),
                    "psqt": frac(o["p"], 2 * n),
                    "deep": frac(o["d"], 2 * n),
                    "exclusive": o["sole"],
                    "availability_oriented": frac(o["availability_right"], na),
                    "availability_disagreements": na,
                    "availability_changed_same_wdl": o["equal_outcome_changed"],
                }
            )
    write(out / "rule-tests.csv", primary)
    write(
        out / "overlap.csv",
        [
            {"fold": f, "first": x, "second": y, "pairs": k}
            for (f, x, y), k in sorted(over.items())
        ],
    )
    write(
        out / "witnesses.csv",
        [
            max(ds, key=lambda r: r["count"])
            for t in RULES
            if (ds := [r for r in witness if r["rule"] == t and r["fold"] == 1])
        ],
    )
    return primary


def read_panels(root):
    ds = defaultdict(list)
    for a, b in MATS:
        path = root / "atlas/metrics" / f"{a}-{b}.npz"
        if not path.exists():
            assert not path.with_suffix(".pairs").stat().st_size
            continue
        data = np.load(path)
        r = data["panel"]
        idx = r[:, 0]
        for k, val in {
            "fold": r[:, 1],
            "hh": r[:, 2],
            "ll": r[:, 3],
            "erased": data["erased"],
            "modern": data["s18"],
            "old": data["s12"],
            "nn": r[:, 6],
            "material": np.repeat(mat(a, b), len(r)),
            "groups": [group(int(i >> 12), int(i >> 6 & 63)) for i in idx],
        }.items():
            ds[k].append(val)
    return {k: np.concatenate(v) for k, v in ds.items()}


def game_panel(root):
    with gzip.open(root / "atlas/game-results.gz", "rt") as f:
        rows = list(map(json.loads, f))
    data = np.load(root / "atlas/game-scores.npz")
    p = defaultdict(list)
    for i, (month, gid, fen, m1, m2, a, b, opts) in enumerate(rows):
        (s1, h1), (s2, h2) = opts
        diff = sign(s1[-1]) - sign(s2[-1])
        if not diff or s1[1] != s2[1]:
            continue
        sg = sign(diff)
        hi, lo = (h1, h2) if sg > 0 else (h2, h1)
        erased = hi[1] & lo[1] & ~((hi[3] & ~lo[3]) | (hi[2] & ~lo[2])) & DOMAIN
        k = chess.Board(fen)
        for key, val in {
            "hh": hi[0],
            "ll": lo[0],
            "erased": erased,
            "modern": data["s18"][i] * sg,
            "old": data["s12"][i] * sg,
            "nn": (s1[2] - s2[2]) * sg,
            "groups": group(k.king(True), k.king(False)),
            "material": mat(a, b),
        }.items():
            p[key].append(val)
    return {k: np.array(v) for k, v in p.items()}, len(rows)


def attribution(scores):
    points = score(scores)
    shap = np.zeros((len(points), 4))
    for j in range(4):
        for s in range(16):
            if s >> j & 1:
                continue
            shap[:, j] += (points[:, s | 1 << j] - points[:, s]) / (
                4 * math.comb(3, s.bit_count())
            )
    assert np.allclose(shap.sum(1), points[:, 15] - points[:, 0])
    return shap


def summarize(root, out):
    compact(root, out)
    data = read_panels(root)
    played, nplayed = game_panel(root)
    train = data["fold"] == 0
    z = bits(data["hh"]) - bits(data["ll"])
    w = fit(z[train])
    write(
        out / "coefficients.csv",
        [
            {"rule": k, "coefficient": float(v), "univariate_black_direction": int(o)}
            for k, v, o in zip(RULES, w, np.sign(z[train].sum(0)))
        ],
    )
    performance = []
    details = []
    local = []
    for cohort, subset in [
        ("discovery", {k: v[train] for k, v in data.items()}),
        ("confirmation", {k: v[~train] for k, v in data.items()}),
        ("played", played),
    ]:
        X = bits(subset["hh"]) - bits(subset["ll"])
        g = subset["groups"]
        s18 = subset["modern"]
        shap = attribution(s18)
        er = bits(subset["erased"]).astype(bool)
        # A small baseline gives one unit to each available capture class or surviving promotion.
        minimal = X[
            :,
            [
                RULES.index(k)
                for k in ("king_takes_actor", "man_takes_actor", "surviving_promotion")
            ],
        ].sum(1)
        predictors = {
            "simple_tactics": minimal,
            "interpretable": X @ w,
            "sf12": subset["old"],
            "sf141": subset["nn"],
            "sf18": s18[:, 15],
            "sf18_without_threats": s18[:, 0],
        }
        for typ, mask in [
            ("all", np.ones(len(g), bool)),
            *[(s, subset["material"] == s) for s in np.unique(subset["material"])],
        ]:
            for k, pred in predictors.items():
                v = score(pred[mask])
                lo, hi = ci(v, g[mask]) if typ == "all" else (None, None)
                performance.append(
                    {
                        "panel": cohort,
                        "material": typ,
                        "policy": k,
                        "n": len(v),
                        "accuracy": mean(v),
                        "cluster_lo": lo,
                        "cluster_hi": hi,
                    }
                )
        for t in ["ALL", *RULES]:
            mask = np.ones(len(g), bool) if t == "ALL" else er[:, RULES.index(t)]
            n = int(mask.sum())
            if not n:
                continue
            ch = score(s18[mask, 15]) - score(s18[mask, 0])
            lo, hi = ci(ch, g[mask]) if cohort != "discovery" else (None, None)
            avg = shap[mask].mean(0)
            details.append(
                {
                    "panel": cohort,
                    "rule": t,
                    "n": n,
                    "groups": len(np.unique(g[mask])),
                    "sf12": mean(score(subset["old"][mask])),
                    "sf141": mean(score(subset["nn"][mask])),
                    "sf18_intact": mean(score(s18[mask, 15])),
                    "sf18_ablated": mean(score(s18[mask, 0])),
                    "channel_gain": mean(ch),
                    "cluster_lo": lo,
                    "cluster_hi": hi,
                    "black_man_attacks": avg[0],
                    "white_man_attacks": avg[1],
                    "hostile_king_contact": avg[2],
                    "friendly_guard": avg[3],
                    "rule_model": mean(score((X @ w)[mask])),
                }
            )
        for j, t in enumerate(RULES):
            active = X[:, j] != 0
            local.append(
                {
                    "panel": cohort,
                    "rule": t,
                    "changed": int(sum(active)),
                    "rule_accuracy": mean(score(X[active, j])),
                    "sf141": mean(score(subset["nn"][active])),
                    "intact18": mean(score(s18[active, 15])),
                }
            )
    write(out / "policies.csv", performance)
    write(out / "factorial.csv", details)
    write(out / "availability.csv", local)
    print("played", nplayed, "informative H ties", len(played["nn"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    summarize(a.root, a.out)
