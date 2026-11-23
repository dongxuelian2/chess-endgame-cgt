"""Certify move witnesses and summarize the exhaustive extension plus external tests."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np

from summarize import (
    MATS,
    attribution,
    bits,
    ci,
    frac,
    game_panel,
    group,
    mat,
    score,
    write,
)
from validate import DOMAIN, RULES, board, init, probe, state


def audit_witness(r, a, b, idx, h, l, src, root, exclusive=False):
    t = RULES.index(r)
    wk, bk, y = idx >> 12, idx >> 6 & 63, idx & 63
    q = board(a, b, wk, bk, y, src)
    q.turn = True
    ds = []
    checks = []
    for dst in (h, l):
        mv = chess.Move(src, dst)
        assert mv in q.legal_moves and not q.is_capture(mv) and not q.gives_check(mv)
        q.push(mv)
        assert not q.has_legal_en_passant()
        mask, moves = probe(a, b, wk, bk, y, dst, True)
        checks.append(mask)
        en = state(a, b, wk, bk, y, dst)
        allowed = [v for u, v, tag in moves if not tag >> t & 1]
        best = max(v for u, v, tag in moves)
        ds.append(
            {
                "uci": mv.uci(),
                "wdl_black": best,
                "restricted_black": max(allowed),
                "classical": int(en["f"]),
                "nnue": int(en["n"]),
                "best_replies": [u for u, v, tag in moves if v == best],
                "deleted_best": [
                    u for u, v, tag in moves if v == best and tag >> t & 1
                ],
            }
        )
        q.pop()
    assert (
        ds[0]["wdl_black"] > ds[1]["wdl_black"]
        and ds[0]["restricted_black"] <= ds[1]["restricted_black"]
    )
    assert ds[0]["classical"] == ds[1]["classical"]
    if exclusive:
        h, l = checks
        assert (h[1] & l[1] & ~((h[3] & ~l[3]) | (h[2] & ~l[2])) & DOMAIN) == 1 << t
    else:
        assert ds[0]["nnue"] > ds[1]["nnue"]
    return {"rule": r, "material": mat(a, b), "fen": q.fen(), "moves": ds}


def finite(root, out):
    atlas = root / "atlas"
    rows = []
    coh = []
    drows = []
    best = {}
    over = defaultdict(int)
    allpairs = []
    for a, b in MATS:
        stem = f"{a}-{b}"
        family = mat(a, b)
        for name, base in [("pilot", atlas), ("remaining", atlas / "remaining")]:
            lines = (base / "metrics" / (stem + ".metrics")).read_text().splitlines()
            nodes, edges = map(
                int, (base / (stem + ".reply-count")).read_text().split()[-2:]
            )
            coh.append(
                {
                    "panel": name,
                    "material": family,
                    "unique_positions": nodes,
                    "legal_replies": edges,
                }
            )
            sample = base / "metrics" / (stem + ".pairs")
            if sample.stat().st_size:
                data = np.loadtxt(sample, dtype="i8", ndmin=2)
                allpairs.append((name, a, b, data))
            for line in lines[1:]:
                fold, tag, *values = line.split()
                v = list(map(int, values))
                idx, hi, lo, src = v[-4:]
                rows.append(
                    {
                        "panel": name,
                        "material": family,
                        "fold": int(fold),
                        "rule": tag,
                        "available": v[0],
                        "required": v[1],
                        "compulsory": v[2],
                        "all_strict_pairs": v[3],
                        "erased": v[4],
                        "nnue_credit": v[5] / 2,
                        "released_credit": v[6] / 2,
                        "psqt_credit": v[7] / 2,
                        "deep_credit": v[8] / 2,
                        "exclusive": v[9],
                        "availability_correct": v[10],
                        "availability_incorrect": v[11],
                        "equal_outcome_opportunities": v[12],
                        "all_equal_pairs": v[13],
                        "king_placements": v[14],
                    }
                )
                if idx >= 0 and (tag not in best or v[4] > best[tag][0]):
                    best[tag] = (v[4], a, b, idx, hi, lo, src)
            for line in (
                (base / "metrics" / (stem + ".overlap")).read_text().splitlines()
            ):
                _, t, u, n = line.split()
                over[t, u] += int(n)
            for line in (
                (base / "metrics" / (stem + ".decisions")).read_text().splitlines()
            ):
                policy, n, t, m, s, bet, eq, wo, changed, d = line.split()
                drows.append(
                    {
                        "panel": name,
                        "material": family,
                        "policy": policy,
                        "parents": int(n),
                        "ties": int(t),
                        "mixed": int(m),
                        "base_sum": float(s),
                        "better": int(bet),
                        "equal": int(eq),
                        "worse": int(wo),
                        "narrowed": int(changed),
                        "gain_sum": float(d),
                    }
                )
    write(out / "complete-rules.csv", rows)
    write(out / "reply-coverage.csv", coh)
    write(out / "complete-decisions.csv", drows)
    write(
        out / "complete-overlap.csv",
        [{"first": t, "second": u, "pairs": n} for (t, u), n in sorted(over.items())],
    )
    certs = [audit_witness(tag, *best[tag][1:], root) for tag in RULES if tag in best]
    (out / "move-certificates.json").write_text(json.dumps(certs, indent=1) + "\n")
    exclusive = {}
    for a, b in MATS:
        for base in (atlas, atlas / "remaining"):
            for line in (
                (base / "metrics" / f"{a}-{b}.exclusive").read_text().splitlines()
            ):
                rule, *v = line.split()
                if rule not in exclusive:
                    exclusive[rule] = audit_witness(
                        rule, a, b, *map(int, v), root, True
                    )
    assert set(exclusive) == {z["rule"] for z in rows if z["exclusive"]}
    (out / "exclusive-certificates.json").write_text(
        json.dumps(list(exclusive.values()), indent=1) + "\n"
    )
    return rows, drows, allpairs


def moves_played(root):
    with gzip.open(root / "atlas/game-decisions.gz", "rt") as f:
        rs = list(map(json.loads, f))
    ws, ori = np.loadtxt(root / "atlas/weights.txt", dtype=int).T
    names = [*RULES, "fitted", "simple_tactics", "native141", "oracle"]
    results = defaultdict(list)
    for r in rs:
        if not r["mixed"]:
            continue
        ys = np.array([z[1] for z in r["choices"]])
        n = np.array([z[2] for z in r["choices"]])
        X = bits([z[3] for z in r["choices"]])
        costs = np.column_stack((X * ori, X @ ws, X[:, [0, 1, 8]].sum(1), n, -ys))
        for j, policy in enumerate(names):
            picked = ys[costs[:, j] == costs[:, j].min()]
            gain = (int(picked.sum()) * len(ys) - int(ys.sum()) * len(picked)) / (
                len(ys) * len(picked)
            )
            results[policy].append((gain, r["group"], len(picked) < len(ys)))
    out = []
    for k, z in results.items():
        v, g, changed = np.array(z).T
        lo, hi = ci(v, g)
        out.append(
            {
                "policy": k,
                "parents": len(rs),
                "tied": sum(r["leaders"] > 1 for r in rs),
                "mixed": len(v),
                "better": int(sum(v > 0)),
                "equal": int(sum(v == 0)),
                "worse": int(sum(v < 0)),
                "narrowed": int(sum(changed)),
                "mean_all": v.sum() / len(rs),
                "mean_mixed": v.mean(),
                "cluster_lo": lo,
                "cluster_hi": hi,
            }
        )
    return out


def simultaneous(v_indicators, delta, clusters, rng):
    # Ratio-of-sums cluster influence function, centered at each observed ratio.
    keys, inv = np.unique(clusters, return_inverse=True)
    cnt = np.zeros((len(keys), len(RULES)))
    val = np.zeros_like(cnt)
    np.add.at(cnt, inv, v_indicators)
    np.add.at(val, inv, v_indicators * delta[:, None])
    n = cnt.sum(0)
    means = val.sum(0) / np.maximum(n, 1)
    influences = (val - cnt * means) / np.maximum(n, 1)
    se = np.sqrt((influences**2).sum(0) * len(keys) / (len(keys) - 1))
    use = (n >= 30) & ((cnt > 0).sum(0) >= 10) & (se > 0)
    quant = None
    if use.any():
        signs = rng.choice([-1.0, 1.0], (4999, len(keys)))
        bound = np.max(abs(signs @ (influences[:, use] / se[use])), axis=1)
        quant = float(np.quantile(bound, 0.95))
    return [
        (means[t] - quant * se[t], means[t] + quant * se[t]) if use[t] else (None, None)
        for t in range(len(RULES))
    ]


def subpanels(root, out, allpairs):
    w, _ = np.loadtxt(root / "atlas/weights.txt", dtype=int).T
    performance = []
    mechanisms = []
    twins = []
    effects = []
    parts = defaultdict(list)
    for name, a, b, data in allpairs:
        prefix = root / "atlas"
        if name == "remaining":
            prefix = prefix / "remaining"
        x = bits(data[:, 2]) - bits(data[:, 3])
        hard = (x == 0).all(1)
        for sub, mask in [
            ("all", np.ones(len(hard), bool)),
            ("identical_rules", hard),
            ("different_rules", ~hard),
        ]:
            if not mask.any():
                continue
            for k, val in [
                ("sf141", data[:, 6]),
                ("positional", data[:, 7]),
                ("fitted", x @ w),
            ]:
                twins.append(
                    {
                        "panel": name,
                        "material": mat(a, b),
                        "subset": sub,
                        "policy": k,
                        "n": int(mask.sum()),
                        "points": float(score(val[mask]).sum()),
                    }
                )
        if name != "remaining":
            continue
        path = prefix / "metrics" / f"{a}-{b}.npz"
        if not path.exists():
            assert not np.any(
                np.random.default_rng(20261122 + 31 * a + b).integers(
                    0, 8, size=len(data)
                )
                == 0
            )
            continue
        z = np.load(path)
        d = z["panel"]
        idx = d[:, 0]
        parts["g"].append([group(int(i >> 12), int(i >> 6 & 63)) for i in idx])
        parts["z"].append(z["s18"])
        parts["er"].append(z["erased"])
        parts["old"].append(z["s12"])
        parts["nn"].append(d[:, 6])
        parts["hh"].append(d[:, 2])
        parts["ll"].append(d[:, 3])
    remaining = {k: np.concatenate(v) for k, v in parts.items()}
    game, _ = game_panel(root)
    for tag, s in [
        ("replication", remaining),
        (
            "played",
            {
                "g": game["groups"],
                "z": game["modern"],
                "er": game["erased"],
                "old": game["old"],
                "nn": game["nn"],
                "hh": game["hh"],
                "ll": game["ll"],
            },
        ),
    ]:
        X = bits(s["hh"]) - bits(s["ll"])
        g = s["g"]
        z = s["z"]
        critical = bits(s["er"])
        coef = attribution(z)
        scores = {
            "sf12": score(s["old"]),
            "sf141": score(s["nn"]),
            "sf18": score(z[:, 15]),
            "sf18_ablated": score(z[:, 0]),
            "fitted": score(X @ w),
            "simple_tactics": score(X[:, [0, 1, 8]].sum(1)),
        }
        for k, v in scores.items():
            lo, hi = ci(v, g)
            performance.append(
                {
                    "panel": tag,
                    "policy": k,
                    "n": len(v),
                    "king_groups": len(np.unique(g)),
                    "accuracy": v.mean(),
                    "lo": lo,
                    "hi": hi,
                }
            )
        for k, j in [
            ("fitted", "simple_tactics"),
            ("fitted", "sf141"),
            ("fitted", "sf18"),
            ("sf18", "sf18_ablated"),
            ("sf18", "sf141"),
        ]:
            v = scores[k] - scores[j]
            lo, hi = ci(v, g)
            effects.append(
                {
                    "panel": tag,
                    "first": k,
                    "second": j,
                    "n": len(v),
                    "effect": v.mean(),
                    "lo": lo,
                    "hi": hi,
                }
            )
        diff = scores["sf18"] - scores["sf18_ablated"]
        bands = simultaneous(critical, diff, g, np.random.default_rng(318712))
        for j, t in enumerate(["ALL", *RULES]):
            sel = np.ones(len(g), bool) if j == 0 else critical[:, j - 1].astype(bool)
            if not sel.any():
                continue
            lo, hi = bands[j - 1] if j else ci(diff, g)
            avg = coef[sel].mean(0)
            mechanisms.append(
                dict(
                    panel=tag,
                    rule=t,
                    n=int(sel.sum()),
                    groups=len(np.unique(g[sel])),
                    **{k: float(v[sel].mean()) for k, v in scores.items()},
                    net_effect=float(diff[sel].mean()),
                    lo=lo,
                    hi=hi,
                    attack_black=avg[0],
                    attack_white=avg[1],
                    enemy_king=avg[2],
                    own_guard=avg[3],
                )
            )
    write(out / "replication.csv", performance)
    write(out / "paired-effects.csv", effects)
    write(out / "channel-interventions.csv", mechanisms)
    write(out / "rule-collisions.csv", twins)
    return mechanisms


def main(root, out):
    init(root)
    rows, ds, allpairs = finite(root, out)
    played = moves_played(root)
    write(out / "played-decisions.csv", played)
    m = subpanels(root, out, allpairs)
    tests = []
    for tag in RULES:
        r = [z for z in rows if z["rule"] == tag]
        d = [z for z in ds if z["policy"] == tag]
        p = next(z for z in played if z["policy"] == tag)
        n = sum(z["erased"] for z in r)
        mx = sum(z["mixed"] for z in d)
        parent = sum(z["parents"] for z in d)
        gain = sum(z["gain_sum"] for z in d)
        ex = next(
            (z for z in m if z["panel"] == "replication" and z["rule"] == tag), {}
        )
        tests.append(
            {
                "rule": tag,
                "native_required": sum(z["required"] for z in r),
                "forced": sum(z["compulsory"] for z in r),
                "erased": n,
                "materials": len({z["material"] for z in r if z["erased"]}),
                "unique_to_class": sum(z["exclusive"] for z in r),
                "accuracy141": frac(sum(z["nnue_credit"] for z in r), n),
                "psqt": frac(sum(z["psqt_credit"] for z in r), n),
                "deep": frac(sum(z["deep_credit"] for z in r), n),
                "narrowing_roots": sum(z["narrowed"] for z in d),
                "better": sum(z["better"] for z in d),
                "worse": sum(z["worse"] for z in d),
                "single_rule_gain_per_mixed": gain / mx,
                "single_rule_gain_all": gain / parent,
                "replication_n": ex.get("n"),
                "delta18": ex.get("net_effect"),
                "simultaneous_lo": ex.get("lo"),
                "simultaneous_hi": ex.get("hi"),
                "played_better": p["better"],
                "played_worse": p["worse"],
                "played_gain_per_mixed": p["mean_mixed"],
            }
        )
    write(out / "findings.csv", tests)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
