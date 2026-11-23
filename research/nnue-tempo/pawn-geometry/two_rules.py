"""Add only the two discovered relations to an otherwise fixed classical evaluator."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np
from channels import fold
from classical_terms import features
from collect import king_group
from measure import load, table
from threat_reversal import interval
from transfer import MATERIALS, square_class


def data(root, mat, random=False):
    rows = load(root, ("random-" if random else "") + mat)
    tags = []
    for r in rows:
        dest = [chess.Move.from_uci(u).to_square for u in r["moves"]]
        t = []
        for fen, a in zip(r["children"] + r["edited_children"], dest * 2):
            b = chess.Board(fen)
            p = next(iter(b.pieces(chess.PAWN, False)))
            capture, prom = square_class(b, a, p)
            t.append([capture, prom == "free"])
        tags.append(t)
    x = np.array(tags, dtype=float).reshape(-1, 2, 2, 2)
    _, full = features(root, rows, True)
    score = np.array(
        [[v for v in r["sf141"] + r["edited_sf141"]] for r in rows]
    ).reshape(-1, 2, 2, 2)
    delta = lambda v: v[:, :, 1] - v[:, :, 0]
    native = delta(score)
    delta_full = delta(full)[:, :, 0]
    labs = np.array([r["labels"] for r in rows]).reshape(-1, 2, 2)
    return (
        np.array([fold(r) for r in rows]),
        np.array([r["stratum"] for r in rows]),
        delta(x),
        native,
        delta_full,
        -delta(labs),
        np.array(
            [
                str(king_group(b.king(True), b.king(False)))
                for b in map(chess.Board, [r["parent"] for r in rows])
            ]
        ),
    )


def fit(x, base, label):
    y = np.sign(label.ravel())
    keep = y != 0
    b = (base.ravel()[keep] / 208) * y[keep]
    z = x.reshape(-1, 2)[keep] * y[keep, None]
    w = np.zeros(2)
    ridge = 0.01
    for _ in range(50):
        p = 1 / (1 + np.exp(np.clip(b + z @ w, -40, 40)))
        g = -z.T @ p / len(p) + ridge * w
        step = np.linalg.solve(
            (z.T * (p * (1 - p))) @ z / len(p) + ridge * np.eye(2), g
        )
        w -= step
        if np.max(abs(step)) < 1e-10:
            break
    assert max(abs(g)) < 1e-7
    return w * 208


def run(args):
    root, mat, weights = args
    results = []
    for random in (False, True):
        folds, gs, x, base, full, lab, groups = data(root, mat, random)
        for group in sorted(set(gs)):
            sel = (gs == group) & (folds == 1)
            y = lab[sel]
            usable = y != 0
            both = y[:, 0] * y[:, 1] < 0
            d = x[sel]
            s = base[sel]
            h = full[sel]
            methods = {
                "hce": s[:, :, 0],
                "full_pregrain": h,
                "nnue": s[:, :, 1],
                "capture_only": s[:, :, 0] + d[:, :, 0] * weights[0],
                "promotion_only": s[:, :, 0] + d[:, :, 1] * weights[1],
                "both_rules": s[:, :, 0] + d @ weights,
                "full_plus_rules": h + d @ weights,
            }
            group_ids = np.repeat(groups[sel, None], 2, axis=1)[usable]
            hce_credit = (methods["hce"] * y > 0) + 0.5 * (methods["hce"] * y == 0)
            stable = usable & np.all(d == 0, axis=2)
            for name, ds in methods.items():
                r = ds * y
                credit = (r > 0) + 0.5 * (r == 0)
                ci = (
                    interval((credit - hce_credit)[usable], group_ids)
                    if usable.any()
                    else [None] * 3
                )
                results.append(
                    {
                        "unchanged_terms_n": int(stable.sum()),
                        "unchanged_terms_rate": float(credit[stable].mean())
                        if stable.any()
                        else None,
                        "diff_from_hce": ci[0],
                        "ci_low": ci[1],
                        "ci_high": ci[2],
                        "material": mat,
                        "design": group,
                        "method": name,
                        "n": len(y),
                        "wdl_informative": int(usable.sum()),
                        "move_rate": float(
                            np.mean((r[usable] > 0) + 0.5 * (r[usable] == 0))
                        )
                        if usable.any()
                        else None,
                        "strict_reversals": int(both.sum()),
                        "reversals_correct": int(np.sum(both & np.all(r > 0, axis=1))),
                    }
                )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    folds, _, x, base, _, y, _ = data(a.root, "KRvKP")
    tr = folds == 0
    w = fit(x[tr], base[tr, :, 0], y[tr])
    with ProcessPoolExecutor(5) as pool:
        rows = [
            r
            for batch in pool.map(run, [(a.root, m, w) for m in MATERIALS])
            for r in batch
        ]
    table(a.dest / "two-rules.csv", rows)
    (a.dest / "rule-weights.json").write_text(
        json.dumps({"n": int(tr.sum()), "capture": w[0], "promotion": w[1]}) + "\n"
    )
