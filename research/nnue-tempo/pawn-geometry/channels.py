"""Train a pawn-danger direction on KRvKP and intervene on its frozen feature slots."""

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
from analyze import stack_weights
from collect import king_group
from feature_routes import ft_weights, get_features
from interaction_layers import equalize
from measure import load, table
from transfer import MATERIALS, STRATA


def fold(r):
    b = chess.Board(r["parent"])
    k = king_group(b.king(True), b.king(False))
    return hashlib.sha256(bytes(k)).digest()[0] % 2


def identity(r):
    return hashlib.sha256(("|".join([r["parent"], *r["moves"]])).encode()).digest()


def arrays(rows, root, stage):
    xs = []
    ps = []
    labs = []
    b1, w1 = ft_weights(root, stage)
    b2, w2, last, w3 = stack_weights(root, stage)
    for i in range(0, len(rows), 100):
        rs = rows[i : i + 100]
        edits = [dict(r, children=r["edited_children"]) for r in rs]
        x, z, raw = get_features(rs + edits, root, raw=True)

        def pack(v):
            return np.stack(np.split(v, 2), axis=1)

        delta = lambda v: v[:, 0, 1] - v[:, 0, 0] - v[:, 1, 1] + v[:, 1, 0]
        assert not np.any(delta(pack(raw.astype(np.int32))))
        assert np.array_equal(x @ w1.T + b1, z)
        tr = np.array([[r["units"], r["edited_units"]] for r in rs])
        a = (pack(z) >> 6).clip(0, 127)
        h = ((a @ w2.T + b2) >> 6).clip(0, 127)
        assert np.array_equal(a, tr[:, :, :, 2:10]) and np.array_equal(
            h, tr[:, :, :, 10:]
        )
        assert np.array_equal(h @ w3 + last, tr[:, :, :, 1])
        xs.append(pack(x))
        ps.append(tr[:, :, :, 0])
        labs.extend(r["labels"] for r in rs)
    return np.concatenate(xs), np.concatenate(ps), np.array(labs)


def mixed(x):
    return x[:, 0, 1] - x[:, 0, 0] - x[:, 1, 1] + x[:, 1, 0]


def fit(x):
    scale = np.sqrt(np.mean(x**2, axis=0))
    scale[scale == 0] = 1
    z = x / scale
    # Regress +1 on the oriented difference, equivalent to balanced +/- reversed pairs.
    penalty = 0.2
    w = np.linalg.solve(z.T @ z + len(x) * penalty * np.eye(z.shape[1]), z.sum(0))
    return w / scale, abs(w), scale


def rate(x):
    return np.mean((x > 0) + 0.5 * (x == 0))


def patches(x, ps, w, ids, labels):
    b1, w1, b2, w2, w3 = w
    n = len(x)
    result = []
    for i in range(0, n, 80):
        raw = x[i : i + 80].astype(float)
        base = equalize(raw)
        p = base @ w1.T + b1
        changes = raw - base

        def eval_pre(z):
            return (
                ((np.floor(z / 64).clip(0, 127) @ w2.T + b2) // 64).clip(0, 127)
            ) @ w3

        effects = []
        for idx in ids:
            if idx is None:
                pre = raw @ w1.T + b1
            elif len(idx) == 0:
                pre = p
            else:
                pre = p + changes[:, :, :, idx] @ w1[:, idx].T
            effects.append(eval_pre(pre) + ps[i : i + 80])
        result.append(np.stack(effects, axis=1))
    y = np.concatenate(result)
    d = y[:, :, :, 1] - y[:, :, :, 0]
    c = d[:, :, 0] - d[:, :, 1]
    actual = c[:, 0]
    reversal = (labels[:, 0] > labels[:, 1]) & (labels[:, 2] < labels[:, 3])
    power = actual @ actual
    return {
        "interaction_projection": (c.T @ actual / power).tolist(),
        "geometric_direction": [rate(v) for v in c.T],
        "both_on_forward": [
            float(np.mean((v[reversal, 0] > 0) & (v[reversal, 1] < 0)))
            if reversal.any()
            else None
            for v in d.transpose(1, 0, 2)
        ],
    }


def main(root, dest):
    sets = load(root, "KRvKP")
    training = sorted(
        [r for r in sets if not fold(r) and r["stratum"] in ("double", "single")],
        key=identity,
    )
    x, _, _ = arrays(training, root, 0)
    xx = mixed(x.astype(float))
    coef, importance, scale = fit(xx)
    rng = np.random.default_rng(7331)
    placebo, _, _ = fit(xx * rng.choice([-1, 1], len(xx))[:, None])
    order = np.argsort(-importance, kind="stable")
    bins = np.array_split(np.argsort(scale, kind="stable"), 16)
    masks = [None, [], order[:64], order[:256]]
    # Match each random set on RMS of the native interaction in the discovery set.
    for _ in range(8):
        masks.append(
            np.concatenate(
                [
                    rng.choice(b, np.isin(b, order[:256]).sum(), replace=False)
                    for b in bins
                ]
            )
        )
    out = []
    stats = []
    for mat in MATERIALS:
        data = load(root, mat)
        stage = int("P" in mat.split("v")[0])
        b1, w1 = ft_weights(root, stage)
        b2, w2, _, w3 = stack_weights(root, stage)
        for group in STRATA:
            rs = sorted(
                [r for r in data if r["stratum"] == group and fold(r)], key=identity
            )[:500]
            if not rs:
                continue
            x, ps, labels = arrays(rs, root, stage)
            diff = mixed(x.astype(float))
            pred = diff @ coef
            m = patches(x, ps, (b1, w1, b2, w2, w3), masks, labels)
            forward = (labels[:, 0] > labels[:, 1]) & (labels[:, 2] < labels[:, 3])
            out.append(
                {
                    "material": mat,
                    "design": group,
                    "n": len(rs),
                    "n_forward": int(forward.sum()),
                    "probe_rate": rate(pred),
                    "permuted_probe_rate": rate(diff @ placebo),
                    "full_direction": m["geometric_direction"][0],
                    "equalized_direction": m["geometric_direction"][1],
                    "full_both": m["both_on_forward"][0],
                    "equalized_both": m["both_on_forward"][1],
                    "equalized_projection": m["interaction_projection"][1],
                    "restore64_projection": m["interaction_projection"][2],
                    "restore256_projection": m["interaction_projection"][3],
                    "matched256_projection": np.median(m["interaction_projection"][4:]),
                }
            )
            stats.append(dict(material=mat, design=group, **m))
    table(dest / "channels.csv", out)
    (dest / "channel-masks.json").write_text(
        json.dumps(
            {
                "training_n": len(training),
                "order": order.tolist(),
                "coef": coef.tolist(),
                "placebo": placebo.tolist(),
                "measurements": stats,
            },
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    main(args.root, args.dest)
