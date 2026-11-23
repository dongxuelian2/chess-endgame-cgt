"""Patch sets of H1-to-H2 messages at both endpoints of legal move comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze import stack_weights
from collect import EXTENDED, MATERIAL
from feature_routes import slope
from spectra import load, match_rate, paired_rate_change

SIZES = (8, 16, 32, 64, 128, 256)


def arrays(rows, weights):
    b, w, _b3, out = weights
    u = np.array([r["units"] for r in rows], dtype=np.int32)
    h = u[:, :, 2:10]
    pre = h @ w.T + b
    post = (pre >> 6).clip(0, 127)
    assert np.array_equal(post, u[:, :, 10:])
    actual = u[:, 1, 1] - u[:, 0, 1]
    dh = h[:, 1] - h[:, 0]
    g = slope(post[:, 1] - post[:, 0], pre[:, 1] - pre[:, 0]) * out
    assert np.allclose((dh * (g @ w)).sum(1), actual)
    # Weighted combination at both finite-difference endpoints, in H2 x H1 order.
    importance = np.einsum("bi,bj->ji", np.abs(dh), np.abs(g)) * abs(w) / len(rows)
    engine = np.array([r["sf141"] for r in rows])
    return {
        "u": u,
        "dh": dh,
        "pre": pre,
        "post": post,
        "actual": actual,
        "psqt": u[:, 1, 0] - u[:, 0, 0],
        "groups": np.array([r.get("game", str(r.get("group"))) for r in rows]),
        "rankings": importance,
        "hce": engine[:, 1, 0] - engine[:, 0, 0],
        "native": engine[:, 1, 1] - engine[:, 0, 1],
    }


def patch(a, weights, selected):
    _, w, _, out = weights
    step = a["dh"] @ (w * selected).T
    p = a["pre"]
    h = a["post"]
    transferred = (
        (
            (((p[:, 0] + step) >> 6).clip(0, 127) - h[:, 0])
            + (h[:, 1] - ((p[:, 1] - step) >> 6).clip(0, 127))
        )
        @ out
        / 2
    )
    mse = np.mean((transferred - a["actual"]) ** 2) / np.mean(
        a["actual"].astype(float) ** 2
    )
    return np.array([mse, match_rate(transferred + a["psqt"])])


def mask(order, k):
    a = np.zeros(256, dtype=bool)
    a[order[:k]] = True
    return a.reshape(32, 8)


def switches(m, rng):
    m = m.copy()
    # A row pair trades one neighbor in each direction; every degree stays fixed.
    for i, j in rng.integers(0, 32, (256, 2)):
        x = np.flatnonzero(m[i] & ~m[j])
        y = np.flatnonzero(m[j] & ~m[i])
        if len(x) and len(y):
            u, v = rng.choice(x), rng.choice(y)
            m[i, [u, v]] = False, True
            m[j, [u, v]] = True, False
    return m


def compare(rows, weights, ranks, primary):
    a = arrays(rows, weights)
    out = {
        "n": len(rows),
        "classical": match_rate(a["hce"]),
        "native": match_rate(a["native"]),
        "net_minus_classical": paired_rate_change(a["native"], a["hce"], a["groups"]),
        "head_only": match_rate(a["actual"]),
        "psqt_only": match_rate(a["psqt"]),
        "width": {},
    }
    rng = np.random.default_rng(8821)
    weight_order = np.argsort(
        -abs(weights[1] * weights[3][:, None]).ravel(), kind="stable"
    )
    for k in SIZES:
        record = {n: patch(a, weights, mask(o, k)).tolist() for n, o in ranks.items()}
        if k == 256:
            assert all(v[0] == 0 for v in record.values())
        record["weight_magnitude"] = patch(a, weights, mask(weight_order, k)).tolist()
        selected = mask(ranks[primary], k)
        if k < 256:
            controls = [patch(a, weights, switches(selected, rng)) for _ in range(12)]
            record["same_degrees_median"] = np.median(controls, axis=0).tolist()
            record["random_median"] = np.median(
                [patch(a, weights, mask(rng.permutation(256), k)) for _ in range(12)],
                axis=0,
            ).tolist()
        out["width"][k] = record
    return out


def main(root):
    dates = ("2013-01", "2013-02")
    data = {
        (m, d): load(root / "circuits" / f"{m}-{d}.gz")
        for m in ("contacts", "forks")
        for d in dates
    }
    result = {}
    for stage in range(8):
        w = stack_weights(root, stage)
        by = {k: [r for r in v if r["stack"] == stage] for k, v in data.items()}
        ranks = {}
        result[stage] = {}
        for m in ("contacts", "forks"):
            rows = by[m, dates[0]]
            if len(rows) >= 20:
                v = arrays(rows, w)["rankings"]
                ranks[m] = np.argsort(-v.ravel(), kind="stable")
        for m in ("contacts", "forks"):
            rows = by[m, dates[1]]
            if rows:
                result[stage][m] = compare(
                    rows, w, ranks, m if m in ranks else "contacts"
                )
        if stage <= 1:
            rows = []
            for m in MATERIAL if stage == 0 else EXTENDED:
                rows.extend(load(root / "circuits" / f"{m}.siblings.gz"))
            result[stage]["wdl"] = compare(rows, w, ranks, "contacts")
        result[stage]["orders"] = {k: v.tolist() for k, v in ranks.items()}
        print(stage, flush=True)
    (root / "circuits/edge-transfer.json").write_text(
        json.dumps(result, allow_nan=False)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
