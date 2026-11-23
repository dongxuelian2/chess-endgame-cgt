"""Trace endpoint differences through every dense edge and test fixed feature subsets."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from analyze import stack_weights
from collect import EXTENDED, MATERIAL
from spectra import load, match_rate

WIDTHS = (16, 64, 256, 512, 1024, 2048)


def ft_weights(root, stack):
    data = (root / "sf141/src/nn-13406b1dcbe0.nnue").read_bytes()
    size = 4 + 32 + 8 * 2048 + 128 + 1024 + 4 + 32
    start = len(data) - (8 - stack) * size + 4
    bias = np.frombuffer(data, dtype="<i4", count=8, offset=start)
    w = np.frombuffer(data, dtype="i1", count=8 * 2048, offset=start + 32)
    return bias, w.reshape(8, 2048).astype(float)


def get_features(rows, root, raw=False):
    features = []
    accumulators = []
    for i in range(0, len(rows), 128):
        batch = rows[i : i + 128]
        commands = "".join(
            "position fen " + fen + "\neval\n" for r in batch for fen in r["children"]
        )
        run = subprocess.run(
            [root / "circuits/probe/src/stockfish"],
            input=commands + "quit\n",
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "NNUE_CIRCUIT_RAW": "1"} if raw else None,
        )
        for line in run.stdout.splitlines():
            if raw and line.startswith("rawft "):
                packed = np.frombuffer(bytes.fromhex(line.split()[1]), dtype=np.uint8)
                accumulators.append(((packed << 4) | (packed >> 4)).view("<i2"))
            if line.startswith("features "):
                _, hexadecimal, *z = line.split()
                features.append(
                    (
                        np.frombuffer(bytes.fromhex(hexadecimal), dtype=np.uint8),
                        list(map(int, z)),
                    )
                )
    x, z = map(np.array, zip(*features))
    output = (x.reshape(len(rows), 2, 2048), z.reshape(len(rows), 2, 8))
    if raw:
        pre = np.array(accumulators).reshape(len(rows), 2, 2048)
        assert np.array_equal(pre.clip(0, 127), output[0])
        return *output, pre
    return output


def slope(delta_h, delta_p):
    return np.divide(
        delta_h, delta_p, out=np.zeros_like(delta_h, dtype=float), where=delta_p != 0
    )


def analyze(rows, root, stack, order=None):
    b1, w1 = ft_weights(root, stack)
    b2, w2, _b3, w3 = stack_weights(root, stack)
    x, native_pre = get_features(rows, root)
    x = x.astype(float)

    def layers(pre):
        h1 = np.clip(np.floor(pre / 64), 0, 127)
        p2 = h1 @ w2.T + b2
        h2 = np.clip(np.floor(p2 / 64), 0, 127)
        return h1, p2, h2

    p1 = x @ w1.T + b1
    h1, p2, h2 = layers(p1)
    trace = np.array([r["units"] for r in rows])
    assert np.array_equal(p1, native_pre)
    assert np.array_equal(h1, trace[:, :, 2:10]) and np.array_equal(
        h2, trace[:, :, 10:]
    )
    delta = lambda t: t[:, 1] - t[:, 0]
    actual = delta(h2) @ w3
    assert np.array_equal(actual, delta(trace[:, :, 1]))
    dx = delta(x)
    gate2 = slope(delta(h2), delta(p2)) * w3
    gate1 = slope(delta(h1), delta(p1)) * (gate2 @ w2)
    contributions = dx * (gate1 @ w1)
    edges = delta(h1)[:, :, None] * w2.T * gate2[:, None, :]
    assert np.allclose(contributions.sum(1), actual, atol=1e-7)
    assert np.allclose(edges.sum((1, 2)), actual, atol=1e-7)
    means = np.abs(contributions).mean(0)
    if order is None:
        order = np.argsort(-means, kind="stable")
    results = {
        "n": len(rows),
        "head_rate": match_rate(actual),
        "edge_abs_mean": np.abs(edges).mean(0).tolist(),
        "edge_signed_mean": edges.mean(0).tolist(),
    }
    reference = np.mean(actual**2)
    psqt = delta(trace[:, :, 0])

    def patch(subset):
        # Replace exactly this block of differences in the original affine computation.
        step = dx[:, subset] @ w1[:, subset].T
        effect = (
            (
                (layers(p1[:, 0] + step)[2] - h2[:, 0])
                + (h2[:, 1] - layers(p1[:, 1] - step)[2])
            )
            @ w3
            / 2
        )
        return [
            float(np.mean((effect - actual) ** 2) / reference),
            match_rate(effect + psqt),
        ]

    results["width"] = {}
    randoms = [np.random.default_rng(421 + i).permutation(2048) for i in range(16)]
    for k in WIDTHS:
        result = patch(order[:k])
        controls = (
            np.array([patch(z[:k]) for z in randoms])
            if k < 2048
            else np.array([result])
        )
        results["width"][k] = {
            "selected": result,
            "random_median": np.median(controls, 0).tolist(),
            "random_q10": np.quantile(controls, 0.1, axis=0).tolist(),
            "random_q90": np.quantile(controls, 0.9, axis=0).tolist(),
        }
    assert results["width"][2048]["selected"][0] == 0
    return results, order


def take(rows, n, rng):
    ids = rng.choice(len(rows), min(n, len(rows)), replace=False)
    return [rows[i] for i in ids]


def main(root):
    rng = np.random.default_rng(443009)
    months = [
        load(root / "circuits" / f"contacts-2013-{month}.gz") for month in ("01", "02")
    ]
    out = {}
    for stack in range(8):
        ranks = None
        record = {}
        for month, rows in zip(("january", "february"), months):
            group = take([r for r in rows if r["stack"] == stack], 1000, rng)
            record[month], ranks = analyze(group, root, stack, ranks)
        if stack <= 1:
            group = []
            for m in MATERIAL if stack == 0 else EXTENDED:
                group += take(load(root / "circuits" / f"{m}.siblings.gz"), 120, rng)
            record["wdl"], _ = analyze(group, root, stack, ranks)
        record["ranking"] = ranks.tolist()
        out[stack] = record
        print("stack", stack, flush=True)
    (root / "circuits/feature-routes.json").write_text(json.dumps(out, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
