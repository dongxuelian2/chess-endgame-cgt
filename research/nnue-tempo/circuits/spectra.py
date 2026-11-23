"""Exhaust all 256 H1 interchanges and follow their effects through the 32 upper units."""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from analyze import auc, gap, stack_weights
from collect import EXTENDED, MATERIAL
from routing import MASKS, ORDER, PAIR, Cube, compile_kernel, transform


def load(path):
    with gzip.open(path) as f:
        return list(map(json.loads, f))


def match_rate(score):
    return np.mean((score > 0) + 0.5 * (score == 0))


def paired_rate_change(a, b, groups):
    d = ((a > 0).astype(float) + 0.5 * (a == 0)) - (
        (b > 0).astype(float) + 0.5 * (b == 0)
    )
    _, ids = np.unique(groups, return_inverse=True)
    n = ids.max() + 1
    if n < 2:
        return [float(d.mean()), None, None]
    z = np.bincount(ids, weights=d - d.mean()) / len(d)
    err = 1.96 * np.sqrt(z @ z * n / (n - 1))
    return [d.mean(), d.mean() - err, d.mean() + err]


def calculate(rows, cube, tempo):
    if not rows:
        return {}
    trace = np.array([r["units"] for r in rows], dtype=np.int32)
    h = trace[:, :, 2:10]
    cube.verify(h[:2])
    # Endpoints are white/black for tempo and preferred/inferior child for a move.
    if tempo:
        delta = np.array([sum(w if abs(w) == 2 else 0 for w in r["wdl"]) for r in rows])
        eligible = delta >= 0
    else:
        delta = np.ones(len(rows))
        eligible = np.ones(len(rows), dtype=bool)
    h, trace, delta = h[eligible], trace[eligible], delta[eligible]
    rows = [r for r, k in zip(rows, eligible) if k]
    groups = np.array([r["game"] if "game" in r else str(r["group"]) for r in rows])
    labels = delta > 0
    coeffs = []
    route_sums = np.zeros((2, 256, 32), dtype=np.int64)
    sm_error = 0.0
    sm_ref = 0.0
    for start in range(0, len(h), 160):
        stop = min(start + 160, len(h))
        cells, smooth = cube.forward(h[start:stop], smooth=start == 0)
        assert np.array_equal(cells[:, [0, 255]], trace[start:stop, :, 10:])
        # All integer products are exact in float32 here. Integer FWHT follows.
        values = (cells.astype(np.float32) @ cube.out.astype(np.float32)).astype(
            np.int32
        )
        assert np.array_equal(
            values[:, [0, 255]] + cube.out_bias, trace[start:stop, :, 1]
        )
        c = transform(values)
        assert np.array_equal(c.sum(1), 256 * values[:, 0])
        assert np.array_equal(c @ np.where(ORDER % 2, -1, 1), 256 * values[:, 255])
        coeffs.append(c)
        for label in (0, 1):
            route_sums[label] += cells[labels[start:stop] == label].sum(
                0, dtype=np.int64
            )
        if smooth is not None:
            cont = transform(smooth.astype(float) @ cube.out / 64)
            select = (ORDER > 0) & (ORDER % 2 == int(not tempo))
            sm_error = float(np.square(cont[:, select] - c[:, select]).sum())
            sm_ref = float(np.square(cont[:, select]).sum())
    c = np.vstack(coeffs).astype(float) / 256
    keep_parity = ORDER % 2 == int(not tempo)
    c[:, ~keep_parity] = 0
    if tempo:
        score0 = c[:, 0].copy()
        assert np.all(trace[:, 0, 0] + trace[:, 1, 0] == 0)
    else:
        score0 = (trace[:, 1, 0] - trace[:, 0, 0]) / 2
        c *= -1
    values = np.array([r["sf141"] for r in rows])
    classical = values.sum(1)[:, 0] if tempo else values[:, 1, 0] - values[:, 0, 0]
    native = values.sum(1)[:, 1] if tempo else values[:, 1, 1] - values[:, 0, 1]
    if tempo:
        c[:, 0] = 0
    cuts = transform(c, subset_sum=True) + score0[:, None]
    full = cuts[:, 255]
    residual = full - score0
    scale = np.mean(residual**2)
    error = (
        np.mean((cuts - full[:, None]) ** 2, axis=0) / scale
        if scale > 0
        else np.zeros(256)
    )
    metric = (lambda s: auc(labels, s)) if tempo else match_rate
    valid = (np.any(labels) & np.any(~labels)) if tempo else True
    change = (
        (lambda a, b: gap(labels, a, b, groups))
        if tempo
        else (lambda a, b: paired_rate_change(a, b, groups))
    )
    energy = np.square(c).sum(0)
    term_energy = energy / energy.sum() if energy.sum() else np.zeros(256)
    # Linear readout separates each Fourier coefficient by upper-layer destination.
    effects = transform(route_sums.transpose(0, 2, 1)).transpose(0, 2, 1) / 256
    if tempo and valid:
        effects = effects[1] / labels.sum() - effects[0] / (~labels).sum()
    else:
        effects = effects[1] / max(1, labels.sum())
    effects = effects * cube.out * (1 if tempo else -1)
    term_list = PAIR if tempo else MASKS[ORDER == 1]
    routes = effects[term_list]
    if not tempo or valid:
        measured = c[labels].mean(0) - (c[~labels].mean(0) if tempo else 0)
        assert np.allclose(routes.sum(1), measured[term_list])
    first = 2 if tempo else 1
    low = score0 + c[:, ORDER == first].sum(1)
    high = low + c[:, ORDER == first + 2].sum(1)
    out = {
        "n": len(rows),
        "positive": int(labels.sum()),
        "smooth_error_ratio": sm_error / sm_ref if sm_ref else None,
        "order_energy": [term_energy[ORDER == i].sum() for i in range(9)],
        "mask_nmse": error.tolist(),
        "term_energy": term_energy.tolist(),
        "route_means": routes.tolist() if valid else None,
        "order1or2_endpoint_nmse": float(np.mean((low - full) ** 2) / scale)
        if scale
        else None,
    }
    changed = (h[:, 0] != h[:, 1]).sum(1)
    rich = changed >= 5
    out["changed_units_histogram"] = np.bincount(changed, minlength=9).tolist()
    if rich.any():
        e = np.square(c[rich]).sum(0)
        out["leading_energy_at_least_five_changed"] = (
            float(e[ORDER == first].sum() / e.sum()) if e.sum() else None
        )
    if valid:
        h2pair = (
            trace[:, 0, 10:] + trace[:, 1, 10:]
            if tempo
            else trace[:, 1, 10:] - trace[:, 0, 10:]
        )
        direct = 0 if tempo else 2 * score0
        permutations = np.random.default_rng(888).permuted(
            np.tile(cube.out, (128, 1)), axis=1
        )
        shuffled = [metric(h2pair @ v + direct) for v in permutations]
        out["permuted_readout_q05_median_q95"] = np.quantile(
            shuffled, [0.05, 0.5, 0.95]
        ).tolist()
        out.update(
            metric={
                "classical": metric(classical),
                "native": metric(native),
                "base": metric(score0),
                "leading": metric(low),
                "next": metric(high),
                "full": metric(full),
            },
            full_minus_base=change(full, score0),
            full_minus_leading=change(full, low),
            mask_metric=[metric(cuts[:, m]) for m in MASKS],
        )
    return out


def analyze_one(key, root, feb, jan, motif):
    materials = MATERIAL + EXTENDED
    if key in materials:
        stack = int(key in EXTENDED)
        sets = load(root / (key + ".jsonl.gz"))
        cohorts = {
            "fit": [r for r in sets if not r["fold"]],
            "test": [r for r in sets if r["fold"]],
            "jan": [r for r in jan if r["material"] == key],
            "feb": [r for r in feb if r["material"] == key],
        }
        moves = load(root / "circuits" / f"{key}.siblings.gz")
        cohorts.update(siblings=moves, siblings_quiet=[r for r in moves if r["safe"]])
    else:
        stack = int(key)
        cohorts = {}
        for month in ("2013-01", "2013-02"):
            cohorts[month] = [
                r
                for r in load(root / "circuits" / f"{motif}-{month}.gz")
                if r["stack"] == stack
            ]
    cube = Cube(root / "circuits/cube.so", stack_weights(root, stack))
    result = {
        k: calculate(rs, cube, k in ("fit", "test", "jan", "feb"))
        for k, rs in cohorts.items()
    }
    prefix = "fork-" if motif == "forks" else ""
    (root / "circuits" / f"spectrum-{prefix}{key}.json").write_text(
        json.dumps(result, allow_nan=False)
    )
    return key


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--contacts", action="store_true")
    group.add_argument("--forks", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    compile_kernel(args.root)
    feb, jan = (
        ([], [])
        if (args.contacts or args.forks)
        else (
            load(args.root / "circuits/feb.jsonl.gz"),
            load(args.root / "games.jsonl.gz"),
        )
    )
    keys = (
        list(map(str, range(8)))
        if (args.contacts or args.forks)
        else MATERIAL + EXTENDED
    )
    with ProcessPoolExecutor(args.workers) as pool:
        for key in pool.map(
            partial(
                analyze_one,
                root=args.root,
                jan=jan,
                feb=feb,
                motif="forks" if args.forks else "contacts",
            ),
            keys,
        ):
            print(key, flush=True)
