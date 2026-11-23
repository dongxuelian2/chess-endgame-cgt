"""Conditional randomization over eligible leaves with a shared-state feasibility oracle."""

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from search_games import game, init


def feasible(tree, base, alternatives, leader):
    rows = [row for action in tree for row in action]
    ids = np.concatenate(rows)
    row_starts = np.cumsum([0] + [len(r) for r in rows[:-1]])
    action_starts = np.cumsum([0] + [len(a) for a in tree[:-1]])

    def action_max(s):
        best = np.maximum.reduceat(s[ids], row_starts, axis=0)
        score = np.minimum.reduceat(best, action_starts, axis=0)
        return np.max(np.delete(score, leader, axis=0), axis=0)

    lower = np.minimum(base[:, None], alternatives)
    upper = np.maximum(base[:, None], alternatives)
    ceiling = action_max(upper)
    won = np.zeros(len(ceiling), bool)
    for row in tree[leader]:
        floor = np.max(lower[row], axis=0)
        js = np.flatnonzero((ceiling > floor) & ~won)
        if len(js):
            s = upper[:, js].copy()
            s[row] = -100000
            won[js] = action_max(s) > floor[js]
    return won


def simulate(job):
    root, cohort, fen, expected = job
    tree, hi, alt, eligible, leader = game((root, cohort, fen), audit="permute")
    rng = np.random.default_rng(
        int.from_bytes(
            hashlib.sha256(("permute-255-v1" + fen).encode()).digest()[:8], "big"
        )
    )
    diff = (alt - hi)[eligible]
    counts = []
    for batch in range(8):
        x = np.repeat(hi[:, None], 32, axis=1)
        for j in range(32):
            x[eligible, j] += diff if batch == j == 0 else rng.permutation(diff)
        counts.extend(map(int, feasible(tree, hi, x, leader)))
    assert counts[0] == expected
    return counts


def main(root, dest):
    with gzip.open(dest / "root-costs.json.gz") as f:
        cases = json.load(f)
    rs = [r for r in cases if r["unique"] and r["extension"] == "clip_pair"]
    jobs = [(root, r["cohort"], r["fen"], r["k"] is not None) for r in rs]
    with ProcessPoolExecutor(10, initializer=init, initargs=(root,)) as pool:
        samples = np.array(list(pool.map(simulate, jobs, chunksize=1)))
    result = []
    for label in ("all", "played", "random"):
        ids = [i for i, r in enumerate(rs) if label == "all" or r["cohort"] == label]
        counts = samples[ids].sum(axis=0)
        result.append(
            {
                "cohort": label,
                "n": len(ids),
                "actual": int(counts[0]),
                "null": counts[1:].tolist(),
                "mean": float(counts[1:].mean()),
                "quantiles": np.quantile(counts[1:], [0.025, 0.5, 0.975]).tolist(),
                "p_greater": float(np.mean(counts >= counts[0])),
                "p_less": float(np.mean(counts <= counts[0])),
            }
        )
    (dest / "randomization.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    main(a.root, a.dest)
