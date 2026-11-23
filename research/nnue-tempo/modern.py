"""Measure SF18 large-net dependence on its learned FullThreats input weights."""

import argparse
import gzip
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from analyze import auc, gap
from collect import EXTENDED, MATERIAL


def readout(rows, root, knockout):
    env = os.environ.copy()
    env.pop("SF18_ZERO_THREATS", None)
    if knockout:
        env["SF18_ZERO_THREATS"] = "1"
    commands = [
        f"position fen {r['board']} {t} - - 0 1\neval\n" for r in rows for t in "wb"
    ]
    proc = subprocess.run(
        [root / "sf18/src/stockfish"],
        input="".join(commands) + "quit\n",
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    trace = np.array(
        [
            list(map(int, line.split()[1:]))
            for line in proc.stdout.splitlines()
            if line.startswith("raw18 ")
        ]
    )
    return trace.reshape(len(rows), 2, 3)


def evaluate(rows, root, game=False):
    full, ablated = (readout(rows, root, k) for k in (False, True))
    assert np.array_equal(full[:, :, 2], ablated[:, :, 2])
    assert np.array_equal(full[:, 0, 2], full[:, 1, 2])
    inactive = full[:, 0, 2] == 0
    assert np.array_equal(full[inactive], ablated[inactive])
    for trace in (full, ablated):
        assert np.all(trace[:, 0, 0] == -trace[:, 1, 0])
    scores = [tr[:, :, :2].sum(axis=(1, 2)) for tr in (full, ablated)]
    d = np.array([sum(w if abs(w) == 2 else 0 for w in r["wdl"]) for r in rows])
    result = {"n": len(rows), "inactive": int(inactive.sum())}
    if np.all(d < 0):
        return {"negative": len(d), "correct": [int((s < 0).sum()) for s in scores]}
    g = np.array(
        [r["game"] if game else 64 * r["group"][0] + r["group"][1] for r in rows]
    )
    for key, mask in [("all", d >= 0), ("active", (d >= 0) & ~inactive)]:
        y = d[mask] > 0
        if 0 < y.sum() < len(y):
            x, z = [s[mask] for s in scores]
            result[key] = {
                "n": len(y),
                "pos": int(y.sum()),
                "full": auc(y, x),
                "knockout": auc(y, z),
                "gap": gap(y, x, z, g[mask]),
            }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    root = p.parse_args().scratch
    result = {}
    with gzip.open(root / "games.jsonl.gz") as f:
        games = list(map(json.loads, f))
    for m in MATERIAL + EXTENDED:
        with gzip.open(root / (m + ".jsonl.gz")) as f:
            test = [r for r in map(json.loads, f) if r["fold"]]
        result[m] = {
            "uniform": evaluate(test, root),
            "games": evaluate([r for r in games if r["material"] == m], root, True),
        }
    with gzip.open(root / "kpk-negative.json.gz") as f:
        result["census"] = evaluate(json.load(f)["examples"], root)
    print(json.dumps(result, indent=2))
