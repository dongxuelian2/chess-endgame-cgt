"""Disjoint test of a geometric pawn-race rule on minor-piece endings."""

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from measure import load, table

from confirmation import confirm
from search_games import init


def run(job):
    return confirm(job, decompose="race")


def main(root, dest):
    with gzip.open(dest / "root-costs.json.gz") as f:
        old = {r["fen"].split()[0] for r in json.load(f)}
    with gzip.open(root / "relational-tests/confirmation.gz") as f:
        old.update(r["fen"].split()[0] for r in map(json.loads, f))
    jobs = []
    for m in ("KNvKP", "KBvKP"):
        keys = {r["parent"].split()[0] for r in load(root, "random-" + m)} - old
        chosen = sorted(
            keys, key=lambda s: hashlib.sha256(("race-800-v1" + s).encode()).digest()
        )[:800]
        assert len(chosen) == 800
        jobs.extend((root, "random", m, k + " w - - 0 1") for k in chosen)
    with ProcessPoolExecutor(10, initializer=init, initargs=(root,)) as pool:
        ds = list(pool.map(run, jobs, chunksize=1))
    result = []
    alignment = []
    for mat in ("all", "KNvKP", "KBvKP"):
        sub = [r for r in ds if mat in ("all", r["material"])]
        informative = [r for r in sub if r["mixed"]]
        count = sum(r["pairs"] for r in sub)
        for j, head in enumerate(("native", "positional", "psqt")):
            alignment.append(
                {
                    "material": mat,
                    "head": head,
                    "roots_with_pairs": sum(r["pairs"] > 0 for r in sub),
                    "pairs": count,
                    "agree_true": sum(r["credits"][j] for r in sub) / count,
                }
            )
        for policy in ("full", "interception", "reflected_file", "rank"):
            v = np.array([r[policy] for r in informative])
            result.append(
                {
                    "material": mat,
                    "policy": policy,
                    "roots": len(sub),
                    "states": sum(r["states"] for r in sub),
                    "wdl_known": sum(r["known"] for r in sub),
                    "informative": len(v),
                    "improved": int(sum(v > 0)),
                    "equal": int(sum(v == 0)),
                    "worsened": int(sum(v < 0)),
                    "mean_delta": float(v.mean()),
                    "overlap": None
                    if policy == "full"
                    else float(np.mean([r[policy + "_overlap"] for r in informative])),
                    "chance": None
                    if policy == "full"
                    else float(np.mean([r[policy + "_chance"] for r in informative])),
                }
            )
    table(dest / "leaf-alignment.csv", alignment)
    table(dest / "interception.csv", result)
    table(
        dest / "interception-roots.csv",
        [
            {k: v for k, v in r.items() if k not in ("pairs", "credits")}
            for r in ds
            if r["mixed"]
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    main(a.root, a.dest)
