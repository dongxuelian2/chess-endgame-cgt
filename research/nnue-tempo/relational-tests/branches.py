"""Native NNUE readout components on the complete informative replication subgroup."""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from measure import table

from confirmation import confirm
from search_games import init


def decompose(job):
    return confirm(job, decompose=True)


def main(root, dest):
    with gzip.open(root / "relational-tests/confirmation.gz") as f:
        rs = [r for r in map(json.loads, f) if r["mixed"]]
    jobs = [(root, r["cohort"], r["material"], r["fen"]) for r in rs]
    with ProcessPoolExecutor(8, initializer=init, initargs=(root,)) as pool:
        for r, ans in zip(rs, pool.map(decompose, jobs), strict=True):
            assert ans["full"] == r["refined"]
            r.update(ans)
    out = []
    for tag in ("all", "KBvKP", "KNvKP", "KRvKP"):
        group = [r for r in rs if tag in ("all", r["material"])]
        for branch in (
            "full",
            "quiet_full",
            "psqt",
            "positional",
            "background",
            "kings",
            "pawn",
            "actor",
            "additive",
            "interaction",
            "pair_boards_only",
        ):
            ds = np.array([r[branch] for r in group])
            out.append(
                {
                    "panel": tag,
                    "secondary": branch,
                    "n": len(ds),
                    "better": int(sum(ds > 0)),
                    "equal": int(sum(ds == 0)),
                    "worse": int(sum(ds < 0)),
                    "delta": float(ds.mean()),
                }
            )
    table(dest / "branches.csv", out)
    table(
        dest / "mixed-roots.csv",
        [
            {
                k: r[k]
                for k in (
                    "cohort",
                    "material",
                    "fen",
                    "full",
                    "quiet_full",
                    "psqt",
                    "positional",
                    "nnue",
                    "background",
                    "kings",
                    "pawn",
                    "actor",
                    "additive",
                    "interaction",
                    "pair_boards_only",
                )
            }
            for r in rs
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    main(a.root, a.dest)
