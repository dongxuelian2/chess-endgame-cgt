"""Independent integer decision oracle with globally shared edit variables."""

import argparse
import gzip
import json
from pathlib import Path

import z3
from measure import table

from search_games import game, init


def check(tree, hi, alt, leader, k):
    t = z3.Int("threshold")
    switches = {}
    vals = []
    for i, (a, b) in enumerate(zip(hi, alt)):
        if a == b:
            vals.append(z3.IntVal(int(a)))
        else:
            switches[i] = z3.Bool("edit_" + str(i))
            vals.append(z3.If(switches[i], int(b), int(a)))
    passes = [v >= t for v in vals]
    safe = [
        z3.And([z3.Or([passes[i] for i in row]) for row in action]) for action in tree
    ]
    s = z3.Solver()
    s.add(z3.Not(safe[leader]), z3.Or([v for i, v in enumerate(safe) if i != leader]))

    def solve(budget):
        s.push()
        if budget is not None:
            s.add(z3.Sum([z3.If(x, 1, 0) for x in switches.values()]) <= budget)
        res = s.check()
        s.pop()
        return res

    if k is None:
        assert solve(None) == z3.unsat
    else:
        assert solve(k - 1) == z3.unsat and solve(k) == z3.sat


def main(root, dest):
    init(root)
    with gzip.open(dest / "root-costs.json.gz") as f:
        rs = [r for r in json.load(f) if r["unique"]]
    out = []
    for ext in ("king", "pawn", "both", "clip_pair", "clip_permuted"):
        sub = [r for r in rs if r["extension"] == ext]
        tests = [
            max((r for r in sub if r["k"]), key=lambda r: r["k"]),
            next(r for r in sub if r["k"] and not r["complete_displaces"]),
            next(r for r in sub if r["k"] is None and r["eligible"] > 5),
        ]
        if ext == "both":
            tests += [r for r in sub if r["k"] != r["relaxed"]]
        for r in {r["fen"]: r for r in tests}.values():
            args = game((root, r["cohort"], r["fen"]), audit=ext)
            check(*args, r["k"])
            out.append(
                {
                    "fen": r["fen"],
                    "treatment": ext,
                    "minimum": r["k"],
                    "eligible": r["eligible"],
                    "z3": z3.get_version_string(),
                }
            )
    table(dest / "audit.csv", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    main(a.root, a.dest)
