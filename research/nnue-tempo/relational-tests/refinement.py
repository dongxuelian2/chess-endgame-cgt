"""Scale-free minimax tie refinement with two native evaluators on the same tree."""

import argparse
import gzip
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from measure import table

from certificate import evaluate
from search_games import game, init, leaf_score


def test_root(job):
    root, cohort, fen = job
    tree, boards, truth, moves = game(job, audit="classical")
    nn = leaf_score(boards)
    hce = nn.copy()
    indices = [
        i
        for i, b in enumerate(boards)
        if not b.is_check() and not b.is_insufficient_material() and any(b.legal_moves)
    ]
    commands = "".join("position fen " + boards[i].fen() + "\neval\n" for i in indices)
    output = subprocess.run(
        [root / "sf141/src/stockfish"],
        input=commands + "quit\n",
        text=True,
        capture_output=True,
        check=True,
    )
    scores = [
        tuple(map(int, s.split()[1:]))
        for s in output.stdout.splitlines()
        if s.startswith("scores ")
    ]
    assert len(indices) == len(scores)
    for i, (h, n) in zip(indices, scores):
        sign = 1 if boards[i].turn else -1
        assert nn[i] == sign * n
        hce[i] = sign * h
    # Rank encoding realizes exact native lexicographic order including equal pairs.
    _, ranks = np.unique(np.column_stack((hce, nn)), axis=0, return_inverse=True)

    def best(s):
        v = evaluate(tree, s)
        return list(map(int, np.flatnonzero(v == v.max())))

    hb, nb, rb = map(best, (hce, nn, ranks))
    assert set(rb) <= set(hb)

    def utility(indices):
        if any(truth[i] is None for i in indices):
            return None
        return sum(truth[i] for i in indices) / len(indices)

    subset = lambda indices: [moves[i] for i in indices]
    return {
        "fen": fen,
        "cohort": cohort,
        "placements": len(boards),
        "native_verified": len(indices),
        "hce": subset(hb),
        "nnue": subset(nb),
        "refined": subset(rb),
        "hce_wdl": utility(hb),
        "nnue_wdl": utility(nb),
        "refined_wdl": utility(rb),
        "different_wdl_hce_tie": len({truth[i] for i in hb}) > 1
        and all(truth[i] is not None for i in hb),
    }


def summarize(rows, dest):
    out = []
    for tag in ("all", "played", "random"):
        group = [r for r in rows if tag == "all" or r["cohort"] == tag]
        for scope in ("all", "HCE tie", "refinement narrows", "mixed WDL HCE tie"):
            rs = [
                r
                for r in group
                if scope == "all"
                or (scope == "HCE tie" and len(r["hce"]) > 1)
                or (scope == "refinement narrows" and len(r["hce"]) > len(r["refined"]))
                or (scope == "mixed WDL HCE tie" and r["different_wdl_hce_tie"])
            ]
            for name in ("refined", "nnue"):
                eligible = [
                    r
                    for r in rs
                    if r["hce_wdl"] is not None and r[name + "_wdl"] is not None
                ]
                d = np.array([r[name + "_wdl"] - r["hce_wdl"] for r in eligible])
                out.append(
                    {
                        "cohort": tag,
                        "scope": scope,
                        "alternative": name,
                        "n": len(rs),
                        "wdl_complete": len(d),
                        "higher": int(sum(d > 0)),
                        "equal": int(sum(d == 0)),
                        "lower": int(sum(d < 0)),
                        "mean_delta": None if not len(d) else float(d.mean()),
                    }
                )
    table(dest / "refinement.csv", out)


def main(root, dest):
    with gzip.open(dest / "root-costs.json.gz") as f:
        prior = json.load(f)
    jobs = [(root, r["cohort"], r["fen"]) for r in prior if r["extension"] == "both"]
    with ProcessPoolExecutor(10, initializer=init, initargs=(root,)) as pool:
        rows = list(pool.map(test_root, jobs, chunksize=1))
    with gzip.open(root / "relational-tests/refinement.gz", "wt") as f:
        for r in rows:
            print(json.dumps(r), file=f)
    summarize(rows, dest)
    records = [
        r
        for r in rows
        if r["different_wdl_hce_tie"] and len(r["refined"]) < len(r["hce"])
    ]
    sampled = []
    for sign in (-1, 0, 1):
        rs = [r for r in records if np.sign(r["refined_wdl"] - r["hce_wdl"]) == sign]
        sampled += sorted(rs, key=lambda r: hashlib.sha256(r["fen"].encode()).digest())[
            :2
        ]
    (dest / "refinement-examples.json").write_text(json.dumps(sampled, indent=1) + "\n")
    print(
        "roots",
        len(rows),
        "native leaf checks",
        sum(r["native_verified"] for r in rows),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    main(args.root, args.dest)
