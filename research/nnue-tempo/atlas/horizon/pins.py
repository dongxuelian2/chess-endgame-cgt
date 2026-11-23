"""Test whether an absolute pin denies the pawn its outcome-preserving advance."""

import argparse
import csv
import json
from pathlib import Path

import chess
import numpy as np
import validate
from results import load
from summarize import score, write


def main(root, out):
    validate.init(root)
    panel, gids, d, _refs = load(root / "atlas/horizon/all5")
    with (out / "proof-onsets.csv").open() as f:
        onsets = {r["game"]: r for r in csv.DictReader(f)}
    counts = []
    checks = []
    examples = []
    pushbit = 1 << validate.RULES.index("single_push")
    for a in (3, 4, 5):
        hist = sum(
            np.loadtxt(root / f"atlas/horizon/pin-{a}-{fold}.txt", dtype=int)[
                :, -1
            ].reshape(3, 3, 3)
            for fold in ("pilot", "remaining")
        )
        for order in range(-1, 2):
            for idx, desc in enumerate(("rest", "erased", "forced")):
                n = hist[order + 1, idx]
                counts.append(
                    {
                        "white": a,
                        "unpinned_minus_pinned_wdl_order": order,
                        "removal": desc,
                        "pairs": int(n.sum()),
                        "nn_favors_pinned": int(n[0]),
                        "nn_tie": int(n[1]),
                        "nn_favors_unpinned": int(n[2]),
                    }
                )
        for mode in np.unique(panel):
            rec = []
            for i in np.flatnonzero((panel == mode) & (d["a"] == a) & (d["b"] == 1)):
                w, k, y, x, z = map(int, [d[p][i] for p in ("wk", "bk", "y", "x", "z")])
                if y // 8 not in range(2, 6) or y - 8 in (w, k, x, z):
                    continue
                boards = [validate.board(a, 1, w, k, y, s) for s in (x, z)]
                pins = [q.is_pinned(False, y) for q in boards]
                if pins.count(True) != 1:
                    continue
                u, p = pins.index(False), pins.index(True)
                m = chess.Move(y, y - 8)
                if m not in boards[u].legal_moves or m in boards[p].legal_moves:
                    continue
                su = (x, z)[u]
                _mask, replies = validate.probe(a, 1, w, k, y, su, True)
                old = max(v for _, v, _ in replies)
                other = [v for _, v, tag in replies if not (tag & pushbit)]
                pinned = validate.sign(validate.tb.probe_wdl(boards[p]))
                erased = old > pinned and bool(other) and max(other) <= pinned
                rec.append((erased, u == 0, float(score(d["nn"][i])), gids[i]))
                if mode == "fresh" and erased:
                    examples.append(
                        {
                            "game": gids[i],
                            "material": a,
                            "unpinned_fen": boards[u].fen(),
                            "pinned_fen": boards[p].fen(),
                            "push": m.uci(),
                            "original": old,
                            "after_removing_push": max(other),
                            "pinned": pinned,
                            "minimum_neutral_depth": onsets[gids[i]][
                                "first_neutral_order"
                            ],
                        }
                    )
            checks.append(
                {
                    "panel": mode,
                    "white": a,
                    "qualified": len(rec),
                    "unpinned_better": sum(r[1] for r in rec),
                    "erased": sum(r[0] for r in rec),
                    "nnue_on_erased": np.mean([r[2] for r in rec if r[0]])
                    if any(r[0] for r in rec)
                    else None,
                    "erased_beyond_13": sum(
                        r[0] and not onsets[r[3]]["first_neutral_order"] for r in rec
                    )
                    if mode != "enumerated"
                    else None,
                }
            )
    write(out / "pin-census.csv", counts)
    write(out / "pin-replication.csv", checks)
    (out / "played-pins.json").write_text(json.dumps(examples, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
