"""Move the pawn to a neighboring file, preserving candidate moves, kings, and pawn ranks."""

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from channels import arrays, fold, identity, mixed
from geometry import sign
from measure import load
from siblings import trace_positions
from transfer import MATERIALS, square_class


def placebo(r, tb):
    b = chess.Board(r["parent"])
    m, z = map(chess.Move.from_uci, r["moves"])
    pawn = next(iter(b.pieces(chess.PAWN, False)))
    piece = b.remove_piece_at(pawn)
    col = chess.square_file(pawn)
    rank = chess.square_rank(pawn)
    rng = random.Random("file-control/" + str(identity(r)))
    files = [f for f in range(8) if f != col]
    rng.shuffle(files)
    files.sort(key=lambda f: abs(f - col))
    for f in files:
        pp = chess.square(f, rank)
        children = []
        parents = []
        if any(s in (pp, pp - 8) for s in (m.to_square, z.to_square)) or any(
            b.piece_at(s) for s in (pp, pp - 8)
        ):
            continue
        for p in (pp, pp - 8):
            q = b.copy()
            q.set_piece_at(p, piece)
            if q.is_check() or not q.is_valid():
                break
            parents.append(q.fen())
            for move in (m, z):
                if not q.is_legal(move) or q.gives_check(move):
                    break
                q.push(move)
                capture, prom = square_class(q, move.to_square, p)
                good = not capture and any(q.legal_moves)
                if r["stratum"] == "promotion" and p == pp - 8:
                    good = good and prom == "free"
                if good:
                    children.append(q.fen())
                q.pop()
        if len(children) != 4:
            continue
        labels = [-sign(tb.probe_wdl(chess.Board(fen))) for fen in children]
        return dict(
            r,
            parent=parents[0],
            edited_parent=parents[1],
            children=children[:2],
            edited_children=children[2:],
            labels=labels,
        )


def dev(v):
    return (v[:, 0] - v[:, 1]) - (v[:, 2] - v[:, 3])


def summarize(orig, sham, root, stage, coef, readout):
    from collect import king_group
    from threat_reversal import interval

    result = {}
    groups = []
    for r in orig:
        b = chess.Board(r["parent"])
        groups.append(str(king_group(b.king(True), b.king(False))))
    for name, get in [
        ("wdl", lambda rows: dev(np.array([r["labels"] for r in rows]))),
        (
            "native",
            lambda rows: (
                -dev(
                    np.array(
                        [[s[1] for s in r["sf141"] + r["edited_sf141"]] for r in rows]
                    )
                )
            ),
        ),
        (
            "classical",
            lambda rows: (
                -dev(
                    np.array(
                        [[s[0] for s in r["sf141"] + r["edited_sf141"]] for r in rows]
                    )
                )
            ),
        ),
    ]:
        a, b = get(orig), get(sham)
        p = (a > b) + 0.5 * (a == b)
        result[name + "_effect_mean"] = float(np.mean(a - b))
        result[name + "_matched_rate"] = float(p.mean())
        result[name + "_matched_interval"] = interval(p, np.array(groups))[1:]
        result[name + "_real_positive"] = float(np.mean((a > 0) + 0.5 * (a == 0)))
        result[name + "_sham_positive"] = float(np.mean((b > 0) + 0.5 * (b == 0)))
    if readout:
        vals = []
        for rows in (orig, sham):
            x, _, _ = arrays(rows, root, stage)
            vals.append(mixed(x.astype(float)) @ coef)
        a, b = vals
        p = (a > b) + 0.5 * (a == b)
        result["probe_matched_rate"] = p.mean()
        result["probe_matched_interval"] = interval(p, np.array(groups))[1:]
        result["probe_real_positive"] = float(np.mean((a > 0) + 0.5 * (a == 0)))
        result["probe_sham_positive"] = float(np.mean((b > 0) + 0.5 * (b == 0)))
    return result


def run(args):
    root, dest, mat = args
    coef = np.array(json.loads((dest / "channel-masks.json").read_text())["coef"])
    allrows = load(root, mat)
    results = []
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        tb.add_directory(str(root / "tb5"))
        for group in ("double", "single", "promotion"):
            cases = sorted(
                [r for r in allrows if r["stratum"] == group and fold(r)], key=identity
            )[:500]
            if not cases:
                continue
            orig = []
            sham = []
            for r in cases:
                ctrl = placebo(r, tb)
                if ctrl:
                    orig.append(r)
                    sham.append(ctrl)
            for i in range(0, len(sham), 128):
                batch = sham[i : i + 128]
                tr = trace_positions(
                    [f for r in batch for f in r["children"] + r["edited_children"]],
                    root,
                )
                for j, r in enumerate(batch):
                    for prefix, k in (("", 0), ("edited_", 2)):
                        r[prefix + "units"], r[prefix + "sf141"] = map(
                            list, zip(*tr[4 * j + k : 4 * j + k + 2])
                        )
            results.append(
                dict(
                    material=mat,
                    design=group,
                    attempted=len(cases),
                    n=len(orig),
                    **summarize(
                        orig, sham, root, int("P" in mat.split("v")[0]), coef, True
                    ),
                )
            )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    with ProcessPoolExecutor(4) as pool:
        rows = [
            r
            for group in pool.map(run, [(args.root, args.dest, m) for m in MATERIALS])
            for r in group
        ]
    (args.dest / "matched-files.json").write_text(json.dumps(rows, indent=1) + "\n")
