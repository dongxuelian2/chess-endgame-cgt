"""Enumerate legal king placements for each original rook / pawn / move template."""

import argparse
import csv
import gzip
import itertools
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.syzygy
from geometry import formula, sign
from measure import table
from siblings import trace_positions


def sweep(args):
    root, key, excluded = args
    source, p, first, second = key
    actor = chess.Piece(chess.ROOK, True)
    pawn = chess.Piece(chess.PAWN, False)
    moves = [chess.Move(source, sq) for sq in (first, second)]
    # The source can coincide with neither initial nor edited pawn square.
    free = set(range(64)) - {source, p, p - 8, first, second}
    rows = []
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        for wk, bk in itertools.permutations(sorted(free), 2):
            if chess.square_distance(wk, bk) <= 1 or (wk, bk) in excluded:
                continue
            b = chess.Board.empty()
            for sq, man in [
                (wk, chess.Piece(chess.KING, True)),
                (bk, chess.Piece(chess.KING, False)),
                (source, actor),
            ]:
                b.set_piece_at(sq, man)
            fens = []
            scores = []
            tags = []
            for location in (p, p - 8):
                b.set_piece_at(location, pawn)
                if not b.is_valid() or b.is_check():
                    break
                for m in moves:
                    if not b.is_legal(m) or b.gives_check(m):
                        break
                    b.push(m)
                    if any(b.legal_moves):
                        fens.append(b.fen())
                        scores.append(-sign(tb.probe_wdl(b)))
                        tags.append(formula(b))
                    b.pop()
                b.remove_piece_at(location)
            if len(fens) == 4:
                rows.append((wk, bk, fens, scores, tags))
    counts = defaultdict(int)
    fp = root / "pawn-geometry" / ("king-" + "-".join(map(str, key)) + ".gz")
    with gzip.open(fp, "wt") as out:
        for start in range(0, len(rows), 128):
            batch = rows[start : start + 128]
            t = trace_positions([f for _, _, fs, _, _ in batch for f in fs], root)
            for i, (wk, bk, _, w, tags) in enumerate(batch):
                s = [v[1] for v in t[4 * i : 4 * i + 4]]
                h, n = zip(*s)
                assert tuple(v[1] for v in tags) == h
                g = [w[0] - w[1], w[2] - w[3]]
                d = [n[1] - n[0], n[3] - n[2]]
                counts["valid"] += 1
                counts["nnue_direction"] += d[0] > d[1]
                if g[0] > 0 > g[1]:
                    counts["wdl_forward"] += 1
                    counts["nnue_both_forward"] += d[0] > 0 > d[1]
                    counts["hce_both_forward"] += h[1] > h[0] and h[3] < h[2]
                if g[0] < 0 < g[1]:
                    counts["wdl_opposite"] += 1
                print(json.dumps([wk, bk, w, n, h]), file=out)
    return {
        "template": "/".join(chess.square_name(s) for s in key),
        "valid": counts["valid"],
        "wdl_forward": counts["wdl_forward"],
        "wdl_opposite": counts["wdl_opposite"],
        "nnue_both_forward": counts["nnue_both_forward"],
        "hce_both_forward": counts["hce_both_forward"],
        "nnue_direction": counts["nnue_direction"],
    }


def main(root, dest):
    c = (
        Path(__file__).resolve().parents[1]
        / "circuits/results/handwritten-collisions.csv"
    )
    keys = defaultdict(set)
    seeds = defaultdict(list)
    for r in csv.DictReader(c.open()):
        if r["material"] != "KRvKP":
            continue
        board = chess.Board(r["parent"])
        a, b = [chess.Move.from_uci(r[k]) for k in ("move_a", "move_b")]
        k = (
            a.from_square,
            chess.Move.from_uci(r["relocation"]).from_square,
            a.to_square,
            b.to_square,
        )
        keys[k].add((board.king(True), board.king(False)))
        seeds[k].append(int(int(r["nnue_original"]) > 0 > int(r["nnue_edited"])))
    with ProcessPoolExecutor(8) as pool:
        out = list(pool.map(sweep, [(root, k, ks) for k, ks in keys.items()]))
    for k, r in zip(keys, out):
        r.update(original_n=len(seeds[k]), original_both_correct=sum(seeds[k]))
    table(dest / "king-sweep.csv", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    main(args.root, args.dest)
