"""Keep two choices fixed and relocate one enemy piece until their exact WDL order reverses."""

import argparse
import gzip
import json
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from collect import EXTENDED, MATERIAL
from siblings import trace_positions
from spectra import load


def reverse(row, tb, rng):
    board = chess.Board(row["parent"])
    a, b = map(chess.Move.from_uci, row["moves"])
    if board.piece_type_at(a.from_square) == chess.KING:
        return
    candidates = []
    empty = ~board.occupied & chess.BB_ALL
    for source in chess.scan_forward(board.occupied_co[not board.turn] & ~board.kings):
        p = board.piece_at(source)
        if p.piece_type == chess.PAWN:
            dest = source + (8 if p.color else -8)
            squares = chess.BB_SQUARES[dest] & ~(chess.BB_RANK_1 | chess.BB_RANK_8)
        else:
            squares = board.attacks_mask(source)
        for dest in chess.scan_forward(squares & empty):
            if dest not in (a.to_square, b.to_square):
                candidates.append((source, dest))
    rng.shuffle(candidates)
    for source, dest in candidates:
        edited = board.copy(stack=False)
        edited.set_piece_at(dest, edited.remove_piece_at(source))
        if not edited.is_valid() or edited.is_check():
            continue
        outcomes = []
        children = []
        safe = True
        for move in (a, b):
            if not edited.is_legal(move) or edited.gives_check(move):
                break
            edited.push(move)
            if edited.has_legal_en_passant() or not any(edited.legal_moves):
                break
            # From the player who made this move.
            outcomes.append(-int(np.sign(tb.probe_wdl(edited))))
            children.append(edited.fen())
            safe = safe and not any(edited.generate_legal_captures())
            edited.pop()
        if len(outcomes) == 2 and outcomes[1] > outcomes[0]:
            return dict(
                row,
                stack=int(row["material"] in EXTENDED),
                edit=chess.square_name(source) + chess.square_name(dest),
                edited_parent=edited.fen(),
                edited_children=children,
                edited_outcomes=outcomes,
                capture_free=row["safe"] and safe,
            )


def collect_one(root, material):
    directory = root / "circuits"
    rows = load(directory / f"{material}.siblings.gz")
    rng = random.Random("exact-relative-v1/" + material)
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        tb.add_directory(str(root / "tb5"))
        result = [r for row in rows if (r := reverse(row, tb, rng)) is not None]
    for i in range(0, len(result), 256):
        group = result[i : i + 256]
        traces = trace_positions(
            [fen for r in group for fen in r["edited_children"]], root
        )
        for j, r in enumerate(group):
            r["edited_units"], r["edited_sf141"] = map(
                list, zip(*traces[2 * j : 2 * j + 2])
            )
    return material, result


def generate(root):
    directory = root / "circuits"
    with (
        ProcessPoolExecutor(8) as pool,
        gzip.open(directory / "reversals-wdl.gz", "wt") as f,
    ):
        for material, rows in pool.map(partial(collect_one, root), MATERIAL + EXTENDED):
            for r in rows:
                print(json.dumps(r, separators=(",", ":")), file=f)
            print(material, len(rows), flush=True)


def summarize(root):
    from threat_reversal import interval, statistics

    directory = root / "circuits"
    rows = load(directory / "reversals-wdl.gz")
    out = {}
    for key in ["all", "capture_free"] + MATERIAL + EXTENDED:
        rs = [
            r
            for r in rows
            if key == "all"
            or key == r["material"]
            or (key == "capture_free" and r["capture_free"])
        ]
        if not rs:
            out[key] = {"n": 0}
            continue
        strategies = {}
        strict = {}
        for i, k in enumerate(("classical", "native")):
            pairs = np.stack(
                [
                    np.array([r[col][1][i] - r[col][0][i] for r in rs])
                    for col in ("sf141", "edited_sf141")
                ]
            )
            strategies[k] = statistics(pairs)
            strict[k] = ((pairs[0] > 0) & (pairs[1] < 0)).astype(float)
        groups = np.array([str(r["group"]) for r in rs])
        out[key] = {
            "n": len(rs),
            "metrics": strategies,
            "both_net_minus_hce": interval(
                strict["native"] - strict["classical"], groups
            ),
        }
    (directory / "wdl-reversal.json").write_text(json.dumps(out, allow_nan=False))
    print({k: out[k] for k in ["all", "capture_free"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    if not args.summarize_only:
        generate(args.root)
    summarize(args.root)
