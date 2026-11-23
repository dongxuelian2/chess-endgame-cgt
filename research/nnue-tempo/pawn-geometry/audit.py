"""Replay the main and random samples and independently verify Syzygy and engine labels."""

import argparse
import re
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from channels import identity
from geometry import formula, sign
from measure import load, table
from transfer import MATERIALS


def verify(args):
    root, mat, random = args
    rows = load(root, ("random-" if random else "") + mat)
    counts = Counter()
    seen = set()
    test = []
    report = Counter()
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        tb.add_directory(str(root / "tb5"))
        for r in rows:
            key = (r["parent"], r["stratum"])
            assert key not in seen
            seen.add(key)
            parent = chess.Board(r["parent"])
            p = next(iter(parent.pieces(chess.PAWN, False)))
            edit = parent.copy()
            edit.set_piece_at(p - 8, edit.remove_piece_at(p))
            assert edit.fen() == r["edited_parent"]
            path = []
            promotion_reply = []
            for i, (board, loc) in enumerate([(parent, p), (edit, p - 8)]):
                assert board.is_valid() and not board.is_check()
                for j, uci in enumerate(r["moves"]):
                    move = chess.Move.from_uci(uci)
                    assert (
                        move in board.legal_moves
                        and not board.is_capture(move)
                        and not board.gives_check(move)
                    )
                    child = board.copy()
                    child.push(move)
                    prefix = "edited_" if i else ""
                    assert child.fen() == r[prefix + "children"][j]
                    val = -sign(tb.probe_wdl(child))
                    assert val == r["labels"][2 * i + j]
                    moves = list(child.legal_moves)
                    assert moves
                    pawn_takes = any(
                        m.from_square == loc
                        and m.to_square == move.to_square
                        and child.is_capture(m)
                        for m in moves
                    )
                    path.append(pawn_takes)
                    queen = next(
                        (
                            m
                            for m in moves
                            if m.from_square == loc
                            and m.to_square == loc - 8
                            and m.promotion == chess.QUEEN
                        ),
                        None,
                    )
                    if queen:
                        child.push(queen)
                        available = any(
                            m.from_square == move.to_square
                            and m.to_square == queen.to_square
                            for m in child.legal_moves
                        )
                        child.pop()
                    else:
                        available = None
                    promotion_reply.append(
                        "covered"
                        if available or move.to_square == loc - 8
                        else ("free" if queen else "other")
                    )
                    cut = r["restricted"][2 * i + j]
                    if cut is not None:
                        assert cut >= val
                    h, n = r[prefix + "sf141"][j]
                    units = r[prefix + "units"][j]
                    assert int((units[0] + units[1]) / 16) == n
                    if mat == "KRvKP":
                        assert h == formula(child)[1]
                    if len(test) < 180 and identity(r)[0] < 8:
                        test.append(
                            (
                                child.fen(),
                                np.array([h, n]) * (1 if child.turn else -1) / 208,
                            )
                        )
            ds = tuple(path)
            double = ds in [(0, 1, 1, 0), (1, 0, 0, 1)]
            single = sum(ds) == 1
            promo = not any(ds) and set(promotion_reply[2:]) == {"free", "covered"}
            if r["stratum"] == "double":
                assert ds == (0, 1, 1, 0)
            if r["stratum"] == "single":
                assert ds in ((0, 1, 0, 0), (0, 0, 1, 0))
            if r["stratum"] == "promotion":
                assert promo and promotion_reply[2] == "free"
            if r["stratum"] == "control":
                assert not any(ds) and chess.square_rank(p) > 2
            kind = (
                "double"
                if double
                else "single"
                if single
                else "promotion"
                if promo
                else "other"
            )
            counts[kind] += 1
            w = r["labels"]
            if (w[0] - w[1]) * (w[2] - w[3]) < 0:
                report[kind] += 1
    commands = "".join("position fen " + fen + "\neval\n" for fen, _ in test) + "quit\n"
    result = subprocess.run(
        [root / "clean-sf141/src/stockfish"],
        input=commands,
        capture_output=True,
        text=True,
        check=True,
    )
    native = []
    for name in ("Classical", "NNUE"):
        native.append(
            list(
                map(
                    float,
                    re.findall(
                        "^" + name + r" evaluation:?\s+([+-]?[\d.]+) \(white side\)",
                        result.stdout,
                        flags=re.MULTILINE,
                    ),
                )
            )
        )
    error = np.max(np.abs(np.array(native).T - np.array([v for _, v in test])))
    assert error <= 0.00500001
    return {
        "material": mat,
        "sample": "unconditioned" if random else "designed",
        "n": len(rows),
        "double": counts["double"],
        "single": counts["single"],
        "promotion": counts["promotion"],
        "other": counts["other"],
        "double_reversal": report["double"],
        "single_reversal": report["single"],
        "promotion_reversal": report["promotion"],
        "other_reversal": report["other"],
        "clean_engine_positions": len(test),
        "max_error_display_pawns": error,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    with ProcessPoolExecutor(8) as pool:
        rows = list(
            pool.map(verify, [(a.root, m, r) for m in MATERIALS for r in (False, True)])
        )
    table(a.dest / "audit.csv", rows)
