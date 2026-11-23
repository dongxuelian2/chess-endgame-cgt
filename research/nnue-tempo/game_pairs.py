"""Replicate the paired-turn ranking on placements reached in rated Lichess games."""

import argparse
import gzip
import io
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import chess.syzygy
import numpy as np
from analyze import auc, gap
from collect import EXTENDED, MATERIAL, evaluate


def positions(movetext, rng):
    game = chess.pgn.read_game(io.StringIO(movetext))
    assert not game.errors
    board = game.board()
    chosen, seen = {}, Counter()
    for move in game.mainline_moves():
        board.push(move)
        if (
            len(board.piece_map()) > 5
            or board.has_legal_en_passant()
            or board.castling_rights
        ):
            continue
        key = chess.syzygy.calc_key(board)
        b = board.copy(stack=False)
        if key not in MATERIAL + EXTENDED:
            key = chess.syzygy.calc_key(board, mirror=True)
            if key not in MATERIAL + EXTENDED:
                continue
            b = board.mirror()
        b.ep_square = None
        b.halfmove_clock = 0
        for turn in (True, False):
            b.turn = turn
            if not b.is_valid() or b.is_check() or not any(b.legal_moves):
                break
        else:
            seen[key] += 1
            if rng.randrange(seen[key]) == 0:
                chosen[key] = b.board_fen()
    return chosen


def main(root, month, output):
    rng = random.Random(15012013)
    rows, games, hits = [], 0, 0
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        tb.add_directory(str(root / "tb5"))
        with subprocess.Popen(
            ["zstd", "-dc", root / f"lichess-{month}.pgn.zst"],
            text=True,
            stdout=subprocess.PIPE,
        ) as proc:
            headers = 0
            for line in proc.stdout:
                headers += line.startswith("[Event ")
                if not line.startswith("1."):
                    continue
                games += 1
                if line.count("x") < 27:
                    continue
                by_material = positions(line, rng)
                hits += bool(by_material)
                for m, fen in by_material.items():
                    wdl = [
                        tb.probe_wdl(chess.Board(f"{fen} {t} - - 0 1")) for t in "wb"
                    ]
                    rows.append(
                        {"material": m, "board": fen, "game": games, "wdl": wdl}
                    )
            assert proc.wait() == 0 and headers == games
    for i in range(0, len(rows), 512):
        evaluate(rows[i : i + 512], root / "sf141/src/stockfish", True)
    print("games", games, "eligible games", hits, "game-family pairs", len(rows))
    with gzip.open(root / output, "wt") as f:
        for row in rows:
            print(json.dumps(row), file=f)
    for m in MATERIAL + EXTENDED:
        sample = [r for r in rows if r["material"] == m]
        y = np.array([sum(w if abs(w) == 2 else 0 for w in r["wdl"]) for r in sample])
        scores = np.array([r["sf141"] for r in sample]).sum(axis=1)
        keep = y >= 0
        if np.any(y > 0) and np.any(y == 0):
            h, n = scores[keep].T
            labels = y[keep] > 0
            print(
                m,
                len(y),
                int(labels.sum()),
                int((~keep).sum()),
                auc(labels, h),
                auc(labels, n),
                gap(labels, n, h, np.arange(len(h))),
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    p.add_argument("--month", default="2013-01")
    p.add_argument("--output", default="games.jsonl.gz")
    args = p.parse_args()
    main(args.scratch, args.month, args.output)
