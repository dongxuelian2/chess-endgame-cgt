"""Census of WDL-reversing negative-tempo K+P versus K boards."""

import argparse
import gzip
import itertools
import json
from collections import Counter
from pathlib import Path

import chess
import chess.syzygy
from collect import evaluate


def main(root):
    count = 0
    rows = []
    negatives = []
    confusion = Counter()

    def flush():
        evaluate(rows, root / "sf12/src/stockfish", False)
        evaluate(rows, root / "sf141/src/stockfish", True)
        for row in rows:
            print(json.dumps([row["board"], row["wdl"]]), file=all_positions)
            truth = sum(row["wdl"])
            for engine in ("sf12", "sf141"):
                for col in (0, 1):
                    val = sum(s[col] for s in row[engine])
                    confusion[(engine, col, truth, (val > 0) - (val < 0))] += 1
            if truth < 0:
                negatives.append(row)
        rows.clear()

    with (
        chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb,
        gzip.open(root / "kpk-census.gz", "wt") as all_positions,
    ):
        b = chess.Board.empty()
        for pawn in range(8, 56):
            for wk, bk in itertools.permutations(range(64), 2):
                if pawn in (wk, bk) or chess.square_distance(wk, bk) <= 1:
                    continue
                b.clear_board()
                b.set_piece_at(wk, chess.Piece(chess.KING, True))
                b.set_piece_at(bk, chess.Piece(chess.KING, False))
                b.set_piece_at(pawn, chess.Piece(chess.PAWN, True))
                labels = []
                for turn in (True, False):
                    b.turn = turn
                    if not b.is_valid() or b.is_check() or not any(b.legal_moves):
                        break
                    labels.append(tb.probe_wdl(b))
                if len(labels) == 2:
                    count += 1
                    rows.append({"board": b.board_fen(), "wdl": labels})
                    if len(rows) >= 512:
                        flush()
        if rows:
            flush()
    rows = negatives
    for key, value in sorted(confusion.items()):
        print(*key, value)
    with gzip.open(root / "kpk-negative.json.gz", "wt") as out:
        json.dump(
            {
                "eligible": count,
                "examples": rows,
                "confusion": [[*k, v] for k, v in sorted(confusion.items())],
            },
            out,
        )
    print("eligible", count, "negative", len(rows))
    for engine in ("sf12", "sf141"):
        for column in (0, 1):
            d = [sum(x[column] for x in r[engine]) for r in rows]
            print(
                engine,
                column,
                "correct/tied/reversed",
                sum(x < 0 for x in d),
                d.count(0),
                sum(x > 0 for x in d),
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    main(p.parse_args().scratch)
