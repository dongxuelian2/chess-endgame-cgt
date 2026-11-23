"""Transfer a frozen four-piece promotion term after adding a white pawn."""

import csv
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import credit
from summarize import write

from cohorts import orbit
from effects import batches, ci


def attached(a, row, rng):
    w, k, y, h, l, s = map(int, row)
    source = v.board(a, 1, w, k, y, s)
    source.turn = chess.WHITE
    for p in rng.permutation(np.arange(8, 56)):
        p = int(p)
        if p in (w, k, y, h, l, s):
            continue
        parent = source.copy(stack=False)
        parent.set_piece_at(p, chess.Piece(chess.PAWN, chess.WHITE))
        if not parent.is_valid():
            continue
        children = []
        for d in (h, l):
            m = chess.Move(s, d)
            if not parent.is_legal(m):
                break
            child = parent.copy(stack=False)
            child.push(m)
            if child.is_check() or not child.is_valid() or not any(child.legal_moves):
                break
            children.append(child)
        if len(children) == 2:
            yield parent.fen(), children


def promotion_options(board):
    surviving = False
    denies_reply = False
    for m in tuple(board.generate_legal_captures()):
        if m.promotion not in (chess.QUEEN, chess.ROOK):
            continue
        board.push(m)
        try:
            if not board.is_stalemate() and not any(
                board.generate_legal_captures(to_mask=chess.BB_SQUARES[m.to_square])
            ):
                surviving = True
                white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
                if not any(
                    move.promotion
                    for move in board.generate_legal_moves(from_mask=white_pawns)
                ):
                    denies_reply = True
        finally:
            board.pop()
    return surviving, denies_reply


def option(board):
    return promotion_options(board)[0]


def eval_positions(engine, fens):
    scores = []
    for group in batches(fens, 512):
        res = subprocess.run(
            [engine],
            input="".join(f"position fen {fen}\neval\n" for fen in group) + "quit\n",
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [
            list(map(int, line.split()[1:]))
            for line in res.stdout.splitlines()
            if line.startswith("scores ")
        ]
        assert len(rows) == len(group), res.stdout[-200:]
        scores.extend(rows)
    return np.array(scores)


def main(root, out):
    v.init(root)
    v.tb.add_directory(root / "tb5")
    with (out / "prediction-promotion.csv").open() as f:
        weight = next(
            int(r["weight"])
            for r in csv.DictReader(f)
            if r["sample"] == "discovery"
            and r["white"] == "0"
            and r["stratum"] == "all"
            and r["variant"] == "surviving_capture"
        )
    engine = root / "sf141/src/stockfish"
    stats, flags, witness = [], [], []
    for a in (2, 3, 4):
        panel = np.load(root / "manuscript" / f"validation-{a}-1.npz")
        rng = np.random.default_rng(26014 + a)
        truth = []
        positions = []
        option_count = np.zeros(3, dtype=int)
        examples = 0
        n_positions = 0
        for row in panel["rows"]:
            for parent, kids in zip(range(2), attached(a, row, rng)):
                fen, pair = kids
                outcomes = np.array([v.sign(v.tb.probe_wdl(q)) for q in pair])
                feat = np.array([option(q) for q in pair], dtype=int)
                n_positions += 2
                option_count += np.bincount(outcomes[feat == 1] + 1, minlength=3)
                if examples < 3 and np.any((feat == 1) & (outcomes <= 0)):
                    witness.append(
                        {
                            "white": a,
                            "parent": fen,
                            "child": pair[
                                int(np.flatnonzero((feat == 1) & (outcomes <= 0))[0])
                            ].fen(),
                            "black_wdl": int(
                                outcomes[(feat == 1) & (outcomes <= 0)][0]
                            ),
                        }
                    )
                    examples += 1
                direction = np.sign(outcomes[0] - outcomes[1])
                if direction:
                    truth.append(
                        (
                            orbit(*map(int, row[:2])),
                            direction,
                            direction * (feat[0] - feat[1]),
                        )
                    )
                    positions.extend(q.fen() for q in pair)
        flags.append(
            {
                "white": a,
                "positions": n_positions,
                "losses_with_option": int(option_count[0]),
                "draws_with_option": int(option_count[1]),
                "wins_with_option": int(option_count[2]),
            }
        )
        scores = eval_positions(engine, positions).reshape(-1, 2, 2)
        z = np.array(truth)
        m = (scores[:, 0, 1] - scores[:, 1, 1]) * z[:, 1]
        for label, mask in [
            ("all", np.ones(len(z), bool)),
            ("classical_tie", scores[:, 0, 0] == scores[:, 1, 0]),
            ("option_favors_H", z[:, 2] > 0),
            ("option_favors_L", z[:, 2] < 0),
            ("same", z[:, 2] == 0),
        ]:
            native = credit(m[mask])
            alt = credit(m[mask] + weight * z[mask, 2])
            if len(native):
                lo, hi = ci(z[mask, 0], alt - native)
                stats.append(
                    {
                        "white": a,
                        "stratum": label,
                        "weight": weight,
                        "n": len(native),
                        "native": native.mean(),
                        "adjusted": alt.mean(),
                        "corrected": int(((native < 1) & (alt == 1)).sum()),
                        "harmed": int(((native == 1) & (alt < 1)).sum()),
                        "lower": lo,
                        "upper": hi,
                    }
                )
    write(out / "prediction-five-piece.csv", stats)
    write(out / "prediction-five-piece-options.csv", flags)
    if witness:
        write(out / "prediction-five-piece-witnesses.csv", witness)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
