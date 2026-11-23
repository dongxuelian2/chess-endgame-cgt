"""Replay all four boards, verify exchange labels, and compare native evaluation with upstream."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from spectra import load


def capture_label(board, to, attacker):
    captures = [
        m for m in board.legal_moves if m.to_square == to and board.is_capture(m)
    ]
    vulnerable = False
    for m in captures:
        if m.from_square != attacker:
            continue
        board.push(m)
        vulnerable = not any(
            x.to_square == to and board.is_capture(x) for x in board.legal_moves
        )
        board.pop()
    return not captures, vulnerable


def verify(root, dataset):
    rows = load(root / "circuits" / f"reversals-{dataset}.gz")
    inputs = []
    rng = np.random.default_rng(11107)
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        tb.add_directory(str(root / "tb5"))
        for r in rows:
            a, b = map(chess.Move.from_uci, r["moves"])
            relocate = chess.Move.from_uci(r["edit"])
            original, edited = [
                chess.Board(r[key]) for key in ["parent", "edited_parent"]
            ]
            check = original.copy()
            piece = check.remove_piece_at(relocate.from_square)
            assert piece.color != original.turn and piece.piece_type != chess.KING
            assert check.piece_at(relocate.to_square) is None
            check.set_piece_at(relocate.to_square, piece)
            assert check.fen() == edited.fen()
            assert a.from_square == b.from_square and a.to_square != b.to_square
            for context, parent, prefix, attacker in [
                (0, original, "", relocate.from_square),
                (1, edited, "edited_", relocate.to_square),
            ]:
                assert parent.is_valid() and not parent.is_check()
                wdl = []
                for i, m in enumerate([a, b]):
                    assert (
                        parent.is_legal(m)
                        and not parent.is_capture(m)
                        and not parent.gives_check(m)
                    )
                    child = parent.copy()
                    child.push(m)
                    assert child.fen() == r[prefix + "children"][i]
                    assert any(child.legal_moves)
                    if dataset == "wdl":
                        wdl.append(-int(np.sign(tb.probe_wdl(child))))
                    else:
                        safe, loose = capture_label(child, m.to_square, attacker)
                        assert safe if i == context else loose
                    trace = r[prefix + "units"][i]
                    hce, nnue = r[prefix + "sf141"][i]
                    assert int((trace[0] + trace[1]) / 16) == nnue
                    if rng.random() < 0.03:
                        inputs.append(
                            (
                                child.fen(),
                                np.array([hce, nnue]) * (1 if child.turn else -1) / 208,
                            )
                        )
                if wdl:
                    assert wdl == r[prefix + "outcomes"] and (
                        wdl[0] > wdl[1] if context == 0 else wdl[1] > wdl[0]
                    )
    error = 0.0
    for i in range(0, len(inputs), 128):
        sample = inputs[i : i + 128]
        commands = (
            "".join("position fen " + fen + "\neval\n" for fen, _ in sample) + "quit\n"
        )
        run = subprocess.run(
            [root / "clean-sf141/src/stockfish"],
            input=commands,
            capture_output=True,
            text=True,
            check=True,
        )
        native = []
        for key in ("Classical", "NNUE"):
            native.append(
                [
                    float(v)
                    for v in re.findall(
                        "^" + key + r" evaluation:?\s+([+-]?[\d.]+) \(white side\)",
                        run.stdout,
                        flags=re.MULTILINE,
                    )
                ]
            )
        expected = np.array([v for _, v in sample])
        assert np.array(native).T.shape == expected.shape
        error = max(error, np.max(abs(np.array(native).T - expected)))
    assert error <= 0.00500001
    out = {
        "quadruples": len(rows),
        "independent_engine_positions": len(inputs),
        "max_error_display_pawns": error,
    }
    (root / "circuits" / f"audit-{dataset}.json").write_text(json.dumps(out))
    print(dataset, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("dataset")
    args = p.parse_args()
    verify(args.root, args.dataset)
