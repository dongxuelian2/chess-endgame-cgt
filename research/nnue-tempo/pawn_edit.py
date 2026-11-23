"""Shift one black pawn, re-evaluate, and transplant H1 coordinates across the edit."""

import argparse
import gzip
import json
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from analyze import auc, stack_weights
from collect import evaluate


def paired_effects(old, new, weights):
    b2, w2, _, w3 = weights

    def score(x):
        return (np.clip((x @ w2.T + b2) >> 6, 0, 127) @ w3).sum(1) / 16

    old_score, new_score = score(old), score(new)
    values = []
    frozen = None
    for j in range(8):
        fwd, rev = old.copy(), new.copy()
        fwd[:, :, j], rev[:, :, j] = new[:, :, j], old[:, :, j]
        values.append(((score(fwd) - old_score + new_score - score(rev)) / 2).mean())
        if j == 2:
            frozen = score(rev) - old_score
    return new_score - old_score, values, frozen


def process(rows, rank, root, tb, weights):
    sources, targets = [], []
    for row in rows:
        if not row["fold"]:
            continue
        b = chess.Board(row["board"] + " b - - 0 1")
        pawn = b.pieces(chess.PAWN, False).pop()
        if chess.square_rank(pawn) + 1 != rank or b.piece_at(pawn - 8):
            continue
        b.remove_piece_at(pawn)
        b.set_piece_at(pawn - 8, chess.Piece(chess.PAWN, False))
        labels = []
        for i, turn in enumerate((True, False)):
            b.turn = turn
            if not b.is_valid() or b.is_check() or not any(b.legal_moves):
                break
            if any(b.generate_legal_captures()) != row["capture"][i]:
                break
            labels.append(tb.probe_wdl(b))
        if len(labels) == 2:
            sources.append(row)
            targets.append({"board": b.board_fen(), "wdl": labels})
    evaluate(targets, root / "sf141/src/stockfish", True)
    old, new = (
        np.array([r["units"] for r in batch])[:, :, 2:10]
        for batch in (sources, targets)
    )
    changes, patches, frozen = paired_effects(old, new, weights)
    before, after = [
        np.array([sum(r["wdl"]) for r in batch]) for batch in (sources, targets)
    ]
    retained = after >= before
    y = after[retained] > before[retained]
    return {
        "n": len(targets),
        "source_means": old[:, :, 2].mean(0).tolist(),
        "target_means": new[:, :, 2].mean(0).tolist(),
        "delta_internal_units": changes.mean(),
        "symmetric_patch_effects": patches,
        "auc": auc(y, changes[retained]),
        "frozen_auc": auc(y, frozen[retained]),
        "increased_equal_decreased": [
            int((after > before).sum()),
            int((after == before).sum()),
            int((after < before).sum()),
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    root = p.parse_args().scratch
    with gzip.open(root / "KRvKP.jsonl.gz") as f:
        rows = list(map(json.loads, f))
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        print(
            json.dumps(
                {
                    rank: process(rows, rank, root, tb, stack_weights(root, 0))
                    for rank in (3, 5)
                },
                indent=2,
            )
        )
