"""Replace KRvKP unit 2 with an additive, chess-readable lookup function."""

import argparse
import gzip
import json
from pathlib import Path

import chess
import numpy as np
from analyze import auc, stack_weights


def concepts(fen):
    b = chess.Board(fen + " w - - 0 1")
    (pawn,) = b.pieces(chess.PAWN, False)
    (rook,) = b.pieces(chess.ROOK, True)
    same_file = chess.square_file(rook) == chess.square_file(pawn)
    return (
        chess.square_rank(pawn),
        chess.square_distance(b.king(True), pawn),
        chess.square_distance(b.king(False), pawn),
        int(same_file and rook < pawn),
        int(same_file and rook > pawn),
        int(b.is_attacked_by(True, pawn)),
        int(b.is_attacked_by(False, rook)),
    )


def main(root):
    with gzip.open(root / "KRvKP.jsonl.gz") as f:
        rows = list(map(json.loads, f))
    synthetic_count = len(rows)
    with gzip.open(root / "games.jsonl.gz") as f:
        rows.extend(r for r in map(json.loads, f) if r["material"] == "KRvKP")
    x = np.array([concepts(r["board"]) for r in rows])
    h = np.array([r["units"] for r in rows])[:, :, 2:10]
    t = h[:, :, 2]
    train = np.array([r.get("fold") == 0 for r in rows])
    played = np.arange(len(rows)) >= synthetic_count
    held = (~train) & (~played)
    y = np.array([sum(r["wdl"]) > 0 for r in rows])
    b2, w2, _, w3 = stack_weights(root, 0)
    out = {}
    for name, cols in [
        ("mean", 0),
        ("pawn_rank", 1),
        ("king_distances_and_rank", 3),
        ("all", 7),
    ]:
        # Reference-cell coding gives one intercept and a per-category adjustment.
        blocks = [np.ones((len(x), 1))]
        levels_used = []
        for j in range(cols):
            levels = np.unique(x[train, j])
            levels_used.append(levels.tolist())
            blocks.append((x[:, j, None] == levels[1:]).astype(float))
        design = np.column_stack(blocks)
        coef = np.linalg.lstsq(design[train], t[train], rcond=None)[0]
        prediction = np.rint(np.clip(design @ coef, 0, 127)).astype(int)
        out[name] = {
            "parameters_per_turn": design.shape[1],
            "levels": levels_used,
            "coefficients": coef.tolist(),
        }
        for cohort, test in [("held", held), ("played", played)]:
            h_new = h[test].copy()
            h_new[:, :, 2] = prediction[test]
            score = (np.clip((h_new @ w2.T + b2) >> 6, 0, 127) @ w3).sum(1)
            r2 = 1 - ((t[test] - prediction[test]) ** 2).sum(0) / (
                (t[test] - t[test].mean(0)) ** 2
            ).sum(0)
            out[name][cohort] = {"auc": auc(y[test], score), "r2": r2.tolist()}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    main(p.parse_args().scratch)
