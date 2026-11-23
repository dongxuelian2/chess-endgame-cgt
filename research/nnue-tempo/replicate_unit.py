"""Freeze the KRvKP hypothesis and retest on 24,000 new placements."""

import argparse
import gzip
import json
import random
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from analyze import auc, stack_weights
from collect import collect, evaluate
from pawn_edit import process
from surrogate import concepts


def mirror_key(board_fen):
    board = chess.Board(board_fen + " w - - 0 1")
    return min(board_fen, board.transform(chess.flip_horizontal).board_fen())


def replicate(root):
    with gzip.open(root / "KRvKP.jsonl.gz") as stream:
        original = list(map(json.loads, stream))
    seen = {mirror_key(r["board"]) for r in original}
    weights = stack_weights(root, 0)
    b2, w2, _, w3 = weights
    fit = json.loads((root / "surrogate.json").read_text())["all"]
    mean = json.loads((root / "results.json").read_text())["KRvKP"]["lesions"]["1"][
        "means"
    ][2]
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        new = []
        for row in collect(
            "KRvKP", 100000, tb, random.Random("KRvKP-frozen-verification-01")
        ):
            if not row["fold"]:
                continue
            key = mirror_key(row["board"])
            if key in seen:
                continue
            seen.add(key)
            new.append(row)
            if len(new) == 24000:
                break
        assert len(new) == 24000
        for i in range(0, len(new), 512):
            evaluate(new[i : i + 512], root / "sf141/src/stockfish", True)
        edits = process(new, 3, root, tb, weights)
    h = np.array([r["units"] for r in new])[:, :, 2:10]
    x = np.array([concepts(r["board"]) for r in new])
    blocks = [np.ones((len(x), 1))] + [
        x[:, j, None] == levels[1:] for j, levels in enumerate(fit["levels"])
    ]
    pred = np.rint(
        np.clip(np.column_stack(blocks) @ np.array(fit["coefficients"]), 0, 127)
    ).astype(int)
    delta = np.array([sum(r["wdl"]) for r in new])
    rates = {}
    for name, replacement in [
        ("intact", h[:, :, 2].copy()),
        ("clamp", mean),
        ("surrogate", pred),
    ]:
        h[:, :, 2] = replacement
        s = (np.clip((h @ w2.T + b2) >> 6, 0, 127) @ w3).sum(1)
        rates[name] = auc(delta[delta >= 0] > 0, s[delta >= 0])
    return {
        "n": len(new),
        "positive": int((delta > 0).sum()),
        "negative": int((delta < 0).sum()),
        "auc": rates,
        "pawn_edit": edits,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scratch", type=Path)
    print(json.dumps(replicate(parser.parse_args().scratch), indent=2))
