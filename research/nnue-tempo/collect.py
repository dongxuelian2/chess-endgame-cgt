"""Sample paired endgames and record native static scores and Syzygy labels."""

import argparse
import gzip
import hashlib
import json
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import chess
import chess.syzygy

MATERIAL = ["KPvK", "KPvKP", "KNvKP", "KBvKP", "KRvKP", "KRvKR"]


EXTENDED = [
    "KRPvKR",
    "KNPvKP",
    "KBPvKP",
    "KPPvKP",
    "KQPvKQ",
    "KRPvKP",
    "KRPPvK",
    "KRNvKR",
]


def king_group(white, black):
    # Fold board-axis reflections and interchange of the kings.
    return min(tuple(sorted((white ^ mask, black ^ mask))) for mask in (0, 7, 56, 63))


def collect(material, n, tb, rng):
    white, black = material.split("v")
    pieces = [chess.Piece.from_symbol(p) for p in white + black.lower()]
    used = set()
    while len(used) < n:
        squares = tuple(rng.sample(range(64), len(pieces)))
        key = tuple(sorted(zip(white + black.lower(), squares)))
        if key in used:
            continue
        b = chess.Board.empty()
        for p, sq in zip(pieces, squares):
            b.set_piece_at(sq, p)
        labels, captures, promotions = [], [], []
        for turn in (chess.WHITE, chess.BLACK):
            b.turn = turn
            if not b.is_valid() or b.is_check() or not any(b.legal_moves):
                break
            labels.append(tb.probe_wdl(b))
            captures.append(any(b.generate_legal_captures()))
            promotions.append(any(m.promotion for m in b.legal_moves))
        if len(labels) != 2:
            continue
        used.add(key)
        group = king_group(b.king(chess.WHITE), b.king(chess.BLACK))
        fold = hashlib.sha256(bytes(group)).digest()[0] % 2
        yield {
            "board": b.board_fen(),
            "material": material,
            "wdl": labels,
            "capture": captures,
            "promotion": promotions,
            "fold": fold,
            "group": group,
        }


def evaluate(rows, engine, units):
    commands = ["uci", "isready"]
    for r in rows:
        for turn in "wb":
            commands.extend([f"position fen {r['board']} {turn} - - 0 1", "eval"])
    commands.append("quit")
    res = subprocess.run(
        [engine.resolve()],
        input="\n".join(commands) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    scores, activations = [], []
    for line in res.stdout.splitlines():
        if line.startswith("scores "):
            scores.append(list(map(int, line.split()[1:])))
        if line.startswith("units "):
            activations.append(list(map(int, line.split()[1:])))
    assert len(scores) == 2 * len(rows), res.stdout[-1000:]
    if units:
        assert len(activations) == len(scores) and all(
            len(v) == 42 for v in activations
        )
    for i, row in enumerate(rows):
        pair = slice(2 * i, 2 * i + 2)
        row["sf141" if units else "sf12"] = scores[pair]
        if units:
            row["units"] = activations[pair]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    p.add_argument("--count", type=int, default=16000)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--extended", action="store_true")
    args = p.parse_args()
    root = args.scratch
    materials = EXTENDED if args.extended else MATERIAL
    with ProcessPoolExecutor(args.workers) as pool:
        list(
            pool.map(
                partial(run_material, root=root, count=args.count, seed=args.seed),
                materials,
            )
        )


def run_material(material, root, count, seed):
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        if material in EXTENDED:
            tb.add_directory(str(root / "tb5"))
        rng = random.Random(f"{seed}:{material}")
        rows = collect(material, count, tb, rng)
        with gzip.open(root / f"{material}.jsonl.gz", "wt") as out:
            for i in range(0, count, 512):
                batch = [next(rows) for _ in range(min(512, count - i))]
                evaluate(batch, root / "sf12/src/stockfish", False)
                evaluate(batch, root / "sf141/src/stockfish", True)
                for row in batch:
                    print(json.dumps(row, separators=(",", ":")), file=out)
        print(material, count, flush=True)


if __name__ == "__main__":
    main()
