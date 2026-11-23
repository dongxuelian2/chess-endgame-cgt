"""Quiet moves by the same piece from a common parent, labeled by Syzygy."""

import argparse
import gzip
import json
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import chess
import chess.syzygy
from collect import EXTENDED, MATERIAL


def outcomes(board, tb):
    target = (tb.probe_wdl(board) > 0) - (tb.probe_wdl(board) < 0)
    options = defaultdict(list)
    for move in list(board.legal_moves):
        if move.promotion or board.is_capture(move) or board.gives_check(move):
            continue
        board.push(move)
        if not board.has_legal_en_passant() and any(board.legal_moves):
            outcome = tb.probe_wdl(board)
            options[move.from_square].append(
                (
                    move.uci(),
                    board.fen(),
                    int(outcome < 0) - int(outcome > 0),
                    not any(board.generate_legal_captures()),
                )
            )
        board.pop()
    return [
        (best, worse)
        for moves in options.values()
        for best in moves
        if best[2] == target
        for worse in moves
        if worse[2] < target
    ]


def trace_positions(fens, root):
    commands = "".join(f"position fen {fen}\neval\n" for fen in fens)
    run = subprocess.run(
        [root / "sf141/src/stockfish"],
        input=commands + "quit\n",
        text=True,
        capture_output=True,
        check=True,
    )
    units = [
        list(map(int, s.split()[1:]))
        for s in run.stdout.splitlines()
        if s.startswith("units ")
    ]
    scores = [
        list(map(int, s.split()[1:]))
        for s in run.stdout.splitlines()
        if s.startswith("scores ")
    ]
    assert len(units) == len(scores) == len(fens)
    return list(zip(units, scores))


def sample(material, root, limit):
    rng = random.Random("quiet-siblings-v1/" + material)
    with gzip.open(root / (material + ".jsonl.gz")) as f:
        rows = [r for r in map(json.loads, f) if r["fold"] == 1]
    rng.shuffle(rows)
    selected = []
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        tb.add_directory(str(root / "tb5"))
        for row in rows[:limit]:
            turn = rng.choice("wb")
            b = chess.Board(row["board"] + " " + turn + " - - 0 1")
            choices = outcomes(b, tb)
            if choices:
                a, z = rng.choice(choices)
                selected.append(
                    {
                        "parent": b.fen(),
                        "group": row["group"],
                        "material": material,
                        "moves": [a[0], z[0]],
                        "children": [a[1], z[1]],
                        "outcomes": [a[2], z[2]],
                        "safe": a[3] and z[3],
                    }
                )
    for i in range(0, len(selected), 512):
        batch = selected[i : i + 512]
        traces = trace_positions([f for r in batch for f in r["children"]], root)
        for j, r in enumerate(batch):
            r["units"], r["sf141"] = map(list, zip(*traces[2 * j : 2 * j + 2]))
    with gzip.open(root / "circuits" / f"{material}.siblings.gz", "wt") as f:
        for row in selected:
            print(json.dumps(row, separators=(",", ":")), file=f)
    return material, len(selected)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--screen", type=int, default=6000)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    (args.root / "circuits").mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(
            partial(sample, root=args.root, limit=args.screen), MATERIAL + EXTENDED
        ):
            print(*result, flush=True)
