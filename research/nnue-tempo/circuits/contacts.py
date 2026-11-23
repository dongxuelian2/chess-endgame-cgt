"""Matched quiet moves that expose or shelter the moved piece in played positions."""

import argparse
import gzip
import io
import json
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import chess
import chess.pgn
from siblings import trace_positions


def matched_pair(board, rng, forks):
    options = defaultdict(lambda: [[], []])
    for move in list(board.legal_moves):
        kind = board.piece_type_at(move.from_square)
        if (
            kind in (chess.PAWN, chess.KING)
            or board.is_capture(move)
            or board.gives_check(move)
        ):
            continue
        board.push(move)
        square = chess.BB_SQUARES[move.to_square]
        captures = list(board.generate_legal_captures(to_mask=square))
        if forks:
            if not captures and any(board.legal_moves):
                targets = (
                    board.attacks_mask(move.to_square)
                    & board.occupied_co[board.turn]
                    & ~(board.kings | board.pawns)
                )
                count = targets.bit_count()
                if count != 1:
                    options[move.from_square][int(count == 0)].append(move)
        elif not captures:
            if any(board.legal_moves):
                options[move.from_square][0].append(move)
        else:
            for capture in captures:
                board.push(capture)
                loose = not any(board.generate_legal_captures(to_mask=square))
                board.pop()
                if loose:
                    options[move.from_square][1].append(move)
                    break
        board.pop()
    pairs = [(a, b) for safe, loose in options.values() for a in safe for b in loose]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    fens = []
    for move in (a, b):
        board.push(move)
        fens.append(board.fen())
        board.pop()
    return {
        "moves": [a.uci(), b.uci()],
        "children": fens,
        "parent": board.fen(),
        "piece": board.piece_at(a.from_square).symbol().upper(),
    }


def process_batch(items, forks):
    result = []
    for game_id, text in items:
        rng = random.Random("loose-v1/" + game_id)
        game = chess.pgn.read_game(io.StringIO(text))
        if game.errors:
            continue
        board = game.board()
        chosen, seen = {}, defaultdict(int)
        for move in game.mainline_moves():
            board.push(move)
            if board.is_check():
                continue
            stack = (board.occupied.bit_count() - 1) // 4
            seen[stack] += 1
            if rng.randrange(seen[stack]) == 0:
                chosen[stack] = board.copy(stack=False)
        for stack, b in chosen.items():
            pair = matched_pair(
                b, random.Random(f"{game_id}/{stack}/fork") if forks else rng, forks
            )
            if pair:
                result.append(dict(**pair, game=game_id, stack=stack))
    return result


def batches(path, stride):
    batch = []
    count = 0
    with subprocess.Popen(
        ["zstd", "-dc", path], stdout=subprocess.PIPE, text=True
    ) as p:
        for line in p.stdout:
            if line.startswith('[Site "'):
                game_id = line.split('"')[1].rsplit("/", 1)[-1]
            if not line.startswith("1."):
                continue
            count += 1
            if count % stride:
                continue
            batch.append((game_id, line))
            if len(batch) == 64:
                yield batch
                batch = []
        if batch:
            yield batch
        assert p.wait() == 0
    print("games screened", count, "sampled", count // stride, flush=True)


def main(root, month, workers, stride, forks):
    name = "forks" if forks else "contacts"
    output = root / "circuits" / f"{name}-{month}.gz"
    counts = defaultdict(int)
    with ProcessPoolExecutor(workers) as pool, gzip.open(output, "wt") as out:
        for batch in pool.map(
            partial(process_batch, forks=forks),
            batches(root / f"lichess-{month}.pgn.zst", stride),
        ):
            if not batch:
                continue
            traces = trace_positions(
                [fen for r in batch for fen in r["children"]], root
            )
            for i, r in enumerate(batch):
                r["units"], r["sf141"] = map(list, zip(*traces[2 * i : 2 * i + 2]))
                print(json.dumps(r, separators=(",", ":")), file=out)
                counts[r["stack"]] += 1
    print(sorted(counts.items()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("month")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--forks", action="store_true")
    args = p.parse_args()
    main(args.root, args.month, args.workers, args.stride, args.forks)
