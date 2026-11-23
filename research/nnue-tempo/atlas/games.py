"""One played quiet man move per game, paired with a quiet alternative from the same square."""

import argparse
import gzip
import io
import json
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from pathlib import Path

import chess
import chess.pgn


def batches(root, months=range(4, 10)):
    for month in months:
        path = root / "atlas" / f"games-{month:02}.zst"
        games = screened = 0
        batch = []
        with subprocess.Popen(
            ["zstd", "-dc", path], text=True, stdout=subprocess.PIPE
        ) as run:
            for line in run.stdout:
                if line.startswith('[Site "'):
                    gid = line.split('"')[1].rsplit("/", 1)[-1]
                if not line.startswith("1."):
                    continue
                games += 1
                if line.count("x") < 28:
                    continue
                screened += 1
                batch.append((month, gid, line))
                if len(batch) == 80:
                    yield batch
                    batch = []
            if batch:
                yield batch
            assert run.wait() == 0
        print(month, games, screened, flush=True)


def extract(batch):
    output = []
    for month, gid, line in batch:
        rng = random.Random("atlas-games-v1" + gid)
        chosen = None
        n = 0
        game = chess.pgn.read_game(io.StringIO(line))
        if game.errors:
            continue
        b = game.board()
        for m in game.mainline_moves():
            if b.occupied.bit_count() < 4:
                break
            if (
                b.occupied.bit_count() == 4
                and b.occupied_co[b.turn].bit_count() == 2
                and b.piece_type_at(m.from_square) != 6
                and not (
                    b.castling_rights
                    or b.has_legal_en_passant()
                    or b.is_check()
                    or b.is_capture(m)
                    or b.gives_check(m)
                    or m.promotion
                )
            ):
                options = []
                for z in b.generate_legal_moves(
                    from_mask=chess.BB_SQUARES[m.from_square]
                ):
                    if b.is_capture(z) or b.gives_check(z) or z.promotion:
                        continue
                    b.push(z)
                    if any(b.legal_moves) and not b.has_legal_en_passant():
                        options.append(z)
                    b.pop()
                if m in options and len(options) > 1:
                    n += 1
                    if rng.randrange(n) == 0:
                        alt = rng.choice([z for z in options if z != m])
                        white = b if b.turn else b.mirror()
                        moves = [
                            q.uci()
                            if b.turn
                            else chess.Move(q.from_square ^ 56, q.to_square ^ 56).uci()
                            for q in (m, alt)
                        ]
                        chosen = (month, gid, white.fen(), *moves)
            b.push(m)
        if chosen:
            output.append(chosen)
    return output


def main(root):
    with (
        gzip.open(root / "atlas/played.gz", "wt") as f,
        ProcessPoolExecutor(10) as pool,
    ):
        for chunk in batched(batches(root), 80):
            for rows in pool.map(extract, chunk):
                for r in rows:
                    print(json.dumps(r), file=f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
