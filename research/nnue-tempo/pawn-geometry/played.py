"""Test the frozen penalties on a played quiet move versus a legal alternative at the actual parent."""

import argparse
import gzip
import io
import json
import random
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.pgn
import chess.syzygy
import numpy as np
from channels import fold
from geometry import sign
from measure import load, table
from siblings import trace_positions
from threat_reversal import interval
from transfer import MATERIALS, square_class


def batches(root):
    for month in ("01", "02", "03"):
        count = 0
        screened = 0
        batch = []
        with subprocess.Popen(
            ["zstd", "-dc", root / f"lichess-2013-{month}.pgn.zst"],
            text=True,
            stdout=subprocess.PIPE,
        ) as p:
            for line in p.stdout:
                if line.startswith('[Site "'):
                    gid = line.split('"')[1].rsplit("/", 1)[-1]
                if not line.startswith("1."):
                    continue
                count += 1
                if line.count("x") < 27:
                    continue
                batch.append((month, gid, line))
                screened += 1
                if len(batch) >= 64:
                    yield batch
                    batch = []
            if batch:
                yield batch
            assert p.wait() == 0
        print(month, count, screened, flush=True)


def extract(batch):
    found = []
    for month, gid, line in batch:
        game = chess.pgn.read_game(io.StringIO(line))
        if game.errors:
            continue
        rng = random.Random("played-pawn-v1/" + gid)
        winners = {}
        counts = Counter()
        board = game.board()
        for actual in game.mainline_moves():
            count = board.occupied.bit_count()
            if count < 4:
                break
            if (
                count in (4, 5)
                and board.pawns
                and not (
                    board.castling_rights
                    or board.has_legal_en_passant()
                    or board.is_check()
                )
            ):
                kind = board.piece_type_at(actual.from_square)
                other = board.occupied_co[not board.turn]
                ours = board.occupied_co[board.turn]
                if (
                    kind in (chess.ROOK, chess.QUEEN, chess.KNIGHT, chess.BISHOP)
                    and other.bit_count() == 2
                    and other & board.pawns
                    and (count == 4 or ours & board.pawns)
                ):
                    mat = (
                        "K"
                        + chess.piece_symbol(kind).upper()
                        + ("P" if count == 5 else "")
                        + "vKP"
                    )
                    if (
                        mat in MATERIALS
                        and not board.is_capture(actual)
                        and not board.gives_check(actual)
                    ):
                        white = board if board.turn else board.mirror()
                        m = (
                            actual
                            if board.turn
                            else chess.Move(
                                chess.square_mirror(actual.from_square),
                                chess.square_mirror(actual.to_square),
                            )
                        )
                        choices = []
                        assert m in white.legal_moves
                        for move in list(
                            white.generate_legal_moves(
                                from_mask=chess.BB_SQUARES[m.from_square]
                            )
                        ):
                            if white.is_capture(move) or white.gives_check(move):
                                continue
                            white.push(move)
                            if any(white.legal_moves):
                                choices.append(move)
                            white.pop()
                        if m in choices and len(choices) >= 2:
                            counts[mat] += 1
                            if rng.randrange(counts[mat]) == 0:
                                alt = rng.choice([z for z in choices if z != m])
                                winners[mat] = {
                                    "month": month,
                                    "game": gid,
                                    "material": mat,
                                    "parent": white.fen(),
                                    "moves": [m.uci(), alt.uci()],
                                }
            board.push(actual)
        found.extend(winners.values())
    return found


def main(root, dest):
    known = set()
    for r in load(root, "KRvKP"):
        if fold(r) == 0:
            known.update([r["parent"].split()[0], r["edited_parent"].split()[0]])
    weights = json.loads((dest / "rule-weights.json").read_text())
    w = np.array([weights["capture"], weights["promotion"]])
    with ProcessPoolExecutor(10) as pool:
        rs = [
            r for batch in pool.map(extract, batches(root), chunksize=1) for r in batch
        ]
    rs = [
        r for r in rs if r["material"] != "KRvKP" or r["parent"].split()[0] not in known
    ]
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        tb.add_directory(str(root / "tb5"))
        for start in range(0, len(rs), 256):
            positions = []
            batch = rs[start : start + 256]
            for r in batch:
                board = chess.Board(r["parent"])
                pawn = next(iter(board.pieces(chess.PAWN, False)))
                labels = []
                tags = []
                for u in r["moves"]:
                    m = chess.Move.from_uci(u)
                    board.push(m)
                    positions.append(board.fen())
                    labels.append(-sign(tb.probe_wdl(board)))
                    cap, promo = square_class(board, m.to_square, pawn)
                    tags.append([int(cap), int(promo == "free")])
                    board.pop()
                r.update(labels=labels, tags=tags)
            tr = trace_positions(positions, root)
            for i, r in enumerate(batch):
                h, n = np.diff(np.array([tr[2 * i][1], tr[2 * i + 1][1]]), axis=0)[0]
                r["margins"] = [
                    float(h),
                    float(n),
                    float(h + (np.array(r["tags"][1]) - r["tags"][0]) @ w),
                ]
    with gzip.open(root / "pawn-geometry/played.gz", "wt") as f:
        for r in rs:
            print(json.dumps(r), file=f)
    out = []
    for mat in ["all", *MATERIALS]:
        data = [r for r in rs if mat == "all" or r["material"] == mat]
        if not data:
            continue
        y = np.array([np.sign(r["labels"][0] - r["labels"][1]) for r in data])
        groups = np.array([r["game"] for r in data])
        touched = np.array([r["tags"][0] != r["tags"][1] for r in data])
        score = np.array([r["margins"] for r in data]) * y[:, None]
        cr = (score > 0) + 0.5 * (score == 0)
        for k, mask in [
            ("all", y != 0),
            ("rule_differs", (y != 0) & touched),
            ("same_features", (y != 0) & ~touched),
        ]:
            if not mask.any():
                continue
            chance = cr[mask].mean(0)
            ci = interval((cr[:, 2] - cr[:, 0])[mask], groups[mask])
            out.append(
                {
                    "material": mat,
                    "subset": k,
                    "n_game_family": len(data),
                    "n_games": len(set(groups)),
                    "n_distinct_parents": len({r["parent"].split()[0] for r in data}),
                    "n_informative": int(mask.sum()),
                    "hce": chance[0],
                    "nnue": chance[1],
                    "two_rules": chance[2],
                    "gain": ci[0],
                    "low": ci[1],
                    "high": ci[2],
                }
            )
    table(dest / "played.csv", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    main(a.root, a.dest)
