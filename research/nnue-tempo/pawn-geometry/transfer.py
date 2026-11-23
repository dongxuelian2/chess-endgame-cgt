"""Generate and label the frozen pawn displacement transfer experiment."""

import argparse
import csv
import gzip
import itertools
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.syzygy
from geometry import formula, pawn_path, sign
from siblings import trace_positions

MATERIALS = ("KRvKP", "KQvKP", "KBvKP", "KNvKP", "KRPvKP", "KBPvKP", "KNPvKP")
STRATA = ("double", "single", "promotion", "control")
LIMIT = 1500


def square_class(b, actor, pawn):
    legal_capture = any(
        m.from_square == pawn
        for m in b.generate_legal_captures(to_mask=chess.BB_SQUARES[actor])
    )
    if chess.square_rank(pawn) != 1:
        return legal_capture, None
    end = pawn - 8
    if actor == end:
        return legal_capture, "blocked"
    move = chess.Move(pawn, end, promotion=chess.QUEEN)
    if not b.is_legal(move):
        return legal_capture, "other"
    b.push(move)
    recapture = any(
        m.from_square == actor
        for m in b.generate_legal_captures(to_mask=chess.BB_SQUARES[end])
    )
    b.pop()
    return legal_capture, "recapture" if recapture else "free"


def choices(b, e, source, p, needed, rng):
    options = []
    source_mask = chess.BB_SQUARES[source]
    for m in b.generate_legal_moves(from_mask=source_mask):
        if (
            b.is_capture(m)
            or not e.is_legal(m)
            or e.is_capture(m)
            or b.gives_check(m)
            or e.gives_check(m)
        ):
            continue
        b.push(m)
        e.push(m)
        if any(b.legal_moves) and any(e.legal_moves):
            orig, _ = square_class(b, m.to_square, p)
            edit, promo = square_class(e, m.to_square, p - 8)
            cover = pawn_path(e, m.to_square, p - 8)[-1]
            options.append((m, orig, edit, promo, cover))
        b.pop()
        e.pop()
    if "unconditioned" in needed:
        if len(options) < 2:
            return {}
        a, z = rng.sample(options, 2)
        return {"unconditioned": (a[0], z[0])}
    pools = {k: [] for k in STRATA if k in needed}
    for a, z in itertools.permutations(options, 2):
        pat = (a[1], z[1], a[2], z[2])
        if pat == (0, 1, 1, 0):
            name = "double"
        elif pat in ((0, 1, 0, 0), (0, 0, 1, 0)):
            name = "single"
        elif any(pat):
            continue
        elif (
            chess.square_rank(p) == 2
            and a[3] == "free"
            and z[3] in ("recapture", "blocked")
        ):
            name = "promotion"
        elif chess.square_rank(p) > 2 and a[4] == z[4]:
            name = "control"
        else:
            continue
        if name in pools:
            pools[name].append((a[0], z[0]))
    return {k: rng.choice(s) for k, s in pools.items() if s}


def restricted(board, tb, actor, pawn, kind, native):
    def excluded(m):
        return m.from_square == pawn and (
            m.to_square == actor if kind == "capture" else bool(m.promotion)
        )

    moves = list(board.legal_moves)
    if not any(map(excluded, moves)):
        return native
    remaining = [m for m in moves if not excluded(m)]
    if not remaining:
        return None
    best = -1
    missing = False
    for m in sorted(remaining, key=lambda x: bool(x.promotion)):
        board.push(m)
        try:
            value = -sign(tb.probe_wdl(board))
        except chess.syzygy.MissingTableError:
            missing = True
        else:
            best = max(best, value)
        board.pop()
        if best == native:
            return best
    return None if missing else best


def record(b, e, p, ms, tb, stratum):
    children = []
    labels = []
    cut = []
    branches = []
    for parent, pawn in ((b, p), (e, p - 8)):
        for m in ms:
            parent.push(m)
            children.append(parent.fen())
            value = sign(tb.probe_wdl(parent))
            labels.append(-value)
            kind = "promotion" if stratum == "promotion" else "capture"
            v = restricted(parent, tb, m.to_square, pawn, kind, value)
            cut.append(-v if v is not None else None)
            if len(parent.piece_map()) == 4 and parent.rooks:
                branches.append(formula(parent))
            parent.pop()
    return {
        "parent": b.fen(),
        "edited_parent": e.fen(),
        "moves": [m.uci() for m in ms],
        "children": children[:2],
        "edited_children": children[2:],
        "labels": labels,
        "restricted": cut,
        "stratum": stratum,
        "branches": branches,
    }


def collect(args):
    root, material, unconditioned = args
    rng = random.Random(
        ("random-pair-v1/" if unconditioned else "pawn-geometry-v1/") + material
    )
    actor_piece = chess.Piece.from_symbol(material[1])
    extra = material.split("v")[0].endswith("P")
    path = root / "pawn-geometry"
    prior = (
        Path(__file__).resolve().parents[1]
        / "circuits/results/handwritten-collisions.csv"
    )
    exclude = set()
    for row in csv.DictReader(prior.open()):
        if row["material"] == "KRvKP":
            q = chess.Board(row["parent"])
            exclude.update(
                [q.board_fen(), q.transform(chess.flip_horizontal).board_fen()]
            )
    output = []
    counts = Counter()
    seen = set()
    available = set(STRATA)
    if actor_piece.piece_type in (chess.BISHOP, chess.KNIGHT):
        available.remove("double")
    if unconditioned:
        available = {"unconditioned"}
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        tb.add_directory(str(root / "tb5"))
        for it in range(90000):
            needed = {
                k for k in available if counts[k] < (4000 if unconditioned else LIMIT)
            }
            if not needed:
                break
            p = rng.randrange(16, 56)
            free = sorted(set(range(64)) - {p, p - 8})
            wk, bk, s, *wp = rng.sample(free, 4 if extra else 3)
            if chess.square_distance(wk, bk) <= 1 or (wp and not 8 <= wp[0] < 56):
                continue
            if needed == {"promotion"} and chess.square_rank(p) != 2:
                continue
            key = (p, wk, bk, s, *wp)
            if key in seen:
                continue
            seen.add(key)
            b = chess.Board.empty()
            for sq, piece in [
                (wk, chess.Piece(chess.KING, True)),
                (bk, chess.Piece(chess.KING, False)),
                (s, actor_piece),
                (p, chess.Piece(chess.PAWN, False)),
            ]:
                b.set_piece_at(sq, piece)
            if wp:
                b.set_piece_at(wp[0], chess.Piece(chess.PAWN, True))
            e = b.copy(stack=False)
            e.set_piece_at(p - 8, e.remove_piece_at(p))
            if not b.is_valid() or not e.is_valid() or b.is_check() or e.is_check():
                continue
            if material == "KRvKP" and (
                b.board_fen() in exclude or e.board_fen() in exclude
            ):
                continue
            for k, ms in choices(b, e, s, p, needed, rng).items():
                output.append(record(b, e, p, ms, tb, k))
                counts[k] += 1
    with gzip.open(
        path / (("random-" if unconditioned else "") + material + ".gz"), "wt"
    ) as f:
        for start in range(0, len(output), 128):
            batch = output[start : start + 128]
            traced = trace_positions(
                [fen for r in batch for fen in r["children"] + r["edited_children"]],
                root,
            )
            for i, r in enumerate(batch):
                for prefix, offset in (("", 0), ("edited_", 2)):
                    r[prefix + "units"], r[prefix + "sf141"] = map(
                        list, zip(*traced[4 * i + offset : 4 * i + offset + 2])
                    )
                if r["branches"]:
                    assert [v for _, v in r["branches"]] == [
                        x[0] for x in r["sf141"] + r["edited_sf141"]
                    ]
                print(json.dumps(r, separators=(",", ":")), file=f)
    return material, dict(counts), it + 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--unconditioned", action="store_true")
    args = p.parse_args()
    root = args.root
    (root / "pawn-geometry").mkdir(exist_ok=True)
    with ProcessPoolExecutor(7) as pool:
        for result in pool.map(
            collect, [(root, m, args.unconditioned) for m in MATERIALS]
        ):
            print(*result, flush=True)
