"""Score all eligible same-piece moves at actual parents with frozen tie breakers."""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from pathlib import Path

import chess

from summarize import group, mat
from validate import init, probe, sign, state


def run(batch):
    result = []
    for month, gid, fen, u, v in batch:
        b = chess.Board(fen)
        wk = b.king(True)
        bk = b.king(False)
        src = chess.parse_square(u[:2])
        a = b.piece_type_at(src)
        y = next(iter(chess.SquareSet(b.occupied_co[False] & ~b.kings)))
        e = b.piece_type_at(y)
        moves = []
        for mv in b.generate_legal_moves(from_mask=1 << src):
            if mv.promotion or b.is_capture(mv) or b.gives_check(mv):
                continue
            b.push(mv)
            ok = any(b.legal_moves) and not b.has_legal_en_passant()
            b.pop()
            if ok:
                x = mv.to_square
                en = state(a, e, wk, bk, y, x)
                assert en["v"]
                moves.append((mv.uci(), int(en["f"]), -sign(en["w"]), int(en["n"])))
        assert len(moves) >= 2
        leaders = [r for r in moves if r[1] == min(z[1] for z in moves)]
        w = [r[2] for r in leaders]
        mixed = min(w) != max(w)
        entry = {
            "month": month,
            "game": gid,
            "fen": fen,
            "material": mat(a, e),
            "group": group(wk, bk),
            "count": len(moves),
            "leaders": len(leaders),
            "sum_w": sum(w),
            "mixed": mixed,
        }
        if mixed:
            entry["choices"] = [
                (u, utility, n, probe(a, e, wk, bk, y, chess.parse_square(u[2:4]))[0])
                for u, _, utility, n in leaders
            ]
        result.append(entry)
    return result


def main(root):
    with (
        gzip.open(root / "atlas/played.gz", "rt") as f,
        gzip.open(root / "atlas/game-decisions.gz", "wt") as out,
        ProcessPoolExecutor(8, initializer=init, initargs=(root,)) as pool,
    ):
        for rows in pool.map(run, batched(map(json.loads, f), 32)):
            for r in rows:
                print(json.dumps(r), file=out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
