"""Separate pawn attack geometry from legality by pinning the pawn in both contexts."""

import argparse
import gzip
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from channels import arrays, identity, mixed
from collect import king_group
from geometry import sign
from siblings import trace_positions
from threat_reversal import interval
from transfer import choices, square_class


def sample(args):
    root, mat = args
    rng = random.Random("pin-specificity-v1/" + mat)
    piece = chess.Piece.from_symbol(mat[2])
    need = {"single": 1000}
    if piece.piece_type == chess.ROOK:
        need["double"] = 1000
    out = []
    seen = set()
    directory = root / "pawn-geometry"
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        tb.add_directory(str(directory))
        for attempt in range(150000):
            if not any(need.values()):
                break
            file = rng.randrange(8)
            unit = rng.choice([v for v in (-1, 1) if 0 <= file + 2 * v < 8])
            p = chess.square(file, rng.randrange(2, 7))
            below = [s for s in range(file, p - 8, 8)]
            above = [s for s in range(p + 8, 64, 8)]
            bk = rng.choice(below)
            slider = rng.choice(above)
            if rng.randrange(2):
                bk, slider = slider, bk
            positions = (slider, slider + unit, slider + 2 * unit)
            free = sorted(set(range(64)) - {p, p - 8, bk, *positions})
            wk, actor = rng.sample(free, 2)
            if chess.square_distance(wk, bk) < 2:
                continue
            key = (p, bk, wk, actor, *positions)
            if key in seen:
                continue
            seen.add(key)
            boards = []
            for s in positions:
                b = chess.Board.empty()
                for sq, man in [
                    (p, chess.Piece(chess.PAWN, False)),
                    (bk, chess.Piece(chess.KING, False)),
                    (wk, chess.Piece(chess.KING, True)),
                    (actor, piece),
                    (s, chess.Piece(chess.ROOK, True)),
                ]:
                    b.set_piece_at(sq, man)
                e = b.copy()
                e.set_piece_at(p - 8, e.remove_piece_at(p))
                if not all(v.is_valid() and not v.is_check() for v in (b, e)):
                    break
                boards.append((b, e))
            if len(boards) != 3:
                continue
            pairs = choices(*boards[1], actor, p, {k for k in need if need[k]}, rng)
            for group, ms in pairs.items():
                states = []
                for j, (b, e) in enumerate(boards):
                    fs = []
                    vals = []
                    tags = []
                    for board, pawn in ((b, p), (e, p - 8)):
                        for move in ms:
                            if (
                                not board.is_legal(move)
                                or board.is_capture(move)
                                or board.gives_check(move)
                            ):
                                break
                            board.push(move)
                            if (
                                board.is_pinned(False, pawn) == (j == 0)
                                and any(board.legal_moves)
                                and not any(
                                    board.generate_legal_captures(
                                        to_mask=chess.BB_SQUARES[positions[j]]
                                    )
                                )
                            ):
                                fs.append(board.fen())
                                vals.append(-sign(tb.probe_wdl(board)))
                                tags.append(
                                    square_class(board, move.to_square, pawn)[0]
                                )
                            board.pop()
                    if len(fs) != 4:
                        break
                    states.append(
                        {
                            "children": fs[:2],
                            "edited_children": fs[2:],
                            "labels": vals,
                            "tags": tags,
                            "parent": b.fen(),
                            "edited_parent": e.fen(),
                            "moves": [m.uci() for m in ms],
                        }
                    )
                if len(states) != 3:
                    continue
                assert not any(states[0]["tags"])
                if states[1]["tags"] != states[2]["tags"]:
                    continue
                out.append({"group": group, "states": states})
                need[group] -= 1
    for i in range(0, len(out), 64):
        group = out[i : i + 64]
        items = [s for r in group for s in r["states"]]
        trace = trace_positions(
            [f for s in items for f in s["children"] + s["edited_children"]], root
        )
        for j, s in enumerate(items):
            for prefix, k in (("", 0), ("edited_", 2)):
                s[prefix + "units"], s[prefix + "sf141"] = map(
                    list, zip(*trace[4 * j + k : 4 * j + k + 2])
                )
    with gzip.open(directory / f"pins-{mat}.gz", "wt") as f:
        for r in out:
            print(json.dumps(r, separators=(",", ":")), file=f)
    return mat, len(out)


def summarize(root, dest):
    from measure import table

    coef = np.array(json.loads((dest / "channel-masks.json").read_text())["coef"])
    result = []
    for mat in ("KRRvKP", "KRBvKP", "KRNvKP"):
        with gzip.open(root / "pawn-geometry" / f"pins-{mat}.gz") as f:
            data = list(map(json.loads, f))
        for g in sorted({r["group"] for r in data}):
            cases = [r for r in data if r["group"] == g]
            d = {}
            for j, name in enumerate(("pinned", "free1", "free2")):
                states = [r["states"][j] for r in cases]
                wdl = -np.array([r["labels"] for r in states]).reshape(-1, 2, 2)
                n = np.array(
                    [[v[1] for v in r["sf141"] + r["edited_sf141"]] for r in states]
                ).reshape(-1, 2, 2)
                d[name] = (mixed(wdl), mixed(n))
            ix = sorted(
                range(len(cases)), key=lambda i: identity(cases[i]["states"][1])
            )[:400]
            probes = {}
            for j, name in enumerate(d):
                x, _, _ = arrays([cases[i]["states"][j] for i in ix], root, 1)
                probes[name] = mixed(x.astype(float)) @ coef
            ids = np.array(
                [
                    str(king_group(b.king(True), b.king(False)))
                    for b in [chess.Board(r["states"][0]["parent"]) for r in cases]
                ]
            )
            for k, series, ii in [
                ("wdl", {name: v[0] for name, v in d.items()}, ids),
                ("native", {name: v[1] for name, v in d.items()}, ids),
                ("probe", probes, ids[ix]),
            ]:
                rate = lambda v: float(((v > 0) + 0.5 * (v == 0)).mean())
                pin, free, control = [series[s] for s in ("pinned", "free1", "free2")]
                contrast = (free - pin) - (control - free)
                ci = interval(contrast, ii)
                result.append(
                    {
                        "material": mat,
                        "design": g,
                        "measure": k,
                        "n": len(pin),
                        "pinned_direction": rate(pin),
                        "released_direction": rate(free),
                        "sham_direction": rate(control),
                        "released_minus_pinned": float(np.mean(free - pin)),
                        "sham_minus_released": float(np.mean(control - free)),
                        "corrected_mean": ci[0],
                        "corrected_low": ci[1],
                        "corrected_high": ci[2],
                        "released_larger_rate": rate(free - pin),
                        "equal_when_pinned": float(np.mean(pin == 0)),
                    }
                )
    table(dest / "pins.csv", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    with ProcessPoolExecutor(3) as pool:
        for out in pool.map(
            sample, [(a.root, m) for m in ("KRRvKP", "KRBvKP", "KRNvKP")]
        ):
            print(*out, flush=True)
    summarize(a.root, a.dest)
