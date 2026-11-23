"""Independent FEN, full WDL and move-class replay using python-chess."""

import argparse
import gzip
import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import batched
from pathlib import Path

import chess
import chess.syzygy
import numpy as np

HERE = Path(__file__).resolve().parent
RULES = re.findall(r"^RULE\((\w+),", (HERE / "rules.def").read_text(), re.MULTILINE)
DOMAIN = (1 << len(RULES)) - 1
DT = np.dtype(
    [
        ("h", "<i2"),
        ("f", "<i2"),
        ("n", "<i2"),
        ("p", "<i2"),
        ("d", "<i2"),
        ("w", "i1"),
        ("v", "u1"),
    ]
)
REPLY = np.dtype([("index", "<u4"), ("mask", "<u8", (64, 4))])
dist = chess.square_distance
sign = lambda x: (int(x) > 0) - (int(x) < 0)


def init(root):
    global scratch, tb, maps
    scratch = root
    tb = chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular")
    maps = {}


def state(a, b, wk, bk, y, x):
    if (a, b) not in maps:
        maps[a, b] = np.memmap(
            scratch / "atlas" / f"{a}-{b}.bin",
            mode="r",
            dtype=DT,
            shape=(64, 64, 64, 64),
        )
    return maps[a, b][wk, bk, y, x]


def board(a, b, wk, bk, y, x):
    q = chess.Board.empty()
    for s, k, side in ((wk, 6, True), (bk, 6, False), (y, b, False), (x, a, True)):
        q.set_piece_at(s, chess.Piece(k, side))
    q.turn = False
    return q


def cut(k, l, s, arms):
    for shift, ray in [
        (lambda q: q // 8, chess.BB_RANKS[s // 8]),
        (lambda q: q % 8, chess.BB_FILES[s % 8]),
    ]:
        if (
            min(shift(k), shift(l)) < shift(s) < max(shift(k), shift(l))
            and (arms | 1 << s) & ray == ray
        ):
            return True
    return False


def opp(k, l):
    v = sorted((abs(k % 8 - l % 8), abs(k // 8 - l // 8)))
    return (v == [0, 2], v == [2, 2])


@lru_cache(8192)
def probe(a, b, wk, bk, y, x, record_moves=False):
    q = board(a, b, wk, bk, y, x)
    entry = state(a, b, wk, bk, y, x)
    assert q.is_valid() and not q.is_check() and entry["v"]
    wdl = tb.probe_wdl(q)
    assert wdl == entry["w"]
    old_w = int(q.attacks(x))
    old_b = int(q.attacks(y))
    oldpin = q.is_pinned(False, y)
    oldwpin = q.is_pinned(True, x)
    oldguard = dist(bk, y) == 1
    olddanger = q.is_attacked_by(True, y)
    masks = [0] * 4
    best = -2
    detail = []
    for m in list(q.legal_moves):
        king = m.from_square == bk
        live = m.to_square != x
        take = not live
        promo = bool(m.promotion)
        quiet = not (king or take or promo)
        q.push(m)
        k = q.king(False)
        s = y if king else m.to_square
        attacks = int(q.attacks(s))
        wa = int(q.attacks(x)) if live else 0
        check = q.is_check()
        replies = list(q.legal_moves)
        end = not replies
        recap = any(r.to_square == s for r in replies)
        guard = dist(k, s) == 1
        enemy_guard = live and dist(wk, x) == 1
        targets = live and bool(attacks & (1 << x))
        fresh = targets and not (old_b & (1 << x))
        approach = live and dist(k, x) == 1 and dist(bk, x) > 1
        wp = live and a == 1
        major = b in (4, 5)
        skewer = False
        if (
            check
            and live
            and chess.between(s, x) & (1 << wk)
            and q.piece_type_at(s) in (3, 4, 5)
        ):
            df = abs(s % 8 - x % 8)
            dr = abs(s // 8 - x // 8)
            skewer = (
                q.piece_type_at(s) == 5
                or (q.piece_type_at(s) == 3 and df == dr)
                or (q.piece_type_at(s) == 4 and df * dr == 0)
            )
        pred = [
            king and take,
            not king and take,
            not king and take and recap,
            not king and take and not recap,
            promo and live,
            promo and take,
            promo and m.promotion != 5,
            promo and check,
            promo and not recap,
            quiet and b == 1 and y - s == 16,
            quiet and b == 1 and y - s == 8,
            quiet and b == 1 and targets,
            king and b == 1 and bk == y - 8,
            king and b == 1 and dist(k, y % 8) == 1 and dist(bk, y % 8) > 1,
            king and b == 1 and guard and not oldguard,
            not king and check,
            king and check,
            not king and check and dist(s, wk) == 1,
            not king and check and dist(s, wk) >= 3,
            check and guard,
            check and targets,
            skewer,
            live and q.is_pinned(True, x) and not oldwpin,
            king and live and oldpin and not q.is_pinned(False, s),
            not king and oldpin and bool(chess.ray(bk, y) & (1 << s)),
            quiet and olddanger and not q.is_attacked_by(True, s),
            king and live and olddanger and guard and not oldguard,
            king and approach and guard,
            king and approach and not enemy_guard,
            king and approach and enemy_guard,
            quiet and olddanger and targets,
            fresh and not enemy_guard,
            fresh and enemy_guard,
            king
            and live
            and bool(old_w & (1 << y))
            and bool(chess.between(x, y) & (1 << k))
            and not wa & (1 << y),
            king and opp(wk, k)[0] and not opp(wk, bk)[0],
            king and opp(wk, k)[1] and not opp(wk, bk)[1],
            quiet and major and cut(wk, k, s, attacks) and not cut(wk, bk, y, old_b),
            king
            and a in (4, 5)
            and live
            and cut(bk, y, x, old_w)
            and not cut(k, y, x, wa)
            and (k % 8 == x % 8 or k // 8 == x // 8),
            wp and not king and s == x + 8,
            wp and king and k == x + 8,
            wp
            and bool(attacks & (1 << (56 + x % 8)))
            and not old_b & (1 << (56 + x % 8)),
            wp and bool(attacks & (1 << (x + 8))) and not old_b & (1 << (x + 8)),
            wp and king and dist(k, x) < dist(bk, x),
            quiet and b == 1 and dist(wk, y % 8) > s // 8 and dist(wk, y % 8) <= y // 8,
            wp and fresh and major and s % 8 == x % 8 and s // 8 < x // 8,
            king and k in (27, 28, 35, 36) and bk not in (27, 28, 35, 36),
            end and check,
            end and not check,
        ]
        assert len(pred) == len(RULES)
        tag = sum(1 << i for i, f in enumerate(pred) if f)
        v = -sign(tb.probe_wdl(q))
        best = max(best, v)
        if record_moves:
            detail.append((m.uci(), v, tag))
        masks[0] |= tag
        masks[1] |= DOMAIN ^ tag
        if v >= 0:
            masks[2] |= DOMAIN ^ tag
        if v > 0:
            masks[3] |= DOMAIN ^ tag
        q.pop()
    assert best == sign(wdl)
    return (masks, detail) if record_moves else masks


def played(batch):
    result = []
    for month, gid, fen, first, second in batch:
        parent = chess.Board(fen)
        opts = []
        for m in (first, second):
            b = parent.copy()
            b.push_uci(m)
            x = chess.parse_square(m[2:4])
            wk, bk = b.king(True), b.king(False)
            y = next(iter(chess.SquareSet(b.occupied_co[False] & ~b.kings)))
            a, e = b.piece_type_at(x), b.piece_type_at(y)
            assert b.turn == False and not b.has_legal_en_passant()
            entry = state(a, e, wk, bk, y, x)
            opts.append(
                ([int(entry[k]) for k in DT.names[:-1]], probe(a, e, wk, bk, y, x))
            )
        result.append([month, gid, fen, first, second, a, e, opts])
    return result


def audit(root):
    init(root)
    rng = np.random.default_rng(971146)
    totals = [0] * 4
    for a in range(1, 6):
        for e in range(1, 6):
            rr = np.memmap(root / "atlas" / f"{a}-{e}.replies", mode="r", dtype=REPLY)
            fens = []
            entries = []
            for g in rng.choice(len(rr), 128, replace=False):
                idx = int(rr[g]["index"])
                wk = idx >> 12
                bk = idx >> 6 & 63
                y = idx & 63
                x = int(rng.integers(64))
                q = board(a, e, wk, bk, y, x)
                entry = state(a, e, wk, bk, y, x)
                ok = (
                    len({wk, bk, y, x}) == 4
                    and q.is_valid()
                    and not q.is_check()
                    and any(q.legal_moves)
                )
                assert bool(entry["v"]) == bool(ok)
                totals[0] += 1
                if not ok:
                    continue
                masks = probe(a, e, wk, bk, y, x)
                assert masks == list(map(int, rr[g]["mask"][x])), (
                    a,
                    e,
                    q.fen(),
                    [
                        RULES[j]
                        for j in range(len(RULES))
                        if any(
                            (m ^ int(n)) >> j & 1
                            for m, n in zip(masks, rr[g]["mask"][x])
                        )
                    ],
                )
                fens.append(q.fen())
                entries.append(entry)
                totals[1] += 1
            cmd = "".join("position fen " + f + "\neval\n" for f in fens) + "quit\n"
            for full in (False, True):
                engine = root / (
                    "circuits/probe/src/stockfish" if full else "sf141/src/stockfish"
                )
                text = subprocess.check_output(
                    [engine],
                    input=cmd,
                    text=True,
                    env={
                        **os.environ,
                        "NNUE_CIRCUIT_CLASSICAL": "1",
                        "NNUE_CIRCUIT_FULL_HCE": "1",
                    },
                )
                scores = [
                    list(map(int, s.split()[1:]))
                    for s in text.splitlines()
                    if s.startswith("scores ")
                ]
                units = [
                    list(map(int, s.split()[1:3]))
                    for s in text.splitlines()
                    if s.startswith("units ")
                ]
                pre = iter(
                    int(s.split()[1])
                    for s in text.splitlines()
                    if s.startswith("pregrain ")
                )
                special = [
                    int(s.split()[2])
                    for s in text.splitlines()
                    if s.startswith("terms ")
                ]
                for j, (en, (h, n), (p, d)) in enumerate(
                    zip(entries, scores, units, strict=True)
                ):
                    assert (
                        n == en["n"]
                        and int(p / 16) == en["p"]
                        and int(d / 16) == en["d"]
                    )
                    if full:
                        assert (h if special[j] else next(pre)) == en["f"]
                    else:
                        assert h == en["h"]
                    totals[2 + full] += 1
    print("validity, masks, native outputs, full classical", *totals)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--games", action="store_true")
    args = p.parse_args()
    if not args.games:
        audit(args.root)
    else:
        with (
            gzip.open(args.root / "atlas/played.gz", "rt") as f,
            gzip.open(args.root / "atlas/game-results.gz", "wt") as out,
            ProcessPoolExecutor(10, initializer=init, initargs=(args.root,)) as pool,
        ):
            for rows in pool.map(played, batched(map(json.loads, f), 32)):
                for r in rows:
                    print(json.dumps(r), file=out)
