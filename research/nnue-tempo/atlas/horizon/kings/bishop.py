"""Replay every bare-race reversal and relocate one king at a time."""

import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import chess
import numpy as np
import validate as v
from race_results import independent
from summarize import write

sq = chess.square_name
sign = v.sign


@lru_cache(None)
def race(obstacles):
    return independent(chess.BISHOP, obstacles)


def options(q):
    result = {}
    for m in list(q.legal_moves):
        q.push(m)
        result[m] = sign(v.tb.probe_wdl(q)) * (-1 if q.turn else 1)
        q.pop()
    return result


def run(root, out):
    v.init(root)
    home = root / "atlas/horizon"
    work = home / "kings"
    raw = np.memmap(
        home / "3.raw",
        dtype=[("ps", "<i4"), ("pos", "<i4"), ("h", "u1", (8,))],
        mode="r",
        shape=(64, 64, 64, 64),
    )
    weights = np.loadtxt(home / "head.txt", dtype="i4")
    bias = weights[:32]
    coef = weights[32:288].reshape(32, 8)
    offset = weights[288]
    last = weights[289:]
    profiles = np.memmap(
        work / "rank-3.bin", dtype="i1", mode="r", shape=(32, 64, 8, 64, 6)
    )

    def score(a, h):
        return int(
            (int(a["ps"]) + int(((coef @ h + bias) >> 6).clip(0, 127) @ last + offset))
            / 16
        )

    def swap(w, k, y, h, l):
        a, b = raw[w, k, y, [h, l]]
        ah, bh = a["h"].astype("i4"), b["h"].astype("i4")
        aa, bb = score(a, ah), score(b, bh)
        ab = np.r_[bh[:6], ah[6:]]
        ba = np.r_[ah[:6], bh[6:]]
        keep = (sign(score(a, ab) - bb), sign(aa - score(b, ba)))
        drop = (sign(score(a, ba) - bb), sign(aa - score(b, ab)))
        return aa - bb, keep, drop, ah - bh

    rows = []
    edits = []
    seen = set()
    for fold in ("pilot", "remaining"):
        for values in np.loadtxt(work / f"exceptions-{fold}", dtype=int, ndmin=2):
            w, k, y, h, l, s, margin, native, keep, drop, empty, d6, d7 = map(
                int, values
            )
            assert (w, k, y, h, l) not in seen
            seen.add((w, k, y, h, l))
            q = v.board(3, 1, w, k, y, s)
            q.turn = True
            assert all(chess.Move(s, t) in q.legal_moves for t in (h, l))
            boards = [v.board(3, 1, w, k, y, t) for t in (h, l)]
            black = [options(b) for b in boards]
            assert [max(a.values()) for a in black] == [1, 0]
            wins = [m for m, z in black[0].items() if z == 1]
            assert len(wins) == 1 and y // 8 == 2
            push = chess.Move(y, y - 8)
            blockers = []
            for obs in ((), (w,), (k,), tuple(sorted((w, k)))):
                f = race(obs)
                blockers.append((int(f(y, h)), int(f(y, l))))
            assert blockers[0] == blockers[2] == (0, 1)
            hs = boards[0].san(wins[0])
            saves = []
            if wins == [push]:
                assert abs(chess.square_file(h) - chess.square_file(y - 8)) == abs(
                    chess.square_rank(h) - chess.square_rank(y - 8)
                )
                assert chess.between(h, y - 8) & (1 << w)
                assert blockers[1] == blockers[3] == (1, 1) and black[1][push] == 0
                b = boards[1]
                b.push(push)
                kinds = set()
                for m, z in options(b).items():
                    if z:
                        continue
                    saves.append(b.san(m))
                    b.push(m)
                    if m.from_square == w and b.is_check():
                        assert chess.between(l, k) & (1 << w)
                        kinds.add("discovered")
                    elif b.is_check() and m.from_square == l:
                        kinds.add("direct_check")
                    elif m.from_square == l and b.is_pinned(False, y - 8):
                        assert k // 8 == 0 and abs(k % 8 - y % 8) == 1
                        kinds.add("quiet_pin")
                    else:
                        raise ValueError(b.fen())
                    b.pop()
                family = (
                    "pin_or_check"
                    if kinds == {"quiet_pin", "direct_check"}
                    else next(iter(kinds))
                )
                assert len(kinds) == 1 or family == "pin_or_check"
                b.pop()
            else:
                family = "royal_fork"
                assert blockers[1] == blockers[3] == (0, 1) and [
                    a[push] for a in black
                ] == [0, 0]
                assert y % 8 in (0, 7) and k == y + (1 if y % 8 == 0 else -1)
                assert (
                    wins[0] == chess.Move(k, k - 8)
                    and chess.square_distance(k - 8, h)
                    == chess.square_distance(k - 8, y % 8)
                    == 1
                )
            profile = profiles[w // 8 * 4 + w % 8, k, y - 16]
            assert profile[h, 5] and profile[l, 5]
            if family == "royal_fork":
                assert profile[h, 4]
            else:
                assert profile[h, 2]
                assert (
                    profile[l, 3]
                    == (
                        {
                            "discovered": 1,
                            "direct_check": 2,
                            "quiet_pin": 4,
                            "pin_or_check": 6,
                        }[family]
                    )
                )
            m, ke, dr, ds = swap(w, k, y, h, l)
            assert (
                m,
                2 + 2 * sign(m),
                2 + sum(ke),
                2 + sum(dr),
                *map(int, ds[6:]),
            ) == (margin, native, keep, drop, d6, d7)
            a, b = raw[w, k, y, [h, l]]
            singles = []
            for u in (6, 7):
                ah, bh = a["h"].astype("i4"), b["h"].astype("i4")
                left, right = bh.copy(), ah.copy()
                left[u], right[u] = ah[u], bh[u]
                singles.append(
                    2
                    + sign(score(a, left) - score(b, bh))
                    + sign(score(a, ah) - score(b, right))
                )
            rows.append(
                {
                    "direct_only_quarters": empty,
                    "high6": int(a["h"][6]),
                    "low6": int(b["h"][6]),
                    "high7": int(a["h"][7]),
                    "low7": int(b["h"][7]),
                    "keep6_quarters": singles[0],
                    "keep7_quarters": singles[1],
                    "fold": fold,
                    "family": family,
                    "parent_fen": q.fen(),
                    "high": sq(h),
                    "low": sq(l),
                    "high_only_win": hs,
                    "low_saves_after_push": " ".join(sorted(saves)),
                    "native_margin": m,
                    "keep_high_recipient": ke[0],
                    "keep_low_recipient": ke[1],
                    "drop_high_recipient": dr[0],
                    "drop_low_recipient": dr[1],
                    "delta6": d6,
                    "delta7": d7,
                    "passes_rank_census": int(
                        family == "royal_fork" or profile[l, 3] != 0
                    ),
                }
            )
            for side in ("white", "black"):
                old = w if side == "white" else k
                for t in chess.SquareSet(chess.BB_KING_ATTACKS[old]):
                    nw, nk = (t, k) if side == "white" else (w, t)
                    if nw % 8 >= 4 or len({nw, nk, y, s, h, l}) != 6:
                        continue
                    pr = profiles[nw // 8 * 4 + nw % 8, nk, y - 16]
                    if not (pr[h, 5] and pr[l, 5]):
                        continue
                    parent = v.board(3, 1, nw, nk, y, s)
                    parent.turn = True
                    if not (
                        parent.is_valid()
                        and not parent.is_check()
                        and all(chess.Move(s, u) in parent.legal_moves for u in (h, l))
                    ):
                        continue
                    a, b = [v.state(3, 1, nw, nk, y, u) for u in (h, l)]
                    if a["f"] != b["f"]:
                        continue
                    order = sign(sign(a["w"]) - sign(b["w"]))
                    val, keep2, _, _ = swap(nw, nk, y, h, l)
                    assert val == int(a["n"]) - int(b["n"])
                    edits.append(
                        {
                            "family": family,
                            "king": side,
                            "truth_order": order,
                            "native_before": sign(m),
                            "native_after": sign(val),
                            "keep_before": sum(ke),
                            "keep_after": sum(keep2),
                        }
                    )
    assert len(rows) == 295
    write(out / "bishop-295.csv", rows)
    summary = []
    for family in sorted({r["family"] for r in rows}):
        subset = [r for r in rows if r["family"] == family]
        assert all(
            r["keep_high_recipient"]
            == r["keep_low_recipient"]
            == sign(r["native_margin"])
            for r in subset
        )
        summary.append(
            {
                "family": family,
                "n": len(subset),
                "correct": sum(r["native_margin"] > 0 for r in subset),
                "tied": sum(r["native_margin"] == 0 for r in subset),
                "wrong": sum(r["native_margin"] < 0 for r in subset),
                "both7zero": sum(r["high7"] == r["low7"] == 0 for r in subset),
                "keep6_points": sum(r["keep6_quarters"] for r in subset) / 4,
                "keep7_points": sum(r["keep7_quarters"] for r in subset) / 4,
                "direct_points": sum(r["direct_only_quarters"] for r in subset) / 4,
            }
        )
    write(out / "bishop-families.csv", summary)
    keys = list(edits[0])
    counts = Counter(tuple(r[k] for k in keys) for r in edits)
    write(
        out / "king-relocations.csv",
        [dict(zip(keys, k), n=n) for k, n in sorted(counts.items())],
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
