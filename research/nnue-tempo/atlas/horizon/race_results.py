"""Compare the pawn race, native rankings, and exchanged neural units."""

import argparse
from functools import lru_cache
from pathlib import Path

import chess
import numpy as np
from results import load, values
from summarize import ci, group, score, write


def independent(a, obstacles):
    @lru_cache(None)
    def escapes(pawn, minor):
        q = chess.Board.empty()
        for s in obstacles:
            q.set_piece_at(s, chess.Piece(chess.KING, True))
        q.set_piece_at(pawn, chess.Piece(chess.PAWN, False))
        q.set_piece_at(minor, chess.Piece(a, True))
        if minor in q.attacks(pawn):
            return True
        # Check every intermediate square on the black first double move.
        for step in (8, 16):
            t = pawn - step
            if step == 16 and pawn // 8 != 6:
                break
            if q.piece_at(t):
                break
            q.remove_piece_at(pawn)
            q.set_piece_at(t, chess.Piece(chess.PAWN, False))
            moves = set(q.attacks(minor)) - set(obstacles)
            q.remove_piece_at(t)
            q.set_piece_at(pawn, chess.Piece(chess.PAWN, False))
            if t in moves:
                continue
            if t // 8 == 0:
                return True
            if all(escapes(t, z) for z in (moves | {minor})):
                return True
        return False

    return escapes


def main(root, out):
    home = root / "atlas/horizon"
    patch = []
    hist = []
    chessrows = []
    effects = []
    matched = []
    panel, _, d, r = load(home / "all5")
    q = np.load(home / "patches.npz")["utility"]
    rng = np.random.default_rng(6619)
    for a in (2, 3):
        hashed = sorted(home.glob(f"race-{a}-*.txt"))
        assert len(hashed) == 2
        H = sum(np.loadtxt(f, dtype=int)[:, -1].reshape(2, 3, 3, 3) for f in hashed)
        P = sum(
            np.loadtxt(str(f) + ".patch", dtype="i8")[:, 2:].reshape(2, 4, 11)
            for f in hashed
        )
        ws = np.fromfile(home / f"race-{a}.bin", dtype="<u8").reshape(4097, 64)
        tests = 0
        for block in rng.choice(4096, 45, replace=False):
            w, k = divmod(int(block), 64)
            obs = {w, k}
            for v in (0, 1):
                fn = independent(a, obs if v else set())
                for y, x in rng.integers(0, 64, (80, 2)):
                    y, x = int(y), int(x)
                    if y // 8 not in range(1, 7) or x == y or y in obs or x in obs:
                        continue
                    assert fn(y, x) == bool(int(ws[1 + block if v else 0, y]) >> x & 1)
                    tests += 1
        for v in (0, 1):
            for t, rr, nn in np.ndindex(3, 3, 3):
                hist.append(
                    {
                        "white": a,
                        "obstacles": v,
                        "wdl_order": t - 1,
                        "race_order": rr - 1,
                        "neural_order": nn - 1,
                        "n": H[v, t, rr, nn],
                    }
                )
            for j, c in enumerate(
                ("race_wrong", "race_tie", "race_right", "equal_wdl_race_strict")
            ):
                (
                    n,
                    native,
                    keep,
                    drop,
                    base,
                    six_equal,
                    two_equal,
                    frozen,
                    additive,
                    du6,
                    du7,
                ) = P[v, j]
                if j == 3:
                    assert n == H[v, 1, [0, 2], :].sum()
                else:
                    assert n == sum(
                        H[v, t, rr, :].sum()
                        for t in (0, 2)
                        for rr in range(3)
                        if 1 + (t - 1) * (rr - 1) == j
                    )
                patch.append(
                    {
                        "white": a,
                        "obstacles": v,
                        "condition": c,
                        "n": n,
                        "native": native / (4 * n),
                        "keep_6_7": keep / (4 * n),
                        "remove_6_7": drop / (4 * n),
                        "empty": base / (4 * n),
                        "first_six_equal": six_equal / n,
                        "last_two_equal": two_equal / n,
                        "fixed_two_dimensional": frozen / (2 * n),
                        "additive": additive / (2 * n),
                        "mean_change_u6": du6 / n,
                        "mean_change_u7": du7 / n,
                    }
                )
            for name in np.unique(panel):
                ix = np.flatnonzero((panel == name) & (d["a"] == a) & (d["b"] == 1))
                if not len(ix):
                    continue
                rowindex = (
                    d["wk"][ix] * 64 + d["bk"][ix] + 1
                    if v
                    else np.zeros(len(ix), dtype=int)
                )
                bits = ws[rowindex, d["y"][ix]]
                rate = ((bits >> d["x"][ix].astype("u8")) & 1).astype("i1") - (
                    (bits >> d["z"][ix].astype("u8")) & 1
                ).astype("i1")
                for scope, active in [
                    ("all", np.ones(len(ix), bool)),
                    ("strict", rate != 0),
                    ("ties", rate == 0),
                ]:
                    if not active.any():
                        continue
                    rid = ix[active]
                    chessrows.append(
                        {
                            "panel": name,
                            "white": a,
                            "obstacles": v,
                            "scope": scope,
                            "n": len(rid),
                            "race": score(rate[active]).mean(),
                            "native": q[rid, 255].mean(),
                            "keep_6_7": q[rid, 192].mean(),
                            "remove_6_7": q[rid, 63].mean(),
                            "independent_state_checks": tests,
                        }
                    )

    def distinguishes(mask):
        v = values(r, mask)
        return v[0, :, -1, 1] > v[1, :, -1, 1]

    c = distinguishes(15)
    no_p = distinguishes(11)
    no_e = distinguishes(14)
    typ = {
        "promotion": c & ~no_p & no_e,
        "elementary_capture": c & no_p & ~no_e,
        "both_required": c & ~no_p & ~no_e,
        "neither_required": c & no_p & no_e,
        "unresolved": ~c,
    }
    groups = np.array([group(w, k) for w, k in zip(d["wk"], d["bk"])])
    for name in np.unique(panel):
        for tag, mask in typ.items():
            m = (panel == name) & mask
            if not m.any():
                continue
            gap = q[m, 255] - q[m, 63]
            lo, hi = ci(gap, groups[m])
            effects.append(
                {
                    "panel": name,
                    "depth": 5,
                    "condition": tag,
                    "n": int(m.sum()),
                    "full": q[m, 255].mean(),
                    "keep_6_7": q[m, 192].mean(),
                    "remove_6_7": q[m, 63].mean(),
                    "effect": gap.mean(),
                    "lo": lo,
                    "hi": hi,
                }
            )
        for strat in ("material", "material_and_black_pawn_rank"):
            contrast = []
            mass = []
            rank = (
                np.where(d["b"] == 1, d["y"] // 8, -1)
                if strat.endswith("rank")
                else np.zeros(len(panel), int)
            )
            for a, b, rank_of in set(zip(d["a"], d["b"], rank)):
                s = (panel == name) & (d["a"] == a) & (d["b"] == b) & (rank == rank_of)
                p = s & typ["promotion"]
                e = s & typ["elementary_capture"]
                if not p.any() or not e.any():
                    continue
                diff = q[:, 255] - q[:, 63]
                contrast.append(diff[p].mean() - diff[e].mean())
                mass.append(min(sum(p), sum(e)))
            matched.append(
                {
                    "panel": name,
                    "strata": strat,
                    "common_cells": len(mass),
                    "overlap_weight": sum(mass),
                    "effect_difference": np.average(contrast, weights=mass)
                    if mass
                    else None,
                }
            )
    write(out / "race-order-counts.csv", hist)
    write(out / "race-unit-census.csv", patch)
    write(out / "race-replication.csv", chessrows)
    write(out / "frontier-unit-effects.csv", effects)
    write(out / "matched-unit-effects.csv", matched)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
