"""Exact discrete activation maximization over every quiet move-pair with a common legal source."""

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from collect import king_group

from model import Model, board_for

SQ = np.arange(64)
U, V = np.triu_indices(64, 1)
BETWEEN = np.array(
    [[chess.between(i, j) for j in range(64)] for i in range(64)], dtype=np.uint64
)
R = (SQ[:, None] // 8) == (SQ // 8)
F = (SQ[:, None] % 8) == (SQ % 8)
D = abs(SQ[:, None] // 8 - SQ // 8) == abs(SQ[:, None] % 8 - SQ % 8)
MOVES = {
    4: (R | F),
    3: D,
    5: R | F | D,
    2: np.array(
        [
            [bool(chess.BB_KNIGHT_ATTACKS[i] & 1 << j) for j in range(64)]
            for i in range(64)
        ]
    ),
}
for v in MOVES.values():
    np.fill_diagonal(v, False)
NAMES = ["frozen_probe", "permuted_probe", "engine", "uniform"]


def sample_kings():
    groups = defaultdict(list)
    for a in range(64):
        for b in range(64):
            group = king_group(a, b)
            if (
                chess.square_distance(a, b) > 1
                and hashlib.sha256(bytes(group)).digest()[0] % 2
            ):
                groups[group].append((a, b))
    rng = np.random.default_rng(632907)
    allgroups = sorted(groups)
    return [
        groups[allgroups[i]][rng.integers(len(groups[allgroups[i]]))]
        for i in rng.choice(len(groups), 128, replace=False)
    ]


def ready(args):
    global model, tb
    model = Model(args)
    tb = chess.syzygy.open_tablebase(args / "pychess/data/syzygy/regular")


def scan(args):
    wk, bk, actor = args
    rng = np.random.default_rng(379881 + wk * 9000 + bk * 60 + actor)
    men = [6, -6, actor, -1]
    rows = []
    best = None
    falsifier = None
    good = np.zeros((64, 64), bool)
    wdl = np.zeros((64, 64), int)
    svals = np.zeros((64, 64, 3))
    # Input grid indexed by pawn square, then actor square.
    nvalid = 0
    for p in range(8, 56):
        ss = np.column_stack([np.full(64, wk), np.full(64, bk), SQ, np.full(64, p)])
        if p in (wk, bk) or chess.BB_PAWN_ATTACKS[False][p] & (1 << wk):
            continue
        x, ps = model.read(ss, men)
        scores, *_ = model.score(x, ps, 0)
        svals[p, :, :2] = x @ model.probes
        svals[p, :, 2] = scores
        for s in range(64):
            if s in (wk, bk, p):
                continue
            b = board_for(ss[s], men)
            if not b.is_valid() or b.is_check() or not any(b.legal_moves):
                continue
            good[p, s] = True
            wdl[p, s] = np.sign(tb.probe_wdl(b))
            nvalid += 1
    for p in range(16, 56):
        # Source positions must be valid unchecked parents, equivalent here to
        # child legality except that lack of black moves at the source is harmless.
        if p in (wk, bk) or p - 8 in (wk, bk):
            continue
        occ = sum(1 << k for k in (wk, bk, p, p - 8))
        dest = good[p] & good[p - 8]
        src = np.ones(64, bool)
        for k in (wk, bk, p, p - 8):
            src[k] = False
        if any(chess.BB_PAWN_ATTACKS[False][q] & (1 << wk) for q in [p, p - 8]):
            continue
        for q in (p, p - 8):
            ownblocks = np.uint64(sum(1 << k for k in (wk, bk, q)))
            src &= ~(MOVES[actor][:, bk] & ((BETWEEN[:, bk] & ownblocks) == 0))
        reach = (
            MOVES[actor]
            & ((BETWEEN & np.uint64(occ)) == 0)
            & src[:, None]
            & dest[None, :]
        )
        # Compute the union over all possible source squares; each pair counts once.
        bits = np.sum(
            np.where(
                reach, (np.uint64(1) << SQ.astype(np.uint64))[:, None], np.uint64(0)
            ),
            axis=0,
            dtype=np.uint64,
        )
        shared = bits[U] & bits[V]
        valid = shared != 0
        if not valid.any():
            continue
        a = U[valid]
        b = V[valid]
        shared = shared[valid]
        scores = svals[p - 8] - svals[p]
        dd = scores[a] - scores[b]
        pawn_att = lambda q: np.array(
            [bool(chess.BB_PAWN_ATTACKS[False][q] & (1 << s)) for s in range(64)], int
        )
        rel = pawn_att(p - 8) - pawn_att(p)
        danger = rel[a] - rel[b]
        w = wdl[p - 8] - wdl[p]
        outcome = w[a] - w[b]
        rand = int(rng.integers(len(a)))
        for i, n in enumerate(NAMES):
            if i == 3:
                which = rand
                direction = rng.choice([-1, 1])
                obj = 0.0
            else:
                mx = abs(dd[:, i])
                which = int(rng.choice(np.flatnonzero(mx == mx.max())))
                direction = (
                    1
                    if dd[which, i] > 0
                    else -1
                    if dd[which, i] < 0
                    else rng.choice([-1, 1])
                )
                obj = float(mx[which])
            A = int(a[which] if direction == 1 else b[which])
            B = int(b[which] if direction == 1 else a[which])
            sb = int(shared[which])
            source = (sb & -sb).bit_length() - 1
            item = {
                "wk": wk,
                "bk": bk,
                "actor": actor,
                "pawn": p,
                "source": source,
                "A": A,
                "B": B,
                "target": n,
                "maximum": obj,
                "pawn_relation": int(direction * danger[which]),
                "wdl_direction": int(direction * outcome[which]),
                "true_reversal": int(
                    wdl[p, A] < wdl[p, B] and wdl[p - 8, A] > wdl[p - 8, B]
                ),
                "engine_mixed": float(scores[A, 2] - scores[B, 2]),
                "pairs": len(a),
            }
            rows.append(item)
            if n == "frozen_probe":
                if best is None or best["maximum"] < obj:
                    best = item
                if item["pawn_relation"] <= 0 and (
                    falsifier is None or falsifier["maximum"] < obj
                ):
                    falsifier = item
    return rows, nvalid, best, falsifier


def main(root, dest):
    jobs = [(w, b, a) for w, b in sample_kings() for a in (4, 5, 3, 2)]
    data = []
    examples = []
    counts = defaultdict(int)
    with ProcessPoolExecutor(10, initializer=ready, initargs=(root,)) as pool:
        for job, (rs, n, good, bad) in zip(jobs, pool.map(scan, jobs, chunksize=1)):
            data.extend(rs)
            counts[job[2]] += n
            examples.extend(r for r in (good, bad) if r)
    import gzip

    from measure import table

    with gzip.open(root / "relational-tests/census.gz", "wt") as f:
        for r in data:
            print(json.dumps(r, separators=(",", ":")), file=f)
    result = []
    for a in counts:
        for name in NAMES:
            rs = [r for r in data if r["actor"] == a and r["target"] == name]
            pos = lambda key, rs=rs: float(np.mean([r[key] > 0 for r in rs]))
            result.append(
                {
                    "actor": chess.piece_symbol(a),
                    "target": name,
                    "legal_children": counts[a],
                    "blocks": len(rs),
                    "pairs": sum(r["pairs"] for r in rs),
                    "pawn_agrees": pos("pawn_relation"),
                    "pawn_opposes": float(
                        np.mean([r["pawn_relation"] < 0 for r in rs])
                    ),
                    "wdl_agrees": pos("wdl_direction"),
                    "wdl_opposes": float(np.mean([r["wdl_direction"] < 0 for r in rs])),
                    "engine_agrees": pos("engine_mixed"),
                    "exact_forward_reversal": pos("true_reversal"),
                }
            )
    table(dest / "maximization.csv", result)
    samples = []
    for a in counts:
        for negative in (False, True):
            rs = [
                r
                for r in examples
                if r["actor"] == a and (r["pawn_relation"] <= 0) == negative
            ]
            for r in sorted(rs, key=lambda r: -r["maximum"])[:4]:
                b = board_for(
                    [r["wk"], r["bk"], r["source"], r["pawn"]], [6, -6, a, -1], True
                )
                A = chess.Move(r["source"], r["A"])
                B = chess.Move(r["source"], r["B"])
                for p in (r["pawn"], r["pawn"] - 8):
                    q = b.copy()
                    if p != r["pawn"]:
                        q.set_piece_at(p, q.remove_piece_at(r["pawn"]))
                    assert (
                        q.is_valid()
                        and not q.is_check()
                        and all(
                            q.is_legal(m)
                            and not q.gives_check(m)
                            and not q.is_capture(m)
                            for m in (A, B)
                        )
                    )
                samples.append(dict(**r, fen=b.fen(), moves=[A.uci(), B.uci()]))
    samples = list(
        {(r["fen"], tuple(r["moves"]), r["target"]): r for r in samples}.values()
    )
    (dest / "extrema.json").write_text(json.dumps(samples, indent=1) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    main(a.root, a.dest)
