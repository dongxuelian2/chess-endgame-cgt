"""Test whether an explicit capture option predicts Stockfish 14.1 errors."""

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network, credit
from promotion_geometry import relation
from summarize import write

from cohorts import kings
from effects import batches, ci, delta, patch
from perspectives import channels

RULES = (
    "geometric_capture",
    "legal_capture",
    "surviving_capture",
    "surviving_straight",
)


def board_features(a, w, k, y, x):
    q = v.board(a, 1, w, k, y, x)
    target = chess.BB_SQUARES[x]
    geom = q.attacks_mask(y) & target != 0
    can_capture = False
    survive = [False, False]
    for m in list(q.generate_legal_moves(from_mask=chess.BB_SQUARES[y])):
        if not m.promotion:
            continue
        capturing = q.is_capture(m)
        can_capture |= capturing
        q.push(m)
        survives = (
            m.promotion in (chess.QUEEN, chess.ROOK)
            and not q.is_stalemate()
            and not any(
                q.generate_legal_captures(to_mask=chess.BB_SQUARES[m.to_square])
            )
        )
        q.pop()
        survive[int(not capturing)] |= survives
    return int(geom), int(can_capture), int(survive[0]), int(survive[1])


def loaded(root):
    for a in range(2, 6):
        for phase in ("discovery", "validation"):
            panel = np.load(root / "manuscript" / f"{phase}-{a}-1.npz")
            yield phase, a, panel["rows"], panel["mask"], None
    for phase, name in [
        ("games_A", "atlas/played.gz"),
        ("games_B", "atlas/horizon/played.gz"),
    ]:
        by = defaultdict(list)
        with gzip.open(root / name, "rt") as z:
            for _, gid, fen, m, n, *_ in map(json.loads, z):
                q = chess.Board(fen)
                w = q.king(True)
                k = q.king(False)
                source = chess.parse_square(m[:2])
                a = q.piece_type_at(source)
                pawn = q.pieces(chess.PAWN, False)
                if len(pawn) != 1 or a not in (2, 3, 4, 5):
                    continue
                y = next(iter(pawn))
                if y // 8 != 1:
                    continue
                x0, x1 = [chess.parse_square(u[2:4]) for u in (m, n)]
                h, l = [v.state(a, 1, w, k, y, x) for x in (x0, x1)]
                if (
                    not h["v"]
                    or not l["v"]
                    or h["f"] != l["f"]
                    or np.sign(h["w"]) == np.sign(l["w"])
                ):
                    continue
                if h["w"] < l["w"]:
                    x0, x1 = x1, x0
                by[a].append(([w, k, y, x0, x1, source], gid))
        for a, rows in by.items():
            arrays, gids = zip(*rows)
            yield (
                phase,
                a,
                np.array(arrays),
                np.zeros(len(rows), dtype=bool),
                list(gids),
            )


def prepare(root, out, net):
    data = defaultdict(list)
    game_ids = defaultdict(dict)
    rank = channels(out)
    for phase, a, rows, necessary, gids in loaded(root):
        if gids is None:
            keys = kings(rows)
        else:
            ids = game_ids[phase]
            keys = np.array([ids.setdefault(g, len(ids)) for g in gids])
        for ix in batches(np.arange(len(rows)), 128):
            rr = rows[ix]
            w, k, y, h, l, _ = rr.T
            f = (y ^ (7 * (w % 8 < 4))) % 8
            raw, p, _ = net.encode(rr, a, 1)
            t = net.forward(raw, p)
            m = delta(t["score"])
            hi = v.state(a, 1, w, k, y, h)
            lo = v.state(a, 1, w, k, y, l)
            assert np.array_equal(m, hi["n"].astype(int) - lo["n"].astype(int))
            assert np.all(hi["f"] == lo["f"]) and np.all(
                np.sign(hi["w"]) > np.sign(lo["w"])
            )
            fh = np.array(
                [board_features(a, w, k, y, x) for w, k, y, x in rr[:, :4].tolist()]
            )
            fl = np.array(
                [
                    board_features(a, w, k, y, x)
                    for w, k, y, x in rr[:, [0, 1, 2, 4]].tolist()
                ]
            )
            assert np.all(hi["w"][fh[:, 2] == 1] > 0)
            assert np.all(lo["w"][fl[:, 2] == 1] > 0)
            clear = (relation(a, h, y - 8, w, k) == 4) & (
                relation(a, l, y - 8, w, k) == 2
            )
            res = np.c_[
                keys[ix],
                np.full(len(ix), a),
                m,
                fh - fl,
                fh,
                fl,
                f,
                clear,
                necessary[ix],
            ]
            effects = np.zeros(len(ix))
            if phase != "discovery":
                for file in np.unique(f):
                    sel = f == file
                    tt = {k: u[sel] for k, u in t.items()}
                    after, _ = patch(net, raw[sel].clip(0, 127), tt, rank[file][:4])
                    effects[sel] = after
            data[phase].append(np.c_[res, effects])
    return {phase: np.concatenate(rows) for phase, rows in data.items()}


def fit(data, index):
    native = data[:, 2]
    diff = data[:, index]
    grid = np.arange(0, 4097, 16)
    values = [credit(native + scale * diff).mean() for scale in grid]
    return int(grid[np.argmax(values)])


def report(data, out):
    discovery = data["discovery"]
    fits = {name: fit(discovery, 3 + i) for i, name in enumerate(RULES)}
    records = []
    counts = []
    for phase, rows in data.items():
        for a in (0, 2, 3, 4, 5):
            w = rows if not a else rows[rows[:, 1] == a]
            if not len(w):
                continue
            for group, mask in [
                ("all", np.ones(len(w), bool)),
                ("promotion_necessary", w[:, 17] == 1),
                ("safe_capture_favors_H", w[:, 5] == 1),
                ("safe_capture_favors_L", w[:, 5] == -1),
                ("same_safe_capture", w[:, 5] == 0),
                (
                    "safe_capture_favors_H_H_clear_L_off",
                    (w[:, 5] == 1) & (w[:, 16] == 1),
                ),
                (
                    "safe_capture_favors_H_other_geometry",
                    (w[:, 5] == 1) & (w[:, 16] == 0),
                ),
            ]:
                x = w[mask]
                if not len(x):
                    continue
                native = credit(x[:, 2])
                for variant, i, scale in [
                    ("native", 5, 0),
                    *[(name, 3 + i, fits[name]) for i, name in enumerate(RULES)],
                    (
                        "capture_fit_other_materials",
                        5,
                        fit(discovery[discovery[:, 1] != a], 5),
                    ),
                ]:
                    alt = credit(x[:, 2] + scale * x[:, i])
                    effect = alt - native
                    if a == 0 and phase != "discovery":
                        lower, upper = ci(x[:, 0].astype(int), effect)
                    else:
                        lower = upper = None
                    records.append(
                        {
                            "sample": phase,
                            "white": a,
                            "stratum": group,
                            "variant": variant,
                            "weight": scale,
                            "n": len(x),
                            "native": native.mean(),
                            "adjusted": alt.mean(),
                            "rescued": int(((x[:, 2] <= 0) & (alt == 1)).sum()),
                            "harmed": int(((x[:, 2] > 0) & (alt < 1)).sum()),
                            "lower": lower,
                            "upper": upper,
                        }
                    )
                if phase != "discovery":
                    counts.append(
                        {
                            "sample": phase,
                            "white": a,
                            "stratum": group,
                            "n": len(x),
                            "native": native.mean(),
                            "donors": x[:, -1].mean(),
                            "strict_right": int((x[:, 2] > 0).sum()),
                            "strict_wrong": int((x[:, 2] < 0).sum()),
                        }
                    )
    write(out / "prediction-promotion.csv", records)
    write(out / "prediction-promotion-donors.csv", counts)


if __name__ == "__main__":
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    v.init(root)
    data = prepare(root, out, Network(root))
    report(data, out)
