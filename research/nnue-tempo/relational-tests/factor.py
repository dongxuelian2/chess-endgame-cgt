"""Exact two-feature decomposition of positional-head scores at a fixed pair of kings."""

from collections import defaultdict

import numpy as np

import search_games


def factor(boards, eligible, positional):
    net = search_games.net
    ans = {
        k: positional.copy()
        for k in ("background", "kings", "pawn", "actor", "additive", "interaction")
    }
    pure = np.zeros_like(positional)
    groups = defaultdict(list)
    for i in np.flatnonzero(eligible):
        b = boards[i]
        if b.turn or len(b.piece_map()) != 4:
            continue
        pawns = list(b.pieces(1, False))
        actors = [
            s
            for s, p in b.piece_map().items()
            if p.color and p.piece_type in (2, 3, 4, 5)
        ]
        if len(pawns) == len(actors) == 1:
            s = actors[0]
            groups[b.piece_type_at(s)].append(
                (i, [b.king(True), b.king(False), pawns[0], s])
            )
    for a, rows in groups.items():
        for j in range(0, len(rows), 128):
            ids, squares = zip(*rows[j : j + 128])
            ids = list(ids)
            men = (6, -6, -1, a)
            sq = np.array(squares)

            def score(columns, sq=sq, men=men):
                x, ps = net.read(sq[:, columns], [men[i] for i in columns])
                _, _, _, deep = net.score(x, ps, 0)
                return -np.trunc(deep / 16).astype(int)

            kings = score([0, 1])
            p = score([0, 1, 2])
            actor = score([0, 1, 3])
            full = score([0, 1, 2, 3])
            add = p + actor - kings
            assert np.array_equal(full, positional[ids])
            pure[ids] = full
            for k, values in [
                ("background", 0),
                ("kings", kings),
                ("pawn", p),
                ("actor", actor),
                ("additive", add),
                ("interaction", full - add),
            ]:
                ans[k][ids] = values
    return {**ans, "pair_boards_only": pure}
