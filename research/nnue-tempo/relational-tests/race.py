"""A hand-derived pawn interception margin on an empty board."""

from functools import lru_cache

import chess
import numpy as np


@lru_cache(None)
def distances(piece, src):
    steps = {src: 0}
    front = [src]
    while front:
        s = front.pop(0)
        attacks = (
            chess.BB_KNIGHT_ATTACKS[s] if piece == 2 else chess.BB_DIAG_ATTACKS[s][0]
        )
        for t in chess.scan_forward(attacks):
            if t not in steps:
                steps[t] = steps[s] + 1
                front.append(t)
    return [steps.get(t, 64) for t in range(64)]


@lru_cache(None)
def slack(piece, actor, pawn):
    path = list(range(pawn - 8, -1, -8))
    paths = [path, path[1:]] if pawn // 8 == 6 else [path]
    distance = distances(piece, actor)
    return min(max(j - distance[q] for j, q in enumerate(route, 1)) for route in paths)


def scores(boards, ix):
    normal = np.zeros(len(boards), int)
    reflected = normal.copy()
    rank = normal.copy()
    for i in ix:
        b = boards[i]
        if b.turn or len(b.piece_map()) != 4:
            continue
        p = list(b.pieces(1, False))
        a = [
            (x.piece_type, s)
            for s, x in b.piece_map().items()
            if x.color and x.piece_type in (2, 3)
        ]
        if len(p) == len(a) == 1:
            pawn = p[0]
            piece, actor = a[0]
            normal[i] = slack(piece, actor, pawn)
            reflected[i] = slack(piece, actor, pawn ^ 7)
            rank[i] = pawn // 8
    return {"interception": normal, "reflected_file": reflected, "rank": rank}


def opposed(boards, ix, classical, native, positional, psqt, geometric):
    """Equal-classical-score pairs with opposite true-file and reflected-file predictions."""
    from collections import defaultdict

    groups = defaultdict(list)
    for i in ix:
        b = boards[i]
        pawns = b.pieces(1, False)
        actors = b.pieces(2, True) | b.pieces(3, True)
        if b.turn or len(pawns) != 1 or len(actors) != 1 or len(b.piece_map()) != 4:
            continue
        target = next(iter(actors))
        if any(m.to_square == target for m in b.generate_legal_captures()):
            continue
        groups[
            b.king(True),
            b.king(False),
            int(pawns),
            int(classical[i]),
            b.piece_type_at(target),
        ].append(i)
    total = 0
    credit = np.zeros(3)
    for group in groups.values():
        if len(group) < 2:
            continue
        a, b = np.triu_indices(len(group), 1)
        left = np.array(group)[a]
        right = np.array(group)[b]
        true = np.sign(
            geometric["interception"][left] - geometric["interception"][right]
        )
        fake = np.sign(
            geometric["reflected_file"][left] - geometric["reflected_file"][right]
        )
        mask = true * fake < 0
        if not mask.any():
            continue
        total += sum(mask)
        for j, score in enumerate([native, positional, psqt]):
            relation = (score[left] - score[right])[mask] * true[mask]
            credit[j] += sum(relation > 0) + 0.5 * sum(relation == 0)
    return int(total), credit.tolist()
