"""Fixed-king pawn-shift experiments. All boards use a white candidate piece."""

import chess


def formula(b):
    k = b.king(chess.WHITE)
    d = b.king(chess.BLACK)
    r = next(iter(b.pieces(chess.ROOK, chess.WHITE)))
    p = next(iter(b.pieces(chess.PAWN, chess.BLACK)))
    dist = chess.square_distance
    rank = chess.square_rank
    if chess.square_file(k) == chess.square_file(p) and rank(k) < rank(p):
        tag, v = "front", 1380 - dist(k, p)
    elif dist(d, p) >= 3 + (not b.turn) and dist(d, r) >= 3:
        tag, v = "remote", 1380 - dist(k, p)
    elif rank(d) <= 2 and dist(d, p) == 1 and rank(k) >= 3 and dist(k, p) > 2 + b.turn:
        tag, v = "supported", 80 - 8 * dist(k, p)
    else:
        tag, v = "race", 200 - 8 * (dist(k, p - 8) - dist(d, p - 8) - rank(p))
    return tag, v if b.turn else -v


def pawn_path(b, actor, pawn):
    """Same geometric convention for every moving piece. Remove the pawn when projecting its path."""
    empty = b.copy(stack=False)
    empty.remove_piece_at(pawn)
    hits = empty.attacks_mask(actor)
    path = [
        int(bool(hits & chess.BB_SQUARES[s]) or s == actor) for s in range(pawn, -1, -8)
    ]
    return path


def describe(b, actor, pawn):
    path = pawn_path(b, actor, pawn)
    captures = list(b.generate_legal_captures(to_mask=chess.BB_SQUARES[actor]))
    pawn_takes = any(m.from_square == pawn for m in captures)
    king_takes = any(m.from_square == b.king(chess.BLACK) for m in captures)
    return {
        "current": path[0],
        "next": path[1],
        "promotion": path[-1],
        "second": path[min(2, len(path) - 1)],
        "route": sum(path),
        "pawn_takes": int(pawn_takes),
        "king_takes": int(king_takes),
        "loose": int(bool(captures)),
    }


def sign(n):
    return int(n > 0) - int(n < 0)


def outcomes(b, tb):
    """All replies and WDL from Black's perspective, including terminal positions."""
    replies = []
    for m in list(b.legal_moves):
        b.push(m)
        value = -sign(tb.probe_wdl(b))
        b.pop()
        replies.append((m, value))
    return replies


def ordered(m):
    return (m[0] > m[1]) and (m[2] < m[3])


def mixed(v):
    return v[0] - v[1] - v[2] + v[3]
