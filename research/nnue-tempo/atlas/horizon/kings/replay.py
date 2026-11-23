"""Independent move generation, restriction certificates and a size-matched deletion control."""

import csv
import gzip
import json
import sys
from dataclasses import dataclass, field
from math import comb, prod
from pathlib import Path

import chess
import numpy as np
import validate as v
from panel import TAGS
from summarize import write

# One illustrative witness per game. All successful restrictions remain in the census.
WITNESS = {
    "0ykns7zh": "b check",
    "1rb6lo8v": "b push_pawn",
    "2mn4wzs3": "w king_step_under_pawn_file_control",
    "87gsv0w8": "w king_step_under_last_rank_control",
    "880q0c1r": "w file_check",
    "9jz0k85n": "b contact_check",
    "9wbdo6ze": "w private_quiet_man_move",
    "abzx8wn1": "w after_push_uses_last_rank",
    "b0zgvlxp": "b private_king_step",
    "ba4yj2mt": "w king_step_under_pawn_file_control",
    "boieijcd": "b private_pawn_push",
    "d6xmnbot": "b private_king_step",
    "dnp9n41r": "b push_pawn",
    "ke4ai0s1": "b private_pawn_push",
    "lb88grk7": "w private_quiet_man_move",
    "mq25ev7a": "w private_quiet_man_move",
    "mx9gsckz": "w guard_own_man",
    "oc2wmsiy": "w after_push_captures_pawn",
    "oj4ogsus": "w cut_kings",
    "pvsyd1ny": "b push_pawn",
    "q0vkoy4l": "w contact_check",
    "q25jew0x": "w after_push_king_closes",
    "qs22m3il": "w after_push_king_closes",
    "vuhg9que": "w after_push_captures_pawn",
    "whlrjock": "w king_step_toward_pawn_under_last_rank_control",
    "x16c3pdb": "w pin_man",
    "yq6lhrnd": "b push_pawn",
    "zk0d9uor": "w king_step_under_last_rank_control",
}


@dataclass
class Action:
    code: tuple
    value: int
    tags: set
    san: str
    replies: list = field(default_factory=list)


def man(q, side):
    mask = q.occupied_co[side] & ~q.kings
    return chess.msb(mask) if mask else None


def action(q, m):
    us = q.turn
    them = not us
    f, t = m.from_square, m.to_square
    old = man(q, them)
    kind = q.piece_type_at(f)
    king = kind == 6
    take = q.is_capture(m)
    san = q.san(m)
    q.push(m)
    own, enemy = man(q, us), man(q, them)
    k, other = q.king(us), q.king(them)
    control = q.attacks(t)
    direct = other in control
    check = q.is_check()
    rank, file = chess.square_rank, chess.square_file
    dist = chess.square_distance
    tags = set()

    def tag(s, condition):
        if condition:
            tags.add(s)

    tag("step_king", king)
    tag("move_man", not king)
    tag("push_pawn", kind == 1 and not take)
    tag("check", check)
    tag("check_from_far", direct and dist(t, other) >= 3)
    tag("rank_check", direct and rank(t) == rank(other))
    tag("file_check", direct and file(t) == file(other))
    tag(
        "diagonal_check",
        direct
        and q.piece_type_at(t) in (3, 5)
        and rank(t) != rank(other)
        and file(t) != file(other),
    )
    tag("contact_check", direct and dist(t, other) == 1)
    tag(
        "check_and_attack_man",
        check and enemy in control if enemy is not None else False,
    )
    tag("pin_man", enemy is not None and q.is_pinned(them, enemy))
    tag("capture_man", take)
    tag("attack_man", enemy is not None and enemy in control)
    if enemy is not None and q.piece_type_at(enemy) == 1:
        nxt = enemy + (8 if them else -8)
        prom = file(enemy) + (56 if them else 0)
        tag("attack_pawn_next", nxt in control)
        tag("attack_promotion", prom in control)
        tag("occupy_promotion", t == prom)
        tag("occupy_pawn_file", file(t) == file(enemy))
        tag(
            "rear_attack_pawn",
            file(t) == file(enemy)
            and (t < enemy if them else t > enemy)
            and enemy in control,
        )
        tag("blockade_pawn", t == nxt)
        if own is not None and q.piece_type_at(own) in (4, 5):
            back = rank(own) == rank(prom)
            guard = back and (own == prom or prom in q.attacks(own))
            tag("major_on_last_rank", not king and back)
            tag(
                "major_attacks_pawn_from_in_front",
                not king
                and file(own) == file(enemy)
                and (own > enemy if them else own < enemy)
                and enemy in q.attacks(own),
            )
            tag("king_step_under_pawn_file_control", king and file(own) == file(enemy))
            tag("king_step_under_last_rank_control", king and guard)
            tag(
                "king_step_toward_pawn_under_last_rank_control",
                king and guard and dist(t, enemy) < dist(f, enemy),
            )
    tag("guard_own_man", king and own is not None and dist(k, own) == 1)
    tag(
        "king_approach_other_man",
        king and old is not None and dist(t, old) < dist(f, old),
    )
    if king and own is not None and q.piece_type_at(own) == 1:
        prom = file(own) + (56 if us else 0)
        tag("king_approach_own_promotion", dist(t, prom) < dist(f, prom))
        tag("king_guard_own_pawn_next", dist(t, own + (8 if us else -8)) == 1)
        tag(
            "king_ahead_of_own_pawn", rank(t) > rank(own) if us else rank(t) < rank(own)
        )
        tag(
            "king_attack_with_promotion_guard",
            enemy is not None and dist(t, prom) == 1 and dist(t, enemy) == 1,
        )
        tag(
            "king_steps_down_pawn_file",
            enemy is not None and file(t) == file(f) and (t > f if us else t < f),
        )
    tag("cut_kings", q.piece_type_at(t) in (4, 5) and v.cut(k, other, t, int(control)))
    tag("uncover_check", king and check)
    tag("approach_enemy_king", king and dist(t, other) < dist(f, other))
    value = v.sign(v.tb.probe_wdl(q)) * (1 if q.turn == chess.BLACK else -1)
    return Action((king, t, m.promotion), value, tags, san)


def branches(q):
    result = []
    for m in list(q.legal_moves):
        b = action(q, m)
        for u in list(q.legal_moves):
            w = action(q, u)
            if "push_pawn" in b.tags:
                for source, target in [
                    ("capture_man", "captures_pawn"),
                    ("check", "checks"),
                    ("king_approach_other_man", "king_closes"),
                    ("attack_promotion", "man_covers_promotion"),
                    ("occupy_promotion", "blocks_queening"),
                    ("major_on_last_rank", "uses_last_rank"),
                    ("step_king", "king_step"),
                ]:
                    if source in w.tags and not (
                        target == "man_covers_promotion" and w.code[0]
                    ):
                        w.tags.add("after_push_" + target)
            b.replies.append(w)
            q.pop()
        assert b.value == min((w.value for w in b.replies), default=b.value)
        result.append(b)
        q.pop()
    assert max(b.value for b in result) == v.sign(v.tb.probe_wdl(q))
    return result


def private(a):
    a.tags.add("private_destination")
    a.tags.add("private_king_step" if a.code[0] else "private_man_move")
    if "check" in a.tags:
        a.tags.add("private_check")
    elif not a.code[0] and "capture_man" not in a.tags:
        a.tags.add("private_quiet_man_move")
    if "push_pawn" in a.tags:
        a.tags.add("private_pawn_push")


def mark(tree, other):
    ref = {b.code: {w.code for w in b.replies} for b in other}
    for b in tree:
        if b.code not in ref:
            private(b)
        else:
            for w in b.replies:
                if w.code not in ref[b.code]:
                    private(w)


def remaining(actions, tag):
    return [a for a in actions if tag not in a.tags] or actions


def evaluate(tree, side, tag):
    if side == "b":
        return max(b.value for b in remaining(tree, tag))
    return max(
        min((w.value for w in remaining(b.replies, tag)), default=b.value) for b in tree
    )


def distribution(tree, side, tag):
    # Uniform subsets of the observed retained size, independent at every node.
    def proportion(n_ok, n, retain):
        return comb(n_ok, retain) / comb(n, retain)

    def cdf(x):
        if side == "b":
            n = len(tree)
            r = len(remaining(tree, tag))
            return proportion(sum(a.value <= x for a in tree), n, r)
        terms = []
        for b in tree:
            n = len(b.replies)
            r = len(remaining(b.replies, tag))
            terms.append(
                1 - proportion(sum(w.value > x for w in b.replies), n, r)
                if n
                else float(b.value <= x)
            )
        return prod(terms)

    return [cdf(-1), cdf(0) - cdf(-1), 1 - cdf(0)]


def study(trees, side, tag):
    hi, lo = trees
    values = [evaluate(t, side, tag) for t in trees]
    assert values[0] <= values[1]
    if side == "b":
        best = max(b.value for b in hi)
        witness = " ".join(sorted(b.san for b in hi if b.value == best))
    else:
        lines = []
        for b in lo:
            new = min((w.value for w in remaining(b.replies, tag)), default=b.value)
            if new == values[1] and new > b.value:
                opt = " ".join(sorted(w.san for w in b.replies if w.value == b.value))
                lines.append(b.san + " / " + opt)
        witness = " | ".join(sorted(lines))
    ph, pl = [distribution(t, side, tag) for t in trees]
    null = sum(sum(ph[: i + 1]) * p for i, p in enumerate(pl))
    return values, witness, null


def audit(qs, masks):
    trees = [branches(q) for q in qs]
    mark(*trees)
    mark(*trees[::-1])
    for side, native in zip("bw", masks):
        assert {
            t
            for t in TAGS
            if evaluate(trees[0], side, t) <= evaluate(trees[1], side, t)
        } == native, (qs[0].fen(), side, native)
    return trees


def run(root, out):
    v.init(root)
    rows = {r["game"]: r for r in csv.DictReader((out / "residual-28.csv").open())}
    assert rows.keys() == WITNESS.keys()
    ledger = []
    with gzip.open(root / "atlas/horizon/played.gz", "rt") as games:
        for _, gid, fen, u, z, *_ in map(json.loads, games):
            if gid not in rows:
                continue
            row = rows[gid]
            q = chess.Board(fen)
            moves = sorted(
                map(chess.Move.from_uci, (u, z)),
                key=lambda m: m.to_square != int(row["high"]),
            )
            qs = []
            for m in moves:
                b = q.copy()
                b.push(m)
                qs.append(b)
            trees = audit(
                qs, [set(row[f"{s}_erases"].split()) for s in ("black", "white")]
            )
            side, tag = WITNESS[gid].split()
            values, witness, null = study(trees, side, tag)
            ledger.append(
                {
                    "game": gid,
                    "parent_fen": fen,
                    "high": q.san(moves[0]),
                    "low": q.san(moves[1]),
                    "native_margin": row["nnue_margin"],
                    "black_wdl_high": max(b.value for b in trees[0]),
                    "black_wdl_low": max(b.value for b in trees[1]),
                    "restricted_player": side,
                    "action": tag,
                    "restricted_high": values[0],
                    "restricted_low": values[1],
                    "witness": witness,
                    "uniform_erasure_probability": round(null, 8),
                }
            )
    assert len(ledger) == 28
    write(out / "played-certificates.csv", sorted(ledger, key=lambda r: r["game"]))
    rng = np.random.default_rng(692051)
    samples = []
    work = root / "atlas/horizon/kings"
    for a in range(1, 6):
        for b in range(1, 6):
            files = sorted(work.glob(f"pairs-{a}-{b}-*.out"))
            if not files:
                continue
            data = np.concatenate([np.loadtxt(f, dtype="i8", ndmin=2) for f in files])
            for i in rng.choice(len(data), min(6, len(data)), replace=False):
                w, k, y, x, z, vh, vl, m1, m2 = map(int, data[i])
                trees = audit(
                    [v.board(a, b, w, k, y, t) for t in (x, z)],
                    [
                        {label for j, label in enumerate(TAGS) if mask >> j & 1}
                        for mask in (m1, m2)
                    ],
                )
                assert [max(p.value for p in t) for t in trees] == [vh, vl]
            samples.append(
                {
                    "white_man": a,
                    "black_man": b,
                    "pairs": min(6, len(data)),
                    "action_classes": len(TAGS),
                    "players": 2,
                }
            )
    write(out / "independent-replay.csv", samples)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
