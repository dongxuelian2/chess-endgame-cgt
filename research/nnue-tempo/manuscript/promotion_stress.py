"""Stress-test the four-piece promotion correction with an opposing seventh-rank pawn."""

import csv
import sys
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import credit
from summarize import write

from cohorts import orbit
from effects import ci
from promotion_extension import eval_positions, option, promotion_options


def sample(a, rng, held):
    visited = set()
    while True:
        w, k = rng.choice(64, 2, replace=False).tolist()
        group = orbit(w, k)
        if group not in held or chess.square_distance(w, k) < 2:
            continue
        y = int(rng.integers(8, 16))
        back = [d for d in chess.SquareSet(chess.BB_PAWN_ATTACKS[False][y])]
        x = back[int(rng.integers(len(back)))]
        p = int(rng.integers(48, 56))
        if len({w, k, y, x, p}) != 5:
            continue
        target = v.board(a, 1, w, k, y, x)
        target.set_piece_at(p, chess.Piece(chess.PAWN, True))
        if not target.is_valid() or target.is_check() or not option(target):
            continue
        # Generate common legal starts by reversibly moving the nonpawn on an empty square.
        sources = list(chess.SquareSet(target.attacks_mask(x) & ~target.occupied))
        rng.shuffle(sources)
        for s in sources:
            parent = target.copy(stack=False)
            parent.remove_piece_at(x)
            parent.set_piece_at(s, chess.Piece(a, True))
            parent.turn = True
            if not parent.is_valid() or not parent.is_legal(chess.Move(s, x)):
                continue
            alternatives = []
            for mv in list(parent.generate_legal_moves(from_mask=chess.BB_SQUARES[s])):
                if mv.to_square == x or parent.is_capture(mv):
                    continue
                q = parent.copy(stack=False)
                q.push(mv)
                if not q.is_check() and any(q.legal_moves) and not option(q):
                    alternatives.append(q)
            if alternatives:
                other = alternatives[int(rng.integers(len(alternatives)))]
                key = (w, k, y, p, x, other.board_fen())
                if key in visited:
                    continue
                visited.add(key)
                yield group, parent.fen(), target, other
                break


def main(root, out):
    v.init(root)
    v.tb.add_directory(root / "tb5")
    with (out / "prediction-promotion.csv").open() as f:
        weight = next(
            int(r["weight"])
            for r in csv.DictReader(f)
            if r["sample"] == "discovery"
            and r["white"] == "0"
            and r["stratum"] == "all"
            and r["variant"] == "surviving_capture"
        )
    held = set(
        np.random.default_rng(96164).permutation(
            sorted({orbit(w, k) for w in range(64) for k in range(64)})
        )[::2]
    )
    summary = []
    counter = []
    examples = []
    for a in (2, 3, 4):
        rng = np.random.default_rng(17996 + a)
        proposals = []
        positions = []
        groups = []
        reply_filter = []
        total = np.zeros(3, dtype=int)
        tie = 0
        for n, (g, parent, q, r) in zip(range(2400), sample(a, rng, held)):
            h, l = (v.sign(v.tb.probe_wdl(p)) for p in (q, r))
            total[h + 1] += 1
            if h == l:
                tie += 1
                continue
            proposals.append((parent, q.fen(), r.fen(), h, l))
            positions.extend([q.fen(), r.fen()])
            groups.append(g)
            reply_filter.append(promotion_options(q)[1])
        scores = eval_positions(root / "sf141/src/stockfish", positions).reshape(
            -1, 2, 2
        )
        s = np.sign(np.array([(e[3] - e[4]) for e in proposals]))
        m = (scores[:, 0, 1] - scores[:, 1, 1]) * s
        for name, mask in [
            ("all", np.ones(len(s), bool)),
            ("classical_tie", scores[:, 0, 0] == scores[:, 1, 0]),
            ("option_favors_H", s > 0),
            ("option_favors_L", s < 0),
        ]:
            native = credit(m[mask])
            alt = credit(m[mask] + s[mask] * weight)
            if len(native):
                low, high = ci(np.array(groups)[mask], alt - native)
                summary.append(
                    {
                        "white": a,
                        "stratum": name,
                        "generated": int(total.sum()),
                        "equal_wdl": tie,
                        "option_loss": int(total[0]),
                        "option_draw": int(total[1]),
                        "n": len(native),
                        "native": native.mean(),
                        "adjusted": alt.mean(),
                        "rescued": int(((native < 1) & (alt == 1)).sum()),
                        "harmed": int(((native == 1) & (alt < 1)).sum()),
                        "lower": low,
                        "upper": high,
                    }
                )

        deny = np.array(reply_filter, bool)
        for name, use in [
            ("all", np.ones(len(s), bool)),
            ("option_favors_H", s > 0),
            ("option_favors_L", s < 0),
        ]:
            native = credit(m[use])
            for variant, give in [
                ("original", np.ones(len(s), bool)),
                ("filter_white_promotion", deny),
            ]:
                alt = credit(m[use] + weight * s[use] * give[use])
                low, hi = ci(np.array(groups)[use], alt - native)
                counter.append(
                    {
                        "white": a,
                        "stratum": name,
                        "variant": variant,
                        "n": int(use.sum()),
                        "changed_boards": int((use & give).sum()),
                        "native": native.mean(),
                        "adjusted": alt.mean(),
                        "rescued": int(((native < 1) & (alt == 1)).sum()),
                        "harmed": int(((native == 1) & (alt < 1)).sum()),
                        "lower": low,
                        "upper": hi,
                    }
                )
        bad = np.flatnonzero((m > 0) & (m + s * weight < 0))
        for i in bad[:3]:
            parent, opt, other, h, l = proposals[i]
            examples.append(
                {
                    "white": a,
                    "parent": parent,
                    "capture_child": opt,
                    "other_child": other,
                    "capture_wdl": h,
                    "other_wdl": l,
                    "native_capture_minus_other": int(
                        scores[i, 0, 1] - scores[i, 1, 1]
                    ),
                }
            )
    write(out / "prediction-five-piece-stress.csv", summary)
    write(out / "prediction-five-piece-reply-filter.csv", counter)
    write(out / "prediction-five-piece-stress-witnesses.csv", examples)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
