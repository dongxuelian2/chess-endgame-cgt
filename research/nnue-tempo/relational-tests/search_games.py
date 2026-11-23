"""Complete three-ply games and literal one-reply horizon interventions, using true transposition ids."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import chess
import chess.syzygy
import numpy as np

from certificate import certificate, evaluate, frontier
from clip_pair import suppress
from model import Model
from signed import signed_certificate

TREATMENTS = ("king", "pawn", "both", "clip_pair", "clip_permuted")
from measure import load, table
from transfer import MATERIALS


def init(root):
    global net, tb
    net = Model(root)
    tb = chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular")
    tb.add_directory(str(root / "tb5"))


def key(b):
    # Unlimited clock; no castling rights in these roots. Include legal EP.
    return (*b._transposition_key(),)


def leaf_score(boards, components=False):
    groups = defaultdict(list)
    out = np.empty(len(boards), int)
    psqt = np.zeros(len(boards), int)
    positional = np.zeros(len(boards), int)
    for i, b in enumerate(boards):
        if b.is_checkmate():
            out[i] = -30000 if b.turn else 30000
            continue
        if b.is_stalemate() or b.is_insufficient_material():
            out[i] = 0
            continue
        pcs = sorted(
            [
                (c.piece_type if c.color else -c.piece_type, s)
                for s, c in b.piece_map().items()
                if c.piece_type != 6
            ]
        )
        men = (6, -6, *(p for p, s in pcs))
        groups[(men, b.turn)].append(
            (i, [b.king(True), b.king(False), *(s for p, s in pcs)])
        )
    for (men, turn), samples in groups.items():
        for j in range(0, len(samples), 256):
            idx, sq = zip(*samples[j : j + 256])
            x, ps = net.read(sq, men)
            if turn:
                x = np.concatenate([x[:, 1024:], x[:, :1024]], axis=1)
                ps = -ps
            score, _, _, deep = net.score(x, ps, (len(men) - 1) // 4)
            if components:
                psqt[list(idx)] = np.trunc(ps / 16).astype(int) * (1 if turn else -1)
                positional[list(idx)] = np.trunc(deep / 16).astype(int) * (
                    1 if turn else -1
                )
            out[list(idx)] = score if turn else -score
    return (out, psqt, positional) if components else out


def game(job, audit=None):
    _root, cohort, boardfen = job
    b = chess.Board(boardfen)
    ids = {}
    boards = []
    tree = []
    rootmoves = []
    replynames = []
    truth = []

    def insert():
        k = key(b)
        if k not in ids:
            ids[k] = len(boards)
            boards.append(b.copy(stack=False))
        return ids[k]

    for m in list(b.legal_moves):
        b.push(m)
        rootmoves.append(m.uci())
        action = []
        rn = []
        try:
            truth.append(-int(np.sign(tb.probe_wdl(b))))
        except chess.syzygy.MissingTableError:
            truth.append(None)
        replies = [] if b.is_insufficient_material() else list(b.legal_moves)
        if not replies:
            action.append([insert()])
            rn.append("terminal")
        for z in replies:
            b.push(z)
            rn.append(z.uci())
            children = [] if b.is_insufficient_material() else list(b.legal_moves)
            row = []
            if not children:
                row.append(insert())
            for q in children:
                b.push(q)
                row.append(insert())
                b.pop()
            action.append(row)
            b.pop()
        tree.append(action)
        replynames.append(rn)
        b.pop()
    n = len(boards)
    if len(tree) < 2:
        return []
    if audit == "classical":
        return tree, boards, truth, rootmoves
    after = defaultdict(list)
    pairs = defaultdict(set)
    for i in range(n):
        child = boards[i]
        if child.turn or child.is_insufficient_material():
            continue
        for m in child.legal_moves:
            captured = child.piece_type_at(m.to_square)
            if child.piece_type_at(m.from_square) == chess.KING and captured in (
                2,
                3,
                4,
                5,
            ):
                t = "king"
            elif child.piece_type_at(m.from_square) == chess.PAWN and (
                m.promotion or captured in (2, 3, 4, 5)
            ):
                t = "pawn"
            else:
                continue
            if t == "pawn" and captured in (2, 3, 4, 5):
                pairs[i].add((m.from_square, m.to_square))
            child.push(m)
            k = key(child)
            if k not in ids:
                ids[k] = len(boards)
                boards.append(child.copy(stack=False))
            after[i, t].append(ids[k])
            child.pop()
    scores = leaf_score(boards)
    hi = scores[:n]
    base = evaluate(tree, hi)
    leader = int(base.argmax())
    original = np.flatnonzero(base == base.max())
    alts = base.copy()
    alts[leader] = -100000
    occ = sum(len(row) for act in tree for row in act)
    result = []
    pair = suppress(net, boards, pairs, hi)
    placebo = hi.copy()
    eligible = sorted(i for i, p in pairs.items() if len(p) == 1)
    rng = np.random.default_rng(
        int.from_bytes(
            hashlib.sha256(("shuffle-v1" + boardfen).encode()).digest()[:8], "big"
        )
    )
    placebo[eligible] += rng.permutation((pair - hi)[eligible])
    if audit == "permute":
        return tree, hi, pair, eligible, leader
    for which in TREATMENTS:
        if audit is not None and which != audit:
            continue
        fraction_costs = None
        if which.startswith("clip_"):
            lo = pair if which == "clip_pair" else placebo
            best = signed_certificate(tree, hi, lo, leader)
            relaxed = None
            details = None
        else:
            lo = hi.copy()
            for (i, t), lst in after.items():
                if t == which or which == "both":
                    lo[i] = min(lo[i], min(scores[lst]))
            best, relaxed, details = certificate(tree, hi, lo, leader)
            front = frontier(tree[leader], hi, lo, details)
            fraction_costs = [
                min((c for t, c in front if t < Fraction(q, 4)), default=None)
                for q in (1, 2, 3, 4)
            ]
            if len(original) == 1:
                assert fraction_costs[-1] == (len(best[1]) if best else None)
        if audit is not None:
            return tree, hi, lo, leader
        final = evaluate(tree, lo)
        chosen = None if best is None else best[1]
        terminal = None
        witness = None
        if chosen is not None:
            edited = hi.copy()
            edited[chosen] = lo[chosen]
            v = evaluate(tree, edited)
            assert v.max() > v[leader]
            rival = int(v.argmax())
            terminal = truth[rival]
            witness = {
                "fen": boardfen,
                "cohort": cohort,
                "extension": which,
                "leader": rootmoves[leader],
                "challenger": rootmoves[rival],
                "reply": replynames[leader][best[0]],
                "original_values": [int(base[leader]), int(base[rival])],
                "witness_values": [int(v[leader]), int(v[rival])],
                "rows": details,
                "edits": [
                    {"fen": boards[i].fen(), "before": int(hi[i]), "after": int(lo[i])}
                    for i in chosen
                ],
            }
        best_full = np.flatnonzero(final == final.max())
        result.append(
            {
                "fen": boardfen,
                "cohort": cohort,
                "extension": which,
                "roots": len(tree),
                "occurrences": occ,
                "states": n,
                "eligible": int(sum(hi != lo)),
                "positive": int(sum(lo > hi)),
                "negative": int(sum(lo < hi)),
                "quarter_costs": fraction_costs,
                "unique": len(original) == 1,
                "incumbent": rootmoves[leader],
                "k": None if chosen is None else len(chosen),
                "relaxed": None if relaxed is None else int(relaxed),
                "complete_displaces": bool(final[leader] < final.max()),
                "baseline_wdl": truth[leader],
                "witness_wdl": terminal,
                "baseline_margin": int(base[leader] - alts.max()),
                "full_wdl": [truth[i] for i in best_full],
                "witness": witness,
            }
        )
    return result


def main(root, dest, count):
    with gzip.open(root / "pawn-geometry/played.gz") as f:
        played = list(map(json.loads, f))
    jobs = []
    for group in ("played", "random"):
        for m in MATERIALS:
            ds = (
                [r for r in played if r["material"] == m]
                if group == "played"
                else load(root, "random-" + m)
            )
            placements = sorted(
                {r["parent"].split(" ")[0] for r in ds},
                key=lambda x: hashlib.sha256(("tree-v1" + x).encode()).digest(),
            )
            jobs.extend((root, group, x + " w - - 0 1") for x in placements[:count])
    cases = []
    examples = []
    with ProcessPoolExecutor(10, initializer=init, initargs=(root,)) as pool:
        for batch in pool.map(game, jobs, chunksize=1):
            for r in batch:
                ex = r.pop("witness")
                if ex:
                    examples.append(ex)
                cases.append(r)
    with gzip.open(root / "relational-tests/games.gz", "wt") as f:
        for r in cases:
            print(json.dumps(r), file=f)
    out = []
    for co in ("played", "random"):
        for what in TREATMENTS:
            allcases = [
                r for r in cases if r["cohort"] == co and r["extension"] == what
            ]
            rs = [r for r in allcases if r["unique"]]
            yes = [r for r in rs if r["k"] is not None]
            movable = [
                r
                for r in yes
                if r["baseline_wdl"] is not None and r["witness_wdl"] is not None
            ]
            out.append(
                {
                    "cohort": co,
                    "extension": what,
                    "roots": len(allcases),
                    "unique": len(rs),
                    "leaf_occurrences": sum(r["occurrences"] for r in allcases),
                    "distinct_per_root_sum": sum(r["states"] for r in allcases),
                    "possible": len(yes),
                    "singleton": sum(r["k"] == 1 for r in yes),
                    "max_minimum": max([r["k"] for r in yes], default=None),
                    "silent_bulk": sum(not r["complete_displaces"] for r in yes),
                    "sharing_changes_min": None
                    if what.startswith("clip_")
                    else sum(r["k"] != r["relaxed"] for r in rs),
                    "eligible_but_impossible": sum(
                        r["eligible"] > 0 and r["k"] is None for r in rs
                    ),
                    "wdl_improved": sum(
                        r["witness_wdl"] > r["baseline_wdl"] for r in movable
                    ),
                    "wdl_equal": sum(
                        r["witness_wdl"] == r["baseline_wdl"] for r in movable
                    ),
                    "wdl_worsened": sum(
                        r["witness_wdl"] < r["baseline_wdl"] for r in movable
                    ),
                }
            )
    table(dest / "certificates.csv", out)

    def lookup(e):
        return e["fen"], e["cohort"], e["extension"]

    ei = {lookup(e): e for e in examples}
    keep = []
    for normal in (False, True):
        rs = [
            r
            for r in cases
            if r["unique"] and r["k"] and r["complete_displaces"] == normal
        ]
        for r in sorted(rs, key=lambda r: -r["k"])[:8]:
            keep.append(ei[lookup(r)])
    for what in ("clip_pair", "clip_permuted"):
        rs = [r for r in cases if r["unique"] and r["k"] and r["extension"] == what]
        for r in sorted(rs, key=lambda r: -r["k"])[:3]:
            keep.append(ei[lookup(r)])
    keep = list({lookup(e): e for e in keep}.values())
    (dest / "witnesses.json").write_text(json.dumps(keep, indent=1) + "\n")
    # Retain all per-root sizes, including impossibility, for independent checks.
    with gzip.open(dest / "root-costs.json.gz", "wb") as f:
        f.write(json.dumps(cases, separators=(",", ":")).encode())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--count", type=int, default=60)
    a = p.parse_args()
    main(a.root, a.dest, a.count)
