"""Separate straight and capturing promotions on held-out decisions across all eight files."""

import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network, credit
from promotion_geometry import relation
from summarize import write

from cohorts import kings
from effects import batches, ci, patch
from perspectives import channels, played_groups


def labels(job):
    a, r = job
    w, k, y, h, l, _ = r
    vals = []
    for x in (h, l):
        q = v.board(a, 1, w, k, y, x)
        moves = []
        for m in list(q.legal_moves):
            capture = q.is_capture(m)
            q.push(m)
            moves.append(
                (
                    bool(m.promotion and capture),
                    bool(m.promotion and not capture),
                    bool(m.promotion and m.promotion != chess.QUEEN),
                    -v.sign(v.tb.probe_wdl(q)),
                )
            )
            q.pop()
        base = max(z[-1] for z in moves)
        vals.append(
            [max((z[-1] for z in moves if not z[i]), default=base) for i in range(3)]
        )
    return np.array(vals[0]) <= np.array(vals[1])


def groups(root):
    for a in range(2, 6):
        d = np.load(root / "manuscript" / f"validation-{a}-1.npz")
        selected = d["rows"][d["mask"]]
        for f in range(8):
            yield "enumerated", a, f, selected[(selected[:, 2] ^ 7) == 8 + f]
    for period, panel in played_groups(root):
        for (a, f), items in panel.items():
            yield period, a, f, np.array([r for r, _ in items])


def main(root, out):
    v.init(root)
    net = Network(root)
    ranked = channels(out)
    records = []
    by_action = defaultdict(list)
    examples = []
    with ProcessPoolExecutor(
        8, mp_context=get_context("fork"), initializer=v.init, initargs=(root,)
    ) as pool:
        for sample, a, f, r in groups(root):
            vals = np.array(
                list(pool.map(labels, [(a, row) for row in r.tolist()], chunksize=80)),
                bool,
            )
            w, k, y, h, l, _ = r.T
            v.state(a, 1, 0, 0, 0, 0)
            st = v.maps[a, 1]
            margins = st[w, k, y, h]["n"].astype(int) - st[w, k, y, l]["n"].astype(int)
            target = y - 8
            same = relation(a, h, target, w, k) == relation(a, l, target, w, k)
            patched = []
            for batch in batches(r):
                raw, p, _ = net.encode(batch, a, 1)
                t = net.forward(raw, p)
                patched.extend(patch(net, raw.clip(0, 127), t, ranked[f][:4])[0])
            patched = np.array(patched)
            cap, push, minor = vals.T
            for tier, mask in [
                ("all", np.ones(len(r), bool)),
                ("same_coverage", same),
            ]:
                for tag, flag in [
                    ("straight_only", push & ~cap),
                    ("capture_only", cap & ~push),
                    ("both_separately", cap & push),
                    ("neither_separately", ~cap & ~push),
                    ("underpromotion_required", minor),
                ]:
                    z = margins[mask & flag]
                    if (
                        tier == "all"
                        and tag in ("straight_only", "capture_only")
                        and len(z)
                    ):
                        name = "bgh" if f in (1, 6, 7) else "acdef"
                        by_action[sample, name, tag].append(
                            np.c_[kings(r[flag]), credit(z), patched[flag]]
                        )
                    if (
                        sample == "enumerated"
                        and f in (1, 6, 7)
                        and tier == "all"
                        and tag == "capture_only"
                        and np.any(flag & (margins < 0))
                    ):
                        row = r[np.flatnonzero(flag & (margins < 0))[0]]
                        wk, bk, pawn, high, low, source = map(int, row)
                        parent = v.board(a, 1, wk, bk, pawn, source)
                        parent.turn = True
                        for dest in (high, low):
                            board = v.board(a, 1, wk, bk, pawn, dest)
                            score = v.state(a, 1, wk, bk, pawn, dest)
                            actions = []
                            for move in board.legal_moves:
                                san = board.san(move)
                                board.push(move)
                                wd = -v.sign(v.tb.probe_wdl(board))
                                board.pop()
                                if wd == v.sign(score["w"]):
                                    actions.append(san)
                            examples.append(
                                {
                                    "white": a,
                                    "file": chr(97 + f),
                                    "parent": parent.fen(),
                                    "choice": parent.san(chess.Move(source, dest)),
                                    "black_wdl": v.sign(score["w"]),
                                    "native": int(score["n"]),
                                    "optimal_replies": " ".join(sorted(actions)),
                                }
                            )
                    records.append(
                        {
                            "sample": sample,
                            "white": a,
                            "file": chr(97 + f),
                            "coverage": tier,
                            "deletion": tag,
                            "n": len(z),
                            "correct": int((z > 0).sum()),
                            "tie": int((z == 0).sum()),
                            "wrong": int((z < 0).sum()),
                            "points": credit(z).sum(),
                            "patched_points": patched[mask & flag].sum(),
                        }
                    )
    write(out / "promotion-actions.csv", records)
    uncertainty = []
    for (sample, name, tag), parts in sorted(by_action.items()):
        z = np.concatenate(parts)
        effect = z[:, 1] - z[:, 2]
        lo, hi = ci(z[:, 0], effect)
        uncertainty.append(
            {
                "sample": sample,
                "files": name,
                "deletion": tag,
                "n": len(z),
                "native": z[:, 1].mean(),
                "patched": z[:, 2].mean(),
                "loss": effect.mean(),
                "lower": lo,
                "upper": hi,
            }
        )
    write(out / "promotion-action-intervals.csv", uncertainty)
    write(out / "promotion-witnesses.csv", examples)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
