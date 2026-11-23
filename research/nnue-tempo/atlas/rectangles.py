"""Verify and publish same-source reversals that survive all 48 availability controls."""

import argparse
import json
from pathlib import Path

import chess

import validate
from summarize import MATS, mat, write


def replay(a, b, coordinates):
    wk, bk, y, z, src, i, j = coordinates
    parents = []
    signs = []
    for enemy in (y, z):
        q = validate.board(a, b, wk, bk, enemy, src)
        q.turn = chess.WHITE
        assert q.is_valid() and not q.is_check()
        children = []
        for dst in (i, j):
            m = chess.Move(src, dst)
            assert m in q.legal_moves and not q.is_capture(m) and not q.gives_check(m)
            q.push(m)
            assert not q.has_legal_en_passant()
            entry = validate.state(a, b, wk, bk, enemy, dst)
            masks, replies = validate.probe(a, b, wk, bk, enemy, dst, record_moves=True)
            tags = masks[0]
            best = max(v for _, v, _ in replies)
            children.append(
                {
                    "move": m.uci(),
                    "wdl_black": validate.sign(entry["w"]),
                    "classical": int(entry["f"]),
                    "nnue": int(entry["n"]),
                    "psqt": int(entry["p"]),
                    "positional": int(entry["d"]),
                    "available_mask": tags,
                    "optimal_black_replies": [
                        move for move, v, _ in replies if v == best
                    ],
                }
            )
            q.pop()
        hi, lo = children
        assert (
            hi["available_mask"] == lo["available_mask"]
            and hi["classical"] == lo["classical"]
        )
        s = validate.sign(hi["wdl_black"] - lo["wdl_black"])
        assert s and validate.sign(hi["nnue"] - lo["nnue"]) == s
        signs.append(s)
        parents.append({"fen": q.fen(), "children": children})
    all_children = [c for parent in parents for c in parent["children"]]
    assert len({(c["classical"], c["available_mask"]) for c in all_children}) == 1
    assert signs[0] == -signs[1]
    return {"material": mat(a, b), "parents": parents}


def main(root, out):
    validate.init(root)
    examples = []
    rows = []
    fields = (
        "eligible_pairs",
        "rectangles",
        "both_nnue",
        "both_psqt",
        "both_positional",
        "neural_side_quarter_points",
        "identical_all_four",
        "both_nnue_identical",
        "both_positional_identical",
    )
    for a, b in MATS:
        metrics = []
        for folder in (root / "atlas/metrics", root / "atlas/remaining/metrics"):
            stem = f"{a}-{b}"
            metrics.append(
                list(map(int, (folder / (stem + ".cross")).read_text().split()))
            )
            coordinates = (folder / (stem + ".rectangle")).read_text().split()
            if coordinates:
                examples.append(replay(a, b, list(map(int, coordinates))))
        data = dict(zip(fields, map(sum, zip(*metrics)), strict=True))
        assert data["both_psqt"] == 0
        rows.append(dict(material=mat(a, b), **data))
    write(out / "crossovers.csv", rows)
    (out / "crossover-certificates.json").write_text(
        json.dumps(examples, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
