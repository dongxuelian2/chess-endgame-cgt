import argparse
import csv
import json
from pathlib import Path

import chess
import chess.syzygy
from geometry import describe, formula, outcomes, sign


def boards(r):
    b = chess.Board(r["parent"])
    edit = chess.Move.from_uci(r["relocation"])
    e = b.copy()
    e.set_piece_at(edit.to_square, e.remove_piece_at(edit.from_square))
    for parent in (b, e):
        p = next(iter(parent.pieces(chess.PAWN, chess.BLACK)))
        for k in ("move_a", "move_b"):
            m = chess.Move.from_uci(r[k])
            ch = parent.copy()
            assert m in ch.legal_moves
            ch.push(m)
            yield ch, m.to_square, p


def audit(root):
    path = (
        Path(__file__).resolve().parents[1]
        / "circuits/results/handwritten-collisions.csv"
    )
    rows = []
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        for r in csv.DictReader(path.open()):
            if r["material"] != "KRvKP":
                continue
            data = []
            for b, a, p in boards(r):
                ms = outcomes(b, tb)
                best = max(v for _, v in ms)
                assert best == sign(tb.probe_wdl(b))
                data.append(
                    {
                        "fen": b.fen(),
                        "geometry": describe(b, a, p),
                        "formula": formula(b),
                        "white_wdl": -best,
                        "replies": [
                            {"move": m.uci(), "san": b.san(m), "black_wdl": v}
                            for m, v in ms
                        ],
                    }
                )
            assert [d["white_wdl"] for d in data] == [
                int(t) for t in (r["original_wdl"] + " " + r["edited_wdl"]).split()
            ]
            assert [
                data[i]["formula"][1] - data[i + 1]["formula"][1] for i in (0, 2)
            ] == [
                -int(r[k])
                for k in ("classical_pregrain_original", "classical_pregrain_edited")
            ]
            rows.append(
                dict(
                    r,
                    success=int(r["nnue_original"]) > 0 > int(r["nnue_edited"]),
                    children=data,
                )
            )
    from siblings import trace_positions

    traces = trace_positions([d["fen"] for row in rows for d in row["children"]], root)
    for i, row in enumerate(rows):
        scores = [v[1] for v in traces[4 * i : 4 * i + 4]]
        assert [s[0] for s in scores] == [d["formula"][1] for d in row["children"]]
        assert scores[1][1] - scores[0][1] == int(row["nnue_original"])
        assert scores[3][1] - scores[2][1] == int(row["nnue_edited"])
    return rows


def describe_replies(row):
    from geometry import ordered

    data = row["children"]
    roles = []
    optimal = []
    all_tags = []
    for child, (board, actor, pawn) in zip(data, boards(row)):
        tagged = []
        for reply in child["replies"]:
            m = chess.Move.from_uci(reply["move"])
            tag = (
                "promotion"
                if m.promotion
                else "pawn takes rook"
                if m.from_square == pawn and m.to_square == actor
                else "king takes rook"
                if m.from_square == board.king(False) and m.to_square == actor
                else "other"
            )
            tagged.append((tag, reply["black_wdl"], reply["san"]))
        best = max(x[1] for x in tagged)
        types = {x[0] for x in tagged if x[1] == best}
        roles.append(next(iter(types)) if len(types) == 1 else "mixed")
        optimal.append(" ".join(x[2] for x in tagged if x[1] == best))
        all_tags.append(tagged)
    item = {
        "parent": row["parent"],
        "pawn_shift": row["relocation"],
        "move_A": row["move_a"],
        "move_B": row["move_b"],
        "nnue_both": int(row["success"]),
        "formula_branches": " / ".join(d["formula"][0] for d in data),
        "hce_original": row["classical_pregrain_original"],
        "hce_edited": row["classical_pregrain_edited"],
        "nnue_original": row["nnue_original"],
        "nnue_edited": row["nnue_edited"],
        "original_wdl": row["original_wdl"],
        "edited_wdl": row["edited_wdl"],
        "forced_original_B": roles[1],
        "forced_edited_A": roles[2],
        "optimal_original_B": optimal[1],
        "optimal_edited_A": optimal[2],
    }
    for cat, k in [
        ("pawn takes rook", "no_pawn_capture"),
        ("promotion", "no_promotion"),
    ]:
        values = [[-v for t, v, _ in child if t != cat] for child in all_tags]
        item[k + "_still_reverses"] = (
            int(ordered([min(lst) for lst in values])) if all(values) else None
        )
    return item


if __name__ == "__main__":
    from measure import table

    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    data = audit(args.root)
    (args.dest / "seed-replies.json").write_text(
        json.dumps(data, separators=(",", ":")) + "\n"
    )
    table(args.dest / "seed-cases.csv", [describe_replies(r) for r in data])
