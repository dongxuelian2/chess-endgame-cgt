"""Draw the actual parents for a diagonal swap and a promotion interception."""

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import chess
import chess.svg

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)


def element(parent, tag, **attributes):
    return ET.SubElement(
        parent, "{" + NS + "}" + tag, {k: str(v) for k, v in attributes.items()}
    )


def label(parent, x, y, value, size=14):
    element(
        parent, "text", x=x, y=y, fill="#182b38", style=f"font: {size}px sans-serif"
    ).text = value


def main(dest):
    rows = json.loads((dest / "seed-replies.json").read_text())
    out = ET.Element(
        "{" + NS + "}svg", {"width": "780", "height": "846", "viewBox": "0 0 780 846"}
    )
    element(out, "rect", x=0, y=0, width=780, height=846, fill="white")
    for i, (a, b, title) in enumerate(
        [
            (
                "c1c2",
                "c1c3",
                "1. The pawn switches which rook destination it can capture.",
            ),
            ("a6a5", "a6a1", "2. The rook can intercept a newly promoted queen."),
        ]
    ):
        row = next(
            r for r in rows if r["move_a"] == a and r["move_b"] == b and r["success"]
        )
        parent = chess.Board(row["parent"])
        shift = chess.Move.from_uci(row["relocation"])
        y = 420 * i
        label(out, 18, y + 26, title, 19)
        for j in (0, 1):
            board = parent.copy()
            if j:
                board.set_piece_at(
                    shift.to_square, board.remove_piece_at(shift.from_square)
                )
            moves = list(map(chess.Move.from_uci, (a, b)))
            good, bad = (moves[j], moves[1 - j])
            arrows = [
                chess.svg.Arrow(bad.from_square, bad.to_square, color="#9b6868bb"),
                chess.svg.Arrow(good.from_square, good.to_square, color="#248369dd"),
            ]
            if i == 0:
                arrows.append(
                    chess.svg.Arrow(
                        shift.to_square if j else shift.from_square,
                        bad.to_square,
                        color="#ab471fff",
                    )
                )
            diagram = ET.fromstring(
                chess.svg.board(board=board, arrows=arrows, size=320)
            )
            # Make internal SVG references unique for each embedded board.
            prefix = f"panel{i}{j}-"
            for n in diagram.iter():
                if "id" in n.attrib:
                    n.set("id", prefix + n.get("id"))
                for key, val in list(n.attrib.items()):
                    if key.endswith("href") and val.startswith("#"):
                        n.set(key, "#" + prefix + val[1:])
            diagram.set("x", str(30 + 390 * j))
            diagram.set("y", str(y + 60))
            out.append(diagram)
            label(
                out,
                30 + 390 * j,
                y + 49,
                "Pawn on "
                + chess.square_name(shift.to_square if j else shift.from_square),
            )
            label(
                out,
                30 + 390 * j,
                y + 402,
                board.san(good) + " wins; " + board.san(bad) + " loses.",
            )
    ET.ElementTree(out).write(dest / "two-positions.svg", encoding="unicode")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dest", type=Path)
    main(p.parse_args().dest)
