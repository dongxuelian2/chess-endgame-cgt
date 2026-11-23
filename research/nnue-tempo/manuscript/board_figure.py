"""Render the exact two chess positions in the introductory figure."""

from pathlib import Path
from xml.etree import ElementTree as ET

import chess
import chess.svg

FEN = "1Q6/8/8/K7/8/5p2/6k1/8 w - - 0 1"
SVG = "http://www.w3.org/2000/svg"
X_LINK = "http://www.w3.org/1999/xlink"


def mark(square):
    return 17 + chess.square_file(square) * 45, 17 + (
        7 - chess.square_rank(square)
    ) * 45


def render(path, position, pinned):
    colors = {
        "square light": "#f2ede3",
        "square dark": "#adbec5",
        "margin": "#ffffff",
        "outer border": "#ffffff00",
        "inner border": "#3a4c52",
        "coord": "#354950",
    }
    fill = {
        chess.B7 if pinned else chess.B6: "#e5bc6f55",
        chess.F3: "#d5a65165",
        chess.F2: "#276c7c35" if not pinned else "#c56d3d24",
    }
    board = ET.fromstring(
        chess.svg.board(position, colors=colors, fill=fill, borders=True, size=394)
    )
    # Label the moves with restrained rings and a dashed line that stays readable at column width.
    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", X_LINK)
    layer = ET.SubElement(
        board, f"{{{SVG}}}g", {"stroke-linecap": "round", "stroke-linejoin": "round"}
    )
    hx, hy = mark(chess.F2)
    if pinned:
        for flip in [-1, 1]:
            ET.SubElement(
                layer,
                f"{{{SVG}}}line",
                {
                    "x1": str(hx + 15),
                    "y1": str(hy + 22.5 - 7 * flip),
                    "x2": str(hx + 30),
                    "y2": str(hy + 22.5 + 7 * flip),
                    "stroke": "#914a2b",
                    "stroke-width": "3.5",
                },
            )
        for square in (chess.C6, chess.D5, chess.E4):
            x, y = mark(square)
            ET.SubElement(
                layer,
                f"{{{SVG}}}circle",
                {
                    "cx": str(x + 22.5),
                    "cy": str(y + 22.5),
                    "r": "4.5",
                    "fill": "#914a2b",
                },
            )
    else:
        ET.SubElement(
            layer,
            f"{{{SVG}}}path",
            {
                "d": f"M {hx + 22.5} {hy + 4} v 24 m -7 -7 l 7 7 l 7 -7",
                "stroke-width": "3.5",
                "stroke": "#225868",
                "fill": "none",
            },
        )
    ET.SubElement(
        board, f"{{{SVG}}}metadata"
    ).text = "Piece artwork by Colin M. L. Burnett, distributed with python-chess. See the companion README for attribution."
    ET.indent(board)
    path.write_text(ET.tostring(board, encoding="unicode") + "\n")


def main():
    out = Path(__file__).resolve().parent
    for dest in ("b6", "b7"):
        position = chess.Board(FEN)
        position.push_san("Q" + dest)
        pinned = dest == "b7"
        push = chess.Move(chess.F3, chess.F2)
        assert (push in position.legal_moves) == (not pinned)
        assert position.is_pinned(chess.BLACK, chess.F3) == pinned
        render(out / ("board-" + dest + ".svg"), position, pinned)


if __name__ == "__main__":
    main()
