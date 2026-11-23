"""Replay a later-game bishop interception and the corresponding move of the pawn."""

import gzip
import json
import sys
from pathlib import Path

import chess
import chess.syzygy
from validate import sign


def main(root, out):
    tb = chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular")
    result = []
    with gzip.open(root / "atlas/horizon/played.gz", "rt") as f:
        for _, gid, fen, played, alternative, a, b, measure in map(json.loads, f):
            if gid not in ("x413ym4w", "boieijcd", "vl90td4e"):
                continue
            rootboard = chess.Board(fen)
            record = {"game": gid, "normalized_fen": fen, "choices": []}
            for u, (native, _) in zip((played, alternative), measure):
                q = rootboard.copy()
                q.push_uci(u)
                wdl = sign(tb.probe_wdl(q))
                opts = []
                for m in list(q.legal_moves):
                    san = q.san(m)
                    q.push(m)
                    v = -sign(tb.probe_wdl(q))
                    if v == wdl:
                        replies = []
                        for z in list(q.legal_moves):
                            reply = q.san(z)
                            q.push(z)
                            w = sign(tb.probe_wdl(q))
                            q.pop()
                            if w == v:
                                replies.append(reply)
                        opts.append({"black": san, "optimal_white_replies": replies})
                    q.pop()
                record["choices"].append(
                    {
                        "white": rootboard.san(chess.Move.from_uci(u)),
                        "black_wdl": wdl,
                        "hand": native[1],
                        "neural": native[2],
                        "optimal_black_replies": opts,
                    }
                )
            result.append(record)
    assert len(result) == 3
    (out / "played-examples.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
