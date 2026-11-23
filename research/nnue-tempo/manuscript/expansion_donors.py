"""Check both frozen SF14.1 promotion donor directions on the new samples."""

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
from effects import batches, delta, donor_margins
from expansion import bootstrap
from network import Network, credit
from promotion_geometry import relation
from summarize import write


def main(root, out):
    work = root / "manuscript/expansion"
    scores = np.load(work / "scores.npz")["scores"]
    with (out / "promotion.csv").open() as stream:
        channels = {
            (ord(row["file"]) - 97, row["intervention"]): list(
                map(int, row["ft"].split())
            )
            for row in csv.DictReader(stream)
            if row["intervention"] in ("first_four", "activity_four")
        }
    data = defaultdict(list)
    with gzip.open(work / "pairs.jsonl.gz", "rt") as stream:
        for i, row in enumerate(map(json.loads, stream)):
            panel, _, a, group, fen, moves, wdl, features = row
            if panel not in (
                "4p_enriched",
                "4p_random",
                "games_4p",
                "games_4p_exposed",
            ):
                continue
            wdl = np.sign(wdl)
            if wdl[0] == wdl[1]:
                continue
            if wdl[0] < wdl[1]:
                moves, features = moves[::-1], features[::-1]
                native = scores[i, ::-1, 3]
            else:
                native = scores[i, :, 3]
            q = chess.Board(fen)
            y = next(iter(q.pieces(chess.PAWN, chess.BLACK)))
            if chess.square_rank(y) != 1:
                continue
            w, k = q.king(chess.WHITE), q.king(chess.BLACK)
            h, l = [chess.Move.from_uci(m).to_square for m in moves]
            file = (y ^ (7 if w % 8 < 4 else 0)) % 8
            exposed = features[0][1] > features[1][1]
            clear_to_off = (
                relation(a, h, y - 8, w, k) == 4 and relation(a, l, y - 8, w, k) == 2
            )
            data[panel, a, file].append(
                (group, [w, k, y, h, l], native, exposed, clear_to_off)
            )

    net = Network(root)
    by_panel = defaultdict(list)
    for (panel, a, file), rows in data.items():
        for chunk in batches(rows):
            groups, coords, exact, exposed, clear_to_off = zip(*chunk)
            raw, ps, _ = net.encode(np.array(coords), a, chess.PAWN)
            trace = net.forward(raw, ps)
            assert np.array_equal(trace["score"], exact), (panel, a, file)
            x = raw.clip(0, 127)
            native = credit(delta(trace["score"]))
            changes = []
            for kind in ("first_four", "activity_four"):
                left, right = donor_margins(net, x, trace, channels[file, kind])
                changes.extend((credit(left), credit(right)))
            by_panel[panel].append(
                np.column_stack(
                    [
                        groups,
                        np.array(exposed, int),
                        np.array(clear_to_off, int),
                        native,
                        *changes,
                    ]
                )
            )

    summaries = []
    for panel, parts in sorted(by_panel.items()):
        data = np.concatenate(parts)
        exposed = data[:, 1].astype(float) == 1
        clear = data[:, 2].astype(float) == 1
        for stratum, mask in [
            ("all", np.ones(len(data), bool)),
            ("clear_off", exposed & clear),
            ("other_exposed", exposed & ~clear),
            ("same_option", ~exposed),
        ]:
            x = data[mask]
            if not len(x):
                continue
            for rule, col in (("target", 4), ("activity", 6)):
                group, native, left, right = x[:, [0, 3, col, col + 1]].T
                scores = np.asarray([native, left, right], float)
                native, left, right = scores
                effect = (left + right) / 2 - native
                lo, hi = bootstrap(group, effect)
                summaries.append(
                    {
                        "panel": panel,
                        "stratum": stratum,
                        "intervention": rule,
                        "n": len(x),
                        "groups": len(set(group)),
                        "native": native.mean(),
                        "replace_higher": left.mean(),
                        "replace_lower": right.mean(),
                        "gain": effect.mean(),
                        "lower": lo,
                        "upper": hi,
                    }
                )
    write(out / "expansion-donors.csv", summaries)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
