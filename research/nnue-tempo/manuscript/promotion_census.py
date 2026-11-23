"""Enumerate adjacent capture-promotion geometries and check the local win test."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import validate as v
from summarize import write

from promotion_prediction import board_features


def census(root, out):
    v.init(root)
    report = []
    witnesses = []
    for a in range(2, 6):
        v.state(a, 1, 0, 0, 0, 0)
        tab = v.maps[a, 1]
        counters = defaultdict(lambda: np.zeros(3, dtype=int))
        example = {}
        for file in range(8):
            for x in (file - 1, file + 1):
                if x < 0 or x > 7:
                    continue
                y = 8 + file
                w, k = np.where(tab[:, :, y, x]["v"] > 0)
                for i, j in zip(w.tolist(), k.tolist()):
                    if i % 8 >= 4:
                        continue
                    g, legal, survives, _ = board_features(a, i, j, y, x)
                    assert g
                    truth = v.sign(tab[i, j, y, x]["w"])
                    for name, case in [
                        ("geometric", True),
                        ("legal", legal),
                        ("major_survives", survives),
                    ]:
                        if case:
                            counters[name][truth + 1] += 1
                            if truth <= 0 and name not in example:
                                q = v.board(a, 1, i, j, y, x)
                                assert v.sign(v.tb.probe_wdl(q)) == truth
                                example[name] = (q.fen(), truth)
        for name, counts in counters.items():
            report.append(
                {
                    "white": a,
                    "feature": name,
                    "loss": int(counts[0]),
                    "draw": int(counts[1]),
                    "win": int(counts[2]),
                }
            )
        assert counters["major_survives"][:2].sum() == 0
        for name, (fen, truth) in example.items():
            witnesses.append(
                {"white": a, "feature": name, "fen": fen, "black_wdl": truth}
            )
    write(out / "prediction-promotion-census.csv", report)
    write(out / "prediction-promotion-census-witnesses.csv", witnesses)


if __name__ == "__main__":
    census(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
