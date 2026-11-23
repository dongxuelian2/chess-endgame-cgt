"""Paired king-placement cluster bootstrap for two descriptive policy contrasts."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
from collect import king_group
from measure import table


def main(dest):
    with (dest / "interception-roots.csv").open() as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for r in rows:
        b = chess.Board(r["fen"])
        grouped[king_group(b.king(True), b.king(False))].append(r)
    keys = sorted(grouped)
    rng = np.random.default_rng(621104)
    draws = rng.integers(len(keys), size=(10000, len(keys)))
    n = np.array([len(grouped[k]) for k in keys])
    out = []
    for a, b in [("interception", "reflected_file"), ("full", "interception")]:
        total = np.array(
            [sum(float(r[a]) - float(r[b]) for r in grouped[k]) for k in keys]
        )
        boot = total[draws].sum(1) / n[draws].sum(1)
        out.append(
            {
                "first": a,
                "second": b,
                "roots": len(rows),
                "king_groups": len(keys),
                "difference": total.sum() / n.sum(),
                "q025": np.quantile(boot, 0.025),
                "q975": np.quantile(boot, 0.975),
            }
        )
    table(dest / "interception-intervals.csv", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dest", type=Path)
    main(p.parse_args().dest)
