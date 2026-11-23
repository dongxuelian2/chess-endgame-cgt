"""Compare instrumented scores against untouched UCI eval output, in pawn units."""

import argparse
import gzip
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from collect import EXTENDED, MATERIAL
from modern import readout


def verify(root, tag):
    commands, expected = [], []
    for m in MATERIAL:
        with gzip.open(root / f"{m}.jsonl.gz") as f:
            for i, row in enumerate(map(json.loads, f)):
                if i % 997:
                    continue
                for t, turn in enumerate("wb"):
                    commands += [f"position fen {row['board']} {turn} - - 0 1", "eval"]
                    expected.append([(1 - 2 * t) * v / 208 for v in row[tag][t]])
    run = subprocess.run(
        [root / f"clean-{tag}/src/stockfish"],
        input="\n".join(commands + ["quit"]),
        text=True,
        capture_output=True,
        check=True,
    )

    def scores(name):
        return [
            float(x)
            for x in re.findall(
                r"^" + name + r" evaluation:?\s+([+-]?[\d.]+) \(white side\)",
                run.stdout,
                flags=re.MULTILINE,
            )
        ]

    observed = list(zip(scores("Classical"), scores("NNUE"), strict=True))
    assert len(observed) == len(expected)
    error = max(
        abs(a - b) for row, want in zip(observed, expected) for a, b in zip(row, want)
    )
    assert error <= 0.00500001, error
    print(tag, len(expected), "positions, maximum error in displayed pawn units", error)


def verify18(root):
    rows, commands, scales = [], [], []
    for m in MATERIAL + EXTENDED:
        with gzip.open(root / f"{m}.jsonl.gz") as f:
            for i, row in enumerate(map(json.loads, f)):
                if i % 1559:
                    continue
                rows.append(row)
                for t in "wb":
                    commands += [f"position fen {row['board']} {t} - - 0 1", "eval"]
                material = sum(
                    m.count(k) * weight for k, weight in zip("PNBRQ", (1, 3, 3, 5, 9))
                )
                x = max(17, material) / 58
                scales.append(
                    ((-72.32565836 * x + 185.93832038) * x - 144.58862193) * x
                    + 416.44950446
                )
    raw = readout(rows, root, False)[:, :, :2].sum(2)
    want = np.rint(raw * np.array([100, -100]) / np.array(scales)[:, None]).astype(int)
    res = subprocess.run(
        [root / "clean-sf18/src/stockfish"],
        input="\n".join(commands + ["quit"]),
        text=True,
        capture_output=True,
        check=True,
    )
    actual = [
        round(100 * float(x))
        for x in re.findall(
            r"^NNUE evaluation\s+([+-]?[\d.]+) \(white side\)",
            res.stdout,
            flags=re.MULTILINE,
        )
    ]
    assert np.array_equal(want.ravel(), actual)
    print("sf18", len(actual), "positions, all displayed scores match")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    root = p.parse_args().scratch
    for tag in ("sf12", "sf141"):
        verify(root, tag)
    verify18(root)
