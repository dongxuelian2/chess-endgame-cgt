"""Stream each disjoint material census to its own dense file."""

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path


def run(job):
    root, a, b, replies, remaining = job
    directory = root / "atlas"
    target = directory / f"{a}-{b}.bin"
    args = [
        directory / "engine/src/stockfish",
        str(a),
        str(b),
        root / "pychess/data/syzygy/regular",
        target.with_suffix(".part"),
        "64",
    ]
    if replies:
        dense = target
        if remaining:
            directory = directory / "remaining"
        directory.mkdir(exist_ok=True)
        target = directory / f"{a}-{b}.replies"
        args = [
            root / "atlas/engine/src/stockfish",
            str(a),
            str(b),
            root / "pychess/data/syzygy/regular",
            dense,
            target.with_suffix(".rpart"),
            "remaining" if remaining else "replies",
        ]
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    line = next(
        x
        for x in result.stdout.splitlines()
        if x.startswith("replies " if replies else "atlas ")
    )
    temp = target.with_suffix(".rpart" if replies else ".part")
    if not replies:
        assert temp.stat().st_size == 64**4 * 12
    os.replace(temp, target)
    (directory / (f"{a}-{b}.reply-count" if replies else f"{a}-{b}.count")).write_text(
        line + "\n"
    )
    print(line, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--remaining", action="store_true")
    p.add_argument("--replies", action="store_true")
    p.add_argument("--actor", type=int, choices=range(1, 6))
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    with ThreadPoolExecutor(a.workers) as ex:
        list(
            ex.map(
                run,
                [
                    (a.root.resolve(), x, y, a.replies or a.remaining, a.remaining)
                    for x, y in product(
                        [a.actor] if a.actor else range(1, 6), range(1, 6)
                    )
                ],
            )
        )
