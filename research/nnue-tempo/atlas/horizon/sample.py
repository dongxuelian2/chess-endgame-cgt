"""Construct quiet-pair panels and evaluate every distinct requested child."""

import argparse
import csv
import gzip
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import numpy as np
from summarize import MATS
from validate import sign


def comparisons(root, all_games=False):
    for a, b in [] if all_games else MATS:
        path = root / "atlas/remaining/metrics" / f"{a}-{b}.npz"
        if not path.exists():
            continue
        for r in np.load(path)["panel"]:
            i, _, h, l, wh, wl, nn, deep, ps, x, z = map(int, r)
            if (h | l) & 51:
                continue
            yield (
                "enumerated",
                "",
                a,
                b,
                i >> 12,
                (i >> 6) & 63,
                i & 63,
                x,
                z,
                h,
                l,
                sign(wh),
                sign(wl),
                nn,
                deep,
                ps,
            )
    for name, path in [
        ("previous", root / "atlas/game-results.gz"),
        ("fresh", root / "atlas/horizon/played.gz"),
    ]:
        with gzip.open(path, "rt") as f:
            for _, gid, fen, u, v, a, b, ((hi, hm), (lo, lm)) in map(json.loads, f):
                if (
                    hi[1] != lo[1]
                    or (not all_games and sign(hi[-1]) == sign(lo[-1]))
                    or (hm[0] | lm[0]) & 51
                ):
                    continue
                if hi[-1] < lo[-1]:
                    hi, lo, hm, lm, u, v = lo, hi, lm, hm, v, u
                q = chess.Board(fen)
                y = next(iter(chess.SquareSet(q.occupied_co[False] & ~q.kings)))
                yield (
                    name,
                    gid,
                    a,
                    b,
                    q.king(True),
                    q.king(False),
                    y,
                    chess.parse_square(u[2:4]),
                    chess.parse_square(v[2:4]),
                    hm[0],
                    lm[0],
                    sign(hi[-1]),
                    sign(lo[-1]),
                    hi[2] - lo[2],
                    hi[4] - lo[4],
                    hi[3] - lo[3],
                )


def run(root, depth, only_games, workers, all_games):
    directory = root / "atlas/horizon"
    rows = list(comparisons(root, all_games))
    if only_games:
        rows = [r for r in rows if r[0] != "enumerated"]
    prefix = directory / (
        ("unselected" if all_games else "games" if only_games else "all") + str(depth)
    )
    prefix.mkdir(exist_ok=True)
    with (prefix / "panel.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "panel",
                "gid",
                "a",
                "b",
                "wk",
                "bk",
                "y",
                "x",
                "z",
                "h",
                "l",
                "wh",
                "wl",
                "nn",
                "deep",
                "ps",
            ]
        )
        writer.writerows(rows)

    def material(ab):
        a, b = ab
        states = sorted(
            {(r[4], r[5], r[6], s) for r in rows if r[2:4] == ab for s in r[7:9]}
        )
        if not states:
            return
        src = prefix / f"{a}-{b}.in"
        dst = src.with_suffix(".out")
        with src.open("w") as f:
            for state in states:
                print(*state, file=f)
        result = subprocess.run(
            [
                directory / "engine/src/stockfish",
                "frontier",
                root / "pychess/data/syzygy/regular",
                str(depth),
                str(a),
                str(b),
                src,
                dst,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout.splitlines()[-1], flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(material, MATS))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--games", action="store_true")
    p.add_argument("--all-games", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    run(args.root.resolve(), args.depth, args.games, args.workers, args.all_games)
