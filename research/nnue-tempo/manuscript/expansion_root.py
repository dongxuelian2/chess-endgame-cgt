"""Measure WDL-optimal full-root play on independent four-piece game positions."""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from multiprocessing import get_context
from pathlib import Path

import chess
import chess.engine
import numpy as np
import validate as v
from expansion import RELEASES, bootstrap, init, pair_key
from expansion_search import BINARIES, DEPTHS
from summarize import write


def wdl_after(board, move):
    board.push(move)
    try:
        return -v.sign(v.tb.probe_wdl(board))
    finally:
        board.pop()


def run(job):
    root, binary, rows = job
    init(root)
    observations = []
    with chess.engine.SimpleEngine.popen_uci(root / binary / "src/stockfish") as engine:
        engine.configure({"Threads": 1, "Hash": 8, "SyzygyProbeLimit": 0})
        for row in rows:
            group, fen, played = row
            board = chess.Board(fen)
            optimum = v.sign(v.tb.probe_wdl(board))
            human = wdl_after(board, chess.Move.from_uci(played))
            assert optimum >= human
            for depth in DEPTHS:
                result = engine.analyse(
                    board, chess.engine.Limit(depth=depth), game=object()
                )
                assert result.get("tbhits", 0) == 0 and result["depth"] == depth
                decision = wdl_after(board, result["pv"][0])
                assert optimum >= decision, (row, result)
                observations.append((group, depth, optimum, human, decision))
    return observations


def sample(work):
    rng = np.random.default_rng(61042)
    with gzip.open(work / "pairs.jsonl.gz", "rt") as stream:
        pairs = list(map(json.loads, stream))
    samples = {}
    seen = set()
    for panel in ("games_4p", "games_4p_exposed"):
        candidates = [
            [r[3], r[4], r[5][0]]
            for r in pairs
            if r[0] == panel and v.sign(r[6][0]) != v.sign(r[6][1])
        ]
        selected = []
        for i in rng.permutation(len(candidates)):
            row = candidates[i]
            _, fen, _ = row
            position = pair_key(fen, [])[0]
            if position not in seen:
                seen.add(position)
                selected.append(row)
            if len(selected) == 768:
                break
        samples[panel] = selected
    return samples


def main(root, out, workers):
    cases = sample(root / "manuscript/expansion")
    jobs = [
        (name, root, binary, rows)
        for name, candidates in cases.items()
        for binary in BINARIES
        for rows in batched(candidates, 96)
    ]
    by_panel = {(name, binary): [] for name in cases for binary in BINARIES}
    with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as pool:
        for (name, _, binary, _), result in zip(
            jobs, pool.map(run, (j[1:] for j in jobs))
        ):
            by_panel[name, binary].extend(result)
    stats = []
    for (panel, binary), result in by_panel.items():
        release = RELEASES[BINARIES.index(binary)]
        for depth in DEPTHS:
            group, _, truth, played, chosen = zip(*(r for r in result if r[1] == depth))
            truth, played, chosen = (np.array(x) for x in [truth, played, chosen])
            native = chosen == truth
            baseline = played == truth
            lo, hi = bootstrap(group, native.astype(int) - baseline)
            stats.append(
                {
                    "panel": panel,
                    "release": release,
                    "depth": depth,
                    "n": len(truth),
                    "wins": int((truth == 1).sum()),
                    "draws": int((truth == 0).sum()),
                    "human": baseline.mean(),
                    "optimal": native.mean(),
                    "lower": lo,
                    "upper": hi,
                }
            )
    write(out / "expansion-root.csv", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    main(args.root.resolve(), args.results, args.workers)
