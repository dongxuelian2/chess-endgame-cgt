"""Compare static pair ordering with native Stockfish search under fixed depths."""

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
from expansion import RELEASES, counts
from network import credit
from summarize import write

DEPTHS = (1, 2, 4, 8)
BINARIES = ("clean-sf12", "clean-sf141", "clean-sf18")


def run(job):
    root, release, rows = job
    engine_path = root / release / "src/stockfish"
    outcomes = []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        engine.configure({"Threads": 1, "Hash": 8, "SyzygyProbeLimit": 0})
        for i, row in rows:
            board = chess.Board(row[4])
            assert board.turn == chess.WHITE
            board.halfmove_clock = 0
            first, second = [chess.Move.from_uci(m) for m in row[5]]
            assert board.is_legal(first) and board.is_legal(second)
            scores = []
            for depth in DEPTHS:
                results = engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=2,
                    game=object(),
                    root_moves=[first, second],
                )
                move_scores = {
                    r["pv"][0]: r["score"].white().score(mate_score=100_000)
                    for r in results
                }
                assert set(move_scores) == {first, second}, results
                assert all(r.get("tbhits", 0) == 0 for r in results)
                assert all(r["depth"] == depth for r in results), results
                # Black's ordering is the negative of the root White scores.
                scores.append(move_scores[second] - move_scores[first])
            outcomes.append((i, scores))
    return outcomes


def choose(records, scores):
    rng = np.random.default_rng(48829)
    panels = np.array([r[0] for r in records])
    labels = np.sign([r[6] for r in records])
    direction = np.sign(labels[:, 0] - labels[:, 1])
    classical_tie = scores[:, 0, 2] == scores[:, 1, 2]
    survive = np.array([r[7] for r in records])[:, :, 1]
    option = (survive[:, 0] - survive[:, 1]) * direction
    samples = {}
    for panel in (
        "4p_random",
        "4p_enriched",
        "5p_random",
        "5p_enriched",
        "6p_random",
        "6p_enriched",
        "games_4p",
        "games_5p",
        "games_4p_exposed",
        "games_5p_exposed",
    ):
        for name, mask in [
            ("classical_tie", classical_tie & (option == 0)),
            ("favorable", option > 0),
            ("adverse", option < 0),
        ]:
            ix = np.flatnonzero((panels == panel) & (direction != 0) & mask)
            if len(ix):
                samples[panel, name] = rng.permutation(ix)[:384]
    return direction, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / "manuscript/expansion"
    with gzip.open(work / "pairs.jsonl.gz", "rt") as stream:
        records = list(map(json.loads, stream))
    scores = np.load(work / "scores.npz")["scores"]
    direction, samples = choose(records, scores)
    ix = sorted(set(np.concatenate(list(samples.values()))))
    print("Selected pairs", len(ix), "strata", len(samples), flush=True)
    jobs = [
        (root, binary, batch)
        for binary in BINARIES
        for batch in batched(((i, records[i]) for i in ix), 128)
    ]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("fork")) as pool:
        result = list(pool.map(run, jobs))
    search = {
        release: np.zeros((len(records), len(DEPTHS)), int) for release in BINARIES
    }
    for (_, release, _), batch in zip(jobs, result):
        for i, values in batch:
            search[release][i] = values
    baseline = (scores[:, 0] - scores[:, 1]) * direction[:, None]
    groups = np.array([r[3] for r in records])
    rows = []
    for (panel, stratum), ids in samples.items():
        for release, binary, col in zip(RELEASES, BINARIES, (1, 3, 4)):
            x = credit(baseline[ids, col])
            for j, depth in enumerate(DEPTHS):
                y = credit(search[binary][ids, j] * direction[ids])
                result = counts(groups[ids], x, y, np.ones(len(ids), bool))
                rows.append(
                    dict(
                        panel=panel,
                        stratum=stratum,
                        release=release,
                        depth=depth,
                        **result,
                    )
                )
    write(args.results / "expansion-search.csv", rows)


if __name__ == "__main__":
    main()
