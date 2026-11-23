"""Paired equal-node choice test on roots admitting both optimal and inferior moves."""

import argparse
import gzip
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import chess
import chess.engine
import chess.syzygy
from collect import MATERIAL

BUDGETS = (64, 512, 4096)


def effective_wdl(b, tb):
    if b.is_checkmate():
        return -1
    w = tb.probe_wdl(b)
    if abs(w) != 2:
        return 0
    if b.halfmove_clock:
        dtz = abs(tb.probe_dtz(b))
        assert dtz + b.halfmove_clock < 100, (b.fen(), dtz)
    return w // 2


def run(material, root, count):
    rng = random.Random("play:" + material)
    with gzip.open(root / (material + ".jsonl.gz")) as f:
        rows = [r for r in map(json.loads, f) if r["fold"]]
    rng.shuffle(rows)
    engines = [
        chess.engine.SimpleEngine.popen_uci(root / "clean-sf141/src/stockfish")
        for _ in range(2)
    ]
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        try:
            for i, engine in enumerate(engines):
                engine.configure(
                    {
                        "Threads": 1,
                        "Hash": 4,
                        "Use NNUE": bool(i),
                        "SyzygyProbeLimit": 0,
                    }
                )
            checked = 0
            totals = Counter()
            with gzip.open(root / (material + ".moves.gz"), "wt") as out:
                for row in rows:
                    turn = rng.choice("wb")
                    b = chess.Board(row["board"] + " " + turn + " - - 0 1")
                    best = effective_wdl(b, tb)
                    labels = {}
                    for m in list(b.legal_moves):
                        b.push(m)
                        labels[m] = -effective_wdl(b, tb)
                        b.pop()
                    assert max(labels.values()) == best
                    checked += 1
                    if len(set(labels.values())) == 1:
                        continue
                    record = {}
                    for nodes in BUDGETS:
                        moves = []
                        for engine in engines:
                            result = engine.play(
                                b,
                                chess.engine.Limit(nodes=nodes),
                                game=object(),
                                info=chess.engine.INFO_BASIC,
                            )
                            moves.append(
                                [
                                    result.move.uci(),
                                    labels[result.move],
                                    result.info["nodes"],
                                ]
                            )
                        record[nodes] = moves
                        good = tuple(int(x[1] == best) for x in moves)
                        totals[nodes, *good] += 1
                    print(json.dumps([b.fen(), best, record, row["group"]]), file=out)
                    if sum(v for k, v in totals.items() if k[0] == BUDGETS[0]) == count:
                        break
            print(material, "screened", checked, sorted(totals.items()), flush=True)
        finally:
            for e in engines:
                e.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--count", type=int, default=1000)
    args = p.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(partial(run, root=args.scratch, count=args.count), MATERIAL))
