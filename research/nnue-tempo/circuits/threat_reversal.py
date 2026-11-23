"""Relocate one opponent attacker so the exposed destination flips for fixed moves."""

import argparse
import gzip
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np
from analyze import stack_weights
from edge_transfer import arrays, mask, switches
from siblings import trace_positions
from spectra import load, match_rate


def loose_attackers(board, destination):
    sources = []
    for capture in list(
        board.generate_legal_captures(to_mask=chess.BB_SQUARES[destination])
    ):
        board.push(capture)
        if not any(
            board.generate_legal_captures(to_mask=chess.BB_SQUARES[destination])
        ):
            sources.append(capture.from_square)
        board.pop()
    return sources


def edit(row):
    root = chess.Board(row["parent"])
    if root.ep_square is not None:
        return
    a, b = map(chess.Move.from_uci, row["moves"])
    original = chess.Board(row["children"][1])
    rng = random.Random(row["game"] + "/" + str(row["stack"]) + "/reverse")
    candidates = []
    for source in loose_attackers(original, b.to_square):
        piece = root.piece_at(source)
        if (
            piece is None
            or piece.piece_type in (chess.PAWN, chess.KING)
            or root.castling_rights & chess.BB_SQUARES[source]
        ):
            continue
        for dest in chess.scan_forward(root.attacks_mask(source) & ~root.occupied):
            if dest not in (a.to_square, b.to_square):
                candidates.append((source, dest))
    rng.shuffle(candidates)
    for source, dest in candidates:
        board = root.copy(stack=False)
        attacker = board.remove_piece_at(source)
        board.set_piece_at(dest, attacker)
        if (
            not board.is_valid()
            or board.is_check()
            or not board.is_legal(a)
            or not board.is_legal(b)
        ):
            continue
        if board.gives_check(a) or board.gives_check(b):
            continue
        board.push(a)
        if dest not in loose_attackers(board, a.to_square):
            continue
        a_fen = board.fen()
        board.pop()
        board.push(b)
        if any(
            board.generate_legal_captures(to_mask=chess.BB_SQUARES[b.to_square])
        ) or not any(board.legal_moves):
            continue
        b_fen = board.fen()
        board.pop()
        return dict(
            row,
            edit=chess.square_name(source) + chess.square_name(dest),
            attacker=attacker.symbol().upper(),
            edited_parent=board.fen(),
            edited_children=[a_fen, b_fen],
        )


def find_batch(rows):
    return [result for row in rows if (result := edit(row)) is not None]


def generate(root, workers, month):
    directory = root / "circuits"
    rows = load(directory / f"contacts-{month}.gz")
    with (
        ProcessPoolExecutor(workers) as pool,
        gzip.open(directory / f"reversals-{month}.gz", "wt") as f,
    ):
        for group in pool.map(
            find_batch, (rows[i : i + 256] for i in range(0, len(rows), 256))
        ):
            if not group:
                continue
            traces = trace_positions(
                [fen for r in group for fen in r["edited_children"]], root
            )
            for i, r in enumerate(group):
                r["edited_units"], r["edited_sf141"] = map(
                    list, zip(*traces[2 * i : 2 * i + 2])
                )
                print(json.dumps(r, separators=(",", ":")), file=f)


def predictions(rows, weights, ranks, controls):
    records = arrays(rows, weights)
    result = {
        "classical": records["hce"],
        "native": records["native"],
        "deep": records["actual"],
        "psqt": records["psqt"],
    }
    for name, selected in [("jan64", mask(ranks, 64)), *controls]:
        _, w, _, out = weights
        step = records["dh"] @ (w * selected).T
        lo, hi = records["pre"].transpose(1, 0, 2)
        h1, h2 = records["post"].transpose(1, 0, 2)
        effect = (
            (
                (((lo + step) >> 6).clip(0, 127) - h1)
                + (h2 - ((hi - step) >> 6).clip(0, 127))
            )
            @ out
            / 2
        )
        result[name] = records["psqt"] + effect
    return result


def statistics(pairs):
    original, edited = pairs
    both = (original > 0) & (edited < 0)
    return {
        "original": match_rate(original),
        "edited": match_rate(-edited),
        "both": float(both.mean()),
        "positive_interaction": match_rate(original - edited),
    }


def interval(d, groups):
    _, ix = np.unique(groups, return_inverse=True)
    g = ix.max() + 1
    v = np.bincount(ix, weights=d - d.mean()) / len(d)
    err = 1.96 * np.sqrt(v @ v * g / (g - 1)) if g > 1 else np.nan
    return (
        [d.mean(), d.mean() - err, d.mean() + err] if g > 1 else [d.mean(), None, None]
    )


def summarize(root, month):
    directory = root / "circuits"
    orders = json.loads((directory / "edge-transfer.json").read_text())
    rows = load(directory / f"reversals-{month}.gz")
    pooled = defaultdict(list)
    stages = {}
    by_stage = {}
    identifiers = []
    for stage in range(8):
        data = [r for r in rows if r["stack"] == stage]
        if not data:
            continue
        edited = [
            dict(r, units=r["edited_units"], sf141=r["edited_sf141"]) for r in data
        ]
        rank = np.array(orders[str(stage)]["orders"]["contacts"])
        w = stack_weights(root, stage)
        rng = np.random.default_rng(1729 + stage)
        controls = [(f"degree{i}", switches(mask(rank, 64), rng)) for i in range(12)]
        lo, hi = [predictions(d, w, rank, controls) for d in (data, edited)]
        pairs = {k: np.array([lo[k], hi[k]]) for k in lo}
        game = np.array([r["game"] for r in data])
        identifiers.extend(
            tuple(r["parent"].split()[:4] + r["moves"] + [r["edit"]]) for r in data
        )
        by_stage[stage] = {
            "n": len(data),
            "metrics": {k: statistics(v) for k, v in pairs.items()},
        }
        for k, v in pairs.items():
            pooled[k].append(v)
        stages[stage] = game
    pairs = {k: np.concatenate(v, axis=1) for k, v in pooled.items()}
    games = np.concatenate(list(stages.values()))
    both = lambda v: ((v[0] > 0) & (v[1] < 0)).astype(float)
    result = {
        "n": len(games),
        "games": len(set(games)),
        "by_stage": by_stage,
        "metrics": {k: statistics(v) for k, v in pairs.items()},
        "both_net_minus_hce": interval(
            both(pairs["native"]) - both(pairs["classical"]), games
        ),
    }
    first = np.array(
        list({k: i for i, k in reversed(list(enumerate(identifiers)))}.values())
    )
    result["distinct_comparisons"] = len(first)
    result["distinct_metrics"] = {k: statistics(v[:, first]) for k, v in pairs.items()}
    mixed = pairs["psqt"][0] - pairs["psqt"][1]
    assert np.max(abs(mixed)) <= 2
    result["psqt_mixed_histogram"] = {
        int(x): int(n) for x, n in zip(*np.unique(mixed, return_counts=True))
    }
    (directory / f"threat-reversal-{month}.json").write_text(
        json.dumps(result, allow_nan=False)
    )
    print(result["n"], result["games"], result["both_net_minus_hce"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--month", default="2013-02")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    if not args.summarize_only:
        generate(args.root, args.workers, args.month)
    summarize(args.root, args.month)
