"""Replay fixed promotion rules across networks, materials and a fresh game archive."""

import argparse
import gzip
import io
import json
import os
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from itertools import batched, product
from multiprocessing import get_context
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import validate as v
from atlas.panels import raw
from cohorts import orbit
from network import credit
from promotion_prediction import fit, loaded
from summarize import write

RELEASES = ("SF12", "SF14.1", "SF18")
RULES = ("legal", "surviving", "reply_filtered")
DEFENDERS = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def init(root):
    v.init(root)
    v.tb.add_directory(root / "tb5")
    v.tb.add_directory(root / "manuscript/expansion/tables")


def capture_flags(board):
    legal = surviving = reply_filtered = False
    pawns = board.pieces_mask(chess.PAWN, chess.BLACK) & chess.BB_RANK_2
    for move in tuple(board.generate_legal_captures(from_mask=pawns)):
        legal |= move.promotion is not None
        if move.promotion not in (chess.QUEEN, chess.ROOK):
            continue
        board.push(move)
        try:
            if board.is_stalemate() or any(
                board.generate_legal_captures(to_mask=chess.BB_SQUARES[move.to_square])
            ):
                continue
            surviving = True
            white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
            reply_filtered |= not any(
                reply.promotion
                for reply in board.generate_legal_moves(from_mask=white_pawns)
            )
        finally:
            board.pop()
    return [int(legal), int(surviving), int(reply_filtered)]


def alternatives(parent, source):
    choices = []
    for move in parent.generate_legal_moves(from_mask=chess.BB_SQUARES[source]):
        if parent.is_capture(move) or move.promotion:
            continue
        child = parent.copy(stack=False)
        child.push(move)
        if (
            child.is_check()
            or child.has_legal_en_passant()
            or not any(child.legal_moves)
        ):
            continue
        choices.append((move.uci(), child))
    return choices


def make_pair(panel, cell, a, parent, choices, group=None, features=None):
    moves, kids = zip(*choices)
    if group is None:
        group = orbit(parent.king(chess.WHITE), parent.king(chess.BLACK))
    if features is None:
        features = [capture_flags(child) for child in kids]
    wdl = [v.tb.probe_wdl(child) for child in kids]
    return [panel, cell, a, group, parent.fen(), list(moves), wdl, features]


def synthetic(params):
    a, count, race, enriched = params
    panel = f"{count}p_{'enriched' if enriched else 'random'}"
    cell = "seventh" if race else "ordinary"
    rng = random.Random(f"promotion-expansion-v1-{a}-{count}-{race}-{enriched}")
    visited = set()
    rows = []
    n = 3000 if enriched else 4000
    while len(rows) < n:
        w, k, s = rng.sample(range(64), 3)
        if chess.square_distance(w, k) < 2:
            continue
        y = rng.randrange(8, 16)
        white_pawns = rng.sample(range(8, 56 if race else 48), count - 4)
        if race and not any(p >= 48 for p in white_pawns):
            continue
        if len({w, k, y, s, *white_pawns}) != count:
            continue
        parent = v.board(a, chess.PAWN, w, k, y, s)
        parent.turn = chess.WHITE
        for pawn in white_pawns:
            parent.set_piece_at(pawn, chess.Piece(chess.PAWN, chess.WHITE))
        if not parent.is_valid() or parent.is_check():
            continue
        choices = alternatives(parent, s)
        if len(choices) < 2:
            continue
        if enriched:
            features = [capture_flags(board) for _, board in choices]
            eligible = [i for i, flags in enumerate(features) if flags[1]]
            others = [i for i, flags in enumerate(features) if not flags[1]]
            if not eligible or not others:
                continue
            ix = [rng.choice(eligible), rng.choice(others)]
        else:
            ix = rng.sample(range(len(choices)), 2)
            features = [None] * len(choices)
        pair = [choices[i] for i in ix]
        key = (parent.board_fen(), *sorted(move for move, _ in pair))
        if key in visited:
            continue
        visited.add(key)
        rows.append(
            make_pair(
                panel,
                cell,
                a,
                parent,
                pair,
                features=[features[i] for i in ix] if enriched else None,
            )
        )
    return rows


def game_batches(path):
    gid = None
    with subprocess.Popen(
        ["zstd", "-dc", path], stdout=subprocess.PIPE, text=True
    ) as run:
        batch = []
        games = screened = 0
        for line in run.stdout:
            if line.startswith('[Site "'):
                gid = line.split('"')[1].rsplit("/", 1)[-1]
            if not line.startswith("1."):
                continue
            games += 1
            if line.count("x") < 27:
                continue
            assert gid
            screened += 1
            batch.append((path.stem, gid, line))
            if len(batch) == 80:
                yield batch
                batch = []
        if batch:
            yield batch
        assert run.wait() == 0
    print(path.stem, games, screened, flush=True)


def played(batch):
    rows = []
    for month, gid, line in batch:
        game = chess.pgn.read_game(io.StringIO(line))
        if game.errors:
            continue
        rng = random.Random(f"promotion-games-v1-{gid}")
        board = game.board()
        selected = {}
        totals = {}
        for actual in game.mainline_moves():
            count = board.occupied.bit_count()
            if count < 4:
                break
            defender = board.turn
            source = actual.from_square
            a = board.piece_type_at(source)
            enemy = board.occupied_co[not defender]
            if (
                count in (4, 5)
                and a in DEFENDERS
                and enemy.bit_count() == 2
                and board.pieces(chess.PAWN, not defender)
                and len(board.pieces(chess.PAWN, defender)) == count - 4
                and not (
                    board.castling_rights
                    or board.has_legal_en_passant()
                    or board.is_check()
                    or board.is_capture(actual)
                    or board.gives_check(actual)
                )
            ):
                parent = board.copy(stack=False) if defender else board.mirror()
                if not defender:
                    source ^= 56
                observed = (
                    actual if defender else chess.Move(source, actual.to_square ^ 56)
                ).uci()
                parent.halfmove_clock = 0
                parent.fullmove_number = 1
                choices = alternatives(parent, source)
                observed_child = next((q for m, q in choices if m == observed), None)
                if observed_child is not None and len(choices) > 1:
                    pawn = next(iter(parent.pieces(chess.PAWN, chess.BLACK)))
                    near = pawn in chess.SquareSet(chess.BB_RANK_2)
                    panels = [
                        (
                            f"games_{count}p",
                            [i for i, (m, _) in enumerate(choices) if m != observed],
                        )
                    ]
                    if near:
                        actual_flag = capture_flags(observed_child)[1]
                        differing = [
                            i
                            for i, (m, child) in enumerate(choices)
                            if m != observed and capture_flags(child)[1] != actual_flag
                        ]
                        if differing:
                            panels.append((f"games_{count}p_exposed", differing))
                    for panel, indices in panels:
                        totals[panel] = totals.get(panel, 0) + 1
                        if rng.randrange(totals[panel]) == 0:
                            pair = [
                                (observed, observed_child),
                                choices[rng.choice(indices)],
                            ]
                            selected[panel] = (
                                panel,
                                "second_rank" if near else "other_ranks",
                                a,
                                parent,
                                pair,
                                gid,
                            )
            board.push(actual)
        rows.extend(make_pair(*args) for args in selected.values())
    return rows


def pair_key(fen, moves):
    parent = chess.Board(fen)
    reflected = parent.transform(chess.flip_horizontal)
    mapped = [
        chess.Move(chess.parse_square(m[:2]) ^ 7, chess.parse_square(m[2:]) ^ 7).uci()
        for m in moves
    ]
    return min(
        (parent.board_fen(), tuple(sorted(moves))),
        (reflected.board_fen(), tuple(sorted(mapped))),
    )


def original_keys(root, discovery_rows):
    seen = {pair_key(r[4], r[5]) for r in discovery_rows}
    for a in DEFENDERS:
        panel = np.load(root / f"manuscript/validation-{a}-1.npz")
        for w, k, y, h, l, s in panel["rows"].tolist():
            parent = v.board(a, chess.PAWN, w, k, y, s)
            parent.turn = chess.WHITE
            moves = [chess.Move(s, dest).uci() for dest in (h, l)]
            seen.add(pair_key(parent.fen(), moves))
    for archive in ("atlas/played.gz", "atlas/horizon/played.gz"):
        with gzip.open(root / archive, "rt") as stream:
            for _, _, fen, first, second, *_ in map(json.loads, stream):
                seen.add(pair_key(fen, [first, second]))
    return seen


def unique_pairs(rows, original):
    seen = set()
    for row in rows:
        panel, _, _, _, fen, moves, *_ = row
        pair = pair_key(fen, moves)
        if pair in original or (panel, pair) in seen:
            continue
        seen.add((panel, pair))
        yield row


def save_candidates(root, work, kind, workers, pieces, archive):
    name = f"synthetic-{pieces}" if kind == "synthetic" else archive
    path = work / f"{name}.jsonl.gz"
    partial = path.with_suffix(".gz.part")
    with (
        gzip.open(partial, "wt") as stream,
        ProcessPoolExecutor(
            workers, mp_context=get_context("fork"), initializer=init, initargs=(root,)
        ) as pool,
    ):
        if kind == "synthetic":
            jobs = [
                (a, count, race, enriched)
                for a, count, race, enriched in product(
                    DEFENDERS, (pieces,), (False, True), (False, True)
                )
                if count > 4 or not race
            ]
            for job, rows in zip(jobs, pool.map(synthetic, jobs)):
                print(job, len(rows), flush=True)
                for row in rows:
                    print(json.dumps(row), file=stream)
        else:
            archive_path = work / "archives" / f"{archive}.zst"
            for chunk in batched(game_batches(archive_path), workers * 2):
                for rows in pool.map(played, chunk):
                    for row in rows:
                        print(json.dumps(row), file=stream)
    partial.replace(path)


def child_fens(row):
    board = chess.Board(row[4])
    result = []
    for uci in row[5]:
        move = board.push_uci(uci)
        assert move.promotion is None
        result.append(board.fen())
        board.pop()
    return result


def release18(root, fens):
    env = os.environ.copy()
    env["ATLAS_RELEASE_ONLY"] = "1"
    text = "".join(f"position fen {fen}\neval\n" for fen in fens) + "quit\n"
    result = subprocess.run(
        [root / "atlas/engine18/src/stockfish"],
        input=text,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    values = [
        int(s.split()[1])
        for s in result.stdout.splitlines()
        if s.startswith("release18 ")
    ]
    assert len(values) == len(fens), result.stdout[-400:]
    return np.array(values)


def score_position_batch(root, fens):
    s12 = raw(root / "sf12/src/stockfish", fens, "scores")
    s14 = raw(root / "manuscript/expansion/classical/src/stockfish", fens, "scores")
    s18 = raw(root / "sf18/src/stockfish", fens, "raw18")[:, :2].sum(axis=1)
    off18 = raw(root / "sf18/src/stockfish", fens, "raw18", knockout=True)[:, :2].sum(
        axis=1
    )
    # The release score selects and scales the network as in engine evaluation.
    full18 = release18(root, fens)
    return np.column_stack([s12, s14, full18, s18, off18])


def score_chunk(job):
    root, rows = job
    fens = [fen for row in rows for fen in child_fens(row)]
    return score_position_batch(root, fens).reshape(-1, 2, 7)


def discovery(root):
    rows = []
    for phase, a, panel, _, _ in loaded(root):
        if phase != "discovery":
            continue
        for w, k, y, h, l, s in panel.tolist():
            parent = v.board(a, chess.PAWN, w, k, y, s)
            parent.turn = chess.WHITE
            choices = [
                (chess.Move(s, x).uci(), v.board(a, chess.PAWN, w, k, y, x))
                for x in (h, l)
            ]
            kids = [q for _, q in choices]
            # The discovery atlas supplies the original WDL order and tie filter.
            rows.append(
                [
                    "discovery",
                    "ordinary",
                    a,
                    orbit(w, k),
                    parent.fen(),
                    [m for m, _ in choices],
                    [1, 0],
                    [capture_flags(q) for q in kids],
                ]
            )
    return rows


def bootstrap(groups, effect):
    _, inverse = np.unique(groups, return_inverse=True)
    mass = np.bincount(inverse)
    sums = np.bincount(inverse, weights=effect)
    rng = np.random.default_rng(61031)
    draws = []
    for _ in range(40):
        ix = rng.integers(len(mass), size=(50, len(mass)))
        draws.extend(sums[ix].sum(axis=1) / mass[ix].sum(axis=1))
    return np.percentile(draws, [2.5, 97.5])


def counts(groups, baseline, adjusted, mask):
    x, y = baseline[mask], adjusted[mask]
    if len(x) == 0:
        return None
    lo, hi = bootstrap(groups[mask], y - x)
    return {
        "n": len(x),
        "clusters": len(np.unique(groups[mask])),
        "credit": x.mean(),
        "adjusted": y.mean(),
        "gain": (y - x).mean(),
        "lower": lo,
        "upper": hi,
        "rescued": int(((x < 1) & (y == 1)).sum()),
        "harmed": int(((x == 1) & (y < 1)).sum()),
    }


def load(root, work, months):
    init(root)
    records = discovery(root)
    originals = original_keys(root, records)
    for pieces in (4, 5, 6):
        with gzip.open(work / f"synthetic-{pieces}.jsonl.gz", "rt") as stream:
            records.extend(unique_pairs(map(json.loads, stream), originals))

    def archive_rows():
        for month in months:
            with gzip.open(work / f"{month}.jsonl.gz", "rt") as stream:
                yield from map(json.loads, stream)

    records.extend(unique_pairs(archive_rows(), originals))
    return records


def scored(root, work, records, workers):
    path = work / "scores.npz"
    with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as pool:
        scores = np.concatenate(
            list(
                pool.map(
                    score_chunk,
                    ((root, rows) for rows in batched(records, 1024)),
                )
            )
        )
    np.savez_compressed(path, scores=scores)
    return scores


def report(records, scores, out, work):
    panels, cells, defenders, groups = (
        np.array([r[i] for r in records]) for i in range(4)
    )
    labels = np.sign(np.array([r[6] for r in records]))
    direction = np.sign(labels[:, 0] - labels[:, 1])
    flags = np.array([r[7] for r in records])
    changes = (flags[:, 0] - flags[:, 1]) * direction[:, None]
    margins = (scores[:, 0] - scores[:, 1]) * direction[:, None]
    train = panels == "discovery"
    cols = (1, 3, 4)
    weights = {}
    for release, col in zip(RELEASES, cols):
        data = np.column_stack([defenders, direction, margins[:, col], changes])
        legal = fit(data[train], 3)
        surviving = fit(data[train], 4)
        weights[release] = (legal, surviving, surviving)

    # The two pre-existing SF14.1 frozen weights are recovered exactly.
    assert weights["SF14.1"] == (2800, 2800, 2800), weights
    rows, flow, threat = [], [], []
    for panel in np.unique(panels):
        for cell in ("all", *np.unique(cells[panels == panel])):
            eligible = (panels == panel) & ((cells == cell) if cell != "all" else True)
            for a in (0, *DEFENDERS):
                subset = eligible & ((defenders == a) if a else True)
                strict = subset & (direction != 0)
                ties14 = margins[:, 2] == 0
                ties12 = margins[:, 0] == 0
                active = changes[:, 1] != 0
                if not strict.any():
                    continue
                flow.append(
                    {
                        "panel": panel,
                        "cell": cell,
                        "defender": a,
                        "proposed": int(subset.sum()),
                        "strict": int(strict.sum()),
                        "sf14_tied": int((strict & ties14).sum()),
                        "sf12_tied": int((strict & ties12).sum()),
                        "favorable": int((strict & (changes[:, 1] > 0)).sum()),
                        "adverse": int((strict & (changes[:, 1] < 0)).sum()),
                        "tied_active": int((strict & ties14 & active).sum()),
                    }
                )
                if a != 0:
                    continue
                strata = [
                    ("all", strict),
                    ("classical_tie", strict & ties14),
                    ("same_option", strict & ~active),
                    ("favorable", strict & (changes[:, 1] > 0)),
                    ("adverse", strict & (changes[:, 1] < 0)),
                ]
                for stratum, mask in strata:
                    for release, col in zip(RELEASES, cols):
                        for i, (rule, weight) in enumerate(
                            zip(RULES, weights[release])
                        ):
                            baseline = credit(margins[:, col])
                            adjusted = credit(margins[:, col] + weight * changes[:, i])
                            result = counts(groups, baseline, adjusted, mask)
                            if result is not None:
                                rows.append(
                                    dict(
                                        panel=panel,
                                        cell=cell,
                                        stratum=stratum,
                                        release=release,
                                        rule=rule,
                                        weight=weight,
                                        **result,
                                    )
                                )
                    result = counts(
                        groups, credit(margins[:, 5]), credit(margins[:, 6]), mask
                    )
                    if result is not None:
                        threat.append(
                            dict(panel=panel, cell=cell, stratum=stratum, **result)
                        )
                sf12_result = counts(
                    groups,
                    credit(margins[:, 1]),
                    credit(margins[:, 1]),
                    strict & ties12,
                )
                if sf12_result is not None:
                    rows.append(
                        dict(
                            panel=panel,
                            cell=cell,
                            stratum="sf12_own_tie",
                            release="SF12",
                            rule="native",
                            weight=0,
                            **sf12_result,
                        )
                    )
    write(out / "expansion-flow.csv", flow)
    write(out / "expansion-static.csv", rows)
    write(out / "expansion-threats.csv", threat)
    # The raw archive preserves the scored order used by the search sampler.
    with gzip.open(work / "pairs.jsonl.gz", "wt") as stream:
        for row in records:
            print(json.dumps(row), file=stream)
    print(weights, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("step", choices=["synthetic", "games", "static"])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pieces", type=int, choices=(4, 5, 6))
    parser.add_argument("--archive")
    parser.add_argument(
        "--cached",
        action="store_true",
        help="summarize the existing aligned score archive",
    )
    parser.add_argument(
        "--months", nargs="+", default=[f"2014-{m:02}" for m in range(1, 7)]
    )
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / "manuscript/expansion"
    if args.step == "static":
        records = load(root, work, args.months)
        if args.cached:
            with gzip.open(work / "pairs.jsonl.gz", "rt") as stream:
                assert records == list(map(json.loads, stream))
            scores = np.load(work / "scores.npz")["scores"]
        else:
            scores = scored(root, work, records, args.workers)
        report(records, scores, args.results, work)
    else:
        if (args.step == "synthetic" and args.pieces is None) or (
            args.step == "games" and args.archive is None
        ):
            parser.error("supply --pieces for synthetic or --archive for games")
        save_candidates(root, work, args.step, args.workers, args.pieces, args.archive)


if __name__ == "__main__":
    main()
