"""Explain the four-channel ordering gap using common-support comparisons."""

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import validate as v
from atlas.panels import raw
from effects import batches, delta, donor_margins
from expansion import (
    alternatives,
    bootstrap,
    capture_flags,
    child_fens,
    discovery,
    init,
    original_keys,
    pair_key,
    unique_pairs,
)
from network import Network, credit
from promotion_geometry import relation
from summarize import write

MONTHS = tuple(f"2014-{m:02d}" for m in range(7, 13))
STRATA = ("source", "2014_H1", "2014_H2")
# Maximum Manhattan displacement of the kings and the off-line defender.
# Native margins must have the same sign and lie within the stated caliper.
SPECS = (
    ("primary", (4, 4, 5), 24, 0.25),
    ("tight", (3, 3, 4), 16, 0.15),
    ("wide", (6, 6, 7), 48, 0.40),
)
NEIGHBORS = 3


@dataclass(frozen=True)
class Candidate:
    cohort: str
    group: str
    material: int
    coords: tuple
    geometry: tuple
    classic_tie: bool
    margin: int
    actual_higher: bool


@dataclass(frozen=True)
class Replay:
    values: np.ndarray
    logits: np.ndarray
    piece_square: np.ndarray
    drive: np.ndarray


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        yield from map(json.loads, stream)


def analyze_pair(row, scores, cohort):
    _, _, material, group, fen, moves, wdl, options = row
    labels = np.sign(wdl)
    if labels[0] == labels[1]:
        return None
    high = int(labels[1] > labels[0])
    low = 1 - high
    if options[high][1] != 1 or options[low][1] != 0:
        return None
    board = chess.Board(fen)
    w, k = board.king(chess.WHITE), board.king(chess.BLACK)
    y = next(iter(board.pieces(chess.PAWN, chess.BLACK)))
    h, l = [chess.Move.from_uci(moves[i]).to_square for i in (high, low)]
    if not (
        chess.square_rank(y) == 1
        and relation(material, h, y - 8, w, k) == 4
        and relation(material, l, y - 8, w, k) == 2
    ):
        return None
    flip = 7 if w % 8 < 4 else 0
    coords = (w, k, y, h, l)
    geometry = ((y ^ flip) % 8, (h ^ flip) - ((y ^ flip) - 8), int(labels[low]))
    return Candidate(
        cohort,
        str(group),
        material,
        coords,
        geometry,
        bool(scores[0, 0] == scores[1, 0]),
        int(scores[high, 1] - scores[low, 1]),
        high == 0,
    )


def prior(work):
    scores = np.load(work / "scores.npz")["scores"]
    reference, exposed_parents = [], []
    old_parents = set()
    prior_ids = set()
    for i, row in enumerate(read_rows(work / "pairs.jsonl.gz")):
        panel, _, _, gid, fen, _, _, _ = row
        if panel.startswith("games_"):
            old_parents.add(pair_key(fen, [])[0])
            prior_ids.add(gid)
        if panel == "games_4p_exposed":
            exposed_parents.append(row)
        if panel in ("4p_enriched", "games_4p_exposed"):
            cohort = "source" if panel == "4p_enriched" else "2014_H1"
            item = analyze_pair(row, scores[i, :, 2:4], cohort)
            if item is not None:
                reference.append(item)
    assert sum(r.cohort == "source" for r in reference) == 2016
    assert sum(r.cohort == "2014_H1" for r in reference) == 235
    return reference, old_parents, prior_ids, exposed_parents


def replicate(root, work, excluded, old_parents, prior_ids):
    def fresh_rows():
        for month in MONTHS:
            for row in read_rows(work / f"{month}.jsonl.gz"):
                if row[0] == "games_4p_exposed":
                    assert row[3] not in prior_ids
                    yield row

    counters = {
        "exposed_proposed": 0,
        "new_parent": 0,
        "distinct_pairs": 0,
        "strict": 0,
        "clear_off": 0,
    }

    def eligible():
        for row in fresh_rows():
            counters["exposed_proposed"] += 1
            if pair_key(row[4], [])[0] not in old_parents:
                counters["new_parent"] += 1
                yield row

    exe = work / "classical/src/stockfish"
    fresh = []
    exposed_parents = list(unique_pairs(eligible(), excluded))
    for group in batches(exposed_parents, 1024):
        counters["distinct_pairs"] += len(group)
        strict = [r for r in group if np.sign(r[6][0]) != np.sign(r[6][1])]
        counters["strict"] += len(strict)
        fens = [fen for row in strict for fen in child_fens(row)]
        if not fens:
            continue
        scores = raw(exe, fens, "scores").reshape(-1, 2, 2)
        for row, score in zip(strict, scores):
            item = analyze_pair(row, score, "2014_H2")
            if item is not None:
                fresh.append(item)
    counters["clear_off"] = len(fresh)
    return fresh, counters, exposed_parents


def traces(net, out, cases):
    with (out / "promotion.csv").open() as stream:
        channel_map = {
            (ord(row["file"]) - ord("a"), row["intervention"]): list(
                map(int, row["ft"].split())
            )
            for row in csv.DictReader(stream)
            if row["intervention"] in ("first_four", "activity_four")
        }
    columns = np.zeros((len(cases), 6), dtype=int)
    logits = np.empty((len(cases), 2, 8))
    piece_square = np.empty((len(cases), 2), dtype=int)
    drive = np.empty((len(cases), 8))
    keys = [(r.material, r.geometry[0]) for r in cases]
    for material, file in sorted(set(keys)):
        indices = [i for i, key in enumerate(keys) if key == (material, file)]
        for ix in batches(indices):
            coords = np.array([cases[i].coords for i in ix])
            raw_ft, ps, _ = net.encode(coords, material, chess.PAWN)
            trace = net.forward(raw_ft, ps)
            margins = delta(trace["score"])
            assert np.array_equal(margins, [cases[i].margin for i in ix])
            logits[ix] = trace["z"]
            piece_square[ix] = ps
            activation = raw_ft.clip(0, 127)
            for j, channels in enumerate(("first_four", "activity_four")):
                ids = channel_map[file, channels]
                hi, lo = donor_margins(net, activation, trace, ids)
                if channels == "first_four":
                    difference = delta(activation)[:, ids]
                    columns[ix, 4] = abs(difference).sum(axis=1)
                    drive[ix] = difference @ net.w1[:, ids].T
                    columns[ix, 5] = abs(drive[ix]).sum(axis=1)
                columns[ix, 2 * j : 2 * j + 2] = np.column_stack(
                    [hi - margins, lo - margins]
                )
    return Replay(columns, logits, piece_square, drive)


def effect(margin, shift):
    return credit(margin[..., None] + shift).mean(axis=-1) - credit(margin)


def features(cases):
    squares = np.array([(r.coords[0], r.coords[1], r.coords[4]) for r in cases])
    flips = 7 * (squares[:, 0] % 8 < 4)
    canonical = squares ^ flips[:, None]
    return np.stack([canonical % 8, canonical // 8], axis=-1)


def nearest(cases, source, games, spec):
    """Match at most three neighbors with replacement on the fixed design."""
    _, geometry_limit, floor, relative = spec
    xy = features(cases)
    margins = np.array([r.margin for r in cases])
    partitions = {}
    for i in source:
        row = cases[i]
        key = (row.material, *row.geometry)
        partitions.setdefault(key, []).append(i)
    designs = {"geometry": {}, "sign": {}, "margin": {}}
    for i in games:
        row = cases[i]
        key = (row.material, *row.geometry)
        pool = np.array(partitions.get(key, []), dtype=int)
        distance = abs(xy[pool] - xy[i]).sum(axis=2)
        same_children = (distance == 0).all(axis=1)
        okay = (distance <= geometry_limit).all(axis=1) & ~same_children
        pool, distance = pool[okay], distance[okay]
        shape = distance @ np.array([2, 2, 1])
        error = abs(margins[pool] - margins[i])
        same_sign = np.sign(margins[pool]) == np.sign(margins[i])
        caliper = max(floor, relative * abs(margins[i]))
        for stage, mask, cost in (
            ("geometry", np.ones(len(pool), bool), shape),
            ("sign", same_sign, shape),
            ("margin", same_sign & (error <= caliper), shape + 4 * error / caliper),
        ):
            order = np.argsort(cost[mask], kind="stable")[:NEIGHBORS]
            neighbors = pool[mask][order]
            if len(neighbors):
                designs[stage][i] = neighbors
    return designs


def edge_interval(groups, game, synth, weights, values):
    """Reweight matched edges by independent game and synthetic king clusters."""
    game_index = np.unique(groups[game], return_inverse=True)[1]
    synth_index = np.unique(groups[synth], return_inverse=True)[1]
    rng = np.random.default_rng(824739)
    draws = []
    for _ in range(40):
        gw = rng.exponential(size=(50, game_index.max() + 1))[:, game_index]
        sw = rng.exponential(size=(50, synth_index.max() + 1))[:, synth_index]
        weight = gw * sw * weights
        draws.extend((weight @ values) / weight.sum(axis=1)[:, None])
    return np.quantile(draws, [0.025, 0.975], axis=0)


def crossed_drive(net, replay, groups, synth, game, weights, cohort):
    """Apply each incoming logit change in each of the two matched contexts."""

    def score_change(context, donor):
        logits = replay.logits[context]
        ps = replay.piece_square[context]
        change = replay.drive[donor]
        shifted = logits + np.stack([-change, change], axis=1)
        return (
            delta(net.upper(shifted, ps)["score"]) / 2
            - delta(net.upper(logits, ps)["score"]) / 2
        )

    ss, gs, sg, gg = [
        score_change(context, donor)
        for context, donor in (
            (synth, synth),
            (game, synth),
            (synth, game),
            (game, game),
        )
    ]
    assert np.array_equal(ss, replay.values[synth, :2].mean(1))
    assert np.array_equal(gg, replay.values[game, :2].mean(1))
    incoming = ((ss - sg) + (gs - gg)) / 2
    downstream = ((ss - gs) + (sg - gg)) / 2
    assert np.allclose(incoming + downstream, ss - gg)
    estimates = np.column_stack([ss, gs, sg, gg, incoming, downstream])
    means = np.average(estimates, weights=weights, axis=0)
    lower, upper = edge_interval(groups, game, synth, weights, estimates[:, 4:])
    return {
        "cohort": cohort,
        "games": len(np.unique(game)),
        "synth_context_synth_drive": means[0],
        "game_context_synth_drive": means[1],
        "synth_context_game_drive": means[2],
        "game_context_game_drive": means[3],
        "incoming_component": means[4],
        "incoming_lower": lower[0],
        "incoming_upper": upper[0],
        "downstream_component": means[5],
        "downstream_lower": lower[1],
        "downstream_upper": upper[1],
    }


def summarize_matches(net, cases, replay, source, games, cohort):
    values = replay.values
    material = np.array([r.material for r in cases])
    groups = np.array([r.group for r in cases])
    m = np.array([r.margin for r in cases])
    shift = values[:, :2]
    gain = effect(m, shift)
    xyz = features(cases)
    summaries, swaps, paths = [], [], []
    for spec in SPECS:
        name = spec[0]
        designs = nearest(cases, source, games, spec)
        keep = np.array(sorted(designs["margin"]), dtype=int)
        if not len(keep):
            continue
        for stage in ("material", "geometry", "sign", "margin"):
            if stage == "material":
                matches = [source[material[source] == material[i]] for i in keep]
                available = len(games)
            else:
                matches = [designs[stage][i] for i in keep]
                available = len(designs[stage])
            length = np.array([len(r) for r in matches])
            si = np.concatenate(matches)
            gi = np.repeat(keep, length)
            weights = np.repeat(1 / length, length) / len(keep)
            spatial_diff = np.average(
                abs(xyz[si] - xyz[gi]).sum(axis=2), axis=0, weights=weights
            )
            contributions = np.bincount(si, weights=weights)
            low, high = (
                edge_interval(
                    groups, gi, si, weights, (gain[si] - gain[gi])[:, None]
                ).ravel()
                if stage == "margin"
                else (None, None)
            )
            summaries.append(
                {
                    "cohort": cohort,
                    "spec": name,
                    "stage": stage,
                    "games_all": len(games),
                    "available": available,
                    "games_matched": len(keep),
                    "synth_unique": len(np.unique(si)),
                    "synth_effective": 1 / (contributions @ contributions),
                    "native_synth": np.average(credit(m[si]), weights=weights),
                    "native_games": credit(m[keep]).mean(),
                    "wrong_synth": np.average(m[si] < 0, weights=weights),
                    "wrong_games": (m[keep] < 0).mean(),
                    "synth_abs_margin": np.average(abs(m[si]), weights=weights),
                    "games_abs_margin": abs(m[keep]).mean(),
                    "abs_margin_difference": np.average(
                        abs(m[si] - m[gi]), weights=weights
                    ),
                    "wk_distance": spatial_diff[0],
                    "bk_distance": spatial_diff[1],
                    "off_distance": spatial_diff[2],
                    "synth_ft_difference": np.average(values[si, 4], weights=weights),
                    "games_ft_difference": values[keep, 4].mean(),
                    "synth_h1_drive": np.average(values[si, 5], weights=weights),
                    "games_h1_drive": values[keep, 5].mean(),
                    "synth_shift": np.average(shift[si].mean(axis=1), weights=weights),
                    "games_shift": shift[keep].mean(),
                    "gain_synth": np.average(gain[si], weights=weights),
                    "gain_games": gain[keep].mean(),
                    "delta": np.average(gain[si] - gain[gi], weights=weights),
                    "lower": low,
                    "upper": high,
                }
            )
            if name == "primary" and stage == "margin":
                paths.append(
                    crossed_drive(net, replay, groups, si, gi, weights, cohort)
                )
            if name == "primary" and stage == "geometry":
                ss, gs, sg, gg = [
                    effect(m[i], shift[j])
                    for i, j in ((si, si), (gi, si), (si, gi), (gi, gi))
                ]
                margin_component = ((ss - gs) + (sg - gg)) / 2
                response_component = ((ss - sg) + (gs - gg)) / 2
                assert np.allclose(margin_component + response_component, ss - gg)
                estimates = np.column_stack(
                    [ss, gs, sg, gg, margin_component, response_component]
                )
                means = np.average(estimates, weights=weights, axis=0)
                lower, upper = edge_interval(groups, gi, si, weights, estimates[:, 4:])
                swaps.append(
                    {
                        "cohort": cohort,
                        "games": len(keep),
                        "synth_gap_synth_shift": means[0],
                        "game_gap_synth_shift": means[1],
                        "synth_gap_game_shift": means[2],
                        "game_gap_game_shift": means[3],
                        "native_margin_component": means[4],
                        "native_lower": lower[0],
                        "native_upper": upper[0],
                        "patch_response_component": means[5],
                        "response_lower": lower[1],
                        "response_upper": upper[1],
                    }
                )

    return summaries, swaps, paths


def raw_summary(cases, values):
    records = []
    groups = np.array([x.group for x in cases])
    margins = np.array([x.margin for x in cases])
    piece = np.array([x.material for x in cases])
    cohort = np.array([x.cohort for x in cases])
    played = np.array([x.actual_higher for x in cases])
    tied = np.array([x.classic_tie for x in cases])
    gain = effect(margins, values[:, :2])
    activity = effect(margins, values[:, 2:])
    for name in STRATA:
        splits = [
            ("all", np.ones(len(cases), bool)),
            ("rook", piece == 4),
            ("queen", piece == 5),
            ("wrong", margins < 0),
            ("right", margins > 0),
            ("tie", margins == 0),
            ("classical_tie", tied),
        ]
        if name != "source":
            splits += [("played_higher", played), ("played_lower", ~played)]
        for stratum, subset in splits:
            keep = (cohort == name) & subset
            n = int(keep.sum())
            if not n:
                continue
            base = credit(margins[keep]).mean()
            lo, hi = bootstrap(groups[keep], gain[keep])
            records.append(
                {
                    "cohort": name,
                    "stratum": stratum,
                    "n": n,
                    "native": base,
                    "wrong": (margins[keep] < 0).mean(),
                    "mean_abs_margin": abs(margins[keep]).mean(),
                    "median_margin": np.median(margins[keep]),
                    "ft_difference": values[keep, 4].mean(),
                    "h1_drive": values[keep, 5].mean(),
                    "mean_shift": values[keep, :2].mean(),
                    "positive_shift": (values[keep, :2] > 0).mean(),
                    "gain": gain[keep].mean(),
                    "patched": base + gain[keep].mean(),
                    "activity_gain": activity[keep].mean(),
                    "lower": lo,
                    "upper": hi,
                }
            )
    return records


def within_parent(net, out, work, parents, cohort):
    """Compare the observed move to other off-line moves against the same risky child."""
    comparisons = []
    groups = []
    seen = set()
    for panel, cell, a, gid, fen, moves, *_ in parents:
        if a not in (chess.ROOK, chess.QUEEN):
            continue
        parent = chess.Board(fen)
        observed = moves[0]
        source = chess.Move.from_uci(observed).from_square
        canonical = pair_key(fen, [observed])
        if canonical in seen:
            continue
        seen.add(canonical)
        w, k = parent.king(chess.WHITE), parent.king(chess.BLACK)
        y = next(iter(parent.pieces(chess.PAWN, chess.BLACK)))
        children = []
        for move, child in alternatives(parent, source):
            target = chess.Move.from_uci(move).to_square
            shape = int(relation(a, target, y - 8, w, k))
            if shape in (2, 4):
                wd = v.tb.probe_wdl(child)
                feature = capture_flags(child)
                if (shape == 4) == bool(feature[1]):
                    children.append((move, wd, feature, shape))
        actual = next(
            (row for row in children if row[0] == observed and row[3] == 2), None
        )
        if actual is None:
            continue
        others = [
            r
            for r in children
            if r[3] == 2 and r[0] != observed and np.sign(r[1]) == np.sign(actual[1])
        ]
        if not others:
            continue
        for move, wd, feature, shape in children:
            if shape == 4 and np.sign(wd) > np.sign(actual[1]):
                anchor = []
                for alternate, wdl, flags, _ in (actual, *others):
                    anchor.append(len(comparisons))
                    comparisons.append(
                        (
                            panel,
                            cell,
                            a,
                            gid,
                            fen,
                            [move, alternate],
                            [wd, wdl],
                            [feature, flags],
                        )
                    )
                groups.append((gid, anchor))

    exe = work / "classical/src/stockfish"
    cases = []
    for group in batches(comparisons, 1024):
        fens = [f for row in group for f in child_fens(row)]
        scores = raw(exe, fens, "scores").reshape(-1, 2, 2)
        for row, score in zip(group, scores):
            item = analyze_pair(row, score, cohort)
            assert item is not None
            cases.append(item)
    values = traces(net, out, cases).values
    margin = np.array([r.margin for r in cases])
    metrics = np.column_stack(
        [credit(margin), effect(margin, values[:, :2]), values[:, :2].mean(1)]
    )
    games = {}
    for gid, (observed, *alternatives_) in groups:
        games.setdefault(gid, []).append(
            [metrics[observed], metrics[alternatives_].mean(0)]
        )
    game_summary = np.array([np.mean(v, axis=0) for v in games.values()])
    difference = game_summary[:, 1] - game_summary[:, 0]
    observed, alternatives_ = game_summary.mean(axis=0)
    low, high = bootstrap(np.array(list(games)), difference[:, 1])
    return {
        "cohort": cohort,
        "games": len(games),
        "anchors": len(groups),
        "played_comparisons": len(groups),
        "unplayed_comparisons": len(cases) - len(groups),
        "native_played": observed[0],
        "native_unplayed": alternatives_[0],
        "gain_played": observed[1],
        "gain_unplayed": alternatives_[1],
        "shift_played": observed[2],
        "shift_unplayed": alternatives_[2],
        "gain_difference": difference[:, 1].mean(),
        "lower": low,
        "upper": high,
    }


def older_parent_check(root, net, cases, replay, source, parents):
    original_parents = {
        pair_key(row[2], [])[0]
        for name in ("atlas/played.gz", "atlas/horizon/played.gz")
        for row in read_rows(root / name)
    }
    repeated = {r[3] for r in parents if pair_key(r[4], [])[0] in original_parents}
    games = np.array(
        [
            i
            for i, r in enumerate(cases)
            if r.cohort == "2014_H2" and r.group not in repeated
        ],
        dtype=int,
    )
    records, _, _ = summarize_matches(net, cases, replay, source, games, "2014_H2")
    margin = np.array([cases[i].margin for i in games])
    gain = effect(margin, replay.values[games, :2])
    lower, upper = bootstrap(np.array([cases[i].group for i in games]), gain)
    matched = next(
        r for r in records if r["spec"] == "primary" and r["stage"] == "margin"
    )
    return {
        "n": len(games),
        "gain": gain.mean(),
        "lower": lower,
        "upper": upper,
        "matched": matched["games_matched"],
        "matched_gap": matched["delta"],
        "matched_lower": matched["lower"],
        "matched_upper": matched["upper"],
    }


def main(root, out, baseline_only):
    work = root / "manuscript/expansion"
    cases, seen_parents, prior_ids, early_exposure = prior(work)
    if not baseline_only:
        init(root)
        prior_cases = discovery(root)
        originals = original_keys(root, prior_cases)
        new, counters, late_exposure = replicate(
            root, work, originals, seen_parents, prior_ids
        )
        cases.extend(new)
        write(out / "promotion-match-flow.csv", [counters])
    net = Network(root)
    replay = traces(net, out, cases)
    summary = raw_summary(cases, replay.values)
    found = {r["cohort"]: r["gain"] for r in summary if r["stratum"] == "all"}
    assert np.isclose(found["source"], 0.21155753968253968)
    assert np.isclose(found["2014_H1"], -0.007446808510638298)
    source = np.array([i for i, r in enumerate(cases) if r.cohort == "source"])
    results, swaps, paths = [], [], []
    for name in STRATA[1:]:
        games = np.array(
            [i for i, r in enumerate(cases) if r.cohort == name], dtype=int
        )
        if len(games):
            rows, swap, path = summarize_matches(
                net, cases, replay, source, games, name
            )
            results.extend(rows)
            swaps.extend(swap)
            paths.extend(path)
    write(out / "promotion-match-strata.csv", summary)
    write(out / "promotion-matched.csv", results)
    write(out / "promotion-match-swap.csv", swaps)
    write(out / "promotion-match-path.csv", paths)
    if not baseline_only:
        selections = [(early_exposure, "2014_H1"), (late_exposure, "2014_H2")]
        rows = [
            within_parent(net, out, work, parents, cohort)
            for parents, cohort in selections
        ]
        write(out / "promotion-match-played-choice.csv", rows)
        row = older_parent_check(root, net, cases, replay, source, late_exposure)
        write(out / "promotion-match-history.csv", [row])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve(), args.baseline_only)
