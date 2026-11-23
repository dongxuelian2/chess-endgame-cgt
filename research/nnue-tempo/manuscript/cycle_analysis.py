"""Compute exact cycle-mean constraints and locate the corresponding network signal."""

import sys
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network
from summarize import write

from circulation import (
    directed_cycles,
    energy_certificate,
    interaction_projection,
    verify_energy,
)
from effects import delta, secant


def cycle_minimum(start, end, margins):
    """Minimum mean of an integer-weight directed multigraph, as a fraction."""
    nodes = sorted(set(start) | set(end))
    index = {x: i for i, x in enumerate(nodes)}
    n = len(nodes)
    a = np.array([index[x] for x in start])
    b = np.array([index[x] for x in end])
    missing = n * (abs(margins).max() + 1) * 2
    state = np.full((n, n), missing, dtype=np.int64)
    np.fill_diagonal(state, 0)
    best = None
    for k in range(1, n + 1):
        extension = state[:, a] + margins
        extension[state[:, a] == missing] = missing
        nxt = np.full_like(state, missing)
        np.minimum.at(nxt.T, b, extension.T)
        value = int(nxt.diagonal().min())
        if value != missing:
            mean = Fraction(value, k)
            best = mean if best is None else min(best, mean)
        state = nxt
    assert best is not None
    return best


def additive_errors(start, end):
    """Exact weighted feedback-arc optimum with an independent small-component check."""
    nodes = sorted(set(start) | set(end))
    pos = {x: i for i, x in enumerate(nodes)}
    a, b = [[pos[x] for x in endpoints] for endpoints in (start, end)]
    cnt = np.zeros((len(nodes), len(nodes)), dtype=int)
    np.add.at(cnt, (a, b), 1)
    reached = cnt > 0
    np.fill_diagonal(reached, True)
    for k in range(len(nodes)):
        reached |= reached[:, k, None] & reached[k, None, :]
    components = {
        tuple(np.flatnonzero(reached[k] & reached[:, k])) for k in range(len(nodes))
    }
    missed = 0
    for comp in components:
        q = cnt[np.ix_(comp, comp)]
        n = len(comp)
        best = [0] * (1 << n)
        for subset in range(1, len(best)):
            present = [i for i in range(n) if subset >> i & 1]
            best[subset] = max(
                best[subset ^ (1 << last)] + sum(q[i, last] for i in present)
                for last in present
            )
        brute = max(
            sum(q[i, j] for l, i in enumerate(order) for j in order[l + 1 :])
            for order in permutations(range(n))
        )
        assert best[-1] == brute
        missed += int(q.sum()) - best[-1]
    return missed, max(map(len, components))


def split(path):
    result = defaultdict(list)
    for line in path.read_text().splitlines():
        fields = tuple(map(int, line.split()))
        result[fields[:5]].append(fields[5:])
    return result


def trace_cycle(net, key, a, b, path):
    w, k, s, _, _ = key
    rows = np.array([(w, k, y, h, l, s) for _, _, _, y, l, h, *_ in path])
    raw, ps, _ = net.encode(rows, a, b)
    x = raw.clip(0, 127)
    t = net.forward(raw, ps)
    assert np.array_equal(t["score"], np.array([[hn, ln] for *_, ln, hn in path]))
    assert not delta(raw).sum(0).any()
    slopes = [secant(x, raw), secant(t["h1"], t["z"]), secant(t["h2"], t["z2"])]
    means = [d.mean(0) for d in slopes]
    signal = delta(raw)
    ledgers = [np.zeros_like(signal, dtype=float) for _ in slopes]
    weights = [None, net.w1, net.w2]
    outputs = []
    for i, (d, mean, weight) in enumerate(zip(slopes, means, weights)):
        if weight is not None:
            signal = signal @ weight.T
            ledgers = [z @ weight.T for z in ledgers]
        ledgers = [z * mean for z in ledgers]
        ledgers[i] += (d - mean) * signal
        signal = d * signal
        assert np.allclose(sum(ledgers).sum(0), signal.sum(0))
    for z in ledgers:
        outputs.append(float(z.sum(0) @ net.out / 16))
    assert np.isclose(sum(outputs), delta(t["pos"]).sum() / 16)
    # Integer PSQT by itself preserves the weak order of its additive precursor.
    assert abs(delta(ps).sum()) <= len(rows)
    return outputs, float(delta(ps).sum() / 16), int(delta(t["score"]).sum())


def frozen_readout(net, key, a, b, arc_array):
    w, k, s, _, _ = key
    y, start, end, _, native, _, _ = arc_array.T
    rows = np.column_stack(
        [np.full(len(y), w), np.full(len(y), k), y, end, start, np.full(len(y), s)]
    )
    raw, ps, _ = net.encode(rows, a, b)
    t = net.forward(raw, ps)
    assert np.array_equal(delta(t["score"]), native)
    clipped = raw.clip(0, 127)
    h1_gain = secant(t["h1"], t["z"]).mean(0)
    h2_gain = secant(t["h2"], t["z2"]).mean(0)
    readout = (h2_gain * net.out) @ net.w2
    from_ft = (delta(clipped) @ net.w1.T * h1_gain) @ readout
    from_h1 = delta(t["h1"]) @ readout
    final = delta(t["pos"])
    stages = np.column_stack([from_ft, from_h1 - from_ft, final - from_h1]) / 16
    interaction = interaction_projection(start, end, stages)
    summed = interaction.sum(1)
    cross_ft_h1 = float(interaction[:, 0] @ interaction[:, 1])
    cross_lower_h2 = float((interaction[:, 0] + interaction[:, 1]) @ interaction[:, 2])
    cycle = directed_cycles(start, end)
    assert cycle
    for weights in (native, final):
        expected = min(
            Fraction(sum(int(weights[i]) for i in path), len(path)) for path in cycle
        )
        assert cycle_minimum(start, end, weights) == expected
    desired = np.ones(len(start), dtype=int)
    reference, additive_proof = energy_certificate(cycle, desired)
    verify_energy(start, end, cycle, desired, reference, additive_proof)
    deficit, proof = energy_certificate(cycle, desired - native)
    verify_energy(start, end, cycle, desired - native, deficit, proof)
    certificate = "|".join(
        f"{c}@" + ";".join(f"{y[i]}:{start[i]}-{end[i]}" for i in cycle[j])
        for j, c in proof
    )
    # Round to a public fixed grid before the exact graph calculation.
    # Each true mean lies within one half-step of the rounded-arm mean.
    unit = 1024 * 16
    return {
        "ft_repaired_margin": str(
            cycle_minimum(start, end, np.rint(from_ft * 1024).astype("i8")) / unit
        ),
        "h1_repaired_margin": str(
            cycle_minimum(start, end, np.rint(from_h1 * 1024).astype("i8")) / unit
        ),
        "h2_repaired_margin": str(cycle_minimum(start, end, final) / 16),
        "directed_cycles": len(cycle),
        "unit_gap_energy_additive": str(reference),
        "unit_gap_energy_native": str(deficit),
        "native_energy_certificate": certificate,
        "cycle_energy_ft": float(interaction[:, 0] @ interaction[:, 0]),
        "cycle_energy_h1": float(interaction[:, 1] @ interaction[:, 1]),
        "cycle_energy_h2": float(interaction[:, 2] @ interaction[:, 2]),
        "cycle_energy_native_positional": float(summed @ summed),
        "cycle_ft_h1_inner_product": cross_ft_h1,
        "cycle_lower_h2_inner_product": cross_lower_h2,
    }


def audit_cycle(key, a, b, path):
    w, k, s, _, _ = key
    witness = []
    for _, _, _, y, l, h, low, high, ln, hn in path:
        p = v.board(a, b, w, k, y, s)
        p.turn = chess.WHITE
        entry = {"parent": p.fen(), "low": ln, "high": hn, "white": a, "black": b}
        for label, target, expected in [("from", l, low), ("to", h, high)]:
            move = chess.Move(s, target)
            assert move in p.legal_moves
            entry[label + "_move"] = p.san(move)
            p.push(move)
            assert v.tb.probe_wdl(p) == expected
            entry[label + "_wdl"] = expected
            p.pop()
        witness.append(entry)
    return witness


def audit_arcs(a, b, key, records, graph):
    w, k, source, classical, signature = key
    ledger = []
    for y, low, high, gap, native, positional, direct in records:
        parent = v.board(a, b, w, k, y, source)
        parent.turn = chess.WHITE
        moves = [chess.Move(source, target) for target in (low, high)]
        wdl = []
        for move in moves:
            assert parent.is_legal(move)
            parent.push(move)
            wdl.append(v.tb.probe_wdl(parent))
            parent.pop()
        assert v.sign(wdl[1]) - v.sign(wdl[0]) == gap
        ledger.append(
            {
                "graph": graph,
                "parent": parent.fen(),
                "lower_move": moves[0].uci(),
                "higher_move": moves[1].uci(),
                "lower_wdl": wdl[0],
                "higher_wdl": wdl[1],
                "classical": classical,
                "reply_signature": signature,
                "native_margin": native,
                "positional_margin": positional,
                "direct_margin": direct,
            }
        )
    return ledger


def main(root, out):
    net = Network(root)
    v.init(root)
    directory = root / "manuscript/cycles"
    census, records, witnesses, arcs = [], [], [], []
    checked = Counter()
    for f in sorted(directory.glob("*.cycles")):
        a, b = map(int, f.stem.split("-"))
        for line in f.read_text().splitlines():
            truth, neural, count = map(int, line.split(","))
            census.append(
                {
                    "white": a,
                    "black": b,
                    "shortest": truth,
                    "neural_shortest": neural,
                    "graphs": count,
                }
            )
        arc_data = split(f.with_suffix(".arcs"))
        assert len(arc_data) == sum(
            r["graphs"]
            for r in census
            if (r["white"], r["black"]) == (a, b) and r["shortest"] >= 3
        )
        for key, path in split(f.with_suffix(".paths")).items():
            length, success, *_ = path[0]
            assert length == len(path)
            assert all(path[i][5] == path[(i + 1) % length][4] for i in range(length))
            checked[length, success] += 1
            entries = audit_cycle(key, a, b, path)
            if checked[length, success] <= 4:
                witnesses.extend(
                    dict(
                        shortest=length,
                        neural=success,
                        witness=checked[length, success],
                        **entry,
                    )
                    for entry in entries
                )
            arcs.extend(audit_arcs(a, b, key, arc_data[key], len(records) + 1))
            flow, ps, native_sum = trace_cycle(net, key, a, b, path)
            _, start, end, truth, native, positional, direct = np.asarray(
                arc_data[key]
            ).T
            record = {
                "white": a,
                "black": b,
                "wk": key[0],
                "bk": key[1],
                "source": key[2],
                "classical": key[3],
                "signature": key[4],
                "edges": len(start),
                "shortest": length,
                "correct_shortest": success,
                "truth_uniform_cost": str(-cycle_minimum(start, end, -truth)),
                "native_correct_all": int(native.min() > 0),
                "native_repaired_margin": str(cycle_minimum(start, end, native)),
                "positional_repaired_margin": str(
                    cycle_minimum(start, end, positional)
                ),
                "direct_repaired_margin": str(cycle_minimum(start, end, direct)),
                "cycle_ft": flow[0],
                "cycle_h1": flow[1],
                "cycle_h2": flow[2],
                "cycle_ps": ps,
                "cycle_native": native_sum,
            }
            minimum, max_component = additive_errors(start, end)
            record["additive_min_errors"] = minimum
            record["largest_scc"] = max_component
            record["native_strict_errors_or_ties"] = int((native <= 0).sum())
            record["native_strict_errors"] = int((native < 0).sum())
            record.update(frozen_readout(net, key, a, b, np.asarray(arc_data[key])))
            records.append(record)
    write(out / "preference-graphs.csv", census)
    write(out / "preference-cycles.csv", records)
    write(out / "cycle-witnesses.csv", witnesses)
    write(out / "cycle-arcs.csv", arcs)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
