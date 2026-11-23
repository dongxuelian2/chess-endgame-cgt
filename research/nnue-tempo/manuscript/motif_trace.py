"""Connect matched chess motifs to projected network margins and exact repair costs."""

import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network, credit, trunc
from promotion_geometry import relation
from study import check_native
from summarize import write

from circulation import energy_certificate, verify_energy
from cohorts import kings
from effects import batches, ci, delta, secant
from perspectives import channels
from tactical import replace_h1


def rows_for(rect):
    w, k, y, z, h, l, s = rect.T
    return np.stack(
        [np.column_stack([w, k, y, h, l, s]), np.column_stack([w, k, z, l, h, s])],
        axis=1,
    )


def trace(net, rect, a, b):
    rows = rows_for(rect).reshape(-1, 6)
    raw, ps, _ = net.encode(rows, a, b)
    x = raw.clip(0, 127)
    t = net.forward(raw, ps)
    check_native(a, b, rows[::2], {**t, "score": t["score"][::2]})
    w, k, y, h, l, _ = rows[1::2].T
    st = v.maps[a, b]
    assert np.array_equal(st[w, k, y, h]["n"], t["score"][1::2, 0])
    assert np.array_equal(st[w, k, y, l]["n"], t["score"][1::2, 1])
    assert not delta(raw).reshape(-1, 2, 2048).sum(1).any()
    return x, t


def contributions(net, x, t):
    d1 = secant(t["h1"], t["z"]).reshape(-1, 2, 8).mean(1)
    d2 = secant(t["h2"], t["z2"]).reshape(-1, 2, 32).mean(1)
    back = (d2 * net.out) @ net.w2
    from_ft = np.einsum(
        "nri,ni->nr", (delta(x) @ net.w1.T).reshape(-1, 2, 8) * d1[:, None, :], back
    )
    from_h1 = np.einsum("nri,ni->nr", delta(t["h1"]).reshape(-1, 2, 8), back)
    final = delta(t["pos"]).reshape(-1, 2)
    core = np.stack([from_ft, from_h1 - from_ft, final - from_h1], axis=-1)
    direct = delta(t["ps"]).reshape(-1, 2)
    whole = delta(t["score"]).reshape(-1, 2)
    out = np.concatenate(
        [
            core / 16,
            direct[:, :, None] / 16,
            (whole - (final + direct) / 16)[:, :, None],
        ],
        axis=-1,
    )
    assert np.allclose(out.sum(2), whole)
    return out


def ft_remove(net, x, t, ids):
    dz = delta(x)[:, ids] @ net.w1[:, ids].T
    z = t["z"]
    alt = net.upper(z + np.stack([-dz, dz], axis=1), t["ps"])["score"]
    return (alt[:, 0] - t["score"][:, 1]).reshape(-1, 2), (
        t["score"][:, 0] - alt[:, 1]
    ).reshape(-1, 2)


def unit_scores(net, t, donor=None):
    h = t["h2"][:, :, 2]
    replacement = 0 if donor is None else donor["h2"][:, :, 2]
    return trunc(t["ps"] + t["pos"] + (replacement - h) * net.out[2], 16)


def gate_swap(net, t, replacement):
    h = t["h1"].copy()
    h[:, :, [1, 3]] = replacement["h1"][:, :, [1, 3]]
    return replace_h1(net, t, h)


def cost(margins, target):
    return np.maximum(target - margins.sum(1), 0).astype(float) ** 2 / 2


def case_rows(net, rect, a, b, key, x, t, effects, variants, ids, swap, donor):
    w, k, _y, _z, h, l, s = map(int, rect)
    rows = rows_for(rect[None, :])[0]
    if b == 4:
        ids = [1691, 577, 961]
    details = []
    for j, (_, _, opp, high, low, _) in enumerate(rows.tolist()):
        p = v.board(a, b, w, k, opp, s)
        p.turn = chess.WHITE
        for side, square in enumerate((high, low)):
            move = chess.Move(s, square)
            assert move in p.legal_moves
            q = p.copy()
            q.push(move)
            truth = v.tb.probe_wdl(q)
            st = v.state(a, b, w, k, opp, square)
            assert truth == st["w"] and t["score"][j, side] == st["n"]
            bef = q.copy()
            best = []
            for reply in list(bef.legal_moves):
                san = bef.san(reply)
                bef.push(reply)
                if -np.sign(v.tb.probe_wdl(bef)) == np.sign(truth):
                    best.append(san)
                bef.pop()
            record = {
                "case": key,
                "white": a,
                "arc": j + 1,
                "end": "plus" if not side else "minus",
                "parent": p.fen(),
                "move": p.san(move),
                "wdl": int(np.sign(truth)),
                "optimal_black": " ".join(best),
                "hand": int(st["f"]),
                "score": int(st["n"]),
                "ft_indices": " ".join(map(str, ids)),
                "ft_values": " ".join(map(str, x[j, side, ids])),
                "h1_values": " ".join(map(str, t["h1"][j, side])),
                "h1_1_logit": int(t["z"][j, side, 1]),
                "h1_3_logit": int(t["z"][j, side, 3]),
                "h2_2_logit": int(t["z2"][j, side, 2]),
                "h2_2_byte": int(t["h2"][j, side, 2]),
                "h2_2_with_swapped_gate": int(swap["h2"][j, side, 2]) if swap else "",
                "h1_1_other_material": int(donor["h1"][j, side, 1]) if donor else "",
                "h1_3_other_material": int(donor["h1"][j, side, 3]) if donor else "",
                "direct_psqt": int(t["ps"][j, side]),
                "positional": int(t["pos"][j, side]),
            }
            details.append(record)
    target = [1, int(b == 1)]
    u = np.array([l, h])
    vv = np.array([h, l])
    records = []
    for name, val in variants.items():
        m = val[0]
        gap = np.asarray(target) - m
        norm, certificate = energy_certificate([(0, 1)], gap)
        verify_energy(u, vv, [(0, 1)], gap, norm, certificate)
        records.append(
            {
                "case": key,
                "white": a,
                "variant": name,
                "anchor": int(m[0]),
                "control_signed": int(m[1]),
                "circulation": int(sum(m)),
                "hodge": str(Fraction(int(sum(m)), 2)),
                "energy": str(norm),
                "dual_multiplier": str(certificate[0][1]) if certificate else "0",
                "raw_ft_projection": 0,
                "ft_projection": effects[0, :, 0].mean(),
                "h1_projection": effects[0, :, 1].mean(),
                "h2_projection": effects[0, :, 2].mean(),
                "psqt_projection": effects[0, :, 3].mean(),
                "rounding_projection": effects[0, :, 4].mean(),
            }
        )
    return details, records


def evaluate(net, rect, a, b, ids):
    x, t = trace(net, rect, a, b)
    effects = contributions(net, x, t)
    variants = {"native": delta(t["score"]).reshape(-1, 2)}
    swap = donor = None
    if b == 1:
        variants["ft_donor_plus"], variants["ft_donor_minus"] = ft_remove(
            net, x, t, ids
        )
    else:
        _, donor = trace(net, rect, 8 - a, b)
        swap = gate_swap(net, t, donor)
        variants["without_h2_2"] = delta(unit_scores(net, t)).reshape(-1, 2)
        variants["swap_h1_gate_only_h2_2"] = delta(unit_scores(net, t, swap)).reshape(
            -1, 2
        )
        variants["swap_h1_gate_all"] = delta(swap["score"]).reshape(-1, 2)
    return x, t, effects, variants, swap, donor


def neutral_capture(rect):
    w, k, _, z, h, l, _ = map(int, rect)
    have = []
    for piece in (3, 5):
        n = 0
        for d in (h, l):
            q = v.board(piece, 4, w, k, z, d)
            n += any(q.generate_legal_captures(to_mask=chess.BB_SQUARES[d]))
        have.append(n)
    return have[0] if have[0] == have[1] else 3


def run(root, out):
    v.init(root)
    net = Network(root)
    ranked = channels(out)
    all_stats = defaultdict(list)
    examples = {}
    for base, b in [(i, 1) for i in range(2, 6)] + [(3, 4)]:
        d = np.load(root / "manuscript/motifs" / f"panel-{base}-{b}.npz")
        for held in (False, True):
            rows = d["rows"][d["validation"] == held]
            if not len(rows):
                continue
            w, k, y, _z, h, l, _s = rows.T
            if b == 1:
                regime = (relation(base, h, y - 8, w, k) == 4) & (
                    relation(base, l, y - 8, w, k) == 2
                )
                file = (y ^ 7) % 8
                named = [
                    ("bgh_ray_reversal", regime & np.isin(file, [1, 6, 7])),
                    ("acdef_ray_reversal", regime & ~np.isin(file, [1, 6, 7])),
                    ("other_capture", ~regime),
                ]
            else:
                file = np.zeros(len(rows), dtype=int)
                named = [("skewer_vs_neutral", np.ones(len(rows), dtype=bool))]
            for regime, mask in named:
                for f in np.unique(file[mask]):
                    picked = rows[mask & (file == f)]
                    ids = ranked[f][:4] if b == 1 else []
                    for a in [base] if b == 1 else [3, 5]:
                        for rect in batches(picked, 128):
                            x, t, parts, variants, swap, donor = evaluate(
                                net, rect, a, b, ids
                            )
                            orig = variants["native"]
                            if b == 4:
                                captures = np.array([neutral_capture(r) for r in rect])
                                strata = [(regime, a, np.ones(len(rect), bool))]
                                for n, tag in enumerate(
                                    [
                                        "neutral_no_capture",
                                        "neutral_one_capture",
                                        "neutral_both_capture",
                                        "neutral_different_captures",
                                    ]
                                ):
                                    strata.append((tag, a, captures == n))
                            else:
                                keep = np.ones(len(rect), bool)
                                strata = [(regime, 0, keep), (regime, a, keep)]
                            term = (
                                delta(t["h2"])[:, 2].reshape(-1, 2) * net.out[2] / 16
                            ).sum(1) / 2
                            for label, m in variants.items():
                                z = np.c_[
                                    kings(rect),
                                    m,
                                    m.sum(1) / 2,
                                    cost(m, 2 if b == 1 else 1),
                                    credit(m[:, 0]),
                                    (m[:, 0] > 0) & (m[:, 1] > 0),
                                    parts.mean(1),
                                    term,
                                    (m.sum(1) - orig.sum(1)) / 2,
                                    ((m.sum(1) > 0) != (orig.sum(1) > 0)),
                                    delta(t["h2"])[::2, 2] * net.out[2] / 16,
                                    (t["h2"][::2, :, 2] == 0).all(1),
                                    abs(delta(t["h2"])[::2, 2]),
                                ]
                                for tag, piece, selection in strata:
                                    if selection.any():
                                        all_stats[tag, held, piece, label].append(
                                            z[selection]
                                        )
                            if held:
                                continue
                            gap = cost(orig, 2 if b == 1 else 1)
                            if b == 1:
                                alt1, alt2 = (
                                    variants["ft_donor_plus"],
                                    variants["ft_donor_minus"],
                                )
                                qualifies = (
                                    (orig[:, 0] < 0)
                                    & (orig.sum(1) < 2)
                                    & (alt1.sum(1) >= 2)
                                    & (alt2.sum(1) >= 2)
                                )
                                value = gap
                                case = regime
                            else:
                                # Use one shared rectangle for both white pieces.
                                if a != 3:
                                    continue
                                q = delta(donor["score"]).reshape(-1, 2)
                                test = variants["without_h2_2"]
                                qq = delta(
                                    unit_scores(net, donor, gate_swap(net, donor, t))
                                ).reshape(-1, 2)
                                qualifies = (
                                    (orig.sum(1) >= 1)
                                    & (test.sum(1) < 1)
                                    & (q.sum(1) < 1)
                                    & (qq.sum(1) >= 1)
                                )
                                value = cost(q, 1)
                                case = regime
                            if qualifies.any():
                                index = np.argmax(np.where(qualifies, value, -1))
                                if (
                                    case not in examples
                                    or examples[case][0] < value[index]
                                ):
                                    examples[case] = (
                                        value[index],
                                        rect[index].copy(),
                                        a,
                                        b,
                                        ids,
                                    )
    reports = []
    for (regime, held, a, label), chunks in sorted(all_stats.items()):
        z = np.concatenate(chunks)
        lo, hi = ci(z[:, 0], z[:, 13])
        reports.append(
            {
                "regime": regime,
                "cohort": "validation" if held else "discovery",
                "white": a,
                "variant": label,
                "rectangles": len(z),
                "king_orbits": len(set(z[:, 0])),
                "anchor_margin": z[:, 1].mean(),
                "signed_control_margin": z[:, 2].mean(),
                "hodge": z[:, 3].mean(),
                "positive_circulation": (z[:, 3] > 0).mean(),
                "can_calibrate": (z[:, 4] == 0).mean(),
                "mean_energy": z[:, 4].mean(),
                "median_energy": np.median(z[:, 4]),
                "anchor_score": z[:, 5].mean(),
                "both_strict": z[:, 6].mean(),
                "ft_projection": z[:, 7].mean(),
                "h1_projection": z[:, 8].mean(),
                "h2_projection": z[:, 9].mean(),
                "psqt_projection": z[:, 10].mean(),
                "rounding_projection": z[:, 11].mean(),
                "native_h2_2_projection": z[:, 12].mean(),
                "intervention_hodge_shift": z[:, 13].mean(),
                "shift_lower": lo,
                "shift_upper": hi,
                "changes_sign": z[:, 14].mean(),
                "native_anchor_h2_2_margin": z[:, 15].mean(),
                "native_anchor_h2_2_both_floor": z[:, 16].mean(),
                "native_anchor_h2_2_abs_gap": z[:, 17].mean(),
            }
        )
    positions, certificates = [], []
    for key, (_, rect, a, b, ids) in sorted(examples.items()):
        for piece in [a] if b == 1 else [3, 5]:
            x, t, parts, variants, swap, donor = evaluate(
                net, rect[None, :], piece, b, ids
            )
            detail, score = case_rows(
                net, rect, piece, b, key, x, t, parts, variants, ids, swap, donor
            )
            positions.extend(detail)
            certificates.extend(score)
    write(out / "motif-transfer.csv", reports)
    write(out / "motif-witness-positions.csv", positions)
    write(out / "motif-witness-certificates.csv", certificates)


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
