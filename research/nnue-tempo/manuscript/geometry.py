"""Hold kings and one man fixed in the knight-fork geometry and audit skewer edges."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network
from summarize import write

from cohorts import orbit
from effects import batches, ci, delta, secant
from tactical import load


def fork(root, out, net, ids):
    # Potential fork squares shared by the king and rook; the knight must reach one now.
    attacks = np.array(chess.BB_KNIGHT_ATTACKS, dtype="uint64")
    v.state(4, 2, 0, 0, 0, 0)
    st = v.maps[4, 2]
    sumstats = defaultdict(lambda: np.zeros(len(ids) + 2))
    for w in range(64):
        if w % 8 >= 4:
            continue
        common = attacks[w] & attacks
        reached = (attacks[:, None] & common[None, :]) != 0
        for k in range(64):
            valid = st[w, k]["v"] > 0
            y, x = np.where(valid)
            if not len(y):
                continue
            features = []
            for u in ids:
                black = u < 1024
                king = k if black else w
                flip = 56 * int(black) + 7 * (king % 8 < 4)
                canon = king ^ flip
                bucket = 31 - 4 * (canon // 8) - (canon % 8 - 4)
                W = net.w[bucket, :, :, u % 1024]
                raw = (
                    net.bias[u % 1024]
                    + W[10, w ^ flip]
                    + W[10, k ^ flip]
                    + W[2 + int(not black), y ^ flip]
                    + W[6 + int(black), x ^ flip]
                )
                M = np.zeros((64, 64))
                M[y, x] = raw.clip(0, 127)
                features.append(M)
            F = np.stack(features, axis=-1)
            for axis, name in [(0, "move_knight"), (1, "move_rook")]:
                on = valid & reached
                off = valid & ~reached & (common != 0)
                n, m = on.sum(axis), off.sum(axis)
                use = (n * m) > 0
                if not use.any():
                    continue
                change = (F * on[:, :, None]).sum(axis)[use] / n[use, None] - (
                    F * off[:, :, None]
                ).sum(axis)[use] / m[use, None]
                counts = (n * m)[use]
                weights = counts[:, None]
                sumstats[name, "signed"] += np.r_[
                    use.sum(), counts.sum(), (weights * change).sum(0)
                ]
                sumstats[name, "abs_context_mean"] += np.r_[
                    use.sum(), counts.sum(), (weights * abs(change)).sum(0)
                ]
    records = []
    for (edit, stat), vls in sumstats.items():
        for u, score in zip(ids, vls[2:]):
            records.append(
                {
                    "edit": edit,
                    "stat": stat,
                    "ft": u,
                    "fixed_contexts": int(vls[0]),
                    "contrasts": int(vls[1]),
                    "mean_contrast": score / vls[1],
                }
            )
    write(out / "fork-geometry.csv", records)


def skewer(root, out, net, ids):
    rows = load(root, "validation", 3, 4)
    records = []
    sums = defaultdict(float)
    for r in batches(rows):
        for a in (3, 5):
            raw, p, _ = net.encode(r, a, 4)
            x = raw.clip(0, 127)
            t = net.forward(raw, p)
            sec = secant(t["h2"], t["z2"])[:, 2]
            edges = delta(t["h1"]) * net.w2[2] * sec[:, None] * net.out[2] / 16
            assert np.allclose(edges.sum(1), delta(t["h2"])[:, 2] * net.out[2] / 16)
            for i in range(8):
                sums[a, "H1_to_H2", i] += np.array(
                    [
                        edges[:, i].sum(),
                        abs(edges[:, i]).sum(),
                        t["h1"][:, :, i].mean(1).sum(),
                    ]
                )
            for u in ids:
                sums[a, "FT", u] += np.array(
                    [
                        delta(x)[:, u].sum(),
                        abs(delta(x)[:, u]).sum(),
                        x[:, :, u].mean(1).sum(),
                    ]
                )
    for (a, l, i), z in sums.items():
        records.append(
            {
                "white": a,
                "component": l,
                "unit": i,
                "n": len(rows),
                "signed": z[0] / len(rows),
                "absolute": z[1] / len(rows),
                "mean_byte": z[2] / len(rows),
            }
        )
    write(out / "skewer-edges.csv", records)


def fork_rectangles(out, net, ids):
    """Test 2x2 fork crossover rectangles where both pieces are two knight moves from K."""
    attacks = np.array(chess.BB_KNIGHT_ATTACKS, dtype="uint64")
    v.state(4, 2, 0, 0, 0, 0)
    st = v.maps[4, 2]
    rng = np.random.default_rng(42)
    sample_rows = []
    sample_orbits = []
    for w in range(64):
        if w % 8 >= 4:
            continue
        common_wx = attacks[w] & attacks
        has_wx = common_wx != 0
        true_fork = (attacks[:, None] & common_wx[None, :]) != 0
        for k in range(0, 64, 2):
            valid = st[w, k]["v"] > 0
            good = valid & has_wx[:, None] & has_wx[None, :]
            ys = np.where((good & true_fork).any(1) & (good & ~true_fork).any(1))[0]
            if len(ys) < 2:
                continue
            orb = orbit(w, k)
            for _ in range(30):
                y1, y2 = rng.choice(ys, 2, replace=False)
                x1s = np.where(good[y1] & true_fork[y1] & good[y2] & ~true_fork[y2])[0]
                x2s = np.where(good[y2] & true_fork[y2] & good[y1] & ~true_fork[y1])[0]
                if not len(x1s) or not len(x2s):
                    continue
                x1 = int(rng.choice(x1s))
                x2 = int(rng.choice(x2s))
                sample_rows.append([w, k, int(y1), x1, x2, 0])
                sample_rows.append([w, k, int(y2), x2, x1, 0])
                sample_orbits.append(orb)
    arr = np.array(sample_rows, dtype=int)
    keys = np.array(sample_orbits, dtype=int)
    n_rect = len(keys)
    raw_mix_list, ft_mix_list, h1_4_list, score_list = [], [], [], []
    for i in range(0, len(arr), 256):
        batch = arr[i : i + 256]
        raw, ps, _ = net.encode(batch, 4, 2)
        x = raw.clip(0, 127)
        t = net.forward(raw, ps)
        r_pair = delta(raw[:, :, ids]).reshape(-1, 2, len(ids)).sum(1)
        assert np.all(r_pair == 0)
        raw_mix_list.append(r_pair)
        ft_mix_list.append(delta(x[:, :, ids]).reshape(-1, 2, len(ids)).sum(1))
        h1_4_list.append(delta(t["h1"][:, :, 4]).reshape(-1, 2).sum(1))
        score_list.append(delta(t["score"]).reshape(-1, 2).sum(1))
    ft_mix = np.concatenate(ft_mix_list)
    h1_4_mix = np.concatenate(h1_4_list)
    score_mix = np.concatenate(score_list)
    records = []
    for idx, u in enumerate(ids):
        lo, hi = ci(keys, ft_mix[:, idx])
        records.append(
            {
                "component": f"FT{u}",
                "rectangles": n_rect,
                "king_orbits": len(set(sample_orbits)),
                "mean_mixed_diff": float(ft_mix[:, idx].mean()),
                "lower": float(lo),
                "upper": float(hi),
            }
        )
    for cname, vals in [("H1_unit_4", h1_4_mix), ("score", score_mix)]:
        lo, hi = ci(keys, vals)
        records.append(
            {
                "component": cname,
                "rectangles": n_rect,
                "king_orbits": len(set(sample_orbits)),
                "mean_mixed_diff": float(vals.mean()),
                "lower": float(lo),
                "upper": float(hi),
            }
        )
    write(out / "fork-rectangles.csv", records)


def fork_colocalization(out, net, ids):
    """Correlate Black-knight and White-rook weights across White-king checking squares."""
    attacks = np.array(chess.BB_KNIGHT_ATTACKS, dtype="uint64")
    corrs = []
    for rank in range(8):
        for file in range(4, 8):
            canon = 8 * rank + file
            bucket = 31 - 4 * (canon // 8) - (canon % 8 - 4)
            csqs = [s for s in range(64) if (int(attacks[canon]) >> s) & 1]
            if len(csqs) < 3:
                continue
            bn = np.stack(
                [
                    net.w[
                        bucket, 3, [q for q in range(64) if (int(attacks[s]) >> q) & 1], :
                    ].mean(0)
                    for s in csqs
                ]
            )
            wr = np.stack(
                [
                    net.w[
                        bucket, 6, [q for q in range(64) if (int(attacks[s]) >> q) & 1], :
                    ].mean(0)
                    for s in csqs
                ]
            )
            bn_c = bn - bn.mean(0, keepdims=True)
            wr_c = wr - wr.mean(0, keepdims=True)
            denom = np.sqrt((bn_c**2).sum(0) * (wr_c**2).sum(0))
            c = np.divide(
                (bn_c * wr_c).sum(0), denom, where=denom > 0, out=np.zeros(1024)
            )
            corrs.append(c)
    all_corrs = np.stack(corrs, axis=1)
    mean_corr = all_corrs.mean(1)
    order = np.argsort(-mean_corr, kind="stable")
    ranks = np.empty(1024, dtype=int)
    ranks[order] = np.arange(1, 1025)
    top_set = set(ids)
    records = []
    for i in range(1024):
        u = 1024 + i
        pct = float((mean_corr < mean_corr[i]).mean() * 100)
        records.append(
            {
                "ft": u,
                "is_fork_top8": int(u in top_set),
                "mean_corr": float(mean_corr[i]),
                "positive_buckets": int((all_corrs[i] > 0).sum()),
                "buckets_tested": all_corrs.shape[1],
                "percentile": pct,
                "rank": int(ranks[i]),
            }
        )
    write(out / "fork-colocalization.csv", records)


if __name__ == "__main__":
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    v.init(root)
    net = Network(root)
    with (out / "tactical-inputs.csv").open() as f:
        ranks = list(csv.DictReader(f))
    ids = lambda b: [int(r["ft"]) for r in ranks if int(r["black"]) == b]
    fork(root, out, net, ids(2))
    skewer(root, out, net, ids(4))
    fork_rectangles(out, net, ids(2))
    fork_colocalization(out, net, ids(2))
