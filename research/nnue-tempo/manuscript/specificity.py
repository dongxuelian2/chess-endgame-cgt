"""Measure motif specificity, confidence intervals, and activity-matched controls for skewers and forks."""

import csv
import sys
from pathlib import Path

import numpy as np
import validate as v
from network import Network, credit
from study import check_native
from summarize import write

from cohorts import kings
from effects import batches, ci, delta
from tactical import FORK, drop_h2, load, replace_h1


def skewer_specificity(root, out, net):
    """Evaluate H2 unit 2 on matched/outside skewer and non-skewer cohorts plus H1_{1,3} hybrids."""
    rows = []
    cohorts = [
        ("matched_B_skewer", 3, "validation-3-4.npz", True),
        ("matched_Q_skewer", 5, "validation-3-4.npz", True),
        ("matched_B_nonskewer", 3, "validation-3-4.npz", False),
        ("outside_B_skewer", 3, "transfer-3-4.npz", True),
        ("outside_B_nonskewer", 3, "transfer-3-4.npz", False),
        ("outside_Q_skewer", 5, "transfer-5-4.npz", True),
    ]
    for name, a, fname, use_mask in cohorts:
        d = np.load(root / "manuscript" / fname)
        r = d["rows"][d["mask"] if use_mask else ~d["mask"]]
        ks = kings(r)
        nat_all, pat_all, mag_all, c2_all, gap_all = [], [], [], [], []
        h2_sum = np.zeros(32)
        for chunk in batches(r):
            raw, ps, _ = net.encode(chunk, a, 4)
            t = net.forward(raw, ps)
            check_native(a, 4, chunk, t)
            nat_all.append(credit(delta(t["score"])))
            pat_all.append(drop_h2(net, t, 2))
            mag_all.append(np.abs(delta(t["score"])))
            c2_all.append(delta(t["h2"])[:, 2] * net.out[2] / 16)
            gap_all.append(np.abs(delta(t["h2"])[:, 2]))
            h2_sum += (delta(t["h2"]) * net.out / 16).sum(0)
        nat = np.concatenate(nat_all)
        pat = np.concatenate(pat_all)
        mag = np.concatenate(mag_all)
        c2 = np.concatenate(c2_all)
        gap = np.concatenate(gap_all)
        dif = nat - pat
        lo, hi = ci(ks, dif)
        n = len(r)
        rows.append(
            {
                "cohort": name,
                "white": a,
                "n": n,
                "king_orbits": len(set(ks)),
                "native_or_hybrid": float(nat.mean()),
                "without_h2_2": float(pat.mean()),
                "loss": float(dif.mean()),
                "lower": float(lo),
                "upper": float(hi),
                "median_abs_margin": float(np.median(mag)),
                "mean_c2_score": float(c2.mean()),
                "h2_2_abs_gap": float(gap.mean()),
                "u2": float(h2_sum[2] / n),
                "u13": float(h2_sum[13] / n),
                "u14": float(h2_sum[14] / n),
                "u25": float(h2_sum[25] / n),
                "u26": float(h2_sum[26] / n),
                "u30": float(h2_sum[30] / n),
            }
        )

    val = load(root, "validation", 3, 4)
    ks = kings(val)
    for label, target, donor in [
        ("matched_B_hybrid_Q13", 3, 5),
        ("matched_Q_hybrid_B13", 5, 3),
    ]:
        nat_all, pat_all, mag_all, c2_all, gap_all = [], [], [], [], []
        h2_sum = np.zeros(32)
        for chunk in batches(val):
            rt, pt, _ = net.encode(chunk, target, 4)
            tt = net.forward(rt, pt)
            rd, pd, _ = net.encode(chunk, donor, 4)
            td = net.forward(rd, pd)
            h = tt["h1"].copy()
            h[:, :, [1, 3]] = td["h1"][:, :, [1, 3]]
            th = replace_h1(net, tt, h)
            nat_all.append(credit(delta(th["score"])))
            pat_all.append(drop_h2(net, th, 2))
            mag_all.append(np.abs(delta(th["score"])))
            c2_all.append(delta(th["h2"])[:, 2] * net.out[2] / 16)
            gap_all.append(np.abs(delta(th["h2"])[:, 2]))
            h2_sum += (delta(th["h2"]) * net.out / 16).sum(0)
        nat = np.concatenate(nat_all)
        pat = np.concatenate(pat_all)
        mag = np.concatenate(mag_all)
        c2 = np.concatenate(c2_all)
        gap = np.concatenate(gap_all)
        dif = nat - pat
        lo, hi = ci(ks, dif)
        n = len(val)
        rows.append(
            {
                "cohort": label,
                "white": target,
                "n": n,
                "king_orbits": len(set(ks)),
                "native_or_hybrid": float(nat.mean()),
                "without_h2_2": float(pat.mean()),
                "loss": float(dif.mean()),
                "lower": float(lo),
                "upper": float(hi),
                "median_abs_margin": float(np.median(mag)),
                "mean_c2_score": float(c2.mean()),
                "h2_2_abs_gap": float(gap.mean()),
                "u2": float(h2_sum[2] / n),
                "u13": float(h2_sum[13] / n),
                "u14": float(h2_sum[14] / n),
                "u25": float(h2_sum[25] / n),
                "u26": float(h2_sum[26] / n),
                "u30": float(h2_sum[30] / n),
            }
        )
    write(out / "skewer-specificity.csv", rows)


def fork_controls(root, out, net):
    """Compare discovery-ranked fork FT channels against activity-matched controls and non-forks."""
    with (out / "tactical-inputs.csv").open() as f:
        ids = [int(r["ft"]) for r in csv.DictReader(f) if int(r["black"]) == 2]
    disc = load(root, "discovery", 4, 2)
    act = np.zeros(2048)
    for r in batches(disc):
        raw, _, _ = net.encode(r, 4, 2)
        act += np.abs(delta(raw.clip(0, 127))).sum(0)
    used = set(ids)
    act_ctrl = []
    for u in ids:
        pool = np.array(
            [q for q in range(2048) if q // 1024 == u // 1024 and q not in used]
        )
        q = int(pool[np.argmin(np.abs(act[pool] - act[u]))])
        act_ctrl.append(q)
        used.add(q)

    cohorts = [
        ("matched_R_fork", 4, "validation-4-2.npz", True),
        ("outside_R_fork", 4, "transfer-4-2.npz", True),
        ("outside_R_nonfork", 4, "transfer-4-2.npz", False),
        ("outside_Q_fork", 5, "transfer-5-2.npz", True),
    ]
    rows = []
    for name, a, fname, use_mask in cohorts:
        d = np.load(root / "manuscript" / fname)
        r = d["rows"][d["mask"] if use_mask else ~d["mask"]]
        ks = kings(r)
        for ft_set, tids in [("top8", ids), ("activity8", act_ctrl)]:
            for route, ridx in [
                ("all", list(range(8))),
                ("fork014", FORK),
                ("complement", [2, 3, 5, 6, 7]),
            ]:
                nat_all, pat_all, rem_all = [], [], []
                for chunk in batches(r):
                    raw, ps, _ = net.encode(chunk, a, 2)
                    x = raw.clip(0, 127)
                    t = net.forward(raw, ps)
                    check_native(a, 2, chunk, t)
                    dz = delta(x)[:, tids] @ net.w1[:, tids].T
                    change = np.zeros_like(dz)
                    change[:, ridx] = dz[:, ridx]
                    alt = net.upper(t["z"] + np.stack([-change, change], axis=1), ps)
                    left = alt["score"][:, 0] - t["score"][:, 1]
                    right = t["score"][:, 0] - alt["score"][:, 1]
                    nat_all.append(credit(delta(t["score"])))
                    pat_all.append((credit(left) + credit(right)) / 2)
                    rem_all.append(delta(t["score"]) - (left + right) / 2)
                nat = np.concatenate(nat_all)
                pat = np.concatenate(pat_all)
                rem = np.concatenate(rem_all)
                dif = nat - pat
                lo, hi = ci(ks, dif)
                rows.append(
                    {
                        "cohort": name,
                        "white": a,
                        "n": len(r),
                        "king_orbits": len(set(ks)),
                        "ft_set": ft_set,
                        "ft_ids": " ".join(map(str, tids)),
                        "through": route,
                        "native": float(nat.mean()),
                        "patched": float(pat.mean()),
                        "loss": float(dif.mean()),
                        "lower": float(lo),
                        "upper": float(hi),
                        "score_removed": float(rem.mean()),
                    }
                )
    write(out / "fork-controls.csv", rows)


def main(root, out):
    v.init(root)
    net = Network(root)
    skewer_specificity(root, out, net)
    fork_controls(root, out, net)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
