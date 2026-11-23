"""Stratify promotion interventions by straight-ray coverage, material, and game sampling."""

import gzip
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network, credit
from promotion_geometry import relation
from promotion_trial import restrict
from study import check_native
from summarize import write

from cohorts import kings
from effects import batches, ci, delta, patch
from perspectives import channels, played_groups
from promotion_errors import labels


def all_played_alternatives(root):
    """Collect every legal quiet classically tied alternative at played parents."""
    groups = defaultdict(list)
    for archive in ("atlas/played.gz", "atlas/horizon/played.gz"):
        with gzip.open(root / archive, "rt") as stream:
            for _, gid, fen, m_uci, *_ in map(json.loads, stream):
                q = chess.Board(fen)
                w, k = q.king(True), q.king(False)
                s = chess.parse_square(m_uci[:2])
                a = q.piece_type_at(s)
                pawn = q.pieces(chess.PAWN, False)
                if len(pawn) != 1 or a not in (2, 3, 4, 5):
                    continue
                y = next(iter(pawn))
                if y // 8 != 1:
                    continue
                f = (y ^ (7 if w % 8 < 4 else 0)) % 8
                played_dest = chess.parse_square(m_uci[2:4])
                h0 = v.state(a, 1, w, k, y, played_dest)
                for mv in q.generate_legal_moves(from_mask=chess.BB_SQUARES[s]):
                    if (
                        mv.to_square == played_dest
                        or q.is_capture(mv)
                        or q.gives_check(mv)
                        or mv.promotion
                    ):
                        continue
                    q.push(mv)
                    ok = any(q.legal_moves) and not q.has_legal_en_passant()
                    q.pop()
                    if not ok:
                        continue
                    l0 = v.state(a, 1, w, k, y, mv.to_square)
                    if h0["f"] != l0["f"] or np.sign(h0["w"]) == np.sign(l0["w"]):
                        continue
                    if h0["w"] > l0["w"]:
                        row = [w, k, y, played_dest, mv.to_square, s]
                        played_h = 1.0
                    else:
                        row = [w, k, y, mv.to_square, played_dest, s]
                        played_h = 0.0
                    if restrict((a, *row)):
                        groups[a, f].append((row, gid, played_h))
    return groups


def single_played_pairs(root):
    """Collect the single sampled played alternative per game from both 2013 panels."""
    played_move = {}
    for archive in ("atlas/played.gz", "atlas/horizon/played.gz"):
        with gzip.open(root / archive, "rt") as stream:
            for _, gid, fen, m_uci, *_ in map(json.loads, stream):
                played_move[gid, fen] = chess.parse_square(m_uci[2:4])
    groups = defaultdict(list)
    for _, panel in played_groups(root):
        for (a, f), items in panel.items():
            for r, gid in items:
                q = v.board(a, 1, *r[:3], r[5])
                q.turn = True
                dest = played_move.get((gid, q.fen()))
                played_h = 1.0 if dest == r[3] else (0.0 if dest == r[4] else np.nan)
                groups[a, f].append((r, gid, played_h))
    return groups


def regime_tag(gh, gl):
    """Classify the White destination pair by straight-ray promotion coverage."""
    if gh == 4 and gl == 2:
        return "H_clr_L_off"
    if gh == 4 and gl == 4:
        return "H_clr_L_clr"
    if gh == 4 and gl == 0:
        return "H_clr_L_occ"
    return "other"


def record_rows(net, ranked, pool, groups, sample_name, cluster_mode):
    """Compute regime-stratified accuracy and four-channel ablation statistics."""
    buckets = defaultdict(list)
    gid_map = {}
    for a, f, r, gids, played_hs in groups:
        if not len(r):
            continue
        act_flags = np.array(
            list(pool.map(labels, [(a, row) for row in r.tolist()], chunksize=80)),
            bool,
        )
        w, k, y, h, l, _ = r.T
        gh = relation(a, h, y - 8, w, k)
        gl = relation(a, l, y - 8, w, k)
        nat_list, pat_list, dm_list, mar_list = [], [], [], []
        ft1_h_list, ft1_l_list, dft1_list, dft4_list = [], [], [], []
        for batch in batches(r):
            raw, p, _ = net.encode(batch, a, 1)
            x = raw.clip(0, 127)
            t = net.forward(raw, p)
            check_native(a, 1, batch, t)
            pt, ef = patch(net, x, t, ranked[f][:4])
            m = delta(t["score"])
            nat_list.append(credit(m))
            pat_list.append(pt)
            dm_list.append(ef)
            mar_list.append(np.abs(m))
            ft1_h_list.append(x[:, 0, ranked[f][0]])
            ft1_l_list.append(x[:, 1, ranked[f][0]])
            dft1_list.append(delta(x)[:, ranked[f][0]])
            dft4_list.append(delta(x)[:, ranked[f][:4]].mean(1))
        nat = np.concatenate(nat_list)
        pat = np.concatenate(pat_list)
        dm = np.concatenate(dm_list)
        mar = np.concatenate(mar_list)
        ft1_h = np.concatenate(ft1_h_list)
        ft1_l = np.concatenate(ft1_l_list)
        dft1 = np.concatenate(dft1_list)
        dft4 = np.concatenate(dft4_list)
        if cluster_mode == "kings":
            cl = kings(r)
        else:
            cl = np.array([gid_map.setdefault(g, len(gid_map)) for g in gids])
        fg = "bgh" if f in (1, 6, 7) else "acdef"
        cap, push, _ = act_flags.T
        for i in range(len(r)):
            if push[i] and not cap[i]:
                action = "straight_only"
            elif cap[i] and not push[i]:
                action = "capture_only"
            else:
                continue
            reg = regime_tag(int(gh[i]), int(gl[i]))
            row_vals = (
                int(cl[i]),
                float(nat[i]),
                float(pat[i]),
                float(dm[i]),
                float(mar[i]),
                float(ft1_h[i]),
                float(ft1_l[i]),
                float(dft1[i]),
                float(dft4[i]),
                float(played_hs[i]),
                float(a == 5),
            )
            for fgrp in (fg, "all"):
                for reg_key in ("all", reg):
                    for w_key in (0, a):
                        buckets[sample_name, fgrp, action, reg_key, w_key].append(
                            row_vals
                        )
    rows = []
    for (samp, fgrp, action, reg_key, w_key), items in sorted(buckets.items()):
        arr = np.array(items, dtype=float)
        dif = arr[:, 1] - arr[:, 2]
        lo, hi = ci(arr[:, 0].astype(int), dif)
        ph = arr[:, 9]
        ph_mean = float(np.nanmean(ph)) if np.any(~np.isnan(ph)) else None
        rows.append(
            {
                "sample": samp,
                "files": fgrp,
                "action": action,
                "regime": reg_key,
                "white": w_key,
                "n": len(arr),
                "clusters": len(set(arr[:, 0].astype(int))),
                "native": float(arr[:, 1].mean()),
                "patched": float(arr[:, 2].mean()),
                "loss": float(dif.mean()),
                "lower": float(lo),
                "upper": float(hi),
                "mean_dm": float(arr[:, 3].mean()),
                "median_abs_margin": float(np.median(arr[:, 4])),
                "mean_ft1_h": float(arr[:, 5].mean()),
                "mean_ft1_l": float(arr[:, 6].mean()),
                "mean_dft1": float(arr[:, 7].mean()),
                "mean_dft4": float(arr[:, 8].mean()),
                "played_is_h_rate": ph_mean,
                "queen_share": float(arr[:, 10].mean()),
            }
        )
    return rows


def specificity_control(net, ranked):
    """Compare four-channel intervention when first-ply promotion is vs is not necessary."""
    buckets = defaultdict(list)
    for a in range(2, 6):
        d = np.load(v.scratch / "manuscript" / f"validation-{a}-1.npz")
        rows_all, mask_all = d["rows"], d["mask"]
        for label, flag in [("necessary", mask_all), ("not_necessary", ~mask_all)]:
            sub = rows_all[flag]
            for f in range(8):
                r = sub[(sub[:, 2] ^ 7) == 8 + f]
                if not len(r):
                    continue
                fg = "bgh" if f in (1, 6, 7) else "acdef"
                for batch in batches(r):
                    raw, p, _ = net.encode(batch, a, 1)
                    t = net.forward(raw, p)
                    check_native(a, 1, batch, t)
                    pt, ef = patch(net, raw.clip(0, 127), t, ranked[f][:4])
                    nat = credit(delta(t["score"]))
                    ks = kings(batch)
                    packed = np.c_[ks, nat, pt, ef]
                    for fgrp in (fg, "all"):
                        buckets[fgrp, label].append(packed)
    out_rows = []
    for (fgrp, label), chunks in sorted(buckets.items()):
        arr = np.concatenate(chunks)
        dif = arr[:, 1] - arr[:, 2]
        lo, hi = ci(arr[:, 0].astype(int), dif)
        out_rows.append(
            {
                "files": fgrp,
                "promotion_status": label,
                "n": len(arr),
                "king_orbits": len(set(arr[:, 0].astype(int))),
                "native": float(arr[:, 1].mean()),
                "patched": float(arr[:, 2].mean()),
                "loss": float(dif.mean()),
                "lower": float(lo),
                "upper": float(hi),
                "mean_dm": float(arr[:, 3].mean()),
            }
        )
    return out_rows


def main(root, out):
    v.init(root)
    net = Network(root)
    ranked = channels(out)

    enum_groups = []
    for a in range(2, 6):
        d = np.load(root / "manuscript" / f"validation-{a}-1.npz")
        sel = d["rows"][d["mask"]]
        for f in range(8):
            r = sel[(sel[:, 2] ^ 7) == 8 + f]
            enum_groups.append((a, f, r, [None] * len(r), np.full(len(r), np.nan)))

    single = single_played_pairs(root)
    single_groups = [
        (
            a,
            f,
            np.array([x[0] for x in items]),
            [x[1] for x in items],
            np.array([x[2] for x in items]),
        )
        for (a, f), items in sorted(single.items())
    ]

    all_alts = all_played_alternatives(root)
    all_groups = [
        (
            a,
            f,
            np.array([x[0] for x in items]),
            [x[1] for x in items],
            np.array([x[2] for x in items]),
        )
        for (a, f), items in sorted(all_alts.items())
    ]

    with ProcessPoolExecutor(
        8, mp_context=get_context("fork"), initializer=v.init, initargs=(root,)
    ) as pool:
        rows = []
        rows.extend(record_rows(net, ranked, pool, enum_groups, "enumerated", "kings"))
        rows.extend(
            record_rows(net, ranked, pool, single_groups, "played_single", "games")
        )
        rows.extend(
            record_rows(net, ranked, pool, all_groups, "played_all_alts", "games")
        )
    write(out / "capture-regimes.csv", rows)
    write(out / "promotion-specificity.csv", specificity_control(net, ranked))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
