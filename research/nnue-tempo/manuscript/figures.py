"""Publication figures from frozen paired results and native validation traces."""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from network import Network

from effects import batches
from tactical import load

TEAL = "#19586a"
GOLD = "#b76828"
DARK = "#263740"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "svg.hashsalt": "chess-nnue",
    }
)


def save(fig, path):
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def draw(root, out):
    fig, ax = plt.subplots(figsize=(7.1, 2.45))
    with (out / "results/promotion.csv").open() as f:
        rows = list(csv.DictReader(f))
    for name, shift, color, marker, title in [
        ("activity_four", -0.19, "#86898a", "s", "Four activity-matched controls"),
        ("first", 0, GOLD, "o", "Leading channel"),
        ("first_four", 0.19, TEAL, "D", "Four leading channels"),
    ]:
        ss = sorted(
            (r for r in rows if r["intervention"] == name), key=lambda r: r["file"]
        )
        val, lo, hi = np.array(
            [[float(r[k]) * 100 for k in ("drop", "lower", "upper")] for r in ss]
        ).T
        ax.errorbar(
            np.arange(8) + shift,
            val,
            yerr=np.array([val - lo, hi - val]),
            fmt=marker,
            color=color,
            ms=4,
            capsize=2,
            lw=1,
            label=title,
        )
    ax.set_xticks(range(8), list("abcdefgh"))
    ax.set_xlabel("Pawn file in the white king’s orientation")
    ax.set_ylabel("Ordering points lost (pp)")
    ax.axhline(0, c="gray", lw=0.6)
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.20),
    )
    save(fig, out / "promotion.svg")
    net = Network(root)
    r = load(root, "validation", 3, 4)
    fig, (ax, bar) = plt.subplots(
        1, 2, figsize=(7.15, 2.25), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    for a, c, l in [(3, TEAL, "Bishop"), (5, GOLD, "Queen")]:
        values = []
        for chunk in batches(r):
            raw, p, _ = net.encode(chunk, a, 4)
            values.extend(net.forward(raw, p)["z2"][:, :, 2].ravel() / 64)
        ax.hist(
            values,
            bins=np.arange(-132, 205, 4),
            weights=np.full(len(values), 100 / len(values)),
            histtype="step",
            color=c,
            lw=1.5,
            label=l,
        )
    ax.axvline(0, ls="--", lw=0.7, color=DARK)
    ax.axvline(127, ls="--", lw=0.7, color=DARK)
    ax.set_xlabel("H2 unit 2 before clipping (byte scale)")
    ax.set_ylabel("Endpoints per bin (%)")
    ax.legend(frameon=False, fontsize=8)
    with (out / "results/verdicts.csv").open() as f:
        verdict = [r for r in csv.DictReader(f) if r["sample"] == "validation"]
    bottom = np.zeros(2)
    for field, c, l in [
        ("zero", "#d6e1e4", "Zero signal"),
        ("nonzero_both_orders_preserved", "#7297a3", "Signal, order stable"),
        ("some_verdict_changes", GOLD, "Order changed"),
    ]:
        v = np.array([100 * int(r[field]) / int(r["n"]) for r in verdict])
        bar.bar([0, 1], v, bottom=bottom, color=c, width=0.58, label=l)
        bottom += v
    bar.set_xticks([0, 1], ["Bishop", "Queen"])
    bar.set_ylabel("Pairs (%)")
    bar.legend(
        fontsize=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.42),
        ncol=1,
    )
    fig.subplots_adjust(wspace=0.45)
    save(fig, out / "skewer.svg")

    with (out / "results/capture-regimes.csv").open() as f:
        crows = {
            (r["sample"], r["files"], r["action"], r["regime"], r["white"]): r
            for r in csv.DictReader(f)
        }
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    regimes = [
        ("straight_only", "all", "Straight\n(all)"),
        ("capture_only", "H_clr_L_off", "Capture\nclr→off"),
        ("capture_only", "H_clr_L_clr", "Capture\nclr→clr"),
        ("capture_only", "H_clr_L_occ", "Capture\nclr→occ"),
    ]
    cohorts = [
        ("enumerated", "bgh", TEAL, "Enum. b,g,h"),
        ("enumerated", "acdef", "#5e8f9e", "Enum. a,c–f"),
        ("played_all_alts", "all", GOLD, "Played (all alts)"),
    ]
    x = np.arange(len(regimes))
    w = 0.24
    for idx, (samp, fgrp, col, label) in enumerate(cohorts):
        vals = [
            float(crows[samp, fgrp, act, reg, "0"]["mean_dm"])
            for act, reg, _ in regimes
        ]
        ax1.bar(x + (idx - 1) * w, vals, width=w, color=col, label=label)
    ax1.axhline(0, color=DARK, lw=0.7)
    ax1.set_xticks(x, [lbl for _, _, lbl in regimes], fontsize=8)
    ax1.set_ylabel("Removed score ΔM (units)")
    ax1.legend(
        frameon=False,
        fontsize=7.5,
        loc="upper right",
        ncol=1,
    )

    comp_cols = [
        ("enumerated", "bgh", "Enum.\nb,g,h"),
        ("enumerated", "acdef", "Enum.\na,c–f"),
        ("played_single", "bgh", "Played 1\nb,g,h"),
        ("played_all_alts", "bgh", "Played all\nb,g,h"),
    ]
    q_off, nonq_off, clr_occ, oth = [], [], [], []
    for samp, fgrp, _ in comp_cols:
        tot = float(crows[samp, fgrp, "capture_only", "all", "0"]["n"])
        row_off = crows.get((samp, fgrp, "capture_only", "H_clr_L_off", "0"))
        row_q_off = crows.get((samp, fgrp, "capture_only", "H_clr_L_off", "5"))
        n_off = float(row_off["n"]) if row_off else 0.0
        n_q_off = float(row_q_off["n"]) if row_q_off else 0.0
        row_cc = crows.get((samp, fgrp, "capture_only", "H_clr_L_clr", "0"))
        row_co = crows.get((samp, fgrp, "capture_only", "H_clr_L_occ", "0"))
        n_cc = float(row_cc["n"]) if row_cc else 0.0
        n_co = float(row_co["n"]) if row_co else 0.0
        row_oth = crows.get((samp, fgrp, "capture_only", "other", "0"))
        n_oth = float(row_oth["n"]) if row_oth else 0.0
        q_off.append(100 * n_q_off / tot)
        nonq_off.append(100 * (n_off - n_q_off) / tot)
        clr_occ.append(100 * (n_cc + n_co) / tot)
        oth.append(100 * n_oth / tot)
    cx = np.arange(len(comp_cols))
    bot = np.zeros(len(comp_cols))
    for arr, col, lbl in [
        (q_off, GOLD, "Q: clr→off"),
        (nonq_off, "#e4bb7d", "R/B/N: clr→off"),
        (clr_occ, TEAL, "clr→clr / occ"),
        (oth, "#9cb3b9", "Other"),
    ]:
        arr = np.array(arr)
        ax2.bar(cx, arr, bottom=bot, width=0.56, color=col, label=lbl)
        bot += arr
    ax2.set_xticks(cx, [lbl for _, _, lbl in comp_cols], fontsize=7.5)
    ax2.set_ylabel("Capture pairs (%)")
    ax2.legend(
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
    )
    fig.subplots_adjust(wspace=0.34)
    save(fig, out / "capture.svg")

    with (out / "results/fork-colocalization.csv").open() as f:
        fcol = list(csv.DictReader(f))
    all_corr = [float(r["mean_corr"]) for r in fcol]
    top_corr = sorted(
        [r for r in fcol if r["is_fork_top8"] == "1"],
        key=lambda r: float(r["mean_corr"]),
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.05, 1.25]}
    )
    ax1.hist(
        all_corr,
        bins=np.linspace(-0.85, 0.85, 35),
        color="#c6d6da",
        edgecolor=DARK,
        linewidth=0.4,
    )
    for r in top_corr:
        c = float(r["mean_corr"])
        ax1.plot([c, c], [0, 86], color=GOLD if int(r["ft"]) != 1760 else "#86898a", lw=1.3)
    ax1.axvline(np.median(all_corr), color=DARK, ls="--", lw=0.9)
    ax1.set_ylim(0, 130)
    ax1.text(
        -0.06,
        123,
        f"Median\n{np.median(all_corr):+.3f}",
        ha="right",
        va="top",
        fontsize=7.2,
        color=DARK,
    )
    ax1.annotate(
        "6 of 7 White-view\nfork FTs (96–99.8%)",
        xy=(0.56, 86),
        xytext=(0.50, 113),
        ha="center",
        va="center",
        fontsize=7.2,
        color=GOLD,
        arrowprops={"arrowstyle": "->", "color": GOLD, "lw": 0.9},
    )
    ax1.set_xlabel("Mean check-square correlation (30 buckets)")
    ax1.set_ylabel("White-view FT channels")

    with (out / "results/fork-controls.csv").open() as f:
        fctrl = {
            (r["cohort"], r["ft_set"], r["through"]): r for r in csv.DictReader(f)
        }
    f_cohorts = [
        ("matched_R_fork", "Matched\nforks (R)"),
        ("outside_R_fork", "Outside\nforks (R)"),
        ("outside_R_nonfork", "Non-forks\n(R, 34k)"),
        ("outside_Q_fork", "Outside\nforks (Q)"),
    ]
    fx = np.arange(len(f_cohorts))
    fw = 0.34
    for idx, (fset, col, lbl) in enumerate(
        [("top8", TEAL, "Top-8 FT via H1 0/1/4"), ("activity8", "#86898a", "Activity-matched 8 via H1 0/1/4")]
    ):
        vals, lo_e, hi_e = [], [], []
        for cname, _ in f_cohorts:
            row = fctrl[cname, fset, "fork014"]
            v = float(row["loss"]) * 100
            vals.append(v)
            lo_e.append(v - float(row["lower"]) * 100)
            hi_e.append(float(row["upper"]) * 100 - v)
        ax2.bar(
            fx + (idx - 0.5) * fw,
            vals,
            yerr=[lo_e, hi_e],
            width=fw,
            color=col,
            capsize=2,
            error_kw={"lw": 0.8},
            label=lbl,
        )
    ax2.axhline(0, color=DARK, lw=0.6)
    ax2.set_xticks(fx, [lbl for _, lbl in f_cohorts], fontsize=7.5)
    ax2.set_ylabel("Ordering loss via H1 0/1/4 (pp)")
    ax2.legend(
        frameon=False,
        fontsize=7.3,
        loc="upper right",
    )
    fig.subplots_adjust(wspace=0.35)
    save(fig, out / "fork.svg")


if __name__ == "__main__":
    draw(Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent)
