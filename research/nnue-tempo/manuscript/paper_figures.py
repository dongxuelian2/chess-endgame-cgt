"""Generate the statistical figures from the archived results and pinned NNUE trace."""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from effects import batches
from network import Network
from tactical import load

NAVY = "#24587E"
TEAL = "#17776F"
RUST = "#AD502F"
INK = "#192E3A"
GRAY = "#657884"
PALE = "#DDE6EB"
GRID = "#E6ECEF"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.4,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "svg.fonttype": "none",
        "svg.hashsalt": "nnue-manuscript-figures",
        "axes.axisbelow": True,
    }
)


def records(folder, name):
    with (folder / (name + ".csv")).open() as source:
        return list(csv.DictReader(source))


def select(rows, **where):
    matches = [r for r in rows if all(r[key] == str(val) for key, val in where.items())]
    assert len(matches) == 1, (where, len(matches))
    return matches[0]


def interval(ax, points, pos, color, marker="o", orientation="x", **style):
    center, lo, hi = np.asarray(points).T
    assert np.all(lo <= center) and np.all(center <= hi)
    err = np.array([center - lo, hi - center])
    xy = (center, pos) if orientation == "x" else (pos, center)
    uncertainty = {orientation + "err": err}
    ax.errorbar(
        *xy,
        **uncertainty,
        fmt=marker,
        color=color,
        capsize=2.2,
        markersize=4.4,
        elinewidth=1.25,
        **style,
    )


def panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", pad=9)


def baseline(ax, orientation="x"):
    if orientation == "x":
        ax.axvline(0, color=GRAY, lw=0.75)
    else:
        ax.axhline(0, color=GRAY, lw=0.75)
    ax.grid(axis=orientation, color=GRID, linewidth=0.6)


def finish(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.055, metadata={"Date": None})
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def promotion_inputs(out):
    routes = out.parent / "atlas/horizon/kings/routes/results"
    selection = records(routes, "promotion-coverage")
    trials = records(out / "results", "promotion")
    fig, (ax, bx) = plt.subplots(
        1,
        2,
        figsize=(7.02, 2.43),
        layout="constrained",
        gridspec_kw={"width_ratios": [1, 1.33]},
    )
    panel(ax, "a", "FT1424 and the promotion line")
    labels = ["Off the line", "Clear line", "King blocks line"]
    classes = ["off_ray", "clear_ray", "king_blocks_ray"]
    vals = [
        select(selection, white=4, ft=1424, oriented_pawn=10, coverage=k)
        for k in classes
    ]
    byte = [float(r["mean_byte"]) for r in vals]
    ax.barh(labels, byte, height=0.49, color=[NAVY, TEAL, RUST])
    ax.set(xlim=(0, 143), xlabel="Mean FT1424 activation (byte)")
    ax.set_xticks([0, 32, 64, 96, 127])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.6)
    for idx, x in enumerate(byte):
        ax.text(x + 2, idx, f"{x:.1f}", va="center", fontsize=7.8)
    panel(bx, "b", "Promotion-channel interventions")
    baseline(bx, "y")
    for name, shift, color, marker, legend in [
        ("activity_four", -0.22, GRAY, "s", "Four matched controls"),
        ("first", 0, RUST, "^", "Leading channel"),
        ("first_four", 0.22, NAVY, "o", "Four leading channels"),
    ]:
        rows = [select(trials, file=f, intervention=name) for f in "abcdefgh"]
        points = [[100 * float(r[k]) for k in ("drop", "lower", "upper")] for r in rows]
        interval(bx, points, np.arange(8) + shift, color, marker, "y", label=legend)
    bx.set(
        xticks=range(8),
        xticklabels=list("abcdefgh"),
        xlabel="Pawn file in White’s orientation",
        ylabel="Ordering credit lost (pp)",
        ylim=(-2, 43),
    )
    bx.legend(loc="upper left", borderaxespad=0.25, fontsize=7.3)
    finish(fig, out / "paper-promotion.svg")


def capture_regimes(out):
    rows = records(out / "results", "capture-regimes")
    prediction = records(out / "results", "prediction-promotion")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.02, 2.6), layout="constrained")
    panel(ax, "a", "Removing the four channels")
    baseline(ax)
    cases = [
        ("Straight promotion", "straight_only", "all", NAVY),
        ("Capture promotion, pooled", "capture_only", "all", GRAY),
        ("Capture, clear / clear", "capture_only", "H_clr_L_clr", NAVY),
        ("Capture, clear / off-line", "capture_only", "H_clr_L_off", RUST),
    ]
    labs = []
    for i, (title, action, regime, color) in enumerate(cases):
        row = select(
            rows,
            sample="enumerated",
            files="all",
            white=0,
            action=action,
            regime=regime,
        )
        point = [[-100 * float(row[key]) for key in ("loss", "upper", "lower")]]
        interval(ax, point, [i], color, marker="D")
        labs.append(f"{title}\n$n$ = {int(row['n']):,}")
    ax.set(
        yticks=range(len(labs)),
        yticklabels=labs,
        xlabel="Change in ordering credit (pp)",
        xlim=(-29, 18),
    )
    ax.invert_yaxis()

    panel(bx, "b", "Testing the missing capture option")
    cases = [
        ("All validation pairs", "validation", "all"),
        ("Surviving capture favors H", "validation", "safe_capture_favors_H"),
        ("Same, clear / off-line", "validation", "safe_capture_favors_H_H_clear_L_off"),
        ("Played alternatives, Apr–Sep", "games_A", "all"),
        ("Played alternatives, Oct–Dec", "games_B", "all"),
    ]
    bx.grid(axis="x", color=GRID, lw=0.6)
    labels = []
    for i, (title, sample, stratum) in enumerate(cases):
        row = select(
            prediction,
            sample=sample,
            stratum=stratum,
            white=0,
            variant="surviving_capture",
        )
        before, after = [100 * float(row[k]) for k in ("native", "adjusted")]
        bx.plot([before, after], [i, i], color=PALE, linewidth=2.3)
        bx.scatter(
            [before],
            [i],
            color=GRAY,
            s=19,
            marker="s",
            label="Native" if i == 0 else None,
        )
        bx.scatter(
            [after], [i], color=TEAL, s=22, label="Fixed correction" if i == 0 else None
        )
        labels.append(f"{title}\n$n$ = {int(row['n']):,}")
    bx.set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xlabel="Ordering credit (%)",
        xlim=(19, 104),
        ylim=(4.65, -0.62),
    )
    bx.legend(loc="center left", bbox_to_anchor=(0.02, 0.26))
    finish(fig, out / "paper-capture.svg")


def transfer_tradeoff(out):
    folder = out / "results"
    promotion = records(folder, "prediction-promotion")
    added = records(folder, "prediction-five-piece")
    stress = records(folder, "prediction-five-piece-reply-filter")
    cases = [
        (
            "Four-piece validation",
            [
                select(
                    promotion,
                    sample="validation",
                    white=0,
                    stratum="all",
                    variant="surviving_capture",
                )
            ],
            "rescued",
        ),
        ("Added pawn", [r for r in added if r["stratum"] == "all"], "corrected"),
        (
            "Seventh-rank stress",
            [r for r in stress if r["stratum"] == "all" and r["variant"] == "original"],
            "rescued",
        ),
        (
            "Stress with reply filter",
            [
                r
                for r in stress
                if r["stratum"] == "all" and r["variant"] == "filter_white_promotion"
            ],
            "rescued",
        ),
    ]
    fig, ax = plt.subplots(figsize=(6.94, 1.83), layout="constrained")
    baseline(ax)
    labels = []
    for i, (name, rows, key) in enumerate(cases):
        n = sum(int(r["n"]) for r in rows)
        gains, harms = (
            sum(int(r[key]) for r in rows),
            sum(int(r["harmed"]) for r in rows),
        )
        ax.barh(
            i,
            gains / n * 100,
            color=TEAL,
            height=0.48,
            label="Rescued" if i == 0 else None,
        )
        ax.barh(
            i,
            -harms / n * 100,
            color=RUST,
            height=0.48,
            label="Harmed" if i == 0 else None,
        )
        ax.text(gains / n * 100 + 0.5, i, f"{gains:,}", va="center", fontsize=8)
        if harms:
            ax.text(
                -harms / n * 100 - 0.55,
                i,
                f"{harms:,}",
                va="center",
                ha="right",
                fontsize=8,
            )
        labels.append(f"{name}  ($n$ = {n:,})")
    ax.set(
        yticks=range(len(labels)),
        yticklabels=labels,
        ylim=(3.55, -0.55),
        xlim=(-13, 50),
        xticks=[-10, 0, 10, 20, 30, 40, 50],
        xlabel="Share of comparisons (%)",
    )
    ax.set_xticklabels(["10 harmed", "0", "10 rescued", "20", "30", "40", "50"])
    finish(fig, out / "paper-transfer.svg")


def skewer_gate(root, out):
    rows = load(root, "validation", 3, 4)
    net = Network(root)
    verdicts = records(out / "results", "verdicts")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.02, 2.35), layout="constrained")
    panel(ax, "a", "The same readout crosses a clipping floor")
    ax.axvspan(-140, 0, facecolor=PALE, alpha=0.6)
    ax.axvline(0, color=GRAY, ls="--", lw=0.8)
    ax.axvline(127, color=GRAY, ls="--", lw=0.8)
    ax.text(-67, 43, "Clipped to zero", ha="center", fontsize=8)
    for white, color, line, label in [
        (3, NAVY, "-", "Bishop"),
        (5, RUST, "--", "Queen"),
    ]:
        x = []
        for batch in batches(rows):
            raw, ps, _ = net.encode(batch, white, 4)
            x.extend(net.forward(raw, ps)["z2"][:, :, 2].ravel() / 64)
        ax.hist(
            x,
            bins=np.arange(-140, 213, 4),
            weights=np.full(len(x), 100 / len(x)),
            histtype="step",
            color=color,
            linestyle=line,
            linewidth=1.35,
            label=label,
        )
    ax.set(
        xlim=(-140, 210),
        ylim=(0, 52),
        ylabel="Endpoints per bin (%)",
        xlabel="H2 unit 2 before clipping (byte scale)",
    )
    ax.legend(loc="upper right")
    panel(bx, "b", "Equalizing unit 2 on the matched pairs")
    y = np.arange(2)
    offset = np.zeros(2)
    rows = [select(verdicts, sample="validation", white=white) for white in (3, 5)]
    for key, color, label, foreground in [
        ("zero", PALE, "Zero difference", INK),
        ("nonzero_both_orders_preserved", NAVY, "Difference, order stable", "white"),
        ("some_verdict_changes", RUST, "Order changed", "white"),
    ]:
        vals = [100 * int(r[key]) / int(r["n"]) for r in rows]
        bx.barh(y, vals, left=offset, height=0.44, label=label, color=color)
        for i, (left, width) in enumerate(zip(offset, vals)):
            if width > 12:
                bx.text(
                    left + width / 2,
                    i,
                    f"{width:.1f}%",
                    va="center",
                    ha="center",
                    color=foreground,
                    fontsize=7.8,
                )
        offset += vals
    bx.set(
        yticks=y,
        yticklabels=["Bishop", "Queen"],
        xlim=(0, 100),
        ylim=(2.26, -0.53),
        xlabel="Pairs (%)",
    )
    bx.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=1,
        fontsize=7.5,
        labelspacing=0.27,
    )
    finish(fig, out / "paper-skewer.svg")


def readout_controls(out):
    rows = records(out / "results", "prediction-skewer-controls")
    fig, axes = plt.subplots(
        1, 2, figsize=(7.02, 2.0), sharey=True, layout="constrained"
    )
    materials = ["Knight", "Bishop", "Rook", "Queen"]
    for ax, digits, letter, title in zip(
        axes,
        ["all", "abs_native_1_31"],
        "ab",
        ["All eligible margins", "Native absolute margin 1–31"],
    ):
        panel(ax, letter, title)
        baseline(ax)
        for name, offset, color, marker, label in [
            ("direct_capture", -0.2, RUST, "s", "Direct capture"),
            ("skewer", 0, NAVY, "o", "Skewer"),
            ("quiet_neither", 0.2, TEAL, "^", "Quiet control"),
        ]:
            sample = [
                select(rows, white=a, stratum=name, margin_band=digits, unit=2)
                for a in range(2, 6)
            ]
            points = [
                [
                    100 * (float(r["native"]) - float(r["without"])),
                    100 * float(r["lower"]),
                    100 * float(r["upper"]),
                ]
                for r in sample
            ]
            interval(ax, points, np.arange(4) + offset, color, marker, label=label)
        ax.set(
            yticks=range(4),
            yticklabels=materials,
            ylim=(3.55, -0.55),
            xlim=(-7, 40),
            xlabel="Ordering credit lost (pp)",
        )
    axes[0].legend(loc="lower right", ncol=1, fontsize=7.4, borderaxespad=0.15)
    finish(fig, out / "paper-rook-controls.svg")


def knight_fork(out):
    folder = out / "results"
    weights = records(folder, "fork-colocalization")
    effects = records(folder, "fork-controls")
    fig, (ax, bx) = plt.subplots(
        1,
        2,
        figsize=(7.02, 2.3),
        layout="constrained",
        gridspec_kw={"width_ratios": [0.92, 1.25]},
    )
    panel(ax, "a", "Weights align on a common fork square")
    values = sorted(float(r["mean_corr"]) for r in weights)
    active = sorted(float(r["mean_corr"]) for r in weights if r["is_fork_top8"] == "1")
    ax.plot(
        values,
        np.arange(1, len(values) + 1) / len(values) * 100,
        lw=1.5,
        color=NAVY,
    )
    for i, v in enumerate(active):
        ax.plot(
            [v, v],
            [0, 7 + 5 * (i % 2)],
            color=RUST,
            linewidth=1.15,
        )
    ax.axhline(95, linewidth=0.65, color=GRAY, linestyle="--")
    ax.set(
        xlabel="Mean weight correlation across king buckets",
        ylabel="Cumulative share of channels (%)",
        ylim=(0, 103),
    )
    ax.text(-0.71, 83, "All White-view\nchannels", color=NAVY, fontsize=7.3, va="top")
    ax.text(0.10, 17, "Selected inputs", color=RUST, fontsize=7.3, va="center")
    panel(bx, "b", "Which first-layer edges carry the decision?")
    baseline(bx)
    panel_rows = [
        ("Matched rook fork", "matched_R_fork"),
        ("Outside rook fork", "outside_R_fork"),
        ("Outside rook, no fork", "outside_R_nonfork"),
        ("Outside queen fork", "outside_Q_fork"),
    ]
    for key, through, off, color, marker, label in [
        ("top8", "fork014", -0.19, NAVY, "o", "H1 units 0, 1, 4"),
        ("top8", "complement", 0, RUST, "s", "Other H1 units"),
        ("activity8", "fork014", 0.19, GRAY, "^", "Activity-matched controls"),
    ]:
        chosen = [
            select(effects, cohort=cohort, ft_set=key, through=through)
            for _, cohort in panel_rows
        ]
        interval(
            bx,
            [[100 * float(r[c]) for c in ("loss", "lower", "upper")] for r in chosen],
            np.arange(4) + off,
            color,
            marker,
            label=label,
        )
    sizes = [
        int(select(effects, cohort=c, ft_set="top8", through="all")["n"])
        for _, c in panel_rows
    ]
    bx.set(
        yticks=range(4),
        yticklabels=[f"{a}\n$n$ = {n:,}" for (a, _), n in zip(panel_rows, sizes)],
        xlabel="Ordering credit lost (pp)",
        ylim=(3.89, -0.61),
        xlim=(-2, 23),
    )
    bx.legend(loc="lower right", fontsize=7.3, borderaxespad=0.15, labelspacing=0.27)
    finish(fig, out / "paper-fork.svg")


def search_depths(out):
    results = records(out / "results", "expansion-search")
    fig, axes = plt.subplots(
        1, 3, figsize=(7.02, 2.42), sharey=True, layout="constrained"
    )
    for ax, (letter, title, cohort, stratum) in zip(
        axes,
        [
            ("a", "Four pieces, favorable", "4p_enriched", "favorable"),
            ("b", "Five pieces, adverse", "5p_enriched", "adverse"),
            ("c", "Six pieces, adverse", "6p_enriched", "adverse"),
        ],
    ):
        panel(ax, letter, title)
        for release, color, marker, style in [
            ("SF12", NAVY, "o", "--"),
            ("SF14.1", TEAL, "s", "-"),
            ("SF18", RUST, "^", "-."),
        ]:
            rows = [
                select(results, panel=cohort, stratum=stratum, release=release, depth=d)
                for d in (1, 2, 4, 8)
            ]
            rates = [float(rows[0]["credit"]), *(float(r["adjusted"]) for r in rows)]
            ax.plot(
                range(5),
                np.array(rates) * 100,
                marker=marker,
                linestyle=style,
                color=color,
                linewidth=1.2,
                markersize=3.5,
                label=release,
            )
        ax.set(
            xticks=range(5),
            xticklabels=["Static", "1", "2", "4", "8"],
            ylim=(45, 102),
            yticks=[50, 60, 70, 80, 90, 100],
        )
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.text(
            0.98,
            0.045,
            f"$n$ = {rows[0]['n']}",
            transform=ax.transAxes,
            ha="right",
            color=GRAY,
        )
    axes[0].set_ylabel("Ordering credit (%)")
    axes[1].set_xlabel("Search depth (plies)")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 0.09),
        fontsize=7,
        labelspacing=0.32,
        handlelength=1.7,
    )
    finish(fig, out / "paper-search.svg")


def main(root, out):
    promotion_inputs(out)
    capture_regimes(out)
    transfer_tradeoff(out)
    skewer_gate(root, out)
    readout_controls(out)
    knight_fork(out)
    search_depths(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent)
