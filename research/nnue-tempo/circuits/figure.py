"""Plot the evaluator controls and the feature-interaction intervention."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(directory):
    def rows(filename):
        with (directory / filename).open() as f:
            return list(csv.DictReader(f))

    data = rows("handwritten-controls.csv")
    reversal = {r["cohort"]: r for r in rows("reversals.csv") if r["stage"] == "all"}
    cohorts = ["2013-02", "2013-03", "wdl"]
    names = ["February threats", "March threats", "Syzygy outcomes"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), sharey=True)
    colors = ["#264d75", "#ba7236", "#619982"]

    def bars(ax, values, names, colors):
        y = np.arange(3)
        height = 0.18
        for i, (v, name, color) in enumerate(zip(values, names, colors)):
            p = y + (i - (len(values) - 1) / 2) * height
            ax.barh(p, v, height=0.15, label=name, color=color)
            for x, at in zip(v, p):
                ax.text(x + 0.45, at, f"{x:.1f}", va="center", fontsize=8)
        ax.set_xlim(0, 46)
        ax.set_xlabel("Both choices strictly correct (%)")
        ax.xaxis.grid(True, color="#eeeeee")
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), frameon=False, fontsize=9)

    selections = [
        ("stock", "native_nnue"),
        ("stock", "native_classical"),
        ("disabled", "native_classical"),
    ]
    values = [
        [
            100
            * float(
                next(
                    r["both_strict"]
                    for r in data
                    if r["cohort"] == c and (r["cutoffs"], r["scoring"]) == pair
                )
            )
            for c in cohorts
        ]
        for pair in selections
    ]
    bars(
        axes[0],
        values,
        ["Released NNUE", "Released classical", "Classical with both cutoffs disabled"],
        colors,
    )
    keys = ["intact_unrounded_both", "equalized_both", "restore_256_both"]
    bars(
        axes[1],
        [[100 * float(reversal[c][k]) for c in cohorts] for k in keys],
        [
            "Intact internal score",
            "Equalized feature increments",
            "256 January-ranked slots restored",
        ],
        colors,
    )
    axes[0].set_title(
        "Released evaluators and classical cutoffs", loc="left", fontsize=11
    )
    axes[1].set_title(
        "Interventions on the NNUE feature transform", loc="left", fontsize=11
    )
    axes[0].set_yticks(np.arange(3), names)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.15, right=0.985, top=0.90, bottom=0.32, wspace=0.14)
    matplotlib.rcParams["svg.hashsalt"] = "nnue-threat-reversal"
    matplotlib.rcParams["svg.fonttype"] = "none"
    fig.savefig(directory / "reversal-controls.svg", metadata={"Date": None})
    plt.close(fig)
    path = directory / "reversal-controls.svg"
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    draw(p.parse_args().directory)
