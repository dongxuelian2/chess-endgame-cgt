"""Assemble compact public results and the manuscript figure from row-level evidence."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze import auc, gap
from collect import EXTENDED, MATERIAL
from move_test import BUDGETS


def summarize(root):
    static = {}
    for f in ("results.json", "results5.json"):
        static.update(json.loads((root / f).read_text()))
    modern = json.loads((root / "modern.json").read_text())
    with gzip.open(root / "games.jsonl.gz") as f:
        games = list(map(json.loads, f))
    cohorts, move = {}, {}
    for m in static:
        rows = [r for r in games if r["material"] == m]
        delta = np.array([sum(w if abs(w) == 2 else 0 for w in r["wdl"]) for r in rows])
        valid = delta >= 0
        labels = delta[valid] > 0
        h, n = np.array([r["sf141"] for r in rows])[valid].sum(1).T
        entry = {
            "n": len(rows),
            "pos": int(labels.sum()),
            "negative": int((~valid).sum()),
        }
        if 0 < labels.sum() < len(labels):
            entry.update(
                h=auc(labels, h),
                nnet=auc(labels, n),
                gap=gap(labels, n, h, np.arange(len(h))),
            )
        cohorts[m] = entry
    for k in BUDGETS:
        node_counts = []
        residuals = defaultdict(float)
        by_material = {}
        for m in MATERIAL:
            with gzip.open(root / (m + ".moves.gz")) as f:
                rows = list(map(json.loads, f))
            correct, groups = [], []
            for fen, best, record, group in rows:
                decisions = record[str(k)]
                node_counts.append([x[2] for x in decisions])
                correct.append([int(x[1] == best) for x in decisions])
                groups.append(tuple(group))
            y = np.array(correct)
            d = y[:, 1] - y[:, 0]
            by_material[m] = [
                int(y[:, 0].sum()),
                int(y[:, 1].sum()),
                int((d > 0).sum()),
                int((d < 0).sum()),
            ]
            for g, v in zip(groups, d - d.mean()):
                residuals[g] += v
        n = len(node_counts)
        total = np.sum(list(by_material.values()), axis=0)
        g = len(residuals)
        se = np.sqrt(sum(x * x for x in residuals.values()) * g / (g - 1)) / n
        diff = (total[1] - total[0]) / n
        move[k] = {
            "n": n,
            "by_material": by_material,
            "total": total.tolist(),
            "difference": diff,
            "ci": [diff - 1.96 * se, diff + 1.96 * se],
            "node_quantiles": np.quantile(
                node_counts, [0, 0.5, 0.99, 1], axis=0
            ).tolist(),
        }
    with gzip.open(root / "kpk-negative.json.gz") as f:
        census = json.load(f)
    return {
        "static": static,
        "games": cohorts,
        "modern": modern,
        "move": move,
        "pawn_edits": json.loads((root / "pawn-edits.json").read_text()),
        "replication": json.loads((root / "replication.json").read_text()),
        "surrogate": json.loads((root / "surrogate.json").read_text()),
        "kpk": {
            "n": census["eligible"],
            "matrix": census["confusion"],
            "sf18": json.loads((root / "census18.json").read_text()),
        },
    }


def figure(data, path):
    mats = MATERIAL + EXTENDED
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for r, (field, label, color) in enumerate(
        [
            ("static", "Sampled placements", "#125d72"),
            ("games", "Game placements", "#bb591e"),
        ]
    ):
        for j, m in enumerate(mats):
            v = data[field][m]
            if field == "static":
                v = v["held"]
            if "gap" not in v:
                continue
            mid, lo, hi = v["gap"]
            y = len(mats) - 1 - j + (0.13 if r == 0 else -0.13)
            ax.plot([lo, hi], [y, y], color=color, lw=1.3)
            ax.scatter(
                mid, y, color=color, s=16, label=label if j == 0 else None, zorder=3
            )
    ax.axvline(0, color="#666666", lw=0.8)
    ax.set_yticks(range(len(mats)), mats[::-1], fontsize=8)
    ax.set_xlabel("AUC difference, Stockfish 14.1 NNUE minus classical", fontsize=9)
    ax.tick_params(length=0, labelsize=8)
    ax.grid(axis="x", alpha=0.16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=215)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    data = summarize(args.scratch)
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "results.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    figure(data, args.destination / "tempo-ranking.png")
