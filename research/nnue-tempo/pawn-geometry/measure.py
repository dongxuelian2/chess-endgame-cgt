import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from transfer import MATERIALS, STRATA


def metrics(rows):
    labels = np.array([r["labels"] for r in rows])
    delta = lambda v: np.stack([v[:, 0] - v[:, 1], v[:, 2] - v[:, 3]], axis=1)
    w = delta(labels)
    mixed = w[:, 0] - w[:, 1]
    direct = (w[:, 0] > 0) & (w[:, 1] < 0)
    reverse = (w[:, 0] < 0) & (w[:, 1] > 0)
    reverses = direct | reverse
    out = {
        "n": len(rows),
        "wdl_positive": int(sum(mixed > 0)),
        "wdl_negative": int(sum(mixed < 0)),
        "wdl_tie": int(sum(mixed == 0)),
        "forward_reversals": int(sum(direct)),
        "opposite_reversals": int(sum(reverse)),
    }
    valid = np.array([None not in r["restricted"] for r in rows])
    cw = delta(
        np.array([[0 if v is None else v for v in r["restricted"]] for r in rows])
    )
    out["forward_evaluable"] = int(sum(direct & valid))
    out["forward_needing_action"] = int(
        sum(direct & valid & ~((cw[:, 0] > 0) & (cw[:, 1] < 0)))
    )
    out["restrictions_unknown"] = int(sum(~valid))
    for i, k in enumerate(("classical", "nnue")):
        scores = -np.array(
            [[x[i] for x in r["sf141"] + r["edited_sf141"]] for r in rows]
        )
        d = delta(scores)
        dm = d[:, 0] - d[:, 1]
        out[k + "_positive"] = int(sum(dm > 0))
        out[k + "_negative"] = int(sum(dm < 0))
        out[k + "_tie"] = int(sum(dm == 0))
        correct = np.all(w * d > 0, axis=1)
        out[k + "_both_forward"] = int(sum(direct & correct))
        out[k + "_both_on_reversals"] = int(sum(reverses & correct))
        out[k + "_both_zero_margins"] = int(sum(np.all(d == 0, axis=1)))
    return out


def load(root, m):
    with gzip.open(root / "pawn-geometry" / f"{m}.gz") as f:
        return list(map(json.loads, f))


def table(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(root, dest):
    result = []
    for m in MATERIALS:
        rows = load(root, m)
        for g in STRATA:
            group = [r for r in rows if r["stratum"] == g]
            if group:
                result.append(dict(material=m, design=g, **metrics(group)))
    table(dest / "transfer.csv", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    main(args.root, args.dest)
