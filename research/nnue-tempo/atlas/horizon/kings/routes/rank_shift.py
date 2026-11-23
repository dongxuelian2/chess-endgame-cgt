"""Hold the men and both kings fixed and exchange only the pawn's rank."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import validate as v
from interventions import primary
from network import Network, relocate
from summarize import write


def metrics(t):
    H = t["h1"]
    return np.c_[
        abs(H[:, 0] - H[:, 1]),
        (H[:, 0] == 0) & (H[:, 1] == 0),
        (H[:, 0] == 127) & (H[:, 1] == 127),
    ]


def main(root, out):
    v.init(root)
    net = Network(root)
    original = primary(root / "atlas/horizon/kings/routes")
    report = []
    columns = []
    for rank in (2, 4, 5, 6, 7):
        before, after = relocate(original, rank)
        for a in (4, 5):
            result = defaultdict(lambda: np.zeros(24))
            weights = np.zeros((8, 2048))
            for start in range(0, len(before), 192):
                r, p, _ = net.encode(before[start : start + 192], a, 1)
                s, _, _ = net.encode(after[start : start + 192], a, 1)
                shift = s - r
                assert np.array_equal(shift[:, 0], shift[:, 1])
                t = net.forward(r, p)
                u = net.forward(s, p)
                delta = s.clip(0, 127) - r.clip(0, 127)
                weights += net.w1 * delta.mean(1).sum(0) / 64
                # Exchange effective midpoint and effective displacement independently at H1.
                midpoint = lambda t: t["z"].mean(1, keepdims=True)
                for label, vv in [
                    ("before", t),
                    ("after", u),
                    ("midpoint_only", net.upper(t["z"] + midpoint(u) - midpoint(t), p)),
                    (
                        "increment_only",
                        net.upper(u["z"] + midpoint(t) - midpoint(u), p),
                    ),
                    (
                        "black_view_only",
                        net.forward(
                            np.concatenate([s[:, :, :1024], r[:, :, 1024:]], axis=2), p
                        ),
                    ),
                    (
                        "white_view_only",
                        net.forward(
                            np.concatenate([r[:, :, :1024], s[:, :, 1024:]], axis=2), p
                        ),
                    ),
                ]:
                    result[label] += metrics(vv).sum(0)
            for label, vals in result.items():
                for i in range(8):
                    gap, floor, ceil = (vals[j + i] / len(before) for j in (0, 8, 16))
                    report.append(
                        {
                            "rank": rank,
                            "piece": a,
                            "n": len(before),
                            "variant": label,
                            "unit": i,
                            "gap": gap,
                            "both_zero": floor,
                            "both_ceiling": ceil,
                        }
                    )
            for i in (2, 4, 6, 7):
                sh = weights[i] / len(before)
                for f in np.argsort(-abs(sh), kind="stable")[:5]:
                    columns.append(
                        {
                            "rank": rank,
                            "piece": a,
                            "unit": i,
                            "ft": f,
                            "midpoint_component": sh[f],
                            "total_shift": sh.sum(),
                        }
                    )
    write(out / "rank-shift.csv", report)
    write(out / "rank-columns.csv", columns)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
