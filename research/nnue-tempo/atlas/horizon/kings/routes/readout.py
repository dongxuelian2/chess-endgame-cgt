"""Test the necessity and sufficiency of every readout unit on fixed chess mechanisms."""

import sys
from pathlib import Path

import numpy as np
from network import Network, credit, retain, trunc
from study import panels
from summarize import write


def compare(left, right, p, scores):
    return (
        credit(trunc(p[:, 0] + left, 16) - scores[:, 1])
        + credit(scores[:, 0] - trunc(p[:, 1] + right, 16))
    ) / 2


def main(root, out):
    net = Network(root)
    report = []
    for panel, pieces, b, rank, flag, rows in panels(
        root / "atlas/horizon/kings/routes"
    ):
        if not (
            (flag == "escort" and panel == "4-1")
            or (panel in ("3-4", "4-4") and flag == "check_from_far")
            or (panel == "4-2" and flag == "check_and_attack_man")
        ) or not len(rows):
            continue
        for a in pieces:
            total = np.zeros((40, 4))
            for start in range(0, len(rows), 192):
                raw, p, _ = net.encode(rows[start : start + 192], a, b)
                t = net.forward(raw, p)
                h2 = t["h2"]
                diff = (h2[:, 0] - h2[:, 1]) * net.out
                scores = t["score"]
                PS = t["pos"]
                native = credit(scores[:, 0] - scores[:, 1])
                for i in range(8):
                    total[i] += np.array(
                        [
                            native.sum(),
                            retain(net, t, [i]).sum(),
                            retain(net, t, [j for j in range(8) if i != j]).sum(),
                            len(t["ps"]),
                        ]
                    )
                for i, change in enumerate(diff.T):
                    only = compare(PS[:, 1] + change, PS[:, 0] - change, p, scores)
                    without = compare(PS[:, 0] - change, PS[:, 1] + change, p, scores)
                    total[8 + i] += [
                        native.sum(),
                        only.sum(),
                        without.sum(),
                        len(t["ps"]),
                    ]
            for i, (n, x, y, d) in enumerate(total):
                report.append(
                    {
                        "panel": panel,
                        "rule": flag,
                        "white": a,
                        "n": len(rows),
                        "layer": "H1" if i < 8 else "H2",
                        "unit": i if i < 8 else i - 8,
                        "native": n / d,
                        "only": x / d,
                        "without": y / d,
                        "removal_loss": (n - y) / d,
                    }
                )
    write(out / "readout-interchange.csv", report)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
