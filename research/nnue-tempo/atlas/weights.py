"""Freeze the fitted rule weights and single-rule signs on discovery king groups."""

import argparse
from pathlib import Path

import numpy as np

from summarize import bits, fit, read_panels


def main(root):
    panel = read_panels(root)
    ix = panel["fold"] == 0
    X = bits(panel["hh"][ix]) - bits(panel["ll"][ix])
    w = fit(X)
    np.savetxt(
        root / "atlas/weights.txt",
        np.column_stack([np.rint(1e6 * w), np.sign(X.sum(0))]).astype(int),
        fmt="%d",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
