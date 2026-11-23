"""Plot the integer head surface alongside the occupied cells of that same surface."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from compress import RAW, grid
from matplotlib.colors import LogNorm
from validate import DT


def main(root, out):
    table, _ = grid(root)
    densities = []
    for a in (2, 3):
        raw = np.memmap(
            root / f"atlas/horizon/{a}.raw", dtype=RAW, mode="r", shape=(64, 64, 64, 64)
        )
        st = np.memmap(root / f"atlas/{a}-1.bin", dtype=DT, mode="r", shape=raw.shape)
        hist = np.zeros((128, 128), int)
        for w in range(64):
            if w % 8 >= 4:
                continue
            h = raw[w]["h"]
            keep = (st[w]["v"] != 0) & np.all(
                h[:, :, :, :6] == [0, 127, 0, 127, 0, 127], axis=-1
            )
            selected = h[keep]
            np.add.at(hist, (selected[:, 6], selected[:, 7]), 1)
        densities.append(hist)
    fig, axes = plt.subplots(
        1, 3, figsize=(11, 4.25), layout="constrained", sharey=True
    )
    im = axes[0].imshow(
        table.T / 16,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        vmin=790,
        vmax=1870,
    )
    fig.colorbar(
        im,
        ax=axes[0],
        orientation="horizontal",
        shrink=0.85,
        label="Positional output / 16",
    )
    axes[0].set_title("Learned scoring function")
    norm = LogNorm(vmin=1, vmax=max(h.max() for h in densities))
    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("#e9edf1")
    for ax, h, title in zip(axes[1:], densities, ["Knight vs pawn", "Bishop vs pawn"]):
        im = ax.imshow(
            np.ma.masked_equal(h.T, 0),
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_title(title)
    fig.colorbar(
        im,
        ax=axes[1:],
        orientation="horizontal",
        shrink=0.65,
        label="Boards per occupied cell",
    )
    for ax in axes:
        ax.set(
            xlabel="Unit 6 activation",
            xticks=[0, 32, 64, 96, 127],
            yticks=[0, 32, 64, 96, 127],
        )
    axes[0].set_ylabel("Unit 7 activation")
    fig.savefig(out / "two-coordinate-head.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
