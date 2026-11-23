"""Evaluate a shared two-coordinate head on every legal mirrored minor-pawn representative."""

import argparse
from pathlib import Path

import numpy as np
from bottleneck import stack_weights
from summarize import write
from validate import DT

RAW = np.dtype([("ps", "<i4"), ("pos", "<i4"), ("h", "u1", (8,))])


def grid(root):
    b, w, last, v = stack_weights(root, 0)
    h = np.zeros((128, 128, 8), dtype=np.int32)
    h[:, :, :6] = [0, 127, 0, 127, 0, 127]
    h[:, :, 6], h[:, :, 7] = np.indices((128, 128))
    table = ((h @ w.T + b) >> 6).clip(0, 127) @ v + last
    # 128^2 times the least-squares additive approximation under uniform grid weights.
    additive = 128 * (table.sum(0)[None, :] + table.sum(1)[:, None]) - table.sum()
    return table, additive


def main(root, out):
    table, additive = grid(root)
    study = root / "atlas/horizon"
    np.savetxt(
        study / "head.txt",
        np.concatenate([x.ravel() for x in stack_weights(root, 0)]),
        fmt="%d",
    )
    # The largest mixed difference lower-bounds uniform error of any additive map.
    largest = 0
    example = None
    for i in range(128):
        v = table[i, None, :] - table
        spread = np.ptp(v, axis=1)
        j = spread.argmax()
        if spread[j] > largest:
            largest = spread[j]
            example = (i, int(j), int(v[j].argmax()), int(v[j].argmin()))
    write(
        out / "head-grid.csv",
        [
            {
                "cells": table.size,
                "largest_abs_mixed": int(largest),
                "u1": example[0],
                "u2": example[1],
                "v1": example[2],
                "v2": example[3],
                "minimum_uniform_raw_error": largest / 4,
                "uniform_additive_rmse": np.mean((table - additive / 16384) ** 2)
                ** 0.5,
            }
        ],
    )
    summary = []
    histrows = []
    realized = []
    for a in (2, 3):
        raw = np.memmap(study / f"{a}.raw", dtype=RAW, mode="r", shape=(64, 64, 64, 64))
        dense = np.memmap(
            root / f"atlas/{a}-1.bin", dtype=DT, mode="r", shape=raw.shape
        )
        modal_joint = np.zeros((128, 128), dtype="i8")
        hist = np.zeros((8, 128), dtype="i8")
        joint = np.zeros((128, 128), dtype="i8")
        exact = face = close = total = 0
        mae = addmae = addexact = 0
        maxerr = 0
        for w in range(64):
            if w % 8 >= 4:
                continue
            valid = dense[w]["v"].astype(bool)
            r = raw[w][valid]
            scores = dense[w]["n"][valid].astype("i4")
            h = r["h"]
            for i in range(8):
                hist[i] += np.bincount(h[:, i], minlength=128)
            np.add.at(joint, (h[:, 6], h[:, 7]), 1)
            fixed = np.all(h[:, :6] == [0, 127, 0, 127, 0, 127], axis=1)
            face += fixed.sum()
            total += len(h)
            np.add.at(modal_joint, (h[fixed, 6], h[fixed, 7]), 1)
            pos = table[h[:, 6], h[:, 7]]
            guess = np.trunc((pos + r["ps"]) / 16).astype("i4")
            err = abs(guess - scores)
            exact += sum(err == 0)
            close += sum(err <= 1)
            mae += err.sum()
            maxerr = max(maxerr, err.max())
            assert np.all(r["pos"][fixed] == pos[fixed])
            pred = np.trunc((additive[h[:, 6], h[:, 7]] / 16384 + r["ps"]) / 16).astype(
                "i4"
            )
            ad = abs(pred - scores)
            addmae += ad.sum()
            addexact += sum(ad == 0)
        xx, yy = np.where(modal_joint)
        X = np.zeros((len(xx), 256))
        X[np.arange(len(xx)), xx] = 1
        X[np.arange(len(xx)), 128 + yy] = 1
        wt = np.sqrt(modal_joint[xx, yy] / modal_joint.sum())
        truth = table[xx, yy]
        coef = np.linalg.lstsq(X * wt[:, None], truth * wt, rcond=None)[0]
        rmse = np.linalg.norm(wt * (X @ coef - truth))
        top = 0
        corners = None
        for i in range(128):
            for j in range(i):
                ys = np.flatnonzero((modal_joint[i] >= 10) & (modal_joint[j] >= 10))
                if len(ys) < 2:
                    continue
                delta = table[i, ys] - table[j, ys]
                mixed = np.ptp(delta)
                if mixed > top:
                    top = mixed
                    corners = (i, j, int(ys[delta.argmax()]), int(ys[delta.argmin()]))
        i, j, s, t = corners
        realized.append(
            {
                "white": a,
                "modal_boards": int(modal_joint.sum()),
                "modal_cells": len(xx),
                "min_additive_raw_rmse": rmse,
                "largest_mixed": int(top),
                "min_uniform_raw_error": top / 4,
                "u1": i,
                "u2": j,
                "v1": s,
                "v2": t,
                "least_corner_count": min(
                    modal_joint[k, z] for k in (i, j) for z in (s, t)
                ),
            }
        )
        for i, h in enumerate(hist):
            mode = h.argmax()
            histrows.append(
                {
                    "white": a,
                    "unit": i,
                    "n": total,
                    "modal_value": mode,
                    "modal_count": h[mode],
                    "zero": h[0],
                    "saturated": h[-1],
                    "unique": np.count_nonzero(h),
                }
            )
        summary.append(
            {
                "white": a,
                "n": total,
                "modal_face": face,
                "unique_u6_u7": np.count_nonzero(joint),
                "exact_score": exact,
                "within_one": close,
                "mae": mae / total,
                "max_error": maxerr,
                "exact_additive": addexact,
                "additive_mae": addmae / total,
            }
        )
    write(out / "realized-head.csv", realized)
    write(out / "compressed-head.csv", summary)
    write(out / "unit-occupancy.csv", histrows)
    np.savetxt(
        study / "grid.txt", np.column_stack((table.ravel(), additive.ravel())), fmt="%d"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.root, a.out)
