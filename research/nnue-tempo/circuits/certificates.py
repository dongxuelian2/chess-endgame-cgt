"""Exhaustively certify move preferences under complementary H1 interchanges."""

import argparse
import json
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from analyze import stack_weights
from collect import EXTENDED, MATERIAL
from routing import MASKS, ORDER, Cube, compile_kernel
from spectra import load


def certificates(rows, cube):
    certified = np.zeros(256, dtype=int)
    histogram = np.zeros(9, dtype=int)
    n_correct = 0
    for start in range(0, len(rows), 128):
        trace = np.array([r["units"] for r in rows[start : start + 128]])
        activations, _ = cube.forward(trace[:, :, 2:10])
        values = (activations.astype(np.float32) @ cube.out.astype(np.float32)).astype(
            np.int32
        )
        margin = values[:, ::-1] - values + (trace[:, 1, 0] - trace[:, 0, 0])[:, None]
        assert np.array_equal(margin[:, 0], (trace[:, 1, :2] - trace[:, 0, :2]).sum(1))
        margin = margin[margin[:, 0] > 0]
        if len(margin) == 0:
            continue
        n_correct += len(margin)
        worst = margin.copy()
        for i in range(8):
            view = worst.reshape(-1, 256 >> (i + 1), 2, 1 << i)
            np.minimum(view[:, :, 1], view[:, :, 0], out=view[:, :, 1])
        sufficient = worst[:, 255 - MASKS] > 0
        assert sufficient[:, -1].all()
        if start == 0:
            direct = [np.min(margin[0, (MASKS & keep) == 0]) > 0 for keep in MASKS]
            assert np.array_equal(direct, sufficient[0])
        certified += sufficient.sum(0)
        min_size = np.min(np.where(sufficient, ORDER, 9), axis=1)
        histogram += np.bincount(min_size, minlength=9)
    assert sum(histogram) == certified[255] == n_correct
    return {
        "n": len(rows),
        "correct": n_correct,
        "min_size": histogram.tolist(),
        "mask_count": certified.tolist(),
    }


def validate_motif(rows, tb):
    counts = [0, 0, 0]
    for r in rows:
        boards = [chess.Board(fen) for fen in r["children"]]
        if max(b.occupied.bit_count() for b in boards) > 5 or any(
            b.castling_rights for b in boards
        ):
            continue
        try:
            wdl = [int(np.sign(tb.probe_wdl(b))) for b in boards]
        except chess.syzygy.MissingTableError:
            continue
        # Lower WDL for the opponent is better for the player making the move.
        counts[int(np.sign(wdl[1] - wdl[0])) + 1] += 1
    return {"worse": counts[0], "equal": counts[1], "better": counts[2]}


def main(root):
    directory = root / "circuits"
    lib = compile_kernel(root)
    cubes = [Cube(lib, stack_weights(root, stage)) for stage in range(8)]
    out = {"siblings": {}, "motifs": {}, "motif_wdl": {}}
    for m in MATERIAL + EXTENDED:
        rows = load(directory / f"{m}.siblings.gz")
        out["siblings"][m] = certificates(rows, cubes[int(m in EXTENDED)])
    with chess.syzygy.open_tablebase(root / "pychess/data/syzygy/regular") as tb:
        tb.add_directory(str(root / "tb5"))
        for task in ("contacts", "forks"):
            for month in ("2013-01", "2013-02"):
                rows = load(directory / f"{task}-{month}.gz")
                key = f"{task}-{month}"
                out["motifs"][key] = {
                    str(i): certificates([r for r in rows if r["stack"] == i], cube)
                    for i, cube in enumerate(cubes)
                }
                out["motif_wdl"][key] = validate_motif(
                    [r for r in rows if r["stack"] <= 1], tb
                )
    (directory / "certificates.json").write_text(json.dumps(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
