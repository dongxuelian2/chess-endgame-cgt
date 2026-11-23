"""Apply the unchanged Syzygy move restrictions to both material substitutions."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from summarize import write


def run(root, outdir):
    root = root.resolve()
    work = root / "atlas/horizon/kings/routes"
    exe = work.parent / "engine/src/stockfish"
    tb = root / "pychess/data/syzygy/regular"

    def solve(job):
        key, p, rows, a, b, method = job
        inp = work / f"{key}-{p}-{a}-{method}.in"
        out = inp.with_suffix(".out")
        np.savetxt(inp, rows[:, :5], fmt="%d")
        subprocess.run(
            [exe, method, tb, "0", str(a), str(b), inp, out],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        data = np.loadtxt(out, dtype="i8", ndmin=2)
        assert np.array_equal(data[:, :5], rows[:, :5])
        return data[:, 5:]

    counts = []
    with ThreadPoolExecutor(10) as pool:
        for f in sorted(work.glob("*.pairs")):
            if not f.stat().st_size:
                counts.append((f.stem, 0, 0))
                continue
            key = f.stem
            solo = key.startswith("solo")
            pair = key.split("-")
            a, b = (
                (int(pair[1]), 1)
                if solo or pair[0] == "relaxed"
                else tuple(map(int, pair))
            )
            rows = np.loadtxt(f, dtype="i8", ndmin=2)
            res = {"rows": rows}
            counts.append((key, len(rows), len(set(map(tuple, rows[:, :3])))))
            method = "third" if b == 1 else "restrictions"
            for piece in [a] if solo else [a, 5]:
                jobs = [
                    (key, p, rows[p : p + 600], piece, b, method)
                    for p in range(0, len(rows), 600)
                ]
                res[str(piece)] = np.concatenate(list(pool.map(solve, jobs)))
            np.savez_compressed(work / f"{key}.npz", **res)

    write(
        outdir / "screen-counts.csv",
        [{"panel": key, "pairs": n, "layouts": g} for key, n, g in counts],
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
