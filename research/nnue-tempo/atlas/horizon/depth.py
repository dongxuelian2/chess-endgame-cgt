"""Find the minimum horizon proving four player/outcome thresholds."""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from results import load, values
from summarize import score, write


def main(root, out, cap, population):
    scratch = root / "atlas/horizon"
    folder = scratch / population
    work = folder / "threshold-shards"
    work.mkdir(exist_ok=True)
    tasks = []
    outputs = []
    for inp in sorted(folder.glob("*.in")):
        lines = inp.read_text().splitlines()
        dst = inp.with_suffix(".proof")
        shards = []
        for start in range(0, len(lines), 160):
            sub = work / f"{inp.stem}-{start}.in"
            sub.write_text("\n".join(lines[start : start + 160]) + "\n")
            shards.append(sub.with_suffix(".out"))
            tasks.append((inp.stem, sub))
        outputs.append((dst, shards))

    def job(spec):
        stem, sub = spec
        a, b = stem.split("-")
        run = subprocess.run(
            [
                scratch / "engine/src/stockfish",
                "proof",
                root / "pychess/data/syzygy/regular",
                str(cap),
                a,
                b,
                sub,
                sub.with_suffix(".out"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(run.stdout.splitlines()[-1], flush=True)

    with ThreadPoolExecutor(14) as pool:
        list(pool.map(job, tasks))
    for dst, shards in outputs:
        temp = dst.with_suffix(".tmp")
        temp.write_text("".join(f.read_text() for f in shards))
        temp.replace(dst)
    panel, gids, ds, refs = load(folder)
    depthmap = {}
    for f in folder.glob("*.proof"):
        a, b = map(int, f.stem.split("-"))
        for row in np.loadtxt(f, dtype=int, ndmin=2):
            depthmap[(a, b, *row[:4])] = row[4:]
    proof = np.array(
        [
            [
                depthmap[k]
                for k in zip(ds["a"], ds["b"], ds["wk"], ds["bk"], ds["y"], ds[t])
            ]
            for t in ("x", "z")
        ]
    )
    check = values(refs, 15)
    truth = np.stack((ds["wh"], ds["wl"]))
    result = []
    onsets = []
    for h in range(cap + 1):
        by = proof <= h
        lower = by[:, :, :2].sum(2) - 1
        upper = 1 - by[:, :, 2:].sum(2)
        assert np.all((lower <= truth) & (truth <= upper))
        neutral = np.clip(0, lower, upper)
        if h < check.shape[2]:
            assert (
                np.array_equal(lower, check[:, :, h, 0])
                and np.array_equal(upper, check[:, :, h, 2])
                and np.array_equal(neutral, check[:, :, h, 1])
            )
        strict = lower[0] > upper[1]
        weak = neutral[0] > neutral[1]
        onsets.append((strict, weak))
        for k in np.unique(panel):
            t = panel == k
            unresolved = t & ~weak
            cert = t & strict
            nn = score(ds["nn"])
            result.append(
                {
                    "panel": k,
                    "depth": h,
                    "n": int(sum(t)),
                    "separated": int(sum(cert)),
                    "neutral_strict": int(sum(weak & t)),
                    "neutral_ties": int(sum(unresolved)),
                    "nnue_on_ties": nn[unresolved].mean() if unresolved.any() else None,
                    "hybrid": np.where(weak[t], 1, nn[t]).mean(),
                }
            )
    write(
        out
        / ("proof-depths.csv" if population == "games7" else "enumerated-depths.csv"),
        result,
    )

    def first(i, k):
        return next(
            (h for h, (s, w) in enumerate(onsets) if (s if k == 0 else w)[i]), None
        )

    np.savez_compressed(folder / "proofs.npz", proof=proof)
    if population == "all5":
        return
    write(
        out / "proof-onsets.csv",
        [
            {
                "panel": p,
                "game": gid,
                "first_separation": first(i, 0),
                "first_neutral_order": first(i, 1),
            }
            for i, (p, gid) in enumerate(zip(panel, gids))
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--cap", type=int, default=13)
    p.add_argument("--population", choices=["games7", "all5"], default="games7")
    a = p.parse_args()
    main(a.root.resolve(), a.out, a.cap, a.population)
