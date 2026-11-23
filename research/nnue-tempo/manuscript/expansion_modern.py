"""Test the Stockfish 18 material-dependent choice of evaluation network."""

import argparse
import gzip
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from atlas.panels import raw
from expansion import child_fens, counts
from expansion_classical import replace
from network import credit
from summarize import write

EXPERIMENT = Path("manuscript/expansion")


def prepare(root):
    clean = root / "clean-sf18"
    revision = subprocess.check_output(
        ["git", "-C", clean, "rev-parse", "HEAD"], text=True
    )
    assert revision.strip() == "cb3d4ee9b47d0c5aae855b12379378ea1439675c"
    target = root / EXPERIMENT / "modern"
    target.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(
        ["git", "-C", clean, "archive", "HEAD", "src", "scripts"]
    )
    subprocess.run(["tar", "xf", "-", "-C", target], input=archive, check=True)
    src = target / "src"
    for name in ("nn-c288c895ea92.nnue", "nn-37f18f62d772.nnue"):
        shutil.copyfile(clean / "src" / name, src / name)
    path = src / "evaluate.cpp"
    replace(
        path,
        "namespace Stockfish {",
        """namespace Stockfish {
static bool forceBig = false;
static bool usedSmall = false;""",
    )
    replace(
        path,
        "bool smallNet           = use_smallnet(pos);",
        "bool smallNet           = !forceBig && use_smallnet(pos);",
    )
    replace(path, "    return v;\n}", "    usedSmall = smallNet;\n    return v;\n}")
    replace(
        path,
        "    auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();",
        """    static auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
    static auto gateCaches = std::make_unique<Eval::NNUE::AccumulatorCaches>(networks);
    accumulators->reset();
    const auto native = evaluate(networks, pos, *accumulators, *gateCaches, VALUE_ZERO);
    const bool small = usedSmall;
    forceBig = true;
    accumulators->reset();
    const auto big = evaluate(networks, pos, *accumulators, *gateCaches, VALUE_ZERO);
    forceBig = false;
    return "gate18 " + std::to_string(native) + " " + std::to_string(big) + " " + std::to_string(small);""",
    )
    subprocess.run(
        [
            "make",
            "-C",
            src,
            "-j" + os.environ.get("JOBS", "10"),
            "build",
            "ARCH=" + os.environ.get("ARCH", "x86-64-avx2"),
        ],
        check=True,
    )


def group(job):
    exe, rows = job
    fens = [fen for row in rows for fen in child_fens(row)]
    return raw(exe, fens, "gate18").reshape(-1, 2, 3)


def measure(root, output, workers):
    work = root / EXPERIMENT
    with gzip.open(work / "pairs.jsonl.gz", "rt") as stream:
        records = list(map(json.loads, stream))
    baseline = np.load(work / "scores.npz")["scores"][:, :, 4]
    wdl = np.sign([r[6] for r in records])
    direction = np.sign(wdl[:, 0] - wdl[:, 1])
    panel = np.array([r[0] for r in records])
    select = (direction != 0) & (panel != "discovery")
    ids = np.flatnonzero(select)
    exe = work / "modern/src/stockfish"
    batches = ((exe, rows) for rows in batched((records[i] for i in ids), 2048))
    with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as pool:
        scored = np.concatenate(list(pool.map(group, batches)))
    assert np.array_equal(scored[:, :, 0], baseline[ids])
    small = scored[:, :, 2] > 0
    assert np.array_equal(scored[:, :, 0][~small], scored[:, :, 1][~small])
    margins = (scored[:, 0, :2] - scored[:, 1, :2]) * direction[ids, None]
    option = np.array([records[i][7] for i in ids])[:, :, 1]
    flag = (option[:, 0] - option[:, 1]) * direction[ids]
    panels = panel[ids]
    material = np.array([records[i][2] for i in ids])
    groups = np.array([records[i][3] for i in ids])
    results = []
    for name in sorted(set(panels)):
        for a in (0, 2, 3, 4, 5):
            for stratum, subset in [
                ("all", np.ones(len(panels), bool)),
                ("favorable", flag > 0),
                ("adverse", flag < 0),
            ]:
                relevant = (panels == name) & subset & ((material == a) if a else True)
                if not relevant.any():
                    continue
                result = counts(
                    groups, credit(margins[:, 0]), credit(margins[:, 1]), relevant
                )
                results.append(
                    dict(
                        panel=name,
                        defender=a,
                        stratum=stratum,
                        used_small=int(small[relevant].any(axis=1).sum()),
                        **result,
                    )
                )
    write(output / "expansion-sf18-selector.csv", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    prepare(args.root.resolve())
    measure(args.root.resolve(), args.results, args.workers)
