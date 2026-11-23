"""Compute exact finite-domain counts and sibling tests from dense score and reply files."""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path


def main(root, remaining=False, kind="measure"):
    here = Path(__file__).resolve().parent
    atlas = root / "atlas"
    exe = atlas / kind
    out = (atlas / "remaining" if remaining else atlas) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-march=native",
            here / (kind + ".cpp"),
            "-o",
            exe,
        ],
        check=True,
    )

    def run(ab):
        args = [exe, atlas, out, *map(str, ab)]
        if kind == "decisions":
            args.append(atlas / "weights.txt")
        if remaining:
            args.append("remaining")
        subprocess.run(args, check=True)

    with ThreadPoolExecutor(10) as ex:
        list(ex.map(run, product(range(1, 6), repeat=2)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--remaining", action="store_true")
    p.add_argument("--decisions", action="store_true")
    p.add_argument("--crossover", action="store_true")
    a = p.parse_args()
    kind = "crossover" if a.crossover else "decisions" if a.decisions else "measure"
    main(a.root, a.remaining, kind)
