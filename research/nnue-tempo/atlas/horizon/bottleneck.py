"""Transfer subsets of the eight bottleneck activations between paired native positions."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from math import comb
from pathlib import Path

import numpy as np
from results import load, values
from summarize import ci, group, score, write
from validate import board

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY / "circuits"))
sys.path.insert(0, str(STUDY))
from analyze import stack_weights
from routing import Cube, compile_kernel


def read_batch(root, states):
    fs = [board(*state).fen() for state in states]
    instructions = "".join("position fen " + fen + "\neval\n" for fen in fs) + "quit\n"
    run = subprocess.run(
        [root / "sf141/src/stockfish"],
        input=instructions,
        text=True,
        capture_output=True,
        check=True,
    )
    units = [
        list(map(int, s.split()[1:]))
        for s in run.stdout.splitlines()
        if s.startswith("units ")
    ]
    assert len(units) == len(states)
    return dict(zip(states, units))


def main(root, out):
    panel, _, data, refs = load(root / "atlas/horizon/all5")
    game_panel, _, _gd, gr = load(root / "atlas/horizon/games7")
    keys = [
        [
            tuple(map(int, t))
            for t in zip(*[data[k] for k in ("a", "b", "wk", "bk", "y", dst)])
        ]
        for dst in ("x", "z")
    ]
    states = sorted(set(keys[0] + keys[1]))
    traces = {}
    with ThreadPoolExecutor(6) as pool:
        for block in pool.map(
            lambda chunk: read_batch(root, chunk), batched(states, 768)
        ):
            traces.update(block)
    trace = np.array([[traces[k] for k in pair] for pair in zip(*keys)], dtype="i4")
    cube = Cube(compile_kernel(root), stack_weights(root, 0))
    cube.verify(trace[:20, :, 2:10])
    utility = np.zeros((len(panel), 256))
    for lo in range(0, len(panel), 256):
        t = trace[lo : lo + 256]
        cells, _ = cube.forward(t[:, :, 2:10])
        assert np.array_equal(cells[:, [0, 255]], t[:, :, 10:])
        pos = cells.astype("i4") @ cube.out + cube.out_bias
        assert np.array_equal(pos[:, [0, 255]], t[:, :, 1])

        def scaled(u):
            return np.trunc(u / 16).astype("i4")

        # With low as donor, the retained high coordinates are the complement
        # of the cube's low-coordinate bits. Use the other donor symmetrically.
        native = scaled(t[:, :, 0] + t[:, :, 1])
        delta_low = scaled(t[:, 0, :1] + pos[:, ::-1]) - native[:, 1, None]
        delta_high = native[:, 0, None] - scaled(t[:, 1, :1] + pos)
        assert np.array_equal(delta_low[:, -1], data["nn"][lo : lo + len(t)])
        assert np.array_equal(delta_high[:, -1], delta_low[:, -1])
        utility[lo : lo + len(t)] = (score(delta_low) + score(delta_high)) / 2
    vec = values(refs, 15)
    neutral = vec[0, :, -1, 1] != vec[1, :, -1, 1]
    game_vec = values(gr, 15)
    game_neutral = game_vec[0, :, -1, 1] != game_vec[1, :, -1, 1]
    assert np.array_equal(panel[panel != "enumerated"], game_panel)
    neutral[panel != "enumerated"] = game_neutral
    kinggroups = np.array([group(w, k) for w, k in zip(data["wk"], data["bk"])])
    selected = []
    units = []
    shapley = np.zeros((len(panel), 8))
    for i in range(8):
        for m in range(256):
            if not (m >> i) & 1:
                shapley[:, i] += (utility[:, m | 1 << i] - utility[:, m]) / (
                    8 * comb(7, m.bit_count())
                )
    assert np.allclose(shapley.sum(1), utility[:, -1] - utility[:, 0])
    masks = np.arange(256)
    sizes = np.array([i.bit_count() for i in range(256)])
    mean = utility[panel == "enumerated"].mean(0)
    best = [(c := masks[sizes == k])[mean[c].argmax()] for k in range(9)]
    for name in np.unique(panel):
        for cond, subset in [
            ("all", panel == name),
            ("neutral_tie", (panel == name) & ~neutral),
            ("neutral_strict", (panel == name) & neutral),
        ]:
            v = utility[subset]
            if not len(v):
                continue
            for k, m in enumerate(best):
                lower, upper = ci(v[:, m] - v[:, -1], kinggroups[subset])
                selected.append(
                    {
                        "panel": name,
                        "condition": cond,
                        "k": k,
                        "retained": " ".join(str(i) for i in range(8) if m >> i & 1),
                        "n": len(v),
                        "accuracy": v[:, m].mean(),
                        "full": v[:, -1].mean(),
                        "gap_lo": lower,
                        "gap_hi": upper,
                    }
                )
            for i in range(8):
                effect = v[:, -1] - v[:, 255 ^ (1 << i)]
                lower, upper = ci(effect, kinggroups[subset])
                units.append(
                    {
                        "panel": name,
                        "condition": cond,
                        "unit": i,
                        "n": len(v),
                        "shapley": shapley[subset, i].mean(),
                        "leave_one_out": effect.mean(),
                        "lo": lower,
                        "hi": upper,
                        "singleton": v[:, 1 << i].mean(),
                        "base": v[:, 0].mean(),
                    }
                )
    write(out / "unit-subsets.csv", selected)
    write(out / "unit-effects.csv", units)
    np.savez_compressed(
        root / "atlas/horizon/patches.npz", utility=utility, shapley=shapley
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    main(args.root, args.out)
