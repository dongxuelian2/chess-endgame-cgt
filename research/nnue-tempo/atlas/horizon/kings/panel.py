"""Replay the deep residuals, then test identical restrictions on untouched contexts."""

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from results import load
from summarize import MATS, write

HERE = Path(__file__).resolve().parent
TAGS = re.findall(r"RULE\((\w+)\)", (HERE / "moves.def").read_text())


def samples(root):
    home = root / "atlas/horizon"
    needed = {}
    for prefix in ("games7", "all5"):
        panel, gids, d, _refs = load(home / prefix)
        t = np.load(home / prefix / "proofs.npz")["proof"] <= 13
        low = t[:, :, :2].sum(2) - 1
        high = 1 - t[:, :, 2:].sum(2)
        opaque = (low[0] < high[1]) & (low[1] < high[0])
        for n, name in enumerate(panel):
            if prefix == "all5" and name != "enumerated":
                continue
            key = tuple(int(d[s][n]) for s in ("a", "b", "wk", "bk", "y", "x", "z"))
            needed.setdefault(key, []).append(
                (
                    name,
                    gids[n],
                    bool(opaque[n]),
                    int(d["nn"][n]),
                    int(d["wh"][n]),
                    int(d["wl"][n]),
                )
            )
    for a, b in MATS:
        f = root / "atlas/remaining/metrics" / f"{a}-{b}.pairs"
        if not f.stat().st_size:
            continue
        for row in np.loadtxt(f, dtype="i8", ndmin=2):
            idx, _, h, l, wh, wl, nn, _, _, x, z = map(int, row)
            if (h | l) & 51:
                continue
            key = (a, b, idx >> 12, idx >> 6 & 63, idx & 63, x, z)
            if any(t[0] == "enumerated" for t in needed.get(key, ())):
                continue
            needed.setdefault(key, []).append(
                ("extension", "", None, nn, int(np.sign(wh)), int(np.sign(wl)))
            )
    return needed


def main(root, out):
    root = root.resolve()
    home = root / "atlas/horizon/kings"
    needed = samples(root)
    tasks = []
    for a, b in MATS:
        keys = sorted(k[2:] for k in needed if k[:2] == (a, b))
        for start in range(0, len(keys), 1600):
            f = home / f"pairs-{a}-{b}-{start}.in"
            f.write_text(
                "".join(
                    " ".join(map(str, k)) + "\n" for k in keys[start : start + 1600]
                )
            )
            tasks.append((a, b, f))

    def run(task):
        a, b, f = task
        subprocess.run(
            [
                home / "engine/src/stockfish",
                "restrictions",
                root / "pychess/data/syzygy/regular",
                "0",
                str(a),
                str(b),
                f,
                f.with_suffix(".out"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(run, tasks))
    rows = []
    stats = {}
    cases = []
    for a, b, f in tasks:
        for row in np.loadtxt(f.with_suffix(".out"), dtype="i8", ndmin=2):
            w, k, y, x, z, vh, vl, black, white = map(int, row)
            for name, gid, opaque, nn, wh, wl in needed[(a, b, w, k, y, x, z)]:
                assert (vh, vl) == (wh, wl)
                correct = (int(np.sign(nn)) + 1) / 2
                if name == "fresh" and opaque:
                    cases.append(
                        {
                            "game": gid,
                            "white_man": a,
                            "black_man": b,
                            "wk": w,
                            "bk": k,
                            "y": y,
                            "high": x,
                            "low": z,
                            "nnue_margin": nn,
                            "black_erases": " ".join(
                                t for i, t in enumerate(TAGS) if black >> i & 1
                            ),
                            "white_erases": " ".join(
                                t for i, t in enumerate(TAGS) if white >> i & 1
                            ),
                        }
                    )
                for subset in ["all", "unresolved13"] if opaque else ["all"]:
                    for material in ("all", f"{a}-{b}"):
                        key = name, subset, material
                        if key not in stats:
                            stats[key] = np.zeros((2, len(TAGS) + 1, 2))
                        arr = stats[key]
                        for i, mask in enumerate((black, white)):
                            arr[i, -1] += (1, correct)
                            while mask:
                                bit = mask & -mask
                                arr[i, bit.bit_length() - 1] += (1, correct)
                                mask ^= bit
    for (name, subset, mat), arr in sorted(stats.items()):
        for i, player in enumerate(("black", "white")):
            for t, label in enumerate(TAGS + ["all_pairs"]):
                n, points = arr[i, t]
                if n == 0 and label != "all_pairs":
                    continue
                rows.append(
                    {
                        "panel": name,
                        "selection": subset,
                        "material": mat,
                        "restricted_player": player,
                        "action": label,
                        "n": int(n),
                        "neural_points": points,
                    }
                )
    write(out / "response-census.csv", rows)
    write(out / "residual-28.csv", sorted(cases, key=lambda r: r["game"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
