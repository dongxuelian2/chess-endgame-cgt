"""Independent legal move-set and policy replay on selected complete four-man blocks."""

import argparse
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from summarize import MATS, write
from validate import REPLY, RULES, board, init, sign, state


def check(job):
    root, a, b = job
    init(root)
    atlas = (root / "atlas").resolve()
    stem = f"{a}-{b}"
    rng = np.random.default_rng(43716 + 27 * a + b)
    src = np.memmap(atlas / "remaining" / f"{stem}.replies", mode="r", dtype=REPLY)
    indices = np.flatnonzero(np.any(src["mask"][:, :, 1] != 0, axis=1))
    sample = src[rng.choice(indices, 16, replace=False)]
    weights, orient = np.loadtxt(atlas / "weights.txt", dtype=int).T
    n = ties = mixed = 0
    utility = 0.0
    names = RULES + [
        "fitted",
        "simple_tactics",
        "native141",
        "psqt",
        "positional",
        "oracle",
        "released_classical",
    ]
    control = np.zeros((len(names), 5))
    with tempfile.TemporaryDirectory(dir=atlas) as tmp:
        path = Path(tmp)
        sample.tofile(path / f"{stem}.replies")
        (path / f"{stem}.bin").symlink_to(atlas / f"{stem}.bin")
        subprocess.run(
            [atlas / "decisions", path, path, str(a), str(b), atlas / "weights.txt"],
            check=True,
        )
        for row in sample:
            i = int(row["index"])
            wk, bk, y = i >> 12, i >> 6 & 63, i & 63
            for x in range(64):
                if not state(a, b, wk, bk, y, x)["v"]:
                    continue
                q = board(a, b, wk, bk, y, x)
                q.turn = True
                opts = []
                for m in list(q.generate_legal_moves(from_mask=1 << x)):
                    if m.promotion or q.is_capture(m) or q.gives_check(m):
                        continue
                    q.push(m)
                    legal = not q.has_legal_en_passant() and any(q.legal_moves)
                    q.pop()
                    if legal:
                        opts.append(m.to_square)
                if len(opts) < 2:
                    continue
                n += 1
                es = [state(a, b, wk, bk, y, s) for s in opts]
                lowest = min(e["f"] for e in es)
                best = [(e, s) for e, s in zip(es, opts) if e["f"] == lowest]
                w = np.array([-sign(e["w"]) for e, _ in best])
                ties += len(w) > 1
                utility += w.mean()
                if min(w) == max(w):
                    continue
                mixed += 1
                masks = np.array([row["mask"][s][0] for _, s in best], dtype="u8")
                z = ((masks[:, None] >> np.arange(len(RULES), dtype="u8")) & 1).astype(
                    int
                )
                costs = np.column_stack(
                    [
                        z * orient,
                        z @ weights,
                        z[:, [0, 1, 8]].sum(1),
                        *[np.array([e[k] for e, _ in best]) for k in ("n", "p", "d")],
                        -w,
                        np.array([e["h"] for e, _ in best]),
                    ]
                )
                for j, c in enumerate(costs.T):
                    v = w[c == c.min()]
                    gain = (int(v.sum()) * len(w) - int(w.sum()) * len(v)) / (
                        len(v) * len(w)
                    )
                    control[j] += [gain > 0, gain == 0, gain < 0, len(v) < len(w), gain]
        for j, line in enumerate((path / f"{stem}.decisions").read_text().splitlines()):
            name, *v = line.split()
            z = np.array(v, float)
            assert (
                name == names[j]
                and np.allclose(z[:4], [n, ties, mixed, utility])
                and np.allclose(z[4:], control[j])
            ), (a, b, name, z, control[j])
    return {
        "actor": a,
        "opponent": b,
        "blocks": len(sample),
        "parents": n,
        "mixed": mixed,
        "policies": len(names),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    arg = p.parse_args()
    with ProcessPoolExecutor(8) as ex:
        write(
            arg.out / "choice-audit.csv",
            list(ex.map(check, [(arg.root, a, b) for a, b in MATS])),
        )
