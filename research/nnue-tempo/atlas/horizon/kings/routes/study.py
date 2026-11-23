"""Layer interventions on matched geometries and single-material rank controls."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import validate as v
from network import Network, credit, retain
from panel import TAGS
from summarize import write


def panels(work):
    guard = TAGS.index("king_guard_own_pawn_next")
    for f in sorted(work.glob("*.npz")):
        d = np.load(f)
        rows = d["rows"]
        pieces = [int(k) for k in d.files[1:]]
        pawn = d[str(pieces[0])].shape[1] == 1
        if pawn:
            match = np.logical_and.reduce(
                [(d[str(a)][:, 0] >> guard & 1).astype(bool) for a in pieces]
            )
            for rank in range(2, 8):
                for flag, selection in [("escort", match), ("other", ~match)]:
                    sel = (rows[:, 2] // 8 + 1 == rank) & selection
                    yield f.stem, pieces, 1, rank, flag, rows[sel]
        else:
            b = int(f.stem[-1])
            both = np.bitwise_and.reduce([d[str(a)][:, 2] for a in pieces])
            for tag in (
                "all",
                "check",
                "contact_check",
                "check_from_far",
                "cut_kings",
                "check_and_attack_man",
            ):
                sel = (
                    np.ones(len(rows), bool)
                    if tag == "all"
                    else (both >> TAGS.index(tag) & 1).astype(bool)
                )
                yield f.stem, pieces, b, 0, tag, rows[sel]


def describe(net, t):
    h = t["h1"]
    d = h[:, 0] - h[:, 1]
    h2 = t["h2"]
    delta = h2[:, 0] - h2[:, 1]
    z = t["z"]
    z2 = t["z2"]
    dz = z2[:, 0] - z2[:, 1]
    sec = np.divide(delta, dz, out=np.zeros(dz.shape), where=dz != 0)
    edges = (
        d[:, :, None]
        * net.w2.T[None, :, :]
        * sec[:, None, :]
        * net.out[None, None, :]
        / 16
    )
    actual = t["pos"][:, 0] - t["pos"][:, 1]
    assert np.allclose(edges.sum(axis=(1, 2)) * 16, actual)
    result = {
        "native": credit(t["score"][:, 0] - t["score"][:, 1]),
        "keep67": retain(net, t, [6, 7]),
        "without2": retain(net, t, [0, 1, 3, 4, 5, 6, 7]),
        "only2": retain(net, t, [2]),
    }
    for j in range(8):
        result[f"u{j}_byte"] = abs(d[:, j])
        result[f"u{j}_floor"] = (h[:, 0, j] == 0) & (h[:, 1, j] == 0)
        result[f"u{j}_ceiling"] = (h[:, 0, j] == 127) & (h[:, 1, j] == 127)
        result[f"u{j}_pre_gap"] = abs(z[:, 0, j] - z[:, 1, j]) / 64
        result[f"u{j}_score"] = edges[:, j].sum(1)
    return result, edges


def check_native(a, b, rows, t):
    w, k, y, h, l = rows[:, :5].T
    v.state(a, b, 0, 0, 0, 0)
    st = v.maps[a, b]
    for i, x in enumerate((h, l)):
        assert np.array_equal(t["score"][:, i], st[w, k, y, x]["n"])
    assert np.array_equal(
        np.sign(st[w, k, y, h]["w"]) > np.sign(st[w, k, y, l]["w"]),
        np.ones(len(rows), bool),
    )


def main(root, out):
    root = root.resolve()
    v.init(root)
    net = Network(root)
    work = root / "atlas/horizon/kings/routes"
    counts = []
    routes = []
    layer = []
    edge_rows = []
    for panel, pieces, b, rank, flag, rows in panels(work):
        name = {"panel": panel, "black": b, "rank": rank, "rule": flag}
        contexts = len(set(map(tuple, rows[:, :3])))
        counts.append(dict(**name, n=len(rows), contexts=contexts))
        if not len(rows):
            continue
        for a in pieces:
            sums = defaultdict(float)
            edge = np.zeros((8, 32))
            absolute = np.zeros_like(edge)
            for j in range(0, len(rows), 256):
                batch = rows[j : j + 256]
                raw, ps, _ = net.encode(batch, a, b)
                t = net.forward(raw, ps)
                check_native(a, b, batch, t)
                metrics, E = describe(net, t)
                for k, value in metrics.items():
                    sums[k] += value.sum()
                edge += E.sum(0)
                absolute += abs(E).sum(0)
            avg = {k: value / len(rows) for k, value in sums.items()}
            routes.append(
                dict(
                    **name,
                    white=a,
                    n=len(rows),
                    contexts=contexts,
                    **{k: avg[k] for k in ("native", "keep67", "only2", "without2")},
                )
            )
            for i in range(8):
                layer.append(
                    dict(
                        **name,
                        white=a,
                        n=len(rows),
                        unit=i,
                        **{
                            k: avg[f"u{i}_{k}"]
                            for k in ("score", "byte", "floor", "ceiling", "pre_gap")
                        },
                    )
                )
            if (
                flag in ("escort", "check_and_attack_man", "check_from_far")
                and len(rows) >= 5
            ):
                for i in range(8):
                    for j in np.argsort(-absolute[i], kind="stable")[:4]:
                        edge_rows.append(
                            dict(
                                **name,
                                white=a,
                                n=len(rows),
                                h1=i,
                                h2=j,
                                signed=edge[i, j] / len(rows),
                                mean_abs=absolute[i, j] / len(rows),
                                fraction_abs=absolute[i, j] / max(1, absolute[i].sum()),
                            )
                        )
    write(out / "support.csv", counts)
    write(out / "transfer.csv", routes)
    write(out / "hidden-units.csv", layer)
    write(out / "edges.csv", edge_rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
