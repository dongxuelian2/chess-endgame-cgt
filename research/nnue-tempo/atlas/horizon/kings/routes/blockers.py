"""Move one king off a promotion ray and compare sibling edits which keep it blocked."""

import sys
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
import validate as v
from network import Network
from promotion_geometry import relation
from summarize import write


def activation(net, feature, oriented, a, w, k, x):
    bucket = 31 - 4 * (w // 8) - ((w ^ 7) % 8 - 4)
    u = feature - 1024
    W = net.w[bucket, :, :, u]
    return int(
        np.clip(
            net.bias[u]
            + W[10, w ^ 7]
            + W[10, k ^ 7]
            + W[1, oriented]
            + W[2 * (a - 1), x ^ 7],
            0,
            127,
        )
    )


def main(root, out):
    net = Network(root)
    v.init(root)
    rows = []
    for a in (4, 5):
        v.state(a, 1, 0, 0, 0, 0)
        st = v.maps[a, 1]
        for oriented, feature in ((10, 1424), (13, 1196)):
            y = oriented ^ 7
            t = y - 8
            data = defaultdict(list)
            joint = defaultdict(list)

            for w in range(64):
                if w % 8 >= 4:
                    continue
                ks, xs = np.where(st[w, :, y, :]["v"] > 0)
                m = relation(a, xs, t, w, ks) == 3
                for k, x in zip(ks[m].tolist(), xs[m].tolist()):
                    base = activation(net, feature, oriented, a, w, k, x)
                    for side, old in [("white", w), ("black", k)]:
                        children = defaultdict(list)
                        for z in chess.SquareSet(chess.BB_KING_ATTACKS[old]):
                            nw, nk = (z, k) if side == "white" else (w, z)
                            if nw % 8 >= 4 or not st[nw, nk, y, x]["v"]:
                                continue
                            c = int(relation(a, x, t, nw, nk))
                            if c in (3, 4):
                                children[c].append(
                                    activation(net, feature, oriented, a, nw, nk, x)
                                )
                        for c, values in children.items():
                            data[side, c].extend(v - base for v in values)
                        if 3 in children and 4 in children:
                            joint[side].append(
                                np.mean(children[4]) - np.mean(children[3])
                            )
            for (side, c), ds in sorted(data.items()):
                rows.append(
                    {
                        "white": a,
                        "ft": feature,
                        "king": side,
                        "comparison": "unblocks" if c == 4 else "stays_blocked",
                        "n": len(ds),
                        "mean_change": np.mean(ds),
                        "decrease": np.mean(np.array(ds) < 0),
                    }
                )
            for side, ds in sorted(joint.items()):
                rows.append(
                    {
                        "white": a,
                        "ft": feature,
                        "king": side,
                        "comparison": "within_parent_open_minus_blocked",
                        "n": len(ds),
                        "mean_change": np.mean(ds),
                        "decrease": np.mean(np.array(ds) < 0),
                    }
                )
    write(out / "blocker-edits.csv", rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
