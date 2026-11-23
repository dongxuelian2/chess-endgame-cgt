"""Relocate the black king across its orientation boundary within the same king bucket."""

import sys
from pathlib import Path

import numpy as np
import validate as v
from network import Network
from promotion_geometry import relation
from summarize import write


def channel(net, u, a, w, k, y, x):
    black = u < 1024
    king = k if black else w
    flip = 56 * int(black) + 7 * (king % 8 < 4)
    canon = king ^ flip
    bucket = 31 - 4 * (canon // 8) - (canon % 8 - 4)
    W = net.w[bucket, :, :, u % 1024]
    return (
        net.bias[u % 1024]
        + W[10, w ^ flip]
        + W[10, k ^ flip]
        + W[int(not black), y ^ flip]
        + W[2 * (a - 1) + int(black), x ^ flip]
    ).clip(0, 127)


def main(root, out):
    v.init(root)
    net = Network(root)
    records = []
    for a in range(2, 6):
        v.state(a, 1, 0, 0, 0, 0)
        st = v.maps[a, 1]
        for file in [1, 6]:
            y = 8 + (file ^ 7)
            for sf, u in [("black", 273), ("white", 1869)]:
                total = np.zeros((5, 5))
                for w in range(64):
                    if w % 8 >= 4:
                        continue
                    for rank in range(8):
                        # Order king cells so Black sees b7 at the first endpoint.
                        k = 8 * rank + (3 if file == 1 else 4)
                        other = k ^ 7
                        x = np.flatnonzero(
                            (st[w, k, y, :]["v"] != 0) & (st[w, other, y, :]["v"] != 0)
                        )
                        if not len(x):
                            continue
                        first = channel(net, u, a, w, k, y, x)
                        second = channel(net, u, a, w, other, y, x)
                        g = relation(a, x, y - 8, w, k)
                        gg = relation(a, x, y - 8, w, other)
                        truth = np.sign(st[w, k, y, x]["w"]) == np.sign(
                            st[w, other, y, x]["w"]
                        )
                        for h in (2, 3, 4):
                            m = (g == h) & (gg == h)
                            total[h] += [
                                m.sum(),
                                first[m].sum(),
                                second[m].sum(),
                                (first[m] > second[m]).sum(),
                                truth[m].sum(),
                            ]
                for h in (2, 3, 4):
                    n, t1, t2, dec, same = total[h]
                    if n:
                        records.append(
                            {
                                "white": a,
                                "file": chr(97 + file),
                                "view": sf,
                                "ft": u,
                                "geometry": {2: "off", 3: "blocked", 4: "clear"}[h],
                                "edits": int(n),
                                "black_view_b": t1 / n,
                                "black_view_g": t2 / n,
                                "strict_decreases": int(dec),
                                "same_wdl": int(same),
                            }
                        )
    write(out / "king-crossing.csv", records)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
