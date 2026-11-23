"""Read the pinned disk layout and replay the four-man HalfKAv2_hm network."""

import struct

import numpy as np
from bottleneck import stack_weights
from feature_routes import ft_weights


def trunc(x, n):
    return np.trunc(x / n).astype("i4")


class Network:
    def __init__(self, root):
        file = root / "sf141/src/nn-13406b1dcbe0.nnue"
        with file.open("rb") as f:
            _, _, desc = struct.unpack("<3I", f.read(12))
        start = 16 + desc
        self.bias = np.memmap(
            file, dtype="<i2", mode="r", offset=start, shape=(1024,)
        ).astype("i4")
        start += 2048
        self.w = np.memmap(
            file, dtype="<i2", mode="r", offset=start, shape=(32, 11, 64, 1024)
        )
        start += self.w.nbytes
        self.ps = np.memmap(
            file, dtype="<i4", mode="r", offset=start, shape=(32, 11, 64, 8)
        )
        self.b1, self.w1 = ft_weights(root, 0)
        self.w1 = self.w1.astype("f8")
        self.b2, self.w2, self.offset, self.out = stack_weights(root, 0)
        self.means = self.w.mean(axis=2)

    def encode(self, rows, a, b):
        w, k, y, h, l = rows[:, :5].T
        t = np.stack([h, l], axis=1)
        buffers = []
        ps = []
        buckets = []
        for black, king in ((True, k), (False, w)):
            flip = 56 * int(black) + 7 * (king % 8 < 4)
            canon = king ^ flip
            bucket = 31 - 4 * (canon // 8) - (canon % 8 - 4)
            buckets.append(bucket)

            def col(table, piece, coords, bucket=bucket, flip=flip):
                return table[bucket, piece, coords ^ flip]

            own = 2 * (a - 1) + int(black)
            other = 2 * (b - 1) + int(not black)
            common = (
                self.bias
                + col(self.w, 10, w)
                + col(self.w, 10, k)
                + col(self.w, other, y)
            )
            buffers.append(
                common[:, None, :] + self.w[bucket[:, None], own, t ^ flip[:, None]]
            )
            direct = (
                col(self.ps, 10, w)[:, 0]
                + col(self.ps, 10, k)[:, 0]
                + col(self.ps, other, y)[:, 0]
            )
            ps.append(
                direct[:, None] + self.ps[bucket[:, None], own, t ^ flip[:, None], 0]
            )
        raw = np.concatenate(buffers, axis=-1)
        assert raw.min() >= -32768 and raw.max() <= 32767
        return raw, trunc(ps[0] - ps[1], 2), buckets

    def offset_change(self, buckets, a):
        return np.concatenate(
            [
                np.rint(self.means[bs, 8 + i] - self.means[bs, 2 * (a - 1) + i])
                for bs, i in zip(buckets, (1, 0))
            ],
            axis=-1,
        ).astype("i4")[:, None, :]

    def forward(self, raw, ps):
        z = raw.clip(0, 127).astype("f8") @ self.w1.T + self.b1
        return self.upper(z, ps)

    def upper(self, z, ps):
        h1 = np.floor(z / 64).clip(0, 127).astype("i4")
        z2 = h1 @ self.w2.T + self.b2
        h2 = (z2 >> 6).clip(0, 127)
        pos = h2 @ self.out + self.offset
        return {
            "z": z,
            "h1": h1,
            "z2": z2,
            "h2": h2,
            "pos": pos,
            "ps": ps,
            "score": trunc(ps + pos, 16),
        }


def credit(m):
    return (1 + np.sign(m)) / 2


def retain(net, trace, indices):
    h = trace["h1"]
    mask = np.zeros(8, bool)
    mask[indices] = True
    ab = np.where(mask, h[:, 0], h[:, 1])
    ba = np.where(mask, h[:, 1], h[:, 0])

    def pos(x):
        return ((x @ net.w2.T + net.b2) >> 6).clip(0, 127) @ net.out + net.offset

    ps = trace["ps"]
    native = trace["score"]
    return (
        credit(trunc(ps[:, 0] + pos(ab), 16) - native[:, 1])
        + credit(native[:, 0] - trunc(ps[:, 1] + pos(ba), 16))
    ) / 2


def relocate(rows, rank):
    """Keep the same two legal moves in both materials after a pawn relocation."""
    import validate as v

    moved = rows.copy()
    moved[:, 2] = 8 * (rank - 1) + moved[:, 2] % 8
    w, k, y, h, l, s = moved.T
    valid = np.ones(len(w), bool)
    for a in (4, 5):
        v.state(a, 1, 0, 0, 0, 0)
        st = v.maps[a, 1]
        for x in (s, h, l):
            valid &= st[w, k, y, x]["v"].astype(bool)
    for x in (h, l):
        same_rank = (
            (x // 8 == s // 8)
            & (y // 8 == s // 8)
            & ((x % 8 - y % 8) * (s % 8 - y % 8) < 0)
        )
        same_file = (
            (x % 8 == s % 8)
            & (y % 8 == s % 8)
            & ((x // 8 - y // 8) * (s // 8 - y // 8) < 0)
        )
        valid &= ~(same_rank | same_file)
    return rows[valid], moved[valid]
