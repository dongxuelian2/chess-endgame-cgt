"""Exact four/five-man inference with separately selectable HalfKAv2_hm king addresses."""

import json
import struct
from pathlib import Path

import chess
import numpy as np
from analyze import stack_weights
from feature_routes import ft_weights


class Model:
    def __init__(self, root):
        data = np.memmap(root / "sf141/src/nn-13406b1dcbe0.nnue", mode="r", dtype="u1")
        n = struct.unpack_from("<I", data, 8)[0]
        start = 12 + n
        assert struct.unpack_from("<I", data, start)[0] == (0x7F234CB8 ^ 2048)
        start += 4
        self.bias = np.frombuffer(data, dtype="<i2", count=1024, offset=start)
        start += 2048
        self.weights = np.frombuffer(
            data, dtype="<i2", count=22528 * 1024, offset=start
        ).reshape(22528, 1024)
        start += 22528 * 2048
        self.ps = np.frombuffer(
            data, dtype="<i4", count=22528 * 8, offset=start
        ).reshape(22528, 8)
        self.head = []
        for stage in (0, 1):
            b, w = ft_weights(root, stage)
            c, v, e, z = stack_weights(root, stage)
            self.head.append((b.astype(np.int32), w.T.astype(np.float32), c, v.T, e, z))
        cfg = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "pawn-geometry/results/channel-masks.json"
            ).read_text()
        )
        self.probes = np.stack([cfg["coef"], cfg["placebo"]], axis=-1)

    def indices(self, squares, men, keys=None):
        # All examples have Black to move. First two columns are WK and BK.
        squares = np.asarray(squares, dtype=np.int32)
        anchors = squares[:, :2] if keys is None else np.asarray(keys)
        parts = []
        for side in (0, 1):  # Black input first, then White
            anchor = anchors[:, 1 - side]
            orient = 56 * (1 - side) ^ ((anchor % 8 < 4) * 7)
            canon = anchor ^ orient
            bucket = 31 - (canon // 8) * 4 - (canon % 8 - 4)
            pcs = np.array(
                [
                    10 if abs(m) == 6 else 2 * (abs(m) - 1) + int((m > 0) != bool(side))
                    for m in men
                ]
            )
            parts.append(bucket[:, None] * 704 + 64 * pcs + (squares ^ orient[:, None]))
        return parts

    def read(self, squares, men, keys=None, need_raw=False):
        stage = (len(men) - 1) // 4
        parts = self.indices(squares, men, keys)
        acc = []
        q = []
        for ids in parts:
            # Summation in int32 provides the pre-clip oracle checks.
            raw = np.broadcast_to(self.bias, (len(ids), 1024)).astype(np.int32).copy()
            for i in range(ids.shape[1]):
                raw += self.weights[ids[:, i]]
            acc.append(raw)
            q.append(self.ps[ids, stage].sum(1))
        raw = np.concatenate(acc, axis=1)
        ps = np.trunc((q[0] - q[1]) / 2).astype(np.int32)
        return (raw if need_raw else raw.clip(0, 127).astype(np.uint8)), ps

    def score(self, x, ps, stage):
        b, w, c, v, e, z = self.head[stage]
        h = (x.astype(np.float32) @ w).astype(np.int32) + b
        h = (h >> 6).clip(0, 127)
        h2 = ((h @ v + c) >> 6).clip(0, 127)
        deep = h2 @ z + e
        return np.trunc((deep + ps) / 16).astype(np.int32), h, h2, deep


def board_for(squares, men, turn=False):
    b = chess.Board.empty()
    b.turn = turn
    for s, m in zip(squares, men):
        b.set_piece_at(int(s), chess.Piece(abs(m), m > 0))
    return b


def verify(root):
    import gzip

    from feature_routes import get_features
    from siblings import trace_positions

    rng = np.random.default_rng(72313)
    model = Model(root)
    counts = 0
    for mat in ("KRvKP", "KQvKP", "KBvKP", "KNvKP", "KRPvKP", "KBPvKP", "KNPvKP"):
        with gzip.open(root / "pawn-geometry" / f"random-{mat}.gz") as fp:
            rs = list(map(json.loads, fp))
        rs = [rs[i] for i in rng.choice(len(rs), 48, replace=False)]
        _x, _z, raw = get_features(rs, root, raw=True)
        fs = [f for r in rs for f in r["children"]]
        men = [6, -6, chess.Piece.from_symbol(mat[1]).piece_type, -1] + (
            [1] if len(mat) == 6 else []
        )
        ss = [
            [next(iter(b.pieces(abs(m), m > 0))) for m in men]
            for b in map(chess.Board, fs)
        ]
        own, ps = model.read(ss, men, need_raw=True)
        assert np.array_equal(own, raw.reshape(own.shape))
        native = trace_positions(fs, root)
        v, h, h2, deep = model.score(own.clip(0, 127), ps, (len(men) - 1) // 4)
        assert np.array_equal(
            np.column_stack([ps, deep, h, h2]), np.array([r[0] for r in native])
        )
        assert np.array_equal(v, [r[1][1] for r in native])
        counts += len(ss)
    print("native endorsements", counts)


if __name__ == "__main__":
    import sys

    verify(Path(sys.argv[1]))
