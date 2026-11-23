"""Projected removal of the clipping interaction of a legal pawn capture pair."""

from collections import defaultdict

import numpy as np


def suppress(net, boards, pairs, baseline):
    groups = defaultdict(list)
    for i, choices in pairs.items():
        if len(choices) != 1:
            continue
        pawn, target = next(iter(choices))
        b = boards[i]
        others = sorted(
            (p.piece_type if p.color else -p.piece_type, s)
            for s, p in b.piece_map().items()
            if p.piece_type != 6 and s not in (pawn, target)
        )
        men = (6, -6, -1, b.piece_type_at(target), *(p for p, s in others))
        groups[men].append(
            (i, [b.king(True), b.king(False), pawn, target, *(s for p, s in others)])
        )
    alt = baseline.copy()
    for men, samples in groups.items():
        for start in range(0, len(samples), 128):
            idx, sq = zip(*samples[start : start + 128])
            raw, ps = net.read(sq, men, need_raw=True)
            addresses = net.indices(sq, men)
            pawn, target = (
                np.concatenate([net.weights[a[:, j]] for a in addresses], axis=1)
                for j in (2, 3)
            )

            def clip(x):
                return np.clip(x, 0, 127)

            separated = clip(
                clip(raw - pawn) + clip(raw - target) - clip(raw - pawn - target)
            )
            stage = (len(men) - 1) // 4
            native, *_ = net.score(clip(raw), ps, stage)
            assert np.array_equal(-native, baseline[list(idx)])
            score, *_ = net.score(separated, ps, stage)
            alt[list(idx)] = -score
    return alt
