"""Minimum-cost, state-consistent, one-sided leaf intervention in a MAX/MIN/MAX game."""

import itertools
from fractions import Fraction

import numpy as np


def evaluate(tree, values):
    return np.array(
        [min(max(values[l] for l in row) for row in action) for action in tree]
    )


def certificate(tree, high, low, leader):
    """Rows are sets of distinct state ids. Exclude each candidate row from all rivals."""
    assert np.all(low <= high)
    others = [act for i, act in enumerate(tree) if i != leader]
    best = None
    relaxed = None
    rival = max(evaluate(others, high))
    details = []
    for j, row in enumerate(tree[leader]):
        row = set(row)
        escaping = max(
            min(
                max(
                    (int(high[l]) for l in branch if l not in row),
                    default=-float("inf"),
                )
                for branch in act
            )
            for act in others
        )
        floor = max(int(low[l]) for l in row)
        if floor < rival:
            cost = sum(high[l] >= rival for l in row)
            relaxed = cost if relaxed is None else min(relaxed, cost)
        required = None
        if floor < escaping:
            required = sorted(l for l in row if high[l] >= escaping)
            if best is None or len(required) < len(best[1]):
                best = (j, required)
        details.append(
            (
                j,
                None if escaping == -float("inf") else int(escaping),
                floor,
                None if required is None else len(required),
            )
        )
    return best, relaxed, details


def frontier(rows, hi, lo, details):
    front = []
    for j, escape, _, cost in details:
        if cost is None or cost == 0:
            continue
        required = {i for i in rows[j] if hi[i] >= escape}
        cut = max(Fraction(int(hi[i]) - escape, int(hi[i] - lo[i])) for i in required)
        front.append((cut, cost))
    return front


def verify_random():
    rng = np.random.default_rng(38824)
    tested = 0
    for _ in range(3000):
        n = int(rng.integers(2, 10))
        high = rng.integers(-5, 6, n)
        low = high - rng.integers(0, 8, n)
        tree = [
            [
                sorted(set(map(int, rng.integers(n, size=rng.integers(1, 5)))))
                for _ in range(rng.integers(1, 4))
            ]
            for _ in range(rng.integers(2, 5))
        ]
        values = evaluate(tree, high)
        leader = int(values.argmax())
        best, _, details = certificate(tree, high, low, leader)
        target = None
        for k in range(n + 1):
            for selected in itertools.combinations(range(n), k):
                v = high.copy()
                ix = list(selected)
                v[ix] = low[ix]
                scores = evaluate(tree, v)
                if scores.max() > scores[leader]:
                    target = k
                    break
            if target is not None:
                break
        assert (None if best is None else len(best[1])) == target
        if tested < 400:
            front = frontier(tree[leader], high, low, details)
            cuts = {Fraction(j, 4) for j in range(5)} | {t for t, _ in front}
            for strength in cuts:
                d, n = strength.denominator, strength.numerator
                partial = high * d - (high - low) * n
                reference, _, _ = certificate(tree, high * d, partial, leader)
                expect = None if reference is None else len(reference[1])
                assert (
                    min((cost for t, cost in front if t < strength), default=None)
                    == expect
                )
        tested += 1
    print("exact subset enumeration:", tested)


if __name__ == "__main__":
    verify_random()
