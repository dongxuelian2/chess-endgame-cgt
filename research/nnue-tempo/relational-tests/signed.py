"""Exact signed binary interventions on shared states at depth three."""

from functools import lru_cache
from itertools import combinations

import numpy as np

from certificate import evaluate


def cover(rows):
    """Minimum unit-cost cover of reply rows by distinct eligible leaf states."""
    masks = {}
    for j, row in enumerate(rows):
        if not row:
            return None
        for leaf in row:
            masks[leaf] = masks.get(leaf, 0) | (1 << j)
    # Equal and dominated masks have an equally cheap superset representative.
    representative = {mask: leaf for leaf, mask in masks.items()}
    maximal = []
    for mask in sorted(representative, key=int.bit_count, reverse=True):
        if not any(mask & m == mask for m in maximal):
            maximal.append(mask)
    options = [[m for m in maximal if m & (1 << j)] for j in range(len(rows))]

    @lru_cache(None)
    def solve(unmet):
        if not unmet:
            return ()
        choices = min((ms for j, ms in enumerate(options) if unmet >> j & 1), key=len)
        return min(((representative[m],) + solve(unmet & ~m) for m in choices), key=len)

    return list(solve((1 << len(rows)) - 1))


def signed_certificate(tree, base, alternative, leader):
    best = None
    lower = np.minimum(base, alternative)
    upper = np.maximum(base, alternative)
    high = evaluate(tree, upper)
    for j, row0 in enumerate(tree[leader]):
        row = set(row0)
        floor = max(lower[l] for l in row)
        ceiling = max(base[l] for l in row)
        thresholds = sorted(
            {
                int(s) + 1
                for l in row
                for s in (base[l], alternative[l])
                if floor <= s <= ceiling
            },
            reverse=True,
        )
        for t in thresholds:
            lower_set = {l for l in row if base[l] >= t}
            if any(alternative[l] >= t for l in lower_set):
                continue
            if best is not None and len(lower_set) >= len(best[1]):
                continue
            for rival, action in enumerate(tree):
                if rival == leader or high[rival] < t:
                    continue
                needed = []
                for branch in action:
                    if any(base[l] >= t and l not in lower_set for l in branch):
                        continue
                    possible = {
                        l for l in branch if l not in row and alternative[l] >= t
                    }
                    if not possible:
                        break
                    needed.append(possible)
                else:
                    upgrades = cover(needed)
                    if upgrades is None:
                        continue
                    selected = sorted(lower_set | set(upgrades))
                    if best is None or len(selected) < len(best[1]):
                        best = (j, selected, rival, t)
    return best


def verify():
    rng = np.random.default_rng(187011)
    for it in range(5000):
        n = int(rng.integers(2, 10))
        tree = [
            [
                sorted(set(map(int, rng.integers(n, size=rng.integers(1, 6)))))
                for _ in range(rng.integers(1, 5))
            ]
            for _ in range(rng.integers(2, 5))
        ]
        base = rng.integers(-5, 6, n)
        alt = rng.integers(-5, 6, n)
        leader = int(evaluate(tree, base).argmax())
        w = signed_certificate(tree, base, alt, leader)
        gold = None
        for k in range(n + 1):
            for s in combinations(range(n), k):
                x = base.copy()
                s = list(s)
                x[s] = alt[s]
                v = evaluate(tree, x)
                if v.max() > v[leader]:
                    gold = k
                    break
            if gold is not None:
                break
        assert (None if w is None else len(w[1])) == gold
        if w:
            x = base.copy()
            x[w[1]] = alt[w[1]]
            v = evaluate(tree, x)
            assert v[w[2]] >= w[3] > v[leader]
    print("signed exact subset checks", it + 1)


if __name__ == "__main__":
    verify()
