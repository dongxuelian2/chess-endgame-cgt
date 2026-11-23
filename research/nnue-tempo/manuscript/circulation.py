"""Graph projections and rational minimum-energy interaction certificates."""

from fractions import Fraction
from itertools import combinations

import numpy as np


def directed_cycles(start, end):
    """Enumerate simple directed cycles, retaining the identities of parallel arcs."""
    adjacency = {int(u): [] for u in set(start) | set(end)}
    for i, (u, v) in enumerate(zip(start, end)):
        adjacency[u].append((int(v), i))

    def visit(first, last, seen, path):
        for vertex, edge in adjacency[last]:
            if vertex == first:
                yield path + (edge,)
            elif vertex > first and vertex not in seen:
                yield from visit(first, vertex, seen | {vertex}, path + (edge,))

    return [
        path for root in sorted(adjacency) for path in visit(root, root, {root}, ())
    ]


def linear_solve(matrix, rhs):
    """Solve an independent small system over the rationals."""
    aug = [
        [Fraction(int(v)) for v in row] + [Fraction(int(b))]
        for row, b in zip(matrix, rhs)
    ]
    n = len(aug)
    for j in range(n):
        pivot = next((i for i in range(j, n) if aug[i][j]), None)
        if pivot is None:
            return None
        aug[j], aug[pivot] = aug[pivot], aug[j]
        d = aug[j][j]
        aug[j] = [e / d for e in aug[j]]
        for i in range(n):
            if i != j and aug[i][j]:
                f = aug[i][j]
                aug[i] = [x - f * y for x, y in zip(aug[i], aug[j])]
    return [row[-1] for row in aug]


def energy_certificate(cycles, gap):
    """Return exact squared norm and a sparse nonnegative cycle decomposition."""
    edge_sets = [set(path) for path in cycles]
    target = [sum(int(gap[i]) for i in path) for path in cycles]
    if max(target) <= 0:
        return Fraction(0), []
    matrix = [[len(c & d) for d in edge_sets] for c in edge_sets]
    for width in range(1, len(cycles) + 1):
        for idx in combinations(range(len(cycles)), width):
            beta = linear_solve(
                [[matrix[i][j] for j in idx] for i in idx], [target[i] for i in idx]
            )
            if beta is None or min(beta) < 0:
                continue
            tot = [
                sum(matrix[i][j] * b for j, b in zip(idx, beta))
                for i in range(len(cycles))
            ]
            if any(z < wanted for z, wanted in zip(tot, target)):
                continue
            norm = sum(target[j] * b for j, b in zip(idx, beta))
            assert norm > 0
            return norm, [(j, b) for j, b in zip(idx, beta) if b]
    raise AssertionError("Missing energy certificate")


def verify_energy(start, end, cycles, gap, norm, certificate):
    """Reconstruct a feasible primal potential and check dual equality exactly."""
    correction = [Fraction(0) for _ in gap]
    for j, coefficient in certificate:
        for i in cycles[j]:
            correction[i] += coefficient
    assert sum(r * r for r in correction) == norm
    assert sum(r * int(b) for r, b in zip(correction, gap)) == norm
    gain = [int(b) - r for b, r in zip(gap, correction)]
    potentials = {u: Fraction(0) for u in set(start) | set(end)}
    for _ in range(len(potentials) - 1):
        for u, v, w in zip(start, end, gain):
            potentials[v] = max(potentials[v], potentials[u] + w)
    assert all(potentials[v] - potentials[u] >= w for u, v, w in zip(start, end, gain))


def interaction_projection(start, end, margin):
    """Orthogonally remove all edge margins realizable by a common destination table."""
    nodes = sorted(set(start) | set(end))
    index = {a: i for i, a in enumerate(nodes)}
    B = np.zeros((len(start), len(nodes)))
    for e, (u, v) in enumerate(zip(start, end)):
        B[e, index[u]], B[e, index[v]] = -1, 1
    fit = B @ np.linalg.lstsq(B, margin, rcond=None)[0]
    res = margin - fit
    assert np.allclose(B.T @ res, 0, atol=1e-7)
    assert np.allclose(np.sum(margin**2, axis=0), np.sum(fit**2 + res**2, axis=0))
    return res
