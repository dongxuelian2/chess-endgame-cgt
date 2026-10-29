"""Exact finite residue certificate for the single stopped-file formulas.

This substitutes closed-form candidates into Elkies' specialized recurrence.
It does not compute a finite prefix and infer periodicity from that prefix.
The proof that [0,24]^2 covers all cases is in single-stop-classification.md.
Python 3, standard library only. Run from any directory; --table prints masks.
"""
import argparse

H = (1, 3, 0, 2, 1, 0, 2, 1, 3, 0)
X = (3, 0, 1, 1, 0, 2, 1, 0, 0, 1)


def zero(n):
    assert n >= 0
    return int(n % 10 not in (0, 2, 3, 6, 9))


def available_zero(n):
    assert n >= 1
    return n % 5 in (0, 2, 3)


def endpoint(n):
    assert n >= 0
    return 1 if n == 0 else 0 if n == 1 else H[n % 10]


def near_endpoint(n):
    assert n >= 0
    return 0 if n == 0 else X[n % 10]


def candidate(a, b):
    """F(a,b); a=-1 or b=-1 is the explicitly defined zero-tail sentinel."""
    if a == -1:
        assert b >= 0
        return zero(b)
    if b == -1:
        assert a >= 0
        return zero(a)
    assert a >= 0 and b >= 0
    if a == 0:
        return endpoint(b)
    if b == 0:
        return endpoint(a)
    if a == 1:
        return near_endpoint(b)
    if b == 1:
        return near_endpoint(a)
    return zero(a) ^ zero(b) ^ int(a % 5 == b % 5 == 0)


def available(a, b):
    """Candidate C(a,b): non-loony move to ordinary colon 0^a 1 0^b."""
    assert a >= 0 and b >= 0
    if a == 0:
        return b >= 1
    if a == 1:
        return b % 5 in (0, 1, 3)
    if a == 2:
        return b != 1
    if b == 0:
        return a != 3
    if b == 1:
        return a % 5 != 4
    if a % 5 in (0, 3):
        return True
    if a % 5 == 4:
        return False
    return (b % 5 == 0) if a % 5 == 1 else (b % 5 != 0)


def substituted_options(a, b):
    """All six move classes, evaluated on the closed-form candidates."""
    assert a >= 0 and b >= 0
    if a == b == 0:
        return {0}
    out = set()
    if a >= 1 and available(a - 1, b):
        out.add(candidate(a - 2, b))
    if b >= 1 and available(b - 1, a):
        out.add(candidate(a, b - 2))
    if a >= 2 and available_zero(a - 1):
        out.add(zero(a - 2) ^ zero(b))
    if b >= 2 and available_zero(b - 1):
        out.add(zero(a) ^ zero(b - 2))
    for r in range(a - 2):
        p = a - 3 - r
        if available_zero(p + 1) and available(r + 1, b):
            out.add(zero(p) ^ candidate(r, b))
    for s in range(b - 2):
        q = b - 3 - s
        if available_zero(q + 1) and available(s + 1, a):
            out.add(candidate(a, s) ^ zero(q))
    return out


def mex(values):
    n = 0
    while n in values:
        n += 1
    return n


def verify():
    # Known Theorem 2 also implies every stopped-colon zero tail is loony.
    for t in range(1, 11):
        assert not (available_zero(t) and zero(t - 1) == zero(t + 1))
    for a in range(25):
        for b in range(25):
            if a == 0:
                assert available(a, b) == (b >= 1)
            else:
                rhs = not (available(a - 1, b)
                           and candidate(a - 2, b) == candidate(a - 1, b))
                assert available(a, b) == rhs, ('C', a, b)
            moves = substituted_options(a, b)
            assert all(0 <= v <= 3 for v in moves)
            assert mex(moves) == candidate(a, b), ('F', a, b, moves)
    return [[sum(1 << v for v in substituted_options(a, b))
             for b in range(25)] for a in range(25)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', action='store_true')
    args = parser.parse_args()
    masks = verify()
    print('PASS: 625 C identities, 625 mex identities, stopped-colon lemma.')
    print('Infinite coverage requires the residue-reduction proof in the note.')
    if args.table:
        print('Rows a=0..24; columns b=0..24; hex mask=sum(2**v for v in options).')
        for a, row in enumerate(masks):
            print(f'{a:2d} ' + ''.join(format(mask, 'x') for mask in row))
