"""Finite closure certificate for the two-stopped-file theorem.

Python 3, standard library only. Run from any directory.
The infinite coverage proof is in ../two-stop-proof.md. This program alone
does not infer an infinite period from a computed prefix.

`reference_*` transcribes the full word recurrence of Elkies Theorem 3,
retaining ordinary and stopped colons. It generates only finite seed data.
The independently defined periodic extension is then substituted into every
recurrence on the complete representative box, including both colon rules.
None means a loony MOVE; it is never assigned an ordinary SG value.
"""
from collections import Counter
from functools import lru_cache
from hashlib import sha256

from single_stop_certificate import candidate as single
from single_stop_certificate import available as single_available
from single_stop_certificate import available_zero, zero, mex, verify as verify_single


def word(a, b, c):
    return '0' * a + '1' + '0' * b + '1' + '0' * c


def parameters(w):
    i, j = (k for k, bit in enumerate(w) if bit == '1')
    return i, j - i - 1, len(w) - j - 1


def word_options(w, epsilon, ordinary, stopped):
    """All initial advances, with the two special neighbouring-stop cases."""
    if len(w) <= 1:
        return {0} if w else set()
    out = set()
    for bit, tail in ((w[0], w[1:]), (w[-1], w[-2::-1])):
        value = (ordinary if bit == '0' else stopped)(tail)
        if value is not None:
            out.add(value)
    for k in range(1, len(w) - 1):
        left, right = w[:k-1], w[k+2:]
        neighbours = w[k-1] + w[k+1]
        if neighbours == '11':
            out.add(epsilon(left) ^ epsilon(right))
        elif neighbours == '10':
            value = ordinary('0' + right)
            if value is not None:
                out.add(epsilon(left) ^ value)
        elif neighbours == '01':
            value = ordinary('0' + left[::-1])
            if value is not None:
                out.add(value ^ epsilon(right))
        else:
            colon = ordinary if w[k] == '0' else stopped
            if (colon('0' + left[::-1]) is not None
                    and colon('0' + right) is not None):
                out.add(epsilon(left) ^ epsilon(right))
    return out


@lru_cache(None)
def reference_e(w):
    return mex(word_options(w, reference_e, reference_a, reference_b))


@lru_cache(None)
def reference_a(w):
    if len(w) <= 1:
        return None
    tail = w[1:]
    previous = (reference_a if w[0] == '0' else reference_b)(tail)
    return None if previous == reference_e(tail) else reference_e(tail)


@lru_cache(None)
def reference_b(w):
    assert w.startswith('0')
    if len(w) <= 2:
        return None
    tail = w[2:]
    previous = (reference_a if w[1] == '0' else reference_b)(tail)
    return reference_e(w[1:]) if previous == reference_e(w[1:]) else None


def rho(n, threshold):
    return n if n < threshold else threshold + (n - threshold) % 10


@lru_cache(None)
def extension_f(a, b, c):
    assert a >= 0 and b >= 1 and c >= 0
    return reference_e(word(rho(a, 2), rho(b, 11), rho(c, 2)))


@lru_cache(None)
def extension_e(w):
    stops = w.count('1')
    assert stops <= 2 and '11' not in w
    if stops == 0:
        return zero(len(w))
    if stops == 1:
        a = w.index('1')
        return single(a, len(w) - a - 1)
    return extension_f(*parameters(w))


@lru_cache(None)
def extension_a(w):
    if len(w) <= 1:
        return None
    stops = w.count('1')
    if stops == 0:
        allowed = available_zero(len(w))
    elif stops == 1:
        a = w.index('1')
        allowed = single_available(a, len(w) - a - 1)
    else:
        a, b, c = parameters(w)
        # Finite leading-state calculation, then the proposed period at a=3.
        previous = single(b, c)  # A(1 0^b 1 0^c), always non-loony.
        for j in range(1, rho(a, 3) + 1):
            target = extension_f(j - 1, b, c)
            previous = None if previous == target else target
        allowed = previous is not None
    return extension_e(w[1:]) if allowed else None


def extension_b(w):
    assert w.startswith('0')
    return None


def verify():
    verify_single()
    # Seeds are exact finite recurrence evaluations, without either closed form.
    seeds = [reference_e(word(a, b, c))
             for a in range(12) for b in range(1, 21) for c in range(12)]
    assert set(seeds) == set(range(6))
    print('Seed domain [0,11] x [1,20] x [0,11]:', len(seeds))
    print('Seed value counts:', dict(sorted(Counter(seeds).items())))
    print('Seed SHA256 (a,b,c order; one ASCII digit per value):',
          sha256(''.join(map(str, seeds)).encode('ascii')).hexdigest())
    counts = Counter()
    masks = []
    for a in range(24):
        for b in range(1, 28):
            for c in range(24):
                w = word(a, b, c)
                options = word_options(w, extension_e, extension_a, extension_b)
                assert mex(options) == extension_e(w), ('mex', a, b, c, options)
                assert extension_e(w) == extension_e(w[::-1]), ('reflection', a,b,c)
                masks.append(sum(1 << value for value in options))
                tail = w[1:]
                previous = (extension_a if w[0] == '0' else extension_b)(tail)
                rhs_a = None if previous == extension_e(tail) else extension_e(tail)
                assert extension_a(w) == rhs_a, ('A', a, b, c)
                counts['mex'] += 1
                counts['A'] += 1
                if a >= 1:
                    tail = w[2:]
                    previous = (extension_a if w[1] == '0' else extension_b)(tail)
                    rhs_b = (extension_e(w[1:])
                             if previous == extension_e(w[1:]) else None)
                    assert extension_b(w) == rhs_b, ('B', a, b, c)
                    counts['B'] += 1
    print('PASS: complete representative identities:', dict(counts))
    print('Option-mask SHA256 (one byte per mask):', sha256(bytes(masks)).hexdigest())
    assert extension_f(0, 5, 5) == 4
    assert extension_f(0, 5, 10) == 5
    assert extension_f(0, 1, 1) != extension_f(10, 1, 1)
    assert extension_f(0, 10, 0) != extension_f(0, 20, 0)
    high = {
        4: {(0,5,5), (0,15,0), (0,20,0), (4,1,5), (9,1,10)},
        5: {(0,5,10), (4,1,10), (5,1,9)},
    }
    for value, representatives in high.items():
        actual = {(a,b,c) for a in range(12) for b in range(1,21)
                  for c in range(12) if extension_f(a,b,c) == value}
        reflected = {(c,b,a) for a,b,c in representatives}
        assert actual == representatives | reflected
    for a in range(2, 12):
        for b in range(6, 21):
            for c in range(2, 12):
                da, dc = a % 5 == 0, c % 5 == 0
                rhs = (zero(a) ^ zero(b) ^ zero(c)
                       ^ int(b % 5 == 0 and (da or dc))
                       ^ int(b % 5 == 3 and da and dc))
                assert extension_f(a, b, c) == rhs
    print('PASS: complete high-value classes and 1500 interior identities.')
    print('Witnesses F(0,5,5)=4; F(0,5,10)=5. Exact capacity M_2=5.')
    print('Infinite conclusion uses the coverage and joint-induction proof.')


if __name__ == '__main__':
    verify()
