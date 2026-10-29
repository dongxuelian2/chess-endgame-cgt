"""Independent self-contained two-stop residue certificate.

Seeds use a separately written per-file Theorem 3 implementation retaining B.
The verification uses the specialized parameter recurrence and periodic candidates.
This is an exact computer-assisted finite check; infinite coverage is proved in
research/two-stop-proof.md. No repository check module is imported.
"""
from collections import Counter
from functools import cache
from hashlib import sha256

BOT = -1


def mex(values):
    n = 0
    while n in values:
        n += 1
    return n


@cache
def raw_a(t):
    if len(t) < 2:
        return BOT
    value = raw_d(t[0], t[1:])
    target = raw_g(t[1:])
    return BOT if value == target else target


@cache
def raw_b(t):
    assert t and t[0] == '0'
    if len(t) < 3:
        return BOT
    value = raw_d(t[1], t[2:])
    target = raw_g(t[1:])
    return target if value == target else BOT


def raw_d(bit, t):
    return raw_b(t) if bit == '1' else raw_a(t)


@cache
def raw_g(t):
    if not t:
        return 0
    if len(t) == 1:
        return 1
    out = set()
    for j, bit in enumerate(t):
        if j == 0:
            value = raw_d(bit, t[1:])
        elif j == len(t) - 1:
            value = raw_d(bit, t[:j][::-1])
        else:
            left, right = t[:j], t[j+1:]
            if bit == '0' and left[-1] == right[0] == '1':
                value = raw_g(left[:-1]) ^ raw_g(right[1:])
            elif bit == '0' and left[-1] == '1':
                q = raw_a(right)
                value = BOT if q == BOT else raw_g(left[:-1]) ^ q
            elif bit == '0' and right[0] == '1':
                q = raw_a(left[::-1])
                value = BOT if q == BOT else raw_g(right[1:]) ^ q
            else:
                p, q = raw_d(bit, left[::-1]), raw_d(bit, right)
                value = BOT if BOT in (p, q) else p ^ q
        if value != BOT:
            out.add(value)
    return mex(out)


H = (1, 3, 0, 2, 1, 0, 2, 1, 3, 0)
X = (3, 0, 1, 1, 0, 2, 1, 0, 0, 1)


def zero(n):
    return int(n % 10 not in (0, 2, 3, 6, 9))


def dzero(n):
    return n % 5 in (0, 2, 3)


def single(a, b):
    if a == 0:
        return 1 if b == 0 else 0 if b == 1 else H[b % 10]
    if b == 0:
        return single(0, a)
    if a == 1:
        return X[b % 10]
    if b == 1:
        return single(1, a)
    return zero(a) ^ zero(b) ^ int(a % 5 == b % 5 == 0)


def colon_single(a, b):
    if a == 0:
        return b > 0
    if a == 1:
        return b % 5 in (0, 1, 3)
    if a == 2:
        return b != 1
    if b == 0:
        return a != 3
    if b == 1:
        return a % 5 != 4
    return (a % 5 in (0, 3) or (a % 5 == 1 and b % 5 == 0)
            or (a % 5 == 2 and b % 5 != 0))


def rho(n, threshold):
    return n if n < threshold else threshold + (n-threshold) % 10


def verify():
    seed = {(a, b, c): raw_g('0'*a + '1' + '0'*b + '1' + '0'*c)
            for a in range(12) for b in range(1, 21) for c in range(12)}

    def candidate(a, b, c):
        if a == -1:
            return single(b, c)
        if c == -1:
            return single(a, b)
        return seed[rho(a, 2), rho(b, 11), rho(c, 2)]

    def colon(a, b, c):
        yes = True
        for j in range(1, rho(a, 3)+1):
            yes = not (yes and candidate(j-2, b, c) == candidate(j-1, b, c))
        return yes

    def options(a, b, c):
        out = set()
        if a >= 1 and colon(a-1, b, c):
            out.add(candidate(a-2, b, c))
        if c >= 1 and colon(c-1, b, a):
            out.add(candidate(a, b, c-2))
        if a >= 2 and dzero(a-1):
            out.add(zero(a-2) ^ single(b, c))
        if c >= 2 and dzero(c-1):
            out.add(single(a, b) ^ zero(c-2))
        for r in range(a-2):
            p = a-3-r
            if dzero(p+1) and colon(r+1, b, c):
                out.add(zero(p) ^ candidate(r, b, c))
        for r in range(c-2):
            p = c-3-r
            if dzero(p+1) and colon(r+1, b, a):
                out.add(candidate(a, b, r) ^ zero(p))
        if b == 1:
            out.add(zero(a) ^ zero(c))
        else:
            if colon_single(b-1, c):
                out.add(zero(a) ^ single(b-2, c))
            if colon_single(b-1, a):
                out.add(single(a, b-2) ^ zero(c))
            for p in range(b-2):
                q = b-3-p
                if colon_single(p+1, a) and colon_single(q+1, c):
                    out.add(single(a, p) ^ single(q, c))
        return out

    masks = []
    for a in range(24):
        for b in range(1, 28):
            for c in range(24):
                out = options(a, b, c)
                assert mex(out) == candidate(a, b, c), (a, b, c, out)
                if a:
                    assert colon(a, b, c) == (not (
                        colon(a-1, b, c)
                        and candidate(a-2, b, c) == candidate(a-1, b, c)))
                masks.append(sum(1 << value for value in out))
    for a in range(2, 12):
        for b in range(6, 21):
            for c in range(2, 12):
                formula = (zero(a) ^ zero(b) ^ zero(c)
                           ^ int(b % 5 == 0 and (a % 5 == 0 or c % 5 == 0))
                           ^ int(b % 5 == 3 and a % 5 == c % 5 == 0))
                assert candidate(a, b, c) == formula, ('interior', a, b, c)
    assert candidate(5, 5, 5) == 2
    assert candidate(0, 5, 10) == 5
    print('PASS: 15552 mex and 14904 positive-leading colon identities.')
    print('PASS: 1500 interior closed-form identities.')
    print('Seed distribution:', sorted(Counter(seed.values()).items()))
    print('ASCII-digit seed SHA256:', sha256(''.join(map(str, seed.values())).encode()).hexdigest())
    print('Byte-mask SHA256:', sha256(bytes(masks)).hexdigest())
    return seed, masks


if __name__ == '__main__':
    verify()

