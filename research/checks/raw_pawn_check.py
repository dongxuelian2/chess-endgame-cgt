"""Independently check finite three-row pawn games against the proposed formulas.

Model: White pawns start on row 0, Black pawns on row 2, one per file.
Pawns move one row forward into an empty cell or capture an enemy diagonally;
captures are optional. There are no double steps or en-passant rights. Reaching
the opposite row in an unstopped file wins the entire game immediately. A pawn
reaching that row in a stopped file stays there and has no further moves.
Each global turn consists of one pawn move or a reduction of the external Nim
heap. A player without either kind of move loses.

The raw transition and win/loss recursion below do not use Elkies' Theorem 3.
Only the final comparison imports the candidate value from the companion file.
This is a finite check of the abstract pawn model, not a proof for all widths
and not a chess engine or a legal full-chess embedding.

Default: all-zero words and all single-stopped-file words of widths 1..7,
external heaps 0..5, and both starting players. Python standard library only.
"""

import argparse
from functools import lru_cache

from single_stop_certificate import candidate, zero


def raw_outcomes(word, max_heap):
    """Return outcomes for both starting players, with one cache per word."""
    width = len(word)
    assert width >= 1 and word.count('1') <= 1 and set(word) <= {'0', '1'}
    stopped = frozenset(i for i, bit in enumerate(word) if bit == '1')
    white_initial = (1 << width) - 1
    black_initial = white_initial << (2 * width)

    @lru_cache(maxsize=None)
    def wins(white, black, turn, heap):
        # Square r*width+c; White advances towards row 2, Black towards row 0.
        own, enemy = (white, black) if turn == 0 else (black, white)
        direction = 1 if turn == 0 else -1
        winning_row = 2 if turn == 0 else 0
        occupied = white | black
        unexamined = own
        while unexamined:
            bit = unexamined & -unexamined
            unexamined -= bit
            row, col = divmod(bit.bit_length() - 1, width)
            next_row = row + direction
            if not 0 <= next_row <= 2:
                continue

            destinations = []
            straight = 1 << (next_row * width + col)
            if not occupied & straight:
                destinations.append((col, straight, False))
            for next_col in (col - 1, col + 1):
                if 0 <= next_col < width:
                    target = 1 << (next_row * width + next_col)
                    if enemy & target:
                        destinations.append((next_col, target, True))

            for next_col, target, capture in destinations:
                if next_row == winning_row and next_col not in stopped:
                    return True
                moved = (own ^ bit) | target
                remaining_enemy = enemy ^ target if capture else enemy
                next_white, next_black = ((moved, remaining_enemy) if turn == 0
                                           else (remaining_enemy, moved))
                if not wins(next_white, next_black, 1 - turn, heap):
                    return True

        for smaller_heap in range(heap):
            if not wins(white, black, 1 - turn, smaller_heap):
                return True
        return False

    outcomes = [[wins(white_initial, black_initial, turn, heap)
                 for heap in range(max_heap + 1)] for turn in (0, 1)]
    cache_info = wins.cache_info()
    wins.cache_clear()
    return outcomes, cache_info


def verify(max_width=7, max_heap=5):
    words_checked = 0
    outcomes_checked = 0
    states_checked = 0
    print('word expected losing_heaps_white losing_heaps_black cached_states')
    for width in range(1, max_width + 1):
        words = ['0' * width]
        words += ['0' * a + '1' + '0' * (width - a - 1)
                  for a in range(width)]
        for word in words:
            if '1' in word:
                a = word.index('1')
                expected = candidate(a, width - a - 1)
            else:
                expected = zero(width)

            actual, info = raw_outcomes(word, max_heap)
            losing_heaps = [[heap for heap, won in enumerate(row) if not won]
                            for row in actual]
            predicted = [expected] if expected <= max_heap else []
            assert losing_heaps == [predicted, predicted], (
                word, expected, losing_heaps)
            print(word, expected, losing_heaps[0], losing_heaps[1], info.currsize)
            words_checked += 1
            outcomes_checked += 2 * (max_heap + 1)
            states_checked += info.currsize
    print(f'PASS: {words_checked} words, {outcomes_checked} starting outcomes, '
          f'{states_checked} cached raw states.')
    print('Finite verification only; the general proof is the residue argument.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-width', type=int, default=7)
    parser.add_argument('--max-heap', type=int, default=5)
    args = parser.parse_args()
    if args.max_width < 1 or args.max_heap < 0:
        parser.error('max-width must be positive and max-heap nonnegative')
    verify(args.max_width, args.max_heap)
