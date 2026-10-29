// Independent finite verification of the three-row pawn rules.
// No Theorem 3 recurrence or candidate formula is used in the search.
// Example: clang++ -O3 -std=c++11 raw_pawn_witness.cpp -o raw_pawn_witness
//          ./raw_pawn_witness 00000100000 7
// Only finite Nim tests are checked; the existence of the quiescent nimber
// still comes from Elkies' Theorem 3, not from these finite outcomes.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <stdexcept>

using U = uint64_t;
static unsigned width, stops;
static U mask, stopped;
static std::vector<unsigned> stopped_columns;
struct Key {
    U first, second;
    bool operator==(const Key& other) const {
        return first == other.first && second == other.second;
    }
};
struct KeyHash {
    size_t operator()(const Key& value) const {
        U mixed = value.first ^ (value.second + 0x9e3779b97f4a7c15ULL +
                                 (value.first << 6) + (value.first >> 2));
        mixed ^= mixed >> 30;
        mixed *= 0xbf58476d1ce4e5b9ULL;
        mixed ^= mixed >> 27;
        mixed *= 0x94d049bb133111ebULL;
        return mixed ^ (mixed >> 31);
    }
};
static std::unordered_map<Key, bool, KeyHash> cache;
static size_t state_limit = 8000000;
static U calls, hits, immediate;

static U reverse_rows(U board) {
    return ((board & mask) << (2 * width)) |
           (board & (mask << width)) | (board >> (2 * width));
}

// Normalize the player to move as White, advancing row 0 -> row 2.
// Terminal-row pawns exist only on stopped files. Four ordinary row masks
// and two stopped-file masks therefore encode every reachable raw board.
// Wider words use two full board masks in a 128-bit key instead.
static Key key(U own, U enemy, unsigned heap) {
    if (4 * width + 2 * stops + 4 > 64)
        return {own, enemy | (U(heap) << (3 * width))};
    U result = (own & mask) | (((own >> width) & mask) << width) |
               (((enemy >> width) & mask) << (2 * width)) |
               (((enemy >> (2 * width)) & mask) << (3 * width));
    for (unsigned i = 0; i < stops; ++i) {
        result |= ((own >> (2 * width + stopped_columns[i])) & 1)
                  << (4 * width + i);
        result |= ((enemy >> stopped_columns[i]) & 1)
                  << (4 * width + stops + i);
    }
    return {result | (U(heap) << (4 * width + 2 * stops)), 0};
}

static bool wins(U own, U enemy, unsigned heap) {
    ++calls;
    const U middle = (own >> width) & mask;
    const U enemy_top = (enemy >> (2 * width)) & mask;
    const U occupied_top = ((own | enemy) >> (2 * width)) & mask;
    const U finishes = (middle & ~occupied_top) |
                       ((middle << 1) & enemy_top) |
                       ((middle >> 1) & enemy_top);
    if (finishes & mask & ~stopped) {
        ++immediate;
        return true;
    }
    Key packed = key(own, enemy, heap);
    auto found = cache.find(packed);
    if (found != cache.end()) {
        ++hits;
        return found->second;
    }
    if (cache.size() >= state_limit)
        throw std::runtime_error("raw search reached the finite state budget");
    U occupied = own | enemy;
    U unexamined = own;
    while (unexamined) {
        unsigned square = __builtin_ctzll(unexamined);
        unexamined &= unexamined - 1;
        unsigned row = square / width, col = square % width;
        if (row == 2) continue;
        U from = U(1) << square;
        U straight = from << width;
        if (!(occupied & straight) &&
            !wins(reverse_rows(enemy), reverse_rows((own ^ from) | straight), heap))
            return cache[packed] = true;
        if (col > 0) {
            U target = from << (width - 1);
            if ((enemy & target) &&
                !wins(reverse_rows(enemy ^ target),
                      reverse_rows((own ^ from) | target), heap))
                return cache[packed] = true;
        }
        if (col + 1 < width) {
            U target = from << (width + 1);
            if ((enemy & target) &&
                !wins(reverse_rows(enemy ^ target),
                      reverse_rows((own ^ from) | target), heap))
                return cache[packed] = true;
        }
    }
    for (unsigned smaller = 0; smaller < heap; ++smaller)
        if (!wins(reverse_rows(enemy), reverse_rows(own), smaller))
            return cache[packed] = true;
    return cache[packed] = false;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: raw_pawn_witness WORD [MAX_HEAP=7] [MAX_STATES=8000000]\n";
        return 2;
    }
    std::string word = argv[1];
    width = word.size();
    if (!width || width > 20) return 2;
    unsigned max_heap = argc >= 3 ? std::stoul(argv[2]) : 7;
    if (argc == 4) state_limit = std::stoull(argv[3]);
    for (unsigned i = 0; i < width; ++i) {
        if (word[i] == '1') {
            stopped |= U(1) << i;
            stopped_columns.push_back(i);
        } else if (word[i] != '0') return 2;
    }
    stops = stopped_columns.size();
    if (word.find("11") != std::string::npos || max_heap > 15) return 2;
    mask = (U(1) << width) - 1;
    auto start = std::chrono::steady_clock::now();
    try {
        for (unsigned heap = 0; heap <= max_heap; ++heap) {
            bool won = wins(mask, mask << (2 * width), heap);
            // Vertical rank reflection plus exchanging colours fixes the
            // initial board and every stopped file, so both starters agree.
            std::cout << word << " heap=" << heap << " both_starts="
                      << (won ? "win" : "loss") << " states=" << cache.size()
                      << std::endl;
        }
    } catch (const std::runtime_error& error) {
        std::cerr << error.what() << "; states=" << cache.size() << std::endl;
        return 3;
    }
    std::cout << "calls=" << calls << " cache_hits=" << hits
              << " immediate_wins=" << immediate << " seconds="
              << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << std::endl;
}
