#include "evaluate.h"
#include "movegen.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Stockfish {
int atlas_hand(const Position &, bool);
namespace Eval::NNUE {
extern int atlas_ps, atlas_deep;
}

struct AtlasEntry {
  int16_t classical, full, neural, psqt, deep;
  int8_t wdl;
  uint8_t valid;
};
static_assert(sizeof(AtlasEntry) == 12);

void atlas_main(char **argv) {
  const PieceType actor = PieceType(std::atoi(argv[1])),
                  enemy = PieceType(std::atoi(argv[2]));
  Tablebases::init(argv[3]);
  FILE *out = std::fopen(argv[4], "wb");
  if (!out)
    std::abort();
  const int count = std::atoi(argv[5]);
  uint64_t valid = 0, terminal = 0;
  Position pos;
  StateInfo st;
  std::vector<AtlasEntry> rows(64 * 64 * 64);
  for (int k = 0; k < count; k++) {
    Square wk = Square(k);
    std::fill(rows.begin(), rows.end(), AtlasEntry{});
    for (Square bk = SQ_A1; bk <= SQ_H8; ++bk) {
      if (distance(wk, bk) <= 1)
        continue;
      for (Square y = SQ_A1; y <= SQ_H8; ++y) {
        if (y == bk || y == wk ||
            (enemy == PAWN && (rank_of(y) == RANK_1 || rank_of(y) == RANK_8)))
          continue;
        for (Square x = SQ_A1; x <= SQ_H8; ++x) {
          if (x == wk || x == bk || x == y ||
              (actor == PAWN && (rank_of(x) == RANK_1 || rank_of(x) == RANK_8)))
            continue;
          Bitboard occ =
              square_bb(wk) | square_bb(bk) | square_bb(x) | square_bb(y);
          if ((actor == PAWN ? pawn_attacks_bb(WHITE, x)
                             : attacks_bb(actor, x, occ)) &
              bk)
            continue;
          Bitboard theirs = enemy == PAWN ? pawn_attacks_bb(BLACK, y)
                                          : attacks_bb(enemy, y, occ);
          if (theirs & wk)
            continue;
          pos.atlas_set(wk, bk, make_piece(WHITE, actor), x,
                        make_piece(BLACK, enemy), y, &st, Threads.main());
          if (MoveList<LEGAL>(pos).size() == 0) {
            ++terminal;
            continue;
          }
          Tablebases::ProbeState state;
          int w = int(Tablebases::probe_wdl(pos, &state));
          if (state == Tablebases::FAIL)
            std::abort();
          int h = atlas_hand(pos, false), f = atlas_hand(pos, true),
              n = int(Eval::NNUE::evaluate(pos, false));
          auto &r = rows[(int(bk) * 64 + y) * 64 + x];
          r = {int16_t(h),
               int16_t(f),
               int16_t(n),
               int16_t(Eval::NNUE::atlas_ps),
               int16_t(Eval::NNUE::atlas_deep),
               int8_t(w),
               1};
          ++valid;
        }
      }
    }
    if (std::fwrite(rows.data(), sizeof(AtlasEntry), rows.size(), out) !=
        rows.size())
      std::abort();
  }
  if (std::fclose(out) != 0)
    std::abort();
  std::cout << "atlas " << int(actor) << " " << int(enemy) << " " << valid
            << " " << terminal << std::endl;
}
} // namespace Stockfish
