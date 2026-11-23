#include <algorithm>
#include <fstream>
#include <limits>
#include <string>
namespace Stockfish {
enum Rule {
#define RULE(id, text) id,
#include "rules.def"
#undef RULE
  rules_count
};
using Mask = uint64_t;
static_assert(rules_count < 64);
constexpr Mask domain = (Mask(1) << rules_count) - 1;
Bitboard control(PieceType pt, Color side, Square at, Bitboard occ) {
  return pt == PAWN ? pawn_attacks_bb(side, at) : attacks_bb(pt, at, occ);
}
bool separates(Square left, Square mid, Square right) {
  return left != mid && mid != right && (between_bb(left, right) & mid);
}
bool opposition(Square a, Square b, bool diag) {
  int f = std::abs(file_of(a) - file_of(b)),
      r = std::abs(rank_of(a) - rank_of(b));
  return diag ? f == 2 && r == 2 : (f == 0 && r == 2) || (r == 0 && f == 2);
}
bool cutoff(Square k1, Square k2, Square man, Bitboard arm) {
  const Bitboard area = arm | square_bb(man);
  auto split = [](int a, int b, int c) {
    return (a < b && b < c) || (c < b && b < a);
  };
  return (split(rank_of(k1), rank_of(man), rank_of(k2)) &&
          (area & rank_bb(man)) == rank_bb(man)) ||
         (split(file_of(k1), file_of(man), file_of(k2)) &&
          (area & file_bb(man)) == file_bb(man));
}
struct Geometry {
  Square wk, bk, x, y;
  PieceType a, b;
  Bitboard occ, white, black;
  bool alive, guard, enemy_guard, pinned, enemy_pin, danger;
  Geometry(Square wk_, Square bk_, Square x_, Square y_, PieceType a_,
           PieceType b_, bool live)
      : wk(wk_), bk(bk_), x(x_), y(y_), a(a_), b(b_), alive(live) {
    occ = square_bb(wk) | bk | y;
    if (alive)
      occ |= x;
    white = alive ? control(a, WHITE, x, occ) : 0;
    black = control(b, BLACK, y, occ);
    guard = distance(bk, y) == 1;
    enemy_guard = alive && distance(wk, x) == 1;
    pinned = alive && separates(x, y, bk) &&
             (control(a, WHITE, x, occ ^ square_bb(y)) & bk);
    enemy_pin = alive && separates(y, x, wk) &&
                (control(b, BLACK, y, occ ^ square_bb(x)) & wk);
    danger = (white & y) || distance(wk, y) == 1;
  }
};
Mask classify(const Geometry &g, const Geometry &z, Move m, bool checked,
              bool end, bool recaptured) {
  bool king = from_sq(m) == g.bk, take = !z.alive,
       promotion = type_of(m) == PROMOTION;
  bool quiet = !king && !take && !promotion, pawn = g.b == PAWN,
       white_pawn = z.alive && z.a == PAWN;
  bool attacks = z.alive && (z.black & z.x),
       fresh = attacks && !(g.black & g.x);
  bool approach =
      z.alive && distance(z.bk, z.x) == 1 && distance(g.bk, g.x) > 1;
  bool major = g.b == ROOK || g.b == QUEEN,
       wmajor = g.a == ROOK || g.a == QUEEN;
  bool kcheck = checked && !king;
  Mask mask = 0;
  auto put = [&](Rule r, bool active) {
    if (active)
      mask |= Mask(1) << r;
  };
  put(king_takes_actor, king && take);
  put(man_takes_actor, !king && take);
  put(trade_last_men, !king && take && recaptured);
  put(clean_man_capture, !king && take && !recaptured);
  put(quiet_promotion, promotion && !take);
  put(capture_promotion, promotion && take);
  put(underpromotion, promotion && promotion_type(m) != QUEEN);
  put(checking_promotion, promotion && checked);
  put(surviving_promotion, promotion && !recaptured);
  put(double_push, quiet && pawn && g.y - z.y == 16);
  put(single_push, quiet && pawn && g.y - z.y == 8);
  put(pawn_chase, quiet && pawn && attacks);
  put(king_unblocks_pawn, king && pawn && g.bk == g.y - 8);
  put(escort_promotion, king && pawn &&
                            distance(z.bk, Square(file_of(z.y))) == 1 &&
                            distance(g.bk, Square(file_of(z.y))) > 1);
  put(king_escorts_pawn, king && pawn && z.guard && !g.guard);
  put(direct_check, kcheck);
  put(discovered_check, king && checked);
  put(contact_check, kcheck && distance(z.y, z.wk) == 1);
  put(long_check, kcheck && distance(z.y, z.wk) >= 3);
  put(guarded_check, checked && z.guard);
  put(fork, checked && attacks);
  put(skewer, checked && z.alive && separates(z.y, z.wk, z.x) &&
                  (control(z.b, BLACK, z.y, z.occ ^ square_bb(z.wk)) & z.x));
  put(pin, z.enemy_pin && !g.enemy_pin);
  put(unpin, king && g.pinned && !z.pinned && z.alive);
  put(stay_on_pin_line, !king && g.pinned && aligned(g.bk, g.y, z.y));
  put(escape_attack, quiet && g.danger && !z.danger);
  put(king_rescue, king && g.danger && z.guard && !g.guard && z.alive);
  put(king_fork_defence, king && approach && z.guard);
  put(king_attack_loose, king && approach && !z.enemy_guard);
  put(king_attack_guarded, king && approach && z.enemy_guard);
  put(counterattack, quiet && g.danger && attacks);
  put(attack_loose, fresh && !z.enemy_guard);
  put(attack_guarded, fresh && z.enemy_guard);
  put(interpose_king, king && z.alive && (g.white & g.y) &&
                          separates(z.x, z.bk, z.y) && !(z.white & z.y));
  put(orthogonal_opposition,
      king && opposition(z.wk, z.bk, false) && !opposition(g.wk, g.bk, false));
  put(diagonal_opposition,
      king && opposition(z.wk, z.bk, true) && !opposition(g.wk, g.bk, true));
  put(cut_king, quiet && major && cutoff(z.wk, z.bk, z.y, z.black) &&
                    !cutoff(g.wk, g.bk, g.y, g.black));
  put(release_cut,
      king && wmajor && z.alive && cutoff(g.bk, g.y, g.x, g.white) &&
          !cutoff(z.bk, z.y, z.x, z.white) &&
          (file_of(z.bk) == file_of(g.x) || rank_of(z.bk) == rank_of(g.x)));
  put(blockade_pawn, white_pawn && !king && z.y == z.x + 8);
  put(king_blockade, white_pawn && king && z.bk == z.x + 8);
  if (white_pawn) {
    Square prom = Square(56 + file_of(z.x)), next = Square(z.x + 8);
    put(attack_promotion, (z.black & prom) && !(g.black & prom));
    put(attack_next_push, (z.black & next) && !(g.black & next));
  }
  put(approach_white_pawn,
      white_pawn && king && distance(z.bk, z.x) < distance(g.bk, g.x));
  if (quiet && pawn) {
    const Square prom = Square(file_of(z.y));
    put(leave_pawn_square, distance(z.wk, prom) > rank_of(z.y) &&
                               distance(g.wk, prom) <= rank_of(g.y));
  }
  put(attack_from_behind, white_pawn && fresh && major &&
                              file_of(z.y) == file_of(z.x) &&
                              rank_of(z.y) < rank_of(z.x));
  auto center = [](Square s) {
    return (file_of(s) == 3 || file_of(s) == 4) &&
           (rank_of(s) == 3 || rank_of(s) == 4);
  };
  put(king_invasion, king && center(z.bk) && !center(g.bk));
  put(mate, end && checked);
  put(stalemate, end && !checked);
  return mask;
}

uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
uint64_t king_key(int a, int b) {
  int best = 4096;
  for (int mask : {0, 7, 56, 63}) {
    int x = a ^ mask, y = b ^ mask;
    best = std::min(best, std::min(x, y) * 64 + std::max(x, y));
  }
  return mix64(best + 41981);
}
struct ReplyEntry {
  uint64_t available, without_loss, without_draw, without_win;
};
void atlas_replies(char **argv) {
  const PieceType a = PieceType(std::atoi(argv[1])),
                  b = PieceType(std::atoi(argv[2]));
  Tablebases::init(argv[3]);
  std::ifstream atlas(argv[4], std::ios::binary);
  std::ofstream out(argv[5], std::ios::binary);
  Position pos;
  StateInfo st, next;
  std::array<AtlasEntry, 64> rows;
  const bool remaining = std::string(argv[6]) == "remaining";
  uint64_t nodes = 0, edges = 0;
  for (Square wk = SQ_A1; wk <= SQ_H8; ++wk)
    for (Square bk = SQ_A1; bk <= SQ_H8; ++bk) {
      if (file_of(wk) >= 4 || distance(wk, bk) <= 1 ||
          ((king_key(wk, bk) % 8 == 0) == remaining))
        continue;
      for (Square y = SQ_A1; y <= SQ_H8; ++y) {
        uint32_t index = (wk * 64 + bk) * 64 + y;
        atlas.seekg(uint64_t(index) * 64 * sizeof(AtlasEntry));
        atlas.read(reinterpret_cast<char *>(rows.data()), sizeof(rows));
        if (!atlas)
          std::abort();
        std::array<ReplyEntry, 64> results{};
        for (Square x = SQ_A1; x <= SQ_H8; ++x) {
          if (!rows[x].valid)
            continue;
          pos.atlas_set(wk, bk, make_piece(WHITE, a), x, make_piece(BLACK, b),
                        y, &st, Threads.main());
          Geometry before(wk, bk, x, y, a, b, true);
          int best = -2;
          for (Move m : MoveList<LEGAL>(pos)) {
            const bool king = from_sq(m) == bk, live = to_sq(m) != x;
            const Square newy = king ? y : to_sq(m);
            pos.do_move(m, next);
            bool check = pos.checkers(), capture = false;
            MoveList<LEGAL> answers(pos);
            if (!king && (!live || type_of(m) == PROMOTION))
              for (Move r : answers)
                if (to_sq(r) == newy) {
                  capture = true;
                  break;
                }
            Geometry after(wk, king ? to_sq(m) : bk, x, newy, a,
                           type_of(pos.piece_on(newy)), live);
            Mask tag =
                classify(before, after, m, check, answers.size() == 0, capture);
            Tablebases::ProbeState state;
            int score = -int(Tablebases::probe_wdl(pos, &state));
            if (state == Tablebases::FAIL)
              std::abort();
            score = (score > 0) - (score < 0);
            best = std::max(best, score);
            auto &entry = results[x];
            entry.available |= tag;
            entry.without_loss |= domain ^ tag;
            if (score >= 0)
              entry.without_draw |= domain ^ tag;
            if (score > 0)
              entry.without_win |= domain ^ tag;
            pos.undo_move(m);
            ++edges;
          }
          if (best != ((rows[x].wdl > 0) - (rows[x].wdl < 0)))
            std::abort();
          ++nodes;
        }
        out.write(reinterpret_cast<const char *>(&index), sizeof(index));
        out.write(reinterpret_cast<const char *>(results.data()),
                  sizeof(results));
      }
    }
  out.close();
  if (!out)
    std::abort();
  std::cout << "replies " << int(a) << " " << int(b) << " " << nodes << " "
            << edges << std::endl;
}
} // namespace Stockfish
