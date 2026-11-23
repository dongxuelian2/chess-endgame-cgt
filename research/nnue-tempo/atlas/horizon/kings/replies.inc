// One Black move and one White reply, followed by ordinary tablebase play.
#include <array>
#include <fstream>
#include <vector>
namespace Stockfish::KingStudy {
using Mask = uint64_t;
enum Tag {
#define RULE(t) t,
#include "moves.def"
#undef RULE
  count
};
constexpr Mask Domain = (Mask(1) << count) - 1;
void set(Mask &m, Tag t, bool cond) {
  if (cond)
    m |= Mask(1) << t;
}
struct Action {
  int code, v;
  Mask tags;
};
struct Branch {
  Action black;
  std::vector<Action> white;
};
using Tree = std::vector<Branch>;
Square man(const Position &p, Color c) {
  auto b = p.pieces(c) ^ p.pieces(c, KING);
  return b ? lsb(b) : SQ_NONE;
}
Mask describe(const Position &p, Color us, Square from, Square dest,
              PieceType moved, bool take, Square oldenemy) {
  Color them = ~us;
  Square k = p.square<KING>(us), other = p.square<KING>(them), own = man(p, us),
         enemy = man(p, them);
  PieceType kind = type_of(p.piece_on(dest));
  bool king = moved == KING, check = p.checkers();
  Bitboard control = kind == PAWN ? pawn_attacks_bb(us, dest)
                                  : attacks_bb(kind, dest, p.pieces());
  Mask m = 0;
  set(m, step_king, king);
  set(m, move_man, !king);
  set(m, push_pawn, moved == PAWN && !take);
  set(m, Tag::check, check);
  bool direct = bool(control & other);
  set(m, check_from_far, check && direct && distance(dest, other) >= 3);
  set(m, rank_check, check && direct && rank_of(dest) == rank_of(other));
  set(m, file_check, check && direct && file_of(dest) == file_of(other));
  set(m, diagonal_check,
      check && direct && (kind == QUEEN || kind == BISHOP) &&
          rank_of(dest) != rank_of(other) && file_of(dest) != file_of(other));
  set(m, contact_check, check && direct && distance(dest, other) == 1);
  set(m, check_and_attack_man,
      check && enemy != SQ_NONE && bool(control & enemy));
  set(m, pin_man,
      enemy != SQ_NONE && bool(p.blockers_for_king(them) & enemy) &&
          bool(p.pieces(them) & enemy));
  set(m, capture_man, take);
  set(m, attack_man, enemy != SQ_NONE && bool(control & enemy));
  if (enemy != SQ_NONE && type_of(p.piece_on(enemy)) == PAWN) {
    Square next = enemy + pawn_push(them),
           prom = make_square(file_of(enemy), relative_rank(them, RANK_8));
    set(m, attack_pawn_next, bool(control & next));
    set(m, attack_promotion, bool(control & prom));
    set(m, occupy_promotion, dest == prom);
    set(m, occupy_pawn_file, file_of(dest) == file_of(enemy));
    set(m, rear_attack_pawn,
        file_of(dest) == file_of(enemy) &&
            relative_rank(them, dest) < relative_rank(them, enemy) &&
            bool(control & enemy));
    set(m, blockade_pawn, dest == next);
  }
  set(m, guard_own_man, king && own != SQ_NONE && distance(k, own) == 1);
  set(m, king_approach_other_man,
      king && oldenemy != SQ_NONE &&
          distance(dest, oldenemy) < distance(from, oldenemy));
  if (king && own != SQ_NONE && type_of(p.piece_on(own)) == PAWN) {
    Square prom = make_square(file_of(own), relative_rank(us, RANK_8));
    set(m, king_approach_own_promotion,
        distance(dest, prom) < distance(from, prom));
    set(m, king_guard_own_pawn_next, distance(dest, own + pawn_push(us)) == 1);
    set(m, king_ahead_of_own_pawn,
        relative_rank(us, dest) > relative_rank(us, own));
  }
  if (kind == ROOK || kind == QUEEN) {
    bool f = (file_of(k) < file_of(dest) && file_of(dest) < file_of(other)) ||
             (file_of(other) < file_of(dest) && file_of(dest) < file_of(k));
    bool r = (rank_of(k) < rank_of(dest) && rank_of(dest) < rank_of(other)) ||
             (rank_of(other) < rank_of(dest) && rank_of(dest) < rank_of(k));
    Bitboard cover = control | dest;
    set(m, cut_kings,
        (f && !(file_bb(dest) & ~cover)) || (r && !(rank_bb(dest) & ~cover)));
  }
  auto major = [](PieceType t) { return t == ROOK || t == QUEEN; };
  set(m, uncover_check, king && check);
  set(m, approach_enemy_king,
      king && distance(dest, other) < distance(from, other));
  if (own != SQ_NONE && type_of(p.piece_on(own)) == PAWN && king &&
      enemy != SQ_NONE) {
    Square prom = make_square(file_of(own), relative_rank(us, RANK_8));
    set(m, king_attack_with_promotion_guard,
        distance(dest, prom) == 1 && distance(dest, enemy) == 1);
    set(m, king_steps_down_pawn_file,
        rank_of(dest) != rank_of(from) &&
            relative_rank(us, dest) > relative_rank(us, from) &&
            file_of(dest) == file_of(from));
  }
  if (enemy != SQ_NONE && type_of(p.piece_on(enemy)) == PAWN &&
      own != SQ_NONE && major(type_of(p.piece_on(own)))) {
    Square prom = make_square(file_of(enemy), relative_rank(them, RANK_8));
    Bitboard major_control =
        attacks_bb(type_of(p.piece_on(own)), own, p.pieces());
    bool backrank = rank_of(own) == rank_of(prom);
    set(m, major_on_last_rank, !king && backrank);
    set(m, major_attacks_pawn_from_in_front,
        !king && file_of(own) == file_of(enemy) &&
            relative_rank(them, own) > relative_rank(them, enemy) &&
            bool(major_control & enemy));
    set(m, king_step_under_pawn_file_control,
        king && file_of(own) == file_of(enemy));
    set(m, king_step_under_last_rank_control,
        king && backrank && (own == prom || bool(major_control & prom)));
    set(m, king_step_toward_pawn_under_last_rank_control,
        king && backrank && (own == prom || bool(major_control & prom)) &&
            distance(dest, enemy) < distance(from, enemy));
  }
  return m;
}
Action play(Position &p, Move m, StateInfo &st) {
  Color us = p.side_to_move();
  Square f = from_sq(m), t = to_sq(m), enemy = man(p, ~us);
  PieceType kind = type_of(p.piece_on(f));
  bool take = p.capture(m);
  int code = ((kind == KING ? 1 : 0) * 64 + t) * 5 +
             (type_of(m) == PROMOTION ? int(promotion_type(m)) - 1 : 0);
  p.do_move(m, st);
  int v = Frontier::label(p) * (us == BLACK ? -1 : 1);
  return {code, v, describe(p, us, f, t, kind, take, enemy)};
}
Tree build(Position &p) {
  Tree tree;
  for (Move b : MoveList<LEGAL>(p)) {
    StateInfo st;
    Action action = play(p, b, st);
    Branch branch{action, {}};
    int best = 2;
    for (Move w : MoveList<LEGAL>(p)) {
      StateInfo wt;
      auto reply = play(p, w, wt);
      if (action.tags & (Mask(1) << push_pawn)) {
        auto tagged = [&](Tag t) { return bool(reply.tags & (Mask(1) << t)); };
        set(reply.tags, after_push_captures_pawn, tagged(capture_man));
        set(reply.tags, after_push_checks, tagged(Tag::check));
        set(reply.tags, after_push_king_closes,
            tagged(king_approach_other_man));
        set(reply.tags, after_push_man_covers_promotion,
            !tagged(step_king) && tagged(attack_promotion));
        set(reply.tags, after_push_blocks_queening, tagged(occupy_promotion));
        set(reply.tags, after_push_uses_last_rank, tagged(major_on_last_rank));
        set(reply.tags, after_push_king_step, tagged(step_king));
      }
      p.undo_move(w);
      best = std::min(best, reply.v);
      branch.white.push_back(reply);
    }
    if (!branch.white.empty() && best != action.v)
      std::abort();
    p.undo_move(b);
    tree.push_back(std::move(branch));
  }
  return tree;
}
void mark(Action &a) {
  bool king = a.tags & (Mask(1) << step_king),
       checking = a.tags & (Mask(1) << check);
  set(a.tags, private_destination, true);
  set(a.tags, private_king_step, king);
  set(a.tags, private_man_move, !king);
  set(a.tags, private_check, checking);
  set(a.tags, private_quiet_man_move,
      !king && !checking && !(a.tags & (Mask(1) << capture_man)));
  set(a.tags, private_pawn_push, a.tags & (Mask(1) << push_pawn));
}
void compare(Tree &a, Tree &b) {
  for (auto &p : a) {
    auto q = std::find_if(b.begin(), b.end(), [&](const Branch &other) {
      return other.black.code == p.black.code;
    });
    if (q == b.end()) {
      mark(p.black);
      continue;
    }
    bool available[640]{};
    for (auto &z : q->white)
      available[z.code] = true;
    for (auto &z : p.white)
      if (!available[z.code])
        mark(z);
  }
}
struct Restricted {
  std::array<int, count> black, white;
  int actual;
};
Restricted restrict(const Tree &t) {
  Restricted x;
  x.actual = -2;
  x.black.fill(-2);
  x.white.fill(-2);
  for (auto &b : t) {
    x.actual = std::max(x.actual, b.black.v);
    Mask white[3]{};
    for (auto &z : b.white)
      white[1 + z.v] |= Domain ^ z.tags;
    for (int j = 0; j < count; j++) {
      Mask bit = Mask(1) << j;
      if (!(b.black.tags & bit))
        x.black[j] = std::max(x.black[j], b.black.v);
      int v = (white[0] & bit)   ? -1
              : (white[1] & bit) ? 0
              : (white[2] & bit) ? 1
                                 : b.black.v;
      x.white[j] = std::max(x.white[j], v);
    }
  }
  for (int &v : x.black)
    if (v == -2)
      v = x.actual;
  return x;
}
void batch(char **argv) {
  Tablebases::init(argv[2]);
  int a = std::stoi(argv[4]), b = std::stoi(argv[5]);
  std::ifstream in(argv[6]);
  std::ofstream out(argv[7]);
  if (!in || !out)
    std::abort();
  int w, k, y, i, j;
  Position p;
  StateInfo st;
  auto make = [&](int z) {
    p.atlas_set(Square(w), Square(k), make_piece(WHITE, PieceType(a)),
                Square(z), make_piece(BLACK, PieceType(b)), Square(y), &st,
                Threads.main());
    return build(p);
  };
  while (in >> w >> k >> y >> i >> j) {
    auto hi = make(i), lo = make(j);
    compare(hi, lo);
    compare(lo, hi);
    auto h = restrict(hi), l = restrict(lo);
    if (h.actual <= l.actual)
      std::abort();
    Mask x = 0, z = 0;
    for (int t = 0; t < count; t++) {
      if (h.black[t] <= l.black[t])
        x |= Mask(1) << t;
      if (h.white[t] <= l.white[t])
        z |= Mask(1) << t;
    }
    out << w << " " << k << " " << y << " " << i << " " << j << " " << h.actual
        << " " << l.actual << " " << x << " " << z << "\n";
  }
  if (!in.eof() || !out)
    std::abort();
}
} // namespace Stockfish::KingStudy
