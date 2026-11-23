// Finite-horizon bounds for a four-man board before its first capture or
// promotion.
#include <array>
#include <fstream>
#include <vector>
namespace Stockfish::Frontier {
using Bits = uint16_t;
constexpr Bits All = 65535;
struct Thresholds {
  Bits draws, wins;
};
struct Bounds {
  Thresholds low, neutral, high;
};
constexpr Bounds Unknown{{0, 0}, {All, 0}, {All, All}};

Bounds reveal(int tag, int value) {
  // Bit s represents the experiment revealing the event classes in subset s.
  constexpr Bits masks[]{0xaaaa, 0xcccc, 0xf0f0, 0xff00};
  Bits on = masks[tag], off = ~on;
  Bits d = value >= 0 ? on : 0, w = value > 0 ? on : 0;
  return {{d, w}, {Bits(off | d), w}, {Bits(off | d), Bits(off | w)}};
}
void maximize(Thresholds &best, Thresholds child) {
  best.draws |= Bits(~child.wins);
  best.wins |= Bits(~child.draws);
}
int label(Position &p) {
  Tablebases::ProbeState status;
  int w = int(Tablebases::probe_wdl(p, &status));
  if (status == Tablebases::FAIL)
    std::abort();
  return (w > 0) - (w < 0);
}
int elementary(Position &p) {
  Bitboard man = p.pieces() ^ p.pieces(KING);
  if (!man)
    return 0;
  Piece pc = p.piece_on(lsb(man));
  if (type_of(pc) == KNIGHT || type_of(pc) == BISHOP)
    return 0;
  if (color_of(pc) == p.side_to_move())
    return 1;
  MoveList<LEGAL> options(p);
  if (!options.size())
    return p.checkers() ? -1 : 0;
  for (Move m : options)
    if (p.capture(m))
      return 0;
  return -1;
}
struct Entry {
  uint32_t key;
  Bounds value;
};
constexpr unsigned Slots = 1 << 18;
uint32_t placement(const Position &p) {
  uint32_t key = p.square<KING>(WHITE);
  key = (key << 6) | unsigned(p.square<KING>(BLACK));
  for (Color c : {WHITE, BLACK})
    key = (key << 6) | unsigned(lsb(p.pieces(c) ^ p.pieces(c, KING)));
  key = ((key * 2 + int(p.side_to_move())) * 9) +
        (p.ep_square() == SQ_NONE ? 8 : int(file_of(p.ep_square())));
  return key;
}
struct Solver {
  std::vector<std::vector<Entry>> cache;
  uint64_t visits = 0, hits = 0, conversions = 0;
  explicit Solver(int depth) : cache(depth + 1, std::vector<Entry>(Slots)) {}
  Bounds solve(Position &p, int depth) {
    ++visits;
    // Fixed material per process. Encode squares, turn and a possible ep file
    // exactly.
    const auto key = placement(p);
    auto &entry = cache[depth][(uint32_t(key * 2654435761U)) >> (32 - 18)];
    if (entry.key == key + 1) {
      ++hits;
      return entry.value;
    }
    Bounds best{};
    MoveList<LEGAL> moves(p);
    if (!moves.size())
      best = reveal(3, p.checkers() ? -1 : 0);
    else if (!depth)
      best = Unknown;
    else
      for (Move m : moves) {
        const bool promo = type_of(m) == PROMOTION, take = p.capture(m);
        StateInfo child;
        p.do_move(m, child);
        Bounds v;
        if (promo || take) {
          ++conversions;
          const int tag = promo ? 2 : (p.pieces(PAWN) ? 1 : 0);
          v = reveal(tag, tag == 0 ? elementary(p) : label(p));
        } else
          v = solve(p, depth - 1);
        p.undo_move(m);
        maximize(best.low, v.high);
        maximize(best.neutral, v.neutral);
        maximize(best.high, v.low);
      }
    entry = {key + 1, best};
    return best;
  }
};
void batch(char **argv) {
  Tablebases::init(argv[2]);
  const int limit = std::stoi(argv[3]), a = std::stoi(argv[4]),
            b = std::stoi(argv[5]);
  Solver solver(limit);
  std::ifstream in(argv[6]);
  std::ofstream out(argv[7]);
  if (!in || !out)
    std::abort();
  int wk, bk, y, x;
  Position p;
  StateInfo st;
  while (in >> wk >> bk >> y >> x) {
    p.atlas_set(Square(wk), Square(bk), make_piece(WHITE, PieceType(a)),
                Square(x), make_piece(BLACK, PieceType(b)), Square(y), &st,
                Threads.main());
    out << wk << " " << bk << " " << y << " " << x;
    for (int depth = 0; depth <= limit; ++depth) {
      Bounds v = solver.solve(p, depth);
      for (auto t : {v.low, v.neutral, v.high})
        out << " " << t.draws << " " << t.wins;
    }
    out << "\n";
  }
  if (!in.eof() || !out)
    std::abort();
  std::cout << "frontier " << a << " " << b << " " << solver.visits << " "
            << solver.hits << " " << solver.conversions << std::endl;
}
} // namespace Stockfish::Frontier
