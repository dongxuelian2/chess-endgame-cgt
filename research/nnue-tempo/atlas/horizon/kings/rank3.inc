namespace Stockfish::KingStudy {
struct Profile {
  int8_t actual, runner, pawn_best_only, follow, fork_only, valid;
};
Profile profile(Position &p, Square pawn) {
  Profile o{};
  o.actual = -2;
  o.runner = -2;
  int kingbest = -2, otherbest = -2, forkbest = -2;
  bool known = false;
  for (Move b : MoveList<LEGAL>(p)) {
    bool push = from_sq(b) == pawn;
    StateInfo st;
    Action act = play(p, b, st);
    o.actual = std::max(int(o.actual), act.v);
    if (push) {
      o.runner = act.v;
      int best = 2, shape = 0;
      for (Move w : MoveList<LEGAL>(p)) {
        StateInfo wt;
        Action white = play(p, w, wt);
        p.undo_move(w);
        if (white.v > best)
          continue;
        if (white.v < best) {
          shape = 0;
          best = white.v;
        }
        auto tag = [&](Tag t) { return bool(white.tags & (Mask(1) << t)); };
        int type = tag(uncover_check) ? 1
                   : tag(Tag::check)  ? 2
                   : tag(pin_man)     ? 4
                                      : 8;
        shape |= type;
      }
      if (best < 2 && best != act.v)
        std::abort();
      // Deleting the best replies is defined only with a legal fallback.
      known = false;
      for (Move w : MoveList<LEGAL>(p)) {
        StateInfo wt;
        Action white = play(p, w, wt);
        p.undo_move(w);
        bool uncovered = white.tags & (Mask(1) << uncover_check),
             checks = white.tags & (Mask(1) << Tag::check),
             pin = white.tags & (Mask(1) << pin_man);
        int type = uncovered ? 1 : checks ? 2 : pin ? 4 : 8;
        if (!(shape & type)) {
          known = true;
          break;
        }
      }
      o.follow = known ? shape : 0;
    } else
      kingbest = std::max(kingbest, act.v);
    if (act.tags & (Mask(1) << king_attack_with_promotion_guard))
      forkbest = std::max(forkbest, act.v);
    else
      otherbest = std::max(otherbest, act.v);
    p.undo_move(b);
  }
  o.pawn_best_only =
      (o.actual == 1 && o.runner == 1 && kingbest >= -1 && kingbest < 1);
  o.fork_only =
      (o.actual == 1 && forkbest == 1 && otherbest >= -1 && otherbest < 1);
  o.valid = 1;
  return o;
}
void rank_batch(char **argv) {
  Tablebases::init(argv[2]);
  int a = std::stoi(argv[4]);
  std::ifstream in(argv[6], std::ios::binary);
  std::ofstream out(argv[7], std::ios::binary);
  if (!in || !out)
    std::abort();
  using Entry = Stockfish::Frontier::DenseEntry;
  Position p;
  StateInfo st;
  for (int w = 0; w < 64; w++)
    if (w % 8 < 4)
      for (int k = 0; k < 64; k++)
        for (int y = 16; y < 24; y++) {
          std::array<Entry, 64> xs;
          std::array<Profile, 64> r{};
          in.seekg(uint64_t((w * 64 + k) * 64 + y) * sizeof(xs));
          if (!in.read((char *)xs.data(), sizeof(xs)))
            std::abort();
          for (int x = 0; x < 64; x++)
            if (xs[x].valid) {
              p.atlas_set(Square(w), Square(k), make_piece(WHITE, PieceType(a)),
                          Square(x), B_PAWN, Square(y), &st, Threads.main());
              bool quiet = true;
              for (Move m : MoveList<LEGAL>(p))
                if (p.capture_or_promotion(m)) {
                  quiet = false;
                  break;
                }
              if (quiet) {
                r[x] = profile(p, Square(y));
                if (r[x].actual != (xs[x].w > 0) - (xs[x].w < 0))
                  std::abort();
              }
            }
          out.write((char *)r.data(), sizeof(r));
        }
  if (!out)
    std::abort();
}
} // namespace Stockfish::KingStudy
