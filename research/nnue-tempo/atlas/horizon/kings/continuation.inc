// Restrict Black's second move on lines that start with a pawn push.
namespace Stockfish::KingStudy {
std::array<int, count> continuation(Position &p) {
  std::array<int, count> root;
  root.fill(-2);
  for (Move m : MoveList<LEGAL>(p)) {
    StateInfo st;
    auto b = play(p, m, st);
    std::array<int, count> reply;
    reply.fill(2);
    if (b.tags & (Mask(1) << push_pawn)) {
      for (Move z : MoveList<LEGAL>(p)) {
        StateInfo zs;
        auto w = play(p, z, zs);
        Mask surviving[3]{};
        int best = -2;
        for (Move u : MoveList<LEGAL>(p)) {
          StateInfo us;
          auto third = play(p, u, us);
          p.undo_move(u);
          surviving[1 + third.v] |= Domain ^ third.tags;
          best = std::max(best, third.v);
        }
        if (best != -2 && best != w.v)
          std::abort();
        for (int j = 0; j < count; j++) {
          Mask bit = Mask(1) << j;
          int value = (surviving[2] & bit)   ? 1
                      : (surviving[1] & bit) ? 0
                      : (surviving[0] & bit) ? -1
                                             : w.v;
          reply[j] = std::min(reply[j], value);
        }
        p.undo_move(z);
      }
    }
    for (int j = 0; j < count; j++)
      root[j] = std::max(root[j], reply[j] == 2 ? b.v : reply[j]);
    p.undo_move(m);
  }
  return root;
}
void third_batch(char **argv) {
  Tablebases::init(argv[2]);
  int a = std::stoi(argv[4]), b = std::stoi(argv[5]);
  if (b != 1)
    std::abort();
  std::ifstream in(argv[6]);
  std::ofstream out(argv[7]);
  int w, k, y, i, j;
  Position p;
  StateInfo st;
  while (in >> w >> k >> y >> i >> j) {
    auto eval = [&](int s) {
      p.atlas_set(Square(w), Square(k), make_piece(WHITE, PieceType(a)),
                  Square(s), B_PAWN, Square(y), &st, Threads.main());
      return continuation(p);
    };
    auto h = eval(i), l = eval(j);
    Mask mask = 0;
    for (int t = 0; t < count; t++)
      if (h[t] <= l[t])
        mask |= Mask(1) << t;
    out << w << " " << k << " " << y << " " << i << " " << j << " " << mask
        << "\n";
  }
  if (!in.eof() || !out)
    std::abort();
}
} // namespace Stockfish::KingStudy
