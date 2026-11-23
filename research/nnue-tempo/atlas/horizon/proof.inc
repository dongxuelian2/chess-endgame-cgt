// Four threshold games for the shortest forced conversion or terminal result.
namespace Stockfish::Frontier {
struct Limit {
  uint32_t key = 0;
  int8_t fail = -1, success = 100;
  Move first = MOVE_NONE;
};
struct Proof {
  std::vector<Limit> slots = std::vector<Limit>(1 << 21);
  uint64_t expanded = 0;
  bool prove(Position &p, Color player, int target, int horizon) {
    const auto key = placement(p) * 4 + 2 * unsigned(player) + target;
    auto &record = slots[uint32_t(key * 2654435761U) >> (32 - 21)];
    Limit node = record.key == key + 1 ? record : Limit{key + 1};
    if (horizon <= node.fail)
      return false;
    if (horizon >= node.success)
      return true;
    bool owns = p.side_to_move() == player;
    // An impossible orthodox result has no sound finite proof. This shortcut
    // discharges failures only. Each successful strategy still reaches exits.
    if (node.fail == -1 && node.success == 100 &&
        (owns ? 1 : -1) * label(p) < target) {
      node.fail = 99;
      record = node;
      return false;
    }
    MoveList<LEGAL> moves(p);
    bool success = !owns;
    if (!moves.size())
      success = (p.checkers() ? (owns ? -1 : 1) : 0) >= target;
    else if (!horizon)
      success = false;
    else {
      ++expanded;
      auto visit = [&](Move m) {
        bool convert = p.capture_or_promotion(m);
        StateInfo child;
        p.do_move(m, child);
        bool child_pass = convert ? ((owns ? -1 : 1) * label(p) >= target)
                                  : prove(p, player, target, horizon - 1);
        p.undo_move(m);
        return child_pass;
      };
      const Move preferred = node.first;
      if (preferred != MOVE_NONE && visit(preferred) == owns)
        success = owns;
      else
        for (Move m : moves)
          if (m != preferred && visit(m) == owns) {
            success = owns;
            node.first = m;
            break;
          }
    }
    if (success)
      node.success = std::min(int(node.success), horizon);
    else
      node.fail = std::max(int(node.fail), horizon);
    record = node;
    return success;
  }
};
void proofs(char **argv) {
  Tablebases::init(argv[2]);
  int limit = std::stoi(argv[3]), a = std::stoi(argv[4]),
      b = std::stoi(argv[5]);
  std::ifstream in(argv[6]);
  std::ofstream out(argv[7]);
  if (!in || !out || limit >= 99)
    std::abort();
  Proof solver;
  Position p;
  StateInfo st;
  int w, k, y, x;
  while (in >> w >> k >> y >> x) {
    p.atlas_set(Square(w), Square(k), make_piece(WHITE, PieceType(a)),
                Square(x), make_piece(BLACK, PieceType(b)), Square(y), &st,
                Threads.main());
    out << w << " " << k << " " << y << " " << x;
    for (Color c : {BLACK, WHITE})
      for (int target : {0, 1}) {
        int depth = 0;
        while (depth <= limit && !solver.prove(p, c, target, depth))
          ++depth;
        out << " " << depth;
      }
    out << "\n";
  }
  if (!in.eof() || !out)
    std::abort();
  std::cout << "proof " << a << " " << b << " " << solver.expanded << std::endl;
}
} // namespace Stockfish::Frontier
