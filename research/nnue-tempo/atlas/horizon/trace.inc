// Export integer endpoints for pairwise native-head interchanges.
namespace Stockfish::Eval::NNUE {
extern uint8_t horizon_h1[8];
extern int horizon_ps, horizon_pos;
} // namespace Stockfish::Eval::NNUE
namespace Stockfish::Frontier {
struct DenseEntry {
  int16_t h, f, n, p, d;
  int8_t w;
  uint8_t valid;
};
struct Raw {
  int32_t ps, pos;
  uint8_t h[8];
};
void traces(char **argv) {
  int a = std::stoi(argv[4]);
  std::ifstream dense(argv[6], std::ios::binary);
  std::ofstream out(argv[7], std::ios::binary);
  if (!dense || !out)
    std::abort();
  Position p;
  StateInfo st;
  std::array<DenseEntry, 64> row;
  uint64_t visits = 0;
  for (int w = 0; w < 64; w++)
    for (int k = 0; k < 64; k++)
      for (int y = 0; y < 64; y++) {
        if (!dense.read(reinterpret_cast<char *>(row.data()), sizeof(row)))
          std::abort();
        std::array<Raw, 64> block{};
        for (int x = 0; x < 64; x++)
          if (w % 8 < 4 && row[x].valid) {
            p.atlas_set(Square(w), Square(k), make_piece(WHITE, PieceType(a)),
                        Square(x), B_PAWN, Square(y), &st, Threads.main());
            int n = Eval::NNUE::evaluate(p, false);
            if (n != row[x].n)
              std::abort();
            auto &r = block[x];
            r.ps = Eval::NNUE::horizon_ps;
            r.pos = Eval::NNUE::horizon_pos;
            std::copy_n(Eval::NNUE::horizon_h1, 8, r.h);
            ++visits;
          }
        out.write(reinterpret_cast<const char *>(block.data()), sizeof(block));
      }
  if (!out)
    std::abort();
  std::cout << "traces " << a << " " << visits << std::endl;
}
} // namespace Stockfish::Frontier
