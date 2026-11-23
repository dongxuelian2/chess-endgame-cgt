// Enumerate sign reversals on four boards with the same two legal White moves.
#include "measure.h"
#include <array>
#include <fstream>
#include <string>
#include <vector>

struct Contrast {
  U sources, signature;
  int classical;
  int y, outcome, neural, psqt, positional;
};

struct Audit {
  uint64_t pairs = 0, rectangles = 0, neural = 0, psqt = 0, positional = 0;
  uint64_t correct_sides = 0, identical = 0, identical_neural = 0,
           identical_positional = 0;
};

int main(int argc, char **argv) {
  if (argc != 5 && argc != 6)
    return 2;
  const std::string root = argv[1], out = argv[2];
  const int a = std::stoi(argv[3]), b = std::stoi(argv[4]);
  const std::string stem = std::to_string(a) + "-" + std::to_string(b);
  std::ifstream dense(root + "/" + stem + ".bin", std::ios::binary);
  std::ifstream reply(root + (argc == 6 ? "/remaining/" : "/") + stem +
                          ".replies",
                      std::ios::binary);
  std::ofstream summary(out + "/" + stem + ".cross"),
      witness(out + "/" + stem + ".rectangle");
  if (!dense || !reply || !summary || !witness)
    return 3;
  Geometry geo;
  Audit stats;
  std::array<std::vector<Contrast>, 64 * 64> prior;
  int last_kings = -1;
  bool witnessed = false;
  uint32_t index;
  while (reply.read(reinterpret_cast<char *>(&index), sizeof(index))) {
    const int kings = index >> 6, wk = kings >> 6, bk = kings & 63,
              y = index & 63;
    if (kings != last_kings) {
      for (auto &v : prior)
        v.clear();
      last_kings = kings;
    }
    std::array<State, 64> x;
    std::array<Replies, 64> r;
    dense.seekg(uint64_t(index) * sizeof(x));
    if (!dense.read(reinterpret_cast<char *>(x.data()), sizeof(x)) ||
        !reply.read(reinterpret_cast<char *>(r.data()), sizeof(r)))
      return 4;
    U sources[64]{};
    const U blockers = (U(1) << wk) | (U(1) << bk) | (U(1) << y);
    for (int src = 0; src < 64; ++src)
      if (x[src].v)
        for (int dst = 0; dst < 64; ++dst)
          if (x[dst].v && legal_candidate(geo, a, b, src, dst, y, blockers))
            sources[dst] |= U(1) << src;
    for (int i = 0; i < 64; ++i)
      for (int j = 0; j < i; ++j) {
        const U common = sources[i] & sources[j];
        if (!common || r[i].available != r[j].available || x[i].f != x[j].f)
          continue;
        const int w = sign(sign(x[i].w) - sign(x[j].w));
        if (!w)
          continue;
        Contrast next{common,
                      r[i].available,
                      x[i].f,
                      y,
                      w,
                      sign(x[i].n - x[j].n) * w,
                      sign(x[i].p - x[j].p) * w,
                      sign(x[i].d - x[j].d) * w};
        ++stats.pairs;
        auto &group = prior[i * 64 + j];
        for (const Contrast &old : group) {
          const U shared = old.sources & common;
          if (old.outcome == w || !shared)
            continue;
          ++stats.rectangles;
          stats.correct_sides += 2 + old.neural + next.neural;
          stats.neural += old.neural > 0 && next.neural > 0;
          stats.psqt += old.psqt > 0 && next.psqt > 0;
          stats.positional += old.positional > 0 && next.positional > 0;
          const bool identical = old.signature == next.signature &&
                                 old.classical == next.classical;
          if (identical) {
            ++stats.identical;
            stats.identical_neural += old.neural > 0 && next.neural > 0;
            stats.identical_positional +=
                old.positional > 0 && next.positional > 0;
          }
          if (!witnessed && identical && old.neural > 0 && next.neural > 0) {
            witness << wk << " " << bk << " " << old.y << " " << y << " "
                    << __builtin_ctzll(shared) << " " << i << " " << j << "\n";
            witnessed = true;
          }
        }
        group.push_back(next);
      }
  }
  summary << stats.pairs << " " << stats.rectangles << " " << stats.neural
          << " " << stats.psqt << " " << stats.positional << " "
          << stats.correct_sides << " " << stats.identical << " "
          << stats.identical_neural << " " << stats.identical_positional
          << "\n";
  return reply.eof() && summary && witness ? 0 : 5;
}
