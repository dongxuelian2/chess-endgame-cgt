// Screen reversals with a shared source and a promotion or skewer on the first
// row.
#include "../atlas/measure.h"
#include <array>
#include <fstream>
#include <tuple>
#include <vector>

using BoardRow = std::array<State, 64>;
using ByOpposingSquare = std::array<BoardRow, 64>;

struct Edge {
  U sources;
  int opposite;
};

bool reachable_skewer(const Geometry &g, int w, int k, int rook, int man) {
  U occupied = (U(1) << w) | (U(1) << k) | (U(1) << rook) | (U(1) << man);
  for (int dst = 0; dst < 64; ++dst)
    if (((occupied >> dst) & 1) == 0 && g.can[4][dst][man] &&
        ((g.between[dst][man] >> w) & 1) && g.can[4][rook][dst] &&
        !(g.between[rook][dst] & occupied))
      return true;
  return false;
}

bool capture_promotion(int pawn, int target) {
  return target / 8 == 0 && std::abs(target % 8 - pawn % 8) == 1;
}

int main(int argc, char **argv) {
  if (argc != 5)
    return 2;
  int a = std::stoi(argv[2]), b = std::stoi(argv[3]);
  bool skewer = b == 4;
  std::string dir = argv[1];
  std::ifstream base(dir + "/" + argv[2] + "-" + argv[3] + ".bin",
                     std::ios::binary);
  std::ifstream queen;
  if (skewer)
    queen.open(dir + "/5-4.bin", std::ios::binary);
  std::ofstream out(argv[4]);
  if (!base || (skewer && !queen) || !out)
    return 3;
  Geometry g;
  ByOpposingSquare p, q;
  for (int w = 0; w < 64; ++w)
    for (int k = 0; k < 64; ++k) {
      if (!base.read(reinterpret_cast<char *>(&p), sizeof(p)) ||
          (skewer && !queen.read(reinterpret_cast<char *>(&q), sizeof(q))))
        return 4;
      if (w % 8 >= 4)
        continue;
      std::array<std::vector<Edge>, 4096> edges;
      bool before[64][64]{};
      U source[64][64]{};
      for (int y = skewer ? 0 : 8; y < (skewer ? 64 : 16); ++y) {
        U occ = (U(1) << w) | (U(1) << k) | (U(1) << y);
        U *src = source[y];
        for (int x = 0; x < 64; ++x) {
          if (!p[y][x].v ||
              (skewer && (!q[y][x].v || sign(p[y][x].w) != sign(q[y][x].w))))
            continue;
          before[y][x] = skewer ? reachable_skewer(g, w, k, y, x)
                                : capture_promotion(y, x);
          for (int s = 0; s < 64; ++s)
            if (p[y][s].v && (!skewer || q[y][s].v) &&
                legal_candidate(g, a, b, s, x, y, occ))
              src[x] |= U(1) << s;
        }
        for (int h = 0; h < 64; ++h)
          for (int l = 0; l < 64; ++l) {
            U common = src[h] & src[l];
            if (!common || p[y][h].f != p[y][l].f ||
                (skewer && q[y][h].f != q[y][l].f) ||
                sign(p[y][h].w) <= sign(p[y][l].w))
              continue;
            edges[h * 64 + l].push_back({common, y});
          }
      }
      for (int h = 0; h < 64; ++h)
        for (int l = 0; l < 64; ++l)
          for (auto [sources, y] : edges[h * 64 + l]) {
            if (!before[y][h] || before[y][l])
              continue;
            if (skewer) {
              for (int z = 0; z < 64; ++z) {
                U shared = sources & source[z][h] & source[z][l];
                if (!shared || y == z || before[z][h] || before[z][l] ||
                    p[z][h].f != p[z][l].f || q[z][h].f != q[z][l].f ||
                    sign(p[z][h].w) != sign(p[z][l].w))
                  continue;
                out << w << ' ' << k << ' ' << y << ' ' << z << ' ' << h << ' '
                    << l << ' ' << __builtin_ctzll(shared) << '\n';
              }
              continue;
            }
            for (auto [other_sources, z] : edges[l * 64 + h]) {
              U shared = sources & other_sources;
              if (!shared || before[z][l])
                continue;
              out << w << ' ' << k << ' ' << y << ' ' << z << ' ' << h << ' '
                  << l << ' ' << __builtin_ctzll(shared) << '\n';
            }
          }
    }
  return out ? 0 : 5;
}
