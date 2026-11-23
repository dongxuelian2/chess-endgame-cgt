// Intersect sibling geometries across piece substitutions and both atlas
// partitions.
#include "../../../measure.h"
#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Material {
  std::ifstream dense;
  std::vector<U> quiet;
  std::array<State, 64> state;
  Material(const std::string &root, int a, int b) : quiet(64 * 64 * 64) {
    auto name = std::to_string(a) + "-" + std::to_string(b);
    dense.open(root + "/" + name + ".bin", std::ios::binary);
    for (auto sub : {"", "remaining/"}) {
      std::ifstream in(root + "/" + sub + name + ".replies", std::ios::binary);
      if (!in)
        std::abort();
      uint32_t idx;
      std::array<Replies, 64> r;
      while (in.read((char *)&idx, 4)) {
        if (idx >= quiet.size() || !in.read((char *)r.data(), sizeof(r)))
          std::abort();
        for (int x = 0; x < 64; x++)
          if (!(r[x].available & 51))
            quiet[idx] |= U(1) << x;
      }
      if (!in.eof())
        std::abort();
    }
  }
  void read() {
    if (!dense.read((char *)state.data(), sizeof(state)))
      std::abort();
  }
};
int main(int argc, char **argv) {
  if (argc != 6 && argc != 7)
    return 2;
  int a = std::stoi(argv[2]), b = std::stoi(argv[3]);
  std::string mode = argv[5];
  bool promotion = mode == "promotion", open = mode == "open",
       solo = mode == "solo" || promotion || open, relaxed = mode == "relaxed";
  Material base(argv[1], a, b), queen(argv[1], solo ? a : 5, b);
  const U seed = argc == 7 ? std::stoull(argv[6]) : 591234;
  Geometry geo;
  std::ofstream out(argv[4]);
  for (int idx = 0; idx < 64 * 64 * 64; idx++) {
    base.read();
    queen.read();
    int w = idx >> 12, k = (idx >> 6) & 63, y = idx & 63;
    if (promotion && y / 8 != 1)
      continue;
    U candidates =
        (promotion || open) ? ~U(0) : base.quiet[idx] & queen.quiet[idx];
    if (!candidates || w % 8 >= 4)
      continue;
    U occ = (U(1) << w) | (U(1) << k) | (U(1) << y), src[64]{};
    for (int t = 0; t < 64; t++)
      if ((candidates >> t & 1) && base.state[t].v && queen.state[t].v &&
          sign(base.state[t].w) == sign(queen.state[t].w))
        for (int s = 0; s < 64; s++)
          if (base.state[s].v && queen.state[s].v &&
              legal_candidate(geo, a, b, s, t, y, occ))
            src[t] |= U(1) << s;
    U best = ~U(0);
    int bh = -1, bl = -1, bs = -1;
    int eligible = 0;
    for (int i = 0; i < 64; i++)
      if (src[i])
        for (int j = 0; j < i; j++) {
          U ss = src[i] & src[j];
          if (!ss || (!relaxed && (base.state[i].f != base.state[j].f ||
                                   queen.state[i].f != queen.state[j].f)))
            continue;
          int truth = sign(base.state[i].w) - sign(base.state[j].w);
          if (!truth)
            continue;
          int h = truth > 0 ? i : j, l = truth > 0 ? j : i,
              s = __builtin_ctzll(ss);
          eligible++;
          if (b == 1 && !solo)
            out << w << " " << k << " " << y << " " << h << " " << l << " " << s
                << "\n";
          else {
            U hash = mix((U(idx) * 4096 + h * 64 + l) + seed);
            if (hash < best) {
              best = hash;
              bh = h;
              bl = l;
              bs = s;
            }
          }
        }
    if ((b != 1 || solo) && eligible)
      out << w << " " << k << " " << y << " " << bh << " " << bl << " " << bs
          << "\n";
  }
  return out ? 0 : 3;
}
