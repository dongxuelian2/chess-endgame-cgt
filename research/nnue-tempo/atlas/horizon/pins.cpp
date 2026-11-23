// Compare a real absolute pin against the same pawn with its single push
// restored.
#include "../measure.h"
#include <array>
#include <fstream>
#include <string>
int main(int argc, char **argv) {
  if (argc != 5)
    return 2;
  std::string root = argv[1], stem = std::string(argv[2]) + "-1",
              folder =
                  root +
                  (std::string(argv[3]) == "remaining" ? "/remaining" : "");
  int a = std::stoi(argv[2]);
  std::ifstream dense(root + "/" + stem + ".bin", std::ios::binary),
      rr(folder + "/" + stem + ".replies", std::ios::binary);
  std::ofstream out(argv[4]);
  if (!dense || !rr || !out)
    return 2;
  int j = 0;
  while (j < K && std::string(names[j]) != "single_push")
    ++j;
  if (j == K)
    return 2;
  const U push = U(1) << j;
  uint64_t counts[3][3][3]{};
  Geometry g;
  uint32_t id;
  while (rr.read(reinterpret_cast<char *>(&id), 4)) {
    std::array<Replies, 64> r;
    std::array<State, 64> x;
    if (!rr.read(reinterpret_cast<char *>(r.data()), sizeof(r)))
      return 3;
    int w = id >> 12, k = (id >> 6) & 63, y = id & 63, t = y - 8;
    if (y / 8 < 2 || y / 8 > 5 || t == w || t == k)
      continue;
    dense.seekg(uint64_t(id) * sizeof(x));
    if (!dense.read(reinterpret_cast<char *>(x.data()), sizeof(x)))
      return 3;
    U src[64]{}, occ = (U(1) << w) | (U(1) << k) | (U(1) << y);
    bool pin[64]{};
    for (int z = 0; z < 64; z++)
      if (x[z].v && z != t && !(r[z].available & 51)) {
        pin[z] = g.can[a][z][k] && ((g.between[z][k] & occ) == (U(1) << y));
        for (int s = 0; s < 64; s++)
          if (x[s].v && legal_candidate(g, a, 1, s, z, y, occ))
            src[z] |= U(1) << s;
      }
    for (int p = 0; p < 64; p++)
      if (src[p] && pin[p] && !(r[p].available & push))
        for (int u = 0; u < 64; u++)
          if ((src[p] & src[u]) && !pin[u] && (r[u].available & push) &&
              x[p].f == x[u].f) {
            int order = sign(sign(x[u].w) - sign(x[p].w)),
                n = sign(x[u].n - x[p].n);
            int after = (r[u].win & push) ? 1 : (r[u].draw & push) ? 0 : -1;
            // Zero denotes a surviving advantage or an initially nonpositive
            // one.
            int result = !(r[u].legal & push)
                             ? 2
                             : ((order > 0 && after <= sign(x[p].w)) ? 1 : 0);
            counts[order + 1][result][n + 1]++;
          }
  }
  for (int order = -1; order <= 1; order++)
    for (int t = 0; t < 3; t++)
      for (int n = -1; n <= 1; n++)
        out << order << " " << t << " " << n << " "
            << counts[order + 1][t][n + 1] << "\n";
  return rr.eof() && out ? 0 : 4;
}
