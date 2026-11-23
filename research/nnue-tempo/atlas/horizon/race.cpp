// Exact reachability game with a running black pawn and a white minor piece.
#include "../measure.h"
#include "head.h"
#include <array>
#include <fstream>
#include <string>

using Table = std::array<U, 64>;
U bit(int s) { return U(1) << s; }
Table race(const Geometry &g, int a, U obstacles) {
  Table escaping{}, moves{};
  for (int x = 0; x < 64; x++)
    if (!(bit(x) & obstacles))
      for (int z = 0; z < 64; z++)
        if (g.can[a][x][z] && !(obstacles & (bit(z) | g.between[x][z])))
          moves[x] |= bit(z);
  for (int y = 8; y < 56; y++)
    if (!(obstacles & bit(y))) {
      for (int x = 0; x < 64; x++)
        if (x != y && !(obstacles & bit(x))) {
          if (x / 8 == y / 8 - 1 && std::abs(x % 8 - y % 8) == 1) {
            escaping[y] |= bit(x);
            continue;
          }
          for (int p = y - 8; p >= 0 && p >= y - 16; p -= 8) {
            if (p != y - 8 && y / 8 != 6)
              break;
            if ((obstacles | bit(x)) & bit(p))
              break;
            if (moves[x] & bit(p))
              continue; // White can take the advanced pawn.
            if (p < 8 || !((moves[x] | bit(x)) & ~escaping[p])) {
              escaping[y] |= bit(x);
              break;
            }
          }
        }
    }
  return escaping;
}
int main(int argc, char **argv) {
  if (argc != 5 && argc != 6)
    return 2;
  std::ofstream exceptions;
  if (argc == 6) {
    exceptions.open(argv[5]);
    if (!exceptions)
      return 2;
  }
  const int a = std::stoi(argv[2]);
  std::string stem = std::to_string(a) + "-1";
  if (std::string(argv[3]) == "tables") {
    Geometry g;
    std::ofstream file(argv[4], std::ios::binary);
    Table bare = race(g, a, 0);
    file.write(reinterpret_cast<const char *>(bare.data()), sizeof(bare));
    for (int w = 0; w < 64; ++w)
      for (int k = 0; k < 64; ++k) {
        Table obs = race(g, a, bit(w) | bit(k));
        file.write(reinterpret_cast<const char *>(obs.data()), sizeof(obs));
      }
    return file ? 0 : 4;
  }
  std::string root = argv[1],
              folder =
                  root +
                  (std::string(argv[3]) == "remaining" ? "/remaining" : "");
  std::ifstream dense(root + "/" + stem + ".bin", std::ios::binary),
      rr(folder + "/" + stem + ".replies", std::ios::binary);
  std::ofstream out(argv[4]);
  if (!dense || !rr || !out)
    return 2;
  // Histogram indexed by rule version, true order, race order and native order.
  uint64_t counts[2][3][3][3]{};
  std::ofstream audit(std::string(argv[4]) + ".audit");
  Geometry g;
  std::ifstream endpoints(root + "/horizon/" + std::to_string(a) + ".raw",
                          std::ios::binary),
      weights(root + "/horizon/head.txt");
  Head head(weights);
  std::ofstream patch(std::string(argv[4]) + ".patch");
  if (!endpoints || !patch)
    return 3;
  int64_t replay[2][4][11]{};
  std::array<std::array<int, 128>, 128> grid;
  std::array<std::array<int64_t, 128>, 128> additive;
  std::ifstream tables(root + "/horizon/grid.txt");
  for (int i = 0; i < 128; i++)
    for (int j = 0; j < 128; j++)
      if (!(tables >> grid[i][j] >> additive[i][j]))
        return 2;
  Table bare = race(g, a, 0), blocked;
  int past = -1;
  uint32_t idx;
  while (rr.read(reinterpret_cast<char *>(&idx), 4)) {
    const int kings = idx >> 6, w = kings >> 6, k = kings & 63, y = idx & 63;
    U kingsBB = bit(w) | bit(k), occ = kingsBB | bit(y);
    if (past != kings) {
      blocked = race(g, a, kingsBB);
      past = kings;
    }
    std::array<State, 64> st;
    std::array<Replies, 64> replies;
    dense.seekg(uint64_t(idx) * sizeof(st));
    if (!dense.read(reinterpret_cast<char *>(st.data()), sizeof(st)) ||
        !rr.read(reinterpret_cast<char *>(replies.data()), sizeof(replies)))
      return 3;
    std::array<RawHead, 64> raw;
    std::array<std::array<int, 32>, 64> pre{};
    endpoints.seekg(uint64_t(idx) * sizeof(raw));
    if (!endpoints.read(reinterpret_cast<char *>(raw.data()), sizeof(raw)))
      return 3;
    for (int x = 0; x < 64; x++)
      if (st[x].v && head.prepare(raw[x], pre[x]) != st[x].n)
        return 5;
    int frozen[64]{}, separable[64]{};
    for (int x = 0; x < 64; x++)
      if (st[x].v) {
        int u = raw[x].h[6], v = raw[x].h[7];
        frozen[x] = (grid[u][v] + raw[x].ps) / 16;
        separable[x] =
            (additive[u][v] + int64_t(raw[x].ps) * 16384) / (16 * 16384);
      }
    uint64_t test[2][3][3]{};
    U src[64]{};
    for (int t = 0; t < 64; t++)
      if (st[t].v && !(replies[t].available & 51))
        for (int s = 0; s < 64; s++)
          if (st[s].v && legal_candidate(g, a, 1, s, t, y, occ))
            src[t] |= bit(s);
    for (int i = 0; i < 64; i++)
      for (int j = 0; j < i; j++)
        if (src[i] & src[j] && st[i].f == st[j].f) {
          const int truth = sign(sign(st[i].w) - sign(st[j].w)),
                    nn = sign(st[i].n - st[j].n);
          for (int v = 0; v < 2; v++) {
            const U win = v ? blocked[y] : bare[y];
            const int r = int((win >> i) & 1) - int((win >> j) & 1);
            counts[v][1 + truth][1 + r][1 + nn]++;
            if (idx % 997 == 37)
              test[v][1 + truth][1 + r]++;
            if (truth || r) {
              const int target = truth ? truth : r;
              auto c = head.credits(raw[i], raw[j], pre[i], pre[j], target);
              if (argc == 6 && v == 0 && truth * r < 0) {
                const int hi = truth > 0 ? i : j, lo = truth > 0 ? j : i;
                exceptions << w << " " << k << " " << y << " " << hi << " "
                           << lo << " " << __builtin_ctzll(src[i] & src[j])
                           << " " << st[hi].n - st[lo].n;
                for (auto credit : c)
                  exceptions << " " << credit;
                exceptions << " " << int(raw[hi].h[6]) - int(raw[lo].h[6])
                           << " " << int(raw[hi].h[7]) - int(raw[lo].h[7])
                           << "\n";
              }
              auto &stats = replay[v][truth ? 1 + truth * r : 3];
              ++stats[0];
              stats[7] += 1 + target * sign(frozen[i] - frozen[j]);
              stats[8] += 1 + target * sign(separable[i] - separable[j]);
              stats[9] += target * (raw[i].h[6] - raw[j].h[6]);
              stats[10] += target * (raw[i].h[7] - raw[j].h[7]);
              for (int q = 0; q < 4; q++)
                stats[1 + q] += c[q];
              stats[5] += std::equal(raw[i].h, raw[i].h + 6, raw[j].h);
              stats[6] += std::equal(raw[i].h + 6, raw[i].h + 8, raw[j].h + 6);
            }
          }
        }
    if (idx % 997 == 37) {
      audit << idx;
      for (auto &version : test)
        for (auto &truth : version)
          for (auto n : truth)
            audit << " " << n;
      audit << "\n";
    }
  }
  for (int v = 0; v < 2; v++)
    for (int t = -1; t <= 1; t++)
      for (int r = -1; r <= 1; r++)
        for (int n = -1; n <= 1; n++)
          out << v << " " << t << " " << r << " " << n << " "
              << counts[v][1 + t][1 + r][1 + n] << "\n";
  for (int v = 0; v < 2; v++)
    for (int c = 0; c < 4; c++) {
      patch << v << " " << c - 1;
      for (auto total : replay[v][c])
        patch << " " << total;
      patch << "\n";
    }
  return rr.eof() && out && patch && audit && (argc == 5 || exceptions) ? 0 : 4;
}
