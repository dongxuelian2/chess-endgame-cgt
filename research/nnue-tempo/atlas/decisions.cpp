#include "measure.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

struct Outcome {
  int sum = 0, n = 0;
  void add(int value) {
    sum += value;
    ++n;
  }
};
struct Effect {
  uint64_t better = 0, equal = 0, worse = 0, changed = 0;
  double utility = 0;
  void record(Outcome choice, Outcome base) {
    int numerator = choice.sum * base.n - base.sum * choice.n;
    better += numerator > 0;
    worse += numerator < 0;
    equal += numerator == 0;
    changed += choice.n < base.n;
    utility += double(numerator) / (base.n * choice.n);
  }
};
int main(int argc, char **argv) {
  if (argc != 6 && argc != 7)
    return 2;
  std::string stem = std::string(argv[3]) + "-" + argv[4], dir = argv[1],
              dest = argv[2];
  int a = std::stoi(argv[3]), b = std::stoi(argv[4]);
  std::ifstream dense(dir + "/" + stem + ".bin", std::ios::binary),
      reply(dir + (argc == 7 ? "/remaining/" : "/") + stem + ".replies",
            std::ios::binary),
      weights(argv[5]);
  std::ofstream out(dest + "/" + stem + ".decisions");
  int w[K], orientation[K];
  for (int j = 0; j < K; j++)
    if (!(weights >> w[j] >> orientation[j]))
      return 3;
  if (!out || !dense || !reply)
    return 4;
  Geometry geo;
  U roots = 0, ties = 0, mixed = 0;
  double hua = 0;
  Effect effects[K + 7];
  uint32_t idx;
  std::array<State, 64> x;
  std::array<Replies, 64> r;
  std::array<int, 64> model{}, baseline{};
  while (reply.read((char *)&idx, 4)) {
    if (!reply.read((char *)r.data(), sizeof(r)))
      return 4;
    dense.seekg(uint64_t(idx) * sizeof(x));
    if (!dense.read((char *)x.data(), sizeof(x)))
      return 4;
    int wk = idx >> 12, bk = idx >> 6 & 63, y = idx & 63;
    U blockers = (U(1) << wk) | (U(1) << bk) | (U(1) << y);
    for (int t = 0; t < 64; t++)
      if (x[t].v) {
        U tags = r[t].available;
        model[t] = 0;
        for (U z = tags; z; z &= z - 1)
          model[t] += w[__builtin_ctzll(z)];
        baseline[t] = ((tags >> 0) & 1) + ((tags >> 1) & 1) + ((tags >> 8) & 1);
      }
    for (int src = 0; src < 64; src++)
      if (x[src].v) {
        int leader = 32767, best = -2, count = 0, sz = 0, dsts[64];
        Outcome base;
        for (int t = 0; t < 64; t++)
          if (x[t].v && legal_candidate(geo, a, b, src, t, y, blockers)) {
            ++count;
            best = std::max(best, -sign(x[t].w));
            if (x[t].f < leader) {
              leader = x[t].f;
              sz = 0;
              base = Outcome{};
            }
            if (x[t].f == leader) {
              dsts[sz++] = t;
              base.add(-sign(x[t].w));
            }
          }
        if (count < 2)
          continue;
        ++roots;
        ties += sz > 1;
        hua += double(base.sum) / sz;
        bool varied = false;
        for (int j = 1; j < sz; j++)
          varied |= sign(x[dsts[j]].w) != sign(x[dsts[0]].w);
        if (!varied)
          continue;
        ++mixed;
        Outcome targets[K]{};
        int mins[7];
        std::fill(mins, mins + 7, std::numeric_limits<int>::max());
        Outcome choices[7]{};
        for (int j = 0; j < sz; j++) {
          int t = dsts[j], v = -sign(x[t].w);
          for (U mask = r[t].available; mask; mask &= mask - 1)
            targets[__builtin_ctzll(mask)].add(v);
          int costs[] = {model[t], baseline[t], x[t].n, x[t].p,
                         x[t].d,   -v,          x[t].h};
          for (int k = 0; k < 7; k++) {
            if (costs[k] < mins[k]) {
              choices[k] = Outcome{};
              mins[k] = costs[k];
            }
            if (costs[k] == mins[k])
              choices[k].add(v);
          }
        }
        for (int j = 0; j < K; j++) {
          Outcome pick = targets[j];
          if (orientation[j] > 0)
            pick = {base.sum - pick.sum, base.n - pick.n};
          if (orientation[j] == 0 || pick.n == 0)
            pick = base;
          effects[j].record(pick, base);
        }
        for (int j = 0; j < 7; j++)
          effects[K + j].record(choices[j], base);
      }
  }
  const char *extra[] = {
      "fitted",     "simple_tactics", "native141",         "psqt",
      "positional", "oracle",         "released_classical"};
  for (int j = 0; j < K + 7; j++) {
    const auto &e = effects[j];
    out.precision(16);
    out << (j < K ? names[j] : extra[j - K]) << " " << roots << " " << ties
        << " " << mixed << " " << hua << " " << e.better << " " << e.equal
        << " " << e.worse << " " << e.changed << " " << e.utility << "\n";
  }
}
