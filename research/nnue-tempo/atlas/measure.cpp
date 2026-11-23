#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "measure.h"
struct Count {
  U available = 0, required = 0, forced = 0, pairs = 0, n = 0, h = 0, d = 0,
    p = 0, only = 0;
  U direction_good = 0, direction_bad = 0, wdl_tie_diff = 0;
  U clusters = 0;
  int witness = -1, hi = -1, lo = -1, source = -1;
};
struct Corpus {
  U valid = 0, h_errors = 0, hn_errors = 0, ambiguous = 0, pairs = 0, n = 0,
    h = 0, p = 0, d = 0, win = 0, loss = 0;
};
template <class Fn> void bits(U v, Fn action) {
  while (v) {
    int b = __builtin_ctzll(v);
    action(b);
    v &= v - 1;
  }
}
int credit(int a, int b) { return 1 + sign(a - b); }
int main(int argc, char **argv) {
  if (argc != 5 && argc != 6)
    return 2;
  std::string dir = argv[1], dest = argv[2];
  int a = std::stoi(argv[3]), b = std::stoi(argv[4]);
  std::string stem = std::to_string(a) + "-" + std::to_string(b);
  std::ifstream dense(dir + "/" + stem + ".bin", std::ios::binary),
      responses(dir + (argc == 6 ? "/remaining/" : "/") + stem + ".replies",
                std::ios::binary);
  std::ofstream rows(dest + "/" + stem + ".metrics"),
      pairsout(dest + "/" + stem + ".overlap");
  if (!dense || !responses || !rows || !pairsout)
    return 2;
  std::ofstream training(dest + "/" + stem + ".pairs");
  std::ofstream exclusiveout(dest + "/" + stem + ".exclusive");
  bool seen_exclusive[K]{};
  Geometry geo;
  Corpus c;
  Count stats[2][K]{};
  U overlaps[2][K][K]{};
  U blockmask = 0, matched[2]{}, equalpairs[2]{};
  uint32_t selected;
  responses.read((char *)&selected, 4);
  std::array<State, 64> x;
  std::array<Replies, 64> r;
  std::array<int, 64> order;
  for (unsigned index = 0; index < 64u * 64 * 64; index++) {
    if (!dense.read((char *)x.data(), sizeof(x)))
      return 3;
    int wk = index >> 12, bk = (index >> 6) & 63, y = index & 63;
    int F = fold(wk, bk);
    bool inspect = responses && index == selected;
    if (inspect && !responses.read((char *)r.data(), sizeof(r)))
      return 3;
    int count = 0;
    for (int i = 0; i < 64; i++)
      if (x[i].v) {
        order[count++] = i;
        ++c.valid;
        int w = sign(x[i].w);
        c.win += w > 0;
        c.loss += w < 0;
        if (inspect) {
          bits(r[i].available, [&](int q) { stats[F][q].available++; });
          bits(r[i].available & ~r[i].legal,
               [&](int q) { stats[F][q].forced++; });
          U required = r[i].legal & ~(w > 0    ? r[i].win
                                      : w == 0 ? r[i].draw
                                               : r[i].legal);
          bits(required, [&](int q) { stats[F][q].required++; });
        }
      }
    // Exact empirical Bayes error in each fixed-background classical score
    // fiber.
    std::sort(order.begin(), order.begin() + count, [&](int i, int j) {
      return x[i].f != x[j].f ? x[i].f < x[j].f : x[i].n < x[j].n;
    });
    for (int s = 0; s < count;) {
      int end = s;
      int w[3] = {};
      while (end < count && x[order[end]].f == x[order[s]].f) {
        int next = end;
        int wn[3] = {};
        while (next < count && x[order[next]].f == x[order[s]].f &&
               x[order[next]].n == x[order[end]].n) {
          int label = 1 + sign(x[order[next]].w);
          ++wn[label];
          ++w[label];
          ++next;
        }
        c.hn_errors += next - end - *std::max_element(wn, wn + 3);
        end = next;
      }
      int errors = end - s - *std::max_element(w, w + 3);
      c.h_errors += errors;
      if (errors) {
        c.ambiguous += end - s;
        for (int i = s; i < end; i++)
          for (int j = i + 1; j < end; j++) {
            auto *hi = &x[order[i]], *lo = &x[order[j]];
            if (sign(hi->w) == sign(lo->w))
              continue;
            if (hi->w < lo->w)
              std::swap(hi, lo);
            ++c.pairs;
            c.n += credit(hi->n, lo->n);
            c.h += credit(hi->h, lo->h);
            c.p += credit(hi->p, lo->p);
            c.d += credit(hi->d, lo->d);
          }
      }
      s = end;
    }
    if (!inspect)
      continue;
    U src[64] = {}, block = (U(1) << wk) | (U(1) << bk) | (U(1) << y);
    for (int i = 0; i < 64; i++)
      if (x[i].v)
        for (int j = 0; j < 64; j++)
          if (x[j].v && legal_candidate(geo, a, b, i, j, y, block))
            src[j] |= U(1) << i;
    U minimum = ~U(0);
    int train_hi = -1, train_lo = -1;
    for (int i = 0; i < 64; i++)
      for (int j = 0; j < i; j++)
        if ((src[i] & src[j]) && x[i].f == x[j].f) {
          int hi = i, lo = j;
          if (sign(x[i].w) == sign(x[j].w)) {
            ++equalpairs[F];
            bits(r[i].available ^ r[j].available,
                 [&](int t) { stats[F][t].wdl_tie_diff++; });
            continue;
          }
          if (x[i].w < x[j].w)
            std::swap(hi, lo);
          ++matched[F];
          U priority =
              mix((U(index) * 4096 + i * 64 + j) ^ (U(a * 6 + b) << 40));
          if (priority < minimum) {
            minimum = priority;
            train_hi = hi;
            train_lo = lo;
          }
          U different = r[hi].available ^ r[lo].available;
          bits(different & r[hi].available,
               [&](int t) { stats[F][t].direction_good++; });
          bits(different & r[lo].available,
               [&](int t) { stats[F][t].direction_bad++; });
          U covered = r[hi].legal & r[lo].legal;
          U still = (r[hi].win & ~r[lo].win) | (r[hi].draw & ~r[lo].draw);
          U erased = covered & ~still & domain;
          if (erased && !(erased & (erased - 1))) {
            int t = __builtin_ctzll(erased);
            if (!seen_exclusive[t]) {
              seen_exclusive[t] = true;
              exclusiveout << names[t] << " " << index << " " << hi << " " << lo
                           << " " << __builtin_ctzll(src[hi] & src[lo]) << "\n";
            }
          }
          blockmask |= erased;
          bits(erased, [&](int t) {
            auto &z = stats[F][t];
            ++z.pairs;
            z.n += credit(x[hi].n, x[lo].n);
            z.h += credit(x[hi].h, x[lo].h);
            z.p += credit(x[hi].p, x[lo].p);
            z.d += credit(x[hi].d, x[lo].d);
            z.only += (erased & (erased - 1)) == 0;
            if (z.witness < 0 && x[hi].n > x[lo].n) {
              z.witness = index;
              z.hi = hi;
              z.lo = lo;
              z.source = __builtin_ctzll(src[hi] & src[lo]);
            }
            bits(erased, [&](int other) { ++overlaps[F][t][other]; });
          });
        }
    if (train_hi >= 0) {
      auto &h = x[train_hi], &l = x[train_lo];
      training << index << " " << F << " " << r[train_hi].available << " "
               << r[train_lo].available << " " << int(h.w) << " " << int(l.w)
               << " " << h.n - l.n << " " << h.d - l.d << " " << h.p - l.p
               << " " << train_hi << " " << train_lo << "\n";
    }
    if (y == 63) {
      bits(blockmask, [&](int t) { stats[F][t].clusters++; });
      blockmask = 0;
    }
    responses.read((char *)&selected, 4);
  }
  if (responses || !responses.eof())
    return 4;
  rows << "c " << c.valid << " " << c.win << " " << c.loss << " " << c.h_errors
       << " " << c.hn_errors << " " << c.ambiguous << " " << c.pairs << " "
       << c.n << " " << c.h << " " << c.p << " " << c.d << "\n";
  for (int f = 0; f < 2; f++)
    for (int t = 0; t < K; t++) {
      const auto &z = stats[f][t];
      rows << f << " " << names[t] << " " << z.available << " " << z.required
           << " " << z.forced << " " << matched[f] << " " << z.pairs << " "
           << z.n << " " << z.h << " " << z.p << " " << z.d << " " << z.only
           << " " << z.direction_good << " " << z.direction_bad << " "
           << z.wdl_tie_diff << " " << equalpairs[f] << " " << z.clusters << " "
           << z.witness << " " << z.hi << " " << z.lo << " " << z.source
           << "\n";
      for (int u = t; u < K; u++)
        if (overlaps[f][t][u])
          pairsout << f << " " << names[t] << " " << names[u] << " "
                   << overlaps[f][t][u] << "\n";
    }
}
