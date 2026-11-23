#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
using U = uint64_t;
const char *names[] = {
#define RULE(id, desc) #id,
#include "rules.def"
#undef RULE
};
constexpr int K = sizeof(names) / sizeof(*names);
constexpr U domain = (U(1) << K) - 1;
struct State {
  int16_t h, f, n, p, d;
  int8_t w;
  uint8_t v;
};
struct Replies {
  U available, legal, draw, win;
};
static_assert(sizeof(State) == 12 && sizeof(Replies) == 32);
int sign(int x) { return (x > 0) - (x < 0); }
U mix(U x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
int fold(int w, int b) {
  int key = 4096;
  for (int m : {0, 7, 56, 63}) {
    int a = w ^ m, c = b ^ m;
    key = std::min(key, std::min(a, c) * 64 + std::max(a, c));
  }
  return (mix(key + 41981) >> 8) & 1;
}
struct Geometry {
  U between[64][64]{};
  bool can[6][64][64]{};
  Geometry() {
    for (int i = 0; i < 64; i++)
      for (int j = 0; j < 64; j++)
        if (i != j) {
          int f = j % 8 - i % 8, r = j / 8 - i / 8;
          bool orth = f == 0 || r == 0, diag = std::abs(f) == std::abs(r);
          can[1][i][j] = f == 0 && (r == 1 || (i / 8 == 1 && r == 2));
          can[2][i][j] = std::abs(f * r) == 2;
          can[3][i][j] = diag;
          can[4][i][j] = orth;
          can[5][i][j] = orth || diag;
          if (orth || diag) {
            int step = sign(f) + 8 * sign(r);
            for (int s = i + step; s != j; s += step)
              between[i][j] |= U(1) << s;
          }
        }
  }
};

bool legal_candidate(const Geometry &geo, int a, int b, int source, int target,
                     int enemy, U blockers) {
  bool ep = a == 1 && b == 1 && target - source == 16 &&
            enemy / 8 == target / 8 && std::abs(enemy % 8 - target % 8) == 1;
  return !ep && geo.can[a][source][target] &&
         !(blockers & geo.between[source][target]);
}
