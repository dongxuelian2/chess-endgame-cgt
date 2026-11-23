// Replay the two upper transforms of material bucket zero.
#pragma once
#include <array>
#include <cstdlib>
#include <fstream>
struct RawHead {
  int32_t ps, pos;
  uint8_t h[8];
};
struct Head {
  int bias[32], coef[32][8], out[32], offset;
  explicit Head(std::istream &weights) {
    for (int &v : bias)
      weights >> v;
    for (auto &row : coef)
      for (int &v : row)
        weights >> v;
    weights >> offset;
    for (int &v : out)
      weights >> v;
    if (!weights)
      std::abort();
  }
  int act(int value) const { return std::clamp(value >> 6, 0, 127); }
  int prepare(const RawHead &h, std::array<int, 32> &pre) const {
    int value = offset;
    for (int j = 0; j < 32; j++) {
      int s = bias[j];
      for (int i = 0; i < 8; i++)
        s += coef[j][i] * h.h[i];
      pre[j] = s;
      value += out[j] * act(s);
    }
    if (value != h.pos)
      std::abort();
    return (h.ps + value) / 16;
  }
  // A pair earns 0..4 quarter-points across the two possible donor endpoints.
  std::array<int, 4> credits(const RawHead &a, const RawHead &b,
                             const std::array<int, 32> &pa,
                             const std::array<int, 32> &pb, int truth) const {
    int ab = offset, ba = offset;
    for (int j = 0; j < 32; j++) {
      const int shift =
          coef[j][6] * (a.h[6] - b.h[6]) + coef[j][7] * (a.h[7] - b.h[7]);
      ab += out[j] * act(pb[j] + shift);
      ba += out[j] * act(pa[j] - shift);
    }
    int ah = (a.ps + a.pos) / 16, bl = (b.ps + b.pos) / 16;
    auto dual = [&](int leftmix, int rightmix) {
      return 2 + truth * (sign((a.ps + leftmix) / 16 - bl) +
                          sign(ah - (b.ps + rightmix) / 16));
    };
    return {2 + 2 * truth * sign(ah - bl), dual(ab, ba), dual(ba, ab),
            dual(b.pos, a.pos)};
  }
};
