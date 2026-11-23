// Count each sibling pair once across all common source squares.
#include "../../measure.h"
#include "../head.h"
#include <array>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <utility>
struct Profile {
  int8_t actual, runner, pawn_best_only, follow, fork_only, valid;
};
int main(int argc, char **argv) {
  if (argc != 7)
    return 2;
  Geometry g;
  int a = std::stoi(argv[1]);
  std::ifstream dense(argv[2], std::ios::binary), in(argv[3], std::ios::binary);
  std::ifstream weights(argv[4]), raw(argv[5], std::ios::binary);
  Head head(weights);
  using Sample = std::pair<U, std::array<int, 5>>;
  std::priority_queue<Sample> samples[7];
  const char *family[] = {"eligible",     "strict",    "discovered",
                          "direct_check", "quiet_pin", "pin_or_check",
                          "royal_fork"};
  uint64_t counts[7][3]{}, keep[7]{}, drop[7]{};
  for (int w = 0; w < 64; w++)
    if (w % 8 < 4)
      for (int k = 0; k < 64; k++)
        for (int y = 16; y < 24; y++) {
          std::array<State, 64> st;
          std::array<Profile, 64> p;
          dense.seekg(uint64_t((w * 64 + k) * 64 + y) * sizeof(st));
          if (!dense.read((char *)st.data(), sizeof(st)) ||
              !in.read((char *)p.data(), sizeof(p)))
            return 3;
          std::array<RawHead, 64> heads;
          std::array<std::array<int, 32>, 64> pre{};
          if (a == 3) {
            raw.seekg(uint64_t((w * 64 + k) * 64 + y) * sizeof(heads));
            if (!raw.read((char *)heads.data(), sizeof(heads)))
              return 3;
            for (int x = 0; x < 64; x++)
              if (p[x].valid && head.prepare(heads[x], pre[x]) != st[x].n)
                return 5;
          }
          U occ = (U(1) << w) | (U(1) << k) | (U(1) << y), src[64]{};
          for (int x = 0; x < 64; x++)
            if (p[x].valid)
              for (int s = 0; s < 64; s++)
                if (st[s].v && legal_candidate(g, a, 1, s, x, y, occ))
                  src[x] |= U(1) << s;
          for (int i = 0; i < 64; i++)
            for (int j = 0; j < i; j++) {
              if (!(src[i] & src[j]) || st[i].f != st[j].f)
                continue;
              int truth = sign(p[i].actual - p[j].actual);
              int pred = 1 + truth * sign(st[i].n - st[j].n);
              counts[0][pred]++;
              if (!truth)
                continue;
              counts[1][pred]++;
              Profile h = p[truth > 0 ? i : j], l = p[truth > 0 ? j : i];
              if (l.actual != 0)
                continue;
              auto record = [&](int f) {
                counts[f][pred]++;
                int hi = truth > 0 ? i : j, lo = truth > 0 ? j : i;
                U key = ((((U(w) * 64 + k) * 64 + y) * 64 + hi) * 64 + lo);
                Sample candidate = {mix(key + 71113), {w, k, y, hi, lo}};
                auto &heap = samples[f];
                if (heap.size() < 256 || candidate < heap.top()) {
                  heap.push(candidate);
                  if (heap.size() > 256)
                    heap.pop();
                }
                if (a == 3) {
                  auto c =
                      head.credits(heads[i], heads[j], pre[i], pre[j], truth);
                  keep[f] += c[1];
                  drop[f] += c[2];
                }
              };
              if (h.pawn_best_only && l.runner == 0) {
                int f = l.follow == 1   ? 2
                        : l.follow == 2 ? 3
                        : l.follow == 4 ? 4
                        : l.follow == 6 ? 5
                                        : 0;
                if (f)
                  record(f);
              }
              if (h.fork_only && h.runner == 0 && l.runner == 0)
                record(6);
            }
        }
  if (in.peek() != EOF)
    return 4;
  std::ofstream sample_out(argv[6]);
  for (int f = 0; f < 7; f++) {
    auto &r = counts[f];
    std::cout << a << "," << family[f] << "," << r[0] + r[1] + r[2] << ",";
    if (f)
      std::cout << r[0] << "," << r[1] << "," << r[2];
    else
      std::cout << ",,";
    std::cout << "," << (a == 3 && f >= 2 ? std::to_string(keep[f]) : "") << ","
              << (a == 3 && f >= 2 ? std::to_string(drop[f]) : "") << "\n";
    while (!samples[f].empty()) {
      sample_out << family[f];
      for (int v : samples[f].top().second)
        sample_out << " " << v;
      sample_out << "\n";
      samples[f].pop();
    }
  }
  return sample_out ? 0 : 4;
}
