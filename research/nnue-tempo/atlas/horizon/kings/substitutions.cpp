// A common king layout, pawn, legal parent and both destinations under R/Q
// substitution.
#include "../../measure.h"
#include <array>
#include <fstream>
#include <string>
struct Profile {
  int8_t actual, runner, pawn_best_only, follow, fork_only, valid;
};
struct Material {
  std::ifstream dense, profiles;
  std::array<State, 64> state;
  std::array<Profile, 64> profile;
  Material(const std::string &root, const std::string &piece)
      : dense(root + "/" + piece + "-1.bin", std::ios::binary),
        profiles(root + "/horizon/kings/rank-" + piece + ".bin",
                 std::ios::binary) {}
  bool read(int index) {
    dense.seekg(uint64_t(index) * sizeof(state));
    return dense.read((char *)state.data(), sizeof(state)) &&
           profiles.read((char *)profile.data(), sizeof(profile));
  }
};
int main(int argc, char **argv) {
  if (argc != 4)
    return 2;
  Material rook(argv[1], "4"), queen(argv[1], "5");
  std::ofstream rows(argv[2]), counts(argv[3]);
  Geometry geo;
  unsigned long long total = 0, contexts = 0;
  for (int w = 0; w < 64; w++)
    if (w % 8 < 4)
      for (int k = 0; k < 64; k++)
        for (int y = 16; y < 24; y++) {
          int index = (w * 64 + k) * 64 + y;
          if (!rook.read(index) || !queen.read(index))
            return 3;
          U src[64]{}, occupied = (U(1) << w) | (U(1) << k) | (U(1) << y);
          for (int x = 0; x < 64; x++)
            if (rook.profile[x].valid && queen.profile[x].valid &&
                rook.profile[x].actual == queen.profile[x].actual) {
              for (int s = 0; s < 64; s++)
                if (rook.state[s].v && queen.state[s].v &&
                    legal_candidate(geo, 4, 1, s, x, y, occupied))
                  src[x] |= U(1) << s;
            }
          bool has = false;
          for (int h = 0; h < 64; h++)
            for (int l = 0; l < 64; l++) {
              U common = src[h] & src[l];
              if (!common)
                continue;
              if (rook.state[h].f != rook.state[l].f ||
                  queen.state[h].f != queen.state[l].f)
                continue;
              if (rook.profile[h].actual <= rook.profile[l].actual)
                continue;
              if (rook.profile[h].runner != rook.profile[h].actual ||
                  queen.profile[h].runner != queen.profile[h].actual)
                continue;
              ++total;
              has = true;
              rows << w << " " << k << " " << y << " " << h << " " << l << " "
                   << __builtin_ctzll(common) << "\n";
            }
          contexts += has;
        }
  counts << "candidates,contexts\n" << total << "," << contexts << "\n";
  return rows && counts ? 0 : 4;
}
