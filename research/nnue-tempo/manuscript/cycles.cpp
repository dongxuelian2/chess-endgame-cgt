// Exact preference graphs indexed by fixed kings, parent source and
// observables.
#include "../atlas/measure.h"
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>

using Mask = std::array<U, 64>;
struct Graph {
  Mask truth{}, neural{};
};

struct Row {
  int y;
  std::array<State, 64> x;
  std::array<Replies, 64> replies;
};

std::vector<int> shortest(const Mask &edges) {
  for (int a = 0; a < 64; a++)
    for (U bits = edges[a]; bits; bits &= bits - 1) {
      int b = __builtin_ctzll(bits);
      if ((edges[b] >> a) & 1)
        return {a, b};
    }
  int indegree[64]{};
  for (U bits : edges)
    for (; bits; bits &= bits - 1)
      ++indegree[__builtin_ctzll(bits)];
  U alive = ~U(0), queue = 0;
  for (int i = 0; i < 64; ++i)
    if (!indegree[i])
      queue |= U(1) << i;
  while (queue) {
    int i = __builtin_ctzll(queue);
    queue &= queue - 1;
    alive ^= U(1) << i;
    for (U bits = edges[i]; bits; bits &= bits - 1) {
      int j = __builtin_ctzll(bits);
      if (!--indegree[j])
        queue |= U(1) << j;
    }
  }
  std::vector<int> best;
  for (U start = alive; start; start &= start - 1) {
    const int a = __builtin_ctzll(start);
    U seen = U(1) << a, frontier = seen;
    int prev[64]{};
    for (int depth = 1; frontier && (best.empty() || depth < int(best.size()));
         ++depth) {
      U next = 0;
      for (U bits = frontier; bits; bits &= bits - 1) {
        int b = __builtin_ctzll(bits);
        if ((edges[b] >> a) & 1) {
          std::vector<int> path{b};
          while (path.back() != a)
            path.push_back(prev[path.back()]);
          std::reverse(path.begin(), path.end());
          best = path;
          next = 0;
          break;
        }
        U unseen = edges[b] & alive & ~seen;
        seen |= unseen;
        next |= unseen;
        for (; unseen; unseen &= unseen - 1)
          prev[__builtin_ctzll(unseen)] = b;
      }
      frontier = next;
    }
    if (best.size() == 3)
      break;
  }
  return best;
}

std::vector<Row> read_rows(const std::string &dir, const std::string &stem,
                           std::ifstream &dense, bool remaining) {
  std::ifstream file(dir + (remaining ? "/remaining/" : "/") + stem +
                         ".replies",
                     std::ios::binary);
  if (!file)
    std::abort();
  std::vector<Row> rows;
  uint32_t index;
  Row data;
  while (file.read(reinterpret_cast<char *>(&index), sizeof(index))) {
    data.y = index;
    dense.seekg(uint64_t(index) * sizeof(data.x));
    if (!dense.read(reinterpret_cast<char *>(data.x.data()), sizeof(data.x)) ||
        !file.read(reinterpret_cast<char *>(data.replies.data()),
                   sizeof(data.replies)))
      std::abort();
    rows.push_back(data);
  }
  if (!file.eof())
    std::abort();
  return rows;
}

int main(int argc, char **argv) {
  if (argc != 5)
    return 2;
  auto a = std::stoi(argv[2]), b = std::stoi(argv[3]);
  std::string dir = argv[1], output = argv[4];
  std::string stem = std::to_string(a) + "-" + std::to_string(b);
  std::ifstream dense(dir + "/" + stem + ".bin", std::ios::binary);
  std::ofstream counts(output + "/" + stem + ".cycles"),
      paths(output + "/" + stem + ".paths"),
      arcs(output + "/" + stem + ".arcs");
  if (!dense || !counts || !paths || !arcs)
    return 3;
  auto rows = read_rows(dir, stem, dense, false);
  auto rest = read_rows(dir, stem, dense, true);
  rows.insert(rows.end(), rest.begin(), rest.end());
  std::sort(rows.begin(), rows.end(),
            [](const Row &x, const Row &y) { return x.y < y.y; });
  size_t pos = 0;
  Geometry geo;
  std::map<std::tuple<int, int>, U> tally;
  while (pos < rows.size()) {
    const size_t initial = pos;
    int kings = rows[pos].y >> 6, wk = kings >> 6, bk = kings & 63;
    std::map<std::tuple<int, int, U>, Graph> graphs;
    U fixed = (U(1) << wk) | (U(1) << bk);
    while (pos < rows.size() && (rows[pos].y >> 6) == kings) {
      const Row &r = rows[pos++];
      if (pos < rows.size() && r.y == rows[pos].y)
        std::abort();
      int y = r.y & 63;
      U blockers = fixed | (U(1) << y);
      for (int s = 0; s < 64; ++s) {
        if (!r.x[s].v)
          continue;
        std::array<int, 64> dest;
        int n = 0;
        for (int x = 0; x < 64; ++x)
          if (r.x[x].v && legal_candidate(geo, a, b, s, x, y, blockers))
            dest[n++] = x;
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < i; ++j) {
            int l = dest[i], h = dest[j];
            int sl = sign(r.x[l].w), sh = sign(r.x[h].w);
            if (sl == sh || r.x[l].f != r.x[h].f ||
                r.replies[l].available != r.replies[h].available)
              continue;
            if (sl > sh)
              std::swap(l, h);
            auto &g = graphs[{s, r.x[l].f, r.replies[l].available}];
            g.truth[l] |= U(1) << h;
            if (r.x[h].n > r.x[l].n)
              g.neural[l] |= U(1) << h;
          }
      }
    }
    for (const auto &[key, g] : graphs) {
      auto path = shortest(g.truth);
      auto neural_path = path.empty() ? std::vector<int>{} : shortest(g.neural);
      ++tally[{path.size(), neural_path.size()}];
      if (path.size() < 3)
        continue;
      const auto [s, classical, flags] = key;
      for (size_t j = initial; j < pos; ++j) {
        const Row &r = rows[j];
        int y = r.y & 63;
        U blockers = fixed | (U(1) << y);
        U valid = 0;
        if (!r.x[s].v)
          continue;
        for (int v = 0; v < 64; ++v)
          if (r.x[v].v && r.x[v].f == classical &&
              r.replies[v].available == flags &&
              legal_candidate(geo, a, b, s, v, y, blockers))
            valid |= U(1) << v;
        for (U lo = valid; lo; lo &= lo - 1) {
          int l = __builtin_ctzll(lo);
          for (U hi = g.truth[l] & valid; hi; hi &= hi - 1) {
            int h = __builtin_ctzll(hi);
            int gap = sign(r.x[h].w) - sign(r.x[l].w);
            if (gap > 0)
              arcs << wk << ' ' << bk << ' ' << s << ' ' << classical << ' '
                   << flags << ' ' << y << ' ' << l << ' ' << h << ' ' << gap
                   << ' ' << r.x[h].n - r.x[l].n << ' ' << r.x[h].d - r.x[l].d
                   << ' ' << r.x[h].p - r.x[l].p << '\n';
          }
        }
      }

      // Prefer a neural-success witness of the same minimum length when
      // available.
      const bool correct = neural_path.size() == path.size();
      if (correct)
        path = neural_path;
      for (size_t i = 0; i < path.size(); ++i) {
        int l = path[i], h = path[(i + 1) % path.size()];
        bool found = false;
        for (size_t j = initial; j < pos; ++j) {
          const Row &r = rows[j];
          int y = r.y & 63;
          U blockers = fixed | (U(1) << y);
          if (!r.x[s].v || !r.x[l].v || !r.x[h].v || r.x[l].f != classical ||
              r.x[h].f != classical || r.replies[l].available != flags ||
              r.replies[h].available != flags ||
              sign(r.x[l].w) >= sign(r.x[h].w) ||
              (correct && r.x[l].n >= r.x[h].n) ||
              !legal_candidate(geo, a, b, s, l, y, blockers) ||
              !legal_candidate(geo, a, b, s, h, y, blockers))
            continue;
          paths << wk << ' ' << bk << ' ' << s << ' ' << classical << ' '
                << flags << ' ' << path.size() << ' ' << correct << ' ' << i
                << ' ' << y << ' ' << l << ' ' << h << ' ' << int(r.x[l].w)
                << ' ' << int(r.x[h].w) << ' ' << r.x[l].n << ' ' << r.x[h].n
                << '\n';
          found = true;
          break;
        }
        if (!found)
          std::abort();
      }
    }
  }
  for (auto [key, n] : tally)
    counts << std::get<0>(key) << ',' << std::get<1>(key) << ',' << n << '\n';
}
