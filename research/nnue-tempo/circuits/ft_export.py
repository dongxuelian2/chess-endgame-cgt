"""Instrument a separate clean SF14.1 tree to expose the transformed features and H1 logits."""

import argparse
from pathlib import Path

from instrument import insert, instrument


def export_accumulators(tree):
    insert(
        tree / "src/nnue/evaluate_nnue.cpp",
        "encoded[4096] = 0;",
        r"""
    if (std::getenv("NNUE_CIRCUIT_RAW")) {
        char raw[8193];
        for (int i=0; i<2048; ++i) {
            const Color c = i<1024 ? pos.side_to_move() : ~pos.side_to_move();
            const uint16_t x = uint16_t(pos.state()->accumulator.accumulation[c][i%1024]);
            for (int j=0; j<4; ++j) raw[4*i+j] = hex[(x>>(4*j))&15];
        }
        raw[8192] = 0;
        std::cout << "rawft " << raw << '\n';
    }""",
    )


def export_classical(tree):
    path = tree / "src/evaluate.cpp"
    insert(
        path,
        "// Evaluation grain",
        r"""
    if constexpr (T) {
        if (std::getenv("NNUE_CIRCUIT_CLASSICAL"))
            std::cout << "pregrain " << int(pos.side_to_move()==WHITE ? v : -v) << '\n';
    }""",
    )
    insert(
        path,
        "v = Evaluation<TRACE>(pos).value();",
        r"""
  if (std::getenv("NNUE_CIRCUIT_CLASSICAL")) {
      const auto mat = Material::probe(pos);
      std::cout << "terms " << int(mat->game_phase()) << " " << int(mat->specialized_eval_exists());
      for (const int term : {int(PAWN),int(KNIGHT),int(BISHOP),int(ROOK),int(QUEEN),int(KING),
                             int(MATERIAL),int(IMBALANCE),int(MOBILITY),int(THREAT),int(PASSED),int(SPACE),int(WINNABLE),int(TOTAL)}) {
          const auto s = Trace::scores[term][WHITE] - Trace::scores[term][BLACK];
          std::cout << " " << int(mg_value(s)) << " " << int(eg_value(s));
      }
      std::cout << '\n';
  }""",
    )


def export_full_classical(tree):
    path = tree / "src/evaluate.cpp"
    source = path.read_text()
    for threshold in ("LazyThreshold1", "LazyThreshold2"):
        old = f"if (lazy_skip({threshold}))"
        assert source.count(old) == 1
        source = source.replace(
            old, f'if (lazy_skip({threshold}) && !std::getenv("NNUE_CIRCUIT_FULL_HCE"))'
        )
    path.write_text(source)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tree", type=Path)
    tree = p.parse_args().tree
    instrument(tree)
    insert(
        tree / "src/nnue/evaluate_nnue.cpp",
        "const auto h1 = h2 + H2::SelfBufferSize + A2::SelfBufferSize;",
        r"""
    const auto pre = reinterpret_cast<const int32_t*>(h1 + Layers::HiddenLayer1::SelfBufferSize);
    constexpr char hex[] = "0123456789abcdef";
    char encoded[4097];
    for (int i=0; i<2048; ++i) {
        const auto x = transformedFeatures[i];
        encoded[2*i] = hex[x>>4];
        encoded[2*i+1] = hex[x&15];
    }
    encoded[4096] = 0;
    std::cout << "features " << encoded;
    for (int i=0; i<8; ++i) std::cout << " " << pre[i];
    std::cout << '\n';""",
    )
    export_accumulators(tree)
    export_classical(tree)
    export_full_classical(tree)
