"""Add integer static-evaluation readouts to disposable Stockfish source trees."""

import argparse
import subprocess
from pathlib import Path


def insert(path, anchor, addition):
    source = path.read_text()
    assert source.count(anchor) == 1, (path, anchor)
    path.write_text(source.replace(anchor, anchor + addition))


def instrument(tree):
    rev = subprocess.check_output(
        ["git", "-C", tree, "rev-parse", "HEAD"], text=True
    ).strip()
    sf14 = rev == "7262fd5d14810b7b495b5038e348a448fda1bcc3"
    assert sf14 or rev == "c306d838697011da0a960758dde3f7ede6849060"
    src = tree / "src"
    call = "NNUE::evaluate(pos, false)" if sf14 else "NNUE::evaluate(pos)"
    insert(
        src / "evaluate.cpp",
        "v = Evaluation<TRACE>(pos).value();",
        f'\n  auto n = {call};\n  return "scores " + std::to_string(int(v)) + " " + std::to_string(int(n));',
    )
    if sf14:
        insert(
            src / "nnue/evaluate_nnue.cpp",
            "const auto positional = network[bucket]->propagate(transformedFeatures, buffer)[0];",
            """
    using H2 = Layers::HiddenLayer2;
    using A2 = Layers::AffineTransform<Layers::HiddenLayer1, 32>;
    const auto h2 = reinterpret_cast<const uint8_t*>(buffer + Network::SelfBufferSize);
    const auto h1 = h2 + H2::SelfBufferSize + A2::SelfBufferSize;
    std::cout << "units " << psqt << " " << positional;
    for (int i = 0; i < 8; ++i) std::cout << " " << int(h1[i]);
    for (int i = 0; i < 32; ++i) std::cout << " " << int(h2[i]);
    std::cout << '\\n';""",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    instrument(parser.parse_args().tree)
