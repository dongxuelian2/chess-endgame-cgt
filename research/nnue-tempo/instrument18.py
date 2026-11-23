"""Expose SF18's large-network readout and a paired FullThreats channel knockout."""

import subprocess
import sys
from pathlib import Path


def edit(path, anchor, replacement):
    s = path.read_text()
    assert s.count(anchor) == 1
    path.write_text(s.replace(anchor, replacement))


tree = Path(sys.argv[1])
assert (
    subprocess.check_output(["git", "-C", tree, "rev-parse", "HEAD"], text=True).strip()
    == "cb3d4ee9b47d0c5aae855b12379378ea1439675c"
)
src = tree / "src"
p = src / "evaluate.cpp"
edit(
    p,
    "auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();",
    "static auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();\n    accumulators->reset();",
)
edit(
    p,
    "auto caches       = std::make_unique<Eval::NNUE::AccumulatorCaches>(networks);",
    """static auto caches = std::make_unique<Eval::NNUE::AccumulatorCaches>(networks);
    const auto [piece, head] = networks.big.evaluate(pos, *accumulators, caches->big);
    NNUE::ThreatFeatureSet::IndexList white, black;
    NNUE::ThreatFeatureSet::append_active_indices(WHITE, pos, white);
    NNUE::ThreatFeatureSet::append_active_indices(BLACK, pos, black);
    return "raw18 " + std::to_string(piece) + " " + std::to_string(head)
         + " " + std::to_string(white.size() + black.size());""",
)
p = src / "nnue/nnue_feature_transformer.h"
anchor = (
    "const auto& threatAccumulatorState = accumulatorStack.latest<ThreatFeatureSet>();"
)
edit(
    p,
    anchor,
    """static const AccumulatorState<ThreatFeatureSet> zeroThreats{};
        static const bool knockout = std::getenv("SF18_ZERO_THREATS") != nullptr;
        const auto& threatAccumulatorState = knockout ? zeroThreats : accumulatorStack.latest<ThreatFeatureSet>();""",
)
