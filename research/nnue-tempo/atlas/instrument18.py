"""Evaluate all subsets of four disjoint FullThreats edge classes from a fresh static accumulator."""

import shutil
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
src = root / "sf18"
dst = root / "atlas/engine18"
rev = subprocess.check_output(
    ["git", "-C", src, "rev-parse", "HEAD"], text=True
).strip()
assert rev == "cb3d4ee9b47d0c5aae855b12379378ea1439675c"
dst.mkdir(exist_ok=True)
subprocess.run(
    ["tar", "xf", "-", "-C", dst],
    input=subprocess.check_output(
        ["git", "-C", src, "archive", "HEAD", "src", "scripts"]
    ),
    check=True,
)
for path in (src / "src").glob("*.nnue"):
    shutil.copy(path, dst / "src")
p = dst / "src/nnue/features/full_threats.cpp"
s = p.read_text()
begin = s.index("void FullThreats::append_active_indices")
end = s.index("// Get a list of indices for recently changed", begin)
body = s[begin:end]
old = "if (index < Dimensions)"
assert body.count(old) == 3
s = (
    s[:begin]
    + """int atlas_mask=15;
bool atlas_keep(Piece attacker, Piece victim) {
    const int group = color_of(attacker)==color_of(victim) ? 3 :
           (type_of(attacker)==KING || type_of(victim)==KING) ? 2 :
           color_of(attacker)==BLACK ? 0 : 1;
    return atlas_mask & (1<<group);
}
"""
    + body.replace(old, "if (index < Dimensions && atlas_keep(attacker,attacked))")
    + s[end:]
)
p.write_text(s)
p = dst / "src/evaluate.cpp"
s = p.read_text().replace(
    "namespace Stockfish {",
    "namespace Stockfish::Eval::NNUE::Features { extern int atlas_mask; }\nnamespace Stockfish {",
    1,
)
s = s.replace(
    "auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();",
    "static auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();",
)
old = "auto caches       = std::make_unique<Eval::NNUE::AccumulatorCaches>(networks);"
assert s.count(old) == 1
s = s.replace(
    old,
    """static auto caches=std::make_unique<Eval::NNUE::AccumulatorCaches>(networks);
    if (std::getenv("ATLAS_RELEASE_ONLY")) {
        NNUE::Features::atlas_mask=15;
        accumulators->reset();
        return "release18 " + std::to_string(Eval::evaluate(networks,pos,*accumulators,*caches,0));
    }
    std::string atlas="atlas18";
    for (int mask=0;mask<16;++mask) {
        NNUE::Features::atlas_mask=mask;
        accumulators->reset();
        const auto [p,h]=networks.big.evaluate(pos,*accumulators,caches->big);
        atlas+=" "+std::to_string(p+h);
    }
    return atlas;""",
)
p.write_text(s)
subprocess.run(
    ["make", "-C", dst / "src", "-j10", "build", "ARCH=x86-64-avx2"], check=True
)
