"""Build the batch evaluator from an isolated checkout of Stockfish 14.1."""

import shutil
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
dst = root / "atlas/engine"
src = root / "clean-sf141"
revision = subprocess.check_output(
    ["git", "-C", src, "rev-parse", "HEAD"], text=True
).strip()
assert revision == "7262fd5d14810b7b495b5038e348a448fda1bcc3"
dst.mkdir(parents=True, exist_ok=True)
archive = subprocess.check_output(["git", "-C", src, "archive", "HEAD", "src"])
subprocess.run(["tar", "xf", "-", "-C", dst], input=archive, check=True)
s = dst / "src"
shutil.copy(src / "src/nn-13406b1dcbe0.nnue", s)


def replace(name, old, new):
    p = s / name
    v = p.read_text()
    assert v.count(old) == 1, (name, old)
    p.write_text(v.replace(old, new))


replace(
    "position.h",
    "  void flip();",
    "  void flip();\n  void atlas_set(Square wk, Square bk, Piece a, Square x, Piece b, Square y, StateInfo* si, Thread* th);",
)
replace(
    "position.cpp",
    "void Position::set_state(StateInfo* si) const {",
    """void Position::atlas_set(Square wk, Square bk, Piece a, Square x, Piece b, Square y, StateInfo* si, Thread* th) {
  std::memset(this,0,sizeof(*this)); std::memset(si,0,sizeof(*si));
  st=si; thisThread=th; sideToMove=BLACK; si->epSquare=SQ_NONE;
  put_piece(W_KING,wk); put_piece(B_KING,bk); put_piece(a,x); put_piece(b,y); set_state(si);
}
void Position::set_state(StateInfo* si) const {""",
)
replace(
    "evaluate.cpp",
    "namespace Stockfish {",
    "namespace Stockfish {\nbool atlas_full=false;",
)
for i in (1, 2):
    old = f"if (lazy_skip(LazyThreshold{i}))"
    replace("evaluate.cpp", old, f"if (!atlas_full && lazy_skip(LazyThreshold{i}))")
replace("evaluate.cpp", "v = (v / 16) * 16;", "if (!atlas_full) v = (v / 16) * 16;")
replace(
    "evaluate.cpp",
    "Value Eval::evaluate(const Position& pos) {",
    """int atlas_hand(const Position& pos, bool full) {
  atlas_full=full; return Evaluation<NO_TRACE>(pos).value();
}
Value Eval::evaluate(const Position& pos) {""",
)
replace(
    "nnue/evaluate_nnue.cpp",
    "  Value evaluate(const Position& pos, bool adjusted) {",
    "  int atlas_ps, atlas_deep;\n  Value evaluate(const Position& pos, bool adjusted) {",
)
replace(
    "nnue/evaluate_nnue.cpp",
    "    // Give more value",
    "    atlas_ps=psqt / OutputScale; atlas_deep=positional / OutputScale;\n    // Give more value",
)
replace(
    "main.cpp",
    "using namespace Stockfish;",
    '#include "atlas.inc"\n#include "reply.inc"\nusing namespace Stockfish;',
)
replace(
    "main.cpp",
    "  UCI::loop(argc, argv);",
    "  if (argc==6) atlas_main(argv); else if(argc==7) atlas_replies(argv); else UCI::loop(argc, argv);",
)
for name in ("atlas.inc", "reply.inc", "rules.def"):
    shutil.copy(Path(__file__).with_name(name), s)
subprocess.run(["make", "-C", s, "-j10", "build", "ARCH=x86-64-avx2"], check=True)
