"""Extend the pinned horizon driver with the two-ply restriction audit."""

import shutil
import subprocess
import sys
from pathlib import Path


def main(root):
    src = root / "atlas/horizon/engine/src"
    dst = root / "atlas/horizon/kings/engine/src"
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("stockfish", "*.o", ".depend"),
    )
    p = dst / "main.cpp"
    s = p.read_text()
    assert '#include "trace.inc"' in s
    s = s.replace(
        '#include "trace.inc"',
        '#include "trace.inc"\n#include "replies.inc"\n#include "rank3.inc"\n#include "continuation.inc"',
    )
    s = s.replace(
        'if (argc==8 && std::string(argv[1])=="frontier")',
        'if(argc==8 && std::string(argv[1])=="third") KingStudy::third_batch(argv); else if(argc==8 && std::string(argv[1])=="restrictions") KingStudy::batch(argv); else if(argc==8 && std::string(argv[1])=="rank3") KingStudy::rank_batch(argv); else if (argc==8 && std::string(argv[1])=="frontier")',
    )
    p.write_text(s)
    for name in ("replies.inc", "moves.def", "rank3.inc", "continuation.inc"):
        shutil.copy(Path(__file__).with_name(name), dst)
    subprocess.run(["make", "-C", dst, "-j10", "build", "ARCH=x86-64-avx2"], check=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
