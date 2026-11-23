"""Build an isolated driver against the prepared, pinned atlas source."""

import shutil
import subprocess
import sys
from pathlib import Path


def main(root):
    src = root / "atlas/engine/src"
    dst = root / "atlas/horizon/engine/src"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("stockfish", ".depend", "*.o"),
    )
    p = dst / "main.cpp"
    text = p.read_text()
    assert '#include "reply.inc"' in text and "if (argc==6)" in text
    text = text.replace(
        '#include "reply.inc"',
        '#include "reply.inc"\n#include "frontier.inc"\n#include "proof.inc"\n#include "trace.inc"',
    ).replace(
        "if (argc==6)",
        'if (argc==8 && std::string(argv[1])=="frontier") Frontier::batch(argv); else if (argc==8 && std::string(argv[1])=="proof") Frontier::proofs(argv); else if (argc==8 && std::string(argv[1])=="traces") Frontier::traces(argv); else if (argc==6)',
    )
    p.write_text(text)
    shutil.copy(Path(__file__).with_name("frontier.inc"), dst)
    shutil.copy(Path(__file__).with_name("proof.inc"), dst)
    shutil.copy(Path(__file__).with_name("trace.inc"), dst)
    nn = dst / "nnue/evaluate_nnue.cpp"
    code = nn.read_text()
    anchor = "    atlas_ps=psqt / OutputScale;"
    assert anchor in code and "  int atlas_ps, atlas_deep;" in code
    code = code.replace(
        "  int atlas_ps, atlas_deep;",
        "  int atlas_ps, atlas_deep, horizon_ps, horizon_pos; uint8_t horizon_h1[8];",
    )
    code = code.replace(
        anchor,
        """    using H2 = Layers::HiddenLayer2;
    using A2 = Layers::AffineTransform<Layers::HiddenLayer1, 32>;
    const auto h2 = reinterpret_cast<const uint8_t*>(buffer + Network::SelfBufferSize);
    const auto h1 = h2 + H2::SelfBufferSize + A2::SelfBufferSize;
    horizon_ps=psqt; horizon_pos=positional; std::copy_n(h1,8,horizon_h1);
"""
        + anchor,
    )
    nn.write_text(code)
    subprocess.run(["make", "-C", dst, "-j10", "build", "ARCH=x86-64-avx2"], check=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
