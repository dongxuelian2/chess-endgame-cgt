"""Expose the full classical score and native integer NNUE on arbitrary FENs."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def replace(path, anchor, replacement):
    source = path.read_text()
    assert source.count(anchor) == 1, (path, anchor)
    path.write_text(source.replace(anchor, replacement))


def main(root):
    clean = root / "clean-sf141"
    revision = subprocess.check_output(
        ["git", "-C", clean, "rev-parse", "HEAD"], text=True
    )
    assert revision.strip() == "7262fd5d14810b7b495b5038e348a448fda1bcc3"
    target = root / "manuscript/expansion/classical"
    target.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "-C", clean, "archive", "HEAD", "src"])
    subprocess.run(["tar", "xf", "-", "-C", target], input=archive, check=True)
    src = target / "src"
    shutil.copyfile(clean / "src/nn-13406b1dcbe0.nnue", src / "nn-13406b1dcbe0.nnue")
    path = src / "evaluate.cpp"
    for threshold in ("LazyThreshold1", "LazyThreshold2"):
        anchor = f"    if (lazy_skip({threshold}))"
        replace(path, anchor, f"    if (!expansionFull && lazy_skip({threshold}))")
    replace(
        path, "    v = (v / 16) * 16;", "    if (!expansionFull) v = (v / 16) * 16;"
    )
    replace(
        path,
        "namespace Stockfish {",
        "namespace Stockfish {\nstatic bool expansionFull = false;",
    )
    replace(
        path,
        "  std::memset(scores, 0, sizeof(scores));",
        """  expansionFull = true;
  const int classical = Evaluation<NO_TRACE>(pos).value();
  expansionFull = false;
  const int neural = NNUE::evaluate(pos, false);
  return "scores " + std::to_string(classical) + " " + std::to_string(neural);
  std::memset(scores, 0, sizeof(scores));""",
    )
    subprocess.run(
        [
            "make",
            "-C",
            src,
            "-j" + os.environ.get("JOBS", "10"),
            "build",
            "ARCH=" + os.environ.get("ARCH", "x86-64-avx2"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
