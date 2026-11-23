"""Count unchanged verdicts separately from missing signals and offsetting verdict shifts."""

import sys
from pathlib import Path

import numpy as np
from network import Network, credit, trunc
from summarize import write

from effects import batches, delta
from tactical import load


def main(root, out):
    net = Network(root)
    rows = []
    for a in [3, 5]:
        for sample in ["validation", "transfer"]:
            r = load(root, sample, 3 if sample == "validation" else a, 4)
            chunks = []
            for batch in batches(r):
                raw, p, _ = net.encode(batch, a, 4)
                t = net.forward(raw, p)
                c = delta(t["h2"])[:, 2] * net.out[2]
                native_full = t["ps"] + t["pos"]
                native = delta(t["score"])
                adjusted = delta(native_full) - c
                left = trunc(native_full[:, 0] - c, 16) - t["score"][:, 1]
                right = t["score"][:, 0] - trunc(native_full[:, 1] + c, 16)
                # Integer quantization can move the difference by strictly less than two score units.
                assert np.all(abs(left - adjusted / 16) < 2) and np.all(
                    abs(right - adjusted / 16) < 2
                )
                payoff = (credit(left) + credit(right)) / 2 - credit(native)
                chunks.append(
                    np.c_[
                        c,
                        payoff,
                        (np.sign(left) == np.sign(native))
                        & (np.sign(right) == np.sign(native)),
                        abs(adjusted) >= 32,
                    ]
                )
            z = np.concatenate(chunks)
            rows.append(
                {
                    "sample": sample,
                    "white": a,
                    "n": len(r),
                    "zero": int((z[:, 0] == 0).sum()),
                    "nonzero_both_orders_preserved": int(
                        ((z[:, 0] != 0) & (z[:, 2] == 1)).sum()
                    ),
                    "some_verdict_changes": int((z[:, 2] == 0).sum()),
                    "mean_credit_gains": int((z[:, 1] > 0).sum()),
                    "mean_credit_losses": int((z[:, 1] < 0).sum()),
                    "certified_adjusted_sign": int(z[:, 3].sum()),
                }
            )
    write(out / "verdicts.csv", rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
