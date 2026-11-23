"""Pair stock and forced-full handwritten traces on the same four positions."""

import argparse
import json
from pathlib import Path

import numpy as np
from classical_terms import features
from spectra import load
from threat_reversal import interval, statistics


def main(root):
    directory = root / "circuits"
    out = {}
    for cohort in ("2013-02", "2013-03", "wdl"):
        rows = load(directory / f"reversals-{cohort}.gz")
        x, y = features(root, rows, False)
        full, fy = features(root, rows, True)
        differs = np.any(full != x, axis=(1, 2, 3)) | np.any(fy != y, axis=(1, 2, 3))
        same = lambda v: np.all(
            v[:, 0, 1] - v[:, 0, 0] == v[:, 1, 1] - v[:, 1, 0], axis=1
        )
        collision = same(x)
        fc = same(full)
        groups = np.array([str(r.get("game", r.get("group"))) for r in rows])
        series = {
            "native_classical": y[:, :, :, 1],
            "native_nnue": y[:, :, :, 2],
            "full_classical": fy[:, :, :, 1],
            "pregrain_classical": y[:, :, :, 0],
            "full_pregrain_classical": fy[:, :, :, 0],
        }
        out[cohort] = {
            "n": len(rows),
            "changed_by_full": int(differs.sum()),
            "stock_collisions": int(collision.sum()),
            "full_collisions": int(fc.sum()),
            "collisions_remaining": int(np.sum(fc & collision)),
            "subsets": {},
        }
        ids = np.flatnonzero(fc)
        if len(ids):
            selected = [rows[i] for i in ids]
            unfaded, _ = features(root, selected, True, apply_phase=False)
            invariant = same(unfaded)
            out[cohort]["raw_collisions"] = int(invariant.sum())
            if cohort == "wdl":
                witnesses = []
                for j, idx in enumerate(ids):
                    r = rows[idx]
                    nmargin = y[idx, :, 1, 2] - y[idx, :, 0, 2]
                    hmargin = fy[idx, :, 1, 0] - fy[idx, :, 0, 0]
                    witnesses.append(
                        {
                            "material": r["material"],
                            "parent": r["parent"],
                            "relocation": r["edit"],
                            "move_a": r["moves"][0],
                            "move_b": r["moves"][1],
                            "original_wdl": " ".join(map(str, r["outcomes"])),
                            "edited_wdl": " ".join(map(str, r["edited_outcomes"])),
                            "identical_raw_mg_eg": bool(invariant[j]),
                            "nnue_original": int(nmargin[0]),
                            "nnue_edited": int(nmargin[1]),
                            "classical_pregrain_original": int(hmargin[0]),
                            "classical_pregrain_edited": int(hmargin[1]),
                        }
                    )
                out[cohort]["witnesses"] = witnesses
        for key, mask in [
            ("all", np.ones(len(rows), dtype=bool)),
            ("affected", differs),
            ("unaffected", ~differs),
            ("full_feature_collision", fc),
        ]:
            if not mask.any():
                continue
            m = {k: (v[mask, :, 1] - v[mask, :, 0]).T for k, v in series.items()}
            both = {k: ((p[0] > 0) & (p[1] < 0)).astype(float) for k, p in m.items()}
            out[cohort]["subsets"][key] = {
                "n": int(mask.sum()),
                "metrics": {k: statistics(p) for k, p in m.items()},
                "full_minus_nnue": interval(
                    both["full_classical"] - both["native_nnue"], groups[mask]
                ),
                "full_minus_stock": interval(
                    both["full_classical"] - both["native_classical"], groups[mask]
                ),
            }
        print(
            cohort,
            out[cohort]["changed_by_full"],
            out[cohort]["stock_collisions"],
            out[cohort]["full_collisions"],
        )
    (directory / "cutoffs.json").write_text(json.dumps(out, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
