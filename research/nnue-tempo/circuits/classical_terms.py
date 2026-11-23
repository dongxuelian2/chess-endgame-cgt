"""Control for output granularity and test how much a refit of the handwritten terms recovers."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from spectra import load
from threat_reversal import interval, statistics

TERMS = (
    "pawns",
    "knights",
    "bishops",
    "rooks",
    "queens",
    "king",
    "material",
    "imbalance",
    "mobility",
    "threat",
    "passed",
    "space",
    "winnable",
)


def features(root, rows, full, apply_phase=True):
    blocks = []
    scores = []
    for start in range(0, len(rows), 128):
        positions = []
        expected = []
        for r in rows[start : start + 128]:
            for prefix in ("", "edited_"):
                positions += r[prefix + "children"]
                expected += r[prefix + "sf141"]
        commands = (
            "".join("position fen " + fen + "\neval\n" for fen in positions) + "quit\n"
        )
        run = subprocess.run(
            [root / "circuits/probe/src/stockfish"],
            input=commands,
            text=True,
            check=True,
            capture_output=True,
            env={
                **os.environ,
                "NNUE_CIRCUIT_CLASSICAL": "1",
                **({"NNUE_CIRCUIT_FULL_HCE": "1"} if full else {}),
            },
        )
        terms = [
            list(map(int, line.split()[1:]))
            for line in run.stdout.splitlines()
            if line.startswith("terms ")
        ]
        pre = iter(
            int(line.split()[1])
            for line in run.stdout.splitlines()
            if line.startswith("pregrain ")
        )
        assert len(terms) == len(expected)
        features = []
        raw = []
        for (phase, special, *pairs), (hce, nnue), fen in zip(
            terms, expected, positions, strict=True
        ):
            h = hce if special else next(pre)
            if not special:
                if full:
                    hce = int(h / 16) * 16
                else:
                    assert int(h / 16) * 16 == hce
            raw.append((h, hce, nnue))
            arr = np.array(pairs).reshape(14, 2)[:-1].astype(float)
            if apply_phase:
                arr *= np.array([phase, 128 - phase]) / 128
            arr *= 1 if fen.split()[1] == "w" else -1
            if special:
                arr[:] = 0
            features.append([*arr.ravel(), h if special else 0])
        assert next(pre, None) is None
        blocks.append(np.array(features).reshape(-1, 2, 2, len(features[0])))
        scores.append(np.array(raw).reshape(-1, 2, 2, 3))
    return np.concatenate(blocks), np.concatenate(scores)


def fit(d):
    # Each row is ordered so a positive margin favors the safe choice.
    x = d.copy()
    x[:, 1] *= -1
    x = x.reshape(-1, x.shape[-1])
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[scale == 0] = 1
    x = x / scale
    w = np.zeros(x.shape[1])
    ridge = 0.05
    for _ in range(40):
        p = 1 / (1 + np.exp(np.clip(x @ w, -40, 40)))
        gradient = -x.T @ p / len(p) + ridge * w
        hess = (x.T * (p * (1 - p))) @ x / len(p) + ridge * np.eye(len(w))
        step = np.linalg.solve(hess, gradient)
        w -= step
        if max(abs(step)) < 1e-9:
            break
    assert max(abs(gradient)) < 1e-7
    return w / scale


def analyze(root, months, full):
    directory = root / "circuits"
    out = {}
    for month in months:
        rows = load(directory / f"reversals-{month}.gz")
        x, y = features(root, rows, full)
        delta = x[:, :, 1] - x[:, :, 0]
        if month == "2013-01":
            weights = fit(delta)
            out["term_weights"] = weights.tolist()
        dense = np.array([[r["units"], r["edited_units"]] for r in rows])
        neural = np.concatenate([dense[:, :, :, :1], dense[:, :, :, 10:]], axis=3)
        stages = np.array([r["stack"] for r in rows])
        if month == "2013-01":
            stage_h = {}
            stage_n = {}
            for stage in range(8):
                subset = stages == stage
                stage_h[stage] = fit(delta[subset])
                difference = neural[:, :, 1] - neural[:, :, 0]
                stage_n[stage] = fit(difference[subset].astype(float))
            out["stage_weights"] = {
                str(s): {"hce": stage_h[s].tolist(), "nnue": stage_n[s].tolist()}
                for s in range(8)
            }
        adapted = {}
        for name, feature, parameters in [
            ("refit_hce_by_stage", x, stage_h),
            ("refit_nnue_by_stage", neural, stage_n),
        ]:
            v = np.empty(feature.shape[:3])
            for stage in set(stages):
                v[stages == stage] = feature[stages == stage] @ parameters[stage]
            adapted[name] = v
        strategies = {}
        raw = y[:, :, :, 0]
        nnue = y[:, :, :, 2]
        scores = {
            "pregrain_classical": raw,
            "native_classical": y[:, :, :, 1],
            "native_nnue": nnue,
            "coarse_nnue": np.trunc(nnue / 16) * 16,
            "coarse_classical": np.trunc(raw / 16) * 16,
            "refit_classical": x @ weights,
            **adapted,
        }
        groups = [str(r.get("game", r.get("group"))) for r in rows]
        both = {}
        margins = {}
        for k, v in scores.items():
            p = (v[:, :, 1] - v[:, :, 0]).T
            if k == "refit_classical":
                p = (delta @ weights).T
            if k in ("refit_hce_by_stage", "refit_nnue_by_stage"):
                features_for_margin, parameters = (
                    (delta, stage_h)
                    if k == "refit_hce_by_stage"
                    else (neural[:, :, 1] - neural[:, :, 0], stage_n)
                )
                for stage in set(stages):
                    p[:, stages == stage] = (
                        features_for_margin[stages == stage] @ parameters[stage]
                    ).T
            margins[k] = p
            strategies[k] = statistics(p)
            tie0 = (p[0] > 0) + 0.5 * (p[0] == 0)
            tie1 = (p[1] < 0) + 0.5 * (p[1] == 0)
            strategies[k]["both_random_ties"] = float((tie0 * tie1).mean())
            both[k] = ((p[0] > 0) & (p[1] < 0)).astype(float)
        hand = delta[:, :, :-1].reshape(-1, 2, len(TERMS), 2).sum(3)
        contrasts = hand[:, 0] - hand[:, 1]
        out[month] = {
            "n": len(rows),
            "metrics": strategies,
            "matched_refit_gap": interval(
                both["refit_nnue_by_stage"] - both["refit_hce_by_stage"], groups
            ),
            "fine_gap": interval(
                both["native_nnue"] - both["pregrain_classical"], groups
            ),
            "coarse_gap": interval(
                both["coarse_nnue"] - both["coarse_classical"], groups
            ),
            "refit_gap": interval(
                both["native_nnue"] - both["refit_classical"], groups
            ),
            "exact_zero_term_interaction": dict(
                zip(TERMS, (contrasts == 0).mean(0).tolist())
            ),
            "positive_term_interaction": dict(
                zip(TERMS, (contrasts > 0).mean(0).tolist())
            ),
        }
        collision = np.all(delta[:, 0] == delta[:, 1], axis=1)
        exact_boards = np.any(np.all(x[:, :, 0] == x[:, :, 1], axis=2), axis=1)
        for label, sel in [
            ("identical_handwritten_move_increments", collision),
            ("identical_handwritten_candidates_in_at_least_one_context", exact_boards),
        ]:
            entry = {"n": int(sel.sum())}
            if sel.any():
                entry["metrics"] = {
                    k: statistics(p[:, sel]) for k, p in margins.items()
                }
            out[month][label] = entry
        if collision.any():
            assert not both["refit_classical"][collision].any()
        print(
            month, [(k, round(v["both"], 4)) for k, v in strategies.items()], flush=True
        )
    (
        directory / ("full-classical-terms.json" if full else "classical-terms.json")
    ).write_text(json.dumps(out, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument(
        "--months", nargs="+", default=["2013-01", "2013-02", "2013-03", "wdl"]
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="evaluate all classical terms, bypassing the two lazy cutoffs",
    )
    args = p.parse_args()
    assert args.months[0] == "2013-01"
    analyze(args.root, args.months, args.full)
