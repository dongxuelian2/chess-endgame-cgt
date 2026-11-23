"""Disjoint-root replication with full, pre-grain classical evaluation and conditional shuffling."""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from measure import load, table
from transfer import MATERIALS

from search_games import game, init, leaf_score


def confirm(job, decompose=False):
    root, cohort, mat, fen = job
    tree, boards, truth, _ = game((root, cohort, fen), audit="classical")
    nn, psqt, positional = leaf_score(boards, components=True)
    hce = nn.copy()
    ix = [
        i
        for i, b in enumerate(boards)
        if not b.is_check() and not b.is_insufficient_material() and any(b.legal_moves)
    ]
    for off in range(0, len(ix), 256):
        batch = ix[off : off + 256]
        commands = "".join(
            "position fen " + boards[i].fen() + "\neval\n" for i in batch
        )
        text = subprocess.run(
            [root / "circuits/probe/src/stockfish"],
            input=commands + "quit\n",
            text=True,
            capture_output=True,
            check=True,
            env={
                **os.environ,
                "NNUE_CIRCUIT_CLASSICAL": "1",
                "NNUE_CIRCUIT_FULL_HCE": "1",
            },
        ).stdout.splitlines()
        pre = iter(int(s.split()[1]) for s in text if s.startswith("pregrain "))
        special = [int(s.split()[2]) for s in text if s.startswith("terms ")]
        scores = [
            list(map(int, s.split()[1:])) for s in text if s.startswith("scores ")
        ]
        for i, sp, (h, n) in zip(batch, special, scores, strict=True):
            sign = 1 if boards[i].turn else -1
            assert nn[i] == sign * n
            raw = h if sp else next(pre)
            if not sp:
                assert int(raw / 16) * 16 == h
            hce[i] = sign * raw
        assert next(pre, None) is None
    allrows = [r for a in tree for r in a]
    ids = np.concatenate(allrows)
    rowstarts = np.cumsum([0] + [len(r) for r in allrows[:-1]])
    actstarts = np.cumsum([0] + [len(a) for a in tree[:-1]])

    def backup(values):
        return np.minimum.reduceat(
            np.maximum.reduceat(values[ids], rowstarts), actstarts
        )

    h = backup(hce)
    leaders = np.flatnonzero(h == max(h))
    n = backup(nn)
    neural = np.flatnonzero(n == max(n))

    def avg(acts):
        return (
            None
            if any(truth[i] is None for i in acts)
            else float(np.mean([truth[i] for i in acts]))
        )

    # Encode the primary and secondary orders exactly with a gap larger than the score range.
    width = int(np.ptp(nn)) + 1

    def refined(secondary):
        score = backup(hce * width + secondary)
        chosen = np.flatnonzero(score == max(score))
        assert set(chosen) <= set(leaders)
        return chosen

    chosen = refined(nn)
    base = avg(leaders)
    delta = lambda u: None if base is None or u is None else u - base
    mix = base is not None and len({truth[i] for i in leaders}) > 1
    if decompose == "race":
        from race import opposed, scores

        record = {
            "fen": fen,
            "material": mat,
            "mixed": mix,
            "states": len(boards),
            "known": base is not None,
        }
        geometry = scores(boards, ix)
        record["pairs"], record["credits"] = opposed(
            boards, ix, hce, nn, positional, psqt, geometry
        )
        if mix:
            for key, secondary in {"full": nn, **geometry}.items():
                span = int(np.ptp(secondary)) + 1
                score = backup(hce * span + secondary)
                acts = np.flatnonzero(score == max(score))
                assert set(acts) <= set(leaders)
                record[key] = delta(avg(acts))
                if key == "full":
                    native_best = set(acts)
                else:
                    record[key + "_overlap"] = len(
                        native_best.intersection(acts)
                    ) / len(native_best)
                    record[key + "_chance"] = len(acts) / len(leaders)
        return record
    if decompose:
        ans = {}
        eligible = np.zeros(len(nn), bool)
        eligible[ix] = True
        # Neutral secondary score at every checked or terminal leaf in the mechanism control.
        from factor import factor

        factors = factor(boards, eligible, np.where(eligible, positional, 0))
        for name, secondary in [
            ("full", nn),
            ("quiet_full", np.where(eligible, nn, 0)),
            ("psqt", np.where(eligible, psqt, 0)),
            ("positional", np.where(eligible, positional, 0)),
            *factors.items(),
        ]:
            span = int(np.ptp(secondary)) + 1
            scores = backup(hce * span + secondary)
            acts = np.flatnonzero(scores == max(scores))
            assert set(acts) <= set(leaders)
            ans[name] = delta(avg(acts))
        return ans
    null = None
    if mix:
        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(("tie-null" + fen).encode()).digest()[:8], "big"
            )
        )
        # Fixed checked/terminal leaves use the same score in both evaluators.
        bins = [np.array(ix)[hce[ix] == v] for v in np.unique(hce[ix])]
        bins = [g for g in bins if len(np.unique(nn[g])) > 1]
        null = []
        for _ in range(255):
            s = nn.copy()
            for g in bins:
                s[g] = rng.permutation(s[g])
            null.append(delta(avg(refined(s))))
    return {
        "cohort": cohort,
        "material": mat,
        "fen": fen,
        "states": len(boards),
        "verified": len(ix),
        "tie": len(leaders) > 1,
        "narrows": len(chosen) < len(leaders),
        "mixed": mix,
        "refined": delta(avg(chosen)),
        "nnue": delta(avg(neural)),
        "null": null,
    }


def main(root, dest):
    with gzip.open(dest / "root-costs.json.gz") as f:
        old = {r["fen"].split()[0] for r in json.load(f)}
    with gzip.open(root / "pawn-geometry/played.gz") as f:
        played = list(map(json.loads, f))
    jobs = []
    for cohort in ("played", "random"):
        for mat in MATERIALS:
            rows = (
                [r for r in played if r["material"] == mat]
                if cohort == "played"
                else load(root, "random-" + mat)
            )
            keys = {r["parent"].split()[0] for r in rows} - old
            ordered = sorted(
                keys,
                key=lambda s: hashlib.sha256(
                    ("confirm-full-100-v1" + s).encode()
                ).digest(),
            )
            assert len(ordered) >= 100
            jobs.extend((root, cohort, mat, k + " w - - 0 1") for k in ordered[:100])
            old.update(ordered[:100])
    assert len({r[3] for r in jobs}) == len(jobs)
    with ProcessPoolExecutor(10, initializer=init, initargs=(root,)) as pool:
        rows = list(pool.map(confirm, jobs, chunksize=1))
    with gzip.open(root / "relational-tests/confirmation.gz", "wt") as f:
        for r in rows:
            print(json.dumps(r), file=f)
    out = []
    nulls = []
    for tag in ["all", "played", "random", *MATERIALS]:
        rs = [r for r in rows if tag in ("all", r["material"], r["cohort"])]
        mixed = [r for r in rs if r["mixed"]]
        for mode in ("refined", "nnue"):
            ds = np.array([r[mode] for r in rs if r[mode] is not None])
            out.append(
                {
                    "panel": tag,
                    "endpoint": mode,
                    "n": len(rs),
                    "leaf_states": sum(r["states"] for r in rs),
                    "native_checks": sum(r["verified"] for r in rs),
                    "tie": sum(r["tie"] for r in rs),
                    "narrows": sum(r["narrows"] for r in rs),
                    "mixed": len(mixed),
                    "wdl_known": len(ds),
                    "better": int(sum(ds > 0)),
                    "equal": int(sum(ds == 0)),
                    "worse": int(sum(ds < 0)),
                    "mean_delta": float(ds.mean()),
                }
            )
        if mixed:
            observed = np.mean([r["refined"] for r in mixed])
            random = np.mean([r["null"] for r in mixed], axis=0)
            nulls.append(
                {
                    "panel": tag,
                    "n": len(mixed),
                    "observed": float(observed),
                    "null_mean": float(random.mean()),
                    "null_q025": float(np.quantile(random, 0.025)),
                    "null_q975": float(np.quantile(random, 0.975)),
                    "p_greater": (1 + int(sum(random >= observed))) / 256,
                }
            )
    table(dest / "confirmation.csv", out)
    table(dest / "conditional-shuffle.csv", nulls)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    main(a.root, a.dest)
