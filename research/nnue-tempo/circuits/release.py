"""Save reproducible aggregate measurements and compact, inspectable result tables."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from collect import EXTENDED, MATERIAL
from routing import MASKS, ORDER


def table(path, records):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main(root, dest):
    dest.mkdir(exist_ok=True, parents=True)
    d = root / "circuits"

    def read(key):
        return json.loads((d / (key + ".json")).read_text())

    sources = [
        "edge-transfer",
        "feature-routes",
        "certificates",
        "classical-terms",
        "full-classical-terms",
        "cutoffs",
        "wdl-reversal",
        "thresholds-wdl",
        "interaction-layers-wdl",
        "audit-wdl",
    ]
    months = ["2013-02", "2013-03"]
    for date in months:
        sources.extend(
            f"{name}-{date}"
            for name in ("thresholds", "interaction-layers", "threat-reversal", "audit")
        )
    sources.extend(p.stem for p in sorted(d.glob("spectrum-*.json")))
    summary = {k: read(k) for k in sources}
    contents = json.dumps(
        summary, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    (dest / "measurements.json.gz").write_bytes(
        gzip.compress(contents, compresslevel=9, mtime=0)
    )

    def units(mask):
        return " ".join(str(i) for i in range(8) if mask & 1 << i)

    ends = []
    transfer = []
    sizes = MASKS[ORDER == 4]
    for name in MATERIAL + EXTENDED:
        cohorts = summary["spectrum-" + name]
        pool = EXTENDED if name in EXTENDED else MATERIAL
        errors = np.array(
            [
                summary["spectrum-" + peer]["fit"]["mask_nmse"]
                for peer in pool
                if peer != name
            ]
        )
        independent = sizes[errors.mean(0)[sizes].argmin()]
        local = sizes[np.argmin(np.array(cohorts["fit"]["mask_nmse"])[sizes])]
        for key, data in cohorts.items():
            if "metric" not in data:
                continue
            m = data["metric"]
            tempo = key in ("fit", "test", "jan", "feb")
            ends.append(
                {
                    "material": name,
                    "cohort": key,
                    "n": data["n"],
                    "metric": "tempo_auc" if tempo else "better_move_rate",
                    "classical": m["classical"],
                    "native": m["native"],
                    "unpatched_base": m["base"],
                    "leading_order": m["leading"],
                    "full_unrounded": m["full"],
                    "leading_order_energy": data["order_energy"][2 if tempo else 1],
                }
            )
            if key in ("test", "feb"):
                err = np.array(data["mask_nmse"])
                transfer.append(
                    {
                        "material": name,
                        "cohort": key,
                        "n": data["n"],
                        "same_family_mask": units(local),
                        "other_families_mask": units(independent),
                        "same_family_nmse": err[local],
                        "other_families_nmse": err[independent],
                        "all_four_unit_masks_median_nmse": np.median(err[sizes]),
                    }
                )
    table(dest / "endgames.csv", ends)
    table(dest / "tempo-transfer.csv", transfer)
    rows = []
    for s, stage in summary["edge-transfer"].items():
        for k, dataset in stage.items():
            if k == "orders":
                continue
            for width, v in dataset["width"].items():
                for rule, (error, rate) in v.items():
                    rows.append(
                        {
                            "stage": s,
                            "task": k,
                            "n": dataset["n"],
                            "edges": width,
                            "selection": rule,
                            "positional_nmse": error,
                            "unrounded_move_rate": rate,
                            "native_rate": dataset["native"],
                            "classical_rate": dataset["classical"],
                            "native_minus_classical_ci_low": dataset[
                                "net_minus_classical"
                            ][1],
                            "native_minus_classical_ci_high": dataset[
                                "net_minus_classical"
                            ][2],
                        }
                    )
    table(dest / "edge-transfer.csv", rows)
    cert = summary["certificates"]
    rows = []
    for name, dataset in cert["motifs"].items():
        if not name.endswith("02"):
            continue
        for stage, r in dataset.items():
            if not r["correct"]:
                continue
            train = cert["motifs"]["contacts-2013-01"][stage]
            best = sizes[np.argmax(np.array(train["mask_count"])[sizes])]
            covered = r["mask_count"][best]
            rows.append(
                {
                    "task": name[:-8],
                    "stage": stage,
                    "n": r["n"],
                    "correct_unrounded": r["correct"],
                    "individually_sufficient_with_two": sum(r["min_size"][:3]),
                    "jan_fixed_four": units(best),
                    "certified_fixed_four": covered,
                }
            )
    table(dest / "certificates.csv", rows)
    reversals = []
    for date in [*months, "wdl"]:
        evaluation = (
            summary["wdl-reversal"]["all"]
            if date == "wdl"
            else summary["threat-reversal-" + date]
        )
        layers = summary["interaction-layers-" + date]
        thresholds = summary["thresholds-" + date]
        for s, v in [("all", evaluation), *evaluation.get("by_stage", {}).items()]:
            m = v["metrics"]
            patch = layers["patch_" + s]
            g = layers[s][0]["share_of_interaction_projection"]
            lo, hi = thresholds[s]["threshold_projection"]
            reversals.append(
                {
                    "cohort": date,
                    "stage": s,
                    "quadruples": v["n"],
                    "classical_both": m["classical"]["both"],
                    "native_both": m["native"]["both"],
                    "native_interaction_sign": m["native"]["positive_interaction"],
                    "intact_unrounded_both": patch["intact"]["both_rate"],
                    "equalized_both": patch["equalized"]["both_rate"],
                    "equalized_projection": patch["equalized"][
                        "interaction_projection"
                    ],
                    "restore_256_both": patch["jan256"]["both_rate"],
                    "restore_random_256_both": patch["random256"]["both_rate"],
                    "feature_projection": g[0],
                    "h1_projection": g[1],
                    "h2_projection": g[2],
                    "lower_threshold_projection": lo,
                    "upper_threshold_projection": hi,
                }
            )
    controls = []
    for setting in ["classical-terms", "full-classical-terms"]:
        for date in ["2013-01", *months, "wdl"]:
            ds = summary[setting][date]
            for name, v in ds["metrics"].items():
                controls.append(
                    {
                        "cohort": date,
                        "cutoffs": "stock"
                        if setting == "classical-terms"
                        else "disabled",
                        "scoring": name,
                        "n": ds["n"],
                        "both_strict": v["both"],
                        "both_random_ties": v["both_random_ties"],
                        "correct_interaction": v["positive_interaction"],
                    }
                )
    table(dest / "handwritten-controls.csv", controls)
    table(dest / "reversals.csv", reversals)
    table(dest / "handwritten-collisions.csv", summary["cutoffs"]["wdl"]["witnesses"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    main(args.root, args.output)
