"""Render the rule-by-rule chess ledger from the committed measurements."""

import csv
import json
import re

from validate import HERE, RULES


def read(filename):
    return list(csv.DictReader((HERE / "results" / filename).open()))


def pct(s):
    return f"{100 * float(s):.1f}" if s != "" else "—"


def count(s):
    return f"{int(s):,}"


def ratio(s):
    return f"{float(s):.2f}" if s and float(s) < 10000 else "—"


def main():
    desc = dict(
        re.findall(r'RULE\((\w+), "([^"]+)"\)', (HERE / "rules.def").read_text())
    )
    tests = {r["rule"]: r for r in read("findings.csv")}
    sample = {
        r["rule"]: r
        for r in read("channel-interventions.csv")
        if r["panel"] == "replication"
    }
    null = {
        r["rule"]: r
        for r in read("size-matched-deletions.csv")
        if r["panel"] == "generated" and r["scope"] == "all"
    }
    witnesses = {
        r["rule"]: r
        for r in json.loads((HERE / "results/exclusive-certificates.json").read_text())
    }
    blocks = {
        "Captures and exchanges": RULES[:4],
        "Promotion and the pawn race": RULES[4:15],
        "Checks, double attacks and pins": RULES[15:25],
        "Attack, support and counterattack": RULES[25:34],
        "King geometry and pawn control": RULES[34:46],
        "Terminal replies": RULES[46:],
    }
    lines = [
        "# Chess findings by reply class",
        "",
        "Each row names a concrete class of legal Black replies. Required counts are distinct canonical boards on which forbidding that class for one move worsens Black’s game-theoretic result while another legal reply remains. Erased counts are common-source White move pairs with equal full classical score and different WDL whose ordering disappears or reverses under the restriction in both children. Counts overlap across rows.",
        "",
        "SF14.1 is native pair accuracy in percent on the complete erased-pair census. SF18 intact and off are raw large-network accuracies on an independent, one-pair-per-context replication subsample. The final column divides observed WDL-order erasures by the exact expectation from deleting equally many random legal replies from each child on that subsample. Values below one indicate weaker selectivity than the size-matched deletion. A dash marks an empty or undefined estimate.",
        "",
        "The [per-rule decision gains](results/findings.csv), [simultaneous channel intervals](results/channel-interventions.csv), [material breakdown](results/complete-rules.csv), [44 move certificates](results/move-certificates.json) and [27 exclusive certificates](results/exclusive-certificates.json) preserve the distinct levels of evidence.",
        "",
    ]
    for heading, items in blocks.items():
        lines.extend(
            [
                "## " + heading,
                "",
                "| Chess test | Required boards | Erased pairs | SF14.1 | Independent pairs | SF18 intact | SF18 off | Erasures / random |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for t in items:
            r = tests[t]
            s = sample.get(t, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        desc[t],
                        count(r["native_required"]),
                        count(r["erased"]),
                        pct(r["accuracy141"]),
                        count(s.get("n", 0)),
                        pct(s.get("sf18", "")),
                        pct(s.get("sf18_ablated", "")),
                        ratio(null[t]["ratio"]),
                    ]
                )
                + " |"
            )
        lines.append("")
        for t in items:
            if t not in witnesses:
                continue
            w = witnesses[t]
            a, b = w["moves"]
            types = {-1: "loss", 0: "draw", 1: "win"}
            lines.append(
                f"- **{t.replace('_', ' ').capitalize()}** in {w['material']}. White’s {a['uci']} and {b['uci']} receive the same classical score ({a['classical']}, Black perspective). Black obtains a {types[a['wdl_black']]} and a {types[b['wdl_black']]}, respectively. Suppressing this reply class changes the two results to {types[a['restricted_black']]} and {types[b['restricted_black']]}. The affected optimal replies after {a['uci']} are {', '.join(a['deleted_best'])}."
            )
        lines.append("")
    lines.extend(
        [
            "## Additional distinctions with nine immediate tactical signals matched",
            "",
            "Both children have the same nine availability flags for captures, exchanges and promotions. A direction was frozen on discovery pairs under that same constraint. Each row below passed a simultaneous 95% lower bound above 50% on the replication panel. Columns count comparisons where this additional flag differs and give the accuracy of its frozen direction. A minus direction means the presence of a legal reply predicts a worse result for Black. This can reflect the situation that necessitates the reply.",
            "",
            "| Available reply | Direction for Black | Replication pairs | Correct % | Simultaneous 95% interval | Played pairs | Played correct % |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    incremental = read("incremental-rules.csv")
    played = {r["rule"]: r for r in incremental if r["panel"] == "played"}
    for r in incremental:
        if r["panel"] != "replication" or not r["lo"] or float(r["lo"]) <= 0.5:
            continue
        g = played[r["rule"]]
        direction = "better" if int(r["direction"]) == 1 else "worse"
        lines.append(
            "| "
            + " | ".join(
                [
                    desc[r["rule"]],
                    direction,
                    count(r["strict_secondary"]),
                    pct(r["rule_accuracy"]),
                    pct(r["lo"]) + "–" + pct(r["hi"]),
                    count(g["strict_secondary"]),
                    pct(g["rule_accuracy"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Only the single pawn push and unpin pass the same interval criterion on played positions. In all 1,293 replication unpin comparisons, the black man is already pinned in precisely one child. Among 634 escape comparisons where the black man is attacked in both children, a safe quiet escape predicts the true ordering in 99.2%. Among 3,972 where the initial attack status differs, accuracy is 49.6%. [All 48 matched tests](results/incremental-rules.csv) · [Prerequisite audit](results/prerequisites.csv)",
            "",
        ]
    )
    (HERE / "FINDINGS.md").write_text("\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    main()
