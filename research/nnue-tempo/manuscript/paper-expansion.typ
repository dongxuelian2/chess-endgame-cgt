#import "paper-matching.typ": matching-section

#let transfer-section(booktabs) = [
= Replication across releases, material and search

== Sampling and score definitions
We generated 140,000 pairs before consulting WDL and retained 139,960 comparison records. We exclude duplicate and mirror-equivalent pairs within each panel and eliminate exact matches to the original samples. The defender has a knight, bishop, rook or queen and zero, one or two extra pawns. The attacker has one pawn on the second rank. For each defender and piece count we place the pieces at random, reject illegal parents and sample two quiet moves by the same nonpawn piece. For five and six pieces we stratify by whether the defender has a pawn on its seventh rank. A separate enriched sample chooses one move that permits a surviving capture-promotion and one that precludes it. These screens use legal moves and board geometry. Syzygy labels enter after the pair is selected. There are 48,696 strict preferences across the synthetic panels, of which 18,587 are in the unenriched panels.

The game sample scans 4,861,993 rated Lichess game records from January to June 2014. For each piece count, we select one eligible parent per game by reservoir sampling and pair the observed move with a uniformly sampled quiet alternative by the same piece. We also sample one exposed pair per game when the observed move and at least one alternative differ in the surviving capture option. Each series is deduplicated under horizontal reflection and excludes exact pairs from discovery, original validation and the 2013 game panels. The ordinary series includes 39,736 eligible pairs and 8,501 strict outcomes. The exposure series includes 2,070 strict outcomes, with possible overlap with the ordinary series. The two denominators describe different sampling questions.

We compare the native integer NNUE readouts from Stockfish 12 and 14.1 and the complete Stockfish 18 static evaluator at zero optimism [4, 13]. The older NNUE scores are extracted before the engine adds its search-facing evaluation adjustments. We use the same original 18,427 discovery comparisons and the same integer grid for each rule and release. The fitted surviving-capture weights are 528, 2,800 and 2,560 in release order. For any legal capture the weights are 384, 2,800 and 2,560. The reply filter inherits the surviving-capture weight. We report the full strict cohorts and the subset tied by the full Stockfish 14.1 classical evaluator. The larger random cohorts have just 39 and 12 such ties for five and six pieces, respectively. The larger-material comparisons therefore chiefly measure static score performance within fixed material.

#figure(
  placement: auto,
  scope: "parent",
  caption: [*Frozen surviving-capture rule on the expanded samples.* Each entry gives native ordering credit and the change in percentage points (pp). The lower two rows deliberately enrich disagreements on the capture option. The game panels use alternatives to an observed move.],
  booktabs(
    (2fr, .65fr, 1fr, 1fr, 1fr),
    (left, right, right, right, right),
    ([Sample], [Pairs], [SF12], [SF14.1], [SF18]),
    (
      ([Four pieces, random], [3,832], [72.85% +2.51], [93.49% +4.29], [93.03% +3.71]),
      ([Five pieces, random], [7,595], [71.99% +3.28], [84.87% +4.51], [85.52% +4.85]),
      ([Six pieces, random], [7,160], [70.57% +3.96], [79.51% +5.50], [79.34% +5.21]),
      ([Four pieces, games], [4,525], [75.48% +0.62], [93.45% +0.67], [91.09% +0.45]),
      ([Five pieces, games], [3,976], [80.94% +0.35], [93.22% +0.30], [93.75% +0.23]),
      ([Four pieces, exposed games], [1,415], [73.50% +26.43], [73.78% +26.22], [83.64% +16.36]),
      ([Five pieces, exposed games], [655], [70.38% +26.56], [80.61% +19.39], [86.87% +13.13]),
    ),
    midrules: (3, 5),
  ),
) <tab:expanded-static>

== Coverage and competing replies
The frozen indicator improves ordering across the three versions on the complete unenriched synthetic and game panels (@tab:expanded-static). On the 2,758 classically tied four-piece synthetic pairs, Stockfish 14.1 improves by 5.93 pp (95% interval 5.07–6.85). On 3,938 classically tied four-piece played comparisons it improves by 0.77 pp (0.51–1.05). In the broader game sample there are 120 informative changes in the option at four pieces and 50 at five pieces. Exposure sampling yields 1,415 and 655 strict choices, respectively. Large gains on these samples estimate the effect conditional on exposure. The four-piece implication between survival and Black's win applies here as well.

For the original four-piece validation panel the legal-capture rule gives the greater average improvement. In the expanded five- and six-piece random panels, the surviving-capture rule gives the greater improvement for Stockfish 14.1, with 359 rescues and 15 harms at five pieces and 409 rescues and 14 harms at six. The reply filter yields 212 and 254 rescues with zero harms on those two panels. With exposure enrichment, the corresponding harm counts fall from 285 to 2 at five pieces and from 272 to 3 at six. Credit still favors the rule with broader coverage. Of 336 five-piece and 322 six-piece enriched decisions where the option points to the worse outcome, 335 and 318 have a white pawn on the seventh rank.

One six-piece example has White Kf6, Rg5 and pawns b5 and b7 against Black Kd4 and Pf2. Rg1 allows ...fxg1=Q without an immediate recapture, and b8=Q keeps the draw. Rg3 gives Black a win even though the capture-promotion disappears. The legal feature is observable in both versions of the board. Its connection to the outcome changes with the competing reply.

#matching-section(booktabs)

== The modern network selector
Stockfish 18 uses two differently sized networks and a material-based selector with a fallback to the large network. We change just the selector so it always uses the large network with the stock mixing and scaling. On the 4,527 exposed synthetic four-piece pairs this increases credit from 57.22% to 72.08% (95% gain interval 13.47–16.27 pp). At least one endpoint used the small network in 3,552 pairs. The largest change comes from queen defenders, where credit rises from 40.53% to 62.97%. On the exposed four-piece played pairs the corresponding change is 83.64% to 91.06%. This identifies part of the gap in the release evaluator and supplies a separate architecture-level intervention.

Removing the complete threat-feature input from the large network gives a second test. On the same exposed synthetic four-piece pairs, the raw large-network score drops from 72.18% to 37.06%. In the enriched five- and six-piece adverse subsets, removing those inputs raises credit from 80.06% to 94.35% and from 79.19% to 95.96%, respectively. In the four-piece random pairs where the option is the same across children, removing the group costs 8.23 pp. The intervention therefore probes a broad source of threat information. Its sign reversal shows that the contribution can conflict with the exact outcome when another reply matters.

== Native shallow search
We sampled 4,718 strict comparisons from the expanded panels, taking up to 384 per panel and stratum. The strata consist of classically tied pairs with the same indicator, pairs where it favors the higher WDL, and pairs where it favors the lower WDL. Each clean release searches both candidate moves in one two-line analysis at depths 1, 2, 4 and 8. We use one thread, an 8 MiB hash, a fresh game state per run and disable engine access to Syzygy. Reported depths include each engine's own quiescence search. WDL scoring remains external.

#figure(
  placement: auto,
  scope: "parent",
  caption: [*Static ordering and shallow search on the same choices.* Static is the per-release readout in @tab:expanded-static. The plotted synthetic samples condition on the direction of the option, and the depths retain the engine's normal quiescence logic. Each point uses exactly the same pairs within its panel.],
  image("paper-search.svg", width: 100%, alt: "Search rapidly repairs four-piece capture errors. Scores for five- and six-piece cases with adverse capture options vary at shallow depths and improve by depth eight."),
) <fig:search>

On the four-piece exposure sample, static credit ranges from 52.34% to 75.13% across the releases and reaches at least 97.79% at depth 1 (@fig:search). The four-piece game exposure subsample reaches at least 99.87% at depth 1. The adverse option cases behave differently. At six pieces the Stockfish 14.1 credit falls from 84.47% statically to 70.03% at depth 1, then reaches 95.03% by depth 8. At depth 8 every release exceeds 91% on this subset. The short search already resolves most of the four-piece issue, while positions with a competing reply can require greater search depth. We also reopened the full legal move set on two disjoint samples of 768 four-piece game parents, one drawn from ordinary alternatives and one from the exposure series. At depth 8, the clean releases choose a WDL-optimal move on 97.27%, 97.40% and 100% of the ordinary sample, respectively. The observed game move preserves the optimal WDL on 94.79%. On the exposure sample, the depth-8 rates are 99.09%, 96.35% and 100%. Full-root results vary with depth. In the exposure sample, Stockfish 14.1 reaches 93.88% at depth 1, 95.83% at depth 2 and 83.59% at depth 4. Restricted pair rankings and full-root WDL preservation are separate measurements.
]
