# When Promotion Cues Mislead Stockfish NNUE

[Research manuscript (PDF)](paper.pdf) · [Typst source](paper.typ) · [Technical notes and proofs](TECHNICAL.md)

The manuscript studies how a cue for the straight promotion square can help or hurt a static evaluator depending on the available legal continuations. The main experiments use Stockfish 12, 14.1 and 18, exact Syzygy outcomes through six pieces, a new calendar of played positions and fixed-depth native search. The targeted 14.1 intervention has near-zero average effect in two separate 2014 game samples. On their common support with synthetic boards, matching the defender, local geometry and signed native score margin leaves a smaller score response in games. The matched logit replay measures a smaller change entering the first hidden layer. The original skewer and knight experiments test how specifically a network path relates to a legal tactic.

## Evidence by claim

| Question | Source and measurement |
|---|---|
| Why is the promotion intervention weaker in played positions? | [Second-half sampling flow](results/promotion-match-flow.csv), [raw strata](results/promotion-match-strata.csv), [match and balance results](results/promotion-matched.csv), [ordering-gain decomposition](results/promotion-match-swap.csv), [crossed logit replay](results/promotion-match-path.csv), [within-parent move-choice control](results/promotion-match-played-choice.csv), and [2013 parent exclusion](results/promotion-match-history.csv). |
| How do frozen rules transfer across releases, material classes and games? | [Sampling flow](results/expansion-flow.csv), [native and corrected static scores](results/expansion-static.csv), and [both frozen 14.1 donor directions](results/expansion-donors.csv). |
| What changes when Stockfish 18 uses the large network? | [Native versus forced large-network evaluator](results/expansion-sf18-selector.csv) and [threat-feature intervention on the raw large network](results/expansion-threats.csv). |
| What survives Stockfish search? | [Restricted two-root ordering at depths 1, 2, 4 and 8](results/expansion-search.csv), [full-root WDL-optimality on played positions](results/expansion-root.csv), and the [depth comparison figure](paper-search.svg). |
| Does the proposed capture option predict the errors? | [Held-out and played prediction](results/prediction-promotion.csv), [paired channel interventions](results/prediction-promotion-donors.csv), [full adjacent-capture census](results/prediction-promotion-census.csv), and [strict error audit](results/prediction-promotion-residuals.csv). |
| Where does the capture correction fail after adding a white pawn? | [Added-pawn transfer](results/prediction-five-piece.csv), [option outcomes](results/prediction-five-piece-options.csv), [seventh-rank stress test](results/prediction-five-piece-stress.csv), [counter-promotion filter](results/prediction-five-piece-reply-filter.csv), and [FENs of overwritten correct rankings](results/prediction-five-piece-stress-witnesses.csv). |
| Does the material-dependent readout persist with direct-capture and small-margin controls? | [Four-material action controls](results/prediction-skewer-controls.csv), with size, uncertainty and sign of each intervention. |
| How much information does the classical scalar lose? | [Classical lookup error and move ordering](../atlas/results/census.csv) and [strict reversal rectangles](../atlas/results/crossovers.csv). Counts cover the stated 25 material classes. |
| How much interaction is necessary across a family of moves? | [Complete preference-graph census](results/preference-graphs.csv), [exact minimum cycle means and additive error bounds](results/preference-cycles.csv) and [Syzygy-replayed witnesses](results/cycle-witnesses.csv). The PDF embeds the [complete arc ledger](results/cycle-arcs.csv) for the 225 graphs and the expanded experiment summaries. A shortest length of zero denotes an acyclic graph. The same table records projected interaction energies and exact rational nonnegative cycle certificates for the minimum comparison-space correction. A certificate lists its cycle coefficient and each edge as enemy-square:lower-destination–higher-destination, using zero-based squares. |
| What chess action drives an escort or a failure on a pin? | [Exact king and pawn tests](../atlas/horizon/kings/README.md) and [matched input and hidden-layer interventions](../atlas/horizon/kings/routes/README.md). |
| Which of the three omitted pawn files nearly pass the single-viewpoint bucket screen? | [Bucket counts and off-ray minus clear-ray contrasts](results/screen.csv). This table lists candidates with at least 24 unique maxima; the full input-weight scan covers all 1,024 coordinates. |
| Which channels contribute on new layouts and pass specificity controls? | [Promotion donor scores and king-group resampling intervals](results/promotion.csv), [coverage-specific strict wins, errors and ties](results/promotion-errors.csv), [promotion-necessary vs non-necessary specificity](results/promotion-specificity.csv), [mean selected-channel activations](results/promotion-geometry.csv), [perspective census](results/perspective-census.csv) and [one-square king relocations between d and e](results/king-crossing.csv). Material codes 2, 3, 4 and 5 denote knight, bishop, rook and queen; 0 denotes a pooled summary in the error table. |
| When do useful promotion channels hurt, and why do pooled played games differ? | [Straight versus capturing promotions](results/promotion-actions.csv), [grouped intervention intervals](results/promotion-action-intervals.csv), [ray-coverage and played-alternative regime stratification](results/capture-regimes.csv), and [misordered FENs with all optimal Black replies](results/promotion-witnesses.csv). The first four deletion labels form a partition. The underpromotion test overlaps that partition. |
| Does the ordering loss repeat in games? | [Two calendar panels](results/played.csv) and [all legal game-parent alternatives](results/capture-regimes.csv). Columns named drop report ordering credit *after* the intervention. |
| Why does H2 unit 2 respond to an attacked bishop and across KBvKR broadly? | [Material and single-unit exchanges](results/tactical-switch.csv), [selected FT weights](results/tactical-inputs.csv), [H1-to-H2 edge contributions](results/skewer-edges.csv), [all H1/H2 donor scores](results/tactical-layers.csv), [per-pair verdict accounting](results/verdicts.csv), and [skewer vs non-skewer and hybrid specificity controls](results/skewer-specificity.csv). |
| Which inputs carry the knight-fork margin on the tested positions? | [Selective FT-to-H1 path interventions](results/fork-path.csv), [activity-matched controls and non-fork specificity](results/fork-controls.csv), [geometric one-piece edits](results/fork-geometry.csv), [2×2 fork crossover rectangles](results/fork-rectangles.csv), and [30-bucket check-square weight co-localization](results/fork-colocalization.csv). |
| How do Hodge margins and correction certificates behave on concrete motifs? | [Candidate and validation counts](results/motif-census.csv), [held-out contrasts and uncertainty intervals](results/motif-transfer.csv), [FENs, legal outcomes and native layer values](results/motif-witness-positions.csv), [rational certificates](results/motif-witness-certificates.csv), and [independent Stockfish audit](results/motif-native-audit.csv). The control margin in the circulation is stored with its sign reversed. |
| Which samples transfer? | [Matched cohort flow](results/cohorts.csv) and [fixed-material cohorts outside the matched placements](results/outside-layouts.csv). The latter can share kings across materials or with discovery; by construction the complete king–king–attacker triple is disjoint from the matched panel. |

## Figures in the manuscript

| Figure | Measurement and source |
|---|---|
| Queen and pawn board | The two legal children after Qb6 and Qb7. [Board renderer](board_figure.py) |
| [Promotion inputs](paper-promotion.svg) | [FT1424 across clear, blocked and off-line placements](../atlas/horizon/kings/routes/results/promotion-coverage.csv) and [held-out donor effects by file](results/promotion.csv). |
| [Capture regimes](paper-capture.svg) | [Move-deletion subsets](results/capture-regimes.csv) and [fixed correction on held-out and played choices](results/prediction-promotion.csv). The two panels use different inclusion criteria. |
| [Five-piece transfer](paper-transfer.svg) | [Four-piece correction](results/prediction-promotion.csv), [added-pawn transfer](results/prediction-five-piece.csv) and [seventh-rank stress with or without the reply filter](results/prediction-five-piece-reply-filter.csv). |
| [Static and shallow search](paper-search.svg) | [Paired depth measurements](results/expansion-search.csv). The full-root game comparison is tabulated separately. |
| [Skewer gates](paper-skewer.svg) | Native H2 unit-2 logits on the matched bishop/queen panel and [its verdict changes](results/verdicts.csv). |
| [Rook action controls](paper-rook-controls.svg) | [Action-specific donor effects across material and margin strata](results/prediction-skewer-controls.csv). |
| [Knight fork route](paper-fork.svg) | [Weight co-localization](results/fork-colocalization.csv) and [selective H1 path interventions](results/fork-controls.csv). |

## Reproduction

Start from the [Stockfish and Syzygy setup](../README.md), the [complete atlas](../atlas/README.md), the [horizon study](../atlas/horizon/README.md), the [king study](../atlas/horizon/kings/README.md) and the [input route experiment](../atlas/horizon/kings/routes/README.md). Those runs produce the pinned network, dense classical/neural/Syzygy board tables, both quiet-reply partitions and game panels. The final study uses a CPU, GNU C++, Python with NumPy, python-chess, Matplotlib, and Typst.

~~~bash
bash research/nnue-tempo/manuscript/run.sh .scratch/nnue
~~~

Reproduce the revised cross-release experiment from that completed scratch tree with Python 3.12 or later, NumPy and the pinned python-chess checkout in the setup. The runner fetches the six new rated PGN archives and the extra Syzygy WDL tables into scratch. It builds the two additional score probes, generates and scores the fresh samples, runs the four search depths, and regenerates the depth figure.

~~~bash
bash research/nnue-tempo/manuscript/expansion.sh .scratch/nnue
~~~

Run individual stages with `inputs`, `prepare`, `synthetic --pieces 4`, `games --archive 2014-01`, `static`, `donors`, `modern`, `search`, or `root`. Use piece counts 4, 5 and 6 and months 01 through 06. The search stages require the aligned pairs and scores from `static`. The experiment uses Stockfish revisions `c306d838697011da0a960758dde3f7ede6849060`, `7262fd5d14810b7b495b5038e348a448fda1bcc3` and `cb3d4ee9b47d0c5aae855b12379378ea1439675c` in release order. The sampling and width/depth controls are specified in the PDF and implemented in [expansion.py](expansion.py), [expansion_search.py](expansion_search.py) and [expansion_root.py](expansion_root.py). Seeds are local to each material or game. The candidate ledgers and score arrays stay in the scratch directory.

For the promotion transfer analysis, run the matching stage after the original expanded run. It reads the frozen channels and the January–June and synthetic ledgers. It downloads July–December 2014 as needed, applies the same exposure sampler, and rebuilds the matching and replication summaries. The paper gives the exact matching calipers and two-way interval procedure.

~~~bash
bash research/nnue-tempo/manuscript/expansion.sh .scratch/nnue matching
~~~

To reproduce the new prediction, transfer and action controls from the completed atlas and motif panels, run `bash research/nnue-tempo/manuscript/prediction.sh .scratch/nnue`. The five-piece tests use the original instrumented Stockfish 14.1 binary and the three added-pawn tables from `setup.sh`. The capture term is applied to one board at a time and has a single weight fitted on discovery comparisons. The white reply filter is an exploratory analysis of the same stress sample.

To reproduce just the graph census and layer decomposition from the completed atlas, run `bash research/nnue-tempo/manuscript/cycles.sh .scratch/nnue`. The scan joins both atlas reply partitions and counts one comparison per common source within each graph. A board or destination pair can recur across graphs. Least cycle means are computed exactly on integer arc scores. The FT- and H1-only surrogate margins are reported on a grid of 1/16,384 final score units.

Reproduce the matched motif contrasts and first chess figure from the completed atlas with `bash research/nnue-tempo/manuscript/motifs.sh .scratch/nnue`. The screening files in scratch contain source-matched alternatives. Each tested stratum retains at most one legal-action-qualified rectangle from a deterministic 16-candidate shortlist. All candidate-level outcomes use fresh-clock Syzygy signs. The confidence intervals resample whole king orbits. For two rows the Hodge coefficient is half the first margin minus the second margin in the same destination order. A zero correction certificate denotes feasibility with a per-rectangle oracle destination table.

The PDF uses a two-column, ACM SIGCONF-inspired Typst layout for this research draft. ICMR's [official author instructions](https://icmr2026.org/calls.html) specify the `acmart` review class and submission length. The chess diagrams are generated with [python-chess SVG rendering](https://python-chess.readthedocs.io/en/stable/svg.html), with piece artwork by [Colin M. L. Burnett](https://commons.wikimedia.org/wiki/User:Cburnett) under the terms provided with python-chess. The figure adds board colors, move annotations and panel captions.

The reproduction script regenerates the statistical figures after all result tables are available. From a completed scratch tree, run [paper_figures.py](paper_figures.py) with the same Python path and scratch argument as in [run.sh](run.sh). Its figure captions distinguish descriptive counts and point estimates from the archived 95% king-orbit bootstrap intervals.

For a PDF build from the archived figures and CSV attachments, run Typst 0.14 or later from the repository root.

~~~bash
typst compile research/nnue-tempo/manuscript/paper.typ
~~~

The discovery and validation king groups are disjoint under dihedral symmetry and king interchange. The split is defined over all square pairs before material or label filtering. The original pair selector uses seed 591234; validation uses 728903 and excludes previously selected destination pairs on the same triple. Each oriented pawn file pools the four defending pieces for discovery. All rankings, including the common-target fork path and the contributors to the bishop-to-queen H2 logit shift, are recomputed from discovery. Intervals in the promotion figure resample king groups within each file, conditional on the chosen channels. The control quadruple matches each target greedily by discovery mean absolute activation difference within its own viewpoint. Alternate selectors and single-coordinate donors can produce different estimates. Floating-point secant accounting is checked against exact integer endpoint margins for each scored promotion batch; chess scores in both enumerated validation and outside-layout tests are checked against the native atlas.

## Revision audit

The revision followed the [Academic Research Skills](https://github.com/Imbad0202/academic-research-skills) argument, integrity and reviewer rubrics at revision `ae243ac`. The paper centers on a chess-specific experimental comparison. The PDF includes the preference-cycle criterion, a concrete legal cycle, the sharp additive margin bound and the integer secant derivation. The evidence audit distinguishes donor attribution from the board-only correction, and both from oracle additive calibration. The held-out king orbits have appeared in prior analyses of the same atlas. The independently generated five-piece stress cohort tests a concrete response failure. The reply filter was evaluated after inspecting the stress failures, and its reported counts describe that cohort. The editorial check is an uncalibrated internal review.
