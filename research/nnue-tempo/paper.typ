#set page(paper: "a4", margin: (x: 22mm, y: 21mm), numbering: "1")
#set text(font: "Libertinus Serif", size: 11pt, lang: "en")
#set par(justify: false, leading: 0.62em)
#set heading(numbering: none)
#set document(title: "The value of having the move in Stockfish NNUE")
#let teal = rgb("#16566b")
#show heading.where(level: 1): it => text(size: 19pt, fill: teal, weight: "bold", it.body)
#show heading.where(level: 2): it => text(size: 13pt, weight: "bold", it.body)
#let report = json("results.json")
#let mats = ("KPvK","KPvKP","KNvKP","KBvKP","KRvKP","KRvKR","KRPvKR","KNPvKP","KBPvKP","KPPvKP","KQPvKQ","KRPvKP","KRPPvK","KRNvKR")
#let fmt(x) = if x == none [—] else [#str(calc.round(x, digits: 3))]
#let tbl(cols, headers, rows) = {
  set text(size: 9.2pt)
  table(columns: cols, inset: (x: 4pt,y: 3.2pt), align: left,
    stroke: none,
    fill: (x,y) => if y == 0 {rgb("#dce8eb")} else if calc.odd(y) {rgb("#f4f6f6")},
    table.header(..headers.map(h=>text(weight:"bold",h))), ..rows.flatten())
}
#text(fill:teal, size:10pt, weight:"bold")[EMPIRICAL STUDY IN CHESS EVALUATION]
#v(4mm)
#text(size:27pt, weight:"bold")[The value of having the move\ in Stockfish NNUE]
#v(2mm)
#text(size:14pt)[Tablebase counterfactuals, hidden-unit interventions, and move selection]
#v(3mm)
#text(size:10pt, fill:rgb("#505b60"))[Research manuscript · November 2026]

== Abstract

A chess evaluator can assign plausible values to two positions while missing the consequence of changing whose turn it is. We keep the pieces fixed, swap the turn, and use independent Syzygy outcomes to measure this effect. Among 352,000 sampled placements in fourteen three- to five-man material classes, Stockfish 14.1 NNUE outranks its classical evaluator on beneficial first moves in thirteen classes. The direction replicates in twelve of thirteen estimable classes from 4,745 played games. An exact king-and-pawn bitbase wins the remaining comparison in both cohorts.

The paired score cancels the 14.1 network’s learned piece-square output exactly before output quantization. In rook versus pawn, clamping one of eight hidden coordinates reduces held-out ranking AUC from 0.847 to 0.438. The effect repeats on 24,000 fresh placements. Legal-placement edits connect this coordinate to the urgency of an advancing pawn. In Stockfish 18, removing a separate encoded threat input reduces AUC on game-derived rook-versus-pawn placements from 0.895 to 0.650. On 6,000 decision roots, enabling the 14.1 production NNUE mode saves a net 94 tablebase outcomes at a 64-node limit and 15 at 4,096 nodes. These tests establish a specific strength, its computational route in two releases, and circumstances in which an explicit rule does better.

== Which question does the comparison answer?

Pálsson and Björnsson studied Stockfish 14.1 using linear surrogates, input and first-hidden-layer probes, and examples of forks, king attacks, and pawn promotions. Their examples suggest a testable question. Can a static network rank the urgency of a threat when the entire piece configuration is held constant? Haworth’s enumerations of mutual zugzwang supply a complementary case in which having the move makes the outcome worse. [#link("https://www.ijcai.org/proceedings/2023/0541.pdf")[Pálsson & Björnsson, §§4.3–4.4] and #link("https://centaur.reading.ac.uk/5950/1/2001g_CoW6_H_3-5-Man_Mutual_Zugzwangs_in_Chess.pdf")[Haworth, §1].]

The experiment measures three different quantities. We measure discrimination against perfect play, dependence of that discrimination on particular computations, and the result preserved by a finite search. Each has its own denominator. In particular, the ability to order positions by urgency and the precision of a rule declaring zugzwang lead to markedly different results.

#pagebreak()
= 1. Matched positions and exact labels

For a placement $B$, let $v_W$ and $v_B$ be White’s outcomes under perfect play with White and Black to move, respectively, in $\{-1,0,1\}$. Define $D(B)=v_W-v_B$. A positive difference means that each side benefits from moving first. A negative difference means that each would prefer to pass. Zero means that the two outcomes agree. The paired positions are legal inputs to an evaluator. The swap itself is a counterfactual, with both positions evaluated from a fresh clock and empty history.

*Sampling.* Fix the material and piece colors and sample distinct squares uniformly. Reject placements with an invalid orientation, either king in check, or a terminal orientation. Deduplicate within class. Use 16,000 placements in each of six three- or four-man classes and 32,000 in each of eight five-man classes. Rights to castle and capture en passant are empty. The Syzygy WDL outcome at clock zero maps cursed wins and blessed losses to draws under the fifty-move rule. Twelve oriented records in the five-man sample required this conversion.

*Readouts.* Obtain the raw NNUE and the classical evaluator from the same release in 12 and 14.1. Dynamic contempt or trend is zero. Keep specialized endgame code, including the exact KPK bitbase. For 18 use the raw large network. Given scores $s_W,s_B$ in side-to-move units, use $S=s_W+s_B$ as a predictor of $D$. The plus sign converts the Black-to-move score to White’s point of view before taking a difference. Use signed scores directly, including Stockfish 12. The sign orientation is fixed across all experiments.

*Primary endpoint.* Within material, rank $D>0$ against $D=0$ by $S$. AUC is the probability that a randomly selected positive receives the higher score, with half credit for a tie. Negative cases have a separate census. Hold out both kings’ square pair as a group, folding horizontal and vertical reflections and king interchange. A deterministic hash partitions the groups into discovery and test halves. The primary test contains 51,221 positive and 119,049 stable placements. Pointwise 95% intervals use the paired AUC influence functions aggregated by king group, with a finite-cluster correction. Family comparisons are exploratory and intervals are marginal. Original model training sets are unknown.

== An exact cancellation in Stockfish 14.1

*Proposition.* On a fixed placement, the entire paired raw score before final integer division is determined by the nonlinear positional head. Write $A,B'$ for the two perspective accumulators, $P$ for the side-relative learned piece-square term, $h(A,B')$ for the last 32 activations, $w$ for the final weight vector, and $b$ for its bias. The paired numerator is
$ 2b + w^top ( h(A,B') + h(B',A) ). $

*Proof.* The release orders the two accumulator halves by turn. The piece-square branch takes half the difference of its perspective totals. Reversal negates it exactly, since C++ truncation toward zero is odd. The material bucket stays fixed. Adding the two affine head outputs cancels $P$ and gives the expression above. Dividing the paired numerator by 16 differs from the sum of the two individually truncated scores by less than two internal score units. We verified the cancellation and reconstructed both dense outputs on all 704,000 oriented observations. #link("https://github.com/official-stockfish/Stockfish/tree/sf_14.1/src/nnue")[Stockfish 14.1 source.]

#pagebreak()
= 2. Which positions reveal an advantage?

#align(center, image("tempo-ranking.png",width:100%))
#text(size:9.4pt)[*Figure 1.* Difference in paired-turn AUC, with pointwise 95% intervals. A positive value favors NNUE. Game intervals use one placement per game within a material class. KRPPvK had 12 random-sample test positives and zero game-cohort positives.]

In rook versus pawn, 14.1 NNUE achieves AUC 0.847 against 0.514 for classical. Restricting the sample to placements where neither side can immediately capture or promote gives 0.740 against 0.635. In knight versus pawn that restriction gives 0.839 against 0.500, and in bishop versus pawn 0.854 against 0.500. The two-player restriction removes immediate material collection as a complete explanation.

Ordinary position ranking tells a similar story. Compare positions of different WDL within the same material and turn, then weight the two turns by their unequal-label pair counts. The 14.1 rook-versus-pawn concordance is 0.945 neural and 0.781 classical. The single-pawn values are 0.996 and 1.000. These comparisons use a static score on one board; the matched-turn test measures a distinct sensitivity.

*A simple rule wins a comparison.* Count the two sides with a legal capture and the two sides with a legal promotion. In rook versus rook, only the capture count varies. Its AUC is 0.948, compared with NNUE’s 0.613 and classical’s 0.500. The same rule reaches 0.871 in rook and knight versus rook, where 14.1 NNUE reaches 0.636. A useful learned response can coexist with a simple and stronger explicit feature on a particular material distribution.

*Played positions.* Screen all 121,332 games in the January 2013 Lichess standard rated archive. Use reservoir sampling to select one eligible occurrence per game and material class, mirror colors when necessary, and standardize rights and clock as in the generated cohort. This gives 6,982 game-class observations from 4,745 games. The same thirteen-class ordering direction holds wherever both labels occur (twelve favor NNUE, KPK favors classical). Repeated placements across games retain their occurrence weights. This cohort measures a distribution of reached boards. Game results play no part in its labels. #link("https://database.lichess.org/")[Lichess open database.]

#pagebreak()
= 3. A pawn-sensitive hidden coordinate

The 14.1 network routes three- and four-man positions through stack 0 and five-man positions through stack 1. Each head compresses two 1,024-coordinate feature vectors to eight units, then expands to 32 and ends in a scalar. For each material and layer, clamp each coordinate to its rounded discovery-set mean for both turns and replay the remaining integer layers. Select the largest discovery AUC loss and measure it on the held-out king geometries. Published arrays give the responses to all eight and all 32 clamps.

#tbl((1.4fr,1fr,1fr,1fr,1fr), ([Material],[Original],[H1 unit → AUC],[H2 unit → AUC],[Stack]), (
  ([KPvKP],[0.809],[5 → 0.654],[12 → 0.637],[0]),
  ([KNvKP],[0.787],[7 → 0.616],[7 → 0.764],[0]),
  ([KBvKP],[0.838],[7 → 0.642],[26 → 0.814],[0]),
  ([KRvKP],[0.847],[2 → 0.438],[14 → 0.711],[0]),
  ([KNPvKP],[0.700],[1 → 0.458],[#str(report.static.KNPvKP.lesions.at("2").unit) → #fmt(report.static.KNPvKP.lesions.at("2").test_auc)],[1]),
))
#text(size:9.4pt)[*Table 1.* Native-weight interventions, zero-based indices. Scores use the unrounded paired numerator. The same numerical index in another stack refers to different weights.]

*Natural donor test.* In held-out KRvKP, keep both kings, the rook, and the pawn’s file fixed. Shift the black pawn from rank 3 to rank 2 if its forward square is empty, both orientations remain eligible, and each side’s binary capture availability is preserved. Both turns are evaluated at each endpoint. This exploratory follow-up was chosen after inspecting the first ablations. Moving from rank 5 to rank 4 serves as a comparison edit.

#tbl((1.5fr,1fr,1fr), ([Quantity],[Rank 3 → 2],[Rank 5 → 4]), (
 ([Matched placements],[836],[879]),
 ([Mean unit 2 activation, Black to move],[52.8 → 16.9],[80.9 → 65.8]),
 ([Mean unit 2 activation, White to move],[121.1 → 124.7],[122.2 → 121.0]),
 ([Exact tempo increased / stable / decreased],[539 / 266 / 31],[114 / 703 / 62]),
 ([Mean paired score change, internal units],[+781.2],[+17.7]),
 ([Symmetric unit 2 transplant effect],[+472.7],[+115.9]),
 ([Edit-ranking AUC, intact / unit 2 frozen],[0.885 / 0.745],[0.637 / 0.354]),
))
#text(size:9.4pt)[*Table 2.* The transplant effect averages the old-to-new replacement and the sign-reversed new-to-old replacement. Edit-ranking AUC compares increased with stable exact tempo. The frozen readout keeps unit 2 at its source value while the other seven units take their target values.]

Unit 2 responds especially strongly when Black can use the advanced pawn immediately. Its signal affects both output magnitude and agreement with perfect play. Other units contribute. The near-promotion unit 7 transplant averages +403.5, and several coordinates oppose the shift. Interventions in a nonlinear head can interact, so the transplant effects need not sum to the total.

*How much can a short chess rule recover?* Fit an additive lookup to this single activation on discovery boards, separately for each turn. Use the pawn rank, the two kings’ Chebyshev distances to the pawn, rook ahead or behind on its file, and attacked status of the rook and pawn (22 coefficients per turn). Substituting turn-specific constants gives AUC 0.481. Adding pawn rank gives 0.647, the two distances 0.674, and all seven features 0.716. The full model explains 51.6% of Black-to-move unit variance and 20.7% for White. Transferred unchanged to 367 played placements, the seven-feature replacement gives 0.752 against the intact network’s 0.888.

*Frozen replication.* Fix the unit, clamp, seven features and coefficients. Sample 24,000 further placements on held-out king geometries, excluding original placements and file mirrors. AUC is 0.839 intact, 0.447 clamped, and 0.724 with the chess-rule replacement. In 2,706 near-promotion edits, retaining the source unit reduces edit-ranking AUC from 0.881 to 0.701. The remaining gap is compatible with interactions among familiar chess concepts.

#pagebreak()
= 4. Three releases and an input intervention

#let scores = mats.map(m=>{
 let s = report.static.at(m).held
 let z = report.modern.at(m).uniform.all
 ([#m],[#s.pos / #s.n],fmt(s.auc.sf141h),fmt(s.auc.sf12n),fmt(s.auc.sf141n),fmt(z.full),fmt(z.knockout))
})
#tbl((1.25fr,1.45fr,1fr,0.85fr,0.85fr,0.85fr,1.05fr),
 ([Material],[Positive / test],[Class. 14.1],[NN 12],[NN 14.1],[NN 18],[18 ablated]), scores)
#text(size:9.4pt)[*Table 3.* Held-out AUC on positive versus stable exact tempo. “18 ablated” zeroes both perspective accumulations of FullThreats, including its piece-square contribution, before the feature transform. The original trained weights and piece-square inputs remain available to the network.]

Stockfish 18 directly encodes a supported set of attacker–target relations as inputs. With its large network, removing this channel lowers rook-versus-pawn AUC by 0.106 [0.095, 0.116] on generated placements and 0.244 [0.188, 0.301] on played placements. In rook-and-pawn versus pawn the game-cohort drop is 0.273 [0.207, 0.339]. The intervention leaves every output exactly equal on all 51,756 generated held-out placements having zero active relation features. A knockout in a trained network measures dependence on this channel. Training a separate model without it would measure adaptation. #link("https://github.com/official-stockfish/Stockfish/tree/sf_18/src/nnue")[Stockfish 18 architecture and FullThreats implementation.]

The historical progression varies by task. In KPPvKP the paired AUC moves from 0.291 (12) to 0.800 (14.1) to 0.613 (18). On the played boards, the corresponding 14.1 and 18 values are 0.710 and 0.731. Training distributions, objective, features, widths and readouts all differ across releases. These comparisons characterize the released evaluators and leave the sources of between-release differences unresolved.

== A concrete chess interpretation

White king d4 and knight d1 face Black king d6 and pawn a3. Neither side can capture or promote immediately. With White to move, Kc3, Ne3 and Nc3 each draw. With Black to move, a2 is the unique winning move. The 14.1 classical evaluator gives zero for both turns. NNUE gives −86 for White to move and +537 for Black to move, both from the mover’s perspective. The tablebase certifies that the urgency is real. This example illustrates the aggregate comparison in which the network ranks 540 quiet positive knight-versus-pawn placements against 4,118 quiet controls.

#pagebreak()
= 5. Where the network fails, and what search gains

== A complete census of negative tempo

Enumerate all 163,306 eligible placements of a white pawn and two kings, counting left–right mirrors separately. There are 160 strict negative-tempo placements, 27,608 positive ones and 135,538 stable ones. Interpret a strictly negative paired score as a declaration of reciprocal zugzwang. The two classical bitbase paths recover the exact sign on the full domain.

#tbl((1.45fr,1fr,1fr,1fr,1fr), ([Evaluator],[True alarms],[False alarms],[Recall],[Precision]), (
 ([Classical 12 / 14.1],[160],[0],[100%],[100%]),
 ([NNUE 12],[9],[5,345],[5.63%],[0.168%]),
 ([NNUE 14.1],[156],[34,178],[97.50%],[0.454%]),
 ([NNUE 18 (large)],[158],[85,830],[98.75%],[0.184%]),
))
#text(size:9.4pt)[*Table 4.* The unconditional prevalence is 0.098%. Threshold zero tests score direction. A tuned decision threshold would define a different detector.]

A score can legitimately prefer one of two winning positions. WDL makes both wins equal. The low precision concerns declaring a change in game-theoretic outcome from the score sign. The four 14.1 misses form two mirror pairs. One has White Kg3 and Pb2 against Black Kg5. White to move draws, Black to move loses, yet the neural prediction favors White having the move. The reference bitbase recognizes the distinction exactly.

== Selecting moves under a finite search budget

Use clean Stockfish 14.1, one thread, 4 MiB hash, external Syzygy disabled, and a fresh game per decision. Sample 1,000 held-out roots per three-/four-man class with at least one optimal and one inferior legal move. Draw the turn before examining successors. Enumerate every child and verify that the best attainable WDL reproduces the root label. Check DTZ where the halfmove clock increments. The two configurations use the same nominal node limits. The default selects between neural and classical evaluation according to the release’s production routing.

#tbl((1fr,1.15fr,1.15fr,1.22fr,1.6fr), ([Nodes],[Classical],[Default],[Only default / class.],[Gain, pp (95% CI)]), (
 ([64],[5,807],[5,901],[132 / 38],[+1.57 (1.14, 2.00)]),
 ([512],[5,903],[5,916],[73 / 60],[+0.22 (−0.14, 0.58)]),
 ([4,096],[5,971],[5,986],[24 / 9],[+0.25 (0.06, 0.44)]),
))
#text(size:9.4pt)[*Table 5.* Success means preserving the exact win, draw or loss. Differences are paired within root. Intervals center within each material class and cluster by king group. UCI stopping may overshoot. Median nodes for both modes were 65, 512, and 4,099 at the respective limits.]

The largest pooled gain appears at the smallest tested budget. The bishop-versus-pawn subset goes the other way at 512 nodes (973 default versus 990 classical). The design weights materials equally, conditions on a meaningful choice, and tests one move. Match Elo, time-to-depth, longer-piece endgames, unseen network training histories, and transfer of the identified unit to middlegames remain open questions.

#v(2mm)
#text(size:9.4pt)[*Reproducibility.* Public companion files include the release-pinned build, exact sampling and evaluation scripts, family-level results and the twelve-source literature review. Native display checks matched 204 positions per older release and 468 for Stockfish 18. The study uses local CPU inference throughout.]
