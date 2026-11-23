#let matching-section(booktabs) = [
== Why the intervention transfers poorly
The original four-piece synthetic exposure panel contains 2,016 clear/off-line comparisons. Equalizing the four coordinates raises ordering credit from 17.09% to 38.24%, with gains in both donor directions. The corresponding 235 played pairs from January to June 2014 show a change of −0.74 percentage points (pp). We investigated the gap and repeated the analysis on six new months of rated games [7]. The July–December archive contains 6,716,411 game records. The frozen exposure screen yields 2,497 candidate pairs. After excluding January–June game parents under horizontal reflection, previously used pairs and within-panel duplicates, 1,456 pairs remain. Of 1,394 strict comparisons, 271 meet the clear/off-line condition. The intervention changes credit by −0.18 pp (95% interval −3.32 to +2.77), confirming the small aggregate effect in the fresh calendar sample.

#figure(
  placement: auto,
  caption: [*The baseline leaves different room for repair.* Rows are four-piece clear/off-line comparisons. Reported gains average the two donor directions.],
  booktabs(
    (1.3fr, .55fr, .7fr, .65fr),
    (left, right, right, right),
    ([Sample], [Pairs], [Native], [Gain, pp]),
    (
      ([Synthetic], [2,016], [17.09%], [+21.16]),
      ([Jan–Jun 2014], [235], [80.64%], [−0.74]),
      ([Jul–Dec 2014], [271], [72.88%], [−0.18]),
    ),
  ),
) <tab:match-raw>

*Material and baseline ordering.* Queens account for 92.9% of the synthetic pairs, 42.1% of the first game sample and 37.3% of the replication. The within-material comparisons retain a synthetic advantage for each piece type. The synthetic effect is +11.19 pp for rooks and +21.92 pp for queens. In the second game sample the respective changes are 0.00 pp and −0.50 pp. The synthetic network ranks 82.84% of the pairs strictly backward, compared with 19.15% and 26.94% in the two game samples. A wrong native order can only gain credit when patched. A correct order can only lose credit. Within the already-wrong subset, the intervention still gives less credit in games, +5.56 and +16.44 pp across the two halves versus +28.89 pp on synthetic pairs. On the initially correct replication pairs it loses 6.22 pp. The aggregate gap therefore combines different opportunities for repair and different responses to the patch.

*Common support.* We match each game pair to up to three synthetic pairs, with replacement. We hold defender type, the pawn file in the White king's orientation, the capture side and the lower WDL sign fixed. We match the locations of both kings and the off-line defender, then the sign and size of the native NNUE margin. The primary design keeps 68 of 235 January–June pairs and 98 of 271 July–December pairs. Matching the margin sign fixes native ordering credit within each calendar sample. Mean absolute margins in the replication are 374 units for games and 376 for their synthetic matches. @tab:match-stages compares different match criteria on this same support.

#figure(
  placement: auto,
  scope: "parent",
  caption: [*How matching changes the estimated ordering gain.* All values are percentage points. Each column uses the same game pairs throughout. Compare the estimates under the last row's signed-margin restriction to the game effect in the same column.],
  booktabs(
    (2fr, 1fr, 1fr),
    (left, right, right),
    ([Matching criterion], [Jan–Jun ($n=68$)], [Jul–Dec ($n=98$)]),
    (
      ([Defender type only], [+17.97], [+17.32]),
      ([Local geometry and defender], [+10.78], [+8.42]),
      ([Geometry and native margin sign], [+15.20], [+10.80]),
      ([Geometry and signed margin size], [+16.79], [+14.54]),
      ([Observed game effect on this support], [+0.74], [+0.26]),
    ),
    midrules: (4,),
  ),
) <tab:match-stages>

The fully matched synthetic effect exceeds the game effect by 16.05 pp in January–June (95% interval 1.72–39.96) and 14.29 pp in July–December (3.10–30.61). The wider match keeps 115 and 162 games and gives differences of +9.31 and +10.55 pp. The tighter match gives +27.78 and +15.48 pp on 42 and 49 games. Excluding parents from the older 2013 game panels leaves 246 replication pairs with exactly zero aggregate gain. The matched gap is +13.83 pp on 91 pairs (95% interval 1.80–30.57). Estimates vary with the overlap definition. Matching balances the number of opportunities for repair within its support, and a synthetic advantage remains in both calendar samples.

*Score shifts locate the remaining gap.* For a native signed margin $m$, denote the two patched margin increments by $d_H$ and $d_L$. Ordering gain averages the credit after adding each increment and subtracts the native credit. We can pair a game's margin with a geometry-matched synthetic patch response, and vice versa, as a numerical decomposition. Holding the game support fixed, the symmetric contribution from native margins is −2.02 pp in January–June and approximately zero in July–December. The contribution from the patch responses is +12.07 and +8.16 pp. These two components sum to the geometry-only gap in each half. The second-half patch-response interval is −0.91 to +17.12 pp.

The activation trace identifies a reason for the weaker responses. On the signed-margin matches, the sum of absolute differences across the four selected bytes is smaller in games, 41.0 versus 64.5 in January–June and 58.9 versus 76.3 in July–December. The mean patch increases the signed margin by 88.2 versus 154.0 native units in the first half and 146.6 versus 250.4 in the second. An exploratory replay sends the same four-channel logit change through each matched receiver's own quantized layers. The part attributable to the incoming change is +79.3 native units (95% interval 27.8–146.2) in the first half and +63.8 (10.4–128.7) in the second. The downstream components are −13.5 and +40.0 units, with intervals spanning zero. On the matched support, the smaller change entering the hidden layers accounts for a measurable part of the score-response gap in both calendar samples.

*Choice of the observed move.* The played move has the lower Black WDL in 235 of 235 initial and 268 of 271 replication comparisons. Within a parent, we held the risky capture-permitting child fixed and replaced the observed off-line move with another off-line move of the same WDL. Across 152 initial and 171 replication games, the resulting intervention gain rises by 1.29 and 1.49 pp, with both intervals spanning zero. The selected move explains little of the point-estimated gap in this restricted within-parent test.
]
