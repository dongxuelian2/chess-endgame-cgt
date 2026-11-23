# When kings change a pawn race

Two kings can reverse a bishop interception by blocking a capture, supplying a target for a pin or a discovered check, or attacking the bishop while supporting promotion. Stockfish 14.1's static NNUE orders all 42 examples of the last pattern correctly and misses 93 of 108 quiet pins in the reversal set. A second experiment follows a black king escorting a pawn against a rook or a queen. On 2,830 matched comparisons the board geometry and the decisive escort rule agree across materials, while different hidden coordinates carry the score difference.

The input-level follow-up traces the material change to the two king perspectives, identifies five promotion-square channels, and tests hidden-layer interventions on 581 played promotion decisions and on rook skewers and knight forks. [From king-relative inputs to endgame decisions](routes/README.md)

## What is compared

White has two legal quiet moves of the same piece from the same starting square. They receive identical full classical scores. Black to move has neither an immediate capture nor a promotion after either choice. The high child H has a better Syzygy outcome for Black than the low child L. Every distinct pair of destinations counts once per fixed material, kings and black piece, across all common sources. Outcomes use the sign of fresh-clock Syzygy WDL as in the parent study. Accuracy gives half credit to a tied score. Coordinate numbers start at zero in the released eight-unit bottleneck. [Atlas and classical evaluation](../../README.md) · [Pawn race, network and thirteen-ply interval definitions](../README.md)

## All 295 reversed bishop comparisons

These 295 pairs exhaust the strict errors of the kingless bishop interception game among 1,359,347 informative bishop–pawn pairs. They span 47 different placements of both kings and the pawn. Every black pawn is on its third rank from White's side (a3–h3), H is a Black win, and L is a draw. In the reduced game the bishop can stop the pawn on H and the pawn escapes on L.

**In 253 pairs the white king lies strictly between the bishop on H and the pawn's next square.** The reduced board offers an immediate bishop capture after the push. The real king physically blocks its diagonal. Restoring just the stationary white king removes the interception on H in all 253. Restoring just the black king leaves both original predictions unchanged. On the orthodox board the pawn push is Black's sole winning first move on H. On L the same push draws, with all saving replies falling into the first four rows below.

**In 42 pairs the black king stands beside a rook pawn on the third rank.** With Kb3/Pa3 against Bb1, or Kg3/Ph3 against Bg1, the sole winning king step is to b2 or g2. From there it attacks the bishop and protects the promotion square. An immediate pawn push draws on both children. Even both kings as fixed obstacles preserve the reversed prediction.

| Complete reversal set | Pairs | NNUE right / tie / wrong | Unit 7 zero at both ends | Further bishop pairs meeting the action test | Native accuracy there |
|---|---:|---:|---:|---:|---:|
| Quiet absolute pin after the push | 108 | 15 / 0 / 93 | 95 | 1,838 | 71.03% |
| White king uncovers a bishop check | 72 | 19 / 0 / 53 | 50 | 1,417 | 50.18% |
| Direct bishop check | 53 | 22 / 1 / 30 | 38 | 48,894 | 76.76% |
| Either an absolute pin or a direct check saves | 20 | 5 / 1 / 14 | 20 | 20 | 67.50% |
| Black king attacks bishop and guards promotion | 42 | 42 / 0 / 0 | 0 | 2,554 | 90.80% |

In all 128 pairs admitting the quiet pin, the black king stands on the first rank one file away from its pawn. Consider White Kd3, Bg8 against Black Kd1, Pc3. After Bh7 c2 the white king blocks the h7–c2 diagonal. After Ba2 c2 the sole drawing reply Bb3 pins the pawn against d1. In the discovered-check family, the white king also stands on the line between the bishop on L and the black king. With White Kc4, Ba4 against Black Ke6, Pe3, the losing Bb5 permits ...e2. After Bb3 e2, either Kc3+ or Kd3+ opens the b3–e6 diagonal and draws. Every FEN and every saving reply after the push is recorded. [295 pair records](results/bishop-295.csv) · [Intervention totals](results/bishop-families.csv)

**What the two units do on these boards.** Let just coordinates 6 and 7 differ across the endpoints and keep the recipient's own direct piece-square evaluation. Replay the two possible donors using the native integer upper head. In each direction all 295 pair signs, including the two ties, equal the original NNUE signs. On the 108 quiet pins the direct branch alone earns 84 points. Adding the difference of unit 6 alone reduces this to 15, the full network's result. Unit 7 alone with the direct branch earns 77. On the 42 king forks the corresponding results are 6, 30.25 and 42 points. In this sample the learned path actively reverses many good direct-branch preferences on pins and repairs many bad ones on king forks. A different network can order the same geometry differently. SF12 earns 2/42 on the forks, and SF18's raw large network earns 40/42. Suppressing the four classes of emitted direct attack features in SF18 brings it to 7.5/42. [Cross-network replays](results/rank-network-controls.csv)

One-square king relocations retain both original bishop moves, their common source, the classical tie and the quiet filter. The white king stays in the canonical a–d half. Real WDL reverses after 118 white-king and 182 black-king edits. Across the edits starting from a correct native order and ending with opposite truth, NNUE changes to the new correct order on 39/41 white and 87/97 black edits. Each count is an original-pair/king-edit incidence. The ledger also includes edits giving equal outcomes or preserving the ordering. [Relocation counts](results/king-relocations.csv)

**Transfer domain.** Enumerating knight, bishop, rook and queen against a third-rank pawn gives 28,443,741 same-source, classically tied, quiet pairs including 2,873,602 with strict WDL order. In the push test H wins solely by pushing, L draws after the same push, and deleting every type of optimal White reply leaves a legal alternative. A quiet pin here means the advanced pawn is absolutely pinned after a nonchecking saving move. The king-fork test requires every winning first move on H to attack the opposing piece with the king and protect promotion, a legal nonwinning alternative, and an immediate push drawing on both children. Subtracting the 295 discovery pairs gives the last two columns of the table. Those columns measure the broader action geometries, including cases where the bare race agrees or ties. The king-fork test also finds 623 knight pairs (94.14% native accuracy) and zero rook or queen pairs. The discovered-check condition is especially difficult even after removing all seeds. [Complete census and neural interventions](results/rank-three.csv)

## What separates the 28 thirteen-ply residuals

In 28 of 240 fresh game comparisons the thirteen-ply conversion intervals still overlap in both directions. The static network orders 24 correctly and four incorrectly. For each case we restricted one legal move class on Black's first turn or on White's reply after each Black move. Both siblings receive the identical rule. Subsequent play uses ordinary Syzygy outcomes. If a restriction removes every legal option at a node, that node retains its original value. The method identifies actions whose availability is necessary to sustain a particular WDL ordering under the specified intervention. The tablebase supplies continuation value even when the actual conversion takes many moves.

| Game or group | Decisive replies |
|---|---|
| 880q0c1r, abzx8wn1 | After ...e4 White needs Rf1+, checking down the file. After ...f2 in the queen ending the winning replies are the quiet last-rank moves Qf1 and Qh1. |
| oc2wmsiy, vuhg9que | The only winning answers to the push are Rxh3 and Rxh4+ respectively. The latter also checks along the rank. |
| q25jew0x, qs22m3il | After the push, only a king step toward the pawn preserves White's win. The winning replies are Ke5/Kf5 and Kd5 respectively. |
| d6xmnbot, b0zgvlxp | After Rg6 the black king's only drawing replies are Kb3, Kb4 and Kb5. Rb6 takes away that file. In the other game the drawing corridor is g3/g4/g5. |
| 9jz0k85n, mx9gsckz | Black needs the contact check ...Rf2+ in the bishop–rook ending. In the rook–pawn ending White's Kc5 after ...Ka3 both guards the rook on c4 and approaches the pawn. |
| 87gsv0w8, zk0d9uor | White needs a king tempo while its rook already controls the promotion square along the last rank. The winning replies to the pushes are Ka6/Kb6/Kc6 and Kb7/Kc7/Kd7. |

Every residual has at least one outcome-erasing action class, including all four NNUE errors. Some action labels overlap, and a class with many legal moves can erase the distinction through its size. For each game's illustrative certificate we retained the same number of moves at each node uniformly at random and computed the exact erasure probability. For a White node with N moves, r retained and g values above a threshold, the probability that its minimum exceeds that threshold is C(g,r)/C(N,r). Products of child CDFs give the distribution at a maximizing Black node. Independent draws on both siblings give the probability the strict order disappears. It is 0.008 for the king approach in q25jew0x and 0.919 for the rook-cut label in oj4ogsus. These are size controls on rules selected after inspecting the games. The high value makes the latter certificate weak evidence of specificity. The complete ledger gives both original and restricted WDL, all optimal witness moves on the listed branches, neural margins, and the four negative cases. [28 certificates](results/played-certificates.csv) · [All overlapping restrictions](results/residual-28.csv)

The four wrong neural orders have concrete witnesses too. In whlrjock White can answer ...h3 with Kf4 or Kf5 while the rook on b1 covers h1. In x16c3pdb White's Qd5 after ...Kg2 pins the f3-pawn. In 0ykns7zh Black has two drawing knight checks after Rb7. In q0vkoy4l the white rook's contact check Rc7+ is the sole winning answer to either ...Bb3 or ...Bc4 on the relevant child. [Counterexamples with both candidate moves](results/played-certificates.csv)

A further 128,921 quiet, hash-selected comparisons from the remaining atlas partition exclude the 18,483 previous enumerated comparisons and all 28 residuals. Three other pairs also appear in the played panels. Under the same filters, removing White's capture after a push erases the order in 4,855 and NNUE earns 93.24% there. Requiring White's king to close on the pawn after a push yields 1,168 at 76.71%. Requiring a rook or queen to move onto or along the last rank after a push yields 214 at 81.07%. In 2,387 bishop–rook comparisons the order requires a quiet nonking destination available only on one sibling after the same Black move. NNUE earns 78.47%. Each action has its own denominator. The 47 restrictions can overlap within a pair. [Transfer by material, player and action](results/response-census.csv)

### Black's second turn reveals an escort requirement

Four correctly scored rook endings initially yielded a shared explanation as broad as “Black must push.” In all four the decisive resource reaches one move further. After a black pawn push and a White reply, suppress Black king moves landing adjacent to the pawn's next square. On both siblings apply this restriction at every such branch. Black's originally drawing sibling now loses in each game. The other sibling was already lost. These four pairs reach the final branch of SF14.1's classical KRKP formula on both endpoints. That branch reads the two kings' distances to the next pawn square and the pawn's distance to promotion. Both rook placements therefore receive the same value. In 1rb6lo8v, for example, Rf2 and Rf1 both score −208 in the classical evaluator. NNUE assigns +86 and −122 respectively, and only Rf1 wins for White. [Classical KRKP source](https://github.com/official-stockfish/Stockfish/blob/7262fd5d14810b7b495b5038e348a448fda1bcc3/src/endgame.cpp#L179-L218)

| Game | One line where the escort matters | NNUE margin in engine units |
|---|---|---:|
| 1rb6lo8v | Rf2 c3 Rc2 Kd3, covering c2 | +208 |
| dnp9n41r | Ra4 h4 Ra3+ Kg2, Kg4 or Kh2, each covering h3 | +62 |
| pvsyd1ny | Rd3 c5 Re3+ Kd4, covering c4 | +227 |
| yq6lhrnd | Ra2 g5 Ra4+ Kf3, covering g4 | +345 |

The alternate White replies above can punish the deletion. The ledger enumerates the ordinary Black answers that avert the worsened outcome along these branches. On the additional atlas comparisons the escort restriction erases 1,138/5,507 rook orders (76.41% NNUE) and 308/486 queen orders (87.66%). Both material-specific complement groups are included in the neural test. [Third-turn certificates](results/played-third.csv) · [Third-turn transfer](results/post-push-third.csv) · [Conditional neural scores](results/escort-paths.csv)

### Change the piece and keep the geometry

Place a rook or a queen on the same starting square. Fix both kings, a black third-rank pawn, both White move destinations, the two true WDL labels and the requirement that both moves tie the classical evaluator within each material. A Black restriction confined to continuations after an initial pawn push can erase H > L only if a push attains H. Indeed, any untouched first move attaining H would keep the restricted H at H, whereas a restriction on Black can only decrease L. Screening this necessary condition leaves 5,630 candidate geometries. Applying the three-ply escort restriction to both materials leaves **2,830 exact matches from 23 fixed king–king–pawn placements**.

| Same 2,830 geometries, pair-weighted | Native | Keep only 6 and 7 | Remove 6 and 7 | Remove 2 |
|---|---:|---:|---:|---:|
| White rook | 90.09% | 53.00% | 89.79% | 53.92% |
| White queen | 89.81% | 87.78% | 78.90% | 89.35% |

For every subset S of the eight hidden coordinates, let M(S) be the signed score margin averaged over the two donors while each child keeps its own direct branch. The mean marginal contribution of coordinate i over the 8! orders of addition is its exact Shapley allocation. Summing over all coordinates telescopes along each order to M(all) − M(empty). We evaluate all 256 subsets with the released integer shifts, clipping and truncation. Every per-pair allocation sums back to the native margin including M(empty). On the rook, coordinate 2 receives a mean +359.82 engine units and coordinate 7 receives +1.08. On the queen the corresponding allocations are +15.97 and +244.71. The queen has a higher keep-6/7 accuracy and larger coordinate-7 allocation, and the rook has a larger coordinate-2 allocation, in **every one of the 23 placements**. Equal weight per placement gives 48.43% versus 87.38% on the keep-6/7 test. The eight coordinates sit in the same material bucket in both experiments. This isolates a change in the score pathway under a legal piece substitution. The board intervention fixes the tested escort requirement while the full legal move trees still depend on the piece. [Matched totals](results/material-substitution.csv) · [All 23 placement-level comparisons](results/matched-escort.csv)

The four original rook games also show within-material variation. In 1rb6lo8v, pvsyd1ny and yq6lhrnd the largest positive margin allocation is to coordinate 2. In dnp9n41r it is to coordinate 6. Across all 28 residuals the intact network earns 24 points. Removing coordinate 2 gives 18 and retaining only 6 and 7 gives 16.75. These interventions locate the output signal conditional on the donor convention. The Syzygy restrictions establish the chess dependency. Whether an individual coordinate computes that dependency across unrelated positions remains an open question. The IJCAI study probes decodability of chess concepts in Stockfish. Here the paired chess intervention and the exact output-path intervention can each fail independently. [All per-game allocations](results/played-paths.csv) · [Pálsson and Björnsson, IJCAI 2023](https://www.ijcai.org/proceedings/2023/541)

## Reproduction and checks

Starting from the [completed horizon experiment](../README.md), run the following from the repository root on a Linux AVX2 CPU.

~~~bash
bash research/nnue-tempo/atlas/horizon/kings/run.sh .scratch/nnue
~~~

The native driver checks each unrestricted backup against Syzygy. Independent python-chess move generation and Syzygy probes replay every reversed bishop case, all 47 action classes on both turns for all 28 residuals and 110 stratified additional pairs, and 96 rook/queen substitutions in both materials. A separate stratified replay verifies the third-rank families outside the seed pairs, with up to 256 comparisons per material and family drawn by a fixed hash before excluding seeds. The residuals served as discovery data. Counts over a complete specified domain, sampled transfer performance and performance on real games have different sampling units. Source code and compact tables are versioned, and the native weights and intermediate board tables are reconstructed in scratch. [Independent replay counts](results/independent-replay.csv) · [Stratified rank-three checks](results/rank-network-controls.csv)
