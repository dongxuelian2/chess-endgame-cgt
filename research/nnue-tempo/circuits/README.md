# How Stockfish 14.1 changes its valuation of the same move

Each comparison fixes the candidate piece and both destinations. The reversal experiment changes one enemy piece's square and measures the four resulting static scores. The aggregate measurements, frozen channel and connection rankings, and layer-specific controls are in [results](results/). The [March specification](march-replication.md) records the primary endpoint chosen after inspecting February.

## Observed results

Both choices must strictly favor the labelled destination before and after relocating the opponent piece. The two threat cohorts are from different months. The WDL cohort reverses the ordering given by the tablebase.

| Sample | Comparisons | Released classical | Released NNUE | Classical with both cutoffs disabled | Full classical before grain truncation |
|---|---:|---:|---:|---:|---:|
| February threats | 3,764 | 25.05% | 29.81% | 36.96% | 40.28% |
| March threats | 4,950 | 26.18% | 29.35% | 37.72% | 40.38% |
| Syzygy outcome reversals | 6,360 | 12.04% | 17.61% | 12.80% | 21.53% |

The March effect meets the criterion fixed before that analysis. The released-network advantage on the strict endpoint was 3.17 percentage points, with a game-clustered 95% interval of [1.61, 4.74]. Enabling all ordinary classical terms at released score granularity gave the classical evaluator an 8.36-point advantage over NNUE [6.78, 9.94]. The 6,360 WDL reversals were selected from 27,846 tablebase-labelled move pairs. Under the stock cutoffs, classical before grain truncation reached 20.90% on WDL reversals. The strict endpoint is sensitive to output ties.

In March the median four-board comparison had 68 of 2,048 native feature increments change across threat contexts. Equalizing these increments reduced joint success from 29.52% to 2.91% at internal score resolution. Feature increments had a 95.54% projection onto the intact positional interaction. H1 and H2 changes in secants had 3.76% and 0.69%. The corresponding projections on outcome reversals were 72.21%, 8.69% and 19.10%. Cohorts have different piece-count distributions. These fractions describe within-cohort decompositions.

On 48 WDL reversals, even the untapered MG/EG term increments matched exactly across contexts with both lazy cutoffs disabled. Specialized endings use their actual specialized scalar. A fixed linear weighting of these effective features necessarily gives the same move margin in both contexts. Native NNUE strictly reversed correctly on 22 of the 48, including 19 rook-versus-pawn cases using the specialized classical evaluator and three other cases. Inspect the [positions and all four labels](results/handwritten-collisions.csv). The table contains all 65 collisions after phase weighting, with a flag for the 48 stronger cases. The endgames in this subset are selected by an exact equality condition.

[Evaluator controls](results/handwritten-controls.csv) · [Layer interventions](results/reversals.csv) · [Connection transfers](results/edge-transfer.csv) · [Endgame families](results/endgames.csv) · [Comparison figure](results/reversal-controls.svg)

## Boards and success criteria

- **Chess threat.** From every eighth game in three standard rated Lichess archives, sample one noncheck position uniformly among positions at each reached piece-count head. Enumerate two quiet, nonchecking moves from the same square by a knight, bishop, rook or queen. One landing square permits no legal capture of the moved piece. On the other, an opposing piece can take it with no immediate legal recapture on that square. Choose one eligible pair uniformly. For a reversal, relocate that opponent knight, bishop, rook or queen along an unobstructed movement path to an empty square, keeping the player and both choices fixed. Accept an edit when the same attacker can take on the previously safe square without immediate recapture and the other square becomes safe. Check both parents and all four legal children. The edited parent is a synthetic counterfactual. The motif describes a local exchange. Sacrifices can make the exposed choice sound.
- **Game-theoretic outcome.** Screen 6,000 held-out placements per endgame family, randomizing the player. Retain one uniformly chosen pair of quiet, nonchecking, nonpromoting moves from the same square with one preserving the root's best Syzygy result and the other worsening it. For the reversal subset, require a nonking mover and relocate one opposing nonking piece along a quiet geometric move. Include single pawn pushes, preserve both kings, and verify that the same candidates now have the opposite strict outcome ordering. Outcome is the sign of Syzygy WDL without a fifty-move draw limit. Generated parents have no castling or en-passant rights. The separate paired-turn test treats cursed wins and blessed losses as draws at a fresh clock.
- **Fork comparison.** From the same January and February game samples, select alternatives where the moved piece is safe on both squares. On one square it geometrically attacks at least two enemy nonpawn, nonking pieces. On the other it attacks zero. This is a distinct motif used to test whether connections transfer.
- Each choice margin is the inferior child's side-to-move evaluation minus the preferred child's. In a reversal, a strict success requires the original ordered margin to be positive and the edited margin negative. Zero fails the joint test. Ordinary one-context rates and the sign of the mixed difference count ties as half. Native NNUE divides the internal positional-plus-PSQT score by 16 and truncates. Intervention rates use the internal score. Report each with its matching intact control. Paired 95% normal intervals cluster by game for played positions or by symmetry-reduced pair of king squares for generated endgames. Family and phase subgroups are descriptive. The observations from related tasks overlap.

## Where the interaction appears

Write the move margin as the difference between children B and A, and the interaction as the original margin minus the edited margin. A component's reported projection is its inner product with the observed interaction, divided by the observed interaction's squared norm across cases. Signed projections add to one for an exhaustive additive decomposition. Large score changes receive greater weight. NMSE is squared reconstruction error divided by the sum of squared native effects.

With fixed kings and material, each raw HalfKAv2_hm accumulator is a sum of piece-square contributions. The mixed difference of a raw accumulator is exactly zero in this four-board design. The code verifies this identity for every slot. For a raw sum (u), the integer input to the dense head is `clip(u, 0, 127) = u + max(-u, 0) - max(u-127, 0)`. Its mixed difference therefore separates exactly into lower- and upper-threshold terms. The 2,048 slots concatenate two perspectives on a shared 1,024-channel transform. Head selection uses `(piece_count - 1) // 4`, with eight sets of dense weights.

The positional move margin factors into the feature increment, the first affine weights and activation secants, the second affine weights and activation secants, and the output weights. For each dense coordinate, a secant is the observed activation difference divided by the preactivation difference. Switch the feature increment and the two banks of secants from edited to original in all six orders and average marginal contributions. Test both zero and the interior unrounded slope when the denominator is zero. Both conventions reconstruct the endpoints. This is a finite-difference attribution to the trained computation. Early feature credit includes the fixed downstream readout required to use it. [DeepLIFT](https://proceedings.mlr.press/v70/shrikumar17a.html)

A separate forward intervention gives each feature the same B-minus-A increment in the two contexts. Use the rounded mean increment, keep each pair close to its native midpoint, and enforce integer values in [0,127]. Features that already have equal increments retain both native pairs. Re-evaluate the original dense network and unchanged PSQT branch. Restore selected channels' original four activations and compare with a seeded random ranking. Mixed activations used in interventions can fall outside the set generated by legal chessboards. The real endpoints all come from replayed moves.

## Connections, spectra and controls

Each of the eight heads has 8 H1 and 32 H2 coordinates. Rank its 256 connections by unlabelled mean absolute endpoint secant contribution in January. Transplant selected messages in both directions, recompute the native H2 clipping and readout, and average the two transferred effects. Compare across motifs and against weight magnitude, 12 random equal-size sets and 12 bipartite rewires preserving both endpoint degrees. The degree control preserves which units carry many connections. The feature-input experiment similarly compares subsets of 2,048 slots against 16 random rankings. These are tests of a fixed model's response. They do not measure inference speed.

For every H1 endpoint pair, enumerate all 256 complementary interchanges and compute the exact Boolean Walsh transform. Exchanging the two endpoints complements the mask. Only even orders survive their sum for the paired-turn test and only odd orders survive their difference for move ordering. The empirical measurements are the energy within each surviving order, the score fidelity and the label agreement. For turn comparisons the base term is the cube mean. For alternative moves it is the PSQT margin. Per-position certificates enumerate all possible interchanges outside a retained set of H1 coordinates and count strict choices that always survive. The minimum retained set can vary across observations. [Analysis of Boolean Functions](https://arxiv.org/abs/2105.10386)

For handwritten evaluation, compare the released output, the value before classical grain truncation, and both evaluators at identical 16-unit granularity. Repeat after disabling the two material-dependent lazy evaluation cutoffs in the classical evaluator. Its weights, specialized endgame evaluators and score granularity otherwise follow the release. Compare the stock and forced-full feature increments case by case. A further control fits a regularized linear weighting of the 13 handwritten term pairs and the specialized endgame evaluator on January's reversal labels. Use phase-weighted MG/EG components, root-mean-square scaling on January, no intercept, and a fixed ridge coefficient of 0.05. Apply frozen weights to later months and the WDL task. Fit separate readouts by head on the same January labels for both the handwritten features and the NNUE H2 activations plus PSQT. The held-out comparisons test which information a linear readout can extract from each representation on these tasks. [Stockfish 14.1 evaluator](https://github.com/official-stockfish/Stockfish/blob/sf_14.1/src/evaluate.cpp)

Pálsson and Björnsson probe Stockfish for chess concepts and compare the network with the classical evaluator. Here the chessboard edit specifies when a preference should reverse, and the intervention tests which internal transformation carries that change. Jenner et al. use causal interventions in a Leela policy network to study look-ahead. The measurements here concern static scores in Stockfish. [IJCAI 2023](https://www.ijcai.org/proceedings/2023/541) · [Jenner et al.](https://arxiv.org/abs/2406.00877)

## Reproduction

Use a CPU with AVX2, C++20, Python 3.11+, NumPy, Matplotlib, GNU Make, curl, zstd and Git. From the repository root, prepare the pinned engines, weights, tablebases and placement sets with the commands below. A prepared scratch directory can be reused.

```bash
study=research/nnue-tempo
data=.scratch/nnue-reproduction
bash "$study/setup.sh" "$data"
export PYTHONPATH="$PWD/$data/pychess"
python "$study/collect.py" "$data" --workers 6
python "$study/collect.py" "$data" --extended --count 32000 --workers 6
python "$study/game_pairs.py" "$data"
bash "$study/circuits/reproduce.sh" "$data" "$study/circuits/results"
```

The runner verifies all three archive digests. It writes per-observation positions and native activations to scratch. Each final audit replays both choices in both contexts, recomputes the chess labels independently, checks native quantization and tests a seeded subsample against the untouched engine binary. [Lichess game archives](https://database.lichess.org/)
