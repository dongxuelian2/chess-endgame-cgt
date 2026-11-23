#set document(date: datetime(year: 2026, month: 11, day: 23), title: "When Promotion Cues Mislead Stockfish NNUE")
#set page(
  paper: "us-letter",
  columns: 2,
  margin: (left: 0.75in, right: 0.75in, top: 0.78in, bottom: 0.88in),
  header: context if counter(page).get().first() > 1 [
    #set text(size: 8pt)
    #align(right)[When Promotion Cues Mislead Stockfish NNUE]
  ],
  footer: context align(center, text(size: 8.5pt, fill: black, counter(page).display())),
)
#set columns(gutter: 0.33in)
#set text(font: "Libertinus Serif", size: 9pt, fill: black, lang: "en")
#set par(leading: 0.51em, spacing: 0.44em, first-line-indent: 0.8em, justify: true)
#set place(clearance: 10pt)
#show raw.where(block: false): set text(font: "Liberation Mono", size: 7.8pt)
#show link: set text(fill: black)
#set list(indent: 0.8em, body-indent: 0.45em, spacing: 0.38em)
#set enum(indent: 0.8em, body-indent: 0.45em, spacing: 0.38em)

#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(
  above: 10pt,
  below: 4pt,
  sticky: true,
  {
    set par(first-line-indent: 0pt, justify: false)
    set text(font: "Libertinus Serif", size: 10.5pt, weight: "bold", fill: black)
    if it.numbering != none {
      counter(heading).display(it.numbering)
      h(0.65em)
    }
    it.body
  },
)
#show heading.where(level: 2): it => block(
  above: 8pt,
  below: 3pt,
  sticky: true,
  {
    set par(first-line-indent: 0pt, justify: false)
    set text(font: "Libertinus Serif", size: 9.6pt, weight: "bold", fill: black)
    if it.numbering != none {
      counter(heading).display(it.numbering)
      h(0.55em)
    }
    it.body
  },
)

#show figure.where(kind: table): set figure.caption(position: top)
#set figure(gap: 5.5pt)
#show figure.caption: it => {
  set text(size: 8.4pt, fill: black)
  set par(leading: 0.44em, first-line-indent: 0pt, justify: true)
  align(left)[*#it.supplement #context it.counter.display(it.numbering):* #it.body]
}

#let booktabs(cols, aligns, headers, rows, size: 8pt, midrules: (), ypad: 2.7pt) = {
  set text(size: size, fill: black)
  set par(leading: 0.40em, first-line-indent: 0pt, justify: false)
  let body = ()
  for (i, row) in rows.enumerate() {
    if i in midrules {
      body.push(table.hline(stroke: 0.35pt + luma(150)))
    }
    body += row
  }
  table(
    columns: cols,
    align: aligns,
    stroke: none,
    inset: (x: 5pt, y: ypad),
    table.hline(stroke: 0.9pt + black),
    table.header(..headers.map(h => text(weight: "bold", h))),
    table.hline(stroke: 0.5pt + black),
    ..body,
    table.hline(stroke: 0.9pt + black),
  )
}

#let refitem(n, body) = block(above: 0pt, below: 2.8pt)[
  #set par(first-line-indent: 0pt, hanging-indent: 2.05em, leading: 0.41em, justify: true)
  #box(width: 2.05em)[[#n]]#body
]

#place(top + center, scope: "parent", float: true, clearance: 12pt)[
  #set par(first-line-indent: 0pt, justify: false)
  #set align(center)
  #block(width: 100%)[
    #text(font: "Libertinus Serif", size: 19pt, weight: "bold")[
      #set par(leading: 0.31em)
      When Promotion Cues Mislead Stockfish NNUE
    ]
    #v(7pt)
    #text(size: 9.5pt)[Anonymous research manuscript]
  ]
]

#heading(numbering: none, outlined: false)[Abstract]
#par(first-line-indent: 0pt)[
A chess evaluator can recognize a defended promotion square and still favor a move that allows promotion by capture. Using legal move pairs and exact endgame outcomes, we trace this conflict in Stockfish 14.1's efficiently updatable neural evaluator (NNUE) and test it across three releases. Equalizing four input channels improves synthetic comparisons and has near-zero average effect in two played samples. Matching defender, geometry and native score margin leaves a gap. The four channels differ less between the played moves and their alternatives. A frozen capture indicator improves ordering by 4.84 percentage points on the original four-piece validation set and by 0.77 on new classically tied played pairs. Tests up to six pieces identify competing promotions that reverse the indicator's implication. Native shallow search resolves most four-piece errors. Rook and knight controls distinguish a broadly used readout from a more tactic-specific path.
]

#heading(numbering: none, outlined: false)[Keywords]
#par(first-line-indent: 0pt)[Chess endgames, static evaluation, NNUE, interpretability, Syzygy tablebases]

= Introduction
In @fig:opening, White chooses between Qb6 and Qb7. The second move pins the f3 pawn and wins, while the first permits the drawing ...f2. Stockfish 14.1 uses a special hand-written rule for queen against pawn. With both kings and the pawn fixed, that rule omits the queen's square and gives both moves −2494 from Black's perspective. The static neural evaluator gives −17 after Qb6 and −889 after Qb7. The same network can also make the opposite kind of mistake. With a pawn on g2 and a white queen on e1, Qf1 attacks g1, yet lets Black play ...gxf1=Q mate. The evaluator prefers Qf1 to a draw.

#figure(
  placement: auto,
  caption: [Two moves from the same parent. Qb6 permits ...f2 and draws. Qb7 pins the f3 pawn to Kg2 and wins. The classical evaluator gives both children the same value.],
  grid(
    columns: (1fr, 1fr),
    column-gutter: 8pt,
    align: center,
    [#image("board-b6.svg", width: 100%, alt: "After Qb6 the black pawn on f3 can advance") #linebreak() #text(size: 8.5pt)[*(a) Qb6* · Draw]],
    [#image("board-b7.svg", width: 100%, alt: "After Qb7 the queen pins the pawn on f3 against the black king") #linebreak() #text(size: 8.5pt)[*(b) Qb7* · White wins]],
  ),
) <fig:opening>

The same geometric cue can improve or degrade an evaluator's ranking when a different legal continuation becomes decisive. We test that proposition using two interventions on the same choices. Deleting a class of chess replies measures its role in the exact outcome. Replacing activations measures their role in the score. A single-board prediction then tests whether the chess interpretation carries across materials, evaluators and search depths. Rook skewers and knight forks serve as comparisons for the specificity of an internal path.

= Related work
Pálsson and Björnsson [1] showed that probes can recover human chess concepts from Stockfish 14.1. McGrath et al. [2] traced chess knowledge during AlphaZero training, and Schut et al. [8] optimized for concepts useful in play. Jenner et al. [3] used interventions to locate information about future moves in Leela's transformer. We start with exact preference labels for two moves from one parent and ask which legal continuation and which part of a static scalar evaluator account for the preference. Classic work on chess search also distinguishes the treatment of tactical continuations from the evaluation of a quiet endpoint [9]. Our search comparison measures that distinction experimentally.

Patching results depend on the donor, metric and target component [10, 11]. We use legal sibling donors in both directions, match control inputs by activation change and check the legal action independently. The results concern the modified computation under those donors. The board-only indicators and exact outcomes provide separate tests wherever activation edits would combine coordinates from two distinct states.

= Positions and tests
*Comparisons.* The four-piece atlas contains both kings and one additional pawn, knight, bishop, rook or queen per side, giving 25 ordered material classes. Black moves in each child. The boards have no check, castling or en passant rights, and use a fresh clock. We collapse Syzygy's five-valued win/draw/loss (WDL) result to three signs [5]. The positive sign favors Black and includes wins classified as cursed under the move clock. Each move pair comes from one legal White parent. Its two children tie in the full classical score before rounding to a multiple of 16 and with early score cutoffs disabled and specialized evaluation rules retained. We give one point for the correct strict ranking, half a point for a tied score, and zero for the reverse. We call the mean *ordering credit*.

*What the classical tie costs.* The atlas contains 214,063,938 boards, or 107,031,969 horizontal-reflection representatives. On 342,175,835 within-context pairs tied by the classical score but with distinct WDL signs, the native neural score earns 69.65%. The context fixes material, both kings and the black man's square. This broad count permits pairs without a common legal White source. Our motif tests require a common source. There are 1,313,752 four-board configurations in which relocating the black man reverses the true ranking of two legal White moves and the classical evaluator ties both rows. A score of the form $f(x)+g(y)$ has a move margin independent of the opposing square $y$. Its order therefore cannot strictly follow both directions. Native NNUE follows both directions in 663,639 configurations. The learned direct piece-square branch follows both in zero. This finite comparison identifies interactions a destination table cannot express.

*Chess and network interventions.* A chess deletion removes all Black replies in a specified class. We consult the tablebase after each remaining reply and take Black's best outcome. If no reply remains, we use the position's ordinary tablebase outcome. We call a move class decisive for the comparison when deletion erases its strict order. A network intervention substitutes selected activations from the sibling child and recomputes downstream layers. We average both donor directions. The hybrid activation vector can be outside the range reached by legal boards. A board-only correction evaluates each child independently.

*Splits and comparisons.* Promotion channels and correction weights are chosen on discovery layouts. The validation layouts use disjoint orbits of king placements under board symmetries and king interchange. A king-layout orbit groups positions related by rotating or reflecting the board and exchanging the king locations. The original donor, geometry and skewer analyses are retrospective uses of the same atlas, including its validation partition. A separately sampled pair on a known atlas layout checks robustness within the atlas. The additional calendar sample and the new material strata test transfer under a different sampling process.


== How a position change reaches the score
The input to a feature coordinate is a bias plus piece-square weights in a bucket indexed by the relevant king. Write its raw value as $R=c+u(x)+v(y)$, where $x$ is the moving white man's destination and $y$ the black man's square. At fixed kings, moving the white man changes the raw sum by $u(i)-u(j)$, regardless of $y$. The network then clips the sum. The same raw change gives different byte differences when the two endpoints cross different clipping thresholds. In the hidden layers a division by 64 with downward integer rounding precedes clipping. Thus a board edit can change sensitivity to a move even when the change it produces in the raw accumulator is identical at both destinations.

To measure the resulting interaction, define $D(y)=S(i,y)-S(j,y)$ with the same destination order on both boards. Comparing the two enemy placements gives
$ q = (D(y)-D(z))/2. $
A shared preference for $i$ over $j$ cancels in this contrast. We use signed score units for this contrast and ordering credit for the ultimate decision. A component can affect the contrast and leave the final decision unchanged if other components supply a larger margin. For a second-layer unit with output weight $w$, its contribution on one row is exactly $w(h_i-h_j)/16$ in pre-rounding score units. Subtracting its contributions across the two rows tests the portion sensitive to the black man's location.

With three or more destinations, even comparisons without any two-move reversal can defeat a shared destination table. @sec:graphs proves the exact criterion and gives a four-move example.

= Promotion channels and capture errors
== A straight-square signal
Stockfish 14.1 adds king-relative piece-square weights. Its feature transformer (FT) clips 1,024 coordinates per king view. We label the next two hidden layers H1 (eight units) and H2 (32 units), using zero-based unit indices. For example, FT625 denotes coordinate 625 of the concatenated transformer output. A scalar positional score joins a learned piece-square score at the output. We use the released four-piece head. For a third-rank pawn escorted by the black king, matching both king locations, both white moves and both outcomes gives 2,830 rook-to-queen comparisons from 23 king–king–pawn layouts. The rook routes most of the ordering through first-layer unit 2. The queen routes it through units 6 and 7. The matched sample is concentrated on the third rank. In a single-material test of 2,586 rook escort layouts across ranks 3–7, the full score earns 77.88%, and retaining only the differences at unit 2 in the hidden path earns 76.28%.

A matched rook-to-queen comparison illustrates where the routes separate. With White Kd4 and Re6 against Black Kh1 and Pf3, moving the rook to e7 or f6 changes first-layer unit 2 from 0 to 30. Units 6 and 7 stay at zero. With a queen at the same source and destinations, unit 2 stays at zero, unit 6 changes from 0 to 31 and unit 7 from 126 to 67. The difference starts in the black king's view. On the original 562 distinct child geometries, FT625 is positive for the rook and zero for the queen. Its weights into units 2, 6 and 7 are +39, −29 and −41. FT961, by comparison, is constant between the two destinations but moves from 0 with the rook to 127 with the queen. Together with FT333, these three coordinates account for a measurable switch. Transplanting them from the queen into the rook lowers the accuracy obtained from unit 2 alone from 91.24% to 49.81%. This test keeps the same choices and isolates an input-driven change in which units respond.

Advancing the seed pawn from the third rank to the second while keeping the other pieces and both candidate moves fixed changes the same rook's first-layer response. The mean absolute difference at unit 7 rises from 0.17 to 104.64 bytes on 2,487 legal paired edits. One input, FT1424, helps explain the change. With a c2 pawn in the white king's orientation, it averages 100.69 when the defending rook is off the line to c1, 2.14 when the rook has a clear line to c1 after the push and 123.49 with a king blocking the rook. Moving the white king one square to open the rook line reduced the activation in all 641 eligible edits. This ties a measured feature to a specific change on the board.

Related signals extend across the eight files. The first white-view screen found five channels with a stable pawn-square maximum and a large clearance response in every tested material. The missing files include a switch of viewpoint. The b and g files in White's orientation can map to the same b-file input in Black's orientation, where FT273 has a stable pawn-square maximum. Relocating Black's king across the d/e boundary holds its weight bucket fixed and mirrors the inputs. FT273 falls from mean 110.20 to zero on 6,454 eligible b-file relocations with the rook off the line, and from 100.08 to zero on 5,583 equivalent g-file relocations. The h-file response is spread over several coordinates. We use four discovery-ranked coordinates per oriented file for the intervention below.

#figure(
  placement: bottom,
  scope: "parent",
  caption: [*Reading the straight promotion square.* (a) FT1424 in legal rook-versus-pawn placements with the pawn on White-oriented c2. A clear rook line to c1 suppresses the feature. A king on the line restores it. (b) Ordering-credit loss from equalizing the leading channel, the four leading channels, or four activity-matched control channels. Discovery fixes the channels. The validation total is 13,523 promotion-dependent comparisons (1,497–1,819 per file). Points give means and bars give 95% king-orbit bootstrap intervals.],
  image("paper-promotion.svg", width: 100%, alt: "A rook with a clear promotion line suppresses FT1424. Removing four leading channels has a larger effect than removing matched controls across all eight files."),
) <fig:promotion-inputs>


The five-file screen required the same unique maximum pawn-square weight across all 32 White-king buckets and a large off-line versus clear-line activation contrast for each of four defending pieces. On b, FT1869 meets the weight criterion but its contrast is just 13.4 bytes for a queen, compared with 42.8–72.4 for the rook, bishop and knight. The best g and h White-view candidates meet the bucket-maximum criterion in 29 and 31 of 32 buckets. These facts explain why the single-view screen missed the three files. The held-out intervention gives a more direct measure of their importance. Removing four channels lowers ordering credit on every file by 10.79–28.33 points, with little change under the activity-matched control (@fig:promotion-inputs). On 1,314 played-position choices where promotion decides the WDL order, the same channel differences account for a 12.44-point loss. Both pieces of evidence establish circuit use without requiring one channel per file.

== What happens when the pawn can capture
A white piece guarding the straight promotion square can itself be captured on a diagonal promotion square. In the opening example of this failure, place White Ka1 and Qe1 against Black Kb3 and Pg2. Qf1 guards g1 and permits ...gxf1=Q mate. Qe4 draws. The Black-side scores are −2403 and +77. With the pawn moved to d2, Qf1 wins for White and Qe4 draws. In that layout the Black-side scores are −304 and +119.

The two-row contrast answers a question a neuron probe alone leaves open. Use the same destination order in both rows and call the margins $m_g$ and $m_d$. A fixed preference for f1 over e4 changes both by the same amount. Their half-difference is unchanged. Here it is $(-2480-(-423))/2=-1028.5$ score units, even though the g2 row should favor f1 more than the d2 row. The negative sign measures an error that depends on the pawn location. The unclipped feature sums have zero residual, because the pawn term cancels when two destinations are subtracted at fixed kings. Restoring FT clipping and then the two hidden quantizers contributes −60.86, −1215.32 and +247.47 in an exact secant accounting with fixed reference slopes. On g2, two implicated bytes, FT273 and FT1282, fall from 127 and 127 on Qe4 to 56 and zero on Qf1. They are zero on both d2 endpoints. The direct output contributes zero to the contrast and rounding accounts for the remaining +0.22.

On the original validation panel for penultimate-rank pawns, 1,593 choices have a surviving capture option on the higher Black outcome only. The native score earns 49.72%. When the higher-WDL destination controls the straight promotion square and the lower-WDL destination lies off its line, the accuracy is 24.62% on 731 pairs. Removing the four chosen channel differences raises this to 38.85%. The other 862 choices in the same capture-option population move in the opposite direction, from 71.00% to 59.66%. The pooled intervention appears almost neutral, 49.72% to 50.11%. The two geometric slices show why. The channels favor a useful cue on some boards and a misleading one on others. Even in this targeted subgroup, most preferences remain incorrect after the intervention. The four coordinates explain one part of the error, conditional on the sibling donor.


A second grouping tests whether this reversal also appears when we classify the decisive action by move deletion. In @fig:capture, panel (a), straight-promotion cases lose 23.30 points after channel equalization. The mean change is only +0.63 points when we pool all capture-promotion cases. Inside that pool, clear/off-line cases improve by 12.60 points and clear/clear cases lose 13.23 points. The sign therefore follows the relation of the defending piece to the straight square. The result also shows why the small pooled effect is an unreliable description of feature use.

== A board-only test
The preceding donor experiment uses both candidate boards. To test the explanation without a donor or a tablebase score, we add a term to the native score when Black has a legal capturing promotion to a queen or rook that survives immediate capture and avoids stalemate. In a four-piece position with a single white man besides the king, that action removes White's last nonking piece and reaches a winning rook or queen ending. This is a local sufficient condition for a Black win in this material domain. We checked all 74,195 horizontal-reflection representatives satisfying the condition in the adjacent-capture geometry. Every board has positive Syzygy WDL. In particular, counting repairs on pairs selected for an indicator that already certifies the winning endpoint measures the size of a known ranking error. The coverage question requires the entire eligible sample. Simply finding a legal capture-promotion also finds 3,971 draws in that geometry.

We fit a single nonnegative weight on 18,427 discovery pairs by maximizing ordering credit over a grid from 0 to 4,096 in steps of 16. The fitted weight is 2,800 native units. We apply it unchanged to each board of every test pair, including those with no promotion-dependent outcome and those from played games. WDL determines test credit. The scoring rule reads only the board and legal continuations up to the immediate recapture.

#figure(
  placement: auto,
  caption: [Ordering credit after applying the frozen capture term. The last row applies the reply filter to the same seventh-rank stress pairs.],
  booktabs(
    (1.48fr, .72fr, .72fr, .72fr),
    (left, right, right, right),
    ([Cohort], [Pairs], [Native], [Adjusted]),
    (
      ([Four-piece validation], [16,518], [91.08%], [95.92%]),
      ([Played Apr–Sep], [870], [97.18%], [98.16%]),
      ([Played Oct–Dec], [809], [97.59%], [98.58%]),
      ([Added white pawn], [18,129], [95.84%], [97.29%]),
      ([Pawn on seventh rank], [3,589], [56.03%], [92.39%]),
      ([Stress with reply filter], [3,589], [56.03%], [72.19%]),
    ),
  ),
) <tab:correction>

On all 16,518 four-piece validation pairs, the surviving-capture rule improves credit by 4.84 percentage points (95% interval 4.51–5.19), with 805 strict rescues and zero harms. The any-legal-capture rule raises credit to 96.30%, with 906 rescues and 38 harms. It gives the larger net gain at the cost of overriding some correct rankings. Both rules use weights chosen on discovery. In the earlier two calendar cohorts, only 75 of 15,000 examined pairs distinguish the surviving-capture indicator. @tab:correction reports the aggregate results on the informative, classically tied pairs in each cohort.

#figure(
  placement: bottom,
  scope: "parent",
  caption: [*When promotion channels help and when they hurt.* (a) Each group is defined by deleting straight or capturing promotions in the chess tree. For the capture-only subsets, the labels give the defender's access to the straight promotion square on the higher- and lower-WDL child for Black, in that order. Positive values mean that removing the channels improves ordering. Bars give 95% king-orbit bootstrap intervals. (b) Native ordering credit and ordering credit after adding the fixed board-only capture term. ‘Clear / off-line’ conditions on a surviving capture being available only on the higher-WDL child. Panel (b) gives point estimates. The panels use different inclusion criteria.],
  image("paper-capture.svg", width: 100%, alt: "Removing the channels harms straight-promotion decisions but helps the clear versus off-line capture regime. The fixed correction restores ordering on the difficult legal-capture subset."),
) <fig:capture>


== What changes when a defender can promote
The four-piece rule rests on the capture leaving White with only a king. To test that assumption we kept the two original White choices and sampled up to two legal locations for an added white pawn before consulting the tablebase. White's moving man was a knight, bishop or rook. Of 40,668 valid child observations, 1,188 offered a surviving capture. Their outcomes included 136 draws and one Black loss. The frozen weight improved ordering on the 18,129 pairs with unequal WDL and restored 262 strict preferences with zero harms in this sample. The 803 pairs also tied by the instrumented hand evaluator moved from 82.32% to 90.78%. The single-board counterexamples already show that the winning certificate is specific to four pieces.

We then required a white pawn on its seventh rank and generated fresh legal common-source alternatives on held-out king layouts. Each pair offers the Black promotion option on exactly one child. The construction selects the two boards before checking WDL. Of 7,200 such pairs, 3,589 have different WDL signs. The option points to the worse Black outcome on 273 of these. Credit rises from 56.03% to 92.39% over the complete enriched sample, with 1,547 rescues and 241 harms. The pooled gain coexists with a systematic failure where the capture points to the worse outcome. For example, with White Kg7, Nf2 and Pe7 against Black Ka5 and Pg2, Nh1 allows the surviving ...gxh1=Q, but e8=Q holds the draw. Nd4 instead leaves Black a win. The native score ranks them correctly and the fixed correction reverses them. The added pawn supplies a reply that surviving the initial recapture does not address. A follow-up on the same stress sample adds the weight only when at least one surviving black promotion leaves White with no immediate promotion. This preserves all 241 native successes overturned by the original term and retains 580 of its 1,547 rescues. Overall credit is 72.19%, compared with 56.03% native and 92.39% after the unrestricted term. The reply filter trades coverage for fewer harmful interventions in this particular stress sample.

#figure(
  placement: bottom,
  scope: "parent",
  caption: [*Rescues and harms after adding a pawn.* The bars count strict ordering rescues and harms as shares of each cohort, with raw counts printed at the ends. A rescue turns an error or tie into a strict success. A harm turns a strict success into an error or tie. The final two rows use the same seventh-rank stress pairs. The reply filter trades some rescues for fewer harms in this sample.],
  image("paper-transfer.svg", width: 100%, alt: "The frozen correction rescues many rankings in the seventh-rank stress cohort and harms 241. A reply filter preserves those rankings while rescuing only 580."),
) <fig:transfer>


#import "paper-expansion.typ": transfer-section
#transfer-section(booktabs)

= How specific is an internal path?
== A shared readout in rook endings
A rook skewer checks the king along the line to a white man behind it. In 3,029 matched comparisons, the bishop and queen versions have the same outcome difference and deleting the skewer removes that difference in both. Equalizing second-layer unit 2 reduces bishop ordering credit from 70.86% to 50.21%. Queen credit moves from 69.10% to 69.02%. Holding the geometry and outcomes fixed gives a direct test of the readout.

In an example with White Kc7 and Bd4 against Black Kb1 and Ra2, Bg7 loses to ...Ra7+ while Bc5 draws. Both moves have the same outcomes with Qd4. At g7, switching to the queen changes FT1691, FT577 and FT961 from $(127,18,0)$ to $(0,43,109)$. The first-layer preactivations at units 1 and 3 move from $(-5138,-3517)$ to $(11631,44181)$. Their bytes switch from zero to 127. These units enter second-layer unit 2 with weights −34 and −17. Swapping only their values subtracts 6477 from each bishop preactivation in the example. The bishop's unit-2 bytes of 80 and 92 both become zero. The reverse transplant changes the queen from $(0,0)$ to $(83,98)$. A material-dependent shift of the baseline can therefore silence a move difference without changing the chess tactic.

#figure(
  placement: bottom,
  scope: "parent",
  caption: [*A material-dependent clipping floor.* The same 3,029 position pairs are evaluated with a bishop and a queen. (a) Distribution of unit-2 preactivations in byte units before clipping. The left shaded region lies below zero. Vertical lines mark 0 and 127. (b) Fractions of move pairs with an exactly zero difference, a nonzero difference that leaves both tested donor orders unchanged, or a changed order.],
  image("paper-skewer.svg", width: 100%, alt: "The queen has more preactivations below the clipping floor and far fewer ordering changes when unit 2 is equalized than the bishop."),
) <fig:skewer>


Across the original matched layouts, 69.2% of the queen pairs have exactly zero difference at H2 unit 2, compared with 23.7% of the bishop pairs (@fig:skewer). In another 919 queen pairs the difference is nonzero but equalizing it leaves both donor orderings intact. This separates two explanations for the small observed intervention effect on the queen. A clipped difference has no route to the output. An active difference can also be too small relative to the rest of the score. Median native absolute margins are 107 for the queen and 11 for the bishop in this matched panel. The controls below examine whether the same suppression persists for small margins.

The gate explains when unit 2 influences a decision. The control boards test its tactical specificity. Move the attacking rook from a2 to g5 in the example. Both moves now lose to direct captures. Subtract the margin on this equal-WDL control from the skewer margin using the same two destinations, as in the promotion contrast. Across 2,576 matched skewer/control layouts, unit 2 supplies +32.14 units to the bishop's skewer margin. Subtracting the direct-capture control gives a mean difference of −7.05. Removing it lowers bishop ordering credit on the skewer row, yet increases the fraction of graphs where a shared table can meet the skewer/neutral target, from 31.3% to 39.1%. In 2,556 of the control rectangles, both destinations allow a direct capture. The reference layout here is a different tactical threat with equal WDL between the two moves. The control tests whether the skewer adds a signal beyond the response to an immediate capture.

== Rook action controls within the atlas
We sampled 18,000 same-source, classically tied comparisons per defending material on held-out king layouts, for 72,000 in total. Unlike the original quiet panels, these permit immediate captures. We independently deleted moves that capture the white nonking piece and moves that skewer it. Each tactical series in @fig:readout isolates cases where one deletion removes the order and the other preserves it. Quiet neither cases lack a direct capture or skewer at both endpoints.

#figure(
  placement: bottom,
  scope: "parent",
  caption: [*Unit 2 across material and decisive actions.* Loss of ordering credit when equalizing the H2 unit-2 difference in both donor directions. The direct-capture and skewer series require the corresponding chess deletion to erase the ordering while the other preserves it. Quiet controls have neither action available. Bars give 95% king-orbit bootstrap intervals. The sample starts from 18,000 comparisons per defender. The small-margin panel retains native absolute margins of 1–31 score units.],
  image("paper-rook-controls.svg", width: 100%, alt: "Readout loss is substantial for bishop and knight skewers and near zero for queen skewers, including when absolute native margins are small."),
) <fig:readout>


The action controls test the readout across materials. In bishop endings, unit 2 affects skewer, immediate capture and quiet controls. In queen endings its mean effect stays near zero even after restricting to small native margins. In that subset the two queen endpoints are both clipped to zero on 70.1% of skewer comparisons, versus 0.3% with a bishop. Clipping therefore persists at small margins, where a competing large native margin is excluded by design. The clipping mechanism explains why a material substitution changes the influence of this general readout. Quiet comparisons substantiate its broader use.

== A narrower path for knight forks
The knight-fork experiment tests whether an input-to-hidden path can localize a single legal tactic. We tested a black knight's checking fork of the white king and rook. For each square $u$ let $N(u)$ be the destinations of a knight jump. If the white king is on $w$, the white man on $x$ and the black knight on $y$, a possible fork destination lies in $N(w) inter N(x) inter N(y)$. This is a geometric screen. The chess test separately requires a legal checking jump and no immediate capture of the knight. A comparison is retained when deleting the qualifying forks erases its strict WDL order.

In the original 2,051 matched rook/queen fork comparisons, retaining the three selected H1 differences gives 72.17% for the rook, close to 72.21% natively. The matched intervention row in @fig:fork uses the 202 comparisons retained by the alternate-pair, validation-king and legal-fork screens. These denominators describe distinct panels. We ranked input coordinates by the signed margin they contribute through those three units on separate discovery layouts. Seven of the leading eight are in the white king's view. To check what these coordinates respond to, we hold the king bucket fixed. Over the squares from which a knight can check the king, we compare the weights on knight locations that can reach each checking square and the weights on rook locations that can be attacked from it. Six of the seven inputs have mean correlations in the 96.1st–99.8th percentiles of the White-view features (@fig:fork, panel (a)). This weight pattern suggests a preference for the same possible fork square in both pieces' placements.

We tested that interpretation with 18,181 legal four-board configurations. Both nonking pieces are two knight jumps from the white king at every endpoint. Only the diagonal boards share a reachable checking fork square. In this construction, simple proximity to the king is matched across the four boards. The raw accumulator mixed difference is zero on every configuration. After clipping, all eight selected coordinates contribute positive mean activation differences. The mean mixed difference of the final score is +83.66 native units (95% king-orbit interval +75.02 to +92.68). This is a test of the geometric association. The next comparison checks whether it matters to decisions that require the legal tactic.

On 972 held-out rook-fork choices outside the matched panel, removing the selected input differences only on edges into H1 units 0, 1 and 4 lowers ordering credit from 75.15% to 66.77%. Removing their edges into the five remaining units leaves mean credit at 75.15%. Activity-matched controls on the three nominated units cost 0.28 points, compared with 8.38 for the selected inputs. The same edit on 34,180 no-fork rook comparisons costs 0.20 points. With a queen as the target of the fork, the loss is 4.99 points through units 0, 1 and 4 and 2.20 points through the complementary edges. This test gives a narrow tactical interpretation for the rook path, with weaker concentration under the material substitution. The counts and effect sizes are more informative about specificity than their ratio, since the no-fork effect is close to zero.

#figure(
  placement: bottom,
  scope: "parent",
  caption: [*A more specific route for knight forks.* (a) Empirical cumulative distribution of the weight correlation across the White-view features. The ticks mark the seven selected White-view inputs. The dashed horizontal line marks the 95th percentile. (b) The matched rook row uses 202 validation comparisons. Ordering-credit loss follows removal of the differences in eight inputs only on their connections to first-layer units 0, 1 and 4, on the remaining first-layer connections, or at activity-matched controls going into 0, 1 and 4. Bars give 95% king-orbit bootstrap intervals.],
  image("paper-fork.svg", width: 100%, alt: "Selected knight-fork features tend to have high cross-piece weight correlations. The nominated first-layer route matters most on fork comparisons with a rook."),
) <fig:fork>


= Scope
In Stockfish 14.1 the channel intervention shows one way a useful cue becomes misleading. The rook controls show why clipping alone gives limited evidence of tactical specificity. The fork study gives stronger evidence for a conditional path because the effect concentrates on selected edges and shrinks on both no-fork and activity-matched controls.

The channel identities and readout explanations apply to the released 14.1 weights. The additional releases test transfer of the decision-level phenomenon. Stockfish 18's threat ablation alters an entire input group. Its effect can therefore combine several changes in the representation. The synthetic samples describe the specified material families and sampling rules. The played comparisons measure alternatives to an observed move in rated online play. Search tests disable tablebase access. Restricted pairs measure the change in ordering from static to search scores. Full-root search measures WDL-preserving choices on four-piece played positions.

#include "paper-appendix.typ"

#heading(numbering: none, outlined: false)[References]
#text(size: 8pt)[
  #refitem(1)[Pálsson, A. and Björnsson, Y. (2023). #link("https://doi.org/10.24963/ijcai.2023/541")[Unveiling Concepts Learned by a World-Class Chess-Playing Agent]. *IJCAI*, 4864–4872.]
  #refitem(2)[McGrath, T., Kapishnikov, A., Tomašev, N., Pearce, A., Wattenberg, M., Hassabis, D., Kim, B., Paquet, U., and Kramnik, V. (2022). #link("https://doi.org/10.1073/pnas.2206625119")[Acquisition of Chess Knowledge in AlphaZero]. *PNAS*, 119(47).]
  #refitem(3)[Jenner, E., Kapur, S., Georgiev, V., Allen, C., Emmons, S., and Russell, S. (2024). #link("https://papers.nips.cc/paper_files/paper/2024/hash/37d9f19150fce07bced2a81fc87d47a6-Abstract-Conference.html")[Evidence of Learned Look-Ahead in a Chess-Playing Neural Network]. *NeurIPS*.]
  #refitem(4)[Stockfish developers. #link("https://github.com/official-stockfish/Stockfish/tree/sf_14.1/src")[Stockfish 14.1 evaluation and NNUE source], network `nn-13406b1dcbe0.nnue`.]
  #refitem(5)[de Man, R. #link("https://github.com/syzygy1/tb")[Syzygy endgame tablebase generator].]
  #refitem(6)[Jiang, X., Lim, L.-H., Yao, Y., and Ye, Y. (2011). #link("https://doi.org/10.1007/s10107-010-0419-x")[Statistical Ranking and Combinatorial Hodge Theory]. *Mathematical Programming*, 127, 203–244.]
  #refitem(7)[Lichess Open Database. #link("https://database.lichess.org/")[Rated standard games, 2013–2014].]
  #refitem(8)[Schut, L., et al. (2025). #link("https://pmc.ncbi.nlm.nih.gov/articles/PMC12002201/")[Bridging the human–AI knowledge gap through concept discovery and transfer in AlphaZero]. *PNAS*, 122.]
  #refitem(9)[Marsland, T. A. (1986). #link("https://doi.org/10.3233/ICG-1986-9102")[A Review of Game-Tree Pruning]. *ICCA Journal*, 9(1), 3–19.]
  #refitem(10)[Zhang, F. and Nanda, N. (2024). #link("https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html")[Towards Best Practices of Activation Patching in Language Models]. *ICLR*.]
  #refitem(11)[Heimersheim, S. and Nanda, N. (2024). #link("https://arxiv.org/abs/2404.15255")[How to use and interpret activation patching]. *arXiv:2404.15255*.]
  #refitem(12)[Karp, R. M. (1978). #link("https://doi.org/10.1016/0012-365X(78)90011-0")[A characterization of the minimum cycle mean in a digraph]. *Discrete Mathematics*, 23(3), 309–311.]
  #refitem(13)[Stockfish developers. #link("https://github.com/official-stockfish/Stockfish/tree/sf_12/src")[Stockfish 12 source] and #link("https://github.com/official-stockfish/Stockfish/tree/sf_18/src")[Stockfish 18 source and evaluation].]

]

#for name in (
  "cycle-arcs",
  "preference-cycles",
  "preference-graphs",
  "expansion-flow",
  "expansion-static",
  "expansion-donors",
  "expansion-threats",
  "expansion-sf18-selector",
  "expansion-search",
  "expansion-root",
  "promotion-match-flow",
  "promotion-match-strata",
  "promotion-matched",
  "promotion-match-swap",
  "promotion-match-path",
  "promotion-match-played-choice",
  "promotion-match-history",
) {
  let filename = name + ".csv"
  pdf.attach(filename, read("results/" + filename, encoding: none),
    relationship: "data", mime-type: "text/csv")
}
