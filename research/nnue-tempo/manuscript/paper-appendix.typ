= Preference graphs and quantized margins <sec:graphs>

== When a shared destination score fails
Fix the two kings, material, White's move source, the tied classical score, and the reply signature used by the atlas screen. The vertices of a graph are White's possible destinations. Each strict comparison adds an arc $u -> v$ from lower to higher Black WDL, labeled by the black man's square. Parallel arcs count as distinct comparisons. A shared destination score $f$ supplies the same difference $f(v)-f(u)$ on each arc with those endpoints.

*Proposition.* A shared destination score satisfies every strict choice exactly when the directed comparison graph is acyclic. The minimum number of strict choices it misses equals the minimum feedback arc set size.

*Proof.* Summing $f(v)-f(u)$ over a directed cycle gives zero, which contradicts positive margins on all its arcs. On an acyclic graph, assigning successive integers in topological order gives positive margins on all arcs. The satisfied arcs of any score therefore form an acyclic subgraph. Conversely, a score from the topological order after deletion of a minimum feedback arc set satisfies every remaining arc. The minimum miss count follows.

Among 1,811,642 nonempty graphs, 225 have cycles and no two-cycle. In 220 the shortest cycle has length three and in five it has length four. Their 3,741 comparison records require at least 229 misses by an optimal shared destination score. Native NNUE misses or ties 1,150. To compute the exact optimum, decompose the graph into strongly connected components. Inside a component let $A(S)$ be the maximum satisfied arc count on a vertex subset $S$. Placing $v$ last contributes every arc from $S-{v}$ to $v$, in addition to $A(S-{v})$. Maximize over $v in S$, starting with $A(emptyset)=0$. A topological order of the components satisfies all intercomponent arcs. Every component in the 225-graph panel has at most six vertices. Exhaustive permutations verify the dynamic program. The PDF attachments include all 3,741 arcs with parent FENs, moves, WDL and score margins, and the summaries for all 225 graphs.

For example, fix White Kd7 and Bb4 and Black Ke2. The table gives four legal White choices from b4, with a different black pawn square on each row. Black draws after the first move and wins after the second on each row. Following the four arcs gives $f("a3")<f("d6")<f("c5")<f("a5")<f("a3")$. The pawn square and fixed pieces reconstruct each parent FEN with White to move, no castling or en passant, and counters 0 and 1.

#figure(
  table(
    columns: (1fr, 1.65fr, 1.65fr),
    align: center, stroke: (y: 0.4pt + luma(215), x: none), inset: 3pt,
    table.header([Pawn], [Draw for Black], [Win for Black]),
    [b3], [Ba3], [Bd6],
    [h3], [Bd6], [Bc5],
    [f3], [Bc5], [Ba5],
    [d3], [Ba5], [Ba3],
  ),
  caption: [A four-cycle on one common move source.],
) <tab:cycle>

== Additive recalibration with a bounded residual
A shared potential added to observed margins has a different feasibility test. Let $m_e$ be an observed margin, $eta_e$ its target, and $c_e>0$ a weight. Consider the smallest $rho>=0$ admitting a potential $f$ and edge corrections $r_e$ with $|r_e|<=rho c_e$ and $m_e+f(v)-f(u)+r_e>=eta_e$ on every edge. The exact answer is
$ rho_* = max(0, max_C frac(sum_(e in C)(eta_e-m_e), sum_(e in C)c_e)), $
where $C$ runs over simple directed cycles and an empty maximum is $-infinity$. Summing the constraints proves the lower bound. For the upper bound, set $r_e=rho_*c_e$ and weight each arc by $eta_e-m_e-r_e$. Every cycle has nonpositive total weight. Add a source of zero-weight arcs to all vertices. Maximum walk weight from that source is finite because a closed subwalk can be removed without decreasing its weight. These maximum weights define a potential with $f(v)-f(u)$ at least the weight of each arc. For $r_e=0$ and a common target, this gives the directed minimum-cycle-mean criterion [12]. The corresponding graph-level projection onto potential differences is the standard gradient-space decomposition [6].

== Secant accounting for integer clipping
For the two comparison rows, let $d_j^l$ be the difference between the endpoint activations after stage $l$ of row $j$. Take $d_1^0=d_2^0$ as the matched raw feature difference. For each layer, let $J_j^l$ be its weight matrix left-multiplied by the diagonal of activation secants. The secant is the difference in output bytes divided by the difference in logits, with zero used when the logits coincide. The FT stage uses the identity weight matrix. Then $d_j^l=J_j^l d_j^(l-1)$ exactly, including when the logits cross a rounding or clipping threshold. Write $A^l=(J_1^l+J_2^l)/2$ and
$ g^l = ((J_1^l-A^l)d_1^(l-1)-(J_2^l-A^l)d_2^(l-1))/2. $
The half-difference $q^l=(d_1^l-d_2^l)/2$ obeys $q^l=A^l q^(l-1)+g^l$. Since $q^0=0$, the positional score half-difference before final rounding is
$ w^top (A^3 A^2 g^1 + A^3 g^2 + g^3)/16, $
where $w$ is the final positional weight vector. The direct piece-square difference is constant across the two enemy placements, and the final integer division adds its own measured rounding residual. This derivation allocates a score contrast for the chosen reference slopes. Donor interventions instead recompute the nonlinear layers with edited activations.

= Replication protocol

*Position generation.* Normalize the side with the penultimate-rank pawn as Black. For a synthetic trial, choose distinct king and defending-piece squares, the black pawn's file, and the added white pawn squares, then reject collisions, invalid parents and checks on either endpoint. White pawn ranks are two through six in the ordinary stratum. In the seventh-rank stratum choose from ranks two through seven and require at least one pawn on rank seven. Two legal, quiet, non-promoting moves by the defending piece give the children, each with at least one legal reply. The random condition samples two distinct moves uniformly. The exposure condition takes one random move from each of the surviving-capture and no-surviving-capture sets. Draw 4,000 random and 3,000 exposure pairs per defender and rank stratum before exclusions. The four-piece condition has a single rank stratum.

*Played positions.* Read the six rated standard monthly archives. Screen move texts with at least 27 captures, then replay the legal game. Retain four- or five-piece parents with the same material restrictions, allowing the enemy pawn on any legal rank. Require the played nonpawn move and at least one quiet legal alternative by the same piece. The game and synthetic children use the same exclusion of checks, castling and en passant. Within each game and panel, sample the parent uniformly from eligible moves by reservoir sampling. Sample the alternative uniformly from eligible alternatives from that same parent. In the exposure panels the alternative must change the survival indicator. Reset the root halfmove clock to zero and evaluate theoretical WDL by its sign. Count canonically mirrored pairs once per panel.

*Scores and uncertainty.* White makes the candidate moves and all child scores use the Black perspective. Use zero for an incorrect ordering, one half for a tie, and one for a strict success. A rescue reaches strict success from a tie or error. A harm leaves strict success for a tie or error. Freeze the lowest maximizing weight on the discovery grid for each release. Compute percentage-point changes from the underlying pair scores before rounding. For intervals resample the entire king-layout group for synthetic data and the entire game for played data, using 2,000 draws for pointwise percentile intervals. These intervals describe variation within the sampling design. All releases use the same labeled choices. The search runs reveal only the chosen roots of interest in the paired test and the full root move set in the additional played-position test.

*Matching and score decomposition.* Restrict all comparisons to the same four-piece clear/off-line configuration. Match defender type, the pawn file relative to the White king's horizontal orientation, capture direction, and the lower of the two WDL signs. Coordinates in that orientation give Manhattan distances $d_W,d_B,d_D$ between candidate matches for the two kings and the off-line destination. The primary cutoffs are 4, 4 and 5 squares. Remove identical child pairs. For the signed-margin stage, require equal native signs including zero and $abs(m_s-m_g) <= c$, where $c = max(24,0.25 abs(m_g))$ and $m_g$ is the native game margin. Choose the three eligible synthetic pairs with the smallest $2d_W+2d_B+d_D+4abs(m_s-m_g)/c$, or all if fewer than three qualify. Omit the last term when matching geometry or geometry with sign. Average neighbors equally before averaging games. Keep the primary-stage supported games in all four rows of @tab:match-stages. The tighter limits are 3, 3 and 4 squares with $c=max(16,0.15 abs(m_g))$. The wider limits are 6, 6 and 7 with $c=max(48,0.40 abs(m_g))$. Use 2,000 exponential multiplier draws, independently by game and by synthetic king orbit. Weight each fixed match edge by the product of its two cluster multipliers and its original match weight, then normalize.

For a margin $m$ and response increments $d=(d_H,d_L)$, define
$ F(m,d) = (C(m+d_H)+C(m+d_L))/2 - C(m), $
where $C$ is the ordering credit function. In a geometry-matched pair let $F_(i j)=F(m_i,d_j)$, with synthetic and game values indexed by $s$ and $g$. The two components are
$ M &= (F_(s s)-F_(g s)+F_(s g)-F_(g g))/2, \
  D &= (F_(s s)-F_(s g)+F_(g s)-F_(g g))/2. $
Their sum is the observed gap $F_(s s)-F_(g g)$. The logit replay uses the same four-cell algebra with continuous score shifts in place of ordering gain. The incoming change is $W_1(a_H-a_L)$ on the selected bytes. Apply its negative to the higher-WDL child's logits and its positive to the lower-WDL child's logits for the two one-sided donor edits. The receiver keeps its own baseline logits, quantizers and direct piece-square term.

#colbreak()

*Engine artifacts.* Stockfish 12 uses `nn-82215d0fd0df.nnue` and 14.1 uses `nn-13406b1dcbe0.nnue`. Stockfish 18 uses `nn-c288c895ea92.nnue` and `nn-37f18f62d772.nnue`. The PDF attachments include the expanded experiment's numerical summaries. The research sources supply pinned revisions, scoring code, seeds and build commands. The release evaluator audit compares the Stockfish 18 selector trace to an independently instrumented build on every scored strict test pair. The Stockfish 14.1 donor replay checks integer endpoint scores against the native trace before changing any activation.
