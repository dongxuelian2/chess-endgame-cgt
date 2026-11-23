# Technical notes on preference graphs and quantized margins

The paper uses a projected contrast between two move margins. This note develops the full graph calculation and gives exact conditions for additive recalibration. The graph experiments concern representational capacity given a separate oracle table per graph. The board-level prediction experiment in the paper fixes its correction on discovery data and applies it to single boards.

## Margins on a preference graph

Fix the two kings, material, White's move source, the classical score and a reply signature. Let a vertex be a White destination. Each enemy square that gives a strict WDL preference between two such destinations contributes an arc $u\to v$ from lower to higher Black WDL. Keep parallel arcs when their opposing squares differ. An additive table contributes the potential difference $f(v)-f(u)$ on every arc. Consequently, the sum of any additive margin round a directed cycle vanishes. A topological order provides strictly correct margins whenever the graph is acyclic.

A two-cycle is the familiar reversal in two opposing layouts. The exact census in [preference-graphs.csv](results/preference-graphs.csv) covers 1,811,642 nonempty graphs. There are 225 cyclic graphs without a directed two-cycle, including 220 whose shortest cycle has length three and five whose shortest has length four. One legal example has White Kd7 and Bb4 against Black Ke2. Put the black pawn on b3, h3, f3 and d3 in successive comparisons. The lower-to-higher Black ordering gives a3 to d6 to c5 to a5 to a3. [Legal moves and native scores](results/cycle-witnesses.csv)

The minimum number of strict preferences missed by a shared table is the minimum feedback arc set size with parallel arcs counted separately. The strict successes of every table form an acyclic subgraph. Conversely, topologically ordering the complement of a feedback arc set satisfies every surviving arc. To compute the exact optimum on each strongly connected component, let $A(S)$ be the maximum number of satisfied arcs with endpoints in $S$. Place a destination $v$ last and add its incoming arcs from $S\setminus\{v\}$ to $A(S\setminus\{v\})$. Maximize over $v$, starting with zero on the empty set. A topological order of the components satisfies every intercomponent arc. All components in the 225-graph panel have at most six vertices, so exhaustive permutations independently verify the recurrence. An optimal shared table misses at least 229 of the 3,741 comparison records. Native NNUE misses or ties 1,150. The greater expressive capacity of NNUE permits both gains and losses on these individual graphs.

## Sharp margin after recalibration

Given observed margins $m_e$ and desired margins $\eta_e$, consider an additive potential and comparison-specific corrections of size at most $\rho c_e$, where every cost scale $c_e$ is positive. The smallest uniform budget is

$$
\rho_* = \max\left(0,\max_C\frac{\sum_{e\in C}(\eta_e-m_e)}{\sum_{e\in C} c_e}\right).
$$

Here $C$ ranges over simple directed cycles and the maximum over an empty collection is $-\infty$. Summation around a cycle proves necessity. For sufficiency, assign the weight $\eta_e-m_e-\rho c_e$ to each arc. Add a root with zero-weight arcs to every destination. Cycle inequalities imply every closed walk has nonpositive weight. The maximum walk weight to a vertex is finite and can be attained on a simple walk. These maximum weights give the required potential. With corrections fixed at zero, the greatest common feasible margin is the minimum directed cycle mean of $m$, an instance of the classical cycle-mean problem. Integer observed margins permit integer maximum-walk potentials for any feasible integer target. [Karp (1978)](https://doi.org/10.1016/0012-365X(78)90011-0)

On the 225-graph panel, 28 graphs have every strict order correct under the original score. A separately fitted additive oracle table can make all orders strict in 161 graphs. The two results measure original rankings and available margin under an oracle calibration, respectively. [Cycle and calibration table](results/preference-cycles.csv)

## Which part of the network supplies a contrast?

Let $B$ be the arc incidence matrix, with $(Bf)_e=f(v)-f(u)$. The weighted least squares projection of the margin onto this gradient space leaves a residual

$$
P_Wm=\left[I-B(B^\top WB)^\dagger B^\top W\right]m,
$$

where $W$ is diagonal and positive. The normal equations give $B^\top WP_Wm=0$. All signed closed-walk sums are preserved. Conversely, zero signed closed-walk sums permit integration along paths to recover a potential in each undirected component. This is the graph-level Hodge decomposition. With two parallel arcs of equal weight, the residual has coordinates $((m_1-m_2)/2,(m_2-m_1)/2)$, which is the contrast used in the paper. [Jiang et al. (2011)](https://doi.org/10.1007/s10107-010-0419-x)

The difference of unclipped feature sums at fixed kings has zero residual because the additive enemy-piece term cancels in a move comparison. In a quantized affine layer the endpoint difference equals its incoming linear difference times a scalar secant for each output coordinate. Use zero when the two preactivations coincide. Let $J_e^l$ denote the resulting linear map on arc $e$ in layer $l$ and $A^l$ its mean over the chosen arcs. With $d_e^l=J_e^ld_e^{l-1}$, set $G_C^l=\sum_{e\in C}(J_e^l-A^l)d_e^{l-1}$. Substitution gives the positional circulation

$$
w^\top(A^3A^2 G_C^1+A^3G_C^2+G_C^3),
$$

where $w$ is the final positional weight vector. The raw destination differences vanish around a cycle, so there is no constant-baseline term. The analogous per-arc expression supplies the FT, H1 and H2 stages of the two-row contrast. The partition depends on the frozen reference slopes. Its sum is exact before output rounding. Native donor interventions evaluate a different quantity and recompute all unpatched downstream coordinates.

## Least squared correction

Write $b=\eta-m$. Allow a potential $f$ and a comparison-specific correction $r$ subject to $Bf+r\ge b$. The minimum squared norm $r^\top Wr$ equals

$$
\left\|\Pi_K(W^{1/2}b)\right\|_2^2,\qquad
K=\{z\ge0:B^\top W^{1/2}z=0\}.
$$

To see this, attach a multiplier $\lambda\ge0$ to the inequalities in the minimization of $r^\top Wr/2$. Minimizing over $f$ requires $B^\top\lambda=0$. Minimizing over $r$ gives $r=W^{-1}\lambda$. The dual is $\max(\lambda^\top b-\lambda^\top W^{-1}\lambda/2)$. A strictly feasible $r$ gives strong duality. Completing the square with $z=W^{-1/2}\lambda$ yields the projection. A nonnegative circulation decomposes into simple directed cycles by following its positive arcs and repeatedly subtracting the smallest flow on a completed cycle.

For two opposite WDL preferences with unit score targets, let $m_A,m_B$ have the same destination order. The squared correction is $[2-(m_A-m_B)]_+^2/2$. For a strict-positive anchor and a zero-margin control, replace 2 by 1. Both follow by eliminating the common potential and projecting onto a half-space in two coordinates. The experiments retain this quantity as a comparative diagnostic. A zero-cost correction after adding a free potential gives no claim about original ordering or deployable accuracy. Requiring one globally consistent board-score correction can only increase the optimum.

In the 225 graphs, 65 need positive squared correction at the one-point target. Ten require multiple active directed cycles. In one rook-versus-knight example, two three-cycles share an edge and their required sums are 55 and 106. The Gram system has diagonal entries 3 and off-diagonal entry 1. Its rational multipliers are $59/8$ and $263/8$, giving exact squared energy $31123/8$. The certificate solves both constraints simultaneously. [Rational solutions and native graph margins](results/preference-cycles.csv)
