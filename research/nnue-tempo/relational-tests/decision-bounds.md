# Decision certificates for a fixed minimax game

## Model

The root maximizes and has at least two actions. Under each action the opponent minimizes over a finite nonempty family of reply rows. A row is a finite nonempty set of leaf state identifiers over which the root player maximizes. A transposition carries one identifier throughout the game. Thus the value of action $a$ is

$$V_a(x)=\min_{R\in a}\max_{i\in R} x_i.$$

Fix an incumbent $A$ of maximum value at the baseline $h$. Each state $i$ has two allowed endpoint scores $h_i,l_i$. An intervention selects a set $S$ and uses its second endpoint everywhere that state occurs. The event is strict displacement, $V_b(x)>V_A(x)$ for some rival $b$. A tie retains the incumbent in the best-action set. The cost is the number of distinct edited states, with optimum infinity when displacement is impossible. Early terminals have equal endpoints.

## Lowering scores at depth three

Assume $l_i\le h_i$. For each reply row of the incumbent define

$$e_R=\max_{b\ne A}\min_{T\in b}\max_{i\in T\setminus R}h_i,
\qquad C_R=\{i\in R:h_i\ge e_R\}.$$

Use negative infinity for an empty inner maximum. The minimum edit cost is

$$\min_{R\in A:\ \max_{i\in R} l_i<e_R}|C_R|.$$

**Proof.** For any successful intervention choose a row $R$ attaining the incumbent value $u$, and a rival beating it. Every row of that rival contains a state scoring strictly above $u$. Such a state lies outside $R$, and its baseline score is at least its edited score. Thus $e_R>u\ge\max_R l_i$. Every state in $C_R$ must have changed to score at most $u$. Conversely edit exactly $C_R$ for any feasible row. Its new maximum falls below $e_R$. A rival attaining $e_R$ retains a witness of score at least $e_R$ outside $R$ in each reply row. This proves necessity, sufficiency and attainment. For nonnegative additive costs replace cardinality with the sum of the costs in $C_R$. Scanning the explicit rows and leaves takes polynomial time.

For a common intervention strength $s\in[0,1]$, set edited scores to $h_i-s(h_i-l_i)$, retaining full real precision. For a baseline incumbent each feasible row has nonempty $C_R$, since $e_R\le\max_{b\ne A}V_b(h)\le V_A(h)\le\max_R h_i$. Every member of $C_R$ in a feasible row has $h_i-l_i>0$. The row becomes feasible exactly when

$$s>\tau_R=\max_{i\in C_R}\frac{h_i-e_R}{h_i-l_i}.$$

The minimum at strength $s$ is the least cost over rows with $s>\tau_R$. At equality a rival can tie. This follows by solving each strict inequality $h_i-s(h_i-l_i)<e_R$. This frontier describes existence of a successful subset at a prescribed strength.

## Signed score changes

Allow either endpoint order. Put $d_i=\min(h_i,l_i)$ and $u_i=\max(h_i,l_i)$. Feasibility alone still takes polynomial time. In the preceding row test replace the floor by $\max_R d_i$ and use $u_i$ in the outside-row witness maximum. Necessity follows from the same separating threshold. For sufficiency set each state inside $R$ to its lower endpoint and each outside state to its upper endpoint.

For the exact minimum, fix $R\in A$, a rival $b$, and an integer threshold $t$. Require the incumbent row to score below $t$ and every reply row of $b$ to reach $t$. Every $i\in R$ with $h_i\ge t$ must be switched and satisfy $l_i<t$. Call this forced set $F$. Keep the remaining members of $R$ at baseline. After switching $F$, retain each rival reply row whose unchanged baseline witnesses fail to reach $t$. Each remaining row needs one selectable state outside $R$ with $l_i\ge t$. A state can cover every such row in which it occurs. Therefore the fixed-triple optimum is $|F|$ plus the minimum set-cover cardinality, or infinity when a row lacks a witness. The union of the forced and covering states realizes the specified strict inequality. Every successful intervention gives such a triple by taking a minimizing incumbent row, a winning rival and $t=1+\max_R x_i$. Upward edits inside $R$ can be undone because every witness above this maximum is external to $R$. It suffices to enumerate one plus the endpoint values between the row's minimum attainable maximum and its baseline maximum. A subset dynamic program on the missed rows takes $O(2^m\operatorname{poly}(L))$ time for $m$ missed rows and explicit game size $L$.

**Directional complexity boundary.** The budget decision problem for integer endpoints is NP-complete already for upward-only edits on general depth-three shared-leaf games with two root actions. Reduce Vertex Cover on a nonempty simple graph. Give the incumbent a constant score of one. Make one rival reply row per edge with a final choice between leaves representing its endpoints. Give each vertex leaf endpoints zero and two. The rival beats one exactly when the edited vertices meet every edge. A selected subset is verifiable by minimax backup in polynomial time. For downward-only edits at depth four, let the incumbent pass through a single MIN choice to a MAX choice of edge and a MIN choice of endpoint. Vertex leaves go from two to zero. A constant rival of one beats the incumbent exactly when every edge is hit. Singleton moves pad fixed branches to the required depth. These reductions concern arbitrary shared-leaf game graphs. The companion chess experiment measures concrete instances at depth three.

## Two representation tests

**Tie refinement at any finite depth.** Give each leaf the ordered pair $(H_i,N_i)$, using the classical score as first coordinate and native NNUE as second. Projection onto the first coordinate commutes with both finite minimum and maximum in lexicographic order. Induction up the game tree shows that every backed-up primary score equals its original classical value. At every node only classically optimal children can determine the secondary result. The refined root best-action set is contained in the original classical best-action set. Any strictly increasing recalibration of either score preserves the result. This establishes precisely which comparisons a tablebase test of this hybrid can credit to NNUE.

**A four-position obstruction.** Fix the kings, side to move and material. Choose two actor squares $A,B$ and two pawn squares $p,q$, with all four child positions generated from the same pair of legal candidate moves in the two contexts. The alternating difference is $D=f(A,q)-f(B,q)-f(A,p)+f(B,p)$. For every additive approximation $g(i,j)=u(i)+v(j)$, its alternating difference vanishes. The triangle inequality gives $\max|f-g|\ge |D|/4$. On these four positions equality is attainable by subtracting the residual $D/4$ times the alternating sign pattern from $f$. The resulting table has zero alternating difference and hence is additive. For a larger domain each observed rectangle supplies a lower bound. The target here is reproduction of the network's static score in native integer units. Tablebase outcomes independently test chess relevance. The general uniform approximation problem has a classical theory. [Diliberto and Straus, 1951](https://msp.org/pjm/1951/1-2/pjm-v1-n2-p04-p.pdf)

## Relationship to prior work

Classical conspiracy numbers count leaves that must change for a prescribed change in the scalar minimax value. Subsequent work used these numbers for move selection. The minimum-proof problem for AND/OR graphs already exhibits hardness with shared structure. Here the object is a strict inequality between two backed-up actions under one assignment, with fixed, state-specific evaluator or chess-rule endpoints. The outside-row formula, its amplitude frontier and the directional comparison specify the exact tractable case used in these measurements. These derivations and the empirical application support a comparison with prior methods. Priority of the precise characterization remains open. [McAllester](https://www.sciencedirect.com/science/article/pii/0004370288900197) · [Conspiracy numbers and move selection](https://dspace.jaist.ac.jp/dspace/bitstream/10119/14764/1/22342.pdf) · [Minimum proof graphs](https://ijcai.org/proceedings/2009/89) · [True proof numbers on DAGs](https://arxiv.org/abs/2102.04907)
