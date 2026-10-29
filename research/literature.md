# 文献调查与问题状态核查

核查日期：2026-10-29。页码均指正式出版物的印刷页码；PDF 阅读器的页序可能不同。下文区分原论文定理、具体构造、计算记录、猜想和仅供借鉴的其他规则集结果。

## 1. 两篇基础文献

### [E96] On Numbers and Endgames

Noam D. Elkies, *On Numbers and Endgames: Combinatorial Game Theory in Chess Endgames*, in *Games of No Chance*, MSRI Publications **29** (1996), pp.135–150。1999 年上传 arXiv，不能因此把出版年份改成 1999。

- [arXiv 记录](https://arxiv.org/abs/math/9905198)
- [正式全文，SLMath](https://library.slmath.org/books/Book29/files/elkies.pdf)

已核读全文。主要结论与边界如下。

| 内容 | 位置、状态与限制 |
|---|---|
| 用局部互逼走把兵的待着走法解释为 normal play | pp.136–138；这是经过论证的局部约化，不是一般棋局的规则。 |
| 整数、`*`、`↑` 及其加法组合的实例 | §2，pp.137–140；§3，pp.140–141 另析实战残局。`n↑` 是 `↑` 的加法倍数，不等于 §5 询问的 higher ups。 |
| `1/2`、`1/4`，以及大棋盘上的任意 dyadic fractions | §4，pp.141–143，Diagrams 6、7；加长棋盘给出更小的二进分数。 |
| switches、tinies 的构造 | §4，pp.143–144，Diagrams 8、9；没有证明任意 short game 均可实现。 |
| loopy 值及分解障碍 | §4，pp.144–148，Diagrams 10、13、14；Diagrams 11、12 显示将军可能破坏局部直觉。 |
| 明确问题清单 | §5，pp.148–149；当前状态须另查，见第 3 节。 |

### [E02] Higher Nimbers

Noam D. Elkies, *Higher Nimbers in Pawn Endgames on Large Chessboards*, in *More Games of No Chance*, MSRI Publications **42** (2002), pp.61–78；arXiv 原稿发表于 2000 年。

- [arXiv 记录](https://arxiv.org/abs/math/0011253)
- [正式全文，SLMath](https://library.slmath.org/books/Book42/files/elkies.pdf)

已核读全文。须分开记录以下四个层次。

| 层次 | 精确结论或记录 |
|---|---|
| 已证定理 | Theorem 1，§3：所定义兵游戏的静止分量等价于 nimber；entailing 应手另行处理。 |
| 已证周期 | Theorem 2，p.70：未设 stopped files 的分量值为 `*ε(m)`；`ε(m)=0` 当 `m mod 10∈{0,2,3,6,9}`，其余为 1。不是所有兵残局的周期 10。 |
| 已证推广 | Theorem 3，pp.75–76：用无相邻 `11` 的二进字编码 stopped files，建立判定 loony 走法和递推值的规则。mex 只能用于定理允许的非 loony 约化后继。 |
| 棋局实现 | §§4–5，pp.70–73：普通宽 n 的兵链可嵌入至少 n+6 列、奇数高度大于 5 的王兵残局；封堵列版本在奇数高度至少 9 的棋盘构造。均附加王与互逼走装置，防止核心终止成为逼和。这不是标准 8×8 的结论。 |
| 计算观察 | §6.3，pp.76–78：记录周期阻塞族的有限计算，包括每六列阻塞的族达到 `*4096`；这不证明值无界，也不证明每个有限 nimber 出现。 |
| 作者猜想 | 所有有限 `*k` 均能实现；无界性和覆盖性均未由上述有限计算建立。 |

因此，不能将此文概括为“已证明任意 nimber 可以由国际象棋兵残局实现”。它提供的是一般结构定理、特定周期定理、显式实例及普适性的计算证据。

本项目分别记录原始 entailing 兵状态与约化后的普通无偏图；$\epsilon(w)$ 是后者的 SG 指标。原始状态的替换只在原文证明覆盖的无偏外部环境中使用，不自动扩张为对所有 partizan 短博弈的 Conway 等价，详见 [definitions.md](definitions.md) 的 L2。

## 2. 关键后续研究

### [DC05] 标准 8×8 上的 `*2`

Gabriel C. Drummond-Cole, *Positions of value \*2 in generalized Domineering and chess*, **Integers 5** (2005), #G06，13 页。

- [期刊全文](https://math.colgate.edu/~integers/fg6/fg6.pdf)
- 已核读 §3，pp.8–13。

Figure 10 是 8×8 任意摆子构造，但棋子配置不可由正常棋局到达。**Figure 11 是可由正统初始局面经合法行棋到达的 `*2` 构造**，pp.11–12 附 67 回合棋谱。p.13 的组合选项为

`{exf7,gxf7,gxh7 | fxe6,fxg6,hxg6} = {*,0,* | *,0,*} = *2`。

作者在 pp.8–9 规定了约化至互逼走的 combinatorial positions；其他合法走法须论证为输棋走法。此结果含非兵棋子，不证明纯王兵残局的 8×8 可实现性。它解决了旧问题中普通棋子、标准棋盘的 `*2` 存在性部分。

### [N25] 近期开放问题清单

Richard J. Nowakowski, *Unsolved problems in combinatorial games*, in *Games of No Chance 6*, MSRI Publications **71** (2025), pp.55–98。

- [正式全文](https://library.slmath.org/books/Book71/files/GONC6-04.pdf)
- **Problem B3(40)，p.70** 仍明确列出：在足够大的有限棋盘上，普通国际象棋兵残局是否能产生每个有限 `*k`。

这是本次调查中确认该问题仍有研究价值的最直接近期文献依据。E19，p.89 讨论普适规则集、构造面积及排除值的一般方法；其中 generalized Konane 等的定理不能直接作为 chess 定理。

### [M02] 全局威胁与兵残局的早期后续

Fabian Mäser, *Global Threats in Combinatorial Games: A Computation Model with Applications to Chess Endgames*, in *More Games of No Chance*, MSRI Publications **42** (2002)。

- [正式全文](https://library.slmath.org/books/Book42/files/maeser.pdf)
- [出版社 DOI](https://doi.org/10.1017/9781316135167.012)
- 已核 §§1–3 及应用范围：用循环值 `on`、`off` 表示全局获胜威胁，给出截断分支的求值方法，§4 应用于适合 CGT 分解的王兵结构。

这项工作直接延续 [E96] 的 entailing 问题。将升变视为全局胜着是其限定模型的假设，不能用于任意兵残局；它也不提供一般标准棋局的独立分量定理。

### [LNS21] Entailing 的 impartial 理论

Urban Larsson, Richard J. Nowakowski, Carlos P. Santos, *Impartial games with entailing moves*, arXiv:2101.11699 (2021)。本次按 arXiv 原稿定位定理。

- [记录](https://arxiv.org/abs/2101.11699)
- [全文](https://arxiv.org/html/2101.11699v1)

Theorem 34 将 affine impartial 值归为 nimbers 与额外的 moon 值；Theorem 39 提供推广的 mex。适用前必须验证 impartial 和 entailing 公理；黑白棋子可以走不同的着法，并不自动满足这些条件。

### [LNS25] Affine normal play

Urban Larsson, Richard J. Nowakowski, Carlos P. Santos, *Affine normal play*, in *Games of No Chance 6*, MSRI Publications **71** (2025), pp.135–190。

- [正式全文](https://library.slmath.org/books/Book71/files/GONC6-06.pdf)
- [arXiv:2402.05732](https://arxiv.org/abs/2402.05732)
- [出版社记录及 DOI](https://doi.org/10.1017/9781009565486.007)

该文用 `±∞` 表示终止胜负，扩展 normal play 的比较与约化理论；经典 normal-play 值嵌入为可逆元素，约化形式一般不唯一。它是处理将杀和强制应手的候选工具，尚不能替代具体棋局的合法性、独立性及和棋机制证明。

相关的 [*A Complete Solution for a Nontrivial Ruleset with Entailing Moves*](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v31i4p65)，同三位作者，**Electronic Journal of Combinatorics 31(4)** (2024), #P4.65，完整求解 Christmas Lights’ Fixture。其对象不是 [E02] 的兵残局族。

### [SS15] Partizan 规则中构造 nimber

Carlos Pereira dos Santos, Jorge Nuno Silva, *Nimbers in partizan games*, in *Games of No Chance 4*, MSRI Publications **63** (2015), pp.215–224。

- [正式全文](https://library.slmath.org/books/Book63/files/131105-PereiradosSantos.pdf)
- pp.217–220 的类与引理利用可逆选项识别 nimber；展示的 `*4` 属于 **Amazons**，不是标准国际象棋。

可借鉴其构造方法，但任何 chess 中的使用都须给出对应的棋子配置与选项论证。

### [S11] Nim-dimension 与 embedding 方法

Carlos Pereira dos Santos, *Embedding processes in combinatorial game theory*, **Discrete Applied Mathematics 159(8)** (2011), pp.675–682。

- [出版社全文及 DOI](https://doi.org/10.1016/j.dam.2010.11.019)
- §1 引入 nim-dimension 背景，§2 讨论 octal games，后文讨论 embedding 方法。

其中 Dawson’s Chess 采用有吃必吃和 normal play。相关周期性不能直接迁移到吃子可选、须应将、允许和棋的通常国际象棋。

### [A+25] 循环与 carry-on

Tomoaki Abuku, Alda Carvalho, Urban Larsson, Richard J. Nowakowski, Carlos P. Santos, Koki Suetsugu, *Cyclic impartial games with carry-on moves*, arXiv:2512.14466 (2025-12-16)。

- [原始预印本](https://arxiv.org/abs/2512.14466)
- 已核摘要：合并循环 impartial 理论与 carry-on 理论；carry-on 是强制应手没有选择的 entailing 特例，示例为 Green-Lime Hackenbush。

本次未逐条审查其证明。它列作工具线索，不能据此声称已有标准国际象棋循环值的完整分类。

### [BT26] 无限国际象棋中的另一种 universality

Matthew Bolan, Andreas Tsevas, *Universality of Infinite Chess*, arXiv:2602.14277v1 (2026-02-15)。

- [原始预印本](https://arxiv.org/abs/2602.14277)
- [全文](https://arxiv.org/html/2602.14277v1)
- 已核摘要、§1 及主结论：作者给出 `Z²` 上允许无限棋子的战略模拟；Corollary 1 涉及全部可数序数，§3 给出仅双王和无限兵的构造。

这里的值是 open-game 的序数胜出距离，等价关系是战略模拟。该预印本不证明有限棋盘的 Conway 加法值普适性，不能用它宣布 [E02] 的问题已解。本次没有独立复核其全部证明。

## 3. 历史问题状态审计

| 问题 | 历史出处 | 本次核查后的状态 |
|---|---|---|
| 普通棋子、8×8 上能否出现 `*2`？ | [E96] p.148 | **已有构造**：[DC05] Figure 11；采用该文的组合局面约化。 |
| 大的有限棋盘上是否能实现每个有限 nimber？ | [E02] 摘要、§6.3 | **近期文献仍列为开放**：[N25] B3，p.70；本次未找到更晚解决。 |
| 纯王兵残局、8×8 上能否实现 `*2`，或更高 nimber？ | 与 [E96] 问题有关的更窄版本 | **当前状态待核**：[DC05] 不解决纯王兵版本；本次未取得完整分类或排除定理。 |
| [E02] 特定周期阻塞族的值是否无界或有可证明的规律？ | [E02] §6.3，pp.77–78 | **历史研究方向；具体子族须再核**。不能把“全体构造普适”与“每六列阻塞族无界”等同。 |
| 标准 8×8 是否出现 `1/8`？ | [E96] p.148 | **历史提出；当前状态待核**。本次未找到可核实的解决论文。 |
| tiny 的非整数参数、higher ups 是否出现？ | [E96] p.148 | **历史提出；当前状态待核**。`n↑` 的已有加法构造不回答 higher ups。 |
| loopy 棋局能否给出 `1/3` 等非 dyadic 值？ | [E96] p.148 | **历史提出；当前状态待核**。须先指定循环、和棋、比较及加法模型。 |
| 是否存在自然适用 chilling 的棋局类？ | [E96] pp.148–149 | **历史提出；当前状态待核**。本次未找到可确认的完整回答。 |
| 系统处理 en passant、将军、强制吃回 | [E96] pp.148–149 | **已有后续一般理论**：[M02]、[LNS21]、[LNS25]；[A+25] 涉及更窄的循环 carry-on 模型。尚不能称为通常国际象棋的统一分解定理。 |

“本次未找到解决”是检索结果，不是未解决性的证明。只有有近期证据支持的条目使用“近期文献仍列为开放”。本项目自己提出的猜想放在 [ideas.md](ideas.md)，不冒充文献问题。

## 4. 本项目采用文献结论时的检查项

以下是由上述文献边界形成的研究规范，不是新定理。

1. 写清楚研究对象是完整棋局、局部 normal-play 分量，还是经强制应手约化的分量。完整合法走法与“合理走法”不能混为一谈。
2. 互逼走装置必须对全部相关后继有效；兵无可走时应触发已证的败局，而不是逼和或死局。
3. 全局每回合仍只走一步。跨分量的将军、攻击线、吃过路兵权和升变可能使直和失效；不能按空间距离直接断言独立。
4. 说明“可合法摆放”与“可从正统初始局面到达”的区别；推广棋盘另需给出初始兵位、双步规则与升变线。
5. 循环模型须交代重复和无吃兵走子的计数。局面状态若包含全局历史，分量各自的棋盘图通常不足以决定后续。
6. abstract SG 值、partizan 的 Conway 等价、受限 universe 中的等价、胜负结果相同以及 ordinal 胜出距离是不同结论。

## 5. 检索记录与继续核查边界

日期：2026-10-29。通过网页学术检索、原论文参考文献、SLMath 正式文集、期刊及作者原始预印本交叉核查；未进行 MathSciNet/zbMATH 全库引用链穷尽检索，也未联系作者。

主要检索式：

- `Elkies nimbers conjecture`
- `chess nimber *4`
- `chess nimbers Drummond`
- `Elkies Higher Nimbers solved`
- `On Numbers and Endgames fractions`
- `chess Elkies 1/8`
- `chess higher ups`
- `chess tiny infinitesimals`
- `chess chilling Elkies`
- `pawn endgames periodicity Elkies`
- `Affine normal play`
- `Cyclic impartial games carry-on moves`
- `Universality of Infinite Chess`

关键核查结果是 [DC05] 对 `*2` 的解决和 [N25] 对大棋盘 nimber 问题的继续列示。一般 CGT、其他棋类规则、Amazons、Konane、Dawson’s Chess、无限棋盘的结果只按各自模型使用。检索到的教学讲义和学生综述没有被用作“当前开放”的唯一证据。
