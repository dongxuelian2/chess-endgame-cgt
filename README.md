# Combinatorial Game Theory and Algebraic Structures in Chess Endgames

研究国际象棋残局中能够实现的组合博弈值、受限兵结构的递推规律，以及棋盘和构造资源对这些值的限制。

## 当前进展

基础调查完成于 **2026-10-29**：核读 Elkies 的两篇基础论文，核查重要后续工作，建立模型、问题状态和初步研究计划。目前没有本项目的新定理或新的计算结果。

- Elkies 已构造分数、无穷小和若干循环值；相应解释依赖明确的正常制转译或循环模型。
- 无 stopped files 的兵链已有严格的 10 周期公式；带 stopped files 的兵链已有含 entailing 状态的递推。
- 普通 8×8 局面的 `*2` 已有 Drummond-Cole（2005）的合法构造；不能再把一般的 8×8 `*2` 存在性列为开放问题。
- “足够大的有限棋盘兵残局实现每个有限 nimber”仍列于 Nowakowski（2025）的问题 B3；本次未检索到其后的解决。

这些结论的出处、适用范围和历史问题审计见 [文献记录](research/literature.md)。标准棋局、约化局部博弈与无限棋盘的序数值不是同一种对象；具体等价关系见 [定义](research/definitions.md)。

## 首选问题

在 Elkies 的 stopped-file 模型中，分类

$$
f(a,b)=\epsilon(0^a10^b),\qquad a,b\in\mathbb Z_{\ge0}.
$$

优先检验是否存在统一阈值，使两个参数最终分别具有周期 10。此为本项目提出的**待检验命题**，尚未证明、尚无本项目计算证据，亦未确认独立新颖性。若不成立，寻找反例及正确的分类。它从已证无缺陷公式出发，只增加一个封堵列，适合先建立可复核的一般结果。

## 文档

- [AGENTS.md](AGENTS.md)：长期数学研究规范。
- [research/literature.md](research/literature.md)：基础论文、后续理论和检索记录。
- [research/definitions.md](research/definitions.md)：模型、符号、等价与分解条件。
- [research/open-problems.md](research/open-problems.md)：三个方向、问题状态和排序。
- [research/ideas.md](research/ideas.md)：首项研究的命题、证明路线和审查要求。

本次初始化止于研究计划。出现可写作的结果后再建立 `papers/`。
