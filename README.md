# Combinatorial Game Theory and Algebraic Structures in Chess Endgames

研究国际象棋残局中能够实现的组合博弈值、受限兵结构的递推规律，以及棋盘和构造资源对这些值的限制。

## 当前进展

研究记录更新于 **2026-10-29**：已从 Elkies 的既有递推得到单封堵列分类，并以计算机辅助证明确定双封堵列的精确容量 **$M_2=5$**，附无限覆盖论证及两份独立有限证书。这些项目推论的新颖性尚未确认。

- Elkies 已构造分数、无穷小和若干循环值；相应解释依赖明确的正常制转译或循环模型。
- 无 stopped files 的兵链已有严格的 10 周期公式；带 stopped files 的兵链已有含 entailing 状态的递推。
- 普通 8×8 局面的 `*2` 已有 Drummond-Cole（2005）的合法构造；不能再把一般的 8×8 `*2` 存在性列为开放问题。
- “足够大的有限棋盘兵残局实现每个有限 nimber”仍列于 Nowakowski（2025）的问题 B3；本次未检索到其后的解决。

这些结论的出处、适用范围和历史问题审计见 [文献记录](research/literature.md)。标准棋局、约化局部博弈与无限棋盘的序数值不是同一种对象；具体等价关系见 [定义](research/definitions.md)。

## 双封堵列容量与周期

在 Elkies 的 stopped-file 模型中，令 $F(a,b,c)=\epsilon(0^a10^b10^c)$，其中 $a,c\ge0,b\ge1$。已证明全部参数下 $F\le5$，且 $F(0,5,10)=5$。结合零／单封堵列结果，有

$$
M_0=1,\qquad M_1=3,\qquad M_2=5.
$$

三个坐标均有统一最终周期 10：外侧长度从 2 起、中间长度从 11 起。这些阈值对周期 10 均最优。证明保留普通冒号、停止冒号及一步记忆，再将全部无限参数的递推恒等式归约到有限代表；两份独立实现分别使用通用词递推和参数特化递推。原始兵规则搜索另确认了两个单封堵列关键实例及双封堵列指标 4 的实例。

详见 [双封堵列分析](research/two-stop-analysis.md) 与 [完整证明](research/two-stop-proof.md)。结论针对单个抽象静止词分量；一般固定封堵列数的容量仍待研究。

## 单封堵列结果

在 Elkies 的 stopped-file 模型中，记

$$
f(a,b)=\epsilon(0^a10^b),\qquad a,b\in\mathbb Z_{\ge0}.
$$

已得端点及邻端点的完整周期表；对 $a,b\ge2$，

$$
f(a,b)=\epsilon(0^a)\mathbin{\mathrm{xor}}\epsilon(0^b)
\mathbin{\mathrm{xor}}[5\mid a\text{ 且 }5\mid b].
$$

整个单封堵列族的值域恰为 `0,*1,*2,*3`。原候选 P1 的最小统一阈值为 2。这些是抽象兵模型的归约值结论；证明保留了 entailing 状态，并用无限剩余类覆盖完成归纳，未将有限样本外推为定理。

完整论证见 [单封堵列分类](research/single-stop-classification.md)，另有 [端点的独立证明](research/endpoint-classification.md)。

## 文档

- [AGENTS.md](AGENTS.md)：长期数学研究规范。
- [research/literature.md](research/literature.md)：基础论文、后续理论和检索记录。
- [research/definitions.md](research/definitions.md)：模型、符号、等价与分解条件。
- [research/two-stop-proof.md](research/two-stop-proof.md)：双封堵列精确容量、统一周期及完整有限状态分类。
- [research/open-problems.md](research/open-problems.md)：三个方向、问题状态和排序。
- [research/ideas.md](research/ideas.md)：首项研究的命题、证明路线和审查要求。
- [research/checks/](research/checks/)：Python 标准库有限证书及独立的 Python/C++ 三行兵规则检查。

下一步研究双封堵列分类的概念性简化及固定封堵列数的结构界，并继续核查新颖性；尚未进行论文投稿。
