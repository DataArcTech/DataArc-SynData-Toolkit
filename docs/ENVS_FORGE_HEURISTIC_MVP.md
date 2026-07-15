# Envs-FORGE Heuristic MVP 建模说明

这份文档记录 `ce16424 feat: add Envs-FORGE MVP synthesis` 这一版的建模方式。它是一个可运行、可验证的启发式 MVP：先用已有 agent 执行结果估计 seed 难度，再用固定规则选择一个生成原语，把任务推向学习前沿附近。

## 大白话版本

我们手里有一批 terminal-bench 风格的原始任务。每个任务都包含四类核心组件：任务说明、输入/环境文件、参考解法、测试脚本。Envs-FORGE 不是只改一句 instruction，而是要求生成后的任务仍然能被参考解法解出来，并且测试能验证这个解法。

当前 MVP 的核心问题是：

> 给定一个原始任务和当前下游 agent 在这个任务上的历史 reward，应该把它变难、变简单，还是保持差不多难度但换一个具体场景？

这一版没有求解一个复杂优化问题，而是使用简单分桶规则：

- 如果 agent 基本都会做，说明任务太容易，就用 `increase_complexity` 增加约束、边界条件或数据规模。
- 如果 agent 基本做不出来，说明任务太难，就用 `reduce_complexity` 降低组合复杂度，把它变成桥接任务。
- 如果 agent 有时能做、有时做不出来，说明任务接近学习前沿，就用 `diversify_frontier` 保持难度附近，换数据、约束或具体业务背景。

生成后必须通过 gold solution 验证。也就是说，合成任务的 `instruction.md`、输入数据、`solution/solve.sh`、`tests/test.sh`、`tests/test_outputs.py`、`environment/Dockerfile` 必须互相一致。

## 变量

令第 \(i\) 个 seed 任务为：

\[
s_i = (I_i, D_i, Sol_i, T_i, E_i)
\]

其中：

- \(I_i\)：原始任务说明，例如 `instruction.md`。
- \(D_i\)：输入数据、配置、fixture、辅助脚本等。
- \(Sol_i\)：原始参考解法，例如 `solution/solve.sh`。
- \(T_i\)：原始测试与 verifier，例如 `tests/test.sh` 和 `tests/test_outputs.py`。
- \(E_i\)：运行环境，例如 `environment/Dockerfile`。

已有 agent 在该 seed 上的 reward 序列为：

\[
R_i = \{r_{i1}, r_{i2}, \dots, r_{in}\}, \quad r_{ik} \in [0, 1]
\]

当前 MVP 使用观测均值估计 pass rate：

\[
\hat p_i =
\begin{cases}
\frac{1}{|R_i|}\sum_{r \in R_i} r, & |R_i| > 0 \\
0.5, & |R_i| = 0
\end{cases}
\]

直觉上，\(\hat p_i\) 越高，任务越容易；越低，任务越难；接近 0.5 时，任务更像当前 agent 的学习前沿。

学习前沿分数定义为：

\[
F(\hat p_i) = \exp\left(-\frac{(\hat p_i - 0.5)^2}{2\sigma^2}\right)
\]

这里 \(\sigma=0.2\)。它不是训练目标，而是一个排序和解释用的分数：越接近 0.5，分数越高。

从 seed 中抽取一个粗粒度 skill profile：

\[
g_i = (\text{domains}, \text{tools}, \text{artifacts}, \text{operations}, \text{constraints})
\]

当前代码用关键词近似实现它。例如 systemd、csv、jsonl、pytest、concurrency、structured output 等会被映射成标签。它还不是完整 skill graph，只是 skill graph 的轻量替代。

动作空间只包含三个生成原语：

\[
A = \{
\text{increase\_complexity},
\text{reduce\_complexity},
\text{diversify\_frontier}
\}
\]

最终生成样本为：

\[
c_i = G(s_i, g_i, a_i)
\]

其中 \(G\) 是 LLM 合成器，\(a_i\) 是启发式规则选择出来的原语。

## 目标函数

这一版的目标可以写成“规则近似”的形式：

\[
\max_{a_i \in A} \ F(\hat p_i')
\]

但当前 MVP 并不会枚举动作并预测 \(\hat p_i'\)。它使用更简单的投影规则，把 seed 按 \(\hat p_i\) 分为三档：

\[
b_i =
\begin{cases}
\text{easy}, & \hat p_i \ge 0.8 \\
\text{hard}, & \hat p_i \le 0.2 \\
\text{frontier}, & 0.2 < \hat p_i < 0.8
\end{cases}
\]

然后固定映射到动作：

\[
a_i =
\begin{cases}
\text{increase\_complexity}, & b_i = \text{easy} \\
\text{reduce\_complexity}, & b_i = \text{hard} \\
\text{diversify\_frontier}, & b_i = \text{frontier}
\end{cases}
\]

大白话说：这不是“算出最优动作”，而是“根据当前 agent 的会不会做，决定往哪个方向推一下”。

## 约束

第一类是可验证约束。它是所有可验证 agentic data 合成方法都应该满足的硬约束，不是 Envs-FORGE 独有：

\[
V_{\text{gold}}(c_i) = 1
\]

含义是：合成任务用自己的参考解法跑完，再跑自己的测试，reward 必须为 1。

第二类是组件一致性约束：

\[
c_i = (I_i', D_i', Sol_i', T_i', E_i')
\]

如果改了任务说明，就必须同步改输入数据、参考解法和测试。Dockerfile 也可能需要改，但应尽量复用 seed 环境，避免不必要依赖。

第三类是 skill 保持约束：

\[
\text{overlap}(g_i, g_i') \ge \tau
\]

当前 MVP 没有用显式相似度阈值求解，而是在 prompt 中约束“preserve the target skill subgraph”。实际代码用 skill tags 写入 prompt，让模型围绕同类 domain/tool/artifact/operation/constraint 做投影。

第四类是测试集隔离约束：

\[
c_i \sim \text{SeedOnly}(\text{TBLite})
\]

生成与决策只看 TBLite seed 和已有执行结果，不看 terminal-bench 测试集任务内容，避免测试集泄露。

第五类是产物规模约束。当前 prompt 显式要求小型、可运行、可验证的任务，例如：

- `solution/solve.sh` 尽量少于 150 行。
- `tests/test_outputs.py` 尽量少于 220 行。
- 不生成大段嵌入数据或长篇说明。
- 如果需要网络安装，尽量复用 seed 依赖类别，并允许使用国内镜像。

## 与四种 baseline 的关系

四种已有 baseline 也都要求生成后验证：

- few-shot：靠示例引导，生成一个结构相近的新任务。
- self-instruct：让模型自发提出同域或相邻任务。
- evol-depth：显式加深任务。
- evol-breadth：显式拓宽任务。

Envs-FORGE Heuristic MVP 的区别是，它把“应该加深、降低还是保持前沿多样化”绑定到当前 agent 的执行结果上。换句话说，它不是单纯扩增数据，而是把合成方向和 agent 当前能力联系起来。

## 局限

这一版是启发式版本，有三个明显限制：

- 它只根据 seed 的历史 pass rate 分桶，没有估计生成后样本的 pass rate。
- 它每个 seed 只选择一个动作方向，不做多候选池选择。
- skill graph 只是关键词标签近似，不是真正的图结构、先修关系或 action dependency graph。

下一版 calibrated heuristic transfer 会在不引入 surrogate、也不扩展三个原语的前提下，把三个动作都枚举一遍，用动作难度迁移量估计生成后 pass rate，再选择预计最接近 0.5 的动作。
