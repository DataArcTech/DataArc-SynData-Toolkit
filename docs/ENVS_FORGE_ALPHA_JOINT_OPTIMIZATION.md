# Envs-FORGE Alpha Binary Integer Programming 建模说明

这份文档记录 Envs-FORGE alpha 的 `binary_integer_programming` 版本：在 calibrated transfer MVP 之上，把 projection / skill subgraph 的选择显式写成 0-1 整数规划问题，并提供可选 PySCIPOpt 求解接口。`artifact_edit_mask` 暂不进入优化器，只作为生成前约束、prompt conditioning 和输出元数据。

代码里这个版本会在 plan、solver result、summary 和合成任务 metadata 中写入：

```json
{
  "optimization_formulation": "binary_integer_programming",
  "optimization_formulation_label": "0-1 integer programming"
}
```

这样后续新增 MIP 版本时，可以继续复用同一组字段，例如把 formulation 扩展为 `mixed_integer_programming`，而不需要改变现有 `projection_policy="joint_optimization"` 的兼容入口。

对应 MVP 文档是：

- `docs/ENVS_FORGE_CALIBRATED_HEURISTIC_TRANSFER.md`

## 大白话版本

MVP 做的是：

> 对每个 seed，根据当前 agent reward 估计，在 `increase_complexity`、`reduce_complexity`、`diversify_frontier` 三个动作里选一个最接近 learning frontier 的动作。

alpha 版继续往前走一步：

> 不只看哪个动作让 reward 更接近 0.5，还把“这批数据要覆盖哪些 skill”放进 batch-level selection。

例如，如果这轮数据合成希望至少覆盖 1 条 `system_admin` skill，那么 alpha 版可能会选择一个 frontier score 稍低的 `reduce_complexity` systemd 任务，而不是只选择局部 frontier score 最高的 JSONL 任务。

这就是 alpha 相比 MVP 的关键提升：从单 seed 的动作投影，变成带约束的 batch-level joint optimization。

## MVP 和 Alpha 的差异

| 维度 | Calibrated Transfer MVP | Alpha Joint Optimization |
|---|---|---|
| 高层动作 | 3 个 projection primitives | 仍是 3 个 projection primitives |
| 动作选择 | 每个 seed 独立选择最接近 frontier 的动作 | 在所有 `(seed, action)` 候选上做 batch 选择 |
| artifact edits | prompt 里要求同步修改 | 仍作为 prompt/metadata，不进入 optimizer |
| edit operator | 没有显式变量 | 记录每类 artifact 允许的 operator family，但不参与求解 |
| skill graph | 关键词 `SkillProfile`，主要用于 prompt conditioning | skill node 作为 solver 变量，edges 由 selected nodes 诱导 |
| 约束 | gold verification、组件同步、测试集隔离 | 额外支持 skill coverage |
| 求解 | calibrated heuristic transfer | 0-1 整数规划；PySCIPOpt 可选，缺失时 exact enumeration / greedy fallback |
| 输出 | projection、predicted pass rate、frontier score | 额外输出 optimization formulation、solver backend、artifact edit mask、skill subgraph |

## 变量

第 \(i\) 个 seed task 写成：

\[
s_i=(I_i,D_i,Sol_i,T_i,E_i)
\]

动作集合仍然是：

\[
A=\{
\text{increase\_complexity},
\text{reduce\_complexity},
\text{diversify\_frontier}
\}
\]

alpha 版的高层选择变量是：

\[
x_{i,a}\in\{0,1\}
\]

表示是否选择 seed \(i\) 的 action \(a\)。

artifact edit mask 暂不建模为 solver 变量。当前只把它写成 action 派生的生成前合约：

\[
M^{req}_{i,a},\quad M^{opt}_{i,a},\quad O_{i,a,k}
\]

其中 \(M^{req}_{i,a}\) 是必须同步编辑的 artifact 集合，\(M^{opt}_{i,a}\) 是可选 artifact 集合，\(O_{i,a,k}\) 是 artifact \(k\) 上允许的 operator family。它们会进入 prompt 和 metadata，但不会作为 optimizer 的自由变量或硬约束。

skill graph 先从 seed 和 action 得到候选模板：

\[
\bar g_{i,a}=(\bar V_{i,a},\bar E_{i,a})
\]

模板里包含：

- `nodes`
- `edges`
- `preserve_nodes`
- `allowed_added_nodes`
- `allowed_removed_nodes`

真正进入 solver 的 skill node 变量是：

\[
u_{i,a,v}\in\{0,1\}
\]

表示如果选择 seed \(i\) 的 action \(a\)，目标任务是否激活 skill node \(v\)。当前 alpha 不单独建 edge 变量 \(q\)，而是用被选中的 nodes 对模板边 \(\bar E_{i,a}\) 做 induced subgraph。

## Objective

当前 reward estimate 仍继承 MVP 的 calibrated transfer：

\[
\tilde p_{i,a}=\operatorname{clip}(\hat p_i+\Delta_a,0,1)
\]

默认：

\[
\Delta_{\text{increase}}=-0.25,\quad
\Delta_{\text{reduce}}=+0.25,\quad
\Delta_{\text{diversify}}=0
\]

frontier score 为：

\[
F(\tilde p_{i,a})=
\exp\left(
-\frac{(\tilde p_{i,a}-0.5)^2}{2\sigma^2}
\right)
\]

batch-level objective 是：

\[
\max_x \sum_i\sum_a x_{i,a}F(\tilde p_{i,a})
\]

大白话说：在满足覆盖约束的前提下，尽量选择预计更接近 learning frontier 的合成动作。

## Constraints

每个 seed 最多选一个 action：

\[
\sum_{a\in A}x_{i,a}\le 1
\]

总共选择 \(N\) 条：

\[
\sum_i\sum_a x_{i,a}=N
\]

skill coverage 约束：

\[
\sum_i\sum_a u_{i,a,v}\ge m_v
\]

skill node 和 action selection 的绑定是：

\[
u_{i,a,v}\le x_{i,a}
\]

必须保留的 core / non-removable nodes：

\[
u_{i,a,v}=x_{i,a},\quad v\in R_{i,a}
\]

其中：

\[
R_{i,a}=(\bar V_{i,a}\setminus A^-_{i,a})\cup P_{i,a}
\]

\(P_{i,a}\) 是 preserve nodes，\(A^-_{i,a}\) 是 allowed removed nodes。

可选 nodes 包括 allowed added nodes 和 allowed removed nodes：

\[
v\in A^+_{i,a}\cup A^-_{i,a}\Rightarrow u_{i,a,v}\in\{0,x_{i,a}\}
\]

其他不允许的节点不能被激活：

\[
u_{i,a,v}=0,\quad v\notin R_{i,a}\cup A^+_{i,a}\cup A^-_{i,a}
\]

为了避免没有 coverage 要求时 solver 随手打开可选节点，目标函数里加了一个极小 penalty：

\[
\max_{x,u}\sum_i\sum_a x_{i,a}F(\tilde p_{i,a})
-\epsilon\sum_i\sum_a\sum_{v\in A^+_{i,a}\cup A^-_{i,a}}u_{i,a,v}
\]

当前实现中 \(\epsilon=10^{-6}\)，只用于 tie-break，不改变 frontier objective 的主要排序。

artifact mask 不进入当前 alpha 的 optimization constraints。原因是 instruction、data、oracle solution、tests、environment 之间存在强语义依赖，如果靠人工枚举所有 implication rule，很容易漏掉边界情况，反而给 optimizer 一个错误的可行域。

当前策略回退到 MVP 风格：在 prompt 中要求同步生成完整 terminal-bench artifact，并在生成后用 gold verification 过滤。也就是不让 solver 判断“哪些 artifact 组合可行”，而让真正的 `solution/solve.sh + tests/test.sh` 验证任务自洽性。

gold verification 仍然是生成后硬约束：

\[
V_{\text{gold}}(c_{i,a})=1
\]

也就是说，solver 决定两层东西：选择哪些 `(seed, action)`，以及这些候选里哪些可选 skill nodes 被激活。真正进入数据集仍然必须通过 terminal-bench 风格 gold solution validation。

## 求解实现

入口仍是：

```python
plan_envs_forge_batch(
    seeds,
    rewards_by_seed=rewards,
    target_count=10,
    projection_policy="joint_optimization",
    min_skill_coverage={"system_admin": 1},
)
```

求解顺序是：

1. 尝试使用 `pyscipopt`。
2. 如果 `pyscipopt` 不可用，且候选数不超过 `EXACT_ENUMERATION_LIMIT=24`，使用 exact enumeration。
3. 如果候选数更大，使用 deterministic greedy fallback。

如果本地安装了 `pyscipopt`，planner 会优先走 PySCIPOpt；否则自动 fallback 到 exact enumeration 或 deterministic greedy。fallback 不是形式上的：测试覆盖了一个场景，要求 solver 为了满足 `system_admin` skill 约束，选择 frontier score 较低的 systemd reduce candidate，而不是局部最优 JSONL candidate。

当前还额外覆盖了两个 PySCIPOpt 行为示例：

- `artifact_edit_mask` 不参与 optimizer，所以如果两个候选只在 mask 上不同，solver 仍按 frontier objective 和 skill constraints 选择。这是刻意的保守回退。
- skill node 是决策变量。如果 coverage 要求 `edge_cases`，而 seed 本身没有该节点，但 `increase_complexity` 的 `allowed_added_nodes` 允许新增它，那么 solver 会激活 \(u_{i,a,\text{edge\_cases}}=1\)，并把 `edge_cases` 写入最终 `skill_subgraph.nodes`。

## CLI 用法

最小 dry-run：

```bash
.venv/bin/python examples/syn_agentic_data/run_envs_forge_tblite.py \
  --source-root /Volumes/PSSD/code/My-Agent/Agent-Data/OpenThoughts-TBLite \
  --output-root /tmp/envs_forge_alpha_joint_smoke \
  --target-count 1 \
  --limit-seeds 5 \
  --projection-policy joint_optimization \
  --dry-run
```

带 coverage 约束：

```bash
.venv/bin/python examples/syn_agentic_data/run_envs_forge_tblite.py \
  --source-root /Volumes/PSSD/code/My-Agent/Agent-Data/OpenThoughts-TBLite \
  --output-root /tmp/envs_forge_alpha_joint_smoke \
  --target-count 1 \
  --limit-seeds 5 \
  --projection-policy joint_optimization \
  --min-skill-coverage system_admin=1 \
  --dry-run
```

`--min-skill-coverage` 可以重复传入，格式是 `name=count`。

## 输出字段

`envs_forge_plan.json` 和 record metadata 会新增：

- `optimization_formulation`
- `optimization_formulation_label`
- `solver_backend`
- `artifact_edit_mask`
- `skill_subgraph`

其中 `artifact_edit_mask` 包含：

- `required_artifacts`
- `optional_artifacts`
- `operators`

`skill_subgraph` 包含：

- `nodes`
- `edges`
- `preserve_nodes`
- `allowed_added_nodes`
- `allowed_removed_nodes`

这些字段让后续分析可以回答：

- 这条合成任务计划改哪些 artifact？
- 为什么 environment 必须改？
- 哪些 skill 被保留？
- 哪些 skill 是新增的？
- 哪些复杂外围 skill 被删除或模拟？
- batch selection 是 PySCIPOpt、exact enumeration 还是 greedy 得到的？

## 当前限制

alpha 版已经把 \(x,u\) 显式化，并把 artifact mask/operator 作为 metadata 记录下来，但还不是最终形态：

- \(\Delta_a\) 仍是固定 heuristic，不是按 skill/domain 校准出的表。
- artifact mask 暂不进入 optimizer；同步性主要靠 prompt contract 和 gold verification。
- edit operator 当前由 action deterministic 派生，还不是 solver 自由选择的变量。
- 还没有从一个 seed 生成多候选后做 candidate pool selection。
- skill graph 模板仍由关键词近似，不是真正从大量任务轨迹中学习出来的 graph；当前只让 node 进 solver，edge 还是 induced subgraph。
- `pyscipopt` 是可选后端；未安装时自动 fallback。

下一步可以把 \(\Delta_a\) 升级为 \(\Delta_{a,g}\)。artifact mask 是否重新进入 optimizer，需要先有更可靠的 artifact dependency schema 或从真实 verified tasks 中学习出的可行 edit graph。
