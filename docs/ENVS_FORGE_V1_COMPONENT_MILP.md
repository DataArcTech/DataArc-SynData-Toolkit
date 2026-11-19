# Envs-FORGE v1：分量编辑的混合整数规划

这份文档记录 `envs-forge-v1`。它取代 `envs-forge-alpha` 作为任务合成的选择器，并且**改变的是决策变量的层级**，不是动作菜单。

对应代码：

- 核心：`sdgsystem/agentic_data/envs_forge_v1.py`
- 运行：`examples/syn_agentic_data/run_envs_forge_v1_tblite.py`
- 测试：`tests/agentic_data/test_envs_forge_v1.py`

`envs-forge-alpha` 保留在 `sdgsystem/agentic_data/envs_forge.py`，`METHOD_VERSION = "envs-forge-alpha"`。

## 为什么 alpha 不是混合整数规划

alpha 的决策变量是「选哪个预设动作」：

\[
x_{i,a,d}\in\{0,1\},\quad \sum_{a,d}x_{i,a,d}\le 1
\]

动作的难度效果 \(\Delta_a\gamma_d\) 是外生常数，所以目标在 \(x\) 上线性、可行域是单纯形、顶点即整数。后果有两条，都是可验证的：

1. **实际跑的实例没有连续变量。** `slack_variables` 只在 `min_skill_coverage` 非空时创建，而两次正式跑批都是 `{}`。我把真实参数重建后交给 SCIP：变量类型统计 `{'BINARY': 6}`，连续变量 0 个。
2. **LP 松弛天然整数。** 同一个实例 SCIP 报告 `nodes = 0`、`lp_iterations = 0` —— 根节点 LP 直接给出整数解，整数约束是可删的。

所以 alpha 的 `optimization_formulation = "mixed_integer_programming"` 是名字，不是事实。它的实体是退化的 0-1 规划，等价于 argmax。

**补充一条 alpha 的隐性缺陷**：`estimate_pass_rate`（beta 平滑）有定义、有测试，但生产路径调的是 `_observed_reward_mean` 原始均值。探针只有一次时原始均值恰好是 0 或 1，log-odds 发散，92% 的种子撞在目标函数下限上。v1 把平滑接进主路径（见下）。

## v1 的决策变量

不再选动作，而是决定**编辑哪些组件、编辑多深**：

| 变量 | 类型 | 含义 |
|---|---|---|
| \(y_v\in\{0,1\}\) | 二元 | 是否触碰组件 \(v\in\{\text{instruction},\text{data},\text{solution},\text{tests},\text{environment}\}\) |
| \(p_v\in[0,e_{\max}]\) | 连续 | 正向编辑强度（加难） |
| \(n_v\in[0,e_{\max}]\) | 连续 | 负向编辑强度（降难） |
| \(\xi\ge 0\) | 连续 | 前沿带外弛豫 |

二元与连续通过 big-M 耦合：

\[
p_v\le e_{\max}y_v,\qquad n_v\le e_{\max}y_v
\]

目标同时含定费用与线性费用：

\[
\min\ \Lambda\xi+\sum_v\big(f_v y_v+u_v(p_v+n_v)\big)
\]

其中 \(f_v\) 是触碰组件的**定费用**（重新生成 + 重建镜像 + 重新验证的固定开销），\(u_v\) 是编辑强度的线性费用。

### 为什么这真的是 MILP

定费用 + big-M 耦合让 LP 松弛必然分数化：分数 \(y_v\) 可以拿到整份编辑效果却只付部分固定费用。实测（`test_model_is_mixed_integer_programming` 守着这条）：

```
z0 = +0.693   IP = 2.4767   LP = 2.2208   间隙 0.2558   LP 松弛 y = [0.488, 0.023, 0.488]
```

而且整数最优处的连续变量落在区间内部，不是贴边界：

```
z0 = +0.693 → e* = [0.977]          内部
z0 = +0.000 → e* = [0.0]（diversify，靠 y 承担决策）
```

`0 < e* < e_max` 意味着连续变量承载真实决策，不能被离散成几个整数档位——否则又会退回 BIP。

### 前沿带而不是前沿点

约束写成带，而不是等式：

\[
|z_0-\textstyle\sum_v\alpha_v(p_v-n_v)|\le \text{band}+\xi,\qquad \text{band}=0.4
\]

- 带内任何位置都算到达前沿，模型因此有自由度去选**更便宜**的编辑方案，而不是死磕 \(\tau=0.5\) 这一个点。
- \(\xi\) 带 \(\Lambda=100\) 的罚，远大于任何现实编辑成本。所以模型**先到前沿、再省成本**；前沿不可达时最小化残余距离，而不是直接 infeasible（对比 alpha 的硬约束 coverage 会因为不可达而无解）。
- \(\xi>0\) 直接等价于「这轮推不到，需要多轮」，`rounds_needed` 字段记录轮数。

### 一致性约束

改了指令就必须改测试，否则任务不可评分。这条在 alpha 里靠 prompt 祈使句嘱咐，v1 里是模型结构：

\[
y_{\text{instruction}}\le y_{\text{tests}},\qquad
y_{\text{solution}}\le y_{\text{tests}},\qquad
y_{\text{data}}\le y_{\text{tests}}
\]

这三个蕴含把可行子集从 \(2^5=32\) 压到 15 个。

### 都在前沿上的种子怎么办

若 \(|z_0|\le\text{band}\)，加一条锚点约束：

\[
\sum_{v\in\{\text{instruction},\text{data}\}} y_v\ge 1
\]

因为流水线要求每个种子都必须产出新环境，「什么都不做」不是合法解。碰 instruction 或 data 才能保证改动了具体需求或数据，而不是只动验证器——这正是 alpha 的 `diversify`（\(\Delta=0\)）看不见难度塌陷的地方。

## 动作标签是解的输出

v1 不再接收 `increase/reduce/diversify`。标签由求解结果反推：

| 条件 | 标签 |
|---|---|
| \(\sum_v\alpha_v e_v > 0.05\) | `increase_complexity` |
| \(\sum_v\alpha_v e_v < -0.05\) | `reduce_complexity` |
| 否则 | `diversify_frontier` |
| \(y\) 只落在 `{instruction, tests}` | `in_depth` |
| 否则 | `in_breadth` |

这样 `diversify` 不再是「难度不变的常数」，而是「\(y\) 变了但净 \(\Delta z\approx 0\)」。如果物料化后难度塌了，说明 \(\alpha_v\) 估错了，这个错误在 `z_predicted` 与实测通过率的对比里能被看到，而不是被 \(\Delta_{\text{diversify}}=0\) 掩盖。

## 灵敏度 \(\alpha_v\)

默认值（nats per unit strength）：

| 组件 | \(\alpha_v\) | 依据 |
|---|---:|---|
| instruction | 0.30 | 加约束/边界/输出契约是最直接的难度来源 |
| tests | 0.30 | 同步加强验证器 |
| solution | 0.20 | 跟随编辑 |
| data | 0.20 | fixture 规模与边界覆盖 |
| environment | 0.10 | 镜像本身对难度贡献最小 |

五个加起来 1.10 nats，正好等于 alpha 在 \(p=0.5\) 处一步 \(\Delta=0.25\) 的 log-odds 位移。所以「全组件满强度」和「alpha 一步」等价，可对照。

这些是**先验，不是估计值**。要换成估计值：

```bash
--sensitivity-file path/to/sensitivity.json
```

格式是 `{"instruction": 0.3, "tests": 0.3, ...}`。真实的估计方法是留一法消融（对同一批种子分别只改一个组件，测实测通过率），但当前仓库里**没有可用数据**——现有 100 个环境只有 gold verification，从没跑过 policy 探针。

## 探针与平滑

\[
\hat p_i^{\text{smoothed}}=\frac{1+\sum_t \mathbb 1[r_{i,t}\ge 1]}{2+n_i}
\]

v1 用平滑后的通过率算 \(z_0=\mathrm{logit}(\hat p)\)。原因是：\(n_i=1\) 时原始均值是 0 或 1，\(\mathrm{logit}\) 发散，所有边界种子需要无限编辑预算，全部落到同一个「最大编辑」解。平滑保证 \(z_0\) 有限，编辑预算才有东西可推。

`n_probes` 与 `pass_rate_raw` 都写进 plan 和 metadata，探针成本可以单独核算。

## 只用局部产物

这是 v1 与 alpha 最重要的工程差异。alpha 的 prompt 要求五个文件全改，`artifact_edit_mask.required_artifacts` 对每个投影强制 4–5 个组件，repair 明写 `Do not return patch fragments`。所以在 alpha 里 \(y_v\) 没有自由度。

v1 的生成契约是：

- prompt 只列出 `touched_components`，并明确禁止返回未触碰组件的文件；
- 校验用 `validate_subset_artifact`：文件必须属于被触碰的组件，每个被触碰组件至少一个文件，`tests` 必须给 `test.sh` + `test_outputs.py`；
- 落盘前用 `expand_subset_artifact` 把未触碰组件从种子原样继承，补成完整目录，再走原有 `materialize_artifact` + gold 验证。

端到端实测（`csv-json-jsonl-merger`，触 `tests`）：

```
touched = ['tests']
generated_files = ['tests/test.sh', 'tests/test_outputs.py']
status = gold_solution_validated
```

只生成了 2 个文件，其余 7 个从种子继承，gold solution 通过。

## 求解

- 默认 `--backend pyscipopt`，`PySCIPOpt` 不可用时自动退回精确枚举（`_solve_with_enumeration`）。
- 两条路径在 40 组随机 reward 上**结果完全一致**（子集、成本、净位移），由 `test_solver_and_enumeration_agree` 守护。
- 精确枚举的复杂度是可行子集数（15 个）而不是 \(2^5\)，所以它是正确的兜底，不是降级。

关于 `nodes = 1`：接受。LP 间隙真实存在（0.2558），但 SCIP 在根节点就闭合了。这不影响「这是 MILP」的判定——变量类型混合、整数约束不可省，两条都成立。`nodes` 与 `lp_iterations` 都写进 `solver_result` 里可供核查。

## 运行

```bash
PYTHONPATH=. .venv/bin/python examples/syn_agentic_data/run_envs_forge_v1_tblite.py \
  --source-root /Volumes/PSSD/code/My-Agent/Agent-Data/OpenThoughts-TBLite \
  --output-root outputs/envs_forge_v1_tblite \
  --performance-result outputs/envs_forge_tblite_milp_joint_full/logs/bip_plan_seed_rewards_as_harbor.json \
  --target-count 10 \
  --completion-max-tokens 40000 \
  --dry-run
```

注意 `--completion-max-tokens`：v1 的 prompt 更短（只给局部文件），但推理型模型会先把 20000 token 花在 reasoning 上、content 返回空。默认已设 40000。

输出：

| 文件 | 内容 |
|---|---|
| `envs_forge_v1_plan.json` | 每个种子的分量编辑方案（`touched`、`strengths`、`cost`、`z0`、`z_predicted`、`rounds_needed`） |
| `envs_forge_v1_solver_result.json` | 批量汇总 + SCIP 轨迹（`nodes`、`lp_iterations`、`band_slack`） |
| `terminal_bench_records.json` | 每次生成的 `generated_files`、`component_coverage`、验证结果 |
| `summary.json` | `mean_edit_cost`、`reachable_fraction`、`sensitivity`、`backend` |

## 与 alpha 的对照

| 项目 | envs-forge-alpha | envs-forge-v1 |
|---|---|---|
| 决策变量 | \(x\in\{0,1\}\)，6 选 1 | \(y\in\{0,1\}\), \(e\in\mathbb R\)，分量级 |
| 连续变量数（实跑） | 0 | 5 |
| LP 松弛 | 天然整数，`nodes=0` | 分数，间隙 0.2558 |
| 难度模型 | 概率空间加常数 \(\Delta\) | log-odds 平移 |
| 通过率估计 | 原始均值（端点发散） | beta 平滑 |
| 前沿约束 | 高斯核打分 + argmax | \(|z|\le\) band，带外罚 |
| 可行域 | 6 个动作 | 15 个一致子集 × 连续强度 |
| 生成范围 | 强制 5 文件全改 | 只改被触碰组件 |
| 不可达处理 | coverage 硬约束 → infeasible | 弛豫 \(\xi\)，记录 `rounds_needed` |
| 动作标签 | 输入 | 输出 |

## 已知边界

- **\(\alpha_v\) 未校准**。默认值是先验。校准需要留一法消融数据，当前没有。
- **探针仍可以只有 1 次**。平滑让端点可用，但 \(n_i=1\) 的分辨率上限仍在：真实 \(p=0.5\) 的种子单次探针下只能落在 0 或 1，平滑后 \(z_0=\pm0.693\)。提高 `n_i` 才是根治，v1 只是不再因此发散。
- **`nodes=1`**。间隙存在但根节点闭合。不声称需要分支。
- **`environment` 组件的 \(\alpha_v\) 最小**（0.10），因为它对难度的影响最间接。这个排序如果有实测数据可能被推翻。
