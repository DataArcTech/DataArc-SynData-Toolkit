# Envs-FORGE Calibrated Heuristic Transfer 建模说明

这份文档记录 Envs-FORGE 在 Heuristic MVP 之后的第二个小版本：`calibrated_transfer`。它仍然只使用三个生成原语，不引入 learned surrogate，不做多候选池筛选，而是在生成前估计每个原语会把任务难度往哪个方向推，然后选择预计最靠近学习前沿的原语。

## 大白话版本

第一版 Heuristic MVP 的规则很直接：

- seed 太容易，就加难。
- seed 太难，就降难。
- seed 在中间，就换个类似难度的新场景。

这个规则能跑，但它没有真正比较三个动作。比如一个 seed 的 pass rate 是 0.7，它还没到 easy 档，但适度增加一点复杂度可能会更接近 0.5；第一版仍然会把它当 frontier，选择 `diversify_frontier`。

`calibrated_transfer` 做的事情是：

> 对同一个 seed，把三个原语都想象执行一次，估计它们会把 pass rate 推到哪里，然后选离 0.5 最近的那个。

它不是训练一个预测模型，只使用固定的动作难度迁移量：

- `increase_complexity`：预计让任务更难，所以 pass rate 减少。
- `reduce_complexity`：预计让任务更容易，所以 pass rate 增加。
- `diversify_frontier`：预计保持难度附近，所以 pass rate 基本不变。

默认迁移量是：

\[
\Delta_{\text{increase}}=-0.25,\quad
\Delta_{\text{reduce}}=+0.25,\quad
\Delta_{\text{diversify}}=0
\]

这是一种 calibrated heuristic：它仍然是启发式，但相比固定分桶规则，它显式估计“动作执行后的任务会不会更靠近学习前沿”。

## 变量

第 \(i\) 个 seed 任务仍然写成：

\[
s_i = (I_i, D_i, Sol_i, T_i, E_i)
\]

含义分别是任务说明、输入/环境数据、参考解法、测试/verifier、Docker 环境。

已有 agent reward 序列为：

\[
R_i = \{r_{i1}, r_{i2}, \dots, r_{in}\}
\]

观测 pass rate 为：

\[
\hat p_i =
\begin{cases}
\frac{1}{|R_i|}\sum_{r \in R_i} r, & |R_i| > 0 \\
0.5, & |R_i| = 0
\end{cases}
\]

动作空间仍然只有三个：

\[
A = \{
\text{increase\_complexity},
\text{reduce\_complexity},
\text{diversify\_frontier}
\}
\]

每个动作有一个固定迁移量：

\[
\Delta_a,\quad a \in A
\]

动作执行后的 pass rate 估计为：

\[
\tilde p_i(a)=\operatorname{clip}(\hat p_i+\Delta_a,0,1)
\]

这里 `clip` 的意思是把数值限制在 \([0,1]\) 内。例如 \(\hat p_i=0.9\)，选择 `increase_complexity`，默认估计是：

\[
\tilde p_i(\text{increase})=\operatorname{clip}(0.9-0.25,0,1)=0.65
\]

学习前沿分数为：

\[
\tilde F_i(a)=\exp\left(-\frac{(\tilde p_i(a)-0.5)^2}{2\sigma^2}\right)
\]

当前实现中 \(\sigma=0.2\)。\(\tilde p_i(a)\) 越接近 0.5，\(\tilde F_i(a)\) 越高。

## 目标函数

`calibrated_transfer` 的生成前目标是：

\[
a_i^\star = \arg\max_{a \in A}\ \tilde F_i(a)
\]

也就是：

> 在三个原语里，选一个预计生成后 pass rate 最接近 0.5 的动作。

然后只生成一个任务：

\[
c_i = G(s_i, g_i, a_i^\star)
\]

其中 \(g_i\) 是从 seed 中抽取的 skill profile，\(G\) 是 LLM 合成器。

这一版不是：

\[
\arg\max_{c \in \mathcal{C}_i} F(c)
\]

因为它不会为一个 seed 先生成多个候选再挑最优。多候选池选择是未来规划里的另一个优化问题，不属于当前 MVP。

## 约束

第一类是验证约束：

\[
V_{\text{gold}}(c_i)=1
\]

生成任务必须通过自己的 gold solution 验证。也就是说，参考解法跑完后，测试脚本必须给出 reward 1。这个约束是四种 baseline 和 Envs-FORGE 都共享的要求。

第二类是组件同步约束：

\[
c_i=(I_i',D_i',Sol_i',T_i',E_i')
\]

如果任务说明改了，输入数据、参考解法、测试和环境也要同步改。特别是 terminal-bench 风格任务中，不能只改 instruction，否则 agent 按新 instruction 做对了，也可能被旧 tests 判错。

第三类是 skill 保持约束：

\[
\text{overlap}(g_i,g_i') \ge \tau
\]

当前代码仍然用关键词标签近似 \(g_i\)，包括 domain、tool、artifact、operation、constraint。它不是完整 skill graph，但足够让 prompt 约束模型不要跳到无关任务。

第四类是测试集隔离约束：

\[
a_i^\star,\ c_i \text{ only depend on TBLite seeds and seed-side execution statistics}
\]

生成前估计只使用 TBLite seed 的历史执行结果，不使用 terminal-bench 测试集内容。这样不会泄露测试集。

第五类是动作空间约束：

\[
a_i^\star \in A,\quad |A|=3
\]

当前版本明确不新增第四、第五个原语。它只是把已有三个原语从“分桶固定选择”升级为“枚举后选择”。

## 和 Heuristic MVP 的区别

Heuristic MVP 使用 band rule：

\[
a_i =
\begin{cases}
\text{increase\_complexity}, & \hat p_i \ge 0.8 \\
\text{reduce\_complexity}, & \hat p_i \le 0.2 \\
\text{diversify\_frontier}, & 0.2 < \hat p_i < 0.8
\end{cases}
\]

`calibrated_transfer` 使用 action transfer：

\[
a_i^\star=\arg\max_{a\in A}
\exp\left(-\frac{(\operatorname{clip}(\hat p_i+\Delta_a,0,1)-0.5)^2}{2\sigma^2}\right)
\]

大白话说，前者是“看 seed 属于哪一档”，后者是“看哪个动作推完以后更像前沿任务”。

## 计算例子

假设某个 seed 的 pass rate 是：

\[
\hat p_i=0.7
\]

Heuristic MVP 中它属于 frontier 档，所以会选择：

\[
a_i=\text{diversify\_frontier}
\]

calibrated transfer 会比较：

\[
\tilde p_i(\text{increase})=0.45
\]

\[
\tilde p_i(\text{reduce})=0.95
\]

\[
\tilde p_i(\text{diversify})=0.7
\]

其中 0.45 离 0.5 最近，所以选择：

\[
a_i^\star=\text{increase\_complexity}
\]

这就是它比第一版更像优化问题的地方：仍然很轻量，但已经从固定规则变成了对动作效果的显式比较。

## 三个真实合成例子

下面三条是这轮 TBLite smoke 中真实物化出来的任务。它们展示的是 Envs-FORGE 的合成过程本身：先用 seed-side 历史 reward 选择 projection，再让 LLM 同步生成 instruction、输入数据、gold solution、tests 和必要的 Docker 环境，最后用 gold solution 跑 verifier。这里的 accepted 指的是 gold verification 通过，不等同于下游 agent 一定能做出来。

| 难度定位 | Seed task | Seed 观测 reward | Projection | 合成任务 | Gold verification |
| --- | --- | ---: | --- | --- | ---: |
| hard | `systemd-log-monitoring` | 0.0 | `reduce_complexity` | `log-event-validator` | 1 |
| frontier | `pgn-chess-repair-puzzles` | 0.5333 | `diversify_frontier` | `pgn-chess-repair-diversified` | 1.0 |
| easy | `jsonl-aggregator` | 1.0 | `increase_complexity` | `jsonl-aggregator-ts-filter` | 1 |

记录来源：`systemd-log-monitoring` 和 `jsonl-aggregator` 来自 `outputs/envs_forge_tblite_calibrated_transfer_smoke10/envs_forge/`；`pgn-chess-repair-puzzles` 来自 `outputs/envs_forge_tblite_mvp_gold_smoke/envs_forge/`。PGN 这条在 band rule 和 calibrated transfer 下都会选择 `diversify_frontier`，因此仍然可以作为三原语流程里的 frontier 例子。

### 1. Hard seed: systemd-log-monitoring -> log-event-validator

原任务要求搭一整套 systemd service、journald、rsyslog、logrotate 和 health check pipeline。这个 seed 的历史 reward 是 0.0，因此 policy 认为它对当前 agent 太难，选择：

\[
a^\star=\text{reduce\_complexity}
\]

合成时保留了原任务的核心 skill：结构化日志、JSON 解析、错误统计、脚本验证；但去掉了 systemd、rsyslog、logrotate 这些环境集成负担。生成后的任务变成：给定 `/opt/data/input.log`，要求写 `/opt/myapp/parse_logs.py`，逐行验证 JSON log，统计 valid/invalid、INFO/WARN/ERROR 数量，并收集 ERROR timestamp。

四大件同步变化如下：

- instruction：从“部署完整日志管线”改为“实现一个可执行的日志验证脚本”。
- 输入数据：从服务运行和日志系统状态，改为确定性的 `/opt/data/input.log`，其中包含合法 JSON、非法 JSON、空行和缺字段样例。
- solution：生成 `/opt/myapp/parse_logs.py`，只用 Python 标准库完成解析和统计。
- tests：检查脚本存在且可执行，验证固定 fixture 的精确 JSON 输出，并增加 empty file 和 all invalid 的边界测试。

这个例子符合预期：它不是随便换题，而是把 hard 环境集成任务压缩成同 skill family 下的可执行、可验证日志解析任务。

### 2. Frontier seed: pgn-chess-repair-puzzles -> pgn-chess-repair-diversified

原任务要求修复损坏 PGN、分析关键局面，并解 FEN puzzle。seed 的历史 reward 是 0.5333，本来就在 0.5 附近，因此 policy 选择：

\[
a^\star=\text{diversify\_frontier}
\]

合成任务保留 PGN repair + puzzle solving 的主结构，但改了具体数据和约束：输入变成 3 局带 intentional corruptions 的 PGN 和 5 个 puzzle；instruction 明确列出 corruption 类型，包括 illegal move、missing capture、impossible castling、off-board rank；测试则用 `expected_analysis.json` 和 grader 给 `pgn_repair`、`error_detection`、`puzzle_solving` 三个子分数。

四大件同步变化如下：

- instruction：保持“修 PGN + 解 puzzle”的任务类型，但换成更明确的数据规模、corruption 类型和输出 schema。
- 输入数据：同步生成新的 `games.pgn` 和 `puzzles.json`，避免只改 instruction。
- solution：使用 `python-chess` 解析和修复棋谱，并对 5 个 puzzle 给出 expected best move/continuation。
- tests：不只检查 JSON schema，还用 grader 分项评估 PGN 修复、错误检测和 puzzle 解答；该 accepted 版本的 gold reward 为 1.0，详细分数是 `pgn_repair=1.0`、`error_detection=1.0`、`puzzle_solving=1.0`。

这个例子也符合预期：它没有明显升难或降难，而是在相同 skill cluster 里换一组可验证棋局和 puzzle，让任务仍然停在 learning frontier 附近。

### 3. Easy seed: jsonl-aggregator -> jsonl-aggregator-ts-filter

原任务是聚合多个 JSONL 文件，输出 top users 和 top tags。seed 历史 reward 是 1.0，因此 policy 认为它太容易，选择：

\[
a^\star=\text{increase\_complexity}
\]

合成任务保留 JSONL 聚合的核心，但增加了时间过滤和确定性排序约束：每条记录新增 `ts` 字段，只统计 `ts >= 1700000000` 的记录；用户排序从单一金额排序变成 total amount、total items、username 三级 tie-break；tag 排序也增加 count、tag name 的 deterministic tie-break；根节点 key order 也被测试约束。

四大件同步变化如下：

- instruction：从“直接聚合所有 JSONL”升级为“过滤时间窗口后聚合，并满足确定性排序和 key order”。
- 输入数据：合成 `records_1.jsonl`、`records_2.jsonl` 的 deterministic fixture，包含时间戳、金额、items、tags。
- solution：用 Python 读取记录、过滤 `ts`、累加用户和 tag，再按多级排序输出 `/app/aggregates.json`。
- tests：检查 root key order、top 5 users、top 5 tags、金额 rounding 和整数类型。

这个例子同样符合预期：它在原始 JSONL aggregation skill 上增加过滤、tie-break 和格式约束，属于典型 easy-to-frontier 的复杂化。

### 下游 qwen3-8b smoke 结果的解读

后续用 `terminus-2 + qwen3-8b` 做 180 秒预算的单次 smoke 时，三组 before/after 的 reward 没有呈现出稳定推向 0.5 的趋势。这不应该直接解读为这三条合成任务不符合预期，因为 gold verification 已经证明任务自身是可解且测试同步的；更合理的解释是：当前固定 downstream policy 太弱或预算太短，导致 reward 估计没有校准好。例如 qwen3-8b 在原始 `jsonl-aggregator` 上反复写错 `jq` 表达式而 timeout，但在合成后的 Python 解法路径上反而拿到 1.0。这个现象说明后续做 frontier optimization 时，需要先用同一个 downstream policy 做小规模校准，再用它的 reward 估计驱动 projection。

## 局限与下一步

这个版本仍然有几个限制：

- \(\Delta_a\) 是人工设置的固定迁移量，不是从大量历史数据中自动校准出来的。
- 它只预测动作方向对 pass rate 的平均影响，不知道某个具体 seed 的结构性风险。
- 它只做生成前单动作选择，不做一个 seed 多候选生成后的全局 selection。
- skill profile 仍然是关键词标签，不是完整 skill graph。

下一步可以把 \(\Delta_a\) 从经验常数升级为按 domain/tool/action 分组统计得到的 calibrated table，例如 systemd 任务的 `reduce_complexity` 可能比 csv 任务的 `reduce_complexity` 带来更大的 pass-rate 提升。但这仍然可以先保持非 learned surrogate，只用历史执行统计做校准。
