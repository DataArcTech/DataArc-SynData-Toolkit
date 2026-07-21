# Agentic Data 合成 Prompt 与三原语对齐说明

这份文档整理两个问题：

- 从 prompt 角度看，few-shot、self-instruct、evol-instruct 到底有什么差异。
- Envs-FORGE 当前三个合成原语的 prompt 是什么，它们能否覆盖四种 baseline 操作。

这里的“四种 baseline 操作”指：

- `few_shot`
- `self_instruct`
- `evol_instruct_in_depth`
- `evol_instruct_in_breadth`

## 1. 从 Prompt 看 Four Baselines

### 1.1 few-shot

普通 DataArc 文本合成里，`few_shot` 使用 `SDGDistillation`。它的 prompt 信息量最多：

- 有 `Task Instruction`。
- 有 `Input Format Instruction`。
- 有 `Output Format Instruction`。
- 可以有从 demo 中抽取的 `Extracted Patterns`。
- 有 demonstration examples。
- 要求生成 `DIVERSE training examples`。
- 明确说可以用 pattern/example 做 guidance，但要生成新内容，不能 copy/paraphrase demo。

大白话说，few-shot 是“按例子和模式仿写”。它最强的约束是结构约束和风格约束，所以通常生成的是 close variant。

terminal-bench 版本里的 few-shot 语义更直接：

```text
Create a close variant that preserves the seed task structure while changing concrete requirements and verifier expectations.
```

所以在 agentic terminal-bench 场景里，few-shot 的核心操作是：

\[
\text{preserve structure} + \text{change concrete requirements/verifier expectations}
\]

它不是显式“加深”或“拓宽”，而是围绕 seed 结构做局部变化。

### 1.2 self-instruct

`self_instruct` 使用 `SelfInstructDistillation`。它的 prompt 比 few-shot 少一些结构锚点：

- 有 `Task Instruction`。
- 有 demonstration examples。
- 要求生成 `DIVERSE training examples`。
- 要求 `cover DIFFERENT aspects of the task`。
- 要求 examples detailed、comprehensive、high-quality。
- 明确说用 examples 做 guidance，但生成 completely new and different examples。

普通 self-instruct 没有 few-shot 那种显式 input/output format section，也没有 extracted patterns section。因此它更像“同域自举生成”，比 few-shot 更自由。

terminal-bench 版本里的 self-instruct 语义是：

```text
Invent a related terminal task in the same domain with its own task description, solution, and verifier.
```

所以 self-instruct 的核心操作是：

\[
\text{same domain} + \text{invent related task} + \text{own solution/verifier}
\]

它可能生成加深任务，也可能生成拓宽任务，但 prompt 本身没有指定方向。也就是说，self-instruct 有可能隐式 deepen 或 broaden，但不稳定、不可控。

### 1.3 evol-instruct in-depth

`evol_instruct_in_depth` 使用 Evol-Instruct 的 depth prompt。它最明确的目标是：

```text
perform IN-DEPTH EVOLUTION
rewrite or extend them into more COMPLEX and CHALLENGING versions
```

它列出的加深方式包括：

- add constraints
- deepen reasoning scope
- concretize abstract descriptions
- increase reasoning steps
- complicate input with structured data, code snippets, tables, formulas

terminal-bench 版本里的 depth 语义是：

```text
Deepen the seed by adding a harder constraint or edge case and update solution and tests accordingly.
```

所以 evol-depth 的核心操作是：

\[
\text{same core topic} + \text{harder constraint/edge case} + \text{more reasoning/input complexity}
\]

它和 Envs-FORGE 的 `increase_complexity` 基本一一对应。

### 1.4 evol-instruct in-breadth

`evol_instruct_in_breadth` 使用 Evol-Instruct 的 breadth prompt。它最明确的目标是：

```text
perform IN-BREADTH EVOLUTION
create NEW but RELATED instructions that explore different directions or perspectives of the same general domain
```

它列出的拓宽方式包括：

- another subtask or variation of the same problem type
- a new angle, user intent, or application scenario
- a different format, such as transformation, critique, generation, explanation
- keep difficulty and length roughly similar to base examples

terminal-bench 版本里的 breadth 语义是：

```text
Broaden the seed into a neighboring task type while keeping the same terminal-bench directory contract.
```

所以 evol-breadth 的核心操作是：

\[
\text{same general domain} + \text{neighboring task type} + \text{similar difficulty}
\]

它和 Envs-FORGE 的 `diversify_frontier` 高度接近，但不是完全等价：breadth 更强调“邻近任务类型”，`diversify_frontier` 更强调“保持学习前沿附近，不要跳到无关方向”。

## 2. Envs-FORGE 三个合成原语的 Prompt

Envs-FORGE 先根据 seed 的 agent pass rate 或 calibrated transfer policy 选择一个 projection，然后把 projection 写入 prompt。

共同的生成前约束包括：

- 生成一个完整 terminal-bench-style task directory。
- 返回 JSON，包含 `task_name`、`instruction_md`、`files`、`metadata`。
- 必须同步更新 `instruction.md`、`solution/solve.sh`、`tests/test.sh`、`tests/test_outputs.py`、`environment/Dockerfile`。
- 不允许只改 instruction。
- 不允许隐藏测试约束。
- `tests/test.sh` 必须总是写 `/logs/verifier/reward.txt`。
- Dockerfile 不能从 task root 复制 `tests/`、`solution/`、`instruction.md`、`task.toml`。
- 生成前要“脑内执行”：build environment、run solution、run tests，结果应为 reward 1。

也就是说，Envs-FORGE 的原语不是单独一句“变难/变简单/换方向”，而是：

\[
\text{projection primitive} + \text{component consistency constraints} + \text{gold verification constraints}
\]

### 2.1 increase_complexity

实际 projection instruction：

```text
The seed appears easy for the current agent. Increase complexity with one or two concrete constraints, extra edge cases, stricter output requirements, or a larger fixture while preserving the seed skill subgraph.
```

核心含义：

\[
\text{easy seed} \rightarrow \text{harder medium/frontier task}
\]

可用操作：

- 增加一两个具体约束。
- 加 edge cases。
- 加 stricter output requirements。
- 加 larger fixture。
- 保持 seed skill subgraph。

它覆盖的是 evol-depth 方向。

### 2.2 reduce_complexity

实际 projection instruction：

```text
The seed appears beyond the current agent. Reduce compositional complexity, split away secondary systems, or create a smaller bridge task while preserving the target skill subgraph and its most important artifacts.
```

核心含义：

\[
\text{hard seed} \rightarrow \text{easier bridge task}
\]

可用操作：

- 降低组合复杂度。
- 拆掉次要系统。
- 创建更小的 bridge task。
- 保留核心 skill subgraph 和关键 artifacts。

这个原语不是四种 baseline 里的传统操作。它是 Envs-FORGE 为“hard seed 拉回学习前沿”额外加入的方向。

### 2.3 diversify_frontier

实际 projection instruction：

```text
The seed is near the learning frontier. Keep difficulty similar, diversify data and constraints, and avoid large simplification or unrelated breadth jumps.
```

核心含义：

\[
\text{frontier seed} \rightarrow \text{similar difficulty, different concrete task}
\]

可用操作：

- 保持难度相近。
- 换数据。
- 换约束。
- 换 fixture。
- 避免大幅简化。
- 避免跳到无关 breadth。

它覆盖的是 few-shot、self-instruct、evol-breadth 的共同可控子集：同域、相近难度、具体变化。

## 3. 三原语能否覆盖四种 Baseline 操作

结论：可以覆盖主要操作，但不是一比一等价。

| Baseline 操作 | Prompt 核心 | 对应 Envs-FORGE 原语 | 覆盖关系 |
|---|---|---|---|
| `few_shot` | preserve structure, change concrete requirements/verifier | `diversify_frontier` | few-shot 是更窄的 diversify，强调结构贴近 seed |
| `self_instruct` | same domain, invent related task | `diversify_frontier` | self-instruct 更自由，diversify 是受 skill/frontier 约束的 self-instruct |
| `evol_instruct_in_depth` | add constraints, deepen, complicate input | `increase_complexity` | 基本一一对应 |
| `evol_instruct_in_breadth` | neighboring task type, new angle, similar difficulty | `diversify_frontier` | 高度接近，但 Envs-FORGE 更限制“不要无关 breadth jump” |

`reduce_complexity` 是额外能力。它不对应传统四 baseline 中的某一个，但在 Envs-FORGE 里很重要，因为 agentic RL 环境合成不是只追求多样性或更难，而是要把任务投影到当前 agent 的学习前沿。

## 4. 为什么三原语足够做当前 MVP

当前 MVP 的优化问题是生成前选择一个 projection：

\[
a_i^\star \in \{
\text{increase\_complexity},
\text{reduce\_complexity},
\text{diversify\_frontier}
\}
\]

这三个方向刚好覆盖学习前沿调度里的三种基本情况：

- seed 太容易：用 `increase_complexity`。
- seed 太难：用 `reduce_complexity`。
- seed 接近前沿：用 `diversify_frontier`。

从 baseline 角度看：

- `increase_complexity` 覆盖 depth。
- `diversify_frontier` 覆盖 close variant、self-instruct related task、breadth variant。
- `reduce_complexity` 补足传统 baseline 没有的 hard-to-frontier 投影。

所以当前三原语不需要扩展成四个或五个动作，也能表达四种 baseline 的主要合成方向。

## 5. 关键差异：Baseline Prompt 是生成策略，Envs-FORGE 原语是调度动作

四种 baseline 更像“怎么生成”：

- few-shot：照着例子生成。
- self-instruct：同域自举生成。
- evol-depth：往深处变难。
- evol-breadth：往邻近方向拓宽。

Envs-FORGE 三原语更像“根据当前 agent 能力，选择往哪里投影”：

- increase：太容易，推难。
- reduce：太难，降到桥接任务。
- diversify：刚好在前沿附近，换数据和场景。

因此，Envs-FORGE 可以看成 baseline 生成方法之前的一层 action policy，也可以看成 baseline 生成方法之后的一层 verified projection contract。

更准确地说：

\[
\text{baseline prompt} = \text{candidate generation style}
\]

\[
\text{Envs-FORGE primitive} = \text{frontier-aware projection action}
\]

二者并不冲突。few-shot/self-instruct/evol 都可以接入 Envs-FORGE 的生成前约束和生成后验证，只是当前 MVP 先把 Envs-FORGE 做成一套独立的三原语 policy。

## 6. 对后续实现的启发

如果后续要把 Envs-FORGE 适配回三类 baseline，可以有两种实现方式。

第一种是 prompt-level adapter：

| Envs-FORGE 原语 | few-shot adapter | self-instruct adapter | evol adapter |
|---|---|---|---|
| `increase_complexity` | close variant + harder verifier/edge case | related task + harder constraint | evol-depth |
| `reduce_complexity` | close variant + smaller fixture/bridge verifier | related bridge task in same domain | inverse-depth / simplification adapter |
| `diversify_frontier` | close variant + changed data/requirements | related same-domain task | evol-breadth |

第二种是 pipeline-level adapter：

1. 先用 Envs-FORGE policy 根据 pass rate 选择 action。
2. 再把 action 填进 few-shot/self-instruct/evol 的 strategy prompt。
3. 生成完整 terminal-bench artifact。
4. 做 schema validation。
5. 做 gold solution validation。
6. 只接受 reward 1 的 verified task。

当前先不实现这层 adapter，因为 MVP 重点是证明三原语的 frontier-aware projection 能跑通；baseline adapter 可以作为下一阶段工程化扩展。
