# Envs-FORGE MILP Joint Optimization

这份文档记录 Envs-FORGE 的 `mixed_integer_programming` 版本。它保留 `projection_policy="joint_optimization"` 作为兼容入口，在 BIP alpha 的基础上做两点扩展：

1. action 从 3 个 projection 扩展为 `projection x evolution_direction`。
2. skill coverage 从硬约束扩展为 soft coverage，用连续 slack 避免 coverage 要求不可达时直接 infeasible。

## 大白话版本

BIP alpha 只在每个 seed 的 3 个 projection 里选：

- `increase_complexity`
- `reduce_complexity`
- `diversify_frontier`

MILP 版把每个 projection 再拆成两个演化方向：

- `in_depth`：沿同一技能链加深，强化约束、边界条件、结构化输出等。
- `in_breadth`：在相近技能邻域扩宽，换 fixture、换近邻约束、换任务场景但不大幅改变难度。

因此一个候选 action 变成：

```text
increase_complexity:in_depth
increase_complexity:in_breadth
reduce_complexity:in_depth
reduce_complexity:in_breadth
diversify_frontier:in_depth
diversify_frontier:in_breadth
```

Soft coverage 的直觉是：我们仍然希望 batch 覆盖指定 skill，比如至少 1 条 `system_admin`，但如果当前 seed 池里根本没有可行任务，就不要让 solver 崩掉，而是记录 coverage gap，并在目标函数里扣分。

## 和四种 baseline 的关系

把 action 拆成 `projection x evolution_direction` 后，一个很有用的观察是：之前的四种 agentic data 合成 baseline，其实都可以看成这个 action space 里的固定策略或受限策略。

| 方法 | 在 MILP action space 中的近似位置 | 直觉 |
|---|---|---|
| `few_shot` | 通常接近 `diversify_frontier:in_depth` | few-shot 主要模仿种子任务形式，通常不主动换任务族，而是在同一结构里补细节、补约束、补实例。 |
| `self_instruct` | 通常接近 `diversify_frontier:in_breadth` | self-instruct 更像从 seed 周围发散出相邻任务，保留大致难度，但更容易换场景、换 fixture、换具体请求。 |
| `evol-depth` | 显式是 `*:in_depth` | depth 是方向约束：projection 可以是 increase/reduce/diversify，但演化方向固定为加深。 |
| `evol-breadth` | 显式是 `*:in_breadth` | breadth 也是方向约束：projection 可以变化，但演化方向固定为扩宽。 |

这也是 MILP 版相比四个 baseline 更完整的地方：baseline 是固定 policy，FORGE-MILP 把这些 policy 放到同一个候选空间里，让 solver 根据 learning frontier score 和 skill coverage 选择当前 batch 最合适的 `(seed, projection, direction)`。

## Variables

第 \(i\) 个 seed task 写成：

\[
s_i=(I_i,D_i,Sol_i,T_i,E_i)
\]

projection 集合：

\[
A=\{\text{increase},\text{reduce},\text{diversify}\}
\]

evolution direction 集合：

\[
D=\{\text{in\_depth},\text{in\_breadth}\}
\]

候选 action 是笛卡尔积：

\[
c_{i,a,d}=(s_i,a,d),\quad a\in A,\ d\in D
\]

四种 baseline 可以写成 action mask。设 baseline 集合为：

\[
B=\{\text{few\_shot},\text{self\_instruct},\text{evol\_depth},\text{evol\_breadth}\}
\]

定义常量 mask：

\[
\rho_{b,a,d}\in\{0,1\}
\]

表示 baseline \(b\) 是否允许 action variant \((a,d)\)。近似地：

\[
\rho_{\text{few\_shot},a,d}=1
\iff
(a,d)=(\text{diversify},\text{in\_depth})
\]

\[
\rho_{\text{self\_instruct},a,d}=1
\iff
(a,d)=(\text{diversify},\text{in\_breadth})
\]

\[
\rho_{\text{evol\_depth},a,d}=1
\iff
d=\text{in\_depth}
\]

\[
\rho_{\text{evol\_breadth},a,d}=1
\iff
d=\text{in\_breadth}
\]

注意 \(\rho\) 是 baseline 分析用的常量，不是默认 FORGE-MILP 的自由变量。默认 FORGE-MILP 直接在完整 \((a,d)\) 空间上优化。

选择变量：

\[
x_{i,a,d}\in\{0,1\}
\]

表示是否选择 seed \(i\) 的 projection \(a\) 和 direction \(d\)。

skill node 变量：

\[
u_{i,a,d,v}\in\{0,1\}
\]

表示候选 \(c_{i,a,d}\) 是否激活 skill node \(v\)。

soft coverage slack：

\[
\xi_v\ge 0
\]

表示 skill \(v\) 的覆盖缺口。它是连续变量，所以这个版本是 MILP，而不是纯整数规划。

## Objective

仍然先根据 seed reward 估计 action 后的 pass rate：

\[
\tilde p_{i,a,d}
=\operatorname{clip}(\hat p_i+\Delta_a\cdot \gamma_d,0,1)
\]

当前默认：

\[
\Delta_{\text{increase}}=-0.25,\quad
\Delta_{\text{reduce}}=0.25,\quad
\Delta_{\text{diversify}}=0
\]

\[
\gamma_{\text{in\_depth}}=1.0,\quad
\gamma_{\text{in\_breadth}}=0.65
\]

Learning frontier score：

\[
F(\tilde p)=
\exp\left(
-\frac{(\tilde p-0.5)^2}{2\sigma^2}
\right)
\]

MILP 目标函数：

\[
\max_{x,u,\xi}
\sum_{i,a,d}x_{i,a,d}F(\tilde p_{i,a,d})
-\epsilon\sum_{i,a,d}\sum_{v\in O_{i,a,d}}u_{i,a,d,v}
-\lambda\sum_v \xi_v
\]

其中：

- \(\epsilon=10^{-6}\)，只用于避免 solver 无意义打开可选 skill node。
- \(\lambda=0.25\)，用于惩罚 soft coverage 缺口。
- \(O_{i,a,d}\) 是 allowed added / allowed removed skill nodes。

大白话说：优先选更接近 0.5 pass rate 的合成动作；如果能覆盖目标 skill，就尽量覆盖；如果覆盖不了，也可以给出解，但会明确记录缺口。

四种 baseline 在这个目标函数下可以看成“先用 \(\rho_b\) 把候选空间裁掉，再做选择”。例如：

- `few_shot` 只允许 `diversify_frontier:in_depth`，所以目标函数只能在这个固定方向里选 seed。
- `self_instruct` 只允许 `diversify_frontier:in_breadth`，所以它更像同难度邻域扩宽。
- `evol-depth` 允许所有 projection，但只允许 `in_depth`。
- `evol-breadth` 允许所有 projection，但只允许 `in_breadth`。

FORGE-MILP 不预先绑定某个 baseline，而是让 solver 在这四类行为都覆盖到的 action space 中选择。

## Constraints

每个 seed 最多选一个 action variant：

\[
\sum_{a,d}x_{i,a,d}\le 1
\]

batch 总共选 \(N\) 条：

\[
\sum_i\sum_{a,d}x_{i,a,d}=N
\]

required skill nodes 必须随候选一起打开：

\[
u_{i,a,d,v}=x_{i,a,d},\quad v\in R_{i,a,d}
\]

optional skill nodes 可以打开或关闭：

\[
u_{i,a,d,v}\le x_{i,a,d},\quad v\in O_{i,a,d}
\]

其他不允许的 skill nodes 不能打开：

\[
u_{i,a,d,v}=0,\quad v\notin R_{i,a,d}\cup O_{i,a,d}
\]

soft skill coverage：

\[
\sum_{i,a,d}u_{i,a,d,v}+\xi_v\ge m_v
\]

其中 \(m_v\) 是用户指定的最小覆盖数量。相比 BIP alpha 的硬约束，MILP 版允许 \(\xi_v>0\)，所以不会因为 coverage 不可达而无解。

如果要把某个 baseline 写成 FORGE-MILP 的特例，只需要加一个 baseline mask 约束：

\[
x_{i,a,d}\le \rho_{b,a,d},\quad \forall i,a,d
\]

当 \(b=\text{few\_shot}\) 时，这条约束会把所有非 `diversify_frontier:in_depth` 的候选都关掉；当 \(b=\text{evol\_breadth}\) 时，它会把所有 `in_depth` 候选关掉。

所以四种 baseline 不是 FORGE-MILP 之外的东西，而是它的受限可行域：

\[
\mathcal X_b
=
\{x\in\mathcal X_{\text{FORGE}}:
x_{i,a,d}\le \rho_{b,a,d}\}
\]

默认 FORGE-MILP 使用完整可行域 \(\mathcal X_{\text{FORGE}}\)，也就是不加这条 baseline mask，让模型自己决定当前 batch 更应该加深、扩宽、增难、降难还是同难度多样化。

这给了一个很清晰的比较视角：baseline 的差异主要来自 prompt policy 固定，而 FORGE-MILP 的差异来自可验证 reward 估计、skill coverage 和 action variant 的联合选择。

## 当前实现范围

进入 solver 的变量：

- \(x_{i,a,d}\)：选择哪个 seed/action/direction。
- \(u_{i,a,d,v}\)：选择哪些 skill nodes。
- \(\xi_v\)：soft coverage gap。

没有进入 solver 的部分：

- `artifact_edit_mask` 仍然由 action 派生，写进 prompt 和 metadata。
- `instruction/data/solution/tests/environment` 的同步一致性仍然通过生成前约束和生成后 gold verification 保证。
- baseline mask \(\rho_b\) 当前用于方法解释和对比建模；默认全量 FORGE-MILP 不启用它，因为我们的目标是从完整 action space 中选择最接近 learning frontier 的 verified task。

这个取舍是故意的：artifact 之间有强语义依赖，当前不靠人工枚举 implication rules 伪装成优化约束。

## 运行示例

```bash
.venv/bin/python examples/syn_agentic_data/run_envs_forge_tblite.py \
  --source-root /Volumes/PSSD/code/My-Agent/Agent-Data/OpenThoughts-TBLite \
  --output-root outputs/envs_forge_tblite_milp \
  --target-count 10 \
  --projection-policy joint_optimization \
  --optimization-formulation mixed_integer_programming \
  --min-skill-coverage system_admin=1 \
  --dry-run
```

输出会包含：

- `optimization_formulation: mixed_integer_programming`
- `action_variant`
- `evolution_direction`
- `coverage_slack`
- PySCIPOpt 的 `selected_x / selected_u / objective_value`
