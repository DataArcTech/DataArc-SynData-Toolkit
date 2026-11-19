"""Envs-FORGE v1: per-task component-edit MILP.

The alpha selector picks one of six preset actions whose difficulty effect is an
exogenous constant. That makes the objective linear over a simplex, so the LP
relaxation is integral and the model collapses to an argmax. v1 moves the
decision variable down one level: instead of choosing an action, it decides
which task components to edit (binary) and how strongly (continuous), with a
fixed charge per touched component. The fixed charge plus the big-M coupling
`e_v <= e_max * y_v` is what makes the LP relaxation fractional, so the model is
a genuine mixed-integer program rather than a relabelled argmax.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Sequence

from .terminal_bench import TerminalBenchSeedTask, _include_seed_file, _shorten

METHOD_VERSION = "envs-forge-v1"

EDIT_COMPONENTS: tuple[str, ...] = ("instruction", "data", "solution", "tests", "environment")

COMPONENT_LABELS: dict[str, str] = {
    "instruction": "task statement",
    "data": "fixture and data bundle",
    "solution": "oracle solution",
    "tests": "verifier tests",
    "environment": "execution environment",
}

# Difficulty sensitivity in log-odds units per unit of edit strength.
# Sum over all five components equals one full alpha step (0.25 probability at
# p=0.5 is about 1.1 nats), so a v1 plan that touches everything at full
# strength moves a seed as far as one alpha projection does.
DEFAULT_SENSITIVITY: dict[str, float] = {
    "instruction": 0.30,
    "data": 0.20,
    "solution": 0.20,
    "tests": 0.30,
    "environment": 0.10,
}
DEFAULT_E_MAX = 1.0
DEFAULT_LINEAR_COST = 1.0
# |z| in log-odds must land inside this band to count as being on the frontier.
FRONTIER_BAND_NATS = 0.4
# Weight on band violation. It dominates any realistic edit cost, so the model
# first reaches the frontier and only then minimises cost; when the frontier is
# unreachable it minimises the residual distance instead of going infeasible.
BAND_SLACK_PENALTY = 100.0

# If a component changes, these must change with it or the task stops being
# gradeable. Changing the statement without touching the tests leaves new
# requirements unchecked; changing the solution without touching the tests
# leaves the verifier asserting the old output shape.
CONSISTENCY_IMPLICATIONS: tuple[tuple[str, str], ...] = (
    ("instruction", "tests"),
    ("solution", "tests"),
    ("data", "tests"),
)

# A seed already sitting on the frontier still has to produce a new environment,
# so at least one component must be touched; a diversifying edit must change the
# concrete requirements or the fixtures, not just the verifier.
DIVERSIFY_ANCHORS: tuple[str, ...] = ("instruction", "data")

NET_DIRECTION_THRESHOLD = 0.05
DEPTH_COMPONENTS: frozenset[str] = frozenset({"instruction", "tests"})


@dataclass(frozen=True)
class ComponentEditPlan:
    seed_name: str
    n_probes: int
    pass_rate_raw: float
    pass_rate_smoothed: float
    z0: float
    touched: tuple[str, ...]
    strengths: dict[str, float]
    fixed_cost: float
    linear_cost: float
    cost: float
    delta_logit: float
    z_predicted: float
    band_violation: float
    reachable: bool
    rounds_needed: int
    projection: str
    evolution_direction: str
    backend: str
    solver_result: dict[str, object] = field(default_factory=dict)

    @property
    def edit_strength(self) -> float:
        return sum(self.strengths.values())

    def as_json(self) -> dict[str, object]:
        return {
            "method_version": METHOD_VERSION,
            "seed_name": self.seed_name,
            "n_probes": self.n_probes,
            "pass_rate_raw": self.pass_rate_raw,
            "pass_rate_smoothed": self.pass_rate_smoothed,
            "z0": self.z0,
            "touched": self.touched,
            "strengths": dict(sorted(self.strengths.items())),
            "fixed_cost": self.fixed_cost,
            "linear_cost": self.linear_cost,
            "cost": self.cost,
            "delta_logit": self.delta_logit,
            "z_predicted": self.z_predicted,
            "band_violation": self.band_violation,
            "reachable": self.reachable,
            "rounds_needed": self.rounds_needed,
            "projection": self.projection,
            "evolution_direction": self.evolution_direction,
            "backend": self.backend,
            "solver_result": self.solver_result,
        }


def component_of_path(path: str) -> str | None:
    """Map a generated file path onto the edit component it belongs to."""
    if path == "instruction.md":
        return "instruction"
    if path.startswith("solution/"):
        return "solution"
    if path.startswith("tests/"):
        return "tests"
    if path == "environment/Dockerfile":
        return "environment"
    if path.startswith("environment/"):
        return "data"
    return None


def allowed_components(seed: TerminalBenchSeedTask) -> tuple[str, ...]:
    present = {component_of_path(path) for path in seed.files}
    return tuple(component for component in EDIT_COMPONENTS if component in present)


def estimate_component_costs(seed: TerminalBenchSeedTask) -> dict[str, float]:
    """Fixed regeneration cost per component.

    Rewriting a component means regenerating it, rebuilding the image, and
    re-verifying the task, so the fixed charge tracks how much machinery the
    component drags along: the verifier and the image cost more than the
    statement, and a larger fixture bundle costs more than a single file.
    """
    dockerfile = seed.files.get("environment/Dockerfile", "")
    costs = {
        "instruction": 1.0,
        "solution": 1.0,
        "tests": 1.5 if "tests/test_outputs.py" in seed.files else 2.0,
        "environment": 3.0 if re.search(r"apt-get install|pip install|npm install", dockerfile) else 2.0,
    }
    data_files = [path for path in seed.files if component_of_path(path) == "data"]
    costs["data"] = 1.0 + 0.25 * len(data_files)
    return costs


def smoothed_pass_rate(rewards: Sequence[float], *, alpha: float = 1.0, beta: float = 1.0) -> float:
    """Laplace-smoothed pass rate.

    With one probe per seed the raw mean is exactly 0 or 1 for most seeds, the
    log-odds are infinite, and every such seed needs an unbounded edit. The
    prior keeps z0 finite so the edit budget has something to act on.
    """
    passes = sum(1 for reward in rewards if float(reward) >= 1.0)
    return (alpha + passes) / (alpha + beta + len(rewards))


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _movement_capacity(
    allowed: Sequence[str],
    sensitivity: dict[str, float],
    e_max: float,
) -> float:
    return sum(sensitivity[component] * e_max for component in allowed)


def _rounds_needed(z0: float, capacity: float, band: float) -> int:
    needed = max(0.0, abs(z0) - band)
    if needed <= 1e-9:
        return 1
    if capacity <= 1e-9:
        return -1
    return int(math.ceil(needed / capacity))


def _derive_labels(touched: Sequence[str], strengths: dict[str, float], delta_logit: float) -> tuple[str, str]:
    if delta_logit > NET_DIRECTION_THRESHOLD:
        projection = "increase_complexity"
    elif delta_logit < -NET_DIRECTION_THRESHOLD:
        projection = "reduce_complexity"
    else:
        projection = "diversify_frontier"
    if set(touched) <= DEPTH_COMPONENTS:
        direction = "in_depth"
    else:
        direction = "in_breadth"
    return projection, direction


def _feasible_subsets(allowed: Sequence[str], require_anchor: bool) -> list[tuple[str, ...]]:
    allowed_set = set(allowed)
    implications = [
        (source, target)
        for source, target in CONSISTENCY_IMPLICATIONS
        if source in allowed_set and target in allowed_set
    ]
    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(allowed) + 1):
        for combo in combinations(allowed, size):
            chosen = set(combo)
            if any(source in chosen and target not in chosen for source, target in implications):
                continue
            if require_anchor and not (chosen & (set(DIVERSIFY_ANCHORS) & allowed_set)):
                continue
            subsets.append(combo)
    return subsets


def _greedy_allocation(
    subset: Sequence[str],
    required: float,
    sensitivity: dict[str, float],
    e_max: float,
) -> tuple[dict[str, float], float] | None:
    """Minimise sum|e_v| over `subset` subject to sum(alpha_v e_v) = required.

    With one equality and box constraints the optimum fills the components with
    the best sensitivity-per-unit-cost ratio first, so the greedy pass is exact.
    Returns None when the subset cannot supply the movement.
    """
    if abs(required) <= 1e-12:
        return {component: 0.0 for component in subset}, 0.0
    sign = 1.0 if required > 0 else -1.0
    ordered = sorted(subset, key=lambda component: sensitivity[component], reverse=True)
    remaining = abs(required)
    strengths = {component: 0.0 for component in subset}
    for component in ordered:
        if remaining <= 1e-12:
            break
        available = sensitivity[component] * e_max
        take = min(available, remaining)
        strengths[component] = sign * (take / sensitivity[component])
        remaining -= take
    if remaining > 1e-9:
        return None
    return strengths, remaining


def _solve_with_enumeration(
    allowed: Sequence[str],
    *,
    z0: float,
    sensitivity: dict[str, float],
    fixed_cost: dict[str, float],
    linear_cost: dict[str, float],
    e_max: float,
    band: float,
) -> ComponentEditPlan | None:
    require_anchor = abs(z0) <= band
    target = max(0.0, abs(z0) - band)
    direction = 1.0 if z0 >= 0 else -1.0
    best: tuple[float, float, tuple[str, ...], dict[str, float], float, float, float] | None = None
    for subset in _feasible_subsets(allowed, require_anchor):
        fixed = sum(fixed_cost[component] for component in subset)
        if target <= 1e-12:
            strengths = {component: 0.0 for component in subset}
            violation = 0.0
            achieved = 0.0
        else:
            allocation = _greedy_allocation(subset, target * direction, sensitivity, e_max)
            if allocation is None:
                capacity = sum(sensitivity[component] * e_max for component in subset)
                strengths = {component: direction * e_max for component in subset}
                achieved = capacity
                violation = abs(target - capacity)
            else:
                strengths, _residual = allocation
                achieved = target
                violation = 0.0
        variable = sum(linear_cost[component] * abs(strengths.get(component, 0.0)) for component in subset)
        score = (BAND_SLACK_PENALTY * violation + fixed + variable, fixed + variable, subset)
        candidate = (score[0], fixed, subset, strengths, variable, achieved, violation)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        return None
    _total, fixed, subset, strengths, variable, achieved, violation = best
    delta_logit = direction * achieved
    return _finalize_plan(
        seed_name="",
        n_probes=0,
        pass_rate_raw=0.0,
        pass_rate_smoothed=0.0,
        z0=z0,
        touched=subset,
        strengths=strengths,
        fixed_cost=fixed,
        linear_cost=variable,
        delta_logit=delta_logit,
        band_violation=violation,
        band=band,
        capacity=_movement_capacity(allowed, sensitivity, e_max),
        backend="exact_enumeration",
        solver_result={
            "backend": "exact_enumeration",
            "feasible_subsets": len(_feasible_subsets(allowed, require_anchor)),
            "unit_capacity": 2 ** len(allowed),
        },
    )


def _solve_with_pyscipopt(
    allowed: Sequence[str],
    *,
    z0: float,
    sensitivity: dict[str, float],
    fixed_cost: dict[str, float],
    linear_cost: dict[str, float],
    e_max: float,
    band: float,
) -> ComponentEditPlan | None:
    try:
        from pyscipopt import Model, quicksum  # type: ignore
    except Exception:
        return None

    model = Model("envs_forge_v1")
    model.hideOutput()
    touch = {component: model.addVar(vtype="B", name=f"y_{component}") for component in allowed}
    pos = {component: model.addVar(vtype="C", lb=0.0, ub=e_max, name=f"p_{component}") for component in allowed}
    neg = {component: model.addVar(vtype="C", lb=0.0, ub=e_max, name=f"n_{component}") for component in allowed}
    slack = model.addVar(vtype="C", lb=0.0, name="band_slack")

    for component in allowed:
        model.addCons(pos[component] <= e_max * touch[component])
        model.addCons(neg[component] <= e_max * touch[component])
    model.addCons(quicksum(touch[component] for component in allowed) >= 1)
    require_anchor = abs(z0) <= band
    anchors = [component for component in DIVERSIFY_ANCHORS if component in allowed]
    if require_anchor and anchors:
        model.addCons(quicksum(touch[component] for component in anchors) >= 1)
    allowed_set = set(allowed)
    for source, target in CONSISTENCY_IMPLICATIONS:
        if source in allowed_set and target in allowed_set:
            model.addCons(touch[source] <= touch[target])

    movement = quicksum(
        sensitivity[component] * (pos[component] - neg[component]) for component in allowed
    )
    z_predicted = z0 - movement
    model.addCons(z_predicted <= band + slack)
    model.addCons(-z_predicted <= band + slack)

    edit_cost = quicksum(
        fixed_cost[component] * touch[component]
        + linear_cost[component] * (pos[component] + neg[component])
        for component in allowed
    )
    model.setObjective(BAND_SLACK_PENALTY * slack + edit_cost, "minimize")
    model.optimize()
    if str(model.getStatus()) != "optimal":
        return None

    touched = tuple(component for component in allowed if model.getVal(touch[component]) > 0.5)
    strengths = {
        component: float(model.getVal(pos[component]) - model.getVal(neg[component]))
        for component in touched
    }
    fixed = sum(fixed_cost[component] for component in touched)
    variable = sum(linear_cost[component] * abs(strengths[component]) for component in touched)
    delta_logit = sum(sensitivity[component] * strengths[component] for component in touched)
    violation = float(model.getVal(slack))
    return _finalize_plan(
        seed_name="",
        n_probes=0,
        pass_rate_raw=0.0,
        pass_rate_smoothed=0.0,
        z0=z0,
        touched=touched,
        strengths=strengths,
        fixed_cost=fixed,
        linear_cost=variable,
        delta_logit=delta_logit,
        band_violation=violation,
        band=band,
        capacity=_movement_capacity(allowed, sensitivity, e_max),
        backend="pyscipopt",
        solver_result={
            "backend": "pyscipopt",
            "status": str(model.getStatus()),
            "objective_value": float(model.getObjVal()),
            "nodes": int(model.getNNodes()),
            "lp_iterations": int(model.getNLPIterations()),
            "band_slack": violation,
            "optimization_formulation": "mixed_integer_programming",
        },
    )


def _finalize_plan(
    *,
    seed_name: str,
    n_probes: int,
    pass_rate_raw: float,
    pass_rate_smoothed: float,
    z0: float,
    touched: Sequence[str],
    strengths: dict[str, float],
    fixed_cost: float,
    linear_cost: float,
    delta_logit: float,
    band_violation: float,
    band: float,
    capacity: float,
    backend: str,
    solver_result: dict[str, object],
) -> ComponentEditPlan:
    z_predicted = z0 - delta_logit
    projection, direction = _derive_labels(touched, strengths, delta_logit)
    return ComponentEditPlan(
        seed_name=seed_name,
        n_probes=n_probes,
        pass_rate_raw=pass_rate_raw,
        pass_rate_smoothed=pass_rate_smoothed,
        z0=z0,
        touched=tuple(touched),
        strengths=dict(strengths),
        fixed_cost=fixed_cost,
        linear_cost=linear_cost,
        cost=fixed_cost + linear_cost,
        delta_logit=delta_logit,
        z_predicted=z_predicted,
        band_violation=band_violation,
        reachable=band_violation <= 1e-6,
        rounds_needed=_rounds_needed(z0, capacity, band),
        projection=projection,
        evolution_direction=direction,
        backend=backend,
        solver_result=solver_result,
    )


def plan_component_edit(
    seed: TerminalBenchSeedTask,
    *,
    rewards: Sequence[float],
    sensitivity: dict[str, float] | None = None,
    fixed_cost: dict[str, float] | None = None,
    linear_cost: dict[str, float] | None = None,
    e_max: float = DEFAULT_E_MAX,
    band: float = FRONTIER_BAND_NATS,
    prefer_solver: bool = True,
) -> ComponentEditPlan:
    allowed = allowed_components(seed)
    if not allowed:
        raise ValueError(f"seed {seed.name} has no editable components")
    alpha = sensitivity or DEFAULT_SENSITIVITY
    missing = [component for component in allowed if component not in alpha]
    if missing:
        raise ValueError(f"sensitivity missing for components: {missing}")
    rates = fixed_cost or estimate_component_costs(seed)
    unit = linear_cost or {component: DEFAULT_LINEAR_COST for component in EDIT_COMPONENTS}

    raw = (sum(float(reward) for reward in rewards) / len(rewards)) if rewards else 0.5
    smoothed = smoothed_pass_rate(rewards) if rewards else 0.5
    z0 = _logit(smoothed)

    solved: ComponentEditPlan | None = None
    if prefer_solver:
        solved = _solve_with_pyscipopt(
            allowed,
            z0=z0,
            sensitivity=alpha,
            fixed_cost=rates,
            linear_cost=unit,
            e_max=e_max,
            band=band,
        )
    if solved is None:
        solved = _solve_with_enumeration(
            allowed,
            z0=z0,
            sensitivity=alpha,
            fixed_cost=rates,
            linear_cost=unit,
            e_max=e_max,
            band=band,
        )
    if solved is None:
        raise RuntimeError(f"no feasible edit plan for seed {seed.name}")
    return ComponentEditPlan(
        seed_name=seed.name,
        n_probes=len(rewards),
        pass_rate_raw=raw,
        pass_rate_smoothed=smoothed,
        z0=solved.z0,
        touched=solved.touched,
        strengths=solved.strengths,
        fixed_cost=solved.fixed_cost,
        linear_cost=solved.linear_cost,
        cost=solved.cost,
        delta_logit=solved.delta_logit,
        z_predicted=solved.z_predicted,
        band_violation=solved.band_violation,
        reachable=solved.reachable,
        rounds_needed=solved.rounds_needed,
        projection=solved.projection,
        evolution_direction=solved.evolution_direction,
        backend=solved.backend,
        solver_result={**solved.solver_result, "method_version": METHOD_VERSION},
    )


def plan_envs_forge_v1_batch(
    seeds: Sequence[TerminalBenchSeedTask],
    *,
    rewards_by_seed: dict[str, Sequence[float]] | None = None,
    target_count: int = 10,
    **kwargs,
) -> list[ComponentEditPlan]:
    rewards_by_seed = rewards_by_seed or {}
    plans: list[ComponentEditPlan] = []
    for seed in seeds:
        plans.append(plan_component_edit(seed, rewards=_lookup_rewards(seed.name, rewards_by_seed), **kwargs))
    # Reachable plans first: an unreachable seed still consumes a synthesis slot
    # but cannot land on the frontier in one round, so spend the budget on the
    # seeds the current round can actually place.
    ordered = sorted(
        plans,
        key=lambda plan: (
            0 if plan.reachable else 1,
            plan.band_violation,
            plan.cost,
            plan.seed_name,
        ),
    )
    return ordered[: max(0, target_count)]


def _lookup_rewards(seed_name: str, rewards_by_seed: dict[str, Sequence[float]]) -> Sequence[float]:
    if seed_name in rewards_by_seed:
        return rewards_by_seed[seed_name]
    matches = [
        values
        for key, values in rewards_by_seed.items()
        if seed_name == key.split("__", 1)[0]
    ]
    merged: list[float] = []
    for values in matches:
        merged.extend(float(value) for value in values)
    return merged


def _strength_word(value: float, e_max: float) -> str:
    magnitude = abs(value) / e_max if e_max else 0.0
    if magnitude < 1e-6:
        return "none"
    if magnitude < 0.34:
        return "light"
    if magnitude < 0.67:
        return "moderate"
    return "strong"


def _component_instruction(component: str, value: float, e_max: float) -> str:
    label = COMPONENT_LABELS[component]
    strength = _strength_word(value, e_max)
    if strength == "none":
        if component == "instruction":
            return f"keep the difficulty of the {label} unchanged while changing its concrete requirements."
        if component == "data":
            return f"replace the {label} with equivalent-scale but different fixtures; keep the difficulty unchanged."
        if component == "solution":
            return f"update the {label} to match the new fixtures without changing the difficulty."
        if component == "tests":
            return f"update the {label} to the new expected outputs; keep the grading granularity unchanged."
        return f"reuse the seed {label} as-is unless the other edits require it."
    direction = "harden" if value > 0 else "simplify"
    if component == "instruction":
        if value > 0:
            return f"{strength} {direction} the {label}: add concrete constraints, edge cases, or stricter output requirements."
        return f"{strength} {direction} the {label}: drop secondary systems and state the remaining contract exactly."
    if component == "data":
        if value > 0:
            return f"{strength} {direction} the {label}: grow or complicate the fixtures so the new constraints are exercised."
        return f"{strength} {direction} the {label}: replace live inputs with smaller deterministic fixtures."
    if component == "solution":
        return f"{strength} {direction} the {label}: keep it consistent with the edited statement and fixtures."
    if component == "tests":
        return f"{strength} {direction} the {label}: keep grading exact and state every threshold the verifier enforces."
    return f"{strength} {direction} the {label}: keep the image minimal and pinned."


def build_component_edit_prompt(
    plan: ComponentEditPlan,
    seed: TerminalBenchSeedTask,
    *,
    model: str,
    e_max: float = DEFAULT_E_MAX,
    band: float = FRONTIER_BAND_NATS,
) -> str:
    editable_files = {
        path: _shorten(text, 2500)
        for path, text in seed.files.items()
        if _include_seed_file(path) and component_of_path(path) in plan.touched
    }
    untouched = [component for component in EDIT_COMPONENTS if component not in plan.touched]
    payload = {
        "method_version": METHOD_VERSION,
        "seed": seed.name,
        "n_probes": plan.n_probes,
        "pass_rate_raw": round(plan.pass_rate_raw, 4),
        "pass_rate_smoothed": round(plan.pass_rate_smoothed, 4),
        "z0_log_odds": round(plan.z0, 4),
        "target_band_abs_log_odds": band,
        "projection": plan.projection,
        "evolution_direction": plan.evolution_direction,
        "touched_components": list(plan.touched),
        "edit_strengths": {key: round(value, 4) for key, value in sorted(plan.strengths.items())},
        "predicted_log_odds_after_edit": round(plan.z_predicted, 4),
        "edit_cost": round(plan.cost, 4),
        "reachable_in_one_round": plan.reachable,
        "rounds_needed": plan.rounds_needed,
        "untouched_components": untouched,
    }
    directives = "\n".join(
        f"- {component}: {_component_instruction(component, plan.strengths.get(component, 0.0), e_max)}"
        for component in plan.touched
    )
    untouched_text = (
        "\n".join(f"- {component} ({COMPONENT_LABELS[component]})" for component in untouched)
        or "- (none: every component is part of this plan)"
    )
    return f"""Synthesize one terminal-bench-style task directory using Envs-FORGE v1.

Return exactly one JSON object, no markdown.
Required top-level keys: task_name, files, metadata.

This run uses a component-edit plan, not a preset projection. A mixed-integer
program decided which components to edit and how strongly, so the action label
below is a consequence of that plan rather than an instruction to maximise or
minimise difficulty.

Plan:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Edit directives (apply every one, at the stated strength):
{directives}

Return ONLY files belonging to the touched components. For each touched
component return at least one file:
- instruction -> instruction.md
- data -> one or more files under environment/ (not environment/Dockerfile)
- solution -> solution/solve.sh
- tests -> tests/test.sh and tests/test_outputs.py
- environment -> environment/Dockerfile

Do NOT return files for components that were not touched:
{untouched_text}

Unchanged files are inherited from the seed, so the returned subset must stay
internally consistent with them. Concretely:
- instruction.md and the tests must agree on every requirement, threshold, file
  name, output key, and tie-break rule.
- solution/solve.sh must solve exactly the instruction that ends up on disk.
- tests/test.sh must always write /logs/verifier/reward.txt: 1 on pass, 0 on
  fail. Capture the pytest exit status explicitly so `set -e` cannot skip it.
- Do NOT copy the seed tests/test.sh verbatim if it installs a runtime from an
  external release script. Concretely, never run
  `curl ... astral.sh/uv/install.sh`, `bootstrap.pypa.io/get-pip.py`, or any
  GitHub release installer inside tests/test.sh. Install verifier dependencies in
  environment/Dockerfile with mirrors, or use interpreters the base image
  already ships. A compliant pattern when the image already has pytest is:

  #!/bin/bash
  set -u
  cd /app 2>/dev/null || cd .
  pytest -q /tests/test_outputs.py
  status=$?
  if [ "$status" -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
  exit 0

- environment/Dockerfile is built with environment/ as the Docker build context.
  Never COPY task-root paths such as tests/, solution/, instruction.md, or
  task.toml. If the Dockerfile needs fixture data, keep it under environment/.
- Pin base images to explicit tags already used by the seed. Do not use :latest.
  Do not change apt sources. Do not add installs the seed did not already need.
- Keep the artifacts small: solution/solve.sh under 150 lines,
  tests/test_outputs.py under 220 lines, no large generated datasets.
- Do not include terminal-bench canary comments or API keys.
- Before returning, mentally execute: build environment/Dockerfile, run
  solution/solve.sh, then run tests/test.sh. The result must be reward 1.

Augmentation model: {model}

Return JSON shape:
{{
  "task_name": "lowercase-hyphenated-name",
  "files": {{
    "<only paths from the touched components>": "..."
  }},
  "metadata": {{
    "summary": "one sentence describing the component edit that was applied",
    "envs_forge_v1": {json.dumps(payload, ensure_ascii=False)}
  }}
}}

Seed task name: {seed.name}
Seed instruction.md for style:
{_shorten(seed.instruction, 3000)}
Seed files for the touched components:
{json.dumps(editable_files, ensure_ascii=False, indent=2)}
"""


def validate_subset_artifact(artifact: dict | None, touched: Sequence[str]) -> list[str]:
    """Validate a partial artifact that only carries the touched components."""
    if artifact is None:
        return ["missing generated JSON object"]
    errors: list[str] = []
    if not isinstance(artifact.get("task_name"), str) or not artifact.get("task_name", "").strip():
        errors.append("task_name must be a non-empty string")
    files = artifact.get("files")
    if isinstance(files, list):
        normalized: dict[str, str] = {}
        for entry in files:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                normalized[entry["path"]] = str(entry.get("content", ""))
        files = normalized
    if not isinstance(files, dict) or not files:
        return errors + ["files must be a non-empty object of path -> content"]

    touched_set = set(touched)
    covered: set[str] = set()
    for path, content in files.items():
        if not isinstance(path, str) or not path.strip():
            errors.append("file paths must be non-empty strings")
            continue
        if not isinstance(content, str) or not content.strip():
            errors.append(f"empty generated file: {path}")
            continue
        component = component_of_path(path)
        if component is None:
            errors.append(f"file does not belong to any editable component: {path}")
            continue
        if component not in touched_set:
            errors.append(f"file belongs to untouched component {component}: {path}")
            continue
        covered.add(component)
    for component in touched:
        if component not in covered:
            errors.append(f"touched component {component} has no generated file")
    if "tests" in touched_set:
        for required in ("tests/test.sh", "tests/test_outputs.py"):
            if required not in files:
                errors.append(f"missing generated file: {required}")
    if "environment" in touched_set and "environment/Dockerfile" not in files:
        errors.append("missing generated file: environment/Dockerfile")
    errors.extend(_validate_generated_content(files))
    return errors


def _validate_generated_content(files: dict) -> list[str]:
    """Apply the content rules to generated files only.

    Inherited seed files are not regenerated by this method, so they are not
    held to the generation contract; the seed already passed its own review.
    """
    from .terminal_bench import (
        _contains_canary,
        _contains_forbidden_bootstrap_url,
        _contains_forbidden_runtime_installer,
        _is_safe_relative_path,
        _test_runner_can_skip_reward_on_pytest_failure,
        _validate_environment_dockerfile,
    )

    errors: list[str] = []
    for path, content in files.items():
        if not isinstance(content, str):
            errors.append(f"generated file content must be a string: {path}")
            continue
        if not _is_safe_relative_path(path):
            errors.append(f"unsafe generated file path: {path}")
        elif _contains_canary(content):
            errors.append(f"generated file contains terminal-bench canary text: {path}")
        elif _contains_forbidden_bootstrap_url(content):
            errors.append(f"generated file must not depend on external bootstrap URL: {path}")
        elif path == "tests/test.sh" and _contains_forbidden_runtime_installer(content):
            errors.append(f"tests/test.sh must not install uv from external release scripts: {path}")
        elif path == "tests/test.sh" and _test_runner_can_skip_reward_on_pytest_failure(content):
            errors.append("tests/test.sh must write /logs/verifier/reward.txt even when pytest fails")
        elif path == "environment/Dockerfile":
            errors.extend(_validate_environment_dockerfile(content))
    return errors


def expand_subset_artifact(
    seed: TerminalBenchSeedTask,
    artifact: dict,
    *,
    touched: Sequence[str],
) -> dict:
    """Complete a partial v1 artifact by inheriting the untouched components.

    The v1 generator only returns the components the MILP decided to edit. The
    materialiser and gold verifier still expect a full task directory, so the
    untouched files are carried over verbatim from the seed.
    """
    files = artifact.get("files")
    if isinstance(files, list):
        normalized: dict[str, str] = {}
        for entry in files:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                normalized[entry["path"]] = str(entry.get("content", ""))
        files = normalized
    generated = dict(files or {})
    touched_set = set(touched)
    inherited: dict[str, str] = {}
    for path, content in seed.files.items():
        component = component_of_path(path)
        if component is None or component in touched_set:
            continue
        inherited[path] = content
    merged = {**inherited, **generated}
    if "instruction.md" not in generated and isinstance(artifact.get("instruction_md"), str):
        merged["instruction.md"] = artifact["instruction_md"]
    return {
        "task_name": artifact.get("task_name", ""),
        "instruction_md": merged.get("instruction.md", seed.files.get("instruction.md", "")),
        "files": merged,
        "metadata": artifact.get("metadata") or {},
    }


def component_coverage(artifact: dict | None) -> dict[str, int]:
    files = (artifact or {}).get("files")
    if isinstance(files, list):
        paths = [entry.get("path") for entry in files if isinstance(entry, dict)]
    elif isinstance(files, dict):
        paths = list(files)
    else:
        paths = []
    counts: dict[str, int] = {}
    for path in paths:
        component = component_of_path(str(path))
        if component:
            counts[component] = counts.get(component, 0) + 1
    return counts


def build_component_repair_prompt(
    plan: ComponentEditPlan,
    seed: TerminalBenchSeedTask,
    *,
    previous_files: dict[str, str],
    failure_summary: str,
    model: str,
    repair_index: int,
    e_max: float = DEFAULT_E_MAX,
) -> str:
    """Repair a partial v1 artifact without widening the touched components."""
    directives = "\n".join(
        f"- {component}: {_component_instruction(component, plan.strengths.get(component, 0.0), e_max)}"
        for component in plan.touched
    )
    return f"""Repair a failed Envs-FORGE v1 task that only edits these components: {list(plan.touched)}.

Return exactly one JSON object, no markdown. Do not return patch fragments.
Required top-level keys: task_name, files, metadata.

Repair attempt: {repair_index}
Augmentation model: {model}

The component plan is fixed; do not add or remove components.
{directives}

Rules the previous attempt broke or must keep:
{failure_summary}

Constraints:
- Return only files inside the touched components {list(plan.touched)}.
- instruction.md and the tests must agree on every requirement, threshold, file name, output key, and tie-break rule.
- solution/solve.sh must solve exactly the instruction that ends up on disk.
- tests/test.sh must always write /logs/verifier/reward.txt: 1 on pass, 0 on fail, and must not install tools from external release scripts (no astral.sh/uv installer, no bootstrap.pypa.io).
- environment/Dockerfile uses environment/ as build context; never COPY tests/, solution/, instruction.md, or task.toml.
- Before returning, mentally execute: build the image, run solution/solve.sh, run tests/test.sh. The result must be reward 1.

Return JSON shape:
{{
  "task_name": "lowercase-hyphenated-name",
  "files": {{"<touched component path>": "..."}},
  "metadata": {{"summary": "what changed in this repair"}}
}}

Seed task name: {seed.name}
Previous generated files:
{json.dumps({path: _shorten(text, 6000) for path, text in previous_files.items()}, ensure_ascii=False, indent=2)}

Seed instruction.md for reference:
{_shorten(seed.instruction, 2500)}
"""


def load_component_sensitivities(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sensitivity file must contain an object of component -> nats per unit strength")
    unknown = [key for key in payload if key not in EDIT_COMPONENTS]
    if unknown:
        raise ValueError(f"unknown components in sensitivity file: {unknown}")
    return {key: float(value) for key, value in payload.items()}
