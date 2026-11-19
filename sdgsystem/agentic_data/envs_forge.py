from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

from .terminal_bench import TerminalBenchSeedTask, _include_seed_file, _shorten

PROJECTIONS: tuple[str, ...] = ("increase_complexity", "reduce_complexity", "diversify_frontier")
EVOLUTION_DIRECTIONS: tuple[str, ...] = ("in_depth", "in_breadth")
# The action-selection selector in this module is the alpha line: it chooses one
# of six preset actions whose difficulty effect is an exogenous constant. The
# component-edit MILP that replaced it lives in envs_forge_v1.py.
METHOD_VERSION = "envs-forge-alpha"
DEFAULT_ACTION_SHIFTS: dict[str, float] = {
    "increase_complexity": -0.25,
    "reduce_complexity": 0.25,
    "diversify_frontier": 0.0,
}
DEFAULT_DIRECTION_SCALES: dict[str, float] = {
    "in_depth": 1.0,
    "in_breadth": 0.65,
}
CALIBRATED_TIE_BREAK: dict[str, int] = {
    "diversify_frontier": 0,
    "increase_complexity": 1,
    "reduce_complexity": 2,
}
EXACT_ENUMERATION_LIMIT = 24
SKILL_NODE_PENALTY = 1e-6
SKILL_COVERAGE_SLACK_PENALTY = 0.25
ENVS_FORGE_ALPHA_OPTIMIZATION_FORMULATION = "binary_integer_programming"
ENVS_FORGE_MILP_OPTIMIZATION_FORMULATION = "mixed_integer_programming"
ENVS_FORGE_ALPHA_OPTIMIZATION_FORMULATION_LABEL = "0-1 integer programming"
ENVS_FORGE_MILP_OPTIMIZATION_FORMULATION_LABEL = "mixed integer programming"


@dataclass(frozen=True)
class SkillProfile:
    domains: tuple[str, ...]
    tools: tuple[str, ...]
    artifacts: tuple[str, ...]
    operations: tuple[str, ...]
    constraints: tuple[str, ...]

    def as_tags(self) -> tuple[str, ...]:
        seen: set[str] = set()
        tags: list[str] = []
        for group in [self.domains, self.tools, self.artifacts, self.operations, self.constraints]:
            for item in group:
                if item not in seen:
                    seen.add(item)
                    tags.append(item)
        return tuple(tags)


@dataclass(frozen=True)
class ArtifactEditMask:
    required_artifacts: tuple[str, ...]
    optional_artifacts: tuple[str, ...]
    operators: dict[str, tuple[str, ...]]

    def as_json(self) -> dict[str, object]:
        return {
            "required_artifacts": self.required_artifacts,
            "optional_artifacts": self.optional_artifacts,
            "operators": self.operators,
        }


@dataclass(frozen=True)
class SkillEdge:
    source: str
    target: str
    relation: str

    def as_json(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target, "relation": self.relation}


@dataclass(frozen=True)
class SkillSubgraph:
    nodes: tuple[str, ...]
    edges: tuple[SkillEdge, ...]
    preserve_nodes: tuple[str, ...]
    allowed_added_nodes: tuple[str, ...]
    allowed_removed_nodes: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "nodes": self.nodes,
            "edges": tuple(edge.as_json() for edge in self.edges),
            "preserve_nodes": self.preserve_nodes,
            "allowed_added_nodes": self.allowed_added_nodes,
            "allowed_removed_nodes": self.allowed_removed_nodes,
        }


@dataclass(frozen=True)
class EnvsForgePlanItem:
    seed: TerminalBenchSeedTask
    profile: SkillProfile
    pass_rate: float
    frontier_score: float
    band: str
    projection: str
    target_difficulty: str
    sample_index: int
    evolution_direction: str | None = None
    projection_policy: str = "band_rule"
    predicted_pass_rate: float | None = None
    predicted_frontier_score: float | None = None
    artifact_edit_mask: ArtifactEditMask | None = None
    skill_subgraph: SkillSubgraph | None = None
    solver_backend: str | None = None
    solver_result: dict[str, object] | None = None
    optimization_formulation: str | None = None

    @property
    def action_variant(self) -> str:
        if self.evolution_direction:
            return f"{self.projection}:{self.evolution_direction}"
        return self.projection


KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "domains": {
        "system_admin": ("systemd", "journald", "rsyslog", "logrotate", "service", "timer", "daemon"),
        "data_processing": ("csv", "json", "jsonl", "parquet", "dataframe", "merge", "aggregate", "etl"),
        "testing": ("test case", "pytest", "tsl", "category partition", "unit test", "coverage"),
        "security": ("jwt", "token", "sql injection", "forensics", "malicious", "xss", "xxe", "permission"),
        "ml_ops": ("mlflow", "model registry", "classifier", "train model", "prediction", "sklearn"),
        "game_reasoning": ("chess", "pgn", "fen", "mate", "puzzle"),
    },
    "tools": {
        "bash": ("bash", "shell", ".sh", "awk", "sed", "grep"),
        "python": ("python", "pytest", ".py", "pandas", "numpy"),
        "systemd": ("systemd", "systemctl", "journalctl", ".service", ".timer"),
        "docker": ("dockerfile", "container", "apt-get", "pip install", "npm"),
        "sql": ("sql", "sqlite", "database", "query"),
        "node": ("node", "npm", "javascript", "typescript"),
    },
    "artifacts": {
        "logs": ("log", "logs", "journal", "auditd", "netflow"),
        "csv": ("csv", ".csv"),
        "json": ("json", ".json"),
        "jsonl": ("jsonl", ".jsonl"),
        "source_code": ("source code", ".py", ".js", ".ts", ".c", ".cpp", "repository"),
        "config": ("config", "yaml", "toml", "service", "timer", "dockerfile"),
    },
    "operations": {
        "merge": ("merge", "join", "combine", "union"),
        "parse": ("parse", "extract", "read", "normalize"),
        "repair": ("fix", "repair", "debug", "bug"),
        "configure": ("configure", "install", "enable", "setup", "deploy"),
        "generate_tests": ("generate test", "test case", "tsl", "category partition"),
        "rank": ("rank", "sort", "top", "tie-break"),
        "validate": ("validate", "verify", "check", "assert"),
    },
    "constraints": {
        "concurrency": ("concurrent", "parallel", "race", "lock", "atomic", "mutex"),
        "atomicity": ("atomic", "lock", "idempotent", "transaction"),
        "deterministic_sorting": ("sort", "ordered", "deterministic", "tie-break"),
        "edge_cases": ("edge case", "invalid", "malformed", "missing", "duplicate"),
        "scheduling": ("timer", "schedule", "cron", "periodic"),
        "structured_output": ("json output", "csv output", "schema", "exact format"),
    },
}


def estimate_pass_rate(rewards: Sequence[float], *, alpha: float = 1.0, beta: float = 1.0) -> float:
    passes = sum(1 for reward in rewards if float(reward) >= 1.0)
    return (alpha + passes) / (alpha + beta + len(rewards))


def learning_frontier_score(pass_rate: float, *, target: float = 0.5, sigma: float = 0.2) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return math.exp(-((pass_rate - target) ** 2) / (2 * sigma**2))


def select_calibrated_projection(
    pass_rate: float,
    *,
    action_shifts: dict[str, float] | None = None,
    target: float = 0.5,
    sigma: float = 0.2,
) -> tuple[str, float, float]:
    shifts = action_shifts or DEFAULT_ACTION_SHIFTS
    scored: list[tuple[float, int, str, float]] = []
    for projection in PROJECTIONS:
        predicted_pass_rate = _clip_probability(pass_rate + shifts.get(projection, 0.0))
        score = learning_frontier_score(predicted_pass_rate, target=target, sigma=sigma)
        scored.append((-score, CALIBRATED_TIE_BREAK[projection], projection, predicted_pass_rate))
    _negative_score, _tie_break, projection, predicted_pass_rate = min(scored)
    return projection, predicted_pass_rate, learning_frontier_score(predicted_pass_rate, target=target, sigma=sigma)


def _default_evolution_direction_for_band(band: str, projection: str) -> str:
    if band == "hard":
        return "in_breadth"
    if band == "frontier":
        return "in_breadth" if projection == "diversify_frontier" else "in_depth"
    return "in_depth"


def _action_shift_for_variant(
    projection: str,
    evolution_direction: str | None,
    *,
    action_shifts: dict[str, float] | None = None,
) -> float:
    base_shift = (action_shifts or DEFAULT_ACTION_SHIFTS).get(projection, 0.0)
    if evolution_direction is None:
        return base_shift
    return base_shift * DEFAULT_DIRECTION_SCALES.get(evolution_direction, 1.0)


def _evolution_direction_instruction(evolution_direction: str | None) -> str:
    if evolution_direction == "in_depth":
        return (
            "Prefer a deeper edit: strengthen constraints, extend edge cases, and make the task structurally richer "
            "without changing the core skill graph."
        )
    if evolution_direction == "in_breadth":
        return (
            "Prefer a broader edit: vary fixtures, nearby constraints, or surrounding context while keeping the "
            "core difficulty in the same neighborhood."
        )
    return (
        "Choose the evolution direction that best matches the projection. If you need a default, stay consistent "
        "with the skill subgraph and the learning-frontier objective."
    )


def artifact_edit_mask_for_projection(projection: str) -> ArtifactEditMask:
    if projection == "increase_complexity":
        return ArtifactEditMask(
            required_artifacts=("instruction", "data", "solution", "tests"),
            optional_artifacts=("environment", "helper_files"),
            operators={
                "instruction": ("add_constraints", "state_edge_cases", "tighten_output_contract"),
                "data": ("add_edge_cases", "increase_fixture_size", "change_schema"),
                "solution": ("update_algorithm", "handle_new_constraints"),
                "tests": ("strengthen_verifier", "check_edge_cases", "check_deterministic_outputs"),
                "environment": ("reuse_or_pin_dependencies",),
                "helper_files": ("add_grader_or_fixture_helpers",),
            },
        )
    if projection == "reduce_complexity":
        return ArtifactEditMask(
            required_artifacts=("instruction", "data", "solution", "tests", "environment"),
            optional_artifacts=("helper_files",),
            operators={
                "instruction": ("split_secondary_systems", "define_bridge_task", "make_hidden_assumptions_explicit"),
                "data": ("replace_live_system_with_fixture", "shrink_fixture", "make_inputs_deterministic"),
                "solution": ("rewrite_solution_for_bridge_task", "remove_privileged_operations"),
                "tests": ("verify_core_skill_only", "add_boundary_cases", "write_reward_on_failure"),
                "environment": ("simplify_environment", "remove_unneeded_services", "pin_minimal_runtime"),
                "helper_files": ("add_small_fixture_generator_or_grader",),
            },
        )
    if projection != "diversify_frontier":
        raise ValueError(f"Unknown Envs-FORGE projection: {projection}")
    return ArtifactEditMask(
        required_artifacts=("instruction", "data", "solution", "tests"),
        optional_artifacts=("environment", "helper_files"),
        operators={
            "instruction": ("same_difficulty_reframe", "change_concrete_requirements"),
            "data": ("replace_fixtures", "change_values", "preserve_fixture_scale"),
            "solution": ("update_expected_logic", "preserve_solution_shape"),
            "tests": ("update_expected_outputs", "preserve_grading_granularity"),
            "environment": ("reuse_seed_runtime",),
            "helper_files": ("update_grader_or_fixture_helpers",),
        },
    )


def build_skill_subgraph(profile: SkillProfile, projection: str, evolution_direction: str | None = None) -> SkillSubgraph:
    nodes = _unique(profile.as_tags() or ("general_terminal",))
    edges = _skill_edges(profile)
    if projection == "reduce_complexity":
        preserve_nodes = _unique((*profile.domains, *profile.artifacts, *profile.operations))
        if evolution_direction == "in_breadth":
            allowed_added_nodes = ("alternate_fixture", "neighboring_constraint")
        else:
            allowed_added_nodes = ("file_level_simulation", "deterministic_fixture")
        removable = {"systemd", "docker", "scheduling", "concurrency", "atomicity"}
        allowed_removed_nodes = tuple(node for node in nodes if node in removable)
    elif projection == "increase_complexity":
        preserve_nodes = nodes
        if evolution_direction == "in_breadth":
            allowed_added_nodes = ("alternate_fixture", "neighboring_constraint")
        else:
            allowed_added_nodes = ("edge_cases", "structured_output", "deterministic_sorting")
        allowed_removed_nodes = ()
    elif projection == "diversify_frontier":
        preserve_nodes = nodes
        if evolution_direction == "in_depth":
            allowed_added_nodes = ("edge_cases", "structured_output", "deterministic_sorting")
        else:
            allowed_added_nodes = ("alternate_fixture", "neighboring_constraint")
        allowed_removed_nodes = ()
    else:
        raise ValueError(f"Unknown Envs-FORGE projection: {projection}")
    return SkillSubgraph(
        nodes=nodes,
        edges=edges,
        preserve_nodes=preserve_nodes or nodes,
        allowed_added_nodes=allowed_added_nodes,
        allowed_removed_nodes=allowed_removed_nodes,
    )


def extract_skill_profile(seed: TerminalBenchSeedTask) -> SkillProfile:
    text = "\n".join([seed.instruction, seed.task_toml, *seed.files.values()]).lower()
    return SkillProfile(
        domains=_match_category(text, KEYWORDS["domains"]),
        tools=_match_category(text, KEYWORDS["tools"]),
        artifacts=_match_category(text, KEYWORDS["artifacts"]),
        operations=_match_category(text, KEYWORDS["operations"]),
        constraints=_match_category(text, KEYWORDS["constraints"]),
    )


def plan_envs_forge_batch(
    seeds: Sequence[TerminalBenchSeedTask],
    *,
    rewards_by_seed: dict[str, Sequence[float]] | None = None,
    target_count: int = 10,
    low: float = 0.2,
    high: float = 0.8,
    target: float = 0.5,
    sigma: float = 0.2,
    projection_policy: str = "band_rule",
    action_shifts: dict[str, float] | None = None,
    min_skill_coverage: dict[str, int] | None = None,
    optimization_formulation: str = ENVS_FORGE_ALPHA_OPTIMIZATION_FORMULATION,
) -> list[EnvsForgePlanItem]:
    if projection_policy not in {"band_rule", "calibrated_transfer", "joint_optimization"}:
        raise ValueError("projection_policy must be 'band_rule', 'calibrated_transfer', or 'joint_optimization'")
    if optimization_formulation not in {
        ENVS_FORGE_ALPHA_OPTIMIZATION_FORMULATION,
        ENVS_FORGE_MILP_OPTIMIZATION_FORMULATION,
    }:
        raise ValueError(
            "optimization_formulation must be 'binary_integer_programming' or 'mixed_integer_programming'"
        )
    rewards_by_seed = rewards_by_seed or {}
    candidates: list[EnvsForgePlanItem] = []
    for seed in seeds:
        reward_values = _lookup_seed_rewards(seed.name, rewards_by_seed)
        pass_rate = _observed_reward_mean(reward_values)
        frontier_score = learning_frontier_score(pass_rate, target=target, sigma=sigma) if reward_values else 0.0
        band = classify_band(pass_rate, low=low, high=high)
        profile = extract_skill_profile(seed)
        predicted_pass_rate = None
        predicted_frontier_score = None
        if projection_policy == "calibrated_transfer":
            projection, predicted_pass_rate, predicted_frontier_score = select_calibrated_projection(
                pass_rate,
                action_shifts=action_shifts,
                target=target,
                sigma=sigma,
            )
            if reward_values:
                frontier_score = predicted_frontier_score
            target_difficulty = target_difficulty_for_projection(projection)
        elif projection_policy == "joint_optimization":
            if optimization_formulation == ENVS_FORGE_MILP_OPTIMIZATION_FORMULATION:
                for projection in PROJECTIONS:
                    for evolution_direction in EVOLUTION_DIRECTIONS:
                        predicted_pass_rate = _clip_probability(
                            pass_rate
                            + _action_shift_for_variant(
                                projection,
                                evolution_direction,
                                action_shifts=action_shifts,
                            )
                        )
                        predicted_frontier_score = learning_frontier_score(predicted_pass_rate, target=target, sigma=sigma)
                        candidates.append(
                            _make_plan_item(
                                seed=seed,
                                profile=profile,
                                pass_rate=pass_rate,
                                frontier_score=predicted_frontier_score,
                                band=band,
                                projection=projection,
                                evolution_direction=evolution_direction,
                                target_difficulty=target_difficulty_for_projection(projection),
                                sample_index=0,
                                projection_policy=projection_policy,
                                predicted_pass_rate=predicted_pass_rate,
                                predicted_frontier_score=predicted_frontier_score,
                                optimization_formulation=optimization_formulation,
                            )
                        )
                continue
            for projection in PROJECTIONS:
                predicted_pass_rate = _clip_probability(pass_rate + (action_shifts or DEFAULT_ACTION_SHIFTS).get(projection, 0.0))
                predicted_frontier_score = learning_frontier_score(predicted_pass_rate, target=target, sigma=sigma)
                candidates.append(
                    _make_plan_item(
                        seed=seed,
                        profile=profile,
                        pass_rate=pass_rate,
                        frontier_score=predicted_frontier_score,
                        band=band,
                        projection=projection,
                        evolution_direction=_default_evolution_direction_for_band(band, projection),
                        target_difficulty=target_difficulty_for_projection(projection),
                        sample_index=0,
                        projection_policy=projection_policy,
                        predicted_pass_rate=predicted_pass_rate,
                        predicted_frontier_score=predicted_frontier_score,
                        optimization_formulation=optimization_formulation,
                    )
                )
            continue
        else:
            projection = projection_for_band(band)
            target_difficulty = target_difficulty_for_band(band)
        candidates.append(
            _make_plan_item(
                seed=seed,
                profile=profile,
                pass_rate=pass_rate,
                frontier_score=frontier_score,
                band=band,
                projection=projection,
                evolution_direction=_default_evolution_direction_for_band(band, projection),
                target_difficulty=target_difficulty,
                sample_index=0,
                projection_policy=projection_policy,
                predicted_pass_rate=predicted_pass_rate,
                predicted_frontier_score=predicted_frontier_score,
            )
        )

    if projection_policy == "joint_optimization":
        ordered = _solve_joint_projection_plan(
            candidates,
            target_count=target_count,
            min_skill_coverage=min_skill_coverage or {},
            optimization_formulation=optimization_formulation,
        )
    else:
        ordered = _balanced_order(candidates)
    selected: list[EnvsForgePlanItem] = []
    for index, item in enumerate(ordered[: max(0, target_count)]):
        selected.append(
            _make_plan_item(
                seed=item.seed,
                profile=item.profile,
                pass_rate=item.pass_rate,
                frontier_score=item.frontier_score,
                band=item.band,
                projection=item.projection,
                evolution_direction=item.evolution_direction,
                target_difficulty=item.target_difficulty,
                sample_index=index,
                projection_policy=item.projection_policy,
                predicted_pass_rate=item.predicted_pass_rate,
                predicted_frontier_score=item.predicted_frontier_score,
                artifact_edit_mask=item.artifact_edit_mask,
                skill_subgraph=item.skill_subgraph,
                solver_backend=item.solver_backend,
                solver_result=item.solver_result,
                optimization_formulation=item.optimization_formulation,
            )
        )
    return selected


def _make_plan_item(
    *,
    seed: TerminalBenchSeedTask,
    profile: SkillProfile,
    pass_rate: float,
    frontier_score: float,
    band: str,
    projection: str,
    target_difficulty: str,
    sample_index: int,
    projection_policy: str,
    evolution_direction: str | None = None,
    predicted_pass_rate: float | None = None,
    predicted_frontier_score: float | None = None,
    artifact_edit_mask: ArtifactEditMask | None = None,
    skill_subgraph: SkillSubgraph | None = None,
    solver_backend: str | None = None,
    solver_result: dict[str, object] | None = None,
    optimization_formulation: str | None = None,
) -> EnvsForgePlanItem:
    return EnvsForgePlanItem(
        seed=seed,
        profile=profile,
        pass_rate=pass_rate,
        frontier_score=frontier_score,
        band=band,
        projection=projection,
        evolution_direction=evolution_direction,
        target_difficulty=target_difficulty,
        sample_index=sample_index,
        projection_policy=projection_policy,
        predicted_pass_rate=predicted_pass_rate,
        predicted_frontier_score=predicted_frontier_score,
        artifact_edit_mask=artifact_edit_mask or artifact_edit_mask_for_projection(projection),
        skill_subgraph=skill_subgraph or build_skill_subgraph(profile, projection, evolution_direction),
        solver_backend=solver_backend,
        solver_result=solver_result,
        optimization_formulation=optimization_formulation or _optimization_formulation_for_policy(projection_policy),
    )


def _solve_joint_projection_plan(
    candidates: Sequence[EnvsForgePlanItem],
    *,
    target_count: int,
    min_skill_coverage: dict[str, int],
    optimization_formulation: str,
) -> list[EnvsForgePlanItem]:
    count = min(max(0, target_count), len({item.seed.name for item in candidates}))
    if count == 0:
        return []

    pyscipopt_solution = _solve_joint_with_pyscipopt(
        candidates,
        target_count=count,
        min_skill_coverage=min_skill_coverage,
        optimization_formulation=optimization_formulation,
    )
    if pyscipopt_solution is not None:
        return _with_solver_backend(pyscipopt_solution, "pyscipopt")

    if len(candidates) <= EXACT_ENUMERATION_LIMIT:
        exact_solution = _solve_joint_with_exact_enumeration(
            candidates,
            target_count=count,
            min_skill_coverage=min_skill_coverage,
            optimization_formulation=optimization_formulation,
        )
        if exact_solution is not None:
            return _with_solver_backend(exact_solution, "exact_enumeration")

    return _with_solver_backend(
        _solve_joint_with_greedy(
            candidates,
            target_count=count,
            min_skill_coverage=min_skill_coverage,
            optimization_formulation=optimization_formulation,
        ),
        "greedy",
    )


def _solve_joint_with_pyscipopt(
    candidates: Sequence[EnvsForgePlanItem],
    *,
    target_count: int,
    min_skill_coverage: dict[str, int],
    optimization_formulation: str,
) -> list[EnvsForgePlanItem] | None:
    try:
        from pyscipopt import Model, quicksum  # type: ignore
    except Exception:
        return None

    model = Model("envs_forge_alpha")
    model.hideOutput()
    variables = [model.addVar(vtype="B", name=f"x_{index}") for index, _item in enumerate(candidates)]
    skill_nodes = _skill_node_universe(candidates, min_skill_coverage)
    skill_variables = {
        (index, node): model.addVar(vtype="B", name=f"u_{index}_{node}")
        for index, _item in enumerate(candidates)
        for node in skill_nodes
    }
    slack_variables = (
        {
            skill: model.addVar(vtype="C", lb=0.0, name=f"slack_{skill}")
            for skill in min_skill_coverage
        }
        if optimization_formulation == ENVS_FORGE_MILP_OPTIMIZATION_FORMULATION
        else {}
    )
    model.addCons(quicksum(variables) == target_count)
    for seed_name in sorted({item.seed.name for item in candidates}):
        model.addCons(quicksum(variables[index] for index, item in enumerate(candidates) if item.seed.name == seed_name) <= 1)
    for index, item in enumerate(candidates):
        required_nodes = _required_skill_nodes(item)
        optional_nodes = _optional_skill_nodes(item)
        for node in skill_nodes:
            skill_variable = skill_variables[index, node]
            if node in required_nodes:
                model.addCons(skill_variable == variables[index])
            elif node in optional_nodes:
                model.addCons(skill_variable <= variables[index])
            else:
                model.addCons(skill_variable == 0)
    for skill, minimum in min_skill_coverage.items():
        covered = quicksum(skill_variables[index, skill] for index, _item in enumerate(candidates))
        if optimization_formulation == ENVS_FORGE_MILP_OPTIMIZATION_FORMULATION:
            model.addCons(covered + slack_variables[skill] >= minimum)
        else:
            model.addCons(covered >= minimum)
    optional_skill_penalty = quicksum(
        skill_variables[index, node]
        for index, item in enumerate(candidates)
        for node in _optional_skill_nodes(item)
        if node in skill_nodes
    )
    slack_penalty = quicksum(slack_variables.values()) if slack_variables else 0
    model.setObjective(
        quicksum(_candidate_objective(item) * variables[index] for index, item in enumerate(candidates))
        - SKILL_NODE_PENALTY * optional_skill_penalty
        - SKILL_COVERAGE_SLACK_PENALTY * slack_penalty,
        "maximize",
    )
    model.optimize()
    if model.getStatus() != "optimal":
        return None
    objective_value = float(model.getObjVal())
    solver_result_base = {
        "backend": "pyscipopt",
        "optimization_formulation": optimization_formulation,
        "optimization_formulation_label": _optimization_formulation_label(optimization_formulation),
        "status": str(model.getStatus()),
        "objective_value": objective_value,
        "target_count": target_count,
        "candidate_count": len(candidates),
        "skill_nodes": skill_nodes,
        "min_skill_coverage": dict(sorted(min_skill_coverage.items())),
        "coverage_slack": {
            skill: float(model.getVal(variable))
            for skill, variable in slack_variables.items()
        },
    }
    selected: list[EnvsForgePlanItem] = []
    for index, item in enumerate(candidates):
        if model.getVal(variables[index]) <= 0.5:
            continue
        selected_nodes = {
            node
            for node in skill_nodes
            if model.getVal(skill_variables[index, node]) > 0.5
        }
        solver_result = {
            **solver_result_base,
            "selected_x": {
                "candidate_index": index,
                "seed_task": item.seed.name,
                "projection": item.projection,
                "evolution_direction": item.evolution_direction,
                "action_variant": item.action_variant,
                "objective": _candidate_objective(item),
                "x": 1.0,
            },
            "selected_u": [
                {"skill_node": node, "u": 1.0}
                for node in sorted(selected_nodes)
            ],
        }
        selected.append(_with_selected_skill_nodes(item, selected_nodes, solver_result=solver_result))
    return selected


def _solve_joint_with_exact_enumeration(
    candidates: Sequence[EnvsForgePlanItem],
    *,
    target_count: int,
    min_skill_coverage: dict[str, int],
    optimization_formulation: str,
) -> list[EnvsForgePlanItem] | None:
    best: tuple[float, tuple[str, ...], tuple[EnvsForgePlanItem, ...]] | None = None
    for combo in combinations(candidates, target_count):
        if len({item.seed.name for item in combo}) != len(combo):
            continue
        selected_with_skills = _apply_skill_graph_selection(combo, min_skill_coverage)
        if selected_with_skills is None:
            selected_with_skills = list(combo)
        objective = sum(_candidate_objective(item) for item in selected_with_skills)
        if optimization_formulation == ENVS_FORGE_MILP_OPTIMIZATION_FORMULATION:
            objective -= SKILL_COVERAGE_SLACK_PENALTY * sum(
                _skill_coverage_shortage(selected_with_skills, min_skill_coverage).values()
            )
        tie_break = tuple(f"{item.seed.name}:{item.projection}" for item in sorted(selected_with_skills, key=lambda value: value.seed.name))
        score = (objective, tie_break, tuple(selected_with_skills))
        if best is None or score[:2] > best[:2]:
            best = score
    return list(best[2]) if best is not None else None


def _solve_joint_with_greedy(
    candidates: Sequence[EnvsForgePlanItem],
    *,
    target_count: int,
    min_skill_coverage: dict[str, int],
    optimization_formulation: str,
) -> list[EnvsForgePlanItem]:
    selected: list[EnvsForgePlanItem] = []
    ordered = sorted(candidates, key=_candidate_objective, reverse=True)

    for skill, minimum in min_skill_coverage.items():
        while _coverage_count(selected, lambda item, value=skill: _candidate_can_cover_skill(item, value)) < minimum:
            candidate = _best_available(ordered, selected, lambda item, value=skill: _candidate_can_cover_skill(item, value))
            if candidate is None:
                break
            selected.append(candidate)

    for candidate in ordered:
        if len(selected) >= target_count:
            break
        if _candidate_available(candidate, selected):
            selected.append(candidate)
    selected_with_skills = _apply_skill_graph_selection(selected[:target_count], min_skill_coverage)
    return selected_with_skills or selected[:target_count]


def _skill_coverage_shortage(
    items: Sequence[EnvsForgePlanItem],
    min_skill_coverage: dict[str, int],
) -> dict[str, int]:
    shortage: dict[str, int] = {}
    selected_nodes_by_index = {index: _required_skill_nodes(item) for index, item in enumerate(items)}
    for skill, minimum in min_skill_coverage.items():
        current = sum(1 for nodes in selected_nodes_by_index.values() if skill in nodes)
        shortage[skill] = max(0, minimum - current)
    return shortage


def _candidate_objective(item: EnvsForgePlanItem) -> float:
    if item.predicted_frontier_score is not None:
        return float(item.predicted_frontier_score)
    return float(item.frontier_score)


def _optimization_formulation_label(formulation: str | None) -> str | None:
    if formulation == ENVS_FORGE_ALPHA_OPTIMIZATION_FORMULATION:
        return ENVS_FORGE_ALPHA_OPTIMIZATION_FORMULATION_LABEL
    if formulation == ENVS_FORGE_MILP_OPTIMIZATION_FORMULATION:
        return ENVS_FORGE_MILP_OPTIMIZATION_FORMULATION_LABEL
    return None


def _candidate_can_cover_skill(item: EnvsForgePlanItem, skill: str) -> bool:
    return skill in _required_skill_nodes(item) | _optional_skill_nodes(item)


def _required_skill_nodes(item: EnvsForgePlanItem) -> set[str]:
    subgraph = item.skill_subgraph or build_skill_subgraph(item.profile, item.projection, item.evolution_direction)
    return (set(subgraph.nodes) - set(subgraph.allowed_removed_nodes)) | set(subgraph.preserve_nodes)


def _optional_skill_nodes(item: EnvsForgePlanItem) -> set[str]:
    subgraph = item.skill_subgraph or build_skill_subgraph(item.profile, item.projection, item.evolution_direction)
    return set(subgraph.allowed_added_nodes) | set(subgraph.allowed_removed_nodes)


def _skill_node_universe(
    candidates: Sequence[EnvsForgePlanItem],
    min_skill_coverage: dict[str, int],
) -> tuple[str, ...]:
    nodes: set[str] = set(min_skill_coverage)
    for item in candidates:
        nodes.update(_required_skill_nodes(item))
        nodes.update(_optional_skill_nodes(item))
    return tuple(sorted(nodes))


def _apply_skill_graph_selection(
    items: Sequence[EnvsForgePlanItem],
    min_skill_coverage: dict[str, int],
) -> list[EnvsForgePlanItem] | None:
    selected_nodes_by_index = {index: _required_skill_nodes(item) for index, item in enumerate(items)}
    for skill, minimum in min_skill_coverage.items():
        current = sum(1 for nodes in selected_nodes_by_index.values() if skill in nodes)
        while current < minimum:
            candidate_index = _best_skill_node_host(items, selected_nodes_by_index, skill)
            if candidate_index is None:
                return None
            selected_nodes_by_index[candidate_index].add(skill)
            current += 1
    return [
        _with_selected_skill_nodes(item, selected_nodes_by_index[index])
        for index, item in enumerate(items)
    ]


def _best_skill_node_host(
    items: Sequence[EnvsForgePlanItem],
    selected_nodes_by_index: dict[int, set[str]],
    skill: str,
) -> int | None:
    best: tuple[float, str, int] | None = None
    for index, item in enumerate(items):
        if skill in selected_nodes_by_index[index] or skill not in _optional_skill_nodes(item):
            continue
        score = (_candidate_objective(item), item.seed.name, index)
        if best is None or score > best:
            best = score
    return best[2] if best is not None else None


def _with_selected_skill_nodes(
    item: EnvsForgePlanItem,
    selected_nodes: set[str],
    solver_result: dict[str, object] | None = None,
) -> EnvsForgePlanItem:
    subgraph = item.skill_subgraph or build_skill_subgraph(item.profile, item.projection, item.evolution_direction)
    ordered_nodes = _ordered_selected_nodes(subgraph, selected_nodes)
    selected_node_set = set(ordered_nodes)
    selected_subgraph = SkillSubgraph(
        nodes=ordered_nodes,
        edges=tuple(edge for edge in subgraph.edges if edge.source in selected_node_set and edge.target in selected_node_set),
        preserve_nodes=subgraph.preserve_nodes,
        allowed_added_nodes=subgraph.allowed_added_nodes,
        allowed_removed_nodes=subgraph.allowed_removed_nodes,
    )
    return _make_plan_item(
        seed=item.seed,
        profile=item.profile,
        pass_rate=item.pass_rate,
        frontier_score=item.frontier_score,
        band=item.band,
        projection=item.projection,
        evolution_direction=item.evolution_direction,
        target_difficulty=item.target_difficulty,
        sample_index=item.sample_index,
        projection_policy=item.projection_policy,
        predicted_pass_rate=item.predicted_pass_rate,
        predicted_frontier_score=item.predicted_frontier_score,
        artifact_edit_mask=item.artifact_edit_mask,
        skill_subgraph=selected_subgraph,
        solver_backend=item.solver_backend,
        solver_result=solver_result or item.solver_result,
        optimization_formulation=item.optimization_formulation,
    )


def _ordered_selected_nodes(subgraph: SkillSubgraph, selected_nodes: set[str]) -> tuple[str, ...]:
    order = [*subgraph.nodes, *subgraph.allowed_added_nodes, *subgraph.allowed_removed_nodes]
    ordered = [node for node in _unique(order) if node in selected_nodes]
    leftovers = sorted(selected_nodes - set(ordered))
    return tuple([*ordered, *leftovers])


def _coverage_count(items: Sequence[EnvsForgePlanItem], predicate) -> int:
    return sum(1 for item in items if predicate(item))


def _best_available(candidates: Sequence[EnvsForgePlanItem], selected: Sequence[EnvsForgePlanItem], predicate) -> EnvsForgePlanItem | None:
    for candidate in candidates:
        if predicate(candidate) and _candidate_available(candidate, selected):
            return candidate
    return None


def _candidate_available(candidate: EnvsForgePlanItem, selected: Sequence[EnvsForgePlanItem]) -> bool:
    return candidate.seed.name not in {item.seed.name for item in selected}


def _with_solver_backend(items: Sequence[EnvsForgePlanItem], backend: str) -> list[EnvsForgePlanItem]:
    result: list[EnvsForgePlanItem] = []
    for item in items:
        solver_result = dict(item.solver_result or {})
        solver_result.setdefault("backend", backend)
        formulation = item.optimization_formulation or _optimization_formulation_for_policy(item.projection_policy)
        if formulation:
            solver_result.setdefault("optimization_formulation", formulation)
            solver_result.setdefault(
                "optimization_formulation_label",
                ENVS_FORGE_MILP_OPTIMIZATION_FORMULATION_LABEL
                if formulation == ENVS_FORGE_MILP_OPTIMIZATION_FORMULATION
                else ENVS_FORGE_ALPHA_OPTIMIZATION_FORMULATION_LABEL,
            )
        result.append(
            _make_plan_item(
                seed=item.seed,
                profile=item.profile,
                pass_rate=item.pass_rate,
                frontier_score=item.frontier_score,
                band=item.band,
                projection=item.projection,
                evolution_direction=item.evolution_direction,
                target_difficulty=item.target_difficulty,
                sample_index=item.sample_index,
                projection_policy=item.projection_policy,
                predicted_pass_rate=item.predicted_pass_rate,
                predicted_frontier_score=item.predicted_frontier_score,
                artifact_edit_mask=item.artifact_edit_mask,
                skill_subgraph=item.skill_subgraph,
                solver_backend=backend,
                solver_result=solver_result,
                optimization_formulation=formulation,
            )
        )
    return result


def _optimization_formulation_for_policy(projection_policy: str) -> str | None:
    if projection_policy == "joint_optimization":
        return ENVS_FORGE_ALPHA_OPTIMIZATION_FORMULATION
    return None


def _lookup_seed_rewards(seed_name: str, rewards_by_seed: dict[str, Sequence[float]]) -> Sequence[float]:
    if seed_name in rewards_by_seed:
        return rewards_by_seed[seed_name]
    matches = [
        values
        for key, values in rewards_by_seed.items()
        if seed_name.startswith(key) or key.startswith(seed_name)
    ]
    if not matches:
        return []
    merged: list[float] = []
    for values in matches:
        merged.extend(float(value) for value in values)
    return merged


def _observed_reward_mean(rewards: Sequence[float]) -> float:
    if not rewards:
        return 0.5
    return sum(float(reward) for reward in rewards) / len(rewards)


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def classify_band(pass_rate: float, *, low: float = 0.2, high: float = 0.8) -> str:
    if pass_rate >= high:
        return "easy"
    if pass_rate <= low:
        return "hard"
    return "frontier"


def projection_for_band(band: str) -> str:
    if band == "easy":
        return "increase_complexity"
    if band == "hard":
        return "reduce_complexity"
    return "diversify_frontier"


def target_difficulty_for_projection(projection: str) -> str:
    if projection == "reduce_complexity":
        return "easy"
    return "medium"


def target_difficulty_for_band(band: str) -> str:
    if band == "easy":
        return "medium"
    if band == "hard":
        return "easy"
    return "medium"


def build_envs_forge_prompt(item: EnvsForgePlanItem, *, model: str) -> str:
    seed_files = {path: _shorten(text, 3500) for path, text in item.seed.files.items() if _include_seed_file(path)}
    projection_instruction = _projection_instruction(item.projection)
    artifact_edit_mask = item.artifact_edit_mask or artifact_edit_mask_for_projection(item.projection)
    skill_subgraph = item.skill_subgraph or build_skill_subgraph(item.profile, item.projection, item.evolution_direction)
    payload = {
        "seed": item.seed.name,
        "band": item.band,
        "pass_rate": round(item.pass_rate, 4),
        "learning_frontier_score": round(item.frontier_score, 4),
        "projection": item.projection,
        "evolution_direction": item.evolution_direction,
        "action_variant": item.action_variant,
        "projection_policy": item.projection_policy,
        "optimization_formulation": item.optimization_formulation,
        "target_difficulty": item.target_difficulty,
        "skill_profile": {
            "domains": item.profile.domains,
            "tools": item.profile.tools,
            "artifacts": item.profile.artifacts,
            "operations": item.profile.operations,
            "constraints": item.profile.constraints,
        },
        "artifact_edit_mask": artifact_edit_mask.as_json(),
        "skill_subgraph": skill_subgraph.as_json(),
    }
    if item.solver_backend:
        payload["solver_backend"] = item.solver_backend
    if item.predicted_pass_rate is not None:
        payload["predicted_pass_rate_after_projection"] = round(item.predicted_pass_rate, 4)
    if item.predicted_frontier_score is not None:
        payload["predicted_frontier_score_after_projection"] = round(item.predicted_frontier_score, 4)
    evolution_direction_instruction = _evolution_direction_instruction(item.evolution_direction)
    return f"""Synthesize one terminal-bench-style task directory using Envs-FORGE.

Return exactly one JSON object, no markdown.
Required top-level keys: task_name, instruction_md, files, metadata.

Envs-FORGE objective:
- Project the seed task onto the current agent's learning frontier.
- preserve the target skill subgraph instead of jumping to an unrelated easy task.
- For easy seeds, increase complexity toward a medium learning-frontier task.
- For hard seeds, reduce compositional complexity while preserving the target skill subgraph.
- For frontier seeds, diversify concrete data, constraints, and fixtures while keeping difficulty near the learning frontier.

Projection decision:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Projection instruction:
{projection_instruction}

Evolution direction:
{evolution_direction_instruction}

Artifact edit mask:
{json.dumps(artifact_edit_mask.as_json(), ensure_ascii=False, indent=2)}

Skill subgraph:
{json.dumps(skill_subgraph.as_json(), ensure_ascii=False, indent=2)}

The generated task must be internally consistent:
- Treat artifact_edit_mask.required_artifacts as mandatory synchronized edits. If an artifact is required, change it to match the projected task rather than copying the seed unchanged.
- Use artifact_edit_mask.operators as the allowed edit operator families for each artifact.
- Preserve skill_subgraph.preserve_nodes in the generated task.
- Only add skill nodes that are consistent with skill_subgraph.allowed_added_nodes.
- For reduce_complexity, you may remove or simulate skill_subgraph.allowed_removed_nodes only when the core preserved nodes remain testable.
- Rewrite the user-facing task description in instruction.md.
- Update the reference solution in solution/solve.sh.
- Update the test runner in tests/test.sh.
- Update the verifier tests in tests/test_outputs.py.
- Update the execution environment in environment/Dockerfile.
- If the changed task needs fixture/helper changes, include those files too, for example tests/grader.py or environment/data/input.txt.
- Keep task.toml changes minimal unless they are required by the new task.
- Keep the generated artifact small: prefer compact fixtures, solution/solve.sh under 150 lines, tests/test_outputs.py under 220 lines, and no long embedded essays or large generated datasets.
- Reuse the seed environment/Dockerfile as much as possible. Do not modify apt sources. Do not add apt/pip/npm installs unless the seed already needs that dependency class and the generated task cannot run without it.
- For reduce_complexity projections, it is acceptable to turn a system-level hard task into a file-level simulation that preserves the target skill subgraph. Prefer deterministic file/config/log fixtures over creating real OS users, real systemd services, or privileged daemons.
- Do not include terminal-bench canary comments or API keys.
- Do not merely restate the seed task; create a small but real Envs-FORGE projection.
- environment/Dockerfile is built with environment/ as the Docker build context. Do not COPY files from the task root such as tests/, solution/, instruction.md, or task.toml.
- If Dockerfile needs fixture data, include it under environment/ and COPY it from the environment/ build context.
- Pin base images to explicit tags already known to work in the seed, such as ubuntu:22.04, ubuntu:24.04, or python:3.13-slim-bookworm. Do not use :latest.
- Domestic mirrors are allowed and recommended for apt/pip/npm when network installs are needed.
- tests/test.sh must always write /logs/verifier/reward.txt: write 1 on pass and 0 on fail. Do not let set -e exit before writing reward on pytest failure; capture the pytest status explicitly.
- Do not hide constraints in tests. If tests require exact thresholds, tie-breaking, file names, output keys, or fixture values, state those requirements explicitly in instruction.md.
- Before returning, mentally execute: build environment/Dockerfile, run solution/solve.sh, then run tests/test.sh. The result must be reward 1.

Augmentation model: {model}
Sample index: {item.sample_index}

Return JSON shape:
{{
  "task_name": "lowercase-hyphenated-name",
  "instruction_md": "full generated task instruction",
  "files": {{
    "instruction.md": "...",
    "solution/solve.sh": "...",
    "tests/test.sh": "...",
    "tests/test_outputs.py": "...",
    "environment/Dockerfile": "..."
  }},
  "metadata": {{
    "difficulty": "{item.target_difficulty}",
    "summary": "short description of the learning-frontier projection",
    "envs_forge": {json.dumps(payload, ensure_ascii=False)}
  }}
}}

Seed task name: {item.seed.name}
Seed files for style and structure:
{json.dumps(seed_files, ensure_ascii=False, indent=2)}
"""


def load_rewards_from_harbor_result(path: str | Path) -> dict[str, list[float]]:
    result_path = Path(path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rewards: dict[str, list[float]] = {}
    evals = ((payload.get("stats") or {}).get("evals") or {})
    if not isinstance(evals, dict):
        return rewards
    for eval_payload in evals.values():
        if not isinstance(eval_payload, dict):
            continue
        reward_stats = ((eval_payload.get("reward_stats") or {}).get("reward") or {})
        if not isinstance(reward_stats, dict):
            continue
        for reward_text, trial_names in reward_stats.items():
            try:
                reward = float(reward_text)
            except (TypeError, ValueError):
                continue
            if not isinstance(trial_names, list):
                continue
            for trial_name in trial_names:
                if not isinstance(trial_name, str):
                    continue
                seed_name = trial_name.split("__", 1)[0]
                rewards.setdefault(seed_name, []).append(reward)
    return rewards


def _match_category(text: str, keyword_map: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    matched = sorted(label for label, needles in keyword_map.items() if any(needle in text for needle in needles))
    return tuple(matched)


def _skill_edges(profile: SkillProfile) -> tuple[SkillEdge, ...]:
    edges: list[SkillEdge] = []
    edges.extend(_connect(profile.domains, profile.tools, "uses_tool"))
    edges.extend(_connect(profile.tools, profile.artifacts, "reads_or_writes"))
    edges.extend(_connect(profile.artifacts, profile.operations, "operation_on_artifact"))
    edges.extend(_connect(profile.operations, profile.constraints, "constrained_by"))
    return tuple(edges)


def _connect(sources: Sequence[str], targets: Sequence[str], relation: str) -> tuple[SkillEdge, ...]:
    return tuple(SkillEdge(source=source, target=target, relation=relation) for source in sources for target in targets)


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def _balanced_order(items: Sequence[EnvsForgePlanItem]) -> list[EnvsForgePlanItem]:
    by_band = {
        "frontier": sorted((item for item in items if item.band == "frontier"), key=lambda item: -item.frontier_score),
        "hard": sorted((item for item in items if item.band == "hard"), key=lambda item: item.pass_rate),
        "easy": sorted((item for item in items if item.band == "easy"), key=lambda item: -item.pass_rate),
    }
    order = ["frontier", "hard", "easy"]
    result: list[EnvsForgePlanItem] = []
    while any(by_band.values()):
        for band in order:
            bucket = by_band[band]
            if bucket:
                result.append(bucket.pop(0))
    return result


def _projection_instruction(projection: str) -> str:
    if projection == "increase_complexity":
        return (
            "The seed appears easy for the current agent. Increase complexity with one or two concrete constraints, "
            "extra edge cases, stricter output requirements, or a larger fixture while preserving the seed skill subgraph."
        )
    if projection == "reduce_complexity":
        return (
            "The seed appears beyond the current agent. Reduce compositional complexity, split away secondary systems, "
            "or create a smaller bridge task while preserving the target skill subgraph and its most important artifacts."
        )
    return (
        "The seed is near the learning frontier. Keep difficulty similar, diversify data and constraints, and avoid "
        "large simplification or unrelated breadth jumps."
    )
