from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .terminal_bench import TerminalBenchSeedTask, _include_seed_file, _shorten


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
class EnvsForgePlanItem:
    seed: TerminalBenchSeedTask
    profile: SkillProfile
    pass_rate: float
    frontier_score: float
    band: str
    projection: str
    target_difficulty: str
    sample_index: int


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
) -> list[EnvsForgePlanItem]:
    rewards_by_seed = rewards_by_seed or {}
    candidates: list[EnvsForgePlanItem] = []
    for seed in seeds:
        reward_values = _lookup_seed_rewards(seed.name, rewards_by_seed)
        pass_rate = _observed_reward_mean(reward_values)
        frontier_score = learning_frontier_score(pass_rate, target=target, sigma=sigma) if reward_values else 0.0
        band = classify_band(pass_rate, low=low, high=high)
        candidates.append(
            EnvsForgePlanItem(
                seed=seed,
                profile=extract_skill_profile(seed),
                pass_rate=pass_rate,
                frontier_score=frontier_score,
                band=band,
                projection=projection_for_band(band),
                target_difficulty=target_difficulty_for_band(band),
                sample_index=0,
            )
        )

    ordered = _balanced_order(candidates)
    selected: list[EnvsForgePlanItem] = []
    for index, item in enumerate(ordered[: max(0, target_count)]):
        selected.append(
            EnvsForgePlanItem(
                seed=item.seed,
                profile=item.profile,
                pass_rate=item.pass_rate,
                frontier_score=item.frontier_score,
                band=item.band,
                projection=item.projection,
                target_difficulty=item.target_difficulty,
                sample_index=index,
            )
        )
    return selected


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


def target_difficulty_for_band(band: str) -> str:
    if band == "easy":
        return "medium"
    if band == "hard":
        return "easy"
    return "medium"


def build_envs_forge_prompt(item: EnvsForgePlanItem, *, model: str) -> str:
    seed_files = {path: _shorten(text, 3500) for path, text in item.seed.files.items() if _include_seed_file(path)}
    projection_instruction = _projection_instruction(item.projection)
    payload = {
        "seed": item.seed.name,
        "band": item.band,
        "pass_rate": round(item.pass_rate, 4),
        "learning_frontier_score": round(item.frontier_score, 4),
        "projection": item.projection,
        "target_difficulty": item.target_difficulty,
        "skill_profile": {
            "domains": item.profile.domains,
            "tools": item.profile.tools,
            "artifacts": item.profile.artifacts,
            "operations": item.profile.operations,
            "constraints": item.profile.constraints,
        },
    }
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

The generated task must be internally consistent:
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
