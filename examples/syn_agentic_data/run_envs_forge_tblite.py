from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdgsystem.agentic_data.envs_forge import (  # noqa: E402
    ArtifactEditMask,
    EnvsForgePlanItem,
    SkillSubgraph,
    artifact_edit_mask_for_projection,
    build_envs_forge_prompt,
    build_skill_subgraph,
    load_rewards_from_harbor_result,
    plan_envs_forge_batch,
)
from sdgsystem.agentic_data.envs_forge import METHOD_VERSION as ENVS_FORGE_ALPHA_METHOD_VERSION  # noqa: E402
from sdgsystem.agentic_data.terminal_bench import (  # noqa: E402
    TerminalBenchSeedTask,
    build_terminal_bench_repair_prompt,
    load_seed_task,
    materialize_artifact,
    parse_json_object,
    summarize_terminal_bench_failure,
    validate_materialized_task,
    validate_terminal_bench_artifact,
    write_terminal_bench_outputs,
)


DEFAULT_TBLITE_ROOT = "/Volumes/PSSD/code/My-Agent/Agent-Data/OpenThoughts-TBLite"
DEFAULT_HARBOR_BIN = "/Volumes/PSSD/code/My-Agent/Agent-Benchmark/terminal-bench-science/.venv/bin/harbor"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/envs_forge_tblite_mvp"


class OpenAICompatibleModelClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        retries: int,
        retry_delay: float,
    ) -> None:
        self.model = model
        self.retries = retries
        self.retry_delay = retry_delay
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        max_tokens = int(kwargs.pop("max_tokens", 20000))
        temperature = float(kwargs.pop("temperature", 0.7))
        timeout = kwargs.pop("timeout", None)
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    **kwargs,
                )
                return (completion.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                print(f"[api-retry] attempt={attempt}/{self.retries} error={exc}", flush=True)
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
        raise RuntimeError(f"API generation failed after {self.retries} attempts") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Envs-FORGE MVP synthesis on OpenThoughts-TBLite seeds.")
    parser.add_argument("--source-root", default=DEFAULT_TBLITE_ROOT)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--performance-result", action="append", default=[])
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--limit-seeds", type=int, default=None)
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--synthesis-model", default="deepseek-v4-flash")
    parser.add_argument("--projection-policy", choices=["band_rule", "calibrated_transfer", "joint_optimization"], default="band_rule")
    parser.add_argument(
        "--optimization-formulation",
        choices=["binary_integer_programming", "mixed_integer_programming"],
        default="binary_integer_programming",
    )
    parser.add_argument("--min-skill-coverage", action="append", default=[])
    parser.add_argument("--downstream-model", default="openai/qwen3-8b")
    parser.add_argument("--completion-max-tokens", type=int, default=20000)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--api-retries", type=int, default=8)
    parser.add_argument("--api-retry-delay", type=float, default=20.0)
    parser.add_argument("--sample-attempts", type=int, default=2)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--validation-mode", choices=["static", "gold_solution"], default="gold_solution")
    parser.add_argument("--harbor-bin", default=DEFAULT_HARBOR_BIN)
    parser.add_argument("--harbor-env", default="docker")
    parser.add_argument("--gold-validation-timeout", type=int, default=None)
    parser.add_argument("--gold-validation-retries", type=int, default=2)
    parser.add_argument("--gold-validation-retry-delay", type=float, default=20.0)
    parser.add_argument("--run-downstream", action="store_true")
    parser.add_argument("--downstream-timeout", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = load_env_values(Path(args.env_file))
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    method_dir = output_root / "envs_forge"
    tasks_dir = method_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    rewards = load_performance_rewards(args.performance_result)
    seeds = discover_seed_tasks(source_root, task_names=args.task_names, limit=args.limit_seeds)
    min_skill_coverage = parse_count_constraints(args.min_skill_coverage)
    plan = plan_envs_forge_batch(
        seeds,
        rewards_by_seed=rewards,
        target_count=args.target_count,
        projection_policy=args.projection_policy,
        min_skill_coverage=min_skill_coverage,
        optimization_formulation=args.optimization_formulation,
    )
    write_json(method_dir / "envs_forge_plan.json", [plan_item_to_json(item) for item in plan])
    write_json(method_dir / "envs_forge_solver_result.json", batch_solver_result_to_json(plan))
    print_plan(plan, method_dir)
    if args.dry_run:
        return

    client = OpenAICompatibleModelClient(
        model=args.synthesis_model,
        api_key=required_env(env, "DEEPSEEK_API_KEY"),
        base_url=required_env(env, "DEEPSEEK_BASE_URL"),
        retries=args.api_retries,
        retry_delay=args.api_retry_delay,
    )
    started_at = utc_now()
    records = load_json_list(method_dir / "terminal_bench_records.json")
    accepted = [record for record in records if is_matched(record, args.validation_mode)]
    completed_plan_keys = {
        (record.get("seed_task"), int(record.get("sample_index") or 0))
        for record in records
        if record.get("strategy") == "envs_forge" and is_matched(record, args.validation_mode)
    }

    for item in plan:
        if len(accepted) >= args.target_count:
            break
        key = (item.seed.name, item.sample_index)
        if key in completed_plan_keys:
            print(f"[skip] accepted seed={item.seed.name} sample={item.sample_index}", flush=True)
            continue
        record = synthesize_with_retries(
            item=item,
            model=args.synthesis_model,
            client=client,
            tasks_dir=tasks_dir,
            completion_max_tokens=args.completion_max_tokens,
            request_timeout=args.request_timeout,
            sample_attempts=args.sample_attempts,
            repair_attempts=args.repair_attempts,
            validation_mode=args.validation_mode,
            harbor_bin=args.harbor_bin,
            harbor_env=args.harbor_env,
            gold_validation_timeout=args.gold_validation_timeout,
            gold_validation_retries=args.gold_validation_retries,
            gold_validation_retry_delay=args.gold_validation_retry_delay,
        )
        records.append(record)
        if is_matched(record, args.validation_mode):
            accepted.append(record)
        write_terminal_bench_outputs(method_dir, records, accepted, started_at)
        print(
            f"[record] seed={item.seed.name} band={item.band} projection={item.projection} "
            f"matched={is_matched(record, args.validation_mode)} accepted={len(accepted)}/{args.target_count}",
            flush=True,
        )

    summary = write_terminal_bench_outputs(method_dir, records, accepted, started_at)
    eval_root = build_accepted_eval_root(method_dir, accepted[: args.target_count])
    summary["envs_forge"] = {
        "method_version": ENVS_FORGE_ALPHA_METHOD_VERSION,
        "target_count": args.target_count,
        "eval_root": str(eval_root),
        "projection_policy": args.projection_policy,
        "optimization_formulation": single_value(item.optimization_formulation for item in plan),
        "min_skill_coverage": min_skill_coverage,
        "synthesis_model": args.synthesis_model,
        "downstream_model": args.downstream_model,
    }
    write_json(method_dir / "summary.json", summary)

    if args.run_downstream and accepted:
        run_downstream_eval(
            eval_root=eval_root,
            output_root=output_root,
            harbor_bin=args.harbor_bin,
            harbor_env=args.harbor_env,
            model=args.downstream_model,
            base_url=required_env(env, "BIANLIAN_BASE_URL"),
            api_key=required_env(env, "BIANLIAN_API_KEY"),
            timeout=args.downstream_timeout,
            n_tasks=min(args.target_count, len(accepted)),
        )


def synthesize_with_retries(
    *,
    item: EnvsForgePlanItem,
    model: str,
    client: OpenAICompatibleModelClient,
    tasks_dir: Path,
    completion_max_tokens: int,
    request_timeout: int,
    sample_attempts: int,
    repair_attempts: int,
    validation_mode: str,
    harbor_bin: str,
    harbor_env: str,
    gold_validation_timeout: int | None,
    gold_validation_retries: int,
    gold_validation_retry_delay: float,
) -> dict[str, Any]:
    last_record: dict[str, Any] | None = None
    for attempt in range(1, sample_attempts + 1):
        record = synthesize_once(
            item=item,
            model=model,
            client=client,
            tasks_dir=tasks_dir,
            completion_max_tokens=completion_max_tokens,
            request_timeout=request_timeout,
            validation_mode=validation_mode,
            harbor_bin=harbor_bin,
            harbor_env=harbor_env,
            gold_validation_timeout=gold_validation_timeout,
            gold_validation_retries=gold_validation_retries,
            gold_validation_retry_delay=gold_validation_retry_delay,
        )
        record["attempt"] = attempt
        if is_matched(record, validation_mode):
            return record
        last_record = record
        for repair_index in range(1, repair_attempts + 1):
            record = repair_once(
                item=item,
                previous_record=last_record,
                model=model,
                client=client,
                tasks_dir=tasks_dir,
                completion_max_tokens=completion_max_tokens,
                request_timeout=request_timeout,
                validation_mode=validation_mode,
                harbor_bin=harbor_bin,
                harbor_env=harbor_env,
                gold_validation_timeout=gold_validation_timeout,
                gold_validation_retries=gold_validation_retries,
                gold_validation_retry_delay=gold_validation_retry_delay,
                repair_index=repair_index,
            )
            record["attempt"] = attempt
            last_record = record
            if is_matched(record, validation_mode):
                return record
            verification = record.get("verification") if isinstance(record.get("verification"), dict) else {}
            print(
                f"[repair-retry] seed={item.seed.name} repair={repair_index}/{repair_attempts} "
                f"status={verification.get('status')}",
                flush=True,
            )
    return last_record or failure_record(item, "generation failed")


def synthesize_once(
    *,
    item: EnvsForgePlanItem,
    model: str,
    client: OpenAICompatibleModelClient,
    tasks_dir: Path,
    completion_max_tokens: int,
    request_timeout: int,
    validation_mode: str,
    harbor_bin: str,
    harbor_env: str,
    gold_validation_timeout: int | None,
    gold_validation_retries: int,
    gold_validation_retry_delay: float,
) -> dict[str, Any]:
    prompt = build_envs_forge_prompt(item, model=model)
    raw = client.generate(prompt, max_tokens=completion_max_tokens, timeout=request_timeout)
    return materialize_and_validate_record(
        item=item,
        raw=raw,
        model=model,
        tasks_dir=tasks_dir,
        validation_mode=validation_mode,
        harbor_bin=harbor_bin,
        harbor_env=harbor_env,
        gold_validation_timeout=gold_validation_timeout,
        gold_validation_retries=gold_validation_retries,
        gold_validation_retry_delay=gold_validation_retry_delay,
        repair_index=None,
    )


def repair_once(
    *,
    item: EnvsForgePlanItem,
    previous_record: dict[str, Any],
    model: str,
    client: OpenAICompatibleModelClient,
    tasks_dir: Path,
    completion_max_tokens: int,
    request_timeout: int,
    validation_mode: str,
    harbor_bin: str,
    harbor_env: str,
    gold_validation_timeout: int | None,
    gold_validation_retries: int,
    gold_validation_retry_delay: float,
    repair_index: int,
) -> dict[str, Any]:
    previous_artifact = previous_record.get("artifact")
    if not isinstance(previous_artifact, dict):
        return previous_record
    prompt = build_terminal_bench_repair_prompt(
        seed=item.seed,
        artifact=previous_artifact,
        failure_summary=summarize_terminal_bench_failure(previous_record),
        strategy="envs_forge",
        evol_direction=item.evolution_direction,
        repair_index=repair_index,
        model=model,
    )
    raw = client.generate(prompt, max_tokens=completion_max_tokens, timeout=request_timeout)
    return materialize_and_validate_record(
        item=item,
        raw=raw,
        model=model,
        tasks_dir=tasks_dir,
        validation_mode=validation_mode,
        harbor_bin=harbor_bin,
        harbor_env=harbor_env,
        gold_validation_timeout=gold_validation_timeout,
        gold_validation_retries=gold_validation_retries,
        gold_validation_retry_delay=gold_validation_retry_delay,
        repair_index=repair_index,
    )


def materialize_and_validate_record(
    *,
    item: EnvsForgePlanItem,
    raw: str,
    model: str,
    tasks_dir: Path,
    validation_mode: str,
    harbor_bin: str,
    harbor_env: str,
    gold_validation_timeout: int | None,
    gold_validation_retries: int,
    gold_validation_retry_delay: float,
    repair_index: int | None,
) -> dict[str, Any]:
    artifact, parse_error = parse_json_object(raw)
    schema_errors = validate_terminal_bench_artifact(artifact)
    task_dir = None
    verification = None
    if artifact and not schema_errors:
        metadata = artifact.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("source_seed", item.seed.name)
            metadata.setdefault("synthetic_strategy", "envs_forge")
            metadata.setdefault("method_version", ENVS_FORGE_ALPHA_METHOD_VERSION)
            metadata.setdefault("envs_forge_projection", item.projection)
            metadata.setdefault("envs_forge_evolution_direction", item.evolution_direction)
            metadata.setdefault("envs_forge_action_variant", item.action_variant)
            metadata.setdefault("envs_forge_projection_policy", item.projection_policy)
            metadata.setdefault("envs_forge_band", item.band)
            metadata.setdefault("envs_forge_pass_rate", item.pass_rate)
            metadata.setdefault("envs_forge_frontier_score", item.frontier_score)
            if item.predicted_pass_rate is not None:
                metadata.setdefault("envs_forge_predicted_pass_rate", item.predicted_pass_rate)
            if item.predicted_frontier_score is not None:
                metadata.setdefault("envs_forge_predicted_frontier_score", item.predicted_frontier_score)
            if item.solver_backend:
                metadata.setdefault("envs_forge_solver_backend", item.solver_backend)
            if item.optimization_formulation:
                metadata.setdefault("envs_forge_optimization_formulation", item.optimization_formulation)
            optimization_label = (item.solver_result or {}).get("optimization_formulation_label")
            if optimization_label:
                metadata.setdefault("envs_forge_optimization_formulation_label", optimization_label)
            metadata.setdefault("envs_forge_artifact_edit_mask", resolved_artifact_edit_mask(item).as_json())
            metadata.setdefault("envs_forge_skill_subgraph", resolved_skill_subgraph(item).as_json())
            metadata.setdefault("augmentation_model", model)
            metadata.setdefault("date_created", datetime.now(timezone.utc).date().isoformat())
            if repair_index is not None:
                metadata.setdefault("repair_attempt", repair_index)
        artifact_name = str(artifact.get("task_name") or "")
        task_dir = materialize_artifact(
            item.seed,
            artifact,
            tasks_dir,
            directory_name=task_directory_name(item.seed.name, artifact_name, item.sample_index, repair_index),
        )
        verification = validate_materialized_task(
            task_dir=task_dir,
            artifact=artifact,
            validation_mode=validation_mode,
            jobs_dir=tasks_dir.parent / "gold_solution_jobs",
            harbor_bin=harbor_bin,
            environment=harbor_env,
            force_build=True,
            timeout=gold_validation_timeout,
            gold_validation_retries=gold_validation_retries,
            gold_validation_retry_delay=gold_validation_retry_delay,
        )
    return {
        "seed_task": item.seed.name,
        "task_name": str((artifact or {}).get("task_name") or ""),
        "task_path": str(task_dir) if task_dir else None,
        "strategy": "envs_forge",
        "method_version": ENVS_FORGE_ALPHA_METHOD_VERSION,
        "evol_direction": item.evolution_direction,
        "action_variant": item.action_variant,
        "sample_index": item.sample_index,
        "instruction": str((artifact or {}).get("instruction_md") or ""),
        "raw_response": raw,
        "artifact": artifact,
        "parse_error": parse_error,
        "schema_errors": schema_errors,
        "verification": verification,
        "envs_forge": plan_item_to_json(item),
        "completed_at": utc_now(),
    }


def build_accepted_eval_root(method_dir: Path, accepted: list[dict[str, Any]]) -> Path:
    eval_root = method_dir / "eval/envs_forge_verified_task_root"
    eval_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, record in enumerate(accepted, start=1):
        task_path = Path(str(record["task_path"]))
        link_path = eval_root / f"{index:03d}-{task_path.name}"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(os.path.relpath(task_path, start=eval_root), target_is_directory=True)
        manifest.append({"index": index, "eval_task_dir": str(link_path), "source_task_path": str(task_path)})
    write_json(eval_root / "envs_forge_manifest.json", {"num_tasks": len(manifest), "tasks": manifest})
    return eval_root


def run_downstream_eval(
    *,
    eval_root: Path,
    output_root: Path,
    harbor_bin: str,
    harbor_env: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int | None,
    n_tasks: int,
) -> None:
    jobs_dir = output_root / "eval/envs_forge_terminus2_qwen3_8b/jobs"
    logs_dir = output_root / "eval/envs_forge_terminus2_qwen3_8b/logs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    command = [
        harbor_bin,
        "run",
        "--path",
        str(eval_root),
        "--jobs-dir",
        str(jobs_dir),
        "--agent",
        "terminus-2",
        "--model",
        model,
        "--agent-kwarg",
        f"api_base={base_url}",
        "--agent-kwarg",
        "collect_rollout_details=True",
        "--agent-kwarg",
        "store_all_messages=True",
        "--agent-include-logs",
        "**/*",
        "--env",
        harbor_env,
        "--force-build",
        "--n-concurrent",
        "1",
        "--n-tasks",
        str(n_tasks),
        "--yes",
    ]
    write_json(logs_dir / "downstream_command.json", {"command": sanitize_command(command), "model": model, "n_tasks": n_tasks})
    run_env = dict(os.environ)
    run_env["OPENAI_API_KEY"] = api_key
    run_env["OPENAI_BASE_URL"] = base_url
    run_env["HARBOR_API_KEY"] = api_key
    run_env["HARBOR_OPENAI_BASE_URL"] = base_url
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, env=run_env)
    (logs_dir / "downstream_stdout.log").write_text(completed.stdout[-20000:], encoding="utf-8")
    (logs_dir / "downstream_stderr.log").write_text(completed.stderr[-20000:], encoding="utf-8")
    write_json(logs_dir / "downstream_result.json", {"returncode": completed.returncode, "jobs_dir": str(jobs_dir)})


def load_env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return {**values, **{key: value for key, value in os.environ.items() if key.endswith(("_API_KEY", "_BASE_URL"))}}


def required_env(env: dict[str, str], name: str) -> str:
    value = env.get(name)
    if not value:
        raise RuntimeError(f"{name} is required in the environment or .env file")
    return value


def discover_seed_tasks(source_root: Path, *, task_names: list[str] | None, limit: int | None) -> list[TerminalBenchSeedTask]:
    names = [
        path.name
        for path in sorted(source_root.iterdir())
        if path.is_dir() and (path / "instruction.md").is_file() and (path / "task.toml").is_file()
    ]
    if task_names:
        missing = sorted(set(task_names) - set(names))
        if missing:
            raise ValueError(f"Unknown task names: {missing}")
        names = [name for name in task_names if name in names]
    if limit is not None:
        names = names[:limit]
    return [load_seed_task(source_root / name) for name in names]


def load_performance_rewards(paths: list[str]) -> dict[str, list[float]]:
    rewards: dict[str, list[float]] = {}
    for path_text in paths:
        for seed, values in load_rewards_from_harbor_result(path_text).items():
            rewards.setdefault(seed, []).extend(values)
    return rewards


def is_matched(record: dict[str, Any], validation_mode: str) -> bool:
    verification = record.get("verification") if isinstance(record.get("verification"), dict) else {}
    if not verification.get("matched"):
        return False
    if validation_mode == "gold_solution":
        return verification.get("status") == "gold_solution_validated"
    return True


def plan_item_to_json(item: EnvsForgePlanItem) -> dict[str, Any]:
    return {
        "seed_task": item.seed.name,
        "pass_rate": item.pass_rate,
        "frontier_score": item.frontier_score,
        "band": item.band,
        "projection": item.projection,
        "evolution_direction": item.evolution_direction,
        "action_variant": item.action_variant,
        "projection_policy": item.projection_policy,
        "target_difficulty": item.target_difficulty,
        "sample_index": item.sample_index,
        "predicted_pass_rate": item.predicted_pass_rate,
        "predicted_frontier_score": item.predicted_frontier_score,
        "optimization_formulation": item.optimization_formulation,
        "optimization_formulation_label": (item.solver_result or {}).get("optimization_formulation_label"),
        "solver_backend": item.solver_backend,
        "solver_result": item.solver_result,
        "artifact_edit_mask": resolved_artifact_edit_mask(item).as_json(),
        "skill_subgraph": resolved_skill_subgraph(item).as_json(),
        "skills": {
            "domains": item.profile.domains,
            "tools": item.profile.tools,
            "artifacts": item.profile.artifacts,
            "operations": item.profile.operations,
            "constraints": item.profile.constraints,
        },
    }


def resolved_artifact_edit_mask(item: EnvsForgePlanItem) -> ArtifactEditMask:
    return item.artifact_edit_mask or artifact_edit_mask_for_projection(item.projection)


def resolved_skill_subgraph(item: EnvsForgePlanItem) -> SkillSubgraph:
    return item.skill_subgraph or build_skill_subgraph(item.profile, item.projection, item.evolution_direction)


def batch_solver_result_to_json(plan: list[EnvsForgePlanItem]) -> dict[str, Any]:
    solver_items = [item for item in plan if item.solver_result]
    if not solver_items:
        return {"backend": None, "selected_x": [], "selected_u": []}

    first = solver_items[0].solver_result or {}
    selected_x = []
    selected_u = []
    for item in solver_items:
        result = item.solver_result or {}
        x_payload = result.get("selected_x")
        if isinstance(x_payload, dict):
            selected_x.append(x_payload)
            candidate_index = x_payload.get("candidate_index")
        else:
            candidate_index = None
        u_payload = result.get("selected_u")
        if isinstance(u_payload, list):
            for row in u_payload:
                if not isinstance(row, dict):
                    continue
                selected_u.append(
                    {
                        "candidate_index": candidate_index,
                        "seed_task": item.seed.name,
                        "projection": item.projection,
                        **row,
                    }
                )

    return {
        "backend": first.get("backend"),
        "optimization_formulation": first.get("optimization_formulation"),
        "optimization_formulation_label": first.get("optimization_formulation_label"),
        "status": first.get("status"),
        "objective_value": first.get("objective_value"),
        "target_count": first.get("target_count"),
        "candidate_count": first.get("candidate_count"),
        "skill_nodes": first.get("skill_nodes"),
        "min_skill_coverage": first.get("min_skill_coverage"),
        "coverage_slack": first.get("coverage_slack"),
        "selected_x": selected_x,
        "selected_u": selected_u,
    }


def failure_record(item: EnvsForgePlanItem, error: str) -> dict[str, Any]:
    return {
        "seed_task": item.seed.name,
        "task_name": "",
        "task_path": None,
        "strategy": "envs_forge",
        "method_version": ENVS_FORGE_ALPHA_METHOD_VERSION,
        "evol_direction": item.evolution_direction,
        "action_variant": item.action_variant,
        "sample_index": item.sample_index,
        "instruction": "",
        "raw_response": "",
        "artifact": None,
        "parse_error": error,
        "schema_errors": ["generation failed"],
        "verification": {"matched": False, "status": "generation_failed", "errors": [error]},
        "envs_forge": plan_item_to_json(item),
        "completed_at": utc_now(),
    }


def print_plan(plan: list[EnvsForgePlanItem], method_dir: Path) -> None:
    print(json.dumps({"output_dir": str(method_dir), "planned": [plan_item_to_json(item) for item in plan]}, indent=2), flush=True)


def parse_count_constraints(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Count constraint must use name=count format: {value}")
        name, count_text = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Count constraint name cannot be empty: {value}")
        count = int(count_text)
        if count < 0:
            raise ValueError(f"Count constraint must be non-negative: {value}")
        result[name] = count
    return result


def single_value(values) -> Any:
    unique = sorted({value for value in values if value})
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return unique


def task_directory_name(seed_name: str, artifact_name: str, sample_index: int, repair_index: int | None) -> str:
    parts = [seed_name, artifact_name, "envs_forge", str(sample_index)]
    if repair_index is not None:
        parts.append(f"repair{repair_index}")
    return "-".join(part for part in parts if part)[:180]


def sanitize_command(command: list[str]) -> list[str]:
    return ["<redacted>" if "api_key" in item.lower() else item for item in command]


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
