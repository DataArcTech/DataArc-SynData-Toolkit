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

from sdgsystem.agentic_data.envs_forge import load_rewards_from_harbor_result  # noqa: E402
from sdgsystem.agentic_data.envs_forge_v1 import (  # noqa: E402
    METHOD_VERSION,
    ComponentEditPlan,
    build_component_edit_prompt,
    build_component_repair_prompt,
    component_coverage,
    expand_subset_artifact,
    load_component_sensitivities,
    plan_envs_forge_v1_batch,
    validate_subset_artifact,
)
from sdgsystem.agentic_data.terminal_bench import (  # noqa: E402
    TerminalBenchSeedTask,
    load_seed_task,
    materialize_artifact,
    parse_json_object,
    validate_materialized_task,
    write_terminal_bench_outputs,
)


DEFAULT_TBLITE_ROOT = "/Volumes/PSSD/code/My-Agent/Agent-Data/OpenThoughts-TBLite"
DEFAULT_HARBOR_BIN = "/Volumes/PSSD/code/My-Agent/Agent-Benchmark/terminal-bench-science/.venv/bin/harbor"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/envs_forge_v1_tblite"
STRATEGY = "envs_forge_v1"


class OpenAICompatibleModelClient:
    def __init__(self, *, model: str, api_key: str, base_url: str, retries: int, retry_delay: float) -> None:
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
    parser = argparse.ArgumentParser(description="Run Envs-FORGE v1 component-edit synthesis on TBLite seeds.")
    parser.add_argument("--source-root", default=DEFAULT_TBLITE_ROOT)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--performance-result", action="append", default=[])
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--limit-seeds", type=int, default=None)
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--synthesis-model", default="deepseek-v4-flash")
    parser.add_argument("--sensitivity-file", default=None, help="JSON of component -> nats per unit strength")
    parser.add_argument("--e-max", type=float, default=1.0)
    parser.add_argument("--band", type=float, default=0.4, help="frontier band in |log-odds|")
    parser.add_argument("--backend", choices=["pyscipopt", "exact_enumeration"], default="pyscipopt")
    parser.add_argument("--include-unreachable", action="store_true")
    parser.add_argument("--downstream-model", default="openai/qwen3-8b")
    parser.add_argument("--completion-max-tokens", type=int, default=40000, help="reasoning models need headroom; 20000 is consumed by reasoning before any JSON is emitted")
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
    method_dir = output_root / METHOD_VERSION
    tasks_dir = method_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    rewards = load_performance_rewards(args.performance_result)
    seeds = discover_seed_tasks(source_root, task_names=args.task_names, limit=args.limit_seeds)
    sensitivity = load_component_sensitivities(args.sensitivity_file) if args.sensitivity_file else None

    plan = plan_envs_forge_v1_batch(
        seeds,
        rewards_by_seed=rewards,
        target_count=10 ** 9 if args.include_unreachable else args.target_count,
        sensitivity=sensitivity,
        e_max=args.e_max,
        band=args.band,
        prefer_solver=args.backend == "pyscipopt",
    )
    plan = plan[: args.target_count]
    seeds_by_name = {seed.name: seed for seed in seeds}
    write_json(method_dir / "envs_forge_v1_plan.json", [item.as_json() for item in plan])
    write_json(method_dir / "envs_forge_v1_solver_result.json", batch_solver_result(plan))
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
    done = {
        (record.get("seed_task"), int(record.get("sample_index") or 0))
        for record in records
        if record.get("strategy") == STRATEGY and is_matched(record, args.validation_mode)
    }

    for index, item in enumerate(plan):
        if len(accepted) >= args.target_count:
            break
        key = (item.seed_name, index)
        if key in done:
            print(f"[skip] accepted seed={item.seed_name} sample={index}", flush=True)
            continue
        record = synthesize_with_retries(
            item=item,
            seed=seeds_by_name[item.seed_name],
            sample_index=index,
            model=args.synthesis_model,
            client=client,
            tasks_dir=tasks_dir,
            e_max=args.e_max,
            band=args.band,
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
            f"[record] seed={item.seed_name} touched={list(item.touched)} "
            f"projection={item.projection} matched={is_matched(record, args.validation_mode)} "
            f"accepted={len(accepted)}/{args.target_count}",
            flush=True,
        )

    summary = write_terminal_bench_outputs(method_dir, records, accepted, started_at)
    eval_root = build_accepted_eval_root(method_dir, accepted[: args.target_count])
    summary[METHOD_VERSION] = {
        "method_version": METHOD_VERSION,
        "target_count": args.target_count,
        "eval_root": str(eval_root),
        "e_max": args.e_max,
        "band": args.band,
        "backend": args.backend,
        "sensitivity": sensitivity or "default",
        "synthesis_model": args.synthesis_model,
        "downstream_model": args.downstream_model,
        "mean_edit_cost": round(sum(item.cost for item in plan) / len(plan), 4) if plan else None,
        "reachable_fraction": round(sum(1 for item in plan if item.reachable) / len(plan), 4) if plan else None,
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
    item: ComponentEditPlan,
    seed: TerminalBenchSeedTask,
    sample_index: int,
    model: str,
    client: OpenAICompatibleModelClient,
    tasks_dir: Path,
    e_max: float,
    band: float,
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
            seed=seed,
            sample_index=sample_index,
            model=model,
            client=client,
            tasks_dir=tasks_dir,
            e_max=e_max,
            band=band,
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
                seed=seed,
                previous_record=last_record,
                sample_index=sample_index,
                repair_index=repair_index,
                model=model,
                client=client,
                tasks_dir=tasks_dir,
                e_max=e_max,
                band=band,
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
            last_record = record
            if is_matched(record, validation_mode):
                return record
            verification = record.get("verification") if isinstance(record.get("verification"), dict) else {}
            print(
                f"[repair-retry] seed={item.seed_name} repair={repair_index}/{repair_attempts} "
                f"status={verification.get('status')}",
                flush=True,
            )
    return last_record or failure_record(item, sample_index, "generation failed")


def synthesize_once(
    *,
    item: ComponentEditPlan,
    seed: TerminalBenchSeedTask,
    sample_index: int,
    model: str,
    client: OpenAICompatibleModelClient,
    tasks_dir: Path,
    e_max: float,
    band: float,
    completion_max_tokens: int,
    request_timeout: int,
    validation_mode: str,
    harbor_bin: str,
    harbor_env: str,
    gold_validation_timeout: int | None,
    gold_validation_retries: int,
    gold_validation_retry_delay: float,
) -> dict[str, Any]:
    prompt = build_component_edit_prompt(item, seed, model=model, e_max=e_max, band=band)
    raw = client.generate(prompt, max_tokens=completion_max_tokens, timeout=request_timeout)
    return materialize_and_validate(
        item=item,
        seed=seed,
        raw=raw,
        sample_index=sample_index,
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
    item: ComponentEditPlan,
    seed: TerminalBenchSeedTask,
    previous_record: dict[str, Any],
    sample_index: int,
    repair_index: int,
    model: str,
    client: OpenAICompatibleModelClient,
    tasks_dir: Path,
    e_max: float,
    band: float,
    completion_max_tokens: int,
    request_timeout: int,
    validation_mode: str,
    harbor_bin: str,
    harbor_env: str,
    gold_validation_timeout: int | None,
    gold_validation_retries: int,
    gold_validation_retry_delay: float,
) -> dict[str, Any]:
    previous = previous_record.get("generated_files")
    if not isinstance(previous, dict) or not previous:
        return previous_record
    prompt = build_component_repair_prompt(
        item,
        seed,
        previous_files=previous,
        failure_summary=summarize_failure(previous_record),
        model=model,
        repair_index=repair_index,
        e_max=e_max,
    )
    raw = client.generate(prompt, max_tokens=completion_max_tokens, timeout=request_timeout)
    return materialize_and_validate(
        item=item,
        seed=seed,
        raw=raw,
        sample_index=sample_index,
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


def materialize_and_validate(
    *,
    item: ComponentEditPlan,
    seed: TerminalBenchSeedTask,
    raw: str,
    sample_index: int,
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
    schema_errors = validate_subset_artifact(artifact, item.touched)
    task_dir: Path | None = None
    verification: dict[str, Any] | None = None
    generated_files: dict[str, str] = {}
    if artifact and not schema_errors:
        raw_files = artifact.get("files")
        if isinstance(raw_files, dict):
            generated_files = {str(path): str(content) for path, content in raw_files.items()}
        elif isinstance(raw_files, list):
            generated_files = {
                str(entry.get("path")): str(entry.get("content", ""))
                for entry in raw_files
                if isinstance(entry, dict) and entry.get("path")
            }
        full = expand_subset_artifact(seed, artifact, touched=item.touched)
        metadata = full.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("source_seed", seed.name)
            metadata.setdefault("synthetic_strategy", STRATEGY)
            metadata.setdefault("method_version", METHOD_VERSION)
            metadata.setdefault("envs_forge_v1_plan", item.as_json())
            metadata.setdefault("touched_components", list(item.touched))
            metadata.setdefault("edit_strengths", {k: round(v, 4) for k, v in sorted(item.strengths.items())})
            metadata.setdefault("edit_cost", round(item.cost, 4))
            metadata.setdefault("z0_log_odds", round(item.z0, 4))
            metadata.setdefault("z_predicted_log_odds", round(item.z_predicted, 4))
            metadata.setdefault("reachable_in_one_round", item.reachable)
            metadata.setdefault("solver_backend", item.backend)
            metadata.setdefault("augmentation_model", model)
            metadata.setdefault("date_created", datetime.now(timezone.utc).date().isoformat())
            if repair_index is not None:
                metadata.setdefault("repair_attempt", repair_index)
        task_dir = materialize_artifact(
            seed,
            full,
            tasks_dir,
            directory_name=task_directory_name(seed.name, str(artifact.get("task_name") or ""), sample_index, repair_index),
        )
        verification = validate_materialized_task(
            task_dir=task_dir,
            artifact=full,
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
        "seed_task": seed.name,
        "task_name": str((artifact or {}).get("task_name") or ""),
        "task_path": str(task_dir) if task_dir else None,
        "strategy": STRATEGY,
        "method_version": METHOD_VERSION,
        "sample_index": sample_index,
        "touched_components": list(item.touched),
        "projection": item.projection,
        "evolution_direction": item.evolution_direction,
        "component_coverage": component_coverage(artifact),
        "generated_files": generated_files,
        "raw_response": raw,
        "artifact": artifact,
        "parse_error": parse_error,
        "schema_errors": schema_errors,
        "verification": verification,
        "envs_forge_v1": item.as_json(),
        "completed_at": utc_now(),
    }


def summarize_failure(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("parse_error",):
        if record.get(key):
            parts.append(f"{key}: {shorten(str(record[key]), 1200)}")
    if record.get("schema_errors"):
        parts.append("schema_errors: " + json.dumps(record["schema_errors"], ensure_ascii=False))
    verification = record.get("verification") or {}
    for key in ("status", "errors"):
        if verification.get(key):
            parts.append(f"verification.{key}: {json.dumps(verification[key], ensure_ascii=False, default=str)[:1500]}")
    gold = verification.get("gold_solution") or {}
    if gold.get("status"):
        parts.append(f"gold_solution.status: {gold['status']}")
    log_tail = gold.get("log_tail")
    if log_tail:
        parts.append("gold_solution log tail:\n" + shorten(str(log_tail), 2000))
    return "\n".join(parts) or "no recorded failure detail"


def build_accepted_eval_root(method_dir: Path, accepted: list[dict[str, Any]]) -> Path:
    eval_root = method_dir / "eval/envs_forge_v1_verified_task_root"
    eval_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, record in enumerate(accepted, start=1):
        task_path = Path(str(record["task_path"]))
        link_path = eval_root / f"{index:03d}-{task_path.name}"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(os.path.relpath(task_path, start=eval_root), target_is_directory=True)
        manifest.append({"index": index, "eval_task_dir": str(link_path), "source_task_path": str(task_path)})
    write_json(eval_root / "envs_forge_v1_manifest.json", {"num_tasks": len(manifest), "tasks": manifest})
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
    jobs_dir = output_root / "eval/envs_forge_v1_downstream/jobs"
    logs_dir = output_root / "eval/envs_forge_v1_downstream/logs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    command = [
        harbor_bin, "run",
        "--path", str(eval_root),
        "--jobs-dir", str(jobs_dir),
        "--agent", "terminus-2",
        "--model", model,
        "--agent-kwarg", f"api_base={base_url}",
        "--agent-kwarg", "collect_rollout_details=True",
        "--agent-kwarg", "store_all_messages=True",
        "--agent-include-logs", "**/*",
        "--env", harbor_env,
        "--force-build",
        "--n-concurrent", "1",
        "--n-tasks", str(n_tasks),
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


def batch_solver_result(plan: list[ComponentEditPlan]) -> dict[str, Any]:
    if not plan:
        return {"method_version": METHOD_VERSION, "plans": []}
    first = plan[0].solver_result or {}
    return {
        "method_version": METHOD_VERSION,
        "backend": first.get("backend"),
        "optimization_formulation": "mixed_integer_programming",
        "plans": [item.as_json() for item in plan],
        "mean_cost": round(sum(item.cost for item in plan) / len(plan), 4),
        "mean_edit_strength": round(sum(item.edit_strength for item in plan) / len(plan), 4),
        "touched_component_counts": _component_counts(plan),
        "unreachable": [item.seed_name for item in plan if not item.reachable],
    }


def _component_counts(plan: list[ComponentEditPlan]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in plan:
        for component in item.touched:
            counts[component] = counts.get(component, 0) + 1
    return dict(sorted(counts.items()))


def print_plan(plan: list[ComponentEditPlan], method_dir: Path) -> None:
    print(
        json.dumps(
            {
                "method_version": METHOD_VERSION,
                "output_dir": str(method_dir),
                "backend": plan[0].backend if plan else None,
                "planned": [item.as_json() for item in plan],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def failure_record(item: ComponentEditPlan, sample_index: int, error: str) -> dict[str, Any]:
    return {
        "seed_task": item.seed_name,
        "task_name": "",
        "task_path": None,
        "strategy": STRATEGY,
        "method_version": METHOD_VERSION,
        "sample_index": sample_index,
        "touched_components": list(item.touched),
        "projection": item.projection,
        "evolution_direction": item.evolution_direction,
        "component_coverage": {},
        "generated_files": {},
        "raw_response": "",
        "artifact": None,
        "parse_error": error,
        "schema_errors": ["generation failed"],
        "verification": {"matched": False, "status": "generation_failed", "errors": [error]},
        "envs_forge_v1": item.as_json(),
        "completed_at": utc_now(),
    }


def task_directory_name(seed_name: str, artifact_name: str, sample_index: int, repair_index: int | None) -> str:
    parts = [seed_name, artifact_name, METHOD_VERSION, str(sample_index)]
    if repair_index is not None:
        parts.append(f"repair{repair_index}")
    return "-".join(part for part in parts if part)[:180]


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


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sanitize_command(command: list[str]) -> list[str]:
    return ["<redacted>" if "api_key" in item.lower() else item for item in command]


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
