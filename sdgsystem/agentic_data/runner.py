from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import SynAgenticDataConfig
from .distillation_bridge import generate_candidate_samples
from .prompts import build_agentic_completion_prompt, build_repair_prompt
from .schema import agentic_to_dataarc, fill_default_metadata, parse_json_object, validate_agentic_sample
from .verifier import verify_rationale


def run_syn_agentic_data(config: SynAgenticDataConfig, model_client: Any) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timed_model_client = RequestTimeoutModelClient(model_client, timeout=config.model_request_timeout)
    started_at = utc_now()
    records: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for domain in config.domains:
        seed_item = config.load_seed_items(domain)[config.seed_index]
        for strategy in config.strategies:
            candidate_samples = generate_candidate_samples(
                strategy=strategy,
                domain=domain,
                seed_item=seed_item,
                config=config,
                model_client=timed_model_client,
            )
            candidate_samples = normalize_candidate_samples(candidate_samples, seed_item)
            for candidate_index, candidate in enumerate(candidate_samples):
                evol_direction = candidate_evol_direction(candidate)
                candidates.append(
                    {
                        "domain": domain,
                        "strategy": strategy,
                        "candidate_index": candidate_index,
                        "evol_direction": evol_direction,
                        "candidate": candidate,
                    }
                )

                record = complete_and_verify(
                    config=config,
                    model_client=timed_model_client,
                    domain=domain,
                    strategy=strategy,
                    seed_item=seed_item,
                    candidate_sample=candidate,
                    candidate_index=candidate_index,
                    evol_direction=evol_direction,
                    output_dir=output_dir,
                )
                record = maybe_repair(config=config, model_client=timed_model_client, record=record, output_dir=output_dir)
                records.append(record)
                if (record.get("verification") or {}).get("matched"):
                    accepted.append(record)
                write_outputs(output_dir=output_dir, candidates=candidates, records=records, accepted=accepted, started_at=started_at)

    summary = write_outputs(output_dir=output_dir, candidates=candidates, records=records, accepted=accepted, started_at=started_at)
    return {"records": records, "accepted": accepted, "summary": summary}


def complete_and_verify(
    *,
    config: SynAgenticDataConfig,
    model_client: Any,
    domain: str,
    strategy: str,
    seed_item: dict[str, Any],
    candidate_sample: dict[str, Any],
    candidate_index: int,
    evol_direction: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    model_name = str(config.model.get("model", ""))
    prompt = build_agentic_completion_prompt(
        strategy=strategy,
        domain=domain,
        seed_item=seed_item,
        candidate_sample=candidate_sample,
        model=model_name,
    )
    raw = model_client.generate(prompt, n=1, max_tokens=config.completion_max_tokens)
    item, parse_error = parse_json_object(raw)
    if item is not None:
        fill_default_metadata(item, domain=domain, strategy=strategy, model=model_name)
        apply_candidate_metadata(item, candidate_sample)
    schema_errors = validate_agentic_sample(item)
    verification = None
    if item and not schema_errors:
        verification = verify_and_canonicalize(
            item,
            output_dir / f"verify_{verification_suffix(strategy, domain, candidate_index, evol_direction)}.py",
            timeout=config.verifier_timeout,
        )
    return {
        "domain": domain,
        "strategy": strategy,
        "seed_index": config.seed_index,
        "candidate_index": candidate_index,
        "evol_direction": evol_direction,
        "candidate_sample": candidate_sample,
        "raw_response": raw,
        "generated_item": item,
        "parse_error": parse_error,
        "schema_errors": schema_errors,
        "verification": verification,
        "accepted_from": "initial",
        "completed_at": utc_now(),
    }


def maybe_repair(
    *,
    config: SynAgenticDataConfig,
    model_client: Any,
    record: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if (record.get("verification") or {}).get("matched"):
        return record
    repaired = record
    repairs: list[dict[str, Any]] = []
    for attempt in range(1, config.repair_attempts + 1):
        raw = model_client.generate(build_repair_prompt(repaired), n=1, max_tokens=config.completion_max_tokens)
        item, parse_error = parse_json_object(raw)
        if item is not None:
            fill_default_metadata(
                item,
                domain=record["domain"],
                strategy=record["strategy"],
                model=str(config.model.get("model", "")),
            )
            apply_candidate_metadata(item, record["candidate_sample"])
        schema_errors = validate_agentic_sample(item)
        verification = None
        if item and not schema_errors:
            verification = verify_and_canonicalize(
                item,
                output_dir
                / f"verify_repair_{attempt}_{verification_suffix(record['strategy'], record['domain'], record.get('candidate_index', 0), record.get('evol_direction'))}.py",
                timeout=config.verifier_timeout,
            )
        repair = {
            "attempt": attempt,
            "raw_response": raw,
            "generated_item": item,
            "parse_error": parse_error,
            "schema_errors": schema_errors,
            "verification": verification,
        }
        repairs.append(repair)
        if verification and verification.get("matched"):
            return {
                **record,
                "raw_response": raw,
                "generated_item": item,
                "parse_error": parse_error,
                "schema_errors": schema_errors,
                "verification": verification,
                "accepted_from": "repair",
                "repair_records": repairs,
                "completed_at": utc_now(),
            }
        repaired = {**repaired, "repair_records": repairs}
    return repaired


def verify_and_canonicalize(item: dict[str, Any], code_path: Path, *, timeout: int) -> dict[str, Any]:
    verification = verify_rationale(item, code_path, timeout=timeout)
    observed = str(verification.get("observed_final_line") or "").strip()
    if not verification.get("matched") and verification.get("returncode") == 0 and observed:
        item["final_answer"] = observed
        verification = verify_rationale(item, code_path, timeout=timeout)
        verification["canonicalized_final_answer"] = True
    return verification


def write_outputs(
    *,
    output_dir: Path,
    candidates: list[dict[str, Any]],
    records: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    accepted_items = [record["generated_item"] for record in accepted if record.get("generated_item")]
    dataarc_items = [agentic_to_dataarc(item) for item in accepted_items]
    summary = {
        "started_at": started_at,
        "completed_at": utc_now(),
        "num_candidates": len(candidates),
        "num_records": len(records),
        "num_accepted": len(accepted),
        "domains": sorted({record["domain"] for record in records}),
        "strategies": sorted({record["strategy"] for record in records}),
    }
    write_json(output_dir / "candidate_results.json", candidates)
    write_json(output_dir / "baseline_results.json", records)
    write_json(output_dir / "accepted_results.json", accepted)
    write_json(output_dir / "accepted_agentic_items.json", accepted_items)
    write_jsonl(output_dir / "dataarc_train.jsonl", dataarc_items)
    write_json(output_dir / "summary.json", summary)
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, default=str) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RequestTimeoutModelClient:
    def __init__(self, model_client: Any, *, timeout: int | None) -> None:
        self._model_client = model_client
        self._timeout = timeout

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if self._timeout is not None and "timeout" not in kwargs:
            kwargs["timeout"] = self._timeout
        return self._model_client.generate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model_client, name)


def candidate_evol_direction(candidate: dict[str, Any]) -> str | None:
    return candidate_metadata_value(candidate, "evol_direction")


def normalize_candidate_samples(candidate_samples: list[dict[str, Any]], seed_item: dict[str, Any]) -> list[dict[str, Any]]:
    valid_candidates = [
        candidate
        for candidate in candidate_samples
        if isinstance(candidate, dict) and "input" in candidate and "output" in candidate
    ]
    if valid_candidates:
        return valid_candidates
    return [{"input": str(seed_item["question"]), "output": str(seed_item.get("final_answer", ""))}]


def apply_candidate_metadata(item: dict[str, Any], candidate_sample: dict[str, Any]) -> None:
    for key in ["evol_direction", "candidate_source"]:
        value = candidate_metadata_value(candidate_sample, key)
        if value:
            item["metadata"][key] = value


def candidate_metadata_value(candidate: dict[str, Any], key: str) -> str | None:
    metadata = candidate.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    return str(value) if value else None


def verification_suffix(strategy: str, domain: str, candidate_index: int, evol_direction: str | None) -> str:
    parts = [strategy, domain]
    if evol_direction:
        parts.append(evol_direction)
    elif candidate_index:
        parts.append(str(candidate_index))
    return "_".join(parts)
