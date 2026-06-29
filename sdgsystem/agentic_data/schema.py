from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

REQUIRED_FIELDS = ["question", "rationale", "final_answer", "metadata"]
REQUIRED_METADATA = ["license", "source", "domain", "required_dependencies", "name", "contributor", "date_created"]
EXECUTE_PYTHON_TOOL = [
    {
        "name": "execute_python",
        "description": "Execute Python code and return stdout, stderr, and return code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. The code should print the final answer on its last line.",
                }
            },
            "required": ["code"],
        },
    }
]


def parse_json_object(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None, str(exc)
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as inner:
            return None, str(inner)
    if not isinstance(parsed, dict):
        return None, "top-level JSON value is not an object"
    return parsed, None


def validate_agentic_sample(item: dict[str, Any] | None) -> list[str]:
    if item is None:
        return ["missing generated JSON object"]
    errors = [f"missing top-level field: {field}" for field in REQUIRED_FIELDS if field not in item]
    for field in ["question", "rationale", "final_answer"]:
        if field in item and not isinstance(item[field], str):
            errors.append(f"{field} must be a string")
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
    else:
        for field in REQUIRED_METADATA:
            if field not in metadata:
                errors.append(f"missing metadata field: {field}")
        if not isinstance(metadata.get("required_dependencies"), list):
            errors.append("metadata.required_dependencies must be a list")
    return errors


def fill_default_metadata(item: dict[str, Any], *, domain: str, strategy: str, model: str) -> dict[str, Any]:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        item["metadata"] = metadata
    metadata.setdefault("license", "MIT")
    metadata.setdefault("source", "synthetic")
    metadata.setdefault("domain", domain)
    metadata.setdefault("required_dependencies", [])
    metadata.setdefault("name", f"syn_agentic_{strategy}_{domain}")
    metadata.setdefault("contributor", "DataArc-SynData-Toolkit")
    metadata.setdefault("date_created", date.today().isoformat())
    metadata.setdefault("synthetic_strategy", strategy)
    metadata.setdefault("synthetic_from_domain", domain)
    metadata.setdefault("augmentation_model", model)
    metadata.setdefault("verifier", "python_rationale_stdout")
    return item


def agentic_to_dataarc(item: dict[str, Any]) -> dict[str, Any]:
    final_answer = str(item["final_answer"])
    function_call = {
        "name": "execute_python",
        "arguments": {"code": str(item["rationale"])},
    }
    observation = {
        "stdout": f"{final_answer}\n",
        "stderr": "",
        "returncode": 0,
        "final_line": final_answer,
    }
    return {
        "conversations": [
            {
                "from": "human",
                "value": str(item["question"]),
            },
            {
                "from": "gpt",
                "value": "I will solve this by executing a concise Python rationale.",
            },
            {
                "from": "function_call",
                "value": json.dumps(function_call, ensure_ascii=False),
            },
            {
                "from": "observation",
                "value": json.dumps(observation, ensure_ascii=False),
            },
            {
                "from": "gpt",
                "value": f"The final answer is: {final_answer}",
            },
        ],
        "tools": json.dumps(EXECUTE_PYTHON_TOOL, ensure_ascii=False),
    }
