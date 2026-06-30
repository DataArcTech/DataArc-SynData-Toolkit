from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

REQUIRED_FIELDS = ["question", "rationale", "final_answer", "metadata"]
REQUIRED_METADATA = ["license", "source", "domain", "required_dependencies", "name", "contributor", "date_created"]
DEFAULT_SYSTEM_PROMPT = (
    "You are a methodical and expert assistant. Your primary goal is to solve user requests by leveraging a set of "
    "available tools. You must reason for the best course of action in a structured manner before responding."
)
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
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    if isinstance(metadata.get("tool_call"), dict):
        return _agentic_to_custom_tool_dataarc(item, metadata)

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
    return _tool_trace_record(
        question=str(item["question"]),
        reasoning="I will solve this by executing a concise Python rationale and checking the final printed line.",
        function_call=function_call,
        observation=observation,
        final_reasoning=f"The tool returned final_line={final_answer!r}, which matches the expected answer.",
        final_text=f"The final answer is: {final_answer}",
        tools=EXECUTE_PYTHON_TOOL,
    )


def _agentic_to_custom_tool_dataarc(item: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    tool_call = metadata["tool_call"]
    observation = metadata.get("tool_observation")
    if not isinstance(observation, dict):
        observation = {"status": str(item["final_answer"])}
    return _tool_trace_record(
        question=str(item["question"]),
        reasoning=str(metadata.get("assistant_plan") or "I will use the configured tool to solve and verify the task."),
        function_call=tool_call,
        observation=observation,
        final_reasoning=str(
            metadata.get("assistant_final_reasoning")
            or "The tool observation contains the verifier result needed to answer the user."
        ),
        final_text=str(metadata.get("assistant_final") or f"The final answer is: {item['final_answer']}"),
        tools=_metadata_tool_spec(metadata),
    )


def _tool_trace_record(
    *,
    question: str,
    reasoning: str,
    function_call: dict[str, Any],
    observation: dict[str, Any],
    final_reasoning: str,
    final_text: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "value": DEFAULT_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "value": question}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "value": reasoning},
                    {"type": "function_call", "value": json.dumps(function_call, ensure_ascii=False)},
                ],
            },
            {
                "role": "tool",
                "content": [{"type": "observation", "value": json.dumps(observation, ensure_ascii=False)}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "value": final_reasoning},
                    {"type": "text", "value": final_text},
                ],
            },
        ],
        "tools": json.dumps(_as_openai_function_tools(tools), ensure_ascii=False),
    }


def _metadata_tool_spec(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    tool_spec = metadata.get("tool_spec")
    if isinstance(tool_spec, dict):
        return [tool_spec]
    if isinstance(tool_spec, list) and all(isinstance(tool, dict) for tool in tool_spec):
        return tool_spec
    tool_call = metadata.get("tool_call")
    if isinstance(tool_call, dict) and isinstance(tool_call.get("name"), str):
        return [
            {
                "name": tool_call["name"],
                "description": "Execute the configured external agentic tool.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ]
    return EXECUTE_PYTHON_TOOL


def _as_openai_function_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wrapped = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            wrapped.append(tool)
            continue
        wrapped.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
                },
            }
        )
    return wrapped
