from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr

DomainName = str
StrategyName = Literal["few_shot", "self_instruct", "evol_instruct"]
EvolDirection = Literal["in_depth", "in_breadth"]


class SynAgenticDataConfig(BaseModel):
    output_dir: str = "./outputs/syn_agentic_data"
    model: dict
    domains: list[DomainName] = Field(default_factory=lambda: ["finance", "programming", "mathematical_programming"])
    strategies: list[StrategyName] = Field(default_factory=lambda: ["few_shot", "self_instruct", "evol_instruct"])
    seed_paths: dict[DomainName, str]
    seed_index: int = 0
    num_samples_per_strategy: int = 1
    evol_directions: list[EvolDirection] = Field(default_factory=lambda: ["in_depth", "in_breadth"])
    candidate_generation_attempts: int = 2
    repair_attempts: int = 1
    verifier_timeout: int = 30
    model_request_timeout: int = 120
    completion_max_tokens: int = 8192
    temperature: float = 0.7

    _config_dir: Path = PrivateAttr(default=Path("."))

    @classmethod
    def from_yaml(cls, path: str) -> "SynAgenticDataConfig":
        config_path = Path(path).resolve()
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        instance = cls(**data)
        instance._config_dir = config_path.parent
        return instance

    def resolve_seed_path(self, domain: DomainName) -> Path:
        path = Path(self.seed_paths[domain])
        return path if path.is_absolute() else self._config_dir / path

    def load_seed_items(self, domain: DomainName) -> list[dict]:
        path = self.resolve_seed_path(domain)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        return data if isinstance(data, list) else [data]


def to_dataarc_demo(seed_item: dict) -> dict[str, str]:
    return {
        "input": str(seed_item["question"]),
        "output": f"Rationale:\n{seed_item['rationale']}\n\nFinal Answer: {seed_item['final_answer']}",
    }
