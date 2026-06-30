from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any
from pathlib import Path

import openai


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdgsystem.agentic_data.terminal_bench import TerminalBenchSynthesisConfig, run_terminal_bench_synthesis


class OpenAICompatibleModelClient:
    def __init__(self, model_config: dict[str, Any]) -> None:
        self.model = str(model_config.get("model", ""))
        self.max_retry_attempts = int(model_config.get("max_retry_attempts", 2))
        self.retry_delay = float(model_config.get("retry_delay", 1))
        api_key = model_config.get("api_key") or os.getenv("API_KEY")
        base_url = model_config.get("base_url") or os.getenv("BASE_URL")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> str:
        max_tokens = int(kwargs.pop("max_tokens", 16000))
        temperature = float(kwargs.pop("temperature", 0.7))
        timeout = kwargs.pop("timeout", None)
        last_error: Exception | None = None
        for _ in range(self.max_retry_attempts):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=n,
                    timeout=timeout,
                    **kwargs,
                )
                return completion.choices[0].message.content.strip()
            except Exception as exc:
                last_error = exc
                time.sleep(self.retry_delay)
        raise RuntimeError(f"Failed to get response from API after {self.max_retry_attempts} attempts") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run terminal-bench-style Syn Agentic Data task synthesis.")
    parser.add_argument("config", help="Path to terminal-bench synthesis YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TerminalBenchSynthesisConfig.from_yaml(args.config)
    model_client = OpenAICompatibleModelClient(config.model)
    result = run_terminal_bench_synthesis(config, model_client)
    print(
        "Terminal-bench Syn Agentic Data completed: "
        f"{result['summary']['num_accepted']}/{result['summary']['num_records']} accepted"
    )
    print(f"Output directory: {config.output_dir}")


if __name__ == "__main__":
    main()
