from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdgsystem.agentic_data.terminal_bench import run_harbor_verification


DEFAULT_HARBOR_BIN = "/Volumes/PSSD/code/My-Agent/Agent-Benchmark/terminal-bench-science/.venv/bin/harbor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Harbor smoke verification for one generated terminal-bench task.")
    parser.add_argument("records", help="Path to terminal_bench_records.json.")
    parser.add_argument("--index", type=int, default=0, help="Record index to run.")
    parser.add_argument("--harbor-bin", default=DEFAULT_HARBOR_BIN, help="Path to Harbor executable.")
    parser.add_argument("--jobs-dir", default=None, help="Directory for Harbor job outputs.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional subprocess timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_path = Path(args.records)
    records = json.loads(records_path.read_text(encoding="utf-8"))
    record = records[args.index]
    harbor = record.get("harbor") or {}
    output_dir = records_path.parent
    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else output_dir / "harbor_jobs"
    result = run_harbor_verification(
        task_path=Path(record["task_path"]),
        jobs_dir=jobs_dir,
        harbor_bin=args.harbor_bin,
        agent=str(harbor.get("agent", "codex")),
        model=str(harbor.get("model", "gpt-5.5")),
        environment=str(harbor.get("environment", "docker")),
        force_build=bool(harbor.get("force_build", True)),
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
