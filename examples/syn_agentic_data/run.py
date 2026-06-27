from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdgsystem.agentic_data.config import SynAgenticDataConfig
from sdgsystem.agentic_data.runner import run_syn_agentic_data
from sdgsystem.configs.config import ModelConfig
from sdgsystem.models import ModelClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Syn Agentic Data synthesis as a sidecar workflow.")
    parser.add_argument("config", help="Path to syn_agentic_data YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SynAgenticDataConfig.from_yaml(args.config)
    model_client = ModelClient(ModelConfig.from_dict(config.model))
    result = run_syn_agentic_data(config, model_client)
    print(f"Syn Agentic Data completed: {result['summary']['num_accepted']}/{result['summary']['num_records']} accepted")
    print(f"Output directory: {config.output_dir}")


if __name__ == "__main__":
    main()
