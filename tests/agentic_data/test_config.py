import json
import tempfile
import unittest
from pathlib import Path

from sdgsystem.agentic_data.config import SynAgenticDataConfig, to_dataarc_demo


class SynAgenticConfigTests(unittest.TestCase):
    def test_loads_config_and_seed_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = {
                "question": "Calculate 1 + 1.",
                "rationale": "print(1 + 1)",
                "final_answer": "2",
                "metadata": {"domain": "finance", "required_dependencies": []},
            }
            (root / "finance.json").write_text(json.dumps([seed]), encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                """
output_dir: ./outputs/syn_agentic_data
model:
  provider: openai
  model: deepseek-v4-pro
  base_url: https://api.deepseek.com
domains: [finance, programming, mathematical_programming]
strategies: [few_shot, self_instruct, evol_instruct]
seed_paths:
  finance: finance.json
  programming: finance.json
  mathematical_programming: finance.json
seed_index: 0
num_samples_per_strategy: 1
repair_attempts: 1
verifier_timeout: 30
""",
                encoding="utf-8",
            )

            config = SynAgenticDataConfig.from_yaml(str(config_path))

            self.assertEqual(config.domains, ["finance", "programming", "mathematical_programming"])
            self.assertEqual(config.strategies, ["few_shot", "self_instruct", "evol_instruct"])
            self.assertEqual(config.model["model"], "deepseek-v4-pro")
            self.assertEqual(config.load_seed_items("finance")[0]["final_answer"], "2")

    def test_loads_environment_backed_domain_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = {
                "question": "Run a terminal-bench task.",
                "rationale": "Run an external terminal agent and verifier.",
                "final_answer": "passed",
                "metadata": {"domain": "terminal_bench_query_optimize", "required_dependencies": ["harbor"]},
            }
            (root / "terminal.json").write_text(json.dumps([seed]), encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                """
output_dir: ./outputs/syn_agentic_data
model:
  provider: openai
  model: deepseek-v4-pro
  base_url: https://api.deepseek.com
domains: [terminal_bench_query_optimize]
strategies: [self_instruct]
seed_paths:
  terminal_bench_query_optimize: terminal.json
seed_index: 0
num_samples_per_strategy: 1
repair_attempts: 1
verifier_timeout: 30
""",
                encoding="utf-8",
            )

            config = SynAgenticDataConfig.from_yaml(str(config_path))

            self.assertEqual(config.domains, ["terminal_bench_query_optimize"])
            self.assertEqual(config.load_seed_items("terminal_bench_query_optimize")[0]["final_answer"], "passed")

    def test_to_dataarc_demo_keeps_question_and_answer_context(self):
        demo = to_dataarc_demo(
            {
                "question": "Q?",
                "rationale": "print(3)",
                "final_answer": "3",
                "metadata": {},
            }
        )
        self.assertEqual(demo["input"], "Q?")
        self.assertIn("print(3)", demo["output"])
        self.assertIn("Final Answer: 3", demo["output"])


if __name__ == "__main__":
    unittest.main()
