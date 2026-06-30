import json
import unittest
from pathlib import Path

from sdgsystem.agentic_data.terminal_bench import load_seed_task, terminal_bench_record_to_dataarc


class SynAgenticTerminalBenchSeedTests(unittest.TestCase):
    def test_terminal_bench_seeds_export_harbor_tool_without_private_artifacts(self):
        seed_root = Path("examples/syn_agentic_data/terminal-bench-seed")
        seed_names = ["portfolio-optimization", "cancel-async-tasks", "constraints-scheduling"]

        for seed_name in seed_names:
            seed = load_seed_task(seed_root / seed_name)
            record = {
                "task_name": f"{seed_name}-synthetic",
                "task_path": f"/tmp/{seed_name}-synthetic",
                "strategy": "few_shot",
                "instruction": seed.instruction,
                "verification": {"matched": True, "status": "static_validated"},
                "harbor": {"agent": "codex", "model": "gpt-5.5", "environment": "docker"},
            }
            converted = terminal_bench_record_to_dataarc(record)
            blob = json.dumps(converted, ensure_ascii=False).lower()

            self.assertNotIn("terminal-bench-canary", json.dumps(seed.files, ensure_ascii=False).lower())
            self.assertIn("messages", converted)
            self.assertNotIn("conversations", converted)
            self.assertEqual(converted["messages"][2]["content"][0]["type"], "reasoning")
            self.assertEqual(converted["messages"][2]["content"][1]["type"], "function_call")
            self.assertEqual(converted["messages"][3]["role"], "tool")
            self.assertEqual(converted["messages"][3]["content"][0]["type"], "observation")
            self.assertIn("run_harbor_terminal_bench", blob)
            self.assertIn("gpt-5.5", blob)
            self.assertNotIn("solution/solve.sh", blob)
            self.assertNotIn("tests/test_outputs.py", blob)
            self.assertNotIn("terminal-bench-canary", blob)


if __name__ == "__main__":
    unittest.main()
