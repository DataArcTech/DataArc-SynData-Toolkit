import tempfile
import unittest
from pathlib import Path

from sdgsystem.agentic_data.verifier import answers_match, verify_agentic_item, verify_rationale


class SynAgenticVerifierTests(unittest.TestCase):
    def test_answers_match_numeric_and_literal_values(self):
        self.assertTrue(answers_match("1.0000000001", "1.0"))
        self.assertTrue(answers_match("[('x', 1, 1.0)]", "[('x', 1, 1.0)]"))
        self.assertFalse(answers_match("2", "3"))

    def test_verify_rationale_matches_last_stdout_line(self):
        item = {"question": "q", "rationale": "print('scratch')\nprint(5)", "final_answer": "5", "metadata": {}}
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_rationale(item, Path(tmp) / "verify.py")
        self.assertTrue(result["matched"])
        self.assertEqual(result["observed_final_line"], "5")

    def test_verify_rationale_reports_timeout(self):
        item = {"question": "q", "rationale": "while True:\n    pass", "final_answer": "never", "metadata": {}}
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_rationale(item, Path(tmp) / "verify.py", timeout=1)
        self.assertFalse(result["matched"])
        self.assertEqual(result["error"], "timeout")

    def test_verify_agentic_item_accepts_terminal_harness_summary_without_python_execution(self):
        item = {
            "question": "Run a terminal-bench task.",
            "rationale": "Run an external terminal agent and verifier.",
            "final_answer": "passed",
            "metadata": {
                "verifier": "terminal_bench_harness",
                "tool_observation": {
                    "status": "passed",
                    "reward": 1,
                    "summary": "solution passed correctness, format, runtime, and integrity checks",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            code_path = Path(tmp) / "verify.py"
            result = verify_agentic_item(item, code_path, timeout=1)

        self.assertTrue(result["matched"])
        self.assertEqual(result["verifier"], "terminal_bench_harness")
        self.assertFalse(code_path.exists())


if __name__ == "__main__":
    unittest.main()
