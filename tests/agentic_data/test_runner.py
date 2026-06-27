import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sdgsystem.agentic_data.config import SynAgenticDataConfig
from sdgsystem.agentic_data.runner import run_syn_agentic_data


class FakeModelClient:
    def __init__(self):
        self.kwargs = []

    def generate(self, prompt, n=1, **kwargs):
        self.kwargs.append(kwargs)
        return json.dumps(
            {
                "question": "Compute 10 + 5.",
                "rationale": "print(10 + 5)",
                "final_answer": "15",
                "metadata": {},
            }
        )


class FakeRepairModelClient:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, n=1, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"question": "Compute 1 + 1.", "rationale": "print(2)", "metadata": {}})
        return json.dumps(
            {
                "question": "Compute 1 + 1.",
                "rationale": "print(2)",
                "final_answer": "2",
                "metadata": {},
            }
        )


class SynAgenticRunnerTests(unittest.TestCase):
    def test_runner_accepts_three_domains_by_three_strategies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = [{"question": "Compute 1 + 1.", "rationale": "print(1 + 1)", "final_answer": "2", "metadata": {}}]
            for domain in ["finance", "programming", "mathematical_programming"]:
                (root / f"{domain}.json").write_text(json.dumps(seed), encoding="utf-8")
            config = SynAgenticDataConfig(
                output_dir=str(root / "out"),
                model={"provider": "openai", "model": "fake"},
                seed_paths={
                    "finance": str(root / "finance.json"),
                    "programming": str(root / "programming.json"),
                    "mathematical_programming": str(root / "mathematical_programming.json"),
                },
            )

            with patch(
                "sdgsystem.agentic_data.runner.generate_candidate_samples",
                return_value=[{"input": "candidate", "output": "rough"}],
            ):
                fake_model = FakeModelClient()
                result = run_syn_agentic_data(config, fake_model)

            self.assertEqual(result["summary"]["num_records"], 9)
            self.assertEqual(result["summary"]["num_accepted"], 9)
            self.assertTrue(all(kwargs.get("max_tokens", 0) >= 8192 for kwargs in fake_model.kwargs))
            self.assertTrue((root / "out" / "accepted_agentic_items.json").exists())
            self.assertTrue((root / "out" / "dataarc_train.jsonl").exists())

    def test_runner_applies_request_timeout_to_all_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = [{"question": "Compute 1 + 1.", "rationale": "print(1 + 1)", "final_answer": "2", "metadata": {}}]
            (root / "finance.json").write_text(json.dumps(seed), encoding="utf-8")
            config = SynAgenticDataConfig(
                output_dir=str(root / "out"),
                model={"provider": "openai", "model": "fake"},
                domains=["finance"],
                strategies=["self_instruct"],
                seed_paths={
                    "finance": str(root / "finance.json"),
                    "programming": str(root / "finance.json"),
                    "mathematical_programming": str(root / "finance.json"),
                },
                model_request_timeout=12,
            )

            def generate_candidate(**kwargs):
                kwargs["model_client"].generate("candidate prompt", n=1)
                return [{"input": "candidate", "output": "rough"}]

            with patch("sdgsystem.agentic_data.runner.generate_candidate_samples", side_effect=generate_candidate):
                fake_model = FakeModelClient()
                run_syn_agentic_data(config, fake_model)

            self.assertGreaterEqual(len(fake_model.kwargs), 2)
            self.assertTrue(all(kwargs.get("timeout") == 12 for kwargs in fake_model.kwargs))

    def test_runner_processes_both_evol_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = [{"question": "Compute 1 + 1.", "rationale": "print(1 + 1)", "final_answer": "2", "metadata": {}}]
            (root / "finance.json").write_text(json.dumps(seed), encoding="utf-8")
            config = SynAgenticDataConfig(
                output_dir=str(root / "out"),
                model={"provider": "openai", "model": "fake"},
                domains=["finance"],
                strategies=["evol_instruct"],
                seed_paths={
                    "finance": str(root / "finance.json"),
                    "programming": str(root / "finance.json"),
                    "mathematical_programming": str(root / "finance.json"),
                },
            )
            candidates = [
                {
                    "input": "depth candidate",
                    "output": "rough",
                    "metadata": {"evol_direction": "in_depth", "candidate_source": "evol_instruct"},
                },
                {
                    "input": "breadth candidate",
                    "output": "rough",
                    "metadata": {"evol_direction": "in_breadth", "candidate_source": "evol_instruct"},
                },
            ]

            with patch("sdgsystem.agentic_data.runner.generate_candidate_samples", return_value=candidates):
                result = run_syn_agentic_data(config, FakeModelClient())

            self.assertEqual(result["summary"]["num_records"], 2)
            self.assertEqual(result["summary"]["num_accepted"], 2)
            self.assertEqual(
                [record["generated_item"]["metadata"]["evol_direction"] for record in result["records"]],
                ["in_depth", "in_breadth"],
            )
            self.assertEqual(
                [record["generated_item"]["metadata"]["candidate_source"] for record in result["records"]],
                ["evol_instruct", "evol_instruct"],
            )

    def test_runner_filters_invalid_candidates_before_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = [{"question": "Compute 1 + 1.", "rationale": "print(1 + 1)", "final_answer": "2", "metadata": {}}]
            (root / "finance.json").write_text(json.dumps(seed), encoding="utf-8")
            config = SynAgenticDataConfig(
                output_dir=str(root / "out"),
                model={"provider": "openai", "model": "fake"},
                domains=["finance"],
                strategies=["self_instruct"],
                seed_paths={
                    "finance": str(root / "finance.json"),
                    "programming": str(root / "finance.json"),
                    "mathematical_programming": str(root / "finance.json"),
                },
            )

            with patch("sdgsystem.agentic_data.runner.generate_candidate_samples", return_value=[None]):
                result = run_syn_agentic_data(config, FakeModelClient())

            self.assertEqual(result["summary"]["num_records"], 1)
            self.assertEqual(result["records"][0]["candidate_sample"]["input"], "Compute 1 + 1.")

    def test_repair_preserves_candidate_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = [{"question": "Compute 1 + 1.", "rationale": "print(1 + 1)", "final_answer": "2", "metadata": {}}]
            (root / "finance.json").write_text(json.dumps(seed), encoding="utf-8")
            config = SynAgenticDataConfig(
                output_dir=str(root / "out"),
                model={"provider": "openai", "model": "fake"},
                domains=["finance"],
                strategies=["evol_instruct"],
                seed_paths={
                    "finance": str(root / "finance.json"),
                    "programming": str(root / "finance.json"),
                    "mathematical_programming": str(root / "finance.json"),
                },
            )
            candidates = [
                {
                    "input": "depth candidate",
                    "output": "rough",
                    "metadata": {"evol_direction": "in_depth", "candidate_source": "seed_fallback"},
                }
            ]

            with patch("sdgsystem.agentic_data.runner.generate_candidate_samples", return_value=candidates):
                result = run_syn_agentic_data(config, FakeRepairModelClient())

            self.assertEqual(result["records"][0]["accepted_from"], "repair")
            self.assertEqual(result["records"][0]["generated_item"]["metadata"]["evol_direction"], "in_depth")
            self.assertEqual(result["records"][0]["generated_item"]["metadata"]["candidate_source"], "seed_fallback")


if __name__ == "__main__":
    unittest.main()
