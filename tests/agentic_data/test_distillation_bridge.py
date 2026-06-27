import unittest
from unittest.mock import patch

from sdgsystem.agentic_data.config import SynAgenticDataConfig
from sdgsystem.agentic_data.distillation_bridge import build_text_distill_config, generate_candidate_samples, get_distillation_class


class FakeEvolModelClient:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, n=1, **kwargs):
        self.prompts.append(prompt)
        if "IN-DEPTH EVOLUTION" in prompt:
            return '[{"input": "depth input", "output": "depth output"}]'
        if "IN-BREADTH EVOLUTION" in prompt:
            return '[{"input": "breadth input", "output": "breadth output"}]'
        return "[]"


class FakePartiallyInvalidEvolModelClient:
    def generate(self, prompt, n=1, **kwargs):
        if "IN-DEPTH EVOLUTION" in prompt:
            return '[{"input": "depth input", "output": "depth output"}]'
        return "not json"


class FakeFlakyEvolModelClient:
    def __init__(self):
        self.depth_calls = 0

    def generate(self, prompt, n=1, **kwargs):
        if "IN-DEPTH EVOLUTION" in prompt:
            self.depth_calls += 1
            if self.depth_calls == 1:
                return "not json"
            return '[{"input": "retried depth input", "output": "retried depth output"}]'
        return '[{"input": "breadth input", "output": "breadth output"}]'


class CrashingGenerator:
    def __init__(self, model, config):
        pass

    def generate(self, demo_examples=None):
        raise TypeError("bad generated sample")


class SynAgenticDistillationBridgeTests(unittest.TestCase):
    def test_strategy_maps_to_existing_distillation_classes(self):
        self.assertEqual(get_distillation_class("few_shot").__name__, "SDGDistillation")
        self.assertEqual(get_distillation_class("self_instruct").__name__, "SelfInstructDistillation")
        self.assertEqual(get_distillation_class("evol_instruct").__name__, "EvolInstructDistillation")

    def test_build_text_distill_config_is_three_subset_scoped(self):
        config = SynAgenticDataConfig(
            model={"provider": "openai", "model": "fake"},
            seed_paths={"finance": "finance.json", "programming": "programming.json", "mathematical_programming": "mp.json"},
        )
        distill_config = build_text_distill_config("finance", "self_instruct", config)
        self.assertEqual(distill_config.domain, "finance")
        self.assertEqual(distill_config.num_samples, 1)
        self.assertIn("finance", distill_config.task_instruction.lower())

    def test_evol_generation_covers_depth_and_breadth(self):
        config = SynAgenticDataConfig(
            model={"provider": "openai", "model": "fake"},
            seed_paths={"finance": "finance.json", "programming": "programming.json", "mathematical_programming": "mp.json"},
        )
        fake_model = FakeEvolModelClient()

        samples = generate_candidate_samples(
            strategy="evol_instruct",
            domain="finance",
            seed_item={"question": "Q?", "rationale": "print(1)", "final_answer": "1", "metadata": {}},
            config=config,
            model_client=fake_model,
        )

        self.assertEqual([sample["metadata"]["evol_direction"] for sample in samples], ["in_depth", "in_breadth"])
        self.assertEqual([sample["input"] for sample in samples], ["depth input", "breadth input"])
        self.assertTrue(any("IN-DEPTH EVOLUTION" in prompt for prompt in fake_model.prompts))
        self.assertTrue(any("IN-BREADTH EVOLUTION" in prompt for prompt in fake_model.prompts))

    def test_evol_generation_falls_back_when_direction_is_unparseable(self):
        config = SynAgenticDataConfig(
            model={"provider": "openai", "model": "fake"},
            seed_paths={"finance": "finance.json", "programming": "programming.json", "mathematical_programming": "mp.json"},
        )

        with self.assertLogs("sdgsystem.agentic_data.distillation_bridge", level="WARNING") as logs:
            samples = generate_candidate_samples(
                strategy="evol_instruct",
                domain="finance",
                seed_item={"question": "Q?", "rationale": "print(1)", "final_answer": "1", "metadata": {}},
                config=config,
                model_client=FakePartiallyInvalidEvolModelClient(),
            )

        self.assertEqual(len(samples), 2)
        self.assertEqual([sample["metadata"]["evol_direction"] for sample in samples], ["in_depth", "in_breadth"])
        self.assertEqual(samples[0]["metadata"]["candidate_source"], "evol_instruct")
        self.assertEqual(samples[1]["metadata"]["candidate_source"], "seed_fallback")
        self.assertEqual(samples[1]["input"], "Q?")
        self.assertIn("in_breadth", logs.output[0])

    def test_evol_generation_retries_unparseable_direction_before_fallback(self):
        config = SynAgenticDataConfig(
            model={"provider": "openai", "model": "fake"},
            seed_paths={"finance": "finance.json", "programming": "programming.json", "mathematical_programming": "mp.json"},
            candidate_generation_attempts=2,
        )
        fake_model = FakeFlakyEvolModelClient()

        with self.assertLogs("sdgsystem.agentic_data.distillation_bridge", level="WARNING") as logs:
            samples = generate_candidate_samples(
                strategy="evol_instruct",
                domain="finance",
                seed_item={"question": "Q?", "rationale": "print(1)", "final_answer": "1", "metadata": {}},
                config=config,
                model_client=fake_model,
            )

        self.assertEqual(fake_model.depth_calls, 2)
        self.assertEqual(samples[0]["input"], "retried depth input")
        self.assertEqual(samples[0]["metadata"]["candidate_source"], "evol_instruct")
        self.assertIn("attempt 1", logs.output[0])

    def test_generation_errors_return_empty_candidates_for_runner_fallback(self):
        config = SynAgenticDataConfig(
            model={"provider": "openai", "model": "fake"},
            seed_paths={"finance": "finance.json", "programming": "programming.json", "mathematical_programming": "mp.json"},
        )

        with self.assertLogs("sdgsystem.agentic_data.distillation_bridge", level="WARNING") as logs, patch(
            "sdgsystem.agentic_data.distillation_bridge.get_distillation_class", return_value=CrashingGenerator
        ):
            samples = generate_candidate_samples(
                strategy="few_shot",
                domain="finance",
                seed_item={"question": "Q?", "rationale": "print(1)", "final_answer": "1", "metadata": {}},
                config=config,
                model_client=FakeEvolModelClient(),
            )

        self.assertEqual(samples, [])
        self.assertIn("candidate generation failed", logs.output[0])


if __name__ == "__main__":
    unittest.main()
