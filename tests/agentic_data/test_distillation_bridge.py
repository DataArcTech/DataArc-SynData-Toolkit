import unittest

from sdgsystem.agentic_data.config import SynAgenticDataConfig
from sdgsystem.agentic_data.distillation_bridge import build_text_distill_config, get_distillation_class


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


if __name__ == "__main__":
    unittest.main()
