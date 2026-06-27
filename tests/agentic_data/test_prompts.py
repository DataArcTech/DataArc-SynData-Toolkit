import unittest

from sdgsystem.agentic_data.prompts import build_agentic_completion_prompt, build_repair_prompt


class SynAgenticPromptTests(unittest.TestCase):
    def test_completion_prompt_adds_generation_pre_constraints(self):
        prompt = build_agentic_completion_prompt(
            strategy="self_instruct",
            domain="finance",
            seed_item={"question": "seed", "rationale": "print(1)", "final_answer": "1", "metadata": {}},
            candidate_sample={"input": "candidate", "output": "rough answer"},
            model="deepseek-v4-pro",
        )
        self.assertIn("question, rationale, final_answer, metadata", prompt)
        self.assertIn("rationale must be executable Python", prompt)
        self.assertIn("final_answer must exactly match the final line printed", prompt)
        self.assertIn("Keep rationale concise", prompt)
        self.assertIn('metadata.synthetic_strategy must be "self_instruct"', prompt)

    def test_repair_prompt_requires_same_verifiable_contract(self):
        prompt = build_repair_prompt({"schema_errors": ["bad"]})
        self.assertIn("failed schema validation or execution", prompt)
        self.assertIn("executable Python", prompt)

    def test_repair_prompt_compacts_large_failed_records(self):
        prompt = build_repair_prompt(
            {
                "domain": "programming",
                "strategy": "evol_instruct",
                "raw_response": "x" * 5000,
                "generated_item": {
                    "question": "q" * 5000,
                    "rationale": "r" * 5000,
                    "final_answer": "True",
                    "metadata": {},
                },
                "verification": {"stderr": "SyntaxError: bad string"},
            }
        )
        self.assertLess(len(prompt), 6000)
        self.assertNotIn("x" * 1000, prompt)
        self.assertIn("SyntaxError", prompt)


if __name__ == "__main__":
    unittest.main()
