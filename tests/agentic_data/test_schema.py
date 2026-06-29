import unittest

from sdgsystem.agentic_data.schema import (
    agentic_to_dataarc,
    fill_default_metadata,
    parse_json_object,
    validate_agentic_sample,
)


class SynAgenticSchemaTests(unittest.TestCase):
    def test_parse_json_object_handles_markdown_wrapper(self):
        item, error = parse_json_object(
            '```json\n{"question":"q","rationale":"print(1)","final_answer":"1","metadata":{}}\n```'
        )
        self.assertIsNone(error)
        self.assertEqual(item["question"], "q")

    def test_validate_requires_agentic_fields(self):
        errors = validate_agentic_sample({"question": "q"})
        self.assertIn("missing top-level field: rationale", errors)
        self.assertIn("missing top-level field: final_answer", errors)

    def test_fill_metadata_and_convert_to_dataarc(self):
        item = {"question": "Calculate 2 + 3.", "rationale": "print(2 + 3)", "final_answer": "5", "metadata": {}}
        fill_default_metadata(item, domain="finance", strategy="few_shot", model="deepseek-v4-pro")
        self.assertEqual(validate_agentic_sample(item), [])
        converted = agentic_to_dataarc(item)
        self.assertEqual(converted["conversations"][0]["from"], "human")
        self.assertEqual(converted["conversations"][0]["value"], "Calculate 2 + 3.")
        self.assertEqual(converted["conversations"][2]["from"], "function_call")
        self.assertIn("execute_python", converted["conversations"][2]["value"])
        self.assertEqual(converted["conversations"][3]["from"], "observation")
        self.assertIn('"stdout": "5\\n"', converted["conversations"][3]["value"])
        self.assertIn("execute_python", converted["tools"])


if __name__ == "__main__":
    unittest.main()
