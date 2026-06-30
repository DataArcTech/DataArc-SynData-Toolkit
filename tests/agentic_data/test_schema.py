import unittest
import json

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
        self.assertNotIn("conversations", converted)
        self.assertEqual(converted["messages"][0]["role"], "system")
        self.assertEqual(converted["messages"][1]["role"], "user")
        self.assertEqual(converted["messages"][1]["content"][0]["value"], "Calculate 2 + 3.")
        self.assertEqual(converted["messages"][2]["role"], "assistant")
        self.assertEqual(converted["messages"][2]["content"][0]["type"], "reasoning")
        self.assertEqual(converted["messages"][2]["content"][1]["type"], "function_call")
        function_call = json.loads(converted["messages"][2]["content"][1]["value"])
        observation = json.loads(converted["messages"][3]["content"][0]["value"])
        self.assertEqual(function_call["name"], "execute_python")
        self.assertIn('"stdout": "5\\n"', json.dumps(observation))
        self.assertEqual(converted["messages"][4]["content"][0]["type"], "reasoning")
        self.assertIn("execute_python", converted["tools"])
        self.assertEqual(json.loads(converted["tools"])[0]["type"], "function")

    def test_convert_to_dataarc_uses_custom_terminal_tool_metadata(self):
        item = {
            "question": "Optimize the local terminal-bench SQLite query task with a terminal agent.",
            "rationale": "Run a Codex terminal agent through Harbor and verify the produced artifact.",
            "final_answer": "passed",
            "metadata": {
                "tool_spec": [
                    {
                        "name": "run_codex_terminal_task",
                        "description": "Run Codex as a terminal agent on an environment-backed task.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_path": {"type": "string"},
                                "agent": {"type": "string"},
                                "model": {"type": "string"},
                            },
                            "required": ["task_path", "agent", "model"],
                        },
                    }
                ],
                "tool_call": {
                    "name": "run_codex_terminal_task",
                    "arguments": {
                        "task_path": "${TERMINAL_BENCH_QUERY_OPTIMIZE_PATH}",
                        "agent": "codex",
                        "model": "gpt-5.5",
                    },
                },
                "tool_observation": {
                    "status": "passed",
                    "reward": 1,
                    "summary": "solution passed correctness, format, runtime, and integrity checks",
                },
                "assistant_plan": "I will run Codex with GPT-5.5 in the terminal-bench environment.",
                "assistant_final": "The verifier passed and the solution artifact was produced.",
            },
        }

        converted = agentic_to_dataarc(item)

        function_call = json.loads(converted["messages"][2]["content"][1]["value"])
        observation = json.loads(converted["messages"][3]["content"][0]["value"])
        tools = json.loads(converted["tools"])
        self.assertNotIn("conversations", converted)
        self.assertEqual(converted["messages"][2]["content"][0]["type"], "reasoning")
        self.assertEqual(converted["messages"][2]["content"][1]["type"], "function_call")
        self.assertEqual(converted["messages"][3]["role"], "tool")
        self.assertEqual(converted["messages"][3]["content"][0]["type"], "observation")
        self.assertEqual(function_call["name"], "run_codex_terminal_task")
        self.assertEqual(function_call["arguments"]["model"], "gpt-5.5")
        self.assertEqual(observation["status"], "passed")
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "run_codex_terminal_task")
        self.assertNotIn("execute_python", converted["tools"])


if __name__ == "__main__":
    unittest.main()
