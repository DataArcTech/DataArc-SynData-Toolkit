"""
Filter out data suitable for the task and convert the format
"""

import re
import json
import logging
from typing import Dict

# from ..configs.constants import DEFAULT_WEB_TASK_INPUT_FORMAT, DEFAULT_WEB_TASK_OUTPUT_FORMAT
from ..models import ModelClient
from ..prompts import FIELD_FILTER_PROMPT, INSTRUCTION_JUDGE_PROMPT, SOLVABLE_JUDGE_PROMPT, FORMAT_CONVERSION_PROMPT

logger = logging.getLogger(__name__)


# Filter out data or texts
class DataFilter:
    def __init__(self, llm: ModelClient) -> None:
        self.llm = llm

    def field_filter(self,
        row: str,
        legal_keys
    ) -> Dict[str, str]:
        # Convert legal_keys to list for proper formatting in prompt
        legal_keys_list = list(legal_keys) if not isinstance(legal_keys, list) else legal_keys

        prompt = FIELD_FILTER_PROMPT.format(row=row, legal_keys=legal_keys_list)
        output: str = self.llm.generate(prompt)

        # Try to extract JSON from response (handle markdown code blocks)
        # First try to find ```json ... ``` blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.S)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
            try:
                result = json.loads(json_str)
                logger.info(f"Successfully parsed: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from code block: {e}")

        # Fallback: find raw JSON object
        match = re.search(r'\{.*\}', output, re.S)
        if match:
            json_str = match.group().strip()
            try:
                result = json.loads(json_str)
                logger.info(f"Successfully parsed: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse raw JSON: {e}")
                return {"input": None, "output": None}

        logger.warning(f"No JSON found in LLM output")
        return {"input": None, "output": None}

    def instruction_judge(self, 
        task_description: str, 
        instruction_sample: Dict[str, str]
    ) -> Dict[str, int]:
        prompt = INSTRUCTION_JUDGE_PROMPT.format(task_description=task_description, instruction_sample=instruction_sample)

        output: str = self.llm.generate(prompt)
        match = re.search(r'\{.*\}', output, re.S)
        if match:
            try:
                return json.loads(match.group().strip())
            except:
                return {"Relevance": 5, "Correctness": 5, "Helpfulness": 5, "Clarity": 5, "Difficulty": 5}
        return {"Relevance": 5, "Correctness": 5, "Helpfulness": 5, "Clarity": 5, "Difficulty": 5}

    def solvable_judge(self, 
        instruction_sample: Dict[str, str]
    ) -> bool:
        solve_prompt = f"Please think step by step and answer this question.\n{instruction_sample['input']}"
        solution: str = self.llm.generate(solve_prompt)

        judge_prompt = SOLVABLE_JUDGE_PROMPT.format(instruction_sample=instruction_sample, solution=solution)
        judge_output: str = self.llm.generate(judge_prompt)

        return "true" in judge_output.lower()


# Convert format
class Formatter:
    def __init__(self, llm: ModelClient) -> None:
        self.llm = llm

    def format_conversion(self,
        input: str,
        output: str,
        input_format: str,
        output_format: str
    ) -> Dict[str, str]:
        # Combine output_format with answer extraction config (e.g., <answer> tags)
        combined_output_format = self.llm.answer_extractor.format_prompts(output_format)

        prompt = FORMAT_CONVERSION_PROMPT.format(
            input=input,
            output=output,
            input_format=input_format,
            output_format=combined_output_format
        )

        output_text: str = self.llm.generate(prompt)

        match = re.search(r'\{.*\}', output_text, re.S)
        if match:
            try:
                result = json.loads(match.group().strip())
                return result
            except:
                return {"input": None, "output": None}
        return {"input": None, "output": None}