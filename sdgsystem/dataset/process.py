import re
import json
import logging
from typing import Dict

from ..models import ModelClient
from ..prompts import FIELD_FILTER_PROMPT, IMAGE_FIELD_FILTER_PROMPT, INSTRUCTION_JUDGE_PROMPT, FORMAT_CONVERSION_PROMPT

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

    def image_field_filter(self, legal_keys) -> Dict[str, str]:
        """
        Identify which fields in an image dataset contain image, input, and output.

        Uses LLM to analyze field names and determine the mapping for:
        - image: field containing image data
        - input: field containing question/prompt
        - output: field containing answer/response

        Args:
            legal_keys: List of field names from the dataset

        Returns:
            Dict with 'image', 'input', 'output' keys mapped to field names (or None)
        """
        legal_keys_list = list(legal_keys) if not isinstance(legal_keys, list) else legal_keys

        prompt = IMAGE_FIELD_FILTER_PROMPT.format(legal_keys=legal_keys_list)
        output: str = self.llm.generate(prompt)

        # Try to extract JSON from response (handle markdown code blocks)
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.S)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
            try:
                result = json.loads(json_str)
                logger.info(f"Image field filter result: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from code block: {e}")

        # Fallback: find raw JSON object
        match = re.search(r'\{.*\}', output, re.S)
        if match:
            json_str = match.group().strip()
            try:
                result = json.loads(json_str)
                logger.info(f"Image field filter result: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse raw JSON: {e}")
                return {"image": None, "input": None, "output": None}

        logger.warning(f"No JSON found in LLM output for image field filter")
        return {"image": None, "input": None, "output": None}


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