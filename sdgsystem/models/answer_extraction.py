from typing import List, Optional, Union
import logging

from ..configs.config import AnswerExtractionConfig

logger = logging.getLogger(__name__)


class AnswerExtractor:
    def __init__(self, config: AnswerExtractionConfig) -> None:
        self.config = config

    def format_prompts(self, base_instructions: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Combine base instruction with answer extraction instruction.

        Args:
            base_instruction: Task-specific output instruction

        Returns:
            Complete instruction with answer format
        """
        if self.config is None or not self.config.enabled or not self.config.instruction:
            return base_instructions
        else:
            is_single_instruction = isinstance(base_instructions, str)
            base_instructions = [base_instructions] if is_single_instruction else base_instructions
            instructions: List[str] = []
            for base_instruction in base_instructions:
                separator = '' if not base_instruction else ' '
                instructions.append(f"{base_instruction}{separator}{self.config.instruction}")
            return instructions[0] if is_single_instruction else instructions

    def extract_answers(self, response: Union[str, List[str], List[List[str]]]) -> Optional[Union[str, List[str], List[List[str]]]]:
        """
        Extract answer from LLM response.

        Supports two extraction modes:
        1. XML-style tags (e.g., "<answer>"): Extracts content between paired tags
           - If closing tag found: extracts between <answer> and </answer>
           - If no closing tag: extracts everything after <answer>
        2. Simple markers (e.g., "####"): Extracts everything after the marker

        Args:
            response: LLM output text

        Returns:
            Extracted answer or None if tag not found
        """
        if self.config is None or not self.config.enabled:
            return response  # Return full response if extraction disabled

        def extract_answer_per_response(tag: str, response: str) -> Optional[str]:
            if tag not in response:
                return None

            # Check if tag is XML-style (starts with < and ends with >)
            if tag.startswith("<") and tag.endswith(">") and len(tag) > 2:
                # Extract tag name (e.g., "answer" from "<answer>")
                tag_name = tag[1:-1]
                closing_tag = f"</{tag_name}>"

                # Try to find content between opening and closing tags
                start_idx = response.find(tag)
                if start_idx != -1:
                    content_start = start_idx + len(tag)
                    end_idx = response.find(closing_tag, content_start)

                    if end_idx != -1:
                        # Found paired tags, extract content between them
                        answer = response[content_start:end_idx].strip()
                        return answer
                    else:
                        # No closing tag found, fall back to extracting everything after opening tag
                        answer = response[content_start:].strip()
                        return answer

            # For non-XML tags (like "####"), extract content after first tag
            parts = response.split(tag)
            if len(parts) > 1:
                # Take content after first tag occurrence
                answer = parts[1].strip()
                return answer

            return None

        tag = self.config.tag
        if isinstance(response, str):
            return extract_answer_per_response(tag, response)
        elif isinstance(response, List):
            answers = []
            for single_response in response:
                if isinstance(single_response, str):
                    answers.append(extract_answer_per_response(tag, single_response))
                elif isinstance(single_response, List):
                    sub_answers = []
                    for resp in single_response:
                        if isinstance(resp, str):
                            sub_answers.append(extract_answer_per_response(tag, resp))
                        else:
                            logger.warning(f"Unexpected type in List[List]: {type(resp)}, returning None")
                            sub_answers.append(None)
                    answers.append(sub_answers)
                else:
                    logger.warning(f"Unexpected type in List: {type(single_response)}, returning None")
                    answers.append(None)

            return answers
        else:
            logger.warning(f"Unexpected response type: {type(response)}, returning None")
            return None