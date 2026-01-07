from typing import List, Dict, Optional

from ..prompts import DEFAULT_KEYWORD_EXTRACTION_PROMPT, HF_KEYWORD_EXTRACTION_PROMPT
from ..models import ModelClient, ModelUsageCounter
from ..configs.constants import DEFAULT_KEYWORDS_EXTRACT_EXAMPLES, DEFAULT_HF_KEYWORDS_EXTRACT_EXAMPLES


class KeywordExtractor:
    """
    Extracts domain keywords using LLM based on task instruction and demo examples.
    """

    def __init__(self, llm: ModelClient):
        """
        Initialize keyword extractor.

        Args:
            llm: ModelClient instance for LLM calls
        """
        self.llm = llm

    def extract_keywords(
        self,
        task_instruction: str,
        demo_examples: Optional[List[Dict[str, str]]] = None,
        usage_counter: ModelUsageCounter = None,
        for_huggingface: bool = False
    ) -> List[str]:
        """
        Extract multiple domain keywords by analyzing the task using LLM.

        Args:
            task_instruction: Task instruction text
            demo_examples: List of demo examples with 'input' and 'output' keys
            usage_counter: Optional usage counter to track token and time usage
            for_huggingface: If True, use broader keywords optimized for HuggingFace search

        Returns:
            List of domain keywords (e.g., ["mathematics", "arithmetic", "algebra"])
        """
        if for_huggingface:
            prompt_template = HF_KEYWORD_EXTRACTION_PROMPT
            examples = demo_examples if demo_examples is not None else DEFAULT_HF_KEYWORDS_EXTRACT_EXAMPLES
        else:
            prompt_template = DEFAULT_KEYWORD_EXTRACTION_PROMPT
            examples = demo_examples if demo_examples is not None else DEFAULT_KEYWORDS_EXTRACT_EXAMPLES

        prompt = prompt_template.format(
            task_instruction=task_instruction,
            demo_examples=examples
        )

        response: str = self.llm.generate(prompt, usage_counter=usage_counter)
        keywords = self._parse_keyword_list(response)

        if usage_counter:
            usage_counter.estimate_usage(n=1)

        return keywords

    def _parse_keyword_list(self, response: str) -> List[str]:
        """
        Parse LLM response to extract list of keywords.

        Args:
            response: LLM response text

        Returns:
            List of extracted keywords

        Raises:
            ValueError: If response format is invalid
        """
        try:
            # Find the list in the response
            start_idx = response.find('[')
            end_idx = response.find(']')

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No list found in response")

            list_str = response[start_idx:end_idx + 1]

            # Evaluate as Python list
            keywords = eval(list_str)

            if not isinstance(keywords, list):
                raise ValueError("Parsed result is not a list")

            # Convert all to strings and clean
            return [str(kw).strip() for kw in keywords if kw]

        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid response format for extracting keywords: {e}")
