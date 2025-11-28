"""
Base synthetic data generator for distillation pipeline.

This module provides a simple LLM-based generator that creates synthetic training
data based on task instructions and optional demonstration examples.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path

from ..models import ModelClient
from ..configs.config import DistillTaskConfig

logger = logging.getLogger(__name__)


class BaseDistillation(ABC):
    """
    Base generator for creating synthetic training data using LLMs.

    This generator creates synthetic question-answer pairs based on:
    - Task instruction (required)
    - Input/output format instructions (optional)
    - Demonstration examples (optional)

    No passage retrieval - pure instruction-based generation.
    Uses batch generation with patterns for better diversity.
    """

    def __init__(
        self,
        model: ModelClient,
        config: DistillTaskConfig,
        buffer_dir: str = "buffer"
    ):
        """
        Initialize the synthetic data generator.

        Args:
            model: ModelClient instance
            config: DistillTaskConfig instance containing task configuration
            buffer_dir: Directory for saving buffer/checkpoint files
        """
        self.model = model
        self.config = config
        self.task_instruction = config.task_instruction
        self.buffer_dir = buffer_dir

    @abstractmethod
    def generate(
        self,
    ) -> List[Dict]:
        pass
    
    @abstractmethod
    def _build_batch_prompt(
        self,
    ) -> str:
        pass
    
    def _parse_batch_response(self, response: str) -> List[Dict]:
        """
        Parse LLM response to extract list of {'input': ..., 'output': ...}.

        Args:
            response: Raw LLM response string

        Returns:
            List of parsed dicts, empty list if parsing fails
        """
        # Try to find JSON array in response
        try:
            # Find first [ and last ]
            start = response.find('[')
            end = response.rfind(']') + 1

            if start == -1 or end == 0:
                # Maybe it's a single object, try that
                return [self._parse_single_response(response)]

            json_str = response[start:end]

            # Try to parse as array
            parsed = json.loads(json_str)

            if not isinstance(parsed, list):
                return []

            # Validate each item has required keys
            results = []
            for item in parsed:
                if isinstance(item, dict) and 'input' in item and 'output' in item:
                    results.append({
                        'input': item['input'],
                        'output': item['output']
                    })

            return results

        except (json.JSONDecodeError, ValueError):
            # Try parsing as single object
            single = self._parse_single_response(response)
            return [single] if single else []

    def _parse_single_response(self, response: str) -> Optional[Dict]:
        """
        Parse LLM response to extract single {'input': ..., 'output': ...}.

        Args:
            response: Raw LLM response string

        Returns:
            Parsed dict or None if parsing fails
        """
        try:
            # Find first { and last }
            start = response.find('{')
            end = response.rfind('}') + 1

            if start == -1 or end == 0:
                return None

            json_str = response[start:end]

            # Try to parse
            parsed = json.loads(json_str)

            # Validate has required keys
            if 'input' in parsed and 'output' in parsed:
                return {
                    'input': parsed['input'],
                    'output': parsed['output']
                }

        except (json.JSONDecodeError, ValueError):
            pass

        return None
    
    def export_to_jsonl(self, samples: List[Dict], output_path: str):
        """
        Export generated samples to JSONL file.

        Args:
            samples: List of samples to export
            output_path: Path to output JSONL file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        logger.info(f"Exported {len(samples)} samples to {output_path}")
