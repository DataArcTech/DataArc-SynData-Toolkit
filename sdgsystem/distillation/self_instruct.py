import logging
from typing import List, Dict, Optional

from ..models import ModelClient
from ..configs.config import TextDistillConfig
from ..prompts import SELF_INSTRUCT_BATCH_GENERATION_PROMPT
from .base import BaseDistillation

logger = logging.getLogger(__name__)


class SelfInstructDistillation(BaseDistillation):
    """
    Self-Instruct Generator for creating synthetic training data using LLMs.

    This generator creates synthetic question-answer pairs based on:
    - Task instruction (required)
    - Demonstration examples (required)

    No passage retrieval - pure instruction-based generation.
    """

    def __init__(
        self,
        model: ModelClient,
        config: TextDistillConfig
    ):
        """
        Initialize the synthetic data generator.

        Args:
            model: ModelClient instance
            config: TextDistillConfig instance containing task configuration
        """
        super().__init__(model, config)

    def generate(
        self,
        demo_examples: Optional[List[Dict]] = None,
        max_tokens: int = 4096
    ) -> List[Dict]:
        """
        Generate synthetic data samples using batch generation.

        Args:
            demo_examples: Optional list of demo examples (dict with 'input', 'output')
            max_tokens: Maximum tokens per generation

        Returns:
            List of generated samples, each a dict with 'input' and 'output' keys
        """
        # Get parameters from config
        num_samples = self.config.num_samples
        temperature = self.config.temperature
        batch_size = self.config.batch_size
        results = []

        logger.info(f"Generating {num_samples} synthetic samples in batches of {batch_size}...")

        # Calculate number of batches needed
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            # Calculate batch size for this iteration
            remaining = num_samples - len(results)
            current_batch_size = min(batch_size, remaining)

            logger.info(f"Batch {batch_idx + 1}/{num_batches}: Generating {current_batch_size} samples...")

            # Build prompt for batch generation (provide all demo examples to every batch)
            prompt = self._build_batch_prompt(
                demo_examples=demo_examples,
                batch_size=current_batch_size
            )

            # Generate using model client
            try:
                response = self.model.generate(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1
                )

                # Parse batch response
                batch_samples = self._parse_batch_response(response)

                if batch_samples:
                    results.extend(batch_samples[:current_batch_size])
                    logger.info(f"Successfully generated {len(batch_samples)} samples in this batch")
                else:
                    logger.warning(f"Failed to parse batch {batch_idx + 1}, skipping")

            except Exception as e:
                logger.error(f"Error generating batch {batch_idx + 1}: {e}")

        logger.info(f"Successfully generated {len(results)}/{num_samples} samples")

        return results[:num_samples]
    
    def _build_batch_prompt(
        self,
        demo_examples: Optional[List[Dict]],
        batch_size: int
    ) -> str:
        """Build prompt for batch generation."""

        # Build demo examples section (show all demo examples if provided)
        if demo_examples:
            demo_section = "## Demonstration Examples\n"
            demo_section += "Here are example inputs and outputs for reference:\n\n"
            for idx, example in enumerate(demo_examples, 1):
                demo_section += f"Example {idx}:\n"
                demo_section += f"Input: {example.get('input', '')}\n"
                demo_section += f"Output: {example.get('output', '')}\n\n"
            demo_section += "IMPORTANT: Do NOT copy or paraphrase these examples. Generate completely NEW and DIFFERENT examples!\n"
        else:
            demo_section = ""

        # Format the main prompt
        prompt = SELF_INSTRUCT_BATCH_GENERATION_PROMPT.format(
            task_instruction=self.task_instruction,
            demo_examples_section=demo_section,
            batch_size=batch_size
        )

        return prompt
