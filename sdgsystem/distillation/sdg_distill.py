"""
SDG Synthetic data generator for distillation pipeline.

This module provides a simple LLM-based generator that creates synthetic training
data based on task instructions and optional demonstration examples.
"""

import logging
from typing import List, Dict, Optional
from tqdm import tqdm

from ..models import ModelClient, ModelUsageCounter
from ..configs.config import DistillTaskConfig
from ..prompts import SDG_DISTILL_BATCH_GENERATION_PROMPT, PATTERN_GENERATION_PROMPT
from .base import *

logger = logging.getLogger(__name__)


class SDGDistillation(BaseDistillation):
    """
    SDG Generator for creating synthetic training data using LLMs.

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
        config: DistillTaskConfig
    ):
        """
        Initialize the synthetic data generator.

        Args:
            model: ModelClient instance
            config: DistillTaskConfig instance containing task configuration
        """
        super().__init__(model, config)

        self.input_instruction = config.input_instruction
        self.output_instruction = config.output_instruction

    def generate(
        self,
        demo_examples: Optional[List[Dict]] = None,
        max_tokens: int = 4096,
        parallel_executor = None
    ) -> List[Dict]:
        """
        Generate synthetic data samples using batch generation.

        Args:
            demo_examples: Optional list of demo examples (dict with 'input', 'output')
            max_tokens: Maximum tokens per generation
            parallel_executor: Optional ParallelExecutor for parallel batch generation

        Returns:
            List of generated samples, each a dict with 'input' and 'output' keys
        """
        # Get parameters from config
        num_samples = self.config.num_samples
        temperature = self.config.temperature
        batch_size = self.config.batch_size
        results = []

        # Calculate total iterations: 1 (pattern extraction if demos) + num_batches
        num_batches = (num_samples + batch_size - 1) // batch_size
        total_iterations = (1 if demo_examples else 0) + num_batches
        usage_counter = ModelUsageCounter(total=total_iterations, name="Distillation-Generation")

        # Extract patterns from demo examples if provided
        patterns = ""
        if demo_examples:
            patterns = self.generate_patterns(demo_examples, usage_counter)

        logger.info(f"Generating {num_samples} synthetic samples in batches of {batch_size}...")

        # Build batch configs for all batches
        batch_configs = []
        for batch_idx in range(num_batches):
            remaining = num_samples - (batch_idx * batch_size)
            current_batch_size = min(batch_size, remaining)
            batch_configs.append({
                "batch_idx": batch_idx,
                "batch_size": current_batch_size,
                "demo_examples": demo_examples,
                "patterns": patterns,
                "temperature": temperature,
                "max_tokens": max_tokens
            })

        # Use parallel execution if available and n_workers > 1
        if parallel_executor and parallel_executor.n_workers > 1:
            logger.info(f"Using parallel execution with {parallel_executor.n_workers} workers")

            # Execute batches in parallel
            batch_results = parallel_executor.execute(
                iterable_inputs=batch_configs,
                process_function=self._generate_single_batch,
                usage_counter=usage_counter,
                n=1  # Each batch is 1 iteration
            )

            # Collect results from all batches
            for batch_samples in batch_results:
                if batch_samples:
                    results.extend(batch_samples)

        else:
            # Sequential execution
            logger.info("Using sequential execution")
            with tqdm(total=num_samples, desc="Generating samples", unit="sample") as pbar:
                for batch_config in batch_configs:
                    batch_samples = self._generate_single_batch(batch_config, usage_counter=usage_counter)

                    if batch_samples:
                        results.extend(batch_samples)
                        pbar.update(len(batch_samples))

                    # Estimate usage after each batch
                    if usage_counter:
                        usage_counter.estimate_usage(n=1)

        logger.info(f"Successfully generated {len(results)}/{num_samples} samples")

        return results[:num_samples]

    def _generate_single_batch(
        self,
        batch_config: Dict,
        usage_counter: ModelUsageCounter = None
    ) -> List[Dict]:
        """
        Generate a single batch of samples.

        Args:
            batch_config: Dict containing batch configuration:
                - batch_idx: batch index
                - batch_size: number of samples to generate
                - demo_examples: optional demo examples
                - patterns: extracted patterns
                - temperature: generation temperature
                - max_tokens: max tokens per generation
            usage_counter: Optional ModelUsageCounter for tracking token usage

        Returns:
            List of generated samples
        """
        batch_idx = batch_config["batch_idx"]
        batch_size = batch_config["batch_size"]
        demo_examples = batch_config["demo_examples"]
        patterns = batch_config["patterns"]
        temperature = batch_config["temperature"]
        max_tokens = batch_config["max_tokens"]

        logger.info(f"Generating batch {batch_idx + 1} with {batch_size} samples...")

        # Build prompt for batch generation
        prompt = self._build_batch_prompt(
            demo_examples=demo_examples,
            patterns=patterns,
            batch_size=batch_size
        )

        # Generate using model client
        try:
            response = self.model.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                usage_counter=usage_counter
            )

            # Parse batch response
            batch_samples = self._parse_batch_response(response)

            if batch_samples:
                logger.info(f"Successfully generated {len(batch_samples)} samples in batch {batch_idx + 1}")
                return batch_samples[:batch_size]
            else:
                logger.warning(f"Failed to parse batch {batch_idx + 1}, returning empty list")
                return []

        except Exception as e:
            logger.error(f"Error generating batch {batch_idx + 1}: {e}")
            return []

    def generate_patterns(self, demo_examples: List[Dict], usage_counter: ModelUsageCounter = None) -> str:
        """
        Extract general patterns from demonstration examples.

        Args:
            demo_examples: List of demo examples
            usage_counter: Optional ModelUsageCounter for tracking token usage

        Returns:
            Pattern summary as string
        """
        if not demo_examples:
            return ""

        logger.info("Extracting patterns from demonstration examples...")

        # Format demo examples for pattern extraction
        demo_text = ""
        for idx, example in enumerate(demo_examples, 1):
            demo_text += f"Example {idx}:\n"
            demo_text += f"Input: {example.get('input', '')}\n"
            demo_text += f"Output: {example.get('output', '')}\n\n"

        # Build pattern extraction prompt
        prompt = PATTERN_GENERATION_PROMPT.format(
            task_instruction=self.task_instruction,
            input_instruction=self.input_instruction or "No specific format",
            output_instruction=self.output_instruction or "No specific format",
            demo_examples=demo_text
        )

        try:
            response, _ = self.model.generate(
                prompt,
                temperature=0.3,
                max_tokens=1024,
                n=1,
                usage_counter=usage_counter
            )

            # Report usage for pattern extraction (1 iteration)
            if usage_counter:
                usage_counter.estimate_usage(1)

            logger.info("Pattern extraction completed")
            return response.strip()

        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
            return ""

    

    def _build_batch_prompt(
        self,
        demo_examples: Optional[List[Dict]],
        patterns: str,
        batch_size: int
    ) -> str:
        """Build prompt for batch generation."""

        # Build input instruction section
        if self.input_instruction:
            input_section = f"""## Input Format Instruction
{self.input_instruction}
You MUST follow this input format strictly!
"""
        else:
            input_section = ""

        # Build output instruction section
        if self.output_instruction:
            # Use format_prompts to combine output_instruction with answer_config
            formatted_output = self.model.answer_extractor.format_prompts(self.output_instruction)
            output_section = f"""## Output Format Instruction
{formatted_output}
You MUST follow this output format strictly!
"""
        else:
            output_section = ""

        # Build pattern section
        if patterns:
            pattern_section = f"""## Extracted Patterns
Based on the demonstration examples, here are the general patterns to follow:
{patterns}

Use these patterns as guidance to create diverse examples.
"""
        else:
            pattern_section = ""

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
        prompt = SDG_DISTILL_BATCH_GENERATION_PROMPT.format(
            task_instruction=self.task_instruction,
            input_instruction_section=input_section,
            output_instruction_section=output_section,
            pattern_section=pattern_section,
            demo_examples_section=demo_section,
            batch_size=batch_size
        )

        return prompt
