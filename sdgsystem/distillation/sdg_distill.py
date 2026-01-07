import os
import logging
from typing import List, Dict, Optional
from tqdm import tqdm

from ..models import ModelClient, ModelUsageCounter
from ..configs.config import TextDistillConfig
from ..prompts import SDG_DISTILL_BATCH_GENERATION_PROMPT, PATTERN_GENERATION_PROMPT
from ..buffer import TaskBuffer
from .base import BaseDistillation

logger = logging.getLogger(__name__)


class SDGDistillation(BaseDistillation):
    """
    Distillation Generator for creating synthetic data using LLMs.
    """

    def __init__(
        self,
        model: ModelClient,
        config: TextDistillConfig,
        buffer_dir: str = "buffer"
    ):
        """
        Initialize the synthetic data generator.

        Args:
            model: ModelClient instance
            config: TextDistillConfig instance containing task configuration
            buffer_dir: Directory for saving buffer/checkpoint files
        """
        super().__init__(model, config, buffer_dir)

        self.input_instruction = config.input_instruction
        self.output_instruction = config.output_instruction

    def generate(
        self,
        demo_examples: Optional[List[Dict]] = None,
        max_tokens: int = 4096,
        parallel_executor = None,
        reporter = None
    ) -> List[Dict]:
        """
        Generate synthetic data samples using batch generation.

        Args:
            demo_examples: Optional list of demo examples (dict with 'input', 'output')
            max_tokens: Maximum tokens per generation
            parallel_executor: Optional ParallelExecutor for parallel batch generation
            reporter: Optional progress reporter for SSE updates

        Returns:
            List of generated samples, each a dict with 'input' and 'output' keys
        """
        # Get parameters from config
        num_samples = self.config.num_samples
        temperature = self.config.temperature
        batch_size = self.config.batch_size
        results = []

        # Calculate num_batches
        num_batches = (num_samples + batch_size - 1) // batch_size

        # Extract patterns (from demo examples if provided, or from instructions)
        if reporter:
            reporter.start_step("pattern_generation", "Generating Patterns", "Extracting patterns...")
        usage_counter_pattern = ModelUsageCounter(total=1, name="Distillation-Pattern")
        patterns = self.generate_patterns(demo_examples, usage_counter_pattern)
        if reporter:
            reporter.complete_step({"message": "Patterns extracted"})

        logger.info(f"Generating {num_samples} synthetic samples in batches of {batch_size}...")

        # Start generation step
        if reporter:
            reporter.start_step(
                "sample_generation", "Generating Samples",
                message=f"Generating {num_samples} samples...",
                total=num_samples,
                unit="samples"
            )

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

        # Initialize usage_counter for batch generation (tracks batches to match buffer)
        usage_counter = ModelUsageCounter(total=num_batches, name="Distillation-Generation")
        # Initialize buffer
        buffer = TaskBuffer(total=num_batches, save_dir=os.path.join(self.buffer_dir, "Distillation-Generation"))

        # Use parallel execution if available and n_workers > 1
        if parallel_executor and parallel_executor.n_workers > 1:
            logger.info(f"Using parallel execution with {parallel_executor.n_workers} workers")

            # Set up progress callback for parallel processing
            if reporter:
                def on_gen_progress(uc: ModelUsageCounter):
                    samples_generated = min(uc.completed * batch_size, num_samples)
                    reporter.update_step(
                        message=f"Generated batch {uc.completed}/{uc.total}",
                        completed=samples_generated,
                        batch_current=uc.completed,
                        batch_total=uc.total,
                        batch_size=batch_size,
                        tokens=uc.token,
                        time_elapsed=uc.time,
                        estimated_remaining_tokens=uc.estimated_remaining_tokens,
                        estimated_remaining_time=uc.estimated_remaining_time,
                    )
                usage_counter.set_on_update(on_gen_progress)

            # Execute batches in parallel
            batch_results = parallel_executor.execute(
                iterable_inputs=batch_configs,
                process_function=self._generate_single_batch,
                usage_counter=usage_counter,
                n=1,  # track per-batch completion to match buffer
                buffer=buffer
            )

            # Collect results from all batches
            for batch_samples in batch_results:
                if batch_samples:
                    results.extend(batch_samples)

        else:
            # Sequential execution
            logger.info("Using sequential execution")
            batch_results: List[List[Dict]] = buffer.load(usage_counter)
            with tqdm(total=num_samples, desc="Generating samples", unit="sample") as pbar:
                for batch_idx, batch_config in enumerate(batch_configs):
                    if buffer and buffer.detail_progress[batch_idx]:
                        continue

                    batch_samples = self._generate_single_batch(batch_config, usage_counter=usage_counter)
                    batch_results.append(batch_samples)

                    if batch_samples:
                        results.extend(batch_samples)
                        pbar.update(len(batch_samples))

                    # Estimate usage after each batch
                    if usage_counter:
                        usage_counter.estimate_usage(n=1)  # track per-batch completion to match buffer
                    # Save buffer
                    if buffer:
                        buffer.add_progress([batch_idx])
                        buffer.save(batch_results, usage_counter)

                    # Report progress
                    if reporter:
                        samples_generated = min((batch_idx + 1) * batch_size, num_samples)
                        reporter.update_step(
                            message=f"Generated batch {batch_idx + 1}/{num_batches}",
                            completed=samples_generated,
                            batch_current=batch_idx + 1,
                            batch_total=num_batches,
                            batch_size=batch_size,
                            tokens=usage_counter.token,
                            time_elapsed=usage_counter.time,
                            estimated_remaining_tokens=usage_counter.estimated_remaining_tokens,
                            estimated_remaining_time=usage_counter.estimated_remaining_time,
                        )

        logger.info(f"Successfully generated {len(results)}/{num_samples} samples")

        # Complete generation step
        if reporter:
            reporter.complete_step({
                "message": f"Generated {len(results)} samples",
                "samples_generated": len(results)
            })

        # Step 3: Validate samples with majority voting
        if reporter:
            reporter.start_step(
                "validation", "Validating Samples",
                message="Starting validation...",
                total=len(results), unit="samples"
            )

        usage_counter_val = ModelUsageCounter(total=len(results), name="Distillation-Validation")
        buffer_val = TaskBuffer(total=len(results), save_dir=os.path.join(self.buffer_dir, "Distillation-Validation"))

        validated_samples: List[Dict] = self.parse_and_validate_samples(
            response_strings=results,
            output_instruction=self.output_instruction or "",
            usage_counter=usage_counter_val,
            parallel_executor=parallel_executor,
            buffer=buffer_val,
            reporter=reporter
        )

        if reporter:
            reporter.complete_step({
                "valid": len(validated_samples),
                "invalid": len(results) - len(validated_samples)
            })

        logger.info(f"Validation complete: {len(validated_samples)}/{len(results)} samples passed")

        return validated_samples[:num_samples]

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

    def generate_patterns(self, demo_examples: Optional[List[Dict]], usage_counter: ModelUsageCounter = None) -> str:
        """
        Extract general patterns from task instructions and optional demonstration examples.

        Args:
            demo_examples: Optional list of demo examples
            usage_counter: Optional ModelUsageCounter for tracking token usage

        Returns:
            Pattern summary as string
        """
        logger.info("Extracting patterns...")

        # Format demo examples for pattern extraction (if provided)
        demo_text = "No demonstration examples provided."
        if demo_examples:
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
            response = self.model.generate(
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
        input_section = ""
        if self.input_instruction:
            input_section = (
                "## Input Format Instruction\n"
                f"{self.input_instruction}\n"
                "You MUST follow this input format strictly!\n"
            )

        # Build output_instruction section with answer_instruction
        output_section = ""
        formatted_output = self.model.answer_extractor.format_prompts(self.output_instruction)
        if formatted_output:
            output_section = (
                "## Output Format Instruction\n"
                f"{formatted_output}\n"
                "You MUST follow this output format strictly!\n"
            )

        # Build pattern section
        pattern_section = ""
        if patterns:
            pattern_section = (
                "## Extracted Patterns\n"
                "Based on the demonstration examples, here are the general patterns to follow:\n"
                f"{patterns}\n\n"
                "Use these patterns as guidance to create diverse examples.\n"
            )

        # Build demo examples section
        demo_section = ""
        if demo_examples:
            demo_lines = ["## Demonstration Examples", "Here are example inputs and outputs for reference:\n"]
            for idx, example in enumerate(demo_examples, 1):
                demo_lines.append(f"Example {idx}:")
                demo_lines.append(f"Input: {example.get('input', '')}")
                demo_lines.append(f"Output: {example.get('output', '')}\n")
            demo_lines.append("IMPORTANT: Do NOT copy or paraphrase these examples. Generate completely NEW and DIFFERENT examples!")
            demo_section = "\n".join(demo_lines) + "\n"

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
