"""
Core data generator module for synthetic data generation.

This module provides functionality to generate synthetic training data
using LLMs with pattern-based generation.
"""

from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm

from ..configs.config import GenerationConfig
from ..models import ModelClient, ModelUsageCounter
from ..dataset.dataset import Dataset
# Import centralized prompts
from ..prompts import (
    META_PROMPT,
    PATTERN_GENERATION_PROMPT,
)
from ..parallel import ParallelExecutor
from .base import BaseGenerator


logger = logging.getLogger(__name__)


class DataGenerator(BaseGenerator):
    """
    Main class for generating synthetic training data.

    This class handles:
    - Pattern-based data generation using LLM
    - Harder/simpler sample generation

    Note: Random seed should be set globally before using this class.
    """

    def __init__(self, model: ModelClient, config: GenerationConfig):
        """Initialize the DataGenerator."""
        super(DataGenerator, self).__init__(model, config)
        self.config = config

    def generate(self,
        task_definition: str,
        demo_examples: List[Dict[str, str]],
        passages: Optional[List[str]]=None,
        usage_counter: ModelUsageCounter = None, 
        parallel_executor: ParallelExecutor = None, 
    ) -> Dataset:
        
        synthetic_dataset = Dataset()

        input_instruction = self.config.input_instruction
        output_instruction = self.config.output_instruction

        # step1. get pattern
        pattern = self._generate_pattern(
            task_instruction=task_definition,
            input_instruction=input_instruction,
            output_instruction=output_instruction,
            demo_examples=demo_examples,
            usage_counter=usage_counter
        )

        if usage_counter:
            usage_counter.estimate_usage(n=1)

        # step2. let LLM generate samples and get LLM responses (in batches)
        batch_size = getattr(self.config, 'batch_size', 5)  # Default batch size of 5
        sample_strings: List[str] = []

        if parallel_executor and parallel_executor.n_workers > 1:
            # parallel processing
            batch_idxes: List[Tuple[int, int]] = []
            for batch_start in range(0, self.config.num_samples, batch_size):
                batch_end = min(batch_start + batch_size, self.config.num_samples)
                batch_idxes.append((batch_start, batch_end))

            result_batches: List[List[str]] = parallel_executor.execute(
                iterable_inputs=batch_idxes, 
                process_function=self._generate_batch, 
                usage_counter=usage_counter, 
                n=batch_size, 
                # additional fixed arguments
                task_definition=task_definition, 
                input_instruction=input_instruction, 
                output_instruction=output_instruction, 
                pattern=pattern, 
                demo_examples=demo_examples, 
                passages=passages
            )

            for batch in result_batches:
                sample_strings.extend(batch)

        else:
            # sequential processing
            with tqdm(total=self.config.num_samples, desc="Generating samples", unit="sample") as pbar:
                for batch_start in range(0, self.config.num_samples, batch_size):
                    batch_end = min(batch_start + batch_size, self.config.num_samples)
                    batch_length = batch_end - batch_start
                    
                    batch_responses = self._generate_batch(
                        batch_start_end=(batch_start, batch_end), 
                        task_definition=task_definition, 
                        input_instruction=input_instruction, 
                        output_instruction=output_instruction, 
                        pattern=pattern, 
                        demo_examples=demo_examples, 
                        passages=passages, 
                        usage_counter=usage_counter
                    )

                    sample_strings.extend(batch_responses)
                    pbar.update(batch_length)

                    if usage_counter:
                        usage_counter.estimate_usage(n=batch_length)

        # step3. Parse and validate
        samples: List[Dict] = self.parse_and_validate_samples(
            response_strings=sample_strings,
            output_instruction=output_instruction,
            usage_counter=usage_counter, 
            parallel_executor=parallel_executor
        )
        synthetic_dataset.add_samples(samples)

        return synthetic_dataset
    
    def _generate_batch(self, 
        batch_start_end: Tuple[int, int], 
        task_definition: str, 
        input_instruction: str, 
        output_instruction: str, 
        pattern: str, 
        demo_examples: List[Dict[str, str]],
        passages: Optional[List[str]]=None, 
        usage_counter: ModelUsageCounter = None, 
    ) -> List[str]:
        batch_start, batch_end = batch_start_end
        batch_prompts = []

        # Build prompts for this batch (each with its own passage)
        for i in range(batch_start, batch_end):
            passage = passages[i % len(passages)] if passages else None
            prompt = self._build_sample_generation_prompt(
                task_instruction=task_definition,
                input_instruction=input_instruction,
                output_instruction=output_instruction,
                reference_passage=passage,
                demo_examples=demo_examples,
                pattern=pattern
            )
            batch_prompts.append(prompt)

        # Generate batch responses
        batch_responses: List[str] = self.model.generate(
            prompts=batch_prompts,
            n=1,
            usage_counter=usage_counter,
            temperature=self.config.temperature
        )

        return batch_responses


    def _generate_pattern(
        self,
        task_instruction: str,
        input_instruction: str,
        output_instruction: str,
        demo_examples: List[Dict],
        max_examples: int = 50,
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> str:
        """
        Generate a pattern summary from demonstration examples.

        Args:
            task_instruction: The task description
            input_instruction: Input format instruction
            output_instruction: Output format instruction (includes answer formatting)
            demo_examples: List of demonstration examples
            max_examples: Maximum number of examples to use for pattern generation
            usage_counter: Optional usage counter to track token and time usage

        Returns:
            pattern_string
        """
        # Use format_prompts to combine output_instruction with answer_config
        combined_output_instruction = self.model.answer_extractor.format_prompts(output_instruction)

        part_demo_examples = demo_examples[:max_examples]
        prompt = PATTERN_GENERATION_PROMPT.format(
            task_instruction=task_instruction,
            input_instruction=input_instruction,
            output_instruction=combined_output_instruction,
            demo_examples=part_demo_examples
        )

        response: str = self.model.generate(
            prompt,
            n=1,
            usage_counter=usage_counter,
            **kwargs
        )
        return response

    def _build_sample_generation_prompt(
        self,
        task_instruction: str,
        input_instruction: str,
        output_instruction: str,
        reference_passage: str,
        demo_examples: Optional[List[Dict]] = None,
        pattern: Optional[str] = None
    ) -> str:
        """
        Build prompt for sample generation (extracted from generate_sample for batching).

        Args:
            task_instruction: The task description
            input_instruction: Input format instruction
            output_instruction: Output format instruction
            reference_passage: Reference passage/knowledge for generation
            demo_examples: Optional demonstration examples
            pattern: Optional pattern description

        Returns:
            prompt string
        """
        # Use format_prompts to combine output_instruction with answer_config
        combined_output_instruction = self.model.answer_extractor.format_prompts(output_instruction)

        template = META_PROMPT
        template += 'You must consider the task instruction (task knowledge), and the passage (domain knowledge) to generate your training data.'
        template += f""" Here is the task instruction:{task_instruction}\n"""

        if input_instruction:
            template += f""" Here is the input instruction:{input_instruction}\n. You should follow the input format in the instruction strictly to generate data!!!"""

        template += f""" Here is the output instruction:{combined_output_instruction}\n. You should follow the output format in the instruction strictly to generate data!!!"""

        if demo_examples and pattern:
            template += f"""Here is the sample pattern {pattern}"""
            template += """ You can refer to the provided examples. """

            for idx, example in enumerate(demo_examples):
                template += f'Demo Example {idx}: {example}'

        template += " Here is some related knowledge passage that you must refer to. Your generated example must base on the knowledge/information of the passage."
        template += f"Related Objects or Passages:{reference_passage[:min(2048, len(reference_passage))]}"
        template += "Before generating the new example, ensure that you strictly adhere to the rules mentioned in the [Requirement] and follow the format of the [high-quality examples]. Think twice before generating a new example. New example (in JSON):"

        return template

