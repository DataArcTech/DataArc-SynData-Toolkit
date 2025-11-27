import json
from typing import List, Dict, Union, Tuple
from tqdm import tqdm
from ..models import ModelClient, ProcessorArgs, ModelUsageCounter
from ..prompts import SAFETY_SUFFIX
from ..parallel import ParallelExecutor
import logging

logger = logging.getLogger(__name__)


class BaseGenerator:
    """
    Base Class for generating synthetic training data
    """

    def __init__(self, model: ModelClient, config=None) -> None:
        self.model = model
        self.config = config

    def parse_and_validate_samples(self,
        response_strings: List[Union[Dict, str]],
        output_instruction: str, 
        usage_counter: ModelUsageCounter = None, 
        parallel_executor: ParallelExecutor = None, 
    ) -> List[Dict]:
        """
        Parse raw LLM response strings and validate with majority voting.

        This is equivalent to Data_Synthesis_RL's precise_check function with voting=True:
        1. Parse JSON strings into dicts (like example_check)
        2. Filter out unparseable responses
        3. Apply majority voting to all valid samples (always enabled)

        Args:
            response_strings: List of raw LLM response strings (potentially JSON)
            output_instruction: Output format instruction (from config: evaluation.output_instruction)
            answer_config: AnswerExtractionConfig for extracting answers from responses
            voting_config: MajorityVotingConfig for voting configuration
            n_votes: Number of votes per sample (default: 16, following Data_Synthesis_RL)

        Returns:
            List of parsed and validated samples with voting applied
        """
        # Step 1: Parse JSON strings (like example_check in Data_Synthesis_RL)
        parsed_samples = []
        parse_failed_count = 0

        for idx, response_str in enumerate(response_strings):
            if isinstance(response_str, str):
                if not response_str:
                    parse_failed_count += 1
                    continue

                # Try to extract JSON from response
                response_str = response_str.strip()

                # Find JSON object boundaries
                start_idx = response_str.find('{')
                end_idx = response_str.rfind('}')

                if start_idx == -1 or end_idx == -1:
                    parse_failed_count += 1
                    continue

                json_str = response_str[start_idx:end_idx + 1]

                # Try to parse JSON
                try:
                    sample = json.loads(json_str)
                    if isinstance(sample, dict) and 'input' in sample and 'output' in sample:
                        parsed_samples.append(sample)
                    else:
                        parse_failed_count += 1
                except (json.JSONDecodeError, ValueError):
                    # Try eval as fallback
                    try:
                        sample = eval(json_str)
                        if isinstance(sample, dict) and 'input' in sample and 'output' in sample:
                            parsed_samples.append(sample)
                        else:
                            parse_failed_count += 1
                    except:
                        parse_failed_count += 1
            else:
                # already sample in dictionary type
                parsed_samples.append(response_str)

        logger.info(f"  Parsing: {len(parsed_samples)}/{len(response_strings)} valid, {parse_failed_count} failed")

        if not parsed_samples:
            return []

        # Step 2: Apply majority voting (always enabled, following Data_Synthesis_RL)
        validated_samples = self._validate_samples(
            samples=parsed_samples,
            output_instruction=output_instruction, 
            usage_counter=usage_counter, 
            parallel_executor=parallel_executor
        )

        return validated_samples


    def _validate_batch(self,
        batch_samples: List[Dict],
        output_instruction: str = None,
        usage_counter: ModelUsageCounter = None,
    ) -> Tuple[List[Dict], int]:
        """
        Validate and improve samples using majority voting (always enabled).

        Args:
            batch_samples: a batch of samples
            output_instruction: Output format instruction (from config: evaluation.output_instruction)
            usage_counter: instance to count and estimate the token and time usage

        Returns:
            List of validated samples with improved outputs (keeps original on voting failure)
        """
        validated_samples: List[Dict] = []
        failed_count: int = 0
        batch_prompts = []
        for sample in batch_samples:
            input_content = sample["input"]
            # Use format_prompts to combine output_instruction with answer_config
            formatted_instruction = self.model.answer_extractor.format_prompts(output_instruction)
            prompt = str(input_content) + "\n" + formatted_instruction + "\n" + SAFETY_SUFFIX
            batch_prompts.append(prompt)

        # Generate batch responses with majority voting
        batch_outputs = self.model.generate(
            prompts=batch_prompts,
            n=1,
            processor_args=ProcessorArgs.from_dict({
                "majority_voting": {"samples": batch_samples}
            }),
            usage_counter=usage_counter
        )

        # Process batch results
        for sample, selected_output in zip(batch_samples, batch_outputs):
            if selected_output:
                validated_samples.append({
                    "input": sample["input"],
                    "output": selected_output
                })
            else:
                failed_count += 1
                validated_samples.append({
                    "input": sample["input"],
                    "output": sample["output"]
                })

        return validated_samples, failed_count

        
    def _validate_samples(self,
        samples: List[Dict],
        output_instruction: str,
        usage_counter: ModelUsageCounter = None, 
        parallel_executor: ParallelExecutor = None, 
    ) -> List[Dict]:
        """
        Validate and improve samples using majority voting (always enabled).
        Processes samples in batches for efficiency.

        Args:
            samples: List of sample dicts to validate
            output_instruction: Output format instruction (from config: evaluation.output_instruction)
            usage_counter: instance to count and estimate the token and time usage

        Returns:
            List of validated samples with improved outputs (keeps original on voting failure)
        """
        validated_samples: List[Dict] = []
        failed_count: int = 0

        # Get batch_size from config (default to 5 if not available)
        batch_size = getattr(self.config, 'batch_size', 5) if self.config else 5

        # Process samples in batches
        if parallel_executor and parallel_executor.n_workers > 1:
            # parallel processing
            batches: List[List[Dict]] = []
            for batch_start in range(0, len(samples), batch_size):
                batch_end = min(batch_start + batch_size, len(samples))
                batches.append(samples[batch_start:batch_end])

            result_batches: List[Tuple[List[Dict], int]] = parallel_executor.execute(
                iterable_inputs=batches,
                process_function=self._validate_batch,
                usage_counter=usage_counter,
                n=batch_size,
                output_instruction=output_instruction
            )
            for batch_validated_samples, batch_failed_count in result_batches:
                validated_samples.extend(batch_validated_samples)
                failed_count += batch_failed_count
                
        else:
            # sequential processing
            with tqdm(total=len(samples), desc="Validating samples", unit="sample") as pbar:
                for batch_start in range(0, len(samples), batch_size):
                    batch_end = min(batch_start + batch_size, len(samples))
                    batch_samples = samples[batch_start:batch_end]

                    # Build prompts for this batch
                    batch_validated_samples, batch_failed_count = self._validate_batch(
                        batch_samples=batch_samples, 
                        output_instruction=output_instruction, 
                        usage_counter=usage_counter
                    )

                    validated_samples.extend(batch_validated_samples)
                    failed_count += batch_failed_count

                    pbar.update(len(batch_samples))

                    # Estimate usage for the batch
                    if usage_counter:
                        usage_counter.estimate_usage(n=len(batch_samples))

        # Summary
        success_count = len(samples) - failed_count
        logger.info(f"Validation Summary:")
        logger.info(f"Successful: {success_count}/{len(samples)}")
        logger.info(f"Failed: {failed_count}/{len(samples)}")
        logger.info(f"Kept samples: {len(validated_samples)}/{len(samples)}")

        return validated_samples