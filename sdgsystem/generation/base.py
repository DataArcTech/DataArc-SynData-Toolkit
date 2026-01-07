import os
import json
from typing import List, Dict, Union, Tuple, Optional
from tqdm import tqdm
from ..models import ModelClient, ProcessorArgs, ModelUsageCounter
from ..parallel import ParallelExecutor
from ..buffer import TaskBuffer
import logging

logger = logging.getLogger(__name__)


class BaseGenerator:
    """
    Base Class for generating synthetic training data
    """

    def __init__(self, 
        model: ModelClient, 
        config=None, 
        buffer_dir: str = "buffer"
    ) -> None:
        self.model = model
        self.config = config
        self.buffer_dir = buffer_dir

    def parse_and_validate_samples(self,
        response_strings: List[Union[Dict, str]],
        output_instruction: str,
        usage_counter: ModelUsageCounter = None,
        parallel_executor: ParallelExecutor = None,
        buffer: TaskBuffer = None,
        reporter=None,
        modality: str = "text",
        output_dir: Optional[str] = None,
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
            usage_counter: Usage counter for tracking token and time usage
            parallel_executor: Optional parallel executor for batch processing
            buffer: Optional task buffer for caching progress
            reporter: Optional progress reporter
            modality: "text" for text-only validation, "image" for VLM validation
            output_dir: Output directory for resolving relative image paths (required for image modality)

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

        logger.info(f"Parsing: {len(parsed_samples)}/{len(response_strings)} valid, {parse_failed_count} failed")

        if not parsed_samples:
            return []

        # Step 2: Apply majority voting (always enabled, following Data_Synthesis_RL)
        validated_samples = self._validate_samples(
            samples=parsed_samples,
            output_instruction=output_instruction,
            usage_counter=usage_counter,
            parallel_executor=parallel_executor,
            buffer=buffer,
            reporter=reporter,
            modality=modality,
            output_dir=output_dir
        )

        return validated_samples


    def _validate_batch(self, 
        batch_samples: List[Dict], 
        output_instruction: str,
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
            prompt = str(input_content) + "\n" + output_instruction
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

        # Handle None response - voting failed due to insufficient valid answers
        if batch_outputs is None:
            raise RuntimeError(
                "Majority voting failed: not enough valid answers could be extracted from model responses. "
                "Please optimize your task instructions to help the model generate responses in the expected format, "
                "or increase 'n_voting' in postprocess.majority_voting config to get more candidate answers."
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

    def _validate_batch_with_images(self,
        batch_samples: List[Dict],
        output_instruction: str,
        usage_counter: ModelUsageCounter = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[Dict], int]:
        """
        Validate and improve samples with images using VLM and majority voting.

        Args:
            batch_samples: a batch of samples (each containing 'image' key)
            output_instruction: Output format instruction
            usage_counter: instance to count and estimate the token and time usage
            output_dir: Output directory for resolving relative image paths

        Returns:
            List of validated samples with improved outputs (keeps original on voting failure)
        """
        validated_samples: List[Dict] = []
        failed_count: int = 0
        batch_prompts = []
        batch_images = []

        for sample in batch_samples:
            input_content = sample["input"]
            prompt = str(input_content) + "\n" + output_instruction
            batch_prompts.append(prompt)

            # Resolve image path
            image_path = sample.get("image", "")
            if output_dir and image_path and not os.path.isabs(image_path):
                image_path = os.path.join(output_dir, image_path)
            batch_images.append(image_path)

        # Generate batch responses with VLM and majority voting
        batch_outputs = self.model.generate_with_images(
            prompts=batch_prompts,
            images=batch_images,
            n=1,
            processor_args=ProcessorArgs.from_dict({
                "majority_voting": {"samples": batch_samples}
            }),
            usage_counter=usage_counter
        )

        # Handle None response - voting failed due to insufficient valid answers
        if batch_outputs is None:
            raise RuntimeError(
                "Majority voting failed: not enough valid answers could be extracted from VLM responses. "
                "Please optimize your instructions to help the model generate responses in the expected format, "
                "or increase 'n_voting' in postprocess.majority_voting config to get more candidate answers."
            )

        # Process batch results
        for sample, selected_output in zip(batch_samples, batch_outputs):
            if selected_output:
                validated_samples.append({
                    "input": sample["input"],
                    "output": selected_output,
                    "image": sample.get("image", "")
                })
            else:
                failed_count += 1
                validated_samples.append({
                    "input": sample["input"],
                    "output": sample["output"],
                    "image": sample.get("image", "")
                })

        return validated_samples, failed_count

    def _validate_samples(self,
        samples: List[Dict],
        output_instruction: str,
        usage_counter: ModelUsageCounter = None,
        parallel_executor: ParallelExecutor = None,
        buffer: TaskBuffer = None,
        reporter=None,
        modality: str = "text",
        output_dir: Optional[str] = None,
    ) -> List[Dict]:
        """
        Validate and improve samples using majority voting (always enabled).
        Processes samples in batches for efficiency.

        Args:
            samples: List of sample dicts to validate
            output_instruction: Output format instruction (from config: evaluation.output_instruction)
            usage_counter: instance to count and estimate the token and time usage
            parallel_executor: Optional parallel executor for batch processing
            buffer: Optional task buffer for caching progress
            reporter: Optional progress reporter
            modality: "text" for text-only validation, "image" for VLM validation
            output_dir: Output directory for resolving relative image paths (required for image modality)

        Returns:
            List of validated samples with improved outputs (keeps original on voting failure)
        """
        validated_samples: List[Dict] = []
        failed_count: int = 0

        # Get batch_size from config (default to 5 if not available)
        batch_size = getattr(self.config, 'batch_size', 5) if self.config else 5

        # Process samples in batches
        batches: List[List[Dict]] = []
        for batch_start in range(0, len(samples), batch_size):
            batch_end = min(batch_start + batch_size, len(samples))
            batches.append(samples[batch_start:batch_end])

        # Resize buffer and usage_counter to match actual batch count (based on parsed_samples)
        buffer.resize_total(total=len(batches))
        if usage_counter:
            usage_counter.resize_total(total=len(batches))

        # Select validation function based on modality
        if modality == "image":
            validate_func = self._validate_batch_with_images
            validate_kwargs = {"output_dir": output_dir}
        else:
            validate_func = self._validate_batch
            validate_kwargs = {}

        if parallel_executor and parallel_executor.n_workers > 1:
            # set up progress callback for parallel processing
            if reporter and usage_counter:
                def on_val_progress(uc: ModelUsageCounter):
                    samples_completed = min(uc.completed * batch_size, len(samples))
                    reporter.update_step(
                        message=f"Validated batch {uc.completed}/{uc.total}",
                        completed=samples_completed,
                        batch_current=uc.completed,
                        batch_total=uc.total,
                        batch_size=batch_size,
                        tokens=uc.token,
                        time_elapsed=uc.time,
                        estimated_remaining_tokens=uc.estimated_remaining_tokens,
                        estimated_remaining_time=uc.estimated_remaining_time,
                    )
                usage_counter.set_on_update(on_val_progress)

            # parallel processing
            result_batches: List[Tuple[List[Dict], int]] = parallel_executor.execute(
                iterable_inputs=batches,
                process_function=validate_func,
                output_instruction=output_instruction,
                usage_counter=usage_counter,
                n=1,  # track per-batch completion to match buffer
                buffer=buffer,
                **validate_kwargs
            )

            validated_batches: List[List[Dict]] = []
            for batch_validated_samples, batch_failed_count in result_batches:
                validated_batches.append(batch_validated_samples)
                failed_count += batch_failed_count

        else:
            # sequential processing
            validated_batches: List[List[Dict]] = buffer.load(usage_counter)
            validated_count = 0
            with tqdm(total=len(samples), desc="Validating samples", unit="sample") as pbar:
                for sample_idx, batch_samples in enumerate(batches):
                    if buffer and buffer.detail_progress[sample_idx]:
                        validated_count += len(batch_samples)
                        continue

                    # Validate this batch
                    batch_validated_samples, batch_failed_count = validate_func(
                        batch_samples=batch_samples,
                        output_instruction=output_instruction,
                        usage_counter=usage_counter,
                        **validate_kwargs
                    )

                    validated_batches.append(batch_validated_samples)
                    failed_count += batch_failed_count
                    validated_count += len(batch_samples)

                    pbar.update(len(batch_samples))

                    # Estimate usage first (updates completed count for accurate estimates)
                    if usage_counter:
                        usage_counter.estimate_usage(n=1)  # track per-batch completion to match buffer

                    # Report progress after estimate_usage so token/time estimates are accurate
                    if reporter:
                        reporter.update_step(
                            message=f"Validated batch {sample_idx + 1}/{len(batches)}",
                            completed=validated_count,
                            batch_current=sample_idx + 1,
                            batch_total=len(batches),
                            batch_size=batch_size,
                            tokens=usage_counter.token if usage_counter else None,
                            time_elapsed=usage_counter.time if usage_counter else None,
                            estimated_remaining_tokens=usage_counter.estimated_remaining_tokens if usage_counter else None,
                            estimated_remaining_time=usage_counter.estimated_remaining_time if usage_counter else None,
                        )

                    # save buffer
                    if buffer:
                        buffer.add_progress([sample_idx])
                        buffer.save(validated_batches, usage_counter)

        for batch in validated_batches:
            validated_samples.extend(batch)

        # Summary
        success_count = len(samples) - failed_count
        logger.info(f"Validation Summary:")
        logger.info(f"  Successful: {success_count}/{len(samples)}")
        logger.info(f"  Failed: {failed_count}/{len(samples)}")
        logger.info(f"  Kept samples: {len(validated_samples)}/{len(samples)}")

        return validated_samples