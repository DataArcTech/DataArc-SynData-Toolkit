import os
import json
from typing import List, Dict, Optional, Tuple
import logging

from ..configs.config import GenerationConfig
from ..models import ModelClient, ModelUsageCounter
from ..dataset.dataset import Dataset
from ..prompts import TEXT_META_PROMPT, PATTERN_GENERATION_PROMPT, IMAGE_META_PROMPT
from ..parallel import ParallelExecutor
from ..buffer import TaskBuffer
from .base import BaseGenerator

logger = logging.getLogger(__name__)


class TextDataGenerator(BaseGenerator):
    """
    Main class for generating synthetic training data.

    This class handles:
    - Pattern-based data generation using LLM
    - Harder/simpler sample generation
    """

    def __init__(self, model: ModelClient, config: GenerationConfig):
        """Initialize the TextDataGenerator."""
        super(TextDataGenerator, self).__init__(model, config)
        self.config = config

    def generate(self,
        task_definition: str,
        demo_examples: List[Dict[str, str]],
        passages: Optional[List[str]]=None,
        usage_counter: ModelUsageCounter = None,
        parallel_executor: ParallelExecutor = None,
        reporter=None,
    ) -> Dataset:
        synthetic_dataset = Dataset()

        input_instruction = self.config.input_instruction
        output_instruction = self.config.output_instruction

        # initial usage_counter
        task_name = usage_counter.name if usage_counter else "Generator"

        # step1. get pattern (separate counter for pattern generation)
        if reporter:
            reporter.start_step("pattern_generation", "Generating Pattern", "Extracting patterns of data generation...")

        usage_counter_pattern = ModelUsageCounter(total=1, name=task_name + "-Pattern") if usage_counter else None
        pattern = self._generate_pattern(
            task_instruction=task_definition,
            input_instruction=input_instruction,
            output_instruction=output_instruction,
            demo_examples=demo_examples,
            usage_counter=usage_counter_pattern
        )

        if usage_counter_pattern:
            usage_counter_pattern.estimate_usage(n=1)

        if reporter:
            reporter.complete_step()

        # step2. let LLM generate samples and get LLM responses (in batches)
        batch_size = getattr(self.config, 'batch_size', 5)  # Default batch size of 5
        batch_idxes: List[Tuple[int, int]] = []
        for batch_start in range(0, self.config.num_samples, batch_size):
            batch_end = min(batch_start + batch_size, self.config.num_samples)
            batch_idxes.append((batch_start, batch_end))

        if reporter:
            reporter.start_step(
                "sample_generation", "Generating Samples",
                message="Starting sample generation...",
                total=self.config.num_samples, unit="samples"
            )

        # initialize usage_counter for batch generation (tracks batches to match buffer)
        usage_counter_gen = ModelUsageCounter(total=len(batch_idxes), name=f"{task_name}-Generation") if usage_counter else None
        # initialize buffer
        buffer_gen = TaskBuffer(total=len(batch_idxes), save_dir=os.path.join(self.buffer_dir, f"{task_name}-Generation"))
        # generate
        string_batches: List[List[str]] = []
        generated_count = 0

        if parallel_executor and parallel_executor.n_workers > 1:
            # set up progress callback for parallel processing
            if reporter and usage_counter_gen:
                def on_gen_progress(uc: ModelUsageCounter):
                    samples_completed = min(uc.completed * batch_size, self.config.num_samples)
                    reporter.update_step(
                        message=f"Generated batch {uc.completed}/{uc.total}",
                        completed=samples_completed,
                        batch_current=uc.completed,
                        batch_total=uc.total,
                        batch_size=batch_size,
                        tokens=uc.token,
                        time_elapsed=uc.time,
                        estimated_remaining_tokens=uc.estimated_remaining_tokens,
                        estimated_remaining_time=uc.estimated_remaining_time,
                    )
                usage_counter_gen.set_on_update(on_gen_progress)

            # parallel processing
            string_batches: List[List[str]] = parallel_executor.execute(
                iterable_inputs=batch_idxes,
                process_function=self._generate_batch,
                usage_counter=usage_counter_gen,
                n=1,  # track per-batch completion to match buffer
                buffer=buffer_gen,
                # additional fixed arguments
                task_definition=task_definition,
                input_instruction=input_instruction,
                output_instruction=output_instruction,
                pattern=pattern,
                demo_examples=demo_examples,
                passages=passages
            )

        else:
            # sequential processing
            string_batches: List[List[str]] = buffer_gen.load(usage_counter_gen)
            for sample_idx, (batch_start, batch_end) in enumerate(batch_idxes):
                if buffer_gen and buffer_gen.detail_progress[sample_idx]:
                    generated_count += (batch_end - batch_start)
                    continue

                batch_length = batch_end - batch_start
                batch_responses = self._generate_batch(
                    batch_start_end=(batch_start, batch_end),
                    task_definition=task_definition,
                    input_instruction=input_instruction,
                    output_instruction=output_instruction,
                    pattern=pattern,
                    demo_examples=demo_examples,
                    passages=passages,
                    usage_counter=usage_counter_gen
                )

                string_batches.append(batch_responses)
                generated_count += batch_length

                # Estimate usage first (updates completed count for accurate estimates)
                if usage_counter_gen:
                    usage_counter_gen.estimate_usage(n=1)  # track per-batch completion to match buffer

                # Report progress after estimate_usage so token/time estimates are accurate
                if reporter:
                    reporter.update_step(
                        message=f"Generated batch {sample_idx + 1}/{len(batch_idxes)}",
                        completed=generated_count,
                        batch_current=sample_idx + 1,
                        batch_total=len(batch_idxes),
                        batch_size=batch_size,
                        tokens=usage_counter_gen.token if usage_counter_gen else None,
                        time_elapsed=usage_counter_gen.time if usage_counter_gen else None,
                        estimated_remaining_tokens=usage_counter_gen.estimated_remaining_tokens if usage_counter_gen else None,
                        estimated_remaining_time=usage_counter_gen.estimated_remaining_time if usage_counter_gen else None,
                    )

                if buffer_gen:
                    buffer_gen.add_progress([sample_idx])
                    buffer_gen.save(string_batches, usage_counter_gen)

        sample_strings: List[str] = []
        for batch in string_batches:
            sample_strings.extend(batch)

        if reporter:
            reporter.complete_step({"generated": len(sample_strings)})

        # step3. Parse and validate
        if reporter:
            reporter.start_step(
                "validation", "Validating Samples",
                message="Starting validation...",
                total=len(sample_strings), unit="samples"
            )

        # initialize usage_counter and buffer
        usage_counter_val = ModelUsageCounter(total=len(sample_strings), name=f"{task_name}-Validation")
        buffer_val = TaskBuffer(total=len(sample_strings), save_dir=os.path.join(self.buffer_dir, f"{task_name}-Validation"))
        # parse and validate
        samples: List[Dict] = self.parse_and_validate_samples(
            response_strings=sample_strings,
            output_instruction=output_instruction,
            usage_counter=usage_counter_val,
            parallel_executor=parallel_executor,
            buffer=buffer_val,
            reporter=reporter
        )
        synthetic_dataset.add_samples(samples)

        if reporter:
            reporter.complete_step({"valid": len(samples), "invalid": len(sample_strings) - len(samples)})

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

        template = TEXT_META_PROMPT
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


class ImageDataGenerator(BaseGenerator):
    """
    Generator for creating QA data from images using VLM.

    Each image generates ceil(num_samples / num_images) QA pairs in one VLM call.
    Images are processed in batches of batch_size.
    """

    def __init__(self, model: ModelClient, config: GenerationConfig):
        super(ImageDataGenerator, self).__init__(model, config)
        self.config = config

    def generate(
        self,
        task_instruction: str,
        image_paths: List[str],
        image_contexts: Dict[str, str] = None,
        usage_counter: ModelUsageCounter = None,
        parallel_executor: ParallelExecutor = None,
        reporter=None,
    ) -> Dataset:
        synthetic_dataset = Dataset()

        input_instruction = self.config.input_instruction
        output_instruction = self.config.output_instruction
        num_samples = self.config.num_samples
        num_images = len(image_paths)
        image_contexts = image_contexts or {}

        task_name = usage_counter.name if usage_counter else "Image-Generator"

        import math

        # Limit images to num_samples to avoid generating excess data
        # If num_images > num_samples, only use first num_samples images (1 sample per image)
        if num_images > num_samples:
            logger.info(f"Limiting images from {num_images} to {num_samples} (num_samples)")
            image_paths = image_paths[:num_samples]
            num_images = num_samples

        samples_per_image = math.ceil(num_samples / num_images)

        batch_size = getattr(self.config, 'batch_size', 5)
        batch_idxes: List[Tuple[int, int]] = []
        for batch_start in range(0, num_images, batch_size):
            batch_end = min(batch_start + batch_size, num_images)
            batch_idxes.append((batch_start, batch_end))

        if reporter:
            reporter.start_step(
                "sample_generation", "Generating QA Samples",
                message=f"Generating {samples_per_image} samples per image...",
                total=num_images, unit="images"
            )

        usage_counter_gen = ModelUsageCounter(total=len(batch_idxes), name=task_name + "-Generation") if usage_counter else None
        buffer_gen = TaskBuffer(total=len(batch_idxes), save_dir=os.path.join(self.buffer_dir, task_name + "-Generation"))

        sample_batches: List[List[Dict]] = []
        generated_count = 0

        if parallel_executor and parallel_executor.n_workers > 1:
            if reporter and usage_counter_gen:
                def on_gen_progress(uc: ModelUsageCounter):
                    images_completed = min(uc.completed * batch_size, num_images)
                    reporter.update_step(
                        message=f"Processed {images_completed}/{num_images} images",
                        completed=images_completed,
                        batch_current=uc.completed,
                        batch_total=uc.total,
                        batch_size=batch_size,
                        tokens=uc.token,
                        time_elapsed=uc.time,
                        estimated_remaining_tokens=uc.estimated_remaining_tokens,
                        estimated_remaining_time=uc.estimated_remaining_time,
                    )
                usage_counter_gen.set_on_update(on_gen_progress)

            sample_batches = parallel_executor.execute(
                iterable_inputs=batch_idxes,
                process_function=self._generate_batch,
                usage_counter=usage_counter_gen,
                n=1,
                buffer=buffer_gen,
                task_instruction=task_instruction,
                input_instruction=input_instruction,
                output_instruction=output_instruction,
                image_paths=image_paths,
                image_contexts=image_contexts,
                samples_per_image=samples_per_image
            )

        else:
            sample_batches = buffer_gen.load(usage_counter_gen)
            for batch_idx, (batch_start, batch_end) in enumerate(batch_idxes):
                if buffer_gen and buffer_gen.detail_progress[batch_idx]:
                    generated_count += (batch_end - batch_start)
                    continue

                batch_length = batch_end - batch_start
                batch_samples = self._generate_batch(
                    batch_start_end=(batch_start, batch_end),
                    task_instruction=task_instruction,
                    input_instruction=input_instruction,
                    output_instruction=output_instruction,
                    image_paths=image_paths,
                    image_contexts=image_contexts,
                    samples_per_image=samples_per_image,
                    usage_counter=usage_counter_gen
                )

                sample_batches.append(batch_samples)
                generated_count += batch_length

                if usage_counter_gen:
                    usage_counter_gen.estimate_usage(n=1)

                if reporter:
                    reporter.update_step(
                        message=f"Processed {generated_count}/{num_images} images",
                        completed=generated_count,
                        batch_current=batch_idx + 1,
                        batch_total=len(batch_idxes),
                        batch_size=batch_size,
                        tokens=usage_counter_gen.token if usage_counter_gen else None,
                        time_elapsed=usage_counter_gen.time if usage_counter_gen else None,
                        estimated_remaining_tokens=usage_counter_gen.estimated_remaining_tokens if usage_counter_gen else None,
                        estimated_remaining_time=usage_counter_gen.estimated_remaining_time if usage_counter_gen else None,
                    )

                if buffer_gen:
                    buffer_gen.add_progress([batch_idx])
                    buffer_gen.save(sample_batches, usage_counter_gen)

        all_samples: List[Dict] = []
        for batch in sample_batches:
            all_samples.extend(batch)

        if reporter:
            reporter.complete_step({"generated": len(all_samples)})

        if reporter:
            reporter.start_step(
                "validation", "Validating Samples",
                message="Starting validation...",
                total=len(all_samples), unit="samples"
            )

        usage_counter_val = ModelUsageCounter(total=len(all_samples), name=task_name + "-Validation")
        buffer_val = TaskBuffer(total=len(all_samples), save_dir=os.path.join(self.buffer_dir, task_name + "-Validation"))

        validated_samples: List[Dict] = self.parse_and_validate_samples(
            response_strings=all_samples,
            output_instruction=output_instruction,
            usage_counter=usage_counter_val,
            parallel_executor=parallel_executor,
            buffer=buffer_val,
            reporter=reporter,
            modality="image"
        )

        validated_samples = validated_samples[:num_samples]
        synthetic_dataset.add_samples(validated_samples)

        if reporter:
            reporter.complete_step({"valid": len(validated_samples), "invalid": len(all_samples) - len(validated_samples)})

        return synthetic_dataset

    def _generate_batch(
        self,
        batch_start_end: Tuple[int, int],
        task_instruction: str,
        input_instruction: str,
        output_instruction: str,
        image_paths: List[str],
        image_contexts: Dict[str, str],
        samples_per_image: int,
        usage_counter: ModelUsageCounter = None,
    ) -> List[Dict]:
        """Generate QA samples for a batch of images."""
        batch_start, batch_end = batch_start_end
        batch_prompts = []
        batch_images = []

        for i in range(batch_start, batch_end):
            image_path = image_paths[i]
            context = image_contexts.get(image_path, None)
            prompt = self._build_image_generation_prompt(
                task_instruction=task_instruction,
                input_instruction=input_instruction,
                output_instruction=output_instruction,
                num_samples=samples_per_image,
                context=context
            )
            batch_prompts.append(prompt)
            batch_images.append(image_path)

        batch_responses: List[str] = self.model.generate_with_images(
            prompts=batch_prompts,
            images=batch_images,
            usage_counter=usage_counter,
            temperature=self.config.temperature
        )

        results: List[Dict] = []
        for response, image_path in zip(batch_responses, batch_images):
            parsed_samples = self._parse_json_array(response)
            for sample in parsed_samples:
                sample['image'] = image_path
                results.append(sample)

        return results

    def _build_image_generation_prompt(
        self,
        task_instruction: str,
        input_instruction: str,
        output_instruction: str,
        num_samples: int,
        context: Optional[str] = None
    ) -> str:
        """Build prompt for image-based QA generation.

        Args:
            task_instruction: Task description
            input_instruction: Input format instruction
            output_instruction: Output format instruction
            num_samples: Number of QA pairs to generate
            context: Optional surrounding text context from the document (None if images provided without documents)

        Returns:
            Complete prompt string
        """
        output_instruction = self.model.answer_extractor.format_prompts(output_instruction)

        # Build optional sections
        input_section = f"Input format instruction: {input_instruction}\n" if input_instruction else ""
        context_section = ""
        if context:
            # Only include context section when context is available (from PDF extraction)
            context_section = (
                f"\nDocument context (text surrounding this image in the source document):\n"
                f"{context}\n"
                f"Use this context to better understand what the image represents and to generate more relevant questions.\n\n"
            )
        example_section = ""
        if self.model.answer_extractor.config and self.model.answer_extractor.config.enabled:
            tag = self.model.answer_extractor.config.tag
            # Check if tag is XML-style (e.g., <answer>) or simple marker (e.g., ####)
            if tag.startswith("<") and tag.endswith(">") and len(tag) > 2:
                # XML-style tag: wrap answer inside tags
                tag_name = tag[1:-1]
                closing_tag = f"</{tag_name}>"
                example_output = f"Category A shows 45% {tag}45%{closing_tag}"
            else:
                # Simple marker: answer follows tag
                example_output = f"Category A shows 45% {tag} 45%"
            example_section = (
                f'\nExample of correct output format:\n'
                f'[{{"input": "What percentage is shown for category A?", "output": "{example_output}"}}]\n'
            )

        # Build prompt
        template = IMAGE_META_PROMPT.format(num_samples=num_samples)
        template += context_section  # Context first to set the scene
        template += f"Task instruction: {task_instruction}\n"
        template += input_section
        template += f"Output format instruction: {output_instruction}\n"
        template += f"\nGenerate {num_samples} diverse question-answer pairs as a JSON array.\n"
        template += "Each element should be a dictionary with 'input' and 'output' keys.\n"
        template += "Each 'input' value MUST follow the input format instruction.\n"
        template += "Each 'output' value MUST follow the output format instruction.\n"
        template += example_section
        template += "\nOutput ONLY the JSON array, nothing else.\n\n"
        template += "JSON Array Output:"

        return template

    def _validate_batch(
        self,
        batch_samples: List[Dict],
        output_instruction: str,
        usage_counter: ModelUsageCounter = None,
    ) -> Tuple[List[Dict], int]:
        """Validate samples using VLM with image context and majority voting."""
        from ..models import ProcessorArgs

        validated_samples: List[Dict] = []
        failed_count: int = 0
        batch_prompts = []
        batch_images = []

        for sample in batch_samples:
            input_content = sample["input"]
            image_path = sample.get("image", "")
            prompt = str(input_content) + "\n" + output_instruction
            batch_prompts.append(prompt)
            batch_images.append(image_path)

        # Generate batch responses with majority voting
        batch_outputs = self.model.generate_with_images(
            prompts=batch_prompts,
            images=batch_images,
            n=1,
            processor_args=ProcessorArgs.from_dict({
                "majority_voting": {"samples": batch_samples}
            }),
            usage_counter=usage_counter
        )

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

    def _parse_json_array(self, response: str) -> List[Dict]:
        """Parse VLM response containing JSON array of samples."""
        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']')

            if start_idx == -1 or end_idx == -1:
                obj_start = response.find('{')
                obj_end = response.rfind('}')
                if obj_start != -1 and obj_end != -1:
                    sample = json.loads(response[obj_start:obj_end + 1])
                    if isinstance(sample, dict) and 'input' in sample and 'output' in sample:
                        return [sample]
                return []

            json_str = response[start_idx:end_idx + 1]
            samples = json.loads(json_str)

            if isinstance(samples, list):
                valid = [s for s in samples if isinstance(s, dict) and 'input' in s and 'output' in s]
                return valid
            return []

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON array from VLM response: {e}")
            return []

