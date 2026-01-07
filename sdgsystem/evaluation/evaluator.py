from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

from ..configs.config import EvaluationConfig
from ..dataset.dataset import Dataset
from ..models import ModelClient, ModelUsageCounter
from .answer_comparison import AnswerComparer

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for assessing synthetic data quality with the working model.

    This class handles:
    1. Batch inference on synthetic data
    2. Answer comparison against ground truth

    Attributes:
        base_model: BaseModel instance for inference
        input_instruction: Instruction prepended to input
        output_instruction: Instruction for output format
        comparison_config: AnswerComparisonConfig for comparing answers
    """

    def __init__(
        self,
        base_model: ModelClient,
        llm: ModelClient,
        config: EvaluationConfig
    ):
        """
        Initialize the evaluator.

        Args:
            base_model: BaseModel instance for inference
            llm: LLM client for answer comparison (if using llm_judge)
            config: EvaluationConfig with all evaluation settings
        """
        self.base_model = base_model
        self.input_instruction = config.input_instruction
        self.output_instruction = config.output_instruction
        self.inference_config = config.inference
        self.scoring_config = config.scoring
        self.answer_comparer = AnswerComparer(
            config=config.answer_comparison_config,
            llm=llm.get_model() if llm is not None else None
        )

    def evaluate(
        self,
        dataset: Dataset,
        mode: str = "inference",
        modality: str = "text",
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Evaluate, filter, and split samples from dataset.

        Args:
            dataset: Dataset, samples with 'input' and 'output' keys (and 'image' key for image modality)
            mode: "inference" for binary evaluation (n=1) or "scoring" for pass@n
            modality: "text" for text-only evaluation, "image" for VLM evaluation
            output_dir: Output directory for resolving relative image paths (required for image modality)

        Returns:
            Evaluation results of dataset, e.g.:
            {
                "scores": [s0, s1, ..., s_i, ...],
            }
            i is the index of sample in dataset
        """
        config = self.inference_config if mode == "inference" else self.scoring_config

        usage_counter = ModelUsageCounter(total=len(dataset), name="Evaluator")

        if modality == "image":
            evaluated_samples = self.evaluate_samples_with_images(
                dataset.samples,
                usage_counter,
                output_dir=output_dir,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                n=config.n
            )
        else:
            evaluated_samples = self.evaluate_samples(
                dataset.samples,
                usage_counter,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                n=config.n
            )
        scores = [s for s, _ in evaluated_samples]

        # Log scoring summary
        total = len(scores)
        if total > 0:
            solved = sum(1 for s in scores if s == 1.0)
            learnable = sum(1 for s in scores if 0 < s < 1.0)
            unsolved = sum(1 for s in scores if s == 0.0)
            logger.info(f"Evaluation Summary ({mode}, {modality}): Total={total}, "
                       f"Solved={solved} ({solved/total*100:.1f}%), "
                       f"Learnable={learnable} ({learnable/total*100:.1f}%), "
                       f"Unsolved={unsolved} ({unsolved/total*100:.1f}%)")

        return {"scores": scores}

    def evaluate_samples(
        self,
        samples: List[Dict[str, str]],
        usage_counter: ModelUsageCounter = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        n: int = 1
    ) -> List[Tuple[float, Dict[str, str]]]:
        """
        Evaluate samples by running inference and comparing with ground truth.

        Args:
            samples: List of samples with 'input' and 'output' keys
            usage_counter: Counter for time tracking
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            n: Number of responses to generate per sample

        Returns:
            List of (score, sample) tuples where:
            - score: float (0.0 to 1.0) = correct_count / n
            - sample: dict with added 'score', 'correct_predictions', 'all_predictions'

            Note: When n=1, score will be either 0.0 or 1.0
        """
        # Build all prompts
        prompts = [self._build_prompt(sample['input']) for sample in samples]

        # Run inference - vLLM handles batching internally
        desc = "Evaluating samples" if n == 1 else f"Scoring samples (n={n})"
        logger.info(f"{desc}: {len(samples)} samples")

        predictions = self.base_model.generate(
            prompts,
            n=n,
            usage_counter=usage_counter,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Track time after inference
        if usage_counter:
            usage_counter.estimate_usage(n=len(samples))

        # Evaluate each sample
        results = []
        for sample, responses in tqdm(zip(samples, predictions), total=len(samples), desc="Comparing answers"):
            ground_truth = sample['output']

            # Ensure responses is a list
            if not isinstance(responses, list):
                responses = [responses]

            # Evaluate each of the n responses
            eval_results = [
                self._evaluate_single(sample['input'], ground_truth, pred)
                for pred in responses
            ]

            # Calculate score as average
            score = sum(eval_results) / len(eval_results)

            # Store correct predictions
            correct_preds = [pred for pred, res in zip(responses, eval_results) if res]

            # Create sample with score info
            sample_with_score = {
                **sample,
                'score': score,
                'correct_predictions': correct_preds,
                'all_predictions': responses
            }

            results.append((score, sample_with_score))

        return results

    def evaluate_samples_with_images(
        self,
        samples: List[Dict[str, str]],
        usage_counter: ModelUsageCounter = None,
        output_dir: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        n: int = 1
    ) -> List[Tuple[float, Dict[str, str]]]:
        """
        Evaluate samples with images by running VLM inference and comparing with ground truth.

        Args:
            samples: List of samples with 'input', 'output', and 'image' keys
            usage_counter: Counter for time tracking
            output_dir: Output directory for resolving relative image paths
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            n: Number of responses to generate per sample

        Returns:
            List of (score, sample) tuples where:
            - score: float (0.0 to 1.0) = correct_count / n
            - sample: dict with added 'score', 'correct_predictions', 'all_predictions'
        """
        import os

        # Build all prompts and collect image paths
        prompts = [self._build_prompt(sample['input']) for sample in samples]

        # Resolve image paths (relative to output_dir if provided)
        image_paths = []
        for sample in samples:
            image_path = sample.get('image', '')
            if output_dir and not os.path.isabs(image_path):
                image_path = os.path.join(output_dir, image_path)
            image_paths.append(image_path)

        # Run VLM inference
        desc = "Evaluating samples (VLM)" if n == 1 else f"Scoring samples (VLM, n={n})"
        logger.info(f"{desc}: {len(samples)} samples")

        predictions = self.base_model.generate_with_images(
            prompts=prompts,
            images=image_paths,
            n=n,
            usage_counter=usage_counter,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Track time after inference
        if usage_counter:
            usage_counter.estimate_usage(n=len(samples))

        # Evaluate each sample
        results = []
        for sample, responses in tqdm(zip(samples, predictions), total=len(samples), desc="Comparing answers"):
            ground_truth = sample['output']

            # Ensure responses is a list
            if not isinstance(responses, list):
                responses = [responses]

            # Evaluate each of the n responses
            eval_results = [
                self._evaluate_single(sample['input'], ground_truth, pred)
                for pred in responses
            ]

            # Calculate score as average
            score = sum(eval_results) / len(eval_results)

            # Store correct predictions
            correct_preds = [pred for pred, res in zip(responses, eval_results) if res]

            # Create sample with score info
            sample_with_score = {
                **sample,
                'score': score,
                'correct_predictions': correct_preds,
                'all_predictions': responses
            }

            results.append((score, sample_with_score))

        return results

    def _build_prompt(self, sample_input: str) -> str:
        """
        Build prompt from sample input and instructions.

        Args:
            sample_input: The input/question from sample

        Returns:
            Complete prompt for model inference
        """
        prompt = str(sample_input)

        # Use format_prompts to combine output_instruction with answer_instruction
        formatted_instruction = self.base_model.answer_extractor.format_prompts(
            self.output_instruction
        )
        if formatted_instruction:
            prompt += " " + formatted_instruction

        return prompt

    def _evaluate_single(
        self,
        sample_input: str,
        ground_truth: str,
        prediction: str, 
        usage_counter: ModelUsageCounter = None, 
    ) -> bool:
        """
        Evaluate a single prediction against ground truth.

        Args:
            sample_input: Original input/question
            ground_truth: Ground truth answer
            prediction: Model prediction

        Returns:
            True if prediction matches ground truth, False otherwise
        """
        # Extract answer from ground truth
        gt_answer = self.base_model.answer_extractor.extract_answers(ground_truth)
        gt_extracted = gt_answer is not None

        # Extract answer from prediction
        pred_answer = self.base_model.answer_extractor.extract_answers(prediction)
        pred_extracted = pred_answer is not None

        # Decision logic: if both succeed, compare extracted; if either fails, compare full texts
        if not gt_extracted or not pred_extracted:
            # At least one extraction failed - compare full texts for fairness
            gt_answer = str(ground_truth).strip() if ground_truth is not None else ""
            pred_answer = str(prediction).strip() if prediction is not None else ""

        # Compare answers using configured method
        is_correct = self.answer_comparer.compare_answers(
            predicted=pred_answer,
            ground_truth=gt_answer, 
            usage_counter=usage_counter, 
            question=sample_input,
        )

        return is_correct