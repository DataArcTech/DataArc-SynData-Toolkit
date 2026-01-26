"""
Main DeepEval evaluator for post-training model evaluation.

Orchestrates the evaluation pipeline:
1. Load test dataset
2. Run inference on post-trained model (and base model for pairwise)
3. Build DeepEval test cases
4. Run evaluation with configured metrics
5. Output results
"""

import os
import logging
from typing import List, Dict, Any, Optional
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from .config import DeepEvalConfig
from .metrics import (
    create_correctness_metric,
    create_pairwise_metric,
    create_format_compliance_metric,
)
from .metrics.pairwise import interpret_pairwise_score
from ..models import ModelClient
from ..dataset.dataset import Dataset
from ..utils import load_jsonl

logger = logging.getLogger(__name__)


class DeepEvalEvaluator:
    """
    Main evaluator class for DeepEval-based post-training evaluation.

    Handles:
    - Loading test dataset
    - Running inference on models
    - Building DeepEval test cases
    - Running evaluation with metrics
    - Outputting results
    """

    def __init__(self, config: DeepEvalConfig):
        """
        Initialize the evaluator.

        Args:
            config: DeepEvalConfig with all evaluation settings
        """
        self.config = config

        # Models will be loaded on-demand to avoid OOM
        self.post_trained_model = None
        self.base_model = None

        # Initialize metrics
        self._init_metrics()

    def _init_metrics(self):
        """Initialize enabled metrics."""
        if self.config.correctness.enabled:
            logger.info("Initializing Answer Correctness metric...")
            self.correctness_metric = create_correctness_metric(
                config=self.config.correctness,
                judge_model=self.config.judge_model
            )

        if self.config.pairwise.enabled:
            logger.info("Initializing Pairwise Preference metric...")
            self.pairwise_metric = create_pairwise_metric(
                config=self.config.pairwise,
                judge_model=self.config.judge_model
            )

        if self.config.format_compliance.enabled:
            logger.info("Initializing Format Compliance metric...")
            self.format_compliance_metric = create_format_compliance_metric(
                config=self.config.format_compliance,
                judge_model=self.config.judge_model
            )

    def load_dataset(self) -> Dataset:
        """Load test dataset from configured path."""
        logger.info(f"Loading test dataset from {self.config.dataset.path}...")
        samples = load_jsonl(self.config.dataset.path)
        dataset = Dataset.from_list(samples)
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset

    def run_inference(
        self,
        dataset: Dataset,
        model: ModelClient,
        model_name: str = "model"
    ) -> List[str]:
        """
        Run inference on dataset using the specified model.

        Args:
            dataset: Test dataset with 'input' field
            model: ModelClient for inference
            model_name: Name for logging

        Returns:
            List of model outputs
        """
        logger.info(f"Running inference with {model_name}...")

        # Build prompts with output instruction
        output_instruction = self.config.inference.output_instruction
        prompts = []
        for sample in dataset.samples:
            prompt = sample["input"]
            if output_instruction:
                prompt = f"{prompt}\n\n{output_instruction}"
            prompts.append(prompt)

        # Run batch inference
        outputs = model.generate(
            prompts,
            temperature=self.config.inference.temperature,
            max_tokens=self.config.inference.max_tokens
        )

        logger.info(f"Completed inference for {len(outputs)} samples")
        return outputs

    def build_test_cases(
        self,
        dataset: Dataset,
        post_trained_outputs: List[str],
        base_outputs: Optional[List[str]] = None
    ) -> tuple[List[LLMTestCase], List[LLMTestCase]]:
        """
        Build DeepEval test cases from dataset and model outputs.

        Args:
            dataset: Test dataset with 'input' and 'output' (ground truth)
            post_trained_outputs: Outputs from post-trained model
            base_outputs: Outputs from base model (for pairwise comparison)

        Returns:
            Tuple of (standard_test_cases, pairwise_test_cases)
        """
        standard_test_cases = []
        pairwise_test_cases = []

        output_instruction = self.config.inference.output_instruction

        for i, sample in enumerate(dataset.samples):
            input_text = sample["input"]
            expected_output = sample["output"]
            actual_output = post_trained_outputs[i]

            # For format compliance, include output instruction in input
            input_with_instruction = input_text
            if output_instruction:
                input_with_instruction = f"{input_text}\n\n{output_instruction}"

            # Standard test case for correctness and format compliance
            standard_test_cases.append(LLMTestCase(
                input=input_with_instruction,
                actual_output=actual_output,
                expected_output=expected_output
            ))

            # Pairwise test case: actual_output = post-trained, expected_output = base
            if self.config.pairwise.enabled and base_outputs is not None:
                pairwise_test_cases.append(LLMTestCase(
                    input=input_with_instruction,
                    actual_output=actual_output,  # Post-trained model output
                    expected_output=base_outputs[i]  # Base model output
                ))

        return standard_test_cases, pairwise_test_cases

    def run(self) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline.

        Returns:
            Evaluation results dictionary
        """
        # 1. Load dataset
        dataset = self.load_dataset()

        # 2. Load and run inference on post-trained model, then cleanup to free GPU memory
        logger.info("Loading post-trained model...")
        self.post_trained_model = ModelClient(self.config.post_trained_model)
        post_trained_outputs = self.run_inference(
            dataset,
            self.post_trained_model,
            model_name="post-trained model"
        )
        self.post_trained_model.get_model().cleanup()

        # 3. Load and run inference on base model (if pairwise enabled), then cleanup
        base_outputs = None
        if self.config.pairwise.enabled and self.config.pairwise.base_model is not None:
            logger.info("Loading base model...")
            self.base_model = ModelClient(self.config.pairwise.base_model)
            base_outputs = self.run_inference(
                dataset,
                self.base_model,
                model_name="base model"
            )
            self.base_model.get_model().cleanup()

        # 4. Build test cases
        logger.info("Building DeepEval test cases...")
        standard_test_cases, pairwise_test_cases = self.build_test_cases(
            dataset,
            post_trained_outputs,
            base_outputs
        )

        # Ensure output directory exists
        os.makedirs(self.config.output.dir, exist_ok=True)

        # 5. Run each metric individually and save results
        results = {}

        # 5a. Answer Correctness
        if self.config.correctness.enabled:
            logger.info("Running Answer Correctness evaluation...")
            correctness_result = evaluate(
                test_cases=standard_test_cases,
                metrics=[self.correctness_metric]
            )
            results["correctness"] = correctness_result
            self._save_metric_results("correctness", correctness_result)

        # 5b. Format Compliance
        if self.config.format_compliance.enabled:
            logger.info("Running Format Compliance evaluation...")
            format_result = evaluate(
                test_cases=standard_test_cases,
                metrics=[self.format_compliance_metric]
            )
            results["format_compliance"] = format_result
            self._save_metric_results("format_compliance", format_result)

        # 5c. Pairwise Preference (uses different test cases)
        if self.config.pairwise.enabled and pairwise_test_cases and self.pairwise_metric:
            logger.info("Running Pairwise Preference evaluation...")
            pairwise_result = evaluate(
                test_cases=pairwise_test_cases,
                metrics=[self.pairwise_metric]
            )
            results["pairwise"] = pairwise_result

            # Parse and print pairwise summary
            pairwise_summary = self._parse_pairwise_results(pairwise_result)
            results["pairwise_summary"] = pairwise_summary

            self._save_pairwise_results(pairwise_result, pairwise_summary)

        logger.info("Evaluation completed!")
        return results

    def _parse_pairwise_results(self, evaluation_result) -> Dict[str, Any]:
        """
        Parse pairwise evaluation results to determine winners.

        Args:
            evaluation_result: Result from evaluate() call

        Returns:
            Summary with winner counts
        """
        winners = {"base_model": 0, "post_trained_model": 0, "tie": 0}

        for test_result in evaluation_result.test_results:
            # Get the first metric's score (we only run one metric for pairwise)
            if test_result.metrics_data:
                metric_data = test_result.metrics_data[0]
                score = metric_data.score
                if score is not None:
                    winner = interpret_pairwise_score(score)
                    winners[winner] += 1

        total = sum(winners.values())
        return {
            "total": total,
            "base_model_wins": winners["base_model"],
            "post_trained_model_wins": winners["post_trained_model"],
            "ties": winners["tie"],
            "post_trained_win_rate": winners["post_trained_model"] / total if total > 0 else 0
        }

    def _save_metric_results(
        self,
        metric_name: str,
        evaluation_result
    ):
        """Save metric evaluation results to JSON file."""
        import json

        results = []
        for test_result in evaluation_result.test_results:
            metric_data = test_result.metrics_data[0] if test_result.metrics_data else None

            result = {
                "input": test_result.input,
                "expected_output": test_result.expected_output,
                "actual_output": test_result.actual_output,
                "score": metric_data.score if metric_data else None,
                "reason": metric_data.reason if metric_data else None,
                "success": metric_data.success if metric_data else None
            }
            results.append(result)

        # Calculate summary
        scores = [r["score"] for r in results if r["score"] is not None]
        summary = {
            "metric": metric_name,
            "total_samples": len(results),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "pass_rate": sum(1 for r in results if r["success"]) / len(results) if results else 0
        }

        output = {"summary": summary, "results": results}
        output_path = os.path.join(self.config.output.dir, f"{metric_name}_results.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {metric_name} results to {output_path}")

    def _save_pairwise_results(
        self,
        evaluation_result,
        pairwise_summary: Dict[str, Any]
    ):
        """Save pairwise evaluation results to JSON file."""
        import json

        results = []
        for test_result in evaluation_result.test_results:
            metric_data = test_result.metrics_data[0] if test_result.metrics_data else None
            score = metric_data.score if metric_data else None
            winner = interpret_pairwise_score(score) if score is not None else None

            result = {
                "input": test_result.input,
                "post_trained_output": test_result.actual_output,
                "base_output": test_result.expected_output,
                "score": score,
                "winner": winner,
                "reason": metric_data.reason if metric_data else None
            }
            results.append(result)

        output = {"summary": pairwise_summary, "results": results}
        output_path = os.path.join(self.config.output.dir, "pairwise_results.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved pairwise results to {output_path}")
