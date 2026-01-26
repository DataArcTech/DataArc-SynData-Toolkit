"""
Answer Correctness metric using DeepEval's G-Eval.

Compares model's actual output against expected output (ground truth)
using LLM-as-a-Judge with customizable evaluation criteria and rubrics.
"""

from typing import List

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric

from ..config import CorrectnessMetricConfig, RubricItem


def create_correctness_metric(
    config: CorrectnessMetricConfig,
    judge_model: str = "gpt-4.1"
) -> GEval:
    """
    Create Answer Correctness G-Eval metric.

    Args:
        config: CorrectnessMetricConfig with criteria, steps, rubric, threshold
        judge_model: Judge LLM model name (e.g., gpt-4o, gpt-4o-mini)

    Returns:
        Configured GEval metric instance
    """
    # Convert RubricItem to DeepEval Rubric format
    rubric = _convert_rubric(config.rubric)

    return GEval(
        name="AnswerCorrectness",
        criteria=config.criteria,
        evaluation_steps=config.evaluation_steps,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        rubric=rubric,
        threshold=config.threshold,
        model=judge_model,
        async_mode=True
    )


def _convert_rubric(rubric_items: List[RubricItem]) -> List[Rubric]:
    """Convert config RubricItem list to DeepEval Rubric list."""
    return [
        Rubric(
            score_range=tuple(item.score_range),
            expected_outcome=item.expected_outcome
        )
        for item in rubric_items
    ]
