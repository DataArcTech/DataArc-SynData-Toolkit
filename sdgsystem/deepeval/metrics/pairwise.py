"""
Pairwise Preference metric using DeepEval's G-Eval.

Compares base model vs post-trained model outputs using LLM-as-a-Judge
to determine which response is better based on a strict priority order.

Uses discrete scoring:
- Score 10: Response A (post-trained model) wins
- Score 5: Tie (both are equivalent)
- Score 0: Response B (base model) wins
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from ..config import PairwiseMetricConfig


def create_pairwise_metric(
    config: PairwiseMetricConfig,
    judge_model: str = "gpt-4.1"
) -> GEval:
    """
    Create Pairwise Preference G-Eval metric.

    Args:
        config: PairwiseMetricConfig with criteria and evaluation_steps
        judge_model: Judge LLM model name (e.g., gpt-4.1)

    Returns:
        Configured GEval metric instance for pairwise comparison
    """
    return GEval(
        name="PairwisePreference",
        criteria=config.criteria,
        evaluation_steps=config.evaluation_steps,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,  # Not used for win/loss determination
        model=judge_model,
        async_mode=True
    )


def interpret_pairwise_score(score: float) -> str:
    """
    Interpret pairwise GEval score to determine winner.

    Args:
        score: GEval score (0-1 scale, converted from 0-10)

    Returns:
        Winner string: "post_trained_model", "base_model", or "tie"
    """
    # Convert 0-1 scale back to 0-10
    s10 = int(round(score * 10))

    if s10 <= 2:  # Score 0
        return "base_model"
    elif s10 >= 8:  # Score 10
        return "post_trained_model"
    else:  # Score 5
        return "tie"
