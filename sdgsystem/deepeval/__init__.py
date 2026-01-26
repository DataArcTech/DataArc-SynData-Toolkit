"""
DeepEval-based model evaluation module for post-training assessment.

This module provides:
- Answer Correctness evaluation using G-Eval
- Pairwise Preference evaluation using G-Eval
- Format Compliance evaluation using G-Eval
"""

from .config import DeepEvalConfig
from .evaluator import DeepEvalEvaluator

__all__ = [
    "DeepEvalConfig",
    "DeepEvalEvaluator",
]
