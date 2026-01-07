import time
from abc import ABC, abstractmethod
from typing import Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity

from ..models.models import BaseLanguageModel
from ..models.usage_counter import ModelUsageCounter
from ..prompts import LLM_JUDGE_COMPARISON_PROMPT
from ..configs.config import (
    BaseComparisonConfig,
    ExactMatchComparisonConfig,
    SemanticComparisonConfig,
    LLMJudgeComparisonConfig
)

logger = logging.getLogger(__name__)


class AnswerComparer:
    def __init__(self,
        config: BaseComparisonConfig,
        llm: Optional[BaseLanguageModel] = None
    ) -> None:
        self.config = config
        self.llm = llm

        if self.config is None:
            raise Exception(f"Set AnswerComparer, but the config is None.")

        self.comparison: BaseComparison = BaseComparison.from_config(
            self.config,
            model=self.llm
        )

    def compare_answers(self, 
        predicted: str, 
        ground_truth: str, 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> bool:
        """
        Compare two answers using the specified method.

        This is a convenience wrapper that routes to the appropriate comparison function
        based on the method parameter.

        Args:
            predicted: Predicted answer from model
            ground_truth: Ground truth answer
            **kwargs: Method-specific parameters (e.g., Optional "question" for context (used by llm_judge))

        Returns:
            True if answers match according to the chosen method, False otherwise
        """
        result = self.comparison.compare(predicted, ground_truth, usage_counter, **kwargs)

        return result


class BaseComparison(ABC):
    def __init__(self, config: BaseComparisonConfig) -> None:
        self.config = config

    @staticmethod
    def from_config(config: BaseComparisonConfig, **kwargs) -> "BaseComparison":
        if isinstance(config, ExactMatchComparisonConfig):
            return ExactMatchComparison(config)

        if isinstance(config, SemanticComparisonConfig):
            return SemanticComparison(config)

        if isinstance(config, LLMJudgeComparisonConfig):
            return LLMJudgeComparison(config, model=kwargs.pop("model"))

        raise Exception(f"Comparison method {config.method} is not supported.")

    @abstractmethod
    def compare(self, 
        predicted: str, 
        ground_truth: str, 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> bool:
        pass

    def _exact_match_comparison(self, 
        predicted: str, 
        ground_truth: str, 
        numeric_tolerance: float = 1e-3, 
        usage_counter: ModelUsageCounter = None, 
    ) -> bool:
        st = time.time()
        pred = predicted.strip()
        gt = ground_truth.strip()

        # Try numeric comparison if tolerance is provided
        if numeric_tolerance is not None and numeric_tolerance > 0:
            try:
                pred_num = float(pred)
                gt_num = float(gt)
                return abs(pred_num - gt_num) < numeric_tolerance
            except (ValueError, TypeError):
                # Not numeric, fall through to string comparison
                pass
        
        if usage_counter:
            usage_counter.add_usage(0, time.time() - st)
        # Exact string match
        return pred == gt


class ExactMatchComparison(BaseComparison):
    def __init__(self, config: ExactMatchComparisonConfig) -> None:
        super(ExactMatchComparison, self).__init__(config)
        self.config: ExactMatchComparisonConfig

    def compare(self, 
        predicted: str, 
        ground_truth: str, 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> bool:
        """
        Compare two answers using exact match or numeric tolerance.

        Algorithm:
        1. Try to parse both as numbers
        2. If both are numeric, use tolerance-based comparison
        3. Otherwise, use exact string match (after stripping whitespace)

        Args:
            predicted: Predicted answer from model
            ground_truth: Ground truth answer
            numeric_tolerance: Tolerance for numeric comparison (e.g., 1e-3)
                            Set to 0 or None for strict string matching

        Returns:
            True if answers match, False otherwise
        """
        return self._exact_match_comparison(predicted, ground_truth, self.config.numeric_tolerance, usage_counter)


class SemanticComparison(BaseComparison):
    def __init__(self, config: SemanticComparisonConfig) -> None:
        super(SemanticComparison, self).__init__(config)
        self.config: SemanticComparisonConfig
        self.model = None

    def compare(self, 
        predicted: str, 
        ground_truth: str, 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> bool:
        """
        Compare two answers using embedding-based semantic similarity.

        Algorithm:
        1. Encode both answers using BGE embeddings
        2. Compute cosine similarity
        3. Return True if similarity >= threshold

        Args:
            predicted: Predicted answer from model
            ground_truth: Ground truth answer
            model_path: Path to SentenceTransformer model
            similarity_threshold: Cosine similarity threshold (0.0-1.0)

        Returns:
            True if answers are semantically similar, False otherwise
        """
        st = time.time()
        try:
            if self.model is None:
                from sentence_transformers import SentenceTransformer
                device = self.config.device
                logger.info(f"Loading SentenceTransformer model: {self.config.model_path} on device: {device}")
                self.model = SentenceTransformer(self.config.model_path, device=device)
                logger.info(f"SentenceTransformer model loaded successfully")

            # Encode both answers using pre-loaded model
            embeddings = self.model.encode([predicted, ground_truth], convert_to_numpy=True)

            # Compute cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            logger.debug(f"Semantic similarity: {similarity:.4f} (threshold: {self.config.similarity_threshold})")
            
            if usage_counter:
                usage_counter.add_usage(0, time.time() - st)

            return similarity >= self.config.similarity_threshold

        except Exception as e:
            logger.warning(f"semantic_comparison failed ({e}), falling back to exact_match")
            return self._exact_match_comparison(predicted, ground_truth, usage_counter=usage_counter)

    def cleanup(self):
        """
        Release SentenceTransformer model and free GPU memory.

        Call this method when you're done with semantic comparison to release GPU resources.
        """
        if self.model is not None:
            logger.info(f"Releasing SentenceTransformer model and freeing GPU memory...")

            # Delete the model instance
            del self.model
            self.model = None

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CUDA cache
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info(f"SentenceTransformer GPU memory released successfully!")
            except ImportError:
                logger.info(f"SentenceTransformer model released (torch not available for cache clearing)")


class LLMJudgeComparison(BaseComparison):
    def __init__(self, config: LLMJudgeComparisonConfig, model: BaseLanguageModel) -> None:
        super(LLMJudgeComparison, self).__init__(config)
        self.config: LLMJudgeComparisonConfig
        self.model = model

    def compare(self, 
        predicted: str, 
        ground_truth: str, 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> bool:
        """
        Compare two answers using LLM judge.

        Algorithm:
        1. Present both answers to LLM judge with question context
        2. Ask LLM to determine if predicted answer is semantically equivalent to ground truth
        3. Parse LLM response (correct/incorrect)

        Args:
            predicted: Predicted answer from model
            ground_truth: Ground truth answer
            question: Optional question/input for context
            temperature: Temperature for LLM judge

        Returns:
            True if LLM judges answers as equivalent, False otherwise
        """
        try:
            question: str = kwargs.pop("question", "N/A")
            # Build the judge prompt
            prompt = LLM_JUDGE_COMPARISON_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                predicted=predicted
            )

            # Query LLM judge (model inherited from global config via query_openai)
            response = self.model.generate(prompt, n=1, usage_counter=usage_counter, temperature=self.config.temperature)

            # Parse response
            response_lower = response.strip().lower()

            # Check for "correct" or "incorrect" in response
            if "correct" in response_lower and "incorrect" not in response_lower:
                return True
            elif "incorrect" in response_lower:
                return False
            else:
                # Unclear response, fall back to exact match
                logger.warning(f"Warning: Unclear LLM judge response '{response}', falling back to exact_match")
                return self._exact_match_comparison(predicted, ground_truth, usage_counter=usage_counter)

        except Exception as e:
            logger.warning(f"Warning: llm_judge_comparison failed ({e}), falling back to exact_match")
            return self._exact_match_comparison(predicted, ground_truth, usage_counter=usage_counter)
