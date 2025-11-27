"""
Pipeline for obtaining trainnig data according to input task definition.
input task definition -> local / web / distill task -> evaluate -> output: training dataset
"""
import os
import logging

from .configs.config import *
from .models import ModelClient
from .tasks import TaskExecutor
from .evaluation import Evaluator
from .evaluation.answer_comparison import SemanticComparison
from .generation import BaseRewriter
from .translation.translator import Translator
from .parallel import ParallelExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)



class Pipeline:
    def __init__(self,
        config: SDGSConfig
    ) -> None:
        self.config = config

        # parallel executor
        self.parallel_executor = ParallelExecutor(n_workers=self.config.n_workers)

        # models
        self.llm = ModelClient(self.config.generator_config, self.config.answer_config, self.config.postprocess_config)
        self.base_model = ModelClient(self.config.base_model_config, self.config.answer_config)
        # tasks
        self.task_executor = TaskExecutor(self.config.task_config, self.llm)
        # evaluation
        self.evaluator = Evaluator.from_config(
            base_model=self.base_model,
            llm=self.llm,
            config=self.config.evaluation_config
        )
        # rewrite
        self.rewriter = BaseRewriter.get_specific_rewriter(self.llm, self.config.rewrite_config)
        # translation (lazy initialization - only if target language is not English)
        self.translator = None
        if self.config.translation_config.language.lower() != "english":
            self.translator = Translator(self.config.translation_config)

    def run(self, save_path: str = None, export_format: str = None):
        """
        Run the pipeline with two-stage evaluation process:
        1. Generate initial dataset (logged in task executor)
        2. Binary evaluation with inference config (temp=0.0, n=1)
        3. Rewrite samples based on difficulty (make harder/easier)
        4. Re-evaluate with scoring config (temp=1.2, n=64)
        5. Categorize and save only learnable samples (0 < score < 1)
        6. Translate to target language if needed
        7. Save final datasets

        Args:
            save_path: Path to save the final dataset
            export_format: Format to export (jsonl, json, etc.)
        """
        # 1. Generate initial dataset (task-specific steps logged in executor)
        dataset = self.task_executor.execute(parallel_executor=self.parallel_executor)

        # 2. Initial evaluation with inference config (binary: solved vs unsolved)
        logger.info("=== Step: Evaluating Initial Dataset ===")
        initial_evaluation = self._evaluate_with_inference(dataset)
        logger.info(f"Initial evaluation completed: {len(dataset)} samples evaluated")

        # 3. Rewrite samples based on difficulty
        logger.info("=== Step: Rewriting Samples Based on Difficulty ===")
        rewritten_dataset = self.rewriter.rewrite(dataset, initial_evaluation, parallel_executor=self.parallel_executor)
        logger.info(f"Rewriting completed: {len(rewritten_dataset)} samples processed")

        # 4. Re-evaluate rewritten samples with scoring config (granular scoring)
        logger.info("=== Step: Re-evaluating Rewritten Samples ===")
        final_evaluation = self._evaluate_with_scoring(rewritten_dataset)
        logger.info(f"Re-evaluation completed with scoring config")

        # 5. Categorize by score into three datasets
        logger.info("=== Step: Categorizing Results ===")
        solved, learnable, unsolved = rewritten_dataset.categorize_by_score(final_evaluation['scores'])
        logger.info(f"Categorization: {len(solved)} solved, {len(learnable)} learnable, {len(unsolved)} unsolved")

        # 6. Translate to target language if needed
        if self.translator is not None:
            target_language = self.config.translation_config.language
            logger.info(f"=== Step: Translating Datasets to {target_language} ===")

            solved = self.translator.translate_dataset(solved, target_lang=target_language)
            learnable = self.translator.translate_dataset(learnable, target_lang=target_language)
            unsolved = self.translator.translate_dataset(unsolved, target_lang=target_language)
            logger.info(f"Translation to {target_language} completed")

        # 7. Save all three datasets to separate files
        logger.info("=== Step: Saving Datasets ===")
        final_save_path = save_path if save_path else os.path.join(
            self.config.output_dir,
            f"{self.config.task_config.name}_final.{export_format if export_format else self.config.export_format}"
        )
        final_export_format = export_format if export_format else self.config.export_format

        rewritten_dataset.save_categorized(solved, learnable, unsolved, final_save_path, final_export_format)
        logger.info(f"Datasets saved to {final_save_path}")

        # 8. Cleanup: Release models and free GPU memory
        logger.info("=== Step: Cleaning up Models ===")
        self._cleanup_models()
        logger.info("Model cleanup completed")

    def _evaluate_with_inference(self, dataset):
        """
        Evaluate with inference config for binary classification.

        Returns scores of 0.0 (unsolved) or 1.0 (solved).
        """
        inference_params = self._get_inference_params()
        return self.evaluator.evaluate(dataset, **inference_params)

    def _evaluate_with_scoring(self, dataset):
        """
        Evaluate with scoring config for granular difficulty assessment.

        Returns scores between 0.0 and 1.0 for pass@n evaluation.
        """
        scoring_params = self._get_scoring_params()
        return self.evaluator.evaluate(dataset, **scoring_params)

    def _get_inference_params(self):
        """Get inference parameters from base_model config."""
        inference_config = self.config.base_model_config.config.inference
        return {
            'temperature': inference_config.temperature,
            'max_tokens': inference_config.max_tokens,
            'n': inference_config.n,
        }

    def _get_scoring_params(self):
        """Get scoring parameters from base_model config."""
        scoring_config = self.config.base_model_config.config.scoring
        return {
            'temperature': scoring_config.temperature,
            'max_tokens': scoring_config.max_tokens,
            'n': scoring_config.n,
        }

    def _cleanup_models(self):
        """Release vLLM, SentenceTransformer, and Translation models, free GPU memory after evaluation completes."""
        try:
            # 1. Cleanup vLLM base model
            if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'model'):
                base_language_model = self.base_model.model.model
                if hasattr(base_language_model, 'cleanup'):
                    base_language_model.cleanup()

            # 2. Cleanup semantic comparison model (only if using semantic method)
            if (hasattr(self.evaluator, 'answer_comparer') and
                isinstance(self.evaluator.answer_comparer, SemanticComparison) and
                hasattr(self.evaluator.answer_comparer, 'cleanup')):
                self.evaluator.answer_comparer.cleanup()

            # 3. Cleanup semantic clustering voting model (only if using semantic_clustering method)
            if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'processor'):
                processor = self.llm.model.processor
                # Check if processor is MajorityVotingProcessor
                if hasattr(processor, 'voting'):
                    from .models.postprocess.majority_voting import SemanticClusteringVoting
                    if isinstance(processor.voting, SemanticClusteringVoting) and hasattr(processor.voting, 'cleanup'):
                        processor.voting.cleanup()

            # 4. Cleanup translation model (only if translator was initialized)
            if self.translator is not None and hasattr(self.translator, 'cleanup'):
                self.translator.cleanup()

        except Exception as e:
            print(f"Warning: Failed to cleanup models: {e}")
