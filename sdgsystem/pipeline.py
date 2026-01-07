import os
import shutil
import logging

from .configs.config import *
from .models import ModelClient
from .tasks import TaskExecutor
from .evaluation import Evaluator
from .evaluation.answer_comparison import SemanticComparison
from .generation import BaseRewriter
from .translation.translator import ArabicTranslator
from .parallel import ParallelExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self,
        config: SDGSConfig
    ) -> None:
        self.config = config
        self.parallel_executor = ParallelExecutor(n_workers=self.config.n_workers)
        self.llm = ModelClient(self.config.generator_config, self.config.answer_config, self.config.postprocess_config)
        self.base_model = ModelClient(self.config.base_model_config, self.config.answer_config)
        self.task_executor = TaskExecutor(self.config.task_config, self.llm)
        self.evaluator = Evaluator(self.base_model, self.llm, self.config.evaluation_config)
        self.rewriter = BaseRewriter.get_specific_rewriter(self.llm, self.config.rewrite_config)
        self.translator = None
        if self.config.translation_config.language.lower() != "english":
            self.translator = ArabicTranslator(self.config.translation_config)

    def run(self, save_path: str = None, export_format: str = None):
        """
        Run the pipeline with two-stage process:
        1. Generate initial dataset (logged in task executor)
        2. Initial dataset evaluating and rewriting

        Args:
            save_path: Path to save the final dataset
            export_format: Format to export (jsonl, json, etc.)
        """
        # Detect modality from task config
        modality = "text"
        if self.config.task_config.image is not None:
            modality = "image"

        # Generate initial dataset (task-specific steps logged in executor)
        dataset = self.task_executor.execute(parallel_executor=self.parallel_executor)

        # Initial evaluation (binary: solved vs unsolved)
        initial_evaluation = self.evaluator.evaluate(
            dataset,
            mode="inference",
            modality=modality,
            output_dir=self.config.output_dir
        )
        logger.info(f"Initial evaluation completed: {len(dataset)} samples evaluated")

        # Rewrite samples based on difficulty
        rewritten_dataset = self.rewriter.rewrite(
            dataset,
            initial_evaluation,
            parallel_executor=self.parallel_executor,
            modality=modality,
            output_dir=self.config.output_dir
        )
        logger.info(f"Rewriting completed: {len(rewritten_dataset)} samples processed")

        # Re-evaluate rewritten samples with scoring (pass@n)
        final_evaluation = self.evaluator.evaluate(
            rewritten_dataset,
            mode="scoring",
            modality=modality,
            output_dir=self.config.output_dir
        )
        logger.info(f"Evaluation rewritten samples completed")

        # Categorize by score into three datasets
        solved, learnable, unsolved = rewritten_dataset.categorize_by_score(final_evaluation['scores'])
        logger.info(f"Categorization: {len(solved)} solved, {len(learnable)} learnable, {len(unsolved)} unsolved")

        # Translate to target language if needed
        if self.translator is not None:
            target_language = self.config.translation_config.language

            solved = self.translator.translate_dataset(solved, target_lang=target_language)
            learnable = self.translator.translate_dataset(learnable, target_lang=target_language)
            unsolved = self.translator.translate_dataset(unsolved, target_lang=target_language)
            logger.info(f"Translation to {target_language} completed")

        # Save all three datasets to separate files
        final_save_path = save_path if save_path else os.path.join(
            self.config.output_dir,
            f"{self.config.task_config.name}_final.{export_format if export_format else self.config.export_format}"
        )
        final_export_format = export_format if export_format else self.config.export_format

        rewritten_dataset.save_categorized(solved, learnable, unsolved, final_save_path, final_export_format)
        logger.info(f"Datasets saved to {final_save_path}")

        # Cleanup: Release models and free GPU memory
        self._cleanup_models()
        logger.info("Model cleanup completed")

        # Clear buffer directories
        self._clear_buffers()
        logger.info("Buffer cleanup completed")

    def _cleanup_models(self):
        """Release vLLM, SentenceTransformer, and Translation models, free GPU memory after evaluation completes."""
        try:
            # Cleanup vLLM base model
            if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'model'):
                base_language_model = self.base_model.model.model
                if hasattr(base_language_model, 'cleanup'):
                    base_language_model.cleanup()

            # Cleanup semantic comparison model (only if using semantic method)
            if (hasattr(self.evaluator, 'answer_comparer') and
                isinstance(self.evaluator.answer_comparer, SemanticComparison) and
                hasattr(self.evaluator.answer_comparer, 'cleanup')):
                self.evaluator.answer_comparer.cleanup()

            # Cleanup semantic clustering voting model (only if using semantic_clustering method)
            if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'processor'):
                processor = self.llm.model.processor
                # Check if processor is MajorityVotingProcessor
                if hasattr(processor, 'voting'):
                    from .models.postprocess.majority_voting import SemanticClusteringVoting
                    if isinstance(processor.voting, SemanticClusteringVoting) and hasattr(processor.voting, 'cleanup'):
                        processor.voting.cleanup()

            # Cleanup translation model (only if translator was initialized)
            if self.translator is not None and hasattr(self.translator, 'cleanup'):
                self.translator.cleanup()

        except Exception as e:
            print(f"Warning: Failed to cleanup models: {e}")

    def _clear_buffers(self):
        """Clear all buffer directories after successful pipeline completion."""
        buffer_dir = "buffer"
        if not os.path.exists(buffer_dir):
            return

        buffer_patterns = [
            # Text modality buffers
            "Text-Local-Generation", "Text-Local-Validation", "Text-Local-Keywords",
            "Text-Distillation-Generation", "Text-Distillation-Validation",
            "Text-rewrite-generation", "Text-rewrite-validation",
            # Image modality buffers
            "Image-Local-Generation", "Image-Local-Validation",
            "Image-rewrite-generation", "Image-rewrite-validation",
        ]

        for pattern in buffer_patterns:
            subdir = os.path.join(buffer_dir, pattern)
            if os.path.exists(subdir):
                shutil.rmtree(subdir)

        # Remove buffer dir if empty
        if os.path.exists(buffer_dir) and not os.listdir(buffer_dir):
            os.rmdir(buffer_dir)
