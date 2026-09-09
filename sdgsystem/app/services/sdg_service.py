"""SDG Service wrapper for the pipeline with progress reporting."""
import os
import shutil
import logging
from typing import Dict, Any, Optional

from ..core.progress import SDGProgressReporter
from ..api.schemas import SDGJobOutput, SDGJobOutputCounts

from ...configs.config import SDGSConfig
from ...pipeline import Pipeline

logger = logging.getLogger(__name__)


class SDGService:
    """Wraps the SDG Pipeline with progress reporting for SSE."""

    def __init__(self, config: Dict[str, Any]):
        self.raw_config = config
        self.config = SDGSConfig.from_dict(config)
        self.pipeline: Optional[Pipeline] = None
        self._dataset = None

    def run_generation(self, reporter: SDGProgressReporter):
        """Run the generation phase with progress reporting."""
        task_type = reporter.task_type
        total_steps = self._get_generation_step_count(task_type)

        reporter.start_phase("generation", "Generating Dataset", total_steps)

        try:
            self.pipeline = Pipeline(self.config)

            # Execute task - task executor will call reporter methods
            self._dataset = self.pipeline.task_executor.execute(
                parallel_executor=self.pipeline.parallel_executor,
                reporter=reporter
            )

            # Save raw dataset
            raw_path = self._get_output_path("raw")
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            self._dataset.save(raw_path, self.config.export_format)

            # Clear generation buffers after successful completion
            self._clear_generation_buffer()

            # Complete generation phase - stops here, user decides to continue
            output = SDGJobOutput(
                raw_dataset=raw_path,
                counts=SDGJobOutputCounts(raw=len(self._dataset) if self._dataset else 0)
            )
            reporter.complete_phase(output)

        except Exception as e:
            # Cleanup models on generation failure since refinement won't run
            self._cleanup_models()
            reporter.fail("generation_error", str(e), {"type": type(e).__name__})
            raise

    def run_refinement(self, reporter: SDGProgressReporter):
        """Run the refinement phase with progress reporting."""
        # Determine if translation should be skipped based on config
        # Skip if language is 'english' or not configured
        translation_config = self.config.translation_config
        skip_translation = (
            translation_config is None or
            translation_config.language.lower() == "english"
        )
        total_steps = 5 if skip_translation else 6

        reporter.start_phase("refinement", "Evaluating & Rewriting Dataset", total_steps)

        try:
            # Initial evaluation
            reporter.start_step("initial_evaluation", "Evaluating Dataset", message="Loading base model...")

            # Trigger base model loading first
            base_model = self.pipeline.base_model.get_model()
            if hasattr(base_model, '_load_model'):
                base_model._load_model()

            reporter.update_step(message="Evaluating samples...")
            initial_evaluation = self.pipeline.evaluator.evaluate(self._dataset)
            reporter.complete_step()

            # Rewriting + Validating samples
            reporter.start_step("rewriting", "Rewriting Samples", message="Rewriting samples...")
            rewritten_dataset = self.pipeline.rewriter.rewrite(
                self._dataset,
                initial_evaluation,
                parallel_executor=self.pipeline.parallel_executor,
                reporter=reporter
            )

            # Scoring
            reporter.start_step("scoring", "Scoring Samples", message="Scoring rewritten samples...")
            final_evaluation = self.pipeline.evaluator.evaluate(rewritten_dataset, mode="scoring")
            reporter.complete_step()

            # Categorization
            reporter.start_step("categorization", "Categorizing Results", message="Categorizing by score...")
            solved, learnable, unsolved = rewritten_dataset.categorize_by_score(
                final_evaluation['scores']
            )
            reporter.complete_step({
                "solved": len(solved),
                "learnable": len(learnable),
                "unsolved": len(unsolved)
            })

            # Step 6: Translation (if not skipped)
            if not skip_translation and self.pipeline.translator:
                target_lang = self.config.translation_config.language
                reporter.start_step("translation", "Translating", message=f"Translating to {target_lang}...")
                solved = self.pipeline.translator.translate_dataset(solved, target_lang)
                learnable = self.pipeline.translator.translate_dataset(learnable, target_lang)
                unsolved = self.pipeline.translator.translate_dataset(unsolved, target_lang)
                reporter.complete_step()

            # Save datasets
            save_path = self._get_output_path("final")
            rewritten_dataset.save_categorized(
                solved, learnable, unsolved,
                save_path, self.config.export_format
            )

            output = SDGJobOutput(
                raw_dataset=self._get_output_path("raw"),
                solved=self._get_output_path("solved") if len(solved) > 0 else None,
                learnable=self._get_output_path("learnable") if len(learnable) > 0 else None,
                unsolved=self._get_output_path("unsolved") if len(unsolved) > 0 else None,
                counts=SDGJobOutputCounts(
                    raw=len(self._dataset) if self._dataset else 0,
                    solved=len(solved),
                    learnable=len(learnable),
                    unsolved=len(unsolved),
                )
            )

            # Clear refinement buffers after successful completion
            self._clear_refinement_buffer()

            # Cleanup models and free GPU memory
            self._cleanup_models()

            reporter.complete_phase(output)

        except Exception as e:
            # Cleanup models even on failure to free GPU memory
            self._cleanup_models()
            reporter.fail("refinement_error", str(e), {"type": type(e).__name__})
            raise

    def _get_generation_step_count(self, task_type: str) -> int:
        """Get total step count for generation phase based on task type.

        Local task steps:
        1. loading - Loading Documents
        2. parsing - Parsing Documents (conditional)
        3. keyword_extraction - Extracting Keywords
        4. passage_retrieval - Retrieving Passages
        5. pattern_generation - Generating Pattern
        6. sample_generation - Generating Samples
        7. validation - Validating Samples

        Distill task steps:
        1. loading - Loading Demo Examples
        2. initialization - Initializing Generator
        3. pattern_generation - Generating Patterns
        4. sample_generation - Generating Samples
        5. validation - Validating Samples
        """
        counts = {"local": 7, "web": 4, "distill": 5}
        return counts.get(task_type, 7)

    def _get_output_path(self, dataset_type: str) -> str:
        """Get output file path for a dataset type.

        Note: Categorized files (solved/learnable/unsolved) are saved by save_categorized()
        using the 'final' base path with category suffix appended.
        """
        if dataset_type in ("solved", "learnable", "unsolved"):
            # Categorized files: {name}_final_{category}.{format}
            return os.path.join(
                self.config.output_dir,
                f"{self.config.task_config.name}_final_{dataset_type}.{self.config.export_format}"
            )
        else:
            # Raw and other files: {name}_{type}.{format}
            return os.path.join(
                self.config.output_dir,
                f"{self.config.task_config.name}_{dataset_type}.{self.config.export_format}"
            )

    def _cleanup_models(self):
        """Release vLLM and other models, free GPU memory."""
        if self.pipeline is None:
            return
        try:
            self.pipeline._cleanup_models()
        except Exception as e:
            logger.warning(f"Failed to cleanup models: {e}")

    def _clear_generation_buffer(self):
        """Clear generation phase buffers after successful completion."""
        buffer_dir = "buffer"
        # Generation buffers: Local-Generation, Local-Validation, Distillation-Generation, etc.
        generation_patterns = [
            "Local-Generation", "Local-Validation", "Local-Pattern",
            "Distillation-Generation", "Distillation-Pattern",
            # Web task doesn't use buffers for generation
        ]
        self._clear_buffer_dirs(buffer_dir, generation_patterns)

    def _clear_refinement_buffer(self):
        """Clear refinement phase buffers after successful completion."""
        buffer_dir = "buffer"
        # Refinement buffers: rewrite-generation, rewrite-validation
        refinement_patterns = [
            "rewrite-generation", "rewrite-validation"
        ]
        self._clear_buffer_dirs(buffer_dir, refinement_patterns)

    def _clear_buffer_dirs(self, buffer_dir: str, patterns: list):
        """Clear specific buffer subdirectories matching the patterns."""
        if not os.path.exists(buffer_dir):
            return
        for pattern in patterns:
            subdir = os.path.join(buffer_dir, pattern)
            if os.path.exists(subdir):
                shutil.rmtree(subdir)
        # Remove buffer dir if empty
        if os.path.exists(buffer_dir) and not os.listdir(buffer_dir):
            os.rmdir(buffer_dir)
