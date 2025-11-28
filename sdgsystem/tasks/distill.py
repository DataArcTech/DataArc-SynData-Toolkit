import logging

from .base import *
from ..distillation.sdg_distill import SDGDistillation
from ..distillation.sdg_mm_distill import SDGMMDistillation
from ..documents.load import DocumentLoader

logger = logging.getLogger(__name__)


class DistillTaskExecutor(BaseTaskExecutor):
    def __init__(self, 
        config: DistillTaskConfig, 
        llm: ModelClient
    ) -> None:
        super(DistillTaskExecutor, self).__init__(config, llm)

    def execute(self, parallel_executor=None) -> Dataset:
        """Execute distillation task: pure synthetic data generation without retrieval."""

        logger.info("=== Step: Loading Demo Examples (if provided) ===")
        demo_examples = []
        if self.config.demo_examples_path:
            loader = DocumentLoader(corpus_paths=[])
            demo_examples = loader.load_demo_examples(self.config.demo_examples_path)
            logger.info(f"Loaded {len(demo_examples)} demo examples")
        else:
            logger.info("No demo examples provided, using zero-shot generation")

        logger.info("=== Step: Initializing multi-modal data generation (if provided) ===")
        image_paths = []
        if self.config.image_dir:
            loader = DocumentLoader(corpus_paths=[])
            image_paths = loader.load_image_paths(self.config.image_dir)
            logger.info(f"Loaded {len(image_paths)} images for multi-modal generation")
        else:
            logger.info("No image directory provided, proceeding with text-only generation")

        if image_paths:
            logger.info("=== Step: Initializing Multi-Modal Distillation Generator ===")
            distillation = SDGMMDistillation(
                model=self.llm,
                config=self.config
            )
        else:
            logger.info("=== Step: Initializing Distillation Generator ===")
            distillation = SDGDistillation(
                model=self.llm,
                config=self.config
            )
        logger.info("Distillation generator initialized")

        logger.info("=== Step: Generating Synthetic Dataset ===")
        if image_paths:
            samples = distillation.generate(
                image_paths=image_paths,
                demo_examples=demo_examples if demo_examples else None,
                parallel_executor=parallel_executor
            )
        else:
            samples = distillation.generate(
                demo_examples=demo_examples if demo_examples else None,
                parallel_executor=parallel_executor
            )
        logger.info(f"Generated {len(samples)} synthetic samples")

        logger.info("=== Step: Creating Dataset ===")
        dataset = Dataset.from_list(samples)
        logger.info(f"Created dataset with {len(dataset)} samples")

        return dataset