import logging

from ...configs.config import TextDistillConfig
from ...models import ModelClient
from ...dataset.dataset import Dataset
from ..base import BaseTaskExecutor
from ...distillation.sdg_distill import SDGDistillation
from ...documents.load import DocumentLoader

logger = logging.getLogger(__name__)


class DistillTaskExecutor(BaseTaskExecutor):
    def __init__(self,
        config: TextDistillConfig,
        llm: ModelClient
    ) -> None:
        super(DistillTaskExecutor, self).__init__(config, llm)
        self.config: TextDistillConfig

    def execute(self, parallel_executor=None, reporter=None) -> Dataset:
        """Execute distillation task: pure synthetic data generation without retrieval."""

        # Loading Demo Examples
        if reporter:
            reporter.start_step("loading", "Loading Demo Examples", "Loading demo examples...")

        demo_examples = []
        if self.config.demo_examples_path:
            loader = DocumentLoader(corpus_paths=[])
            demo_examples = loader.load_demo_examples(self.config.demo_examples_path)
        else:
            logger.info("No demo examples provided, using zero-shot generation")

        if reporter:
            reporter.complete_step({
                "message": f"Loaded {len(demo_examples)} demo examples" if demo_examples else "Using zero-shot generation",
                "demo_count": len(demo_examples)
            })

        # Initializing Distillation Generator
        if reporter:
            reporter.start_step("initialization", "Initializing Generator", "Starting distillation...")

        distillation = SDGDistillation(
            model=self.llm,
            config=self.config
        )
        logger.info("Distillation generator initialized")

        if reporter:
            reporter.complete_step({"message": "Distillation Generator initialized"})

        # Generating Synthetic Dataset
        samples = distillation.generate(
            demo_examples=demo_examples if demo_examples else None,
            parallel_executor=parallel_executor,
            reporter=reporter
        )
        logger.info(f"Generated {len(samples)} synthetic samples")

        # Create and return dataset
        dataset = Dataset.from_list(samples)
        logger.info(f"Created dataset with {len(dataset)} samples")

        return dataset
