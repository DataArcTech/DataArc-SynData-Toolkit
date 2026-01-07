from ..configs.config import SDGSTaskConfig
from ..models import ModelClient
from ..dataset.dataset import Dataset
from .base import BaseTaskExecutor
from .text import LocalTaskExecutor, WebTaskExecutor, DistillTaskExecutor
from .image import ImageLocalTaskExecutor, ImageWebTaskExecutor


class TaskExecutor:
    def __init__(self,
        config: SDGSTaskConfig,
        llm: ModelClient
    ) -> None:
        self.config = config
        self.executor: BaseTaskExecutor = TaskExecutor.get_executor(llm, self.config)

    @staticmethod
    def get_executor(llm: ModelClient, config: SDGSTaskConfig) -> BaseTaskExecutor:
        """
        Factory method to create appropriate executor based on config.

        Modality structure:
        - text.local -> LocalTaskExecutor
        - text.web -> WebTaskExecutor
        - text.distill -> DistillTaskExecutor
        - image.local -> ImageLocalTaskExecutor
        - image.web -> ImageWebTaskExecutor
        """
        executor: BaseTaskExecutor = None

        # Check for text modality
        if config.text is not None:
            if config.text.local is not None:
                executor = LocalTaskExecutor(config.text.local, llm)
            elif config.text.web is not None:
                executor = WebTaskExecutor(config.text.web, llm)
            elif config.text.distill is not None:
                executor = DistillTaskExecutor(config.text.distill, llm)
            else:
                raise Exception("text modality configured but no source specified (local/web/distill)")
        # Check for image modality
        elif config.image is not None:
            if config.image.local is not None:
                executor = ImageLocalTaskExecutor(config.image.local, llm)
            elif config.image.web is not None:
                executor = ImageWebTaskExecutor(config.image.web, llm)
            else:
                raise Exception("image modality configured but no source specified (local/web)")
        else:
            raise Exception("No modality configured. Please specify 'text' or 'image' in your config.")

        return executor

    def execute(self, parallel_executor=None, reporter=None) -> Dataset:
        dataset = Dataset()

        # Check which source is being used
        if self.config.text is not None:
            if self.config.text.web is not None:
                # WebTaskExecutor doesn't use parallel_executor
                dataset.extend(self.executor.execute(reporter=reporter))
            else:
                # LocalTaskExecutor and DistillTaskExecutor use parallel_executor
                dataset.extend(self.executor.execute(parallel_executor=parallel_executor, reporter=reporter))
        elif self.config.image is not None:
            if self.config.image.web is not None:
                # ImageWebTaskExecutor doesn't use parallel_executor
                dataset.extend(self.executor.execute(reporter=reporter))
            else:
                # ImageLocalTaskExecutor uses parallel_executor
                dataset.extend(self.executor.execute(parallel_executor=parallel_executor, reporter=reporter))
        else:
            dataset.extend(self.executor.execute(parallel_executor=parallel_executor, reporter=reporter))

        return dataset
