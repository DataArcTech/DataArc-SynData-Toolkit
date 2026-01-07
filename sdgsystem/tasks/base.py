from abc import ABC, abstractmethod

from ..models import ModelClient
from ..dataset.dataset import Dataset
from ..configs.config import BaseTaskConfig


class BaseTaskExecutor(ABC):
    """Base class for all task executors across modalities."""

    def __init__(self, config: BaseTaskConfig, llm: ModelClient) -> None:
        self.config = config
        self.llm = llm

    @abstractmethod
    def execute(self, reporter=None) -> Dataset:
        pass
