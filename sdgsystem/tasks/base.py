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


class BaseSynthesisTaskExecutor(ABC):
    """Base class for synthesis executor with fixed steps."""

    def __init__(self, 
        config: BaseTaskConfig, 
        llm: ModelClient, 
        parallel_executor = None, 
        reporter = None, 
    ) -> None:
        self.config = config
        self.llm = llm

        # executor for processing data parallelly. None means sequential processing
        self.parallel_executor = parallel_executor
        # reporter that report messages in code running to backend logs
        self.reporter = reporter

    @abstractmethod
    def task_parsing(self) -> dict:
        """
        Parsing configurations in self.config to the information required in synthesis, like task instruction, domain, parameters for model, ...

        Input: self.config
        Output: parsed_information in dictionary: {
            "task_instruction": ..., 
            "domain": ..., 
            ...
        }
        """
        pass

    @abstractmethod
    def prepare(self): 
        """
        Prepare the sources where data will be acquired. (local corpora / web dataset / teacher model)
        
        Input: self.config
        Output: set the source as the class member, e.g., self.sources / self.corpora / self.teacher_model / ...
        """
        pass

    @abstractmethod
    def construct_constraints(self, parsed_information: dict) -> dict:
        """
        construct constraints for synthesis from the parsed_information.
        constraints like reference passages / web dataset fields / generation patterns / ...

        Input: 
            self.config
            parsed_information: from self.task_parsing()
        Output: constraints in dictionary
        """
        pass

    @abstractmethod
    def data_acquisition(self, parsed_informatino: dict, constraints: dict) -> Dataset:
        """
        Synthesizing / Filtering data from sources.

        Input: 
            self.config
            self.sources: from self.prepare()
            parsed_information: from self.task_parsing()
            constraints: from self.construct_constraints()

        Output:
            raw synthetic dataset
        """
        pass

    @abstractmethod
    def structure_process(self, dataset: Dataset) -> Dataset:
        """
        Processing the raw synthetic data into unified formats.
        
        Input: raw synthetic dataset
        Output: structured dataset, each in format: {
            "input": ..., 
            "output": ..., 
            "image" / "audio" / ...: ...    # multimodality
            ...
        }
        """
        pass

    def execute(self) -> Dataset:
        # Step (1): parsing configurations
        parsed_information = self.task_parsing()
        
        # Step (2): prepare data sources
        self.prepare()

        # Step (3): get constraints for synthesis
        constraints = self.construct_constraints(parsed_information)

        # Step (4): synthesizing / filtering data from sources
        raw_data = self.data_acquisition(
            parsed_information, 
            constraints
        )

        # Step (5): processing data into a unified structure
        syn_data = self.structure_process(raw_data)

        return syn_data