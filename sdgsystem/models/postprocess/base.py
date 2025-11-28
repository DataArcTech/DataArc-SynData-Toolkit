"""
Base Class of postprocessor of LLMs' responses
"""
from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from pathlib import Path

from ...configs.config import BasePostProcessConfig
from ..models import BaseLanguageModel
from ..processor_arguments import ProcessorArgs
from ..answer_extraction import AnswerExtractor
from ..usage_counter import ModelUsageCounter



class BasePostProcessor(ABC):
    def __init__(self, 
        processor: Union[BaseLanguageModel, "BasePostProcessor"], 
        config: BasePostProcessConfig
    ) -> None:
        self.config = config
        self.processor = processor
        self._processor_is_model = isinstance(self.processor, BaseLanguageModel)

    def get_model(self) -> BaseLanguageModel:
        if isinstance(self.processor, BaseLanguageModel):
            return self.processor
        else:
            return self.processor.get_model()

    def _generate(self, 
        prompts: Union[str, List[str]], 
        n: int = 1, 
        answer_extractor: AnswerExtractor = None, 
        processor_args: ProcessorArgs = ProcessorArgs(), 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Get processed responses from LLM.
        Args:
            prompts: the prompts
            n: repeated num for each prompt
            answer_extractor: instance to extract answer from response according to specific format
            processor_args: additional arguments for postprocessor
            usage_counter: instance to count and estimate token and time usage of models
        """
        if self._processor_is_model:
            return self.processor.generate(prompts, n, answer_extractor, usage_counter, **kwargs)
        else:
            self.processor: BasePostProcessor
            return self.processor.generate(prompts, n, answer_extractor, processor_args, usage_counter, **kwargs)

    def _multimodal_generate(self, 
        prompts: Union[str, List[str]],
        image_paths: Union[Path, List[Path]], 
        n: int = 1, 
        answer_extractor: AnswerExtractor = None, 
        processor_args: ProcessorArgs = ProcessorArgs(), 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Get processed responses from LLM with images.
        Args:
            prompts: the prompts
            image_paths: the image paths
            n: repeated num for each prompt
            answer_extractor: instance to extract answer from response according to specific format
            processor_args: additional arguments for postprocessor
            usage_counter: instance to count and estimate token and time usage of models
        """
        if self._processor_is_model:
            return self.processor.multimodal_generate(prompts, image_paths, n, answer_extractor, usage_counter, **kwargs)
        else:
            self.processor: BasePostProcessor
            return self.processor.multimodal_generate(prompts, image_paths, n, answer_extractor, processor_args, usage_counter, **kwargs)

    @abstractmethod
    def generate(self, 
        prompts: Union[str, List[str]], 
        n: int = 1, 
        answer_extractor: AnswerExtractor = None, 
        processor_args: ProcessorArgs = ProcessorArgs(), 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        return self._generate(prompts, n, answer_extractor, processor_args, usage_counter, **kwargs)
    
    @abstractmethod
    def multimodal_generate(self, 
        prompts: Union[str, List[str]],
        image_paths: Union[Path, List[Path]], 
        n: int = 1, 
        answer_extractor: AnswerExtractor = None, 
        processor_args: ProcessorArgs = ProcessorArgs(), 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        return self._multimodal_generate(prompts, image_paths, n, answer_extractor, processor_args, usage_counter, **kwargs)


class NonePostProcessor(BasePostProcessor):
    """
    Class for directly use original model without any postprocess method
    """
    def __init__(self, processor: Union[BaseLanguageModel, BasePostProcessor]) -> None:
        super(NonePostProcessor, self).__init__(processor, None)

    def generate(self, 
        prompts: Union[str, List[str]], 
        n: int = 1, 
        answer_extractor: AnswerExtractor = None, 
        processor_args: ProcessorArgs = ProcessorArgs(), 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        return self._generate(prompts, n, answer_extractor, processor_args, usage_counter, **kwargs)

    def multimodal_generate(self, 
        prompts: Union[str, List[str]],
        image_paths: Union[Path, List[Path]], 
        n: int = 1, 
        answer_extractor: AnswerExtractor = None, 
        processor_args: ProcessorArgs = ProcessorArgs(), 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        return self._multimodal_generate(prompts, image_paths, n, answer_extractor, processor_args, usage_counter, **kwargs)