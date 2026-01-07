from abc import ABC, abstractmethod
from typing import List, Union

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

    def _generate_with_images(self,
        prompts: Union[str, List[str]],
        images: Union[str, List[str]],
        n: int = 1,
        answer_extractor: AnswerExtractor = None,
        processor_args: ProcessorArgs = ProcessorArgs(),
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Get processed responses from VLM with image inputs.
        Args:
            prompts: the prompts
            images: image paths (one per prompt)
            n: repeated num for each prompt
            answer_extractor: instance to extract answer from response according to specific format
            processor_args: additional arguments for postprocessor
            usage_counter: instance to count and estimate token and time usage of models
        """
        if self._processor_is_model:
            return self.processor.generate_with_images(prompts, images, n, answer_extractor, usage_counter, **kwargs)
        else:
            self.processor: BasePostProcessor
            return self.processor.generate_with_images(prompts, images, n, answer_extractor, processor_args, usage_counter, **kwargs)

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

    def generate_with_images(self,
        prompts: Union[str, List[str]],
        images: Union[str, List[str]],
        n: int = 1,
        answer_extractor: AnswerExtractor = None,
        processor_args: ProcessorArgs = ProcessorArgs(),
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        return self._generate_with_images(prompts, images, n, answer_extractor, processor_args, usage_counter, **kwargs)


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

    def generate_with_images(self,
        prompts: Union[str, List[str]],
        images: Union[str, List[str]],
        n: int = 1,
        answer_extractor: AnswerExtractor = None,
        processor_args: ProcessorArgs = ProcessorArgs(),
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        return self._generate_with_images(prompts, images, n, answer_extractor, processor_args, usage_counter, **kwargs)