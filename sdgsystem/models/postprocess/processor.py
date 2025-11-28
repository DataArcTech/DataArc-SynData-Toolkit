from typing import List, Tuple, Union
from pathlib import Path

from ...configs.config import (
    PostProcessConfig, 
    BasePostProcessConfig
)
from ..answer_extraction import AnswerExtractor
from ..models import BaseLanguageModel
from ..usage_counter import ModelUsageCounter
from ..processor_arguments import ProcessorArgs
from .base import BasePostProcessor, NonePostProcessor
from .majority_voting import MajorityVotingConfig, MajorityVotingProcessor




class PostProcessor:
    def __init__(self, 
        model: BaseLanguageModel, 
        config: PostProcessConfig
    ) -> None:
        self.config = config
        self.model = model
        self.processor = PostProcessor.get_postprocessor_from_config(model, config)

    def get_model(self) -> BaseLanguageModel:
        return self.model

    @staticmethod
    def get_specific_postprocessor(processor: BasePostProcessor, config: BasePostProcessConfig) -> BasePostProcessor:
        if config is None:
            return NonePostProcessor(processor)

        if isinstance(config, MajorityVotingConfig):
            return MajorityVotingProcessor(processor, config)

        raise Exception(f"Postprocessor of {config.method} is not supported.")

    @staticmethod
    def get_postprocessor_from_config(model: BaseLanguageModel, config: PostProcessConfig) -> BasePostProcessor:
        # initial model
        processor = model

        if config is None or config.configs is None or len(config.methods) == 0:
            return NonePostProcessor(processor)

        # get processor
        configs: List[BasePostProcessConfig] = []
        for m in config.methods:
            if m in config.configs:
                configs.append(config.configs[m])
        configs.reverse()
        for postprocess_config in configs:
            processor = PostProcessor.get_specific_postprocessor(processor, postprocess_config)
        
        return processor
    
    def generate(self, 
        prompts: Union[str, List[str]], 
        n: int = 1, 
        answer_extractor: AnswerExtractor = None, 
        processor_args: ProcessorArgs = ProcessorArgs(), 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        return self.processor.generate(prompts, n, answer_extractor, processor_args, usage_counter, **kwargs)

    def multimodal_generate(self, 
        prompts: Union[str, List[str]], 
        image_paths: Union[Path, List[Path]],
        n: int = 1, 
        answer_extractor: AnswerExtractor = None, 
        processor_args: ProcessorArgs = ProcessorArgs(), 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        return self.processor.multimodal_generate(prompts, image_paths, n, answer_extractor, processor_args, usage_counter, **kwargs)