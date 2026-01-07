from typing import List, Union

from ..configs.config import ModelConfig, AnswerExtractionConfig, PostProcessConfig
from .answer_extraction import AnswerExtractor
from .models import BaseLanguageModel
from .postprocess import PostProcessor
from .processor_arguments import ProcessorArgs
from .usage_counter import ModelUsageCounter


class ModelClient:
    def __init__(self, 
        model_config: ModelConfig, 
        answer_config: AnswerExtractionConfig = None, 
        postprocess_config: PostProcessConfig = None
    ) -> None:
        self.model_config = model_config  
        self.model = PostProcessor(
            model=BaseLanguageModel.get_model_from_config(self.model_config), 
            config=postprocess_config
        )
        self.answer_extractor = AnswerExtractor(answer_config)
        self.token_usage: int = 0

    def generate(self,
        prompts: Union[str, List[str]],
        n: int = 1,
        processor_args: ProcessorArgs = ProcessorArgs(),
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Generate responses from LLM model.

        Args:
            prompts: Single prompt string or list of prompts
            n: Number of responses to generate per prompt
            processor_args: Arguments for post-processing (e.g., majority voting)
            usage_counter: Instance to count and estimate token/time usage
            **kwargs: Additional arguments for LLM inference

        Returns:
            Response string(s) from LLM:
            - Single prompt, n=1: str
            - Batch prompts, n=1: List[str]
            - Any prompts, n>1: List[List[str]]
        """
        if processor_args.answer_extraction is None or not processor_args.answer_extraction.enable:
            answer_extractor = None
        else:
            prompts = self.answer_extractor.format_prompts(prompts)
            answer_extractor = self.answer_extractor

        responses = self.model.generate(prompts, n, answer_extractor, processor_args, usage_counter, **kwargs)
        return responses

    def generate_with_images(self,
        prompts: Union[str, List[str]],
        images: Union[str, List[str]],
        n: int = 1,
        processor_args: ProcessorArgs = ProcessorArgs(),
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Generate responses from VLM model with image inputs.

        Args:
            prompts: Single prompt string or list of prompts
            images: Single image path or list of image paths (one per prompt)
            n: Number of responses to generate per prompt
            processor_args: Arguments for post-processing (e.g., majority voting)
            usage_counter: Instance to count and estimate token/time usage
            **kwargs: Additional arguments for LLM inference

        Returns:
            Response string(s) from VLM:
            - Single prompt, n=1: str
            - Batch prompts, n=1: List[str]
            - Any prompts, n>1: List[List[str]]
        """
        if processor_args.answer_extraction is None or not processor_args.answer_extraction.enable:
            answer_extractor = None
        else:
            prompts = self.answer_extractor.format_prompts(prompts)
            answer_extractor = self.answer_extractor

        responses = self.model.generate_with_images(prompts, images, n, answer_extractor, processor_args, usage_counter, **kwargs)
        return responses

    def get_model(self) -> BaseLanguageModel:
        return self.model.get_model()