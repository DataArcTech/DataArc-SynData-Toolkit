from pathlib import Path

from .answer_extraction import *
from .models import *
from .postprocess import *
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
        get response from model
        Output:
            responses: str | List[str], tokens_used: int
        """
        if processor_args.answer_extraction is None or not processor_args.answer_extraction.enable:
            answer_extractor = None
        else:
            prompts = self.answer_extractor.format_prompts(prompts)
            answer_extractor = self.answer_extractor

        responses = self.model.generate(prompts, n, answer_extractor, processor_args, usage_counter, **kwargs)
        return responses

    def multimodal_generate(self, 
        prompts: Union[str, List[str]], 
        image_paths: Union[Path, List[Path]],
        n: int = 1, 
        processor_args: ProcessorArgs = ProcessorArgs(), 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        get response from model
        Output:
            responses: str | List[str], tokens_used: int
        """
        if processor_args.answer_extraction is None or not processor_args.answer_extraction.enable:
            answer_extractor = None
        else:
            prompts = self.answer_extractor.format_prompts(prompts)
            answer_extractor = self.answer_extractor

        responses = self.model.multimodal_generate(prompts, image_paths, n, answer_extractor, processor_args, usage_counter, **kwargs)
        return responses

    def get_model(self) -> BaseLanguageModel:
        return self.model.get_model()
        
    def report_token_usage(self, done, total):
        used = self.token_usage

        if done > 0:
            estimated_total = used / done * total
        else:
            estimated_total = 0

        logging.info(
            f"\n[Token Usage] Processed {done}/{total} samples | "
            f"Current token usage = {used} | "
            f"Estimated total token = {estimated_total:.2f}"
        )