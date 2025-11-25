import os
import time
from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from vllm import LLM, SamplingParams
import openai
from openai.types.chat.chat_completion import ChatCompletion
import logging

from ..configs.config import *
from ..configs.constants import *
from .answer_extraction import AnswerExtractor
from .usage_counter import ModelUsageCounter


logger = logging.getLogger(__name__)


class BaseLanguageModel(ABC):
    def __init__(self, config) -> None:
        self.config = config

    @staticmethod
    def get_model_from_config(config: ModelConfig) -> "BaseLanguageModel":
        if config.provider == "local":
            return LocalModel(config.config)
        else:
            return APIModel(config.config)

    @abstractmethod
    def _get_responses(self, 
        prompts: Union[str, List[str]], 
        n: int=1, 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        get response from model
        Args: 
            prompt: str, the prompt
            kwargs: Arguments for LLM inference
            - temperature: float
            - top_p: float
            - ...

        Returns:
            Tuple of (response, token_usage)
        """
        pass

    def generate(self, 
        prompts: Union[str, List[str]], 
        n: int = 1, 
        answer_extractor: AnswerExtractor = None, 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        get extracter answers from model
        Args: 
            prompt: str, the prompt
            answer_extractor: instance to extractor answer from models' responses
            usage_counter: instance to count and estimate the token and time usage of model
            kwargs: Arguments for LLM inference
            - temperature: float
            - top_p: float
            - ...

        Returns:
            Tresponse
        """
        responses = self._get_responses(prompts, n, usage_counter, **kwargs)
        if answer_extractor is not None:
            responses = answer_extractor.extract_answers(responses)
        return responses


class APIModel(BaseLanguageModel):
    def __init__(self, config: APIModelConfig) -> None:
        super(APIModel, self).__init__(config)
        self.provider = config.provider
        self.model_name = config.model

        self.api_key = config.api_key
        self.base_url = config.base_url

        self.max_retry = config.max_retry_attempts
        self.retry_delay = config.retry_delay

        self.model = APIModel.get_model_by_provider(self.provider, self.model_name, self.api_key, self.base_url)

    @staticmethod
    def get_model_by_provider(provider: str, model_name: str, api_key: str = None, base_url: str = None) -> openai.OpenAI:
        try:
            if provider == "openai":
                api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
                base_url = base_url if base_url else os.getenv("OPENAI_BASE_URL")
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                return client

            if provider == "ollama":
                api_key = api_key if api_key else "ollama"
                base_url = base_url if base_url else "http://localhost:11434/v1"
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                return client

        except Exception as e:
            raise Exception(f"Unknown provider error for {provider} / {model_name}: {e}")

    def _get_responses(self, 
        prompts: Union[str, List[str]], 
        n: int=1, 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        temperature: float = kwargs.pop("temperature", 0.0)
        max_tokens: int = kwargs.pop("max_tokens", 4000)

        is_single_prompt = isinstance(prompts, str)
        prompts = [prompts] if is_single_prompt else prompts
        final_responses = []
        for prompt in prompts:
            retry_count = 0
            responses: List[str] = None

            st = time.time()
            token_used: int = 0

            while responses is None and retry_count < self.max_retry:
                retry_count += 1
                try:
                    completion: ChatCompletion = self.model.chat.completions.create(
                        model = self.model_name, 
                        messages = [{"role": ROLE_USER, "content": prompt}], 
                        temperature=temperature, 
                        max_tokens=max_tokens, 
                        n=n, 
                        **kwargs
                    )
                    token_used += completion.usage.total_tokens
                    responses = [choice.message.content.strip() for choice in completion.choices]
                except Exception as e:
                    logger.warning(f"API call failed (attempt {retry_count}/{self.max_retry}): {e}")
                    time.sleep(self.retry_delay)
            
            if usage_counter:
                usage_counter.add_usage(token_used, time.time() - st)

            # Check if responses is still None after all retries
            if responses is None:
                raise RuntimeError(f"Failed to get response from API after {self.max_retry} retries")

            if n == 1:
                final_responses.append(responses[0])    # List[str]
            else:
                final_responses.append(responses)       # List[List[str]]

        final_responses = final_responses[0] if is_single_prompt else final_responses
        return final_responses


class LocalModel(BaseLanguageModel):
    """
    Base model wrapper using vLLM for efficient inference.

    This class wraps the vLLM library to provide efficient batch inference
    for the working model (student model). It supports both deterministic
    inference and sampling for pass@n evaluation.
    """

    def __init__(self, config: LocalModelConfig):
        """
        Initialize the base model configuration (lazy loading).

        Args:
            config: LocalModelConfig instance containing model path,
                   max_model_len, gpu_memory_utilization, and device
        """
        super(LocalModel, self).__init__(config)

        self.model_path = config.path
        self.device = config.device
        self.max_model_len = config.max_model_len
        self.gpu_memory_utilization = config.gpu_memory_utilization

        self.inference_config = config.inference
        self.scoring_config = config.scoring

        # Lazy loading: model is loaded on first use
        self.llm = None

    def _load_model(self):
        """
        Load vLLM model on first use (lazy loading).

        This method is called automatically when the model is first needed.
        Uses the GPU specified in self.device.
        """
        if self.llm is not None:
            return  # Already loaded

        # Set CUDA_VISIBLE_DEVICES before loading vLLM model
        if self.device.startswith("cuda:"):
            gpu_id = self.device.split(":")[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            logger.info(f"[vLLM] Setting CUDA_VISIBLE_DEVICES={gpu_id}")

        # Log before loading
        logger.info(f"Loading vLLM model from {self.model_path} on device {self.device}")

        # Load model with vLLM
        self.llm = LLM(
            model=self.model_path,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=1,  # Single GPU
            trust_remote_code=True
        )

        logger.info(f"vLLM model loaded successfully!")
        logger.info(f"  Max length: {self.max_model_len}")
        logger.info(f"  GPU utilization: {self.gpu_memory_utilization}")

    def cleanup(self):
        """
        Manually release vLLM model and free GPU memory.

        Call this method when you're done with the model to release GPU resources.
        Useful for Gradio apps or when running multiple pipelines sequentially.
        """
        if self.llm is not None:
            logger.info(f"Releasing vLLM model and freeing GPU memory...")

            # Delete the vLLM instance
            del self.llm
            self.llm = None

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CUDA cache
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info(f"GPU memory released successfully!")
            except ImportError:
                logger.info(f"vLLM model released (torch not available for cache clearing)")

    def _get_responses(self, 
        prompts: Union[str, List[str]], 
        n: int=1, 
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        temperature: float = kwargs.pop("temperature", self.inference_config.temperature)
        max_tokens: int = kwargs.pop("max_tokens", self.inference_config.max_tokens)
        top_p: float = kwargs.pop("top_p", self.inference_config.top_p)
        
        response = self.generate_with_prompts(prompts, usage_counter, temperature, max_tokens, top_p, n)
        
        return response

    def generate_with_prompts(self,
        prompts: Union[str, List[str]], 
        usage_counter: ModelUsageCounter = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        top_p: float = 0.95,
        n: int = 1
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Generate responses for input prompts.

        This method handles three output formats:
        1. Single prompt, n=1: Returns string
        2. Batch prompts, n=1: Returns List[str]
        3. Batch prompts, n>1: Returns List[List[str]]

        Args:
            prompts: Single prompt string or list of prompts
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling threshold
            n: Number of samples per prompt

        Returns:
            Generated text(s) in appropriate format
        """
        # Lazy load model on first use
        self._load_model()

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n
        )

        # Ensure prompts is a list
        is_single_prompt = isinstance(prompts, str)
        prompt_list = [prompts] if is_single_prompt else prompts

        # Generate responses
        st = time.time()
        token_usage: int = 0
        responses = self.llm.generate(prompt_list, sampling_params)
        if usage_counter:
            usage_counter.add_usage(token_usage, time.time() - st)

        # Format output based on input type and n
        response = None
        if n == 1:
            # Single sample per prompt
            results = [resp.outputs[0].text for resp in responses]
        else:
            # Multiple samples per prompt
            results = [[output.text for output in resp.outputs] for resp in responses]
        
        response = results[0] if is_single_prompt else results

        return response

    def batch_inference(self,
        prompts: List[str], 
        usage_counter: ModelUsageCounter = None, 
        batch_size: int = 100,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        top_p: float = 0.95,
        n: int = 1
    ) -> List[Union[str, List[str]]]:
        """
        Perform batched inference on a large list of prompts.

        This method processes prompts in batches to avoid memory issues
        with very large datasets.

        Args:
            prompts: List of prompts to process
            batch_size: Number of prompts per batch
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling threshold
            n: Number of samples per prompt

        Returns:
            List of generated responses (format depends on n)
        """
        all_responses = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]

            logger.info(f"Processing batch {batch_idx + 1}/{num_batches} "
                    f"({len(batch_prompts)} prompts)...")

            batch_responses, _ = self.generate_with_prompts(
                batch_prompts, 
                usage_counter=usage_counter, 
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n
            )
            # estimate the token and time usage
            if usage_counter:
                usage_counter.estimate_usage(n=len(batch_prompts))

            all_responses.extend(batch_responses)

        return all_responses
