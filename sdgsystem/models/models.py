import os
import time
from abc import ABC, abstractmethod
from typing import List, Union
import base64
from PIL import Image
import openai
from openai.types.chat.chat_completion import ChatCompletion
import logging

from ..configs.config import ModelConfig, APIModelConfig, LocalModelConfig
from ..configs.constants import ROLE_USER
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

    @abstractmethod
    def _get_responses_with_images(self,
        prompts: Union[str, List[str]],
        images: Union[str, List[str]],
        n: int = 1,
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Get response from VLM model with image inputs.

        Args:
            prompts: Single prompt string or list of prompts
            images: Single image path or list of image paths (one per prompt)
            n: Number of responses to generate per prompt
            usage_counter: Instance to count and estimate token/time usage
            **kwargs: Additional arguments for LLM inference
                - temperature: float
                - max_tokens: int

        Returns:
            Response string(s) from VLM:
            - Single prompt, n=1: str
            - Batch prompts, n=1: List[str]
            - Any prompts, n>1: List[List[str]]
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

    def generate_with_images(self,
        prompts: Union[str, List[str]],
        images: Union[str, List[str]],
        n: int = 1,
        answer_extractor: AnswerExtractor = None,
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Generate responses from VLM model with image inputs.

        Args:
            prompts: Single prompt string or list of prompts
            images: Single image path or list of image paths (one per prompt)
            n: Number of responses to generate per prompt
            answer_extractor: Instance to extract answers from model responses
            usage_counter: Instance to count and estimate token/time usage
            **kwargs: Additional arguments for LLM inference

        Returns:
            Response string(s) from VLM:
            - Single prompt, n=1: str
            - Batch prompts, n=1: List[str]
            - Any prompts, n>1: List[List[str]]
        """
        responses = self._get_responses_with_images(prompts, images, n, usage_counter, **kwargs)
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

        self.client = APIModel.get_model_by_provider(self.provider, self.api_key, self.base_url)

    @staticmethod
    def get_model_by_provider(provider: str, api_key: str = None, base_url: str = None):
        """
        Get client for various providers.

        Supported providers: openai, anthropic, deepseek, qwen, ollama

        Environment variables:
        - API_KEY: API key for the provider
        - BASE_URL: Base URL for the API endpoint

        Note: If base_url is specified, OpenAI SDK is used (assumes OpenAI-compatible endpoint).
        """
        # Get API key from config or environment
        api_key = api_key or os.getenv("API_KEY")

        # Get base URL from config or environment
        base_url = base_url or os.getenv("BASE_URL")

        # Anthropic uses its own SDK (only when base_url is not specified)
        if provider == "anthropic" and not base_url:
            import anthropic
            try:
                client = anthropic.Anthropic(api_key=api_key)
                return client
            except Exception as e:
                raise Exception(f"Failed to create Anthropic client: {e}")

        # Provider-specific default base URLs
        default_base_urls = {
            "openai": "https://api.openai.com/v1",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "ollama": "http://localhost:11434/v1",
        }

        if not base_url and provider in default_base_urls:
            base_url = default_base_urls[provider]

        # Ollama doesn't require API key
        if provider == "ollama" and not api_key:
            api_key = "ollama"

        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            return client
        except Exception as e:
            raise Exception(f"Failed to create client for provider '{provider}': {e}")

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
                    if self.provider == "anthropic" and not self.base_url:
                        # Native Anthropic API (only when no custom base_url)
                        message = self.client.messages.create(
                            model=self.model_name,
                            messages=[{"role": ROLE_USER, "content": prompt}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            **kwargs
                        )
                        token_used += message.usage.input_tokens + message.usage.output_tokens
                        responses = [message.content[0].text.strip()]
                    else:
                        # OpenAI-compatible API (includes custom base_url for any provider)
                        completion: ChatCompletion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": ROLE_USER, "content": prompt}],
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

    def _get_responses_with_images(self,
        prompts: Union[str, List[str]],
        images: Union[str, List[str]],
        n: int = 1,
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Get responses from VLM API with image inputs.

        Supports OpenAI-compatible vision APIs (GPT-4V, Qwen-VL, etc.) and Anthropic Claude.

        Args:
            prompts: Single prompt string or list of prompts
            images: Single image path or list of image paths (one per prompt)
            n: Number of responses to generate per prompt
            usage_counter: Instance to count and estimate token/time usage
            **kwargs: Additional arguments for LLM inference

        Returns:
            Response string(s) from VLM:
            - Single prompt, n=1: str
            - Batch prompts, n=1: List[str]
            - Any prompts, n>1: List[List[str]]
        """
        temperature: float = kwargs.pop("temperature", 0.0)
        max_tokens: int = kwargs.pop("max_tokens", 4000)

        is_single_prompt = isinstance(prompts, str)
        prompts = [prompts] if is_single_prompt else prompts
        images = [images] if isinstance(images, str) else images

        if len(prompts) != len(images):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

        final_responses = []

        for prompt, image_path in zip(prompts, images):
            retry_count = 0
            responses: List[str] = None

            st = time.time()
            token_used: int = 0

            # Encode image to base64
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            # Determine media type from extension
            ext = image_path.lower().split('.')[-1]
            media_types = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "gif": "image/gif",
                "bmp": "image/bmp", "webp": "image/webp",
                "tiff": "image/tiff", "tif": "image/tiff"
            }
            media_type = media_types.get(ext, "image/png")

            while responses is None and retry_count < self.max_retry:
                retry_count += 1
                try:
                    if self.provider == "anthropic" and not self.base_url:
                        # Native Anthropic API with vision
                        anthropic_content = [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                        # Anthropic doesn't support n parameter, so we need to call multiple times
                        responses = []
                        for _ in range(n):
                            message = self.client.messages.create(
                                model=self.model_name,
                                messages=[{"role": ROLE_USER, "content": anthropic_content}],
                                temperature=temperature,
                                max_tokens=max_tokens,
                                **kwargs
                            )
                            token_used += message.usage.input_tokens + message.usage.output_tokens
                            responses.append(message.content[0].text.strip())
                    else:
                        # OpenAI-compatible vision API
                        openai_content = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                        completion: ChatCompletion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": ROLE_USER, "content": openai_content}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            n=n,
                            **kwargs
                        )
                        token_used += completion.usage.total_tokens
                        responses = [choice.message.content.strip() for choice in completion.choices]
                except Exception as e:
                    logger.warning(f"VLM API call failed (attempt {retry_count}/{self.max_retry}): {e}")
                    time.sleep(self.retry_delay)

            if usage_counter:
                usage_counter.add_usage(token_used, time.time() - st)

            if responses is None:
                raise RuntimeError(f"Failed to get VLM response after {self.max_retry} retries")

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

        # Lazy loading: model is loaded on first use
        self.llm = None

    def _load_model(self):
        """
        Load vLLM model on first use (lazy loading).

        This method is called automatically when the model is first needed.
        Uses the GPU specified in self.device.
        """
        from vllm import LLM
        
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

    def _get_responses(self,
        prompts: Union[str, List[str]],
        n: int=1,
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        temperature: float = kwargs.pop("temperature", 0.0)
        max_tokens: int = kwargs.pop("max_tokens", 1500)
        top_p: float = kwargs.pop("top_p", 0.95)

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
        from vllm import SamplingParams

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

    def _get_responses_with_images(self,
        prompts: Union[str, List[str]],
        images: Union[str, List[str]],
        n: int = 1,
        usage_counter: ModelUsageCounter = None,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Get responses from local VLM model with image inputs.

        Uses vLLM with multimodal support (requires vLLM >= 0.4.0 for vision models).

        Args:
            prompts: Single prompt string or list of prompts
            images: Single image path or list of image paths (one per prompt)
            n: Number of responses to generate per prompt
            usage_counter: Instance to count and estimate token/time usage
            **kwargs: Additional arguments for LLM inference

        Returns:
            Response string(s) from VLM:
            - Single prompt, n=1: str
            - Batch prompts, n=1: List[str]
            - Any prompts, n>1: List[List[str]]
        """
        from vllm import SamplingParams

        temperature: float = kwargs.pop("temperature", 0.0)
        max_tokens: int = kwargs.pop("max_tokens", 4000)
        top_p: float = kwargs.pop("top_p", 0.95)

        # Lazy load model on first use
        self._load_model()

        is_single_prompt = isinstance(prompts, str)
        prompts = [prompts] if is_single_prompt else prompts
        images = [images] if isinstance(images, str) else images

        if len(prompts) != len(images):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n
        )

        # Build multimodal inputs for vLLM
        # Use chat template format to ensure proper image placeholder handling
        tokenizer = self.llm.get_tokenizer()
        multimodal_inputs = []

        for prompt, image_path in zip(prompts, images):
            image = Image.open(image_path)

            # Build chat messages with image and text content
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Apply chat template to get properly formatted prompt with image placeholder
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            multimodal_inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": image}
            })

        # Generate responses
        st = time.time()
        token_usage: int = 0
        responses = self.llm.generate(multimodal_inputs, sampling_params)

        if usage_counter:
            usage_counter.add_usage(token_usage, time.time() - st)

        # Format output based on input type and n
        if n == 1:
            # Single sample per prompt
            results = [resp.outputs[0].text for resp in responses]
        else:
            # Multiple samples per prompt
            results = [[output.text for output in resp.outputs] for resp in responses]

        return results[0] if is_single_prompt else results

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
