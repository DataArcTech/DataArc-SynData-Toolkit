import torch
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

from ..configs.config import TranslationConfig
from ..dataset.dataset import Dataset

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    """Base class for translators - subclass this to support different translation models."""

    def __init__(self, config: TranslationConfig):
        self.config = config

    @abstractmethod
    def _load_model(self):
        """Load the translation model (lazy loading)."""
        pass

    @abstractmethod
    def translate_text(self, text: str, source_lang: str = "English", target_lang: str = "Arabic") -> str:
        """Translate a single text from source language to target language."""
        pass

    @abstractmethod
    def cleanup(self):
        """Release model and free resources."""
        pass

    def translate_dataset(self, dataset: Dataset, target_lang: str = "Arabic") -> Dataset:
        """
        Translate all samples in a dataset to the target language.

        Args:
            dataset: Dataset to translate
            target_lang: Target language (default: "Arabic")

        Returns:
            New Dataset with translated samples
        """
        logger.info(f"Translating dataset to {target_lang}...")

        translated_dataset = Dataset()
        total_samples = len(dataset.samples)

        for idx, sample in enumerate(dataset.samples):
            translated_sample = self._translate_sample(sample, target_lang)
            translated_dataset.add_sample(translated_sample)
            print(f"    Translated {idx+1}/{total_samples}", end='\r')

        logger.info(f"Translation completed: {total_samples} samples translated")
        return translated_dataset

    def _translate_sample(self, sample: Dict[str, Any], target_lang: str) -> Dict[str, Any]:
        """
        Translate a single sample.

        Translates the 'input' and 'output' fields of the sample.
        """
        translated_sample = sample.copy()

        if "input" in sample and sample["input"]:
            translated_sample["input"] = self.translate_text(sample["input"], target_lang=target_lang)

        if "output" in sample and sample["output"]:
            translated_sample["output"] = self.translate_text(sample["output"], target_lang=target_lang)

        return translated_sample


class ArabicTranslator(BaseTranslator):
    """Translator using Hala-style causal LM models with chat templates."""

    def __init__(self, config: TranslationConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.pipe = None

        if config.language.lower() != "english" and not config.model_path:
            raise ValueError(
                f"Translation to '{config.language}' requires a model_path to be specified."
            )

    def _load_model(self):
        """Load translation model on first use (lazy loading)."""
        if self.pipe is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        logger.info(f"Loading translation model from {self.config.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        logger.info(f"Translation model loaded successfully")

    def translate_text(self, text: str, source_lang: str = "English", target_lang: str = "Arabic") -> str:
        self._load_model()

        messages = [
            {"role": "user", "content": f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}"}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=self.config.max_tokens,
            do_sample=False,
            return_full_text=False
        )

        return outputs[0]["generated_text"].strip()

    def cleanup(self):
        """Release model and free GPU memory."""
        if self.model is not None:
            logger.info("Cleaning up translation model...")
            del self.model
            del self.tokenizer
            del self.pipe
            self.model = None
            self.tokenizer = None
            self.pipe = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("Translation model cleanup completed")
