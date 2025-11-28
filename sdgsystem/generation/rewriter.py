import os
from abc import ABC, abstractmethod
from typing import List, Dict, Union
from tqdm import tqdm

from ..configs.config import BaseRewriteConfig, DifficultyAdjustRewriteConfig
from ..dataset.dataset import Dataset
from ..models import ModelClient, ModelUsageCounter
from ..prompts import HARDER_SAMPLE_PROMPT, SIMPLER_SAMPLE_PROMPT
from ..parallel import ParallelExecutor
from ..buffer import TaskBuffer
from .base import BaseGenerator



class BaseRewriter(ABC, BaseGenerator):
    """
    Base class for rewrite synthetic training data
    """
    def __init__(self,
        model: ModelClient,
        config: BaseRewriteConfig, 
        buffer_dir: str = "buffer"
    ) -> None:
        """Initialize the Rewriter"""
        super(BaseRewriter, self).__init__(model, config, buffer_dir)
        self.config = config
    
    @abstractmethod
    def _rewrite_single_sample(self, 
        sample: Dict, 
        usage_counter: ModelUsageCounter = None
    ) -> Union[Dict, str]:
        """
        Rewrite the given sample.

        Args:
            sample: Original sample dict

        Returns:
            rewrite_sample
        """
        return {}

    @abstractmethod
    def _process_dataset_by_evaluations(self, dataset: Dataset, evaluations: Dict) -> List[Dict]:
        """
        transform samples in dataset to the samples which rewriting demands, according to the evaluations
        """
        return []

    @staticmethod
    def get_specific_rewriter(llm: ModelClient, config: BaseRewriteConfig) -> "BaseRewriter":
        if isinstance(config, DifficultyAdjustRewriteConfig):
            return DifficultyAdjustRewriter(llm, config)
        
        raise Exception(f"Rewriter with method {config.method} is not supported.")

    def rewrite(self, 
        dataset: Dataset, 
        evaluations: Dict, 
        parallel_executor: ParallelExecutor = None
    ) -> Dataset:
        """
        Rewrite dataset based on evaluation results.

        Args:
            dataset: Original dataset
            evaluations: Evaluation results

        Returns:
            Rewritten dataset
        """
        rewrite_dataset = Dataset()

        # step1. process dataset according to the evaluation results
        samples = self._process_dataset_by_evaluations(dataset, evaluations)

        # step2. get rewrite sample responses from LLM
        # initialize the usage counter for rewriter-generate
        # estimate the token and time usage with each sample
        usage_counter_gen = ModelUsageCounter(total=len(samples), name="Rewriter-Generation")
        # initialize buffer
        buffer_gen = TaskBuffer(total=len(samples), save_dir=os.path.join(self.buffer_dir, "rewrite-generation"))
        # rewrite
        if parallel_executor and parallel_executor.n_workers > 1:
            # parallel processing
            rewrite_sample_strings: List[Union[Dict, str]] = parallel_executor.execute(
                iterable_inputs=samples, 
                process_function=self._rewrite_single_sample, 
                usage_counter=usage_counter_gen, 
                n=1, 
                buffer=buffer_gen
            )
        else:
            # sequential processing
            rewrite_sample_strings: List[str] = buffer_gen.load(usage_counter_gen)
            sample_idx: int = 0
            for sample in tqdm(samples, desc="Rewriting samples", unit="sample"):
                if buffer_gen and buffer_gen.detail_progress[sample_idx]:
                    sample_idx += 1
                    continue

                rewrite_sample_strings.append(self._rewrite_single_sample(sample, usage_counter_gen))

                usage_counter_gen.estimate_usage(n=1)
                buffer_gen.add_progress([sample_idx])
                buffer_gen.save(rewrite_sample_strings, usage_counter_gen)
                sample_idx += 1

        # step3. validate sample strings and parse them to samples
        # initialize the usage counter for rewriter-validate
        # estimate the token and time usage with each sample
        usage_counter_val = ModelUsageCounter(total=len(rewrite_sample_strings), name="Rewriter-Validation")
        # initialize buffer
        buffer_val = TaskBuffer(total=len(samples), save_dir=os.path.join(self.buffer_dir, "rewrite-validation"))
        # parse and validate
        rewrite_samples: List[Dict] = self.parse_and_validate_samples(
            response_strings=rewrite_sample_strings,
            output_instruction=self.config.output_instruction, 
            usage_counter=usage_counter_val, 
            parallel_executor=parallel_executor, 
            buffer=buffer_val
        )
        rewrite_dataset.add_samples(rewrite_samples)

        return rewrite_dataset


# Specific class for rewriting samples by adjusting difficulty
class DifficultyAdjustRewriter(BaseRewriter):
    """
    class for rewrite synthetic data by adjust their difficulty
    """
    def __init__(self, 
        llm: ModelClient, 
        config: DifficultyAdjustRewriteConfig
    ) -> None:
        super(DifficultyAdjustRewriter, self).__init__(llm, config)
        self.config: DifficultyAdjustRewriteConfig
        self.harder_temperature: float = self.config.harder_temperature
        self.easier_temperature: float = self.config.easier_temperature

    def _process_dataset_by_evaluations(self, 
        dataset: Dataset, 
        evaluations: Dict
    ) -> List[Dict]:
        """
        transform samples in dataset to the samples which rewriting demands, according to the evaluations
        """
        samples: List[Dict] = []
        for sample, score in zip(dataset.samples, evaluations["scores"]):
            if score == 1.0:
                label = "solved"
            elif score == 0.0:
                label = "unsolved"
            else:
                label = "learnable"
            samples.append({"label": label, "sample": sample})
        return samples

    def _rewrite_single_sample(self, 
        sample: Dict, 
        usage_counter: ModelUsageCounter = None
    ) -> Union[Dict, str]:
        """
        Generate a harder / simpler version of the given sample.

        Args:
            sample: Original sample dict with 'label' and 'sample' keys

        Returns:
            rewrite_sample
        """
        label = sample["label"]
        if label == "learnable":
            sample = sample["sample"]
            return sample

        else:
            is_harder = (label == "solved")
            prompt_template = HARDER_SAMPLE_PROMPT if is_harder else SIMPLER_SAMPLE_PROMPT
            temperature = self.harder_temperature if is_harder else self.easier_temperature

            # Use format_prompts to combine output_instruction with answer_config
            combined_output_instruction = self.model.answer_extractor.format_prompts(
                self.config.output_instruction
            )

            prompt = prompt_template.format(
                sample=sample["sample"],
                input_instruction=self.config.input_instruction,
                output_instruction=combined_output_instruction
            )

            response: str= self.model.generate(prompt, usage_counter=usage_counter, temperature=temperature)

        return response