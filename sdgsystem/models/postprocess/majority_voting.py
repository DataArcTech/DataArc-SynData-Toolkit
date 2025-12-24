"""
File of majority voting methods for LLM responses:
    - Majority Voting
"""
import random
import time
import logging
from typing import List, Dict, Optional, Tuple, Union
from statistics import mode
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ...configs.config import (
    MajorityVotingConfig, 
    BaseVotingConfig, 
    ExactMatchVotingConfig, 
    SemanticClusteringVotingConfig, 
    LLMJudgeVotingConfig
)
from ...prompts import LLM_JUDGE_VOTING_PROMPT
from ..models import BaseLanguageModel
from ..usage_counter import ModelUsageCounter
from ..answer_extraction import AnswerExtractor
from ..processor_arguments import ProcessorArgs
from .base import BasePostProcessor

logger = logging.getLogger(__name__)


class MajorityVotingProcessor(BasePostProcessor):
    def __init__(self, 
        processor: Union[BaseLanguageModel, "BasePostProcessor"], 
        config: MajorityVotingConfig
    ) -> None:
        super(MajorityVotingProcessor, self).__init__(processor, config)
        self.config: MajorityVotingConfig
        self.n_voting = self.config.n_voting

        self.voting_config = self.config.voting_config
        if self.voting_config is None:
            raise Exception(f"Set MajorityVotingProcessor, but the config is None.")
        self.voting: BaseVoting = BaseVoting.from_config(
            self.voting_config,
            model=self.get_model()
        )

    def generate(self, 
        prompts: Union[str, List[str]], 
        n: int = 1, 
        answer_extractor: AnswerExtractor = None, 
        processor_args: ProcessorArgs = ProcessorArgs(), 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        # if processor_args.majority_voting is None or disable, skip majority_voting
        if processor_args.majority_voting is None or not processor_args.majority_voting.enable:
            return self._generate(prompts, n, answer_extractor, processor_args, usage_counter, **kwargs)

        # check whether prompt is single
        is_single_prompt = isinstance(prompts, str)

        # extract samples from kwargs if exist
        samples = processor_args.majority_voting.samples
        samples: List[Dict] = [samples] if isinstance(samples, Dict) else samples

        all_selected_responses = []      # n * len(prompts)

        for _ in range(n):
            # get response from model
            model_responses = self._generate(prompts, self.n_voting, None, processor_args, usage_counter, **kwargs)

            ## if is llm-judge-voting, extract questions from kwargs
            is_llm_judge_voting = isinstance(self.voting_config, LLMJudgeVotingConfig)

            selected_outputs: List[str] = []
            all_responses = [model_responses] if is_single_prompt else model_responses

            for responses, sample in zip(all_responses, samples):
                sample_input: str = sample.get("input")
                sample_output: str = sample.get("output", None)
                if sample_output:
                    responses.append(sample_output)

                answers: List[str] = answer_extractor.extract_answers(responses)

                # Check if answer extraction returned None
                if answers is None:
                    logger.warning(f"  Answer extraction failed - returned None")
                    return None

                # Filter out None values from failed extractions
                valid_answers = [ans for ans in answers if ans is not None]
                if len(valid_answers) < 2:
                    logger.warning(f"  Not enough valid answers for voting: {len(valid_answers)} valid out of {len(responses)} responses")
                    return None

                # Use valid_answers for voting
                answers = valid_answers
                
                if is_llm_judge_voting:
                    self.voting: LLMJudgeVoting
                    mode_value, selected_idx, mode_ids = self.voting.voting(answers, usage_counter, example_input=sample_input)
                else:
                    mode_value, selected_idx, mode_ids = self.voting.voting(answers, usage_counter)

                # Check if voting succeeded
                if mode_value is None or mode_ids is None or selected_idx is None:
                    return None

                # Get the selected output
                selected_output = responses[selected_idx]

                # # Get all matching responses sorted by length (for training diversity)
                # all_matching_outputs = [(responses[idx], len(responses[idx])) for idx in mode_ids]
                # all_matching_outputs.sort(key=lambda x: x[1])

                # return selected_output, all_matching_outputs
                selected_outputs.append(selected_output)

            output = selected_outputs[0] if is_single_prompt else selected_outputs
            all_selected_responses.append(output)

        if is_single_prompt:
            if n == 1:
                final_responses: str = all_selected_responses[0]
            else:
                final_responses: List[str] = all_selected_responses
        else:
            if n == 1:
                final_responses: List[str] = all_selected_responses[0]
            else:
                final_responses: List[List[str]] = list(zip(*all_selected_responses))

        return final_responses


class BaseVoting(ABC):
    def __init__(self, config: BaseVotingConfig) -> None:
        self.config = config

    @staticmethod
    def from_config(config: BaseVotingConfig, **kwargs) -> "BaseVoting":
        if isinstance(config, ExactMatchVotingConfig):
            return ExactMatchVoting(config)

        if isinstance(config, SemanticClusteringVotingConfig):
            return SemanticClusteringVoting(config)

        if isinstance(config, LLMJudgeVotingConfig):
            return LLMJudgeVoting(config, model=kwargs.pop("model"))

        raise Exception(f"Voting method {config.method} is not supported.")

    @abstractmethod
    def voting(self, 
        answers: List[str], 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Tuple[Optional[str], Optional[int], Optional[List[int]]]:
        pass

    # basic voting - exact match
    def _exact_match_voting(self, 
        answers: List[str], 
        usage_counter: ModelUsageCounter = None, 
    ) -> Tuple[Optional[str], Optional[int], Optional[List[int]]]:
        st = time.time()
        try:
            mode_value = mode(answers)
            mode_ids = [idx for idx, value in enumerate(answers) if value == mode_value]
            selected_idx = random.choice(mode_ids) if mode_ids else None
        except:
            mode_value, selected_idx, mode_ids = None, None, None
        
        if usage_counter:
            usage_counter.add_usage(0, time.time() - st)
        return mode_value, selected_idx, mode_ids


class ExactMatchVoting(BaseVoting):
    def __init__(self, config: ExactMatchVotingConfig) -> None:
        super(ExactMatchVoting, self).__init__(config)
        self.config: ExactMatchVotingConfig

    def voting(self, 
        answers: List[str], 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Tuple[Optional[str], Optional[int], Optional[List[int]]]:
        """
        Perform exact match voting - select answers that appear most frequently.

        Args:
            answers: List of extracted answers

        Returns:
            Tuple of (mode_value, selected_indices, mode_indices) or (None, None, None) if no mode found
        """
        return self._exact_match_voting(answers, usage_counter)


class SemanticClusteringVoting(BaseVoting):
    def __init__(self, config: SemanticClusteringVotingConfig) -> None:
        super(SemanticClusteringVoting, self).__init__(config)
        self.config: SemanticClusteringVotingConfig
        self.model = None  # Lazy load on first use

    def voting(self, 
        answers: List[str], 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Tuple[Optional[str], Optional[int], Optional[List[int]]]:
        """
        Perform embedding-based clustering voting using SentenceTransformer/BGE.

        Algorithm:
        1. Encode all answers using BGE embeddings
        2. Compute cosine similarity matrix
        3. Group similar answers into clusters (similarity >= threshold)
        4. Find the largest cluster
        5. Select medoid (most central answer) from largest cluster

        Args:
            answers: List of extracted answers

        Returns:
            Tuple of (medoid_answer, selected_indices, cluster_indices) or (None, None, None) if clustering fails
        """
        if len(answers) < 2:
            return None, None, None

        try:
            st = time.time()
            # Lazy load model on first use
            if self.model is None:
                from sentence_transformers import SentenceTransformer
                device = self.config.device
                logger.info(f"Loading SentenceTransformer model: {self.config.model_path} on device: {device}")
                self.model = SentenceTransformer(self.config.model_path, device=device)
                logger.info(f"SentenceTransformer model loaded successfully")

            # Encode all answers to embeddings
            embeddings = self.model.encode(answers, convert_to_numpy=True)

            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)

            # Perform clustering using similarity threshold
            # Build adjacency graph: answers are connected if similarity >= threshold
            n = len(answers)
            visited = [False] * n
            clusters = []

            for i in range(n):
                if visited[i]:
                    continue

                # Start new cluster with answer i
                cluster = [i]
                visited[i] = True

                # Add all similar answers to this cluster
                for j in range(i + 1, n):
                    if not visited[j] and similarity_matrix[i][j] >= self.config.similarity_threshold:
                        cluster.append(j)
                        visited[j] = True

                clusters.append(cluster)

            # Find the largest cluster
            largest_cluster = max(clusters, key=len)

            if len(largest_cluster) == 0:
                return None, None, None

            # Find medoid: answer with highest average similarity to all others in cluster
            medoid_idx = None
            max_avg_similarity = -1

            for idx in largest_cluster:
                # Compute average similarity to all other answers in cluster
                avg_similarity = np.mean([similarity_matrix[idx][j] for j in largest_cluster if j != idx])

                if avg_similarity > max_avg_similarity:
                    max_avg_similarity = avg_similarity
                    medoid_idx = idx

            if medoid_idx is None:
                # Single answer in cluster
                medoid_idx = largest_cluster[0]

            if usage_counter:
                usage_counter.add_usage(0, time.time() - st)
            return answers[medoid_idx], medoid_idx, largest_cluster

        except Exception as e:
            logger.warning(f"Warning: embedding_clustering failed ({e}), falling back to exact_match")
            return self._exact_match_voting(answers, usage_counter)

    def cleanup(self):
        """Release SentenceTransformer model and free GPU memory."""
        if self.model is not None:
            logger.info(f"Releasing SentenceTransformer model (SemanticClusteringVoting) and freeing GPU memory...")
            del self.model
            self.model = None

            import gc
            gc.collect()

            try:
                import torch
                torch.cuda.empty_cache()
                logger.info(f"SentenceTransformer GPU memory released successfully!")
            except ImportError:
                logger.info(f"SentenceTransformer model released (torch not available)")


class LLMJudgeVoting(BaseVoting):
    def __init__(self, config: LLMJudgeVotingConfig, model: BaseLanguageModel) -> None:
        super(LLMJudgeVoting, self).__init__(config)
        self.config: LLMJudgeVotingConfig
        self.model = model

    def voting(self, 
        answers: List[str], 
        usage_counter: ModelUsageCounter = None, 
        **kwargs
    ) -> Tuple[Optional[str], Optional[int], Optional[List[int]]]:
        """
        Use an LLM to judge which answer is most correct.

        Algorithm:
        1. Present all candidate answers to LLM judge
        2. Ask LLM to select the best/most common correct answer
        3. LLM returns the specific index of the best answer
        4. Also find all exact match duplicates for diversity

        Args:
            answers: List of extracted answers
            example_input: Original input question

        Returns:
            Tuple of (selected_answer, selected_index, all_equivalent_indices) or (None, None, None) if judging fails
            - selected_index: The specific index LLM chose (use this for selected_output)
            - all_equivalent_indices: All indices with exact match to selected answer (for all_outputs)
        """
        if len(answers) < 2:
            return None, None, None

        try:
            # Format candidate answers for the prompt (add index)
            formatted_answers = "\n".join([f"[{i}] {answer}" for i, answer in enumerate(answers)])

            # Build the judge prompt
            example_input = kwargs.pop("example_input")
            prompt = LLM_JUDGE_VOTING_PROMPT.format(
                question=example_input,
                answers=formatted_answers,
                max_index=len(answers) - 1
            )

            # Query LLM judge (model inherited from global config via query_openai)
            response = self.model.generate(prompt, usage_counter=usage_counter, temperature=self.config.temperature)

            # Parse the response to get the selected index
            try:
                # Extract just the number from response
                selected_idx = int(response.strip().split()[0])

                if 0 <= selected_idx < len(answers):
                    selected_answer = answers[selected_idx]

                    # Find all answers that are equivalent to the selected one
                    # Use exact match to find duplicates
                    equivalent_indices = [i for i, ans in enumerate(answers) if ans.strip() == selected_answer.strip()]

                    return selected_answer, selected_idx, equivalent_indices
                else:
                    logger.warning(f"Warning: LLM judge returned invalid index {selected_idx}, falling back to exact_match")
                    mode_value, selected_idx, mode_ids = self._exact_match_voting(answers, usage_counter)
                    return mode_value, None, mode_ids

            except (ValueError, IndexError) as e:
                logger.warning(f"Warning: Failed to parse LLM judge response '{response}' ({e}), falling back to exact_match")
                mode_value, selected_idx, mode_ids = self._exact_match_voting(answers, usage_counter)
                return mode_value, selected_idx, mode_ids

        except Exception as e:
            logger.warning(f"Warning: llm_judge failed ({e}), falling back to exact_match")
            mode_value, selected_idx, mode_ids = self._exact_match_voting(answers, usage_counter)
            return mode_value, selected_idx, mode_ids