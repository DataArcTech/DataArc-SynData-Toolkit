import os
import yaml
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from .constants import *


# Base Configuration
class BaseConfig(BaseModel):
    def update(self, config: Dict):
        for f in self.model_fields.keys():
            if f in config and config[f] is not None:
                if isinstance(self.__dict__[f], BaseConfig):
                    self.__dict__[f].update(config[f])
                else:
                    self.__dict__[f] = config[f]

# Configuration for Task
class BaseTaskConfig(BaseConfig):
    """Base class of task configuration"""
    name: str = Field(..., description="Name of task")
    task_instruction: str = Field(..., description="task description")
    domain: str = Field(default=None, description="Domain of task")
    demo_examples_path: Optional[str] = Field(default=None, description="Path of demo examples for synthetic data.")

# Configuration for Local Task
class ParserConfig(BaseConfig):
    """Configuration for parsing documents"""
    method: str = Field(default=DEFAULT_PARSING_METHOD, description="parsing method")
    document_dir: str = Field(default=None, description="Directory containing PDF documents to parse")
    device: str = Field(default="cuda:0", description="Device to use for parsing (cuda or cpu)")

class RetrievalConfig(BaseConfig):
    """Configuration for retrieval"""
    passages_dir: str = Field(..., description="Directions to document corpora")
    method: str = Field(default=DEFAULT_RETRIEVAL_METHOD, description="retrieval method")
    top_k: int = Field(default=DEFAULT_RETRIEVAL_TOP_K, description="retrieval top_k")

class GenerationConfig(BaseConfig):
    """Configuration for data generation"""
    input_instruction: str = Field(..., description="input instruction")
    output_instruction: str = Field(..., description="output instruction")
    num_samples: int = Field(..., gt=0, description="number of initial data")
    batch_size: int = Field(default=5, gt=0, description="batch size for sample generation")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        gt=0.,
        description="llm temperature of data generation"
    )

class LocalTaskConfig(BaseTaskConfig):
    """Task configuration of generating data from local documents"""
    retrieval: RetrievalConfig = Field(..., description="retrieval configuration")
    parsing: ParserConfig = Field(..., description="parsing configuration")
    generation: GenerationConfig = Field(..., description="generation config")

    @classmethod
    def from_dict(cls, config: Dict) -> "LocalTaskConfig":
        try:
            generation_config_dict: Dict = config["generation"]
            config["task_instruction"] = generation_config_dict["task_instruction"]
            instance = cls(**config)
        except Exception as e:
            raise Exception(f"Error occured when parsing configuration of local task: {str(e)}")

        return instance

    def update(self, config: Dict):
        if "generation" in config:
            if "task_instruction" in config["generation"]:
                self.task_instruction = config["generation"]["task_instruction"]
        super().update(config)


# Configuration for web task
class WebTaskConfig(BaseTaskConfig):
    """Task Configuration of crawling from huggingface"""
    huggingface_token: str = Field(default=os.environ.get("HUGGINGFACE_TOKEN", None), description="huggingface token")
    input_instruction: str = Field(..., description="input instruction")
    output_instruction: str = Field(..., description="output instruction")
    dataset_limit: int = Field(default=1, gt=0, description="number of datasets to crawl per keyword")
    num_samples: int = Field(..., gt=0, description="number of samples to output in final dataset")
    dataset_score_threshold: int = Field(default=30, ge=0, description="minimum overall score (sum of 5 criteria) for a dataset to be valid")

    @classmethod
    def from_dict(cls, config: Dict) -> "WebTaskConfig":
        try:
            instance = cls(**config)
        except Exception as e:
            raise Exception(f"Error occured when parsing configuration of web task: {str(e)}")

        return instance


# Configuration for distll task
class DistillTaskConfig(BaseTaskConfig):
    """Task Configuration of model distll"""
    input_instruction: str = Field(default=None, description="input format instruction")
    output_instruction: str = Field(default=None, description="output format instruction")
    num_samples: int = Field(..., gt=0, description="number of samples to generate")
    batch_size: int = Field(default=5, gt=0, description="batch size for generation")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        gt=0.,
        description="llm temperature of data generation"
    )

    @classmethod
    def from_dict(cls, config: Dict) -> "DistillTaskConfig":
        try:
            instance = cls(**config)
        except Exception as e:
            raise Exception(f"Error occured when parsing configuration of distill task: {str(e)}")

        return instance


class SDGSTaskConfig(BaseConfig):
    """Total Task Configuration"""
    name: str = Field(default=DEFAULT_TASK_NAME)
    task_type: str = Field(default=None)
    local_task_config: Optional[LocalTaskConfig] = Field(default=None)
    web_task_config: Optional[WebTaskConfig] = Field(default=None)
    distill_task_config: Optional[DistillTaskConfig] = Field(default=None)

    @classmethod
    def from_dict(cls, config: Dict) -> "SDGSTaskConfig":
        name: str = config.get("name", DEFAULT_TASK_NAME)
        domain: str = config.get("domain", None)
        demo_examples_path: str = config.get("demo_examples_path", None)
        global_config_dict = {
            "name": name,
            "domain": domain,
            "demo_examples_path": demo_examples_path
        }

        # Use explicitly set task_type if provided, otherwise auto-detect
        task_type: str = config.get("task_type", None)
        if task_type is None:
            for t in ["local", "web", "distill"]:
                if t in config:
                    task_type = t
                    break

        local_task_config = LocalTaskConfig.from_dict({**global_config_dict, **config["local"]}) if "local" in config else None
        web_task_config = WebTaskConfig.from_dict({**global_config_dict, **config["web"]}) if "web" in config else None
        distill_task_config = DistillTaskConfig.from_dict({**global_config_dict, **config["distill"]}) if "distill" in config else None

        return cls(
            name=name,
            task_type=task_type,
            local_task_config=local_task_config,
            web_task_config=web_task_config,
            distill_task_config=distill_task_config
        )

    def update(self, config: Dict):
        name: str = config.get("name", self.name)
        global_config_dict = {"name": name}
        if "domain" in config:
            global_config_dict["domain"] = config["domain"]
        if "demo_examples_path" in config:
            global_config_dict["demo_examples_path"] = config["demo_examples_path"]

        if self.local_task_config:
            self.local_task_config.update({**global_config_dict, **config.get("local", {})})
        
        if self.web_task_config:
            self.web_task_config.update({**global_config_dict, **config.get("web", {})})

        if self.distill_task_config:
            self.distill_task_config.update({**global_config_dict, **config.get("distill", {})})

        self.name = name
        self.task_type = config.get("task_type", self.task_type)


# Model Configuration
class InferenceConfig(BaseConfig):
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=1500)
    top_p: float = Field(default=0.95)
    n: int = Field(default=1)

class LocalModelConfig(BaseConfig):
    path: str = Field(..., description="model name or path (if provider is 'local', this param should be the path)")
    device: str = Field(default="cuda:0", description="CUDA device (e.g., 'cuda:0')")
    max_model_len: int = Field(default=DEFAULT_LOCAL_MODEL_LEN, description="max model length")
    gpu_memory_utilization: float = Field(default=DEFAULT_GPU_UTILIZATION, description="gpu memory utilization")
    inference: InferenceConfig = Field(default=InferenceConfig())
    scoring: InferenceConfig = Field(default=InferenceConfig())

    @classmethod
    def from_dict(cls, config: Dict) -> "LocalModelConfig":
        try:
            instance = cls(**config)
        except Exception as e:
            raise Exception(f"Error occured when parsing configuration of local model: {str(e)}")

        return instance


class APIModelConfig(BaseConfig):
    provider: str = Field(default=DEFAULT_API_PROVIDER, description="provider of LLM, choices=[openai, ollama]")
    model: str = Field(..., description="model name")
    api_key: str = Field(default=None, description="api key")
    base_url: str = Field(default=None, description="base url")
    max_retry_attempts: int = Field(
        default=DEFAULT_MAX_RETRY_ATTEMPTS, 
        ge=0, 
        description="retry number"
    )
    retry_delay: float = Field(
        default=DEFAULT_RETRY_BASE_DELAY, 
        ge=0, 
        description="retry delay"
    )

    @classmethod
    def from_dict(cls, config: Dict) -> "APIModelConfig":
        try:
            instance = cls(**config)
        except Exception as e:
            raise Exception(f"Error occured when parsing configuration of API model: {str(e)}")

        return instance


class ModelConfig(BaseConfig):
    """Configuration for Model (Total)"""
    provider: str = Field(default=DEFAULT_API_PROVIDER, description="provider of LLM, choices=[openai, ollama]")
    config: Union[LocalModelConfig, APIModelConfig] = Field(default=None, description="configuration of model")

    @classmethod
    def from_dict(cls, config: Dict) -> "ModelConfig":
        instance = cls()
        instance.provider = config.get("provider", "local")
        if instance.provider == "local":
            instance.config = LocalModelConfig(**config)
        else:
            instance.config = APIModelConfig(**config)
        return instance

    def update(self, config: Dict):
        self.config.update(config)


# Configuration for Answer Extraction
class AnswerExtractionConfig(BaseConfig):
    """Answer extraction configuration."""
    enabled: bool = Field(default=True, description="Whether answer extraction is enabled.")
    tag: str = Field(...)
    instruction: str = Field(...)


# Configuration for postprocess of LLMs' responses
class BasePostProcessConfig(BaseConfig):
    method: str = Field(default="majority_voting")

    @staticmethod
    def from_dict(config: Dict, method: str) -> "BasePostProcessConfig":
        if method == "majority_voting":
            return MajorityVotingConfig.from_dict(config)
        
        raise Exception(f"Error occured when parsing configuration of postprocess: {method} is not supported for postprocess.")

## Configuration for Majority Voting
class BaseVotingConfig(BaseConfig):
    """Base Configuration for voting method"""
    method: str = Field(...)
    
    @staticmethod
    def from_dict(config: Dict) -> "BaseVotingConfig":
        try:
            method = config.get("method", DEFAULT_VOTING_METHOD)
            addition_config = config.get(method, {})
            total_config = {**config, **addition_config}
        except Exception as e:
            raise Exception(f"Error occured when parsing configuration of majority_voting: {str(e)}")

        # get specific config according to method
        if method == "exact_match":
            return ExactMatchVotingConfig(**total_config)

        if method == "semantic_clustering":
            return SemanticClusteringVotingConfig(**total_config)

        if method == "llm_judge":
            return LLMJudgeVotingConfig(**total_config)

        raise Exception(f"Error occured when parsing configuration of majority_voting: method {method} is not supported for majority_voting.")

    def update(self, config: Dict):
        method = config.get("method", self.method)
        addition_config = config.get(method, {})
        if addition_config:
            super().update({**config, **addition_config})

class ExactMatchVotingConfig(BaseVotingConfig):
    numeric_tolerance: float = Field(default=1e-3)

class SemanticClusteringVotingConfig(BaseVotingConfig):
    model_path: str = Field(default="BAAI/bge-large-zh-v1.5")
    device: str = Field(default="cuda:0", description="CUDA device (e.g., 'cuda:0')")
    similarity_threshold: float = Field(default=0.85)

class LLMJudgeVotingConfig(BaseVotingConfig):
    temperature: float = Field(default=0.3)


### Total configuration for majority voting
class MajorityVotingConfig(BasePostProcessConfig):
    """Configuration for majority voting"""
    n_voting: int = Field(default=DEFAULT_N_VOTING)
    voting_config: BaseVotingConfig = Field(default=ExactMatchVotingConfig(method="exact_match"))

    @classmethod
    def from_dict(cls, config: Dict) -> "MajorityVotingConfig":
        n_voting: int = config.pop("n", DEFAULT_N_VOTING)
        voting_config = BaseVotingConfig.from_dict(config)
        return cls(n_voting=n_voting, voting_config=voting_config)

    def update(self, config: Dict):
        n_voting: int = config.pop("n", None)
        if n_voting:
            self.n_voting = n_voting
        self.voting_config.update(config)


# Total configuration for postprocess
class PostProcessConfig(BaseConfig):
    methods: List[str] = Field(default=[])
    configs: Dict[str, BasePostProcessConfig] = Field(default={})

    @classmethod
    def from_dict(cls, config: Dict) -> "PostProcessConfig":
        methods: List[str] = config["methods"]
        configs: Dict[BasePostProcessConfig] = {}
        for method in methods:
            method_config_dict: Dict = config.get(method, {})
            config = BasePostProcessConfig.from_dict(method_config_dict, method)
            configs[method] = config
        return cls(methods=methods, configs=configs)

    def update(self, config: Dict):
        methods: List[str] = config.get(methods, self.methods)
        for method in methods:
            method_config_dict: Dict = config.get(method, {})
            if method in self.configs:
                self.configs[method].update(method_config_dict)
            else:
                self.configs[method] = BasePostProcessConfig.from_dict(method_config_dict, method)

# Configuration for Evaluaiton
class BaseComparisonConfig(BaseConfig):
    """Base Configuration for answer comparison method"""
    method: str = Field(default=DEFAULT_COMPARISON_METHOD)

    @staticmethod
    def from_dict(config: Dict) -> "BaseComparisonConfig":
        try:
            method = config["method"]
            addition_config = config.get(method, {})
            total_config = {**config, **addition_config}
        except Exception as e:
            raise Exception(f"Error occured when parsing configuration of answer_comparison: {str(e)}")

        # get specific config according to method 
        if method == "exact_match":
            return ExactMatchComparisonConfig(**total_config)
        
        if method == "semantic":
            return SemanticComparisonConfig(**total_config)

        if method == "llm_judge":
            return LLMJudgeComparisonConfig(**total_config)
        
        raise Exception(f"Error occured when parsing configuration of answer_comparison: method {method} is not supported for answer_comparison.")

    def update(self, config: Dict):
        method = config.get("method", self.method)
        addition_config = config.get(method, {})
        if addition_config:
            super().update({**config, **addition_config})

class ExactMatchComparisonConfig(BaseComparisonConfig):
    numeric_tolerance: float = Field(default=1e-3)

class SemanticComparisonConfig(BaseComparisonConfig):
    model_path: str = Field(default="BAAI/bge-large-zh-v1.5")
    device: str = Field(default="cuda:0", description="CUDA device (e.g., 'cuda:0')")
    similarity_threshold: float = Field(default=0.85)

class LLMJudgeComparisonConfig(BaseComparisonConfig):
    temperature: float = Field(default=0.3)

class EvaluationConfig(BaseConfig):
    """Configuration for evaluation"""
    batch_size: int = Field(...)
    input_instruction: str = Field(default=None)
    output_instruction: str = Field(default=None)
    answer_comparison_config: BaseComparisonConfig = Field(default=ExactMatchComparisonConfig(method="exact_match"))

    @classmethod
    def from_dict(cls, config: Dict) -> "EvaluationConfig":
        answer_comparison_config = BaseComparisonConfig.from_dict(config["answer_comparison"])
        total_config = {**config, **{"answer_comparison_config": answer_comparison_config}}
        return cls(**total_config)

    def update(self, config: Dict):
        answer_comparison_config_dict = config.get("answer_comparison", {})
        if answer_comparison_config_dict:
            config = {**config, **{"answer_comparison_config": answer_comparison_config_dict}}
        super().update(config)


# Configuration for Rewrite
class BaseRewriteConfig(BaseConfig):
    method: str = Field(default=DEFAULT_REWRITE_METHOD)
    input_instruction: str = Field(...)
    output_instruction: str = Field(...)

    @staticmethod
    def from_dict(config: Dict) -> "BaseRewriteConfig":
        try:
            method = config["method"]
            addition_config = config.get(method, {})
            total_config = {**config, **addition_config}
        except Exception as e:
            raise Exception(f"Error occured when parsing configuration of rewrite: {str(e)}")
        
        # get specific config according to method
        if method == "difficulty_adjust":
            return DifficultyAdjustRewriteConfig(**total_config)

        raise Exception(f"Error occured when parsing configuration of rewrite: method {method} is not supported for rewrite.")

    def update(self, config: Dict):
        method = config.get("method", self.method)
        addition_config = config.get(method, {})
        if addition_config:
            super().update({**config, **addition_config})

class DifficultyAdjustRewriteConfig(BaseRewriteConfig):
    easier_temperature: float = Field(default=DEFAULT_EASIER_TEMPERATURE)
    harder_temperature: float = Field(default=DEFAULT_HARDER_TEMPERATURE)


# Configuration for Translation
class TranslationConfig(BaseConfig):
    """Configuration for translating generated dataset to target language"""
    language: str = Field(default="english", description="Target language for the final dataset (e.g., 'english', 'arabic')")
    model_path: Optional[str] = Field(default=None, description="Translation model path (auto-determined based on language if not specified)")
    max_tokens: int = Field(default=256, description="Maximum tokens for translation generation")
    batch_size: int = Field(default=1, description="Batch size for translation")

    @classmethod
    def from_dict(cls, config: Dict) -> "TranslationConfig":
        try:
            return cls(**config)
        except Exception as e:
            raise Exception(f"Error occurred when parsing translation configuration: {str(e)}")


# Global
class SDGSConfig(BaseConfig):
    seed: int = Field(...)
    device: str = Field(default="cuda:0", description="CUDA device to use for all GPU operations")
    output_dir: str = Field(..., description="synthetic dataset output directory")
    export_format: str = Field(default=DEFAULT_EXPORT_FORMAT, description="Export dataset format")
    n_workers: int = Field(default=1, description="Number of parallel workers. Default n_workers=1 means sequential processing.")

    task_config: SDGSTaskConfig = Field(...)
    generator_config: ModelConfig = Field(...)
    base_model_config: ModelConfig = Field(...)
    answer_config: AnswerExtractionConfig = Field(default=None)
    postprocess_config: PostProcessConfig = Field(default=None)
    evaluation_config: EvaluationConfig = Field(...)
    rewrite_config: BaseRewriteConfig = Field(default=None)
    translation_config: TranslationConfig = Field(default=None, description="Translation configuration")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SDGSConfig":
        """Load configuration from a YAML file."""
        try:
            with open(yaml_path, encoding="utf-8") as fr:
                config_dict = yaml.safe_load(fr)
        except FileNotFoundError as e:
            raise Exception(f"not found: {yaml_path}")
        except yaml.YAMLError as e:
            raise Exception(f"invalid YAML: {str(e)}")
        except Exception as e:
            raise Exception(f"read error: {str(e)}")

        if not isinstance(config_dict, Dict):
            raise Exception("Error when parsing YAML.")

        # Get global device
        global_device = config_dict.get("device", "cuda:0")
        task_config = SDGSTaskConfig.from_dict(config_dict["task"])

        # Inject global device into parsing config if local task exists
        if task_config.local_task_config and task_config.local_task_config.parsing:
            task_config.local_task_config.parsing.device = global_device

        generator_config = ModelConfig.from_dict(config_dict["llm"])
        base_model_config = ModelConfig.from_dict(config_dict["base_model"])

        # Inject global device into base_model if it's a LocalModelConfig
        if base_model_config.provider == "local" and isinstance(base_model_config.config, LocalModelConfig):
            base_model_config.config.device = global_device

        answer_config = AnswerExtractionConfig(**config_dict["answer_extraction"])
        postprocess_config = PostProcessConfig.from_dict(config_dict["postprocess"])
        evaluation_config = EvaluationConfig.from_dict(config_dict["evaluation"])
        rewrite_config = BaseRewriteConfig.from_dict(config_dict["rewrite"])
        translation_config = TranslationConfig.from_dict(config_dict.get("translation", {}))

        return cls(
            seed=config_dict["seed"],
            device=config_dict.get("device", "cuda:0"),
            output_dir=config_dict["output_dir"],
            export_format=config_dict["export_format"],
            n_workers=config_dict.get("n_workers", 1), 
            task_config=task_config,
            generator_config=generator_config,
            base_model_config=base_model_config,
            answer_config=answer_config,
            postprocess_config=postprocess_config,
            evaluation_config=evaluation_config,
            rewrite_config=rewrite_config,
            translation_config=translation_config
        )

    def update(self, config: Dict):
        if "task" in config:
            self.task_config.update(config.pop("task"))
        
        if "llm" in config:
            self.generator_config.update(config.pop("llm"))
        
        if "base_model" in config:
            self.base_model_config.update(config.pop("base_model"))

        if "answer_extraction" in config:
            self.answer_config.update(config.pop("answer_extraction"))

        if "postprocess" in config:
            self.postprocess_config.update(config.pop("postprocess"))

        if "evaluation" in config:
            self.evaluation_config.update(config.pop("evaluation"))

        if "rewrite" in config:
            self.rewrite_config.update(config.pop("rewrite"))

        super().update(config)