import os
import yaml
from typing import Dict, List, Optional, Union
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


# ============ Base Task Configuration ============
class BaseTaskConfig(BaseConfig):
    """Base class of task configuration"""
    name: str = Field(..., description="Name of task")
    task_instruction: str = Field(..., description="task description")
    input_instruction: Optional[str] = Field(default="", description="input instruction")
    output_instruction: Optional[str] = Field(default="", description="output instruction")
    num_samples: int = Field(..., gt=0, description="number of samples to generate")
    batch_size: int = Field(default=5, gt=0, description="batch size for generation")
    domain: str = Field(default=None, description="Domain of task")
    demo_examples_path: Optional[str] = Field(default=None, description="Path of demo examples for synthetic data.")


# ============ Common Configurations ============

class ParserConfig(BaseConfig):
    """Configuration for parsing documents (used by text and image modalities)"""
    method: str = Field(default=DEFAULT_PARSING_METHOD, description="parsing method")
    document_dir: str = Field(default=None, description="Directory containing PDF documents to parse")
    device: str = Field(default="cuda:0", description="Device to use for parsing (cuda or cpu)")


class GenerationConfig(BaseConfig):
    """Configuration for data generation (used by text and image modalities)"""
    input_instruction: Optional[str] = Field(default="", description="input instruction (inherited from task config)")
    output_instruction: Optional[str] = Field(default="", description="output instruction (inherited from task config)")
    num_samples: int = Field(default=None, gt=0, description="number of samples (inherited from task config)")
    batch_size: int = Field(default=5, gt=0, description="batch size for sample generation")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        gt=0.,
        description="llm temperature of data generation"
    )


# ============ Text Source Configurations ============

class RetrievalConfig(BaseConfig):
    """Configuration for text retrieval"""
    passages_dir: str = Field(..., description="Directions to document corpora")
    method: str = Field(default=DEFAULT_RETRIEVAL_METHOD, description="retrieval method")
    top_k: int = Field(default=DEFAULT_RETRIEVAL_TOP_K, description="retrieval top_k")


class TextLocalConfig(BaseTaskConfig):
    """Configuration for text.local - local document source"""
    retrieval: RetrievalConfig = Field(..., description="retrieval configuration")
    parsing: ParserConfig = Field(..., description="parsing configuration")
    generation: GenerationConfig = Field(..., description="generation config")

    @classmethod
    def from_dict(cls, config: Dict) -> "TextLocalConfig":
        try:
            # Inject task-level config into generation config
            generation_config_dict: Dict = config.get("generation", {})
            if config.get("input_instruction"):
                generation_config_dict["input_instruction"] = config["input_instruction"]
            if config.get("output_instruction"):
                generation_config_dict["output_instruction"] = config["output_instruction"]
            if config.get("num_samples"):
                generation_config_dict["num_samples"] = config["num_samples"]
            if config.get("batch_size"):
                generation_config_dict["batch_size"] = config["batch_size"]
            config["generation"] = generation_config_dict
            instance = cls(**config)
        except Exception as e:
            raise Exception(f"Error occurred when parsing configuration of text.local: {str(e)}")

        return instance


class TextWebConfig(BaseTaskConfig):
    """Configuration for text.web - HuggingFace source"""
    huggingface_token: str = Field(default=os.environ.get("HUGGINGFACE_TOKEN", None), description="huggingface token")
    dataset_limit: int = Field(default=DEFAULT_WEB_DATASET_LIMIT, gt=0, description="number of datasets to crawl per keyword")
    dataset_score_threshold: int = Field(default=30, ge=0, description="minimum overall score (sum of 5 criteria) for a dataset to be valid")

    @classmethod
    def from_dict(cls, config: Dict) -> "TextWebConfig":
        try:
            instance = cls(**config)
        except Exception as e:
            raise Exception(f"Error occurred when parsing configuration of text.web: {str(e)}")

        return instance


class TextDistillConfig(BaseTaskConfig):
    """Configuration for text.distill - distillation source"""
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        gt=0.,
        description="llm temperature of data generation"
    )

    @classmethod
    def from_dict(cls, config: Dict) -> "TextDistillConfig":
        try:
            instance = cls(**config)
        except Exception as e:
            raise Exception(f"Error occurred when parsing configuration of text.distill: {str(e)}")

        return instance


# ============ Image Source Configurations ============

class ImageLocalConfig(BaseTaskConfig):
    """Configuration for image.local - local image source

    Image sources:
    - image_dir: Directory containing user-uploaded images
    - parsing: PDF parsing config (reuses ParserConfig) - extracts images from PDFs via MinerU

    At least one source must be provided. If both are provided, images are combined.
    """
    image_dir: Optional[str] = Field(default=None, description="Directory containing user-uploaded images")
    parsing: Optional[ParserConfig] = Field(default=None, description="PDF parsing config for image extraction")
    generation: GenerationConfig = Field(..., description="Generation config")
    output_dir: str = Field(default=None, description="Output directory for images (injected from global config)")

    @classmethod
    def from_dict(cls, config: Dict) -> "ImageLocalConfig":
        try:
            # Inject task-level config into generation config
            generation_config_dict: Dict = config.get("generation", {})
            if config.get("input_instruction"):
                generation_config_dict["input_instruction"] = config["input_instruction"]
            if config.get("output_instruction"):
                generation_config_dict["output_instruction"] = config["output_instruction"]
            if config.get("num_samples"):
                generation_config_dict["num_samples"] = config["num_samples"]
            if config.get("batch_size"):
                generation_config_dict["batch_size"] = config["batch_size"]
            config["generation"] = generation_config_dict
            instance = cls(**config)
        except Exception as e:
            raise Exception(f"Error occurred when parsing configuration of image.local: {str(e)}")

        return instance


class ImageWebConfig(BaseTaskConfig):
    """Configuration for image.web - HuggingFace image dataset source

    Searches HuggingFace for image datasets, probes them for quality,
    and downloads images with their associated QA pairs.
    """
    huggingface_token: str = Field(default=os.environ.get("HUGGINGFACE_TOKEN", None), description="huggingface token")
    dataset_limit: int = Field(default=1, gt=0, description="number of datasets to crawl per keyword")
    dataset_score_threshold: int = Field(default=30, ge=0, description="minimum overall score (sum of 5 criteria) for a dataset to be valid")
    output_dir: str = Field(default=None, description="Output directory for images (injected from global config)")

    @classmethod
    def from_dict(cls, config: Dict) -> "ImageWebConfig":
        try:
            instance = cls(**config)
        except Exception as e:
            raise Exception(f"Error occurred when parsing configuration of image.web: {str(e)}")

        return instance


# ============ Image Modality Configuration ============

class ImageModalityConfig(BaseConfig):
    """Configuration for image modality - contains local or web source"""
    local: Optional[ImageLocalConfig] = Field(default=None, description="Local image source config")
    web: Optional[ImageWebConfig] = Field(default=None, description="Web/HuggingFace image source config")

    @classmethod
    def from_dict(cls, config: Dict, global_config: Dict) -> "ImageModalityConfig":
        """Parse image modality config with global config injection"""
        # Validate: only one source should be configured
        sources = [key for key in ["local", "web"] if key in config]
        if len(sources) > 1:
            raise Exception(
                f"Multiple image sources configured: {sources}. "
                "Please specify only one of 'local' or 'web' under 'image'."
            )
        if len(sources) == 0:
            raise Exception(
                "image modality configured but no source specified. "
                "Please specify 'local' or 'web' under 'image'."
            )

        local_config = None
        web_config = None

        if "local" in config:
            local_config = ImageLocalConfig.from_dict({**global_config, **config["local"]})

        if "web" in config:
            web_config = ImageWebConfig.from_dict({**global_config, **config["web"]})

        return cls(local=local_config, web=web_config)


# ============ Text Modality Configuration ============

class TextModalityConfig(BaseConfig):
    """Configuration for text modality - contains local, web, or distill source"""
    local: Optional[TextLocalConfig] = Field(default=None, description="Local document source config")
    web: Optional[TextWebConfig] = Field(default=None, description="Web/HuggingFace source config")
    distill: Optional[TextDistillConfig] = Field(default=None, description="Distillation source config")

    @classmethod
    def from_dict(cls, config: Dict, global_config: Dict) -> "TextModalityConfig":
        """Parse text modality config with global config injection"""
        # Validate: only one source should be configured
        sources = [key for key in ["local", "web", "distill"] if key in config]
        if len(sources) > 1:
            raise Exception(
                f"Multiple text sources configured: {sources}. "
                "Please specify only one of 'local', 'web', or 'distill' under 'text'."
            )
        if len(sources) == 0:
            raise Exception(
                "text modality configured but no source specified. "
                "Please specify one of 'local', 'web', or 'distill' under 'text'."
            )

        local_config = None
        web_config = None
        distill_config = None

        if "local" in config:
            local_config = TextLocalConfig.from_dict({**global_config, **config["local"]})

        if "web" in config:
            web_config = TextWebConfig.from_dict({**global_config, **config["web"]})

        if "distill" in config:
            distill_config = TextDistillConfig.from_dict({**global_config, **config["distill"]})

        return cls(local=local_config, web=web_config, distill=distill_config)


# ============ Task Configuration ============

class SDGSTaskConfig(BaseConfig):
    """Total Task Configuration with modality-based structure"""
    name: str = Field(default=DEFAULT_TASK_NAME)
    text: Optional[TextModalityConfig] = Field(default=None, description="Text modality configuration")
    image: Optional[ImageModalityConfig] = Field(default=None, description="Image modality configuration")

    @classmethod
    def from_dict(cls, config: Dict) -> "SDGSTaskConfig":
        # Validate: only one modality should be configured
        modalities = [key for key in ["text", "image"] if key in config]
        if len(modalities) > 1:
            raise Exception(
                f"Multiple modalities configured: {modalities}. "
                "Please specify only one of 'text' or 'image' in your config."
            )
        if len(modalities) == 0:
            raise Exception(
                "No modality configured. "
                "Please specify 'text' or 'image' in your config."
            )

        # Extract global task config
        name: str = config.get("name", DEFAULT_TASK_NAME)
        domain: str = config.get("domain", None)
        demo_examples_path: str = config.get("demo_examples_path", None)
        task_instruction: str = config.get("task_instruction", None)
        input_instruction: str = config.get("input_instruction") or ""
        output_instruction: str = config.get("output_instruction") or ""
        num_samples: int = config.get("num_samples", None)
        batch_size: int = config.get("batch_size", None)

        global_config_dict = {
            "name": name,
            "domain": domain,
            "demo_examples_path": demo_examples_path,
            "task_instruction": task_instruction,
            "input_instruction": input_instruction,
            "output_instruction": output_instruction,
            "num_samples": num_samples,
            "batch_size": batch_size,
        }

        # Parse text modality
        text_config = None
        if "text" in config:
            text_config = TextModalityConfig.from_dict(config["text"], global_config_dict)

        # Parse image modality
        image_config = None
        if "image" in config:
            image_config = ImageModalityConfig.from_dict(config["image"], global_config_dict)

        return cls(name=name, text=text_config, image=image_config)

    def update(self, config: Dict):
        name: str = config.get("name", self.name)
        global_config_dict = {"name": name}
        if "domain" in config:
            global_config_dict["domain"] = config["domain"]
        if "demo_examples_path" in config:
            global_config_dict["demo_examples_path"] = config["demo_examples_path"]
        if "task_instruction" in config:
            global_config_dict["task_instruction"] = config["task_instruction"]
        if "input_instruction" in config:
            global_config_dict["input_instruction"] = config["input_instruction"]
        if "output_instruction" in config:
            global_config_dict["output_instruction"] = config["output_instruction"]
        if "num_samples" in config:
            global_config_dict["num_samples"] = config["num_samples"]
        if "batch_size" in config:
            global_config_dict["batch_size"] = config["batch_size"]

        if self.text:
            if self.text.local:
                self.text.local.update({**global_config_dict, **config.get("text", {}).get("local", {})})
            if self.text.web:
                self.text.web.update({**global_config_dict, **config.get("text", {}).get("web", {})})
            if self.text.distill:
                self.text.distill.update({**global_config_dict, **config.get("text", {}).get("distill", {})})

        if self.image:
            if self.image.local:
                self.image.local.update({**global_config_dict, **config.get("image", {}).get("local", {})})
            if self.image.web:
                self.image.web.update({**global_config_dict, **config.get("image", {}).get("web", {})})

        self.name = name


# ============ Model Configuration ============

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

    @classmethod
    def from_dict(cls, config: Dict) -> "LocalModelConfig":
        try:
            instance = cls(**config)
        except Exception as e:
            raise Exception(f"Error occurred when parsing configuration of local model: {str(e)}")

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
            raise Exception(f"Error occurred when parsing configuration of API model: {str(e)}")

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


# ============ Answer Extraction Configuration ============

class AnswerExtractionConfig(BaseConfig):
    """Answer extraction configuration."""
    enabled: bool = Field(default=True, description="Whether answer extraction is enabled.")
    tag: str = Field(default=DEFAULT_ANSWER_TAG)
    instruction: str = Field(default=DEFAULT_ANSWER_INSTRUCTION)


# ============ Post-process Configuration ============

class BasePostProcessConfig(BaseConfig):
    method: str = Field(default="majority_voting")

    @staticmethod
    def from_dict(config: Dict, method: str) -> "BasePostProcessConfig":
        if method == "majority_voting":
            return MajorityVotingConfig.from_dict(config)

        raise Exception(f"Error occurred when parsing configuration of postprocess: {method} is not supported for postprocess.")


class BaseVotingConfig(BaseConfig):
    """Base Configuration for voting method"""
    method: str = Field(...)

    @staticmethod
    def from_dict(config: Dict) -> "BaseVotingConfig":
        # Infer method from which config key is present
        valid_methods = ["exact_match", "semantic_clustering", "llm_judge"]
        found_methods = [m for m in valid_methods if m in config]

        if len(found_methods) > 1:
            raise ValueError(
                f"Multiple voting methods configured: {found_methods}. "
                "Please uncomment only ONE method in majority_voting config."
            )

        method = found_methods[0] if found_methods else DEFAULT_VOTING_METHOD

        addition_config = config.get(method, {})
        total_config = {**config, **addition_config, "method": method}

        # get specific config according to method
        if method == "exact_match":
            return ExactMatchVotingConfig(**total_config)

        if method == "semantic_clustering":
            return SemanticClusteringVotingConfig(**total_config)

        if method == "llm_judge":
            return LLMJudgeVotingConfig(**total_config)

        raise Exception(f"Error occurred when parsing configuration of majority_voting: method {method} is not supported for majority_voting.")

    def update(self, config: Dict):
        # Infer method from config structure
        method = self.method
        if "exact_match" in config:
            method = "exact_match"
        elif "semantic_clustering" in config:
            method = "semantic_clustering"
        elif "llm_judge" in config:
            method = "llm_judge"
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


# ============ Evaluation Configuration ============

class BaseComparisonConfig(BaseConfig):
    """Base Configuration for answer comparison method"""
    method: str = Field(default=DEFAULT_COMPARISON_METHOD)

    @staticmethod
    def from_dict(config: Dict) -> "BaseComparisonConfig":
        # Infer method from which config key is present
        valid_methods = ["exact_match", "semantic", "llm_judge"]
        found_methods = [m for m in valid_methods if m in config]

        if len(found_methods) > 1:
            raise ValueError(
                f"Multiple comparison methods configured: {found_methods}. "
                "Please uncomment only ONE method in answer_comparison config."
            )

        method = found_methods[0] if found_methods else DEFAULT_COMPARISON_METHOD

        addition_config = config.get(method, {})
        total_config = {**config, **addition_config, "method": method}

        # get specific config according to method
        if method == "exact_match":
            return ExactMatchComparisonConfig(**total_config)

        if method == "semantic":
            return SemanticComparisonConfig(**total_config)

        if method == "llm_judge":
            return LLMJudgeComparisonConfig(**total_config)

        raise Exception(f"Error occurred when parsing configuration of answer_comparison: method {method} is not supported for answer_comparison.")

    def update(self, config: Dict):
        # Infer method from config structure
        method = self.method
        if "exact_match" in config:
            method = "exact_match"
        elif "semantic" in config:
            method = "semantic"
        elif "llm_judge" in config:
            method = "llm_judge"
        addition_config = config.get(method, {})
        if addition_config:
            super().update({**config, **addition_config})


class ExactMatchComparisonConfig(BaseComparisonConfig):
    numeric_tolerance: float = Field(default=1e-3)


class SemanticComparisonConfig(BaseComparisonConfig):
    model_path: str = Field(default="BAAI/bge-m3")
    device: str = Field(default="cuda:0", description="CUDA device (e.g., 'cuda:0')")
    similarity_threshold: float = Field(default=0.85)


class LLMJudgeComparisonConfig(BaseComparisonConfig):
    temperature: float = Field(default=0.3)


class EvaluationConfig(BaseConfig):
    """Configuration for evaluation"""
    batch_size: int = Field(...)
    input_instruction: Optional[str] = Field(default="", description="input instruction (inherited from task config)")
    output_instruction: Optional[str] = Field(default="", description="output instruction (inherited from task config)")
    answer_comparison_config: BaseComparisonConfig = Field(default=ExactMatchComparisonConfig(method="exact_match"))
    inference: InferenceConfig = Field(default=InferenceConfig())
    scoring: InferenceConfig = Field(default=InferenceConfig(temperature=1.2, n=8))

    @classmethod
    def from_dict(cls, config: Dict) -> "EvaluationConfig":
        answer_comparison_config = BaseComparisonConfig.from_dict(config["answer_comparison"])
        total_config = {**config, **{"answer_comparison_config": answer_comparison_config}}
        instance = cls(**total_config)
        return instance

    def update(self, config: Dict):
        answer_comparison_config_dict = config.get("answer_comparison", {})
        if answer_comparison_config_dict:
            config = {**config, **{"answer_comparison_config": answer_comparison_config_dict}}
        super().update(config)


# ============ Rewrite Configuration ============

class BaseRewriteConfig(BaseConfig):
    method: str = Field(default=DEFAULT_REWRITE_METHOD)
    input_instruction: Optional[str] = Field(default="", description="input instruction (inherited from task config)")
    output_instruction: Optional[str] = Field(default="", description="output instruction (inherited from task config)")
    batch_size: int = Field(default=5, gt=0, description="batch size for rewriting (inherited from task config)")

    @staticmethod
    def from_dict(config: Dict) -> "BaseRewriteConfig":
        # Infer method from which config key is present
        valid_methods = ["difficulty_adjust"]
        found_methods = [m for m in valid_methods if m in config]

        if len(found_methods) > 1:
            raise ValueError(
                f"Multiple rewrite methods configured: {found_methods}. "
                "Please uncomment only ONE method in rewrite config."
            )

        method = found_methods[0] if found_methods else DEFAULT_REWRITE_METHOD

        addition_config = config.get(method, {})
        total_config = {**config, **addition_config, "method": method}

        # get specific config according to method
        if method == "difficulty_adjust":
            instance = DifficultyAdjustRewriteConfig(**total_config)
            return instance

        raise Exception(f"Error occurred when parsing configuration of rewrite: method {method} is not supported for rewrite.")

    def update(self, config: Dict):
        # Infer method from config structure
        method = self.method
        if "difficulty_adjust" in config:
            method = "difficulty_adjust"
        addition_config = config.get(method, {})
        if addition_config:
            super().update({**config, **addition_config})


class DifficultyAdjustRewriteConfig(BaseRewriteConfig):
    easier_temperature: float = Field(default=DEFAULT_EASIER_TEMPERATURE)
    harder_temperature: float = Field(default=DEFAULT_HARDER_TEMPERATURE)


# ============ Translation Configuration ============

class TranslationConfig(BaseConfig):
    """Configuration for translating generated dataset to target language"""
    language: str = Field(default="english", description="Target language for the final dataset (e.g., 'english', 'arabic')")
    model_path: Optional[str] = Field(default=None, description="Translation model path (auto-determined based on language if not specified)")
    max_tokens: int = Field(default=256, description="Maximum tokens for translation generation")

    @classmethod
    def from_dict(cls, config: Dict) -> "TranslationConfig":
        try:
            return cls(**config)
        except Exception as e:
            raise Exception(f"Error occurred when parsing translation configuration: {str(e)}")


# ============ Global Configuration ============

class SDGSConfig(BaseConfig):
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

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "SDGSConfig":
        """Load configuration from a dictionary."""
        # Get global device
        global_device = config_dict.get("device", "cuda:0")
        task_config = SDGSTaskConfig.from_dict(config_dict["task"])

        # Inject global device into parsing config if text.local exists
        if task_config.text and task_config.text.local and task_config.text.local.parsing:
            task_config.text.local.parsing.device = global_device

        # Inject output_dir and device into image config if exists
        if task_config.image:
            if task_config.image.local:
                task_config.image.local.output_dir = config_dict["output_dir"]
                if task_config.image.local.parsing:
                    task_config.image.local.parsing.device = global_device
            if task_config.image.web:
                task_config.image.web.output_dir = config_dict["output_dir"]

        generator_config = ModelConfig.from_dict(config_dict["llm"])
        base_model_config = ModelConfig.from_dict(config_dict["base_model"])

        # Inject global device into base_model if it's a LocalModelConfig
        if base_model_config.provider == "local" and isinstance(base_model_config.config, LocalModelConfig):
            base_model_config.config.device = global_device

        answer_config = AnswerExtractionConfig(**config_dict["answer_extraction"])
        postprocess_config = PostProcessConfig.from_dict(config_dict["postprocess"])

        # Get instructions from task-level config
        task_dict = config_dict["task"]
        input_instruction = task_dict.get("input_instruction") or ""
        output_instruction = task_dict.get("output_instruction") or ""

        # Inject instructions and batch_size into evaluation config dict before parsing
        evaluation_config_dict = config_dict["evaluation"].copy()
        evaluation_config_dict["input_instruction"] = input_instruction
        evaluation_config_dict["output_instruction"] = output_instruction
        batch_size = task_dict.get("batch_size")
        if batch_size and "batch_size" not in evaluation_config_dict:
            evaluation_config_dict["batch_size"] = batch_size
        evaluation_config = EvaluationConfig.from_dict(evaluation_config_dict)

        # Inject instructions and batch_size into rewrite config dict before parsing
        rewrite_config_dict = config_dict["rewrite"].copy()
        rewrite_config_dict["input_instruction"] = input_instruction
        rewrite_config_dict["output_instruction"] = output_instruction
        batch_size = task_dict.get("batch_size")
        if batch_size:
            rewrite_config_dict["batch_size"] = batch_size
        rewrite_config = BaseRewriteConfig.from_dict(rewrite_config_dict)

        translation_config = TranslationConfig.from_dict(config_dict.get("translation", {}))

        return cls(
            device=config_dict.get("device", "cuda:0"),
            output_dir=config_dict["output_dir"],
            export_format=config_dict.get("export_format", "jsonl"),
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
