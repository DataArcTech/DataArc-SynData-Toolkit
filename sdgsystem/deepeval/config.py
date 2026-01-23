"""
Configuration schemas for DeepEval-based evaluation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from ..configs.config import BaseConfig, ModelConfig


# ============ Rubric Configuration ============

class RubricItem(BaseModel):
    """Single rubric scoring range for G-Eval metrics"""
    score_range: List[int] = Field(..., description="Score range [min, max] on 0-10 scale, e.g., [0, 2]")
    expected_outcome: str = Field(..., description="Expected outcome description for this score range")


# Default rubrics
DEFAULT_CORRECTNESS_RUBRIC = [
    RubricItem(score_range=[0, 2], expected_outcome="Factually incorrect or completely wrong answer"),
    RubricItem(score_range=[3, 5], expected_outcome="Partially correct with significant errors or omissions"),
    RubricItem(score_range=[6, 8], expected_outcome="Mostly correct with minor errors or missing details"),
    RubricItem(score_range=[9, 10], expected_outcome="Fully correct and complete answer")
]

DEFAULT_FORMAT_COMPLIANCE_RUBRIC = [
    RubricItem(score_range=[0, 2], expected_outcome="Output does not follow the required format at all"),
    RubricItem(score_range=[3, 5], expected_outcome="Output partially follows format but missing key elements"),
    RubricItem(score_range=[6, 8], expected_outcome="Output follows format with minor deviations"),
    RubricItem(score_range=[9, 10], expected_outcome="Output perfectly follows the required format")
]


# ============ Metric Configurations ============

class CorrectnessMetricConfig(BaseConfig):
    """Configuration for Answer Correctness metric using G-Eval"""
    enabled: bool = Field(default=True, description="Whether this metric is enabled")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Passing threshold for correctness (0-1 scale)")
    criteria: str = Field(
        default="Determine whether the actual output is factually correct based on the expected output.",
        description="Evaluation criteria"
    )
    evaluation_steps: List[str] = Field(
        default=[
            "Compare actual output directly with expected output for factual accuracy",
            "Check if all key elements from expected output are present in actual output",
            "Assess discrepancies in details, values, or information",
            "Penalize factual errors and significant omissions"
        ],
        description="Evaluation steps for G-Eval (customizable)"
    )
    rubric: List[RubricItem] = Field(
        default=DEFAULT_CORRECTNESS_RUBRIC,
        description="Rubric for discrete scoring on 0-10 scale (customizable)"
    )


class FormatComplianceMetricConfig(BaseConfig):
    """Configuration for Format Compliance metric using custom G-Eval"""
    enabled: bool = Field(default=True, description="Whether this metric is enabled")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Passing threshold for format compliance (0-1 scale)")
    criteria: str = Field(
        default="Evaluate whether the actual output follows the output instruction format.",
        description="Evaluation criteria (customizable)"
    )
    evaluation_steps: List[str] = Field(
        default=[
            "Identify the output format requirements from the output instruction",
            "Check if the actual output structure matches the required format",
            "Verify all required elements or sections are present",
            "Assess proper formatting and syntax compliance"
        ],
        description="Evaluation steps for format compliance (customizable)"
    )
    rubric: List[RubricItem] = Field(
        default=DEFAULT_FORMAT_COMPLIANCE_RUBRIC,
        description="Rubric for discrete scoring on 0-10 scale (customizable)"
    )


class PairwiseMetricConfig(BaseConfig):
    """Configuration for Pairwise Preference metric using G-Eval.

    Compares post-trained model output (Response A) vs base model output (Response B).
    Uses discrete scoring: 10 = A wins, 5 = tie, 0 = B wins.
    """
    enabled: bool = Field(default=True, description="Whether this metric is enabled")
    base_model: Optional[ModelConfig] = Field(
        default=None,
        description="Base model configuration for comparison (required if pairwise is enabled)"
    )
    criteria: str = Field(
        default="Compare Response A (post-trained model) vs Response B (base model) to determine which is better.",
        description="Evaluation criteria"
    )
    evaluation_steps: List[str] = Field(
        default=[
            "Compare factual and logical correctness: any mistakes make that response worse.",
            "Check for hallucinations or unsupported claims: fabricating information makes that response worse.",
            "Evaluate instruction/format compliance: violations are penalized heavily.",
            "Assess helpfulness, completeness, and clarity.",
            "Output exactly 10 if Response A wins, 0 if Response B wins, or 5 if tie. No other scores allowed."
        ],
        description="Evaluation steps for pairwise comparison (customizable)"
    )


# ============ Main Configuration ============

class InferenceConfig(BaseConfig):
    """Configuration for model inference during evaluation"""
    temperature: float = Field(default=0.0, ge=0.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Maximum tokens to generate")
    output_instruction: str = Field(
        default="",
        description="Output instruction appended to input during inference (used for format compliance evaluation)"
    )


class DatasetConfig(BaseConfig):
    """Configuration for evaluation dataset"""
    path: str = Field(..., description="Path to test dataset (JSONL format with 'input' and 'output' fields)")


class OutputConfig(BaseConfig):
    """Configuration for evaluation output"""
    dir: str = Field(default="./deepeval_results", description="Output directory for results")


class DeepEvalConfig(BaseConfig):
    """Main configuration for DeepEval-based post-training evaluation

    Supports three metrics (all enabled by default):
    1. Answer Correctness - G-Eval comparing actual vs expected output
    2. Pairwise Preference - G-Eval comparing base vs post-trained model
    3. Format Compliance - G-Eval checking output format adherence
    """
    # Dataset configuration
    dataset: DatasetConfig = Field(..., description="Test dataset configuration")

    # Post-trained model to evaluate
    post_trained_model: ModelConfig = Field(..., description="Post-trained model configuration")

    # Judge model (uses DeepEval built-in API, reads OPENAI_API_KEY and OPENAI_BASE_URL from .env)
    judge_model: str = Field(
        default="gpt-4o",
        description="Judge LLM model name for G-Eval (e.g., gpt-4o, gpt-4o-mini)"
    )

    # Inference settings
    inference: InferenceConfig = Field(default_factory=InferenceConfig, description="Inference configuration")

    # Metrics configuration (all enabled by default, customizable)
    correctness: CorrectnessMetricConfig = Field(
        default_factory=CorrectnessMetricConfig,
        description="Answer Correctness metric configuration"
    )
    pairwise: PairwiseMetricConfig = Field(
        default_factory=PairwiseMetricConfig,
        description="Pairwise Preference metric configuration"
    )
    format_compliance: FormatComplianceMetricConfig = Field(
        default_factory=FormatComplianceMetricConfig,
        description="Format Compliance metric configuration"
    )

    # Output configuration
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DeepEvalConfig":
        """Load configuration from YAML file"""
        import yaml

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config: Dict) -> "DeepEvalConfig":
        """Parse configuration from dictionary"""
        try:
            # Parse dataset config
            dataset_config = DatasetConfig(**config["dataset"])

            # Parse post-trained model config
            post_trained_model_config = ModelConfig.from_dict(config["post_trained_model"])

            # Parse inference config
            inference_config = InferenceConfig(**config.get("inference", {}))

            # Parse correctness metric config with custom rubrics
            correctness_dict = config.get("correctness", {})
            if "rubric" in correctness_dict:
                correctness_dict["rubric"] = [RubricItem(**item) for item in correctness_dict["rubric"]]
            correctness_config = CorrectnessMetricConfig(**correctness_dict)

            # Parse pairwise metric config with base_model
            pairwise_dict = config.get("pairwise", {})
            if "base_model" in pairwise_dict and pairwise_dict["base_model"] is not None:
                pairwise_dict["base_model"] = ModelConfig.from_dict(pairwise_dict["base_model"])
            pairwise_config = PairwiseMetricConfig(**pairwise_dict)

            # Parse format compliance metric config with custom rubrics
            format_compliance_dict = config.get("format_compliance", {})
            if "rubric" in format_compliance_dict:
                format_compliance_dict["rubric"] = [RubricItem(**item) for item in format_compliance_dict["rubric"]]
            format_compliance_config = FormatComplianceMetricConfig(**format_compliance_dict)

            # Parse output config
            output_config = OutputConfig(**config.get("output", {}))

            # Create main config
            instance = cls(
                dataset=dataset_config,
                post_trained_model=post_trained_model_config,
                judge_model=config.get("judge_model", "gpt-4o"),
                inference=inference_config,
                correctness=correctness_config,
                pairwise=pairwise_config,
                format_compliance=format_compliance_config,
                output=output_config
            )

            # Validate: base_model is required if pairwise is enabled
            if instance.pairwise.enabled and instance.pairwise.base_model is None:
                raise ValueError(
                    "Pairwise metric is enabled but base_model is not configured. "
                    "Please provide base_model in pairwise configuration or set pairwise.enabled to false."
                )

            return instance

        except Exception as e:
            raise Exception(f"Error parsing DeepEval configuration: {str(e)}")
