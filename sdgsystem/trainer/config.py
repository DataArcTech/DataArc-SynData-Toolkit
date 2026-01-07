import logging
import yaml
from enum import Enum
from typing import Optional, Literal, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrainingMethod(str, Enum):
    """Supported training methods."""
    SFT = "sft"
    GRPO = "grpo"


# ============== Verl Config Models ==============

class DataConfig(BaseModel):
    """Data configuration."""
    train_files: str = Field(..., description="Path to training dataset")
    val_files: Optional[str] = Field(default=None, description="Path to validation dataset")
    train_batch_size: int = Field(default=32, gt=0)
    micro_batch_size_per_gpu: int = Field(default=4, gt=0)
    max_length: Optional[int] = Field(default=2048, gt=0, description="Max sequence length (SFT)")
    max_prompt_length: Optional[int] = Field(default=512, gt=0, description="Max prompt length (GRPO)")
    max_response_length: Optional[int] = Field(default=1024, gt=0, description="Max response length (GRPO)")
    truncation: Literal["left", "right", "error"] = Field(default="right")
    filter_overlong_prompts: bool = Field(default=True)
    prompt_key: str = Field(default="input", description="Key for prompt in source JSONL")
    response_key: str = Field(default="output", description="Key for response in source JSONL")


class FSDPConfig(BaseModel):
    """FSDP configuration."""
    param_offload: bool = Field(default=False)
    optimizer_offload: bool = Field(default=False)


class ModelConfig(BaseModel):
    """Model configuration."""
    path: str = Field(..., description="Model path or HuggingFace ID")
    enable_gradient_checkpointing: bool = Field(default=True)
    use_remove_padding: bool = Field(default=True)
    use_liger: bool = Field(default=False)
    lora_rank: int = Field(default=0, description="LoRA rank (0 = disabled)")
    lora_alpha: int = Field(default=16)
    target_modules: str = Field(default="all-linear")


class OptimConfig(BaseModel):
    """Optimizer configuration."""
    lr: float = Field(default=1e-5, gt=0)


class TrainerConfig(BaseModel):
    """Trainer configuration."""
    project_name: str = Field(default="sdg-training")
    experiment_name: Optional[str] = Field(default=None)
    logger: str = Field(default="console,wandb")
    total_epochs: int = Field(default=3, gt=0)
    save_freq: int = Field(default=100)
    test_freq: int = Field(default=-1)
    nnodes: int = Field(default=1, gt=0)
    n_gpus_per_node: int = Field(default=1, gt=0)
    default_local_dir: str = Field(default="./checkpoints")
    val_before_train: bool = Field(default=False, description="Run validation before training (uses extra memory)")
    critic_warmup: int = Field(default=0)
    wandb_proxy: Optional[str] = Field(default=None, description="Wandb proxy URL (e.g., http://proxy.example.com:8080)")


class ActorOptimConfig(BaseModel):
    """Actor optimizer config."""
    lr: float = Field(default=1e-6, gt=0)


class ActorConfig(BaseModel):
    """Actor configuration (GRPO)."""
    optim: ActorOptimConfig = Field(default_factory=ActorOptimConfig)
    ppo_mini_batch_size: int = Field(default=64, gt=0)
    ppo_micro_batch_size_per_gpu: int = Field(default=8, gt=0)
    use_kl_loss: bool = Field(default=True)
    kl_loss_coef: float = Field(default=0.001)
    kl_loss_type: str = Field(default="low_var_kl")
    entropy_coeff: float = Field(default=0.0)
    fsdp_config: FSDPConfig = Field(default_factory=FSDPConfig)


class TraceConfig(BaseModel):
    """Trace configuration for Weave."""
    backend: Optional[str] = Field(default=None, description="Trace backend (e.g., 'weave')")


class RolloutConfig(BaseModel):
    """Rollout configuration (GRPO)."""
    n: int = Field(default=5, gt=0, description="Number of rollout samples per prompt")
    gpu_memory_utilization: float = Field(default=0.6, gt=0, le=1)
    tensor_model_parallel_size: int = Field(default=1, gt=0)
    mode: str = Field(default="async", description="Rollout mode: 'async' (required for vLLM)")
    trace: Optional[TraceConfig] = Field(default=None, description="Trace configuration for Weave")


class RefFSDPConfig(BaseModel):
    """Reference model FSDP config."""
    param_offload: bool = Field(default=True)


class RefConfig(BaseModel):
    """Reference model configuration (GRPO)."""
    fsdp_config: RefFSDPConfig = Field(default_factory=RefFSDPConfig)


class AlgorithmConfig(BaseModel):
    """Algorithm configuration (GRPO)."""
    adv_estimator: str = Field(default="grpo")
    use_kl_in_reward: bool = Field(default=False)


class CustomRewardFunctionConfig(BaseModel):
    """Custom reward function config."""
    path: str = Field(..., description="Path to reward function file")
    name: str = Field(default="compute_score")


class RewardModelSubConfig(BaseModel):
    """Reward model sub-config."""
    enable: bool = Field(default=False)
    path: Optional[str] = Field(default=None)
    micro_batch_size_per_gpu: int = Field(default=4, gt=0, description="Batch size per GPU for reward model inference")


class RewardConfig(BaseModel):
    """Reward configuration (GRPO)."""
    data_source: Optional[str] = Field(default=None, description="Dataset source for auto reward function")
    custom_reward_function: Optional[CustomRewardFunctionConfig] = Field(default=None)
    reward_model: Optional[RewardModelSubConfig] = Field(default=None)


class WandbConfig(BaseModel):
    """Wandb configuration for logging and Weave tracing."""
    api_key: Optional[str] = Field(default=None, description="Wandb API key (optional, can use WANDB_API_KEY env var)")
    entity: Optional[str] = Field(default=None, description="Wandb entity (username or team)")
    enable_weave: bool = Field(default=True, description="Enable Weave tracing (GRPO only, enabled by default)")


# ============== Top-Level Config Classes ==============

class SFTConfig(BaseModel):
    """SFT configuration using nested structure."""
    method: Literal["sft"] = Field(default="sft")
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig = Field(default_factory=OptimConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    ulysses_sequence_parallel_size: int = Field(default=1)
    wandb: Optional[WandbConfig] = Field(default=None, description="Wandb configuration")

    def to_verl_args(self, train_parquet: str, val_parquet: Optional[str] = None) -> dict:
        """Convert to verl trainer command arguments."""
        args = {
            # Data config
            "data.train_files": train_parquet,
            "data.val_files": val_parquet,
            "data.prompt_key": self.data.prompt_key,
            "data.response_key": self.data.response_key,
            "data.max_length": self.data.max_length,
            "data.truncation": self.data.truncation,
            "data.train_batch_size": self.data.train_batch_size,
            "data.micro_batch_size_per_gpu": self.data.micro_batch_size_per_gpu,

            # Model config
            "model.partial_pretrain": self.model.path,
            "model.use_liger": str(self.model.use_liger).lower(),
            "model.enable_gradient_checkpointing": str(self.model.enable_gradient_checkpointing).lower(),

            # Optimizer config
            "optim.lr": self.optim.lr,

            # Trainer config
            "trainer.default_local_dir": self.trainer.default_local_dir,
            "trainer.project_name": self.trainer.project_name,
            "trainer.experiment_name": self.trainer.experiment_name or f"sft-{self.model.path.split('/')[-1]}",
            "trainer.logger": "[" + self.trainer.logger + "]",
            "trainer.total_epochs": self.trainer.total_epochs,
            "trainer.save_freq": self.trainer.save_freq,
            "trainer.nnodes": self.trainer.nnodes,
            "trainer.n_gpus_per_node": self.trainer.n_gpus_per_node,

            # Optimizations
            "use_remove_padding": str(self.model.use_remove_padding).lower(),
            "ulysses_sequence_parallel_size": self.ulysses_sequence_parallel_size,
        }

        # Add wandb proxy if configured (use + prefix to add new field to Hydra config)
        if self.trainer.wandb_proxy:
            args["+trainer.wandb_proxy"] = self.trainer.wandb_proxy

        # Add LoRA config if enabled
        if self.model.lora_rank > 0:
            args.update({
                "model.lora_rank": self.model.lora_rank,
                "model.lora_alpha": self.model.lora_alpha,
                "model.target_modules": self.model.target_modules,
            })

        return args


class GRPOConfig(BaseModel):
    """GRPO configuration using nested structure."""
    method: Literal["grpo"] = Field(default="grpo")
    data: DataConfig
    model: ModelConfig
    actor: ActorConfig = Field(default_factory=ActorConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    ref: RefConfig = Field(default_factory=RefConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    reward: RewardConfig
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    wandb: Optional[WandbConfig] = Field(default=None, description="Wandb configuration")

    def to_verl_args(self, train_parquet: str, val_parquet: Optional[str] = None) -> dict:
        """Convert to verl main_ppo command arguments."""
        args = {
            # Algorithm
            "algorithm.adv_estimator": self.algorithm.adv_estimator,
            "algorithm.use_kl_in_reward": str(self.algorithm.use_kl_in_reward).lower(),

            # Data config
            "data.train_files": train_parquet,
            "data.train_batch_size": self.data.train_batch_size,
            "data.max_prompt_length": self.data.max_prompt_length,
            "data.max_response_length": self.data.max_response_length,
            "data.filter_overlong_prompts": str(self.data.filter_overlong_prompts).lower(),
            "data.truncation": self.data.truncation,

            # Actor/Model config
            "actor_rollout_ref.model.path": self.model.path,
            "actor_rollout_ref.model.use_remove_padding": str(self.model.use_remove_padding).lower(),
            "actor_rollout_ref.model.enable_gradient_checkpointing": str(self.model.enable_gradient_checkpointing).lower(),

            # Actor training config
            "actor_rollout_ref.actor.optim.lr": self.actor.optim.lr,
            "actor_rollout_ref.actor.ppo_mini_batch_size": self.actor.ppo_mini_batch_size,
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": self.actor.ppo_micro_batch_size_per_gpu,
            "actor_rollout_ref.actor.use_kl_loss": str(self.actor.use_kl_loss).lower(),
            "actor_rollout_ref.actor.kl_loss_coef": self.actor.kl_loss_coef,
            "actor_rollout_ref.actor.kl_loss_type": self.actor.kl_loss_type,
            "actor_rollout_ref.actor.entropy_coeff": self.actor.entropy_coeff,

            # Actor FSDP config
            "actor_rollout_ref.actor.fsdp_config.param_offload": str(self.actor.fsdp_config.param_offload).lower(),
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload": str(self.actor.fsdp_config.optimizer_offload).lower(),

            # Rollout config
            "actor_rollout_ref.rollout.name": "vllm",
            "actor_rollout_ref.rollout.n": self.rollout.n,
            "actor_rollout_ref.rollout.mode": self.rollout.mode,
            "actor_rollout_ref.rollout.gpu_memory_utilization": self.rollout.gpu_memory_utilization,
            "actor_rollout_ref.rollout.tensor_model_parallel_size": self.rollout.tensor_model_parallel_size,
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": self.actor.ppo_micro_batch_size_per_gpu,

            # Reference model config
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": self.actor.ppo_micro_batch_size_per_gpu,
            "actor_rollout_ref.ref.fsdp_config.param_offload": str(self.ref.fsdp_config.param_offload).lower(),

            # Trainer config
            "trainer.critic_warmup": self.trainer.critic_warmup,
            "trainer.default_local_dir": self.trainer.default_local_dir,
            "trainer.project_name": self.trainer.project_name,
            "trainer.experiment_name": self.trainer.experiment_name or f"grpo-{self.model.path.split('/')[-1]}",
            "trainer.logger": "[" + self.trainer.logger + "]",
            "trainer.n_gpus_per_node": self.trainer.n_gpus_per_node,
            "trainer.nnodes": self.trainer.nnodes,
            "trainer.total_epochs": self.trainer.total_epochs,
            "trainer.save_freq": self.trainer.save_freq,
            "trainer.test_freq": self.trainer.test_freq,
            "trainer.val_before_train": str(self.trainer.val_before_train).lower(),
        }

        # Add validation data if provided
        if val_parquet:
            args["data.val_files"] = val_parquet

        # Reward configuration
        if self.reward.reward_model and self.reward.reward_model.enable:
            args["reward_model.enable"] = "true"
            args["reward_model.model.path"] = self.reward.reward_model.path
            args["reward_model.micro_batch_size_per_gpu"] = self.reward.reward_model.micro_batch_size_per_gpu
        else:
            args["reward_model.enable"] = "false"
            if self.reward.custom_reward_function:
                args["custom_reward_function.path"] = self.reward.custom_reward_function.path
                args["custom_reward_function.name"] = self.reward.custom_reward_function.name

        # Weave trace configuration (if wandb is configured and weave is enabled)
        if self.wandb and self.wandb.enable_weave:
            if self.rollout.trace and self.rollout.trace.backend:
                args["actor_rollout_ref.rollout.trace.backend"] = self.rollout.trace.backend

        # Add wandb proxy if configured (use + prefix to add new field to Hydra config)
        if self.trainer.wandb_proxy:
            args["+trainer.wandb_proxy"] = self.trainer.wandb_proxy

        # Add LoRA config if enabled
        if self.model.lora_rank > 0:
            args.update({
                "actor_rollout_ref.model.lora_rank": self.model.lora_rank,
                "actor_rollout_ref.model.lora_alpha": self.model.lora_alpha,
                "actor_rollout_ref.model.target_modules": self.model.target_modules,
            })

        return args


def get_config_class(method: str):
    """Get the appropriate config class for a training method."""
    method_map = {
        "sft": SFTConfig,
        "grpo": GRPOConfig,
    }
    if method.lower() not in method_map:
        raise ValueError(f"Unknown training method: {method}. Supported: {list(method_map.keys())}")
    return method_map[method.lower()]


def load_training_config(yaml_path: str) -> Union[SFTConfig, GRPOConfig]:
    """
    Load training configuration from a YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Appropriate config class based on method
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if "method" not in config_dict:
        raise ValueError("YAML config must specify 'method' field")

    config_class = get_config_class(config_dict["method"])
    return config_class(**config_dict)
