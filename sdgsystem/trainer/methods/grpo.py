import logging
from typing import Optional

from ..config import GRPOConfig

logger = logging.getLogger(__name__)


class GRPOMethod:
    """GRPO training method using verl's main_ppo."""

    @staticmethod
    def validate_config(config: GRPOConfig) -> tuple[bool, Optional[str]]:
        """
        Validate GRPO configuration.

        Args:
            config: GRPO configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if not config.data.train_files:
            return False, "data.train_files is required"

        if not config.data.val_files:
            return False, "data.val_files is required"

        if not config.model.path:
            return False, "model.path is required"

        if not config.trainer.default_local_dir:
            return False, "trainer.default_local_dir is required"

        # Validate learning rate
        if config.actor.optim.lr <= 0:
            return False, "actor.optim.lr must be positive"

        # Validate rollout tensor parallel size
        if config.rollout.tensor_model_parallel_size > config.trainer.n_gpus_per_node:
            return False, f"rollout.tensor_model_parallel_size ({config.rollout.tensor_model_parallel_size}) cannot exceed trainer.n_gpus_per_node ({config.trainer.n_gpus_per_node})"

        # Validate GPU memory utilization
        if config.rollout.gpu_memory_utilization <= 0 or config.rollout.gpu_memory_utilization > 1:
            return False, "rollout.gpu_memory_utilization must be between 0 and 1"

        # Validate reward configuration
        # Must have one of: data_source, custom_reward_function, or reward_model
        has_data_source = bool(config.reward.data_source)
        has_custom_fn = config.reward.custom_reward_function is not None
        has_reward_model = config.reward.reward_model is not None and config.reward.reward_model.enable

        reward_options = sum([has_data_source, has_custom_fn, has_reward_model])
        if reward_options == 0:
            return False, "Reward configuration required: specify reward.data_source, reward.custom_reward_function, or reward.reward_model"

        if has_reward_model and (has_data_source or has_custom_fn):
            return False, "reward.reward_model cannot be combined with data_source or custom_reward_function"

        return True, None

    @staticmethod
    def build_command(
        config: GRPOConfig,
        train_parquet: str,
        val_parquet: Optional[str] = None,
    ) -> list[str]:
        """
        Build the python command for GRPO training.

        GRPO uses verl.trainer.main_ppo directly (not torchrun).

        Args:
            config: GRPO configuration
            train_parquet: Path to training Parquet file
            val_parquet: Optional path to validation Parquet file

        Returns:
            List of command arguments for subprocess
        """
        # GRPO uses python directly with main_ppo
        cmd = [
            "python3",
            "-m", "verl.trainer.main_ppo",
        ]

        # Get verl arguments from config
        verl_args = config.to_verl_args(train_parquet, val_parquet)

        # Convert to command line arguments
        for key, value in verl_args.items():
            if isinstance(value, bool):
                cmd.append(f"{key}={str(value).lower()}")
            elif isinstance(value, (int, float)):
                cmd.append(f"{key}={value}")
            else:
                cmd.append(f"{key}={value}")

        return cmd
