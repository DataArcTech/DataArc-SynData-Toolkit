import logging
from typing import Optional

from ..config import SFTConfig

logger = logging.getLogger(__name__)


class SFTMethod:
    """SFT training method using verl's fsdp_sft_trainer."""

    @staticmethod
    def validate_config(config: SFTConfig) -> tuple[bool, Optional[str]]:
        """
        Validate SFT configuration.

        Args:
            config: SFT configuration to validate

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
        if config.optim.lr <= 0:
            return False, "optim.lr must be positive"

        # Validate sequence parallel size with GPU count
        if config.ulysses_sequence_parallel_size > config.trainer.n_gpus_per_node:
            return False, f"ulysses_sequence_parallel_size ({config.ulysses_sequence_parallel_size}) cannot exceed n_gpus_per_node ({config.trainer.n_gpus_per_node})"

        return True, None

    @staticmethod
    def build_command(
        config: SFTConfig,
        train_parquet: str,
        val_parquet: Optional[str] = None,
    ) -> list[str]:
        """
        Build the torchrun command for SFT training.

        Args:
            config: SFT configuration
            train_parquet: Path to training Parquet file
            val_parquet: Optional path to validation Parquet file

        Returns:
            List of command arguments for subprocess
        """
        # Build torchrun command
        cmd = [
            "torchrun",
            "--standalone",
            f"--nnodes={config.trainer.nnodes}",
            f"--nproc_per_node={config.trainer.n_gpus_per_node}",
            "-m", "verl.trainer.fsdp_sft_trainer",
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
