import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Callable

from .config import SFTConfig, GRPOConfig, get_config_class
from .data_preprocessing import jsonl_to_parquet, prepare_grpo_data
from .methods.sft import SFTMethod
from .methods.grpo import GRPOMethod

logger = logging.getLogger(__name__)


class TrainingLauncher:
    """Launch training jobs using verl framework."""

    def __init__(self, verl_path: Optional[str] = None):
        """
        Initialize the training launcher.

        Args:
            verl_path: Optional path to verl installation. If None, assumes verl is installed.
        """
        self.verl_path = verl_path
        self._process: Optional[subprocess.Popen] = None
        self._cancelled = False

    def cancel(self) -> bool:
        """
        Cancel the running training process.

        Returns:
            True if process was terminated, False if no process running.
        """
        self._cancelled = True
        if self._process and self._process.poll() is None:
            logger.info("Terminating training process...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process did not terminate, killing...")
                self._process.kill()
                self._process.wait()
            logger.info("Training process terminated")
            return True
        return False

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def run_sft(
        self,
        config: SFTConfig,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        Run SFT training.

        Args:
            config: SFT configuration
            log_callback: Optional callback for each log line

        Returns:
            Process return code (0 = success)
        """
        # Validate config
        is_valid, error = SFTMethod.validate_config(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error}")

        # Convert data to Parquet format (save next to source JSONL)
        logger.info("Converting dataset to Parquet format...")
        train_parquet = jsonl_to_parquet(config.data.train_files)
        val_parquet = jsonl_to_parquet(config.data.val_files) if config.data.val_files else None
        logger.info(f"Training data: {train_parquet}")
        if val_parquet:
            logger.info(f"Validation data: {val_parquet}")

        # Build command
        cmd = SFTMethod.build_command(config, train_parquet, val_parquet)
        logger.info(f"Training command: {' '.join(cmd)}")

        # Setup log file
        log_dir = Path(config.trainer.default_local_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "training.log"

        # Set environment
        env = os.environ.copy()
        if self.verl_path:
            env["PYTHONPATH"] = f"{self.verl_path}:{env.get('PYTHONPATH', '')}"

        # Set Wandb API key if configured
        if config.wandb:
            # Use API key from config, or fall back to environment variable
            api_key = config.wandb.api_key or os.environ.get("WANDB_API_KEY")
            if api_key:
                env["WANDB_API_KEY"] = api_key
                logger.info("Wandb logging enabled")
            else:
                logger.warning("Wandb config provided but no API key found (neither in config nor WANDB_API_KEY env var)")

        # Run training
        logger.info("Starting training...")
        self._cancelled = False
        with open(log_file, "w") as lf:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in self._process.stdout:
                if self._cancelled:
                    break
                lf.write(line)
                lf.flush()
                if log_callback:
                    log_callback(line.rstrip())
                else:
                    print(line, end="")

            self._process.wait()

        return_code = self._process.returncode
        self._process = None

        if self._cancelled:
            logger.info("Training was cancelled")
            return -1
        elif return_code == 0:
            logger.info("Training completed successfully")
        else:
            logger.error(f"Training failed with return code {return_code}")

        return return_code

    def run_grpo(
        self,
        config: GRPOConfig,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        Run GRPO training.

        Args:
            config: GRPO configuration
            log_callback: Optional callback for each log line

        Returns:
            Process return code (0 = success)
        """
        # Validate config
        is_valid, error = GRPOMethod.validate_config(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error}")

        # Auto-configure Weave if enabled
        if config.wandb and config.wandb.enable_weave:
            if not config.rollout.trace or config.rollout.trace.backend != "weave":
                from .config import TraceConfig
                config.rollout.trace = TraceConfig(backend="weave")
                logger.info("Auto-configured rollout.trace.backend='weave' for Weave tracing")

        # Convert data to Parquet format
        logger.info("Converting dataset to Parquet format...")
        # For reward model, no ground truth needed; for rule-based reward, use response_key
        use_reward_model = config.reward.reward_model and config.reward.reward_model.enable
        response_key = None if use_reward_model else config.data.response_key

        # Determine data_source: use configured value, or "custom" for custom reward functions
        data_source = config.reward.data_source
        if not data_source and config.reward.custom_reward_function:
            data_source = "custom"

        train_parquet, val_parquet = prepare_grpo_data(
            train_jsonl=config.data.train_files,
            val_jsonl=config.data.val_files,
            output_dir=None,
            prompt_key=config.data.prompt_key,
            response_key=response_key,
            data_source=data_source,
            use_reward_model=use_reward_model,
        )
        logger.info(f"Training data: {train_parquet}")
        if val_parquet:
            logger.info(f"Validation data: {val_parquet}")

        # Build command
        cmd = GRPOMethod.build_command(config, train_parquet, val_parquet)
        logger.info(f"Training command: {' '.join(cmd)}")

        # Setup log file
        log_dir = Path(config.trainer.default_local_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "training.log"

        # Set environment
        env = os.environ.copy()
        if self.verl_path:
            env["PYTHONPATH"] = f"{self.verl_path}:{env.get('PYTHONPATH', '')}"

        # Set Wandb API key if configured
        if config.wandb:
            # Use API key from config, or fall back to environment variable
            api_key = config.wandb.api_key or os.environ.get("WANDB_API_KEY")
            if api_key:
                env["WANDB_API_KEY"] = api_key
                logger.info("Wandb logging enabled")
            else:
                logger.warning("Wandb config provided but no API key found (neither in config nor WANDB_API_KEY env var)")

        # Run training
        logger.info("Starting GRPO training...")
        self._cancelled = False
        with open(log_file, "w") as lf:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in self._process.stdout:
                if self._cancelled:
                    break
                lf.write(line)
                lf.flush()
                if log_callback:
                    log_callback(line.rstrip())
                else:
                    print(line, end="")

            self._process.wait()

        return_code = self._process.returncode
        self._process = None

        if self._cancelled:
            logger.info("GRPO training was cancelled")
            return -1
        elif return_code == 0:
            logger.info("GRPO training completed successfully")
        else:
            logger.error(f"GRPO training failed with return code {return_code}")

        return return_code


def run_training_from_config(
    config_dict: dict,
    verl_path: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Run training from a configuration dictionary.

    Args:
        config_dict: Configuration dictionary (nested verl-style format)
        verl_path: Optional path to verl installation
        log_callback: Optional callback for each log line

    Returns:
        Process return code (0 = success)
    """
    method = config_dict.get("method")
    if not method:
        raise ValueError("Config must specify 'method' field")

    # Get config class and create config
    config_class = get_config_class(method)
    config = config_class(**config_dict)

    # Launch training
    launcher = TrainingLauncher(verl_path=verl_path)

    if method == "sft":
        return launcher.run_sft(config, log_callback)
    elif method == "grpo":
        return launcher.run_grpo(config, log_callback)
    else:
        raise NotImplementedError(f"Training method '{method}' not yet implemented")
