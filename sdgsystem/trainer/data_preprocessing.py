import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _read_jsonl(jsonl_path: str) -> list[dict]:
    """
    Read samples from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of sample dictionaries
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    if not samples:
        raise ValueError(f"No valid samples found in {jsonl_path}")

    logger.info(f"Loaded {len(samples)} samples from {jsonl_path}")
    return samples


def _save_parquet(samples: list[dict], output_path: str) -> str:
    """
    Save samples to Parquet file.

    Args:
        samples: List of sample dictionaries
        output_path: Output Parquet file path

    Returns:
        Path to saved file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(samples)
    df.to_parquet(path, index=False)

    logger.info(f"Saved {len(samples)} samples to {output_path}")
    return str(path)


def jsonl_to_parquet(
    jsonl_path: str,
    output_dir: Optional[str] = None,
) -> str:
    """
    Convert JSONL file to Parquet format.

    Simply converts format without modifying keys. verl will use
    prompt_key/response_key from config to read the correct columns.

    Args:
        jsonl_path: Path to JSONL file
        output_dir: Directory for output Parquet file (default: same as input)

    Returns:
        Path to output Parquet file
    """
    if output_dir is None:
        output_dir = Path(jsonl_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_jsonl(jsonl_path)

    output_name = Path(jsonl_path).stem
    output_path = str(output_dir / f"{output_name}.parquet")
    return _save_parquet(samples, output_path)


def prepare_grpo_data(
    train_jsonl: str,
    val_jsonl: Optional[str] = None,
    output_dir: Optional[str] = None,
    prompt_key: str = "input",
    response_key: Optional[str] = "output",
    data_source: Optional[str] = None,
    use_reward_model: bool = False,
) -> tuple[str, Optional[str]]:
    """
    Prepare training and validation data for GRPO (Group Relative Policy Optimization).

    Converts JSONL files to Parquet format expected by verl GRPO trainer.
    GRPO only needs prompts - responses are generated via rollout during training.

    Two reward modes:
    1. Rule-based reward (data_source or custom_reward_function):
       - response_key specifies ground truth field for reward verification
       - Ground truth stored in parquet "reward_model" field
    2. Reward model:
       - No ground truth needed, reward model scores generated responses directly
       - Set response_key=None and use_reward_model=True
       - Prompts are converted to chat template format (list of messages)

    Input JSONL format (same as SFT):
        {"input": "question", "output": "answer", ...}

    Output Parquet format (rule-based):
        {"prompt": "question", "data_source": "openai/gsm8k", "reward_model": {"ground_truth": "answer"}, "extra_info": {...}}

    Output Parquet format (reward model):
        {"prompt": [{"role": "user", "content": "question"}], "extra_info": {...}}

    Args:
        train_jsonl: Path to training JSONL file
        val_jsonl: Optional path to validation JSONL file
        output_dir: Directory for output Parquet files (default: same as input)
        prompt_key: Key name for prompt in JSONL (default: "input")
        response_key: Key name for ground truth in JSONL (default: "output"), None if using reward model
        data_source: Dataset source identifier for reward function routing
        use_reward_model: Whether using a reward model (requires chat template format)

    Returns:
        Tuple of (train_parquet_path, val_parquet_path or None)
    """
    if output_dir is None:
        output_dir = Path(train_jsonl).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert training data
    train_samples = _read_jsonl(train_jsonl)
    train_converted = []
    for i, sample in enumerate(train_samples):
        if prompt_key not in sample:
            logger.warning(f"Train sample {i} missing '{prompt_key}' key, skipping")
            continue

        # Format prompt based on reward mode
        prompt_content = sample[prompt_key]
        if use_reward_model:
            # Reward model requires chat template format (list of messages)
            prompt_value = [{"role": "user", "content": prompt_content}]
        else:
            # Rule-based reward uses plain string
            prompt_value = prompt_content

        converted = {
            "prompt": prompt_value,
        }

        # Add data_source for reward function routing
        if data_source:
            converted["data_source"] = data_source

        # Add reward_model field (required by verl)
        if use_reward_model:
            # Reward model mode: indicate style="model" (no ground truth needed)
            converted["reward_model"] = {"style": "model"}
        elif response_key and response_key in sample:
            # Rule-based reward: include ground truth for verification
            converted["reward_model"] = {"ground_truth": sample[response_key]}

        # Preserve additional metadata
        exclude_keys = [prompt_key] + ([response_key] if response_key else [])
        extra_info = {k: v for k, v in sample.items() if k not in exclude_keys}
        if extra_info:
            converted["extra_info"] = extra_info

        train_converted.append(converted)

    train_name = Path(train_jsonl).stem
    train_parquet = str(output_dir / f"{train_name}_grpo.parquet")
    train_path = _save_parquet(train_converted, train_parquet)

    # Convert validation data if provided
    val_path = None
    if val_jsonl:
        val_samples = _read_jsonl(val_jsonl)
        val_converted = []
        for i, sample in enumerate(val_samples):
            if prompt_key not in sample:
                logger.warning(f"Val sample {i} missing '{prompt_key}' key, skipping")
                continue

            # Format prompt based on reward mode
            prompt_content = sample[prompt_key]
            if use_reward_model:
                # Reward model requires chat template format (list of messages)
                prompt_value = [{"role": "user", "content": prompt_content}]
            else:
                # Rule-based reward uses plain string
                prompt_value = prompt_content

            converted = {
                "prompt": prompt_value,
            }

            if data_source:
                converted["data_source"] = data_source

            # Add reward_model field (required by verl)
            if use_reward_model:
                # Reward model mode: indicate style="model" (no ground truth needed)
                converted["reward_model"] = {"style": "model"}
            elif response_key and response_key in sample:
                # Rule-based reward: include ground truth for verification
                converted["reward_model"] = {"ground_truth": sample[response_key]}

            exclude_keys = [prompt_key] + ([response_key] if response_key else [])
            extra_info = {k: v for k, v in sample.items() if k not in exclude_keys}
            if extra_info:
                converted["extra_info"] = extra_info

            val_converted.append(converted)

        val_name = Path(val_jsonl).stem
        val_parquet = str(output_dir / f"{val_name}_grpo.parquet")
        val_path = _save_parquet(val_converted, val_parquet)

    return train_path, val_path
