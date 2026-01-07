import os
import logging
from typing import List, Dict, Tuple

from ..utils import save_jsonl

logger = logging.getLogger(__name__)


class Dataset:
    """
    A class to represent a dataset consisting of samples, where each sample contains messages with specific roles.
    """
    def __init__(self) -> None:
        """Initialize an empty dataset."""
        self.samples: List[Dict] = []
        self.failed_samples = []

    @classmethod
    def from_list(cls, sample_list: List[Dict]) -> "Dataset":
        """Create a Dataset instance from a list of samples.

        Args:
            sample_list: List of dictionaries containing the samples.

        Returns:
            A new Dataset instance populated with the provided samples.
        """
        instance = cls()
        for sample in sample_list:
            if cls.validate_sample(sample):
                instance.samples.append(sample)
            else:
                instance.failed_samples.append(sample)

        return instance

    @staticmethod
    def validate_sample(sample: Dict) -> bool:
        """Validate if a sample has the correct format.

        For structured generation samples (from Outlines), validation is minimal
        since Outlines guarantees schema compliance. This is primarily a sanity check.

        Args:
            sample: Dictionary containing the sample data.

        Returns:
            bool: True if the sample is valid, False otherwise.
        """
        # Basic data completeness check
        if not sample or not isinstance(sample, Dict):
            return False

        if "input" not in sample or "output" not in sample:
            return False

        return True

    def add_sample(self, sample: Dict) -> str:
        failure_description = ""
        if self.validate_sample(sample):
            self.samples.append(sample)
        else:
            failure_description = f"Invalid sample format: {sample}"
        return failure_description

    def add_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[str]]:
        failed_samples: List[Dict] = []
        failure_descriptions: List[str] = []

        for sample in samples:
            if self.validate_sample(sample):
                self.samples.append(sample)
            else:
                failed_samples.append(sample)
                failure_descriptions.append(f"Invalid sample format: {sample}")
        
        self.failed_samples.extend(failed_samples)

        return failed_samples, failure_descriptions

    def save(self, save_path: str, export_format: str):
        """Save the dataset to a JSONL file.

        Args:
            save_path: Path where the JSONL file should be saved.
        """
        if export_format == "jsonl":
            save_jsonl(self.samples, save_path)
        elif export_format == "json":
            pass
        else:
            raise Exception(f"Export format of {export_format} is not supported.")

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.samples)

    def extend(self, other: "Dataset"):
        """Extend self from other Dataset"""
        self.samples += other.samples
        self.failed_samples += other.failed_samples

    def categorize_by_score(self, scores: List[float]) -> Tuple["Dataset", "Dataset", "Dataset"]:
        """
        Categorize samples into three datasets based on evaluation scores.

        Args:
            scores: List of scores (0.0 to 1.0) corresponding to each sample

        Returns:
            Tuple of (solved_dataset, learnable_dataset, unsolved_dataset):
            - solved_dataset: Samples with score = 1.0 (too easy)
            - learnable_dataset: Samples with 0 < score < 1.0 (appropriate difficulty)
            - unsolved_dataset: Samples with score = 0.0 (too hard)
        """
        if len(scores) != len(self.samples):
            raise ValueError(f"Number of scores ({len(scores)}) must match number of samples ({len(self.samples)})")

        solved_samples = []
        learnable_samples = []
        unsolved_samples = []

        for sample, score in zip(self.samples, scores):
            # Add score to sample metadata
            sample_with_score = sample.copy()
            sample_with_score['score'] = score

            if score == 0.0:
                unsolved_samples.append(sample_with_score)
            elif score == 1.0:
                solved_samples.append(sample_with_score)
            else:  # 0 < score < 1
                learnable_samples.append(sample_with_score)

        logger.info(f"Categorized {len(self.samples)} samples:")
        logger.info(f"  Solved (score=1.0): {len(solved_samples)} samples (too easy)")
        logger.info(f"  Learnable (0<score<1): {len(learnable_samples)} samples (appropriate difficulty)")
        logger.info(f"  Unsolved (score=0.0): {len(unsolved_samples)} samples (too hard)")

        return (
            Dataset.from_list(solved_samples),
            Dataset.from_list(learnable_samples),
            Dataset.from_list(unsolved_samples)
        )

    def save_categorized(self, solved: "Dataset", learnable: "Dataset", unsolved: "Dataset",
                         save_path: str, export_format: str):
        """
        Save all three categorized datasets to separate files.

        Args:
            solved: Dataset with solved samples (score=1.0)
            learnable: Dataset with learnable samples (0<score<1)
            unsolved: Dataset with unsolved samples (score=0.0)
            save_path: Full path for the base output file
            export_format: Export format (jsonl, json, etc.)
        """
        # Extract directory and base name from save_path
        output_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
        base_name = os.path.splitext(os.path.basename(save_path))[0]

        # Save solved dataset
        if len(solved) > 0:
            solved_path = os.path.join(output_dir, f"{base_name}_solved.{export_format}")
            solved.save(solved_path, export_format)
            logger.info(f"Saved {len(solved)} solved samples to: {solved_path}")
        else:
            logger.info(f"No solved samples to save")

        # Save learnable dataset
        if len(learnable) > 0:
            learnable_path = os.path.join(output_dir, f"{base_name}_learnable.{export_format}")
            learnable.save(learnable_path, export_format)
            logger.info(f"Saved {len(learnable)} learnable samples to: {learnable_path}")
        else:
            logger.info(f"No learnable samples to save")

        # Save unsolved dataset
        if len(unsolved) > 0:
            unsolved_path = os.path.join(output_dir, f"{base_name}_unsolved.{export_format}")
            unsolved.save(unsolved_path, export_format)
            logger.info(f"Saved {len(unsolved)} unsolved samples to: {unsolved_path}")
        else:
            logger.info(f"No unsolved samples to save")