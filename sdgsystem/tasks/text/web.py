import logging
import re
from tqdm import tqdm
from typing import List, Dict, Optional

from ...configs.config import TextWebConfig
from ...models import ModelClient
from ...dataset.dataset import Dataset
from ..base import BaseTaskExecutor
from ..keyword_extractor import KeywordExtractor
from ...huggingface import HFCrawler
from ...dataset.process import DataFilter, Formatter

logger = logging.getLogger(__name__)


class WebTaskExecutor(BaseTaskExecutor):
    def __init__(self,
        config: TextWebConfig,
        llm: ModelClient,
    ) -> None:
        super(WebTaskExecutor, self).__init__(config, llm)

        self.config: TextWebConfig = config
        self.keyword_extractor = KeywordExtractor(llm)
        self.crawler = HFCrawler(self.config.huggingface_token)
        self.filter = DataFilter(llm)
        self.formatter = Formatter(llm)

    def extract_keywords(self, task_definition: str) -> List[str]:
        """
        Extract keywords for HuggingFace dataset search with domain-first priority.

        Strategy:
        1. Domain field (if provided) - primary user-controlled keyword
        2. LLM extraction from task_instruction - supplementary keywords for breadth
        3. Search with ALL keywords to maximize dataset discovery
        """
        domain = self.config.domain
        keywords = []

        # Priority 1: Use domain as primary keywords if provided
        # Split by comma or space to allow multiple keywords (e.g., "mathematics, gsm8k" or "mathematics gsm8k")
        if domain:
            domain_keywords = [kw.strip() for kw in re.split(r'[,\s]+', domain) if kw.strip()]
            keywords.extend(domain_keywords)
            logger.info(f"Using domain keywords: {domain_keywords}")

        # Priority 2: Extract additional keywords from task instruction
        llm_keywords = self.keyword_extractor.extract_keywords(
            task_instruction=task_definition,
            for_huggingface=True
        )

        # Add LLM keywords, avoiding duplicates
        for kw in llm_keywords:
            if kw not in keywords:
                keywords.append(kw)

        if llm_keywords:
            logger.info(f"LLM extracted supplementary keywords: {llm_keywords}")

        if not keywords:
            raise ValueError(
                "No keywords available for dataset search. "
                "Please provide 'domain' field or ensure task_instruction is descriptive enough for keyword extraction."
            )

        logger.info(f"Final search keywords: {keywords}")
        return keywords

    def search_datasets(self) -> List[Dict]:
        """
        Search datasets from HuggingFace based on task keywords.
        Returns a list of dataset metadata (without rows).
        """
        task_keywords = self.extract_keywords(task_definition=self.config.task_instruction)
        logger.info(f"Searching with {len(task_keywords)} keywords: {task_keywords}")

        datasets = []
        for task_keyword in tqdm(task_keywords, desc="Searching datasets", unit="keyword"):
            task_datasets = self.crawler.search_datasets(query=task_keyword, limit=self.config.dataset_limit)
            if not task_datasets:
                continue

            for task_dataset in task_datasets:
                dataset_id = task_dataset.id
                dataset_splits = self.crawler.get_splits(dataset_id)
                logger.info(f"Found dataset: {dataset_id}")

                datasets.append({
                    "keyword": task_keyword,
                    "id": dataset_id,
                    "splits": dataset_splits
                })

        logger.info(f"Found {len(datasets)} datasets")

        return datasets

    def probe_dataset(self, dataset: Dict) -> Optional[Dict]:
        """
        Phase 1: Probe a dataset with 1 sample to validate quality.
        Returns dataset with score if valid, None otherwise.
        """
        dataset_id = dataset["id"]
        splits = dataset["splits"]

        # Get 1 sample to probe
        first_rows = self.crawler.get_first_rows(dataset_id, splits, sample_limit=1)
        if not first_rows:
            logger.info(f"Skipping {dataset_id}: no rows available")
            return None

        row = first_rows[0]
        # Handle nested row structure from HuggingFace API
        if isinstance(row, dict) and "row" in row:
            row = row["row"]

        legal_keys = list(row.keys())

        # Check which field maps "Input" and "Output"
        fields = self.filter.field_filter(row=row, legal_keys=legal_keys)
        if fields["input"] is None or fields["output"] is None:
            logger.info(f"Skipping {dataset_id}: no input/output fields found")
            return None

        # Get the original Input text and Output text
        input_text = row.get(fields["input"])
        output_text = row.get(fields["output"])
        if input_text is None or output_text is None:
            logger.info(f"Skipping {dataset_id}: None input/output values")
            return None

        # Judge the sample and calculate overall score
        sample_scores = self.filter.instruction_judge(
            self.config.task_instruction,
            f"input: {input_text}\noutput: {output_text}"
        )
        overall_score = sum(int(score) for score in sample_scores.values())
        logger.info(f"Dataset {dataset_id} probe score: {overall_score}")

        # Store field mapping and score for later use
        dataset["fields"] = fields
        dataset["overall_score"] = overall_score

        return dataset

    def validate_datasets(self, datasets: List[Dict]) -> List[Dict]:
        """
        Phase 1: Probe all datasets and score them.
        Returns list of scored datasets sorted by score (highest first).
        """
        scored_datasets = []
        for dataset in tqdm(datasets, desc="Validating datasets", unit="dataset"):
            result = self.probe_dataset(dataset)
            if result is not None:
                scored_datasets.append(result)

        # Sort all scored datasets by score (highest first)
        scored_datasets.sort(key=lambda x: x["overall_score"], reverse=True)

        logger.info(f"Validated: {len(scored_datasets)} datasets with scores")

        return scored_datasets

    def calculate_distribution(self, scored_datasets: List[Dict], num_samples: int) -> List[Dict]:
        """
        Phase 2: Calculate how many samples to extract from each dataset.
        Uses sequential score-based approach: extract from highest-scored datasets first
        until we have enough samples. This ensures maximum quality.

        Args:
            scored_datasets: All datasets sorted by score (highest first)
            num_samples: Total number of samples needed

        Returns:
            List of datasets with samples_to_extract field set
        """
        if not scored_datasets:
            return []

        result = []
        remaining = num_samples

        for dataset in scored_datasets:
            if remaining <= 0:
                break

            # Get available rows from dataset
            available = dataset.get("num_rows", float('inf'))
            to_extract = min(remaining, available)

            dataset["samples_to_extract"] = to_extract
            result.append(dataset)
            remaining -= to_extract

            logger.info(f"Will extract {to_extract} samples from {dataset['id']} (score: {dataset['overall_score']})")

        if remaining > 0:
            logger.warning(f"Could not fulfill request: {remaining} samples still needed after exhausting all datasets")

        logger.info(f"Distribution: {len(result)} datasets selected, {num_samples - remaining}/{num_samples} samples available")

        return result

    def extract_and_format(self, dataset: Dict) -> List[Dict]:
        """
        Phase 3: Extract samples from a dataset and format them.
        """
        dataset_id = dataset["id"]
        splits = dataset["splits"]
        fields = dataset["fields"]
        samples_to_extract = dataset["samples_to_extract"]

        # Get rows from dataset
        rows = self.crawler.get_first_rows(dataset_id, splits, sample_limit=samples_to_extract)
        logger.info(f"Extracted {len(rows)} rows from {dataset_id}")

        formatted_samples = []
        for row in rows:
            # Handle nested row structure
            if isinstance(row, dict) and "row" in row:
                row = row["row"]

            input_text = row.get(fields["input"])
            output_text = row.get(fields["output"])

            if input_text is None or output_text is None:
                continue

            # Format conversion
            formatted = self.formatter.format_conversion(
                input_text,
                output_text,
                self.config.input_instruction,
                self.config.output_instruction
            )

            if formatted.get("input") is None or formatted.get("output") is None:
                logger.warning(f"Skipping row: format conversion failed")
                continue

            formatted_samples.append({
                "keyword": dataset["keyword"],
                "id": dataset_id,
                "original_input": input_text,
                "original_output": output_text,
                "input": formatted["input"],
                "output": formatted["output"],
                "dataset_score": dataset["overall_score"]
            })

        return formatted_samples

    def execute(self, reporter=None) -> Dataset:
        """
        Execute the full web task pipeline:
        1. Search datasets from HuggingFace
        2. Probe each dataset with 1 sample to validate quality
        3. Calculate sample distribution across valid datasets
        4. Extract and format samples from valid datasets
        """
        logger.info("=== Step: Starting Web Task Pipeline ===")

        # Search datasets
        if reporter:
            reporter.start_step("searching", "Searching Datasets", "Searching HuggingFace for relevant datasets...")

        datasets = self.search_datasets()

        if not datasets:
            error_msg = (
                "No datasets found for the given keywords. "
                "Please adjust your task_instruction or domain to match available HuggingFace datasets."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if reporter:
            reporter.complete_step({
                "message": f"Found {len(datasets)} datasets",
                "datasets_found": len(datasets),
                "datasets": [{"id": d["id"], "keyword": d["keyword"]} for d in datasets]
            })

        # Probe and validate datasets
        if reporter:
            reporter.start_step(
                "validating", "Validating Datasets",
                message="Probing datasets and scoring quality...",
                total=len(datasets),
                unit="datasets"
            )

        scored_datasets = self.validate_datasets(datasets)

        if not scored_datasets:
            error_msg = (
                "No valid datasets found after probing. "
                "All datasets failed validation (missing input/output fields or empty values)."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate distribution using sequential score-based approach
        datasets_to_use = self.calculate_distribution(scored_datasets, self.config.num_samples)

        if reporter:
            reporter.complete_step({
                "message": f"Validated {len(scored_datasets)} datasets, using top {len(datasets_to_use)}",
                "total_datasets": len(scored_datasets),
                "datasets_to_use": [{"id": d["id"], "score": d["overall_score"], "samples_to_extract": d["samples_to_extract"]} for d in datasets_to_use]
            })

        # Extract and format samples
        if reporter:
            reporter.start_step(
                "extracting", "Extracting Samples",
                message=f"Extracting samples from {len(datasets_to_use)} datasets...",
                total=len(datasets_to_use),
                unit="datasets"
            )

        all_samples = []
        for idx, dataset in enumerate(tqdm(datasets_to_use, desc="Extracting samples", unit="dataset")):
            samples = self.extract_and_format(dataset)
            all_samples.extend(samples)
            logger.info(f"Extracted {len(samples)} samples from {dataset['id']} (score: {dataset['overall_score']})")

            if reporter:
                reporter.update_step(
                    message=f"Extracted samples from {dataset['id']}",
                    completed=idx + 1,
                    current_item_name=dataset['id'],
                    current_item_index=idx
                )

        if reporter:
            reporter.complete_step({
                "message": f"Extracted {len(all_samples)} samples from {len(datasets_to_use)} datasets",
                "samples_extracted": len(all_samples),
                "datasets_used": len(datasets_to_use)
            })

        # Build final dataset
        final_dataset = Dataset()
        final_dataset.add_samples(samples=all_samples)

        logger.info(f"Web task complete: {len(all_samples)} samples from {len(datasets_to_use)} datasets")

        return final_dataset
