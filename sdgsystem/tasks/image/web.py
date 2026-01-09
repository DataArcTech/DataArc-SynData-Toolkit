import io
import logging
import requests
from pathlib import Path
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

from ...configs.config import ImageWebConfig
from ...models import ModelClient
from ...dataset.dataset import Dataset
from ..base import BaseTaskExecutor
from ..keyword_extractor import KeywordExtractor
from ...huggingface import HFCrawler
from ...dataset.process import DataFilter, Formatter

logger = logging.getLogger(__name__)


class ImageWebTaskExecutor(BaseTaskExecutor):
    """
    Task executor for image.web - fetches image QA data from HuggingFace.

    Pipeline:
    1. Extract keywords from task instruction
    2. Search HuggingFace for VQA datasets
    3. Probe datasets for quality (LLM identifies image/input/output fields)
    4. Load datasets using HuggingFace datasets library (handles all image formats)
    5. Download images and extract QA pairs
    """

    def __init__(self, config: ImageWebConfig, llm: ModelClient) -> None:
        super(ImageWebTaskExecutor, self).__init__(config, llm)
        self.config: ImageWebConfig = config
        self.keyword_extractor = KeywordExtractor(llm)
        self.crawler = HFCrawler(self.config.huggingface_token)
        self.filter = DataFilter(llm)
        self.formatter = Formatter(llm)

    def extract_keywords(self, task_definition: str) -> List[str]:
        """Extract domain keywords using LLM based on task instruction."""
        domain = self.config.domain

        keywords = self.keyword_extractor.extract_keywords(
            task_instruction=task_definition,
            for_huggingface=True
        )

        # If domain is non-empty and not already in keywords, add it
        if domain and domain not in keywords:
            keywords.append(domain)

        return keywords

    def search_datasets(self) -> List[Dict]:
        """
        Search image datasets from HuggingFace based on task keywords.
        Returns a list of dataset metadata (with splits info).
        """
        task_keywords = self.extract_keywords(task_definition=self.config.task_instruction)
        logger.info(f"Searching with {len(task_keywords)} keywords: {task_keywords}")

        datasets = []
        for task_keyword in tqdm(task_keywords, desc="Searching datasets", unit="keyword"):
            task_datasets = self.crawler.search_image_datasets(
                query=task_keyword,
                limit=self.config.dataset_limit
            )
            if not task_datasets:
                continue

            for task_dataset in task_datasets:
                dataset_id = task_dataset.id
                dataset_splits = self.crawler.get_splits(dataset_id)
                logger.info(f"Found dataset: {dataset_id}")

                datasets.append({
                    "keyword": task_keyword,
                    "id": dataset_id,
                    "splits": dataset_splits,
                })

        # Deduplicate by dataset id
        seen_ids = set()
        unique_datasets = []
        for ds in datasets:
            if ds["id"] not in seen_ids:
                seen_ids.add(ds["id"])
                unique_datasets.append(ds)

        logger.info(f"Found {len(unique_datasets)} unique datasets")
        return unique_datasets

    def _detect_image_type(self, image_data, dataset_id: str, config_name: str = None) -> Optional[str]:
        """
        Detect the type of image data and validate it can be retrieved.

        Args:
            image_data: The image data from the dataset sample
            dataset_id: Dataset ID for downloading path-based images
            config_name: Dataset config name for path construction

        Returns:
            Image type string ("pil", "url", "path", "repo_path", "list") or None if invalid
        """
        # Case 0: List of images - use first image
        if isinstance(image_data, list):
            if not image_data:
                logger.warning("Empty image list")
                return None
            # Recursively detect type of first image
            first_image_type = self._detect_image_type(image_data[0], dataset_id, config_name)
            if first_image_type:
                return f"list_{first_image_type}"
            return None

        # Case 1: PIL Image - directly usable
        if isinstance(image_data, Image.Image):
            return "pil"

        # Case 2: String - could be URL or path
        if isinstance(image_data, str):
            # Check if it's a URL
            if image_data.startswith(("http://", "https://")):
                return "url"

            # It's a filename/path - try common patterns for HuggingFace datasets
            # Pattern 1: Direct path in repo root
            # Pattern 2: Path under config-specific directory (e.g., "config_name/images/filename.png")
            # Pattern 3: Path under common image directories (e.g., "images/filename.png", "data/images/filename.png")

            possible_paths = [
                image_data,  # Direct path as-is
            ]

            # Add config-based paths
            if config_name:
                possible_paths.extend([
                    f"{config_name}/{image_data}",
                    f"{config_name}/images/{image_data}",
                    f"data/{config_name}/{image_data}",
                ])

            # Add common directory patterns
            possible_paths.extend([
                f"images/{image_data}",
                f"data/{image_data}",
                f"data/images/{image_data}",
            ])

            for path in possible_paths:
                try:
                    local_path = hf_hub_download(
                        repo_id=dataset_id,
                        filename=path,
                        repo_type="dataset"
                    )
                    # Verify it's a valid image
                    Image.open(local_path)
                    logger.info(f"Found image at path: {path}")
                    return "repo_path"
                except Exception:
                    continue

            logger.warning(f"Failed to find image '{image_data}' in {dataset_id}")
            return None

        # Case 3: Raw bytes - directly usable
        if isinstance(image_data, bytes):
            try:
                Image.open(io.BytesIO(image_data))
                return "bytes"
            except Exception:
                logger.warning("Invalid raw bytes image data")
                return None

        # Case 4: Dict with image data (some datasets use this format)
        if isinstance(image_data, dict):
            if "bytes" in image_data and image_data["bytes"]:
                return "dict_bytes"
            if "path" in image_data and image_data["path"]:
                # Recursively check the path
                return self._detect_image_type(image_data["path"], dataset_id, config_name)

        logger.warning(f"Unknown image data type: {type(image_data)}")
        return None

    def _find_image_path(self, image_data: str, dataset_id: str, config_name: str = None) -> Optional[str]:
        """
        Find the correct path for an image filename in a HuggingFace repo.

        Args:
            image_data: The image filename/path from the dataset
            dataset_id: Dataset ID
            config_name: Dataset config name for path construction

        Returns:
            The working path or None if not found
        """
        possible_paths = [image_data]

        if config_name:
            possible_paths.extend([
                f"{config_name}/{image_data}",
                f"{config_name}/images/{image_data}",
                f"data/{config_name}/{image_data}",
            ])

        possible_paths.extend([
            f"images/{image_data}",
            f"data/{image_data}",
            f"data/images/{image_data}",
        ])

        for path in possible_paths:
            try:
                local_path = hf_hub_download(
                    repo_id=dataset_id,
                    filename=path,
                    repo_type="dataset"
                )
                Image.open(local_path)
                return path
            except Exception:
                continue

        return None

    def _load_image(self, image_data, dataset_id: str, image_type: str, config_name: str = None) -> Optional[Image.Image]:
        """
        Load image data into PIL Image based on detected type.

        Args:
            image_data: The image data from the dataset
            dataset_id: Dataset ID for path-based images
            image_type: Type detected by _detect_image_type
            config_name: Dataset config name for path construction

        Returns:
            PIL Image or None if failed
        """
        try:
            # Handle list types - extract first image and get base type
            if image_type.startswith("list_"):
                if not isinstance(image_data, list) or not image_data:
                    return None
                base_type = image_type[5:]  # Remove "list_" prefix
                return self._load_image(image_data[0], dataset_id, base_type, config_name)

            if image_type == "pil":
                return image_data

            if image_type == "url":
                response = requests.get(image_data, timeout=30)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))

            if image_type == "repo_path":
                # Find the correct path and download
                working_path = self._find_image_path(image_data, dataset_id, config_name)
                if working_path:
                    local_path = hf_hub_download(
                        repo_id=dataset_id,
                        filename=working_path,
                        repo_type="dataset"
                    )
                    return Image.open(local_path)
                return None

            if image_type == "bytes":
                return Image.open(io.BytesIO(image_data))

            if image_type == "dict_bytes":
                return Image.open(io.BytesIO(image_data["bytes"]))

        except Exception as e:
            logger.warning(f"Failed to load image: {e}")
            return None

        return None

    def probe_dataset(self, dataset: Dict) -> Optional[Dict]:
        """
        Probe a dataset with 1 sample to validate it has image + QA fields.
        Uses LLM to identify which fields map to image/input/output.
        Validates that image data can actually be retrieved.

        Returns dataset with score, field mapping, and image_type if valid, None otherwise.
        """
        dataset_id = dataset["id"]

        # Use splits info from search_datasets
        dataset_splits = dataset.get("splits", [])
        if not dataset_splits:
            logger.warning(f"Skipping {dataset_id}: no splits available")
            return None

        # Try to load using splits from API
        # Prefer train/validation splits over test - test splits often have withheld labels
        def split_priority(split_info):
            split_name = split_info.get("split", "").lower()
            if split_name.startswith("train"): # train, training, etc.
                return 0
            elif split_name.startswith("val") or split_name.startswith("valid"): # val, valid, validation, etc.
                return 1
            elif split_name.startswith("dev"):
                return 2
            elif split_name.startswith("test"): # Test splits last
                return 3
            return 4

        sorted_splits = sorted(dataset_splits, key=split_priority)

        ds = None
        used_split = None
        used_config = None
        for split_info in sorted_splits:
            split_name = split_info.get("split")
            config_name = split_info.get("config")
            try:
                ds = load_dataset(dataset_id, name=config_name, split=split_name, streaming=True, trust_remote_code=True)
                used_split = split_name
                used_config = config_name
                break
            except Exception:
                continue

        if ds is None:
            logger.warning(f"Skipping {dataset_id}: failed to load with available splits")
            return None

        try:
            ds_iter = iter(ds)
            sample = next(ds_iter)
        except Exception as e:
            logger.warning(f"Skipping {dataset_id}: failed to iterate - {e}")
            return None

        # Get field names and let LLM identify image/input/output fields
        legal_keys = list(sample.keys())
        fields = self.filter.image_field_filter(legal_keys=legal_keys)

        if fields["image"] is None:
            logger.warning(f"Skipping {dataset_id}: no image field identified in {legal_keys}")
            return None
        if fields["input"] is None or fields["output"] is None:
            logger.warning(f"Skipping {dataset_id}: no input/output fields identified in {legal_keys}")
            return None

        # Try multiple samples to find one with valid image/input/output values
        max_probe_samples = 10
        valid_sample = None

        for probe_idx in range(max_probe_samples):
            if probe_idx > 0:
                try:
                    sample = next(ds_iter)
                except StopIteration:
                    break

            image_data = sample.get(fields["image"])
            input_text = sample.get(fields["input"])
            output_text = sample.get(fields["output"])

            # Convert output to string if needed (could be list or other type)
            if isinstance(output_text, list):
                output_text = output_text[0] if output_text else ""
            output_text = str(output_text) if output_text else ""

            # Check if this sample has all required fields with values
            if image_data is not None and input_text and output_text:
                valid_sample = {
                    "image_data": image_data,
                    "input_text": input_text,
                    "output_text": output_text
                }
                logger.info(f"Dataset {dataset_id}: found valid sample at index {probe_idx}")
                break

        if valid_sample is None:
            logger.warning(f"Skipping {dataset_id}: no valid sample found in first {max_probe_samples} rows")
            return None

        image_data = valid_sample["image_data"]
        input_text = valid_sample["input_text"]
        output_text = valid_sample["output_text"]

        # Detect and validate image type
        image_type = self._detect_image_type(image_data, dataset_id, used_config)
        if image_type is None:
            logger.warning(f"Skipping {dataset_id}: cannot retrieve image data")
            return None

        logger.info(f"Dataset {dataset_id} image type: {image_type}")

        # Judge the sample and calculate overall score
        sample_scores = self.filter.instruction_judge(
            self.config.task_instruction,
            f"input: {input_text}\noutput: {output_text}"
        )
        overall_score = sum(int(score) for score in sample_scores.values())
        logger.info(f"Dataset {dataset_id} probe score: {overall_score}")

        # Store field mapping, score, image type, and split/config info
        dataset["fields"] = fields
        dataset["overall_score"] = overall_score
        dataset["image_type"] = image_type
        dataset["used_split"] = used_split
        dataset["used_config"] = used_config

        return dataset

    def validate_datasets(self, datasets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Probe all datasets and filter by score threshold.
        Returns tuple of (valid_datasets, fallback_datasets) sorted by score.
        """
        scored_datasets = []
        for dataset in tqdm(datasets, desc="Validating datasets", unit="dataset"):
            result = self.probe_dataset(dataset)
            if result is not None:
                scored_datasets.append(result)

        # Sort by score (highest first)
        scored_datasets.sort(key=lambda x: x["overall_score"], reverse=True)

        # Separate by threshold
        above_threshold = [d for d in scored_datasets if d["overall_score"] >= self.config.dataset_score_threshold]
        below_threshold = [d for d in scored_datasets if d["overall_score"] < self.config.dataset_score_threshold]

        logger.info(f"Validated: {len(above_threshold)} valid, {len(below_threshold)} fallback datasets")

        return above_threshold, below_threshold

    def calculate_distribution(
        self,
        valid_datasets: List[Dict],
        fallback_datasets: List[Dict],
        num_samples: int
    ) -> List[Dict]:
        """
        Calculate how many samples to extract from each dataset.
        """
        all_datasets = valid_datasets.copy()

        if not all_datasets:
            logger.warning("No valid datasets available. Using fallback datasets.")
            all_datasets = fallback_datasets.copy()

        if not all_datasets:
            return []

        # Distribute evenly
        n_datasets = len(all_datasets)
        base_count = num_samples // n_datasets
        remainder = num_samples % n_datasets

        for i, dataset in enumerate(all_datasets):
            dataset["samples_to_extract"] = base_count + (1 if i < remainder else 0)

        return all_datasets

    def extract_and_format(self, dataset: Dict, image_counter: int) -> Tuple[List[Dict], int]:
        """
        Extract samples from a dataset, save images, and format them.

        Handles multiple image storage formats:
        - PIL Image (embedded in parquet)
        - URL strings (fetch via requests)
        - Path strings (download via hf_hub_download)
        - Dict with bytes

        Args:
            dataset: Dataset dict with id, fields, samples_to_extract, image_type
            image_counter: Current image counter for sequential naming

        Returns:
            Tuple of (formatted_samples, updated_image_counter)
        """
        dataset_id = dataset["id"]
        fields = dataset["fields"]
        samples_to_extract = dataset["samples_to_extract"]
        image_type = dataset["image_type"]
        used_split = dataset.get("used_split", "train")
        used_config = dataset.get("used_config")

        # Create images directory
        output_dir = Path(self.config.output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load dataset with streaming using the same split/config from probing
            ds = load_dataset(dataset_id, name=used_config, split=used_split, streaming=True, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {e}")
            return [], image_counter

        formatted_samples = []
        for i, sample in enumerate(ds):
            if i >= samples_to_extract:
                break

            image_data = sample.get(fields["image"])
            input_text = sample.get(fields["input"])
            output_text = sample.get(fields["output"])

            # Convert output to string if needed
            if isinstance(output_text, list):
                output_text = output_text[0] if output_text else ""
            output_text = str(output_text) if output_text else ""

            if image_data is None or not input_text or not output_text:
                continue

            # Load image using detected type
            pil_image = self._load_image(image_data, dataset_id, image_type, used_config)
            if pil_image is None:
                continue

            # Save image with sequential naming
            image_filename = f"{image_counter:06d}.png"
            image_path = images_dir / image_filename
            relative_image_path = f"images/{image_filename}"

            try:
                # Convert to RGB if necessary (for PNG compatibility)
                # Handles RGBA, LA, P (palette), CMYK, and other non-RGB modes
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                pil_image.save(image_path)
            except Exception as e:
                logger.warning(f"Failed to save image: {e}")
                continue

            image_counter += 1

            # Format conversion for text fields
            formatted = self.formatter.format_conversion(
                input_text,
                output_text,
                self.config.input_instruction or "",
                self.config.output_instruction or ""
            )

            if formatted.get("input") is None or formatted.get("output") is None:
                continue

            formatted_samples.append({
                "keyword": dataset["keyword"],
                "id": dataset_id,
                "original_input": input_text,
                "original_output": output_text,
                "input": formatted["input"],
                "output": formatted["output"],
                "image": relative_image_path,
                "dataset_score": dataset["overall_score"]
            })

        logger.info(f"Extracted {len(formatted_samples)} samples from {dataset_id}")
        return formatted_samples, image_counter

    def execute(self, reporter=None) -> Dataset:
        """
        Execute the full image web task pipeline:
        1. Search image datasets from HuggingFace
        2. Probe each dataset (LLM identifies fields) and validate quality
        3. Calculate sample distribution across valid datasets
        4. Load images using datasets library and extract QA pairs
        """
        logger.info("=== Starting Image Web Task Pipeline ===")

        # Step 1: Search datasets
        if reporter:
            reporter.start_step(
                "searching", "Searching Datasets",
                message="Searching HuggingFace for VQA datasets..."
            )

        datasets = self.search_datasets()

        if not datasets:
            error_msg = (
                "No datasets found for the given keywords. "
                "Please adjust your task_instruction or domain to match available HuggingFace VQA datasets."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if reporter:
            reporter.complete_step({
                "message": f"Found {len(datasets)} datasets",
                "datasets_found": len(datasets),
                "datasets": [{"id": d["id"], "keyword": d["keyword"]} for d in datasets]
            })

        # Step 2: Probe and validate datasets
        if reporter:
            reporter.start_step(
                "validating", "Validating Datasets",
                message="Probing datasets for image QA fields...",
                total=len(datasets),
                unit="datasets"
            )

        valid_datasets, fallback_datasets = self.validate_datasets(datasets)

        if not valid_datasets and not fallback_datasets:
            error_msg = (
                "No valid image datasets found after probing. "
                "All datasets failed validation (missing image/input/output fields or images not retrievable)."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Step 3: Calculate distribution
        datasets_to_use = self.calculate_distribution(
            valid_datasets,
            fallback_datasets,
            self.config.num_samples
        )

        if reporter:
            reporter.complete_step({
                "message": f"Found {len(valid_datasets)} valid, {len(fallback_datasets)} fallback",
                "valid_count": len(valid_datasets),
                "fallback_count": len(fallback_datasets),
                "datasets_to_use": [{"id": d["id"], "samples": d["samples_to_extract"]} for d in datasets_to_use]
            })

        # Step 4: Extract and format samples
        if reporter:
            reporter.start_step(
                "extracting", "Extracting Samples",
                message=f"Downloading images from {len(datasets_to_use)} datasets...",
                total=len(datasets_to_use),
                unit="datasets"
            )

        all_samples = []
        image_counter = 0

        for idx, dataset in enumerate(tqdm(datasets_to_use, desc="Extracting samples", unit="dataset")):
            samples, image_counter = self.extract_and_format(dataset, image_counter)
            all_samples.extend(samples)

            if reporter:
                reporter.update_step(
                    message=f"Extracted from {dataset['id']}",
                    completed=idx + 1,
                    current_item_name=dataset['id']
                )

        # Use fallback datasets if needed
        if len(all_samples) < self.config.num_samples and fallback_datasets:
            remaining = self.config.num_samples - len(all_samples)
            logger.warning(f"Using fallback datasets for {remaining} more samples")

            for dataset in fallback_datasets:
                if len(all_samples) >= self.config.num_samples:
                    break

                dataset["samples_to_extract"] = self.config.num_samples - len(all_samples)
                samples, image_counter = self.extract_and_format(dataset, image_counter)
                all_samples.extend(samples)

        if reporter:
            reporter.complete_step({
                "message": f"Extracted {len(all_samples)} samples with {image_counter} images",
                "samples_extracted": len(all_samples),
                "images_downloaded": image_counter
            })

        # Build final dataset
        final_dataset = Dataset()
        final_dataset.add_samples(samples=all_samples)

        logger.info(f"=== Image Web Task Complete: {len(all_samples)} samples ===")

        return final_dataset
