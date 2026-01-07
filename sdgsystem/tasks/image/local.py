import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

from ...configs.config import ImageLocalConfig
from ...models import ModelClient, ModelUsageCounter
from ...dataset.dataset import Dataset
from ..base import BaseTaskExecutor
from ...documents.parse import MinerUParser
from ...generation.generator import ImageDataGenerator

logger = logging.getLogger(__name__)


class ImageLocalTaskExecutor(BaseTaskExecutor):
    """
    Task executor for image.local - generates QA data from local images.

    Image sources:
    - image_dir: Directory containing user-uploaded images
    - parsing: PDF parsing config - extracts images from PDFs via MinerU

    At least one source must be provided. If both are provided, images are combined.
    """

    def __init__(self, config: ImageLocalConfig, llm: ModelClient) -> None:
        super(ImageLocalTaskExecutor, self).__init__(config, llm)
        self.config: ImageLocalConfig = config

    def _load_images_from_dir(self, image_dir: str) -> List[str]:
        """Load image paths from a directory.

        Args:
            image_dir: Path to directory containing images

        Returns:
            List of absolute image paths
        """
        image_dir_path = Path(image_dir)
        if not image_dir_path.exists():
            logger.warning(f"Image directory does not exist: {image_dir}")
            return []

        # Support common image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        image_paths = []

        for file_path in image_dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path.absolute()))

        logger.info(f"Loaded {len(image_paths)} images from {image_dir}")
        return image_paths

    def _organize_images(
        self,
        source_images: List[str],
        source_contexts: Dict[str, str] = None
    ) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
        """Copy images to output_dir/images/ with sequential naming.

        Args:
            source_images: List of source image paths
            source_contexts: Optional dict mapping source image paths to their text context

        Returns:
            Tuple of:
            - List of new absolute image paths in output_dir/images/
            - Dict mapping new absolute paths to relative paths (e.g., "images/000000.png")
            - Dict mapping new absolute paths to their text context
        """
        output_dir = Path(self.config.output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        source_contexts = source_contexts or {}

        new_paths = []
        path_mapping = {}  # new_abs_path -> relative_path
        context_mapping = {}  # new_abs_path -> context

        for idx, source_path in enumerate(source_images):
            source = Path(source_path)
            # Use .png extension for consistency
            new_filename = f"{idx:06d}.png"
            new_abs_path = images_dir / new_filename
            relative_path = f"images/{new_filename}"

            # Copy the image
            shutil.copy2(source, new_abs_path)

            # Use absolute path and normalize for consistent lookup
            new_abs_path_str = str(new_abs_path.absolute())
            new_paths.append(new_abs_path_str)
            path_mapping[new_abs_path_str] = relative_path

            # Map context from source to new path
            if source_path in source_contexts:
                context_mapping[new_abs_path_str] = source_contexts[source_path]

        logger.info(f"Organized {len(new_paths)} images to {images_dir}")
        logger.info(f"Images with context: {len(context_mapping)}")

        return new_paths, path_mapping, context_mapping

    def _extract_context_from_pdf_info(
        self,
        pdf_info_path: Path,
        context_blocks: int = 3
    ) -> Dict[str, str]:
        """Extract surrounding text context for each image from pdf_info.json.

        Args:
            pdf_info_path: Path to pdf_info.json file
            context_blocks: Number of text blocks before/after image to include

        Returns:
            Dict mapping image filename to its surrounding text context
        """
        image_contexts = {}

        if not pdf_info_path.exists():
            logger.warning(f"pdf_info.json not found: {pdf_info_path}")
            return image_contexts

        try:
            with open(pdf_info_path, 'r', encoding='utf-8') as f:
                pdf_info = json.load(f)

            # Collect all blocks across pages in document order
            all_blocks = []
            for page in pdf_info:
                page_idx = page.get('page_idx', 0)
                preproc_blocks = page.get('preproc_blocks', [])
                for block in preproc_blocks:
                    block['_page_idx'] = page_idx
                    all_blocks.append(block)

            # Find image blocks and their context
            for i, block in enumerate(all_blocks):
                block_type = block.get('type', '')

                if block_type == 'image':
                    # Get image filename from block
                    # MinerU structure: block.blocks[].lines[].spans[].image_path
                    img_path = ''

                    # Try MinerU nested structure first
                    inner_blocks = block.get('blocks', [])
                    for inner_block in inner_blocks:
                        if inner_block.get('type') == 'image_body':
                            lines = inner_block.get('lines', [])
                            for line in lines:
                                spans = line.get('spans', [])
                                for span in spans:
                                    if span.get('type') == 'image' and span.get('image_path'):
                                        img_path = span.get('image_path')
                                        break
                                if img_path:
                                    break
                        if img_path:
                            break

                    # Fallback: try legacy structure (img_body.img_path)
                    if not img_path:
                        img_body = block.get('img_body', {})
                        img_path = img_body.get('img_path', '') if isinstance(img_body, dict) else ''

                    # Fallback: try direct image_path field
                    if not img_path:
                        img_path = block.get('image_path', '')

                    if not img_path:
                        continue

                    img_filename = Path(img_path).name

                    # Collect text from surrounding blocks
                    context_parts = []

                    # Get blocks before
                    start_idx = max(0, i - context_blocks)
                    for j in range(start_idx, i):
                        text = self._extract_text_from_block(all_blocks[j])
                        if text:
                            context_parts.append(text)

                    # Get blocks after
                    end_idx = min(len(all_blocks), i + context_blocks + 1)
                    for j in range(i + 1, end_idx):
                        text = self._extract_text_from_block(all_blocks[j])
                        if text:
                            context_parts.append(text)

                    context = ' '.join(context_parts).strip()
                    if context:
                        image_contexts[img_filename] = context

        except Exception as e:
            logger.error(f"Failed to parse pdf_info.json: {e}")

        return image_contexts

    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract text content from a block.

        Args:
            block: Block dict from pdf_info

        Returns:
            Extracted text or empty string
        """
        block_type = block.get('type', '')

        if block_type == 'text':
            # Text block: extract from lines -> spans -> content
            lines = block.get('lines', [])
            texts = []
            for line in lines:
                spans = line.get('spans', [])
                for span in spans:
                    content = span.get('content', '')
                    if content:
                        texts.append(content)
            return ' '.join(texts)

        elif block_type == 'title':
            # Title block: similar structure
            lines = block.get('lines', [])
            texts = []
            for line in lines:
                spans = line.get('spans', [])
                for span in spans:
                    content = span.get('content', '')
                    if content:
                        texts.append(content)
            return ' '.join(texts)

        return ''

    def _extract_images_from_pdfs(self, reporter=None) -> Tuple[List[str], Dict[str, str]]:
        """Extract images from PDFs using MinerU parser with context.

        Args:
            reporter: Optional progress reporter

        Returns:
            Tuple of:
            - List of extracted image paths
            - Dict mapping image path to its surrounding text context
        """
        if not self.config.parsing or not self.config.parsing.document_dir:
            return [], {}

        document_dir = Path(self.config.parsing.document_dir)
        if not document_dir.exists():
            logger.warning(f"Document directory does not exist: {document_dir}")
            return [], {}

        # Find PDF files
        pdf_files = list(document_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {document_dir}")
            return [], {}

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Initialize MinerU parser
        parser = MinerUParser(config=self.config.parsing)

        extracted_images = []
        image_contexts = {}  # image_path -> context

        for idx, pdf_file in enumerate(pdf_files):
            if reporter:
                reporter.update_step(
                    message=f"Extracting images from {pdf_file.name}",
                    completed=idx + 1,
                    current_item_name=pdf_file.name,
                    current_item_index=idx
                )

            try:
                # Parse PDF - MinerU extracts images to output_dir/pdf_name/images/
                parser.parse_document(str(pdf_file))

                # Find extracted images and pdf_info.json
                pdf_stem = pdf_file.stem
                base_dir = Path(parser.output_dir) / pdf_stem / parser.parse_method
                images_dir = base_dir / "images"
                pdf_info_path = base_dir / "pdf_info.json"

                # Extract context for images
                filename_to_context = self._extract_context_from_pdf_info(pdf_info_path)

                if images_dir.exists():
                    for img_path in images_dir.iterdir():
                        if img_path.is_file():
                            abs_path = str(img_path.absolute())
                            extracted_images.append(abs_path)

                            # Map context by filename
                            if img_path.name in filename_to_context:
                                image_contexts[abs_path] = filename_to_context[img_path.name]

                    logger.info(f"Extracted {len(list(images_dir.iterdir()))} images from {pdf_file.name}")
                else:
                    logger.warning(f"No images directory found for {pdf_file.name}")

            except Exception as e:
                logger.error(f"Failed to extract images from {pdf_file.name}: {e}")
                continue

        logger.info(f"Total extracted images from PDFs: {len(extracted_images)}")
        logger.info(f"Images with context: {len(image_contexts)}")

        return extracted_images, image_contexts

    def execute(self, parallel_executor=None, reporter=None) -> Dataset:
        """Execute image local task: load images / extract from PDFs → generate QA data.

        Args:
            parallel_executor: Optional parallel executor for batch processing
            reporter: Optional progress reporter

        Returns:
            Dataset containing generated QA samples with image paths
        """
        logger.info("=== Starting Image Local Task Pipeline ===")

        all_images = []
        all_contexts = {}  # image_path -> context

        # Check if at least one source is configured
        has_image_dir = self.config.image_dir is not None
        has_parsing = self.config.parsing is not None and self.config.parsing.document_dir is not None

        if not has_image_dir and not has_parsing:
            error_msg = (
                "No image source configured. Please provide either 'image_dir' for user images "
                "or 'parsing.document_dir' for PDF image extraction."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Step 1: Load user-uploaded images (if configured)
        if has_image_dir:
            if reporter:
                reporter.start_step(
                    "loading_images", "Loading Images",
                    message="Loading user-uploaded images..."
                )

            user_images = self._load_images_from_dir(self.config.image_dir)
            all_images.extend(user_images)
            # User images don't have context

            if reporter:
                reporter.complete_step({
                    "message": f"Loaded {len(user_images)} images from {self.config.image_dir}",
                    "images_loaded": len(user_images)
                })

        # Step 2: Extract images from PDFs (if configured)
        if has_parsing:
            document_dir = Path(self.config.parsing.document_dir)
            pdf_files = list(document_dir.glob("*.pdf")) if document_dir.exists() else []

            if reporter:
                reporter.start_step(
                    "extracting_images", "Extracting Images from PDFs",
                    message="Loading parser model...",
                    total=len(pdf_files),
                    unit="files"
                )

            pdf_images, pdf_contexts = self._extract_images_from_pdfs(reporter)
            all_images.extend(pdf_images)
            all_contexts.update(pdf_contexts)

            if reporter:
                reporter.complete_step({
                    "message": f"Extracted {len(pdf_images)} images from {len(pdf_files)} PDFs",
                    "images_extracted": len(pdf_images),
                    "pdfs_processed": len(pdf_files),
                    "images_with_context": len(pdf_contexts)
                })

        # Validate we have images
        if not all_images:
            error_msg = "No images found from any source. Please check your image_dir and/or document_dir paths."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Total images collected: {len(all_images)}")

        # Step 3: Organize images to output_dir/images/ with sequential naming
        if reporter:
            reporter.start_step(
                "organizing_images", "Organizing Images",
                message=f"Copying {len(all_images)} images to output directory..."
            )

        organized_paths, path_mapping, context_mapping = self._organize_images(
            all_images, all_contexts
        )

        if reporter:
            reporter.complete_step({
                "message": f"Organized {len(organized_paths)} images to {self.config.output_dir}/images/",
                "images_organized": len(organized_paths),
                "images_with_context": len(context_mapping)
            })

        # Step 4: Generate QA data using VLM
        usage_counter = ModelUsageCounter(total=1, name="Image-Local")

        image_generator = ImageDataGenerator(self.llm, self.config.generation)

        dataset = image_generator.generate(
            task_instruction=self.config.task_instruction,
            image_paths=organized_paths,
            image_contexts=context_mapping,
            usage_counter=usage_counter,
            parallel_executor=parallel_executor,
            reporter=reporter
        )

        # Step 5: Update image paths in dataset to relative paths
        for sample in dataset.samples:
            if 'image' in sample and sample['image'] in path_mapping:
                sample['image'] = path_mapping[sample['image']]

        logger.info(f"Generated dataset with {len(dataset)} samples")
        return dataset
