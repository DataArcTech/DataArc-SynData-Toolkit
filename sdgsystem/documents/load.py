import json
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads PDF documents from specified directories.
    """

    def __init__(self, corpus_paths: List[str]):
        """
        Initialize document loader.

        Args:
            corpus_paths: List of directory paths containing PDF files
        """
        self.corpus_paths = [Path(path) for path in corpus_paths]

    def load_documents(self) -> List[Path]:
        """
        Load all PDF documents from configured corpus paths.

        Returns:
            List of PDF file paths
        """
        collected_pdfs = []

        for path in self.corpus_paths:
            if not path.exists():
                logger.warning(f"Warning: Directory not found: {path}")
                continue

            if not path.is_dir():
                logger.warning(f"Warning: Not a directory: {path}")
                continue

            pdf_files = list(path.glob("*.pdf"))

            if not pdf_files:
                logger.warning(f"Warning: No PDF files found in: {path}")
                continue

            for pdf_file in pdf_files:
                logger.info(f"Loaded: {pdf_file}")
                collected_pdfs.append(pdf_file)

        return collected_pdfs

    def count_documents(self) -> int:
        """
        Count total PDF documents across all corpus paths.

        Returns:
            Total number of PDF documents
        """
        total = 0

        for path in self.corpus_paths:
            if not path.exists() or not path.is_dir():
                continue

            pdf_files = list(path.glob("*.pdf"))
            total += len(pdf_files)

        return total

    def load_demo_examples(self, demo_examples_path: Optional[str]) -> List[dict]:
        """
        Load demo examples from JSONL file.

        Args:
            demo_examples_path: Path to the JSONL file containing demo examples (optional, can be None or empty string)

        Returns:
            List of demo example dicts (empty list if path is None/empty, file not found, or error occurs)
        """
        demo_examples = []

        # Handle None or empty string - check before creating Path object
        if not demo_examples_path:
            logger.info(f"No demo examples path provided")
            logger.info(f"Using empty demo examples list")
            return demo_examples

        demo_path = Path(demo_examples_path)
        if not demo_path.exists():
            logger.info(f"Demo examples file not found: {demo_examples_path}")
            logger.info(f"Using empty demo examples list")
            return demo_examples

        try:
            with open(demo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        example = json.loads(line)
                        demo_examples.append(example)

            logger.info(f"Loaded {len(demo_examples)} demo examples from {demo_examples_path}")
        except Exception as e:
            logger.error(f"Error loading demo examples: {e}")
            logger.error(f"Using empty demo examples list")

        return demo_examples
