import random
import logging
from pathlib import Path
from tqdm import tqdm

from .base import *
from ..documents.parse import DotsOCRParser, MinerUParser
from ..documents.chunk import RecursiveChunker
from ..documents.export import export_chunks_to_jsonl
from ..documents.load import DocumentLoader
from ..documents.retrieve import BM25Retriever
from ..generation.generator import DataGenerator
from ..models import ModelUsageCounter

logger = logging.getLogger(__name__)


class LocalTaskExecutor(BaseTaskExecutor):
    def __init__(self,
        config: LocalTaskConfig,
        llm: ModelClient
    ) -> None:
        super(LocalTaskExecutor, self).__init__(config, llm)
        self.config: LocalTaskConfig

    def execute(self, parallel_executor=None) -> Dataset:
        """Execute local task: document processing → retrieval → generation → dataset."""

        logger.info("=== Step: Loading Documents and Demo Examples ===")
        document_dir = self.config.parsing.document_dir if self.config.parsing else None
        corpus_paths = [document_dir] if document_dir else []
        loader = DocumentLoader(corpus_paths=corpus_paths)

        pdf_files = loader.load_documents() if corpus_paths else []
        logger.info(f"Loaded {len(pdf_files)} PDF files")

        demo_examples_path = self.config.demo_examples_path if hasattr(self.config, 'demo_examples_path') else None
        demo_examples = loader.load_demo_examples(demo_examples_path) if demo_examples_path else []

        logger.info("=== Step: Document Processing ===")
        if self.config.parsing and self.config.parsing.method and pdf_files:
            logger.info(f"Parser method: {self.config.parsing.method}")
            if self.config.parsing.method == "dotsocr":
                parser = DotsOCRParser(config=self.config.parsing)
            elif self.config.parsing.method == "mineru":
                parser = MinerUParser(config=self.config.parsing)
            else:
                raise ValueError(f"Unknown parser: {self.config.parsing.method}")

            output_dir = Path(self.config.retrieval.passages_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)

            for pdf_file in tqdm(pdf_files, desc="Processing documents"):
                results = parser.parse_document(str(pdf_file))
                combined_content = "\n\n".join([r['content'] for r in results if 'content' in r])
                chunks = chunker.chunk_text(combined_content)
                output_path = output_dir / f"{pdf_file.stem}.jsonl"
                export_chunks_to_jsonl(chunks, str(output_path))
            logger.info("Document processing completed")
        else:
            logger.info("Skipping document processing (no documents or parser not configured)")

        logger.info("=== Step: Extracting Keywords ===")
        usage_counter_keywords = ModelUsageCounter(total=1, name="LocalTask-Keywords")
        keywords = self.extract_keywords(
            self.config.task_instruction,
            demo_examples,
            usage_counter=usage_counter_keywords
        )
        logger.info(f"Extracted keywords: {keywords}")

        logger.info("=== Step: Retrieving Passages ===")
        retriever = BM25Retriever(config=self.config.retrieval, cache_corpus=True)
        reference_passages = retriever.retrieve(keywords)
        reference_passages = list(set(reference_passages))
        random.shuffle(reference_passages)
        logger.info(f"Retrieved {len(reference_passages)} unique passages")

        logger.info("=== Step: Generating Synthetic Dataset ===")
        # Calculate total iterations: 1 (pattern) + num_samples (generation) + num_samples (validation)
        total_iterations = 1 + self.config.generation.num_samples * 2
        usage_counter_generation = ModelUsageCounter(total=total_iterations, name="LocalTask-Generation")

        data_generator = DataGenerator(self.llm, self.config.generation)

        dataset = data_generator.generate(
            task_definition=self.config.task_instruction,
            demo_examples=demo_examples,
            passages=reference_passages,
            usage_counter=usage_counter_generation,
            parallel_executor=parallel_executor
        )
        logger.info(f"Generated dataset with {len(dataset)} samples")

        return dataset
