import random
import logging
import re
from pathlib import Path
from typing import List, Dict

from ...configs.config import TextLocalConfig
from ...configs.constants import DEFAULT_KEYWORDS_EXTRACT_EXAMPLES
from ...models import ModelClient, ModelUsageCounter
from ...dataset.dataset import Dataset
from ..base import BaseTaskExecutor
from ..keyword_extractor import KeywordExtractor
from ...documents.parse import MinerUParser
from ...documents.chunk import RecursiveChunker
from ...documents.export import export_chunks_to_jsonl
from ...documents.load import DocumentLoader
from ...documents.retrieve import BM25Retriever
from ...generation.generator import TextDataGenerator

logger = logging.getLogger(__name__)


class LocalTaskExecutor(BaseTaskExecutor):
    def __init__(self,
        config: TextLocalConfig,
        llm: ModelClient
    ) -> None:
        super(LocalTaskExecutor, self).__init__(config, llm)
        self.config: TextLocalConfig
        self.keyword_extractor = KeywordExtractor(llm)

    def extract_keywords(
        self,
        task_instruction: str,
        demo_examples: List[Dict[str, str]] = DEFAULT_KEYWORDS_EXTRACT_EXAMPLES,
        usage_counter: ModelUsageCounter = None
    ) -> List[str]:
        """Extract domain keywords using LLM based on task instruction and demo examples."""
        domain = self.config.domain

        keywords = self.keyword_extractor.extract_keywords(
            task_instruction=task_instruction,
            demo_examples=demo_examples,
            usage_counter=usage_counter
        )

        # Add domain keywords (split by comma or space) if not already in keywords
        if domain:
            domain_keywords = [kw.strip() for kw in re.split(r'[,\s]+', domain) if kw.strip()]
            for kw in domain_keywords:
                if kw not in keywords:
                    keywords.append(kw)

        return keywords

    def execute(self, parallel_executor=None, reporter=None) -> Dataset:
        """Execute local task: document processing → retrieval → generation → dataset."""

        # Loading Documents (quick - just file listing)
        if reporter:
            reporter.start_step("loading", "Loading Documents", "Loading documents and demo examples...")

        document_dir = self.config.parsing.document_dir if self.config.parsing else None
        corpus_paths = [document_dir] if document_dir else []
        loader = DocumentLoader(corpus_paths=corpus_paths)

        pdf_files = loader.load_documents() if corpus_paths else []
        logger.info(f"Loaded {len(pdf_files)} PDF files")

        demo_examples_path = self.config.demo_examples_path if hasattr(self.config, 'demo_examples_path') else None
        demo_examples = loader.load_demo_examples(demo_examples_path) if demo_examples_path else []

        if reporter:
            reporter.complete_step({
                "message": f"Loaded {len(pdf_files)} documents and {len(demo_examples)} demo examples",
                "pdf_count": len(pdf_files),
                "demo_count": len(demo_examples)
            })

        # Parsing Documents (includes model loading on first document)
        if self.config.parsing and self.config.parsing.method and pdf_files:
            if reporter:
                reporter.start_step("parsing", "Parsing Documents",
                                   message="Loading parser model...",
                                   total=len(pdf_files), unit="files")

            logger.info(f"Parser method: {self.config.parsing.method}")
            if self.config.parsing.method == "mineru":
                parser = MinerUParser(config=self.config.parsing)
            else:
                raise ValueError(f"Unknown parser: {self.config.parsing.method}")

            output_dir = Path(self.config.retrieval.passages_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)

            for idx, pdf_file in enumerate(pdf_files):
                results = parser.parse_document(str(pdf_file))
                combined_content = "\n\n".join([r['content'] for r in results if 'content' in r])
                chunks = chunker.chunk_text(combined_content)
                output_path = output_dir / f"{pdf_file.stem}.jsonl"
                export_chunks_to_jsonl(chunks, str(output_path))

                # Report progress after each file is parsed
                if reporter:
                    reporter.update_step(
                        message=f"Parsed {pdf_file.name}",
                        completed=idx + 1,
                        current_item_name=pdf_file.name,
                        current_item_index=idx
                    )

            if reporter:
                reporter.complete_step({"files_parsed": len(pdf_files)})
            logger.info("Document processing completed")
        else:
            logger.info("Skipping document processing (no documents or parser not configured)")

        # Extracting Keywords
        if reporter:
            reporter.start_step("keyword_extraction", "Extracting Keywords", "Analyzing instructions...")

        usage_counter_keywords = ModelUsageCounter(total=1, name="Text-Local-Keywords")
        keywords = self.extract_keywords(
            self.config.task_instruction,
            demo_examples,
            usage_counter=usage_counter_keywords
        )
        logger.info(f"Extracted keywords: {keywords}")

        if reporter:
            reporter.complete_step({"keywords": keywords})

        # Retrieving Passages
        if reporter:
            reporter.start_step("passage_retrieval", "Retrieving Passages", "Searching for relevant passages...")

        retriever = BM25Retriever(config=self.config.retrieval, cache_corpus=True)
        reference_passages = retriever.retrieve(keywords)
        reference_passages = list(set(reference_passages))
        random.shuffle(reference_passages)
        logger.info(f"Retrieved {len(reference_passages)} unique passages")

        if reporter:
            reporter.complete_step({"passages_count": len(reference_passages)})

        # Generating Samples
        usage_counter_generation = ModelUsageCounter(total=1, name="Text-Local")

        data_generator = TextDataGenerator(self.llm, self.config.generation)

        dataset = data_generator.generate(
            task_definition=self.config.task_instruction,
            demo_examples=demo_examples,
            passages=reference_passages,
            usage_counter=usage_counter_generation,
            parallel_executor=parallel_executor,
            reporter=reporter
        )
        logger.info(f"Generated dataset with {len(dataset)} samples")

        return dataset
