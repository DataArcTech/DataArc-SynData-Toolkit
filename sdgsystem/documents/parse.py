from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
import logging
import re
import os
import json

from ..configs.config import ParserConfig

logger = logging.getLogger(__name__)


# Abstract base class of parser
class BaseParser(ABC):
    def __init__(self, config: ParserConfig) -> None:
        """
        Initialize parser with configuration.

        Args:
            config: Parser configuration containing device, document_dir and other settings
        """
        self.config = config
        self.device = config.device
        self.document_dir = Path(config.document_dir) if config.document_dir else None

    @abstractmethod
    def parse_document(self, document_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Parse a document and extract structured content.

        Args:
            document_path: Path to the document file inside config.document_dir
            **kwargs: Additional parser-specific arguments

        Returns:
            List of dicts containing parsed content (format varies by parser)
        """
        return []


# class of MinerU Parser
class MinerUParser(BaseParser):
    """
    MinerU parser for PDF document processing using pipeline mode.
    Uses MinerU Python API directly for better control and features.

    Workflow:
    1. Initialize MinerU with pipeline backend configuration
    2. Parse PDF to extract structured content using pipeline
    3. Generate markdown output
    """

    def __init__(
        self,
        config: ParserConfig,
        output_dir: str = "./outputs/mineru",
        parse_method: str = "auto",
        model_source: str = "modelscope",
        table_enable: bool = True,
        formula_enable: bool = True,
        lang: str = "en"
    ) -> None:
        """
        Initialize the MinerU parser with pipeline mode.

        Args:
            config: Parser configuration (contains device and document_dir)
            output_dir: Directory to save output files (default: "./output/mineru")
            parse_method: Parse method - "auto", "txt", "ocr" (default: "auto")
            model_source: Model source - "modelscope" or "huggingface" (default: "modelscope")
            table_enable: Enable table extraction (default: True)
            formula_enable: Enable formula extraction (default: True)
            lang: Document language - "en" or "ch" (default: "en")
        """
        super(MinerUParser, self).__init__(config)
        self.output_dir = Path(output_dir)
        self.parse_method = parse_method
        self.model_source = model_source
        self.table_enable = table_enable
        self.formula_enable = formula_enable
        self.lang = lang
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set ModelScope as model source via environment variable
        os.environ["MINERU_MODEL_SOURCE"] = self.model_source

       # Device configuration for MinerU
        if self.device and self.device.lower() == "cpu":
            # For CPU mode, use MINERU_DEVICE_MODE
            os.environ["MINERU_DEVICE_MODE"] = "cpu"
            logger.info(f"Loading MinerU model - Device: CPU (MINERU_DEVICE_MODE=cpu)")
        else:
            # CUDA mode - Set CUDA_VISIBLE_DEVICES before MinerU initialization
            if self.device and self.device.startswith("cuda:"):
                gpu_id = self.device.split(":")[1]
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                logger.info(f"[MinerU] Setting CUDA_VISIBLE_DEVICES={gpu_id}")
            logger.info(f"Loading MinerU model - Device: {self.device or 'cuda:0'}")

    def parse_document(self, document_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Parse a PDF document using MinerU Python API (pipeline mode).

        Args:
            document_path: Path to PDF file inside config.document_dir
            **kwargs: Additional options:
                - start_page: Start page ID (default: 0)
                - end_page: End page ID (default: None for all pages)

        Returns:
            List of dicts with 'page_no' and 'content' for each page
        """
        pdf_path = Path(document_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        start_page = kwargs.get('start_page', 0)
        end_page = kwargs.get('end_page', None)

        try:
            from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
            from mineru.data.data_reader_writer import FileBasedDataWriter
            from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
            from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
            from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
            from mineru.utils.enum_class import MakeMode

            # Read PDF bytes
            pdf_bytes = read_fn(pdf_path)

            # Convert PDF pages if page range specified
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page, end_page)

            # Analyze document using pipeline
            infer_results, image_lists, pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                [pdf_bytes],
                [self.lang],
                parse_method=self.parse_method,
                formula_enable=self.formula_enable,
                table_enable=self.table_enable
            )

            # Log after model loading completes
            logger.info(f"Parsing document with MinerU model...")

            # Process results
            images_list = image_lists[0]
            pdf_doc = pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]

            # Prepare environment for image writing
            file_name = pdf_path.stem
            local_image_dir, _ = prepare_env(self.output_dir, file_name, self.parse_method)
            image_writer = FileBasedDataWriter(local_image_dir)

            # Convert to middle JSON
            middle_json = pipeline_result_to_middle_json(
                infer_results[0], images_list, pdf_doc, image_writer,
                _lang, _ocr_enable, self.formula_enable
            )

            pdf_info = middle_json["pdf_info"]

            # Save pdf_info to JSON file for later use (e.g., image context extraction)
            pdf_info_path = self.output_dir / file_name / self.parse_method / "pdf_info.json"
            pdf_info_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pdf_info_path, 'w', encoding='utf-8') as f:
                json.dump(pdf_info, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved pdf_info to {pdf_info_path}")

            # Generate markdown content
            image_dir = str(os.path.basename(local_image_dir))
            md_content = pipeline_union_make(
                pdf_info,
                MakeMode.MM_MD,
                image_dir
            )

            # Clean markdown content
            cleaned_content = self._clean_markdown(md_content)

            logger.info(f"Parsing completed: {file_name}")
            logger.info(f"Processed {len(images_list)} page(s)")
            logger.info(f"Original size: {len(md_content)} chars, Cleaned size: {len(cleaned_content)} chars")

            # Return in unified format
            results = [{
                'page_no': 0,
                'content': cleaned_content,
                'parse_method': f'mineru-pipeline-{self.parse_method}'
            }]

            return results

        except Exception as e:
            logger.error(f"MinerU parsing failed: {e}")
            raise RuntimeError(f"Failed to parse PDF {pdf_path}: {e}")

    def _clean_markdown(self, text: str) -> str:
        """Remove image references, table tags, and clean up markdown for text retrieval."""
        # Remove base64 image data URLs (data:image/...;base64,...)
        text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '', text)

        # Remove markdown image syntax with file paths or URLs
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'[Image: \1]', text)

        # Remove HTML image tags
        text = re.sub(r'<img[^>]*>', '', text)

        # Remove standalone base64 strings (very long alphanumeric sequences)
        text = re.sub(r'\b[A-Za-z0-9+/]{200,}={0,2}\b', '', text)

        # Remove HTML tables (everything inside <table></table> tags)
        text = re.sub(r'<table[^>]*>.*?</table>', '', text, flags=re.DOTALL)

        # Remove other common HTML tags
        text = re.sub(r'</?div[^>]*>', '', text)
        text = re.sub(r'</?span[^>]*>', '', text)

        # Remove excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()
