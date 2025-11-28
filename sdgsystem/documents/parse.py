from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
import logging
import io
import re
import os
import fitz
import base64
from io import BytesIO

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


# class of DotsOCR Parser
class DotsOCRParser(BaseParser):
    """
    DotsOCR parser using vLLM for PDF document processing.

    Workflow:
    1. Convert PDF pages to images
    2. Process each page through DotsOCR via vLLM API
    3. Extract text and layout information
    """

    DEFAULT_PROMPT = """OCR with format: <|layout|><|text|>"""

    def __init__(
        self,
        config: ParserConfig,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "model",
        dpi: int = 200,
        max_workers: int = 4,
        temperature: float = 0.0,
        max_tokens: int = 16384
    ) -> None:
        super(DotsOCRParser, self).__init__(config)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.dpi = dpi
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"DotsOCR Parser initialized - Model: {model_name}, DPI: {dpi}, Device: {self.device}")

    def parse_document(self, document_path: str, prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse a PDF document and extract content from all pages.

        Args:
            document_path: Path to PDF file
            prompt: Custom prompt for parsing (uses default if not provided)

        Returns:
            List of dicts with 'page_no' and 'content' for each page
        """
        pdf_path = Path(document_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        prompt = prompt or self.DEFAULT_PROMPT
        logger.info(f"Parsing PDF: {pdf_path}")

        # Convert PDF pages to images
        pdf_doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            images.append(Image.open(io.BytesIO(img_data)))
        pdf_doc.close()

        logger.info(f"Processing {len(images)} pages")

        # Process pages in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_page, img, prompt, i): i
                for i, img in enumerate(images)
            }

            with tqdm(total=len(images), desc="Parsing pages") as pbar:
                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Error on page {page_num}: {e}")
                        results.append({'page_no': page_num, 'content': '', 'error': str(e)})
                    pbar.update(1)

        results.sort(key=lambda x: x['page_no'])
        logger.info("Parsing completed")
        return results

    def _process_page(self, image: Image.Image, prompt: str, page_num: int) -> Dict[str, Any]:
        """Process a single page through vLLM."""
        content = self._call_vllm(image, prompt)
        return {'page_no': page_num, 'content': content}

    def _call_vllm(self, image: Image.Image, prompt: str) -> str:
        """Call the vLLM server with an image and prompt."""
        image_url = self._image_to_base64_url(image)

        # DotsOCR expects image tokens before the prompt
        full_prompt = f"<|img|><|imgpad|><|endofimg|>{prompt}"

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                },
                {
                    "type": "text",
                    "text": full_prompt
                }
            ]
        }]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def _image_to_base64_url(self, image: Image.Image, format: str = 'PNG') -> str:
        """Convert PIL Image to base64 data URL."""
        buffered = BytesIO()
        image.save(buffered, format=format)
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{base64_str}"

    def save_as_markdown(self, results: List[Dict[str, Any]], output_path: str) -> str:
        """
        Combine all pages and save as a single clean markdown file.

        Args:
            results: List of page results from parse_document()
            output_path: Path to save the markdown file

        Returns:
            Path to the saved markdown file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Combine all pages
        all_content = []
        for result in results:
            if 'error' in result:
                logger.warning(f"Skipping page {result['page_no']} due to error")
                continue

            content = result.get('content', '')
            if content:
                # Add page separator
                all_content.append(f"\n\n---\n**Page {result['page_no']}**\n---\n\n")
                all_content.append(content)

        combined_text = ''.join(all_content)

        # Clean the content: remove base64 images and image references
        cleaned_text = self._clean_markdown(combined_text)

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        logger.info(f"Saved clean markdown to {output_path}")
        logger.info(f"Original size: {len(combined_text)} chars, Cleaned size: {len(cleaned_text)} chars")

        return str(output_path)

    def _clean_markdown(self, text: str) -> str:
        """Remove base64 images and clean up markdown for text retrieval."""
        # Remove base64 image data URLs (data:image/...;base64,...)
        text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '[IMAGE_REMOVED]', text)

        # Remove markdown image syntax with base64 or long URLs
        text = re.sub(r'!\[([^\]]*)\]\([^\)]{100,}\)', r'[Image: \1]', text)

        # Remove standalone base64 strings (very long alphanumeric sequences)
        text = re.sub(r'\b[A-Za-z0-9+/]{200,}={0,2}\b', '[BASE64_REMOVED]', text)

        # Remove excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        return text.strip()


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

            # Generate markdown content
            image_dir = str(os.path.basename(local_image_dir))
            md_content = pipeline_union_make(
                pdf_info,
                MakeMode.MM_MD,
                image_dir
            )

            # Clean markdown content
            cleaned_content = self._clean_markdown(md_content)

            # Collect image file paths
            image_paths = []
            for img_path in sorted(os.listdir(local_image_dir)):
                if img_path.endswith(('.jpg')):
                    image_paths.append(os.path.join(os.path.basename(local_image_dir), img_path))

            logger.info(f"Parsing completed: {file_name}")
            logger.info(f"Processed {len(images_list)} page(s)")
            logger.info(f"Original size: {len(md_content)} chars, Cleaned size: {len(cleaned_content)} chars")

            # Return in unified format
            results = [{
                'page_no': 0,
                'content': cleaned_content,
                'image_paths': image_paths,
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
        # text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'[Image: \1]', text)

        # Remove HTML image tags
        # text = re.sub(r'<img[^>]*>', '', text)

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

    def save_as_markdown(self, results: List[Dict[str, Any]], output_path: str) -> str:
        """
        Save parsed results as markdown file.

        Args:
            results: Parsed results from parse_document()
            output_path: Path to save markdown file

        Returns:
            Path to saved markdown file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not results:
            logger.warning("No results to save")
            return str(output_path)

        # MinerU returns combined content for all pages in a single result (already cleaned)
        content = results[0].get('content', '')

        if not content:
            logger.warning("Empty content in results")
            return str(output_path)

        # Save to file (content is already cleaned in parse_document)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Saved markdown to {output_path}")
        logger.info(f"Content size: {len(content)} chars")

        return str(output_path)
