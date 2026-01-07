import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def export_chunks_to_jsonl(
    chunks: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    Export chunks to JSONL format for passage dataset.

    Each chunk is converted to the format: {"text": "chunk_content"}
    Empty or whitespace-only chunks are skipped.

    Args:
        chunks: List of chunk dicts from RecursiveChunker.chunk_text()
        output_path: Path to save the JSONL file

    Returns:
        Path to the saved JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert chunks to JSONL format
    valid_chunks = 0
    skipped_chunks = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            content = chunk.get('content', '').strip()

            # Skip empty chunks
            if not content:
                skipped_chunks += 1
                continue

            # Write in JSONL format: {"text": "content"}
            json_line = json.dumps({"text": content}, ensure_ascii=False)
            f.write(json_line + '\n')
            valid_chunks += 1

    logger.info(f"Exported {valid_chunks} chunks to {output_path}")
    if skipped_chunks > 0:
        logger.info(f"Skipped {skipped_chunks} empty chunks")

    return str(output_path)
