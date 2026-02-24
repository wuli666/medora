import asyncio
import base64
import logging
import re
import tempfile
from pathlib import Path

import magic_pdf.model as _model_cfg

# MinerU v1.x: enable local model inference
_model_cfg.__use_inside_model__ = True

from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset, SupportedPdfParseMethod
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

logger = logging.getLogger(__name__)


def _parse_pdf_sync(pdf_path: str) -> dict:
    """Synchronous MinerU-based PDF parsing with OCR and layout analysis."""
    pdf_bytes = Path(pdf_path).read_bytes()

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_dir = Path(tmp_dir) / "images"
        image_dir.mkdir()
        image_writer = FileBasedDataWriter(str(image_dir))

        ds = PymuDocDataset(pdf_bytes)

        if ds.classify() == SupportedPdfParseMethod.TXT:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)

        md_content = pipe_result.get_markdown("images")

        # Strip image references from markdown (images are returned separately as base64)
        text = re.sub(r"!\[.*?\]\(images/[^)]*\)", "", md_content).strip()

        # Collect all extracted images as base64
        images = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_file in sorted(image_dir.glob(ext)):
                img_b64 = base64.b64encode(img_file.read_bytes()).decode()
                images.append(img_b64)

        return {"text": text, "images": images}


async def parse_pdf(pdf_path: str) -> dict:
    """Async wrapper around MinerU PDF parsing to avoid blocking the event loop."""
    return await asyncio.to_thread(_parse_pdf_sync, pdf_path)
