import asyncio
import base64
import logging
import re
import tempfile
from pathlib import Path

from mineru.cli.common import do_parse, read_fn
from mineru.utils.enum_class import MakeMode

logger = logging.getLogger(__name__)


def _parse_pdf_sync(pdf_path: str) -> dict:
    """Synchronous MinerU v2.x PDF parsing with OCR and layout analysis."""
    pdf_bytes = read_fn(Path(pdf_path))
    pdf_file_name = Path(pdf_path).stem

    with tempfile.TemporaryDirectory() as tmp_dir:
        do_parse(
            output_dir=tmp_dir,
            pdf_file_names=[pdf_file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            backend="pipeline",
            parse_method="auto",
            formula_enable=False,
            table_enable=False,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            f_make_md_mode=MakeMode.MM_MD,
        )

        # Read generated markdown
        md_dir = Path(tmp_dir) / pdf_file_name / "auto"
        md_file = md_dir / f"{pdf_file_name}.md"
        md_content = md_file.read_text(encoding="utf-8") if md_file.exists() else ""

        # Strip image references from markdown (images are returned separately as base64)
        text = re.sub(r"!\[.*?\]\(images/[^)]*\)", "", md_content).strip()

        # Collect all extracted images as base64
        images = []
        image_dir = md_dir / "images"
        if image_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for img_file in sorted(image_dir.glob(ext)):
                    img_b64 = base64.b64encode(img_file.read_bytes()).decode()
                    images.append(img_b64)

        return {"text": text, "images": images}


async def parse_pdf(pdf_path: str) -> dict:
    """Async wrapper around MinerU PDF parsing to avoid blocking the event loop."""
    return await asyncio.to_thread(_parse_pdf_sync, pdf_path)
