from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

import fitz

from src.config.logger import get_logger

logger = get_logger(__name__)

_TITLE_SIZE = 22
_SUBTITLE_SIZE = 11
_HEADING_SIZE = 14
_BODY_SIZE = 11
_LINE_HEIGHT = 18
_PAGE_WIDTH = 595
_PAGE_HEIGHT = 842
_MARGIN_X = 50
_MARGIN_TOP = 60
_MARGIN_BOTTOM = 60
_META_COLOR = (0.35, 0.35, 0.35)
_TEXT_COLOR = (0.1, 0.1, 0.1)
_RULE_COLOR = (0.75, 0.75, 0.75)


def _contains_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # CJK Extension A
            or 0x3000 <= code <= 0x303F  # CJK Symbols and Punctuation
            or 0xFF00 <= code <= 0xFFEF  # Fullwidth/Halfwidth Forms
        ):
            return True
    return False


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _split_markdown_sections(summary_text: str) -> dict[str, str]:
    text = (summary_text or "").strip()
    if not text:
        return {}
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current = "Body"
    sections[current] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            current = stripped[3:].strip() or "Body"
            sections.setdefault(current, [])
            continue
        if stripped.startswith("# "):
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def _build_sections(summary_struct: dict | None, summary_text: str) -> tuple[str, str, list[tuple[str, list[str]]]]:
    data = summary_struct if isinstance(summary_struct, dict) else {}

    title = str(data.get("report_title", "")).strip() or "Health Management & Follow-up Report"
    brief = str(data.get("brief_summary", "")).strip() or "A report has been generated from the available information."

    key_findings = _coerce_list(data.get("key_findings"))
    medication = _coerce_list(data.get("medication_reminders"))
    follow_up = _coerce_list(data.get("follow_up_tips"))

    if not data:
        parsed = _split_markdown_sections(summary_text)
        brief = parsed.get("Summary", parsed.get("摘要", brief))
        key_findings = _coerce_list(parsed.get("Key Findings", parsed.get("关键发现", ""))) or [
            "No clear key findings at this time."
        ]
        medication = _coerce_list(parsed.get("Medication Reminders", parsed.get("用药提醒", "")))
        follow_up = _coerce_list(parsed.get("Follow-up Tips", parsed.get("随访提示", "")))

    if not key_findings:
        key_findings = ["No clear key findings at this time."]

    sections: list[tuple[str, list[str]]] = [("Key Findings", key_findings)]
    if medication:
        sections.append(("Medication Reminders", medication))
    if follow_up:
        sections.append(("Follow-up Tips", follow_up))
    return title, brief, sections


def _draw_wrapped_text(
    page: fitz.Page,
    text: str,
    x: float,
    y: float,
    width: float,
    fontsize: int,
    fontname: str,
    *,
    align: int = 0,
    color: tuple[float, float, float] = _TEXT_COLOR,
) -> float:
    rect = fitz.Rect(x, y, x + width, _PAGE_HEIGHT - _MARGIN_BOTTOM)
    overflow = page.insert_textbox(
        rect,
        text,
        fontsize=fontsize,
        fontname=fontname,
        align=align,
        color=color,
    )
    used_height = (rect.height - overflow) if overflow >= 0 else rect.height
    return max(_LINE_HEIGHT, used_height)


def _pick_font(primary: str, probe: str, fallback: str) -> str:
    try:
        _ = fitz.get_text_length(probe, fontname=primary, fontsize=_BODY_SIZE)
        return primary
    except Exception:
        return fallback


def build_report_pdf_bytes(summary_struct: dict | None, summary_text: str, run_id: str) -> bytes:
    title, brief, sections = _build_sections(summary_struct, summary_text)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_no = (run_id or "").strip()[:8] or "N/A"

    doc = fitz.open()
    page = doc.new_page(width=_PAGE_WIDTH, height=_PAGE_HEIGHT)
    y = _MARGIN_TOP
    text_width = _PAGE_WIDTH - _MARGIN_X * 2

    # Choose font by content to avoid stretched English glyph spacing with CJK font.
    combined_text = "\n".join(
        [title, brief, *(section_title for section_title, _ in sections), *(item for _, items in sections for item in items)]
    )
    if _contains_cjk(combined_text):
        try:
            base_font = "china-s"
            _ = fitz.get_text_length("中文", fontname=base_font, fontsize=_BODY_SIZE)
        except Exception:
            logger.warning("[report_pdf] fallback to helv due to unavailable CJK font")
            base_font = "helv"
        title_font = base_font
        heading_font = base_font
        body_font = base_font
        meta_font = base_font
    else:
        title_font = _pick_font("Times-Bold", "Title", "helv")
        heading_font = _pick_font("Times-Bold", "Heading", "helv")
        body_font = _pick_font("Times-Roman", "Body", "helv")
        meta_font = _pick_font("Helvetica", "Meta", body_font)

    def ensure_space(need: float) -> None:
        nonlocal page, y
        if y + need <= _PAGE_HEIGHT - _MARGIN_BOTTOM:
            return
        page = doc.new_page(width=_PAGE_WIDTH, height=_PAGE_HEIGHT)
        y = _MARGIN_TOP

    ensure_space(80)
    y += _draw_wrapped_text(page, title, _MARGIN_X, y, text_width, _TITLE_SIZE, title_font, color=_TEXT_COLOR)
    y += 10
    subtitle = f"Generated at: {created_at}    Report ID: {report_no}"
    y += _draw_wrapped_text(page, subtitle, _MARGIN_X, y, text_width, _SUBTITLE_SIZE, meta_font, color=_META_COLOR)
    y += 8
    page.draw_line(
        fitz.Point(_MARGIN_X, y),
        fitz.Point(_PAGE_WIDTH - _MARGIN_X, y),
        color=_RULE_COLOR,
        width=0.8,
    )
    y += 14

    ensure_space(60)
    y += _draw_wrapped_text(page, "Summary", _MARGIN_X, y, text_width, _HEADING_SIZE, heading_font, color=_TEXT_COLOR)
    y += 6
    y += _draw_wrapped_text(page, brief, _MARGIN_X, y, text_width, _BODY_SIZE, body_font, color=_TEXT_COLOR)
    y += 16

    for section_title, items in sections:
        ensure_space(60)
        y += _draw_wrapped_text(page, section_title, _MARGIN_X, y, text_width, _HEADING_SIZE, heading_font, color=_TEXT_COLOR)
        y += 6
        page.draw_line(
            fitz.Point(_MARGIN_X, y),
            fitz.Point(_PAGE_WIDTH - _MARGIN_X, y),
            color=_RULE_COLOR,
            width=0.6,
        )
        y += 6
        for item in items:
            ensure_space(40)
            bullet = f"- {item}"
            y += _draw_wrapped_text(page, bullet, _MARGIN_X + 8, y, text_width - 8, _BODY_SIZE, body_font, color=_TEXT_COLOR)
        y += 12

    # Footer page number for formal report style.
    for i, p in enumerate(doc, start=1):
        footer = f"Page {i} of {doc.page_count}"
        p.insert_textbox(
            fitz.Rect(_MARGIN_X, _PAGE_HEIGHT - 34, _PAGE_WIDTH - _MARGIN_X, _PAGE_HEIGHT - 18),
            footer,
            fontsize=9,
            fontname=meta_font,
            align=1,
            color=_META_COLOR,
        )

    buffer = BytesIO()
    doc.save(buffer)
    doc.close()
    return buffer.getvalue()
