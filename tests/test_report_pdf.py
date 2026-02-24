from io import BytesIO

import fitz

from src.utils.report_pdf import build_report_pdf_bytes


def test_build_report_pdf_bytes_with_struct() -> None:
    payload = {
        "report_title": "健康管理与随访报告",
        "brief_summary": "当前情况总体稳定，建议继续规律监测并按时复查。",
        "key_findings": ["近期血压偏高，需持续监测。"],
        "medication_reminders": ["氨氯地平 5mg，每日一次，早晨服用。"],
        "follow_up_tips": ["若连续3天血压>=160/100mmHg，请3天内复查。"],
    }
    pdf_bytes = build_report_pdf_bytes(payload, "", "run-test-1")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000

    doc = fitz.open(stream=BytesIO(pdf_bytes).getvalue(), filetype="pdf")
    assert doc.page_count >= 1
    text = doc[0].get_text()
    doc.close()
    assert "健康管理与随访报告" in text
    assert "摘要" in text
    assert "关键发现" in text


def test_build_report_pdf_bytes_with_text_fallback() -> None:
    summary_text = """# 健康管理与随访报告

## 摘要
整体情况稳定。

## 关键发现
- 存在轻度炎症指标升高。
"""
    pdf_bytes = build_report_pdf_bytes(None, summary_text, "run-test-2")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
