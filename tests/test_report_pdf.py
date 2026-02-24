from io import BytesIO

import fitz

from src.utils.report_pdf import build_report_pdf_bytes


def test_build_report_pdf_bytes_with_struct() -> None:
    payload = {
        "report_title": "Health Management & Follow-up Report",
        "brief_summary": "Overall condition is stable; continue routine monitoring and scheduled rechecks.",
        "key_findings": ["Recent blood pressure is elevated and needs continued monitoring."],
        "medication_reminders": ["Amlodipine 5mg once daily in the morning."],
        "follow_up_tips": ["If blood pressure remains >=160/100 mmHg for 3 days, recheck within 3 days."],
    }
    pdf_bytes = build_report_pdf_bytes(payload, "", "run-test-1")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000

    doc = fitz.open(stream=BytesIO(pdf_bytes).getvalue(), filetype="pdf")
    assert doc.page_count >= 1
    text = doc[0].get_text()
    doc.close()
    assert "Health Management & Follow-up Report" in text
    assert "Summary" in text
    assert "Key Findings" in text


def test_build_report_pdf_bytes_with_text_fallback() -> None:
    summary_text = """# Health Management & Follow-up Report

## Summary
Overall condition is stable.

## Key Findings
- Mild inflammatory markers are elevated.
"""
    pdf_bytes = build_report_pdf_bytes(None, summary_text, "run-test-2")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
