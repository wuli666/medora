import asyncio

import api.main as main


def _run(coro):
    return asyncio.run(coro)


def test_download_report_pdf_success(monkeypatch):
    async def _fake_get_run(_run_id: str):
        return {
            "done": True,
            "response": {
                "summary": "# Health Management & Follow-up Report\n\n## Summary\nStable condition.",
                "summary_struct": {
                    "report_title": "Health Management & Follow-up Report",
                    "brief_summary": "Stable condition.",
                    "key_findings": ["Key finding A"],
                    "medication_reminders": [],
                    "follow_up_tips": [],
                },
            },
        }

    monkeypatch.setattr(main, "get_run", _fake_get_run)
    response = _run(main.download_report_pdf("run-123"))
    assert response.status_code == 200
    assert response.media_type == "application/pdf"
    assert "attachment;" in response.headers.get("content-disposition", "")


def test_download_report_pdf_not_found(monkeypatch):
    async def _fake_get_run(_run_id: str):
        return None

    monkeypatch.setattr(main, "get_run", _fake_get_run)
    try:
        _run(main.download_report_pdf("missing"))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 404
    else:
        raise AssertionError("Expected HTTPException 404")


def test_download_report_pdf_not_done(monkeypatch):
    async def _fake_get_run(_run_id: str):
        return {"done": False, "response": {}}

    monkeypatch.setattr(main, "get_run", _fake_get_run)
    try:
        _run(main.download_report_pdf("pending"))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 409
    else:
        raise AssertionError("Expected HTTPException 409")


def test_download_report_pdf_non_formal_report(monkeypatch):
    async def _fake_get_run(_run_id: str):
        return {
            "done": True,
            "response": {
                "summary": "Nice weather today, sunny and clear.",
                "summary_struct": {
                    "report_title": "General Consultation Reply",
                    "brief_summary": "Nice weather today, sunny and clear.",
                    "key_findings": ["Non-medical consultation"],
                    "medication_reminders": [],
                    "follow_up_tips": [],
                },
            },
        }

    monkeypatch.setattr(main, "get_run", _fake_get_run)
    try:
        _run(main.download_report_pdf("non-medical"))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 404
    else:
        raise AssertionError("Expected HTTPException 404")
