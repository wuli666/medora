import asyncio

import api.main as main


def _run(coro):
    return asyncio.run(coro)


def test_download_report_pdf_success(monkeypatch):
    async def _fake_get_run(_run_id: str):
        return {
            "done": True,
            "response": {
                "summary": "# 健康管理与随访报告\n\n## 摘要\n状态稳定。",
                "summary_struct": {
                    "report_title": "健康管理与随访报告",
                    "brief_summary": "状态稳定。",
                    "key_findings": ["关键发现A"],
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
                "summary": "今天天气不错，阳光明媚。",
                "summary_struct": {
                    "report_title": "通用咨询回复",
                    "brief_summary": "今天天气不错，阳光明媚。",
                    "key_findings": ["非医疗咨询"],
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
