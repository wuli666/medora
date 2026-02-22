from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any


def _meta(start_ts: float) -> dict[str, Any]:
    return {
        "version": "1.0",
        "ts": datetime.now(timezone.utc).isoformat(),
        "latency_ms": int((time.perf_counter() - start_ts) * 1000),
    }


def ok_payload(tool: str, data: dict[str, Any], start_ts: float) -> str:
    return json.dumps(
        {
            "tool": tool,
            "ok": True,
            "data": data,
            "error": None,
            "meta": _meta(start_ts),
        },
        ensure_ascii=False,
    )


def error_payload(
    tool: str,
    code: str,
    message: str,
    start_ts: float,
    data: dict[str, Any] | None = None,
) -> str:
    return json.dumps(
        {
            "tool": tool,
            "ok": False,
            "data": data or {},
            "error": {"code": code, "message": message},
            "meta": _meta(start_ts),
        },
        ensure_ascii=False,
    )

