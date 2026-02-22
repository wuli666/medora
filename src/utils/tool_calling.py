from __future__ import annotations

import json
from typing import Any, Mapping

from langchain_core.messages import AIMessage


def latest_ai_message(state: Mapping[str, Any]) -> AIMessage | None:
    messages = state.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None

def parse_tool_payload(tool_name: str, raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        payload = raw
    else:
        text = str(raw) if raw is not None else ""
        try:
            payload = json.loads(text)
        except Exception:
            return {
                "tool": tool_name,
                "ok": False,
                "data": {},
                "error": {"code": "INVALID_JSON", "message": text[:500]},
                "meta": {"version": "1.0"},
            }

    if not isinstance(payload, dict):
        return {
            "tool": tool_name,
            "ok": False,
            "data": {},
            "error": {"code": "INVALID_PAYLOAD", "message": "tool payload must be JSON object"},
            "meta": {"version": "1.0"},
        }

    ok = bool(payload.get("ok"))
    data = payload.get("data")
    if not isinstance(data, dict):
        data = {}
    error = payload.get("error")
    if ok:
        error = None
    elif not isinstance(error, dict):
        error = {"code": "UNKNOWN_ERROR", "message": "tool returned failure without error payload"}
    return {
        "tool": str(payload.get("tool") or tool_name),
        "ok": ok,
        "data": data,
        "error": error,
        "meta": payload.get("meta") if isinstance(payload.get("meta"), dict) else {"version": "1.0"},
    }


def tool_text_or_error(payload: dict[str, Any], key: str = "analysis_text") -> str:
    if payload.get("ok"):
        data = payload.get("data", {})
        value = data.get(key, "")
        return str(value) if value is not None else ""
    error = payload.get("error", {})
    code = error.get("code", "UNKNOWN_ERROR")
    msg = error.get("message", "")
    return f"[{code}] {msg}".strip()
