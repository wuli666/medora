"""Tavily-powered web search tool."""

from __future__ import annotations

import json
from typing import Any, Literal, cast

from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from src.config import settings


def _error_payload(query: str, error: str) -> str:
    """Build a stable error payload for tool responses."""
    return json.dumps(
        {
            "tool": "web_search",
            "ok": False,
            "query": query,
            "results": [],
            "error": error,
        },
        ensure_ascii=True,
    )


def _normalize_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize Tavily response entries into compact result items."""
    raw_results = response.get("results", [])
    normalized: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
        )
    return normalized


@tool
async def web_search(
    query: str,
    max_results: int = settings.TAVILY_MAX_RESULTS,
    search_depth: str = settings.TAVILY_SEARCH_DEPTH,
) -> str:
    """Search the web using Tavily and return compact JSON results.

    Args:
        query: Search query text.
        max_results: Maximum number of results to return.
        search_depth: Tavily search depth (basic or advanced).
    """
    normalized_query = query.strip()
    if not normalized_query:
        return _error_payload(query, "Query must not be empty.")

    if search_depth not in {"basic", "advanced", "fast", "ultra-fast"}:
        return _error_payload(
            normalized_query,
            "search_depth must be one of: basic, advanced, fast, ultra-fast.",
        )

    safe_search_depth = cast(
        Literal["basic", "advanced", "fast", "ultra-fast"],
        search_depth,
    )

    if settings.TAVILY_API_KEY is None:
        return _error_payload(
            normalized_query,
            "TAVILY_API_KEY is not set. Configure it in your environment.",
        )

    try:
        client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
        response = await client.search(
            query=normalized_query,
            max_results=max_results,
            search_depth=safe_search_depth,
        )
    except Exception as exc:  # pragma: no cover - defensive integration guard
        return _error_payload(normalized_query, f"Tavily request failed: {type(exc).__name__}")

    payload = {
        "tool": "web_search",
        "ok": True,
        "query": normalized_query,
        "results": _normalize_results(response),
        "error": None,
    }
    return json.dumps(payload, ensure_ascii=True)
