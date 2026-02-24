"""Web and local-RAG retrieval tools for medical evidence collection."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any

import chromadb
from langchain_core.tools import tool
from tavily import TavilyClient

from src.config.settings import settings
from src.tool.envelope import error_payload, ok_payload

TAVILY_API_KEY = settings.TAVILY_API_KEY
CHROMA_DIR = settings.CHROMA_DIR


@tool("web_search", return_direct=False)
async def web_search(
    query: Annotated[str, "联网检索查询词，必须为非空字符串。"]
) -> str:
    """联网检索工具（Tavily，返回 envelope JSON 字符串）。

    参数要求: `query` 必须是非空字符串。
    成功字段: `data.query`、`data.results`(title/url/snippet)、`data.summary`。
    常见错误码: `INVALID_INPUT`、`MISSING_CONFIG`、`UPSTREAM_ERROR`。
    """
    start_ts = time.perf_counter()
    normalized_query = (query or "").strip()
    if not normalized_query:
        return error_payload("web_search", "INVALID_INPUT", "query must not be empty", start_ts)

    max_results = settings.WEB_SEARCH_MAX_RESULTS
    if not TAVILY_API_KEY:
        return error_payload("web_search", "MISSING_CONFIG", "Tavily API key 未配置", start_ts)
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = await asyncio.to_thread(client.search, normalized_query, max_results=max_results)
        results: list[dict[str, Any]] = []
        summary_lines: list[str] = []
        for item in response.get("results", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", ""))
            url = str(item.get("url", ""))
            snippet = str(item.get("content", ""))[:300]
            results.append({"title": title, "url": url, "snippet": snippet})
            summary_lines.append(f"- {title}: {snippet}\n  来源: {url}")
        return ok_payload(
            "web_search",
            {
                "query": normalized_query,
                "results": results,
                "summary": "\n".join(summary_lines) if summary_lines else "未找到相关结果。",
                "source_count": len(results),
            },
            start_ts,
        )
    except Exception as e:
        return error_payload("web_search", "UPSTREAM_ERROR", f"网络搜索失败: {e}", start_ts)


@tool("rag_search", return_direct=False)
async def rag_search(
    query: Annotated[str, "本地知识库检索查询词，必须为非空字符串。"]
) -> str:
    """本地知识库检索工具（Chroma，返回 envelope JSON 字符串）。

    参数要求: `query` 必须是非空字符串。
    成功字段: `data.query`、`data.documents`、`data.summary`。
    常见错误码: `INVALID_INPUT`、`INTERNAL_ERROR`。
    """
    start_ts = time.perf_counter()
    normalized_query = (query or "").strip()
    if not normalized_query:
        return error_payload("rag_search", "INVALID_INPUT", "query must not be empty", start_ts)

    n_results = settings.RAG_TOP_K
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection("medical_knowledge")
        response = await asyncio.to_thread(
            collection.query,
            query_texts=[normalized_query],
            n_results=n_results,
        )
        docs = []
        if response and response.get("documents") and response["documents"][0]:
            docs = [str(d) for d in response["documents"][0]]
        summary = "\n---\n".join(docs) if docs else "知识库中未找到相关内容。"
        return ok_payload(
            "rag_search",
            {
                "query": normalized_query,
                "documents": docs,
                "summary": summary,
                "source_count": len(docs),
            },
            start_ts,
        )
    except Exception as e:
        return error_payload("rag_search", "INTERNAL_ERROR", f"RAG 检索失败: {e}", start_ts)
