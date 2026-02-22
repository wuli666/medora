"""Medical analysis tools backed by chat models."""

from __future__ import annotations

import base64
import binascii
import time
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.llm.model_factory import get_chat_model
from src.prompts.prompts import MEDGEMMA_IMAGE_PROMPT, MEDGEMMA_TEXT_PROMPT
from src.tool.envelope import error_payload, ok_payload


def _stringify_content(content: Any) -> str:
    """Convert arbitrary model content into a safe string."""
    return str(content) if content is not None else ""


def _normalize_image_base64(image_base64: str) -> str:
    raw = (image_base64 or "").strip()
    if raw.startswith("data:"):
        _, _, tail = raw.partition(",")
        raw = tail.strip()
    return "".join(raw.split())


@tool
async def analyze_medical_text(
    medical_text: Annotated[str, "待分析的医疗文本，必须为非空字符串。"]
) -> str:
    """文本医疗分析工具（返回 envelope JSON 字符串）。

    调用时机: 输入包含医学文本（如病史、检查报告、症状描述等）。
    参数要求: `medical_text` 必须是非空字符串。
    成功字段: `data.analysis_text`。
    常见错误码: `MODEL_UNAVAILABLE`、`UPSTREAM_ERROR`。
    """
    start_ts = time.perf_counter()
    prompt = MEDGEMMA_TEXT_PROMPT.format(medical_text=medical_text)
    llm = get_chat_model("TOOLER_TEXT", default_model="qwen-plus", temperature=0.2)
    if llm is None:
        return error_payload(
            "analyze_medical_text",
            "MODEL_UNAVAILABLE",
            "文本分析模型不可用：请检查对应 agent 的模型配置。",
            start_ts,
        )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return ok_payload(
            "analyze_medical_text",
            {"analysis_text": _stringify_content(response.content)},
            start_ts,
        )
    except Exception as e:
        return error_payload(
            "analyze_medical_text",
            "UPSTREAM_ERROR",
            f"MedGemma 文本分析失败: {e}",
            start_ts,
        )


@tool
async def analyze_medical_image(
    image_base64: Annotated[str, "医疗图像的 base64 字符串（原始内容，不含 markdown 包裹）。"],
    clinical_context: Annotated[str, "可选临床背景信息（主诉/病史/部位/时长等），可为空。"] = "",
) -> str:
    """医疗图像分析工具（返回 envelope JSON 字符串）。

    调用时机: 输入包含医学图像（X 光/CT 截图/皮损照片等）。
    参数要求: `image_base64` 必须是可解码的 base64 图像内容（可传原始 base64 或 data URL）；`clinical_context` 可空。
    成功字段: `data.analysis_text`、`data.key_facts`（当前通常为空列表）。
    常见错误码: `MODEL_UNAVAILABLE`、`UPSTREAM_ERROR`。
    """
    start_ts = time.perf_counter()
    prompt_text = MEDGEMMA_IMAGE_PROMPT.format(
        clinical_context=clinical_context or "无额外临床背景信息"
    )
    llm = get_chat_model("TOOLER_IMAGE", default_model="qwen-plus", temperature=0.2)
    if llm is None:
        return error_payload(
            "analyze_medical_image",
            "MODEL_UNAVAILABLE",
            "图像分析模型不可用：请检查对应 agent 的模型配置。",
            start_ts,
        )
    normalized_image_base64 = _normalize_image_base64(image_base64)
    if not normalized_image_base64:
        return error_payload(
            "analyze_medical_image",
            "INVALID_INPUT",
            "图像分析失败: image_base64 为空。",
            start_ts,
        )
    try:
        base64.b64decode(normalized_image_base64, validate=True)
    except (ValueError, binascii.Error):
        return error_payload(
            "analyze_medical_image",
            "INVALID_INPUT",
            "图像分析失败: image_base64 非法，无法解码。",
            start_ts,
        )
    try:
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{normalized_image_base64}"},
                },
            ]
        )
        response = await llm.ainvoke([message])
        analysis_text = _stringify_content(response.content)
        return ok_payload(
            "analyze_medical_image",
            {
                "key_facts": [],
                "analysis_text": analysis_text,
            },
            start_ts,
        )
    except Exception as e:
        return error_payload(
            "analyze_medical_image",
            "UPSTREAM_ERROR",
            f"MedGemma 图像分析失败: {e}",
            start_ts,
        )
