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
from src.config.logger import get_logger

_logger = get_logger(__name__)

def normalize_image_base64(image_base64: str) -> str:
    raw = (image_base64 or "").strip()
    if raw.startswith("data:"):
        _, _, tail = raw.partition(",")
        raw = tail.strip()
    return "".join(raw.split())


@tool
async def analyze_medical_text(
    medical_text: Annotated[
        str,
        "医学文本分析输入。只要用户提供了症状/病史/检查报告/用药信息中的任一项，就应优先调用本工具；参数必须为非空原文文本。",
    ]
) -> str:
    """医学文本证据提取工具（返回 envelope JSON 字符串）。

    Planner 调用策略:
    1. 只要输入包含医学文字信息（主诉/症状/病史/检查结果/诊断/用药），默认应调用。
    3. 仅在用户完全没有可分析医学文本时才可跳过。

    参数要求:
    - `medical_text`: 非空，尽量保留用户原始医学描述，避免过度改写。

    成功字段:
    - `data.analysis_text`: 结构化医学分析结论（供 planner/reflector/summarizer 使用）。

    常见错误码:
    - `MODEL_UNAVAILABLE`、`UPSTREAM_ERROR`。
    """
    _logger.debug(f"Invoking analyze_medical_text with input: {medical_text[:100]}...")
    start_ts = time.perf_counter()
    prompt = MEDGEMMA_TEXT_PROMPT.format(medical_text=medical_text)
    llm = get_chat_model("TOOLER_TEXT", default_model="qwen-plus", temperature=0.2)
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return ok_payload(
            "analyze_medical_text",
            {"analysis_text": str(response.content) if response.content is not None else ""},
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
    image_base64: Annotated[
        str,
        "医学图像 base64 内容。只要存在影像/照片输入（X光/CT/MRI/超声/皮损/报告截图等），应优先调用本工具；可传原始 base64 或 data URL。",
    ],
    clinical_context: Annotated[
        str,
        "临床上下文（主诉/病史/部位/时长/既往检查），建议尽量填写以提升影像判读准确性；可为空。",
    ] = "",
) -> str:
    """医学图像判读工具（返回 envelope JSON 字符串）。

    Planner 调用策略（提高召回）:
    1. 只要接收到任何医学图像或检查截图，应调用本工具，不要仅依赖文本推断。
    3. 图像存在时，本工具通常应与 `analyze_medical_text` 并行调用以补齐证据。

    参数要求:
    - `image_base64`: 必须是可解码图像内容（支持 data URL）。
    - `clinical_context`: 推荐填写，至少包含症状或检查背景，可显著提高输出质量。

    成功字段:
    - `data.analysis_text`: 图像分析结果。
    - `data.key_facts`: 关键发现（当前通常为空列表）。

    常见错误码:
    - `MODEL_UNAVAILABLE`、`INVALID_INPUT`、`UPSTREAM_ERROR`。
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
    normalized_image_base64 = normalize_image_base64(image_base64)
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
        analysis_text = str(response.content) if response.content is not None else ""
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
