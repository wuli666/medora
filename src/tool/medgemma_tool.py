"""Medical analysis tools backed by chat models."""

from __future__ import annotations

import base64
import binascii
import time
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.config.logger import get_logger
from src.graph.state import MedicalImageAnalysis, MedicalTextAnalysis
from src.llm.model_factory import get_chat_model
from src.prompts.prompts import MEDGEMMA_IMAGE_PROMPT, MEDGEMMA_TEXT_PROMPT
from src.tool.envelope import error_payload, ok_payload

_logger = get_logger(__name__)


def normalize_image_base64(image_base64: str) -> str:
    raw = (image_base64 or "").strip()
    if raw.startswith("data:"):
        _, _, tail = raw.partition(",")
        raw = tail.strip()
    return "".join(raw.split())


def _render_text_analysis_text(analysis: MedicalTextAnalysis) -> str:
    def _join(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items if str(item).strip()) or "- 暂无"

    return (
        "1. 患者基本信息\n"
        f"{_join(analysis.patient_profile)}\n\n"
        "2. 主要诊断\n"
        f"{_join(analysis.main_diagnoses)}\n\n"
        "3. 现有用药方案\n"
        f"{_join(analysis.medications)}\n\n"
        "4. 关键检查指标及异常值\n"
        f"{_join(analysis.abnormal_indicators)}\n\n"
        "5. 风险评估\n"
        f"{_join(analysis.risk_assessment)}\n\n"
        "6. 需要关注的问题\n"
        f"{_join(analysis.follow_up_points)}"
    )


def _render_image_analysis_text(analysis: MedicalImageAnalysis) -> str:
    def _join(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items if str(item).strip()) or "- 暂无"

    return (
        f"1. 图像类型识别\n- {analysis.image_type or '未说明'}\n\n"
        "2. 主要发现\n"
        f"{_join(analysis.key_findings)}\n\n"
        "3. 异常区域描述\n"
        f"{_join(analysis.abnormal_regions)}\n\n"
        "4. 临床意义评估\n"
        f"{_join(analysis.clinical_implications)}\n\n"
        "5. 建议进一步检查\n"
        f"{_join(analysis.recommended_checks)}"
    )


async def _invoke_structured(llm, schema, messages):
    try:
        structured_llm = llm.with_structured_output(schema, include_raw=True)
        response = await structured_llm.ainvoke(messages)
        if isinstance(response, dict):
            parsed = response.get("parsed")
            if isinstance(parsed, schema):
                return parsed
            return schema.model_validate(parsed)
    except TypeError:
        pass

    structured_llm = llm.with_structured_output(schema)
    response = await structured_llm.ainvoke(messages)
    if isinstance(response, schema):
        return response
    return schema.model_validate(response)


@tool
async def analyze_medical_text(
    medical_text: Annotated[
        str,
        "医学文本分析输入。只要用户提供了病历文字信息，就应优先调用本工具；参数必须为非空原文文本。",
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

    normalized_text = (medical_text or "").strip()
    if not normalized_text:
        return error_payload(
            "analyze_medical_text",
            "INVALID_INPUT",
            "medical_text must not be empty",
            start_ts,
        )

    llm = get_chat_model("TOOLER_TEXT", default_model="qwen-plus", temperature=0.2)
    try:
        parsed = await _invoke_structured(
            llm,
            MedicalTextAnalysis,
            [
                SystemMessage(content=MEDGEMMA_TEXT_PROMPT),
                HumanMessage(content=normalized_text),
            ],
        )
        return ok_payload(
            "analyze_medical_text",
            {
                "analysis_struct": parsed.model_dump(),
                "analysis_text": _render_text_analysis_text(parsed),
            },
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
        parsed = await _invoke_structured(
            llm,
            MedicalImageAnalysis,
            [
                SystemMessage(content=MEDGEMMA_IMAGE_PROMPT),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": f"临床背景:\n{clinical_context or '无额外临床背景信息'}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{normalized_image_base64}"
                            },
                        },
                    ]
                ),
            ],
        )

        return ok_payload(
            "analyze_medical_image",
            {
                "analysis_struct": parsed.model_dump(),
                "analysis_text": _render_image_analysis_text(parsed),
                "key_facts": parsed.key_findings,
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
