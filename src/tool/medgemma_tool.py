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
        return "\n".join(f"- {item}" for item in items if str(item).strip()) or "- None"

    return (
        "1. Patient Profile\n"
        f"{_join(analysis.patient_profile)}\n\n"
        "2. Primary Diagnoses\n"
        f"{_join(analysis.main_diagnoses)}\n\n"
        "3. Current Medication Plan\n"
        f"{_join(analysis.medications)}\n\n"
        "4. Key Test Indicators and Abnormal Values\n"
        f"{_join(analysis.abnormal_indicators)}\n\n"
        "5. Risk Assessment\n"
        f"{_join(analysis.risk_assessment)}\n\n"
        "6. Attention Points\n"
        f"{_join(analysis.follow_up_points)}"
    )


def _render_image_analysis_text(analysis: MedicalImageAnalysis) -> str:
    def _join(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items if str(item).strip()) or "- None"

    return (
        f"1. Image Type\n- {analysis.image_type or 'Not specified'}\n\n"
        "2. Primary Findings\n"
        f"{_join(analysis.key_findings)}\n\n"
        "3. Abnormal Regions\n"
        f"{_join(analysis.abnormal_regions)}\n\n"
        "4. Clinical Implications\n"
        f"{_join(analysis.clinical_implications)}\n\n"
        "5. Recommended Further Checks\n"
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
        "Medical text input for analysis. Prioritize this tool whenever medical text is provided; the parameter must be non-empty original text.",
    ]
) -> str:
    """Medical text evidence extraction tool (returns envelope JSON string).

    Planner call strategy:
    1. If the input contains medical text (chief complaint/symptoms/history/exam results/diagnosis/medication), call this tool by default.
    2. Skip only when no analyzable medical text exists.

    Parameters:
    - `medical_text`: Non-empty. Keep the user's original medical description as much as possible.

    Success fields:
    - `data.analysis_text`: Structured medical analysis text for planner/reflector/summarizer.

    Common error codes:
    - `MODEL_UNAVAILABLE`, `UPSTREAM_ERROR`.
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
            f"MedGemma text analysis failed: {e}",
            start_ts,
        )


@tool
async def analyze_medical_image(
    image_base64: Annotated[
        str,
        "Medical image base64 content. If image/photo input exists (X-ray/CT/MRI/ultrasound/skin lesion/report screenshot, etc.), prioritize this tool; raw base64 or data URL is supported.",
    ],
    clinical_context: Annotated[
        str,
        "Clinical context (chief complaint/history/body part/duration/prior tests). Fill when possible to improve interpretation accuracy; can be empty.",
    ] = "",
) -> str:
    """Medical image interpretation tool (returns envelope JSON string).

    Planner call strategy:
    1. If any medical image or exam screenshot is provided, call this tool instead of relying on text-only inference.
    2. When images exist, this tool should usually run in parallel with `analyze_medical_text`.

    Parameters:
    - `image_base64`: Must be decodable image content (data URL supported).
    - `clinical_context`: Recommended; at least symptoms or exam context can improve quality.

    Success fields:
    - `data.analysis_text`: Image analysis result.
    - `data.key_facts`: Key findings.

    Common error codes:
    - `MODEL_UNAVAILABLE`, `INVALID_INPUT`, `UPSTREAM_ERROR`.
    """
    start_ts = time.perf_counter()

    llm = get_chat_model("TOOLER_IMAGE", default_model="qwen-plus", temperature=0.2)
    if llm is None:
        return error_payload(
            "analyze_medical_image",
            "MODEL_UNAVAILABLE",
            "Image analysis model is unavailable. Please check agent model configuration.",
            start_ts,
        )
    normalized_image_base64 = normalize_image_base64(image_base64)
    if not normalized_image_base64:
        return error_payload(
            "analyze_medical_image",
            "INVALID_INPUT",
            "Image analysis failed: image_base64 is empty.",
            start_ts,
        )
    try:
        base64.b64decode(normalized_image_base64, validate=True)
    except (ValueError, binascii.Error):
        return error_payload(
            "analyze_medical_image",
            "INVALID_INPUT",
            "Image analysis failed: image_base64 is invalid or cannot be decoded.",
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
                            "text": f"Clinical Context:\n{clinical_context or 'No additional clinical context provided'}",
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
            f"MedGemma image analysis failed: {e}",
            start_ts,
        )
