"""Image explanation tool backed by MedGemma image model."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.llm import get_llm
from src.config import settings


class KeyFact(BaseModel):
    """Structured key fact extracted from image analysis."""

    fact: str = Field(description="Single factual observation about the image.")
    confidence: Literal["low", "medium", "high"] = Field(
        description="Model confidence for this fact."
    )
    evidence: str = Field(
        description="Grounding clue for the fact from image content. Can be empty."
    )


class ImageExplainResult(BaseModel):
    """Tool response payload for image explanation."""

    tool: Literal["image_explain"] = "image_explain"
    ok: bool
    key_facts: list[KeyFact] = Field(default_factory=list)
    error: str | None = None


class _StructuredKeyFacts(BaseModel):
    """Structured output contract requested from the image model."""

    key_facts: list[KeyFact] = Field(default_factory=list)


def _error_payload(error: str) -> str:
    """Serialize a stable tool error payload."""
    payload = ImageExplainResult(ok=False, key_facts=[], error=error)
    return payload.model_dump_json(ensure_ascii=True)


def _extract_data_url(image_base64: str) -> tuple[str, str]:
    """Extract mime type and base64 body from raw base64 or data URL."""
    normalized = image_base64.strip()
    if normalized.startswith("data:") and ";base64," in normalized:
        header, encoded = normalized.split(";base64,", 1)
        mime = header[5:] or "image/jpeg"
        return mime, encoded.strip()
    return "image/jpeg", normalized


def _validate_base64_image(encoded: str, max_bytes: int) -> None:
    """Validate base64 image payload and enforce size limits."""
    if not encoded:
        raise ValueError("image_base64 must not be empty.")

    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64 content.") from exc

    if not decoded:
        raise ValueError("image_base64 decoded to empty bytes.")
    if len(decoded) > max_bytes:
        raise ValueError(f"image exceeds max allowed size of {max_bytes} bytes.")


async def _invoke_image_model(mime_type: str, encoded_image: str, prompt: str) -> list[KeyFact]:
    """Invoke MedGemma image model and return structured key facts."""
    llm = get_llm(
        model=settings.MEDGEMMA_IMAGE_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.MEDGEMMA_IMAGE_TEMPERATURE,
    )
    structured_llm = llm.llm.with_structured_output(_StructuredKeyFacts)

    instruction = (
        "You are a medical image assistant. Extract concise key facts only. "
        "Return key_facts as JSON objects with fields: fact, confidence (low|medium|high), "
        "and evidence (required, can be empty string). Do not provide diagnosis."
    )

    user_text = f"{instruction}\n\nTask: {prompt}"
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "source_type": "base64",
                "mime_type": mime_type,
                "data": encoded_image,
            },
            {"type": "text", "text": user_text},
        ]
    )

    print("=========== image_explain request content: ==========")
    print(message.content)

    try:
        result = await structured_llm.ainvoke([message])
    except ValueError:
        # Fallback for runtimes that only support image_url content blocks.
        fallback_message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
                },
                {"type": "text", "text": user_text},
            ]
        )
        result = await structured_llm.ainvoke([fallback_message])
    print("=========== Raw model output (for debugging) ==========")
    print(result)

    if isinstance(result, _StructuredKeyFacts):
        return result.key_facts
    validated = _StructuredKeyFacts.model_validate(result)
    return validated.key_facts


@tool
async def image_explain(
    image_base64: str,
    prompt: str = "Generate key medical facts from this image.",
) -> str:
    """Generate structured key facts from a base64-encoded image.

    Args:
        image_base64: Image encoded as base64, optionally as a data URL.
        prompt: Optional focus instruction for key facts.
    """
    try:
        mime_type, encoded = _extract_data_url(image_base64)
        _validate_base64_image(encoded, settings.IMAGE_EXPLAIN_MAX_BYTES)
    except ValueError as exc:
        return _error_payload(str(exc))

    try:
        print("=========== image_explain prompt: ==========")
        print(prompt)
        key_facts = await _invoke_image_model(mime_type, encoded, prompt)
    except Exception as exc:  # pragma: no cover - defensive integration guard
        detail = str(exc).strip() or "no details"
        return _error_payload(f"Image explanation failed: {type(exc).__name__}: {detail}")

    payload = ImageExplainResult(ok=True, key_facts=key_facts, error=None)
    return payload.model_dump_json(ensure_ascii=True)
