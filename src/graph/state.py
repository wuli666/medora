"""State models for MedGemma LangGraph workflows."""

from __future__ import annotations

import operator
from typing import Any
from typing_extensions import Annotated, TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage


class VerificationReport(TypedDict, total=False):
    """Serialized reflect/verify report stored in state."""

    issues: list[dict[str, Any]]
    needs_tool: bool
    needs_clarification: bool
    confidence: str
    is_safe_to_finalize: bool


class GraphState(TypedDict, total=False):
    """Shared state for the medical assistant graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    draft_response: str
    verification_report: VerificationReport
    issues_history: Annotated[list[dict[str, Any]], operator.add]
    final_response: str
    reflect_loop_count: int
    max_reflect_loops: int
    use_llm_reflection: bool
