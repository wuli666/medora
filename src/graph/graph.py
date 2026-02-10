"""LangGraph wiring for planner -> reflect -> summary flow."""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import Any
from typing import Literal
from typing import Mapping
from typing import cast

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from src.agents import reflect_verify_node
from src.config import settings
from src.graph.state import GraphState
from src.llm import get_llm
from src.tool import tools, tools_by_name


DEFAULT_MAX_REFLECT_LOOPS = settings.MAX_REFLECT_LOOPS
DEFAULT_DRAFT_RESPONSE = (
    "I can help with general information, but this is not a diagnosis. "
    "Please share more detail and consult a healthcare professional for "
    "urgent or worsening symptoms."
)
PLANNER_SYSTEM_PROMPT = (
    "You are a medical assistant planner. "
    "Decide whether tools are needed and call them when helpful. "
    "When no tool is needed, provide a concise medically cautious draft response."
)


def _message_text(content: Any) -> str:
    """Convert message content into a text string."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=True)
    except TypeError:
        return str(content)


def _latest_human_text(state: GraphState) -> str:
    """Return latest human message content if present."""
    for message in reversed(state.get("messages", [])):
        if message.type == "human":
            return _message_text(message.content)
    return ""


def _latest_ai_message(state: GraphState) -> AIMessage | None:
    """Return the latest AI message from conversation history."""
    for message in reversed(state.get("messages", [])):
        if message.type == "ai" and isinstance(message, AIMessage):
            return message
    return None


def _has_pending_tool_calls(state: GraphState) -> bool:
    """Check whether planner emitted tool calls in its latest AI message."""
    ai_message = _latest_ai_message(state)
    return bool(ai_message and ai_message.tool_calls)


def _needs_more_planning(report: Mapping[str, Any]) -> bool:
    """Determine if planner should continue reflection loop."""
    return bool(report.get("needs_tool") or report.get("needs_clarification"))


def _reached_reflection_limit(state: GraphState) -> bool:
    """Check whether reflection loop count reached configured maximum."""
    reflect_loop_count = int(state.get("reflect_loop_count", 0))
    max_reflect_loops = int(state.get("max_reflect_loops", DEFAULT_MAX_REFLECT_LOOPS))
    return reflect_loop_count >= max_reflect_loops


@lru_cache(maxsize=1)
def _planner_llm_with_tools():
    """Create a tool-bound planner LLM once per process."""
    return get_llm(
        model=settings.MEDGEMMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.MEDGEMMA_TEMPERATURE,
    ).llm.bind_tools(tools)


def supervisor_node(state: GraphState) -> GraphState:
    """Initialize defaults needed by the graph."""
    return {
        "reflect_loop_count": int(state.get("reflect_loop_count", 0)),
        "max_reflect_loops": int(state.get("max_reflect_loops", DEFAULT_MAX_REFLECT_LOOPS)),
        "use_llm_reflection": bool(
            state.get("use_llm_reflection", settings.USE_LLM_REFLECTION)
        ),
    }


async def planner_node(state: GraphState) -> GraphState:
    """Planner node that decides tool use and drafts responses.

    This node can be extended to request tools and draft responses.
    """
    messages = list(state.get("messages", []))
    if not messages and not _latest_human_text(state):
        return {}

    response = await _planner_llm_with_tools().ainvoke(
        [SystemMessage(content=PLANNER_SYSTEM_PROMPT), *messages]
    )

    update: GraphState = {"messages": [response]}

    tool_calls = response.tool_calls if isinstance(response, AIMessage) else []
    if tool_calls:
        return update

    draft_response = _message_text(response.content).strip()
    if not draft_response:
        latest_user_message = _latest_human_text(state)
        draft_response = DEFAULT_DRAFT_RESPONSE if latest_user_message else ""
    if draft_response:
        update["draft_response"] = draft_response
    return update


async def tool_node(state: GraphState) -> GraphState:
    """Execute tool calls emitted by the planner AI message."""
    ai_message = _latest_ai_message(state)
    if ai_message is None or not ai_message.tool_calls:
        return {}

    semaphore = asyncio.Semaphore(max(1, int(settings.TOOL_NODE_MAX_CONCURRENCY)))

    async def _run_tool_call(idx: int, tool_call: dict[str, Any]) -> tuple[int, ToolMessage]:
        tool_name = str(tool_call.get("name", ""))
        tool_call_id = str(tool_call.get("id", tool_name or "tool_call"))
        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            args = {}

        tool_impl = tools_by_name.get(tool_name)
        if tool_impl is None:
            content = json.dumps(
                {
                    "tool": tool_name,
                    "ok": False,
                    "error": f"Unknown tool: {tool_name}",
                },
                ensure_ascii=True,
            )
            return idx, ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)

        try:
            async with semaphore:
                observation = await tool_impl.ainvoke(args)
            content = observation if isinstance(observation, str) else _message_text(observation)
        except Exception as exc:  # pragma: no cover - defensive integration guard
            content = json.dumps(
                {
                    "tool": tool_name,
                    "ok": False,
                    "error": f"Tool execution failed: {type(exc).__name__}",
                },
                ensure_ascii=True,
            )

        return idx, ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)

    results = await asyncio.gather(
        *[_run_tool_call(idx, tool_call) for idx, tool_call in enumerate(ai_message.tool_calls)]
    )
    ordered = [message for _, message in sorted(results, key=lambda item: item[0])]

    return {"messages": cast(list[AnyMessage], ordered)}


def summary_node(state: GraphState):
    """Build final summary after reflection workflow completes."""
    report = state.get("verification_report", {})
    final_response = state.get("draft_response", "")
    if report.get("issues") and not report.get("is_safe_to_finalize", False):
        final_response = (
            f"{final_response}\n\n"
            "If symptoms are severe, new, or worsening, seek care from a licensed "
            "clinician immediately."
        ).strip()
    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)] if final_response else [],
    }


def route_from_planner(state: GraphState) -> Literal["tool_node", "reflect_verify", "summary"]:
    """Route from planner based on current draft and verification status."""
    if _has_pending_tool_calls(state):
        return "tool_node"

    draft_response = state.get("draft_response", "")
    if not draft_response:
        return "summary"

    report = state.get("verification_report", {})
    if not report:
        return "reflect_verify"

    if _reached_reflection_limit(state):
        return "summary"

    if _needs_more_planning(report):
        return "reflect_verify"
    return "summary"


def build_graph() -> Any:

    builder = StateGraph(GraphState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("planner", planner_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("reflect_verify", reflect_verify_node)
    builder.add_node("summary", summary_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "planner")
    builder.add_conditional_edges(
        "planner",
        route_from_planner,
        ["tool_node", "reflect_verify", "summary"],
    )
    builder.add_edge("tool_node", "planner")
    builder.add_edge("reflect_verify", "planner")
    builder.add_edge("summary", END)
    return builder.compile(name="medgemma-assistant")


graph = build_graph()
