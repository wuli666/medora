import asyncio
import json
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field


from src.config.logger import get_logger, log_stage
from src.config.settings import settings
from src.graph.state import MedAgentState
from src.llm.model_factory import get_chat_model, safe_llm_call
from src.prompts.prompts import (
    INTENT_CLASSIFY_PROMPT,
    MERGE_PROMPT,
    NON_MEDICAL_REPLY_PROMPT,
    PLAN_INIT_PROMPT,
    PLAN_TOOL_PROMPT,
    PLAN_PROMPT,
    REFLECT_PROMPT,
    SEARCH_SUMMARY_PROMPT,
    SUMMARIZE_PROMPT,
)
from src.runtime.progress import mark_stage
from src.tool import tools, tools_by_name
from src.utils.tool_calling import (
    latest_ai_message,
    parse_tool_payload,
    tool_text_or_error,
)

_GREETING_PATTERN = re.compile(
    r"^(你好|您好|hi|hello|hey|在吗|有人吗)[!！。.\s]*$",
    re.IGNORECASE,
)
_MEDICAL_KEYWORDS = {
    "病", "症状", "检查", "报告", "指标", "诊断", "用药", "药", "复诊", "随访",
    "血压", "血糖", "心电", "ct", "mri", "x光", "影像", "化验", "慢病", "高血压",
    "糖尿病", "胸闷", "头痛", "咳嗽", "发热", "疼", "痛", "医生", "住院", "门诊",
    "体检", "检验", "处方", "不适", "炎症",
}

logger = get_logger(__name__)

_SUPERVISOR_MODEL = settings.SUPERVISOR_MODEL or "qwen-plus"
_PLANNER_MODEL = settings.PLANNER_MODEL or "qwen-plus"
_TOOLER_MERGE_MODEL = settings.TOOLER_MERGE_MODEL or "qwen-plus"
_SEARCHER_MODEL = settings.SEARCHER_MODEL or "qwen-plus"
_REFLECTOR_MODEL = settings.REFLECTOR_MODEL or "qwen-plus"
_SUMMARIZER_MODEL = settings.SUMMARIZER_MODEL or "qwen-plus"
_TOOLER_MAX_CONCURRENCY = 4
_MAX_TOOL_CALL_RETRIES = 2
_MAX_REFLECT_ITERATIONS = 2
_NO_TOOL_SEARCH_FALLBACK = "（未执行检索，基于已有信息生成）"

class UserIntent(BaseModel):
    INTENT: Literal["MEDICAL", "NON_MEDICAL"] = Field(
        description="Issue severity level."
    )


class reflect_report(BaseModel):
    completeness_check: str = Field(
        description="Missing key medical history/exam/medication information."
    )
    consistency_check: str = Field(
        description="Whether analysis, search and plan are consistent."
    )
    hallucination_risk: str = Field(
        description="Claims potentially lacking evidence."
    )
    minimal_corrections: list[str] = Field(
        description="1-3 executable minimal corrections."
    )
    quality_conclusion: Literal["PASS", "FAIL"] = Field(
        description="Final quality gate conclusion."
    )


# ── Supervisor ──────────────────────────────────────────────────────
async def supervisor_node(
    state: MedAgentState,
) -> Command[Literal["planner", "summarize"]]:
    """Central router: decide which worker to invoke next."""
    logger.info("[supervisor] enter")
    run_id = state["run_id"]
    intent = state.get("query_intent", "").strip().upper()
    has_images = bool(state.get("has_images"))
    raw_user_text = state.get("raw_text", "")
    messages = list(state.get("messages", []))
    if raw_user_text:
        last = messages[-1] if messages else None
        last_type = getattr(last, "type", "")
        last_content = str(getattr(last, "content", ""))
        if last_type != "human" or last_content != raw_user_text:
            messages.append(HumanMessage(content=raw_user_text))

    if intent == "NON_MEDICAL":
        logger.info("[supervisor] route=summarize (cached non-medical)")
        await mark_stage(run_id, "quick_router", "done")
        return Command(
            update={"query_intent": intent, "messages": messages},
            goto="summarize",
        )

    if not intent:
        await mark_stage(run_id, "quick_router", "running")
        raw_text = state.get("raw_text", "").strip()
        normalized = raw_text.lower()

        if has_images and state.get("images"):
            intent = "MEDICAL"
            logger.info("[supervisor] intent=MEDICAL (has_images=True)")
        elif not raw_text:
            intent = "NON_MEDICAL"
            logger.info("[supervisor] intent=NON_MEDICAL (empty text)")
        else:
            prompt = INTENT_CLASSIFY_PROMPT.format(user_text=raw_text)
            llm = get_chat_model("SUPERVISOR", default_model=_SUPERVISOR_MODEL, temperature=0.1)
            logger.info("[supervisor] intent classify via llm model=%s", _SUPERVISOR_MODEL)
            structured_llm = llm.with_structured_output(UserIntent)
            response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
            log_stage(logger, "supervisor.intent", response)

            if isinstance(response, UserIntent):
                intent = response.INTENT
            else:
                intent = str(response).strip().upper().replace("-", "_")
            
            logger.info("[supervisor] intent=%s (llm classified)", intent)
            if intent not in {"MEDICAL", "NON_MEDICAL"}:
                if _GREETING_PATTERN.match(normalized):
                    intent = "NON_MEDICAL"
                    logger.info("[supervisor] intent=NON_MEDICAL (keyword_fallback)")
                elif any(kw in normalized for kw in _MEDICAL_KEYWORDS):
                    intent = "MEDICAL"
                    logger.info("[supervisor] intent=MEDICAL (keyword_fallback)")
                else:
                    intent = "NON_MEDICAL"
                    logger.info("[supervisor] intent=NON_MEDICAL (keyword_fallback)")

        if intent == "NON_MEDICAL":
            logger.info("[supervisor] route=summarize (entry non-medical)")
            await mark_stage(run_id, "quick_router", "done")
            return Command(
                update={"query_intent": intent, "messages": messages},
                goto="summarize",
            )

        await mark_stage(run_id, "quick_router", "done")
        # Medical query enters planner first (todo list).
        return Command(
            update={
                "query_intent": intent,
                "messages": messages,
                "tool_skipped": False,
                "merged_analysis": state.get("raw_text", "") if not has_images else state.get("merged_analysis", ""),
                "medical_text_analysis": state.get("medical_text_analysis", ""),
                "medical_image_analysis": state.get("medical_image_analysis", ""),
                "planner_decision": "",
                "tools_dispatched": False,
                "planner_tool_attempts": 0,
            },
            goto="planner",
        )

    # Supervisor only handles intent + initialization.
    logger.info("[supervisor] route=planner")
    return Command(goto="planner")


# ── Planner ─────────────────────────────────────────────────────────
async def planner_node(
    state: MedAgentState,
) -> Command[Literal["tooler", "reflector", "summarize"]]:
    """Planner-centric orchestrator with phase-driven routing."""
    logger.info("[planner] enter")
    run_id = state["run_id"]
    await mark_stage(run_id, "planner", "running")

    llm = get_chat_model("PLANNER", default_model=_PLANNER_MODEL, temperature=0.3)

    plan_text = str(state.get("plan", "")).strip()
    tools_dispatched = bool(state.get("tools_dispatched"))
    decision = str(state.get("planner_decision", "")).strip().upper()
    reflection_text = str(state.get("reflection", "")).strip()
    pre_updates: dict[str, Any] = {}

    if decision == "SUMMARY":
        logger.info("[planner] route=summarize (reflector approved)")
        await mark_stage(run_id, "planner", "done")
        return Command(update={"planner_decision": "SUMMARY"}, goto="summarize")

    if decision == "REDO":
        logger.info("[planner] decision=REDO, re-planning before tool dispatch")
        analysis_context = state.get("merged_analysis", "") or state.get("raw_text", "")
        search_context = state.get("search_results", "") or "（未检索到外部证据）"
        reflection_section = (
            f"\n\n## 上一轮质检反馈（请据此修正计划）\n{reflection_text}"
            if reflection_text
            else ""
        )
        prompt = (
            PLAN_PROMPT.format(
                analysis=analysis_context,
                search_results=search_context,
            )
            + reflection_section
        )
        response = await safe_llm_call(llm, prompt, "planner.update")
        log_stage(logger, "planner.update", response)
        plan_text = response
        tools_dispatched = False
        pre_updates.update(
            {
                "plan": response,
                "planner_decision": "",
                "tools_dispatched": False,
                "planner_tool_attempts": 0,
                "tool_skipped": False,
            }
        )

    if not plan_text:
        # INIT_PLAN
        logger.info("[planner] generating initial plan via llm model=%s", _PLANNER_MODEL)
        init_prompt = PLAN_INIT_PROMPT.format(raw_text=state.get("raw_text", ""))
        plan_text = await safe_llm_call(llm, init_prompt, "planner.init")
        log_stage(logger, "planner.init", plan_text)
        tools_dispatched = False
        pre_updates.update(
            {
                "plan": plan_text,
                "tools_dispatched": False,
                "planner_decision": "",
                "planner_tool_attempts": 0,
                "tool_skipped": False,
            }
        )

    if not tools_dispatched:
        # DISPATCH_TOOLS
        llm_with_tools = llm.bind_tools(tools)

        logger.info("[planner] requesting tool calls via llm")
        reflection_section = (
            f"\n\n上一轮质检反馈（请据此调整工具调用）：\n{reflection_text}"
            if reflection_text
            else ""
        )

        planner_human = (
            f"用户输入：{state.get('raw_text', '')}\n\n"
            f"当前执行清单：{plan_text}\n\n"
            f"请根据需要调用工具。{reflection_section}"
        )

        attempts = 0
        for attempt in range(1, _MAX_TOOL_CALL_RETRIES + 2):
            attempts = attempt
            response = await llm_with_tools.ainvoke(
                [
                    SystemMessage(content=PLAN_TOOL_PROMPT),
                    HumanMessage(content=planner_human),
                ]
            )
            ai_response = response if isinstance(response, AIMessage) else None

            if ai_response is not None and ai_response.tool_calls:
                await mark_stage(run_id, "planner", "done")
                return Command(
                    update={
                        **pre_updates,
                        "messages": [ai_response],
                        "tools_dispatched": True,
                        "planner_tool_attempts": attempts,
                    },
                    goto="tooler",
                )

            if attempt <= _MAX_TOOL_CALL_RETRIES:
                logger.info("[planner] no tool calls detected, retrying attempt=%s", attempt)

        logger.warning("[planner] no tool calls after %s attempts, fallback to no-tool planning", attempts)
        fallback_search = state.get("search_results", "") or _NO_TOOL_SEARCH_FALLBACK
        fallback_merged = state.get("merged_analysis", "") or state.get("raw_text", "")
        await mark_stage(run_id, "planner", "done")
        return Command(
            update={
                **pre_updates,
                "tools_dispatched": True,
                "tool_skipped": True,
                "planner_tool_attempts": attempts,
                "search_results": fallback_search,
                "merged_analysis": fallback_merged,
                "planner_decision": "REFLECT",
            },
            goto="reflector",
        )

    await mark_stage(run_id, "planner", "done")
    logger.info("[planner] route=reflector (tools already dispatched)")
    return Command(update={**pre_updates, "planner_decision": "REFLECT"}, goto="reflector")


# ── Tooler (MedGemma text ∥ image → merge) ─────────────────────────
async def tooler_node(state: MedAgentState) -> dict:
    """Execute planner tool calls and map JSON outputs into graph fields."""
    logger.info("[tooler] enter has_images=%s", bool(state.get("has_images")))
    await mark_stage(state["run_id"], "tooler", "running")

    ai_message = latest_ai_message(state)
    if ai_message is None or not ai_message.tool_calls:
        logger.info("[tooler] no pending tool calls, noop")
        await mark_stage(state["run_id"], "tooler", "done")
        return {}

    tool_calls = list(ai_message.tool_calls)
    logger.info("[tooler] executing %s tool calls", len(tool_calls))
    semaphore = asyncio.Semaphore(max(1, _TOOLER_MAX_CONCURRENCY))

    async def _run_tool_call(idx: int, tool_call: dict[str, Any]) -> tuple[int, ToolMessage, dict[str, Any]]:
        tool_name = str(tool_call.get("name", ""))
        tool_call_id = str(tool_call.get("id", tool_name or f"call_{idx}"))
        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        else:
            args = dict(args)

        # Do not trust planner-provided image payloads; always use uploaded state images.
        if tool_name == "analyze_medical_image":
            images = state.get("images", [])
            if isinstance(images, list) and images:
                image_index_raw = args.pop("image_index", 0)
                try:
                    image_index = int(image_index_raw)
                except (TypeError, ValueError):
                    image_index = 0
                image_index = max(0, min(image_index, len(images) - 1))
                args["image_base64"] = str(images[image_index])
                if not isinstance(args.get("clinical_context"), str):
                    args["clinical_context"] = state.get("raw_text", "")

        tool_impl = tools_by_name.get(tool_name)
        if tool_impl is None:
            payload = {
                "tool": tool_name,
                "ok": False,
                "data": {},
                "error": {"code": "UNKNOWN_TOOL", "message": f"Unknown tool: {tool_name}"},
                "meta": {"version": "1.0"},
            }
            content = json.dumps(payload, ensure_ascii=False)
            return idx, ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name), payload

        async with semaphore:
            observation = await tool_impl.ainvoke(args)
        payload = parse_tool_payload(tool_name, observation)
        content = json.dumps(payload, ensure_ascii=False)
        return idx, ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name), payload

    executed = await asyncio.gather(*[_run_tool_call(i, tc) for i, tc in enumerate(tool_calls)])
    ordered = sorted(executed, key=lambda item: item[0])
    tool_messages = [message for _, message, _ in ordered]
    payloads = [payload for _, _, payload in ordered]

    text_payloads = [p for p in payloads if p.get("tool") == "analyze_medical_text"]
    image_payloads = [p for p in payloads if p.get("tool") == "analyze_medical_image"]
    web_payloads = [p for p in payloads if p.get("tool") == "web_search"]
    rag_payloads = [p for p in payloads if p.get("tool") == "rag_search"]

    text_analysis = tool_text_or_error(text_payloads[0]) if text_payloads else ""
    image_analyses = [tool_text_or_error(p) for p in image_payloads]
    image_analyses = [t for t in image_analyses if t]

    web_summary = ""
    if web_payloads:
        web_summary = str(web_payloads[0].get("data", {}).get("summary", ""))
        if not web_summary:
            web_summary = tool_text_or_error(web_payloads[0], key="summary")

    rag_summary = ""
    if rag_payloads:
        rag_summary = str(rag_payloads[0].get("data", {}).get("summary", ""))
        if not rag_summary:
            rag_summary = tool_text_or_error(rag_payloads[0], key="summary")

    combined_search = f"## 网络搜索结果\n{web_summary}\n\n## 知识库搜索结果\n{rag_summary}"
    llm = get_chat_model("SEARCHER", default_model=_SEARCHER_MODEL, temperature=0.2)

    prompt = SEARCH_SUMMARY_PROMPT.format(
        analysis=text_analysis or state.get("raw_text", ""),
        search_results=combined_search,
    )

    logger.info("[tooler] summarizing search results via llm model=%s", _SEARCHER_MODEL)
    search_results = await safe_llm_call(llm, prompt, "tooler.search")

    if image_analyses:
        image_analysis = "\n---\n".join(image_analyses)
        llm = get_chat_model("TOOLER_MERGE", default_model=_TOOLER_MERGE_MODEL, temperature=0.2)
        prompt = MERGE_PROMPT.format(
            text_analysis=text_analysis,
            image_analysis=image_analysis,
        )
        logger.info("[tooler] merging text and image analysis with model %s", _TOOLER_MERGE_MODEL)
        merged = await safe_llm_call(llm, prompt, "tooler.merge")
    else:
        image_analysis = ""
        merged = text_analysis

    log_stage(logger, "tooler.text", text_analysis)
    if image_analysis:
        log_stage(logger, "tooler.image", image_analysis)
    log_stage(logger, "tooler.merge", merged)
    log_stage(logger, "searcher", search_results)
    logger.info("[tooler] exit")
    await mark_stage(state["run_id"], "tooler", "done")
    await mark_stage(state["run_id"], "searcher", "done")

    return {
        "messages": tool_messages,
        "medical_text_analysis": text_analysis,
        "medical_image_analysis": image_analysis,
        "merged_analysis": merged,
        "search_results": search_results,
    }


# ── Reflector ───────────────────────────────────────────────────────
async def reflector_node(state: MedAgentState) -> dict:
    """Review analysis + search + plan for consistency and accuracy."""
    logger.info("[reflector] enter")
    await mark_stage(state["run_id"], "reflector", "running")
    llm = get_chat_model("REFLECTOR", default_model=_REFLECTOR_MODEL, temperature=0.2)
    
    prompt = REFLECT_PROMPT.format(
        analysis=state["merged_analysis"],
        search_results=state["search_results"],
        plan=state["plan"],
    )
    
    logger.info("[reflector] evaluating plan consistency via llm model=%s", _REFLECTOR_MODEL)

    structured_llm = llm.with_structured_output(reflect_report)
    raw_response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
    report = (
        raw_response
        if isinstance(raw_response, reflect_report)
        else reflect_report.model_validate(raw_response)
    )

    next_iteration = int(state.get("iteration", 0)) + 1
    is_fail = report.quality_conclusion == "FAIL"
    hit_limit = next_iteration >= _MAX_REFLECT_ITERATIONS
    planner_decision = "REDO" if (is_fail and not hit_limit) else "SUMMARY"

    fixes = "\n".join(f"- {item}" for item in report.minimal_corrections[:3])
    reflection_text = (
        f"1. 完整性检查：{report.completeness_check}\n"
        f"2. 一致性检查：{report.consistency_check}\n"
        f"3. 幻觉风险：{report.hallucination_risk}\n"
        f"4. 最小修正建议：\n{fixes}\n"
        f"5. 质检结论：{report.quality_conclusion}"
    )

    log_stage(logger, "reflector", reflection_text)
    logger.info("[reflector] exit")
    await mark_stage(state["run_id"], "reflector", "done")
    update: dict[str, Any] = {
        "reflection": reflection_text,
        "iteration": next_iteration,
        "planner_decision": planner_decision,
    }
    return update


# ── Summarize ───────────────────────────────────────────────────────
async def summarize_node(state: MedAgentState) -> dict:
    """Produce a patient-friendly summary incorporating reflection."""
    logger.info("[summarizer] enter")
    await mark_stage(state["run_id"], "summarize", "running")
    # If supervisor already provided a direct summary, return it as-is.
    if state.get("summary"):
        log_stage(logger, "summarizer", state["summary"])
        logger.info("[summarizer] exit (pre-filled summary)")
        await mark_stage(state["run_id"], "summarize", "done")
        return {"summary": state["summary"]}

    llm = get_chat_model("SUMMARIZER", default_model=_SUMMARIZER_MODEL, temperature=0.2)

    if (state.get("query_intent") or "").upper() == "NON_MEDICAL":
        prompt = NON_MEDICAL_REPLY_PROMPT.format(user_text=state.get("raw_text", ""))
        logger.info("[summarizer] generating non-medical reply via llm model=%s", _SUMMARIZER_MODEL)
        response = await safe_llm_call(llm, prompt, "summarizer.non_medical")
        messages = list(state.get("messages", []))
        messages.append(AIMessage(content=response))
        log_stage(logger, "summarizer", response)
        logger.info("[summarizer] exit (non-medical llm reply)")
        await mark_stage(state["run_id"], "summarize", "done")
        return {"summary": response, "messages": messages}
    prompt = SUMMARIZE_PROMPT.format(
        analysis=state["merged_analysis"],
        search_results=state["search_results"],
        plan=state["plan"],
        reflection=state["reflection"],
    )
    
    logger.info("[summarizer] generating final summary via llm model=%s", _SUMMARIZER_MODEL)
    response = await safe_llm_call(llm, prompt, "summarizer")
        
    log_stage(logger, "summarizer", response)
    logger.info("[summarizer] exit")
    await mark_stage(state["run_id"], "summarize", "done")
    return {"summary": response}
