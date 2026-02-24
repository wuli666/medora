import asyncio
import json
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.config.logger import get_logger, log_stage
from src.config.settings import settings
from src.graph.state import (
    CarePlan,
    MedAgentState,
    MergedMedicalAnalysis,
    PatientSummary,
    ReflectReport,
    SearchSummary,
)
from src.llm.model_factory import get_chat_model
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
from src.runtime.progress import mark_stage, mark_substep
from src.tool import tools, tools_by_name
from src.utils.tool_calling import latest_ai_message, parse_tool_payload, tool_text_or_error

_GREETING_PATTERN = re.compile(
    r"^(你好|您好|hi|hello|hey|在吗|有人吗)[!！。.\s]*$",
    re.IGNORECASE,
)
_MEDICAL_KEYWORDS = {
    "病",
    "症状",
    "检查",
    "报告",
    "指标",
    "诊断",
    "用药",
    "药",
    "复诊",
    "随访",
    "血压",
    "血糖",
    "心电",
    "ct",
    "mri",
    "x光",
    "影像",
    "化验",
    "慢病",
    "高血压",
    "糖尿病",
    "胸闷",
    "头痛",
    "咳嗽",
    "发热",
    "疼",
    "痛",
    "医生",
    "住院",
    "门诊",
    "体检",
    "检验",
    "处方",
    "不适",
    "炎症",
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
_NO_TOOL_SEARCH_FALLBACK = "(Search not executed; generated from available evidence)"


class UserIntent(BaseModel):
    INTENT: Literal["MEDICAL", "NON_MEDICAL"] = Field(
        description="User query intent label."
    )


def _join_lines(items: list[str]) -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return "\n".join(f"- {item}" for item in cleaned)


def _to_ai_message_json(model: BaseModel) -> str:
    """Serialize structured output compatibly across pydantic versions."""
    if hasattr(model, "model_dump"):
        payload = model.model_dump(mode="json")
    elif hasattr(model, "dict"):
        payload = model.dict()  # type: ignore[attr-defined]
    else:
        return str(model)
    return json.dumps(payload, ensure_ascii=False)


def _render_care_plan_text(plan: CarePlan) -> str:
    return (
        "1. Condition Analysis\n"
        f"{_join_lines(plan.condition_analysis) or '- None'}\n\n"
        "2. Monitoring Metrics\n"
        f"{_join_lines(plan.monitoring_metrics) or '- None'}\n\n"
        "3. Lifestyle Advice\n"
        f"{_join_lines(plan.lifestyle_advice) or '- None'}"
    )


def _render_reflection_text(report: ReflectReport) -> str:
    fixes = _join_lines(report.minimal_corrections[:3]) or "- None"
    return (
        f"1. Completeness Check: {report.completeness_check}\n"
        f"2. Consistency Check: {report.consistency_check}\n"
        f"3. Hallucination Risk: {report.hallucination_risk}\n"
        f"4. Minimal Corrections:\n{fixes}\n"
        f"5. Quality Conclusion: {report.quality_conclusion}"
    )


def _render_summary_text(summary: PatientSummary) -> str:
    title = summary.report_title.strip() or "Health Management & Follow-up Report"
    brief_summary = summary.brief_summary.strip() or "A health management recommendation has been generated from the available information."
    sections = [
        f"# {title}",
        f"## Summary\n{brief_summary}",
        f"## Key Findings\n{_join_lines(summary.key_findings) or '- None'}",
    ]
    if summary.medication_reminders:
        sections.append(f"## Medication Reminders\n{_join_lines(summary.medication_reminders)}")
    if summary.follow_up_tips:
        sections.append(f"## Follow-up Tips\n{_join_lines(summary.follow_up_tips)}")
    return "\n\n".join(sections)


def _render_merged_analysis_text(merged: MergedMedicalAnalysis) -> str:
    return (
        "1. Primary Diagnoses/Findings\n"
        f"{_join_lines(merged.primary_findings) or '- None'}\n\n"
        "2. Key Indicator Abnormalities\n"
        f"{_join_lines(merged.key_abnormalities) or '- None'}\n\n"
        "3. Risk Assessment\n"
        f"{_join_lines(merged.risk_assessment) or '- None'}\n\n"
        "4. Areas Requiring Further Attention\n"
        f"{_join_lines(merged.attention_points) or '- None'}"
    )


def _render_search_summary_text(summary: SearchSummary) -> str:
    return (
        "## Key Medical Knowledge\n"
        f"{_join_lines(summary.key_points) or '- None'}\n\n"
        "## Source Notes\n"
        f"{_join_lines(summary.source_notes) or '- None'}\n\n"
        f"Evidence Level: {summary.evidence_level or 'Not specified'}"
    )


async def _invoke_structured(
    llm: Any,
    schema: type[BaseModel],
    messages: list,
) -> tuple[BaseModel, AIMessage]:
    if llm is None:
        raise RuntimeError("LLM unavailable for structured invocation")

    parsed_model: BaseModel | None = None
    try:
        structured_llm = llm.with_structured_output(
            schema,
            include_raw=True,
            method="json_schema",
            strict=True,
        )
        response = await structured_llm.ainvoke(messages)
        if isinstance(response, dict):
            parsed = response.get("parsed")
            if isinstance(parsed, schema):
                parsed_model = parsed
            else:
                parsed_model = schema.model_validate(parsed)
    except TypeError:
        parsed_model = None

    if parsed_model is None:
        try:
            structured_llm = llm.with_structured_output(schema, include_raw=True)
            response = await structured_llm.ainvoke(messages)
            if isinstance(response, dict):
                parsed = response.get("parsed")
                if isinstance(parsed, schema):
                    parsed_model = parsed
                else:
                    parsed_model = schema.model_validate(parsed)
        except TypeError:
            parsed_model = None

    if parsed_model is None:
        structured_llm = llm.with_structured_output(schema)
        response = await structured_llm.ainvoke(messages)
        if isinstance(response, schema):
            parsed_model = response
        else:
            parsed_model = schema.model_validate(response)
    return parsed_model, AIMessage(content=_to_ai_message_json(parsed_model))


def _build_plan_payload(
    raw_text: str,
    analysis: str,
    search_results: str,
    reflection: str,
) -> str:
    return (
        f"User Input:\n{raw_text or '(No data)'}\n\n"
        f"Medical Analysis:\n{analysis or '(No data)'}\n\n"
        f"Knowledge Supplement:\n{search_results or '(No data)'}\n\n"
        f"Previous Review Feedback:\n{reflection or '(No data)'}"
    )


async def _mark_user_substep(
    run_id: str,
    stage_key: str,
    substep_id: str,
    label: str,
    status: str,
    detail: str = "",
) -> None:
    await mark_substep(
        run_id=run_id,
        stage_key=stage_key,
        substep_id=substep_id,
        status=status,
        label=label,
        detail=detail,
    )


# ── Supervisor ──────────────────────────────────────────────────────
async def supervisor_node(
    state: MedAgentState,
) -> Command[Literal["planner", "summarize"]]:
    """Central router: decide which worker to invoke next."""
    logger.info("[supervisor] enter")
    run_id = state["run_id"]
    intent = state.get("query_intent", "").strip().upper()
    route_note = ""
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
        await mark_stage(run_id, "quick_router", "running", content="Understanding your request.")
        await _mark_user_substep(run_id, "quick_router", "understand_query", "Understand your request", "done", "Request understanding completed.")
        await _mark_user_substep(run_id, "quick_router", "decide_route", "Decide whether medical analysis is needed", "done", "This request will receive a direct general reply.")
        await _mark_user_substep(run_id, "quick_router", "route_ready", "Route confirmed", "done", "Processing path has been confirmed.")
        await mark_stage(run_id, "quick_router", "done", content="Classified as general consultation; preparing a direct reply.")
        return Command(
            update={"query_intent": intent, "messages": messages},
            goto="summarize",
        )

    if not intent:
        await mark_stage(run_id, "quick_router", "running", content="Understanding your request and selecting the processing route.")
        await _mark_user_substep(run_id, "quick_router", "understand_query", "Understand your request", "running", "Analyzing the information you provided.")
        raw_text = state.get("raw_text", "").strip()
        normalized = raw_text.lower()

        if has_images and state.get("images"):
            intent = "MEDICAL"
            route_note = "Medical information detected; entering medical analysis workflow."
            logger.info("[supervisor] intent=MEDICAL (has_images=True)")
        elif not raw_text:
            intent = "NON_MEDICAL"
            route_note = "No medical content detected; handling as general consultation."
            logger.info("[supervisor] intent=NON_MEDICAL (empty text)")
        else:
            llm = get_chat_model("SUPERVISOR", default_model=_SUPERVISOR_MODEL, temperature=0.1)
            logger.info("[supervisor] intent classify via llm model=%s", _SUPERVISOR_MODEL)
            response, raw_ai = await _invoke_structured(
                llm,
                UserIntent,
                [
                    SystemMessage(content=INTENT_CLASSIFY_PROMPT),
                    HumanMessage(content=raw_text),
                ],
            )
            log_stage(logger, "supervisor.intent", response.model_dump())
            messages.append(raw_ai)

            intent = response.INTENT
            logger.info("[supervisor] intent=%s (llm classified)", intent)
            route_note = "Request type classification completed."
            if intent not in {"MEDICAL", "NON_MEDICAL"}:
                if _GREETING_PATTERN.match(normalized):
                    intent = "NON_MEDICAL"
                    route_note = "Classified as general consultation."
                    logger.info("[supervisor] intent=NON_MEDICAL (keyword_fallback)")
                elif any(kw in normalized for kw in _MEDICAL_KEYWORDS):
                    intent = "MEDICAL"
                    route_note = "Classified as a medical-related query."
                    logger.info("[supervisor] intent=MEDICAL (keyword_fallback)")
                else:
                    intent = "NON_MEDICAL"
                    route_note = "Classified as general consultation."
                    logger.info("[supervisor] intent=NON_MEDICAL (keyword_fallback)")

        await _mark_user_substep(run_id, "quick_router", "understand_query", "Understand your request", "done", "Request understanding completed.")
        await _mark_user_substep(run_id, "quick_router", "decide_route", "Decide whether medical analysis is needed", "running", "Determining the most appropriate processing route.")
        if intent == "NON_MEDICAL":
            logger.info("[supervisor] route=summarize (entry non-medical)")
            await _mark_user_substep(run_id, "quick_router", "decide_route", "Decide whether medical analysis is needed", "done", "Medical analysis is not required for this request.")
            await _mark_user_substep(run_id, "quick_router", "route_ready", "Route confirmed", "done", "A direct general reply will be generated.")
            await mark_stage(
                run_id,
                "quick_router",
                "done",
                content=route_note or "Confirmed as general consultation; preparing a direct reply.",
            )
            return Command(
                update={"query_intent": intent, "messages": messages},
                goto="summarize",
            )

        await _mark_user_substep(run_id, "quick_router", "decide_route", "Decide whether medical analysis is needed", "done", "Entering the medical analysis workflow.")
        await _mark_user_substep(run_id, "quick_router", "route_ready", "Route confirmed", "done", "Proceeding with medical analysis and recommendations.")
        await mark_stage(
            run_id,
            "quick_router",
            "done",
            content=route_note or "Medical analysis workflow confirmed.",
        )
        return Command(
            update={
                "query_intent": intent,
                "messages": messages,
                "tool_skipped": False,
                "merged_analysis": state.get("raw_text", "") if not has_images else state.get("merged_analysis", ""),
                "medical_text_analysis": state.get("medical_text_analysis", ""),
                "medical_image_analysis": state.get("medical_image_analysis", ""),
                "medical_text_analysis_struct": state.get("medical_text_analysis_struct"),
                "medical_image_analysis_struct": state.get("medical_image_analysis_struct", []),
                "merged_analysis_struct": state.get("merged_analysis_struct"),
                "search_results_struct": state.get("search_results_struct"),
                "plan_struct": state.get("plan_struct"),
                "reflection_struct": state.get("reflection_struct"),
                "summary_struct": state.get("summary_struct"),
                "planner_decision": "",
                "tools_dispatched": False,
                "planner_tool_attempts": 0,
            },
            goto="planner",
        )

    logger.info("[supervisor] route=planner")
    return Command(goto="planner")


# ── Planner ─────────────────────────────────────────────────────────
async def planner_node(
    state: MedAgentState,
) -> Command[Literal["tooler", "reflector", "summarize"]]:
    """Planner-centric orchestrator with phase-driven routing."""
    logger.info("[planner] enter")
    run_id = state["run_id"]
    await mark_stage(run_id, "planner", "running", content="Organizing health priorities and drafting management recommendations.")
    await _mark_user_substep(run_id, "planner", "focus_extract", "Extract current health priorities", "running", "Extracting key information from current materials.")

    llm = get_chat_model("PLANNER", default_model=_PLANNER_MODEL, temperature=0.3)

    plan_text = str(state.get("plan", "")).strip()
    plan_struct = state.get("plan_struct")
    tools_dispatched = bool(state.get("tools_dispatched"))
    decision = str(state.get("planner_decision", "")).strip().upper()
    reflection_text = str(state.get("reflection", "")).strip()
    pre_updates: dict[str, Any] = {}
    added_messages: list[AIMessage] = []

    if decision == "SUMMARY":
        logger.info("[planner] route=summarize (reflector approved)")
        await _mark_user_substep(run_id, "planner", "focus_extract", "Extract current health priorities", "done", "Current-round priority extraction completed.")
        await _mark_user_substep(run_id, "planner", "plan_finalize", "Finalize recommendations", "done", "Management recommendations confirmed; ready for summary generation.")
        await mark_stage(run_id, "planner", "done", content=plan_text)
        return Command(update={"planner_decision": "SUMMARY"}, goto="summarize")

    if decision == "REDO":
        logger.info("[planner] decision=REDO, re-planning before tool dispatch")
        await _mark_user_substep(run_id, "planner", "focus_extract", "Extract current health priorities", "done", "Priorities updated using previous-round results.")
        await _mark_user_substep(run_id, "planner", "plan_draft", "Draft initial care recommendations", "running", "Updating recommendations using available information.")
        analysis_context = state.get("merged_analysis", "") or state.get("raw_text", "")
        search_context = state.get("search_results", "") or "(No external evidence retrieved)"

        plan_model, raw_ai = await _invoke_structured(
            llm,
            CarePlan,
            [
                SystemMessage(content=PLAN_PROMPT),
                HumanMessage(
                    content=_build_plan_payload(
                        raw_text=state.get("raw_text", ""),
                        analysis=analysis_context,
                        search_results=search_context,
                        reflection=reflection_text,
                    )
                ),
            ],
        )
        added_messages.append(raw_ai)
        plan_text = _render_care_plan_text(plan_model)
        plan_struct = plan_model.model_dump()
        tools_dispatched = False
        await _mark_user_substep(run_id, "planner", "plan_draft", "Draft initial care recommendations", "done", "Updated recommendation draft generated.")
        pre_updates.update(
            {
                "plan": plan_text,
                "plan_struct": plan_struct,
                "planner_decision": "",
                "tools_dispatched": False,
                "planner_tool_attempts": 0,
                "tool_skipped": False,
            }
        )

    if not plan_text:
        logger.info("[planner] generating initial plan via llm model=%s", _PLANNER_MODEL)
        await _mark_user_substep(run_id, "planner", "focus_extract", "Extract current health priorities", "done", "Current health priorities extracted.")
        await _mark_user_substep(run_id, "planner", "plan_draft", "Draft initial care recommendations", "running", "Generating the first draft of management recommendations.")
        plan_model, raw_ai = await _invoke_structured(
            llm,
            CarePlan,
            [
                SystemMessage(content=PLAN_INIT_PROMPT),
                HumanMessage(content=state.get("raw_text", "")),
            ],
        )
        added_messages.append(raw_ai)
        plan_text = _render_care_plan_text(plan_model)
        plan_struct = plan_model.model_dump()
        tools_dispatched = False
        await _mark_user_substep(run_id, "planner", "plan_draft", "Draft initial care recommendations", "done", "Initial management recommendations generated.")
        pre_updates.update(
            {
                "plan": plan_text,
                "plan_struct": plan_struct,
                "tools_dispatched": False,
                "planner_decision": "",
                "planner_tool_attempts": 0,
                "tool_skipped": False,
            }
        )

    if not tools_dispatched:
        llm_with_tools = llm.bind_tools(tools)
        await _mark_user_substep(run_id, "planner", "plan_finalize", "Finalize recommendations", "running", "Evaluating whether additional evidence is needed before finalizing recommendations.")

        logger.info("[planner] requesting tool calls via llm")
        planner_human = (
            f"User Input: {state.get('raw_text', '')}\n\n"
            f"Current Plan: {plan_text}\n\n"
            f"Previous Review Feedback: {reflection_text or 'None'}\n\n"
            "Call tools as needed."
        )

        attempts = 0
        last_ai: AIMessage | None = None
        for attempt in range(1, _MAX_TOOL_CALL_RETRIES + 2):
            attempts = attempt
            response = await llm_with_tools.ainvoke(
                [
                    SystemMessage(content=PLAN_TOOL_PROMPT),
                    HumanMessage(content=planner_human),
                ]
            )
            ai_response = response if isinstance(response, AIMessage) else AIMessage(content=str(response))
            last_ai = ai_response

            if ai_response.tool_calls:
                await _mark_user_substep(run_id, "planner", "plan_finalize", "Finalize recommendations", "done", "Prepared to gather additional evidence and refine recommendations.")
                await mark_stage(run_id, "planner", "done", content=plan_text)
                return Command(
                    update={
                        **pre_updates,
                        "messages": [*added_messages, ai_response],
                        "tools_dispatched": True,
                        "planner_tool_attempts": attempts,
                    },
                    goto="tooler",
                )

            if attempt <= _MAX_TOOL_CALL_RETRIES:
                logger.info("[planner] no tool calls detected, retrying attempt=%s", attempt)

        logger.warning(
            "[planner] no tool calls after %s attempts, fallback to no-tool planning", attempts
        )
        fallback_search = state.get("search_results", "") or _NO_TOOL_SEARCH_FALLBACK
        fallback_merged = state.get("merged_analysis", "") or state.get("raw_text", "")
        await _mark_user_substep(run_id, "planner", "plan_finalize", "Finalize recommendations", "done", "Recommendations finalized based on available information.")
        await mark_stage(run_id, "planner", "done", content=plan_text)
        return Command(
            update={
                **pre_updates,
                "messages": [*added_messages, *([last_ai] if last_ai else [])],
                "tools_dispatched": True,
                "tool_skipped": True,
                "planner_tool_attempts": attempts,
                "search_results": fallback_search,
                "merged_analysis": fallback_merged,
                "planner_decision": "REFLECT",
            },
            goto="reflector",
        )

    await _mark_user_substep(run_id, "planner", "focus_extract", "Extract current health priorities", "done", "Current-round priority extraction completed.")
    await _mark_user_substep(run_id, "planner", "plan_finalize", "Finalize recommendations", "done", "Recommendations finalized; moving to review.")
    await mark_stage(run_id, "planner", "done", content=plan_text)
    logger.info("[planner] route=reflector (tools already dispatched)")
    return Command(
        update={
            **pre_updates,
            "messages": added_messages,
            "plan": plan_text,
            "plan_struct": plan_struct,
            "planner_decision": "REFLECT",
        },
        goto="reflector",
    )


# ── Tooler (MedGemma text ∥ image → merge) ─────────────────────────
async def tooler_node(state: MedAgentState) -> dict:
    """Execute planner tool calls and map JSON outputs into graph fields."""
    logger.info("[tooler] enter has_images=%s", bool(state.get("has_images")))
    run_id = state["run_id"]
    await mark_stage(run_id, "tooler", "running", content="Reading your materials and extracting key points.")
    await _mark_user_substep(run_id, "tooler", "read_materials", "Read provided materials", "running", "Reading medical text and imaging materials.")

    ai_message = latest_ai_message(state)
    if ai_message is None or not ai_message.tool_calls:
        logger.info("[tooler] no pending tool calls, noop")
        await _mark_user_substep(run_id, "tooler", "read_materials", "Read provided materials", "done", "Material reading completed.")
        await _mark_user_substep(run_id, "tooler", "extract_key_info", "Extract key medical information", "skipped", "No analyzable materials detected in this round.")
        await _mark_user_substep(run_id, "tooler", "build_conclusion", "Build analysis conclusion", "skipped", "Skipped conclusion generation in this round.")
        await mark_stage(run_id, "tooler", "done", content="No additional materials to analyze in this round.")
        return {}

    tool_calls = list(ai_message.tool_calls)
    logger.info("[tooler] executing %s tool calls", len(tool_calls))
    semaphore = asyncio.Semaphore(max(1, _TOOLER_MAX_CONCURRENCY))
    await _mark_user_substep(run_id, "tooler", "read_materials", "Read provided materials", "done", "Material reading complete; starting key information extraction.")
    await _mark_user_substep(run_id, "tooler", "extract_key_info", "Extract key medical information", "running", "Extracting key medical information.")

    async def _run_tool_call(idx: int, tool_call: dict[str, Any]) -> tuple[int, ToolMessage, dict[str, Any]]:
        tool_name = str(tool_call.get("name", ""))
        tool_call_id = str(tool_call.get("id", tool_name or f"call_{idx}"))
        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        else:
            args = dict(args)

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
            await _mark_user_substep(
                run_id,
                "tooler",
                f"extract_{idx}",
                "Extract key medical information",
                "error",
                "Some materials could not be recognized and were skipped; processing continues for the rest.",
            )
            return idx, ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name), payload

        async with semaphore:
            observation = await tool_impl.ainvoke(args)
        payload = parse_tool_payload(tool_name, observation)
        content = json.dumps(payload, ensure_ascii=False)
        ok = bool(payload.get("ok", False))
        await _mark_user_substep(
            run_id,
            "tooler",
            f"extract_{idx}",
            "Extract key medical information",
            "done" if ok else "error",
            "Information extraction completed." if ok else "Some extraction steps failed; workflow continued.",
        )
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
    text_struct = None
    if text_payloads:
        data = text_payloads[0].get("data", {})
        if isinstance(data, dict):
            value = data.get("analysis_struct")
            if isinstance(value, dict):
                text_struct = value
            if not text_analysis:
                text_analysis = str(data.get("analysis_text", ""))

    image_analyses: list[str] = []
    image_structs: list[dict] = []
    for payload in image_payloads:
        data = payload.get("data", {})
        text = tool_text_or_error(payload)
        if isinstance(data, dict):
            struct = data.get("analysis_struct")
            if isinstance(struct, dict):
                image_structs.append(struct)
            if not text:
                text = str(data.get("analysis_text", ""))
        if text:
            image_analyses.append(text)

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

    combined_search = f"## Web Search Results\n{web_summary}\n\n## Knowledge Base Results\n{rag_summary}"

    await _mark_user_substep(run_id, "tooler", "extract_key_info", "Extract key medical information", "done", "Material key-point extraction completed.")
    await _mark_user_substep(run_id, "searcher", "search_collect", "Retrieve related medical evidence", "running", "Retrieving medical evidence relevant to your question.")
    await mark_stage(run_id, "searcher", "running", content="Collecting related medical evidence and organizing notes.")

    searcher_llm = get_chat_model("SEARCHER", default_model=_SEARCHER_MODEL, temperature=0.2)
    search_struct_model, search_raw_ai = await _invoke_structured(
        searcher_llm,
        SearchSummary,
        [
            SystemMessage(content=SEARCH_SUMMARY_PROMPT),
            HumanMessage(
                content=(
                    f"Medical Analysis Result:\n{text_analysis or state.get('raw_text', '')}\n\n"
                    f"Search Results:\n{combined_search}"
                )
            ),
        ],
    )
    search_results = _render_search_summary_text(search_struct_model)
    await _mark_user_substep(run_id, "searcher", "search_collect", "Retrieve related medical evidence", "done", "Related medical evidence retrieved.")
    await _mark_user_substep(run_id, "searcher", "search_filter", "Filter reliable content", "running", "Filtering higher-value reliable content.")
    await _mark_user_substep(run_id, "searcher", "search_filter", "Filter reliable content", "done", "Content filtering completed.")
    await _mark_user_substep(run_id, "searcher", "search_summary", "Organize supplemental notes", "running", "Organizing readable supplemental notes.")

    llm_messages: list[AIMessage] = [search_raw_ai]
    await _mark_user_substep(run_id, "tooler", "build_conclusion", "Build analysis conclusion", "running", "Integrating materials to produce analysis conclusions.")
    if image_analyses:
        image_analysis = "\n---\n".join(image_analyses)
        merge_llm = get_chat_model("TOOLER_MERGE", default_model=_TOOLER_MERGE_MODEL, temperature=0.2)
        merged_model, merge_raw_ai = await _invoke_structured(
            merge_llm,
            MergedMedicalAnalysis,
            [
                SystemMessage(content=MERGE_PROMPT),
                HumanMessage(
                    content=(
                        f"Text Analysis Result:\n{text_analysis}\n\n"
                        f"Image Analysis Result:\n{image_analysis}"
                    )
                ),
            ],
        )
        llm_messages.append(merge_raw_ai)
        merged = _render_merged_analysis_text(merged_model)
        merged_struct = merged_model.model_dump()
    else:
        image_analysis = ""
        if text_struct:
            merged_model = MergedMedicalAnalysis(
                primary_findings=text_struct.get("main_diagnoses", []),
                key_abnormalities=text_struct.get("abnormal_indicators", []),
                risk_assessment=text_struct.get("risk_assessment", []),
                attention_points=text_struct.get("follow_up_points", []),
            )
        else:
            merged_model = MergedMedicalAnalysis(primary_findings=[text_analysis] if text_analysis else [])
        merged = _render_merged_analysis_text(merged_model)
        merged_struct = merged_model.model_dump()
    await _mark_user_substep(run_id, "tooler", "build_conclusion", "Build analysis conclusion", "done", "Analysis conclusions generated.")
    await _mark_user_substep(run_id, "searcher", "search_summary", "Organize supplemental notes", "done", "Supplemental notes prepared.")

    log_stage(logger, "tooler.text", text_analysis)
    if image_analysis:
        log_stage(logger, "tooler.image", image_analysis)
    log_stage(logger, "tooler.merge", merged)
    log_stage(logger, "searcher", search_results)
    logger.info("[tooler] exit")
    await mark_stage(run_id, "tooler", "done", content=merged)
    await mark_stage(run_id, "searcher", "done", content=search_results)

    return {
        "messages": [*tool_messages, *llm_messages],
        "medical_text_analysis": text_analysis,
        "medical_text_analysis_struct": text_struct,
        "medical_image_analysis": image_analysis,
        "medical_image_analysis_struct": image_structs,
        "merged_analysis": merged,
        "merged_analysis_struct": merged_struct,
        "search_results": search_results,
        "search_results_struct": search_struct_model.model_dump(),
    }


# ── Reflector ───────────────────────────────────────────────────────
async def reflector_node(state: MedAgentState) -> dict:
    """Review analysis + search + plan for consistency and accuracy."""
    logger.info("[reflector] enter")
    run_id = state["run_id"]
    await mark_stage(run_id, "reflector", "running", content="Running quality review.")
    await _mark_user_substep(run_id, "reflector", "check_consistency", "Check consistency", "running", "Checking consistency between analysis and recommendations.")
    llm = get_chat_model("REFLECTOR", default_model=_REFLECTOR_MODEL, temperature=0.2)

    report, raw_ai = await _invoke_structured(
        llm,
        ReflectReport,
        [
            SystemMessage(content=REFLECT_PROMPT),
            HumanMessage(
                content=(
                    f"Medical Analysis:\n{state.get('merged_analysis', '')}\n\n"
                    f"Knowledge Supplement:\n{state.get('search_results', '')}\n\n"
                    f"Care Plan:\n{state.get('plan', '')}"
                )
            ),
        ],
    )

    next_iteration = int(state.get("iteration", 0)) + 1
    is_fail = report.quality_conclusion == "FAIL"
    hit_limit = next_iteration >= _MAX_REFLECT_ITERATIONS
    planner_decision = "REDO" if (is_fail and not hit_limit) else "SUMMARY"

    reflection_text = _render_reflection_text(report)
    await _mark_user_substep(run_id, "reflector", "check_consistency", "Check consistency", "done", "Consistency check completed.")
    await _mark_user_substep(run_id, "reflector", "check_actionability", "Check actionability", "running", "Evaluating whether recommendations are clear and actionable.")
    await _mark_user_substep(
        run_id,
        "reflector",
        "check_actionability",
        "Check actionability",
        "done",
        "Actionability check completed." if planner_decision == "SUMMARY" else "Recommendations still need refinement.",
    )
    await _mark_user_substep(run_id, "reflector", "review_done", "Finish quality review", "done", "Quality review completed.")

    log_stage(logger, "reflector", reflection_text)
    logger.info("[reflector] exit")
    await mark_stage(run_id, "reflector", "done", content=reflection_text)
    update: dict[str, Any] = {
        "messages": [raw_ai],
        "reflection": reflection_text,
        "reflection_struct": report.model_dump(),
        "iteration": next_iteration,
        "planner_decision": planner_decision,
    }
    return update


# ── Summarize ───────────────────────────────────────────────────────
async def summarize_node(state: MedAgentState) -> dict:
    """Produce a patient-friendly summary incorporating reflection."""
    logger.info("[summarizer] enter")
    run_id = state["run_id"]
    await mark_stage(run_id, "summarize", "running", content="Generating your summary.")
    await _mark_user_substep(run_id, "summarize", "summary_plain", "Convert to plain language", "running", "Converting key information into easy-to-read language.")

    if state.get("summary"):
        log_stage(logger, "summarizer", state["summary"])
        logger.info("[summarizer] exit (pre-filled summary)")
        await _mark_user_substep(run_id, "summarize", "summary_plain", "Convert to plain language", "done", "Plain-language conversion completed.")
        await _mark_user_substep(run_id, "summarize", "summary_actions", "Generate action suggestions", "done", "Action suggestions are ready.")
        await _mark_user_substep(run_id, "summarize", "summary_done", "Output final summary", "done", "Final summary generated.")
        await mark_stage(run_id, "summarize", "done", content=state["summary"])
        return {
            "summary": state["summary"],
            "summary_struct": state.get("summary_struct"),
            "messages": [AIMessage(content=state["summary"])],
        }

    llm = get_chat_model("SUMMARIZER", default_model=_SUMMARIZER_MODEL, temperature=0.2)

    if (state.get("query_intent") or "").upper() == "NON_MEDICAL":
        logger.info("[summarizer] generating non-medical reply via llm model=%s", _SUMMARIZER_MODEL)
        response = await llm.ainvoke(
            [
                SystemMessage(content=NON_MEDICAL_REPLY_PROMPT),
                HumanMessage(content=state.get("raw_text", "")),
            ]
        )
        ai_response = response if isinstance(response, AIMessage) else AIMessage(content=str(response))
        summary_text = str(ai_response.content)
        await _mark_user_substep(run_id, "summarize", "summary_plain", "Convert to plain language", "done", "Content organization completed.")
        await _mark_user_substep(run_id, "summarize", "summary_actions", "Generate action suggestions", "done", "Actionable recommendations added.")
        await _mark_user_substep(run_id, "summarize", "summary_done", "Output final summary", "done", "Final reply generated.")
        log_stage(logger, "summarizer", summary_text)
        logger.info("[summarizer] exit (non-medical llm reply)")
        await mark_stage(run_id, "summarize", "done", content=summary_text)
        return {
            "summary": summary_text,
            "summary_struct": {
                "report_title": "General Consultation Reply",
                "brief_summary": summary_text[:60],
                "key_findings": ["Non-medical consultation"],
                "medication_reminders": [],
                "follow_up_tips": [],
            },
            "messages": [ai_response],
        }

    summary_struct_model, raw_ai = await _invoke_structured(
        llm,
        PatientSummary,
        [
            SystemMessage(content=SUMMARIZE_PROMPT),
            HumanMessage(
                content=(
                    f"User Medical Input:\n{state.get('raw_text', '')}\n\n"
                    f"Medical Analysis:\n{state.get('merged_analysis', '')}\n\n"
                    f"Knowledge Supplement:\n{state.get('search_results', '')}\n\n"
                    f"Care Plan:\n{state.get('plan', '')}\n\n"
                )
            ),
        ],
    )
    summary_text = _render_summary_text(summary_struct_model)
    await _mark_user_substep(run_id, "summarize", "summary_plain", "Convert to plain language", "done", "Plain-language organization completed.")
    await _mark_user_substep(run_id, "summarize", "summary_actions", "Generate action suggestions", "done", "Executable action suggestions generated.")
    await _mark_user_substep(run_id, "summarize", "summary_done", "Output final summary", "done", "Final summary generated.")

    log_stage(logger, "summarizer", summary_text)
    logger.info("[summarizer] exit")
    await mark_stage(run_id, "summarize", "done", content=summary_text)
    return {
        "summary": summary_text,
        "summary_struct": summary_struct_model.model_dump(),
        "messages": [raw_ai],
    }
