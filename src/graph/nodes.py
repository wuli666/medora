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
from src.runtime.progress import mark_stage
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
_NO_TOOL_SEARCH_FALLBACK = "（未执行检索，基于已有信息生成）"


class UserIntent(BaseModel):
    INTENT: Literal["MEDICAL", "NON_MEDICAL"] = Field(
        description="User query intent label."
    )


def _join_lines(items: list[str]) -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return "\n".join(f"- {item}" for item in cleaned)


def _render_care_plan_text(plan: CarePlan) -> str:
    return (
        "1. 病情分析\n"
        f"{_join_lines(plan.condition_analysis) or '- 暂无'}\n\n"
        "2. 监测指标\n"
        f"{_join_lines(plan.monitoring_metrics) or '- 暂无'}\n\n"
        "3. 生活方式建议\n"
        f"{_join_lines(plan.lifestyle_advice) or '- 暂无'}"
    )


def _render_reflection_text(report: ReflectReport) -> str:
    fixes = _join_lines(report.minimal_corrections[:3]) or "- 暂无"
    return (
        f"1. 完整性检查：{report.completeness_check}\n"
        f"2. 一致性检查：{report.consistency_check}\n"
        f"3. 幻觉风险：{report.hallucination_risk}\n"
        f"4. 最小修正建议：\n{fixes}\n"
        f"5. 质检结论：{report.quality_conclusion}"
    )


def _render_summary_text(summary: PatientSummary) -> str:
    return (
        "1. 关键发现\n"
        f"{_join_lines(summary.key_findings) or '- 暂无'}\n\n"
        "2. 建议行动\n"
        f"{_join_lines(summary.action_items) or '- 暂无'}\n\n"
        "3. 注意事项\n"
        f"{_join_lines(summary.cautions) or '- 暂无'}"
    )


def _render_merged_analysis_text(merged: MergedMedicalAnalysis) -> str:
    return (
        "1. 主要诊断/发现\n"
        f"{_join_lines(merged.primary_findings) or '- 暂无'}\n\n"
        "2. 关键指标异常\n"
        f"{_join_lines(merged.key_abnormalities) or '- 暂无'}\n\n"
        "3. 风险评估\n"
        f"{_join_lines(merged.risk_assessment) or '- 暂无'}\n\n"
        "4. 需要进一步关注的方面\n"
        f"{_join_lines(merged.attention_points) or '- 暂无'}"
    )


def _render_search_summary_text(summary: SearchSummary) -> str:
    return (
        "## 关键医学知识\n"
        f"{_join_lines(summary.key_points) or '- 暂无'}\n\n"
        "## 来源线索\n"
        f"{_join_lines(summary.source_notes) or '- 暂无'}\n\n"
        f"证据等级：{summary.evidence_level or '未说明'}"
    )


async def _invoke_structured(
    llm: Any,
    schema: type[BaseModel],
    messages: list,
) -> tuple[BaseModel, AIMessage]:

    structured_llm = llm.with_structured_output(schema)
    response = await structured_llm.ainvoke(messages)
    if isinstance(response, schema):
        parsed_model = response
    else:
        parsed_model = schema.model_validate(response)
    return parsed_model, AIMessage(content=parsed_model.model_dump_json(ensure_ascii=False))


def _build_plan_payload(
    raw_text: str,
    analysis: str,
    search_results: str,
    reflection: str,
) -> str:
    return (
        f"用户输入:\n{raw_text or '（无）'}\n\n"
        f"医学分析:\n{analysis or '（暂无）'}\n\n"
        f"知识补充:\n{search_results or '（暂无）'}\n\n"
        f"上一轮质检反馈:\n{reflection or '（无）'}"
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
        await mark_stage(run_id, "quick_router", "done", content="意图识别结果：非医疗咨询，直接进入总结回复。")
        return Command(
            update={"query_intent": intent, "messages": messages},
            goto="summarize",
        )

    if not intent:
        await mark_stage(run_id, "quick_router", "running", content="正在识别用户意图并选择执行路径。")
        raw_text = state.get("raw_text", "").strip()
        normalized = raw_text.lower()

        if has_images and state.get("images"):
            intent = "MEDICAL"
            route_note = "检测到影像输入，判定为医疗咨询。"
            logger.info("[supervisor] intent=MEDICAL (has_images=True)")
        elif not raw_text:
            intent = "NON_MEDICAL"
            route_note = "输入为空，按非医疗咨询处理。"
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
            route_note = f"意图识别结果：{intent}（LLM判定）。"
            if intent not in {"MEDICAL", "NON_MEDICAL"}:
                if _GREETING_PATTERN.match(normalized):
                    intent = "NON_MEDICAL"
                    route_note = "检测到问候语，按非医疗咨询处理。"
                    logger.info("[supervisor] intent=NON_MEDICAL (keyword_fallback)")
                elif any(kw in normalized for kw in _MEDICAL_KEYWORDS):
                    intent = "MEDICAL"
                    route_note = "命中医疗关键词，判定为医疗咨询。"
                    logger.info("[supervisor] intent=MEDICAL (keyword_fallback)")
                else:
                    intent = "NON_MEDICAL"
                    route_note = "未命中医疗语义，按非医疗咨询处理。"
                    logger.info("[supervisor] intent=NON_MEDICAL (keyword_fallback)")

        if intent == "NON_MEDICAL":
            logger.info("[supervisor] route=summarize (entry non-medical)")
            await mark_stage(
                run_id,
                "quick_router",
                "done",
                content=route_note or "意图识别结果：非医疗咨询，直接进入总结回复。",
            )
            return Command(
                update={"query_intent": intent, "messages": messages},
                goto="summarize",
            )

        await mark_stage(
            run_id,
            "quick_router",
            "done",
            content=route_note or "意图识别结果：医疗咨询，进入完整诊疗分析流程。",
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
    await mark_stage(run_id, "planner", "running", content="正在生成与评估健康管理计划。")

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
        await mark_stage(run_id, "planner", "done", content=plan_text)
        return Command(update={"planner_decision": "SUMMARY"}, goto="summarize")

    if decision == "REDO":
        logger.info("[planner] decision=REDO, re-planning before tool dispatch")
        analysis_context = state.get("merged_analysis", "") or state.get("raw_text", "")
        search_context = state.get("search_results", "") or "（未检索到外部证据）"

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

        logger.info("[planner] requesting tool calls via llm")
        planner_human = (
            f"用户输入：{state.get('raw_text', '')}\n\n"
            f"当前执行方案：{plan_text}\n\n"
            f"上一轮质检反馈：{reflection_text or '无'}\n\n"
            "请根据需要调用工具。"
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
    await mark_stage(state["run_id"], "tooler", "running", content="正在解析病历文本与医学影像。")

    ai_message = latest_ai_message(state)
    if ai_message is None or not ai_message.tool_calls:
        logger.info("[tooler] no pending tool calls, noop")
        await mark_stage(state["run_id"], "tooler", "done", content="未检测到可执行工具调用，跳过病历/影像解析。")
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

    combined_search = f"## 网络搜索结果\n{web_summary}\n\n## 知识库搜索结果\n{rag_summary}"

    searcher_llm = get_chat_model("SEARCHER", default_model=_SEARCHER_MODEL, temperature=0.2)
    search_struct_model, search_raw_ai = await _invoke_structured(
        searcher_llm,
        SearchSummary,
        [
            SystemMessage(content=SEARCH_SUMMARY_PROMPT),
            HumanMessage(
                content=(
                    f"医学分析结果:\n{text_analysis or state.get('raw_text', '')}\n\n"
                    f"搜索结果:\n{combined_search}"
                )
            ),
        ],
    )
    search_results = _render_search_summary_text(search_struct_model)

    llm_messages: list[AIMessage] = [search_raw_ai]
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
                        f"文本分析结果:\n{text_analysis}\n\n"
                        f"图像分析结果:\n{image_analysis}"
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

    log_stage(logger, "tooler.text", text_analysis)
    if image_analysis:
        log_stage(logger, "tooler.image", image_analysis)
    log_stage(logger, "tooler.merge", merged)
    log_stage(logger, "searcher", search_results)
    logger.info("[tooler] exit")
    await mark_stage(state["run_id"], "tooler", "done", content=merged)
    await mark_stage(state["run_id"], "searcher", "done", content=search_results)

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
    await mark_stage(state["run_id"], "reflector", "running", content="正在校验分析结论与管理建议的一致性。")
    llm = get_chat_model("REFLECTOR", default_model=_REFLECTOR_MODEL, temperature=0.2)

    report, raw_ai = await _invoke_structured(
        llm,
        ReflectReport,
        [
            SystemMessage(content=REFLECT_PROMPT),
            HumanMessage(
                content=(
                    f"医学分析:\n{state.get('merged_analysis', '')}\n\n"
                    f"知识补充:\n{state.get('search_results', '')}\n\n"
                    f"健康管理方案:\n{state.get('plan', '')}"
                )
            ),
        ],
    )

    next_iteration = int(state.get("iteration", 0)) + 1
    is_fail = report.quality_conclusion == "FAIL"
    hit_limit = next_iteration >= _MAX_REFLECT_ITERATIONS
    planner_decision = "REDO" if (is_fail and not hit_limit) else "SUMMARY"

    reflection_text = _render_reflection_text(report)

    log_stage(logger, "reflector", reflection_text)
    logger.info("[reflector] exit")
    await mark_stage(state["run_id"], "reflector", "done", content=reflection_text)
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
    await mark_stage(state["run_id"], "summarize", "running", content="正在生成患者可读摘要。")

    if state.get("summary"):
        log_stage(logger, "summarizer", state["summary"])
        logger.info("[summarizer] exit (pre-filled summary)")
        await mark_stage(state["run_id"], "summarize", "done", content=state["summary"])
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
        log_stage(logger, "summarizer", summary_text)
        logger.info("[summarizer] exit (non-medical llm reply)")
        await mark_stage(state["run_id"], "summarize", "done", content=summary_text)
        return {
            "summary": summary_text,
            "summary_struct": {
                "key_findings": ["非医疗咨询"],
                "action_items": ["已按通用助手方式回复"],
                "cautions": [],
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
                    f"医学分析:\n{state.get('merged_analysis', '')}\n\n"
                    f"知识补充:\n{state.get('search_results', '')}\n\n"
                    f"健康管理方案:\n{state.get('plan', '')}\n\n"
                    f"质量审查意见:\n{state.get('reflection', '')}"
                )
            ),
        ],
    )
    summary_text = _render_summary_text(summary_struct_model)

    log_stage(logger, "summarizer", summary_text)
    logger.info("[summarizer] exit")
    await mark_stage(state["run_id"], "summarize", "done", content=summary_text)
    return {
        "summary": summary_text,
        "summary_struct": summary_struct_model.model_dump(),
        "messages": [raw_ai],
    }
