import asyncio
import re
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, Send

from src.config.logger import get_logger, log_stage
from src.graph.state import MedAgentState
from src.config.settings import settings
from src.llm.model_factory import get_chat_model, safe_llm_call
from src.prompts.prompts import (
    INTENT_CLASSIFY_PROMPT,
    MERGE_PROMPT,
    NON_MEDICAL_REPLY_PROMPT,
    PLAN_INIT_PROMPT,
    PLAN_PROMPT,
    REFLECT_PROMPT,
    SEARCH_SUMMARY_PROMPT,
    SUMMARIZE_PROMPT,
    SUPERVISOR_EVAL_PROMPT,
)
from src.runtime.progress import mark_stage, skip_remaining_after
from src.tool.medgemma_tool import medgemma_analyze_image, medgemma_analyze_text
from src.tool.search_tools import rag_search, web_search

MAX_REDO = 2
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
_ANALYSIS_BUDGET = settings.ANALYSIS_BUDGET_CHARS
_SEARCH_BUDGET = settings.SEARCH_BUDGET_CHARS
_PLAN_BUDGET = settings.PLAN_BUDGET_CHARS
_REFLECTION_BUDGET = settings.REFLECTION_BUDGET_CHARS

_SUPERVISOR_MODEL = settings.SUPERVISOR_MODEL or "qwen-plus"

def _clip_text(text: str, limit: int) -> str:
    s = (text or "").strip()
    if len(s) <= limit:
        return s
    head = int(limit * 0.72)
    tail = max(120, limit - head - 24)
    return f"{s[:head]}\n...[内容已压缩]...\n{s[-tail:]}"


async def _classify_query_intent(state: MedAgentState) -> str:
    """Classify user intent via LLM first; keyword rule as fallback."""
    has_images = bool(state.get("has_images")) and bool(state.get("images"))
    raw_text = state.get("raw_text", "").strip()
    normalized = raw_text.lower()

    if has_images:
        log_stage(logger, "supervisor.intent", "MEDICAL (has_images=True)")
        return "MEDICAL"
    if not raw_text:
        log_stage(logger, "supervisor.intent", "NON_MEDICAL (empty text)")
        return "NON_MEDICAL"

    llm = get_chat_model("SUPERVISOR", default_model=_SUPERVISOR_MODEL, temperature=0.1)
    if llm is not None:
        prompt = INTENT_CLASSIFY_PROMPT.format(user_text=raw_text)
        response = await safe_llm_call(llm, prompt, "supervisor.intent")
        
        label = response.strip().upper().replace("-", "_")

        log_stage(logger, "supervisor.intent", f"{label} (LLM classified)")
        return label

    # LLM unavailable, fallback to keyword + greeting rules.
    if _GREETING_PATTERN.match(normalized):
        label = "NON_MEDICAL"
    elif any(kw in normalized for kw in _MEDICAL_KEYWORDS):
        label = "MEDICAL"
    else:
        label = "NON_MEDICAL"

    log_stage(logger, "supervisor.intent", f"{label} (keyword_fallback)")
    return label


# ── Supervisor ──────────────────────────────────────────────────────
async def supervisor_node(
    state: MedAgentState,
) -> Command[Literal["tooler", "searcher", "planner", "reflector", "summarize", "__end__"]]:
    """Central router: decide which worker to invoke next."""
    logger.info("[supervisor] enter")
    intent = state.get("query_intent", "").strip().upper()
    if not intent:
        await mark_stage(state["run_id"], "quick_router", "running")
        intent = await _classify_query_intent(state)
        if intent == "NON_MEDICAL":
            llm = get_chat_model("SUPERVISOR", default_model=_SUPERVISOR_MODEL, temperature=0.3)
            if llm is not None:
                try:
                    prompt = NON_MEDICAL_REPLY_PROMPT.format(
                        user_text=state.get("raw_text", "")
                    )
                    direct_reply = await safe_llm_call(llm, prompt, "supervisor.direct_reply")
                except Exception:
                    direct_reply = ""
            else:
                direct_reply = ""
            log_stage(logger, "supervisor.route", "goto=END (non-medical@entry)")
            await mark_stage(state["run_id"], "quick_router", "done")
            await skip_remaining_after(state["run_id"], "quick_router")
            return Command(
                goto=END,
                update={
                    "query_intent": intent,
                    "summary": (
                        direct_reply
                        or "我在。你可以直接说你的问题；如果有病历、检查或用药相关内容，也可以发我帮你解读。"
                    ),
                },
            )
        await mark_stage(state["run_id"], "quick_router", "done")
        # Medical query enters planner first (todo list).
        return Command(
            goto="planner",
            update={
                "query_intent": intent,
                "tool_skipped": not bool(state.get("has_images")),
                "merged_analysis": state.get("raw_text", "") if not state.get("has_images") else state.get("merged_analysis", ""),
                "medical_text_analysis": state.get("medical_text_analysis", ""),
                "medical_image_analysis": state.get("medical_image_analysis", ""),
                "plan_updated": False,
                "planner_decision": "",
                "tools_dispatched": False,
            },
        )

    if intent == "NON_MEDICAL":
        log_stage(logger, "supervisor.route", "goto=END (non-medical)")
        await mark_stage(state["run_id"], "quick_router", "done")
        await skip_remaining_after(state["run_id"], "quick_router")
        return Command(goto=END)

    # Supervisor only handles intent + initialization.
    log_stage(logger, "supervisor.route", "goto=planner")
    return Command(goto="planner")


# ── Tooler (MedGemma text ∥ image → merge) ─────────────────────────
async def tooler_node(state: MedAgentState) -> dict:
    """Run tool stage with parallel sub-tools:
    1) text analysis
    2) image analysis (if any)
    3) research search
    """
    logger.info("[tooler] enter has_images=%s", bool(state.get("has_images")))
    await mark_stage(state["run_id"], "tooler", "running")
    await mark_stage(state["run_id"], "searcher", "running")

    async def _run_research() -> str:
        analysis_seed = state.get("raw_text", "")
        planning_hint = state.get("plan", "")
        query = f"{planning_hint}\n{analysis_seed}"[:240]
        web_results = await asyncio.to_thread(web_search.invoke, {"query": f"医学 {query}"})
        rag_results = await asyncio.to_thread(rag_search.invoke, {"query": query})
        combined = f"## 网络搜索结果\n{web_results}\n\n## 知识库搜索结果\n{rag_results}"

        llm = get_chat_model("SEARCHER", default_model="qwen-plus", temperature=0.2)
        if llm is None:
            return combined
        prompt = SEARCH_SUMMARY_PROMPT.format(
            analysis=_clip_text(analysis_seed, _ANALYSIS_BUDGET),
            search_results=_clip_text(combined, _SEARCH_BUDGET),
        )
        try:
            return await safe_llm_call(llm, prompt, "tooler.search")
        except Exception as exc:
            return f"{combined}\n\n## 检索总结\n检索总结模型暂不可用：{str(exc).strip() or exc.__class__.__name__}"

    tasks = [medgemma_analyze_text(state["raw_text"]), _run_research()]
    if state.get("has_images") and state.get("images"):
        for img in state["images"]:
            tasks.append(medgemma_analyze_image(img, state["raw_text"]))

    results = await asyncio.gather(*tasks)

    text_analysis = results[0]
    search_results = results[1]
    image_analyses = list(results[2:])

    if image_analyses:
        image_analysis = "\n---\n".join(image_analyses)
        llm = get_chat_model("TOOLER_MERGE", default_model="qwen-plus", temperature=0.2)
        if llm is None:
            merged = f"{text_analysis}\n\n## 图像分析补充\n{image_analysis}"
        else:
            prompt = MERGE_PROMPT.format(
                text_analysis=text_analysis,
                image_analysis=image_analysis,
            )
            try:
                merged = await safe_llm_call(llm, prompt, "tooler.merge")
            except Exception:
                merged = f"{text_analysis}\n\n## 图像分析补充\n{image_analysis}"
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
        "medical_text_analysis": text_analysis,
        "medical_image_analysis": image_analysis,
        "merged_analysis": merged,
        "search_results": search_results,
    }



# ── Searcher ────────────────────────────────────────────────────────
async def searcher_node(state: MedAgentState) -> dict:
    """Web + RAG search, then LLM summarisation."""
    logger.info("[searcher] enter")
    await mark_stage(state["run_id"], "searcher", "running")
    analysis = state["merged_analysis"] or state.get("raw_text", "")
    planning_hint = state.get("plan", "")
    query = f"{planning_hint}\n{analysis}"[:240]

    logger.info("[searcher] web_search begin")
    web_results = web_search.invoke({"query": f"医学 {query}"})
    logger.info("[searcher] web_search end")
    logger.info("[searcher] rag_search begin")
    rag_results = rag_search.invoke({"query": query})
    logger.info("[searcher] rag_search end")
    combined = f"## 网络搜索结果\n{web_results}\n\n## 知识库搜索结果\n{rag_results}"

    llm = get_chat_model("SEARCHER", default_model="qwen-plus", temperature=0.2)
    if llm is None:
        log_stage(logger, "searcher", combined)
        logger.info("[searcher] exit (llm unavailable)")
        await mark_stage(state["run_id"], "searcher", "done")
        return {"search_results": combined}
    prompt = SEARCH_SUMMARY_PROMPT.format(
        analysis=_clip_text(analysis, _ANALYSIS_BUDGET),
        search_results=_clip_text(combined, _SEARCH_BUDGET),
    )
    try:
        response = await safe_llm_call(llm, prompt, "searcher")
    except Exception as exc:
        fallback = f"{combined}\n\n## 检索总结\n检索总结模型暂不可用：{str(exc).strip() or exc.__class__.__name__}"
        log_stage(logger, "searcher", fallback)
        logger.info("[searcher] exit (llm error)")
        await mark_stage(state["run_id"], "searcher", "done")
        return {"search_results": fallback}
    log_stage(logger, "searcher", response)
    logger.info("[searcher] exit")
    await mark_stage(state["run_id"], "searcher", "done")
    return {"search_results": response}


# ── Planner ─────────────────────────────────────────────────────────
async def planner_node(
    state: MedAgentState,
) -> Command[Literal["tooler", "reflector", "summarize"]]:
    """Planner-centric orchestrator using Command + Send handoff."""
    logger.info("[planner] enter")
    await mark_stage(state["run_id"], "planner", "running")

    llm = get_chat_model("PLANNER", default_model="qwen-plus", temperature=0.3)

    def _fallback_plan() -> str:
        return (
            "当前无法调用规划模型。请先完成以下基础动作：\n"
            "1. 规律作息与补水，记录症状变化\n"
            "2. 规范用药并避免重复叠加\n"
            "3. 症状持续或加重时尽快线下就医"
        )

    # 1) Initial todo list (non-blocking, no model call).
    if not state.get("plan"):
        if llm is None:
            plan_text = _fallback_plan()
        else:
            init_prompt = PLAN_INIT_PROMPT.format(raw_text=_clip_text(state.get("raw_text", ""), 1200))
            try:
                plan_text = await safe_llm_call(llm, init_prompt, "planner.init")
            except Exception as exc:
                plan_text = f"{_fallback_plan()}\n(初始化失败: {str(exc).strip() or exc.__class__.__name__})"
        log_stage(logger, "planner.init", plan_text)
        await mark_stage(state["run_id"], "planner", "done")
        sends: list[Send] = [Send("tooler", dict(state))]
        return Command(
            goto=sends,
            update={
                "plan": plan_text,
                "tools_dispatched": True,
                "plan_updated": False,
                "planner_decision": "",
            },
        )

    # 2) Wait tool outputs; then update plan.
    if not state.get("search_results") or (state.get("has_images") and not state.get("merged_analysis")):
        await mark_stage(state["run_id"], "planner", "done")
        return Command(goto="tooler")

    if not state.get("plan_updated"):
        analysis_context = state["merged_analysis"] or state.get("raw_text", "")
        search_context = state["search_results"] or "（待后续检索补充）"
        feedback = state.get("revision_feedback", "")
        feedback_section = (
            f"\n\n## 上一轮审查反馈（请据此改进方案）\n{feedback}" if feedback else ""
        )
        if llm is None:
            response = state.get("plan", "") or _fallback_plan()
        else:
            prompt = (
                PLAN_PROMPT.format(
                    analysis=_clip_text(analysis_context, _ANALYSIS_BUDGET),
                    search_results=_clip_text(search_context, _SEARCH_BUDGET),
                )
                + feedback_section
            )
            try:
                response = await safe_llm_call(llm, prompt, "planner.update")
            except Exception as exc:
                response = f"{state.get('plan', _fallback_plan())}\n(更新失败: {str(exc).strip() or exc.__class__.__name__})"
        log_stage(logger, "planner.update", response)
        await mark_stage(state["run_id"], "planner", "done")
        return Command(
            goto="reflector",
            update={"plan": response, "plan_updated": True, "planner_decision": "REFLECT"},
        )

    await mark_stage(state["run_id"], "planner", "done")
    if not state.get("reflection"):
        return Command(goto="reflector", update={"planner_decision": "REFLECT"})
    return Command(goto="summarize", update={"planner_decision": "SUMMARY"})


# ── Reflector ───────────────────────────────────────────────────────
async def reflector_node(state: MedAgentState) -> dict:
    """Review analysis + search + plan for consistency and accuracy."""
    logger.info("[reflector] enter")
    await mark_stage(state["run_id"], "reflector", "running")
    llm = get_chat_model("REFLECTOR", default_model="qwen-plus", temperature=0.2)
    if llm is None:
        fallback = "当前无法调用反思校验模型，建议由医生进一步复核方案。"
        log_stage(logger, "reflector", fallback)
        logger.info("[reflector] exit (llm unavailable)")
        await mark_stage(state["run_id"], "reflector", "done")
        return {"reflection": fallback}
    prompt = REFLECT_PROMPT.format(
        analysis=_clip_text(state["merged_analysis"], _ANALYSIS_BUDGET),
        search_results=_clip_text(state["search_results"], _SEARCH_BUDGET),
        plan=_clip_text(state["plan"], _PLAN_BUDGET),
    )
    try:
        response = await safe_llm_call(llm, prompt, "reflector")
    except Exception as exc:
        fallback = "质检暂不可用（不影响主流程）"
        log_stage(logger, "reflector", fallback)
        logger.info("[reflector] exit (llm error): %s", str(exc).strip() or exc.__class__.__name__)
        await mark_stage(state["run_id"], "reflector", "done")
        return {"reflection": fallback}
    log_stage(logger, "reflector", response)
    logger.info("[reflector] exit")
    await mark_stage(state["run_id"], "reflector", "done")
    return {"reflection": response}


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

    llm = get_chat_model("SUMMARIZER", default_model="qwen-plus", temperature=0.2)
    if llm is None:
        fallback = (
            "当前系统无法连接到总结模型，已返回基础流程结果。"
            "请在模型可用后重试，或咨询专业医生获取最终意见。"
        )
        log_stage(logger, "summarizer", fallback)
        logger.info("[summarizer] exit (llm unavailable)")
        await mark_stage(state["run_id"], "summarize", "done")
        return {"summary": fallback}

    if (state.get("query_intent") or "").upper() == "NON_MEDICAL":
        prompt = NON_MEDICAL_REPLY_PROMPT.format(user_text=state.get("raw_text", ""))
        try:
            response = await safe_llm_call(llm, prompt, "summarizer.non_medical")
        except Exception as exc:
            response = (
                "我在。你可以继续告诉我你的问题；如果有病历或检查内容，我也可以帮你解读。"
                f"(错误信息: {str(exc).strip() or exc.__class__.__name__})"
            )
        log_stage(logger, "summarizer", response)
        logger.info("[summarizer] exit (non-medical llm reply)")
        await mark_stage(state["run_id"], "summarize", "done")
        return {"summary": response}
    prompt = SUMMARIZE_PROMPT.format(
        analysis=_clip_text(state["merged_analysis"], _ANALYSIS_BUDGET),
        search_results=_clip_text(state["search_results"], _SEARCH_BUDGET),
        plan=_clip_text(state["plan"], _PLAN_BUDGET),
        reflection=_clip_text(state["reflection"], _REFLECTION_BUDGET),
    )
    try:
        response = await safe_llm_call(llm, prompt, "summarizer")
    except Exception as exc:
        response = (
            "总结模型暂不可用，以下为简要结论：\n"
            f"- 解析结果、检索补充、计划与校验已完成。\n"
            f"- 建议按计划执行，并在症状变化时及时复诊。\n"
            f"(错误信息: {str(exc).strip() or exc.__class__.__name__})"
        )
    log_stage(logger, "summarizer", response)
    logger.info("[summarizer] exit")
    await mark_stage(state["run_id"], "summarize", "done")
    return {"summary": response}
