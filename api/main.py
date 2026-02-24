import os
import tempfile
import time
import re
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from langchain_core.messages import HumanMessage

from api.schemas import IntentRouteResponse, MultiAgentResponse, StageItem
from src.config.logger import configure_logging, get_logger
from src.graph.builder import get_graph_app
from src.llm.model_factory import get_chat_model
from src.prompts.prompts import INTENT_CLASSIFY_PROMPT
from src.runtime.progress import begin_run, complete_run, fail_run, get_run, next_event, subscribe, to_sse, unsubscribe
from src.utils.pdf_parser import parse_pdf
from src.utils.db import init_db
from src.utils.image_utils import image_bytes_to_base64

app = FastAPI(title="MedGemma Multi-Agent System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

configure_logging()
logger = get_logger(__name__)
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


def _looks_medical(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    if _GREETING_PATTERN.match(normalized):
        return False
    return any(k in normalized for k in _MEDICAL_KEYWORDS)


def _route_stages(intent: str) -> list[StageItem]:
    if intent == "NON_MEDICAL":
        return [
            StageItem(
                key="quick_router",
                label="意图识别",
                status="running",
                content="",
            )
        ]
    return [
        StageItem(key="quick_router", label="意图识别", status="running", content=""),
        StageItem(key="planner", label="管理计划生成", status="pending", content=""),
        StageItem(key="tooler", label="病历/影像解析", status="pending", content=""),
        StageItem(key="searcher", label="医学检索补充", status="pending", content=""),
        StageItem(key="reflector", label="一致性校验", status="pending", content=""),
        StageItem(key="summarize", label="患者摘要生成", status="pending", content=""),
    ]


async def _classify_intent_backend(user_text: str, has_image: bool) -> str:
    if has_image:
        return "MEDICAL"
    normalized = (user_text or "").strip()
    if not normalized:
        return "NON_MEDICAL"

    llm = get_chat_model("SUPERVISOR", default_model="qwen-plus", temperature=0.1)
    if llm is not None:
        try:
            prompt = INTENT_CLASSIFY_PROMPT.format(user_text=normalized)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            label = str(response.content).strip().upper().replace("-", "_")
            if label in {"MEDICAL", "NON_MEDICAL"}:
                if label == "NON_MEDICAL" and _looks_medical(normalized):
                    return "MEDICAL"
                return label
        except Exception:
            pass

    if _GREETING_PATTERN.match(normalized):
        return "NON_MEDICAL"
    return "MEDICAL"


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    logger.info("[request.start] %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "[request.end] %s %s status=%s elapsed=%.1fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/multi-agent/intent", response_model=IntentRouteResponse)
async def classify_intent(
    patient_text: str = Form(...),
    has_image: bool = Form(False),
):
    intent = await _classify_intent_backend(patient_text, has_image=has_image)
    route = "direct_reply" if intent == "NON_MEDICAL" else "full_pipeline"
    logger.info("[intent] intent=%s route=%s", intent, route)
    return IntentRouteResponse(intent=intent, route=route, stages=_route_stages(intent))


@app.post("/api/multi-agent/run", response_model=MultiAgentResponse)
async def run_multi_agent(
    patient_text: str = Form(...),
    run_id: str | None = Form(None),
    image: UploadFile | None = File(None),
    pdf: UploadFile | None = File(None),
):
    run_id = (run_id or "").strip() or str(uuid.uuid4())
    await begin_run(run_id)
    logger.info("[run_multi_agent] started text_len=%s has_image=%s has_pdf=%s", len(patient_text or ""), bool(image), bool(pdf))
    raw_text = patient_text
    images: list[str] = []

    # Process PDF
    if pdf:
        content = await pdf.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            pdf_data = await parse_pdf(tmp_path)
            if pdf_data["text"]:
                raw_text += "\n\n" + pdf_data["text"]
            images.extend(pdf_data["images"])
        finally:
            os.unlink(tmp_path)

    # Process image
    if image:
        content = await image.read()
        images.append(image_bytes_to_base64(content))

    initial_state = {
        "run_id": run_id,
        "raw_text": raw_text,
        "images": images,
        "has_images": len(images) > 0,
        "medical_text_analysis": "",
        "medical_image_analysis": "",
        "merged_analysis": "",
        "search_results": "",
        "plan": "",
        "planner_decision": "",
        "tools_dispatched": False,
        "planner_tool_attempts": 0,
        "reflection": "",
        "summary": "",
        "tool_skipped": False,
        "iteration": 0,
        "query_intent": "",
    }

    graph_app = get_graph_app()
    try:
        logger.info("[run_multi_agent] graph_invoke begin")
        result = await graph_app.ainvoke(initial_state)
        logger.info("[run_multi_agent] graph_invoke end")
    except Exception as exc:
        logger.exception("[run_multi_agent] graph failed")
        await fail_run(run_id, f"后端处理失败: {exc}")
        raise HTTPException(status_code=500, detail=f"后端处理失败: {exc}") from exc

    # Build stage outputs
    tool_output = result.get("merged_analysis", "")
    search_output = result.get("search_results", "")
    planner_output = result.get("plan", "")
    reflect_output = result.get("reflection", "")
    summary_output = result.get("summary", "")
    tool_skipped = bool(result.get("tool_skipped"))

    stages = [
        StageItem(
            key="planner",
            label="管理计划生成",
            status="done" if planner_output else "skipped",
            content=planner_output,
        ),
        StageItem(
            key="tooler",
            label="病历/影像解析",
            status="skipped" if tool_skipped else ("done" if tool_output else "skipped"),
            content="" if tool_skipped else tool_output,
        ),
        StageItem(
            key="searcher",
            label="医学检索补充",
            status="done" if search_output else "skipped",
            content=search_output,
        ),
        StageItem(
            key="reflector",
            label="一致性校验",
            status="done" if reflect_output else "skipped",
            content=reflect_output,
        ),
        StageItem(
            key="summarize",
            label="患者摘要",
            status="done" if summary_output else "skipped",
            content=summary_output,
        ),
    ]

    response_payload = MultiAgentResponse(
        run_id=run_id,
        summary=summary_output,
        planner=planner_output,
        tool=tool_output,
        search=search_output,
        reflect_verify=reflect_output,
        stages=stages,
    )
    await complete_run(run_id, response_payload.model_dump())
    logger.info("[run_multi_agent] finished summary_len=%s", len(summary_output or ""))
    return response_payload


@app.get("/api/multi-agent/events/{run_id}")
async def stream_run_events(run_id: str):
    queue = await subscribe(run_id)

    async def event_generator():
        try:
            snapshot = await get_run(run_id)
            if snapshot:
                yield to_sse("snapshot", snapshot)
            while True:
                event = await next_event(queue, timeout=15.0)
                if event is None:
                    yield "event: ping\ndata: {}\n\n"
                    latest = await get_run(run_id)
                    if latest and latest.get("done"):
                        break
                    continue
                yield to_sse(event["event"], event["data"])
                if event["event"] in {"run_completed", "run_failed"}:
                    break
        finally:
            await unsubscribe(run_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Serve frontend static files (mount last so API routes take priority)
# Optional: run API-only mode when `web/` is absent.
if os.path.isdir("web"):
    app.mount("/", StaticFiles(directory="web", html=True), name="static")
else:
    logger.info("[static] skip mount: 'web/' directory not found (API-only mode)")
