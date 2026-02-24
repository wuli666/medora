import asyncio
import json
import time
from copy import deepcopy
from typing import Any

STAGE_META = [
    ("quick_router", "意图识别"),
    ("planner", "管理计划生成"),
    ("tooler", "病历/影像解析"),
    ("searcher", "医学检索补充"),
    ("reflector", "一致性校验"),
    ("summarize", "患者摘要生成"),
]

_RUNS: dict[str, dict[str, Any]] = {}
_SUBSCRIBERS: dict[str, set[asyncio.Queue]] = {}
_LOCK = asyncio.Lock()


def _terminal_status(status: str) -> bool:
    return status in {"done", "skipped", "error"}


def _ensure_stage_shape(stage: dict[str, Any]) -> None:
    stage.setdefault("substeps", [])
    stage.setdefault("current_substep", "")


async def _broadcast(run_id: str, event: str, data: dict[str, Any]) -> None:
    async with _LOCK:
        queues = list(_SUBSCRIBERS.get(run_id, set()))
    if not queues:
        return
    packet = {"event": event, "data": data}
    for q in queues:
        q.put_nowait(packet)

async def begin_run(run_id: str) -> None:
    payload = None

    default_stages = [
        {
            "key": key,
            "label": label,
            "status": "pending",
            "content": "",
            "started_at": None,
            "ended_at": None,
            "substeps": [],
            "current_substep": "",
        }
        for key, label in STAGE_META
    ]

    async with _LOCK:
        _RUNS[run_id] = {
            "run_id": run_id,
            "done": False,
            "error": "",
            "response": None,
            "stages": default_stages,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        payload = deepcopy(_RUNS[run_id])
    await _broadcast(run_id, "run_started", payload)


async def mark_stage(run_id: str, stage_key: str, status: str, content: str = "") -> None:
    payload = None
    async with _LOCK:
        run = _RUNS.get(run_id)
        if not run:
            return
        stage = next((s for s in run["stages"] if s["key"] == stage_key), None)
        if not stage:
            return
        _ensure_stage_shape(stage)
        now = time.time()
        stage["status"] = status
        if status == "running" and not stage["started_at"]:
            stage["started_at"] = now
        if _terminal_status(status):
            stage["ended_at"] = now
            stage["current_substep"] = ""
        if content:
            stage["content"] = content
        run["updated_at"] = now
        payload = {
            "run_id": run_id,
            "stage": deepcopy(stage),
            "stages": deepcopy(run["stages"]),
            "updated_at": run["updated_at"],
        }
    if payload:
        await _broadcast(run_id, "stage_update", payload)


async def set_stage_current_substep(run_id: str, stage_key: str, substep_id: str = "") -> None:
    payload = None
    async with _LOCK:
        run = _RUNS.get(run_id)
        if not run:
            return
        stage = next((s for s in run["stages"] if s["key"] == stage_key), None)
        if not stage:
            return
        _ensure_stage_shape(stage)
        stage["current_substep"] = substep_id or ""
        run["updated_at"] = time.time()
        payload = {
            "run_id": run_id,
            "stage": deepcopy(stage),
            "stages": deepcopy(run["stages"]),
            "updated_at": run["updated_at"],
        }
    if payload:
        await _broadcast(run_id, "stage_update", payload)


async def mark_substep(
    run_id: str,
    stage_key: str,
    substep_id: str,
    status: str,
    label: str = "",
    detail: str = "",
) -> None:
    payload = None
    async with _LOCK:
        run = _RUNS.get(run_id)
        if not run:
            return
        stage = next((s for s in run["stages"] if s["key"] == stage_key), None)
        if not stage:
            return
        _ensure_stage_shape(stage)
        now = time.time()

        substeps: list[dict[str, Any]] = stage["substeps"]
        substep = next((s for s in substeps if s.get("id") == substep_id), None)
        if substep is None:
            substep = {
                "id": substep_id,
                "label": label or substep_id,
                "status": "pending",
                "detail": "",
                "started_at": None,
                "ended_at": None,
            }
            substeps.append(substep)

        if label:
            substep["label"] = label
        substep["status"] = status
        if status == "running" and not substep["started_at"]:
            substep["started_at"] = now
            substep["ended_at"] = None
            stage["current_substep"] = substep_id
        if _terminal_status(status):
            substep["ended_at"] = now
            if stage.get("current_substep") == substep_id:
                stage["current_substep"] = ""
        if detail:
            substep["detail"] = detail

        run["updated_at"] = now
        payload = {
            "run_id": run_id,
            "stage": deepcopy(stage),
            "stages": deepcopy(run["stages"]),
            "updated_at": run["updated_at"],
        }
    if payload:
        await _broadcast(run_id, "stage_update", payload)


async def skip_remaining_after(run_id: str, stage_key: str) -> None:
    payload = None
    async with _LOCK:
        run = _RUNS.get(run_id)
        if not run:
            return
        hit = False
        now = time.time()
        for s in run["stages"]:
            _ensure_stage_shape(s)
            if s["key"] == stage_key:
                hit = True
                continue
            if hit and s["status"] == "pending":
                s["status"] = "skipped"
                s["ended_at"] = now
                s["current_substep"] = ""
        run["updated_at"] = now
        payload = {
            "run_id": run_id,
            "stages": deepcopy(run["stages"]),
            "updated_at": run["updated_at"],
        }
    if payload:
        await _broadcast(run_id, "stage_update", payload)


async def complete_run(run_id: str, response: dict[str, Any]) -> None:
    payload = None
    async with _LOCK:
        run = _RUNS.get(run_id)
        if not run:
            return
        run["done"] = True
        run["response"] = response
        run["updated_at"] = time.time()
        payload = deepcopy(run)
    if payload:
        await _broadcast(run_id, "run_completed", payload)


async def fail_run(run_id: str, error: str) -> None:
    payload = None
    async with _LOCK:
        run = _RUNS.get(run_id)
        if not run:
            return
        run["done"] = True
        run["error"] = error
        run["updated_at"] = time.time()
        payload = deepcopy(run)
    if payload:
        await _broadcast(run_id, "run_failed", payload)


async def get_run(run_id: str) -> dict[str, Any] | None:
    async with _LOCK:
        run = _RUNS.get(run_id)
        return deepcopy(run) if run else None


async def subscribe(run_id: str) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    async with _LOCK:
        _SUBSCRIBERS.setdefault(run_id, set()).add(queue)
    return queue


async def unsubscribe(run_id: str, queue: asyncio.Queue) -> None:
    async with _LOCK:
        queues = _SUBSCRIBERS.get(run_id)
        if not queues:
            return
        queues.discard(queue)
        if not queues:
            _SUBSCRIBERS.pop(run_id, None)


async def next_event(queue: asyncio.Queue, timeout: float = 15.0) -> dict[str, Any] | None:
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


def to_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
