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


def _default_stages():
    return [
        {
            "key": key,
            "label": label,
            "status": "pending",
            "content": "",
            "started_at": None,
            "ended_at": None,
        }
        for key, label in STAGE_META
    ]


def _find_stage(run: dict[str, Any], stage_key: str) -> dict[str, Any] | None:
    for s in run["stages"]:
        if s["key"] == stage_key:
            return s
    return None


async def begin_run(run_id: str) -> None:
    payload = None
    async with _LOCK:
        _RUNS[run_id] = {
            "run_id": run_id,
            "done": False,
            "error": "",
            "response": None,
            "stages": _default_stages(),
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
        stage = _find_stage(run, stage_key)
        if not stage:
            return
        now = time.time()
        stage["status"] = status
        if status == "running" and not stage["started_at"]:
            stage["started_at"] = now
        if status in {"done", "skipped", "error"}:
            stage["ended_at"] = now
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


async def skip_remaining_after(run_id: str, stage_key: str) -> None:
    payload = None
    async with _LOCK:
        run = _RUNS.get(run_id)
        if not run:
            return
        hit = False
        now = time.time()
        for s in run["stages"]:
            if s["key"] == stage_key:
                hit = True
                continue
            if hit and s["status"] == "pending":
                s["status"] = "skipped"
                s["ended_at"] = now
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


async def _broadcast(run_id: str, event: str, data: dict[str, Any]) -> None:
    async with _LOCK:
        queues = list(_SUBSCRIBERS.get(run_id, set()))
    if not queues:
        return
    packet = {"event": event, "data": data}
    for q in queues:
        q.put_nowait(packet)


def to_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
