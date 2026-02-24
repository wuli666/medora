import asyncio
import uuid

from src.runtime import progress


def _run(coro):
    return asyncio.run(coro)


def test_begin_run_initializes_substeps() -> None:
    run_id = f"run-{uuid.uuid4()}"
    _run(progress.begin_run(run_id))
    snapshot = _run(progress.get_run(run_id))
    assert snapshot is not None
    assert snapshot["stages"]
    for stage in snapshot["stages"]:
        assert stage["substeps"] == []
        assert stage["current_substep"] == ""


def test_mark_substep_lifecycle() -> None:
    run_id = f"run-{uuid.uuid4()}"
    _run(progress.begin_run(run_id))

    _run(
        progress.mark_substep(
            run_id=run_id,
            stage_key="planner",
            substep_id="plan_draft",
            status="running",
            label="Draft initial care recommendations",
            detail="Generating recommendations.",
        )
    )
    snapshot = _run(progress.get_run(run_id))
    assert snapshot is not None
    planner = next(s for s in snapshot["stages"] if s["key"] == "planner")
    assert planner["current_substep"] == "plan_draft"
    assert planner["substeps"][0]["status"] == "running"

    _run(
        progress.mark_substep(
            run_id=run_id,
            stage_key="planner",
            substep_id="plan_draft",
            status="done",
            detail="Recommendations generated.",
        )
    )
    snapshot = _run(progress.get_run(run_id))
    assert snapshot is not None
    planner = next(s for s in snapshot["stages"] if s["key"] == "planner")
    assert planner["current_substep"] == ""
    assert planner["substeps"][0]["status"] == "done"
    assert planner["substeps"][0]["detail"] == "Recommendations generated."
