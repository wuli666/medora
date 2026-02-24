from pydantic import BaseModel, Field


class SubStepItem(BaseModel):
    id: str
    label: str
    status: str
    detail: str = ""
    started_at: float | None = None
    ended_at: float | None = None


class StageItem(BaseModel):
    key: str
    label: str
    status: str
    content: str
    substeps: list[SubStepItem] = Field(default_factory=list)
    current_substep: str = ""


class MultiAgentResponse(BaseModel):
    run_id: str
    summary: str
    planner: str
    tool: str
    search: str
    reflect_verify: str
    stages: list[StageItem]


class IntentRouteResponse(BaseModel):
    intent: str
    route: str
    stages: list[StageItem]
