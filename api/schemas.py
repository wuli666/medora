from pydantic import BaseModel


class StageItem(BaseModel):
    key: str
    label: str
    status: str
    content: str


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
