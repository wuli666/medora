from typing import TypedDict
from langgraph.graph import MessagesState

class MedAgentState(MessagesState):
    run_id: str
    # Input
    raw_text: str
    images: list[str]  # base64 encoded
    has_images: bool

    # MedGemma analysis
    medical_text_analysis: str
    medical_image_analysis: str
    merged_analysis: str

    # Search & knowledge
    search_results: str

    # Planning
    plan: str
    plan_updated: bool
    planner_decision: str
    tools_dispatched: bool
    planner_tool_attempts: int

    # Reflection
    reflection: str

    # Final output
    summary: str
    tool_skipped: bool

    # Supervisor loop control
    iteration: int
    revision_feedback: str  # previous reflection carried as feedback for redo
    query_intent: str  # "MEDICAL" | "NON_MEDICAL"
