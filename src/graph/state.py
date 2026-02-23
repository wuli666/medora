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
    planner_decision: str
    tools_dispatched: bool
    planner_tool_attempts: int

    # Reflection
    reflection: str
    iteration: int

    # Final output
    summary: str
    tool_skipped: bool

    # Supervisor loop control
    query_intent: str  # "MEDICAL" | "NON_MEDICAL"
